"""
test_FTM_omega.py
-----------------
Evaluate an omega-conditioned 2D Helmholtz FTM checkpoint and visualize
reconstruction quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import h5py
import matplotlib
import numpy as np
import torch

from test_FTM import _build_mask_view, _plot_compare_case, _plot_error_curves
from train_FTM_GPU import normalize_coords_to_unit
from train_FTM_omega import OmegaConditionedMLP1D, build_phi_conditional

matplotlib.use("Agg")


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _decode_batch_conditional(
    cores_re: np.ndarray,
    cores_im: np.ndarray,
    phi_by_m: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    pred_re_flat = np.einsum("bmr,mpr->bmp", cores_re, phi_by_m, optimize=True)
    pred_im_flat = np.einsum("bmr,mpr->bmp", cores_im, phi_by_m, optimize=True)
    pred = np.stack([pred_re_flat, pred_im_flat], axis=-1).reshape(
        cores_re.shape[0], cores_re.shape[1], H, W, 2
    )
    return pred.astype(np.float32)


def _load_checkpoint(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    required_cfg = [
        "rank_x",
        "rank_y",
        "hidden_dim",
        "hidden_layers",
        "activation",
        "omega_bands",
        "omega_hidden_dim",
        "film_scale",
    ]
    for key in required_cfg:
        if key not in cfg:
            raise KeyError(f"Checkpoint config missing key: {key}")

    net_x = OmegaConditionedMLP1D(
        out_dim=int(cfg["rank_x"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_hidden_layers=int(cfg["hidden_layers"]),
        activation=str(cfg["activation"]),
        omega_bands=int(cfg["omega_bands"]),
        omega_hidden_dim=int(cfg["omega_hidden_dim"]),
        film_scale=float(cfg["film_scale"]),
    ).to(device)
    net_y = OmegaConditionedMLP1D(
        out_dim=int(cfg["rank_y"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_hidden_layers=int(cfg["hidden_layers"]),
        activation=str(cfg["activation"]),
        omega_bands=int(cfg["omega_bands"]),
        omega_hidden_dim=int(cfg["omega_hidden_dim"]),
        film_scale=float(cfg["film_scale"]),
    ).to(device)

    net_x.load_state_dict(ckpt["net_x_state"])
    net_y.load_state_dict(ckpt["net_y_state"])
    net_x.eval()
    net_y.eval()

    cores_real = _to_numpy(ckpt["cores_real"]).astype(np.float32)
    cores_imag = _to_numpy(ckpt["cores_imag"]).astype(np.float32)
    if cores_real.shape != cores_imag.shape:
        raise ValueError(
            f"cores_real and cores_imag shape mismatch: {cores_real.shape} vs {cores_imag.shape}"
        )

    return {
        "config": cfg,
        "net_x": net_x,
        "net_y": net_y,
        "cores_real": cores_real,
        "cores_imag": cores_imag,
        "omega": _to_numpy(ckpt.get("omega", np.array([], dtype=np.float32))).astype(np.float32),
        "omega_min": float(ckpt.get("omega_min", 0.0)),
        "omega_max": float(ckpt.get("omega_max", 1.0)),
        "data_scale": float(ckpt.get("data_scale", 1.0)),
    }


def _normalize_omega(omega: np.ndarray, omega_min: float, omega_max: float) -> np.ndarray:
    den = max(omega_max - omega_min, 1e-12)
    return ((omega - omega_min) / den).astype(np.float32)


def evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    ckpt_info = _load_checkpoint(Path(args.ckpt), device=device)
    cfg = ckpt_info["config"]
    net_x = ckpt_info["net_x"]
    net_y = ckpt_info["net_y"]
    cores_real = ckpt_info["cores_real"]
    cores_imag = ckpt_info["cores_imag"]
    B_ckpt, M_ckpt, Rx, Ry = cores_real.shape
    R = Rx * Ry

    with h5py.File(args.data_h5, "r") as f:
        if "data" not in f or "mask_tr" not in f or "omega" not in f:
            raise KeyError("HDF5 must contain data, mask_tr, omega.")
        data_ds = f["data"]
        mask_ds = f["mask_tr"]
        omega = f["omega"][...].astype(np.float32)

        B_data, M_data, H, W, C = data_ds.shape
        if C != 2:
            raise ValueError(f"Expected channel=2, got data shape {data_ds.shape}")
        if M_data != M_ckpt:
            raise ValueError(f"Frequency length mismatch: data M={M_data}, ckpt M={M_ckpt}")

        B_eval = min(B_data, B_ckpt)
        if args.max_eval_samples > 0:
            B_eval = min(B_eval, args.max_eval_samples)
        if B_eval <= 0:
            raise ValueError("No samples to evaluate.")

        grid_x = f["grid_x"][...].astype(np.float32) if "grid_x" in f else np.linspace(0, 1, H, dtype=np.float32)
        grid_y = f["grid_y"][...].astype(np.float32) if "grid_y" in f else np.linspace(0, 1, W, dtype=np.float32)
        use_norm_coords = bool(cfg.get("normalize_coords", True))
        if use_norm_coords:
            x_coords = normalize_coords_to_unit(grid_x.astype(np.float64)).astype(np.float32)
            y_coords = normalize_coords_to_unit(grid_y.astype(np.float64)).astype(np.float32)
        else:
            x_coords = grid_x.astype(np.float32)
            y_coords = grid_y.astype(np.float32)

        x_t = torch.from_numpy(x_coords).unsqueeze(-1).to(device)
        y_t = torch.from_numpy(y_coords).unsqueeze(-1).to(device)
        omega_norm = _normalize_omega(omega, ckpt_info["omega_min"], ckpt_info["omega_max"])
        omega_t = torch.from_numpy(omega_norm).unsqueeze(-1).to(device)
        with torch.no_grad():
            phi_by_m = build_phi_conditional(net_x, net_y, x_t, y_t, omega_t)
        phi_by_m_np = phi_by_m.detach().cpu().numpy().astype(np.float32)

        if phi_by_m_np.shape[2] != R:
            raise ValueError(f"Rank mismatch: phi has {phi_by_m_np.shape[2]}, core has {R}")

        cores_re_flat = cores_real[:B_eval].reshape(B_eval, M_ckpt, R)
        cores_im_flat = cores_imag[:B_eval].reshape(B_eval, M_ckpt, R)
        data_scale = float(ckpt_info["data_scale"]) if args.denormalize else 1.0
        shared_mask, _ = _build_mask_view(mask_ds, B_eval)

        mse_freq_sum = np.zeros(M_data, dtype=np.float64)
        cnt_freq = np.zeros(M_data, dtype=np.float64)
        mse_obs_sum = np.zeros(M_data, dtype=np.float64)
        cnt_obs = np.zeros(M_data, dtype=np.float64)
        mse_unobs_sum = np.zeros(M_data, dtype=np.float64)
        cnt_unobs = np.zeros(M_data, dtype=np.float64)
        mse_sample = np.zeros(B_eval, dtype=np.float64)
        case_mse = np.zeros((B_eval, M_data), dtype=np.float64)

        if shared_mask is not None:
            obs_shared = shared_mask.reshape(M_data, -1).astype(np.float32)
            unobs_shared = 1.0 - obs_shared

        for s in range(0, B_eval, args.batch_size):
            e = min(B_eval, s + args.batch_size)
            bsz = e - s
            pred = _decode_batch_conditional(
                cores_re=cores_re_flat[s:e],
                cores_im=cores_im_flat[s:e],
                phi_by_m=phi_by_m_np,
                H=H,
                W=W,
            )
            gt = data_ds[s:e].astype(np.float32)
            if data_scale != 1.0:
                pred = pred * data_scale
                gt = gt * data_scale

            err2 = (pred - gt) ** 2
            err2_flat = err2.reshape(bsz, M_data, -1)
            gt2_flat = (gt ** 2).reshape(bsz, M_data, -1)
            sum_err = np.sum(err2_flat, axis=2)
            sum_gt = np.sum(gt2_flat, axis=2)
            rel_err = np.sqrt(sum_err / np.maximum(sum_gt, 1e-12))

            mse_freq_sum += np.sum(rel_err, axis=0)
            cnt_freq += float(bsz)
            case_mse[s:e] = rel_err
            mse_sample[s:e] = np.mean(rel_err, axis=1)

            if shared_mask is not None:
                sum_err_obs = np.sum(err2_flat * obs_shared[None, :, :], axis=2)
                sum_gt_obs = np.sum(gt2_flat * obs_shared[None, :, :], axis=2)
                rel_err_obs = np.sqrt(sum_err_obs / np.maximum(sum_gt_obs, 1e-12))

                sum_err_unobs = np.sum(err2_flat * unobs_shared[None, :, :], axis=2)
                sum_gt_unobs = np.sum(gt2_flat * unobs_shared[None, :, :], axis=2)
                rel_err_unobs = np.sqrt(sum_err_unobs / np.maximum(sum_gt_unobs, 1e-12))

                mse_obs_sum += np.sum(rel_err_obs, axis=0)
                cnt_obs += bsz
                mse_unobs_sum += np.sum(rel_err_unobs, axis=0)
                cnt_unobs += bsz
            else:
                mask_batch = (mask_ds[s:e].astype(np.float32) > 0.5)
                obs_flat = mask_batch.reshape(bsz, M_data, -1).astype(np.float32)
                unobs_flat = 1.0 - obs_flat
                sum_err_obs = np.sum(err2_flat * obs_flat, axis=2)
                sum_gt_obs = np.sum(gt2_flat * obs_flat, axis=2)
                rel_err_obs = np.sqrt(sum_err_obs / np.maximum(sum_gt_obs, 1e-12))
                sum_err_unobs = np.sum(err2_flat * unobs_flat, axis=2)
                sum_gt_unobs = np.sum(gt2_flat * unobs_flat, axis=2)
                rel_err_unobs = np.sqrt(sum_err_unobs / np.maximum(sum_gt_unobs, 1e-12))
                mse_obs_sum += np.sum(rel_err_obs, axis=0)
                cnt_obs += bsz
                mse_unobs_sum += np.sum(rel_err_unobs, axis=0)
                cnt_unobs += bsz

            print(f"Eval progress: {e}/{B_eval} samples")

    rmse_freq_full = mse_freq_sum / np.maximum(cnt_freq, 1.0)
    rmse_freq_obs = np.full_like(rmse_freq_full, np.nan)
    valid_obs = cnt_obs > 0
    rmse_freq_obs[valid_obs] = mse_obs_sum[valid_obs] / cnt_obs[valid_obs]
    rmse_freq_unobs = np.full_like(rmse_freq_full, np.nan)
    valid_unobs = cnt_unobs > 0
    rmse_freq_unobs[valid_unobs] = mse_unobs_sum[valid_unobs] / cnt_unobs[valid_unobs]
    rmse_sample = mse_sample
    global_rmse = float(np.mean(mse_sample))

    _plot_error_curves(
        out_dir=out_dir,
        omega=omega,
        rmse_freq_full=rmse_freq_full,
        rmse_freq_obs=rmse_freq_obs,
        rmse_freq_unobs=rmse_freq_unobs,
        rmse_sample=rmse_sample,
    )

    freq_table = np.stack([omega, rmse_freq_full, rmse_freq_obs, rmse_freq_unobs], axis=1)
    np.savetxt(
        out_dir / "metrics_per_frequency.csv",
        freq_table,
        delimiter=",",
        header="omega,rmse_full,rmse_observed,rmse_unobserved",
        comments="",
    )
    sample_table = np.stack([np.arange(len(rmse_sample)), rmse_sample], axis=1)
    np.savetxt(
        out_dir / "metrics_per_sample.csv",
        sample_table,
        delimiter=",",
        header="sample_idx,rmse",
        comments="",
    )

    flat_case = case_mse.reshape(-1)
    worst_ids = np.argsort(flat_case)[::-1]
    vis_cases = []
    if args.sample_idx >= 0 and args.freq_idx >= 0:
        vis_cases.append((args.sample_idx, args.freq_idx))
    for idx in worst_ids:
        s_idx = int(idx // case_mse.shape[1])
        f_idx = int(idx % case_mse.shape[1])
        pair = (s_idx, f_idx)
        if pair not in vis_cases:
            vis_cases.append(pair)
        if len(vis_cases) >= args.num_visualize:
            break

    with h5py.File(args.data_h5, "r") as f:
        data_ds = f["data"]
        for rank, (s_idx, f_idx) in enumerate(vis_cases, start=1):
            gt = data_ds[s_idx, f_idx].astype(np.float32)
            pred_case = _decode_batch_conditional(
                cores_re=cores_re_flat[s_idx : s_idx + 1, f_idx : f_idx + 1, :],
                cores_im=cores_im_flat[s_idx : s_idx + 1, f_idx : f_idx + 1, :],
                phi_by_m=phi_by_m_np[f_idx : f_idx + 1],
                H=gt.shape[0],
                W=gt.shape[1],
            )[0, 0]
            if args.denormalize and ckpt_info["data_scale"] != 1.0:
                scale = float(ckpt_info["data_scale"])
                gt = gt * scale
                pred_case = pred_case * scale
            out_file = out_dir / f"compare_rank{rank}_sample{s_idx:03d}_freq{f_idx:03d}.png"
            _plot_compare_case(
                out_file=out_file,
                gt=gt,
                pred=pred_case,
                omega_value=float(omega[f_idx]),
                sample_idx=s_idx,
                freq_idx=f_idx,
            )

    summary = {
        "checkpoint": str(args.ckpt),
        "data_h5": str(args.data_h5),
        "evaluated_samples": int(B_eval),
        "global_rmse": global_rmse,
        "rmse_freq_mean": float(np.mean(rmse_freq_full)),
        "rmse_freq_std": float(np.std(rmse_freq_full)),
        "best_freq_idx": int(np.argmin(rmse_freq_full)),
        "worst_freq_idx": int(np.argmax(rmse_freq_full)),
        "best_freq_omega": float(omega[np.argmin(rmse_freq_full)]),
        "worst_freq_omega": float(omega[np.argmax(rmse_freq_full)]),
        "data_scale_used": float(ckpt_info["data_scale"] if args.denormalize else 1.0),
        "output_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    print("\nEvaluation finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved figures and metrics in: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate omega-conditioned FTM checkpoint and visualize reconstruction quality")
    p.add_argument("--ckpt", type=str, default="ckp/ftm_omega_checkpoint.pt")
    p.add_argument("--data_h5", type=str, default="helmholtz_dataset_42.h5")
    p.add_argument("--out_dir", type=str, default="visual_data/ftm_omega_eval")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_eval_samples", type=int, default=0)
    p.add_argument("--num_visualize", type=int, default=12)
    p.add_argument("--sample_idx", type=int, default=-1)
    p.add_argument("--freq_idx", type=int, default=-1)
    p.add_argument("--denormalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", type=str, default="auto")
    return p


def main() -> None:
    args = build_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
