"""
test_FTM.py
-----------
Evaluate a trained frequency-axis FTM checkpoint and visualize reconstruction quality.

Outputs
-------
1) Error curves (frequency RMSE and sample RMSE)
2) Reconstruction-vs-ground-truth comparison figures
3) Metric tables and summary JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import matplotlib
import numpy as np
import torch

from train_FTM import MLP1D, build_phi, normalize_coords_to_unit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _decode_batch(
    cores_re: np.ndarray,  # (B, M, R)
    cores_im: np.ndarray,  # (B, M, R)
    phi_np: np.ndarray,    # (P, R)
    H: int,
    W: int,
) -> np.ndarray:
    pred_re_flat = np.einsum("bmr,pr->bmp", cores_re, phi_np, optimize=True)
    pred_im_flat = np.einsum("bmr,pr->bmp", cores_im, phi_np, optimize=True)
    pred = np.stack([pred_re_flat, pred_im_flat], axis=-1).reshape(cores_re.shape[0], cores_re.shape[1], H, W, 2)
    return pred.astype(np.float32)


def _load_checkpoint(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})

    required_cfg = ["rank_x", "rank_y", "hidden_dim", "hidden_layers", "activation"]
    for k in required_cfg:
        if k not in cfg:
            raise KeyError(f"Checkpoint config missing key: {k}")

    net_x = MLP1D(
        out_dim=int(cfg["rank_x"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_hidden_layers=int(cfg["hidden_layers"]),
        activation=str(cfg["activation"]),
    ).to(device)
    net_y = MLP1D(
        out_dim=int(cfg["rank_y"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_hidden_layers=int(cfg["hidden_layers"]),
        activation=str(cfg["activation"]),
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

    omega_ckpt = _to_numpy(ckpt.get("omega", np.array([], dtype=np.float32))).astype(np.float32)
    data_scale = float(ckpt.get("data_scale", 1.0))

    return {
        "config": cfg,
        "net_x": net_x,
        "net_y": net_y,
        "cores_real": cores_real,
        "cores_imag": cores_imag,
        "omega_ckpt": omega_ckpt,
        "data_scale": data_scale,
    }


def _build_mask_view(mask_ds: h5py.Dataset, B_eval: int) -> Tuple[np.ndarray | None, bool]:
    """
    Returns
    -------
    shared_mask : np.ndarray | None
        If mask is shared across samples, returns bool array (M,H,W,C).
        Otherwise returns None and mask will be read per-batch as (B,M,H,W,C).
    per_sample : bool
        True if mask has sample dimension.
    """
    if mask_ds.ndim == 4:
        return (mask_ds[...].astype(np.float32) > 0.5), False

    if mask_ds.ndim == 5:
        if mask_ds.shape[0] < B_eval:
            raise ValueError(
                f"mask_tr sample dimension {mask_ds.shape[0]} is smaller than eval samples {B_eval}."
            )
        return None, True

    raise ValueError(f"mask_tr must have 4 or 5 dims, got shape {mask_ds.shape}")


def _plot_error_curves(
    out_dir: Path,
    omega: np.ndarray,
    rmse_freq_full: np.ndarray,
    rmse_freq_obs: np.ndarray,
    rmse_freq_unobs: np.ndarray,
    rmse_sample: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(omega, rmse_freq_full, lw=2.0, label="RMSE(full)")

    if np.isfinite(rmse_freq_obs).any():
        ax.plot(omega, rmse_freq_obs, lw=1.8, ls="--", label="RMSE(observed)")
    if np.isfinite(rmse_freq_unobs).any():
        ax.plot(omega, rmse_freq_unobs, lw=1.8, ls=":", label="RMSE(unobserved)")

    ax.set_title("Error Curve vs Frequency")
    ax.set_xlabel("omega")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(np.arange(len(rmse_sample)), rmse_sample, lw=1.8)
    ax.set_title("Per-sample RMSE Curve")
    ax.set_xlabel("sample index")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "error_curves.png", dpi=180)
    plt.close(fig)


def _plot_compare_case(
    out_file: Path,
    gt: np.ndarray,   # (H,W,2)
    pred: np.ndarray, # (H,W,2)
    omega_value: float,
    sample_idx: int,
    freq_idx: int,
) -> None:
    gt_re, gt_im = gt[..., 0], gt[..., 1]
    pd_re, pd_im = pred[..., 0], pred[..., 1]

    gt_amp = np.sqrt(gt_re ** 2 + gt_im ** 2)
    pd_amp = np.sqrt(pd_re ** 2 + pd_im ** 2)

    err_re = np.abs(pd_re - gt_re)
    err_im = np.abs(pd_im - gt_im)
    err_amp = np.abs(pd_amp - gt_amp)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    vmin_re = float(min(np.min(gt_re), np.min(pd_re)))
    vmax_re = float(max(np.max(gt_re), np.max(pd_re)))
    vmin_im = float(min(np.min(gt_im), np.min(pd_im)))
    vmax_im = float(max(np.max(gt_im), np.max(pd_im)))
    vmin_amp = float(min(np.min(gt_amp), np.min(pd_amp)))
    vmax_amp = float(max(np.max(gt_amp), np.max(pd_amp)))

    items = [
        (gt_re, "GT Real", vmin_re, vmax_re),
        (pd_re, "Recon Real", vmin_re, vmax_re),
        (err_re, "Abs Err Real", None, None),
        (gt_im, "GT Imag", vmin_im, vmax_im),
        (pd_im, "Recon Imag", vmin_im, vmax_im),
        (err_im, "Abs Err Imag", None, None),
        (gt_amp, "GT Amplitude", vmin_amp, vmax_amp),
        (pd_amp, "Recon Amplitude", vmin_amp, vmax_amp),
        (err_amp, "Abs Err Amplitude", None, None),
    ]

    for ax, (img, title, vmin, vmax) in zip(axes.flat, items):
        if vmin is None:
            im = ax.imshow(img, cmap="magma", origin="lower")
        else:
            im = ax.imshow(img, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"sample={sample_idx}, freq_idx={freq_idx}, omega={omega_value:.4f}", y=0.99)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

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
        with torch.no_grad():
            phi = build_phi(net_x, net_y, x_t, y_t)
        phi_np = phi.detach().cpu().numpy().astype(np.float32)

        if phi_np.shape[1] != R:
            raise ValueError(f"Rank mismatch: phi has {phi_np.shape[1]}, core has {R}")

        cores_re_flat = cores_real[:B_eval].reshape(B_eval, M_ckpt, R)
        cores_im_flat = cores_imag[:B_eval].reshape(B_eval, M_ckpt, R)

        data_scale = float(ckpt_info["data_scale"]) if args.denormalize else 1.0

        shared_mask, per_sample_mask = _build_mask_view(mask_ds, B_eval)

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
            obs_count_shared = np.sum(obs_shared, axis=1)
            unobs_shared = 1.0 - obs_shared
            unobs_count_shared = np.sum(unobs_shared, axis=1)

        for s in range(0, B_eval, args.batch_size):
            e = min(B_eval, s + args.batch_size)
            bsz = e - s

            pred = _decode_batch(
                cores_re=cores_re_flat[s:e],
                cores_im=cores_im_flat[s:e],
                phi_np=phi_np,
                H=H,
                W=W,
            )

            gt = data_ds[s:e].astype(np.float32)

            if data_scale != 1.0:
                pred = pred * data_scale
                gt = gt * data_scale

            err2 = (pred - gt) ** 2
            err2_flat = err2.reshape(bsz, M_data, -1)
            gt2 = gt ** 2
            gt2_flat = gt2.reshape(bsz, M_data, -1)

            sum_err = np.sum(err2_flat, axis=2)
            sum_gt = np.sum(gt2_flat, axis=2)
            rel_err = np.sqrt(sum_err / sum_gt)

            # mse_freq_sum += np.sum(err2_flat, axis=(0, 2))
            # cnt_freq += float(bsz) * float(H * W * C)
            mse_freq_sum += np.sum(rel_err, axis=0)
            cnt_freq += float(bsz)

            # case_mse[s:e] = np.mean(err2_flat, axis=2)
            # mse_sample[s:e] = np.mean(err2_flat, axis=(1, 2))
            case_mse[s:e] = rel_err
            mse_sample[s:e] = np.mean(rel_err, axis=1)


            if shared_mask is not None:
                # mse_obs_sum += np.sum(err2_flat * obs_shared[None, :, :], axis=(0, 2))
                # cnt_obs += obs_count_shared * float(bsz)

                # mse_unobs_sum += np.sum(err2_flat * unobs_shared[None, :, :], axis=(0, 2))
                # cnt_unobs += unobs_count_shared * float(bsz)
                sum_err_obs = np.sum(err2_flat * obs_shared[None, :, :], axis=2)
                sum_gt_obs = np.sum(gt2_flat * obs_shared[None, :, :], axis=2)
                rel_err_obs = np.sqrt(sum_err_obs / sum_gt_obs )

                sum_err_unobs = np.sum(err2_flat * unobs_shared[None, :, :], axis=2)
                sum_gt_unobs = np.sum(gt2_flat * unobs_shared[None, :, :], axis=2)
                rel_err_unobs = np.sqrt(sum_err_unobs / sum_gt_unobs)

                mse_obs_sum += np.sum(rel_err_obs, axis=0)
                cnt_obs += bsz

                mse_unobs_sum += np.sum(rel_err_unobs, axis=0)
                cnt_unobs += bsz
            else:
                # mask_batch = (mask_ds[s:e].astype(np.float32) > 0.5)
                # obs_flat = mask_batch.reshape(bsz, M_data, -1).astype(np.float32)
                # unobs_flat = 1.0 - obs_flat

                # mse_obs_sum += np.sum(err2_flat * obs_flat, axis=(0, 2))
                # cnt_obs += np.sum(obs_flat, axis=(0, 2))

                # mse_unobs_sum += np.sum(err2_flat * unobs_flat, axis=(0, 2))
                # cnt_unobs += np.sum(unobs_flat, axis=(0, 2))
                mask_batch = (mask_ds[s:e].astype(np.float32) > 0.5)
                obs_flat = mask_batch.reshape(bsz, M_data, -1).astype(np.float32)
                unobs_flat = 1.0 - obs_flat

                # 观测区求和
                sum_err_obs = np.sum(err2_flat * obs_flat, axis=2)    # (bsz, M)
                sum_gt_obs = np.sum(gt2_flat * obs_flat, axis=2)      # (bsz, M)
                rel_err_obs = np.sqrt(sum_err_obs / sum_gt_obs)

                # 非观测区求和
                sum_err_unobs = np.sum(err2_flat * unobs_flat, axis=2)
                sum_gt_unobs = np.sum(gt2_flat * unobs_flat, axis=2)
                rel_err_unobs = np.sqrt(sum_err_unobs / sum_gt_unobs)

                # 累加
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
    global_rmse = np.mean(mse_sample)

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

    vis_cases: List[Tuple[int, int]] = []

    if args.sample_idx >= 0 and args.freq_idx >= 0:
        if not (0 <= args.sample_idx < case_mse.shape[0]):
            raise ValueError(f"sample_idx out of range [0,{case_mse.shape[0]-1}]")
        if not (0 <= args.freq_idx < case_mse.shape[1]):
            raise ValueError(f"freq_idx out of range [0,{case_mse.shape[1]-1}]")
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

            pred_case = _decode_batch(
                cores_re=cores_re_flat[s_idx : s_idx + 1, f_idx : f_idx + 1, :],
                cores_im=cores_im_flat[s_idx : s_idx + 1, f_idx : f_idx + 1, :],
                phi_np=phi_np,
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
    p = argparse.ArgumentParser(description="Evaluate FTM checkpoint and visualize reconstruction quality")

    p.add_argument("--ckpt", type=str, default="heat_data/ckp/ftm_gpu_checkpoint.pt")
    p.add_argument("--data_h5", type=str, default="heat_data/data_for_test/harmonic_heat_dataset_mask10.h5")
    p.add_argument("--out_dir", type=str, default="heat_data/visual_data/ftm_eval")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_eval_samples", type=int, default=0)
    p.add_argument("--num_visualize", type=int, default=12)

    p.add_argument("--sample_idx", type=int, default=-1)
    p.add_argument("--freq_idx", type=int, default=-1)

    p.add_argument("--denormalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", type=str, default="auto")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
