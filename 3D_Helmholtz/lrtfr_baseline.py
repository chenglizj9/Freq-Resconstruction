"""
lrtfr_baseline.py  (3D Helmholtz edition)
------------------------------------------
Low-Rank Tensor Factor Regression baseline for 3D Helmholtz field reconstruction.

Given a pre-trained 3D FTM basis (net_x, net_y, net_z), solve per-(sample, freq)
least-squares from sparse observations — no diffusion prior.

    min_g  || phi_3d[obs_idx] @ g - y_obs ||^2    (for each channel re/im)
    pred   = phi_3d @ g_opt  →  (Nx, Ny, Nz)

This is the deterministic ablation baseline that shows the value of the
diffusion prior over pure basis fitting.

Usage
-----
    python lrtfr_baseline.py \
        --ftm_ckpt ckp/ftm3d.pt \
        --test_h5  helmholtz3d_dataset.h5 \
        --out_dir  visual_data/lrtfr3d_eval
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
import numpy as np
import torch

from train_FTM_GPU import MLP1D, build_phi_3d, normalize_coords_to_unit

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    return np.asarray(x)


def _select_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _rel_rmse(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.sum((pred - gt) ** 2) / max(np.sum(gt ** 2), eps)))


def _masked_rel_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                     eps: float = 1e-12) -> float:
    m = mask.astype(bool)
    if not np.any(m): return float("nan")
    return float(np.sqrt(np.sum((pred - gt) ** 2 * m) / max(np.sum(gt ** 2 * m), eps)))


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_ftm_basis(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"FTM checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt.get("config", {})
    for k in ("rank_x", "rank_y", "rank_z", "hidden_dim", "hidden_layers", "activation"):
        if k not in cfg:
            raise KeyError(f"FTM config missing key: '{k}'")
    if ckpt.get("spatial_dims", 2) != 3:
        raise ValueError("Expected a 3D FTM checkpoint (spatial_dims=3).")

    def _net(rank: int) -> MLP1D:
        net = MLP1D(int(rank), int(cfg["hidden_dim"]), int(cfg["hidden_layers"]),
                    str(cfg["activation"])).to(device)
        net.eval()
        return net

    net_x = _net(cfg["rank_x"]); net_x.load_state_dict(ckpt["net_x_state"])
    net_y = _net(cfg["rank_y"]); net_y.load_state_dict(ckpt["net_y_state"])
    net_z = _net(cfg["rank_z"]); net_z.load_state_dict(ckpt["net_z_state"])

    return {
        "net_x": net_x, "net_y": net_y, "net_z": net_z,
        "normalize_coords": bool(cfg.get("normalize_coords", True)),
        "rank_x": int(cfg["rank_x"]), "rank_y": int(cfg["rank_y"]),
        "rank_z": int(cfg["rank_z"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Least-squares reconstruction per (sample, freq)
# ─────────────────────────────────────────────────────────────────────────────

def _lstsq_reconstruct(
    phi_np: np.ndarray,    # (P, R)  full feature matrix
    gt:     np.ndarray,    # (Nx, Ny, Nz, 2)
    obs_re: np.ndarray,    # bool (P,)
    obs_im: np.ndarray,    # bool (P,)
    rcond: Optional[float],
    ridge: float,
) -> np.ndarray:
    """Return (Nx, Ny, Nz, 2) prediction."""
    Nx, Ny, Nz, _ = gt.shape
    P = Nx * Ny * Nz

    def _solve(phi_obs, y_obs, ridge):
        if ridge > 0:
            # Ridge regression: (A.T A + ridge I) g = A.T y
            AtA = phi_obs.T @ phi_obs + ridge * np.eye(phi_obs.shape[1], dtype=np.float32)
            Aty = phi_obs.T @ y_obs
            return np.linalg.solve(AtA, Aty).astype(np.float32)
        g, *_ = np.linalg.lstsq(phi_obs, y_obs, rcond=rcond)
        return g.astype(np.float32)

    y_re_flat = gt[..., 0].reshape(P)
    y_im_flat = gt[..., 1].reshape(P)

    phi_re = phi_np[obs_re]   # (n_obs_re, R)
    phi_im = phi_np[obs_im]

    g_re = _solve(phi_re, y_re_flat[obs_re], ridge)
    g_im = _solve(phi_im, y_im_flat[obs_im], ridge)

    pred_re = (phi_np @ g_re).reshape(Nx, Ny, Nz)
    pred_im = (phi_np @ g_im).reshape(Nx, Ny, Nz)
    return np.stack([pred_re, pred_im], axis=-1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def _plot_midplane(out_file: Path, gt: np.ndarray, pred: np.ndarray,
                   omega_val: float, sample_idx: int, freq_idx: int) -> None:
    iz = gt.shape[2] // 2
    gt_s, pred_s = gt[:, :, iz, :], pred[:, :, iz, :]
    gt_amp  = np.sqrt(gt_s[..., 0] ** 2 + gt_s[..., 1] ** 2)
    pd_amp  = np.sqrt(pred_s[..., 0] ** 2 + pred_s[..., 1] ** 2)

    items = [
        (gt_s[..., 0],        "GT Real",   "viridis"),
        (pred_s[..., 0],      "Pred Real", "viridis"),
        (np.abs(pred_s[..., 0] - gt_s[..., 0]), "Err Real", "magma"),
        (gt_s[..., 1],        "GT Imag",   "viridis"),
        (pred_s[..., 1],      "Pred Imag", "viridis"),
        (np.abs(pred_s[..., 1] - gt_s[..., 1]), "Err Imag", "magma"),
        (gt_amp,              "GT Amp",    "viridis"),
        (pd_amp,              "Pred Amp",  "viridis"),
        (np.abs(pd_amp - gt_amp),          "Err Amp",  "magma"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, (img, ttl, cmap) in zip(axes.flat, items):
        im = ax.imshow(img, origin="lower", cmap=cmap)
        ax.set_title(ttl, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"LRTFR  s={sample_idx}  f={freq_idx}  ω={omega_val:.4f}  (mid-z slice)")
    fig.tight_layout(); fig.savefig(out_file, dpi=150); plt.close(fig)


def _plot_freq_curve(out_path: Path, omega: np.ndarray,
                     rmse_freq: np.ndarray, obs_freq: np.ndarray,
                     unobs_freq: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(omega, rmse_freq,   label="Full",     lw=2.0)
    ax.plot(omega, obs_freq,    label="Observed", lw=1.5, ls="--")
    ax.plot(omega, unobs_freq,  label="Unobserved", lw=1.5, ls=":")
    ax.set_xlabel("ω"); ax.set_ylabel("Relative RMSE")
    ax.set_title("LRTFR Frequency-wise Error (3D Helmholtz)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device  = _select_device(args.device)

    ftm = _load_ftm_basis(Path(args.ftm_ckpt), device)

    with h5py.File(args.test_h5, "r") as f:
        for k in ("data", "mask_tr", "omega"):
            if k not in f: raise KeyError(f"HDF5 missing '{k}'")

        data_ds = f["data"]
        mask_ds = f["mask_tr"]
        omega   = f["omega"][...].astype(np.float32)

        B_data, M_data, Nx, Ny, Nz, C = data_ds.shape
        assert C == 2, "Only C=2 (real+imag) supported."

        B_eval = B_data if args.max_eval_samples <= 0 else min(B_data, args.max_eval_samples)
        P = Nx * Ny * Nz

        x_grid = (f["grid_x"][...].astype(np.float32) if "grid_x" in f
                   else np.linspace(0, 1, Nx, dtype=np.float32))
        y_grid = (f["grid_y"][...].astype(np.float32) if "grid_y" in f
                   else np.linspace(0, 1, Ny, dtype=np.float32))
        z_grid = (f["grid_z"][...].astype(np.float32) if "grid_z" in f
                   else np.linspace(0, 1, Nz, dtype=np.float32))

        if ftm["normalize_coords"]:
            x_np = normalize_coords_to_unit(x_grid.astype(np.float64)).astype(np.float32)
            y_np = normalize_coords_to_unit(y_grid.astype(np.float64)).astype(np.float32)
            z_np = normalize_coords_to_unit(z_grid.astype(np.float64)).astype(np.float32)
        else:
            x_np, y_np, z_np = x_grid, y_grid, z_grid

        x_t = torch.from_numpy(x_np).unsqueeze(-1).to(device)
        y_t = torch.from_numpy(y_np).unsqueeze(-1).to(device)
        z_t = torch.from_numpy(z_np).unsqueeze(-1).to(device)

        with torch.no_grad():
            phi_t = build_phi_3d(ftm["net_x"], ftm["net_y"], ftm["net_z"], x_t, y_t, z_t)
        phi_np = phi_t.cpu().numpy().astype(np.float32)   # (P, R)

        # Accumulators
        sum_e2     = np.zeros(M_data, dtype=np.float64)
        sum_g2     = np.zeros(M_data, dtype=np.float64)
        sum_e2_obs = np.zeros(M_data, dtype=np.float64)
        sum_g2_obs = np.zeros(M_data, dtype=np.float64)
        sum_e2_un  = np.zeros(M_data, dtype=np.float64)
        sum_g2_un  = np.zeros(M_data, dtype=np.float64)
        rmse_sample = np.zeros(B_eval, dtype=np.float64)
        case_rel    = np.zeros((B_eval, M_data), dtype=np.float64)

        rows: List[Dict] = []
        vis_count = 0

        for b_idx in range(B_eval):
            for m_idx in range(M_data):
                gt        = data_ds[b_idx, m_idx].astype(np.float32)   # (Nx, Ny, Nz, 2)
                omega_val = float(omega[m_idx])

                if mask_ds.ndim == 5:
                    mask_bm = (mask_ds[m_idx].astype(np.float32) > 0.5)
                elif mask_ds.ndim == 6:
                    mask_bm = (mask_ds[b_idx, m_idx].astype(np.float32) > 0.5)
                else:
                    raise ValueError(f"mask_tr must be 5-D or 6-D, got {mask_ds.ndim}")

                obs_re = mask_bm[..., 0].reshape(P).astype(bool)
                obs_im = mask_bm[..., 1].reshape(P).astype(bool)

                if obs_re.sum() < phi_np.shape[1] or obs_im.sum() < phi_np.shape[1]:
                    # Under-determined: skip or use ridge
                    if args.ridge <= 0 and obs_re.sum() < phi_np.shape[1]:
                        print(f"  [WARN] b={b_idx} m={m_idx}: n_obs={obs_re.sum()} < R={phi_np.shape[1]}, "
                              f"using ridge=1e-3")
                        ridge = 1e-3
                    else:
                        ridge = args.ridge
                else:
                    ridge = args.ridge

                pred = _lstsq_reconstruct(phi_np, gt, obs_re, obs_im, args.rcond, ridge)

                rmse = _rel_rmse(pred, gt)
                obs_rmse  = _masked_rel_rmse(pred, gt, mask_bm)
                unobs_mask = 1.0 - mask_bm.astype(np.float32)
                unobs_rmse = _masked_rel_rmse(pred, gt, unobs_mask)

                sum_e2[m_idx]     += np.sum((pred - gt) ** 2)
                sum_g2[m_idx]     += np.sum(gt ** 2)
                sum_e2_obs[m_idx] += np.sum((pred - gt) ** 2 * mask_bm)
                sum_g2_obs[m_idx] += np.sum(gt ** 2 * mask_bm)
                sum_e2_un[m_idx]  += np.sum((pred - gt) ** 2 * unobs_mask)
                sum_g2_un[m_idx]  += np.sum(gt ** 2 * unobs_mask)

                case_rel[b_idx, m_idx] = rmse
                rows.append({"b": b_idx, "m": m_idx, "omega": omega_val,
                              "rmse": rmse, "obs_rmse": obs_rmse, "unobs_rmse": unobs_rmse})

                if vis_count < args.num_visualize:
                    vis_count += 1
                    _plot_midplane(out_dir / f"compare_{vis_count:03d}_s{b_idx:03d}_f{m_idx:03d}.png",
                                   gt, pred, omega_val, b_idx, m_idx)

            rmse_sample[b_idx] = case_rel[b_idx].mean()
            print(f"  Sample {b_idx+1}/{B_eval}  mean_rmse={rmse_sample[b_idx]:.4e}")

    rmse_freq       = np.sqrt(sum_e2     / np.maximum(sum_g2,     1e-30)) / B_eval
    rmse_freq_obs   = np.sqrt(sum_e2_obs / np.maximum(sum_g2_obs, 1e-30)) / B_eval
    rmse_freq_unobs = np.sqrt(sum_e2_un  / np.maximum(sum_g2_un,  1e-30)) / B_eval

    _plot_freq_curve(out_dir / "freq_rmse_curve.png", omega,
                     rmse_freq, rmse_freq_obs, rmse_freq_unobs)

    np.savetxt(out_dir / "metrics_per_frequency.csv",
               np.column_stack([omega, rmse_freq, rmse_freq_obs, rmse_freq_unobs]),
               delimiter=",", header="omega,rmse_full,rmse_obs,rmse_unobs", comments="")
    np.savetxt(out_dir / "metrics_per_sample.csv",
               np.column_stack([np.arange(B_eval), rmse_sample]),
               delimiter=",", header="sample_idx,rmse_mean", comments="")

    summary = {
        "method": "LRTFR_3D", "ftm_ckpt": str(args.ftm_ckpt), "test_h5": str(args.test_h5),
        "evaluated_samples": int(B_eval),
        "mean_rmse": float(np.mean(rmse_sample)),
        "mean_rmse_obs": float(np.nanmean([r["obs_rmse"] for r in rows])),
        "mean_rmse_unobs": float(np.nanmean([r["unobs_rmse"] for r in rows])),
        "output_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print("\nLRTFR 3D Evaluation finished.")
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LRTFR baseline for 3D Helmholtz")
    p.add_argument("--ftm_ckpt",          type=str,   default="ckp/ftm3d.pt")
    p.add_argument("--test_h5",           type=str,   default="helmholtz3d_dataset.h5")
    p.add_argument("--out_dir",           type=str,   default="visual_data/lrtfr3d_eval")
    p.add_argument("--max_eval_samples",  type=int,   default=0)
    p.add_argument("--num_visualize",     type=int,   default=10)
    p.add_argument("--rcond",             type=float, default=None)
    p.add_argument("--ridge",             type=float, default=0.0,
                   help="Ridge coefficient for LS (use >0 for sparse observations).")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
