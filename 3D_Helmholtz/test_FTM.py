"""
test_FTM.py  (3D Helmholtz edition)
-------------------------------------
Evaluate a trained 3D FTM checkpoint on C=2 (real+imag) 3D Helmholtz data.

Outputs
-------
1) Per-frequency RMSE curves (full / observed / unobserved)
2) Per-sample RMSE
3) 3D full-volume comparison plots (worst-case + random)
4) Metric CSV tables and summary JSON

Usage
-----
    python test_FTM.py --ckpt ckp/ftm3d.pt --data_h5 helmholtz3d_dataset.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
import numpy as np
import torch

from train_FTM_GPU import MLP1D, build_phi_3d, normalize_coords_to_unit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vis3d import save_3d_visuals


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_rel(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float64)
    ok  = den > 0
    out[ok] = np.sqrt(num[ok] / den[ok])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_checkpoint(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt.get("config", {})
    for k in ("rank_x", "rank_y", "rank_z", "hidden_dim", "hidden_layers", "activation"):
        if k not in cfg:
            raise KeyError(f"Checkpoint config missing key: '{k}'")

    if ckpt.get("spatial_dims", 2) != 3:
        raise ValueError("This script expects a 3D FTM checkpoint (spatial_dims=3).")

    def _net(rank: int) -> MLP1D:
        net = MLP1D(int(rank), int(cfg["hidden_dim"]), int(cfg["hidden_layers"]),
                    str(cfg["activation"])).to(device)
        net.eval()
        return net

    net_x = _net(cfg["rank_x"]); net_x.load_state_dict(ckpt["net_x_state"])
    net_y = _net(cfg["rank_y"]); net_y.load_state_dict(ckpt["net_y_state"])
    net_z = _net(cfg["rank_z"]); net_z.load_state_dict(ckpt["net_z_state"])

    cores_raw = ckpt["cores"]   # list[C] of tensors (B, M, Rx, Ry, Rz)
    cores     = [_to_numpy(c).astype(np.float32) for c in cores_raw]
    channel_names = list(ckpt.get("channel_names", [f"ch{i}" for i in range(len(cores))]))
    C = len(cores)

    B, M, Rx, Ry, Rz = cores[0].shape
    R = Rx * Ry * Rz
    cores_flat = [c.reshape(B, M, R) for c in cores]

    omega_ckpt  = _to_numpy(ckpt.get("omega", np.array([], dtype=np.float32))).astype(np.float32)
    data_scale  = float(ckpt.get("data_scale", 1.0))

    return {
        "cfg": cfg, "net_x": net_x, "net_y": net_y, "net_z": net_z,
        "cores": cores, "cores_flat": cores_flat,
        "channel_names": channel_names, "C": C,
        "B": B, "M": M, "Rx": Rx, "Ry": Ry, "Rz": Rz, "R": R,
        "omega_ckpt": omega_ckpt, "data_scale": data_scale,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Decoding
# ─────────────────────────────────────────────────────────────────────────────

def _build_phi_3d(net_x, net_y, net_z, x_t, y_t, z_t, device) -> np.ndarray:
    with torch.no_grad():
        phi = build_phi_3d(net_x, net_y, net_z, x_t, y_t, z_t)
    return phi.cpu().numpy().astype(np.float32)


def _decode(cores_batch: List[np.ndarray], phi: np.ndarray,
            Nx: int, Ny: int, Nz: int) -> np.ndarray:
    """
    cores_batch: list[C] of (bsz, M, R)
    phi: (P, R), P = Nx*Ny*Nz
    Returns: (bsz, M, Nx, Ny, Nz, C)
    """
    bsz = cores_batch[0].shape[0]
    M   = cores_batch[0].shape[1]
    C   = len(cores_batch)
    out = np.zeros((bsz, M, Nx, Ny, Nz, C), dtype=np.float32)
    for c, core in enumerate(cores_batch):
        flat = np.einsum("bmr,pr->bmp", core, phi, optimize=True)   # (bsz, M, P)
        out[..., c] = flat.reshape(bsz, M, Nx, Ny, Nz)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────



def _plot_error_curves(
    out_dir: Path,
    omega: np.ndarray,
    rmse_freq: np.ndarray,      # (M, C)
    rmse_freq_obs: np.ndarray,
    rmse_freq_unobs: np.ndarray,
    rmse_sample: np.ndarray,    # (B,)
    channel_names: List[str],
) -> None:
    C = len(channel_names)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(C, 2)))
    if rmse_freq.ndim == 1: rmse_freq = rmse_freq[:, None]
    if rmse_freq_obs.ndim == 1: rmse_freq_obs = rmse_freq_obs[:, None]
    if rmse_freq_unobs.ndim == 1: rmse_freq_unobs = rmse_freq_unobs[:, None]

    for c, cname in enumerate(channel_names):
        col = colors[c]
        ax.plot(omega, rmse_freq[:, c], lw=2.0, color=col, label=f"{cname} full")
        if np.isfinite(rmse_freq_obs[:, c]).any():
            ax.plot(omega, rmse_freq_obs[:, c], lw=1.5, color=col, ls="--", label=f"{cname} obs")
        if np.isfinite(rmse_freq_unobs[:, c]).any():
            ax.plot(omega, rmse_freq_unobs[:, c], lw=1.5, color=col, ls=":", label=f"{cname} unobs")
    ax.set_title("Relative RMSE vs Frequency (per channel)")
    ax.set_xlabel("ω"); ax.set_ylabel("Relative RMSE")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    axes[1].plot(np.arange(len(rmse_sample)), rmse_sample, lw=1.8)
    axes[1].set_title("Per-sample Mean Relative RMSE")
    axes[1].set_xlabel("sample index"); axes[1].set_ylabel("Relative RMSE")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "error_curves.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu"
                            if args.device == "auto" else args.device)

    info = _load_checkpoint(Path(args.ckpt), device)
    cfg          = info["cfg"]
    net_x, net_y, net_z = info["net_x"], info["net_y"], info["net_z"]
    cores_flat   = info["cores_flat"]    # list[C] of (B_ckpt, M_ckpt, R)
    channel_names= info["channel_names"]
    C_ckpt       = info["C"]
    B_ckpt       = info["B"]
    M_ckpt       = info["M"]
    R            = info["R"]
    data_scale   = info["data_scale"]

    with h5py.File(args.data_h5, "r") as f:
        for k in ("data", "mask_tr", "omega"):
            if k not in f: raise KeyError(f"HDF5 missing '{k}'")

        data_ds = f["data"]
        mask_ds = f["mask_tr"]
        omega   = f["omega"][...].astype(np.float32)

        B_data, M_data, Nx, Ny, Nz, C_data = data_ds.shape
        if C_data != C_ckpt:
            raise ValueError(f"Channel mismatch: data C={C_data}, ckpt C={C_ckpt}")
        if M_data != M_ckpt:
            raise ValueError(f"Frequency mismatch: data M={M_data}, ckpt M={M_ckpt}")

        B_eval = min(B_data, B_ckpt)
        if args.max_eval_samples > 0:
            B_eval = min(B_eval, args.max_eval_samples)

        x_grid = (f["grid_x"][...].astype(np.float32) if "grid_x" in f
                   else np.linspace(0, 1, Nx, dtype=np.float32))
        y_grid = (f["grid_y"][...].astype(np.float32) if "grid_y" in f
                   else np.linspace(0, 1, Ny, dtype=np.float32))
        z_grid = (f["grid_z"][...].astype(np.float32) if "grid_z" in f
                   else np.linspace(0, 1, Nz, dtype=np.float32))

        use_norm = bool(cfg.get("normalize_coords", True))
        def _coord(arr):
            arr = arr.astype(np.float64)
            return normalize_coords_to_unit(arr).astype(np.float32) if use_norm else arr.astype(np.float32)

        x_t = torch.from_numpy(_coord(x_grid)).unsqueeze(-1).to(device)
        y_t = torch.from_numpy(_coord(y_grid)).unsqueeze(-1).to(device)
        z_t = torch.from_numpy(_coord(z_grid)).unsqueeze(-1).to(device)

        phi_np = _build_phi_3d(net_x, net_y, net_z, x_t, y_t, z_t, device)   # (P, R)
        P = Nx * Ny * Nz
        if phi_np.shape[1] != R:
            raise ValueError(f"Rank mismatch: phi has R={phi_np.shape[1]}, expected R={R}")

        # ── Mask handling ─────────────────────────────────────────────────
        # mask_ds: (M, Nx, Ny, Nz, C) or (B, M, Nx, Ny, Nz, C)
        if mask_ds.ndim == 5:
            shared_mask = (mask_ds[...].astype(np.float32) > 0.5)   # (M, Nx, Ny, Nz, C)
            per_sample_mask = False
        elif mask_ds.ndim == 6:
            shared_mask = None
            per_sample_mask = True
        else:
            raise ValueError(f"mask_tr must be 5-D or 6-D, got {mask_ds.ndim}")

        # ── Metric accumulators ───────────────────────────────────────────
        sum_e2   = np.zeros((M_data, C_data), dtype=np.float64)
        sum_g2   = np.zeros((M_data, C_data), dtype=np.float64)
        sum_e2o  = np.zeros((M_data, C_data), dtype=np.float64)
        sum_g2o  = np.zeros((M_data, C_data), dtype=np.float64)
        sum_e2u  = np.zeros((M_data, C_data), dtype=np.float64)
        sum_g2u  = np.zeros((M_data, C_data), dtype=np.float64)
        rmse_sample = np.zeros(B_eval, dtype=np.float64)
        case_rel    = np.zeros((B_eval, M_data), dtype=np.float64)

        ds_use = info["data_scale"] if args.denormalize else 1.0

        for s in range(0, B_eval, args.batch_size):
            e   = min(B_eval, s + args.batch_size)
            bsz = e - s

            cores_b = [cf[s:e] for cf in cores_flat]
            pred    = _decode(cores_b, phi_np, Nx, Ny, Nz)   # (bsz, M, Nx, Ny, Nz, C)
            gt      = data_ds[s:e].astype(np.float32)

            if ds_use != 1.0:
                pred *= ds_use; gt *= ds_use

            err2 = (pred - gt) ** 2
            gt2  = gt ** 2
            err2_f = err2.reshape(bsz, M_data, Nx * Ny * Nz, C_data)
            gt2_f  = gt2.reshape(bsz, M_data, Nx * Ny * Nz, C_data)

            sum_e2 += np.sum(err2_f, axis=(0, 2))
            sum_g2 += np.sum(gt2_f,  axis=(0, 2))

            case_e  = np.sum(err2_f, axis=2)
            case_g  = np.sum(gt2_f,  axis=2)
            case_rel_c = np.sqrt(case_e / np.maximum(case_g, 1e-30))
            case_rel[s:e] = case_rel_c.mean(axis=2)
            rmse_sample[s:e] = case_rel[s:e].mean(axis=1)

            if shared_mask is not None:
                obs   = shared_mask.reshape(M_data, Nx * Ny * Nz, C_data).astype(np.float32)
                unobs = 1.0 - obs
                sum_e2o += np.sum(err2_f * obs[None],   axis=(0, 2))
                sum_g2o += np.sum(gt2_f  * obs[None],   axis=(0, 2))
                sum_e2u += np.sum(err2_f * unobs[None], axis=(0, 2))
                sum_g2u += np.sum(gt2_f  * unobs[None], axis=(0, 2))
            else:
                mask_b = (mask_ds[s:e].astype(np.float32) > 0.5)
                obs    = mask_b.reshape(bsz, M_data, Nx * Ny * Nz, C_data).astype(np.float32)
                unobs  = 1.0 - obs
                sum_e2o += np.sum(err2_f * obs,   axis=(0, 2))
                sum_g2o += np.sum(gt2_f  * obs,   axis=(0, 2))
                sum_e2u += np.sum(err2_f * unobs, axis=(0, 2))
                sum_g2u += np.sum(gt2_f  * unobs, axis=(0, 2))

            print(f"  Eval {e}/{B_eval} …")

        rmse_freq       = _safe_rel(sum_e2,  sum_g2)
        rmse_freq_obs   = _safe_rel(sum_e2o, sum_g2o)
        rmse_freq_unobs = _safe_rel(sum_e2u, sum_g2u)

        _plot_error_curves(out_dir, omega, rmse_freq, rmse_freq_obs, rmse_freq_unobs,
                           rmse_sample, channel_names)

        # ── CSV ───────────────────────────────────────────────────────────
        col_hdr = ["omega"] + [
            f"{cn}_{sfx}" for cn in channel_names for sfx in ("full", "obs", "unobs")
        ]
        freq_cols = [omega]
        for c in range(C_data):
            freq_cols += [rmse_freq[:, c], rmse_freq_obs[:, c], rmse_freq_unobs[:, c]]
        np.savetxt(out_dir / "metrics_per_frequency.csv",
                   np.column_stack(freq_cols), delimiter=",",
                   header=",".join(col_hdr), comments="")
        np.savetxt(out_dir / "metrics_per_sample.csv",
                   np.column_stack([np.arange(B_eval), rmse_sample]),
                   delimiter=",", header="sample_idx,rmse_mean", comments="")

        # ── Visualize worst cases ─────────────────────────────────────────
        vis_cases: List[Tuple[int, int]] = []
        if 0 <= args.sample_idx < B_eval and 0 <= args.freq_idx < M_data:
            vis_cases.append((args.sample_idx, args.freq_idx))

        flat_case = case_rel.reshape(-1)
        for idx in np.argsort(flat_case)[::-1]:
            fi = int(idx // B_eval)
            si = int(idx %  B_eval)
            if (si, fi) not in vis_cases:
                vis_cases.append((si, fi))
            if len(vis_cases) >= args.num_visualize:
                break

        for rank, (si, fi) in enumerate(vis_cases, start=1):
            gt_frame   = data_ds[si, fi].astype(np.float32)
            if ds_use != 1.0: gt_frame = gt_frame * ds_use
            cores_one  = [cf[si:si+1, fi:fi+1, :] for cf in cores_flat]
            pred_frame = _decode(cores_one, phi_np, Nx, Ny, Nz)[0, 0]
            if ds_use != 1.0: pred_frame = pred_frame * ds_use
            obs_mask = mask_ds[fi] if shared_mask is not None else mask_ds[si, fi]
            obs_mask = (obs_mask.astype(np.float32) > 0.5)
            if obs_mask.ndim == 4:
                obs_mask = obs_mask[..., 0]
            save_3d_visuals(
                gt_frame, pred_frame, obs_mask,
                stem=out_dir / f"compare_rank{rank:02d}_s{si:03d}_f{fi:03d}",
                title=f"3D FTM  sample={si}  freq_idx={fi}  ω={float(omega[fi]):.4f}",
                channel_names=channel_names,
            )

    global_rmse = float(np.mean(rmse_sample))
    rmse_per_ch = {
        cn: {"freq_mean": float(np.nanmean(rmse_freq[:, c])),
             "obs_mean":  float(np.nanmean(rmse_freq_obs[:, c])),
             "unobs_mean":float(np.nanmean(rmse_freq_unobs[:, c]))}
        for c, cn in enumerate(channel_names)
    }

    summary = {
        "checkpoint": str(args.ckpt), "data_h5": str(args.data_h5),
        "evaluated_samples": int(B_eval),
        "global_rmse": global_rmse, "rmse_per_channel": rmse_per_ch,
        "output_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print("\n" + "─" * 60)
    print("3D FTM Evaluation finished.")
    print(json.dumps(summary, indent=2))
    print(f"Saved to: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate 3D FTM checkpoint (Helmholtz C=2)")
    p.add_argument("--ckpt",             type=str, default="ckp/ftm3d.pt")
    p.add_argument("--data_h5",          type=str, default="helmholtz3d_dataset.h5")
    p.add_argument("--out_dir",          type=str, default="visual_data/ftm3d_eval")
    p.add_argument("--batch_size",       type=int, default=16)
    p.add_argument("--max_eval_samples", type=int, default=0)
    p.add_argument("--num_visualize",    type=int, default=12)
    p.add_argument("--sample_idx",       type=int, default=0)
    p.add_argument("--freq_idx",         type=int, default=2)
    p.add_argument("--denormalize",      action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device",           type=str, default="auto")
    return p


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
