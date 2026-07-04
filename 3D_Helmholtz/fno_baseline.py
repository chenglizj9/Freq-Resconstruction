"""
fno_baseline.py  (3D Helmholtz edition)
-----------------------------------------
3D Fourier Neural Operator baseline for sparse-to-full 3D field reconstruction.

Input channels (7):
  obs_re    : observed real part (masked, 0 elsewhere)
  obs_im    : observed imag part (masked, 0 elsewhere)
  mask      : binary observation mask
  omega_map : ω normalized, broadcast to spatial dims
  x_coord   : x coordinate map
  y_coord   : y coordinate map
  z_coord   : z coordinate map

Output channels (2): pred_re, pred_im

Uses truncated 3D spectral convolution (memory-efficient) followed by
pointwise MLP projection (FNO architecture).

Examples
--------
Train:
    python fno_baseline.py --mode train \
        --train_h5 helmholtz3d_dataset.h5 \
        --out ckp/fno3d.pt

Eval:
    python fno_baseline.py --mode eval \
        --ckpt ckp/fno3d.pt \
        --test_h5 helmholtz3d_dataset.h5
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from vis3d import save_3d_visuals as _save_3d
    _HAS_VIS3D = True
except ImportError:
    _HAS_VIS3D = False


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _rel_rmse(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.sum((pred - gt) ** 2) / max(np.sum(gt ** 2), eps)))


def _masked_rel_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                     eps: float = 1e-12) -> float:
    m = mask.astype(bool)
    if not np.any(m): return float("nan")
    return float(np.sqrt(np.sum((pred - gt) ** 2 * m) / max(np.sum(gt ** 2 * m), eps)))


def _normalize_omega(omega: float, omega_min: float, omega_max: float) -> float:
    return float((omega - omega_min) / max(omega_max - omega_min, 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# 3D Spectral Convolution (truncated FFT modes)
# ─────────────────────────────────────────────────────────────────────────────

class SpectralConv3d(nn.Module):
    """3D Fourier convolution with truncated modes."""
    def __init__(self, in_channels: int, out_channels: int,
                 modes_x: int, modes_y: int, modes_z: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mx, self.my, self.mz = modes_x, modes_y, modes_z

        scale = 1.0 / (in_channels * out_channels)
        # 8 corner blocks in 3D FFT
        self.w = nn.ParameterList([
            nn.Parameter(scale * torch.rand(in_channels, out_channels,
                                            modes_x, modes_y, modes_z, 2))
            for _ in range(8)
        ])

    @staticmethod
    def _complex_mul3d(inp: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # inp: (B, in, mx, my, mz), w: (in, out, mx, my, mz, 2)
        w_c = torch.view_as_complex(w.contiguous())   # (in, out, mx, my, mz)
        return torch.einsum("bixyz,ioxyz->boxyz", inp, w_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, Nx, Ny, Nz)
        B, C, Nx, Ny, Nz = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))   # (B, C, Nx, Ny, Nz//2+1)

        mx, my, mz = self.mx, self.my, self.mz
        out_ft = torch.zeros(B, self.out_channels, Nx, Ny, Nz // 2 + 1,
                              dtype=torch.cfloat, device=x.device)

        blocks = [
            (slice(None, mx),    slice(None, my),    slice(None, mz),    0),
            (slice(None, mx),    slice(None, my),    slice(-mz, None),   1),
            (slice(None, mx),    slice(-my, None),   slice(None, mz),    2),
            (slice(None, mx),    slice(-my, None),   slice(-mz, None),   3),
            (slice(-mx, None),   slice(None, my),    slice(None, mz),    4),
            (slice(-mx, None),   slice(None, my),    slice(-mz, None),   5),
            (slice(-mx, None),   slice(-my, None),   slice(None, mz),    6),
            (slice(-mx, None),   slice(-my, None),   slice(-mz, None),   7),
        ]

        for sx, sy, sz, wi in blocks:
            out_ft[:, :, sx, sy, sz] = self._complex_mul3d(
                x_ft[:, :, sx, sy, sz], self.w[wi])

        return torch.fft.irfftn(out_ft, s=(Nx, Ny, Nz), dim=(-3, -2, -1))


class FNOBlock3d(nn.Module):
    def __init__(self, channels: int, modes_x: int, modes_y: int, modes_z: int):
        super().__init__()
        self.spec = SpectralConv3d(channels, channels, modes_x, modes_y, modes_z)
        self.skip = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.skip(x))


class FNO3d(nn.Module):
    """
    3D FNO for sparse field reconstruction.
    Input:  (B, 7, Nx, Ny, Nz)  [obs_re, obs_im, mask, omega, x, y, z]
    Output: (B, 2, Nx, Ny, Nz)  [pred_re, pred_im]
    """
    def __init__(
        self,
        in_channels:  int = 7,
        out_channels: int = 2,
        width:        int = 32,
        n_layers:     int = 4,
        modes_x:      int = 8,
        modes_y:      int = 8,
        modes_z:      int = 8,
    ):
        super().__init__()
        self.in_proj = nn.Conv3d(in_channels, width, kernel_size=1)
        self.blocks  = nn.Sequential(
            *[FNOBlock3d(width, modes_x, modes_y, modes_z) for _ in range(n_layers)]
        )
        self.out_proj = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=1), nn.GELU(),
            nn.Conv3d(width, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.blocks(h)
        return self.out_proj(h)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class Helmholtz3dDataset(Dataset):
    def __init__(self, h5_path: str, max_samples: int = 0, freq_indices: List[int] = None):
        self.h5_path = h5_path
        self.cases: List[Tuple[int, int]] = []

        with h5py.File(h5_path, "r") as f:
            data_ds = f["data"]
            B, M, Nx, Ny, Nz, C = data_ds.shape
            omega = f["omega"][...].astype(np.float32)
            self.omega_min = float(omega.min())
            self.omega_max = float(omega.max())
            self.omega     = omega

            x_g = (f["grid_x"][...].astype(np.float32) if "grid_x" in f
                    else np.linspace(0, 1, Nx, dtype=np.float32))
            y_g = (f["grid_y"][...].astype(np.float32) if "grid_y" in f
                    else np.linspace(0, 1, Ny, dtype=np.float32))
            z_g = (f["grid_z"][...].astype(np.float32) if "grid_z" in f
                    else np.linspace(0, 1, Nz, dtype=np.float32))

            # Normalize coords to [0, 1]
            def _n(a):
                lo, hi = a.min(), a.max()
                return (a - lo) / max(hi - lo, 1e-12)
            X, Y, Z = np.meshgrid(_n(x_g), _n(y_g), _n(z_g), indexing="ij")
            self.coord_x = X.astype(np.float32)   # (Nx, Ny, Nz)
            self.coord_y = Y.astype(np.float32)
            self.coord_z = Z.astype(np.float32)

            B_use  = B if max_samples <= 0 else min(B, max_samples)
            M_use  = list(range(M)) if freq_indices is None else freq_indices
            for b in range(B_use):
                for m in M_use:
                    self.cases.append((b, m))

            self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
            self.M = M

    def __len__(self) -> int: return len(self.cases)

    def __getitem__(self, idx: int):
        b, m = self.cases[idx]
        with h5py.File(self.h5_path, "r") as f:
            field = f["data"][b, m].astype(np.float32)       # (Nx, Ny, Nz, 2)
            mask_ds = f["mask_tr"]
            if mask_ds.ndim == 5:
                mask = (mask_ds[m].astype(np.float32) > 0.5).astype(np.float32)
            elif mask_ds.ndim == 6:
                mask = (mask_ds[b, m].astype(np.float32) > 0.5).astype(np.float32)
            else:
                raise ValueError(f"Unexpected mask dims: {mask_ds.ndim}")
            omega_val = float(f["omega"][m])

        obs_re = (field[..., 0] * mask[..., 0])    # (Nx, Ny, Nz)
        obs_im = (field[..., 1] * mask[..., 1])
        obs_mask = mask[..., 0]                     # shared channel 0 mask for input

        omega_norm = _normalize_omega(omega_val, self.omega_min, self.omega_max)
        omega_map  = np.full((self.Nx, self.Ny, self.Nz), omega_norm, dtype=np.float32)

        # Build input tensor (7, Nx, Ny, Nz)
        inp = np.stack([obs_re, obs_im, obs_mask, omega_map,
                        self.coord_x, self.coord_y, self.coord_z], axis=0)
        target = np.stack([field[..., 0], field[..., 1]], axis=0)  # (2, Nx, Ny, Nz)

        return torch.from_numpy(inp), torch.from_numpy(target), torch.from_numpy(mask)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _parse_freq_indices(text: str, M: int) -> List[int]:
    if not text.strip(): return list(range(M))
    out = []
    for p in text.split(","):
        p = p.strip()
        if p: out.append(int(p))
    return sorted(set(out))


def train_mode(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"
                          if args.device == "auto" else args.device)

    freq_ids = None  # use all training freqs
    dataset  = Helmholtz3dDataset(args.train_h5, max_samples=args.max_train_samples)
    loader   = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                          drop_last=True)

    model = FNO3d(
        in_channels=7, out_channels=2,
        width=args.width, n_layers=args.n_layers,
        modes_x=args.modes, modes_y=args.modes, modes_z=args.modes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.9, min_lr=1e-6)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf"); loss_hist: List[float] = []

    print("\n" + "─" * 60)
    print(f"3D FNO Training  width={args.width}  modes={args.modes}  layers={args.n_layers}")
    print(f"dataset size={len(dataset)}  device={device}  epochs={args.epochs}")
    print("─" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0; n = 0
        for inp, target, mask in loader:
            inp, target = inp.to(device), target.to(device)
            mask_t = mask[..., 0].unsqueeze(1).to(device)   # (B,1,Nx,Ny,Nz)
            pred = model(inp)
            diff = (pred - target) ** 2
            loss = (diff * mask_t).sum() / (mask_t.sum() * pred.shape[1]).clamp(min=1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += float(loss.item()); n += 1

        epoch_loss = running / max(n, 1)
        loss_hist.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        scheduler.step(epoch_loss)

        if args.log_every > 0 and (epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs):
            print(f"[epoch {epoch:04d}/{args.epochs}] loss={epoch_loss:.6e} best={best_loss:.6e} "
                  f"lr={optimizer.param_groups[0]['lr']:.3e}")

    ckpt = {
        "model_state": model.state_dict(),
        "model_config": {
            "in_channels": 7, "out_channels": 2,
            "width": args.width, "n_layers": args.n_layers,
            "modes_x": args.modes, "modes_y": args.modes, "modes_z": args.modes,
        },
        "omega_stats": {"min": dataset.omega_min, "max": dataset.omega_max},
        "train_info": {"best_loss": float(best_loss), "epochs": args.epochs},
        "loss_history": loss_hist,
        "config": vars(args),
    }
    torch.save(ckpt, out_path)
    print(f"\nSaved checkpoint: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_mode(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"
                          if args.device == "auto" else args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_cfg = ckpt["model_config"]
    model = FNO3d(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    omega_min = float(ckpt["omega_stats"]["min"])
    omega_max = float(ckpt["omega_stats"]["max"])

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    freq_ids = _parse_freq_indices(args.freq_indices, 99)  # will be filtered by actual M

    with h5py.File(args.test_h5, "r") as f:
        data_ds = f["data"]
        mask_ds = f["mask_tr"]
        omega   = f["omega"][...].astype(np.float32)
        B, M, Nx, Ny, Nz, C = data_ds.shape
        B_eval = B if args.max_eval_samples <= 0 else min(B, args.max_eval_samples)
        freq_ids = [fi for fi in freq_ids if fi < M] or list(range(M))

        x_g = (f["grid_x"][...].astype(np.float32) if "grid_x" in f
                else np.linspace(0, 1, Nx, dtype=np.float32))
        y_g = (f["grid_y"][...].astype(np.float32) if "grid_y" in f
                else np.linspace(0, 1, Ny, dtype=np.float32))
        z_g = (f["grid_z"][...].astype(np.float32) if "grid_z" in f
                else np.linspace(0, 1, Nz, dtype=np.float32))
        def _n(a): lo, hi = a.min(), a.max(); return (a - lo) / max(hi - lo, 1e-12)
        X, Y, Z = np.meshgrid(_n(x_g), _n(y_g), _n(z_g), indexing="ij")
        cx, cy, cz = X.astype(np.float32), Y.astype(np.float32), Z.astype(np.float32)

        rows: List[Dict] = []
        vis_count = 0
        freq_rmse: Dict[int, List[float]] = {m: [] for m in freq_ids}

        for b_idx in range(B_eval):
            for m_idx in freq_ids:
                field = data_ds[b_idx, m_idx].astype(np.float32)
                if mask_ds.ndim == 5:
                    mask = (mask_ds[m_idx].astype(np.float32) > 0.5).astype(np.float32)
                else:
                    mask = (mask_ds[b_idx, m_idx].astype(np.float32) > 0.5).astype(np.float32)

                omega_val  = float(omega[m_idx])
                omega_norm = _normalize_omega(omega_val, omega_min, omega_max)
                omega_map  = np.full((Nx, Ny, Nz), omega_norm, dtype=np.float32)

                obs_re = field[..., 0] * mask[..., 0]
                obs_im = field[..., 1] * mask[..., 1]
                inp = np.stack([obs_re, obs_im, mask[..., 0], omega_map, cx, cy, cz], axis=0)
                inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred_t = model(inp_t)   # (1, 2, Nx, Ny, Nz)
                pred = pred_t[0].cpu().numpy()   # (2, Nx, Ny, Nz)
                pred_field = np.stack([pred[0], pred[1]], axis=-1)   # (Nx, Ny, Nz, 2)

                rmse = _rel_rmse(pred_field, field)
                obs_rmse   = _masked_rel_rmse(pred_field, field, mask)
                unobs_mask = 1.0 - mask
                unobs_rmse = _masked_rel_rmse(pred_field, field, unobs_mask)

                freq_rmse[m_idx].append(rmse)
                rows.append({"b": b_idx, "m": m_idx, "omega": omega_val,
                              "rmse": rmse, "obs_rmse": obs_rmse, "unobs_rmse": unobs_rmse})

                if vis_count < args.num_visualize:
                    vis_count += 1
                    case_stem = out_dir / f"case{vis_count:03d}_s{b_idx:03d}_f{m_idx:03d}"
                    info = f"FNO3D  s={b_idx}  f={m_idx}  ω={omega_val:.3f}  RMSE={rmse:.3f}"

                    # 2-D mid-z slice overview (fast reference)
                    iz = Nz // 2
                    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                    for c_idx, ch in enumerate(["Real", "Imag"]):
                        gt_sl = field[:, :, iz, c_idx]; pd_sl = pred_field[:, :, iz, c_idx]
                        for j, (img, ttl, cmap) in enumerate(
                            [(gt_sl, f"GT {ch}", "viridis"),
                             (pd_sl, f"Pred {ch}", "viridis"),
                             (np.abs(pd_sl - gt_sl), f"Err {ch}", "magma")]):
                            ax = axes[c_idx, j]
                            im = ax.imshow(img, origin="lower", cmap=cmap)
                            ax.set_title(ttl, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
                            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    fig.suptitle(info + "  (mid-z slice)")
                    fig.tight_layout()
                    fig.savefig(str(case_stem) + ".png", dpi=150)
                    plt.close(fig)

                    # 3-D orthogonal-slice + sensor-scatter figures
                    if _HAS_VIS3D:
                        obs_mask3d = mask[:, :, :, 0] if mask.ndim == 4 else mask
                        _save_3d(field, pred_field, obs_mask3d,
                                 stem=case_stem, title=info)

            print(f"  Sample {b_idx+1}/{B_eval}")

    omega_curve  = np.array([float(omega[fi]) for fi in freq_ids])
    rmse_curve   = np.array([np.mean(freq_rmse[fi]) if freq_rmse[fi] else np.nan for fi in freq_ids])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(omega_curve, rmse_curve, lw=2.0, label="FNO3D")
    ax.set_xlabel("ω"); ax.set_ylabel("Relative RMSE"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("FNO3D Frequency-wise Error (3D Helmholtz)")
    fig.tight_layout(); fig.savefig(out_dir / "freq_rmse_curve.png", dpi=150); plt.close(fig)

    np.savetxt(out_dir / "metrics_per_frequency.csv",
               np.column_stack([omega_curve, rmse_curve]),
               delimiter=",", header="omega,rmse", comments="")

    summary = {
        "method": "FNO3D", "ckpt": str(args.ckpt), "test_h5": str(args.test_h5),
        "evaluated_samples": int(B_eval),
        "mean_rmse": float(np.nanmean([r["rmse"] for r in rows])),
        "mean_rmse_obs": float(np.nanmean([r["obs_rmse"] for r in rows])),
        "mean_rmse_unobs": float(np.nanmean([r["unobs_rmse"] for r in rows])),
        "output_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print("\n3D FNO Evaluation finished.")
    print(json.dumps(summary, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="3D FNO baseline for Helmholtz field reconstruction")
    p.add_argument("--mode",   type=str, default="eval", choices=["train", "eval"])

    # Train args
    p.add_argument("--train_h5",          type=str,   default="helmholtz3d_dataset.h5")
    p.add_argument("--out",               type=str,   default="ckp/fno3d.pt")
    p.add_argument("--max_train_samples", type=int,   default=0)
    p.add_argument("--epochs",            type=int,   default=50)
    p.add_argument("--batch_size",        type=int,   default=16)
    p.add_argument("--num_workers",       type=int,   default=16)
    p.add_argument("--lr",                type=float, default=1e-3)
    p.add_argument("--wd",                type=float, default=1e-4)
    p.add_argument("--grad_clip",         type=float, default=1.0)
    p.add_argument("--width",             type=int,   default=32)
    p.add_argument("--n_layers",          type=int,   default=4)
    p.add_argument("--modes",             type=int,   default=8,
                   help="Spectral modes per axis (same for x/y/z).")
    p.add_argument("--log_every",         type=int,   default=5)

    # Eval args
    p.add_argument("--ckpt",             type=str, default="ckp/fno3d.pt")
    p.add_argument("--test_h5",          type=str, default="helmholtz3d_dataset_msk0.01.h5")
    p.add_argument("--out_dir",          type=str, default="visual_data/fno3d_eval_msk0.01")
    p.add_argument("--max_eval_samples", type=int, default=0)
    p.add_argument("--freq_indices",     type=str, default="")
    p.add_argument("--num_visualize",    type=int, default=10)

    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    set_seed(args.seed)
    if args.mode == "train":
        train_mode(args)
    else:
        eval_mode(args)
