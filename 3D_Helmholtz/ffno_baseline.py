"""
ffno_baseline.py  (3D Helmholtz edition)
-----------------------------------------
Factorized FNO (F-FNO) baseline for 2-channel 3D Helmholtz field reconstruction.
[Tran et al., "Factorized Fourier Neural Operators", ICLR 2023]

Replaces 3D joint spectral convolution with three separate 1D FFTs along Nx, Ny, Nz.
Memory: O(in·out·(m1+m2+m3)) vs O(in·out·m1·m2·m3) for standard FNO3D.

Input channels (7):
  obs_re, obs_im, mask, omega_map, x_coord, y_coord, z_coord
Output (2): pred_re, pred_im

Usage
-----
Train:
    python ffno_baseline.py --mode train \\
        --train_h5 helmholtz3d_dataset.h5 --out ckp/ffno_3d.pt
Eval:
    python ffno_baseline.py --mode eval \\
        --ckpt ckp/ffno_3d.pt --test_h5 helmholtz3d_dataset.h5
"""

from __future__ import annotations

import argparse
import csv
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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _normalize_omega(omega, omega_min, omega_max):
    return float((omega - omega_min) / max(omega_max - omega_min, 1e-12))


def _rel_rmse(pred, gt, eps=1e-12):
    return float(np.sqrt(np.sum((pred - gt) ** 2) / max(np.sum(gt ** 2), eps)))


def _masked_rel_rmse(pred, gt, mask, eps=1e-12):
    m = mask.astype(bool)
    if not np.any(m): return float("nan")
    return float(np.sqrt(np.sum((pred - gt) ** 2 * m) / max(np.sum(gt ** 2 * m), eps)))


def _safe_load(path):
    try: return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# Dataset  (per-item HDF5 reads, same as existing fno_baseline.py)
# ---------------------------------------------------------------------------

class Helmholtz3dFNOData(Dataset):
    def __init__(self, h5_path: str, max_samples: int = 0, freq_indices: List[int] = None):
        self.h5_path = h5_path
        self.cases: List[Tuple[int, int]] = []

        with h5py.File(h5_path, "r") as f:
            B, M, Nx, Ny, Nz, C = f["data"].shape
            omega = f["omega"][...].astype(np.float32)
            self.omega_min = float(omega.min()); self.omega_max = float(omega.max())
            self.omega = omega; self.C = C
            x_g = f["grid_x"][...].astype(np.float32) if "grid_x" in f else np.linspace(0,1,Nx,dtype=np.float32)
            y_g = f["grid_y"][...].astype(np.float32) if "grid_y" in f else np.linspace(0,1,Ny,dtype=np.float32)
            z_g = f["grid_z"][...].astype(np.float32) if "grid_z" in f else np.linspace(0,1,Nz,dtype=np.float32)
            def _n(a): lo,hi=a.min(),a.max(); return (a-lo)/max(hi-lo,1e-12)
            X, Y, Z = np.meshgrid(_n(x_g), _n(y_g), _n(z_g), indexing="ij")
            self.cx = X.astype(np.float32); self.cy = Y.astype(np.float32); self.cz = Z.astype(np.float32)
            self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
            B_use = B if max_samples <= 0 else min(B, max_samples)
            M_use = list(range(M)) if freq_indices is None else freq_indices
            for b in range(B_use):
                for m in M_use: self.cases.append((b, m))

    def __len__(self): return len(self.cases)

    def __getitem__(self, idx):
        b, m = self.cases[idx]
        with h5py.File(self.h5_path, "r") as f:
            field = f["data"][b, m].astype(np.float32)   # (Nx,Ny,Nz,C)
            mask_ds = f["mask_tr"]
            if mask_ds.ndim == 5: mask = (mask_ds[m].astype(np.float32) > 0.5).astype(np.float32)
            elif mask_ds.ndim == 6: mask = (mask_ds[b,m].astype(np.float32) > 0.5).astype(np.float32)
            else: raise ValueError(f"mask dims {mask_ds.ndim}")
            omega_val = float(f["omega"][m])

        omega_n = _normalize_omega(omega_val, self.omega_min, self.omega_max)
        obs_re = field[..., 0] * mask[..., 0]
        obs_im = field[..., 1] * mask[..., 0] if self.C > 1 else np.zeros_like(field[..., 0])
        obs_mask = mask[..., 0]
        omega_map = np.full((self.Nx, self.Ny, self.Nz), omega_n, dtype=np.float32)
        inp = np.stack([obs_re, obs_im, obs_mask, omega_map, self.cx, self.cy, self.cz], axis=0)
        target = np.stack([field[..., c] for c in range(self.C)], axis=0)
        return (torch.from_numpy(inp), torch.from_numpy(target),
                torch.from_numpy(mask), torch.tensor(omega_n, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Factorized Spectral Conv 3D
# ---------------------------------------------------------------------------

def _cmul1d(x_ft, w):
    xr, xi = x_ft.real, x_ft.imag
    wr, wi = w[..., 0], w[..., 1]
    return torch.complex(
        torch.einsum("bil,iol->bol", xr, wr) - torch.einsum("bil,iol->bol", xi, wi),
        torch.einsum("bil,iol->bol", xr, wi) + torch.einsum("bil,iol->bol", xi, wr))


class FactorizedSpectralConv3d(nn.Module):
    """Three separate 1D spectral convolutions along x, y, z."""
    def __init__(self, in_ch, out_ch, mx, my, mz):
        super().__init__()
        self.in_ch = in_ch; self.out_ch = out_ch
        self.mx = mx; self.my = my; self.mz = mz
        scale = 1.0 / (in_ch * out_ch)
        self.wx = nn.Parameter(scale * torch.randn(in_ch, out_ch, mx, 2))
        self.wy = nn.Parameter(scale * torch.randn(in_ch, out_ch, my, 2))
        self.wz = nn.Parameter(scale * torch.randn(in_ch, out_ch, mz, 2))

    def _1d_conv(self, x, dim, modes, w):
        """FFT along one dim, mix with w, IFFT back. x: (B,C,Nx,Ny,Nz)."""
        B, C, Nx, Ny, Nz = x.shape
        Fx = torch.fft.rfft(x, dim=dim)
        Ls = Fx.shape[dim]
        m  = min(modes, Ls)
        # Flatten all dims except B, C, and the FFT dim into the batch dim
        perm, shape = list(range(5)), list(x.shape)
        fft_axis = dim  # original axis index
        # Move fft axis to last, keep B,C fixed
        # We need (B*other, C, L) where L is the rfft length
        # For dim=2 (x): (B,C,Nx,Ny,Nz) → fold Ny,Nz → (B*Ny*Nz, C, Nx)
        # Strategy: permute so fft_dim comes after B and C, then fold rest
        dims_order = [0, 1, dim] + [d for d in range(2,5) if d != dim]
        xt = x.permute(dims_order)           # (B, C, L, ...)
        Lv  = xt.shape[2]
        rest = xt.shape[3:]
        xflat = xt.reshape(B, C, Lv, -1).permute(0,3,1,2).reshape(-1, C, Lv)  # (B*N, C, L)
        Ff = torch.fft.rfft(xflat, dim=-1)
        Ls2 = Ff.shape[-1]; m2 = min(modes, Ls2)
        out_f = torch.zeros(xflat.shape[0], self.out_ch, Ls2, dtype=torch.cfloat, device=x.device)
        out_f[:, :, :m2] = _cmul1d(Ff[:, :, :m2], w[:, :, :m2, :])
        out_s = torch.fft.irfft(out_f, n=Lv, dim=-1)  # (B*N, out_ch, L)
        N_rest = int(np.prod(rest)) if len(rest) > 0 else 1
        out_r = out_s.reshape(B, N_rest, self.out_ch, Lv).permute(0,2,3,1)  # (B, out_ch, L, N_rest)
        # Reconstruct to (B, out_ch, Nx, Ny, Nz) with L at the right position
        inv_order = [0,1] + [None]*3
        spatial_dims_new = list(rest)
        shape_new = [B, self.out_ch, Lv] + list(rest)
        out_vol = out_r.reshape(shape_new)  # (B, out_ch, L, d1, d2)
        # Restore dim order: currently (B, out_ch, dim, others), need (B, out_ch, Nx, Ny, Nz)
        back_perm = [0, 1] + [None]*3
        orig_spatial = list(range(2, 5))  # [2,3,4] for Nx,Ny,Nz
        out_spatial_dims = [dim] + [d for d in orig_spatial if d != dim]  # e.g. for dim=2: [2,3,4]
        back = [0, 1] + [2 + out_spatial_dims.index(d) for d in orig_spatial]
        return out_vol.permute(*back)

    def forward(self, x):
        return self._1d_conv(x, 2, self.mx, self.wx) + \
               self._1d_conv(x, 3, self.my, self.wy) + \
               self._1d_conv(x, 4, self.mz, self.wz)


class FFNOBlock3D(nn.Module):
    def __init__(self, width, mx, my, mz):
        super().__init__()
        self.sp = FactorizedSpectralConv3d(width, width, mx, my, mz)
        self.w  = nn.Conv3d(width, width, 1)
        self.n  = nn.GroupNorm(min(8, width), width)
    def forward(self, x): return F.gelu(self.n(self.sp(x) + self.w(x)))


class FFNO3d(nn.Module):
    def __init__(self, in_channels=7, out_channels=2, width=32, mx=8, my=8, mz=8, n_layers=4):
        super().__init__()
        self.fc0 = nn.Conv3d(in_channels, width, 1)
        self.blocks = nn.ModuleList([FFNOBlock3D(width, mx, my, mz) for _ in range(n_layers)])
        self.fc1 = nn.Conv3d(width, width*2, 1)
        self.fc2 = nn.Conv3d(width*2, out_channels, 1)

    def forward(self, x):
        x = self.fc0(x)
        for b in self.blocks: x = b(x)
        return self.fc2(F.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_mode(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    ds = Helmholtz3dFNOData(args.train_h5, max_samples=args.max_train_samples)
    n  = len(ds); n_tr = max(1, int(round(n * args.train_ratio)))
    idxs = np.random.default_rng(args.seed).permutation(n)
    tr_ds = torch.utils.data.Subset(ds, idxs[:n_tr].tolist())
    va_ds = torch.utils.data.Subset(ds, idxs[n_tr:].tolist() or idxs[-1:].tolist())
    tr_dl = DataLoader(tr_ds, args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    va_dl = DataLoader(va_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    in_ch, out_ch = ds.C + 5, ds.C

    model = FFNO3d(in_ch, out_ch, args.width, args.modes_x, args.modes_y, args.modes_z, args.n_layers).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    print(f"\nF-FNO 3D | in={in_ch} out={out_ch} width={args.width} device={device}")

    for epoch in range(1, args.epochs + 1):
        model.train(); tl = 0.0; tn = 0
        for inp, tgt, mask, *_ in tr_dl:
            inp, tgt = inp.to(device), tgt.to(device)
            mask_t = mask[..., 0].unsqueeze(1).to(device)   # (B,1,Nx,Ny,Nz)
            pred = model(inp)
            diff = (pred - tgt) ** 2
            loss = (diff * mask_t).sum() / (mask_t.sum() * pred.shape[1]).clamp(min=1)
            opt.zero_grad(set_to_none=True); loss.backward()
            if args.grad_clip > 0: nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); tl += loss.item() * inp.shape[0]; tn += inp.shape[0]
        model.eval(); vl = 0.0; vn = 0
        with torch.no_grad():
            for inp, tgt, mask, *_ in va_dl:
                inp, tgt = inp.to(device), tgt.to(device)
                mask_t = mask[..., 0].unsqueeze(1).to(device)
                pred = model(inp)
                diff = (pred - tgt) ** 2
                vl += ((diff * mask_t).sum() / (mask_t.sum() * pred.shape[1]).clamp(min=1)).item() * inp.shape[0]
                vn += inp.shape[0]
        tl /= max(tn,1); vl /= max(vn,1); sched.step(vl)
        if vl < best_val:
            best_val = vl
            torch.save({"model_state": model.state_dict(),
                        "model_config": {"in_channels": in_ch, "out_channels": out_ch,
                                         "width": args.width, "mx": args.modes_x,
                                         "my": args.modes_y, "mz": args.modes_z,
                                         "n_layers": args.n_layers},
                        "train_config": vars(args), "best_val_loss": best_val}, out_path)
        if args.log_every > 0 and (epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs):
            print(f"[{epoch:04d}/{args.epochs}] train={tl:.4e} val={vl:.4e} best={best_val:.4e}")
    print(f"Done. ckpt={out_path}")


def eval_mode(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    ckpt  = _safe_load(Path(args.ckpt))
    model = FFNO3d(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    ds = Helmholtz3dFNOData(args.test_h5, max_samples=args.max_samples)
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=0)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []; vis_cnt = 0
    with torch.no_grad():
        for case_idx, (inp, tgt, mask, omega_n) in enumerate(dl):
            pred = model(inp.to(device)).cpu().numpy()
            gt = tgt.numpy(); mask_np = mask.numpy(); omega_n = omega_n.numpy()
            for i in range(pred.shape[0]):
                pred_i = pred[i].transpose(1,2,3,0); gt_i = gt[i].transpose(1,2,3,0)
                mask_i = mask_np[i,...,0]
                rmse = _rel_rmse(pred_i, gt_i)
                obs_r = _masked_rel_rmse(pred_i, gt_i, mask_i[..., None])
                unobs_r = _masked_rel_rmse(pred_i, gt_i, (1.0-mask_i)[..., None])
                global_i = case_idx * args.batch_size + i
                rows.append({"case": global_i, "omega_n": float(omega_n[i]),
                             "rmse": rmse, "obs_rmse": obs_r, "unobs_rmse": unobs_r})
                if vis_cnt < args.num_visualize:
                    vis_cnt += 1
                    case_stem = out_dir / f"case{vis_cnt:03d}_{global_i:05d}"
                    info = f"FFNO3D  case={global_i}  ω={float(omega_n[i]):.3f}  RMSE={rmse:.3f}"
                    iz = gt_i.shape[2] // 2
                    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                    for c_idx, ch in enumerate(["Real", "Imag"]):
                        gt_sl = gt_i[:, :, iz, c_idx]; pd_sl = pred_i[:, :, iz, c_idx]
                        for j, (img, ttl, cmap) in enumerate([
                            (gt_sl, f"GT {ch}", "viridis"),
                            (pd_sl, f"Pred {ch}", "viridis"),
                            (np.abs(pd_sl - gt_sl), f"Err {ch}", "magma"),
                        ]):
                            ax = axes[c_idx, j]
                            im = ax.imshow(img, origin="lower", cmap=cmap)
                            ax.set_title(ttl, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
                            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    fig.suptitle(info + "  (mid-z slice)")
                    fig.tight_layout()
                    fig.savefig(str(case_stem) + ".png", dpi=args.vis_dpi)
                    plt.close(fig)
                    if _HAS_VIS3D:
                        _save_3d(gt_i, pred_i, mask_i, stem=case_stem, title=info)
                    rows[-1]["vis_path"] = str(case_stem) + ".png"
    csv_path = out_dir / "metrics_cases.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, ["case","omega_n","rmse","obs_rmse","unobs_rmse","vis_path"])
        w.writeheader()
        for r in rows: w.writerow(r)
    summary = {"ckpt": args.ckpt, "test_h5": args.test_h5, "num_cases": len(rows),
               "mean_rmse": float(np.mean([r["rmse"] for r in rows])),
               "mean_obs_rmse": float(np.mean([r["obs_rmse"] for r in rows])),
               "mean_unobs_rmse": float(np.mean([r["unobs_rmse"] for r in rows]))}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nEvaluation finished."); print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser("F-FNO 3D Helmholtz baseline")
    p.add_argument("--mode", choices=["train","eval"], default="eval")
    p.add_argument("--train_h5", default="helmholtz3d_dataset.h5")
    p.add_argument("--test_h5",  default="helmholtz3d_dataset_msk0.01.h5")
    p.add_argument("--ckpt",     default="ckp/ffno_3d.pt")
    p.add_argument("--out",      default="ckp/ffno_3d.pt")
    p.add_argument("--out_dir",  default="visual_data/ffno_3d_eval_msk0.01")
    p.add_argument("--train_ratio",   type=float, default=0.8)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_samples",   type=int, default=0)
    p.add_argument("--epochs",        type=int, default=50)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--wd",            type=float, default=1e-6)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--log_every",     type=int, default=5)
    p.add_argument("--width",         type=int, default=32)
    p.add_argument("--modes_x",       type=int, default=8)
    p.add_argument("--modes_y",       type=int, default=8)
    p.add_argument("--modes_z",       type=int, default=8)
    p.add_argument("--n_layers",      type=int, default=4)
    p.add_argument("--num_workers",    type=int, default=32)
    p.add_argument("--num_visualize",  type=int, default=10)
    p.add_argument("--vis_dpi",        type=int, default=150)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--device",         type=str, default="auto")
    return p


def main():
    args = build_parser().parse_args()
    train_mode(args) if args.mode == "train" else eval_mode(args)


if __name__ == "__main__":
    main()
