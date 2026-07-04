"""
diffusion_pde_baseline.py  (3D Helmholtz edition)
--------------------------------------------------
Pixel-space diffusion baseline for 2-channel 3D Helmholtz field reconstruction.
DiffusionPDE-style [Huang et al., NeurIPS 2024]:
  - Train: ω-conditional 3D score network (NO obs/mask in model input)
  - Eval:  DDPM + scale-invariant DPS guidance (fixed λ)

Model input: noisy 3D field (C=2 channels) + ω embedding via FiLM
Output: noise prediction (C=2 channels)

NOTE: 3D volumes are large. Default batch_size=1, base_channels=16 to fit memory.
      Use T=200 and --batch_size 1 for evaluation.

Usage
-----
Train:
    python diffusion_pde_baseline.py --mode train \\
        --train_h5 helmholtz3d_dataset.h5 --out ckp/dpde_3d.pt
Eval:
    python diffusion_pde_baseline.py --mode eval \\
        --ckpt ckp/dpde_3d.pt --test_h5 helmholtz3d_dataset.h5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
# Dataset
# ---------------------------------------------------------------------------

class Helmholtz3dDPDEData(Dataset):
    def __init__(self, h5_path: str, max_samples: int = 0, freq_indices: List[int] = None,
                 return_obs: bool = False):
        self.h5_path = h5_path; self.return_obs = return_obs
        self.cases: List[Tuple[int, int]] = []
        with h5py.File(h5_path, "r") as f:
            B, M, Nx, Ny, Nz, C = f["data"].shape
            omega = f["omega"][...].astype(np.float32)
            self.omega_min = float(omega.min()); self.omega_max = float(omega.max())
            self.omega = omega; self.C = C; self.Nx = Nx; self.Ny = Ny; self.Nz = Nz
            B_use = B if max_samples <= 0 else min(B, max_samples)
            M_use = list(range(M)) if freq_indices is None else freq_indices
            for b in range(B_use):
                for m in M_use: self.cases.append((b, m))

    def __len__(self): return len(self.cases)

    def __getitem__(self, idx):
        b, m = self.cases[idx]
        with h5py.File(self.h5_path, "r") as f:
            field = f["data"][b, m].astype(np.float32)   # (Nx,Ny,Nz,C)
            omega_val = float(f["omega"][m])
            mask_ds = f["mask_tr"]
            if mask_ds.ndim == 5: mask = (mask_ds[m].astype(np.float32) > 0.5).astype(np.float32)
            elif mask_ds.ndim == 6: mask = (mask_ds[b,m].astype(np.float32) > 0.5).astype(np.float32)
            else: raise ValueError(f"mask dims {mask_ds.ndim}")

        omega_n = _normalize_omega(omega_val, self.omega_min, self.omega_max)
        target   = np.stack([field[..., c] for c in range(self.C)], axis=0)  # (C, Nx,Ny,Nz)
        mask_ch  = mask[..., 0:1].transpose(3, 0, 1, 2).astype(np.float32)   # (1, Nx,Ny,Nz)
        out = (torch.from_numpy(target),
               torch.tensor(omega_n, dtype=torch.float32).unsqueeze(0),
               torch.from_numpy(mask_ch))
        if self.return_obs:
            obs = np.stack([field[..., c] * mask[..., 0] for c in range(self.C)], axis=0)
            return out + (torch.from_numpy(obs),)
        return out


# ---------------------------------------------------------------------------
# Diffusion schedule
# ---------------------------------------------------------------------------

class DiffusionSchedule:
    def __init__(self, betas):
        self.betas = betas; al = 1.0 - betas; self.alphas = al
        self.alpha_bars = torch.cumprod(al, 0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    @classmethod
    def linear(cls, T, b0=1e-4, b1=2e-2, device="cpu"):
        return cls(torch.linspace(b0, b1, T, device=device))

    def to(self, device):
        for a in ["betas","alphas","alpha_bars","sqrt_alpha_bars","sqrt_one_minus_alpha_bars"]:
            setattr(self, a, getattr(self, a).to(device))
        return self

    def gather(self, vals, t, x):
        out = vals.gather(0, t).to(dtype=x.dtype)
        return out.view(-1, *([1]*(x.ndim-1)))


# ---------------------------------------------------------------------------
# 3D Score network (ω-conditional 3D U-Net)
# ---------------------------------------------------------------------------

def _gn(ch, mg=8):
    g = min(mg, ch)
    while g > 1 and ch % g != 0: g -= 1
    return g


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half-1,1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], -1)
        return F.pad(emb, (0,1)) if self.dim%2==1 else emb


class FourierOmegaEmb(nn.Module):
    def __init__(self, bands=8):
        super().__init__(); self.bands = bands; self.out_dim = 1 + 2*bands
    def forward(self, w):
        feats = [w]
        for k in range(self.bands):
            f = (2**k) * math.pi
            feats += [torch.sin(f*w), torch.cos(f*w)]
        return torch.cat(feats, -1)


class ScoreRes3D(nn.Module):
    def __init__(self, ic, oc, cd):
        super().__init__()
        self.n1 = nn.GroupNorm(_gn(ic), ic); self.c1 = nn.Conv3d(ic, oc, 3, padding=1)
        self.n2 = nn.GroupNorm(_gn(oc), oc); self.c2 = nn.Conv3d(oc, oc, 3, padding=1)
        self.cp = nn.Linear(cd, 2*oc)
        self.sk = nn.Conv3d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x, cond):
        h = self.c1(F.silu(self.n1(x))); h = self.n2(h)
        sc = self.cp(cond); sc, sh = sc.chunk(2, -1)
        h = h * (1 + sc.view(-1, sc.shape[-1], 1, 1, 1)) + sh.view(-1, sh.shape[-1], 1, 1, 1)
        return self.c2(F.silu(h)) + self.sk(x)


class PixelScoreUNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, base_channels=16,
                 cond_dim=128, time_dim=64, omega_bands=8):
        super().__init__()
        self.te = SinusoidalTimeEmb(time_dim); self.oe = FourierOmegaEmb(omega_bands)
        self.cm = nn.Sequential(nn.Linear(time_dim + self.oe.out_dim, cond_dim),
                                nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        c1, c2, c3 = base_channels, base_channels*2, base_channels*4
        self.st  = nn.Conv3d(in_channels, c1, 3, padding=1)
        self.e1  = ScoreRes3D(c1, c1, cond_dim); self.d1 = nn.Conv3d(c1, c2, 3, stride=2, padding=1)
        self.e2  = ScoreRes3D(c2, c2, cond_dim); self.d2 = nn.Conv3d(c2, c3, 3, stride=2, padding=1)
        self.mid = ScoreRes3D(c3, c3, cond_dim)
        self.u2  = ScoreRes3D(c3+c2, c2, cond_dim); self.u1 = ScoreRes3D(c2+c1, c1, cond_dim)
        self.on  = nn.GroupNorm(_gn(c1), c1); self.oc = nn.Conv3d(c1, out_channels, 3, padding=1)

    def forward(self, x, t, omega_norm):
        cond = self.cm(torch.cat([self.te(t), self.oe(omega_norm)], -1))
        e0 = self.st(x); e1 = self.e1(e0, cond)
        e2 = self.e2(self.d1(e1), cond); m = self.mid(self.d2(e2), cond)
        u2 = self.u2(torch.cat([F.interpolate(m, e2.shape[-3:], mode="nearest"), e2], 1), cond)
        u1 = self.u1(torch.cat([F.interpolate(u2, e1.shape[-3:], mode="nearest"), e1], 1), cond)
        return self.oc(F.silu(self.on(u1)))


# ---------------------------------------------------------------------------
# DPS sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def dps_sample_3d(model, sched, omega, y_obs, mask, T, zeta, device):
    """DDPM + scale-invariant DPS for 3D fields.
    omega: (B,1), y_obs: (B,C,Nx,Ny,Nz), mask: (B,1,Nx,Ny,Nz)
    """
    B = y_obs.shape[0]; x = torch.randn_like(y_obs)
    for t_idx in reversed(range(T)):
        t_vec = torch.full((B,), t_idx, device=device, dtype=torch.long)
        eps_pred = model(x, t_vec, omega)
        ab_t = sched.alpha_bars[t_idx]; a_t = sched.alphas[t_idx]; b_t = sched.betas[t_idx]
        x0hat = (x - (1-ab_t).sqrt() * eps_pred) / ab_t.sqrt().clamp(min=1e-8)
        noise = torch.randn_like(x) if t_idx > 0 else torch.zeros_like(x)
        x_prior = (1.0/a_t.sqrt()) * (x - ((1.0-a_t)/(1.0-ab_t).sqrt()) * eps_pred) + b_t.sqrt()*noise
        g_t = mask * (mask * x0hat - y_obs)
        prior_step_norm = (x_prior - x).pow(2).sum(dim=(1,2,3,4), keepdim=True).sqrt().clamp(min=1e-8)
        g_norm = g_t.pow(2).sum(dim=(1,2,3,4), keepdim=True).sqrt().clamp(min=1e-8)
        x = x_prior - zeta * prior_step_norm / g_norm * g_t
    return x


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def _collate_train(batch):
    # dataset now always returns (y, omega, mask); obs only added when return_obs=True
    y     = torch.stack([b[0] for b in batch])
    omega = torch.stack([b[1] for b in batch])
    mask  = torch.stack([b[2] for b in batch])
    return y, omega, mask


def _collate_eval(batch):
    y    = torch.stack([b[0] for b in batch])
    omega = torch.stack([b[1] for b in batch])
    mask  = torch.stack([b[2] for b in batch])
    obs   = torch.stack([b[3] for b in batch])
    return y, omega, mask, obs


def train_mode(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    ds = Helmholtz3dDPDEData(args.train_h5, max_samples=args.max_train_samples, return_obs=False)
    n  = len(ds); n_tr = max(1, int(round(n * args.train_ratio)))
    idxs = np.random.default_rng(args.seed).permutation(n)
    tr_ds = torch.utils.data.Subset(ds, idxs[:n_tr].tolist())
    va_ds = torch.utils.data.Subset(ds, idxs[n_tr:].tolist() or idxs[-1:].tolist())
    tr_dl = DataLoader(tr_ds, args.batch_size, shuffle=True,  num_workers=args.num_workers,
                       collate_fn=_collate_train, pin_memory=True)
    va_dl = DataLoader(va_ds, args.batch_size, shuffle=False, num_workers=args.num_workers,
                       collate_fn=_collate_train, pin_memory=True)

    C = ds.C
    model = PixelScoreUNet3D(C, C, args.base_channels, args.cond_dim, args.time_dim, args.omega_bands).to(device)
    sched = DiffusionSchedule.linear(args.T, args.beta_start, args.beta_end, device=device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_sc = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    print(f"\nDiffusionPDE 3D | C={C} base={args.base_channels} T={args.T} device={device}")

    for epoch in range(1, args.epochs + 1):
        model.train(); tl = 0.0; tn = 0
        for y, omega, mask in tr_dl:
            y, omega, mask = y.to(device), omega.to(device), mask.to(device); B = y.shape[0]
            t = torch.randint(0, args.T, (B,), device=device)
            noise = torch.randn_like(y)
            sa = sched.gather(sched.sqrt_alpha_bars, t, y)
            sb = sched.gather(sched.sqrt_one_minus_alpha_bars, t, y)
            eps_pred = model(sa*y + sb*noise, t, omega)
            # loss only at observed sensor locations
            diff = (eps_pred - noise) ** 2
            n_obs = mask.sum() * y.shape[1]
            loss = (diff * mask).sum() / n_obs.clamp(min=1)
            opt.zero_grad(set_to_none=True); loss.backward()
            if args.grad_clip > 0: nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); tl += loss.item()*B; tn += B
        model.eval(); vl = 0.0; vn = 0
        with torch.no_grad():
            for y, omega, mask in va_dl:
                y, omega, mask = y.to(device), omega.to(device), mask.to(device); B = y.shape[0]
                t = torch.randint(0, args.T, (B,), device=device); noise = torch.randn_like(y)
                sa = sched.gather(sched.sqrt_alpha_bars, t, y); sb = sched.gather(sched.sqrt_one_minus_alpha_bars, t, y)
                eps_pred = model(sa*y+sb*noise, t, omega)
                diff = (eps_pred - noise) ** 2; n_obs = mask.sum() * y.shape[1]
                vl += ((diff * mask).sum() / n_obs.clamp(min=1)).item()*B; vn += B
        tl /= max(tn,1); vl /= max(vn,1); lr_sc.step(vl)
        if vl < best_val:
            best_val = vl
            torch.save({"model_state": model.state_dict(),
                        "model_config": {"in_channels": C, "out_channels": C,
                                         "base_channels": args.base_channels,
                                         "cond_dim": args.cond_dim, "time_dim": args.time_dim,
                                         "omega_bands": args.omega_bands},
                        "diffusion_config": {"T": args.T, "beta_start": args.beta_start, "beta_end": args.beta_end},
                        "train_config": vars(args), "best_val_loss": best_val}, out_path)
        if args.log_every > 0 and (epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs):
            print(f"[{epoch:04d}/{args.epochs}] train={tl:.4e} val={vl:.4e} best={best_val:.4e}")
    print(f"Done. ckpt={out_path}")


def eval_mode(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    ckpt  = _safe_load(Path(args.ckpt))
    model = PixelScoreUNet3D(**ckpt["model_config"]).to(device); model.eval()
    model.load_state_dict(ckpt["model_state"])
    dcfg  = ckpt.get("diffusion_config", {}); T = dcfg.get("T", args.T)
    sched = DiffusionSchedule.linear(T, dcfg.get("beta_start", args.beta_start),
                                     dcfg.get("beta_end", args.beta_end)).to(device)

    ds = Helmholtz3dDPDEData(args.test_h5, max_samples=args.max_samples, return_obs=True)
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate_eval)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []; vis_cnt = 0

    for case_idx, (y, omega, mask, obs) in enumerate(dl):
        y_d = y.to(device); omega_d = omega.to(device)
        obs_d = obs.to(device); mask_d = mask.to(device)
        pred_t = dps_sample_3d(model, sched, omega_d, obs_d, mask_d, T, args.zeta, device)
        pred_np = pred_t.cpu().numpy(); gt_np = y.numpy(); mask_np = mask.numpy()
        for i in range(pred_np.shape[0]):
            pred_i = pred_np[i].transpose(1,2,3,0); gt_i = gt_np[i].transpose(1,2,3,0)
            mask_i = mask_np[i, 0]
            rmse = _rel_rmse(pred_i, gt_i)
            obs_r = _masked_rel_rmse(pred_i, gt_i, mask_i[..., None])
            unobs_r = _masked_rel_rmse(pred_i, gt_i, (1.0-mask_i)[..., None])
            global_i = case_idx * args.batch_size + i
            rows.append({"case": global_i, "omega_n": float(omega[i,0]),
                         "rmse": rmse, "obs_rmse": obs_r, "unobs_rmse": unobs_r})
            if vis_cnt < args.num_visualize:
                vis_cnt += 1
                case_stem = out_dir / f"case{vis_cnt:03d}_{global_i:05d}"
                info = f"DiffusionPDE3D  case={global_i}  ω={float(omega[i,0]):.3f}  RMSE={rmse:.3f}"
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
               "dps_zeta": args.zeta,
               "mean_rmse": float(np.mean([r["rmse"] for r in rows])),
               "mean_obs_rmse": float(np.mean([r["obs_rmse"] for r in rows])),
               "mean_unobs_rmse": float(np.mean([r["unobs_rmse"] for r in rows]))}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nEvaluation finished."); print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser("DiffusionPDE 3D Helmholtz baseline")
    p.add_argument("--mode", choices=["train","eval"], default="eval")
    p.add_argument("--train_h5", default="helmholtz3d_dataset.h5")
    p.add_argument("--test_h5",  default="helmholtz3d_dataset_msk0.01.h5")
    p.add_argument("--ckpt",     default="ckp/dpde_3d.pt")
    p.add_argument("--out",      default="ckp/dpde_3d.pt")
    p.add_argument("--out_dir",  default="visual_data/dpde_3d_eval_msk0.01")
    p.add_argument("--train_ratio",   type=float, default=0.8)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_samples",   type=int, default=0)
    p.add_argument("--epochs",        type=int, default=50)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--wd",            type=float, default=1e-6)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--log_every",     type=int, default=5)
    p.add_argument("--base_channels", type=int, default=16)
    p.add_argument("--cond_dim",      type=int, default=128)
    p.add_argument("--time_dim",      type=int, default=64)
    p.add_argument("--omega_bands",   type=int, default=8)
    p.add_argument("--T",             type=int, default=200)
    p.add_argument("--beta_start",    type=float, default=1e-4)
    p.add_argument("--beta_end",      type=float, default=2e-2)
    p.add_argument("--zeta",          type=float, default=0.3)
    p.add_argument("--num_workers",    type=int, default=16)
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
