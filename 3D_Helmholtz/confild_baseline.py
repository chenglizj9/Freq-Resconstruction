"""
confild_baseline.py  (3D Helmholtz edition)
--------------------------------------------
CoNFiLD-style 3D conditional diffusion baseline for sparse-to-full reconstruction.

Architecture: lightweight 3D U-Net conditioned on sparse observations + omega.
Training: DDPM noise prediction on voxel-space fields.
Inference: DDPM reverse chain with optional DPS observation guidance.

Input channels (9): noisy_re(1) + noisy_im(1) + obs_re(1) + obs_im(1) + mask(1)
                    + omega_map(1) + x_coord(1) + y_coord(1) + z_coord(1)
Output: predicted noise (2 channels)

Examples
--------
Train:
    python confild_baseline.py --mode train \
        --train_h5 helmholtz3d_dataset.h5 --out ckp/confild3d.pt

Eval:
    python confild_baseline.py --mode eval \
        --ckpt ckp/confild3d.pt --test_h5 helmholtz3d_dataset.h5
"""

from __future__ import annotations

import argparse
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
# Diffusion schedule (cosine)
# ─────────────────────────────────────────────────────────────────────────────

def build_cosine_schedule(num_steps: int, s: float = 0.008,
                           device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
    steps = num_steps + 1
    t = torch.linspace(0, num_steps, steps, device=device) / num_steps
    alpha_bars = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bars = (alpha_bars / alpha_bars[0])[:-1]
    alphas = torch.zeros_like(alpha_bars)
    alphas[1:] = alpha_bars[1:] / alpha_bars[:-1]; alphas[0] = alpha_bars[0]
    betas = 1.0 - alphas
    return {
        "betas": betas, "alphas": alphas, "alpha_bars": alpha_bars,
        "sqrt_alpha_bars":           torch.sqrt(alpha_bars),
        "sqrt_one_minus_alpha_bars": torch.sqrt(1.0 - alpha_bars),
    }


def _extract(vals: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    out = vals.gather(0, t).to(dtype=x.dtype)
    return out.view(-1, *([1] * (x.ndim - 1)))


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight 3D U-Net
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32)
                          / max(half - 1, 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1: emb = F.pad(emb, (0, 1))
        return emb


class ResBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        g1 = min(in_ch, 8); g2 = min(out_ch, 8)
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, 2 * out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        sc, sh = torch.chunk(self.time_proj(t_emb), 2, dim=-1)
        h = self.norm2(h) * (1.0 + sc.view(-1, sc.shape[-1], 1, 1, 1)) + sh.view(-1, sh.shape[-1], 1, 1, 1)
        return self.conv2(F.silu(h)) + self.skip(x)


class UNet3d(nn.Module):
    """
    Lightweight 3D U-Net for 3D field diffusion.
    Input:  (B, in_channels, Nx, Ny, Nz)   in_channels = noisy(2) + cond(7) = 9
    Output: (B, out_channels, Nx, Ny, Nz)  out_channels = 2 (predicted noise)
    """
    def __init__(self, in_channels: int = 9, out_channels: int = 2,
                 base_channels: int = 16, time_dim: int = 64):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(time_dim), nn.Linear(time_dim, time_dim * 2), nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.in_conv = nn.Conv3d(in_channels, c1, 3, padding=1)
        self.enc1 = ResBlock3d(c1, c1, time_dim)
        self.down1 = nn.Conv3d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = ResBlock3d(c2, c2, time_dim)
        self.down2 = nn.Conv3d(c2, c3, 3, stride=2, padding=1)
        self.mid   = ResBlock3d(c3, c3, time_dim)
        self.up2   = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.dec2  = ResBlock3d(c2 + c2, c2, time_dim)
        self.up1   = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.dec1  = ResBlock3d(c1 + c1, c1, time_dim)
        self.out_conv = nn.Conv3d(c1, out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = self.in_conv(x)
        e1 = self.enc1(h, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        m  = self.mid(self.down2(e2), t_emb)
        u2 = self.dec2(torch.cat([self.up2(m), e2], dim=1), t_emb)
        u1 = self.dec1(torch.cat([self.up1(u2), e1], dim=1), t_emb)
        return self.out_conv(F.silu(u1))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class Helmholtz3dCoNFiLDDataset(Dataset):
    """Returns (target, cond, mask) tensors."""
    def __init__(self, h5_path: str, max_samples: int = 0):
        self.h5_path = h5_path
        self.cases: List[Tuple[int, int]] = []

        with h5py.File(h5_path, "r") as f:
            B, M, Nx, Ny, Nz, C = f["data"].shape
            omega = f["omega"][...].astype(np.float32)
            self.omega_min = float(omega.min())
            self.omega_max = float(omega.max())
            self.omega     = omega
            self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
            self.C = C

            x_g = (f["grid_x"][...].astype(np.float32) if "grid_x" in f
                    else np.linspace(0, 1, Nx, dtype=np.float32))
            y_g = (f["grid_y"][...].astype(np.float32) if "grid_y" in f
                    else np.linspace(0, 1, Ny, dtype=np.float32))
            z_g = (f["grid_z"][...].astype(np.float32) if "grid_z" in f
                    else np.linspace(0, 1, Nz, dtype=np.float32))

            def _n(a): lo, hi = a.min(), a.max(); return (a - lo) / max(hi - lo, 1e-12)
            X, Y, Z = np.meshgrid(_n(x_g), _n(y_g), _n(z_g), indexing="ij")
            self.cx = X.astype(np.float32)
            self.cy = Y.astype(np.float32)
            self.cz = Z.astype(np.float32)

            B_use = B if max_samples <= 0 else min(B, max_samples)
            for b in range(B_use):
                for m in range(M):
                    self.cases.append((b, m))

    def __len__(self): return len(self.cases)

    def __getitem__(self, idx: int):
        b, m = self.cases[idx]
        with h5py.File(self.h5_path, "r") as f:
            field    = f["data"][b, m].astype(np.float32)   # (Nx, Ny, Nz, 2)
            mask_ds  = f["mask_tr"]
            if mask_ds.ndim == 5:
                mask = (mask_ds[m].astype(np.float32) > 0.5).astype(np.float32)
            elif mask_ds.ndim == 6:
                mask = (mask_ds[b, m].astype(np.float32) > 0.5).astype(np.float32)
            else:
                raise ValueError(f"Unexpected mask dims: {mask_ds.ndim}")
            omega_val = float(f["omega"][m])

        target = np.stack([field[..., 0], field[..., 1]], axis=0).astype(np.float32)   # (2, N, N, N)

        obs_re   = field[..., 0] * mask[..., 0]
        obs_im   = field[..., 1] * mask[..., 1]
        obs_mask = mask[..., 0]
        omega_norm = _normalize_omega(omega_val, self.omega_min, self.omega_max)
        omega_map  = np.full((self.Nx, self.Ny, self.Nz), omega_norm, dtype=np.float32)

        cond = np.stack([obs_re, obs_im, obs_mask, omega_map, self.cx, self.cy, self.cz], axis=0)
        return (torch.from_numpy(target),
                torch.from_numpy(cond),
                torch.from_numpy(mask[..., 0]))   # (2,N,N,N), (7,N,N,N), (N,N,N)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_mode(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"
                          if args.device == "auto" else args.device)

    dataset = Helmholtz3dCoNFiLDDataset(args.train_h5, max_samples=args.max_train_samples)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                         drop_last=True)

    model = UNet3d(in_channels=9, out_channels=2,
                   base_channels=args.base_channels, time_dim=args.time_dim).to(device)
    sched = build_cosine_schedule(args.diffusion_steps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.9, min_lr=1e-6)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf"); loss_hist: List[float] = []

    print(f"\n{'─'*60}")
    print(f"3D CoNFiLD Training  base_ch={args.base_channels}  T={args.diffusion_steps}")
    print(f"dataset={len(dataset)}  device={device}  epochs={args.epochs}")
    print("─" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0; n = 0

        for target, cond, _ in loader:
            target = target.to(device)   # (B, 2, N, N, N)
            cond   = cond.to(device)     # (B, 7, N, N, N)
            bsz    = target.shape[0]

            t = torch.randint(0, args.diffusion_steps, (bsz,), device=device, dtype=torch.long)
            noise    = torch.randn_like(target)
            sqrt_ab  = _extract(sched["sqrt_alpha_bars"],           t, target)
            sqrt_omb = _extract(sched["sqrt_one_minus_alpha_bars"], t, target)
            xt = sqrt_ab * target + sqrt_omb * noise   # noisy field (B, 2, N, N, N)

            inp        = torch.cat([xt, cond], dim=1)  # (B, 9, N, N, N)
            pred_noise = model(inp, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += float(loss.item()); n += 1

        epoch_loss = running / max(n, 1)
        loss_hist.append(epoch_loss)
        if epoch_loss < best_loss: best_loss = epoch_loss
        lr_sched.step(epoch_loss)

        if args.log_every > 0 and (epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs):
            print(f"[epoch {epoch:04d}/{args.epochs}] loss={epoch_loss:.6e} best={best_loss:.6e} "
                  f"lr={optimizer.param_groups[0]['lr']:.3e}")

    ckpt_out = {
        "model_state":  model.state_dict(),
        "model_config": {"in_channels": 9, "out_channels": 2,
                         "base_channels": args.base_channels, "time_dim": args.time_dim},
        "diffusion_config": {"num_steps": args.diffusion_steps, "schedule": "cosine"},
        "omega_stats": {"min": float(dataset.omega_min), "max": float(dataset.omega_max)},
        "train_info":  {"best_loss": float(best_loss), "epochs": args.epochs},
        "loss_history": loss_hist, "config": vars(args),
    }
    torch.save(ckpt_out, out_path)
    print(f"\nSaved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (DDPM + DPS)
# ─────────────────────────────────────────────────────────────────────────────

def _build_ts_seq(total: int, steps: int) -> List[int]:
    if steps <= 0 or steps >= total: return list(range(total - 1, -1, -1))
    seq = np.linspace(total - 1, 0, steps, dtype=np.int64)
    out, seen = [], set()
    for t in seq:
        if int(t) not in seen: out.append(int(t)); seen.add(int(t))
    return out


def eval_mode(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"
                          if args.device == "auto" else args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = UNet3d(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    sched = build_cosine_schedule(ckpt["diffusion_config"]["num_steps"], device=device)
    omega_min = float(ckpt["omega_stats"]["min"])
    omega_max = float(ckpt["omega_stats"]["max"])
    ts_seq = _build_ts_seq(ckpt["diffusion_config"]["num_steps"], args.sample_steps)

    with h5py.File(args.test_h5, "r") as f:
        data_ds = f["data"]
        mask_ds = f["mask_tr"]
        omega   = f["omega"][...].astype(np.float32)
        B, M, Nx, Ny, Nz, C = data_ds.shape
        B_eval = B if args.max_eval_samples <= 0 else min(B, args.max_eval_samples)

        x_g = (f["grid_x"][...].astype(np.float32) if "grid_x" in f
                else np.linspace(0, 1, Nx, dtype=np.float32))
        y_g = (f["grid_y"][...].astype(np.float32) if "grid_y" in f
                else np.linspace(0, 1, Ny, dtype=np.float32))
        z_g = (f["grid_z"][...].astype(np.float32) if "grid_z" in f
                else np.linspace(0, 1, Nz, dtype=np.float32))

        def _n(a): lo, hi = a.min(), a.max(); return (a - lo) / max(hi - lo, 1e-12)
        X, Y, Z  = np.meshgrid(_n(x_g), _n(y_g), _n(z_g), indexing="ij")
        cx, cy, cz = X.astype(np.float32), Y.astype(np.float32), Z.astype(np.float32)

        freq_ids = list(range(M))
        rows: List[Dict] = []
        vis_count = 0
        freq_rmse: Dict[int, List[float]] = {m: [] for m in freq_ids}

        for b_idx in range(B_eval):
            for m_idx in freq_ids:
                field = data_ds[b_idx, m_idx].astype(np.float32)   # (Nx, Ny, Nz, 2)
                if mask_ds.ndim == 5:
                    mask = (mask_ds[m_idx].astype(np.float32) > 0.5).astype(np.float32)
                else:
                    mask = (mask_ds[b_idx, m_idx].astype(np.float32) > 0.5).astype(np.float32)

                omega_val  = float(omega[m_idx])
                omega_norm = _normalize_omega(omega_val, omega_min, omega_max)
                omega_map  = np.full((Nx, Ny, Nz), omega_norm, dtype=np.float32)

                obs_re = field[..., 0] * mask[..., 0]
                obs_im = field[..., 1] * mask[..., 1]
                cond_np = np.stack([obs_re, obs_im, mask[..., 0], omega_map, cx, cy, cz], axis=0)
                cond_t  = torch.from_numpy(cond_np).unsqueeze(0).to(device)   # (1, 7, N, N, N)

                mask_t   = torch.from_numpy(mask[..., 0]).to(device)          # bool after threshold
                y_re_t   = torch.from_numpy(obs_re).to(device)
                y_im_t   = torch.from_numpy(obs_im).to(device)

                g = torch.Generator(device=device)
                g.manual_seed(args.seed + 17 * (b_idx * M + m_idx))
                x_chain = torch.randn(1, 2, Nx, Ny, Nz, generator=g, device=device)

                for i, t_idx in enumerate(ts_seq):
                    t_val = torch.full((1,), t_idx, device=device, dtype=torch.long)
                    with torch.no_grad():
                        inp_t  = torch.cat([x_chain, cond_t], dim=1)
                        eps_pr = model(inp_t, t_val)
                        abar_t = sched["alpha_bars"][t_idx]
                        x0_hat = (x_chain - torch.sqrt(torch.clamp(1 - abar_t, min=1e-12)) * eps_pr) / \
                                  torch.sqrt(torch.clamp(abar_t, min=1e-12))

                    if args.dps_weight > 0:
                        x0_var = x0_hat.detach().requires_grad_(True)
                        m_bool = (mask_t > 0.5)
                        pred_re_obs = x0_var[0, 0][m_bool]
                        pred_im_obs = x0_var[0, 1][m_bool]
                        res_re = (pred_re_obs - y_re_t[m_bool]) ** 2
                        res_im = (pred_im_obs - y_im_t[m_bool]) ** 2
                        y_re_obs_sq = torch.mean(y_re_t[m_bool] ** 2) + 1e-4
                        y_im_obs_sq = torch.mean(y_im_t[m_bool] ** 2) + 1e-4
                        dps_loss = torch.mean(res_re) / y_re_obs_sq + torch.mean(res_im) / y_im_obs_sq
                        grad = torch.autograd.grad(dps_loss, x0_var)[0]
                        tw = 1.0 - (t_idx / max(ts_seq[0], 1))
                        if args.guidance_grad_clip > 0:
                            grad = torch.clamp(grad, -args.guidance_grad_clip, args.guidance_grad_clip)
                        x0_hat = x0_hat - tw * args.dps_weight * grad

                    if i == len(ts_seq) - 1:
                        x_chain = x0_hat
                    else:
                        t_prev  = ts_seq[i + 1]
                        abar_p  = sched["alpha_bars"][t_prev]
                        with torch.no_grad():
                            x_chain = (torch.sqrt(torch.clamp(abar_p, min=1e-12)) * x0_hat +
                                       torch.sqrt(torch.clamp(1 - abar_p, min=1e-12)) * eps_pr)

                pred_np    = x_chain[0].detach().cpu().numpy()    # (2, Nx, Ny, Nz)
                pred_field = np.stack([pred_np[0], pred_np[1]], axis=-1)  # (Nx, Ny, Nz, 2)

                rmse      = _rel_rmse(pred_field, field)
                obs_rmse  = _masked_rel_rmse(pred_field, field, mask)
                unobs_rmse= _masked_rel_rmse(pred_field, field, 1.0 - mask)
                freq_rmse[m_idx].append(rmse)
                rows.append({"b": b_idx, "m": m_idx, "omega": omega_val,
                              "rmse": rmse, "obs_rmse": obs_rmse, "unobs_rmse": unobs_rmse})

                if vis_count < args.num_visualize:
                    vis_count += 1
                    iz = Nz // 2
                    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                    for c_idx, ch in enumerate(["Real", "Imag"]):
                        gt_sl = field[:, :, iz, c_idx]; pd_sl = pred_field[:, :, iz, c_idx]
                        for j, (img, ttl, cmap) in enumerate([
                            (gt_sl, f"GT {ch}", "viridis"),
                            (pd_sl, f"Pred {ch}", "viridis"),
                            (np.abs(pd_sl - gt_sl), f"Err {ch}", "magma"),
                        ]):
                            ax = axes[c_idx, j]
                            im = ax.imshow(img, origin="lower", cmap=cmap)
                            ax.set_title(ttl, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
                            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    fig.suptitle(f"CoNFiLD3D  s={b_idx}  f={m_idx}  ω={omega_val:.3f}  (mid-z)")
                    fig.tight_layout()
                    fig.savefig(out_dir / f"case{vis_count:03d}_s{b_idx:03d}_f{m_idx:03d}.png", dpi=150)
                    plt.close(fig)

            print(f"  Sample {b_idx+1}/{B_eval}")

    omega_curve = np.array([float(omega[fi]) for fi in freq_ids])
    rmse_curve  = np.array([np.mean(freq_rmse[fi]) if freq_rmse[fi] else np.nan for fi in freq_ids])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(omega_curve, rmse_curve, lw=2.0)
    ax.set_xlabel("ω"); ax.set_ylabel("Relative RMSE")
    ax.set_title("CoNFiLD3D Frequency-wise Error (3D Helmholtz)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "freq_rmse_curve.png", dpi=150); plt.close(fig)

    np.savetxt(out_dir / "metrics_per_frequency.csv",
               np.column_stack([omega_curve, rmse_curve]),
               delimiter=",", header="omega,rmse", comments="")

    summary = {
        "method": "CoNFiLD3D", "ckpt": str(args.ckpt), "test_h5": str(args.test_h5),
        "evaluated_samples": int(B_eval),
        "mean_rmse": float(np.nanmean([r["rmse"] for r in rows])),
        "mean_rmse_obs": float(np.nanmean([r["obs_rmse"] for r in rows])),
        "mean_rmse_unobs": float(np.nanmean([r["unobs_rmse"] for r in rows])),
        "dps_weight": float(args.dps_weight),
        "output_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print("\n3D CoNFiLD Evaluation finished.")
    print(json.dumps(summary, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CoNFiLD-style 3D voxel diffusion baseline")
    p.add_argument("--mode",  type=str, default="train", choices=["train", "eval"])
    # Train
    p.add_argument("--train_h5",          type=str,   default="helmholtz3d_dataset.h5")
    p.add_argument("--out",               type=str,   default="ckp/confild3d.pt")
    p.add_argument("--max_train_samples", type=int,   default=0)
    p.add_argument("--epochs",            type=int,   default=200)
    p.add_argument("--batch_size",        type=int,   default=2)
    p.add_argument("--num_workers",       type=int,   default=0)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--wd",                type=float, default=1e-4)
    p.add_argument("--grad_clip",         type=float, default=1.0)
    p.add_argument("--base_channels",     type=int,   default=16)
    p.add_argument("--time_dim",          type=int,   default=64)
    p.add_argument("--diffusion_steps",   type=int,   default=500)
    p.add_argument("--log_every",         type=int,   default=5)
    # Eval
    p.add_argument("--ckpt",             type=str,   default="ckp/confild3d.pt")
    p.add_argument("--test_h5",          type=str,   default="helmholtz3d_dataset.h5")
    p.add_argument("--out_dir",          type=str,   default="visual_data/confild3d_eval")
    p.add_argument("--max_eval_samples", type=int,   default=0)
    p.add_argument("--sample_steps",     type=int,   default=100)
    p.add_argument("--dps_weight",       type=float, default=1.0)
    p.add_argument("--guidance_grad_clip",type=float, default=1e-2)
    p.add_argument("--num_visualize",    type=int,   default=10)
    # Common
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
