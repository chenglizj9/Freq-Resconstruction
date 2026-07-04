"""
train_diffusion.py  (multi-channel edition)
--------------------------------------------
Train a conditional diffusion model p(G_c | omega) on FTM core tensors.

Task setting
------------
All physical field channels share the same spatial basis (net_x / net_y).
Only the core tensors G_c ∈ R^{B × M × Rx × Ry} differ per channel c.
The diffusion model operates on the concatenated core image:

    x0 ∈ R^{C × Rx × Ry}        (C=2 Helmholtz, C=4 elastic wave)

and is conditioned on the normalised angular frequency ω.

Input
-----
FTM checkpoint (from train_FTM_GPU.py) containing EITHER:
  New format (multi-channel):
    "cores"        : list[C] of tensors (B, M, Rx, Ry)
    "channel_names": list[str]
  Old format (C=2 Helmholtz, backward-compatible):
    "cores_real"   : (B, M, Rx, Ry)
    "cores_imag"   : (B, M, Rx, Ry)

Output
------
Diffusion checkpoint with:
  model_state / model_config   — U-Net weights and hyperparameters
  diffusion_config             — noise schedule parameters
  core_stats                   — per-channel mean/std, shape (C, 1, 1)
  channel_names                — list[str] of channel names
  omega_stats                  — {min, max} for ω normalisation
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_omega(
    omega: np.ndarray, omega_min: float, omega_max: float
) -> np.ndarray:
    den = max(omega_max - omega_min, 1e-12)
    return ((omega - omega_min) / den).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (supports old C=2 and new multi-channel checkpoint formats)
# ─────────────────────────────────────────────────────────────────────────────

def load_core_dataset(
    ftm_ckpt: Path,
    max_samples: int = 0,
    max_pairs:   int = 0,
    seed:        int = 42,
) -> Dict[str, Any]:
    """
    Load FTM cores from a checkpoint and prepare (z, omega) pairs for diffusion.

    Returns a dict with:
      z_traj_norm    : (B, M, C, Rx, Ry)  float32  — normalised core trajectories
      z_norm         : (B*M, C, Rx, Ry)   float32  — flattened pairs (for DataLoader)
      omega_grid_norm: (M, 1)             float32  — per-freq normalised ω
      omega_norm     : (B*M, 1)           float32  — pair-level normalised ω
      omega_raw      : (B*M, 1)           float32
      mean           : (C, 1, 1)          float32  — per-channel global mean
      std            : (C, 1, 1)          float32  — per-channel global std
      channel_names  : list[str]
      … (shape metadata)
    """
    if not ftm_ckpt.exists():
        raise FileNotFoundError(f"FTM checkpoint not found: {ftm_ckpt}")

    ckpt = torch.load(ftm_ckpt, map_location="cpu")

    # ── Detect checkpoint format ─────────────────────────────────────────
    if "cores" in ckpt:
        # New multi-channel format
        cores_list: List[np.ndarray] = [
            _to_numpy(c).astype(np.float32) for c in ckpt["cores"]
        ]
        channel_names: List[str] = ckpt.get(
            "channel_names",
            [f"ch{i}" for i in range(len(cores_list))],
        )
    elif "cores_real" in ckpt and "cores_imag" in ckpt:
        # Old C=2 format — backward compatible
        cores_list = [
            _to_numpy(ckpt["cores_real"]).astype(np.float32),
            _to_numpy(ckpt["cores_imag"]).astype(np.float32),
        ]
        channel_names = ["real", "imag"]
    else:
        raise KeyError(
            "FTM checkpoint must contain either 'cores' (new) "
            "or 'cores_real'/'cores_imag' (old)."
        )

    C = len(cores_list)
    # All cores must share the same shape
    shapes = [c.shape for c in cores_list]
    if len(set(shapes)) != 1:
        raise ValueError(f"All cores must have the same shape, got {shapes}")
    if cores_list[0].ndim != 4:
        raise ValueError(f"Expected core shape (B,M,Rx,Ry), got {shapes[0]}")

    B_full, M, Rx, Ry = cores_list[0].shape

    if "omega" not in ckpt:
        raise KeyError("FTM checkpoint missing 'omega'.")
    omega = _to_numpy(ckpt["omega"]).astype(np.float32)
    if omega.shape[0] != M:
        raise ValueError(f"omega length {omega.shape[0]} != core M={M}")

    # ── Optional sub-sampling ────────────────────────────────────────────
    B = min(B_full, max_samples) if max_samples > 0 else B_full
    cores_list = [c[:B] for c in cores_list]

    # ── Stack into (B, M, C, Rx, Ry) ────────────────────────────────────
    # cores_list: C × (B, M, Rx, Ry)  →  stack on axis 2
    z = np.stack(cores_list, axis=2).astype(np.float32)   # (B, M, C, Rx, Ry)

    # ── Per-channel global normalisation ─────────────────────────────────
    # Compute mean/std over (B, M, Rx, Ry) independently for each channel c
    # Result shapes: (C, 1, 1)
    z_bm = z.reshape(B * M, C, Rx, Ry)                    # (B*M, C, Rx, Ry)
    mean = np.mean(z_bm, axis=(0, 2, 3)).astype(np.float32).reshape(C, 1, 1)
    std  = np.std( z_bm, axis=(0, 2, 3)).astype(np.float32).reshape(C, 1, 1)
    std  = np.maximum(std, 1e-6)

    # Normalise: broadcast (C,1,1) over (B, M, C, Rx, Ry)
    z_norm_traj = ((z - mean[None, None, :, :, :])
                   / std[None, None, :, :, :]).astype(np.float32)   # (B,M,C,Rx,Ry)
    z_norm_flat = z_norm_traj.reshape(B * M, C, Rx, Ry)             # (B*M,C,Rx,Ry)

    # ── Omega pairing ─────────────────────────────────────────────────────
    omega_min = float(np.min(omega))
    omega_max = float(np.max(omega))
    omega_grid_norm  = _normalize_omega(omega.reshape(M, 1), omega_min, omega_max)
    omega_pairs_raw  = np.tile(omega.reshape(1, M), (B, 1)).reshape(B * M, 1)
    omega_pairs_norm = _normalize_omega(omega_pairs_raw, omega_min, omega_max)

    # ── Optional random sub-sampling of (sample, freq) pairs ─────────────
    if max_pairs > 0 and max_pairs < B * M:
        rng = np.random.default_rng(seed)
        sel = rng.choice(B * M, size=max_pairs, replace=False)
        z_norm_flat      = z_norm_flat[sel]
        omega_pairs_norm = omega_pairs_norm[sel]
        omega_pairs_raw  = omega_pairs_raw[sel]

    return {
        "z_traj_norm":     z_norm_traj,        # (B, M, C, Rx, Ry)
        "z_norm":          z_norm_flat,         # (B*M or max_pairs, C, Rx, Ry)
        "omega_grid_norm": omega_grid_norm,     # (M, 1)
        "omega_norm":      omega_pairs_norm,    # (N_pairs, 1)
        "omega_raw":       omega_pairs_raw,     # (N_pairs, 1)
        "mean":            mean,                # (C, 1, 1)
        "std":             std,                 # (C, 1, 1)
        "channel_names":   channel_names,
        "norm_mode":       "per_channel_global",
        "omega_min":       omega_min,
        "omega_max":       omega_max,
        "C":               int(C),
        "rx":              int(Rx),
        "ry":              int(Ry),
        "latent_dim":      int(C * Rx * Ry),
        "num_pairs":       int(z_norm_flat.shape[0]),
        "num_samples":     int(B),
        "num_freqs":       int(M),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _group_count(channels: int, max_groups: int = 8) -> int:
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10_000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        if half == 0:
            return t.float().unsqueeze(-1)
        freqs = torch.exp(
            -math.log(float(self.max_period))
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class FourierOmegaEmbedding(nn.Module):
    def __init__(self, num_bands: int = 8):
        super().__init__()
        self.num_bands = num_bands
        self.out_dim   = 1 + 2 * num_bands

    def forward(self, omega_norm: torch.Tensor) -> torch.Tensor:
        feats = [omega_norm]
        for k in range(self.num_bands):
            freq = (2 ** k) * math.pi
            feats.append(torch.sin(freq * omega_norm))
            feats.append(torch.cos(freq * omega_norm))
        return torch.cat(feats, dim=1)


class ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1    = nn.GroupNorm(_group_count(in_ch),  in_ch)
        self.conv1    = nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1)
        self.norm2    = nn.GroupNorm(_group_count(out_ch), out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, 2 * out_ch)
        self.dropout  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip     = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = torch.chunk(self.cond_proj(cond), 2, dim=1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class SelfAttention2D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        heads = max(1, min(num_heads, channels))
        while heads > 1 and channels % heads != 0:
            heads -= 1
        self.norm     = nn.GroupNorm(_group_count(channels), channels)
        self.attn     = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.norm(x).reshape(b, c, h * w).transpose(1, 2)
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return x + self.out_proj(out).transpose(1, 2).reshape(b, c, h, w)


class ConditionalUNet2D(nn.Module):
    """
    Conditional U-Net for score/noise estimation.

    in_channels / out_channels are both set to C (number of core channels).
    For C=2 (Helmholtz) and C=4 (elastic wave) this is the only parameter
    that changes between the two use cases.
    """

    def __init__(
        self,
        in_channels:    int   = 4,     # C
        out_channels:   int   = 4,     # C
        base_channels:  int   = 64,
        cond_dim:       int   = 256,
        time_dim:       int   = 128,
        omega_bands:    int   = 8,
        dropout:        float = 0.0,
        mid_attn_heads: int   = 4,
        mid_attn_dropout: float = 0.0,
    ):
        super().__init__()

        self.time_emb  = SinusoidalTimeEmbedding(time_dim)
        self.omega_emb = FourierOmegaEmbedding(omega_bands)

        cond_in = time_dim + self.omega_emb.out_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.stem  = nn.Conv2d(in_channels, c1, 3, padding=1)

        self.enc1  = ResBlock2D(c1, c1, cond_dim, dropout)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)

        self.enc2  = ResBlock2D(c2, c2, cond_dim, dropout)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)

        self.mid      = ResBlock2D(c3, c3, cond_dim, dropout)
        self.mid_attn = (SelfAttention2D(c3, mid_attn_heads, mid_attn_dropout)
                         if mid_attn_heads > 0 else nn.Identity())

        self.dec2    = ResBlock2D(c3 + c2, c2, cond_dim, dropout)
        self.dec1    = ResBlock2D(c2 + c1, c1, cond_dim, dropout)
        self.out_norm = nn.GroupNorm(_group_count(c1), c1)
        self.out_conv = nn.Conv2d(c1, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,            # (N, C, Rx, Ry)
        t: torch.Tensor,            # (N,)
        omega_norm: torch.Tensor,   # (N, 1)
    ) -> torch.Tensor:
        cond = self.cond_mlp(
            torch.cat([self.time_emb(t), self.omega_emb(omega_norm)], dim=1)
        )
        x0 = self.stem(x)
        e1 = self.enc1(x0, cond)

        d1 = self.down1(e1)
        e2 = self.enc2(d1, cond)

        d2   = self.down2(e2)
        mid  = self.mid_attn(self.mid(d2, cond))

        u2 = self.dec2(
            torch.cat([F.interpolate(mid, e2.shape[-2:], mode="nearest"), e2], dim=1), cond
        )
        u1 = self.dec1(
            torch.cat([F.interpolate(u2,  e1.shape[-2:], mode="nearest"), e1], dim=1), cond
        )
        return self.out_conv(F.silu(self.out_norm(u1)))


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion schedule
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiffusionSchedule:
    betas:                     torch.Tensor
    alphas:                    torch.Tensor
    alpha_bars:                torch.Tensor
    sqrt_alpha_bars:           torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor


def build_linear_schedule(
    num_steps: int, beta_start: float, beta_end: float, device: torch.device
) -> DiffusionSchedule:
    if not (0.0 < beta_start < beta_end < 1.0):
        raise ValueError("Require 0 < beta_start < beta_end < 1")
    betas      = torch.linspace(beta_start, beta_end, num_steps, device=device)
    alphas     = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(
        betas=betas, alphas=alphas, alpha_bars=alpha_bars,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
    )


def build_cosine_schedule(
    num_steps: int, s: float = 0.008, device: torch.device = torch.device("cpu")
) -> DiffusionSchedule:
    t = torch.linspace(0, num_steps, num_steps + 1, device=device) / num_steps
    ab = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    ab = ab / ab[0]
    ab = ab[:-1]
    alphas = torch.zeros_like(ab)
    alphas[1:] = ab[1:] / ab[:-1]; alphas[0] = ab[0]
    betas = 1.0 - alphas
    return DiffusionSchedule(
        betas=betas, alphas=alphas, alpha_bars=ab,
        sqrt_alpha_bars=torch.sqrt(ab),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - ab),
    )


def extract_to_batch(vals: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    out = vals.gather(0, t).to(dtype=x.dtype)
    return out.view(-1, *([1] * (x.ndim - 1)))


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )

    # ── Load core dataset ─────────────────────────────────────────────────
    data = load_core_dataset(
        ftm_ckpt=Path(args.ftm_ckpt),
        max_samples=args.max_samples,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )

    C  = data["C"]    # 2 for Helmholtz, 4 for elastic
    Rx = data["rx"]
    Ry = data["ry"]
    channel_names = data["channel_names"]

    # z_traj_norm: (B, M, C, Rx, Ry) — keep full trajectory for consistency loss
    z_traj_t = torch.from_numpy(data["z_traj_norm"]).float()    # stays on CPU
    omega_grid_t = torch.from_numpy(data["omega_grid_norm"]).float().to(device)  # (M,1)

    dataset = TensorDataset(z_traj_t)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # ── Model ────────────────────────────────────────────────────────────
    base_channels = int(args.hidden_dim) if args.hidden_dim > 0 else int(args.base_channels)

    model = ConditionalUNet2D(
        in_channels=C,          # ← key: C channels not 2
        out_channels=C,
        base_channels=base_channels,
        cond_dim=args.cond_dim,
        time_dim=args.time_dim,
        omega_bands=args.omega_bands,
        dropout=args.dropout,
        mid_attn_heads=args.mid_attn_heads,
        mid_attn_dropout=args.mid_attn_dropout,
    ).to(device)

    # ── Noise schedule ────────────────────────────────────────────────────
    if args.schedule == "linear":
        schedule = build_linear_schedule(
            args.diffusion_steps, args.beta_start, args.beta_end, device
        )
    elif args.schedule == "cosine":
        schedule = build_cosine_schedule(args.diffusion_steps, device=device)
    else:
        raise ValueError("schedule must be 'linear' or 'cosine'")

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.95, eps=1e-12, min_lr=2e-7
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "─" * 72)
    print("Conditional Diffusion Training on FTM Core Tensors (U-Net)")
    print(f"  ftm_ckpt  = {args.ftm_ckpt}")
    print(f"  channels  = {C}  {channel_names}")
    print(f"  core shape: ({C}, {Rx}, {Ry})   latent_dim={data['latent_dim']}")
    print(f"  samples={data['num_samples']}  freqs={data['num_freqs']}  "
          f"pairs={data['num_pairs']}")
    print(f"  norm_mode = {data['norm_mode']}")
    print(f"  device={device}  epochs={args.epochs}  batch={args.batch_size}  "
          f"T={args.diffusion_steps}")
    print(f"  freq_consistency_weight={args.freq_consistency_weight:.2e}")
    print("─" * 72 + "\n")

    best_loss    = float("inf")
    loss_history: List[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_batches = 0.0, 0

        for (x0_traj,) in loader:
            # x0_traj: (Bz, M, C, Rx, Ry)
            x0_traj = x0_traj.to(device, non_blocking=True)
            bsz, M, _C, _Rx, _Ry = x0_traj.shape

            # One diffusion timestep per sample, shared across frequency axis.
            t_sample = torch.randint(0, args.diffusion_steps, (bsz,), device=device)
            t = t_sample[:, None].expand(-1, M).reshape(-1)        # (bsz*M,)

            x0    = x0_traj.reshape(bsz * M, C, Rx, Ry)
            noise = torch.randn_like(x0)

            sqrt_ab  = extract_to_batch(schedule.sqrt_alpha_bars,           t, x0)
            sqrt_omb = extract_to_batch(schedule.sqrt_one_minus_alpha_bars,  t, x0)
            xt = sqrt_ab * x0 + sqrt_omb * noise

            omega_norm = omega_grid_t.unsqueeze(0).expand(bsz, -1, -1) \
                                     .reshape(bsz * M, 1)

            pred_noise = model(xt, t, omega_norm)
            noise_loss = F.mse_loss(pred_noise, noise)

            # Frequency-trajectory consistency on recovered x0_hat
            x0_hat      = (xt - sqrt_omb * pred_noise) / sqrt_ab.clamp(min=1e-12)
            x0_hat_traj = x0_hat.reshape(bsz, M, C, Rx, Ry)

            if M > 1 and args.freq_consistency_weight > 0:
                diff_true = x0_traj[:, 1:] - x0_traj[:, :-1]
                diff_pred = x0_hat_traj[:, 1:] - x0_hat_traj[:, :-1]
                fc_loss   = F.mse_loss(diff_pred, diff_true)
            else:
                fc_loss   = x0.new_zeros(())

            loss = noise_loss + args.freq_consistency_weight * fc_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running  += float(loss.item())
            n_batches += 1

        epoch_loss = running / max(n_batches, 1)
        loss_history.append(epoch_loss)
        best_loss = min(best_loss, epoch_loss)
        scheduler.step(epoch_loss)

        if args.log_every > 0 and (
            epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs
        ):
            print(
                f"[epoch {epoch:04d}/{args.epochs}]  "
                f"loss={epoch_loss:.6e}  best={best_loss:.6e}  "
                f"noise={float(noise_loss.item()):.4e}  "
                f"fc={float(fc_loss.item()):.4e}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    # ── Save checkpoint ───────────────────────────────────────────────────
    ckpt_out: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "model_config": {
            "in_channels":     C,
            "out_channels":    C,
            "base_channels":   base_channels,
            "cond_dim":        args.cond_dim,
            "time_dim":        args.time_dim,
            "omega_bands":     args.omega_bands,
            "dropout":         args.dropout,
            "mid_attn_heads":  args.mid_attn_heads,
            "mid_attn_dropout": args.mid_attn_dropout,
        },
        "diffusion_config": {
            "num_steps":  args.diffusion_steps,
            "beta_start": args.beta_start,
            "beta_end":   args.beta_end,
            "schedule":   args.schedule,
        },
        # Per-channel normalisation stats — shape (C, 1, 1)
        "core_stats": {
            "mean":       torch.from_numpy(data["mean"]),   # (C, 1, 1)
            "std":        torch.from_numpy(data["std"]),    # (C, 1, 1)
            "C":          C,
            "rx":         Rx,
            "ry":         Ry,
            "latent_dim": data["latent_dim"],
            "layout":     f"{C}ch_image",
            "norm_mode":  data["norm_mode"],
        },
        "channel_names": channel_names,
        "omega_stats": {
            "min": data["omega_min"],
            "max": data["omega_max"],
        },
        "train_info": {
            "num_pairs":               data["num_pairs"],
            "num_samples":             data["num_samples"],
            "num_freqs":               data["num_freqs"],
            "freq_consistency_weight": args.freq_consistency_weight,
            "best_loss":               best_loss,
            "final_loss":              loss_history[-1] if loss_history else 0.0,
        },
        "loss_history": loss_history,
        "ftm_ckpt": str(args.ftm_ckpt),
        "config":   vars(args),
    }
    torch.save(ckpt_out, out_path)

    summary = {
        "out":          str(out_path),
        "C":            C,
        "channel_names": channel_names,
        "core_shape":   [C, Rx, Ry],
        "norm_mode":    data["norm_mode"],
        "latent_dim":   data["latent_dim"],
        "best_loss":    best_loss,
        "final_loss":   loss_history[-1] if loss_history else 0.0,
        "epochs":       args.epochs,
    }
    summary_path = out_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print("\nTraining finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved checkpoint : {out_path}")
    print(f"Saved summary    : {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train conditional diffusion on FTM core tensors "
                    "(C=2 Helmholtz or C=4 elastic wave)"
    )
    p.add_argument("--ftm_ckpt",  type=str,   default="ckp/ftm_gpu_checkpoint.pt")
    p.add_argument("--out",       type=str,   default="ckp/diffusion_core.pt")

    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--max_pairs",   type=int, default=0)

    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--num_workers", type=int,   default=16)

    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip",    type=float, default=0.0)

    p.add_argument("--base_channels",    type=int,   default=128)
    p.add_argument("--cond_dim",         type=int,   default=512)
    p.add_argument("--time_dim",         type=int,   default=256)
    p.add_argument("--omega_bands",      type=int,   default=0)
    p.add_argument("--dropout",          type=float, default=0.0)
    p.add_argument("--mid_attn_heads",   type=int,   default=5)
    p.add_argument("--mid_attn_dropout", type=float, default=0.0)

    # Compat args (accepted, not used)
    p.add_argument("--hidden_dim", type=int, default=0)
    p.add_argument("--depth",      type=int, default=0)

    p.add_argument("--diffusion_steps", type=int,   default=500)
    p.add_argument("--beta_start",      type=float, default=1e-4)
    p.add_argument("--beta_end",        type=float, default=2e-2)
    p.add_argument("--schedule",        type=str,   default="linear",
                   choices=["linear", "cosine"])
    p.add_argument("--freq_consistency_weight", type=float, default=0.00)

    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--device",    type=str, default="auto")
    p.add_argument("--log_every", type=int, default=5)
    return p


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()