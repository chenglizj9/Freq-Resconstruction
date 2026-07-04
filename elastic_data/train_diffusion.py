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


def _split_indices(num_samples: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    ids = rng.permutation(num_samples)
    if num_samples <= 1:
        train_ids = ids.tolist()
        val_ids = ids.tolist()
    else:
        n_train = int(round(num_samples * train_ratio))
        n_train = max(1, min(n_train, num_samples - 1))
        train_ids = ids[:n_train].tolist()
        val_ids = ids[n_train:].tolist()
    return train_ids, val_ids


def _split_indices(num_samples: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    ids = rng.permutation(num_samples)
    if num_samples <= 1:
        train_ids = ids.tolist()
        val_ids = ids.tolist()
    else:
        n_train = int(round(num_samples * train_ratio))
        n_train = max(1, min(n_train, num_samples - 1))
        train_ids = ids[:n_train].tolist()
        val_ids = ids[n_train:].tolist()
    return train_ids, val_ids


# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (supports old C=2 and new multi-channel checkpoint formats)
# ─────────────────────────────────────────────────────────────────────────────

def load_core_dataset(
    ftm_ckpt: Path,
    max_samples: int = 0,
    max_pairs:   int = 0,
    seed:        int = 42,
    sample_indices: Optional[List[int]] = None,
    freq_indices: Optional[List[int]] = None,
    norm_stats: Optional[Dict[str, np.ndarray]] = None,
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

    if sample_indices is not None:
        sample_indices = [int(i) for i in sample_indices]
        if not sample_indices:
            raise ValueError("sample_indices must not be empty")
        if min(sample_indices) < 0 or max(sample_indices) >= B_full:
            raise ValueError("sample_indices out of range")
    elif max_samples > 0:
        B = min(B_full, max_samples)
        sample_indices = list(range(B))
    else:
        sample_indices = list(range(B_full))

    if freq_indices is not None:
        freq_indices = [int(i) for i in freq_indices]
        if not freq_indices:
            raise ValueError("freq_indices must not be empty")
        if min(freq_indices) < 0 or max(freq_indices) >= M:
            raise ValueError("freq_indices out of range")
    else:
        freq_indices = list(range(M))

    # ── Select sample/frequency subset ─────────────────────────────────────
    cores_list = [c[np.asarray(sample_indices)][:, np.asarray(freq_indices)] for c in cores_list]
    B = len(sample_indices)
    M = len(freq_indices)

    if "omega" not in ckpt:
        raise KeyError("FTM checkpoint missing 'omega'.")
    omega_all = _to_numpy(ckpt["omega"]).astype(np.float32)
    if omega_all.shape[0] != shapes[0][1]:
        raise ValueError(f"omega length {omega_all.shape[0]} != core M={shapes[0][1]}")
    omega = omega_all[np.asarray(freq_indices)]

    # ── Stack into (B, M, C, Rx, Ry) ────────────────────────────────────
    # cores_list: C × (B, M, Rx, Ry)  →  stack on axis 2
    z = np.stack(cores_list, axis=2).astype(np.float32)   # (B, M, C, Rx, Ry)
    # Compute mean/std over (B, M, Rx, Ry) independently for each channel c
    # Result shapes: (C, 1, 1)
    z_bm = z.reshape(B * M, C, Rx, Ry)                    # (B*M, C, Rx, Ry)
    if norm_stats is None:
        mean = np.mean(z_bm, axis=(0, 2, 3)).astype(np.float32).reshape(C, 1, 1)
        std = np.std(z_bm, axis=(0, 2, 3)).astype(np.float32).reshape(C, 1, 1)
        std = np.maximum(std, 1e-6)
    else:
        mean = np.asarray(norm_stats["mean"], dtype=np.float32).reshape(C, 1, 1)
        std = np.asarray(norm_stats["std"], dtype=np.float32).reshape(C, 1, 1)
        std = np.maximum(std, 1e-6)

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

    all_meta = load_core_dataset(
        ftm_ckpt=Path(args.ftm_ckpt),
        max_samples=args.max_samples,
        seed=args.seed,
    )
    train_sample_ids, val_sample_ids = _split_indices(
        all_meta["num_samples"], args.train_ratio, args.seed
    )

    train_data = load_core_dataset(
        ftm_ckpt=Path(args.ftm_ckpt),
        max_samples=args.max_samples,
        max_pairs=args.max_pairs,
        seed=args.seed,
        sample_indices=train_sample_ids,
    )
    val_data = load_core_dataset(
        ftm_ckpt=Path(args.ftm_ckpt),
        max_samples=args.max_samples,
        seed=args.seed,
        sample_indices=val_sample_ids,
        norm_stats={"mean": train_data["mean"], "std": train_data["std"]},
    )

    C  = train_data["C"]
    Rx = train_data["rx"]
    Ry = train_data["ry"]
    channel_names = train_data["channel_names"]

    x_train = torch.from_numpy(train_data["z_norm"]).float()
    w_train = torch.from_numpy(train_data["omega_norm"]).float()
    x_val = torch.from_numpy(val_data["z_norm"]).float()
    w_val = torch.from_numpy(val_data["omega_norm"]).float()

    train_loader = DataLoader(
        TensorDataset(x_train, w_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, w_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    base_channels = int(args.hidden_dim) if args.hidden_dim > 0 else int(args.base_channels)

    model = ConditionalUNet2D(
        in_channels=C,
        out_channels=C,
        base_channels=base_channels,
        cond_dim=args.cond_dim,
        time_dim=args.time_dim,
        omega_bands=args.omega_bands,
        dropout=args.dropout,
        mid_attn_heads=args.mid_attn_heads,
        mid_attn_dropout=args.mid_attn_dropout,
    ).to(device)

    if args.schedule == "linear":
        schedule = build_linear_schedule(
            args.diffusion_steps, args.beta_start, args.beta_end, device
        )
    elif args.schedule == "cosine":
        schedule = build_cosine_schedule(args.diffusion_steps, device=device)
    else:
        raise ValueError("schedule must be 'linear' or 'cosine'")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=10,
        factor=args.lr_plateau_factor,
        eps=1e-12,
        min_lr=args.min_lr,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "─" * 72)
    print("Conditional Diffusion Training on FTM Core Tensors (U-Net)")
    print(f"  ftm_ckpt  = {args.ftm_ckpt}")
    print(f"  channels  = {C}  {channel_names}")
    print(f"  core shape: ({C}, {Rx}, {Ry})   latent_dim={train_data['latent_dim']}")
    print(
        f"  train samples={train_data['num_samples']}  val samples={val_data['num_samples']}  "
        f"train pairs={train_data['num_pairs']}  val pairs={val_data['num_pairs']}"
    )
    print(f"  norm_mode = {train_data['norm_mode']}")
    print(f"  device={device}  epochs={args.epochs}  batch={args.batch_size}  T={args.diffusion_steps}")
    print(f"  freq_consistency_weight={args.freq_consistency_weight:.2e}")
    print("─" * 72 + "\n")

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    loss_history: List[float] = []
    val_history: List[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_running = 0.0
        train_batches = 0
        last_noise_loss = 0.0
        last_fc_loss = 0.0

        for x0, omega_norm in train_loader:
            x0 = x0.to(device, non_blocking=True)
            omega_norm = omega_norm.to(device, non_blocking=True)
            bsz = x0.shape[0]

            t = torch.randint(0, args.diffusion_steps, (bsz,), device=device)
            noise = torch.randn_like(x0)

            sqrt_ab = extract_to_batch(schedule.sqrt_alpha_bars, t, x0)
            sqrt_omb = extract_to_batch(schedule.sqrt_one_minus_alpha_bars, t, x0)
            xt = sqrt_ab * x0 + sqrt_omb * noise

            pred_noise = model(xt, t, omega_norm)
            noise_loss = F.mse_loss(pred_noise, noise)

            if args.freq_consistency_weight > 0:
                x0_hat = (xt - sqrt_omb * pred_noise) / sqrt_ab.clamp(min=1e-12)
                fc_loss = F.mse_loss(x0_hat, x0)
            else:
                fc_loss = x0.new_zeros(())

            loss = noise_loss + args.freq_consistency_weight * fc_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_running += float(loss.item()) * bsz
            train_batches += bsz
            last_noise_loss = float(noise_loss.item())
            last_fc_loss = float(fc_loss.item())

        train_loss = train_running / max(train_batches, 1)
        loss_history.append(train_loss)
        best_train_loss = min(best_train_loss, train_loss)

        model.eval()
        val_running = 0.0
        val_batches = 0
        with torch.no_grad():
            for x0, omega_norm in val_loader:
                x0 = x0.to(device, non_blocking=True)
                omega_norm = omega_norm.to(device, non_blocking=True)
                bsz = x0.shape[0]

                t = torch.randint(0, args.diffusion_steps, (bsz,), device=device)
                noise = torch.randn_like(x0)
                sqrt_ab = extract_to_batch(schedule.sqrt_alpha_bars, t, x0)
                sqrt_omb = extract_to_batch(schedule.sqrt_one_minus_alpha_bars, t, x0)
                xt = sqrt_ab * x0 + sqrt_omb * noise
                pred_noise = model(xt, t, omega_norm)
                val_loss = F.mse_loss(pred_noise, noise)

                val_running += float(val_loss.item()) * bsz
                val_batches += bsz

        epoch_val_loss = val_running / max(val_batches, 1)
        val_history.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            ckpt_out: Dict[str, Any] = {
                "model_state": model.state_dict(),
                "model_config": {
                    "in_channels": C,
                    "out_channels": C,
                    "base_channels": base_channels,
                    "cond_dim": args.cond_dim,
                    "time_dim": args.time_dim,
                    "omega_bands": args.omega_bands,
                    "dropout": args.dropout,
                    "mid_attn_heads": args.mid_attn_heads,
                    "mid_attn_dropout": args.mid_attn_dropout,
                },
                "diffusion_config": {
                    "num_steps": args.diffusion_steps,
                    "beta_start": args.beta_start,
                    "beta_end": args.beta_end,
                    "schedule": args.schedule,
                },
                "core_stats": {
                    "mean": torch.from_numpy(train_data["mean"]),
                    "std": torch.from_numpy(train_data["std"]),
                    "C": C,
                    "rx": Rx,
                    "ry": Ry,
                    "latent_dim": train_data["latent_dim"],
                    "layout": f"{C}ch_image",
                    "norm_mode": train_data["norm_mode"],
                },
                "channel_names": channel_names,
                "omega_stats": {
                    "min": train_data["omega_min"],
                    "max": train_data["omega_max"],
                },
                "train_info": {
                    "num_pairs": train_data["num_pairs"],
                    "num_samples": train_data["num_samples"],
                    "num_freqs": train_data["num_freqs"],
                    "val_num_pairs": val_data["num_pairs"],
                    "val_num_samples": val_data["num_samples"],
                    "val_num_freqs": val_data["num_freqs"],
                    "freq_consistency_weight": args.freq_consistency_weight,
                    "best_train_loss": best_train_loss,
                    "best_val_loss": best_val_loss,
                    "final_loss": train_loss,
                    "final_val_loss": epoch_val_loss,
                    "train_ratio": args.train_ratio,
                },
                "loss_history": loss_history,
                "val_history": val_history,
                "ftm_ckpt": str(args.ftm_ckpt),
                "config": vars(args),
            }
            torch.save(ckpt_out, out_path)

        if args.log_every > 0 and (
            epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs
        ):
            print(
                f"[epoch {epoch:04d}/{args.epochs}]  "
                f"train={train_loss:.6e}  val={epoch_val_loss:.6e}  best_val={best_val_loss:.6e}  "
                f"noise={last_noise_loss:.4e}  fc={last_fc_loss:.4e}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    summary = {
        "out": str(out_path),
        "C": C,
        "channel_names": channel_names,
        "core_shape": [C, Rx, Ry],
        "norm_mode": train_data["norm_mode"],
        "latent_dim": train_data["latent_dim"],
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "final_val_loss": val_history[-1] if val_history else 0.0,
        "epochs": args.epochs,
        "train_ratio": args.train_ratio,
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

    p.add_argument("--epochs",      type=int,   default=500)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--num_workers", type=int,   default=4)

    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip",    type=float, default=1.0)

    p.add_argument("--base_channels",    type=int,   default=128)
    p.add_argument("--cond_dim",         type=int,   default=256)
    p.add_argument("--time_dim",         type=int,   default=128)
    p.add_argument("--omega_bands",      type=int,   default=8)
    p.add_argument("--dropout",          type=float, default=0.0)
    p.add_argument("--mid_attn_heads",   type=int,   default=0)
    p.add_argument("--mid_attn_dropout", type=float, default=0.0)

    # Compat args (accepted, not used)
    p.add_argument("--hidden_dim", type=int, default=0)
    p.add_argument("--depth",      type=int, default=0)

    p.add_argument("--diffusion_steps", type=int,   default=500)
    p.add_argument("--beta_start",      type=float, default=1e-4)
    p.add_argument("--beta_end",        type=float, default=2e-2)
    p.add_argument("--schedule",        type=str,   default="linear",
                   choices=["linear", "cosine"])
    p.add_argument("--train_ratio",     type=float, default=0.8)
    p.add_argument("--freq_consistency_weight", type=float, default=0.00)
    p.add_argument("--lr_plateau_factor", type=float, default=0.5)
    p.add_argument("--min_lr", type=float, default=2e-7)

    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--device",    type=str, default="auto")
    p.add_argument("--log_every", type=int, default=5)
    return p


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()