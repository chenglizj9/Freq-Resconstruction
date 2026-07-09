"""
train_diffusion.py
------------------
Train a conditional diffusion model p(G | omega) on non-flattened FTM cores.

Each training sample is a 2-channel core image:
  x0 in R^{2 x Rx x Ry}
where channel 0 is real core and channel 1 is imaginary core.

Input
-----
An FTM checkpoint (from train_FTM_GPU.py or compatible) containing:
- cores_real: (B, M, Rx, Ry)
- cores_imag: (B, M, Rx, Ry)
- omega:      (M,)

Output
------
Diffusion checkpoint containing:
- model_state / model_config
- diffusion_config
- core normalization stats (mean/std, shape 2 x Rx x Ry)
- omega normalization range
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


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


def _normalize_omega(omega: np.ndarray, omega_min: float, omega_max: float) -> np.ndarray:
    den = max(omega_max - omega_min, 1e-12)
    return ((omega - omega_min) / den).astype(np.float32)


def load_core_dataset(
    ftm_ckpt: Path,
    max_samples: int = 0,
    max_pairs: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    if not ftm_ckpt.exists():
        raise FileNotFoundError(f"FTM checkpoint not found: {ftm_ckpt}")

    ckpt = torch.load(ftm_ckpt, map_location="cpu")
    if "cores_real" not in ckpt or "cores_imag" not in ckpt:
        raise KeyError("FTM checkpoint must contain 'cores_real' and 'cores_imag'.")

    cores_real = _to_numpy(ckpt["cores_real"]).astype(np.float32)
    cores_imag = _to_numpy(ckpt["cores_imag"]).astype(np.float32)

    if cores_real.shape != cores_imag.shape:
        raise ValueError(
            f"cores_real / cores_imag shape mismatch: {cores_real.shape} vs {cores_imag.shape}"
        )

    if cores_real.ndim != 4:
        raise ValueError(f"Expected core shape (B,M,Rx,Ry), got {cores_real.shape}")

    b, m, rx, ry = cores_real.shape

    if "omega" not in ckpt:
        raise KeyError("FTM checkpoint missing omega.")
    omega = _to_numpy(ckpt["omega"]).astype(np.float32)
    if omega.ndim != 1 or omega.shape[0] != m:
        raise ValueError(f"Invalid omega shape {omega.shape}, expected ({m},)")

    if max_samples > 0:
        b = min(b, max_samples)
        cores_real = cores_real[:b]
        cores_imag = cores_imag[:b]

    # z shape: (B, M, 2, Rx, Ry)
    z = np.stack([cores_real, cores_imag], axis=2).astype(np.float32)
    z_pairs = z.reshape(b * m, 2, rx, ry).astype(np.float32)

    omega_grid = omega.reshape(m, 1).astype(np.float32)
    omega_pairs = np.tile(omega_grid.reshape(1, m, 1), (b, 1, 1)).reshape(b * m, 1).astype(np.float32)

    if max_pairs > 0 and max_pairs < z_pairs.shape[0]:
        rng = np.random.default_rng(seed)
        sel = rng.choice(z_pairs.shape[0], size=max_pairs, replace=False)
        z_pairs = z_pairs[sel]
        omega_pairs = omega_pairs[sel]

    # Channel-global normalization: mean/std are shared across all spatial positions.
    # Shapes: (2, 1, 1) for real/imag channels.
    mean = np.mean(z_pairs, axis=(0, 2, 3), keepdims=False).astype(np.float32).reshape(2, 1, 1)
    std = np.std(z_pairs, axis=(0, 2, 3), keepdims=False).astype(np.float32).reshape(2, 1, 1)
    std = np.maximum(std, 1e-6)

    omega_min = float(np.min(omega))
    omega_max = float(np.max(omega))
    # omega_max = 71.0
    omega_grid_norm = _normalize_omega(omega_grid, omega_min, omega_max)
    omega_pairs_norm = _normalize_omega(omega_pairs, omega_min, omega_max)

    z_traj_norm = ((z - mean[None, None, ...]) / std[None, None, ...]).astype(np.float32)
    z_norm = z_traj_norm.reshape(b * m, 2, rx, ry).astype(np.float32)

    return {
        "z_traj_norm": z_traj_norm,
        "z_norm": z_norm,
        "omega_grid_norm": omega_grid_norm,
        "omega_norm": omega_pairs_norm,
        "omega_raw": omega_pairs,
        "mean": mean,
        "std": std,
        "norm_mode": "channel_global",
        "omega_min": omega_min,
        "omega_max": omega_max,
        "rx": int(rx),
        "ry": int(ry),
        "latent_dim": int(2 * rx * ry),
        "num_pairs": int(z_norm.shape[0]),
        "num_samples": int(b),
        "num_freqs": int(m),
    }


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
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
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class FourierOmegaEmbedding(nn.Module):
    def __init__(self, num_bands: int = 8):
        super().__init__()
        self.num_bands = num_bands
        self.out_dim = 1 + 2 * num_bands
        # self.out_dim = 1 + 2 * num_bands
        # self.out_dim = 0

    def forward(self, omega_norm: torch.Tensor) -> torch.Tensor:
        feats = [omega_norm]
        # feats.append(omega_norm ** 2)
        for k in range(self.num_bands):
            freq = (k) * math.pi
            feats.append(torch.sin(freq * omega_norm))
            feats.append(torch.cos(freq * omega_norm))
        return torch.cat(feats, dim=1)


def _group_count(channels: int, max_groups: int = 8) -> int:
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


class ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(_group_count(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.cond_proj = nn.Linear(cond_dim, 2 * out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        h = self.norm2(h)
        scale_shift = self.cond_proj(cond)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        h = h * (1.0 + scale) + shift

        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class SelfAttention2D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")

        heads = max(1, min(int(num_heads), channels))
        while heads > 1 and channels % heads != 0:
            heads -= 1

        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> tokens: (B, H*W, C)
        b, c, h, w = x.shape
        tokens = self.norm(x).reshape(b, c, h * w).transpose(1, 2)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        attn_out = self.out_proj(attn_out)
        attn_out = attn_out.transpose(1, 2).reshape(b, c, h, w)
        return x + attn_out


class ConditionalUNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 64,
        cond_dim: int = 256,
        time_dim: int = 128,
        omega_bands: int = 8,
        dropout: float = 0.0,
        mid_attn_heads: int = 4,
        mid_attn_dropout: float = 0.0,
    ):
        super().__init__()

        self.time_emb = SinusoidalTimeEmbedding(dim=time_dim)
        self.omega_emb = FourierOmegaEmbedding(num_bands=omega_bands)

        cond_in = time_dim + self.omega_emb.out_dim
        # cond_in = time_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.stem = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        self.enc1 = ResBlock2D(c1, c1, cond_dim=cond_dim, dropout=dropout)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)

        self.enc2 = ResBlock2D(c2, c2, cond_dim=cond_dim, dropout=dropout)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)

        self.mid = ResBlock2D(c3, c3, cond_dim=cond_dim, dropout=dropout)
        if mid_attn_heads > 0:
            self.mid_attn = SelfAttention2D(c3, num_heads=mid_attn_heads, dropout=mid_attn_dropout)
        else:
            self.mid_attn = nn.Identity()

        self.dec2 = ResBlock2D(c3 + c2, c2, cond_dim=cond_dim, dropout=dropout)
        self.dec1 = ResBlock2D(c2 + c1, c1, cond_dim=cond_dim, dropout=dropout)

        self.out_norm = nn.GroupNorm(_group_count(c1), c1)
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, omega_norm: torch.Tensor) -> torch.Tensor:
        cond = self.cond_mlp(torch.cat([self.time_emb(t), self.omega_emb(omega_norm)], dim=1))
        # cond = self.cond_mlp(self.time_emb(t))

        x0 = self.stem(x)
        e1 = self.enc1(x0, cond)

        d1 = self.down1(e1)
        e2 = self.enc2(d1, cond)

        d2 = self.down2(e2)
        m = self.mid(d2, cond)
        m = self.mid_attn(m)

        u2 = F.interpolate(m, size=e2.shape[-2:], mode="nearest")
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec2(u2, cond)

        u1 = F.interpolate(u2, size=e1.shape[-2:], mode="nearest")
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.dec1(u1, cond)

        out = self.out_conv(F.silu(self.out_norm(u1)))
        return out


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor


def build_linear_schedule(
    num_steps: int,
    beta_start: float,
    beta_end: float,
    device: torch.device,
) -> DiffusionSchedule:
    if num_steps <= 1:
        raise ValueError("num_steps must be > 1")
    if not (0.0 < beta_start < beta_end < 1.0):
        raise ValueError("Require 0 < beta_start < beta_end < 1")

    betas = torch.linspace(beta_start, beta_end, num_steps, device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
    )

def build_cosine_schedule(
    num_steps: int,
    s: float = 0.008,
    device: torch.device = torch.device("cpu"),
) -> DiffusionSchedule:
    steps = num_steps + 1
    t = torch.linspace(0, num_steps, steps, device=device) / num_steps
    alpha_bars = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]
    alpha_bars = alpha_bars[:-1]

    alphas = torch.zeros_like(alpha_bars)
    alphas[1:] = alpha_bars[1:] / alpha_bars[:-1]
    alphas[0] = alpha_bars[0]
    betas = 1.0 - alphas

    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
    )
def extract_to_batch(vals: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # vals: (T,), t: (N,), output shape: (N, 1, 1, 1) for x=(N,C,H,W)
    out = vals.gather(0, t).to(dtype=x.dtype)
    return out.view(-1, *([1] * (x.ndim - 1)))


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    data = load_core_dataset(
        ftm_ckpt=Path(args.ftm_ckpt),
        max_samples=args.max_samples,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )

    z_traj_t = torch.from_numpy(data["z_traj_norm"]).float()   # (B,M,2,Rx,Ry)
    omega_grid_t = torch.from_numpy(data["omega_grid_norm"]).float().to(device)  # (M,1)
    dataset = TensorDataset(z_traj_t)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    base_channels = int(args.base_channels)
    if int(args.hidden_dim) > 0:
        base_channels = int(args.hidden_dim)

    model = ConditionalUNet2D(
        in_channels=2,
        out_channels=2,
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
            num_steps=args.diffusion_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            device=device,
        )
    elif args.schedule == "cosine":
        schedule = build_cosine_schedule(
            num_steps=args.diffusion_steps,
            device=device
        )
    else:
        raise ValueError("schedule must be linear or cosine")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.95, eps=1e-12, min_lr=2e-7)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    loss_history = []

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 72)
    print("Conditional Diffusion Training on 2D FTM Cores (U-Net)")
    print(f"ftm_ckpt={args.ftm_ckpt}")
    print(
        f"pairs={data['num_pairs']}, core_shape=(2,{data['rx']},{data['ry']}), "
        f"latent_dim={data['latent_dim']}"
    )
    print(f"normalization={data['norm_mode']}")
    print(f"samples={data['num_samples']}, freqs={data['num_freqs']}")
    print(f"freq_consistency_weight={args.freq_consistency_weight:.2e}")
    print(
        f"device={device}, epochs={args.epochs}, batch_size={args.batch_size}, "
        f"diffusion_steps={args.diffusion_steps}"
    )
    print("-" * 72 + "\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0

        for (x0_traj,) in loader:
            # x0_traj: (Bz, M, 2, Rx, Ry)
            x0_traj = x0_traj.to(device, non_blocking=True)

            bsz, m, c, rx, ry = x0_traj.shape

            # Use one diffusion timestep per sample, shared across that sample's frequency trajectory.
            t_sample = torch.randint(0, args.diffusion_steps, (bsz,), device=device, dtype=torch.long)
            t = t_sample[:, None].expand(-1, m).reshape(-1)

            x0 = x0_traj.reshape(bsz * m, c, rx, ry)
            noise = torch.randn_like(x0)

            sqrt_ab = extract_to_batch(schedule.sqrt_alpha_bars, t, x0)
            sqrt_omb = extract_to_batch(schedule.sqrt_one_minus_alpha_bars, t, x0)
            xt = sqrt_ab * x0 + sqrt_omb * noise

            omega_norm = omega_grid_t.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * m, 1)

            pred_noise = model(xt, t, omega_norm)
            noise_loss = F.mse_loss(pred_noise, noise)

            # Recover x0 estimate and enforce frequency-trajectory consistency.
            x0_hat = (xt - sqrt_omb * pred_noise) / torch.clamp(sqrt_ab, min=1e-12)
            x0_hat_traj = x0_hat.reshape(bsz, m, c, rx, ry)

            if m > 1:
                diff_true = x0_traj[:, 1:, ...] - x0_traj[:, :-1, ...]
                diff_pred = x0_hat_traj[:, 1:, ...] - x0_hat_traj[:, :-1, ...]
                freq_consistency_loss = F.mse_loss(diff_pred, diff_true)
            else:
                freq_consistency_loss = torch.zeros((), dtype=x0.dtype, device=x0.device)

            loss = noise_loss
            # loss = noise_loss + args.freq_consistency_weight * freq_consistency_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            running += float(loss.item())
            n_batches += 1

        epoch_loss = running / max(n_batches, 1)
        loss_history.append(epoch_loss)
        best_loss = min(best_loss, epoch_loss)

        # Update the learning rate scheduler
        # scheduler.step(epoch_loss)

        if args.log_every > 0 and (epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs):
            print(
                f"[epoch {epoch:04d}/{args.epochs}] "
                f"loss={epoch_loss:.6e} best={best_loss:.6e} "
                f"lr={optimizer.param_groups[0]['lr']:.6e}"
            )

    ckpt_out: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "model_config": {
            "in_channels": 2,
            "out_channels": 2,
            "base_channels": int(base_channels),
            "cond_dim": int(args.cond_dim),
            "time_dim": int(args.time_dim),
            "omega_bands": int(args.omega_bands),
            "dropout": float(args.dropout),
            "mid_attn_heads": int(args.mid_attn_heads),
            "mid_attn_dropout": float(args.mid_attn_dropout),
        },
        "diffusion_config": {
            "num_steps": int(args.diffusion_steps),
            "beta_start": float(args.beta_start),
            "beta_end": float(args.beta_end),
            "schedule": str(args.schedule),
        },
        "core_stats": {
            "mean": torch.from_numpy(data["mean"]),  # (2,1,1)
            "std": torch.from_numpy(data["std"]),    # (2,1,1)
            "rx": int(data["rx"]),
            "ry": int(data["ry"]),
            "latent_dim": int(data["latent_dim"]),
            "layout": "2ch_image",
            "norm_mode": data["norm_mode"],
        },
        "omega_stats": {
            "min": float(data["omega_min"]),
            "max": float(data["omega_max"]),
        },
        "train_info": {
            "num_pairs": int(data["num_pairs"]),
            "num_samples": int(data["num_samples"]),
            "num_freqs": int(data["num_freqs"]),
            "freq_consistency_weight": float(args.freq_consistency_weight),
            "best_loss": float(best_loss),
            "final_loss": float(loss_history[-1] if loss_history else 0.0),
        },
        "loss_history": loss_history,
        "ftm_ckpt": str(args.ftm_ckpt),
        "config": vars(args),
    }
    torch.save(ckpt_out, out_path)

    summary = {
        "out": str(out_path),
        "pairs": int(data["num_pairs"]),
        "core_shape": [2, int(data["rx"]), int(data["ry"])],
        "norm_mode": data["norm_mode"],
        "latent_dim": int(data["latent_dim"]),
        "best_loss": float(best_loss),
        "final_loss": float(loss_history[-1] if loss_history else 0.0),
        "epochs": int(args.epochs),
    }
    summary_path = out_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nTraining finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved checkpoint: {out_path}")
    print(f"Saved summary:   {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train conditional U-Net diffusion on 2D FTM core tensors")

    p.add_argument("--ftm_ckpt", type=str, default="heat_data/ckp/ftm_gpu_checkpoint.pt")
    p.add_argument("--out", type=str, default="heat_data/ckp/diffusion_core.pt")

    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--max_pairs", type=int, default=0)

    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=0.0)

    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--time_dim", type=int, default=128)
    p.add_argument("--omega_bands", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--mid_attn_heads", type=int, default=4)
    p.add_argument("--mid_attn_dropout", type=float, default=0.0)

    # Deprecated/compat options: accepted so old command lines do not fail.
    p.add_argument("--hidden_dim", type=int, default=0)
    p.add_argument("--depth", type=int, default=0)

    p.add_argument("--diffusion_steps", type=int, default=500)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)
    p.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"])
    p.add_argument("--freq_consistency_weight", type=float, default=0.05)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--log_every", type=int, default=5)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
