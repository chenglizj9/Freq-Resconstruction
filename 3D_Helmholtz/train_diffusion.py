"""
train_diffusion.py  (3D Helmholtz edition)
-------------------------------------------
Train a conditional diffusion model p(G | omega) on 3D FTM core tensors.

Each training sample is a flattened core pair:
  x0 in R^{2*Rx*Ry*Rz}   (channel 0 = real core, channel 1 = imag core)

Input (FTM checkpoint from train_FTM_GPU.py):
  - cores : list[2] of tensors (B, M, Rx, Ry, Rz)
  - omega : (M,)

Output (diffusion checkpoint):
  - model_state / model_config
  - diffusion_config
  - core normalization stats (mean/std, shape 2*R)
  - omega normalization range

Architecture: 3D U-Net score network with residual blocks and FiLM conditioning.
Input (B, latent_dim) is reshaped to (B, 2, Rx, Ry, Rz) for volumetric convolutions,
then flattened back. Down/up-sampling levels are auto-capped by min(Rx,Ry,Rz).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_omega(omega: np.ndarray, omega_min: float, omega_max: float) -> np.ndarray:
    den = max(omega_max - omega_min, 1e-12)
    return ((omega - omega_min) / den).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────

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

    def forward(self, omega_norm: torch.Tensor) -> torch.Tensor:
        feats = [omega_norm]
        for k in range(self.num_bands):
            freq = (2 ** k) * math.pi
            feats.append(torch.sin(freq * omega_norm))
            feats.append(torch.cos(freq * omega_norm))
        return torch.cat(feats, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3D U-Net score network with residual blocks + FiLM conditioning
# ─────────────────────────────────────────────────────────────────────────────

def _fix_groups(n_ch: int, groups: int) -> int:
    """Return the largest valid GroupNorm group count ≤ groups that divides n_ch."""
    g = groups
    while g > 1 and n_ch % g != 0:
        g //= 2
    return g


class ResBlock3D(nn.Module):
    """3D residual block: GroupNorm → SiLU → Conv3d × 2, with FiLM conditioning."""
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int,
                 dropout: float = 0.0, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(_fix_groups(in_ch, groups), in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_fix_groups(out_ch, groups), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.film  = nn.Linear(cond_dim, 2 * out_ch)
        self.skip  = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.drop  = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        sc, sh = torch.chunk(self.film(cond), 2, dim=-1)
        sc = sc.view(sc.shape[0], -1, 1, 1, 1)
        sh = sh.view(sh.shape[0], -1, 1, 1, 1)
        h = self.conv2(self.drop(F.silu(self.norm2(h) * (1.0 + sc) + sh)))
        return self.skip(x) + h


class UNet3DScoreNet(nn.Module):
    """
    3D U-Net score network for volumetric FTM latent cores.

    Input:  x          (B, latent_dim)   latent_dim = 2 * Rx * Ry * Rz
            t          (B,)              diffusion timestep index
            omega_norm (B, 1)            normalized frequency
    Output: (B, latent_dim)  predicted noise

    Internally reshapes x → (B, 2, Rx, Ry, Rz), runs encoder–bottleneck–decoder
    with skip connections, then flattens back.  Down/up-sampling levels are
    auto-capped so the spatial size never goes below 1.
    """
    def __init__(
        self,
        rx: int,
        ry: int,
        rz: int,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        n_res_blocks:  int  = 2,
        time_dim:      int  = 256,
        omega_bands:   int  = 8,
        cond_dim:      int  = 512,
        dropout:       float = 0.0,
        groups:        int  = 8,
    ):
        super().__init__()
        self.rx, self.ry, self.rz = rx, ry, rz

        # Conditioning embeddings
        self.time_emb  = SinusoidalTimeEmbedding(time_dim)
        self.omega_emb = FourierOmegaEmbedding(omega_bands)
        cond_raw = time_dim + self.omega_emb.out_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_raw, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Auto-cap levels so spatial dims never drop below 1
        min_spatial = min(rx, ry, rz)
        max_levels  = max(0, int(math.log2(min_spatial))) if min_spatial >= 1 else 0
        n_levels    = min(len(channel_mults) - 1, max_levels)
        self.n_levels = n_levels

        chs = [base_channels * m for m in channel_mults[: n_levels + 1]]

        def _rblk(in_ch, out_ch):
            return ResBlock3D(in_ch, out_ch, cond_dim, dropout, groups)

        # Input projection: (B, 2, Rx, Ry, Rz) → (B, chs[0], Rx, Ry, Rz)
        self.in_conv = nn.Conv3d(2, chs[0], kernel_size=3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs       = nn.ModuleList()
        for i in range(n_levels):
            self.enc_blocks.append(nn.ModuleList(
                [_rblk(chs[i], chs[i]) for _ in range(n_res_blocks)]
            ))
            self.downs.append(nn.Conv3d(chs[i], chs[i + 1], kernel_size=2, stride=2))

        # Bottleneck
        mid_ch = chs[n_levels]
        self.mid_blocks = nn.ModuleList(
            [_rblk(mid_ch, mid_ch) for _ in range(n_res_blocks)]
        )

        # Decoder
        self.ups        = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(n_levels - 1, -1, -1):
            self.ups.append(nn.ConvTranspose3d(chs[i + 1], chs[i], kernel_size=2, stride=2))
            blks = nn.ModuleList([_rblk(2 * chs[i], chs[i])])          # first: concat skip
            blks.extend(_rblk(chs[i], chs[i]) for _ in range(n_res_blocks - 1))
            self.dec_blocks.append(blks)

        # Output
        self.out_norm = nn.GroupNorm(_fix_groups(chs[0], groups), chs[0])
        self.out_conv = nn.Conv3d(chs[0], 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, omega_norm: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = x.view(B, 2, self.rx, self.ry, self.rz)

        cond = self.cond_proj(
            torch.cat([self.time_emb(t), self.omega_emb(omega_norm)], dim=-1)
        )  # (B, cond_dim)

        h = self.in_conv(h)

        # Encoder – collect skip connections
        skips: List[torch.Tensor] = []
        for blks, down in zip(self.enc_blocks, self.downs):
            for blk in blks:
                h = blk(h, cond)
            skips.append(h)
            h = down(h)

        # Bottleneck
        for blk in self.mid_blocks:
            h = blk(h, cond)

        # Decoder – concat skip, apply res blocks
        for up, blks, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            h = up(h)
            if h.shape[2:] != skip.shape[2:]:           # guard odd spatial dims
                h = F.interpolate(h, size=skip.shape[2:], mode="trilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            for blk in blks:
                h = blk(h, cond)

        h = self.out_conv(F.silu(self.out_norm(h)))     # (B, 2, Rx, Ry, Rz)
        return h.view(B, -1)                             # (B, latent_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion schedule
# ─────────────────────────────────────────────────────────────────────────────

def build_linear_schedule(num_steps: int, beta_start: float, beta_end: float,
                           device: torch.device):
    betas      = torch.linspace(beta_start, beta_end, num_steps, device=device)
    alphas     = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas, "alphas": alphas, "alpha_bars": alpha_bars,
        "sqrt_alpha_bars":            torch.sqrt(alpha_bars),
        "sqrt_one_minus_alpha_bars":  torch.sqrt(1.0 - alpha_bars),
    }


def build_cosine_schedule(num_steps: int, s: float = 0.008,
                           device: torch.device = torch.device("cpu")):
    steps = num_steps + 1
    t = torch.linspace(0, num_steps, steps, device=device) / num_steps
    alpha_bars = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]
    alpha_bars = alpha_bars[:-1]
    alphas = torch.zeros_like(alpha_bars)
    alphas[1:] = alpha_bars[1:] / alpha_bars[:-1]
    alphas[0]  = alpha_bars[0]
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
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_core_dataset(ftm_ckpt: Path, max_samples: int = 0, seed: int = 42) -> Dict[str, Any]:
    if not ftm_ckpt.exists():
        raise FileNotFoundError(f"FTM checkpoint not found: {ftm_ckpt}")

    ckpt = torch.load(ftm_ckpt, map_location="cpu")

    if "cores" not in ckpt:
        raise KeyError("3D FTM checkpoint must contain 'cores' (list[C]).")

    cores_raw: List[torch.Tensor] = ckpt["cores"]   # list[C] of (B, M, Rx, Ry, Rz)
    C = len(cores_raw)
    if C != 2:
        raise ValueError(f"Expected C=2 (real+imag), got C={C}. "
                         f"Multi-channel 3D diffusion not yet supported.")

    cores = [_to_numpy(c).astype(np.float32) for c in cores_raw]  # list[2] of (B, M, Rx, Ry, Rz)
    B, M, Rx, Ry, Rz = cores[0].shape
    R = Rx * Ry * Rz

    omega = _to_numpy(ckpt["omega"]).astype(np.float32)  # (M,)
    if omega.shape[0] != M:
        raise ValueError(f"omega shape mismatch: {omega.shape}, expected ({M},)")

    if max_samples > 0:
        B = min(B, max_samples)
        cores = [c[:B] for c in cores]

    # Stack channels: (B, M, 2, Rx, Ry, Rz)
    stacked = np.stack(cores, axis=2).astype(np.float32)
    # Flatten spatial: (B, M, 2*R)
    z = stacked.reshape(B, M, 2 * R)

    # Omega broadcast: (B, M, 1)
    omega_tiled = np.tile(omega.reshape(1, M, 1), (B, 1, 1)).astype(np.float32)

    # Flatten to pairs: (B*M, 2*R) and (B*M, 1)
    z_flat      = z.reshape(B * M, 2 * R)
    omega_flat  = omega_tiled.reshape(B * M, 1)

    # Normalization (per-element mean/std across all pairs)
    mean = z_flat.mean(axis=0, keepdims=True).astype(np.float32)   # (1, 2*R)
    std  = z_flat.std(axis=0,  keepdims=True).astype(np.float32)
    std  = np.maximum(std, 1e-6)

    z_norm_flat = ((z_flat - mean) / std).astype(np.float32)
    z_norm      = z_norm_flat.reshape(B, M, 2 * R)

    omega_min = float(np.min(omega))
    omega_max = float(np.max(omega))
    omega_norm_flat = _normalize_omega(omega_flat.reshape(-1), omega_min, omega_max).reshape(B * M, 1)
    omega_norm_traj = _normalize_omega(omega, omega_min, omega_max).reshape(M, 1)

    return {
        "z_traj_norm":  z_norm,                  # (B, M, 2*R)
        "z_norm":       z_norm_flat,              # (B*M, 2*R)
        "omega_norm":   omega_norm_flat,          # (B*M, 1)
        "omega_traj_norm": omega_norm_traj,       # (M, 1)
        "mean": mean,                             # (1, 2*R)
        "std":  std,                              # (1, 2*R)
        "omega_min": omega_min,
        "omega_max": omega_max,
        "rx": int(Rx), "ry": int(Ry), "rz": int(Rz),
        "R": int(R), "latent_dim": int(2 * R),
        "num_pairs": int(B * M), "num_samples": int(B), "num_freqs": int(M),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device)

    data = load_core_dataset(Path(args.ftm_ckpt), max_samples=args.max_samples, seed=args.seed)
    latent_dim = data["latent_dim"]

    z_traj   = torch.from_numpy(data["z_traj_norm"]).float()   # (B, M, latent_dim)
    omega_traj = torch.from_numpy(data["omega_traj_norm"]).float().to(device)  # (M, 1)
    dataset  = TensorDataset(z_traj)
    loader   = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                          drop_last=False)

    channel_mults = tuple(int(x) for x in args.channel_mults.split(","))
    model = UNet3DScoreNet(
        rx            = data["rx"],
        ry            = data["ry"],
        rz            = data["rz"],
        base_channels = args.base_channels,
        channel_mults = channel_mults,
        n_res_blocks  = args.n_res_blocks,
        time_dim      = args.time_dim,
        omega_bands   = args.omega_bands,
        cond_dim      = args.cond_dim,
        dropout       = args.dropout,
        groups        = args.groups,
    ).to(device)

    if args.schedule == "linear":
        sched = build_linear_schedule(args.diffusion_steps, args.beta_start, args.beta_end, device)
    else:
        sched = build_cosine_schedule(args.diffusion_steps, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.95, eps=1e-12, min_lr=2e-7)

    best_loss = float("inf")
    loss_history: List[float] = []

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n" + "─" * 72)
    print("3D FTM Conditional Diffusion Training (3D U-Net score network)")
    print(f"ftm_ckpt   : {args.ftm_ckpt}")
    print(f"pairs      : {data['num_pairs']}  latent_dim={latent_dim}")
    print(f"rx={data['rx']} ry={data['ry']} rz={data['rz']}  R={data['R']}")
    print(f"unet levels: {model.n_levels}  channels: {args.channel_mults}  "
          f"n_res_blocks={args.n_res_blocks}  params={n_params:,}")
    print(f"device={device}  epochs={args.epochs}  batch={args.batch_size}  T={args.diffusion_steps}")
    print("─" * 72 + "\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0; n_batches = 0

        for (x0_traj,) in loader:
            x0_traj = x0_traj.to(device)          # (Bz, M, latent_dim)
            bsz, m, _ = x0_traj.shape

            # One diffusion timestep per sample, same for all freqs in that trajectory
            t_sample = torch.randint(0, args.diffusion_steps, (bsz,), device=device, dtype=torch.long)
            t = t_sample[:, None].expand(-1, m).reshape(-1)

            x0 = x0_traj.reshape(bsz * m, latent_dim)
            noise = torch.randn_like(x0)

            sqrt_ab  = _extract(sched["sqrt_alpha_bars"],           t, x0)
            sqrt_omb = _extract(sched["sqrt_one_minus_alpha_bars"],  t, x0)
            xt = sqrt_ab * x0 + sqrt_omb * noise

            omega_norm = omega_traj.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * m, 1)
            pred_noise = model(xt, t, omega_norm)

            loss = F.mse_loss(pred_noise, noise)

            # Frequency-trajectory consistency
            if args.freq_consistency_weight > 0 and m > 1:
                x0_hat = (xt - sqrt_omb * pred_noise) / torch.clamp(sqrt_ab, min=1e-12)
                x0_hat_traj = x0_hat.reshape(bsz, m, latent_dim)
                diff_true = x0_traj[:, 1:] - x0_traj[:, :-1]
                diff_pred = x0_hat_traj[:, 1:] - x0_hat_traj[:, :-1]
                loss = loss + args.freq_consistency_weight * F.mse_loss(diff_pred, diff_true)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running   += float(loss.item())
            n_batches += 1

        epoch_loss = running / max(n_batches, 1)
        loss_history.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        lr_sched.step(epoch_loss)

        if args.log_every > 0 and (epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs):
            print(f"[epoch {epoch:04d}/{args.epochs}] loss={epoch_loss:.6e} "
                  f"best={best_loss:.6e} lr={optimizer.param_groups[0]['lr']:.3e}")

    # ── Save checkpoint ────────────────────────────────────────────────────
    ckpt_out = {
        "model_state":  model.state_dict(),
        "model_config": {
            "type":          "UNet3D",
            "rx":            int(data["rx"]),
            "ry":            int(data["ry"]),
            "rz":            int(data["rz"]),
            "base_channels": int(args.base_channels),
            "channel_mults": list(channel_mults),
            "n_res_blocks":  int(args.n_res_blocks),
            "time_dim":      int(args.time_dim),
            "omega_bands":   int(args.omega_bands),
            "cond_dim":      int(args.cond_dim),
            "dropout":       float(args.dropout),
            "groups":        int(args.groups),
        },
        "diffusion_config": {
            "num_steps":  int(args.diffusion_steps),
            "beta_start": float(args.beta_start),
            "beta_end":   float(args.beta_end),
            "schedule":   str(args.schedule),
        },
        "core_stats": {
            "mean":       torch.from_numpy(data["mean"]),   # (1, latent_dim)
            "std":        torch.from_numpy(data["std"]),    # (1, latent_dim)
            "rx": int(data["rx"]), "ry": int(data["ry"]), "rz": int(data["rz"]),
            "R": int(data["R"]), "latent_dim": int(latent_dim),
            "layout": "flat_real_imag",
        },
        "omega_stats": {"min": float(data["omega_min"]), "max": float(data["omega_max"])},
        "train_info":  {
            "num_pairs": int(data["num_pairs"]), "num_samples": int(data["num_samples"]),
            "num_freqs": int(data["num_freqs"]), "best_loss": float(best_loss),
            "final_loss": float(loss_history[-1] if loss_history else 0.0),
        },
        "loss_history": loss_history,
        "ftm_ckpt": str(args.ftm_ckpt),
        "config":   vars(args),
        "spatial_dims": 3,
    }
    torch.save(ckpt_out, out_path)

    summary = {
        "out": str(out_path), "latent_dim": int(latent_dim),
        "rx": data["rx"], "ry": data["ry"], "rz": data["rz"], "R": data["R"],
        "best_loss": float(best_loss),
        "final_loss": float(loss_history[-1] if loss_history else 0.0),
        "epochs": int(args.epochs),
    }
    summary_path = out_path.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining finished.")
    print(json.dumps(summary, indent=2))
    print(f"Saved checkpoint : {out_path}")
    print(f"Saved summary    : {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train 3D FTM conditional diffusion (3D U-Net score net)")
    p.add_argument("--ftm_ckpt", type=str, default="ckp/ftm3d.pt")
    p.add_argument("--out",      type=str, default="ckp/diffusion3d.pt")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--epochs",       type=int,   default=1000)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--num_workers",  type=int,   default=0)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip",    type=float, default=0)
    # 3D U-Net architecture
    p.add_argument("--base_channels",  type=int,   default=128,
                   help="Channel count at the first encoder level")
    p.add_argument("--channel_mults",  type=str,   default="1,2,4,8",
                   help="Comma-separated channel multipliers per level, e.g. '1,2,4,8'")
    p.add_argument("--n_res_blocks",   type=int,   default=2,
                   help="Residual blocks per encoder/decoder level")
    p.add_argument("--groups",         type=int,   default=8,
                   help="GroupNorm group count (auto-reduced if channel count is small)")
    # Conditioning
    p.add_argument("--time_dim",    type=int,   default=256)
    p.add_argument("--omega_bands", type=int,   default=8)
    p.add_argument("--cond_dim",    type=int,   default=512)
    p.add_argument("--dropout",     type=float, default=0.0)
    # Diffusion schedule
    p.add_argument("--diffusion_steps", type=int,   default=500)
    p.add_argument("--beta_start",      type=float, default=1e-4)
    p.add_argument("--beta_end",        type=float, default=2e-2)
    p.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine"])
    p.add_argument("--freq_consistency_weight", type=float, default=0.0)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--device",  type=str, default="auto")
    p.add_argument("--log_every", type=int, default=1)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
