"""
confild_baseline.py  (elastic wave edition)
--------------------------------------------
CoNFILD diffusion model baseline for 4-channel elastic wave field reconstruction.

Input channels (conditioning):
- obs_ux_re, obs_ux_im   : observed x-displacement (real/imag)
- obs_uy_re, obs_uy_im   : observed y-displacement (real/imag)
- mask                   : observation mask (shared channel-0 mask)
- omega_norm             : normalised frequency map
- x_coord, y_coord       : spatial coordinate maps
Total: 8 conditioning channels

Model input: [noisy_field(4) | conditioning(8)] = 12 channels
Model output: 4 channels (predicted noise for ux_re, ux_im, uy_re, uy_im)

Examples
--------
Train:
    python confild_baseline.py --mode train \
        --train_h5 elastic_dataset.h5 \
        --out ckp/confild_elastic.pt

Evaluate:
    python confild_baseline.py --mode eval \
        --ckpt ckp/confild_elastic.pt \
        --test_h5 elastic_dataset.h5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


def _safe_torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _normalize_omega(omega: float, omega_min: float, omega_max: float) -> float:
    return (omega - omega_min) / max(float(omega_max - omega_min), 1e-12)


def _parse_indices(text: str, limit: int) -> List[int]:
    if text.strip() == "":
        return list(range(limit))
    out: List[int] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        idx = int(p)
        if idx < 0 or idx >= limit:
            raise ValueError(f"index out of range: {idx}")
        out.append(idx)
    return sorted(set(out))


def _relative_rmse(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.sum((pred - gt) ** 2) / max(np.sum(gt ** 2), eps)))


def _masked_relative_rmse(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps: float = 1e-12
) -> float:
    """mask: (H,W,1) or (H,W,C) – any positive value counts as observed."""
    m = (mask > 0.5)
    # broadcast mask over all channels if needed
    if m.shape[-1] == 1 and pred.ndim == 3:
        m = np.broadcast_to(m, pred.shape)
    if not np.any(m):
        return float("nan")
    num = float(np.sum((pred - gt) ** 2 * m))
    den = float(np.sum(gt ** 2 * m))
    return float(np.sqrt(num / max(den, eps)))


def _split_samples(num: int, ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    ids = np.arange(num)
    rng.shuffle(ids)
    n_train = max(1, min(int(round(num * ratio)), num - 1))
    train_ids = ids[:n_train].tolist()
    val_ids = ids[n_train:].tolist() or train_ids[-1:]
    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

CHANNEL_NAMES = ["ux_re", "ux_im", "uy_re", "uy_im"]


class ElasticDataset(Dataset):
    """
    Loads 4-channel elastic wave data from HDF5 into memory once.

    Item keys:
      x      : (8, H, W)  conditioning input
      y      : (4, H, W)  ground-truth field
      mask   : (1, H, W)  binary observation mask
      omega  : (1,)       normalised frequency
      sample_idx, freq_idx, omega_raw
    """

    def __init__(
        self,
        h5_path: str | Path,
        sample_indices: Optional[Sequence[int]] = None,
        freq_indices: str = "",
    ) -> None:
        self.path = Path(h5_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        with h5py.File(self.path, "r") as f:
            self.omega = f["omega"][...].astype(np.float32)
            self.omega_min = float(np.min(self.omega))
            self.omega_max = float(np.max(self.omega))

            self.data = f["data"][...].astype(np.float32)   # (N,M,H,W,4)
            self.N, self.M, self.H, self.W, self.C = self.data.shape
            assert self.C == 4, f"Expected C=4, got {self.C}"

            self.has_mask = "mask_tr" in f
            self.mask_data = f["mask_tr"][...].astype(np.float32) if self.has_mask else None

            self.gx = f["grid_x"][...].astype(np.float32) if "grid_x" in f else \
                np.linspace(0.0, 1.0, self.H, dtype=np.float32)
            self.gy = f["grid_y"][...].astype(np.float32) if "grid_y" in f else \
                np.linspace(0.0, 1.0, self.W, dtype=np.float32)

        self.sample_indices = list(range(self.N)) if sample_indices is None \
            else [int(i) for i in sample_indices]
        self.freq_indices = _parse_indices(freq_indices, self.M)
        self.pairs: List[Tuple[int, int]] = [
            (b, m) for b in self.sample_indices for m in self.freq_indices
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def _get_mask(self, sample: int, freq_idx: int) -> np.ndarray:
        """Returns (H, W, 1) float32 mask."""
        if not self.has_mask:
            return np.ones((self.H, self.W, 1), dtype=np.float32)
        mask = self.mask_data
        if mask.ndim == 4:       # (M, H, W, C)
            m = mask[freq_idx, ..., :1]
        elif mask.ndim == 5:     # (B, M, H, W, C)
            m = mask[sample, freq_idx, ..., :1]
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")
        return m.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample, freq_idx = self.pairs[idx]
        field = self.data[sample, freq_idx]          # (H, W, 4)
        mask = self._get_mask(sample, freq_idx)      # (H, W, 1)
        omega = float(self.omega[freq_idx])
        omega_norm = float(_normalize_omega(omega, self.omega_min, self.omega_max))

        obs = field * mask                           # (H, W, 4), broadcast mask
        x = np.stack([
            obs[..., 0], obs[..., 1], obs[..., 2], obs[..., 3],  # 4 observed channels
            mask[..., 0],                                          # 1 mask channel
            np.full((self.H, self.W), omega_norm, dtype=np.float32),
            np.broadcast_to(self.gx[:, None], (self.H, self.W)).copy(),
            np.broadcast_to(self.gy[None, :], (self.H, self.W)).copy(),
        ], axis=0).astype(np.float32)               # (8, H, W)

        y = field.transpose(2, 0, 1).astype(np.float32)   # (4, H, W)
        return {
            "x": x,
            "y": y,
            "mask": mask.transpose(2, 0, 1).astype(np.float32),  # (1, H, W)
            "omega": np.array([omega_norm], dtype=np.float32),
            "sample_idx": np.array(sample, dtype=np.int64),
            "freq_idx": np.array(freq_idx, dtype=np.int64),
            "omega_raw": np.array(omega, dtype=np.float32),
        }


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    keys = ["x", "y", "mask", "omega", "sample_idx", "freq_idx", "omega_raw"]
    out = {}
    for k in keys:
        out[k] = torch.from_numpy(np.stack([b[k] for b in batch], axis=0))
    out["sample_idx"] = out["sample_idx"].long()
    out["freq_idx"] = out["freq_idx"].long()
    return out


# ---------------------------------------------------------------------------
# Model components (shared with Helmholtz confild, generalised to C channels)
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
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


def _group_count(channels: int, max_groups: int = 8) -> int:
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


class ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, 2 * out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h)
        sc = self.cond_proj(cond)
        scale, shift = torch.chunk(sc, 2, dim=1)
        h = h * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class SelfAttention2D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        heads = max(1, min(num_heads, channels))
        while heads > 1 and channels % heads != 0:
            heads -= 1
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.norm(x).reshape(b, c, h * w).transpose(1, 2)
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        out = self.out_proj(out).transpose(1, 2).reshape(b, c, h, w)
        return x + out


class ConditionalUNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 4,
        base_channels: int = 64,
        cond_dim: int = 256,
        time_dim: int = 128,
        omega_bands: int = 8,
        dropout: float = 0.0,
        mid_attn_heads: int = 4,
        mid_attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.omega_emb = FourierOmegaEmbedding(omega_bands)
        cond_in = time_dim + self.omega_emb.out_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.stem = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.enc1 = ResBlock2D(c1, c1, cond_dim, dropout)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = ResBlock2D(c2, c2, cond_dim, dropout)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.mid = ResBlock2D(c3, c3, cond_dim, dropout)
        self.mid_attn = SelfAttention2D(c3, mid_attn_heads, mid_attn_dropout) \
            if mid_attn_heads > 0 else nn.Identity()
        self.dec2 = ResBlock2D(c3 + c2, c2, cond_dim, dropout)
        self.dec1 = ResBlock2D(c2 + c1, c1, cond_dim, dropout)
        self.out_norm = nn.GroupNorm(_group_count(c1), c1)
        self.out_conv = nn.Conv2d(c1, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, omega_norm: torch.Tensor) -> torch.Tensor:
        cond = self.cond_mlp(torch.cat([self.time_emb(t), self.omega_emb(omega_norm)], dim=1))
        x0 = self.stem(x)
        e1 = self.enc1(x0, cond)
        d1 = self.down1(e1)
        e2 = self.enc2(d1, cond)
        d2 = self.down2(e2)
        m = self.mid_attn(self.mid(d2, cond))
        u2 = self.dec2(torch.cat([F.interpolate(m, size=e2.shape[-2:], mode="nearest"), e2], 1), cond)
        u1 = self.dec1(torch.cat([F.interpolate(u2, size=e1.shape[-2:], mode="nearest"), e1], 1), cond)
        return self.out_conv(F.silu(self.out_norm(u1)))


# ---------------------------------------------------------------------------
# Diffusion schedule
# ---------------------------------------------------------------------------

@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor


def build_linear_schedule(
    num_steps: int, beta_start: float, beta_end: float, device: torch.device
) -> DiffusionSchedule:
    betas = torch.linspace(beta_start, beta_end, num_steps, device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(
        betas=betas, alphas=alphas, alpha_bars=alpha_bars,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
    )


def extract(vals: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    out = vals.gather(0, t).to(dtype=x.dtype)
    return out.view(-1, *([1] * (x.ndim - 1)))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _plot_elastic_case(
    out_path: Path,
    gt: np.ndarray,    # (H, W, 4): [ux_re, ux_im, uy_re, uy_im]
    pred: np.ndarray,
    sample_idx: int,
    freq_idx: int,
    omega_val: float,
    rmse: float,
    obs_rmse: float,
    unobs_rmse: float,
    dpi: int = 150,
) -> None:
    ch_labels = [("ux", 0, 1), ("uy", 2, 3)]
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    row = 0
    for name, re_idx, im_idx in ch_labels:
        for ch_idx, part in [(re_idx, "Re"), (im_idx, "Im")]:
            g = gt[..., ch_idx]
            p = pred[..., ch_idx]
            err = np.abs(p - g)
            for ax, img, title in zip(
                axes[row],
                [g, p, err],
                [f"GT {name}_{part}", f"Pred {name}_{part}", f"Err {name}_{part}"],
            ):
                im = ax.imshow(img, origin="lower", cmap="viridis" if "Err" not in title else "magma")
                ax.set_title(title, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            row += 1
    fig.suptitle(
        f"sample={sample_idx} freq={freq_idx} ω={omega_val:.4f} | "
        f"rmse={rmse:.3e} obs={obs_rmse:.3e} unobs={unobs_rmse:.3e}",
        y=0.999, fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# HDF5 meta helper
# ---------------------------------------------------------------------------

def _load_h5_meta(h5_path: Path) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as f:
        meta = {}
        if "metadata" in f:
            raw = f["metadata"][()]
            try:
                meta = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, np.bytes_)) else str(raw))
            except Exception:
                pass
        shape = f["data"].shape
        omega = f["omega"][...].astype(np.float32)
    return {"meta": meta, "omega": omega, "shape": shape}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)

    meta = _load_h5_meta(Path(args.train_h5))
    num_samples = int(meta["shape"][0])
    train_ids, val_ids = _split_samples(num_samples, args.train_ratio, args.seed)

    train_ds = ElasticDataset(args.train_h5, train_ids, args.train_freq_indices)
    val_ds = ElasticDataset(args.train_h5, val_ids, args.eval_freq_indices)

    # in_channels = 4 noisy + 8 conditioning = 12
    model = ConditionalUNet2D(
        in_channels=4 + 8,
        out_channels=4,
        base_channels=args.base_channels,
        cond_dim=args.cond_dim,
        time_dim=args.time_dim,
        omega_bands=args.omega_bands,
        dropout=args.dropout,
        mid_attn_heads=args.mid_attn_heads,
        mid_attn_dropout=args.mid_attn_dropout,
    ).to(device)

    schedule = build_linear_schedule(args.diffusion_steps, args.beta_start, args.beta_end, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=_collate, drop_last=False)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=_collate, drop_last=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    print(f"\n{'=' * 60}")
    print("CoNFILD elastic baseline training")
    print(f"train_h5 : {args.train_h5}")
    print(f"train_N  : {len(train_ids)}, val_N : {len(val_ids)}")
    print(f"device   : {device}, epochs : {args.epochs}")
    print(f"{'=' * 60}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            omega = batch["omega"].to(device)
            bsz = x.shape[0]
            t = torch.randint(0, args.diffusion_steps, (bsz,), device=device, dtype=torch.long)
            noise = torch.randn_like(y)
            sqrt_ab = extract(schedule.sqrt_alpha_bars, t, y)
            sqrt_omb = extract(schedule.sqrt_one_minus_alpha_bars, t, y)
            yt = sqrt_ab * y + sqrt_omb * noise
            loss = F.mse_loss(model(torch.cat([yt, x], 1), t, omega), noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            tr_loss += float(loss.item()) * bsz
            tr_n += bsz

        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                omega = batch["omega"].to(device)
                bsz = x.shape[0]
                t = torch.randint(0, args.diffusion_steps, (bsz,), device=device, dtype=torch.long)
                noise = torch.randn_like(y)
                sqrt_ab = extract(schedule.sqrt_alpha_bars, t, y)
                sqrt_omb = extract(schedule.sqrt_one_minus_alpha_bars, t, y)
                yt = sqrt_ab * y + sqrt_omb * noise
                loss = F.mse_loss(model(torch.cat([yt, x], 1), t, omega), noise)
                val_loss += float(loss.item()) * bsz
                val_n += bsz

        tl = tr_loss / max(tr_n, 1)
        vl = val_loss / max(val_n, 1)
        scheduler.step(vl)

        if vl < best_val:
            best_val = vl
            torch.save({
                "model_state": model.state_dict(),
                "model_config": {
                    "in_channels": 4 + 8,
                    "out_channels": 4,
                    "base_channels": args.base_channels,
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
                },
                "train_config": vars(args),
                "best_val_loss": best_val,
                "train_ids": train_ids,
                "val_ids": val_ids,
            }, out_path)

        if args.log_every > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0):
            print(f"[{epoch:04d}/{args.epochs}] train={tl:.6e}  val={vl:.6e}  best={best_val:.6e}  "
                  f"lr={optimizer.param_groups[0]['lr']:.3e}")

    print(f"\nTraining done. Best val loss: {best_val:.6e}")
    print(f"Checkpoint: {out_path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)

    ckpt = _safe_torch_load(Path(args.ckpt))
    model_cfg = dict(ckpt["model_config"])
    model = ConditionalUNet2D(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    diff_cfg = ckpt.get("diffusion_config", {})
    schedule = build_linear_schedule(
        num_steps=diff_cfg.get("num_steps", 500),
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 2e-2),
        device=device,
    )
    num_steps = diff_cfg.get("num_steps", 500)

    data_path = Path(args.test_h5)
    meta = _load_h5_meta(data_path)
    num_samples = int(meta["shape"][0])
    sample_indices = list(range(min(num_samples, args.max_samples) if args.max_samples > 0 else num_samples))

    ds = ElasticDataset(data_path, sample_indices, args.eval_freq_indices)
    loader = DataLoader(ds, args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    vis_count = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            omega = batch["omega"].to(device)
            mask_np = batch["mask"].cpu().numpy()         # (B, 1, H, W)
            sample_idx_np = batch["sample_idx"].cpu().numpy()
            freq_idx_np = batch["freq_idx"].cpu().numpy()
            omega_raw_np = batch["omega_raw"].cpu().numpy()
            bsz = x.shape[0]

            # DDPM reverse sampling
            yt = torch.randn_like(y)
            for t_idx in reversed(range(num_steps)):
                t = torch.full((bsz,), t_idx, device=device, dtype=torch.long)
                eps_pred = model(torch.cat([yt, x], 1), t, omega)
                abar_t = schedule.alpha_bars[t_idx]
                alpha_t = schedule.alphas[t_idx]
                beta_t = schedule.betas[t_idx]
                z = torch.randn_like(yt) if t_idx > 0 else torch.zeros_like(yt)
                yt = (1.0 / torch.sqrt(alpha_t)) * (
                    yt - (1.0 - alpha_t) / torch.sqrt(1.0 - abar_t) * eps_pred
                ) + torch.sqrt(beta_t) * z

            pred_np = yt.cpu().numpy()    # (B, 4, H, W)
            gt_np = y.cpu().numpy()

            for i in range(bsz):
                pred_i = pred_np[i].transpose(1, 2, 0)   # (H, W, 4)
                gt_i = gt_np[i].transpose(1, 2, 0)
                mask_i = mask_np[i].transpose(1, 2, 0)   # (H, W, 1)

                rmse = _relative_rmse(pred_i, gt_i, args.eps)
                obs_rmse = _masked_relative_rmse(pred_i, gt_i, mask_i, args.eps)
                unobs_rmse = _masked_relative_rmse(pred_i, gt_i, 1.0 - mask_i, args.eps)

                # Per-channel RMSE
                ch_rmse = {CHANNEL_NAMES[c]: _relative_rmse(pred_i[..., c], gt_i[..., c], args.eps)
                           for c in range(4)}

                row: Dict[str, Any] = {
                    "sample_idx": int(sample_idx_np[i]),
                    "freq_idx": int(freq_idx_np[i]),
                    "omega": float(omega_raw_np[i]),
                    "rmse": float(rmse),
                    "obs_rmse": float(obs_rmse),
                    "unobs_rmse": float(unobs_rmse),
                }
                row.update({f"rmse_{k}": float(v) for k, v in ch_rmse.items()})
                rows.append(row)

                if vis_count < args.num_visualize:
                    vis_count += 1
                    vis_path = out_dir / (
                        f"case{vis_count:03d}_s{int(sample_idx_np[i]):03d}_f{int(freq_idx_np[i]):03d}.png"
                    )
                    _plot_elastic_case(
                        vis_path, gt_i, pred_i,
                        int(sample_idx_np[i]), int(freq_idx_np[i]),
                        float(omega_raw_np[i]), rmse, obs_rmse, unobs_rmse,
                        dpi=args.vis_dpi,
                    )
                    rows[-1]["vis_path"] = str(vis_path)

    if not rows:
        raise RuntimeError("No evaluation rows produced")

    fieldnames = ["sample_idx", "freq_idx", "omega", "rmse", "obs_rmse", "unobs_rmse"] + \
        [f"rmse_{n}" for n in CHANNEL_NAMES] + ["vis_path"]
    csv_path = out_dir / "metrics_cases.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    rmses = np.array([r["rmse"] for r in rows])
    obs_rmses = np.array([r["obs_rmse"] for r in rows])
    unobs_rmses = np.array([r["unobs_rmse"] for r in rows])

    summary = {
        "ckpt": str(args.ckpt),
        "test_h5": str(data_path),
        "num_cases": len(rows),
        "mean_rmse": float(np.mean(rmses)),
        "mean_obs_rmse": float(np.nanmean(obs_rmses)),
        "mean_unobs_rmse": float(np.nanmean(unobs_rmses)),
        "per_channel": {
            n: float(np.mean([r[f"rmse_{n}"] for r in rows])) for n in CHANNEL_NAMES
        },
        "num_visualized": vis_count,
        "output_dir": str(out_dir),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nEvaluation finished.")
    print(json.dumps(summary, indent=2))
    print(f"Metrics : {csv_path}")
    print(f"Summary : {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CoNFILD diffusion baseline for elastic wave reconstruction")
    p.add_argument("--mode", default="eval", choices=["train", "eval"])

    p.add_argument("--train_h5", default="elastic_dataset.h5")
    p.add_argument("--test_h5",  default="elastic_dataset_msk0.01.h5")
    p.add_argument("--ckpt",     default="ckp/confild_elastic.pt")
    p.add_argument("--out",      default="ckp/confild_elastic.pt")
    p.add_argument("--out_dir",  default="visual_data/confild_elastic_eval_msk0.01")

    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--train_freq_indices", type=str, default="")
    p.add_argument("--eval_freq_indices",  type=str, default="")
    p.add_argument("--max_samples", type=int, default=10)

    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--log_every",    type=int,   default=5)

    p.add_argument("--base_channels",    type=int,   default=64)
    p.add_argument("--cond_dim",         type=int,   default=256)
    p.add_argument("--time_dim",         type=int,   default=128)
    p.add_argument("--omega_bands",      type=int,   default=12)
    p.add_argument("--dropout",          type=float, default=0.0)
    p.add_argument("--mid_attn_heads",   type=int,   default=4)
    p.add_argument("--mid_attn_dropout", type=float, default=0.0)

    p.add_argument("--diffusion_steps", type=int,   default=500)
    p.add_argument("--beta_start",      type=float, default=1e-4)
    p.add_argument("--beta_end",        type=float, default=2e-2)

    p.add_argument("--num_visualize", type=int, default=20)
    p.add_argument("--vis_dpi",       type=int, default=150)
    p.add_argument("--eps",           type=float, default=1e-6)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--device",        type=str,   default="auto")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "train":
        train_model(args)
    else:
        evaluate_model(args)


if __name__ == "__main__":
    main()
