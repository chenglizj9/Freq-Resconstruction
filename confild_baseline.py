"""
confild_baseline.py
---------------
2D Fourier Neural Operator baseline for Helmholtz field reconstruction.

Baseline protocol
------------------
Input channels:
- observed real part
- observed imaginary part
- observation mask
- normalized omega map
- x coordinate map
- y coordinate map

Output channels:
- reconstructed real part
- reconstructed imaginary part

The script supports both training and evaluation on the HDF5 datasets produced by
Generate_dataset.py / the test split files in data_for_test/.

Examples
--------
Train:
    python confild_baseline.py --mode train --train_h5 helmholtz_dataset_42.h5 --out ckp/confild_baseline.pt

Evaluate:
    python confild_baseline.py --mode eval --ckpt ckp/confild_baseline.pt --test_h5 data_for_test/helmholtz_dataset_42_for_test_mask2.h5
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
from physics_metric import evaluate_physics_residual
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------


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


def _normalize_omega(omega: np.ndarray | float, omega_min: float, omega_max: float) -> np.ndarray | float:
    den = max(float(omega_max - omega_min), 1e-12)
    return (omega - omega_min) / den


def _parse_indices(text: str, limit: int) -> List[int]:
    if text.strip() == "":
        return list(range(limit))

    out: List[int] = []
    for part in text.split(","):
        p = part.strip()
        if p == "":
            continue
        idx = int(p)
        if idx < 0 or idx >= limit:
            raise ValueError(f"index out of range: {idx}, valid [0, {limit - 1}]")
        out.append(idx)

    if not out:
        raise ValueError("Parsed empty index list")
    return sorted(set(out))


def _relative_rmse(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    num = float(np.sum((pred - gt) ** 2))
    den = float(np.sum(gt**2))
    return float(np.sqrt(num / max(den, eps)))


def _masked_relative_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> float:
    m = mask.astype(bool)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    if not np.any(m):
        return float("nan")
    diff = (pred - gt) ** 2
    gt_sq = gt**2
    if diff.ndim == 3 and diff.shape[-1] == 2:
        diff = np.sum(diff, axis=-1)
        gt_sq = np.sum(gt_sq, axis=-1)
    num = float(np.sum(diff[m]))
    den = float(np.sum(gt_sq[m]))
    return float(np.sqrt(num / max(den, eps)))


def _symm_norm(data: np.ndarray):
    vmax = np.percentile(np.abs(data), 99)
    return (-vmax, vmax)


def _plot_case(
    out_path: Path,
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    sample_idx: int,
    freq_idx: int,
    omega_val: float,
    rmse: float,
    obs_rmse: float,
    unobs_rmse: float,
    dpi: int = 180,
    pde_res: float = 0.0,
) -> None:
    gt_re, gt_im = gt[..., 0], gt[..., 1]
    pr_re, pr_im = pred[..., 0], pred[..., 1]
    err_re = np.abs(pr_re - gt_re)
    err_im = np.abs(pr_im - gt_im)
    gt_amp = np.sqrt(gt_re**2 + gt_im**2)
    pr_amp = np.sqrt(pr_re**2 + pr_im**2)
    err_amp = np.abs(pr_amp - gt_amp)
    mask_img = mask[..., 0].astype(np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    items = [
        (gt_re, "GT Real", "viridis", None),
        (pr_re, "Pred Real", "viridis", None),
        (err_re, "Abs Err Real", "magma", None),
        (gt_im, "GT Imag", "viridis", None),
        (pr_im, "Pred Imag", "viridis", None),
        (err_im, "Abs Err Imag", "magma", None),
    ]

    for ax, (img, title, cmap, _) in zip(axes.flat, items):
        if title.startswith("GT") or title.startswith("Pred"):
            vmin, vmax = _symm_norm(img)
            im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(img, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"sample={sample_idx}, freq_idx={freq_idx}, omega={omega_val:.4f} | "
        f"rmse={rmse:.3e}, obs_rmse={obs_rmse:.3e}, unobs_rmse={unobs_rmse:.3e}, pde_res={pde_res:.3e}",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class HelmholtzFNOData(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        sample_indices: Optional[Sequence[int]] = None,
        freq_indices: str = "",
        use_data: bool = True,
    ) -> None:
        self.path = Path(h5_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        # 🔥 核心修改：一次性打开并读取所有数据到内存，关闭HDF5
        with h5py.File(self.path, "r") as f:
            # 读取基础参数
            self.omega = f["omega"][...].astype(np.float32)
            self.omega_min = float(np.min(self.omega))
            self.omega_max = float(np.max(self.omega))
            self.data_scale = float(np.asarray(f["data_scale"][()])) if "data_scale" in f else 1.0

            # 读取场数据（全部加载到内存）
            if use_data and "data" in f:
                self.use_data = True
                self.data = f["data"][...].astype(np.float32)  # 全量加载
                self.N, self.M, self.H, self.W, self.C = self.data.shape
            elif "fields_real" in f and "fields_imag" in f:
                self.use_data = False
                self.data = None
                self.re_data = f["fields_real"][...].astype(np.float32)  # 全量加载
                self.im_data = f["fields_imag"][...].astype(np.float32)  # 全量加载
                self.N, self.M, self.H, self.W = self.re_data.shape
                self.C = 2
            else:
                raise KeyError("HDF5 must contain either data or fields_real/fields_imag")

            # 读取掩码（全量加载）
            self.has_mask = "mask_tr" in f
            if self.has_mask:
                self.mask_data = f["mask_tr"][...].astype(np.float32)  # 全量加载
            else:
                self.mask_data = None

            # 读取网格
            self.gx = f["grid_x"][...].astype(np.float32) if "grid_x" in f else np.linspace(0.0, 1.0, self.H, dtype=np.float32)
            self.gy = f["grid_y"][...].astype(np.float32) if "grid_y" in f else np.linspace(0.0, 1.0, self.W, dtype=np.float32)

        # 索引处理
        if sample_indices is None:
            self.sample_indices = list(range(self.N))
        else:
            self.sample_indices = [int(i) for i in sample_indices]
        self.freq_indices = _parse_indices(freq_indices, self.M)
        self.pairs: List[Tuple[int, int]] = [(b, m) for b in self.sample_indices for m in self.freq_indices]

    def __len__(self) -> int:
        return len(self.pairs)

    # 🔥 无磁盘读取！纯内存操作
    def _get_field(self, sample: int, freq_idx: int) -> np.ndarray:
        if self.use_data:
            return self.data[sample, freq_idx]
        return np.stack([self.re_data[sample, freq_idx], self.im_data[sample, freq_idx]], axis=-1)

    # 🔥 无磁盘读取！纯内存操作
    def _get_mask(self, sample: int, freq_idx: int) -> np.ndarray:
        if not self.has_mask:
            return np.ones((self.H, self.W, 1), dtype=np.float32)

        mask = self.mask_data
        if mask.ndim == 4:
            mask = mask[freq_idx]
        elif mask.ndim == 5:
            mask = mask[sample, freq_idx]

        if mask.shape[-1] == 2:
            mask = mask[..., :1]
        return mask

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        sample, freq_idx = self.pairs[idx]
        field = self._get_field(sample, freq_idx)
        mask = self._get_mask(sample, freq_idx)
        omega = float(self.omega[freq_idx])
        omega_norm = float(_normalize_omega(omega, self.omega_min, self.omega_max))

        obs = field * mask
        x = np.stack(
            [
                obs[..., 0], obs[..., 1], mask[..., 0],
                np.full((self.H, self.W), omega_norm, dtype=np.float32),
                np.broadcast_to(self.gx[:, None], (self.H, self.W)),
                np.broadcast_to(self.gy[None, :], (self.H, self.W)),
            ], axis=0,
        )
        y = field.transpose(2, 0, 1)
        return {
            "x": x, "y": y, "mask": mask.transpose(2, 0, 1),
            "omega": np.array([omega_norm]), "sample_idx": sample,
            "freq_idx": freq_idx, "omega_raw": omega,
        }
# class HelmholtzFNOData(Dataset):
#     def __init__(
#         self,
#         h5_path: str | Path,
#         sample_indices: Optional[Sequence[int]] = None,
#         freq_indices: str = "",
#         use_data: bool = True,
#     ) -> None:
#         self.path = Path(h5_path)
#         if not self.path.exists():
#             raise FileNotFoundError(self.path)

#         self.f = h5py.File(self.path, "r")
#         if "omega" not in self.f:
#             raise KeyError("HDF5 must contain omega")

#         self.omega = self.f["omega"][...].astype(np.float32)
#         self.omega_min = float(np.min(self.omega))
#         self.omega_max = float(np.max(self.omega))
#         self.data_scale = float(np.asarray(self.f["data_scale"][()])) if "data_scale" in self.f else 1.0

#         if use_data and "data" in self.f:
#             self.use_data = True
#             self.data = self.f["data"]
#             self.N, self.M, self.H, self.W, self.C = self.data.shape
#             if self.C != 2:
#                 raise ValueError("data must have 2 channels")
#         elif "fields_real" in self.f and "fields_imag" in self.f:
#             self.use_data = False
#             self.data = None
#             self.N, self.M, self.H, self.W = self.f["fields_real"].shape
#             self.C = 2
#         else:
#             raise KeyError("HDF5 must contain either data or fields_real/fields_imag")

#         if sample_indices is None:
#             self.sample_indices = list(range(self.N))
#         else:
#             self.sample_indices = [int(i) for i in sample_indices]

#         self.freq_indices = _parse_indices(freq_indices, self.M)
#         self.gx = self.f["grid_x"][...].astype(np.float32) if "grid_x" in self.f else np.linspace(0.0, 1.0, self.H, dtype=np.float32)
#         self.gy = self.f["grid_y"][...].astype(np.float32) if "grid_y" in self.f else np.linspace(0.0, 1.0, self.W, dtype=np.float32)

#         mask_ds = self.f["mask_tr"] if "mask_tr" in self.f else None
#         self.mask_ds = mask_ds
#         self.has_mask = mask_ds is not None
#         self.mask_shape = tuple(mask_ds.shape) if mask_ds is not None else None

#         self.pairs: List[Tuple[int, int]] = [(b, m) for b in self.sample_indices for m in self.freq_indices]
#         if not self.pairs:
#             raise ValueError("No (sample, freq) pairs selected")

#     def __len__(self) -> int:
#         return len(self.pairs)

#     def close(self) -> None:
#         try:
#             self.f.close()
#         except Exception:
#             pass

#     def __del__(self) -> None:
#         self.close()

#     def _get_field(self, sample: int, freq_idx: int) -> np.ndarray:
#         if self.use_data:
#             x = self.data[sample, freq_idx].astype(np.float32)
#             if x.shape[-1] != 2:
#                 raise ValueError("data must have 2 channels")
#             return x

#         re = self.f["fields_real"][sample, freq_idx].astype(np.float32)
#         im = self.f["fields_imag"][sample, freq_idx].astype(np.float32)
#         return np.stack([re, im], axis=-1).astype(np.float32)

#     def _get_mask(self, sample: int, freq_idx: int) -> np.ndarray:
#         if not self.has_mask:
#             return np.ones((self.H, self.W, 1), dtype=np.float32)

#         mask_ds = self.mask_ds
#         assert mask_ds is not None
#         if mask_ds.ndim == 4:
#             mask = mask_ds[freq_idx].astype(np.float32)
#         elif mask_ds.ndim == 5:
#             mask = mask_ds[sample, freq_idx].astype(np.float32)
#         else:
#             raise ValueError(f"Unsupported mask_tr shape: {mask_ds.shape}")

#         if mask.shape[-1] == 2:
#             mask = mask[..., :1]
#         elif mask.shape[-1] != 1:
#             raise ValueError(f"Unsupported mask channel count: {mask.shape}")
#         return mask.astype(np.float32)

#     def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
#         sample, freq_idx = self.pairs[idx]
#         field = self._get_field(sample, freq_idx)
#         mask = self._get_mask(sample, freq_idx)
#         omega = float(self.omega[freq_idx])
#         omega_norm = float(_normalize_omega(omega, self.omega_min, self.omega_max))

#         obs = field * mask
#         x = np.stack(
#             [
#                 obs[..., 0],
#                 obs[..., 1],
#                 mask[..., 0],
#                 np.full((self.H, self.W), omega_norm, dtype=np.float32),
#                 np.broadcast_to(self.gx[:, None], (self.H, self.W)),
#                 np.broadcast_to(self.gy[None, :], (self.H, self.W)),
#             ],
#             axis=0,
#         ).astype(np.float32)

#         y = field.transpose(2, 0, 1).astype(np.float32)
#         return {
#             "x": x,
#             "y": y,
#             "mask": mask.transpose(2, 0, 1).astype(np.float32),
#             "omega": np.array([omega_norm], dtype=np.float32),
#             "sample_idx": np.array(sample, dtype=np.int64),
#             "freq_idx": np.array(freq_idx, dtype=np.int64),
#             "omega_raw": np.array(omega, dtype=np.float32),
#         }


# ----------------------------------------------------------------------------
# FNO model
# ----------------------------------------------------------------------------


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



# ----------------------------------------------------------------------------
# Loaders / splits
# ----------------------------------------------------------------------------


def _load_h5_meta(h5_path: Path) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as f:
        meta = {}
        if "metadata" in f:
            raw = f["metadata"][()]
            if isinstance(raw, (bytes, np.bytes_)):
                try:
                    meta = json.loads(raw.decode("utf-8"))
                except Exception:
                    meta = {}
            else:
                try:
                    meta = json.loads(str(raw))
                except Exception:
                    meta = {}
        omega = f["omega"][...].astype(np.float32)
        shape = f["data"].shape if "data" in f else f["fields_real"].shape

    return {
        "meta": meta,
        "omega": omega,
        "shape": shape,
    }


def _split_samples(num_samples: int, train_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    ids = np.arange(num_samples)
    rng.shuffle(ids)
    n_train = max(1, int(round(num_samples * train_ratio)))
    n_train = min(n_train, num_samples - 1) if num_samples > 1 else num_samples
    train_ids = ids[:n_train].tolist()
    val_ids = ids[n_train:].tolist()
    if not val_ids:
        val_ids = train_ids[-1:]
    return train_ids, val_ids


# ----------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------


@dataclass
class BatchStats:
    loss: float = 0.0
    rmse: float = 0.0
    n: int = 0


def _collate(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    x = torch.from_numpy(np.stack([b["x"] for b in batch], axis=0)).float()
    y = torch.from_numpy(np.stack([b["y"] for b in batch], axis=0)).float()
    mask = torch.from_numpy(np.stack([b["mask"] for b in batch], axis=0)).float()
    omega = torch.from_numpy(np.stack([b["omega"] for b in batch], axis=0)).float()
    sample_idx = torch.from_numpy(np.stack([b["sample_idx"] for b in batch], axis=0)).long()
    freq_idx = torch.from_numpy(np.stack([b["freq_idx"] for b in batch], axis=0)).long()
    omega_raw = torch.from_numpy(np.stack([b["omega_raw"] for b in batch], axis=0)).float()
    return {
        "x": x,
        "y": y,
        "mask": mask,
        "omega": omega,
        "sample_idx": sample_idx,
        "freq_idx": freq_idx,
        "omega_raw": omega_raw,
    }


def train_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    train_meta = _load_h5_meta(Path(args.train_h5))
    shape = train_meta["shape"]
    if len(shape) == 5:
        num_samples, num_freqs = int(shape[0]), int(shape[1])
    else:
        num_samples, num_freqs = int(shape[0]), int(shape[1])

    train_samples, val_samples = _split_samples(num_samples, args.train_ratio, args.seed)
    train_ds = HelmholtzFNOData(
        args.train_h5,
        sample_indices=train_samples,
        freq_indices=args.train_freq_indices,
        use_data=not args.no_use_data,
    )
    val_ds = HelmholtzFNOData(
        args.train_h5,
        sample_indices=val_samples,
        freq_indices=args.eval_freq_indices,
        use_data=not args.no_use_data,
    )

    in_ch = 6 if args.use_coords else 4
    model = ConditionalUNet2D(
        in_channels=in_ch + 2,
        out_channels=2,
        base_channels=args.base_channels,
        cond_dim=args.cond_dim,
        time_dim=args.time_dim,
        omega_bands=args.omega_bands,
        dropout=args.dropout,
        mid_attn_heads=args.mid_attn_heads,
        mid_attn_dropout=args.mid_attn_dropout,
    ).to(device)

    schedule = build_linear_schedule(
        num_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=32,
        collate_fn=_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=32,
        collate_fn=_collate,
        drop_last=False,
    )

    best_val = float("inf")
    history: List[Dict[str, float]] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 72)
    print("CoNFILD baseline training")
    print(f"train_h5={args.train_h5}")
    print(f"train_samples={len(train_samples)}, val_samples={len(val_samples)}, num_freqs={num_freqs}")
    print(f"input_channels={(6 if args.use_coords else 4)+2}, base_channels={args.base_channels}")
    print(f"device={device}, epochs={args.epochs}, batch_size={args.batch_size}")
    print("-" * 72 + "\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            omega = batch["omega"].to(device)
            if not args.use_coords:
                x = x[:, :4]
            
            bsz = x.shape[0]
            t = torch.randint(0, args.diffusion_steps, (bsz,), device=device, dtype=torch.long)
            noise = torch.randn_like(y)
            
            sqrt_ab = extract_to_batch(schedule.sqrt_alpha_bars, t, y)
            sqrt_omb = extract_to_batch(schedule.sqrt_one_minus_alpha_bars, t, y)
            yt = sqrt_ab * y + sqrt_omb * noise
            
            model_in = torch.cat([yt, x], dim=1)
            pred_noise = model(model_in, t, omega)
            
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss_sum += float(loss.item()) * int(x.shape[0])
            train_n += int(x.shape[0])

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                omega = batch["omega"].to(device)
                if not args.use_coords:
                    x = x[:, :4]
                    
                bsz = x.shape[0]
                t = torch.randint(0, args.diffusion_steps, (bsz,), device=device, dtype=torch.long)
                noise = torch.randn_like(y)
                sqrt_ab = extract_to_batch(schedule.sqrt_alpha_bars, t, y)
                sqrt_omb = extract_to_batch(schedule.sqrt_one_minus_alpha_bars, t, y)
                yt = sqrt_ab * y + sqrt_omb * noise
                
                model_in = torch.cat([yt, x], dim=1)
                pred_noise = model(model_in, t, omega)
                loss = F.mse_loss(pred_noise, noise)
                
                val_loss_sum += float(loss.item()) * int(x.shape[0])
                val_n += int(x.shape[0])

        train_loss = train_loss_sum / max(train_n, 1)
        val_loss = val_loss_sum / max(val_n, 1)
        scheduler.step(val_loss)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "model_config": {
                    "in_channels": (6 if args.use_coords else 4) + 2,
                    "out_channels": 2,
                    "base_channels": int(args.base_channels),
                    "cond_dim": int(args.cond_dim),
                    "time_dim": int(args.time_dim),
                    "omega_bands": int(args.omega_bands),
                    "dropout": float(args.dropout),
                    "mid_attn_heads": int(args.mid_attn_heads),
                    "mid_attn_dropout": float(args.mid_attn_dropout),
                    "use_coords": bool(args.use_coords),
                },
                "diffusion_config": {
                    "num_steps": int(args.diffusion_steps),
                    "beta_start": float(args.beta_start),
                    "beta_end": float(args.beta_end),
                },
                "train_config": vars(args),
                "best_val_loss": float(best_val),
                "train_samples": train_samples,
                "val_samples": val_samples,
                "train_h5": str(args.train_h5),
            }
            torch.save(ckpt, out_path)

        if args.log_every > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0):
            print(
                f"[epoch {epoch:04d}/{args.epochs}] train_loss={train_loss:.6e} "
                f"val_loss={val_loss:.6e} best_val={best_val:.6e} lr={optimizer.param_groups[0]['lr']:.3e}"
            )

    summary = {
        "out": str(out_path),
        "best_val_loss": float(best_val),
        "final_train_loss": float(history[-1]["train_loss"] if history else 0.0),
        "final_val_loss": float(history[-1]["val_loss"] if history else 0.0),
        "epochs": int(args.epochs),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
    }
    summary_path = out_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nTraining finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved checkpoint: {out_path}")
    print(f"Saved summary:     {summary_path}")


def _plot_eval_case(
    out_path: Path,
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    sample_idx: int,
    freq_idx: int,
    omega_val: float,
    rmse: float,
    obs_rmse: float,
    unobs_rmse: float,
    dpi: int,
    pde_res: float = 0.0,
) -> None:
    gt_re, gt_im = gt[..., 0], gt[..., 1]
    pr_re, pr_im = pred[..., 0], pred[..., 1]
    gt_amp = np.sqrt(gt_re**2 + gt_im**2)
    pr_amp = np.sqrt(pr_re**2 + pr_im**2)
    err = np.abs(pr_amp - gt_amp)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    items = [
        (gt_re, "GT Real", "viridis"),
        (pr_re, "Pred Real", "viridis"),
        (np.abs(pr_re - gt_re), "Abs Err Real", "magma"),
        (gt_im, "GT Imag", "viridis"),
        (pr_im, "Pred Imag", "viridis"),
        (np.abs(pr_im - gt_im), "Abs Err Imag", "magma"),
        (gt_amp, "GT Amp", "viridis"),
        (pr_amp, "Pred Amp", "viridis"),
        (err, "Abs Err Amp", "magma"),

    ]
    for ax, (img, title, cmap) in zip(axes.flat, items):
        # if title.startswith("GT") or title.startswith("Pred"):
        #     vmin, vmax = _symm_norm(img)
        #     im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        # else:
        im = ax.imshow(img, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"sample={sample_idx}, freq_idx={freq_idx}, omega={omega_val:.4f} | "
        f"rmse={rmse:.3e}, obs_rmse={obs_rmse:.3e}, unobs_rmse={unobs_rmse:.3e}, pde_res={pde_res:.3e}",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# def evaluate_model(args: argparse.Namespace) -> None:
#     set_seed(args.seed)

#     if args.device == "auto":
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     else:
#         device = torch.device(args.device)
#     ckpt = _safe_torch_load(Path(args.ckpt))
#     if "model_config" not in ckpt or "model_state" not in ckpt:
#         raise KeyError("Checkpoint must contain model_config and model_state")

#     model_cfg = dict(ckpt["model_config"])
#     use_coords = model_cfg.pop("use_coords", True)
#     model = ConditionalUNet2D(**model_cfg).to(device)
#     model.load_state_dict(ckpt["model_state"])
#     model.eval()
    
#     diff_cfg = ckpt.get("diffusion_config", {})
#     schedule = build_linear_schedule(
#         num_steps=diff_cfg.get("num_steps", 500),
#         beta_start=diff_cfg.get("beta_start", 1e-4),
#         beta_end=diff_cfg.get("beta_end", 2e-2),
#         device=device,
#     )

#     data_path = Path(args.test_h5)
#     meta = _load_h5_meta(data_path)
#     shape = meta["shape"]
#     num_samples, num_freqs = int(shape[0]), int(shape[1])
#     sample_indices = list(range(num_samples if args.max_samples <= 0 else min(num_samples, args.max_samples)))
#     freq_indices = _parse_indices(args.eval_freq_indices, num_freqs)
#     ds = HelmholtzFNOData(
#         data_path,
#         sample_indices=sample_indices,
#         freq_indices=args.eval_freq_indices,
#         use_data=not args.no_use_data,
#     )

#     loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=32, collate_fn=_collate)

#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     rows: List[Dict[str, Any]] = []
#     vis_count = 0
#     with torch.no_grad():
#         for batch in loader:
#             x = batch["x"].to(device)
#             y = batch["y"].to(device)
#             mask = batch["mask"].cpu().numpy()
#             sample_idx = batch["sample_idx"].cpu().numpy()
#             freq_idx = batch["freq_idx"].cpu().numpy()
#             omega_raw = batch["omega_raw"].cpu().numpy()
#             if not args.use_coords:
#                 x = x[:, :4]
#             omega = batch["omega"].to(device)
#             bsz = x.shape[0]
            
#             yt = torch.randn_like(y)
#             num_steps = ckpt.get("diffusion_config", {}).get("num_steps", 500)
            
#             for t_idx in reversed(range(num_steps)):
#                 t = torch.full((bsz,), t_idx, device=device, dtype=torch.long)
#                 with torch.no_grad():
#                     eps_pred = model(torch.cat([yt, x], dim=1), t, omega)
                
#                 abar_t = schedule.alpha_bars[t_idx]
#                 alpha_t = schedule.alphas[t_idx]
#                 beta_t = schedule.betas[t_idx]
                
#                 if t_idx > 0:
#                     noise = torch.randn_like(yt)
#                 else:
#                     noise = torch.zeros_like(yt)
                
#                 yt_prev = (1.0 / torch.sqrt(alpha_t)) * (yt - ((1.0 - alpha_t) / torch.sqrt(1.0 - abar_t)) * eps_pred) + torch.sqrt(beta_t) * noise
#                 yt = yt_prev
                
#             pred = yt.detach().cpu().numpy()
#             gt = y.detach().cpu().numpy()

#             for i in range(pred.shape[0]):
#                 pred_i = pred[i].transpose(1, 2, 0)
#                 gt_i = gt[i].transpose(1, 2, 0)
#                 mask_i = mask[i].transpose(1, 2, 0)
#                 rmse = _relative_rmse(pred_i, gt_i, eps=args.eps)
#                 obs_rmse = _masked_relative_rmse(pred_i, gt_i, mask_i, eps=args.eps)
#                 unobs_rmse = _masked_relative_rmse(pred_i, gt_i, 1.0 - mask_i, eps=args.eps)
#                 pde_res = evaluate_physics_residual(pred_i, ds.f, int(sample_idx[i]), float(omega_raw[i]), meta["meta"])
#                 pde_res = evaluate_physics_residual(pred_i, ds.f, int(sample_idx[i]), float(omega_raw[i]), meta["meta"])

#                 rows.append(
#                     {
#                         "sample_idx": int(sample_idx[i]),
#                         "freq_idx": int(freq_idx[i]),
#                         "omega": float(omega_raw[i]),
#                         "rmse": float(rmse),
#                         "obs_rmse": float(obs_rmse),
#                         "unobs_rmse": float(unobs_rmse),
#                         "pde_res": float(pde_res),
#                     }
#                 )

#                 if vis_count < args.num_visualize:
#                     vis_count += 1
#                     vis_path = out_dir / f"case{vis_count:03d}_sample{int(sample_idx[i]):03d}_freq{int(freq_idx[i]):03d}.png"
#                     _plot_eval_case(
#                         out_path=vis_path,
#                         gt=gt_i,
#                         pred=pred_i,
#                         mask=mask_i,
#                         sample_idx=int(sample_idx[i]),
#                         freq_idx=int(freq_idx[i]),
#                         omega_val=float(omega_raw[i]),
#                         rmse=rmse,
#                         obs_rmse=obs_rmse,
#                         unobs_rmse=unobs_rmse,
#                         dpi=args.vis_dpi,
#                         pde_res=pde_res,
#                         pde_res=pde_res,
#                     )
#                     rows[-1]["vis_path"] = str(vis_path)

#     if not rows:
#         raise RuntimeError("No evaluation rows produced")

#     csv_path = out_dir / "metrics_cases.csv"
#     with open(csv_path, "w", encoding="utf-8", newline="") as f:
#         writer = csv.DictWriter(
#             f,
#             fieldnames=["sample_idx", "freq_idx", "omega", "rmse", "obs_rmse", "unobs_rmse", "pde_res", "vis_path"],
#         )
#         writer.writeheader()
#         for row in rows:
#             writer.writerow(row)

#     rmses = np.array([r["rmse"] for r in rows], dtype=np.float64)
#     obs_rmses = np.array([r["obs_rmse"] for r in rows], dtype=np.float64)
#     unobs_rmses = np.array([r["unobs_rmse"] for r in rows], dtype=np.float64)
#     pde_ress = np.array([r["pde_res"] for r in rows if not np.isnan(r.get("pde_res", np.nan))], dtype=np.float64)
#     pde_ress = np.array([r["pde_res"] for r in rows if not np.isnan(r.get("pde_res", np.nan))], dtype=np.float64)

#     summary = {
#         "ckpt": str(args.ckpt),
#         "test_h5": str(data_path),
#         "num_cases": int(len(rows)),
#         "mean_rmse": float(np.mean(rmses)),
#         "mean_obs_rmse": float(np.mean(obs_rmses)),
#         "mean_unobs_rmse": float(np.mean(unobs_rmses)),
#         "mean_pde_res": float(np.mean(pde_ress)) if len(pde_ress) > 0 else 0.0,
#         "mean_pde_res": float(np.mean(pde_ress)) if len(pde_ress) > 0 else 0.0,
#         "num_visualized": int(vis_count),
#         "output_dir": str(out_dir),
#     }
#     summary_path = out_dir / "summary.json"
#     with open(summary_path, "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2, ensure_ascii=False)

#     print("\nEvaluation finished.")
#     print(json.dumps(summary, indent=2, ensure_ascii=False))
#     print(f"Saved metrics: {csv_path}")
#     print(f"Saved summary: {summary_path}")

def evaluate_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    ckpt = _safe_torch_load(Path(args.ckpt))
    if "model_config" not in ckpt or "model_state" not in ckpt:
        raise KeyError("Checkpoint must contain model_config and model_state")

    model_cfg = dict(ckpt["model_config"])
    use_coords = model_cfg.pop("use_coords", True)
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

    data_path = Path(args.test_h5)
    meta = _load_h5_meta(data_path)
    shape = meta["shape"]
    num_samples, num_freqs = int(shape[0]), int(shape[1])
    sample_indices = list(range(num_samples if args.max_samples <= 0 else min(num_samples, args.max_samples)))
    
    # ===================== 修复 1：手动打开 HDF5 文件给残差计算用 =====================
    h5_file = h5py.File(data_path, "r")

    ds = HelmholtzFNOData(
        data_path,
        sample_indices=sample_indices,
        freq_indices=args.eval_freq_indices,
        use_data=not args.no_use_data,
    )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    vis_count = 0
    num_steps = diff_cfg.get("num_steps", 500)

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].cpu().numpy()
            sample_idx = batch["sample_idx"].cpu().numpy()
            freq_idx = batch["freq_idx"].cpu().numpy()
            omega_raw = batch["omega_raw"].cpu().numpy()
            omega = batch["omega"].to(device)
            bsz = x.shape[0]
            
            # 扩散采样
            yt = torch.randn_like(y)
            for t_idx in reversed(range(num_steps)):
                t = torch.full((bsz,), t_idx, device=device, dtype=torch.long)
                eps_pred = model(torch.cat([yt, x], dim=1), t, omega)
                
                abar_t = schedule.alpha_bars[t_idx]
                alpha_t = schedule.alphas[t_idx]
                beta_t = schedule.betas[t_idx]
                
                noise = torch.randn_like(yt) if t_idx > 0 else 0.0
                yt = (1.0 / torch.sqrt(alpha_t)) * (yt - ((1.0 - alpha_t) / torch.sqrt(1.0 - abar_t)) * eps_pred) + torch.sqrt(beta_t) * noise
                
            pred = yt.cpu().numpy()
            gt = y.cpu().numpy()

            for i in range(pred.shape[0]):
                pred_i = pred[i].transpose(1, 2, 0)
                gt_i = gt[i].transpose(1, 2, 0)
                mask_i = mask[i].transpose(1, 2, 0)
                rmse = _relative_rmse(pred_i, gt_i, eps=args.eps)
                obs_rmse = _masked_relative_rmse(pred_i, gt_i, mask_i, eps=args.eps)
                unobs_rmse = _masked_relative_rmse(pred_i, gt_i, 1.0 - mask_i, eps=args.eps)
                
                # ===================== 核心：物理残差计算 =====================
                try:
                    pde_res = evaluate_physics_residual(
                        pred=pred_i,
                        h5_file=h5_file,
                        sample_idx=int(sample_idx[i]),
                        omega=float(omega_raw[i]),
                        h5_meta=meta["meta"]
                    )
                except Exception as e:
                    pde_res = np.nan

                rows.append({
                    "sample_idx": int(sample_idx[i]),
                    "freq_idx": int(freq_idx[i]),
                    "omega": float(omega_raw[i]),
                    "rmse": float(rmse),
                    "obs_rmse": float(obs_rmse),
                    "unobs_rmse": float(unobs_rmse),
                    "pde_res": float(pde_res),
                })

                if vis_count < args.num_visualize:
                    vis_count += 1
                    vis_path = out_dir / f"case{vis_count:03d}_sample{int(sample_idx[i]):03d}_freq{int(freq_idx[i]):03d}.png"
                    _plot_eval_case(
                        out_path=vis_path,
                        gt=gt_i, pred=pred_i, mask=mask_i,
                        sample_idx=int(sample_idx[i]),
                        freq_idx=int(freq_idx[i]),
                        omega_val=float(omega_raw[i]),
                        rmse=rmse, obs_rmse=obs_rmse, unobs_rmse=unobs_rmse,
                        dpi=args.vis_dpi, pde_res=pde_res
                    )
                    rows[-1]["vis_path"] = str(vis_path)

    h5_file.close()

    if not rows:
        raise RuntimeError("No evaluation rows produced")

    csv_path = out_dir / "metrics_cases.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_idx", "freq_idx", "omega", "rmse", "obs_rmse", "unobs_rmse", "pde_res", "vis_path"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    rmses = np.array([r["rmse"] for r in rows], dtype=np.float64)
    obs_rmses = np.array([r["obs_rmse"] for r in rows], dtype=np.float64)
    unobs_rmses = np.array([r["unobs_rmse"] for r in rows], dtype=np.float64)
    pde_ress = np.array([r["pde_res"] for r in rows if not np.isnan(r.get("pde_res", np.nan))], dtype=np.float64)

    summary = {
        "ckpt": str(args.ckpt),
        "test_h5": str(data_path),
        "num_cases": int(len(rows)),
        "mean_rmse": float(np.mean(rmses)),
        "mean_obs_rmse": float(np.mean(obs_rmses)),
        "mean_unobs_rmse": float(np.mean(unobs_rmses)),
        "mean_pde_res": float(np.mean(pde_ress)) if len(pde_ress) > 0 else 0.0,
        "num_visualized": int(vis_count),
        "output_dir": str(out_dir),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nEvaluation finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved metrics: {csv_path}")
# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="2D CoNFILD baseline for Helmholtz reconstruction")
    p.add_argument("--mode", type=str, default="eval", choices=["train", "eval"])

    p.add_argument("--train_h5", type=str, default="helmholtz_dataset_42.h5")
    p.add_argument("--test_h5", type=str, default="data_for_test/helmholtz_dataset_42_for_test_mask1.h5")
    p.add_argument("--ckpt", type=str, default="ckp/confild_baseline.pt")
    p.add_argument("--out", type=str, default="ckp/confild_baseline.pt")
    p.add_argument("--out_dir", type=str, default="visual_data/confild_baseline_eval/mask_ratio1")

    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--train_freq_indices", type=str, default="")
    p.add_argument("--eval_freq_indices", type=str, default="")
    p.add_argument("--max_samples", type=int, default=1)

    p.add_argument("--no_use_data", action="store_true")
    p.add_argument("--use_coords", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=5)

    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--time_dim", type=int, default=128)
    p.add_argument("--omega_bands", type=int, default=12)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--mid_attn_heads", type=int, default=4)
    p.add_argument("--mid_attn_dropout", type=float, default=0.0)

    p.add_argument("--diffusion_steps", type=int, default=500)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)

    p.add_argument("--num_visualize", type=int, default=50)
    p.add_argument("--vis_dpi", type=int, default=180)

    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "train":
        train_model(args)
    else:
        evaluate_model(args)


if __name__ == "__main__":
    main()
