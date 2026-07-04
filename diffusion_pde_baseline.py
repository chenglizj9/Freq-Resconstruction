"""
diffusion_pde_baseline.py
-------------------------
Pixel-space diffusion baseline for 2D Helmholtz reconstruction.
DiffusionPDE-style [Huang et al., NeurIPS 2024]:
  - Train: ω-conditional score network on full fields (NO obs/mask in model input)
  - Eval:  DDPM reverse process + DPS guidance (fixed-ζ observation constraint)
           gradient approximation: g_t ≈ mask^T (mask ⊙ x̂_0 − y) (no autograd through score net)

Key contrast with our FTM method:
  - Operates in PIXEL SPACE (expensive for high resolution, no frequency extrapolation)
  - DPS gradient requires evaluation at full field resolution (vs closed-form Tucker matmul)
  - Fixed ζ requires per-dataset tuning (vs self-calibrated guidance in our method)

Input to score net : [noisy_field_C_channels]  + ω embedding (FiLM)
Output             : predicted noise (C channels)

Usage
-----
Train:
    python diffusion_pde_baseline.py --mode train \\
        --train_h5 helmholtz_dataset_42.h5 --out ckp/dpde_baseline.pt
Eval:
    python diffusion_pde_baseline.py --mode eval \\
        --ckpt ckp/dpde_baseline.pt \\
        --test_h5 data_for_test/helmholtz_dataset_42_for_test_mask1.h5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
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
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _normalize_omega(omega: float, omega_min: float, omega_max: float) -> float:
    return float((omega - omega_min) / max(omega_max - omega_min, 1e-12))


def _parse_indices(text: str, limit: int) -> List[int]:
    if text.strip() == "":
        return list(range(limit))
    out = []
    for p in text.split(","):
        p = p.strip()
        if p:
            idx = int(p)
            if idx < 0 or idx >= limit:
                raise ValueError(f"index {idx} out of [0,{limit-1}]")
            out.append(idx)
    return sorted(set(out))


def _relative_rmse(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.sum((pred - gt) ** 2) / max(np.sum(gt ** 2), eps)))


def _masked_relative_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                           eps: float = 1e-12) -> float:
    m = mask.astype(bool)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    if not np.any(m):
        return float("nan")
    diff = (pred - gt) ** 2; gt_sq = gt ** 2
    if diff.ndim == 3 and diff.shape[-1] == 2:
        diff = np.sum(diff, axis=-1); gt_sq = np.sum(gt_sq, axis=-1)
    return float(np.sqrt(np.sum(diff[m]) / max(np.sum(gt_sq[m]), eps)))


def _safe_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# Dataset  (trains on FULL fields; obs/mask loaded only for eval DPS guidance)
# ---------------------------------------------------------------------------

class HelmholtzDPDEData(Dataset):
    """Returns full field + mask always; obs (masked field) returned only for eval DPS."""

    def __init__(
        self,
        h5_path: str | Path,
        sample_indices: Optional[Sequence[int]] = None,
        freq_indices: str = "",
        return_obs: bool = False,   # True during eval — also returns masked observations
    ) -> None:
        self.path = Path(h5_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.return_obs = return_obs

        with h5py.File(self.path, "r") as f:
            self.omega = f["omega"][...].astype(np.float32)
            self.omega_min = float(np.min(self.omega))
            self.omega_max = float(np.max(self.omega))

            if "data" in f:
                self.data = f["data"][...].astype(np.float32)
                self.N, self.M, self.H, self.W, self.C = self.data.shape
            else:
                re = f["fields_real"][...].astype(np.float32)
                im = f["fields_imag"][...].astype(np.float32)
                self.data = np.stack([re, im], axis=-1)
                self.N, self.M, self.H, self.W = re.shape; self.C = 2

            self.has_mask = "mask_tr" in f
            self.mask_data = f["mask_tr"][...].astype(np.float32) if self.has_mask else None

        self.sample_indices = list(range(self.N)) if sample_indices is None \
            else [int(i) for i in sample_indices]
        self.freq_indices = _parse_indices(freq_indices, self.M)
        self.pairs = [(b, m) for b in self.sample_indices for m in self.freq_indices]

    def __len__(self): return len(self.pairs)

    def _get_mask(self, s, m):
        if not self.has_mask:
            return np.ones((self.H, self.W, 1), dtype=np.float32)
        md = self.mask_data
        if md.ndim == 4:   md = md[m]
        elif md.ndim == 5: md = md[s, m]
        if md.shape[-1] == 2: md = md[..., :1]
        return md

    def __getitem__(self, idx):
        s, m    = self.pairs[idx]
        field   = self.data[s, m]           # (H, W, C)
        omega   = float(self.omega[m])
        omega_n = _normalize_omega(omega, self.omega_min, self.omega_max)

        mask = self._get_mask(s, m)                        # (H, W, 1)
        y    = field.transpose(2, 0, 1).astype(np.float32)  # (C, H, W)
        out = {
            "y":         y,
            "mask":      mask.transpose(2, 0, 1).astype(np.float32),  # (1, H, W)
            "omega":     np.array([omega_n],  dtype=np.float32),
            "sample_idx":np.array(s,          dtype=np.int64),
            "freq_idx":  np.array(m,          dtype=np.int64),
            "omega_raw": np.array(omega,      dtype=np.float32),
        }
        if self.return_obs:
            out["obs"] = (field * mask).transpose(2, 0, 1).astype(np.float32)  # (C, H, W)
        return out


# ---------------------------------------------------------------------------
# Diffusion schedule
# ---------------------------------------------------------------------------

class DiffusionSchedule:
    def __init__(self, betas: torch.Tensor):
        self.betas  = betas
        alphas      = 1.0 - betas
        self.alphas = alphas
        self.alpha_bars            = torch.cumprod(alphas, 0)
        self.sqrt_alpha_bars       = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    @classmethod
    def cosine(cls, T: int, s: float = 0.008, device="cpu") -> "DiffusionSchedule":
        t  = torch.linspace(0, T, T + 1, device=device) / T
        ab = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        ab = ab / ab[0]; ab = ab[:-1]
        al = torch.zeros_like(ab)
        al[1:] = ab[1:] / ab[:-1]; al[0] = ab[0]
        return cls(1.0 - al)

    @classmethod
    def linear(cls, T: int, b0: float = 1e-4, b1: float = 2e-2, device="cpu") -> "DiffusionSchedule":
        return cls(torch.linspace(b0, b1, T, device=device))

    def to(self, device):
        self.betas                     = self.betas.to(device)
        self.alphas                    = self.alphas.to(device)
        self.alpha_bars                = self.alpha_bars.to(device)
        self.sqrt_alpha_bars           = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        return self

    def gather(self, vals: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = vals.gather(0, t).to(dtype=x.dtype)
        return out.view(-1, *([1] * (x.ndim - 1)))


def _extract(sched: DiffusionSchedule, t: torch.Tensor, x: torch.Tensor):
    return sched.gather(sched.alpha_bars, t, x)


# ---------------------------------------------------------------------------
# Score network: ω-conditional U-Net  (no obs/mask in input)
# ---------------------------------------------------------------------------

def _gn(ch: int, max_g: int = 8) -> int:
    g = min(max_g, ch)
    while g > 1 and ch % g != 0: g -= 1
    return g


class FourierOmegaEmb(nn.Module):
    def __init__(self, bands: int = 8):
        super().__init__(); self.bands = bands; self.out_dim = 1 + 2 * bands

    def forward(self, omega_norm: torch.Tensor) -> torch.Tensor:
        feats = [omega_norm]
        for k in range(self.bands):
            f = (2 ** k) * math.pi
            feats += [torch.sin(f * omega_norm), torch.cos(f * omega_norm)]
        return torch.cat(feats, -1)


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb   = torch.cat([torch.sin(args), torch.cos(args)], -1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ScoreResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.n1   = nn.GroupNorm(_gn(in_ch),  in_ch)
        self.c1   = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n2   = nn.GroupNorm(_gn(out_ch), out_ch)
        self.c2   = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cp   = nn.Linear(cond_dim, 2 * out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        h = self.n2(h)
        sc = self.cp(cond)
        sc, sh = sc.chunk(2, -1)
        h = h * (1 + sc.view(-1, sc.shape[-1], 1, 1)) + sh.view(-1, sh.shape[-1], 1, 1)
        h = self.c2(F.silu(h))
        return h + self.skip(x)


class PixelScoreUNet(nn.Module):
    """ω-conditional U-Net score network for pixel-space diffusion (no obs input)."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 64,
        cond_dim: int = 256,
        time_dim: int = 128,
        omega_bands: int = 8,
    ):
        super().__init__()
        self.time_emb  = SinusoidalTimeEmb(time_dim)
        self.omega_emb = FourierOmegaEmb(omega_bands)
        cond_in = time_dim + self.omega_emb.out_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.stem  = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.enc1  = ScoreResBlock(c1, c1, cond_dim)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc2  = ScoreResBlock(c2, c2, cond_dim)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.mid   = ScoreResBlock(c3, c3, cond_dim)
        self.dec2  = ScoreResBlock(c3 + c2, c2, cond_dim)
        self.dec1  = ScoreResBlock(c2 + c1, c1, cond_dim)
        self.out_n = nn.GroupNorm(_gn(c1), c1)
        self.out_c = nn.Conv2d(c1, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, omega_norm: torch.Tensor) -> torch.Tensor:
        cond = self.cond_mlp(torch.cat([self.time_emb(t), self.omega_emb(omega_norm)], -1))
        e0   = self.stem(x)
        e1   = self.enc1(e0, cond)
        e2   = self.enc2(self.down1(e1), cond)
        m    = self.mid(self.down2(e2), cond)
        u2   = self.dec2(torch.cat([F.interpolate(m, e2.shape[-2:], mode="nearest"), e2], 1), cond)
        u1   = self.dec1(torch.cat([F.interpolate(u2, e1.shape[-2:], mode="nearest"), e1], 1), cond)
        return self.out_c(F.silu(self.out_n(u1)))


# ---------------------------------------------------------------------------
# Data splits / loaders
# ---------------------------------------------------------------------------

def _split(n: int, ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    ids = rng.permutation(n)
    k   = max(1, min(int(round(n * ratio)), n - 1) if n > 1 else n)
    return ids[:k].tolist(), (ids[k:].tolist() or ids[-1:].tolist())


def _collate_train(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    keys = ["y", "mask", "omega", "sample_idx", "freq_idx", "omega_raw"]
    out  = {k: torch.from_numpy(np.stack([b[k] for b in batch])) for k in keys}
    out["sample_idx"] = out["sample_idx"].long()
    out["freq_idx"]   = out["freq_idx"].long()
    return out


def _collate_eval(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    keys = ["y", "omega", "sample_idx", "freq_idx", "omega_raw", "mask", "obs"]
    out  = {k: torch.from_numpy(np.stack([b[k] for b in batch])) for k in keys}
    out["sample_idx"] = out["sample_idx"].long()
    out["freq_idx"]   = out["freq_idx"].long()
    return out


def _load_meta(path: Path) -> Dict[str, Any]:
    with h5py.File(path, "r") as f:
        meta = {}
        if "metadata" in f:
            raw = f["metadata"][()]
            try: meta = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
            except Exception: pass
        omega = f["omega"][...].astype(np.float32)
        shape = f["data"].shape if "data" in f else f["fields_real"].shape
    return {"meta": meta, "omega": omega, "shape": shape}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    meta = _load_meta(Path(args.train_h5))
    N    = int(meta["shape"][0])
    C    = int(meta["shape"][-1])
    tr_ids, va_ids = _split(N, args.train_ratio, args.seed)

    tr_ds = HelmholtzDPDEData(args.train_h5, tr_ids, args.train_freq_indices, return_obs=False)
    va_ds = HelmholtzDPDEData(args.train_h5, va_ids, args.eval_freq_indices,  return_obs=False)
    # mask is always returned by the dataset; loss is computed only at observed points

    model = PixelScoreUNet(
        in_channels=C, out_channels=C,
        base_channels=args.base_channels, cond_dim=args.cond_dim,
        time_dim=args.time_dim, omega_bands=args.omega_bands,
    ).to(device)

    sched = DiffusionSchedule.linear(args.T, args.beta_start, args.beta_end, device=device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

    tr_dl = DataLoader(tr_ds, args.batch_size, shuffle=True,  num_workers=4, collate_fn=_collate_train)
    va_dl = DataLoader(va_ds, args.batch_size, shuffle=False, num_workers=4, collate_fn=_collate_train)

    best_val = float("inf")
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nDiffusionPDE training | C={C} base={args.base_channels} T={args.T} device={device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tl = 0.0; tn = 0
        for batch in tr_dl:
            y     = batch["y"].to(device)            # (B, C, H, W)
            mask  = batch["mask"].to(device)          # (B, 1, H, W)
            omega = batch["omega"].to(device)         # (B, 1)
            B     = y.shape[0]
            t     = torch.randint(0, args.T, (B,), device=device)
            noise = torch.randn_like(y)
            sa    = sched.gather(sched.sqrt_alpha_bars,           t, y)
            sb    = sched.gather(sched.sqrt_one_minus_alpha_bars, t, y)
            y_t   = sa * y + sb * noise

            eps_pred = model(y_t, t, omega)
            # loss only at observed sensor locations (fair comparison with sparse-obs methods)
            diff = (eps_pred - noise) ** 2           # (B, C, H, W)
            n_obs = mask.sum() * y.shape[1]          # observed positions × channels
            loss = (diff * mask).sum() / n_obs.clamp(min=1)
            opt.zero_grad(set_to_none=True); loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tl += loss.item() * B; tn += B

        model.eval()
        vl = 0.0; vn = 0
        with torch.no_grad():
            for batch in va_dl:
                y    = batch["y"].to(device); mask = batch["mask"].to(device)
                omega = batch["omega"].to(device)
                B = y.shape[0]
                t = torch.randint(0, args.T, (B,), device=device)
                noise = torch.randn_like(y)
                sa = sched.gather(sched.sqrt_alpha_bars, t, y)
                sb = sched.gather(sched.sqrt_one_minus_alpha_bars, t, y)
                y_t = sa * y + sb * noise
                eps_pred = model(y_t, t, omega)
                diff = (eps_pred - noise) ** 2
                n_obs = mask.sum() * y.shape[1]
                vl += ((diff * mask).sum() / n_obs.clamp(min=1)).item() * B; vn += B

        tl /= max(tn,1); vl /= max(vn,1)
        lr_sched.step(vl)
        if vl < best_val:
            best_val = vl
            torch.save({
                "model_state": model.state_dict(),
                "model_config": {"in_channels": C, "out_channels": C,
                                 "base_channels": args.base_channels,
                                 "cond_dim": args.cond_dim,
                                 "time_dim": args.time_dim,
                                 "omega_bands": args.omega_bands},
                "diffusion_config": {"T": args.T, "beta_start": args.beta_start,
                                     "beta_end": args.beta_end},
                "train_config": vars(args), "best_val_loss": best_val,
                "train_h5": args.train_h5,
            }, out_path)

        if args.log_every > 0 and (epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs):
            print(f"[{epoch:04d}/{args.epochs}] train={tl:.4e} val={vl:.4e} best={best_val:.4e}")

    print(f"Done. best_val={best_val:.4e}  ckpt={out_path}")


# ---------------------------------------------------------------------------
# Evaluation with DPS guidance
# ---------------------------------------------------------------------------

def _plot_eval(out_path, gt, pred, mask, s, f, omega, rmse, obs_r, unobs_r, pde_res=0.0, dpi=180):
    gt_re, gt_im = gt[..., 0], gt[..., 1]
    pr_re, pr_im = pred[..., 0], pred[..., 1]
    gt_amp = np.sqrt(gt_re**2 + gt_im**2); pr_amp = np.sqrt(pr_re**2 + pr_im**2)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    items = [
        (gt_re,"GT Re","viridis"),(pr_re,"Pred Re","viridis"),(np.abs(pr_re-gt_re),"Err Re","magma"),
        (gt_im,"GT Im","viridis"),(pr_im,"Pred Im","viridis"),(np.abs(pr_im-gt_im),"Err Im","magma"),
        (gt_amp,"GT Amp","viridis"),(pr_amp,"Pred Amp","viridis"),(np.abs(pr_amp-gt_amp),"Err Amp","magma"),
    ]
    for ax, (img, title, cmap) in zip(axes.flat, items):
        im = ax.imshow(img, origin="lower", cmap=cmap)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"s={s} f={f} ω={omega:.3f} | rmse={rmse:.2e} obs={obs_r:.2e} unobs={unobs_r:.2e} pde={pde_res:.2e}",
        y=0.995)
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig)


@torch.no_grad()
def dps_sample(
    model: PixelScoreUNet,
    sched: DiffusionSchedule,
    omega: torch.Tensor,         # (B, 1)
    y_obs: torch.Tensor,         # (B, C, H, W) – sparse observation (0 outside mask)
    mask: torch.Tensor,          # (B, 1, H, W) – binary mask
    T: int,
    zeta: float,
    device: torch.device,
) -> torch.Tensor:
    """DDPM reverse + DPS observation guidance.

    Gradient approximation (no autograd through score net):
      x̂_0 = (x_t - sqrt(1-ᾱt) * ε_θ) / sqrt(ᾱt)
      g_t  = mask ⊙ (mask ⊙ x̂_0 − y_obs)   (gradient of ½‖M(x̂_0) − y‖²)
      x_t  ← x_{t-1}^DDPM − ζ_t * g_t

    ζ_t is scale-invariant: ζ_t = zeta * ‖Δx_prior‖ / (‖g_t‖ + eps)
    to match the plan's "scale-invariant normalization" (boundable hyperparameter).
    """
    B = y_obs.shape[0]
    x = torch.randn_like(y_obs)

    for t_idx in reversed(range(T)):
        t_vec = torch.full((B,), t_idx, device=device, dtype=torch.long)

        # Score prediction
        with torch.no_grad():
            eps_pred = model(x, t_vec, omega)

        ab_t  = sched.alpha_bars[t_idx]
        a_t   = sched.alphas[t_idx]
        b_t   = sched.betas[t_idx]

        # DDPM prior step
        x0hat = (x - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt().clamp(min=1e-8)
        noise = torch.randn_like(x) if t_idx > 0 else torch.zeros_like(x)
        x_prior = (1.0 / a_t.sqrt()) * (x - (1.0 - a_t) / (1.0 - ab_t).sqrt() * eps_pred) \
                  + b_t.sqrt() * noise

        # DPS guidance step (linear mask, approximate gradient)
        residual = mask * x0hat - y_obs        # (B, C, H, W)
        g_t      = mask * residual              # gradient of ½‖M(x̂_0) − y‖²

        # Scale-invariant step size
        # .norm() routes to matrix_norm for >2 dims; use explicit L2 instead
        prior_step_norm = (x_prior - x).pow(2).sum(dim=(1,2,3), keepdim=True).sqrt().clamp(min=1e-8)
        g_norm          = g_t.pow(2).sum(dim=(1,2,3), keepdim=True).sqrt().clamp(min=1e-8)


        zeta_t          = zeta * prior_step_norm / g_norm

        x = x_prior - zeta_t * g_t

    return x


def evaluate_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    ckpt = _safe_load(Path(args.ckpt))
    model = PixelScoreUNet(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dcfg = ckpt.get("diffusion_config", {})
    T    = dcfg.get("T", args.T)
    sched = DiffusionSchedule.linear(T, dcfg.get("beta_start", args.beta_start),
                                     dcfg.get("beta_end", args.beta_end)).to(device)

    data_path = Path(args.test_h5)
    meta = _load_meta(data_path)
    N    = int(meta["shape"][0])
    max_s = N if args.max_samples <= 0 else min(N, args.max_samples)
    ds   = HelmholtzDPDEData(data_path, list(range(max_s)), args.eval_freq_indices, return_obs=True)
    dl   = DataLoader(ds, args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate_eval)

    h5_file = h5py.File(data_path, "r")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []; vis_cnt = 0
    for batch in dl:
        y       = batch["y"].to(device)
        omega   = batch["omega"].to(device)
        mask    = batch["mask"].to(device)
        obs     = batch["obs"].to(device)
        s_idx   = batch["sample_idx"].cpu().numpy()
        f_idx   = batch["freq_idx"].cpu().numpy()
        omega_r = batch["omega_raw"].cpu().numpy()

        pred_t = dps_sample(model, sched, omega, obs, mask, T,
                            args.zeta, device)

        pred_np = pred_t.cpu().numpy()
        gt_np   = y.cpu().numpy()
        mask_np = batch["mask"].numpy()

        for i in range(pred_np.shape[0]):
            pred_i  = pred_np[i].transpose(1,2,0)
            gt_i    = gt_np[i].transpose(1,2,0)
            mask_i  = mask_np[i].transpose(1,2,0)
            rmse    = _relative_rmse(pred_i, gt_i)
            obs_r   = _masked_relative_rmse(pred_i, gt_i, mask_i)
            unobs_r = _masked_relative_rmse(pred_i, gt_i, 1.0 - mask_i)
            try:
                pde_res = evaluate_physics_residual(
                    pred=pred_i, h5_file=h5_file,
                    sample_idx=int(s_idx[i]), omega=float(omega_r[i]),
                    h5_meta=meta["meta"])
            except Exception:
                pde_res = float("nan")

            row = {"sample_idx": int(s_idx[i]), "freq_idx": int(f_idx[i]),
                   "omega": float(omega_r[i]), "rmse": rmse,
                   "obs_rmse": obs_r, "unobs_rmse": unobs_r, "pde_res": pde_res}
            rows.append(row)
            if vis_cnt < args.num_visualize:
                vis_cnt += 1
                vp = out_dir / f"case{vis_cnt:03d}_s{int(s_idx[i]):03d}_f{int(f_idx[i]):03d}.png"
                _plot_eval(vp, gt_i, pred_i, mask_i, int(s_idx[i]), int(f_idx[i]),
                           float(omega_r[i]), rmse, obs_r, unobs_r, pde_res, args.vis_dpi)
                rows[-1]["vis_path"] = str(vp)

    h5_file.close()
    if not rows: raise RuntimeError("No evaluation rows")

    csv_path = out_dir / "metrics_cases.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, ["sample_idx","freq_idx","omega","rmse","obs_rmse","unobs_rmse","pde_res","vis_path"])
        w.writeheader()
        for r in rows: w.writerow(r)

    summary = {
        "ckpt": args.ckpt, "test_h5": str(data_path), "num_cases": len(rows),
        "dps_zeta": args.zeta,
        "mean_rmse":       float(np.mean([r["rmse"]      for r in rows])),
        "mean_obs_rmse":   float(np.mean([r["obs_rmse"]  for r in rows])),
        "mean_unobs_rmse": float(np.mean([r["unobs_rmse"]for r in rows])),
        "mean_pde_res":    float(np.nanmean([r["pde_res"]for r in rows])),
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nEvaluation finished.")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("DiffusionPDE-style baseline — 2D Helmholtz")
    p.add_argument("--mode", choices=["train","eval"], default="eval")
    p.add_argument("--train_h5", default="helmholtz_dataset_42.h5")
    p.add_argument("--test_h5",  default="data_for_test/helmholtz_dataset_42_for_test_mask1.h5")
    p.add_argument("--ckpt",     default="ckp/dpde_baseline.pt")
    p.add_argument("--out",      default="ckp/dpde_baseline.pt")
    p.add_argument("--out_dir",  default="visual_data/dpde_baseline_eval/mask_ratio1")
    p.add_argument("--train_ratio",        type=float, default=0.8)
    p.add_argument("--train_freq_indices", type=str,   default="")
    p.add_argument("--eval_freq_indices",  type=str,   default="")
    p.add_argument("--max_samples",        type=int,   default=-1)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--wd",         type=float, default=1e-6)
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--log_every",  type=int,   default=10)
    p.add_argument("--base_channels", type=int,   default=64)
    p.add_argument("--cond_dim",      type=int,   default=256)
    p.add_argument("--time_dim",      type=int,   default=128)
    p.add_argument("--omega_bands",   type=int,   default=8)
    p.add_argument("--T",          type=int,   default=500)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end",   type=float, default=2e-2)
    # DPS guidance weight (scale-invariant: lambda in (0,1))
    p.add_argument("--zeta",       type=float, default=0.3,
                   help="Scale-invariant DPS guidance weight λ ∈ (0,1)")
    p.add_argument("--num_visualize", type=int, default=50)
    p.add_argument("--vis_dpi",  type=int,   default=150)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--device",   type=str,   default="auto")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "train":
        train_model(args)
    else:
        evaluate_model(args)


if __name__ == "__main__":
    main()
