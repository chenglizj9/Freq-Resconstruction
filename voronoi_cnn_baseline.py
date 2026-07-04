"""
voronoi_cnn_baseline.py
-----------------------
Voronoi-CNN baseline for 2D Helmholtz sparse field reconstruction.
[Fukami et al., Nature Machine Intelligence, 2021]

Pre-processing: fill sparse observations with nearest-neighbor (Voronoi) interpolation
                via scipy cKDTree, then refine with a CNN.

Input to CNN  : [voronoi_re, voronoi_im, mask, omega_norm, x_coord, y_coord]  (6 ch)
Output        : [re, im]  (2 ch)

Usage
-----
Train:
    python voronoi_cnn_baseline.py --mode train \\
        --train_h5 helmholtz_dataset_42.h5 --out ckp/voronoi_cnn.pt
Eval:
    python voronoi_cnn_baseline.py --mode eval \\
        --ckpt ckp/voronoi_cnn.pt \\
        --test_h5 data_for_test/helmholtz_dataset_42_for_test_mask1.h5
"""

from __future__ import annotations

import argparse
import csv
import json
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
from scipy.spatial import cKDTree
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
    diff = (pred - gt) ** 2
    gt_sq = gt ** 2
    if diff.ndim == 3 and diff.shape[-1] == 2:
        diff = np.sum(diff, axis=-1)
        gt_sq = np.sum(gt_sq, axis=-1)
    return float(np.sqrt(np.sum(diff[m]) / max(np.sum(gt_sq[m]), eps)))


def _safe_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# Voronoi nearest-neighbor fill
# ---------------------------------------------------------------------------

def voronoi_fill(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Nearest-neighbor Voronoi fill from sparse observed points to full grid.

    Args:
        field : (H, W, C)  full field (only observed positions are reliable)
        mask  : (H, W, 1)  binary observation mask

    Returns:
        filled : (H, W, C)  Voronoi-interpolated field
    """
    H, W, C = field.shape
    obs_mask = mask[..., 0].astype(bool)       # (H, W)
    obs_yx = np.argwhere(obs_mask)             # (K, 2)  [row, col]
    if len(obs_yx) == 0:
        return np.zeros_like(field)
    obs_vals = field[obs_mask]                 # (K, C)

    # Build grid coordinates
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    all_pts = np.stack([rows.ravel(), cols.ravel()], axis=-1)  # (H*W, 2)

    tree = cKDTree(obs_yx)
    _, idx = tree.query(all_pts)               # (H*W,)
    filled = obs_vals[idx].reshape(H, W, C)
    return filled.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HelmholtzVoronoiData(Dataset):
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

            if "data" in f:
                self.data = f["data"][...].astype(np.float32)
                self.N, self.M, self.H, self.W, self.C = self.data.shape
            else:
                re = f["fields_real"][...].astype(np.float32)
                im = f["fields_imag"][...].astype(np.float32)
                self.data = np.stack([re, im], axis=-1)
                self.N, self.M, self.H, self.W = re.shape
                self.C = 2

            self.has_mask = "mask_tr" in f
            self.mask_data = f["mask_tr"][...].astype(np.float32) if self.has_mask else None

            self.gx = f["grid_x"][...].astype(np.float32) if "grid_x" in f \
                else np.linspace(0.0, 1.0, self.H, dtype=np.float32)
            self.gy = f["grid_y"][...].astype(np.float32) if "grid_y" in f \
                else np.linspace(0.0, 1.0, self.W, dtype=np.float32)

        self.sample_indices = list(range(self.N)) if sample_indices is None \
            else [int(i) for i in sample_indices]
        self.freq_indices = _parse_indices(freq_indices, self.M)
        self.pairs: List[Tuple[int, int]] = [
            (b, m) for b in self.sample_indices for m in self.freq_indices
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def _get_mask(self, sample: int, freq_idx: int) -> np.ndarray:
        if not self.has_mask:
            return np.ones((self.H, self.W, 1), dtype=np.float32)
        m = self.mask_data
        if m.ndim == 4:
            m = m[freq_idx]
        elif m.ndim == 5:
            m = m[sample, freq_idx]
        if m.shape[-1] == 2:
            m = m[..., :1]
        return m

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample, freq_idx = self.pairs[idx]
        field = self.data[sample, freq_idx]                        # (H, W, C)
        mask = self._get_mask(sample, freq_idx)                    # (H, W, 1)
        omega = float(self.omega[freq_idx])
        omega_norm = _normalize_omega(omega, self.omega_min, self.omega_max)

        voronoi = voronoi_fill(field, mask)                        # (H, W, C)

        # Input: [voronoi channels, mask, omega_map, x, y]
        x = np.stack(
            [voronoi[..., c] for c in range(self.C)]
            + [
                mask[..., 0],
                np.full((self.H, self.W), omega_norm, dtype=np.float32),
                np.broadcast_to(self.gx[:, None], (self.H, self.W)).copy(),
                np.broadcast_to(self.gy[None, :], (self.H, self.W)).copy(),
            ],
            axis=0,
        ).astype(np.float32)                                       # (C+4, H, W)

        y = field.transpose(2, 0, 1).astype(np.float32)           # (C, H, W)

        return {
            "x": x, "y": y,
            "mask": mask.transpose(2, 0, 1).astype(np.float32),
            "omega": np.array([omega_norm], dtype=np.float32),
            "sample_idx": np.array(sample, dtype=np.int64),
            "freq_idx": np.array(freq_idx, dtype=np.int64),
            "omega_raw": np.array(omega, dtype=np.float32),
        }


# ---------------------------------------------------------------------------
# Voronoi-CNN model  (encoder–decoder with skip connections)
# ---------------------------------------------------------------------------

def _gn(ch: int, max_g: int = 8) -> int:
    g = min(max_g, ch)
    while g > 1 and ch % g != 0:
        g -= 1
    return g


class ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_gn(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_gn(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class VoronoiCNN(nn.Module):
    """Lightweight U-Net CNN for Voronoi-CNN baseline."""

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 2,
        base_channels: int = 64,
    ):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.stem  = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.enc1  = ResBlock2D(c1, c1)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc2  = ResBlock2D(c2, c2)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.mid   = nn.Sequential(ResBlock2D(c3, c3), ResBlock2D(c3, c3))
        self.up2   = ResBlock2D(c3 + c2, c2)
        self.up1   = ResBlock2D(c2 + c1, c1)
        self.head  = nn.Sequential(
            nn.GroupNorm(_gn(c1), c1),
            nn.SiLU(),
            nn.Conv2d(c1, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.stem(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(self.down1(e1))
        m  = self.mid(self.down2(e2))
        d2 = self.up2(torch.cat([F.interpolate(m, e2.shape[-2:], mode="nearest"), e2], 1))
        d1 = self.up1(torch.cat([F.interpolate(d2, e1.shape[-2:], mode="nearest"), e1], 1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# Data splits / loaders
# ---------------------------------------------------------------------------

def _split(n: int, ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    ids = rng.permutation(n)
    k = max(1, int(round(n * ratio)))
    k = min(k, n - 1) if n > 1 else n
    tr, va = ids[:k].tolist(), ids[k:].tolist()
    return tr, (va if va else tr[-1:])


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    keys = ["x", "y", "mask", "omega", "sample_idx", "freq_idx", "omega_raw"]
    out  = {k: torch.from_numpy(np.stack([b[k] for b in batch])) for k in keys}
    out["sample_idx"] = out["sample_idx"].long()
    out["freq_idx"]   = out["freq_idx"].long()
    return out


def _load_meta(path: Path) -> Dict[str, Any]:
    with h5py.File(path, "r") as f:
        meta = {}
        if "metadata" in f:
            raw = f["metadata"][()]
            try:
                meta = json.loads((raw if isinstance(raw, str) else raw.decode("utf-8")))
            except Exception:
                pass
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

    meta  = _load_meta(Path(args.train_h5))
    N     = int(meta["shape"][0])
    tr_ids, va_ids = _split(N, args.train_ratio, args.seed)

    tr_ds = HelmholtzVoronoiData(args.train_h5, tr_ids, args.train_freq_indices)
    va_ds = HelmholtzVoronoiData(args.train_h5, va_ids, args.eval_freq_indices)
    in_ch = tr_ds.C + 4  # voronoi channels + mask + omega + x + y
    out_ch = tr_ds.C

    model = VoronoiCNN(in_ch, out_ch, args.base_channels).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

    tr_dl = DataLoader(tr_ds, args.batch_size, shuffle=True,  num_workers=4,  collate_fn=_collate)
    va_dl = DataLoader(va_ds, args.batch_size, shuffle=False, num_workers=4,  collate_fn=_collate)

    best_val = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nVoronoi-CNN training | in={in_ch} out={out_ch} base={args.base_channels} device={device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl = 0.0; tn = 0
        for b in tr_dl:
            x, y = b["x"].to(device), b["y"].to(device)
            mask = b["mask"].to(device)
            pred = model(x)
            loss = F.mse_loss(pred * mask, y * mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tl += loss.item() * x.shape[0]; tn += x.shape[0]

        model.eval()
        vl = 0.0; vn = 0
        with torch.no_grad():
            for b in va_dl:
                x, y = b["x"].to(device), b["y"].to(device)
                vl += F.mse_loss(model(x), y).item() * x.shape[0]; vn += x.shape[0]

        tl /= max(tn, 1); vl /= max(vn, 1)
        sched.step(vl)
        if vl < best_val:
            best_val = vl
            torch.save({
                "model_state": model.state_dict(),
                "model_config": {"in_channels": in_ch, "out_channels": out_ch,
                                 "base_channels": args.base_channels},
                "train_config": vars(args),
                "best_val_loss": best_val,
                "train_h5": args.train_h5,
            }, out_path)

        if args.log_every > 0 and (epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs):
            print(f"[{epoch:04d}/{args.epochs}] train={tl:.4e} val={vl:.4e} best={best_val:.4e}")

    print(f"Done. best_val={best_val:.4e}  ckpt={out_path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _plot_eval(out_path, gt, pred, mask, sample_idx, freq_idx, omega, rmse, obs_r, unobs_r, pde_res=0.0, dpi=180):
    gt_re, gt_im = gt[..., 0], gt[..., 1]
    pr_re, pr_im = pred[..., 0], pred[..., 1]
    gt_amp = np.sqrt(gt_re**2 + gt_im**2)
    pr_amp = np.sqrt(pr_re**2 + pr_im**2)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    items = [
        (gt_re, "GT Re", "viridis"), (pr_re, "Pred Re", "viridis"),
        (np.abs(pr_re - gt_re), "Err Re", "magma"),
        (gt_im, "GT Im", "viridis"), (pr_im, "Pred Im", "viridis"),
        (np.abs(pr_im - gt_im), "Err Im", "magma"),
        (gt_amp, "GT Amp", "viridis"), (pr_amp, "Pred Amp", "viridis"),
        (np.abs(pr_amp - gt_amp), "Err Amp", "magma"),
    ]
    for ax, (img, title, cmap) in zip(axes.flat, items):
        im = ax.imshow(img, origin="lower", cmap=cmap)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"s={sample_idx} f={freq_idx} ω={omega:.3f} | rmse={rmse:.2e} "
                 f"obs={obs_r:.2e} unobs={unobs_r:.2e} pde={pde_res:.2e}", y=0.995)
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig)


def evaluate_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    ckpt = _safe_load(Path(args.ckpt))
    model = VoronoiCNN(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data_path = Path(args.test_h5)
    meta = _load_meta(data_path)
    N = int(meta["shape"][0])
    max_s = N if args.max_samples <= 0 else min(N, args.max_samples)
    ds = HelmholtzVoronoiData(data_path, list(range(max_s)), args.eval_freq_indices)
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate)

    h5_file = h5py.File(data_path, "r")
    out_dir  = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    vis_cnt = 0
    with torch.no_grad():
        for batch in dl:
            x       = batch["x"].to(device)
            y       = batch["y"].to(device)
            mask    = batch["mask"].cpu().numpy()
            s_idx   = batch["sample_idx"].cpu().numpy()
            f_idx   = batch["freq_idx"].cpu().numpy()
            omega_r = batch["omega_raw"].cpu().numpy()
            pred    = model(x).cpu().numpy()
            gt      = y.cpu().numpy()

            for i in range(pred.shape[0]):
                pred_i  = pred[i].transpose(1, 2, 0)
                gt_i    = gt[i].transpose(1, 2, 0)
                mask_i  = mask[i].transpose(1, 2, 0)
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
    if not rows:
        raise RuntimeError("No evaluation rows")

    csv_path = out_dir / "metrics_cases.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, ["sample_idx","freq_idx","omega","rmse","obs_rmse","unobs_rmse","pde_res","vis_path"])
        w.writeheader()
        for r in rows: w.writerow(r)

    rmses     = np.array([r["rmse"]      for r in rows])
    obs_rmses = np.array([r["obs_rmse"]  for r in rows])
    unobs_rmses = np.array([r["unobs_rmse"] for r in rows])
    pde_ress  = np.array([r["pde_res"]   for r in rows if not np.isnan(r["pde_res"])])

    summary = {
        "ckpt": args.ckpt, "test_h5": str(data_path),
        "num_cases": len(rows),
        "mean_rmse": float(np.mean(rmses)),
        "mean_obs_rmse": float(np.mean(obs_rmses)),
        "mean_unobs_rmse": float(np.mean(unobs_rmses)),
        "mean_pde_res": float(np.mean(pde_ress)) if len(pde_ress) else float("nan"),
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nEvaluation finished.")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Voronoi-CNN baseline — 2D Helmholtz")
    p.add_argument("--mode", choices=["train","eval"], default="eval")
    p.add_argument("--train_h5", default="helmholtz_dataset_42.h5")
    p.add_argument("--test_h5",  default="data_for_test/helmholtz_dataset_42_for_test_mask1.h5")
    p.add_argument("--ckpt",     default="ckp/voronoi_cnn.pt")
    p.add_argument("--out",      default="ckp/voronoi_cnn.pt")
    p.add_argument("--out_dir",  default="visual_data/voronoi_cnn_eval/mask_ratio1")
    p.add_argument("--train_ratio",       type=float, default=0.8)
    p.add_argument("--train_freq_indices",type=str,   default="")
    p.add_argument("--eval_freq_indices", type=str,   default="")
    p.add_argument("--max_samples",       type=int,   default=-1)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--wd",         type=float, default=1e-6)
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--log_every",  type=int,   default=10)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--num_visualize", type=int, default=50)
    p.add_argument("--vis_dpi",   type=int,   default=150)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--device",    type=str,   default="auto")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "train":
        train_model(args)
    else:
        evaluate_model(args)


if __name__ == "__main__":
    main()
