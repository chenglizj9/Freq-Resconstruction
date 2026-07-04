"""
fno_baseline.py  (elastic wave edition)
-----------------------------------------
Fourier Neural Operator baseline for 4-channel elastic wave field reconstruction.

Input channels (8):
- obs_ux_re, obs_ux_im   : observed x-displacement
- obs_uy_re, obs_uy_im   : observed y-displacement
- mask                   : observation mask
- omega_norm             : normalised frequency map
- x_coord, y_coord       : spatial coordinates

Output channels (4): ux_re, ux_im, uy_re, uy_im

Examples
--------
Train:
    python fno_baseline.py --mode train \
        --train_h5 elastic_dataset.h5 \
        --out ckp/fno_elastic.pt

Evaluate:
    python fno_baseline.py --mode eval \
        --ckpt ckp/fno_elastic.pt \
        --test_h5 elastic_dataset.h5
"""

from __future__ import annotations

import argparse
import csv
import json
import random
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
# Utilities (same helpers as confild_baseline)
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        if not (0 <= idx < limit):
            raise ValueError(f"index out of range: {idx}")
        out.append(idx)
    return sorted(set(out))


def _relative_rmse(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.sum((pred - gt) ** 2) / max(np.sum(gt ** 2), eps)))


def _masked_relative_rmse(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps: float = 1e-12
) -> float:
    m = (mask > 0.5)
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


CHANNEL_NAMES = ["ux_re", "ux_im", "uy_re", "uy_im"]


# ---------------------------------------------------------------------------
# Dataset (identical to confild_baseline.ElasticDataset)
# ---------------------------------------------------------------------------

class ElasticDataset(Dataset):
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
            self.data = f["data"][...].astype(np.float32)       # (N,M,H,W,4)
            self.N, self.M, self.H, self.W, self.C = self.data.shape
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
        if not self.has_mask:
            return np.ones((self.H, self.W, 1), dtype=np.float32)
        mask = self.mask_data
        if mask.ndim == 4:
            return mask[freq_idx, ..., :1].astype(np.float32)
        else:  # ndim == 5
            return mask[sample, freq_idx, ..., :1].astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample, freq_idx = self.pairs[idx]
        field = self.data[sample, freq_idx]
        mask = self._get_mask(sample, freq_idx)
        omega = float(self.omega[freq_idx])
        omega_norm = float(_normalize_omega(omega, self.omega_min, self.omega_max))
        obs = field * mask
        x = np.stack([
            obs[..., 0], obs[..., 1], obs[..., 2], obs[..., 3],
            mask[..., 0],
            np.full((self.H, self.W), omega_norm, dtype=np.float32),
            np.broadcast_to(self.gx[:, None], (self.H, self.W)).copy(),
            np.broadcast_to(self.gy[None, :], (self.H, self.W)).copy(),
        ], axis=0).astype(np.float32)
        y = field.transpose(2, 0, 1).astype(np.float32)
        return {
            "x": x, "y": y,
            "mask": mask.transpose(2, 0, 1).astype(np.float32),
            "omega": np.array([omega_norm], dtype=np.float32),
            "sample_idx": np.array(sample, dtype=np.int64),
            "freq_idx": np.array(freq_idx, dtype=np.int64),
            "omega_raw": np.array(omega, dtype=np.float32),
        }


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    keys = ["x", "y", "mask", "omega", "sample_idx", "freq_idx", "omega_raw"]
    out = {k: torch.from_numpy(np.stack([b[k] for b in batch])) for k in keys}
    out["sample_idx"] = out["sample_idx"].long()
    out["freq_idx"] = out["freq_idx"].long()
    return out


# ---------------------------------------------------------------------------
# FNO model
# ---------------------------------------------------------------------------

def _complex_mul2d(input_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    wr, wi = weights[..., 0], weights[..., 1]
    ir, ii = input_ft.real, input_ft.imag
    out_r = torch.einsum("bixy,ioxy->boxy", ir, wr) - torch.einsum("bixy,ioxy->boxy", ii, wi)
    out_i = torch.einsum("bixy,ioxy->boxy", ir, wi) + torch.einsum("bixy,ioxy->boxy", ii, wr)
    return torch.complex(out_r, out_i)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(b, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)
        out_ft[:, :, :m1, :m2] = _complex_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = _complex_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(h, w))


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(min(8, width), width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.w(x)))


class FNO2d(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 4,
        width: int = 64,
        modes1: int = 16,
        modes2: int = 16,
        n_layers: int = 4,
        padding: int = 4,
    ):
        super().__init__()
        self.padding = padding
        self.fc0 = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes1, modes2) for _ in range(n_layers)])
        self.fc1 = nn.Conv2d(width, width * 2, 1)
        self.fc2 = nn.Conv2d(width * 2, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = F.pad(x, (0, self.padding, 0, self.padding))
        x = self.fc0(x)
        for block in self.blocks:
            x = block(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
        return x


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
# Visualization
# ---------------------------------------------------------------------------

def _plot_elastic_case(
    out_path: Path,
    gt: np.ndarray,
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
        f"FNO | sample={sample_idx} freq={freq_idx} ω={omega_val:.4f} | "
        f"rmse={rmse:.3e} obs={obs_rmse:.3e} unobs={unobs_rmse:.3e}",
        y=0.999, fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


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

    model = FNO2d(
        in_channels=8,
        out_channels=4,
        width=args.width,
        modes1=args.modes,
        modes2=args.modes,
        n_layers=args.n_layers,
        padding=args.padding,
    ).to(device)

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
    print("FNO elastic baseline training")
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
            pred = model(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            tr_loss += float(loss.item()) * x.shape[0]
            tr_n += x.shape[0]

        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                loss = F.mse_loss(model(x), y)
                val_loss += float(loss.item()) * x.shape[0]
                val_n += x.shape[0]

        tl = tr_loss / max(tr_n, 1)
        vl = val_loss / max(val_n, 1)
        scheduler.step(vl)

        if vl < best_val:
            best_val = vl
            torch.save({
                "model_state": model.state_dict(),
                "model_config": {
                    "in_channels": 8,
                    "out_channels": 4,
                    "width": args.width,
                    "modes1": args.modes,
                    "modes2": args.modes,
                    "n_layers": args.n_layers,
                    "padding": args.padding,
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
    model = FNO2d(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data_path = Path(args.test_h5)
    meta = _load_h5_meta(data_path)
    num_samples = int(meta["shape"][0])
    sample_indices = list(range(
        min(num_samples, args.max_samples) if args.max_samples > 0 else num_samples
    ))

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
            pred = model(x)

            pred_np = pred.cpu().numpy()
            gt_np = y.cpu().numpy()
            mask_np = batch["mask"].cpu().numpy()
            sample_idx_np = batch["sample_idx"].cpu().numpy()
            freq_idx_np = batch["freq_idx"].cpu().numpy()
            omega_raw_np = batch["omega_raw"].cpu().numpy()

            for i in range(pred_np.shape[0]):
                pred_i = pred_np[i].transpose(1, 2, 0)
                gt_i = gt_np[i].transpose(1, 2, 0)
                mask_i = mask_np[i].transpose(1, 2, 0)

                rmse = _relative_rmse(pred_i, gt_i, args.eps)
                obs_rmse = _masked_relative_rmse(pred_i, gt_i, mask_i, args.eps)
                unobs_rmse = _masked_relative_rmse(pred_i, gt_i, 1.0 - mask_i, args.eps)
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
                    vis_path = out_dir / f"case{vis_count:03d}_s{int(sample_idx_np[i]):03d}_f{int(freq_idx_np[i]):03d}.png"
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
    summary = {
        "ckpt": str(args.ckpt),
        "test_h5": str(data_path),
        "num_cases": len(rows),
        "mean_rmse": float(np.mean(rmses)),
        "mean_obs_rmse": float(np.nanmean([r["obs_rmse"] for r in rows])),
        "mean_unobs_rmse": float(np.nanmean([r["unobs_rmse"] for r in rows])),
        "per_channel": {n: float(np.mean([r[f"rmse_{n}"] for r in rows])) for n in CHANNEL_NAMES},
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
    p = argparse.ArgumentParser(description="FNO baseline for elastic wave reconstruction")
    p.add_argument("--mode", default="eval", choices=["train", "eval"])

    p.add_argument("--train_h5", default="elastic_dataset.h5")
    p.add_argument("--test_h5",  default="elastic_dataset_msk0.01.h5")
    p.add_argument("--ckpt",     default="ckp/fno_elastic.pt")
    p.add_argument("--out",      default="ckp/fno_elastic.pt")
    p.add_argument("--out_dir",  default="visual_data/fno_elastic_eval_msk0.01")

    p.add_argument("--train_ratio",        type=float, default=0.8)
    p.add_argument("--train_freq_indices", type=str,   default="")
    p.add_argument("--eval_freq_indices",  type=str,   default="")
    p.add_argument("--max_samples",        type=int,   default=10)

    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--log_every",    type=int,   default=5)

    p.add_argument("--width",    type=int, default=64)
    p.add_argument("--modes",    type=int, default=16)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--padding",  type=int, default=4)

    p.add_argument("--num_visualize", type=int,   default=20)
    p.add_argument("--vis_dpi",       type=int,   default=150)
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
