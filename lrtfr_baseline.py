"""
lrtfr_baseline.py
-----------------
Low-rank tensor regression baseline for Helmholtz field reconstruction.

This baseline assumes a pretrained universal spatial basis (FTM checkpoint)
is available. For each sparse observation field, it solves a least-squares
problem for the real and imaginary core coefficients separately, then
decodes the recovered core back to the full physical field.

Input protocol
--------------
The test HDF5 file can contain either:
- data      : (B, M, H, W, 2) complex fields already stacked in channels
- fields_real / fields_imag : (B, M, H, W)

The observation mask can be either:
- mask_tr   : (M, H, W, 2)
- mask_tr   : (B, M, H, W, 2)

The script uses the mask as sparse observations and solves

    min_c || A c - y_obs ||_2

where A is the basis matrix restricted to observed grid points.

Example
-------
    python lrtfr_baseline.py \
        --ftm_ckpt ckp/ftm_gpu_checkpoint.pt \
        --test_h5 data_for_test/helmholtz_dataset_42_for_test_mask2.h5 \
        --out_dir visual_data/lrtfr_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib
import numpy as np
from physics_metric import evaluate_physics_residual
import json
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train_FTM_GPU import MLP1D, build_phi, normalize_coords_to_unit


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


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


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _parse_indices(text: str, limit: int) -> List[int]:
    if text.strip() == "":
        return list(range(limit))

    out: List[int] = []
    for part in text.split(","):
        item = part.strip()
        if item == "":
            continue
        idx = int(item)
        if idx < 0 or idx >= limit:
            raise ValueError(f"index out of range: {idx}, valid [0, {limit - 1}]")
        out.append(idx)

    if not out:
        raise ValueError("Parsed empty index list")
    return sorted(set(out))


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _relative_rmse(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    num = float(np.sum((pred - gt) ** 2))
    den = float(np.sum(gt**2))
    return float(np.sqrt(num / max(den, eps)))


def _masked_relative_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> float:
    if pred.ndim == 3 and pred.shape[-1] == 2:
        num = 0.0
        den = 0.0
        if mask.ndim == 2:
            mask_channels = [mask, mask]
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            base = mask[..., 0]
            mask_channels = [base, base]
        elif mask.ndim == 3 and mask.shape[-1] == 2:
            mask_channels = [mask[..., 0], mask[..., 1]]
        else:
            raise ValueError(f"Unsupported mask shape for complex field: {mask.shape}")

        for ch in range(2):
            m = mask_channels[ch].astype(bool)
            if not np.any(m):
                continue
            diff = (pred[..., ch] - gt[..., ch]) ** 2
            gt_sq = gt[..., ch] ** 2
            num += float(np.sum(diff[m]))
            den += float(np.sum(gt_sq[m]))

        return float(np.sqrt(num / max(den, eps)))

    m = mask.astype(bool)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    if not np.any(m):
        return float("nan")
    num = float(np.sum(((pred - gt) ** 2)[m]))
    den = float(np.sum((gt**2)[m]))
    return float(np.sqrt(num / max(den, eps)))


def _sym_limit(arr: np.ndarray, q: float = 99.5) -> float:
    lim = float(np.percentile(np.abs(arr), q))
    return max(lim, 1e-8)


# -----------------------------------------------------------------------------
# Basis loading
# -----------------------------------------------------------------------------


@dataclass
class FTMBasis:
    net_x: torch.nn.Module
    net_y: torch.nn.Module
    rank_x: int
    rank_y: int
    normalize_coords: bool
    omega: Optional[np.ndarray] = None
    grid_x: Optional[np.ndarray] = None
    grid_y: Optional[np.ndarray] = None


def load_ftm_basis(ftm_ckpt_path: Path, device: torch.device) -> FTMBasis:
    if not ftm_ckpt_path.exists():
        raise FileNotFoundError(f"FTM checkpoint not found: {ftm_ckpt_path}")

    ckpt = _safe_torch_load(ftm_ckpt_path)
    cfg = ckpt.get("config", {})

    required = ["rank_x", "rank_y", "hidden_dim", "hidden_layers", "activation"]
    for key in required:
        if key not in cfg:
            raise KeyError(f"FTM checkpoint config missing key: {key}")

    net_x = MLP1D(
        out_dim=int(cfg["rank_x"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_hidden_layers=int(cfg["hidden_layers"]),
        activation=str(cfg["activation"]),
    ).to(device)
    net_y = MLP1D(
        out_dim=int(cfg["rank_y"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_hidden_layers=int(cfg["hidden_layers"]),
        activation=str(cfg["activation"]),
    ).to(device)

    net_x.load_state_dict(ckpt["net_x_state"])
    net_y.load_state_dict(ckpt["net_y_state"])
    net_x.eval()
    net_y.eval()

    omega = _to_numpy(ckpt["omega"]).astype(np.float32) if "omega" in ckpt else None
    grid_x = _to_numpy(ckpt["grid_x"]).astype(np.float32) if "grid_x" in ckpt else None
    grid_y = _to_numpy(ckpt["grid_y"]).astype(np.float32) if "grid_y" in ckpt else None

    return FTMBasis(
        net_x=net_x,
        net_y=net_y,
        rank_x=int(cfg["rank_x"]),
        rank_y=int(cfg["rank_y"]),
        normalize_coords=bool(cfg.get("normalize_coords", True)),
        omega=omega,
        grid_x=grid_x,
        grid_y=grid_y,
    )


def _build_phi_np(basis: FTMBasis, grid_x: np.ndarray, grid_y: np.ndarray, device: torch.device) -> np.ndarray:
    if basis.normalize_coords:
        x_coords = normalize_coords_to_unit(grid_x.astype(np.float64)).astype(np.float32)
        y_coords = normalize_coords_to_unit(grid_y.astype(np.float64)).astype(np.float32)
    else:
        x_coords = grid_x.astype(np.float32)
        y_coords = grid_y.astype(np.float32)

    x_t = torch.from_numpy(x_coords).unsqueeze(-1).to(device)
    y_t = torch.from_numpy(y_coords).unsqueeze(-1).to(device)

    with torch.no_grad():
        phi = build_phi(basis.net_x, basis.net_y, x_t, y_t)
    return phi.detach().cpu().numpy().astype(np.float32)


# -----------------------------------------------------------------------------
# Dataset loader
# -----------------------------------------------------------------------------


@dataclass
class SparseHelmholtzData:
    data: np.ndarray  # (B, M, H, W, 2)
    mask: np.ndarray  # (M, H, W, 2) or (B, M, H, W, 2)
    omega: np.ndarray  # (M,)
    grid_x: np.ndarray  # (H,)
    grid_y: np.ndarray  # (W,)


def _load_sparse_dataset(h5_path: Path, max_samples: int = 0) -> SparseHelmholtzData:
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as f:
        omega = f["omega"][...].astype(np.float32)

        if "grid_x" in f:
            grid_x = f["grid_x"][...].astype(np.float32)
        else:
            if "data" in f:
                grid_x = np.linspace(0.0, 1.0, int(f["data"].shape[2]), dtype=np.float32)
            else:
                grid_x = np.linspace(0.0, 1.0, int(f["fields_real"].shape[2]), dtype=np.float32)

        if "grid_y" in f:
            grid_y = f["grid_y"][...].astype(np.float32)
        else:
            if "data" in f:
                grid_y = np.linspace(0.0, 1.0, int(f["data"].shape[3]), dtype=np.float32)
            else:
                grid_y = np.linspace(0.0, 1.0, int(f["fields_real"].shape[3]), dtype=np.float32)

        if "data" in f:
            data = f["data"][...].astype(np.float32)
        elif "fields_real" in f and "fields_imag" in f:
            real = f["fields_real"][...].astype(np.float32)
            imag = f["fields_imag"][...].astype(np.float32)
            data = np.stack([real, imag], axis=-1).astype(np.float32)
        else:
            raise KeyError("HDF5 must contain either 'data' or 'fields_real'/'fields_imag'.")

        if "mask_tr" not in f:
            raise KeyError("HDF5 must contain 'mask_tr'.")
        mask = f["mask_tr"][...].astype(np.float32)

    if data.ndim != 5 or data.shape[-1] != 2:
        raise ValueError(f"Expected data shape (B,M,H,W,2), got {data.shape}")

    if max_samples > 0:
        data = data[:max_samples]
        if mask.ndim == 5:
            mask = mask[:max_samples]

    if mask.ndim not in (4, 5):
        raise ValueError(f"Expected mask shape (M,H,W,2) or (B,M,H,W,2), got {mask.shape}")

    if not np.all(np.diff(omega) > 0):
        raise ValueError("omega must be strictly increasing.")

    return SparseHelmholtzData(data=data, mask=mask, omega=omega, grid_x=grid_x, grid_y=grid_y)


def _build_channel_views(data: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    B, M, H, W, C = data.shape
    if C != 2:
        raise ValueError("Only 2-channel complex data is supported.")

    P = H * W
    data_re = data[..., 0].reshape(B, M, P)
    data_im = data[..., 1].reshape(B, M, P)

    if mask.ndim == 4:
        mask_re = (mask[..., 0].reshape(M, P) > 0.5).astype(np.float32)
        if mask.shape[-1] == 1:
            mask_im = mask_re.copy()
        else:
            mask_im = (mask[..., 1].reshape(M, P) > 0.5).astype(np.float32)
    else:
        mask_re = (mask[..., 0].reshape(B, M, P) > 0.5).astype(np.float32)
        if mask.shape[-1] == 1:
            mask_im = mask_re.copy()
        else:
            mask_im = (mask[..., 1].reshape(B, M, P) > 0.5).astype(np.float32)

    return data_re, data_im, mask_re, mask_im


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------


def _plot_case(
    out_path: Path,
    gt: np.ndarray,
    obs: np.ndarray,
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
    obs_re, obs_im = obs[..., 0], obs[..., 1]
    pr_re, pr_im = pred[..., 0], pred[..., 1]
    gt_amp = np.sqrt(gt_re**2 + gt_im**2)
    pr_amp = np.sqrt(pr_re**2 + pr_im**2)
    err = np.abs(pr_amp - gt_amp)
    err_re = np.abs(pr_re - gt_re)
    err_im = np.abs(pr_im - gt_im)
    mask_img = mask[..., 0].astype(np.float32)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    items = [
        (gt_re, "GT Real", "viridis"),
        (pr_re, "Pred Real", "viridis"),
        (err_re, "Abs Err Real", "magma"),
        (gt_im, "GT Imag", "viridis"),
        (pr_im, "Pred Imag", "viridis"),
        (err_im, "Abs Err Imag", "magma"),
        (gt_amp, "GT Amplitude", "viridis"),
        (pr_amp, "Pred Amplitude", "viridis"),
        (err, "Abs Err Amplitude", "magma"),

    ]

    for ax, (img, title, cmap) in zip(axes.flat, items):
        # if title.startswith("GT") or title.startswith("Pred"):
        #     vmin, vmax = _sym_limit(img), _sym_limit(img)
        #     im = ax.imshow(img, origin="lower", cmap=cmap, vmin=-vmin, vmax=vmax)
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


# -----------------------------------------------------------------------------
# LRTFR reconstruction
# -----------------------------------------------------------------------------


def _solve_core_from_sparse_obs(
    phi_np: np.ndarray,
    field: np.ndarray,
    mask_re: np.ndarray,
    mask_im: np.ndarray,
    rcond: Optional[float] = None,
) -> np.ndarray:
    if mask_re.ndim != 1 or mask_im.ndim != 1:
        raise ValueError("mask vectors must be flattened 1D arrays")

    obs_re = mask_re > 0.5
    obs_im = mask_im > 0.5
    if not np.any(obs_re):
        raise ValueError("No observed points in real channel")
    if not np.any(obs_im):
        raise ValueError("No observed points in imag channel")

    gt_re = field[..., 0].reshape(-1)
    gt_im = field[..., 1].reshape(-1)

    a_re = phi_np[obs_re]
    a_im = phi_np[obs_im]
    b_re = gt_re[obs_re]
    b_im = gt_im[obs_im]

    core_re, *_ = np.linalg.lstsq(a_re, b_re, rcond=rcond)
    core_im, *_ = np.linalg.lstsq(a_im, b_im, rcond=rcond)
    return np.stack([core_re, core_im], axis=0).astype(np.float32)


def _decode_field_from_core(core_img: np.ndarray, phi_np: np.ndarray, h: int, w: int) -> np.ndarray:
    core_re = core_img[0].reshape(-1)
    core_im = core_img[1].reshape(-1)
    pred_re = phi_np @ core_re
    pred_im = phi_np @ core_im
    return np.stack([pred_re, pred_im], axis=-1).reshape(h, w, 2).astype(np.float32)


def run_eval(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = _select_device(args.device)

    basis = load_ftm_basis(Path(args.ftm_ckpt), device=device)
    ds = _load_sparse_dataset(Path(args.test_h5), max_samples=args.max_samples)

    data = ds.data
    mask = ds.mask
    omega = ds.omega
    grid_x = ds.grid_x
    grid_y = ds.grid_y

    B, M, H, W, _ = data.shape
    freq_indices = _parse_indices(args.eval_freq_indices, M)

    phi_np = _build_phi_np(basis, grid_x, grid_y, device=device)
    expected_rank = basis.rank_x * basis.rank_y
    if phi_np.shape[1] != expected_rank:
        raise ValueError(
            f"Basis rank mismatch: phi has {phi_np.shape[1]} columns, expected {expected_rank}"
        )

    data_re, data_im, mask_re, mask_im = _build_channel_views(data, mask)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    vis_count = 0

    import json
    with h5py.File(str(args.test_h5), "r") as h5_f:
        meta_str = h5_f["metadata"][()].decode("utf-8") if "metadata" in h5_f else "{}"
        try:
            h5_meta = json.loads(meta_str)
        except:
            h5_meta = {}
            
        for b in range(B):
            for m in freq_indices:
                gt = data[b, m]

                if mask.ndim == 4:
                    mask_re_b = mask_re[m]
                    mask_im_b = mask_im[m]
                else:
                    mask_re_b = mask_re[b, m]
                    mask_im_b = mask_im[b, m]

                core = _solve_core_from_sparse_obs(
                    phi_np=phi_np,
                    field=gt,
                    mask_re=mask_re_b,
                    mask_im=mask_im_b,
                    rcond=args.rcond,
                )
                pred = _decode_field_from_core(core, phi_np, H, W)

                obs = np.zeros_like(gt, dtype=np.float32)
                obs[..., 0] = gt[..., 0] * (mask_re_b.reshape(H, W) > 0.5)
                obs[..., 1] = gt[..., 1] * (mask_im_b.reshape(H, W) > 0.5)

                rmse = _relative_rmse(pred, gt, eps=args.eps)
                obs_rmse = _masked_relative_rmse(pred, gt, np.stack([mask_re_b.reshape(H, W), mask_im_b.reshape(H, W)], axis=-1), eps=args.eps)
                unobs_mask = np.stack([
                    1.0 - mask_re_b.reshape(H, W),
                    1.0 - mask_im_b.reshape(H, W),
                ], axis=-1)
                unobs_rmse = _masked_relative_rmse(pred, gt, unobs_mask, eps=args.eps)
                pde_res = evaluate_physics_residual(pred, h5_f, int(b), float(omega[m]), h5_meta)

                rows.append(
                    {
                        "sample_idx": int(b),
                        "freq_idx": int(m),
                        "omega": float(omega[m]),
                        "rmse": float(rmse),
                        "obs_rmse": float(obs_rmse),
                        "unobs_rmse": float(unobs_rmse),
                        "pde_res": float(pde_res),
                    }
                )

                if vis_count < args.num_visualize:
                    vis_count += 1
                    vis_path = out_dir / f"case{vis_count:03d}_sample{b:03d}_freq{m:03d}.png"
                    _plot_case(
                        out_path=vis_path,
                        gt=gt,
                        obs=obs,
                        pred=pred,
                        mask=np.stack([mask_re_b.reshape(H, W), mask_im_b.reshape(H, W)], axis=-1),
                        sample_idx=b,
                        freq_idx=m,
                        omega_val=float(omega[m]),
                        rmse=rmse,
                        obs_rmse=obs_rmse,
                        unobs_rmse=unobs_rmse,
                        dpi=args.vis_dpi,
                        pde_res=pde_res,
                    )
                    rows[-1]["vis_path"] = str(vis_path)

            # Note: h5_f closed here
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
    pde_ress = np.array([r["pde_res"] for r in rows if not np.isnan(r.get("pde_res", np.nan))], dtype=np.float64)

    summary = {
        "ftm_ckpt": str(args.ftm_ckpt),
        "test_h5": str(args.test_h5),
        "num_cases": int(len(rows)),
        "mean_rmse": float(np.mean(rmses)),
        "mean_obs_rmse": float(np.mean(obs_rmses)),
        "mean_unobs_rmse": float(np.mean(unobs_rmses)),
        "mean_pde_res": float(np.mean(pde_ress)) if len(pde_ress) > 0 else 0.0,
        "mean_pde_res": float(np.mean(pde_ress)) if len(pde_ress) > 0 else 0.0,
        "num_visualized": int(vis_count),
        "output_dir": str(out_dir),
        "rank_x": int(basis.rank_x),
        "rank_y": int(basis.rank_y),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nEvaluation finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved metrics: {csv_path}")
    print(f"Saved summary: {summary_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LRTFR baseline for Helmholtz reconstruction")
    p.add_argument("--mode", type=str, default="eval", choices=["eval"])
    p.add_argument("--ftm_ckpt", type=str, default="ckp/ftm_gpu_checkpoint.pt")
    p.add_argument("--test_h5", type=str, default="data_for_test/helmholtz_dataset_42_for_test_mask1.h5")
    p.add_argument("--out_dir", type=str, default="visual_data/lrtfr_baseline_eval/mask_ratio1")
    p.add_argument("--eval_freq_indices", type=str, default="")
    p.add_argument("--max_samples", type=int, default=1)
    p.add_argument("--num_visualize", type=int, default=50)
    p.add_argument("--vis_dpi", type=int, default=180)
    p.add_argument("--rcond", type=float, default=None)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode != "eval":
        raise ValueError("Only eval mode is implemented for the LRTFR baseline.")
    run_eval(args)


if __name__ == "__main__":
    main()