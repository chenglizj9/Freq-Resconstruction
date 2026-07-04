"""
lrtfr_baseline.py  (elastic wave edition)
-------------------------------------------
Low-Rank Tensor Factor Regression baseline for 4-channel elastic wave field
reconstruction.

Given a pre-trained FTM basis (net_x, net_y) and sparse observations, solve
per-channel least-squares problems:

    min_c  || phi[obs] @ c - field_obs ||^2   for each channel c in {0,1,2,3}

then reconstruct the full field as:

    pred[..., c] = phi @ c_opt  → (H*W,) → (H, W)

Supports both shared-basis and split-basis FTM checkpoints.

Example
-------
    python lrtfr_baseline.py \
        --ftm_ckpt ckp/ftm_gpu_checkpoint.pt \
        --test_h5  elastic_dataset.h5 \
        --out_dir  visual_data/lrtfr_elastic_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train_FTM_GPU import MLP1D, build_phi, normalize_coords_to_unit


# ---------------------------------------------------------------------------
# Utilities
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


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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


def _select_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


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


CHANNEL_NAMES = ["ux_real", "ux_imag", "uy_real", "uy_imag"]


# ---------------------------------------------------------------------------
# FTM basis loading
# ---------------------------------------------------------------------------

@dataclass
class FTMBasis:
    nets_x: List[nn.Module]      # one per basis group
    nets_y: List[nn.Module]      # one per basis group
    ch_to_group: List[int]       # index into nets_x/nets_y per channel
    rank_x: int
    rank_y: int
    normalize_coords: bool
    channel_names: List[str]
    omega: Optional[np.ndarray] = None
    grid_x: Optional[np.ndarray] = None
    grid_y: Optional[np.ndarray] = None


def load_ftm_basis(path: Path, device: torch.device) -> FTMBasis:
    if not path.exists():
        raise FileNotFoundError(f"FTM checkpoint not found: {path}")

    ckpt = _safe_torch_load(path)
    cfg = ckpt.get("config", {})
    for k in ("rank_x", "rank_y", "hidden_dim", "hidden_layers", "activation"):
        if k not in cfg:
            raise KeyError(f"FTM checkpoint config missing key: '{k}'")

    def _make_net(rank_out: int) -> MLP1D:
        net = MLP1D(
            out_dim=rank_out,
            hidden_dim=int(cfg["hidden_dim"]),
            num_hidden_layers=int(cfg["hidden_layers"]),
            activation=str(cfg["activation"]),
        ).to(device)
        net.eval()
        return net

    channel_names: List[str] = ckpt.get("channel_names", [f"ch{i}" for i in range(4)])
    C = len(channel_names)

    if "net_x_state_list" in ckpt:
        # Split-basis checkpoint
        states_x = ckpt["net_x_state_list"]
        states_y = ckpt["net_y_state_list"]
        nets_x = [_make_net(int(cfg["rank_x"])) for _ in states_x]
        nets_y = [_make_net(int(cfg["rank_y"])) for _ in states_y]
        for net, sd in zip(nets_x, states_x):
            net.load_state_dict(sd)
        for net, sd in zip(nets_y, states_y):
            net.load_state_dict(sd)
        ch_to_group: List[int] = ckpt.get(
            "ch_to_group", [min(c // 2, len(nets_x) - 1) for c in range(C)]
        )
    else:
        # Shared-basis checkpoint
        net_x = _make_net(int(cfg["rank_x"]))
        net_y = _make_net(int(cfg["rank_y"]))
        net_x.load_state_dict(ckpt["net_x_state"])
        net_y.load_state_dict(ckpt["net_y_state"])
        nets_x = [net_x]
        nets_y = [net_y]
        ch_to_group = [0] * C

    omega = _to_numpy(ckpt["omega"]).astype(np.float32) if "omega" in ckpt else None
    grid_x = _to_numpy(ckpt["grid_x"]).astype(np.float32) if "grid_x" in ckpt else None
    grid_y = _to_numpy(ckpt["grid_y"]).astype(np.float32) if "grid_y" in ckpt else None

    return FTMBasis(
        nets_x=nets_x,
        nets_y=nets_y,
        ch_to_group=ch_to_group,
        rank_x=int(cfg["rank_x"]),
        rank_y=int(cfg["rank_y"]),
        normalize_coords=bool(cfg.get("normalize_coords", True)),
        channel_names=channel_names,
        omega=omega,
        grid_x=grid_x,
        grid_y=grid_y,
    )


def _build_phi_np(
    net_x: nn.Module,
    net_y: nn.Module,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    normalize: bool,
    device: torch.device,
) -> np.ndarray:
    """Build phi matrix (P, R) where P = H*W and R = Rx*Ry."""
    if normalize:
        x_coords = normalize_coords_to_unit(grid_x.astype(np.float64)).astype(np.float32)
        y_coords = normalize_coords_to_unit(grid_y.astype(np.float64)).astype(np.float32)
    else:
        x_coords = grid_x.astype(np.float32)
        y_coords = grid_y.astype(np.float32)

    x_t = torch.from_numpy(x_coords).unsqueeze(-1).to(device)
    y_t = torch.from_numpy(y_coords).unsqueeze(-1).to(device)

    with torch.no_grad():
        phi = build_phi(net_x, net_y, x_t, y_t)
    return phi.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

@dataclass
class ElasticData:
    data: np.ndarray    # (B, M, H, W, 4)
    mask: np.ndarray    # (M, H, W, 4) or (B, M, H, W, 4)
    omega: np.ndarray   # (M,)
    grid_x: np.ndarray  # (H,)
    grid_y: np.ndarray  # (W,)


def load_elastic_data(h5_path: Path, max_samples: int = 0) -> ElasticData:
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as f:
        data = f["data"][...].astype(np.float32)    # (B, M, H, W, 4)
        mask = f["mask_tr"][...].astype(np.float32)  # (B, M, H, W, 4) or (M, H, W, 4)
        omega = f["omega"][...].astype(np.float32)
        grid_x = f["grid_x"][...].astype(np.float32) if "grid_x" in f else \
            np.linspace(0.0, 1.0, data.shape[2], dtype=np.float32)
        grid_y = f["grid_y"][...].astype(np.float32) if "grid_y" in f else \
            np.linspace(0.0, 1.0, data.shape[3], dtype=np.float32)

    if data.ndim != 5:
        raise ValueError(f"Expected (B,M,H,W,C), got {data.shape}")

    if max_samples > 0:
        data = data[:max_samples]
        if mask.ndim == 5:
            mask = mask[:max_samples]

    return ElasticData(data=data, mask=mask, omega=omega, grid_x=grid_x, grid_y=grid_y)


# ---------------------------------------------------------------------------
# Core least-squares solver
# ---------------------------------------------------------------------------

def _solve_channel_core(
    phi_np: np.ndarray,         # (P, R)
    field_flat: np.ndarray,     # (P,)
    mask_flat: np.ndarray,      # (P,) float in {0,1}
    rcond: Optional[float],
) -> np.ndarray:
    """Return core vector (R,) minimising || phi[obs] @ core - field[obs] ||^2."""
    obs = mask_flat > 0.5
    if not np.any(obs):
        raise ValueError("No observed points for this channel")
    phi_obs = phi_np[obs]       # (n_obs, R)
    vals_obs = field_flat[obs]  # (n_obs,)
    core, *_ = np.linalg.lstsq(phi_obs, vals_obs, rcond=rcond)
    return core.astype(np.float32)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _plot_elastic_case(
    out_path: Path,
    gt: np.ndarray,     # (H, W, 4)
    obs: np.ndarray,    # (H, W, 4) - zero outside mask
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
        f"LRTFR | sample={sample_idx} freq={freq_idx} ω={omega_val:.4f} | "
        f"rmse={rmse:.3e} obs={obs_rmse:.3e} unobs={unobs_rmse:.3e}",
        y=0.999, fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = _select_device(args.device)

    basis = load_ftm_basis(Path(args.ftm_ckpt), device)
    ds = load_elastic_data(Path(args.test_h5), args.max_samples)

    data = ds.data           # (B, M, H, W, 4)
    mask = ds.mask
    omega = ds.omega
    grid_x = ds.grid_x if basis.grid_x is None else basis.grid_x
    grid_y = ds.grid_y if basis.grid_y is None else basis.grid_y

    B, M, H, W, C = data.shape
    assert C == 4, f"Expected C=4, got {C}"
    freq_indices = _parse_indices(args.eval_freq_indices, M)

    # Build one phi per basis group
    n_groups = len(basis.nets_x)
    phi_list: List[np.ndarray] = []
    for grp in range(n_groups):
        phi_np = _build_phi_np(
            basis.nets_x[grp], basis.nets_y[grp],
            grid_x, grid_y, basis.normalize_coords, device,
        )
        phi_list.append(phi_np)
        print(f"  Group {grp}: phi shape = {phi_np.shape}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    vis_count = 0

    per_sample_mask = mask.ndim == 5

    for b in range(B):
        for m in freq_indices:
            field = data[b, m]   # (H, W, 4)

            if per_sample_mask:
                mask_bm = mask[b, m]   # (H, W, 4)
            else:
                mask_bm = mask[m]      # (H, W, 4)

            P = H * W
            pred_channels = []
            for c in range(C):
                phi_c = phi_list[basis.ch_to_group[c]]   # (P, R)
                field_flat = field[..., c].flatten()      # (P,)
                mask_flat = mask_bm[..., c].flatten()     # (P,)

                core = _solve_channel_core(phi_c, field_flat, mask_flat, args.rcond)
                pred_c = (phi_c @ core).reshape(H, W)
                pred_channels.append(pred_c)

            pred = np.stack(pred_channels, axis=-1)   # (H, W, 4)

            # Observations (for visualisation)
            obs = field * (mask_bm > 0.5).astype(np.float32)

            # Metrics: use channel-0 mask as the spatial mask indicator
            mask_spatial = mask_bm[..., :1]   # (H, W, 1)
            rmse = _relative_rmse(pred, field, args.eps)
            obs_rmse = _masked_relative_rmse(pred, field, mask_spatial, args.eps)
            unobs_rmse = _masked_relative_rmse(pred, field, 1.0 - mask_spatial, args.eps)
            ch_rmse = {CHANNEL_NAMES[c]: _relative_rmse(pred[..., c], field[..., c], args.eps)
                       for c in range(C)}

            row: Dict[str, Any] = {
                "sample_idx": int(b),
                "freq_idx": int(m),
                "omega": float(omega[m]),
                "rmse": float(rmse),
                "obs_rmse": float(obs_rmse),
                "unobs_rmse": float(unobs_rmse),
            }
            row.update({f"rmse_{k}": float(v) for k, v in ch_rmse.items()})
            rows.append(row)

            if vis_count < args.num_visualize:
                vis_count += 1
                vis_path = out_dir / f"case{vis_count:03d}_s{b:03d}_f{m:03d}.png"
                _plot_elastic_case(
                    vis_path, field, obs, pred,
                    b, m, float(omega[m]),
                    rmse, obs_rmse, unobs_rmse,
                    dpi=args.vis_dpi,
                )
                rows[-1]["vis_path"] = str(vis_path)

        print(f"  [sample {b+1}/{B}] last rmse={rows[-1]['rmse']:.4e}")

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
        "ftm_ckpt": str(args.ftm_ckpt),
        "test_h5": str(args.test_h5),
        "num_cases": len(rows),
        "mean_rmse": float(np.mean(rmses)),
        "mean_obs_rmse": float(np.nanmean([r["obs_rmse"] for r in rows])),
        "mean_unobs_rmse": float(np.nanmean([r["unobs_rmse"] for r in rows])),
        "per_channel": {n: float(np.mean([r[f"rmse_{n}"] for r in rows])) for n in CHANNEL_NAMES},
        "rank_x": basis.rank_x,
        "rank_y": basis.rank_y,
        "n_basis_groups": n_groups,
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
    p = argparse.ArgumentParser(description="LRTFR baseline for elastic wave reconstruction")
    p.add_argument("--ftm_ckpt",           default="ckp/ftm_gpu_checkpoint.pt")
    p.add_argument("--test_h5",            default="elastic_dataset_msk0.01.h5")
    p.add_argument("--out_dir",            default="visual_data/lrtfr_elastic_eval_msk0.01")
    p.add_argument("--eval_freq_indices",  type=str,   default="")
    p.add_argument("--max_samples",        type=int,   default=10)
    p.add_argument("--num_visualize",      type=int,   default=20)
    p.add_argument("--vis_dpi",            type=int,   default=150)
    p.add_argument("--rcond",              type=float, default=None)
    p.add_argument("--eps",                type=float, default=1e-6)
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--device",             type=str,   default="auto")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
