"""
Visualize_core_all_freq.py
--------------------------
Estimate and visualize FTM cores for all frequencies in a test dataset sample.

Workflow
--------
1) Load shared basis networks from a trained FTM checkpoint.
2) Build spatial basis Phi(x, y) on the test grid.
3) For one sample, solve all-frequency real/imag cores by least squares:
      Phi @ core(omega) ~= field(omega)
4) Visualize solved cores across all frequencies.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Ensure project root imports work even when this script is launched from new_idea.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train_FTM_GPU import MLP1D, build_phi, normalize_coords_to_unit


def _safe_torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _sym_limit(arr: np.ndarray, q: float = 99.5) -> float:
    lim = float(np.percentile(np.abs(arr), q))
    return max(lim, 1e-8)


def _pos_limit(arr: np.ndarray, q: float = 99.5) -> float:
    lim = float(np.percentile(arr, q))
    return max(lim, 1e-8)


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_ftm_basis(ftm_ckpt: Path, device: torch.device) -> Dict[str, Any]:
    if not ftm_ckpt.exists():
        raise FileNotFoundError(f"FTM checkpoint not found: {ftm_ckpt}")

    ckpt = _safe_torch_load(ftm_ckpt)
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

    return {
        "net_x": net_x,
        "net_y": net_y,
        "rank_x": int(cfg["rank_x"]),
        "rank_y": int(cfg["rank_y"]),
        "normalize_coords": bool(cfg.get("normalize_coords", True)),
    }


def _load_sample_all_freq_fields(data_h5: Path, sample_idx: int) -> Dict[str, Any]:
    if not data_h5.exists():
        raise FileNotFoundError(f"Dataset not found: {data_h5}")

    with h5py.File(data_h5, "r") as f:
        if "omega" not in f:
            raise KeyError("HDF5 file must contain key 'omega'.")
        omega = f["omega"][...].astype(np.float32)

        if "data" in f:
            ds = f["data"]
            if ds.ndim != 5 or ds.shape[-1] != 2:
                raise ValueError(f"Expected data shape (B,M,H,W,2), got {ds.shape}")
            b, m, h, w, _ = ds.shape
            if not (0 <= sample_idx < b):
                raise ValueError(f"sample_idx out of range [0, {b - 1}]")
            fields = ds[sample_idx].astype(np.float32)
        elif "fields_real" in f and "fields_imag" in f:
            ds_re = f["fields_real"]
            ds_im = f["fields_imag"]
            if ds_re.shape != ds_im.shape or ds_re.ndim != 4:
                raise ValueError(
                    f"Invalid fields_real/fields_imag shapes: {ds_re.shape}, {ds_im.shape}"
                )
            b, m, h, w = ds_re.shape
            if not (0 <= sample_idx < b):
                raise ValueError(f"sample_idx out of range [0, {b - 1}]")
            fields_re = ds_re[sample_idx].astype(np.float32)
            fields_im = ds_im[sample_idx].astype(np.float32)
            fields = np.stack([fields_re, fields_im], axis=-1).astype(np.float32)
        else:
            raise KeyError("HDF5 must contain either 'data' or ('fields_real' and 'fields_imag').")

        if "grid_x" in f:
            grid_x = f["grid_x"][...].astype(np.float32)
        else:
            grid_x = np.linspace(0.0, 1.0, h, dtype=np.float32)

        if "grid_y" in f:
            grid_y = f["grid_y"][...].astype(np.float32)
        else:
            grid_y = np.linspace(0.0, 1.0, w, dtype=np.float32)

    if fields.shape[0] != omega.shape[0]:
        raise ValueError(f"Frequency mismatch: fields M={fields.shape[0]} vs omega={omega.shape[0]}")

    return {
        "fields": fields,       # (M,H,W,2)
        "omega": omega,         # (M,)
        "grid_x": grid_x,       # (H,)
        "grid_y": grid_y,       # (W,)
    }


def _build_phi_np(
    ftm_basis: Dict[str, Any],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if ftm_basis["normalize_coords"]:
        x_coords = normalize_coords_to_unit(grid_x.astype(np.float64)).astype(np.float32)
        y_coords = normalize_coords_to_unit(grid_y.astype(np.float64)).astype(np.float32)
    else:
        x_coords = grid_x.astype(np.float32)
        y_coords = grid_y.astype(np.float32)

    x_t = torch.from_numpy(x_coords).unsqueeze(-1).to(device)
    y_t = torch.from_numpy(y_coords).unsqueeze(-1).to(device)

    with torch.no_grad():
        phi = build_phi(ftm_basis["net_x"], ftm_basis["net_y"], x_t, y_t)
    return phi.detach().cpu().numpy().astype(np.float64)


def _rel_rmse_by_freq(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # pred/gt: (M,H,W,2)
    diff2 = np.sum((pred - gt) ** 2, axis=(1, 2, 3))
    gt2 = np.sum(gt ** 2, axis=(1, 2, 3))
    return np.sqrt(diff2 / np.maximum(gt2, eps)).astype(np.float64)


def solve_cores_all_freq(
    phi_np: np.ndarray,
    fields: np.ndarray,
    rank_x: int,
    rank_y: int,
    lstsq_rcond: float | None,
) -> Dict[str, Any]:
    # fields shape: (M,H,W,2)
    m, h, w, c = fields.shape
    if c != 2:
        raise ValueError("Only 2-channel complex fields are supported.")

    p, r = phi_np.shape
    if p != h * w:
        raise ValueError(f"Phi rows mismatch grid size: phi P={p}, expected H*W={h*w}")
    if r != rank_x * rank_y:
        raise ValueError(f"Phi rank mismatch: phi R={r}, expected rank_x*rank_y={rank_x*rank_y}")

    y_re = fields[..., 0].reshape(m, p).T.astype(np.float64, copy=False)  # (P,M)
    y_im = fields[..., 1].reshape(m, p).T.astype(np.float64, copy=False)  # (P,M)

    sol_re, residual_re, rank_re, s_re = np.linalg.lstsq(phi_np, y_re, rcond=lstsq_rcond)
    sol_im, residual_im, rank_im, s_im = np.linalg.lstsq(phi_np, y_im, rcond=lstsq_rcond)

    # (R,M) -> (M,Rx,Ry)
    cores_re = sol_re.T.reshape(m, rank_x, rank_y).astype(np.float32)
    cores_im = sol_im.T.reshape(m, rank_x, rank_y).astype(np.float32)

    pred_re = (phi_np @ sol_re).T.reshape(m, h, w)
    pred_im = (phi_np @ sol_im).T.reshape(m, h, w)
    pred = np.stack([pred_re, pred_im], axis=-1).astype(np.float32)

    rel_rmse = _rel_rmse_by_freq(pred=pred, gt=fields)

    return {
        "cores_real": cores_re,
        "cores_imag": cores_im,
        "pred_fields": pred,
        "rel_rmse_per_freq": rel_rmse,
        "lstsq_rank_real": int(rank_re),
        "lstsq_rank_imag": int(rank_im),
        "lstsq_singular_values_real": s_re.astype(np.float64),
        "lstsq_singular_values_imag": s_im.astype(np.float64),
        "lstsq_residual_real": residual_re.astype(np.float64),
        "lstsq_residual_imag": residual_im.astype(np.float64),
    }


def plot_core_grid_signed(
    cores: np.ndarray,  # (M,Rx,Ry)
    omega: np.ndarray,  # (M,)
    out_file: Path,
    title: str,
    ncols: int,
    dpi: int,
) -> None:
    m = cores.shape[0]
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(m / ncols))

    vlim = _sym_limit(cores)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.2 * ncols, 2.2 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    im = None
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i >= m:
            ax.axis("off")
            continue

        im = ax.imshow(
            cores[i],
            cmap="RdBu_r",
            origin="lower",
            vmin=-vlim,
            vmax=vlim,
            interpolation="nearest",
        )
        ax.set_title(f"idx={i}\\nomega={float(omega[i]):.4g}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)

    fig.suptitle(title, fontsize=13)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_core_grid_abs(
    cores_abs: np.ndarray,  # (M,Rx,Ry)
    omega: np.ndarray,      # (M,)
    out_file: Path,
    title: str,
    ncols: int,
    dpi: int,
) -> None:
    m = cores_abs.shape[0]
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(m / ncols))

    vmax = _pos_limit(cores_abs)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.2 * ncols, 2.2 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    im = None
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i >= m:
            ax.axis("off")
            continue

        im = ax.imshow(
            cores_abs[i],
            cmap="viridis",
            origin="lower",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(f"idx={i}\\nomega={float(omega[i]):.4g}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)

    fig.suptitle(title, fontsize=13)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_core_norm_curve(
    cores_re: np.ndarray,
    cores_im: np.ndarray,
    omega: np.ndarray,
    out_file: Path,
    dpi: int,
) -> None:
    norm_re = np.linalg.norm(cores_re.reshape(cores_re.shape[0], -1), axis=1)
    norm_im = np.linalg.norm(cores_im.reshape(cores_im.shape[0], -1), axis=1)
    norm_c = np.sqrt(norm_re ** 2 + norm_im ** 2)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(omega, norm_re, lw=1.6, label="||core_real||_F")
    ax.plot(omega, norm_im, lw=1.6, label="||core_imag||_F")
    ax.plot(omega, norm_c, lw=1.8, label="||core_complex||_F")
    ax.set_xlabel("omega")
    ax.set_ylabel("Frobenius norm")
    ax.set_title("Core norm over all frequencies")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Solve and visualize all-frequency cores for one test sample using learned shared basis."
    )
    p.add_argument("--ftm_ckpt", type=str, default="ckp/ftm_gpu_checkpoint.pt")
    p.add_argument(
        "--data_h5",
        type=str,
        default="data_for_test/helmholtz_dataset_42_for_test_mask10.h5",
    )
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="new_idea/core_all_freq_vis")

    p.add_argument("--lstsq_rcond", type=float, default=-1.0)
    p.add_argument("--ncols", type=int, default=10)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--eps", type=float, default=1e-12)

    p.add_argument("--save_npz", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", type=str, default="auto")
    return p


def main() -> None:
    args = build_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _select_device(args.device)
    ftm_basis = _load_ftm_basis(Path(args.ftm_ckpt), device=device)
    data = _load_sample_all_freq_fields(Path(args.data_h5), sample_idx=int(args.sample_idx))

    fields = data["fields"]
    omega = data["omega"]
    grid_x = data["grid_x"]
    grid_y = data["grid_y"]

    phi_np = _build_phi_np(
        ftm_basis=ftm_basis,
        grid_x=grid_x,
        grid_y=grid_y,
        device=device,
    )

    rcond = None if args.lstsq_rcond < 0 else float(args.lstsq_rcond)
    result = solve_cores_all_freq(
        phi_np=phi_np,
        fields=fields,
        rank_x=ftm_basis["rank_x"],
        rank_y=ftm_basis["rank_y"],
        lstsq_rcond=rcond,
    )

    cores_re = result["cores_real"]
    cores_im = result["cores_imag"]
    cores_abs = np.sqrt(cores_re ** 2 + cores_im ** 2)
    rel_rmse = result["rel_rmse_per_freq"]

    sample_tag = f"sample{int(args.sample_idx):03d}"
    out_real = out_dir / f"core_real_allfreq_{sample_tag}.png"
    out_imag = out_dir / f"core_imag_allfreq_{sample_tag}.png"
    out_abs = out_dir / f"core_abs_allfreq_{sample_tag}.png"
    out_norm = out_dir / f"core_norm_curve_{sample_tag}.png"

    plot_core_grid_signed(
        cores=cores_re,
        omega=omega,
        out_file=out_real,
        title=f"Solved real cores for {sample_tag}",
        ncols=args.ncols,
        dpi=args.dpi,
    )
    plot_core_grid_signed(
        cores=cores_im,
        omega=omega,
        out_file=out_imag,
        title=f"Solved imag cores for {sample_tag}",
        ncols=args.ncols,
        dpi=args.dpi,
    )
    plot_core_grid_abs(
        cores_abs=cores_abs,
        omega=omega,
        out_file=out_abs,
        title=f"Solved |complex core| for {sample_tag}",
        ncols=args.ncols,
        dpi=args.dpi,
    )
    plot_core_norm_curve(
        cores_re=cores_re,
        cores_im=cores_im,
        omega=omega,
        out_file=out_norm,
        dpi=args.dpi,
    )

    npz_path = out_dir / f"core_lstsq_{sample_tag}.npz"
    if args.save_npz:
        np.savez_compressed(
            npz_path,
            cores_real=cores_re,
            cores_imag=cores_im,
            omega=omega.astype(np.float32),
            rel_rmse_per_freq=rel_rmse.astype(np.float32),
            lstsq_rank_real=np.array(result["lstsq_rank_real"], dtype=np.int32),
            lstsq_rank_imag=np.array(result["lstsq_rank_imag"], dtype=np.int32),
            lstsq_residual_real=result["lstsq_residual_real"].astype(np.float64),
            lstsq_residual_imag=result["lstsq_residual_imag"].astype(np.float64),
        )

    summary = {
        "ftm_ckpt": str(args.ftm_ckpt),
        "data_h5": str(args.data_h5),
        "sample_idx": int(args.sample_idx),
        "device": str(device),
        "num_freqs": int(fields.shape[0]),
        "field_shape": [int(fields.shape[1]), int(fields.shape[2]), int(fields.shape[3])],
        "rank_x": int(ftm_basis["rank_x"]),
        "rank_y": int(ftm_basis["rank_y"]),
        "lstsq_rcond": None if rcond is None else float(rcond),
        "lstsq_rank_real": int(result["lstsq_rank_real"]),
        "lstsq_rank_imag": int(result["lstsq_rank_imag"]),
        "rel_rmse_mean": float(np.mean(rel_rmse)),
        "rel_rmse_std": float(np.std(rel_rmse)),
        "rel_rmse_max": float(np.max(rel_rmse)),
        "outputs": {
            "core_real_plot": str(out_real),
            "core_imag_plot": str(out_imag),
            "core_abs_plot": str(out_abs),
            "core_norm_curve": str(out_norm),
            "npz": str(npz_path) if args.save_npz else "",
        },
    }

    summary_path = out_dir / f"summary_{sample_tag}.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print("Solved and visualized all-frequency cores.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
