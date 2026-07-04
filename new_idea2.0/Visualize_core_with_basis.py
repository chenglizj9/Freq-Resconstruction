"""
Visualize_core.py
-----------------
Visualize learned FTM core tensors directly from a checkpoint.

This script plots one sample's core matrices across selected frequency indices.
It does NOT decode cores back to physical fields.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import h5py


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


def _load_cores(ckpt_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "cores_real" not in ckpt or "cores_imag" not in ckpt:
        raise KeyError("Checkpoint must contain keys: cores_real and cores_imag")

    cores_real = _to_numpy(ckpt["cores_real"]).astype(np.float32)
    cores_imag = _to_numpy(ckpt["cores_imag"]).astype(np.float32)

    if cores_real.shape != cores_imag.shape:
        raise ValueError(
            f"cores_real and cores_imag shape mismatch: {cores_real.shape} vs {cores_imag.shape}"
        )

    # Expected shape is (B, M, Rx, Ry). If flattened (B, M, R), reshape via config.
    if cores_real.ndim == 4:
        pass
    elif cores_real.ndim == 3:
        cfg = ckpt.get("config", {})
        if "rank_x" not in cfg or "rank_y" not in cfg:
            raise ValueError(
                "Flattened cores require config.rank_x and config.rank_y for reshape."
            )
        rx = int(cfg["rank_x"])
        ry = int(cfg["rank_y"])
        if cores_real.shape[-1] != rx * ry:
            raise ValueError(
                f"Flattened core size {cores_real.shape[-1]} does not match rank_x*rank_y={rx*ry}."
            )
        cores_real = cores_real.reshape(cores_real.shape[0], cores_real.shape[1], rx, ry)
        cores_imag = cores_imag.reshape(cores_imag.shape[0], cores_imag.shape[1], rx, ry)
    else:
        raise ValueError(
            f"Unsupported core tensor shape {cores_real.shape}; expected 3D or 4D array."
        )

    omega = ckpt.get("omega", None)
    omega_np = None if omega is None else _to_numpy(omega).astype(np.float32)
    return cores_real, cores_imag, omega_np


def _load_grid_from_h5(data_h5: Path) -> tuple[np.ndarray, np.ndarray]:
    if not data_h5.exists():
        raise FileNotFoundError(f"Dataset not found: {data_h5}")

    with h5py.File(data_h5, "r") as f:
        if "data" not in f:
            raise KeyError("HDF5 must contain key 'data' to infer grid shape.")
        ds = f["data"]
        if ds.ndim != 5 or ds.shape[-1] != 2:
            raise ValueError(f"Expected data shape (B,M,H,W,2), got {ds.shape}")
        h = int(ds.shape[2])
        w = int(ds.shape[3])

        if "grid_x" in f:
            grid_x = f["grid_x"][...].astype(np.float32)
        else:
            grid_x = np.linspace(0.0, 1.0, h, dtype=np.float32)

        if "grid_y" in f:
            grid_y = f["grid_y"][...].astype(np.float32)
        else:
            grid_y = np.linspace(0.0, 1.0, w, dtype=np.float32)

    return grid_x, grid_y


def _parse_freq_indices(freq_indices: str, m_total: int, num_freq: int) -> List[int]:
    if freq_indices.strip():
        out: List[int] = []
        for chunk in freq_indices.split(","):
            idx = int(chunk.strip())
            if idx < 0 or idx >= m_total:
                raise ValueError(f"freq index {idx} out of range [0, {m_total - 1}]")
            if idx not in out:
                out.append(idx)
        if not out:
            raise ValueError("freq_indices is empty after parsing.")
        return out

    n = max(1, min(int(num_freq), m_total))
    return np.linspace(0, m_total - 1, n, dtype=int).tolist()


def _parse_basis_indices(text: str, total_basis: int, num_basis: int) -> List[int]:
    if text.strip():
        out: List[int] = []
        for chunk in text.split(","):
            idx = int(chunk.strip())
            if idx < 0 or idx >= total_basis:
                raise ValueError(f"basis index {idx} out of range [0, {total_basis - 1}]")
            if idx not in out:
                out.append(idx)
        if not out:
            raise ValueError("basis_indices is empty after parsing.")
        return out

    n = max(1, min(int(num_basis), total_basis))
    return np.linspace(0, total_basis - 1, n, dtype=int).tolist()


def _sym_limit(arr: np.ndarray, q: float = 99.5) -> float:
    lim = float(np.percentile(np.abs(arr), q))
    return max(lim, 1e-8)


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
    return phi.detach().cpu().numpy().astype(np.float32)


def plot_core_sample(
    cores_real: np.ndarray,
    cores_imag: np.ndarray,
    omega: np.ndarray | None,
    sample_idx: int,
    freq_ids: List[int],
    out_file: Path,
    dpi: int,
) -> None:
    # cores_* shape: (B, M, Rx, Ry)
    re_sel = cores_real[sample_idx, freq_ids]  # (F, Rx, Ry)
    im_sel = cores_imag[sample_idx, freq_ids]

    vlim_re = _sym_limit(re_sel)
    vlim_im = _sym_limit(im_sel)

    n_cols = len(freq_ids)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(3.2 * n_cols, 6.2),
        squeeze=False,
        constrained_layout=True,
    )

    im0 = None
    im1 = None
    for c, f_idx in enumerate(freq_ids):
        title = f"freq_idx={f_idx}"
        if omega is not None and f_idx < len(omega):
            title += f"\nomega={float(omega[f_idx]):.4g}"

        ax = axes[0, c]
        im0 = ax.imshow(
            cores_real[sample_idx, f_idx],
            cmap="RdBu_r",
            origin="lower",
            vmin=-vlim_re,
            vmax=vlim_re,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Ry")
        ax.set_ylabel("Rx")

        ax = axes[1, c]
        im1 = ax.imshow(
            cores_imag[sample_idx, f_idx],
            cmap="RdBu_r",
            origin="lower",
            vmin=-vlim_im,
            vmax=vlim_im,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Ry")
        ax.set_ylabel("Rx")

    if im0 is not None:
        fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.02, pad=0.02)
    if im1 is not None:
        fig.colorbar(im1, ax=axes[1, :].tolist(), fraction=0.02, pad=0.02)

    axes[0, 0].text(
        -0.2,
        0.5,
        "Real core",
        transform=axes[0, 0].transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=10,
        fontweight="bold",
    )
    axes[1, 0].text(
        -0.2,
        0.5,
        "Imag core",
        transform=axes[1, 0].transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=10,
        fontweight="bold",
    )

    fig.suptitle(f"FTM cores for sample={sample_idx}", fontsize=13)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_basis_grid(
    phi: np.ndarray,
    basis_ids: List[int],
    grid_shape: tuple[int, int],
    out_file: Path,
    title: str,
    ncols: int,
    dpi: int,
) -> None:
    h, w = grid_shape
    n_basis = len(basis_ids)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n_basis / ncols))

    basis_maps = [phi[:, idx].reshape(h, w) for idx in basis_ids]
    vlim = _sym_limit(np.stack(basis_maps, axis=0))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.6 * ncols, 2.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    im = None
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i >= n_basis:
            ax.axis("off")
            continue

        b_idx = basis_ids[i]
        basis_map = basis_maps[i]
        im = ax.imshow(
            basis_map,
            cmap="RdBu_r",
            origin="lower",
            vmin=-vlim,
            vmax=vlim,
            interpolation="nearest",
        )
        ax.set_title(f"basis_idx={b_idx}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)

    fig.suptitle(title, fontsize=13)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize FTM core tensors and shared basis functions")
    p.add_argument("--ckpt", type=str, default="ckp/ftm_gpu_checkpoint.pt")
    p.add_argument("--ftm_ckpt", type=str, default="ckp/ftm_gpu_checkpoint.pt")
    p.add_argument("--data_h5", type=str, default="data_for_test/helmholtz_dataset_42_for_test_mask10.h5")
    p.add_argument("--out_dir", type=str, default="visual_data/core_vis")
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument(
        "--freq_indices",
        type=str,
        default="",
        help="Comma-separated frequency indices, e.g. '0,2,5,10'. If empty, evenly sample by --num_freq.",
    )
    p.add_argument("--num_freq", type=int, default=8)
    p.add_argument(
        "--basis_indices",
        type=str,
        default="",
        help="Comma-separated basis indices. If empty, evenly sample by --num_basis.",
    )
    p.add_argument("--num_basis", type=int, default=8)
    p.add_argument("--basis_ncols", type=int, default=4)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--device", type=str, default="auto")
    return p


def main() -> None:
    args = build_parser().parse_args()

    ckpt_path = Path(args.ckpt)
    ftm_ckpt_path = Path(args.ftm_ckpt)
    data_h5_path = Path(args.data_h5)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _select_device(args.device)

    cores_real, cores_imag, omega = _load_cores(ckpt_path)
    b_total, m_total, rx, ry = cores_real.shape

    if args.sample_idx < 0 or args.sample_idx >= b_total:
        raise ValueError(f"sample_idx out of range [0, {b_total - 1}]")

    freq_ids = _parse_freq_indices(args.freq_indices, m_total, args.num_freq)

    ftm_basis = _load_ftm_basis(ftm_ckpt_path, device=device)
    grid_x, grid_y = _load_grid_from_h5(data_h5_path)

    # Use the dataset spatial grid so the basis functions are plotted in the same physical coordinates
    # as the fields/cores produced by the main pipeline.
    phi = None
    basis_ids: List[int] = []
    basis_out = out_dir / f"basis_sample{args.sample_idx:03d}.png"

    try:
        phi = _build_phi_np(
            ftm_basis=ftm_basis,
            grid_x=grid_x,
            grid_y=grid_y,
            device=device,
        )
        basis_ids = _parse_basis_indices(args.basis_indices, ftm_basis["rank_x"] * ftm_basis["rank_y"], args.num_basis)
        plot_basis_grid(
            phi=phi,
            basis_ids=basis_ids,
            grid_shape=(len(grid_x), len(grid_y)),
            out_file=basis_out,
            title="Shared FTM basis functions",
            ncols=args.basis_ncols,
            dpi=args.dpi,
        )
    except Exception as exc:
        basis_out = out_dir / f"basis_sample{args.sample_idx:03d}_failed.txt"
        with open(basis_out, "w", encoding="utf-8") as fp:
            fp.write(str(exc))

    out_file = out_dir / f"core_sample{args.sample_idx:03d}.png"
    plot_core_sample(
        cores_real=cores_real,
        cores_imag=cores_imag,
        omega=omega,
        sample_idx=args.sample_idx,
        freq_ids=freq_ids,
        out_file=out_file,
        dpi=args.dpi,
    )

    summary = {
        "checkpoint": str(ckpt_path),
        "output": str(out_file),
        "basis_output": str(basis_out),
        "sample_idx": int(args.sample_idx),
        "core_shape_per_freq": [int(rx), int(ry)],
        "num_total_samples": int(b_total),
        "num_total_freqs": int(m_total),
        "freq_indices_visualized": [int(v) for v in freq_ids],
        "ftm_ckpt": str(ftm_ckpt_path),
        "data_h5": str(data_h5_path),
        "num_basis_visualized": int(len(basis_ids)),
        "basis_indices_visualized": [int(v) for v in basis_ids],
    }
    print("Saved core visualization.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
