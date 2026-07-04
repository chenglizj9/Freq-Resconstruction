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
from pathlib import Path
from typing import Any, List

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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


def _sym_limit(arr: np.ndarray, q: float = 99.5) -> float:
    lim = float(np.percentile(np.abs(arr), q))
    return max(lim, 1e-8)


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize FTM core tensors directly from checkpoint")
    p.add_argument("--ckpt", type=str, default="ckp/ftm_gpu_checkpoint.pt")
    p.add_argument("--out_dir", type=str, default="visual_data/core_vis")
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument(
        "--freq_indices",
        type=str,
        default="",
        help="Comma-separated frequency indices, e.g. '0,2,5,10'. If empty, evenly sample by --num_freq.",
    )
    p.add_argument("--num_freq", type=int, default=8)
    p.add_argument("--dpi", type=int, default=180)
    return p


def main() -> None:
    args = build_parser().parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cores_real, cores_imag, omega = _load_cores(ckpt_path)
    b_total, m_total, rx, ry = cores_real.shape

    if args.sample_idx < 0 or args.sample_idx >= b_total:
        raise ValueError(f"sample_idx out of range [0, {b_total - 1}]")

    freq_ids = _parse_freq_indices(args.freq_indices, m_total, args.num_freq)

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
        "sample_idx": int(args.sample_idx),
        "core_shape_per_freq": [int(rx), int(ry)],
        "num_total_samples": int(b_total),
        "num_total_freqs": int(m_total),
        "freq_indices_visualized": [int(v) for v in freq_ids],
    }
    print("Saved core visualization.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
