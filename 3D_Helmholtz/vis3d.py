"""
vis3d.py — 3D visualization utilities for the 3D Helmholtz baselines.

Two public functions:
    plot_field_3d(gt, pred, mask, *, title, out_path, dpi, ...)
        Produces a figure with 3D volume point-cloud views for GT, Pred, and |Error|,
        one row per displayed field, including amplitude for complex-valued inputs.

    plot_sensors_3d(mask, field, *, title, out_path, dpi, ...)
        Produces a 3D scatter of observed sensor locations coloured by field value.

All functions are non-interactive (Agg backend) and save to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sym_lim(vol: np.ndarray, q: float = 0.98):
    """Symmetric colour limits centred at 0 for a signed field volume."""
    lo, hi = np.nanpercentile(vol, [100*(1-q), 100*q])
    v = max(abs(lo), abs(hi), 1e-12)
    return -v, v


def _draw_volume_points(
    ax,
    vol: np.ndarray,       # (Nx, Ny, Nz)  single channel
    vmin: float,
    vmax: float,
    cmap: str,
    alpha: float = 0.55,
    max_pts: int = 12000,
    quantile: float = 0.82,
) -> None:
    """Draw a sparse 3D point-cloud view of a scalar volume."""
    Nx, Ny, Nz = vol.shape
    vals = np.asarray(vol, dtype=np.float32)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return

    abs_vals = np.abs(vals[finite])
    thresh = float(np.nanquantile(abs_vals, quantile)) if abs_vals.size else 0.0
    keep = finite & (np.abs(vals) >= thresh)
    pts = np.argwhere(keep)
    if pts.size == 0:
        pts = np.argwhere(finite)

    if len(pts) > max_pts:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), max_pts, replace=False)]

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors = plt.get_cmap(cmap)(norm(vals[pts[:, 0], pts[:, 1], pts[:, 2]]))
    colors[:, 3] = alpha

    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=colors, s=4, marker="o", linewidths=0, depthshade=False,
    )

    ax.set_xlim(0, Nx - 1); ax.set_ylim(0, Ny - 1); ax.set_zlim(0, Nz - 1)
    ax.set_xlabel("X", fontsize=7, labelpad=1)
    ax.set_ylabel("Y", fontsize=7, labelpad=1)
    ax.set_zlabel("Z", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=22, azim=-55)


def _add_colorbar(fig, ax, vmin, vmax, cmap, label="", pad=0.02):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=pad, shrink=0.7)
    cb.ax.tick_params(labelsize=6)
    if label:
        cb.set_label(label, fontsize=7)
    return cb


def _build_display_fields(gt: np.ndarray, pred: np.ndarray, channel_names: Sequence[str]):
    fields = []
    for ci, name in enumerate(channel_names[:gt.shape[-1]]):
        g = gt[..., ci]
        p = pred[..., ci]
        err = np.abs(p - g)
        vf_min, vf_max = _sym_lim(g)
        ve_min, ve_max = 0.0, float(np.nanpercentile(err, 98)) + 1e-12
        fields.append((name, g, p, err, vf_min, vf_max, ve_min, ve_max))

    if gt.shape[-1] >= 2:
        gt_amp = np.sqrt(np.sum(gt[..., :2] ** 2, axis=-1))
        pred_amp = np.sqrt(np.sum(pred[..., :2] ** 2, axis=-1))
        err_amp = np.abs(pred_amp - gt_amp)
        va_min, va_max = 0.0, float(np.nanpercentile(gt_amp, 98)) + 1e-12
        vea_min, vea_max = 0.0, float(np.nanpercentile(err_amp, 98)) + 1e-12
        fields.append(("Amplitude", gt_amp, pred_amp, err_amp, va_min, va_max, vea_min, vea_max))

    return fields


# ─────────────────────────────────────────────────────────────────────────────
# Public: orthogonal-slice 3D figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_field_3d(
    gt: np.ndarray,           # (Nx, Ny, Nz, C) — full GT field
    pred: np.ndarray,         # (Nx, Ny, Nz, C) — model prediction
    mask: Optional[np.ndarray] = None,  # (Nx, Ny, Nz) or (Nx, Ny, Nz, 1) binary
    *,
    channel_names: Sequence[str] = ("Real", "Imag"),
    title: str = "",
    out_path: str | Path = "field_3d.png",
    dpi: int = 130,
    cmap_field: str = "RdBu_r",
    cmap_err: str = "magma",
    alpha: float = 0.88,
    levels: int = 48,
) -> None:
    """
    Save a 3D volume-style figure.

    Layout: N rows × 3 columns (GT | Pred | |Error|) per displayed field,
    where N includes the original channels plus amplitude for complex-valued inputs.
    """
    Nx, Ny, Nz, _ = gt.shape
    display_fields = _build_display_fields(gt, pred, channel_names)
    nrows = len(display_fields); ncols = 3
    fig = plt.figure(figsize=(5.5 * ncols, 4.5 * nrows))
    fig.patch.set_facecolor("#1a1a2e")

    for ci, (name, g, p, err, vf_min, vf_max, ve_min, ve_max) in enumerate(display_fields):
        for col, (vol, lbl, cm, vn, vx) in enumerate([
            (g,   name + " — GT",   cmap_field, vf_min, vf_max),
            (p,   name + " — Pred",  cmap_field, vf_min, vf_max),
            (err, name + " — |Err|", cmap_err,   ve_min, ve_max),
        ]):
            ax = fig.add_subplot(nrows, ncols, ci * ncols + col + 1, projection="3d")
            ax.set_facecolor("#1a1a2e")
            _draw_volume_points(ax, vol, vn, vx, cm, alpha=alpha)

            corners = np.array([
                [0, 0, 0], [Nx - 1, 0, 0], [Nx - 1, Ny - 1, 0], [0, Ny - 1, 0], [0, 0, 0],
                [0, 0, Nz - 1], [Nx - 1, 0, Nz - 1], [Nx - 1, Ny - 1, Nz - 1], [0, Ny - 1, Nz - 1], [0, 0, Nz - 1],
            ], dtype=np.float32)
            ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color="white", linewidth=0.5, alpha=0.35)

            ax.set_title(lbl, color="white", fontsize=9, pad=2)
            _add_colorbar(fig, ax, vn, vx, cm)

    if title:
        fig.suptitle(title, color="white", fontsize=10, y=1.01)

    plt.tight_layout(pad=0.5)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Public: sensor-scatter 3D figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensors_3d(
    mask: np.ndarray,         # (Nx, Ny, Nz) or (Nx, Ny, Nz, 1)  binary
    field: np.ndarray,        # (Nx, Ny, Nz, C)  — GT field for colouring
    *,
    channel: int = 0,
    title: str = "",
    out_path: str | Path = "sensors_3d.png",
    dpi: int = 130,
    cmap: str = "RdBu_r",
    max_pts: int = 4000,
    show_volume_outline: bool = True,
) -> None:
    """
    Save a 3D scatter plot of the observation sensor locations.

    Points are coloured by the field value at that location.
    If more than `max_pts` sensors exist, a random subset is shown.
    """
    if mask.ndim == 4: mask = mask[..., 0]
    Nx, Ny, Nz = mask.shape

    obs_xyz = np.argwhere(mask.astype(bool))   # (K, 3) — ix, iy, iz
    K = len(obs_xyz)

    if K == 0:
        # nothing to plot
        return

    rng = np.random.default_rng(0)
    if K > max_pts:
        sel = rng.choice(K, max_pts, replace=False)
        obs_xyz = obs_xyz[sel]

    vals = field[obs_xyz[:, 0], obs_xyz[:, 1], obs_xyz[:, 2], channel]
    vmin, vmax = _sym_lim(vals)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#1a1a2e")

    sc = ax.scatter(
        obs_xyz[:, 0], obs_xyz[:, 1], obs_xyz[:, 2],
        c=vals, cmap=cmap, norm=norm,
        s=12, alpha=0.8, linewidths=0,
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.05, shrink=0.7)
    cb.set_label(f"Field ch{channel}", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    if show_volume_outline:
        # Draw bounding box wireframe
        xs, ys, zs = Nx-1, Ny-1, Nz-1
        corners = [[0,0,0],[xs,0,0],[xs,ys,0],[0,ys,0],[0,0,0],
                   [0,0,zs],[xs,0,zs],[xs,ys,zs],[0,ys,zs],[0,0,zs]]
        bx, by, bz = zip(*corners)
        ax.plot(bx, by, bz, color="white", linewidth=0.6, alpha=0.4)
        ax.plot([xs,xs],[0,0],[0,zs], color="white", lw=0.6, alpha=0.4)
        ax.plot([xs,xs],[ys,ys],[0,zs], color="white", lw=0.6, alpha=0.4)
        ax.plot([0,0],[ys,ys],[0,zs], color="white", lw=0.6, alpha=0.4)
        ax.plot([xs,xs],[0,ys],[zs,zs], color="white", lw=0.6, alpha=0.4)

    ax.set_xlim(0, Nx-1); ax.set_ylim(0, Ny-1); ax.set_zlim(0, Nz-1)
    ax.set_xlabel("X", color="white", fontsize=8)
    ax.set_ylabel("Y", color="white", fontsize=8)
    ax.set_zlabel("Z", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=6)
    ax.view_init(elev=20, azim=-50)

    info = f"{K} sensors ({100*K/(Nx*Ny*Nz):.1f}% of volume)"
    if title:
        info = title + "\n" + info
    ax.set_title(info, color="white", fontsize=9, pad=6)

    plt.tight_layout(pad=0.4)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper: both plots at once
# ─────────────────────────────────────────────────────────────────────────────

def save_3d_visuals(
    gt: np.ndarray,           # (Nx, Ny, Nz, C)
    pred: np.ndarray,         # (Nx, Ny, Nz, C)
    mask: np.ndarray,         # (Nx, Ny, Nz) or (Nx, Ny, Nz, 1)
    *,
    stem: str | Path,         # path prefix (without extension)
    title: str = "",
    dpi: int = 130,
    channel_names: Sequence[str] = ("Real", "Imag"),
    cmap_field: str = "RdBu_r",
    cmap_err: str = "magma",
) -> None:
    """
    Save both 3D visuals for one test case.

    Writes two files:
        <stem>_ortho3d.png   — sparse 3D volume view
        <stem>_sensors3d.png — sensor scatter coloured by field
    """
    stem = Path(stem)
    plot_field_3d(
        gt, pred, mask,
        channel_names=channel_names,
        title=title,
        out_path=stem.with_name(stem.name + "_ortho3d.png"),
        dpi=dpi, cmap_field=cmap_field, cmap_err=cmap_err,
    )
    plot_sensors_3d(
        mask, gt,
        title=title,
        out_path=stem.with_name(stem.name + "_sensors3d.png"),
        dpi=dpi,
    )
