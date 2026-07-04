"""
Visualize_dataset.py  (ocean acoustic)
--------------------------------------
Visualise the generated ocean-acoustic complex-pressure dataset.

For each complex field p(z, r) = Re + j*Im we plot four quantities:
    real( Re )   |   imag( Im )   |   magnitude |p|   |   phase  arg(p)

Two figures are produced:

1. dataset_overview.png   — DIVERSITY across samples.
   rows = several samples (different environments), columns = [Re, Im, |p|, phase],
   all at one showcase frequency.  Each row is annotated with its environment
   (water depth / source depth / seabed / SSP / model).

2. dataset_freq_sweep.png — MULTI-FREQUENCY evolution of one sample.
   rows = [Re, Im, |p|, phase], columns = a spread of frequencies.

The dataset is stored per-frequency normalised, so magnitudes are comparable
across samples at a fixed frequency.

Usage
-----
    python Visualize_dataset.py --data_h5 ocean_dataset.h5
    python Visualize_dataset.py --samples 0 50 120 300 480 --freq_idx 3
    python Visualize_dataset.py --sweep_sample 7 --sweep_freqs 0 2 4 6 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str):
    with h5py.File(path, "r") as f:
        data = f["data"][...].astype(np.float32)          # (N, M, H, W, 2)
        omega = f["omega"][...].astype(np.float64)         # (M,) Hz
        meta = json.loads(f["metadata"][()]) if "metadata" in f else {}
    samples_meta = {s["sample_id"]: s for s in meta.get("samples", [])}
    return data, omega, meta, samples_meta


def complex_field(data, n, m):
    """Return (Re, Im, |p|, phase) each (H, W) for sample n, freq idx m."""
    re = data[n, m, :, :, 0]
    im = data[n, m, :, :, 1]
    p = re + 1j * im
    return re, im, np.abs(p), np.angle(p)


def env_label(sm: dict | None) -> str:
    if not sm:
        return ""
    ssp = sm.get("ssp_mode", "?")
    if ssp == "isovelocity":
        ssp = f"iso {sm.get('ssp_speed', 0):.0f}m/s"
    obs = "  obstacle" if sm.get("has_obstacle") else ""
    return (f"{sm.get('model_name','?')}\n"
            f"wd={sm.get('water_depth_m',0):.0f}m  sd={sm.get('source_depth_m',0):.0f}m\n"
            f"{sm.get('seabed','?')}  {ssp}{obs}")


# ─────────────────────────────────────────────────────────────────────────────
# Panel plotting
# ─────────────────────────────────────────────────────────────────────────────

_QUANTITIES = ["real (Re)", "imag (Im)", "|p| (dB re peak)", "phase arg(p)"]


_DB_FLOOR = -40.0   # dB display floor for magnitude (relative to per-panel peak)


def _draw_panel(ax, field, kind, extent):
    """Draw one quantity.  Robust dynamic range: acoustic fields have a strong
    near-source spike, so Re/Im use a 98th-percentile symmetric scale and |p|
    is shown in dB (re. its own peak) — both reveal the wave/modal structure."""
    if kind == "phase":
        im = ax.imshow(field, origin="upper", aspect="auto", extent=extent,
                       cmap="twilight", vmin=-np.pi, vmax=np.pi)
    elif kind == "mag":
        peak = float(np.max(field)) or 1.0
        db = 20.0 * np.log10(np.maximum(field / peak, 1e-6))
        im = ax.imshow(db, origin="upper", aspect="auto", extent=extent,
                       cmap="magma", vmin=_DB_FLOOR, vmax=0.0)
    else:  # real / imag — symmetric diverging, clipped to 98th percentile
        a = float(np.percentile(np.abs(field), 98)) or float(np.max(np.abs(field))) or 1.0
        im = ax.imshow(field, origin="upper", aspect="auto", extent=extent,
                       cmap="RdBu_r", vmin=-a, vmax=a)
    return im


def _extent_for(meta, sm):
    """Physical extent (range km on x, depth m on y) for axis labelling."""
    r0, r1 = meta.get("range_m", [0.0, 1.0])
    wd = (sm or {}).get("water_depth_m", 1.0)
    return (r0 / 1000.0, r1 / 1000.0, wd, 0.0)   # depth increases downward


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: diversity across samples
# ─────────────────────────────────────────────────────────────────────────────

def plot_overview(data, omega, meta, samples_meta, sample_ids, freq_idx, out_path):
    nrows = len(sample_ids)
    fig = plt.figure(figsize=(15, 2.9 * nrows + 0.6))
    gs = GridSpec(nrows, 5, width_ratios=[1, 1, 1, 1, 0.05], figure=fig,
                  hspace=0.32, wspace=0.18)
    f_hz = omega[freq_idx]
    kinds = ["re", "im", "mag", "phase"]

    for r, n in enumerate(sample_ids):
        sm = samples_meta.get(n)
        re, im, mag, pha = complex_field(data, n, freq_idx)
        fields = [re, im, mag, pha]
        extent = _extent_for(meta, sm)
        for c, (field, kind, title) in enumerate(zip(fields, kinds, _QUANTITIES)):
            ax = fig.add_subplot(gs[r, c])
            imobj = _draw_panel(ax, field, kind, extent)
            if r == 0:
                ax.set_title(title, fontsize=12)
            if c == 0:
                ax.set_ylabel(f"sample {n}\n" + env_label(sm), fontsize=8)
                ax.set_yticks([])
            else:
                ax.set_yticks([])
            if r == nrows - 1:
                ax.set_xlabel("range (km)", fontsize=8)
            else:
                ax.set_xticks([])
            fig.colorbar(imobj, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(f"Ocean acoustic dataset — diversity across samples  "
                 f"(f = {f_hz:.0f} Hz, depth↓ vs range→)", fontsize=14, y=0.995)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: frequency sweep of one sample
# ─────────────────────────────────────────────────────────────────────────────

def plot_freq_sweep(data, omega, meta, samples_meta, sample_id, freq_ids, out_path):
    ncols = len(freq_ids)
    sm = samples_meta.get(sample_id)
    extent = _extent_for(meta, sm)
    row_kinds = ["re", "im", "mag", "phase"]

    fig = plt.figure(figsize=(2.7 * ncols + 1.2, 11))
    gs = GridSpec(4, ncols, figure=fig, hspace=0.22, wspace=0.18)

    for r, kind in enumerate(row_kinds):
        for c, m in enumerate(freq_ids):
            re, im, mag, pha = complex_field(data, sample_id, m)
            field = {"re": re, "im": im, "mag": mag, "phase": pha}[kind]
            ax = fig.add_subplot(gs[r, c])
            imobj = _draw_panel(ax, field, kind, extent)
            if r == 0:
                ax.set_title(f"{omega[m]:.0f} Hz", fontsize=11)
            if c == 0:
                ax.set_ylabel(_QUANTITIES[r], fontsize=11)
                ax.set_yticks([])
            else:
                ax.set_yticks([])
            if r == 3:
                ax.set_xlabel("range (km)", fontsize=8)
            else:
                ax.set_xticks([])
            fig.colorbar(imobj, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(f"Ocean acoustic dataset — frequency sweep, sample {sample_id}\n"
                 + env_label(sm).replace("\n", "  "), fontsize=13, y=0.99)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Sample picker (maximise environment diversity)
# ─────────────────────────────────────────────────────────────────────────────

def pick_diverse_samples(samples_meta, N, k):
    """Pick k samples spread over seabed/water-depth/ssp to showcase diversity."""
    ids = sorted(samples_meta.keys())
    if not ids:
        return list(range(min(k, N)))
    # sort by (seabed, ssp_mode, water_depth) then take evenly spaced picks
    ids.sort(key=lambda i: (samples_meta[i].get("seabed", ""),
                            samples_meta[i].get("ssp_mode", ""),
                            samples_meta[i].get("water_depth_m", 0.0)))
    idx = np.linspace(0, len(ids) - 1, num=min(k, len(ids))).round().astype(int)
    return [ids[j] for j in idx]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Visualise ocean acoustic dataset")
    p.add_argument("--data_h5", type=str, default="ocean_dataset.h5")
    p.add_argument("--out_dir", type=str, default="visual_data")
    # overview (diversity)
    p.add_argument("--samples", type=int, nargs="*", default=None,
                   help="sample ids for the overview grid (default: auto-diverse)")
    p.add_argument("--n_show", type=int, default=5, help="#samples when auto-picking")
    p.add_argument("--freq_idx", type=int, default=-1,
                   help="frequency index for the overview (default: middle)")
    # frequency sweep
    p.add_argument("--sweep_sample", type=int, default=0)
    p.add_argument("--sweep_freqs", type=int, nargs="*", default=None,
                   help="freq indices for the sweep (default: evenly spaced)")
    args = p.parse_args()

    data, omega, meta, samples_meta = load_dataset(args.data_h5)
    N, M = data.shape[0], data.shape[1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    freq_idx = args.freq_idx if args.freq_idx >= 0 else M // 2
    sample_ids = args.samples if args.samples else \
        pick_diverse_samples(samples_meta, N, args.n_show)
    sweep_freqs = args.sweep_freqs if args.sweep_freqs else \
        list(np.linspace(0, M - 1, num=min(5, M)).round().astype(int))

    print(f"Dataset {args.data_h5}: N={N} M={M} grid={data.shape[2]}x{data.shape[3]} "
          f"freqs={omega.tolist()}")
    print(f"  overview: samples={sample_ids} @ {omega[freq_idx]:.0f} Hz")
    plot_overview(data, omega, meta, samples_meta, sample_ids, freq_idx,
                  out_dir / "dataset_overview.png")
    print(f"  sweep: sample={args.sweep_sample} freqs={[int(omega[i]) for i in sweep_freqs]}")
    plot_freq_sweep(data, omega, meta, samples_meta, args.sweep_sample, sweep_freqs,
                    out_dir / "dataset_freq_sweep.png")


if __name__ == "__main__":
    main()
