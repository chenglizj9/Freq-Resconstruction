"""
generate_dataset.py
-------------------
Generate the (N_samples × N_freqs) Helmholtz dataset described in the project doc.

Dataset layout (HDF5)
---------------------
/metadata          — JSON string with all generation parameters
/omega             — (M,)       float64  — frequency values
/sources           — (N, K, 2)  float64  — source positions per sample
/amplitudes        — (N, K, 2)  float64  — (real, imag) source amplitudes per sample
/fields_real       — (N, M, Ng, Ng) float32 — Re(u)
/fields_imag       — (N, M, Ng, Ng) float32 — Im(u)
/grid_x            — (Ng,)      float64  — 1-D x coordinates
/grid_y            — (Ng,)      float64  — 1-D y coordinates

Usage
-----
    python Generate_dataset.py                       # default config (also exports FTM-ready data)
    python Generate_dataset.py --N 50 --M 20 --grid 64 --out my_data.h5
    python Generate_dataset.py --obs_ratio 0.01 --mask_mode per_sample_per_freq
    python Generate_dataset.py --omega_min 2 --omega_max 30 --N 100 --M 40
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from Helmholtz_Solver import HelmholtzSolver


def build_sparse_mask(
    rng: np.random.Generator,
    N: int,
    M: int,
    grid: int,
    channels: int,
    obs_ratio: float,
    mask_mode: str,
) -> np.ndarray:
    """
    Build sparse observation mask for FTM training.

    Returns
    -------
    mask_tr :
        - shared-mask modes:       (M, grid, grid, channels) uint8 in {0,1}
        - per-sample-mask modes:   (N, M, grid, grid, channels) uint8 in {0,1}
    """
    if not (0.0 < obs_ratio <= 1.0):
        raise ValueError("obs_ratio must be in (0, 1].")

    if mask_mode == "fixed":
        # One spatial mask shared by all samples and all frequencies.
        spatial = (rng.random((grid, grid)) < obs_ratio).astype(np.uint8)
        mask = np.repeat(spatial[None, :, :, None], M, axis=0)
    elif mask_mode == "per_freq":
        # One mask per frequency, shared by all samples.
        mask = (rng.random((M, grid, grid, 1)) < obs_ratio).astype(np.uint8)
    elif mask_mode == "per_sample_fixed":
        # One independent mask per sample, shared across that sample's frequencies.
        spatial = (rng.random((N, grid, grid, 1)) < obs_ratio).astype(np.uint8)
        mask = np.repeat(spatial[:, None, :, :, :], M, axis=1)
    elif mask_mode in {"per_sample", "per_sample_per_freq"}:
        # Fully independent masks per sample and per frequency.
        mask = (rng.random((N, M, grid, grid, 1)) < obs_ratio).astype(np.uint8)
    else:
        raise ValueError(
            "mask_mode must be one of: fixed, per_freq, per_sample_fixed, per_sample_per_freq"
        )

    return np.repeat(mask, channels, axis=-1).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Source configuration sampler
# ──────────────────────────────────────────────────────────────────────────────

def sample_sources(
    rng: np.random.Generator,
    L: float,
    pml_frac: float,
    n_sources: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample K point-source positions uniformly inside the physical interior
    (i.e. excluding the PML region) and complex amplitudes with unit magnitude.

    Returns
    -------
    positions  : (K, 2) float64
    amplitudes : (K,)   complex128
    """
    margin = pml_frac * L + 0.05 * L   # keep sources clear of PML
    lo, hi = margin, L - margin
    positions = rng.uniform(lo, hi, size=(n_sources, 2))
    phases = rng.uniform(0, 2 * np.pi, size=n_sources)
    amplitudes = np.exp(1j * phases)
    return positions, amplitudes


# ──────────────────────────────────────────────────────────────────────────────
# Main generation routine
# ──────────────────────────────────────────────────────────────────────────────

def generate(
    N: int = 200,
    M: int = 50,
    grid: int = 128,
    L: float = 1.0,
    c: float = 1.0,
    omega_min: float = 2.0,
    omega_max: float = 51.0,
    n_sources_range: tuple[int, int] = (1, 4),
    pml_width: float = 0.12,
    sigma_max: float = 50.0,
    seed: int = 42,
    out: str = "helmholtz_dataset.h5",
    omega_scale: str = "linear",   # "linear" | "log"
    min_ppw: float = 8.0,
    export_ftm: bool = True,
    metadata_out: str = "",
    obs_ratio: float = 0.01,
    mask_mode: str = "per_sample_per_freq",
    normalize: bool = True,
    source_sigma: float = 0.025,
) -> Path:
    """
    Generate the dataset and save to HDF5.

    Parameters
    ----------
    N              : number of independent samples (different BC / source configs)
    M              : number of frequency points per sample
    grid           : spatial grid resolution (Ng × Ng)
    L              : domain side length
    c              : wave speed
    omega_min/max  : frequency range  [ω_min, ω_max]
    n_sources_range: min and max number of point sources per sample
    pml_width      : PML thickness as fraction of L
    sigma_max      : PML conductivity peak
    seed           : master random seed
    out            : output file path
    omega_scale    : "linear" or "log" spacing of frequency grid
    min_ppw        : minimum points-per-wavelength at omega_max (warning only)
    """
    if N <= 0 or M <= 0 or grid <= 4:
        raise ValueError("N, M must be positive and grid must be > 4.")
    if omega_min <= 0.0 or omega_max <= omega_min:
        raise ValueError("Require 0 < omega_min < omega_max.")
    if n_sources_range[0] <= 0 or n_sources_range[0] > n_sources_range[1]:
        raise ValueError("n_sources_range must satisfy 0 < min_src <= max_src.")

    rng = np.random.default_rng(seed)
    out_path = Path(out)

    # ── Frequency grid ──────────────────────────────────────────────────────
    if omega_scale == "log":
        omegas = np.geomspace(omega_min, omega_max, M)
    else:
        omegas = np.linspace(omega_min, omega_max, M)

    # ── Solver ──────────────────────────────────────────────────────────────
    solver = HelmholtzSolver(N=grid, L=L, c=c, pml_width=pml_width, sigma_max=sigma_max)

    # Quality check: points per wavelength at the highest frequency.
    ppw_at_omega_max = (2.0 * np.pi * c * grid) / (omega_max * L)
    if ppw_at_omega_max < min_ppw:
        print(
            f"[WARN] Low PPW at omega_max: {ppw_at_omega_max:.2f} < {min_ppw:.2f}. "
            "Consider increasing grid or lowering omega_max."
        )

    # ── Pre-allocate storage ─────────────────────────────────────────────────
    max_sources = n_sources_range[1]
    src_pos_arr = np.full((N, max_sources, 2), np.nan, dtype=np.float64)
    src_amp_arr = np.full((N, max_sources, 2), np.nan, dtype=np.float64)
    # We'll store real & imag separately as float32 to save disk
    fields_real = np.zeros((N, M, grid, grid), dtype=np.float32)
    fields_imag = np.zeros((N, M, grid, grid), dtype=np.float32)
    source_fields = []

    # ── Generation loop ─────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Helmholtz 2D Dataset Generation")
    print(f"  N={N} samples  ×  M={M} frequencies  ×  grid={grid}²")
    print(f"  ω ∈ [{omega_min:.1f}, {omega_max:.1f}]  ({omega_scale} scale)")
    print(f"  Output → {out_path}")
    print(f"{'─'*60}\n")

    t0 = time.time()

    t_sample = time.time()
    for i in tqdm(range(N), desc="Sample source configs", unit="sample"):
        # Sample source configuration for this BC/IC
        K = rng.integers(n_sources_range[0], n_sources_range[1] + 1)
        positions, amplitudes = sample_sources(rng, L, pml_width, K)

        src_pos_arr[i, :K] = positions
        src_amp_arr[i, :K, 0] = amplitudes.real
        src_amp_arr[i, :K, 1] = amplitudes.imag

        # Build source field once (all frequencies share this BC).
        source_fields.append(solver.multi_source(positions, amplitudes, sigma=source_sigma))

    sample_elapsed = time.time() - t_sample

    # Solve by frequency so each omega reuses one matrix factorization.
    t_solve = time.time()
    for j, omega in tqdm(enumerate(omegas), total=M, desc="Frequencies", unit="freq"):
        solve_op = solver.factorize_for_omega(float(omega))
        for i in range(N):
            u = solver.solve_with_factorized(solve_op, source_field=source_fields[i])
            fields_real[i, j] = u.real.astype(np.float32)
            fields_imag[i, j] = u.imag.astype(np.float32)

    elapsed = time.time() - t0
    solve_elapsed = time.time() - t_solve
    print(f"\n  Solved {N * M} PDEs in {elapsed:.1f}s  "
          f"({elapsed / (N * M) * 1000:.1f} ms/solve)")
    print(f"  Source setup time: {sample_elapsed:.1f}s")
    print(f"  Linear solves time: {solve_elapsed:.1f}s")

    # ── Save to HDF5 ────────────────────────────────────────────────────────
    meta = dict(
        N=N, M=M, grid=grid, L=L, c=c,
        omega_min=omega_min, omega_max=omega_max,
        omega_scale=omega_scale,
        n_sources_range=list(n_sources_range),
        pml_width=pml_width, sigma_max=sigma_max,
        source_sigma=float(source_sigma),
        equation="laplace(u) + (omega/c)^2 u = -f",
        ppw_at_omega_max=round(float(ppw_at_omega_max), 3),
        min_ppw_recommendation=min_ppw,
        factorization_reuse_by_frequency=True,
        seed=seed,
        elapsed_seconds=round(elapsed, 2),
        sample_setup_seconds=round(sample_elapsed, 2),
        solve_seconds=round(solve_elapsed, 2),
    )

    source_fields_arr = np.stack(source_fields, axis=0).astype(np.complex64)
    source_fields_real = source_fields_arr.real.astype(np.float32)
    source_fields_imag = source_fields_arr.imag.astype(np.float32)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("metadata",    data=json.dumps(meta))
        f.create_dataset("omega",       data=omegas,           dtype=np.float64)
        f.create_dataset("sources",     data=src_pos_arr,      dtype=np.float64)
        f.create_dataset("amplitudes",  data=src_amp_arr,      dtype=np.float64)
        f.create_dataset("source_fields_real", data=source_fields_real,
                         dtype=np.float32, compression="gzip", compression_opts=4)
        f.create_dataset("source_fields_imag", data=source_fields_imag,
                         dtype=np.float32, compression="gzip", compression_opts=4)
        f.create_dataset("fields_real", data=fields_real,
                         dtype=np.float32, compression="gzip", compression_opts=4)
        f.create_dataset("fields_imag", data=fields_imag,
                         dtype=np.float32, compression="gzip", compression_opts=4)
        f.create_dataset("grid_x",      data=solver.x1d,       dtype=np.float64)
        f.create_dataset("grid_y",      data=solver.x1d,       dtype=np.float64)

        scale = 1.0

        if export_ftm:
            data_ftm = np.stack([fields_real, fields_imag], axis=-1).astype(np.float32)
            if normalize:
                max_abs = np.max(np.abs(data_ftm))
                scale = max(max_abs, 1e-8)
                data_ftm = data_ftm / scale
            else:
                scale = 1.0

            mask_tr = build_sparse_mask(
                rng=rng,
                N=N,
                M=M,
                grid=grid,
                channels=data_ftm.shape[-1],
                obs_ratio=obs_ratio,
                mask_mode=mask_mode,
            )

            f.create_dataset(
                "data",
                data=data_ftm,
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset(
                "mask_tr",
                data=mask_tr,
                dtype=np.uint8,
                compression="gzip",
                compression_opts=4,
            )

            meta_ftm = {
                "data": {
                    "u_ind_uni": solver.x1d.astype(np.float32),
                    "v_ind_uni": solver.x1d.astype(np.float32),
                    "w_ind_uni": np.arange(data_ftm.shape[-1], dtype=np.float32),
                    "t_ind_uni": omegas.astype(np.float32),
                    "data_scale": float(scale),
                    "obs_ratio": float(obs_ratio),
                    "mask_mode": mask_mode,
                    "mask_tr_shape": tuple(int(v) for v in mask_tr.shape),
                    "mask_shared_across_samples": bool(mask_tr.ndim == 4),
                    "normalize": bool(normalize),
                }
            }
            if mask_tr.ndim == 4:
                # Keep backward compatibility for small shared-mask case.
                meta_ftm["data"]["mask_tr"] = mask_tr.astype(np.float32)

            if metadata_out:
                meta_path = Path(metadata_out)
            else:
                meta_path = out_path.with_name(f"{out_path.stem}_metadata.npy")
            np.save(meta_path, meta_ftm, allow_pickle=True)
            print(f"  Saved  {meta_path}")

        f.create_dataset("data_scale", data=np.array(scale, dtype=np.float32))

    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved  {out_path}  ({size_mb:.1f} MB)\n")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2D Helmholtz dataset")
    parser.add_argument("--N",           type=int,   default=200)
    parser.add_argument("--M",           type=int,   default=20)
    parser.add_argument("--grid",        type=int,   default=128)
    parser.add_argument("--L",           type=float, default=1.0)
    parser.add_argument("--c",           type=float, default=1.0)
    parser.add_argument("--omega_min",   type=float, default=52.0)
    parser.add_argument("--omega_max",   type=float, default=71.0)
    parser.add_argument("--omega_scale", type=str,   default="linear",
                        choices=["linear", "log"])
    parser.add_argument("--min_src",     type=int,   default=1)
    parser.add_argument("--max_src",     type=int,   default=4)
    parser.add_argument("--pml_width",   type=float, default=0.12)
    parser.add_argument("--sigma_max",   type=float, default=50.0)
    parser.add_argument("--source_sigma", type=float, default=0.025)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--min_ppw",     type=float, default=8.0)
    parser.add_argument("--out",         type=str,   default="data_for_test/helmholtz_dataset_42_mask5_Extra.h5")
    parser.add_argument("--export_ftm",  action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metadata_out", type=str, default="")
    parser.add_argument("--obs_ratio",   type=float, default=0.05)
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="per_sample_fixed",
        choices=["fixed", "per_freq", "per_sample_fixed", "per_sample_per_freq", "per_sample"],
    )
    parser.add_argument("--disable_normalize", action="store_true")
    args = parser.parse_args()

    out_path = args.out.format(seed=args.seed)

    generate(
        N=args.N, M=args.M, grid=args.grid,
        L=args.L, c=args.c,
        omega_min=args.omega_min, omega_max=args.omega_max,
        omega_scale=args.omega_scale,
        n_sources_range=(args.min_src, args.max_src),
        pml_width=args.pml_width, sigma_max=args.sigma_max,
        source_sigma=args.source_sigma,
        seed=args.seed, out=out_path, min_ppw=args.min_ppw,
        export_ftm=args.export_ftm,
        metadata_out=args.metadata_out,
        obs_ratio=args.obs_ratio,
        mask_mode=args.mask_mode,
        normalize=not args.disable_normalize,
    )
