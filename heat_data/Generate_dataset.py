from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np

from Heat_Solver import HarmonicHeatSolver


DATASET_VERSION = "harmonic_heat_v1"


def build_shared_mask(
    n_freq: int,
    n_grid: int,
    mask_ratio: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = max(1, int(round(mask_ratio * n_grid * n_grid)))
    flat = np.zeros(n_grid * n_grid, dtype=np.float32)
    flat[rng.choice(n_grid * n_grid, size=keep, replace=False)] = 1.0
    mask = flat.reshape(n_grid, n_grid)
    mask = np.stack([mask, mask], axis=-1)
    mask = np.repeat(mask[None, ...], n_freq, axis=0)
    return mask.astype(np.float32)


def sample_sources(
    rng: np.random.Generator,
    n_sources: int,
    domain_l: float,
) -> Tuple[np.ndarray, np.ndarray]:
    positions = rng.uniform(0.15 * domain_l, 0.85 * domain_l, size=(n_sources, 2))
    amp_re = rng.uniform(0.5, 1.5, size=n_sources)
    amp_im = rng.uniform(-0.5, 0.5, size=n_sources)
    amplitudes = np.stack([amp_re, amp_im], axis=-1)
    return positions.astype(np.float64), amplitudes.astype(np.float64)


def generate_dataset(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)

    grid = np.linspace(0.0, args.L, args.N, dtype=np.float64)
    omega = np.linspace(args.omega_min, args.omega_max, args.n_freq, dtype=np.float64)
    mask_tr = build_shared_mask(args.n_freq, args.N, args.mask_ratio, args.seed + 17)

    solver = HarmonicHeatSolver(
        N=args.N,
        L=args.L,
        diffusivity=args.diffusivity,
        capacity=args.capacity,
    )

    data = np.zeros((args.n_samples, args.n_freq, args.N, args.N, 2), dtype=np.float32)
    source_fields_real = np.zeros((args.n_samples, args.N, args.N), dtype=np.float32)
    source_fields_imag = np.zeros((args.n_samples, args.N, args.N), dtype=np.float32)
    sources = np.full((args.n_samples, args.max_sources, 2), np.nan, dtype=np.float64)
    amplitudes = np.full((args.n_samples, args.max_sources, 2), np.nan, dtype=np.float64)
    num_sources = np.zeros((args.n_samples,), dtype=np.int32)

    for b in range(args.n_samples):
        k = int(rng.integers(args.min_sources, args.max_sources + 1))
        pos, amp = sample_sources(rng, k, args.L)
        amp_c = amp[:, 0] + 1j * amp[:, 1]
        src = solver.multi_source(pos, amp_c, sigma=args.source_sigma)

        source_fields_real[b] = src.real.astype(np.float32)
        source_fields_imag[b] = src.imag.astype(np.float32)
        sources[b, :k] = pos
        amplitudes[b, :k] = amp
        num_sources[b] = k

        for m, w in enumerate(omega):
            sol = solver.solve(float(w), src)
            data[b, m, ..., 0] = sol.real.astype(np.float32)
            data[b, m, ..., 1] = sol.imag.astype(np.float32)

        if (b + 1) % max(1, args.log_every) == 0 or (b + 1) == args.n_samples:
            print(f"generated {b + 1}/{args.n_samples} samples")

    data_scale = float(np.max(np.abs(data)))
    if data_scale <= 0.0:
        data_scale = 1.0

    if args.normalize_data:
        data = data / data_scale
        source_fields_real = source_fields_real / data_scale
        source_fields_imag = source_fields_imag / data_scale

    out_path = Path(args.out_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "version": DATASET_VERSION,
        "equation": "-diffusivity*Delta(u) + i*omega*capacity*u = f",
        "N": int(args.N),
        "L": float(args.L),
        "n_samples": int(args.n_samples),
        "n_freq": int(args.n_freq),
        "omega_min": float(args.omega_min),
        "omega_max": float(args.omega_max),
        "diffusivity": float(args.diffusivity),
        "capacity": float(args.capacity),
        "source_sigma": float(args.source_sigma),
        "mask_ratio": float(args.mask_ratio),
        "normalize_data": bool(args.normalize_data),
        "data_scale": float(data_scale),
    }

    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=data, compression="gzip")
        f.create_dataset("mask_tr", data=mask_tr, compression="gzip")
        f.create_dataset("omega", data=omega)
        f.create_dataset("grid_x", data=grid)
        f.create_dataset("grid_y", data=grid)
        f.create_dataset("source_fields_real", data=source_fields_real, compression="gzip")
        f.create_dataset("source_fields_imag", data=source_fields_imag, compression="gzip")
        f.create_dataset("sources", data=sources)
        f.create_dataset("amplitudes", data=amplitudes)
        f.create_dataset("num_sources", data=num_sources)
        f.create_dataset("data_scale", data=np.asarray(data_scale, dtype=np.float32))
        f.create_dataset("metadata", data=np.bytes_(json.dumps(metadata, ensure_ascii=False)))

    meta_sidecar = out_path.with_name(f"{out_path.stem}_metadata.npy")
    np.save(meta_sidecar, {"data": metadata}, allow_pickle=True)

    print("saved dataset:", out_path)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate harmonic heat equation dataset")
    p.add_argument("--out_h5", type=str, default="heat_data/data_for_test/harmonic_heat_dataset_mask10.h5")
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--L", type=float, default=1.0)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--n_freq", type=int, default=20)
    p.add_argument("--omega_min", type=float, default=1.0)
    p.add_argument("--omega_max", type=float, default=20.0)
    p.add_argument("--diffusivity", type=float, default=1.0)
    p.add_argument("--capacity", type=float, default=1.0)
    p.add_argument("--min_sources", type=int, default=1)
    p.add_argument("--max_sources", type=int, default=3)
    p.add_argument("--source_sigma", type=float, default=0.04)
    p.add_argument("--mask_ratio", type=float, default=0.10)
    p.add_argument("--normalize_data", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    return p


def main() -> None:
    args = build_parser().parse_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()
