"""
Generate_dataset.py — 3D Helmholtz dataset
-------------------------------------------
Generates a frequency-domain 3D complex pressure field dataset for the pipeline:

    (∇²_PML + k²) u(x,y,z) = -f(x,y,z),   k = ω/c

Domain  : [0,L]³, uniform N×N×N grid.
PML     : Quadratic absorption on all 6 faces.
Sources : K Gaussian point sources with random positions & complex amplitudes
          (same "BC/IC diversity" role as in 2D Helmholtz).

Dataset format (HDF5)
---------------------
  /metadata   — JSON (generation params + per-sample sources)
  /omega      — (M,)            float64
  /grid_x     — (N,)            float64
  /grid_y     — (N,)            float64
  /grid_z     — (N,)            float64
  /data       — (B, M, N, N, N, 2)  float32  [real, imag]  (per-freq normalised)
  /mask_tr    — sparse obs mask
  /data_scale — ()               float32

Usage
-----
    python Generate_dataset.py                      # defaults (N=32, B=100)
    python Generate_dataset.py --N 24 --B 50 --n_jobs 4
    python Generate_dataset.py --N 32 --B 200 --freq_train 2 5 8 11 14 17 20 --freq_extrap 25 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from joblib import Parallel, delayed


# ─────────────────────────────────────────────────────────────────────────────
# 3D Helmholtz solver with PML
# ─────────────────────────────────────────────────────────────────────────────

def _pml_sigma(x: np.ndarray, L: float, pml_d: float, sigma_max: float) -> np.ndarray:
    s = np.zeros_like(x, dtype=float)
    s[x < pml_d]     = sigma_max * ((pml_d - x[x < pml_d]) / pml_d) ** 2
    s[x > L - pml_d] = sigma_max * ((x[x > L - pml_d] - (L - pml_d)) / pml_d) ** 2
    return s


def _build_1d_pml_op(sigma: np.ndarray, h: float, omega: float) -> sp.csc_matrix:
    """Build the 1-D PML-modified second-derivative operator diag(s)*D2 (N×N)."""
    N = len(sigma)
    # Complex stretch on interior faces: s_{i+1/2} = (s_i + s_{i+1})/2
    sx = 1.0 / (1.0 + 1j * sigma / omega).astype(np.complex128)
    sx_ph = 0.5 * (sx[:-1] + sx[1:])         # right-face values  (N-1,)
    sx_lpad = np.r_[sx[:1], sx_ph]            # left-face per node (N,)
    sx_rpad = np.r_[sx_ph, sx[-1:]]           # right-face per node
    d0 = -(sx_lpad + sx_rpad) / h**2
    dp =  sx_ph / h**2
    return sp.diags([dp, d0, dp], [-1, 0, 1], shape=(N, N),
                    format="csc", dtype=np.complex128)


class Helmholtz3DSolver:
    """
    Sparse FD solver for 3D Helmholtz + PML.

    System: (Ax⊗Iy⊗Iz + Ix⊗Ay⊗Iz + Ix⊗Iy⊗Az + k²I) u = -f
    Field u flattened as u[ix, iy, iz] → u_flat[ix*N² + iy*N + iz].
    """

    def __init__(
        self,
        N: int = 32,
        L: float = 1.0,
        c: float = 1.0,
        pml_frac: float = 0.15,
        sigma_max: float = 40.0,
        source_sigma: float = 0.04,
    ):
        self.N, self.L, self.c = N, L, c
        self.h = L / N
        self.pml_d = pml_frac * L
        self.sigma_max = sigma_max
        self.source_sigma = source_sigma

        x1d = np.linspace(self.h / 2, L - self.h / 2, N)
        self.x1d = x1d
        self.sigma = _pml_sigma(x1d, L, self.pml_d, sigma_max)  # same for all dims

    # ------------------------------------------------------------------
    def _build_system(self, omega: float) -> sp.csc_matrix:
        N, h = self.N, self.h
        Ax = _build_1d_pml_op(self.sigma, h, omega)
        Ay = Ax.copy()   # uniform medium → same operator in every direction
        Az = Ax.copy()

        I  = sp.eye(N, format="csc", dtype=np.complex128)
        A  = (sp.kron(sp.kron(Ax, I), I) +
              sp.kron(sp.kron(I, Ay), I) +
              sp.kron(sp.kron(I, I), Az))

        # k² diagonal
        k2 = (omega / self.c) ** 2
        A  = A + k2 * sp.eye(N**3, format="csc", dtype=np.complex128)

        # Dirichlet BC: zero on all 6 boundary faces
        A = A.tolil()
        for ix in range(N):
            for iy in range(N):
                for iz_b in [0, N - 1]:
                    r = ix * N * N + iy * N + iz_b
                    A[r, :] = 0; A[r, r] = 1.0
        for ix in range(N):
            for iz in range(N):
                for iy_b in [0, N - 1]:
                    r = ix * N * N + iy_b * N + iz
                    A[r, :] = 0; A[r, r] = 1.0
        for iy in range(N):
            for iz in range(N):
                for ix_b in [0, N - 1]:
                    r = ix_b * N * N + iy * N + iz
                    A[r, :] = 0; A[r, r] = 1.0
        return A.tocsc()

    def factorize(self, omega: float):
        """Pre-factorize the system matrix for this ω (reuse for many RHS)."""
        A = self._build_system(omega)
        return spla.factorized(A)

    def make_source(self, positions_xyz: np.ndarray,
                    amplitudes: np.ndarray) -> np.ndarray:
        """Build 3D Gaussian source field (N,N,N), complex."""
        N, h, s = self.N, self.h, self.source_sigma
        xs = self.x1d
        f = np.zeros((N, N, N), dtype=np.complex128)
        X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
        for (x0, y0, z0), amp in zip(positions_xyz, amplitudes):
            r2 = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2
            f += amp * np.exp(-r2 / (2 * s**2))
        # zero at boundary faces (Dirichlet)
        f[[0, -1], :, :] = 0; f[:, [0, -1], :] = 0; f[:, :, [0, -1]] = 0
        return f

    def solve_with_factorized(self, solve_op, positions_xyz: np.ndarray,
                               amplitudes: np.ndarray) -> np.ndarray:
        """Solve for a given source configuration using a pre-factorized operator."""
        N = self.N
        f = self.make_source(positions_xyz, amplitudes)
        u_flat = solve_op(-f.ravel())
        return u_flat.reshape(N, N, N).astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# Sparse mask builder
# ─────────────────────────────────────────────────────────────────────────────

def build_sparse_mask(
    rng: np.random.Generator,
    B: int, M: int, N: int,
    obs_ratio: float,
    mask_mode: str,
) -> np.ndarray:
    """mask_tr shape: (M,N,N,N,2) or (B,M,N,N,N,2)."""
    def _spatial(shape):
        return (rng.random(shape) < obs_ratio).astype(np.uint8)

    if mask_mode == "fixed":
        sp = _spatial((N, N, N))
        mask = np.broadcast_to(sp[None, :, :, :, None],
                               (M, N, N, N, 1)).copy()
    elif mask_mode == "per_freq":
        mask = _spatial((M, N, N, N, 1))
    elif mask_mode == "per_sample_fixed":
        sp = _spatial((B, N, N, N, 1))
        mask = np.repeat(sp[:, None, :, :, :, :], M, axis=1)
    else:
        mask = _spatial((B, M, N, N, N, 1))

    return np.repeat(mask, 2, axis=-1).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample worker
# ─────────────────────────────────────────────────────────────────────────────

def _compute_one_sample(
    sample_id: int,
    solver: Helmholtz3DSolver,
    freq_list: List[float],
    seed: int,
    k_min: int,
    k_max: int,
    pml_frac: float,
) -> dict:
    rng  = np.random.default_rng(seed + sample_id * 1000 + 7)
    N, L = solver.N, solver.L
    K    = int(rng.integers(k_min, k_max + 1))

    # Source positions: keep away from PML (inner 70% of domain)
    lo, hi = pml_frac * L + 0.05 * L, (1 - pml_frac) * L - 0.05 * L
    pos  = rng.uniform(lo, hi, (K, 3))
    amps = np.exp(1j * rng.uniform(0, 2 * np.pi, K))

    M = len(freq_list)
    u_real = np.zeros((M, N, N, N), dtype=np.float32)
    u_imag = np.zeros((M, N, N, N), dtype=np.float32)

    for j, omega in enumerate(freq_list):
        solve_op = solver.factorize(float(omega))
        u = solver.solve_with_factorized(solve_op, pos, amps)
        u_real[j] = u.real
        u_imag[j] = u.imag

    return {
        "sample_id": sample_id,
        "u_real": u_real,
        "u_imag": u_imag,
        "positions": pos,
        "amplitudes": np.stack([amps.real, amps.imag], axis=-1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main generation routine
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    B: int,
    freq_train: Sequence[float],
    freq_extrap: Sequence[float],
    N: int,
    L: float,
    c: float,
    pml_frac: float,
    sigma_max: float,
    source_sigma: float,
    k_min: int,
    k_max: int,
    obs_ratio: float,
    mask_mode: str,
    out: str,
    seed: int,
    n_jobs: int,
    normalize: bool,
    metadata_out: str,
) -> Path:
    freq_list = [float(f) for f in list(freq_train) + list(freq_extrap)]
    M = len(freq_list)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    solver = Helmholtz3DSolver(N=N, L=L, c=c, pml_frac=pml_frac,
                               sigma_max=sigma_max, source_sigma=source_sigma)

    print(f"\n{'─'*60}")
    print(f"  3D Helmholtz Dataset  N={N}³ grid  B={B}  M={M}")
    print(f"  freq_train={list(freq_train)}  extrap={list(freq_extrap)}")
    print(f"  c={c}  pml={pml_frac:.2f}  σ_max={sigma_max}  K=[{k_min},{k_max}]")
    print(f"  obs={obs_ratio}  mask={mask_mode}  → {out_path}")
    print(f"{'─'*60}\n")

    t0 = time.time()

    if n_jobs == 1:
        results = []
        for sid in range(B):
            r = _compute_one_sample(sid, solver, freq_list, seed,
                                    k_min, k_max, pml_frac)
            print(f"  [{sid+1:>4}/{B}] sources={len(r['positions'])}", flush=True)
            results.append(r)
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_compute_one_sample)(sid, solver, freq_list, seed,
                                         k_min, k_max, pml_frac)
            for sid in range(B)
        )
        results.sort(key=lambda r: r["sample_id"])

    elapsed = time.time() - t0
    print(f"\n  Solved {B*M} PDEs in {elapsed:.1f}s ({elapsed/B/M*1000:.0f}ms/solve)")

    # ── Assemble (B, M, N, N, N, 2) ─────────────────────────────────────
    u_real = np.stack([r["u_real"] for r in results], axis=0)  # (B,M,N,N,N)
    u_imag = np.stack([r["u_imag"] for r in results], axis=0)
    data   = np.stack([u_real, u_imag], axis=-1).astype(np.float32)  # (B,M,N,N,N,2)

    # Per-frequency normalisation
    scale = 1.0
    if normalize:
        mx = np.max(np.abs(data), axis=(0, 2, 3, 4, 5), keepdims=True)  # (1,M,1,1,1,1)
        mx = np.maximum(mx, 1e-12)
        data = (data / mx).astype(np.float32)

    x1d = solver.x1d.astype(np.float64)

    # Sparse mask
    rng_mask = np.random.default_rng(seed + 999_977)
    mask_tr  = build_sparse_mask(rng_mask, B, M, N, obs_ratio, mask_mode)

    # Metadata
    samples_meta = [
        {"sample_id": r["sample_id"],
         "n_sources": int(len(r["positions"]))}
        for r in results
    ]
    meta = dict(
        equation="3d_helmholtz", channels=["real", "imag"],
        B=B, M=M, N=N, L=L, c=c,
        freq_train=list(freq_train), freq_extrap=list(freq_extrap),
        pml_frac=pml_frac, sigma_max=sigma_max,
        obs_ratio=obs_ratio, mask_mode=mask_mode,
        normalize=normalize, data_scale=float(scale),
        seed=seed, elapsed_s=round(elapsed, 2),
        samples=samples_meta,
    )

    # ── Write HDF5 ──────────────────────────────────────────────────────
    gz = {"compression": "gzip", "compression_opts": 4}
    with h5py.File(out_path, "w") as f:
        f.create_dataset("metadata",   data=json.dumps(meta))
        f.create_dataset("omega",      data=np.array(freq_list, dtype=np.float64))
        f.create_dataset("grid_x",     data=x1d)
        f.create_dataset("grid_y",     data=x1d)
        f.create_dataset("grid_z",     data=x1d)
        f.create_dataset("data",       data=data,    dtype=np.float32, **gz)
        f.create_dataset("mask_tr",    data=mask_tr, dtype=np.uint8,   **gz)
        f.create_dataset("data_scale", data=np.array(scale, dtype=np.float32))

    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved {out_path} ({size_mb:.1f} MB)")

    # ── Sidecar .npy (same convention as 2D) ───────────────────────────
    meta_ftm = {"data": {
        "u_ind_uni": x1d.astype(np.float32),
        "v_ind_uni": x1d.astype(np.float32),
        "w_ind_uni": x1d.astype(np.float32),
        "t_ind_uni": np.array(freq_list, dtype=np.float32),
        "data_scale": float(scale), "obs_ratio": float(obs_ratio),
        "mask_mode": mask_mode,
        "mask_tr_shape": tuple(int(v) for v in mask_tr.shape),
        "mask_shared_across_samples": bool(mask_tr.ndim == 5),
        "normalize": bool(normalize),
        "channels": ["real", "imag"],
        "spatial_dims": 3,
    }}
    if mask_tr.ndim == 5:
        meta_ftm["data"]["mask_tr"] = mask_tr.astype(np.float32)
    meta_path = Path(metadata_out) if metadata_out else \
        out_path.with_name(f"{out_path.stem}_metadata.npy")
    np.save(meta_path, meta_ftm, allow_pickle=True)
    print(f"  Saved {meta_path}\n")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate 3D Helmholtz dataset (HDF5)")
    p.add_argument("--B",           type=int,   default=10)
    p.add_argument("--N",           type=int,   default=32)
    p.add_argument("--L",           type=float, default=1.0)
    p.add_argument("--c",           type=float, default=1.0)
    p.add_argument("--pml_frac",    type=float, default=0.15)
    p.add_argument("--sigma_max",   type=float, default=40.0)
    p.add_argument("--source_sigma",type=float, default=0.04)
    p.add_argument("--k_min",       type=int,   default=1)
    p.add_argument("--k_max",       type=int,   default=3)
    p.add_argument("--freq_train",  type=float, nargs="+",
                   default=[2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14])
    p.add_argument("--freq_extrap", type=float, nargs="*",
                   default=[14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22])
    p.add_argument("--obs_ratio",   type=float, default=0.01)
    p.add_argument("--mask_mode",   type=str,   default="per_sample_fixed",
                   choices=["fixed", "per_freq", "per_sample_fixed", "per_sample_per_freq"])
    p.add_argument("--out",         type=str,   default="helmholtz3d_dataset_msk0.01.h5")
    p.add_argument("--metadata_out",type=str,   default="")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--n_jobs",      type=int,   default=16)
    p.add_argument("--no_normalize",action="store_true")
    a = p.parse_args()
    generate(
        B=a.B, freq_train=a.freq_train, freq_extrap=a.freq_extrap,
        N=a.N, L=a.L, c=a.c, pml_frac=a.pml_frac, sigma_max=a.sigma_max,
        source_sigma=a.source_sigma, k_min=a.k_min, k_max=a.k_max,
        obs_ratio=a.obs_ratio, mask_mode=a.mask_mode, out=a.out,
        seed=a.seed, n_jobs=a.n_jobs, normalize=not a.no_normalize,
        metadata_out=a.metadata_out,
    )
