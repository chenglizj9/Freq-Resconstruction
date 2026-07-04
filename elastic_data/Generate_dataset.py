"""
generate_elastic_dataset.py
---------------------------
Generate 2D frequency-domain elastic wave dataset and save in HDF5 format.

Dataset layout (HDF5)
---------------------
/metadata            — JSON string with all generation parameters
/omega               — (M,)            float64  — frequency values
/grid_x              — (Ng,)           float64  — 1-D x coordinates
/grid_y              — (Ng,)           float64  — 1-D y coordinates

/obstacle_mask       — (N, Ng, Ng)     float32  — per-sample obstacle masks (0/1)
/lambda_field        — (N, Ng, Ng)     float32  — Lamé λ fields
/mu_field            — (N, Ng, Ng)     float32  — Lamé μ fields
/rho_field           — (N, Ng, Ng)     float32  — density ρ fields

/ux_real             — (N, M, Ng, Ng)  float32  — Re(ux)
/ux_imag             — (N, M, Ng, Ng)  float32  — Im(ux)
/uy_real             — (N, M, Ng, Ng)  float32  — Re(uy)
/uy_imag             — (N, M, Ng, Ng)  float32  — Im(uy)

/data                — (N, M, Ng, Ng, 4) float32 — FTM-ready: [ux_re, ux_im, uy_re, uy_im]
                       (optionally normalised by data_scale)
/mask_tr             — sparse observation mask (see mask_mode)
                         shared modes:     (M, Ng, Ng, 4) uint8
                         per-sample modes: (N, M, Ng, Ng, 4) uint8
/data_scale          — () float32 — global normalisation factor (1.0 if disabled)

Usage
-----
    python generate_elastic_dataset.py                            # defaults
    python generate_elastic_dataset.py --N 200 --grid 128 --out elastic_data.h5
    python generate_elastic_dataset.py --obs_ratio 0.05 --mask_mode per_sample_fixed
    python generate_elastic_dataset.py --no_mask --no_fix_lambda --no_fix_mu --no_fix_rho
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from joblib import Parallel, delayed
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Elastic Wave Solver
# ─────────────────────────────────────────────────────────────────────────────

class ElasticWaveSolver:
    def __init__(
        self,
        N: int = 128,
        L: float = 1.0,
        pml_width: int = 10,
        sigma_max: float = 60.0,
        linear_solver: str = "gmres",
        gmres_rtol: float = 1e-6,
        gmres_atol: float = 1e-8,
        gmres_restart: int = 200,
        gmres_maxiter: int | None = 500,
    ):
        self.N = N
        self.L = L
        self.dx = L / N
        self.pml_width = pml_width
        self.shape = (N, N)
        self.num_points = N * N
        self.grid_x, self.grid_y = np.meshgrid(
            np.linspace(0.0, L, N),
            np.linspace(0.0, L, N),
        )
        self.x1d = np.linspace(0.0, L, N)
        self.sigma_max = float(sigma_max)
        self.linear_solver = linear_solver.lower()
        self.gmres_rtol = float(gmres_rtol)
        self.gmres_atol = float(gmres_atol)
        self.gmres_restart = int(gmres_restart)
        self.gmres_maxiter = gmres_maxiter
        self.Dx, self.Dy = self._build_first_derivative_operators()

        if self.linear_solver not in {"gmres", "spsolve"}:
            raise ValueError("linear_solver must be one of {'gmres', 'spsolve'}")

    def _build_first_derivative_1d(self) -> sp.csr_matrix:
        derivative = sp.lil_matrix((self.N, self.N), dtype=np.float32)
        inv_dx = 1.0 / self.dx
        derivative[0, 0] = -inv_dx
        derivative[0, 1] = inv_dx
        derivative[-1, -2] = -inv_dx
        derivative[-1, -1] = inv_dx
        half_inv_dx = 0.5 * inv_dx
        for idx in range(1, self.N - 1):
            derivative[idx, idx - 1] = -half_inv_dx
            derivative[idx, idx + 1] = half_inv_dx
        return derivative.tocsr()

    def _build_first_derivative_operators(self) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        derivative_1d = self._build_first_derivative_1d()
        eye = sp.eye(self.N, format="csr")
        dx_op = sp.kron(eye, derivative_1d, format="csr")
        dy_op = sp.kron(derivative_1d, eye, format="csr")
        return dx_op, dy_op

    def _create_pml_mask(self) -> np.ndarray:
        mask = np.zeros((self.N, self.N), dtype=np.float32)
        width = self.pml_width
        for idx in range(width):
            value = ((width - idx) / width) ** 2
            mask[idx, :] = np.maximum(mask[idx, :], value)
            mask[-(idx + 1), :] = np.maximum(mask[-(idx + 1), :], value)
            mask[:, idx] = np.maximum(mask[:, idx], value)
            mask[:, -(idx + 1)] = np.maximum(mask[:, -(idx + 1)], value)
        return mask

    def generate_mask(
        self,
        rng: np.random.Generator,
        num_circles: int = 3,
        min_r: float = 0.05,
        max_r: float = 0.15,
        safe_dist: float = 0.08,
    ) -> np.ndarray:
        x_src = (self.N // 6) * self.dx
        y_src = (self.N // 2) * self.dx
        mask = np.zeros(self.shape, dtype=np.float32)
        for _ in range(num_circles):
            placed = False
            while not placed:
                cx = rng.uniform(0.2 * self.L, 0.8 * self.L)
                cy = rng.uniform(0.2 * self.L, 0.8 * self.L)
                radius = rng.uniform(min_r, max_r)
                dist = np.sqrt((cx - x_src) ** 2 + (cy - y_src) ** 2)
                if dist <= radius + safe_dist:
                    continue
                placed = True
                distance = np.sqrt((self.grid_x - cx) ** 2 + (self.grid_y - cy) ** 2)
                mask[distance < radius] = 1.0
        return mask

    def _generate_smooth_field(
        self,
        rng: np.random.Generator,
        value_range: Tuple[float, float],
        num_blobs_range: Tuple[int, int],
    ) -> np.ndarray:
        field = np.zeros(self.shape, dtype=np.float32)
        num_blobs = int(rng.integers(num_blobs_range[0], num_blobs_range[1] + 1))
        for _ in range(num_blobs):
            amplitude = rng.uniform(-1.0, 1.0)
            cx = rng.uniform(0.1 * self.L, 0.9 * self.L)
            cy = rng.uniform(0.1 * self.L, 0.9 * self.L)
            sx = rng.uniform(0.06 * self.L, 0.20 * self.L)
            sy = rng.uniform(0.06 * self.L, 0.20 * self.L)
            exponent = ((self.grid_x - cx) ** 2) / (2.0 * sx ** 2) + \
                       ((self.grid_y - cy) ** 2) / (2.0 * sy ** 2)
            field += amplitude * np.exp(-exponent).astype(np.float32)
        field_min, field_max = float(field.min()), float(field.max())
        if field_max - field_min < 1e-8:
            normalized = np.full(self.shape, 0.5, dtype=np.float32)
        else:
            normalized = (field - field_min) / (field_max - field_min)
        low, high = value_range
        return (low + (high - low) * normalized).astype(np.float32)

    def generate_material_fields(
        self,
        rng: np.random.Generator,
        lambda_range: Tuple[float, float],
        mu_range: Tuple[float, float],
        rho_range: Tuple[float, float],
    ) -> Dict[str, np.ndarray]:
        return {
            "lambda": self._generate_smooth_field(rng, lambda_range, (4, 6)),
            "mu":     self._generate_smooth_field(rng, mu_range,     (4, 6)),
            "rho":    self._generate_smooth_field(rng, rho_range,    (3, 6)),
        }

    def solve(
        self,
        omega: float,
        lambda_field: np.ndarray,
        mu_field: np.ndarray,
        rho_field: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> Dict[str, np.ndarray]:
        if mask is None:
            mask = np.zeros(self.shape, dtype=np.float32)

        flat_lambda = lambda_field.reshape(-1).astype(np.complex64)
        flat_mu     = mu_field.reshape(-1).astype(np.complex64)
        flat_rho    = rho_field.reshape(-1).astype(np.complex64)
        flat_mask   = mask.reshape(-1) > 0

        lam_d   = sp.diags(flat_lambda,                    format="csr")
        mu_d    = sp.diags(flat_mu,                        format="csr")
        lp2mu_d = sp.diags(flat_lambda + 2.0 * flat_mu,   format="csr")

        pml = self._create_pml_mask().reshape(-1)
        mass_d = sp.diags(
            flat_rho * (omega ** 2) + 1j * omega * self.sigma_max * pml,
            format="csr",
        )

        a11 = self.Dx @ lp2mu_d @ self.Dx + self.Dy @ mu_d    @ self.Dy + mass_d
        a12 = self.Dx @ lam_d   @ self.Dy + self.Dy @ mu_d    @ self.Dx
        a21 = self.Dx @ mu_d    @ self.Dy + self.Dy @ lam_d   @ self.Dx
        a22 = self.Dx @ mu_d    @ self.Dx + self.Dy @ lp2mu_d @ self.Dy + mass_d

        system = sp.bmat([[a11, a12], [a21, a22]], format="lil", dtype=np.complex64)

        rhs_x = np.zeros(self.num_points, dtype=np.complex64)
        rhs_y = np.zeros(self.num_points, dtype=np.complex64)
        src_idx = (self.N // 2) * self.N + self.N // 6
        rhs_x[src_idx] = np.complex64(1000.0)
        rhs = np.concatenate([rhs_x, rhs_y])

        obstacle_idx = np.flatnonzero(flat_mask)
        if obstacle_idx.size > 0:
            all_idx = np.concatenate([obstacle_idx, obstacle_idx + self.num_points])
            system[all_idx, :] = 0.0
            system[all_idx, all_idx] = 1.0
            rhs[all_idx] = 0.0

        solution = self._solve_linear_system(system, rhs)
        ux = solution[: self.num_points].reshape(self.shape).astype(np.complex64)
        uy = solution[self.num_points :].reshape(self.shape).astype(np.complex64)

        if obstacle_idx.size > 0:
            ux[mask > 0] = 0.0
            uy[mask > 0] = 0.0

        return {"ux": ux, "uy": uy}

    def _solve_linear_system(self, system: sp.spmatrix, rhs: np.ndarray) -> np.ndarray:
        system_csr = system.tocsr()
        if self.linear_solver == "spsolve":
            return spla.spsolve(system_csr, rhs)
        solution, info = spla.gmres(
            system_csr, rhs,
            rtol=self.gmres_rtol, atol=self.gmres_atol,
            restart=self.gmres_restart, maxiter=self.gmres_maxiter,
        )
        if info != 0:
            raise RuntimeError(f"GMRES failed to converge, info={info}")
        return solution


# ─────────────────────────────────────────────────────────────────────────────
# Sparse observation mask builder  (identical API to Helmholtz version)
# ─────────────────────────────────────────────────────────────────────────────

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

    channels = 4  (ux_real, ux_imag, uy_real, uy_imag)

    Returns
    -------
    mask_tr :
      shared-mask modes:       (M, grid, grid, channels) uint8
      per-sample-mask modes:   (N, M, grid, grid, channels) uint8
    """
    if not (0.0 < obs_ratio <= 1.0):
        raise ValueError("obs_ratio must be in (0, 1].")

    if mask_mode == "fixed":
        spatial = (rng.random((grid, grid)) < obs_ratio).astype(np.uint8)
        mask = np.repeat(spatial[None, :, :, None], M, axis=0)          # (M,g,g,1)
    elif mask_mode == "per_freq":
        mask = (rng.random((M, grid, grid, 1)) < obs_ratio).astype(np.uint8)
    elif mask_mode == "per_sample_fixed":
        spatial = (rng.random((N, grid, grid, 1)) < obs_ratio).astype(np.uint8)
        mask = np.repeat(spatial[:, None, :, :, :], M, axis=1)          # (N,M,g,g,1)
    elif mask_mode in {"per_sample", "per_sample_per_freq"}:
        mask = (rng.random((N, M, grid, grid, 1)) < obs_ratio).astype(np.uint8)
    else:
        raise ValueError(
            "mask_mode must be one of: fixed, per_freq, "
            "per_sample_fixed, per_sample_per_freq"
        )

    return np.repeat(mask, channels, axis=-1).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample worker  (returns arrays; no disk I/O)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_one_sample(
    sample_id: int,
    solver: ElasticWaveSolver,
    freq_list: List[float],
    seed: int,
    use_mask: bool,
    lambda_range: Tuple[float, float],
    mu_range: Tuple[float, float],
    rho_range: Tuple[float, float],
    fixed_materials: Dict[str, np.ndarray] | None,
) -> Dict:
    """
    Solve all frequencies for one sample.
    Returns a dict with material fields, mask, and wave fields for every ω.
    """
    rng = np.random.default_rng(seed + int(sample_id))
    M   = len(freq_list)
    Ng  = solver.N

    # ── obstacle mask ──────────────────────────────────────────────────────
    obstacle_mask = np.zeros(solver.shape, dtype=np.float32)
    if use_mask:
        num_circles = int(rng.integers(2, 6))
        obstacle_mask = solver.generate_mask(
            rng=rng, num_circles=num_circles, min_r=0.05, max_r=0.12
        )

    # ── material fields ────────────────────────────────────────────────────
    random_mats = solver.generate_material_fields(
        rng=rng,
        lambda_range=lambda_range,
        mu_range=mu_range,
        rho_range=rho_range,
    )
    if fixed_materials is None:
        materials = random_mats
    else:
        materials = {
            k: (fixed_materials[k].copy() if fixed_materials.get(k) is not None
                else random_mats[k])
            for k in ("lambda", "mu", "rho")
        }

    # ── solve ──────────────────────────────────────────────────────────────
    ux_real = np.zeros((M, Ng, Ng), dtype=np.float32)
    ux_imag = np.zeros((M, Ng, Ng), dtype=np.float32)
    uy_real = np.zeros((M, Ng, Ng), dtype=np.float32)
    uy_imag = np.zeros((M, Ng, Ng), dtype=np.float32)

    for j, omega in enumerate(freq_list):
        fields = solver.solve(
            omega=float(omega),
            lambda_field=materials["lambda"],
            mu_field=materials["mu"],
            rho_field=materials["rho"],
            mask=obstacle_mask if use_mask else None,
        )
        ux_real[j] = fields["ux"].real.astype(np.float32)
        ux_imag[j] = fields["ux"].imag.astype(np.float32)
        uy_real[j] = fields["uy"].real.astype(np.float32)
        uy_imag[j] = fields["uy"].imag.astype(np.float32)

    return {
        "obstacle_mask": obstacle_mask,
        "lambda":        materials["lambda"],
        "mu":            materials["mu"],
        "rho":           materials["rho"],
        "ux_real":       ux_real,
        "ux_imag":       ux_imag,
        "uy_real":       uy_real,
        "uy_imag":       uy_imag,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main generation routine
# ─────────────────────────────────────────────────────────────────────────────

def generate_elastic_dataset(
    num_samples: int = 1000,
    freq_train: Sequence[float] = (10, 20, 30, 40, 50, 60, 70, 80, 90),
    freq_extrap: Sequence[float] = (100, 120, 150, 180),
    N: int = 128,
    L: float = 1.0,
    n_jobs: int = -1,
    out: str = "elastic_dataset.h5",
    seed: int = 0,
    use_mask: bool = True,
    lambda_range: Tuple[float, float] = (0.5, 2.8),
    mu_range:     Tuple[float, float] = (0.5, 2.2),
    rho_range:    Tuple[float, float] = (0.8, 2.5),
    fix_lambda: bool = True,
    fix_mu:     bool = True,
    fix_rho:    bool = True,
    sigma_max:  float = 60.0,
    linear_solver: str = "gmres",
    gmres_rtol:    float = 1e-6,
    gmres_atol:    float = 1e-8,
    gmres_restart: int   = 200,
    gmres_maxiter: int | None = 500,
    # FTM / sparse-mask options
    export_ftm: bool  = True,
    normalize:  bool  = True,
    obs_ratio:  float = 0.1,
    mask_mode:  str   = "per_sample_fixed",
    metadata_out: str = "",
) -> Path:
    """Generate elastic wave dataset and save as a single HDF5 file."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    freq_list: List[float] = list(freq_train) + list(freq_extrap)
    M  = len(freq_list)
    Ng = N

    # ── Solver ──────────────────────────────────────────────────────────────
    solver = ElasticWaveSolver(
        N=N, L=L, pml_width=12, sigma_max=sigma_max,
        linear_solver=linear_solver,
        gmres_rtol=gmres_rtol, gmres_atol=gmres_atol,
        gmres_restart=gmres_restart, gmres_maxiter=gmres_maxiter,
    )

    # ── Fixed material fields (shared across all samples if requested) ──────
    material_rng = np.random.default_rng(seed)
    fixed_materials: Dict[str, np.ndarray] = {}
    if fix_lambda:
        fixed_materials["lambda"] = solver._generate_smooth_field(
            material_rng, lambda_range, (4, 8))
    if fix_mu:
        fixed_materials["mu"]     = solver._generate_smooth_field(
            material_rng, mu_range,     (4, 8))
    if fix_rho:
        fixed_materials["rho"]    = solver._generate_smooth_field(
            material_rng, rho_range,    (3, 7))

    print(f"\n{'─'*60}")
    print(f"  Elastic Wave 2-D Dataset Generation")
    print(f"  N={num_samples} samples  ×  M={M} frequencies  ×  grid={Ng}²")
    print(f"  ω list (train): {list(freq_train)}")
    print(f"  ω list (extrap): {list(freq_extrap)}")
    print(f"  Output → {out_path}")
    print(f"{'─'*60}\n")

    t0 = time.time()

    # ── Parallel solve ───────────────────────────────────────────────────────
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_compute_one_sample)(
            sample_id       = sid,
            solver          = solver,
            freq_list       = freq_list,
            seed            = seed,
            use_mask        = use_mask,
            lambda_range    = lambda_range,
            mu_range        = mu_range,
            rho_range       = rho_range,
            fixed_materials = (
                {k: fixed_materials.get(k, None) for k in ("lambda", "mu", "rho")}
                if fixed_materials else None
            ),
        )
        for sid in tqdm(range(num_samples), desc="Samples", unit="sample")
    )

    elapsed = time.time() - t0
    print(f"\n  Solved {num_samples * M} PDEs in {elapsed:.1f}s "
          f"({elapsed / (num_samples * M) * 1000:.1f} ms/solve)")

    # ── Assemble into arrays ─────────────────────────────────────────────────
    obstacle_masks = np.stack([r["obstacle_mask"] for r in results], axis=0)  # (N,Ng,Ng)
    lambda_fields  = np.stack([r["lambda"]        for r in results], axis=0)
    mu_fields      = np.stack([r["mu"]            for r in results], axis=0)
    rho_fields     = np.stack([r["rho"]           for r in results], axis=0)
    ux_real        = np.stack([r["ux_real"]       for r in results], axis=0)  # (N,M,Ng,Ng)
    ux_imag        = np.stack([r["ux_imag"]       for r in results], axis=0)
    uy_real        = np.stack([r["uy_real"]       for r in results], axis=0)
    uy_imag        = np.stack([r["uy_imag"]       for r in results], axis=0)

    # ── FTM data block: channels = [ux_re, ux_im, uy_re, uy_im] ─────────────
    # shape: (N, M, Ng, Ng, 4)
    data_ftm = np.stack([ux_real, ux_imag, uy_real, uy_imag], axis=-1).astype(np.float32)

    scale = 1.0
    if export_ftm and normalize:
        # max_abs = float(np.max(np.abs(data_ftm)))
        # scale   = max(max_abs, 1e-8)
        # data_ftm = (data_ftm / scale).astype(np.float32)
        max_per_freq = np.max(np.abs(data_ftm), axis=(0,2,3,4), keepdims=True)  # (1,M,1,1,1)
        max_per_freq = np.maximum(max_per_freq, 1e-8)
        data_ftm = (data_ftm / max_per_freq).astype(np.float32)
        scale = 1.0  # 逐频归一后全局scale无意义

    # ── Sparse observation mask ──────────────────────────────────────────────
    rng_mask = np.random.default_rng(seed + 999_999)
    mask_tr = build_sparse_mask(
        rng      = rng_mask,
        N        = num_samples,
        M        = M,
        grid     = Ng,
        channels = data_ftm.shape[-1],   # 4
        obs_ratio= obs_ratio,
        mask_mode= mask_mode,
    )

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta = dict(
        equation            = "frequency_domain_elastic_wave_2d",
        formulation         = "coupled ux/uy displacement system with spatially varying lambda, mu, rho",
        components          = ["ux", "uy"],
        channels            = ["ux_real", "ux_imag", "uy_real", "uy_imag"],
        material_params     = ["lambda", "mu", "rho"],
        N                   = num_samples,
        M                   = M,
        grid                = Ng,
        L                   = L,
        freq_train          = list(freq_train),
        freq_extrap         = list(freq_extrap),
        use_mask            = bool(use_mask),
        obstacle_type       = "hard" if use_mask else "none",
        obstacle_bc         = "dirichlet_zero" if use_mask else "none",
        fix_lambda          = bool(fix_lambda),
        fix_mu              = bool(fix_mu),
        fix_rho             = bool(fix_rho),
        lambda_range        = [float(lambda_range[0]), float(lambda_range[1])],
        mu_range            = [float(mu_range[0]),     float(mu_range[1])],
        rho_range           = [float(rho_range[0]),    float(rho_range[1])],
        sigma_max           = float(sigma_max),
        linear_solver       = linear_solver,
        gmres_rtol          = float(gmres_rtol),
        gmres_atol          = float(gmres_atol),
        gmres_restart       = int(gmres_restart),
        gmres_maxiter       = None if gmres_maxiter is None else int(gmres_maxiter),
        seed                = seed,
        export_ftm          = bool(export_ftm),
        normalize           = bool(normalize) and export_ftm,
        obs_ratio           = float(obs_ratio),
        mask_mode           = mask_mode,
        data_scale          = float(scale),
        elapsed_seconds     = round(elapsed, 2),
    )

    # ── Write HDF5 ────────────────────────────────────────────────────────────
    gz = {"compression": "gzip", "compression_opts": 4}

    with h5py.File(out_path, "w") as f:
        f.create_dataset("metadata",  data=json.dumps(meta))
        f.create_dataset("omega",     data=np.array(freq_list, dtype=np.float64))
        f.create_dataset("grid_x",    data=solver.x1d.astype(np.float64))
        f.create_dataset("grid_y",    data=solver.x1d.astype(np.float64))

        # Material & geometry
        f.create_dataset("obstacle_mask", data=obstacle_masks, dtype=np.float32, **gz)
        f.create_dataset("lambda_field",  data=lambda_fields,  dtype=np.float32, **gz)
        f.create_dataset("mu_field",      data=mu_fields,      dtype=np.float32, **gz)
        f.create_dataset("rho_field",     data=rho_fields,     dtype=np.float32, **gz)

        # Raw complex fields
        f.create_dataset("ux_real", data=ux_real, dtype=np.float32, **gz)
        f.create_dataset("ux_imag", data=ux_imag, dtype=np.float32, **gz)
        f.create_dataset("uy_real", data=uy_real, dtype=np.float32, **gz)
        f.create_dataset("uy_imag", data=uy_imag, dtype=np.float32, **gz)

        # FTM-ready block + sparse mask
        if export_ftm:
            f.create_dataset("data",    data=data_ftm, dtype=np.float32, **gz)
            f.create_dataset("mask_tr", data=mask_tr,  dtype=np.uint8,   **gz)

        f.create_dataset("data_scale", data=np.array(scale, dtype=np.float32))

    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved  {out_path}  ({size_mb:.1f} MB)\n")

    # ── Optional .npy metadata sidecar (mirrors Helmholtz convention) ─────────
    if export_ftm:
        meta_ftm = {
            "data": {
                "u_ind_uni":   solver.x1d.astype(np.float32),
                "v_ind_uni":   solver.x1d.astype(np.float32),
                "w_ind_uni":   np.arange(data_ftm.shape[-1], dtype=np.float32),
                "t_ind_uni":   np.array(freq_list, dtype=np.float32),
                "data_scale":  float(scale),
                "obs_ratio":   float(obs_ratio),
                "mask_mode":   mask_mode,
                "mask_tr_shape": tuple(int(v) for v in mask_tr.shape),
                "mask_shared_across_samples": bool(mask_tr.ndim == 4),
                "normalize":   bool(normalize),
                "channels":    ["ux_real", "ux_imag", "uy_real", "uy_imag"],
            }
        }
        if mask_tr.ndim == 4:
            meta_ftm["data"]["mask_tr"] = mask_tr.astype(np.float32)

        if metadata_out:
            meta_path = Path(metadata_out)
        else:
            meta_path = out_path.with_name(f"{out_path.stem}_metadata.npy")
        np.save(meta_path, meta_ftm, allow_pickle=True)
        print(f"  Saved  {meta_path}")

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2D elastic wave dataset (HDF5)")
    parser.add_argument("--N",            type=int,   default=1500)
    parser.add_argument("--grid",         type=int,   default=128)
    parser.add_argument("--L",            type=float, default=1.0)
    parser.add_argument("--n_jobs",       type=int,   default=8)
    parser.add_argument("--out",          type=str,   default="elastic_dataset.h5")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--sigma_max",    type=float, default=60.0)
    parser.add_argument("--linear_solver",type=str,   default="spsolve",
                        choices=["gmres", "spsolve"])
    parser.add_argument("--gmres_rtol",   type=float, default=1e-6)
    parser.add_argument("--gmres_atol",   type=float, default=1e-8)
    parser.add_argument("--gmres_restart",type=int,   default=200)
    parser.add_argument("--gmres_maxiter",type=int,   default=500)
    # Frequencies
    parser.add_argument("--freq_train",   type=int, nargs="+",
                        default=[2,6,10,14,18,22,26])
    parser.add_argument("--freq_extrap",  type=int, nargs="+",
                        default=[])
    # Obstacle / materials
    parser.add_argument("--no_mask",      action="store_true")
    parser.add_argument("--no_fix_lambda",action="store_true")
    parser.add_argument("--no_fix_mu",    action="store_true")
    parser.add_argument("--no_fix_rho",   action="store_true")
    # FTM / sparse mask
    parser.add_argument("--no_export_ftm",action="store_true")
    parser.add_argument("--disable_normalize", action="store_true")
    parser.add_argument("--obs_ratio",    type=float, default=0.1)
    parser.add_argument("--mask_mode",    type=str,   default="per_sample_fixed",
                        choices=["fixed", "per_freq",
                                 "per_sample_fixed", "per_sample_per_freq"])
    parser.add_argument("--metadata_out", type=str,   default="")
    args = parser.parse_args()

    generate_elastic_dataset(
        num_samples    = args.N,
        freq_train     = args.freq_train,
        freq_extrap    = args.freq_extrap,
        N              = args.grid,
        L              = args.L,
        n_jobs         = args.n_jobs,
        out            = args.out,
        seed           = args.seed,
        use_mask       = not args.no_mask,
        fix_lambda     = not args.no_fix_lambda,
        fix_mu         = not args.no_fix_mu,
        fix_rho        = not args.no_fix_rho,
        sigma_max      = args.sigma_max,
        linear_solver  = args.linear_solver,
        gmres_rtol     = args.gmres_rtol,
        gmres_atol     = args.gmres_atol,
        gmres_restart  = args.gmres_restart,
        gmres_maxiter  = args.gmres_maxiter,
        export_ftm     = not args.no_export_ftm,
        normalize      = not args.disable_normalize,
        obs_ratio      = args.obs_ratio,
        mask_mode      = args.mask_mode,
        metadata_out   = args.metadata_out,
    )