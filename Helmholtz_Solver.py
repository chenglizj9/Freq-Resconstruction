"""
helmholtz_solver.py
-------------------
2D Helmholtz equation solver using sparse finite differences + PML absorbing boundary.

Equation:  ∇²u + k²u = -f(x,y)
           k = ω / c,  c = speed of sound (default 1.0)

Domain:    [0, L] × [0, L],  discretised on an N×N grid.
BCs:       PML (Perfectly Matched Layer) on all four sides + outermost Dirichlet rows.

Usage:
    solver = HelmholtzSolver(N=128, L=1.0, c=1.0, pml_width=0.12)
    u = solver.solve(omega, source_fn)   # returns (N,N) complex ndarray
"""

import numpy as np
from scipy.sparse import diags, kron, eye, lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, factorized
from typing import Callable, Optional


# ──────────────────────────────────────────────────────────────────────────────
# PML profile
# ──────────────────────────────────────────────────────────────────────────────

def _pml_sigma(x: np.ndarray, L: float, pml_d: float, sigma_max: float) -> np.ndarray:
    """
    Quadratic PML conductivity profile σ(x).
    Active only in the layers [0, pml_d] and [L-pml_d, L].
    """
    sigma = np.zeros_like(x, dtype=float)
    # left layer
    mask_l = x < pml_d
    sigma[mask_l] = sigma_max * ((pml_d - x[mask_l]) / pml_d) ** 2
    # right layer
    mask_r = x > L - pml_d
    sigma[mask_r] = sigma_max * ((x[mask_r] - (L - pml_d)) / pml_d) ** 2
    return sigma


# ──────────────────────────────────────────────────────────────────────────────
# Main solver class
# ──────────────────────────────────────────────────────────────────────────────

class HelmholtzSolver:
    """
    Sparse FD solver for 2D Helmholtz + PML.

    Parameters
    ----------
    N          : grid points per side (interior + PML).  Recommended 64–256.
    L          : physical domain side length.
    c          : wave speed.
    pml_width  : PML thickness as a fraction of L  (e.g. 0.12 → 12%).
    sigma_max  : peak PML conductivity.  Higher = stronger absorption.
    """

    def __init__(
        self,
        N: int = 128,
        L: float = 1.0,
        c: float = 1.0,
        pml_width: float = 0.12,
        sigma_max: float = 50.0,
    ):
        self.N = N
        self.L = L
        self.c = c
        self.pml_d = pml_width * L
        self.sigma_max = sigma_max

        # 1-D grid (cell centres)
        self.h = L / N
        self.x1d = np.linspace(self.h / 2, L - self.h / 2, N)
        self.X, self.Y = np.meshgrid(self.x1d, self.x1d, indexing="ij")

        # PML stretch functions s_x(x), s_y(y) — stored as 1-D arrays
        sigma_x = _pml_sigma(self.x1d, L, self.pml_d, sigma_max)
        sigma_y = _pml_sigma(self.x1d, L, self.pml_d, sigma_max)
        # placeholder omega=1; the actual 1/s depends on omega at solve time
        self._sigma_x = sigma_x   # shape (N,)
        self._sigma_y = sigma_y   # shape (N,)

    # ------------------------------------------------------------------
    # Build system matrix for a given omega
    # ------------------------------------------------------------------

    def _build_matrix(self, omega: float) -> csc_matrix:
        N, h = self.N, self.h
        k = omega / self.c

        # Complex stretch 1/s_j(x) = 1 / (1 + i·σ_j / ω)
        sx = 1.0 / (1.0 + 1j * self._sigma_x / omega)   # (N,)
        sy = 1.0 / (1.0 + 1j * self._sigma_y / omega)   # (N,)

        # ── x-direction 1-D operator  ∂/∂x [ sx ∂/∂x ]  ──────────────
        # Face values: s_{i+1/2} = (sx_i + sx_{i+1})/2,  length N-1
        sx_ph = 0.5 * (sx[:-1] + sx[1:])   # right faces  (N-1,)

        # For interior node i the stencil is:
        #   (sx_{i+1/2} * (u_{i+1}-u_i) - sx_{i-1/2} * (u_i - u_{i-1})) / h²
        # main diagonal: -(sx_{i-1/2} + sx_{i+1/2}) / h²
        # We pad: sx_{-1/2} = sx_{0} (wall), sx_{N-1/2} = sx_{N-1} (wall)
        sx_lpad = np.r_[sx[0:1],    sx_ph]   # length N  (left  face of each node)
        sx_rpad = np.r_[sx_ph,      sx[-1:]] # length N  (right face of each node)
        d0_x = -(sx_lpad + sx_rpad) / h**2  # (N,)
        dp_x = sx_ph / h**2                  # (N-1,)  super-diagonal
        dm_x = sx_ph / h**2                  # (N-1,)  sub-diagonal  (same values)

        Ax = diags([dm_x, d0_x, dp_x], [-1, 0, 1], shape=(N, N), format="csc",
                   dtype=complex)

        # ── y-direction 1-D operator  ∂/∂y [ sy ∂/∂y ]  ──────────────
        sy_ph = 0.5 * (sy[:-1] + sy[1:])
        sy_lpad = np.r_[sy[0:1],    sy_ph]
        sy_rpad = np.r_[sy_ph,      sy[-1:]]
        d0_y = -(sy_lpad + sy_rpad) / h**2
        dp_y = sy_ph / h**2
        dm_y = sy_ph / h**2

        Ay = diags([dm_y, d0_y, dp_y], [-1, 0, 1], shape=(N, N), format="csc",
                   dtype=complex)

        # ── 2-D Kronecker product  (PML Laplacian + k² I)  ────────────
        # A_2D = Ax ⊗ I_y  +  I_x ⊗ Ay  +  k² I
        I = eye(N, format="csc", dtype=complex)
        A = kron(Ax, I, format="csc") + kron(I, Ay, format="csc")

        # k² term (uniform, since we absorbed PML into the differential ops)
        k2_diag = diags(np.full(N * N, k**2), 0, format="csc", dtype=complex)
        A = A + k2_diag

        # Dirichlet BC on boundary nodes (overwrite rows)
        # Boundary nodes: i==0, i==N-1, j==0, j==N-1
        A = A.tolil()
        for i in range(N):
            for j_idx, j in enumerate([0, N - 1]):
                row = i * N + j
                A[row, :] = 0
                A[row, row] = 1.0
            for i_idx, ii in enumerate([0, N - 1]):
                row = ii * N + i
                A[row, :] = 0
                A[row, row] = 1.0

        return A.tocsc()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        omega: float,
        source_fn: Optional[Callable] = None,
        source_field: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solve  (∇²_PML + k²) u = -f  and return u as (N, N) complex array.

        Provide exactly one of:
            source_fn    : callable f(X, Y) → ndarray, evaluated on self.X, self.Y
            source_field : pre-computed (N, N) ndarray
        """
        N = self.N
        if source_fn is not None:
            f = source_fn(self.X, self.Y).astype(complex)
        elif source_field is not None:
            f = source_field.astype(complex)
        else:
            raise ValueError("Provide source_fn or source_field")

        # Zero source at Dirichlet boundaries
        f[[0, -1], :] = 0.0
        f[:, [0, -1]] = 0.0

        A = self._build_matrix(omega)
        rhs = -f.ravel()
        u_flat = spsolve(A, rhs)
        return u_flat.reshape(N, N)

    def factorize_for_omega(self, omega: float):
        """
        Build and factorize the linear system matrix for a fixed frequency.

        Returns
        -------
        solve_op : callable
            Callable mapping rhs (flattened) -> solution (flattened).
            Useful when solving many right-hand sides for the same omega.
        """
        A = self._build_matrix(omega)
        return factorized(A)

    def solve_with_factorized(
        self,
        solve_op,
        source_fn: Optional[Callable] = None,
        source_field: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solve using a pre-factorized linear operator produced by factorize_for_omega.

        Provide exactly one of:
            source_fn    : callable f(X, Y) -> ndarray
            source_field : pre-computed (N, N) ndarray
        """
        N = self.N
        if source_fn is not None:
            f = source_fn(self.X, self.Y).astype(complex)
        elif source_field is not None:
            f = source_field.astype(complex)
        else:
            raise ValueError("Provide source_fn or source_field")

        # Zero source at Dirichlet boundaries.
        f[[0, -1], :] = 0.0
        f[:, [0, -1]] = 0.0

        rhs = -f.ravel()
        u_flat = solve_op(rhs)
        return u_flat.reshape(N, N)

    def gaussian_source(
        self,
        x0: float,
        y0: float,
        amplitude: float = 1.0,
        sigma: float = 0.025,
    ) -> np.ndarray:
        """
        Smooth Gaussian approximation of a point source at (x0, y0).
        Keeps the source away from PML to ensure physical excitation.
        """
        r2 = (self.X - x0) ** 2 + (self.Y - y0) ** 2
        return amplitude * np.exp(-r2 / (2 * sigma**2))

    def multi_source(
        self,
        positions: np.ndarray,
        amplitudes: Optional[np.ndarray] = None,
        sigma: float = 0.025,
    ) -> np.ndarray:
        """
        Superposition of Gaussian point sources.

        Parameters
        ----------
        positions  : (K, 2) array of (x, y) source positions, in [0, L]
        amplitudes : (K,) complex amplitudes  (default: random phase)
        """
        K = len(positions)
        if amplitudes is None:
            rng = np.random.default_rng()
            phases = rng.uniform(0, 2 * np.pi, K)
            amplitudes = np.exp(1j * phases)
        f = np.zeros((self.N, self.N), dtype=complex)
        for pos, amp in zip(positions, amplitudes):
            f += amp * self.gaussian_source(pos[0], pos[1], amplitude=1.0, sigma=sigma)
        return f
