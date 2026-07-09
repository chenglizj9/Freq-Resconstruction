import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg


class HarmonicHeatSolver:
    def __init__(self, N: int, L: float = 1.0, diffusivity: float = 1.0, capacity: float = 1.0):
        self.N = int(N)
        self.L = float(L)
        self.diffusivity = float(diffusivity)
        self.capacity = float(capacity)
        self.h = self.L / (self.N - 1)
        self._lap = self._build_laplacian()

    def _build_laplacian(self) -> sparse.csc_matrix:
        n = self.N
        h2 = self.h ** 2
        main = -4.0 * np.ones(n * n, dtype=np.float64)
        off1 = np.ones(n * n - 1, dtype=np.float64)
        offn = np.ones(n * n - n, dtype=np.float64)

        for i in range(1, n):
            off1[i * n - 1] = 0.0

        lap = sparse.diags(
            diagonals=[main, off1, off1, offn, offn],
            offsets=[0, -1, 1, -n, n],
            shape=(n * n, n * n),
            format="lil",
        ) / h2

        for i in range(n):
            for j in range(n):
                if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                    idx = i * n + j
                    lap.rows[idx] = [idx]
                    lap.data[idx] = [1.0]

        return lap.tocsc()

    def _build_matrix(self, omega: float) -> sparse.csc_matrix:
        n2 = self.N * self.N
        mass = sparse.identity(n2, format="csc", dtype=np.complex128)
        a = -self.diffusivity * self._lap.astype(np.complex128) + 1j * float(omega) * self.capacity * mass
        return a

    def gaussian_source(self, pos_xy, amp=1.0 + 0.0j, sigma: float = 0.05):
        x0, y0 = float(pos_xy[0]), float(pos_xy[1])
        grid = np.linspace(0.0, self.L, self.N, dtype=np.float64)
        xx, yy = np.meshgrid(grid, grid, indexing="ij")
        g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma ** 2))
        g = g / max(float(np.sum(g) * self.h * self.h), 1e-12)
        return np.asarray(amp, dtype=np.complex128) * g

    def multi_source(self, positions, amplitudes, sigma: float = 0.05):
        src = np.zeros((self.N, self.N), dtype=np.complex128)
        for pos, amp in zip(positions, amplitudes):
            src += self.gaussian_source(pos, amp=amp, sigma=sigma)
        src[[0, -1], :] = 0.0
        src[:, [0, -1]] = 0.0
        return src

    def solve(self, omega: float, source_field: np.ndarray) -> np.ndarray:
        a = self._build_matrix(omega)
        rhs = np.asarray(source_field, dtype=np.complex128).reshape(-1)
        rhs = rhs.copy()
        n = self.N
        boundary = np.zeros((n, n), dtype=bool)
        boundary[[0, -1], :] = True
        boundary[:, [0, -1]] = True
        rhs[boundary.reshape(-1)] = 0.0
        sol = splinalg.spsolve(a, rhs)
        return sol.reshape(n, n)
