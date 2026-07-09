import numpy as np
import h5py
from Heat_Solver import HarmonicHeatSolver


def evaluate_physics_residual(
    pred: np.ndarray,
    h5_file: h5py.File,
    sample_idx: int,
    omega: float,
    h5_meta: dict,
    eps: float = 1e-12,
) -> float:
    h, w, _ = pred.shape
    if h != w:
        return np.nan

    l_val = float(h5_meta.get("L", 1.0))
    diffusivity = float(h5_meta.get("diffusivity", 1.0))
    capacity = float(h5_meta.get("capacity", 1.0))
    source_sigma = float(h5_meta.get("source_sigma", 0.05))

    solver = HarmonicHeatSolver(
        N=h,
        L=l_val,
        diffusivity=diffusivity,
        capacity=capacity,
    )

    if "source_fields_real" in h5_file and "source_fields_imag" in h5_file:
        f_re = h5_file["source_fields_real"][sample_idx]
        f_im = h5_file["source_fields_imag"][sample_idx]
    else:
        pos = h5_file["sources"][sample_idx]
        amp = h5_file["amplitudes"][sample_idx]
        valid = np.isfinite(pos[:, 0]) & np.isfinite(amp[:, 0])
        pos = pos[valid]
        amp = amp[valid]
        amp_c = amp[:, 0] + 1j * amp[:, 1]
        src = solver.multi_source(pos, amp_c, sigma=source_sigma)
        f_re = src.real
        f_im = src.imag

    a = solver._build_matrix(omega)
    a_re = a.real
    a_im = a.imag

    u_re = pred[..., 0].reshape(-1)
    u_im = pred[..., 1].reshape(-1)

    au_re = a_re @ u_re - a_im @ u_im
    au_im = a_re @ u_im + a_im @ u_re

    res_re = au_re - f_re.reshape(-1)
    res_im = au_im - f_im.reshape(-1)

    interior = np.ones((h, w), dtype=bool)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    interior = interior.reshape(-1)

    res_re = res_re[interior]
    res_im = res_im[interior]
    f_re_i = f_re.reshape(-1)[interior]
    f_im_i = f_im.reshape(-1)[interior]

    res_norm = np.mean(res_re ** 2 + res_im ** 2)
    f_norm = np.mean(f_re_i ** 2 + f_im_i ** 2)
    return float(np.sqrt(res_norm / (f_norm + eps)))
