code = """import numpy as np
import h5py
from typing import Dict, Any
from Helmholtz_Solver import HelmholtzSolver

def evaluate_physics_residual(pred_ri: np.ndarray, h5_f: h5py.File, sample_idx: int, omega: float, h5_meta: Dict[str, Any]) -> float:
    h, w = pred_ri.shape[:2]
    if h != w:
        return float('nan')
    l_val = float(h5_meta.get("L", 1.0))
    c_val = float(h5_meta.get("c", 1.0))
    pml_val = float(h5_meta.get("pml_width", 0.12))
    sigma_val = float(h5_meta.get("sigma_max", 50.0))
    source_sigma = float(h5_meta.get("source_sigma", 0.02))

    solver = HelmholtzSolver(N=h, L=l_val, c=c_val, pml_width=pml_val, sigma_max=sigma_val)
    A = solver._build_matrix(omega)

    if ("source_fields_real" in h5_f):
        src_re = h5_f["source_fields_real"][sample_idx].astype(np.float32)
        src_im = h5_f["source_fields_imag"][sample_idx].astype(np.float32)
        src = (src_re + 1j * src_im).astype(np.complex64)
    elif "sources" in h5_f and "amplitudes" in h5_f:
        pos = h5_f["sources"][sample_idx].astype(np.float64)
        amp = h5_f["amplitudes"][sample_idx].astype(np.float64)
        valid = (np.isfinite(pos[:, 0]) & np.isfinite(amp[:, 0]))
        if np.any(valid):
            positions = pos[valid]
            amplitudes = amp[valid, 0] + 1j * amp[valid, 1]
            src = solver.multi_source(positions, amplitudes, sigma=source_sigma).astype(np.complex64)
        else:
            src = np.zeros((solver.N, solver.N), dtype=np.complex64)
    else:
        src = np.zeros((solver.N, solver.N), dtype=np.complex64)

    data_scale = float(np.asarray(h5_f["data_scale"][()])) if "data_scale" in h5_f else 1.0
    src = src / data_scale

    u = pred_ri[..., 0].ravel() + 1j * pred_ri[..., 1].ravel()
    f = src.ravel()
    res = A.dot(u) + f
    
    mask = np.ones((h, w), dtype=bool)
    mask[[0, -1], :] = False
    mask[:, [0, -1]] = False
    idx = np.flatnonzero(mask)
    
    res = res[idx]
    f = f[idx]
    
    num = np.mean(np.abs(res)**2)
    den = np.mean(np.abs(f)**2)
    den = max(den, 1.0)
    
    return float(np.sqrt(num / max(den, 1e-12)))
"""
with open("physics_metric.py", "w") as f: f.write(code)
