import numpy as np
import h5py
from Helmholtz_Solver import HelmholtzSolver

def evaluate_physics_residual(
    pred: np.ndarray,        # (H, W, 2) 预测场
    h5_file: h5py.File,       # 数据集文件句柄
    sample_idx: int,         # 样本序号
    omega: float,            # 频率
    h5_meta: dict,           # 数据集元数据
    eps: float = 1e-12
) -> float:
    """
    计算亥姆霍兹方程残差，和 diffusion 测试代码完全一致
    """
    H, W, _ = pred.shape
    if H != W:
        return np.nan

    # ============= 1. 从数据集读取物理配置 =============
    L = h5_meta.get("L", 1.0)
    c = h5_meta.get("c", 1.0)
    pml_width = h5_meta.get("pml_width", 0.12)
    sigma_max = h5_meta.get("sigma_max", 50.0)
    source_sigma = h5_meta.get("source_sigma", 0.025)

    # ============= 2. 构建亥姆霍兹求解器 =============
    solver = HelmholtzSolver(
        N=H,
        L=L,
        c=c,
        pml_width=pml_width,
        sigma_max=sigma_max
    )

    # ============= 3. 获取源项 f =============
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

    # ============= 4. 构建 PDE 矩阵 A =============
    A = solver._build_matrix(omega)
    A_re = A.real
    A_im = A.imag

    # ============= 5. 预测场向量化 =============
    u_re = pred[..., 0].reshape(-1)
    u_im = pred[..., 1].reshape(-1)

    # ============= 6. 计算 A·u =============
    au_re = A_re @ u_re - A_im @ u_im
    au_im = A_re @ u_im + A_im @ u_re

    # ============= 7. 残差 = A u + f =============
    res_re = au_re + f_re.reshape(-1)
    res_im = au_im + f_im.reshape(-1)

    # ============= 8. 只算内部（去掉PML边界） =============
    interior = np.ones((H, W), dtype=bool)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    interior = interior.flatten()

    res_re = res_re[interior]
    res_im = res_im[interior]

    res_norm = np.mean(res_re**2 + res_im**2)
    f_norm = np.mean(f_re.reshape(-1)[interior]**2 + f_im.reshape(-1)[interior]**2)

    return float(np.sqrt(res_norm / (f_norm + eps)))