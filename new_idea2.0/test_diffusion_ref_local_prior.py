"""
test_diffusion_ref_local_prior.py
---------------------------------
Evaluate conditional diffusion sampling with DPS + reference-conditioned local prior.

Pipeline
--------
1) Load trained diffusion model p(G | omega) on 2-channel core images.
2) Load FTM basis networks (net_x/net_y) from FTM checkpoint.
3) For each test case (sample, frequency), run:
    - prior sampling (no guidance)
    - guidance sampling using sparse observations, optional PDE, and local ref prior
4) Decode core -> field and report metrics/plots.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train_diffusion import ConditionalUNet2D, build_cosine_schedule, build_linear_schedule
from train_FTM_GPU import MLP1D, build_phi, normalize_coords_to_unit
from Helmholtz_Solver import HelmholtzSolver

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_omega(omega: float, omega_min: float, omega_max: float) -> float:
    den = max(omega_max - omega_min, 1e-12)
    return float((omega - omega_min) / den)


def _parse_freq_indices(text: str, num_freqs: int) -> List[int]:
    if text.strip() == "":
        return list(range(num_freqs))

    out = []
    for part in text.split(","):
        p = part.strip()
        if p == "":
            continue
        idx = int(p)
        if idx < 0 or idx >= num_freqs:
            raise ValueError(f"freq index out of range: {idx}, valid [0,{num_freqs - 1}]")
        out.append(idx)

    if len(out) == 0:
        raise ValueError("freq_indices parsed to empty list")
    return sorted(set(out))


def _load_diffusion(diff_ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    if not diff_ckpt_path.exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {diff_ckpt_path}")

    ckpt = torch.load(diff_ckpt_path, map_location="cpu")
    model_cfg = ckpt["model_config"]
    diff_cfg = ckpt["diffusion_config"]

    model = ConditionalUNet2D(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    schedule_name = str(diff_cfg.get("schedule", "linear")).lower()
    if schedule_name == "linear":
        schedule = build_linear_schedule(
            num_steps=int(diff_cfg["num_steps"]),
            beta_start=float(diff_cfg["beta_start"]),
            beta_end=float(diff_cfg["beta_end"]),
            device=device,
        )
    elif schedule_name == "cosine":
        schedule = build_cosine_schedule(
            num_steps=int(diff_cfg["num_steps"]),
            device=device,
        )
    else:
        raise ValueError(f"Unsupported diffusion schedule in checkpoint: {schedule_name}")

    mean = torch.from_numpy(_to_numpy(ckpt["core_stats"]["mean"]).astype(np.float32)).to(device)
    std = torch.from_numpy(_to_numpy(ckpt["core_stats"]["std"]).astype(np.float32)).to(device)
    rx = int(ckpt["core_stats"]["rx"])
    ry = int(ckpt["core_stats"]["ry"])
    latent_dim = int(ckpt["core_stats"]["latent_dim"])

    # Compatibility: accept shapes (2,Rx,Ry), (2,1,1), and legacy (1,2,1,1).
    if mean.ndim == 4 and mean.shape[0] == 1:
        mean = mean.squeeze(0)
    if std.ndim == 4 and std.shape[0] == 1:
        std = std.squeeze(0)

    valid_shapes = {(2, rx, ry), (2, 1, 1)}
    if tuple(mean.shape) not in valid_shapes or tuple(std.shape) not in valid_shapes:
        raise ValueError(
            f"Invalid mean/std shape in diffusion checkpoint: mean={tuple(mean.shape)} std={tuple(std.shape)} "
            f"expected one of {(2, rx, ry)} or {(2, 1, 1)}"
        )

    omega_min = float(ckpt["omega_stats"]["min"])
    omega_max = float(ckpt["omega_stats"]["max"])

    return {
        "model": model,
        "schedule": schedule,
        "mean": mean,
        "std": std,
        "rx": rx,
        "ry": ry,
        "latent_dim": latent_dim,
        "omega_min": omega_min,
        "omega_max": omega_max,
        "num_steps": int(diff_cfg["num_steps"]),
        "schedule_name": schedule_name,
    }


def _load_ftm_basis(ftm_ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    if not ftm_ckpt_path.exists():
        raise FileNotFoundError(f"FTM checkpoint not found: {ftm_ckpt_path}")

    ckpt = torch.load(ftm_ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})

    required = ["rank_x", "rank_y", "hidden_dim", "hidden_layers", "activation"]
    for k in required:
        if k not in cfg:
            raise KeyError(f"FTM checkpoint config missing key: {k}")

    net_x = MLP1D(
        out_dim=int(cfg["rank_x"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_hidden_layers=int(cfg["hidden_layers"]),
        activation=str(cfg["activation"]),
    ).to(device)
    net_y = MLP1D(
        out_dim=int(cfg["rank_y"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_hidden_layers=int(cfg["hidden_layers"]),
        activation=str(cfg["activation"]),
    ).to(device)

    net_x.load_state_dict(ckpt["net_x_state"])
    net_y.load_state_dict(ckpt["net_y_state"])
    net_x.eval()
    net_y.eval()

    return {
        "net_x": net_x,
        "net_y": net_y,
        "rank_x": int(cfg["rank_x"]),
        "rank_y": int(cfg["rank_y"]),
        "normalize_coords": bool(cfg.get("normalize_coords", True)),
    }


def _build_timestep_sequence(total_steps: int, sample_steps: int) -> List[int]:
    if sample_steps <= 0 or sample_steps >= total_steps:
        return list(range(total_steps - 1, -1, -1))
    seq = np.linspace(total_steps - 1, 0, sample_steps, dtype=np.int64).tolist()
    out = []
    seen = set()
    for t in seq:
        if t not in seen:
            out.append(int(t))
            seen.add(int(t))
    return out


def _load_h5_metadata_dict(f: h5py.File) -> Dict[str, Any]:
    if "metadata" not in f:
        return {}

    raw = f["metadata"][()]
    if isinstance(raw, (bytes, np.bytes_)):
        text = raw.decode("utf-8")
    elif isinstance(raw, np.ndarray) and raw.shape == ():
        scalar = raw.item()
        if isinstance(scalar, (bytes, np.bytes_)):
            text = scalar.decode("utf-8")
        else:
            text = str(scalar)
    else:
        text = str(raw)

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _load_data_scale(h5_path: Path, f: h5py.File) -> float:
    if "data_scale" in f:
        try:
            return float(np.asarray(f["data_scale"][()]))
        except Exception:
            pass

    sidecar = h5_path.with_name(f"{h5_path.stem}_metadata.npy")
    if sidecar.exists():
        try:
            meta = np.load(sidecar, allow_pickle=True).item()
            return float(meta.get("data", {}).get("data_scale", 1.0))
        except Exception:
            pass

    return 1.0


def _csc_to_torch_sparse(mat: Any, device: torch.device) -> torch.Tensor:
    coo = mat.tocoo()
    indices = np.vstack([coo.row, coo.col]).astype(np.int64)
    values = coo.data.astype(np.float32)

    idx_t = torch.from_numpy(indices).to(device)
    val_t = torch.from_numpy(values).to(device)
    return torch.sparse_coo_tensor(idx_t, val_t, size=coo.shape, device=device).coalesce()


def _build_interior_index(h: int, w: int, device: torch.device) -> torch.Tensor:
    mask = np.ones((h, w), dtype=bool)
    mask[[0, -1], :] = False
    mask[:, [0, -1]] = False
    idx = np.flatnonzero(mask.reshape(-1)).astype(np.int64)
    return torch.from_numpy(idx).to(device)


def _load_source_field_for_sample(
    f: h5py.File,
    sample_idx: int,
    solver: HelmholtzSolver,
    source_sigma: float,
) -> np.ndarray:
    if "source_fields_real" in f and "source_fields_imag" in f:
        src_re = f["source_fields_real"][sample_idx].astype(np.float32)
        src_im = f["source_fields_imag"][sample_idx].astype(np.float32)
        return (src_re + 1j * src_im).astype(np.complex64)

    if "sources" not in f or "amplitudes" not in f:
        return np.zeros((solver.N, solver.N), dtype=np.complex64)

    pos = f["sources"][sample_idx].astype(np.float64)
    amp = f["amplitudes"][sample_idx].astype(np.float64)

    valid = (
        np.isfinite(pos[:, 0])
        & np.isfinite(pos[:, 1])
        & np.isfinite(amp[:, 0])
        & np.isfinite(amp[:, 1])
    )
    if not np.any(valid):
        return np.zeros((solver.N, solver.N), dtype=np.complex64)

    positions = pos[valid]
    amplitudes = amp[valid, 0] + 1j * amp[valid, 1]
    src = solver.multi_source(positions, amplitudes, sigma=source_sigma)
    return src.astype(np.complex64)


def _pde_residual_loss(
    core_re: torch.Tensor,
    core_im: torch.Tensor,
    physics: Dict[str, torch.Tensor],
    eps: float,
) -> torch.Tensor:
    phi_full = physics["phi_full"]
    a_re = physics["A_re"]
    a_im = physics["A_im"]
    f_re = physics["f_re"]
    f_im = physics["f_im"]

    u_re = torch.matmul(core_re, phi_full.t()).transpose(0, 1).contiguous()
    u_im = torch.matmul(core_im, phi_full.t()).transpose(0, 1).contiguous()

    au_re = torch.sparse.mm(a_re, u_re) - torch.sparse.mm(a_im, u_im)
    au_im = torch.sparse.mm(a_re, u_im) + torch.sparse.mm(a_im, u_re)

    res_re = au_re.squeeze(1) + f_re
    res_im = au_im.squeeze(1) + f_im

    interior_idx = physics.get("interior_idx")
    if interior_idx is not None:
        res_re = res_re.index_select(0, interior_idx)
        res_im = res_im.index_select(0, interior_idx)

    num = torch.mean(res_re * res_re + res_im * res_im)
    den = physics.get("residual_den")
    if den is None:
        return num
    return num / torch.clamp(den, min=eps)


def _core_vectors_from_image(
    x_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # x_norm: (N,2,Rx,Ry), mean/std: (2,Rx,Ry) or (2,1,1)
    x = x_norm * std.unsqueeze(0) + mean.unsqueeze(0)
    core_re = x[:, 0].reshape(x.shape[0], -1)
    core_im = x[:, 1].reshape(x.shape[0], -1)
    return core_re, core_im


def _decode_field_from_core_image(
    core_img_norm: np.ndarray,  # (2,Rx,Ry)
    phi_np: np.ndarray,         # (P,R)
    mean_np: np.ndarray,        # (2,Rx,Ry) or (2,1,1)
    std_np: np.ndarray,         # (2,Rx,Ry) or (2,1,1)
    h: int,
    w: int,
) -> np.ndarray:
    core_img = core_img_norm * std_np + mean_np
    core_re = core_img[0].reshape(-1)
    core_im = core_img[1].reshape(-1)

    pred_re = phi_np @ core_re
    pred_im = phi_np @ core_im
    return np.stack([pred_re, pred_im], axis=-1).reshape(h, w, 2).astype(np.float32)


def _rel_rmse(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    num = float(np.sum((pred - gt) ** 2))
    den = float(np.sum(gt ** 2))
    return float(np.sqrt(num / max(den, eps)))


def _masked_rel_rmse(pred: np.ndarray, gt: np.ndarray, mask_bool: np.ndarray, eps: float = 1e-12) -> float:
    m = mask_bool.astype(bool)
    if not np.any(m):
        return float("nan")
    diff2 = (pred - gt) ** 2
    gt2 = gt ** 2
    num = float(np.sum(diff2[m]))
    den = float(np.sum(gt2[m]))
    return float(np.sqrt(num / max(den, eps)))


def _solve_lstsq_core_from_gt(
    gt: np.ndarray,          # (H,W,2)
    phi_np: np.ndarray,      # (P,R)
    rx: int,
    ry: int,
    rcond: Optional[float] = None,
) -> np.ndarray:
    gt_re = gt[..., 0].reshape(-1)
    gt_im = gt[..., 1].reshape(-1)

    core_re, *_ = np.linalg.lstsq(phi_np, gt_re, rcond=rcond)
    core_im, *_ = np.linalg.lstsq(phi_np, gt_im, rcond=rcond)
    return np.stack([core_re, core_im], axis=0).reshape(2, rx, ry).astype(np.float32)


def _core_rel_err(pred_core: np.ndarray, gt_core: np.ndarray, eps: float = 1e-12) -> float:
    num = float(np.sum((pred_core - gt_core) ** 2))
    den = float(np.sum(gt_core ** 2))
    return float(np.sqrt(num / max(den, eps)))


def _select_reference_freq_indices(target_idx: int, num_freqs: int, num_each_side: int) -> List[int]:
    n_side = int(max(1, min(3, num_each_side)))
    ref_ids: List[int] = []
    for offset in range(1, n_side + 1):
        left = target_idx - offset
        right = target_idx + offset
        if left >= 0:
            ref_ids.append(left)
        if right < num_freqs:
            ref_ids.append(right)
    return ref_ids


def _compute_ref_weights(
    ref_omegas: np.ndarray,
    target_omega: float,
    mode: str,
    weight_lambda: float,
) -> np.ndarray:
    k = int(ref_omegas.shape[0])
    if k == 0:
        return np.zeros((0,), dtype=np.float32)

    if mode == "uniform":
        return np.full((k,), 1.0 / k, dtype=np.float32)

    if mode == "two_sided_linear":
        left_cands = np.where(ref_omegas < target_omega)[0]
        right_cands = np.where(ref_omegas > target_omega)[0]
        if left_cands.size > 0 and right_cands.size > 0:
            left_idx = int(left_cands[np.argmax(ref_omegas[left_cands])])
            right_idx = int(right_cands[np.argmin(ref_omegas[right_cands])])
            omega_l = float(ref_omegas[left_idx])
            omega_r = float(ref_omegas[right_idx])
            den = max(omega_r - omega_l, 1e-12)
            w = np.zeros((k,), dtype=np.float32)
            w[left_idx] = float((omega_r - target_omega) / den)
            w[right_idx] = float((target_omega - omega_l) / den)
            s = float(np.sum(w))
            if s > 0.0:
                return w / s

    lam = float(max(weight_lambda, 1e-12))
    dist = np.abs(ref_omegas - float(target_omega)).astype(np.float64)
    logits = -lam * dist
    logits = logits - np.max(logits)
    ww = np.exp(logits)
    den = float(np.sum(ww))
    if den <= 0.0:
        return np.full((k,), 1.0 / k, dtype=np.float32)
    return (ww / den).astype(np.float32)


def _build_local_ref_gaussian_prior(
    ref_cores: np.ndarray,
    ref_omegas: np.ndarray,
    target_omega: float,
    weight_mode: str,
    weight_lambda: float,
    var_eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ref_cores: (K,2,Rx,Ry)
    if ref_cores.ndim != 4 or ref_cores.shape[1] != 2:
        raise ValueError(f"Invalid ref_cores shape: {ref_cores.shape}, expected (K,2,Rx,Ry)")

    weights = _compute_ref_weights(
        ref_omegas=ref_omegas.astype(np.float64),
        target_omega=float(target_omega),
        mode=weight_mode,
        weight_lambda=weight_lambda,
    )
    w = weights.reshape(-1, 1, 1, 1).astype(np.float32)
    mu = np.sum(w * ref_cores, axis=0).astype(np.float32)
    centered = ref_cores - mu[None, ...]
    var = (np.sum(w * (centered ** 2), axis=0) + float(var_eps)).astype(np.float32)
    return mu, var, weights


def _plot_case(
    out_path: Path,
    gt: np.ndarray,
    pred_prior: np.ndarray,
    pred_dps: np.ndarray,
    core_gt_lstsq: np.ndarray,
    core_dps: np.ndarray,
    core_rel_err_dps: float,
    core_rel_err_prior: float,
    mask_bool: np.ndarray,
    sample_idx: int,
    freq_idx: int,
    omega_val: float,
) -> None:
    gt_re, gt_im = gt[..., 0], gt[..., 1]
    pr_re, pr_im = pred_prior[..., 0], pred_prior[..., 1]
    dp_re, dp_im = pred_dps[..., 0], pred_dps[..., 1]

    gt_amp = np.sqrt(gt_re ** 2 + gt_im ** 2)
    pr_amp = np.sqrt(pr_re ** 2 + pr_im ** 2)
    dp_amp = np.sqrt(dp_re ** 2 + dp_im ** 2)

    err_pr = np.abs(pr_amp - gt_amp)
    err_dp = np.abs(dp_amp - gt_amp)
    mask_img = mask_bool[..., 0].astype(np.float32)

    cg_re, cg_im = core_gt_lstsq[0], core_gt_lstsq[1]
    cd_re, cd_im = core_dps[0], core_dps[1]
    err_c_re = np.abs(cd_re - cg_re)
    err_c_im = np.abs(cd_im - cg_im)

    fig, axes = plt.subplots(6, 3, figsize=(16, 18))
    items = [
        (gt_re, "GT Real", "viridis"),
        (pr_re, "Prior Real", "viridis"),
        (dp_re, "DPS Real", "viridis"),
        (gt_im, "GT Imag", "viridis"),
        (pr_im, "Prior Imag", "viridis"),
        (dp_im, "DPS Imag", "viridis"),
        (gt_amp, "GT Amp", "viridis"),
        (pr_amp, "Prior Amp", "viridis"),
        (dp_amp, "DPS Amp", "viridis"),
        (err_pr, "Abs Err Prior Amp", "magma"),
        (err_dp, "Abs Err DPS Amp", "magma"),
        (mask_img, "Obs Mask (Re ch)", "gray"),
        (cg_re, "Core GT LSQ Real", "viridis"),
        (cd_re, "Core DPS Real", "viridis"),
        (err_c_re, "Abs Err Core Real", "magma"),
        (cg_im, "Core GT LSQ Imag", "viridis"),
        (cd_im, "Core DPS Imag", "viridis"),
        (err_c_im, "Abs Err Core Imag", "magma"),
    ]

    for ax, (img, title, cmap) in zip(axes.flat, items):
        im = ax.imshow(img, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"sample={sample_idx}, freq_idx={freq_idx}, omega={omega_val:.4f} | "
        f"core_rel_err_prior={core_rel_err_prior:.3e}, core_rel_err_dps={core_rel_err_dps:.3e}",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_freq_curve(out_path: Path, omega: np.ndarray, prior_vals: np.ndarray, dps_vals: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(omega, prior_vals, label="Prior", lw=1.8)
    ax.plot(omega, dps_vals, label="DPS", lw=1.8)
    ax.set_xlabel("omega")
    ax.set_ylabel("Mean relative RMSE")
    ax.set_title("Frequency-wise Reconstruction Error")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def sample_core(
    model: ConditionalUNet2D,
    schedule: Any,
    omega_norm: float,
    mean: torch.Tensor,     # (2,Rx,Ry)
    std: torch.Tensor,      # (2,Rx,Ry)
    a_re: torch.Tensor,     # (n_obs_re, R)
    a_im: torch.Tensor,     # (n_obs_im, R)
    y_re: torch.Tensor,     # (1, n_obs_re)
    y_im: torch.Tensor,     # (1, n_obs_im)
    timestep_seq: Sequence[int],
    dps_weight: float,
    phys_weight: float,
    guidance_scale: float,
    init_noise: torch.Tensor,  # (1,2,Rx,Ry)
    physics: Optional[Dict[str, torch.Tensor]] = None,
    ref_prior: Optional[Dict[str, torch.Tensor]] = None,
    ref_weight: float = 0.0,
    guidance_grad_clip: float = 1e-3,
    log_guidance: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    device = init_noise.device
    x = init_noise.clone()
    omega_cond = torch.tensor([[omega_norm]], device=device, dtype=torch.float32)

    for i, t_idx in enumerate(timestep_seq):
        t = torch.full((1,), int(t_idx), device=device, dtype=torch.long)

        with torch.no_grad():
            eps_pred = model(x, t, omega_cond)
            abar_t = schedule.alpha_bars[t_idx]
            x0_hat = (x - torch.sqrt(torch.clamp(1.0 - abar_t, min=1e-12)) * eps_pred) / torch.sqrt(
                torch.clamp(abar_t, min=1e-12)
            )

        safe_step = len(timestep_seq) * 0.4
        is_safe_step = i < len(timestep_seq) - safe_step
        more_safe_step = i < len(timestep_seq) - safe_step * 0.5
        use_dps = dps_weight > 0.0
        use_phys = phys_weight > 0.0 and physics is not None
        use_ref = ref_weight > 0.0 and ref_prior is not None

        if use_dps or use_phys or use_ref:
            x0_var = x0_hat.detach().requires_grad_(True)
            core_re, core_im = _core_vectors_from_image(x0_var, mean=mean, std=std)

            total_loss = torch.zeros((), device=device, dtype=torch.float32)
            like_loss: Optional[torch.Tensor] = None
            pde_loss: Optional[torch.Tensor] = None
            ref_loss: Optional[torch.Tensor] = None

            if use_dps:
                pred_re_obs = torch.matmul(core_re, a_re.t())
                pred_im_obs = torch.matmul(core_im, a_im.t())

                rel_re = torch.mean((pred_re_obs - y_re) ** 2) / (torch.mean(y_re ** 2) + eps)
                rel_im = torch.mean((pred_im_obs - y_im) ** 2) / (torch.mean(y_im ** 2) + eps)
                like_loss = rel_re + rel_im
                if not more_safe_step:
                    like_loss = like_loss * 0.5
                if omega_cond.item() > 0.65:
                    like_loss = like_loss * 0.1
                total_loss = total_loss + dps_weight * like_loss

            if use_phys and physics is not None:
                pde_loss = _pde_residual_loss(core_re=core_re, core_im=core_im, physics=physics, eps=eps)
                if more_safe_step:
                    pde_loss = pde_loss * 3.0
                if omega_cond.item() > 0.65:
                    pde_loss = pde_loss * 0.05
                total_loss = total_loss + phys_weight * pde_loss

            if use_ref and ref_prior is not None:
                mu_re = ref_prior["mu_re"]
                mu_im = ref_prior["mu_im"]
                var_re = torch.clamp(ref_prior["var_re"], min=eps)
                var_im = torch.clamp(ref_prior["var_im"], min=eps)

                ref_loss_re = torch.mean(((core_re - mu_re) ** 2) / var_re)
                ref_loss_im = torch.mean(((core_im - mu_im) ** 2) / var_im)
                ref_loss = 0.5 * (ref_loss_re + ref_loss_im)
                total_loss = total_loss + ref_weight * ref_loss

            grad = torch.autograd.grad(total_loss, x0_var, retain_graph=False, create_graph=False)[0]

            time_weight = 1.0 - (t_idx / max(timestep_seq[0], 1))
            # time_weight = (t_idx / max(timestep_seq[0], 1))
            # time_weight = 1.0
            #低频的引导强度提高
            if omega_cond.item() < 0.1:
                grad = grad * 3
                time_weight = 1.0

            if not is_safe_step and guidance_grad_clip > 0.0:
                grad = torch.clamp(grad, -guidance_grad_clip, guidance_grad_clip)
            
            if not more_safe_step:
                grad = torch.clamp(grad, -guidance_grad_clip * 0.5, guidance_grad_clip * 0.5)

            
            x0_hat = x0_hat - time_weight * guidance_scale * grad
            if log_guidance:
                msg = f"grad max: {grad.abs().max().item():.3e}"
                if like_loss is not None:
                    msg += f", like_loss: {like_loss.item():.3e}"
                if pde_loss is not None:
                    msg += f", pde_loss: {pde_loss.item():.3e}"
                if ref_loss is not None:
                    msg += f", ref_loss: {ref_loss.item():.3e}"
                print(msg)

        if i == len(timestep_seq) - 1:
            x = x0_hat
        else:
            t_prev = timestep_seq[i + 1]
            abar_prev = schedule.alpha_bars[t_prev]
            with torch.no_grad():
                x = torch.sqrt(torch.clamp(abar_prev, min=1e-12)) * x0_hat + torch.sqrt(
                    torch.clamp(1.0 - abar_prev, min=1e-12)
                ) * eps_pred

    return x.squeeze(0)


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    diff = _load_diffusion(Path(args.diff_ckpt), device=device)
    ftm = _load_ftm_basis(Path(args.ftm_ckpt), device=device)

    model = diff["model"]
    schedule = diff["schedule"]
    mean = diff["mean"]
    std = diff["std"]
    omega_min = diff["omega_min"]
    omega_max = diff["omega_max"]
    rx, ry = diff["rx"], diff["ry"]
    r = rx * ry

    if diff["latent_dim"] != 2 * r:
        raise ValueError(f"latent_dim mismatch: latent_dim={diff['latent_dim']}, expected={2 * r}")

    timestep_seq = _build_timestep_sequence(diff["num_steps"], args.sample_steps)

    data_h5_path = Path(args.data_h5)
    with h5py.File(data_h5_path, "r") as f:
        if "data" not in f or "mask_tr" not in f or "omega" not in f:
            raise KeyError("HDF5 must contain data, mask_tr, omega")

        data_ds = f["data"]
        mask_ds = f["mask_tr"]
        omega = f["omega"][...].astype(np.float32)

        b_data, m_data, h, w, c = data_ds.shape
        if c != 2:
            raise ValueError("Only 2-channel data is supported")

        freq_ids = _parse_freq_indices(args.freq_indices, m_data)
        b_eval = b_data if args.max_samples <= 0 else min(b_data, args.max_samples)
        sample_ids = list(range(b_eval))

        all_cases = [(b, m) for b in sample_ids for m in freq_ids]
        if len(all_cases) == 0:
            raise ValueError("No evaluation cases")

        if args.max_cases > 0 and len(all_cases) > args.max_cases:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(all_cases), size=args.max_cases, replace=False)
            cases = [all_cases[int(i)] for i in idx]
        else:
            cases = all_cases

        grid_x = f["grid_x"][...].astype(np.float32) if "grid_x" in f else np.linspace(0, 1, h, dtype=np.float32)
        grid_y = f["grid_y"][...].astype(np.float32) if "grid_y" in f else np.linspace(0, 1, w, dtype=np.float32)

        if ftm["normalize_coords"]:
            x_coords = normalize_coords_to_unit(grid_x.astype(np.float64)).astype(np.float32)
            y_coords = normalize_coords_to_unit(grid_y.astype(np.float64)).astype(np.float32)
        else:
            x_coords = grid_x
            y_coords = grid_y

        x_t = torch.from_numpy(x_coords).unsqueeze(-1).to(device)
        y_t = torch.from_numpy(y_coords).unsqueeze(-1).to(device)
        with torch.no_grad():
            phi = build_phi(ftm["net_x"], ftm["net_y"], x_t, y_t)
        phi_np = phi.detach().cpu().numpy().astype(np.float32)
        if phi_np.shape[1] != r:
            raise ValueError(f"Phi rank mismatch: phi={phi_np.shape[1]}, expected={r}")

        mean_np = mean.detach().cpu().numpy().astype(np.float32)
        std_np = std.detach().cpu().numpy().astype(np.float32)

        h5_meta = _load_h5_metadata_dict(f)
        data_scale = max(_load_data_scale(data_h5_path, f), 1e-12)
        phi_t_full = torch.from_numpy(phi_np).to(device)

        use_phys_guidance = args.phys_weight > 0.0
        physics_ready = False
        # Auto lambda follows doc suggestion: lambda ~ 1 / average frequency spacing.
        omega_sorted = np.sort(omega.astype(np.float64))
        if omega_sorted.size > 1:
            omega_avg_delta = float(np.mean(np.abs(np.diff(omega_sorted))))
        else:
            omega_avg_delta = 1.0
        ref_weight_lambda = (
            float(args.ref_weight_lambda)
            if args.ref_weight_lambda > 0.0
            else float(1.0 / max(omega_avg_delta, 1e-6))
        )
        source_sigma = float(h5_meta.get("source_sigma", args.source_sigma))
        a_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        source_cache: Dict[int, np.ndarray] = {}
        core_gt_cache: Dict[Tuple[int, int], np.ndarray] = {}
        interior_idx_t: Optional[torch.Tensor] = None
        solver: Optional[HelmholtzSolver] = None

        if use_phys_guidance:
            if h != w:
                print("[WARN] phys_weight>0 but non-square grid, physics guidance is disabled.")
            else:
                l_val = float(h5_meta.get("L", args.helmholtz_L))
                c_val = float(h5_meta.get("c", args.helmholtz_c))
                pml_val = float(h5_meta.get("pml_width", args.helmholtz_pml_width))
                sigma_val = float(h5_meta.get("sigma_max", args.helmholtz_sigma_max))
                solver = HelmholtzSolver(
                    N=h,
                    L=l_val,
                    c=c_val,
                    pml_width=pml_val,
                    sigma_max=sigma_val,
                )
                if args.phys_interior_only:
                    interior_idx_t = _build_interior_index(h, w, device=device)
                physics_ready = True
                print(
                    f"[INFO] physics guidance enabled: L={l_val:.4g}, c={c_val:.4g}, "
                    f"pml_width={pml_val:.4g}, sigma_max={sigma_val:.4g}, "
                    f"source_sigma={source_sigma:.4g}, data_scale={data_scale:.4g}"
                )

        rows = []
        vis_count = 0
        freq_prior = {m: [] for m in freq_ids}
        freq_dps = {m: [] for m in freq_ids}

        for i_case, (b_idx, m_idx) in enumerate(cases, start=1):
            gt = data_ds[b_idx, m_idx].astype(np.float32)
            omega_val = float(omega[m_idx])
            # omega_max = 101
            omega_cond = _normalize_omega(omega_val, omega_min=omega_min, omega_max=omega_max)

            if mask_ds.ndim == 4:
                mask_case = (mask_ds[m_idx].astype(np.float32) > 0.5)
            elif mask_ds.ndim == 5:
                mask_case = (mask_ds[b_idx, m_idx].astype(np.float32) > 0.5)
            else:
                raise ValueError(f"Invalid mask shape: {mask_ds.shape}")

            idx_re = np.flatnonzero(mask_case[..., 0].reshape(-1))
            idx_im = np.flatnonzero(mask_case[..., 1].reshape(-1))
            if idx_re.size == 0 or idx_im.size == 0:
                continue

            a_re = torch.from_numpy(phi_np[idx_re]).to(device)
            a_im = torch.from_numpy(phi_np[idx_im]).to(device)

            y_re = torch.from_numpy(gt[..., 0].reshape(-1)[idx_re]).to(device).view(1, -1)
            y_im = torch.from_numpy(gt[..., 1].reshape(-1)[idx_im]).to(device).view(1, -1)

            ref_ids = _select_reference_freq_indices(
                target_idx=m_idx,
                num_freqs=m_data,
                num_each_side=args.ref_num_each_side,
            )
            ref_prior_case: Optional[Dict[str, torch.Tensor]] = None
            if args.ref_weight > 0.0 and len(ref_ids) > 0:
                ref_cores = []
                ref_omegas = []
                for ref_m in ref_ids:
                    key_ref = (b_idx, ref_m)
                    if key_ref not in core_gt_cache:
                        gt_ref = data_ds[b_idx, ref_m].astype(np.float32)
                        core_gt_cache[key_ref] = _solve_lstsq_core_from_gt(
                            gt=gt_ref,
                            phi_np=phi_np,
                            rx=rx,
                            ry=ry,
                            rcond=args.core_lstsq_rcond,
                        )
                    ref_cores.append(core_gt_cache[key_ref])
                    ref_omegas.append(float(omega[ref_m]))

                ref_cores_np = np.stack(ref_cores, axis=0).astype(np.float32)
                ref_omegas_np = np.asarray(ref_omegas, dtype=np.float32)
                mu_ref, var_ref, _ = _build_local_ref_gaussian_prior(
                    ref_cores=ref_cores_np,
                    ref_omegas=ref_omegas_np,
                    target_omega=omega_val,
                    weight_mode=args.ref_weight_mode,
                    weight_lambda=ref_weight_lambda,
                    var_eps=args.ref_var_eps,
                )
                ref_prior_case = {
                    "mu_re": torch.from_numpy(mu_ref[0].reshape(1, -1)).to(device),
                    "mu_im": torch.from_numpy(mu_ref[1].reshape(1, -1)).to(device),
                    "var_re": torch.from_numpy(var_ref[0].reshape(1, -1)).to(device),
                    "var_im": torch.from_numpy(var_ref[1].reshape(1, -1)).to(device),
                }

            physics_case: Optional[Dict[str, torch.Tensor]] = None
            if physics_ready and solver is not None:
                if m_idx not in a_cache:
                    a_mat = solver._build_matrix(float(omega_val))
                    a_cache[m_idx] = (
                        _csc_to_torch_sparse(a_mat.real.astype(np.float32), device=device),
                        _csc_to_torch_sparse(a_mat.imag.astype(np.float32), device=device),
                    )

                a_re_op, a_im_op = a_cache[m_idx]

                if args.phys_use_source:
                    if b_idx not in source_cache:
                        source_cache[b_idx] = _load_source_field_for_sample(
                            f=f,
                            sample_idx=b_idx,
                            solver=solver,
                            source_sigma=source_sigma,
                        )
                    src = source_cache[b_idx]
                    f_re_np = src.real.reshape(-1).astype(np.float32)
                    f_im_np = src.imag.reshape(-1).astype(np.float32)
                    if args.phys_scale_source_by_data:
                        f_re_np = f_re_np / data_scale
                        f_im_np = f_im_np / data_scale
                else:
                    f_re_np = np.zeros(h * w, dtype=np.float32)
                    f_im_np = np.zeros(h * w, dtype=np.float32)

                f_re_t = torch.from_numpy(f_re_np).to(device)
                f_im_t = torch.from_numpy(f_im_np).to(device)

                if interior_idx_t is not None:
                    f_re_den = f_re_t.index_select(0, interior_idx_t)
                    f_im_den = f_im_t.index_select(0, interior_idx_t)
                else:
                    f_re_den = f_re_t
                    f_im_den = f_im_t

                residual_den = torch.mean(f_re_den * f_re_den + f_im_den * f_im_den)
                residual_den = torch.clamp(residual_den, min=1.0)

                physics_case = {
                    "phi_full": phi_t_full,
                    "A_re": a_re_op,
                    "A_im": a_im_op,
                    "f_re": f_re_t,
                    "f_im": f_im_t,
                    "interior_idx": interior_idx_t,
                    "residual_den": residual_den,
                }

            # Same initialization for fair prior-vs-DPS comparison.
            g = torch.Generator(device=device)
            g.manual_seed(args.seed + 17 * i_case)
            init_noise = torch.randn((1, 2, rx, ry), generator=g, device=device, dtype=torch.float32)

            prior_core = sample_core(
                model=model,
                schedule=schedule,
                omega_norm=omega_cond,
                mean=mean,
                std=std,
                a_re=a_re,
                a_im=a_im,
                y_re=y_re,
                y_im=y_im,
                timestep_seq=timestep_seq,
                dps_weight=0.0,
                phys_weight=0.0,
                guidance_scale=args.guidance_scale,
                init_noise=init_noise,
                physics=None,
                ref_prior=None,
                ref_weight=0.0,
                guidance_grad_clip=args.guidance_grad_clip,
                log_guidance=False,
                eps=args.eps,
            )

            dps_core = sample_core(
                model=model,
                schedule=schedule,
                omega_norm=omega_cond,
                mean=mean,
                std=std,
                a_re=a_re,
                a_im=a_im,
                y_re=y_re,
                y_im=y_im,
                timestep_seq=timestep_seq,
                dps_weight=args.dps_weight,
                phys_weight=args.phys_weight,
                guidance_scale=args.guidance_scale,
                init_noise=init_noise,
                physics=physics_case,
                ref_prior=ref_prior_case,
                ref_weight=args.ref_weight,
                guidance_grad_clip=args.guidance_grad_clip,
                log_guidance=args.log_guidance,
                eps=args.eps,
            )

            prior_np = prior_core.detach().cpu().numpy().astype(np.float32)
            dps_np = dps_core.detach().cpu().numpy().astype(np.float32)

            prior_core_phys = (prior_np * std_np + mean_np).astype(np.float32)
            dps_core_phys = (dps_np * std_np + mean_np).astype(np.float32)
            key_case = (b_idx, m_idx)
            if key_case not in core_gt_cache:
                core_gt_cache[key_case] = _solve_lstsq_core_from_gt(
                    gt=gt,
                    phi_np=phi_np,
                    rx=rx,
                    ry=ry,
                    rcond=args.core_lstsq_rcond,
                )
            core_gt_lstsq = core_gt_cache[key_case]
            core_rel_prior = _core_rel_err(prior_core_phys, core_gt_lstsq, eps=args.eps)
            core_rel_dps = _core_rel_err(dps_core_phys, core_gt_lstsq, eps=args.eps)

            pred_prior = _decode_field_from_core_image(
                core_img_norm=prior_np,
                phi_np=phi_np,
                mean_np=mean_np,
                std_np=std_np,
                h=h,
                w=w,
            )
            pred_dps = _decode_field_from_core_image(
                core_img_norm=dps_np,
                phi_np=phi_np,
                mean_np=mean_np,
                std_np=std_np,
                h=h,
                w=w,
            )

            rmse_prior = _rel_rmse(pred_prior, gt, eps=args.eps)
            rmse_dps = _rel_rmse(pred_dps, gt, eps=args.eps)

            obs_rmse_prior = _masked_rel_rmse(pred_prior, gt, mask_case, eps=args.eps)
            obs_rmse_dps = _masked_rel_rmse(pred_dps, gt, mask_case, eps=args.eps)

            unobs_mask = ~mask_case.astype(bool)
            unobs_rmse_prior = _masked_rel_rmse(pred_prior, gt, unobs_mask, eps=args.eps)
            unobs_rmse_dps = _masked_rel_rmse(pred_dps, gt, unobs_mask, eps=args.eps)

            rows.append(
                {
                    "sample_idx": b_idx,
                    "freq_idx": m_idx,
                    "omega": omega_val,
                    "rmse_prior": rmse_prior,
                    "rmse_dps": rmse_dps,
                    "rmse_obs_prior": obs_rmse_prior,
                    "rmse_obs_dps": obs_rmse_dps,
                    "rmse_unobs_prior": unobs_rmse_prior,
                    "rmse_unobs_dps": unobs_rmse_dps,
                    "core_rel_err_prior": core_rel_prior,
                    "core_rel_err_dps": core_rel_dps,
                    "num_ref_cores": int(len(ref_ids)),
                }
            )

            freq_prior[m_idx].append(rmse_prior)
            freq_dps[m_idx].append(rmse_dps)

            if vis_count < args.num_visualize:
                vis_count += 1
                fig_path = out_dir / f"compare_case{vis_count:03d}_sample{b_idx:03d}_freq{m_idx:03d}.png"
                _plot_case(
                    out_path=fig_path,
                    gt=gt,
                    pred_prior=pred_prior,
                    pred_dps=pred_dps,
                    core_gt_lstsq=core_gt_lstsq,
                    core_dps=dps_core_phys,
                    core_rel_err_dps=core_rel_dps,
                    core_rel_err_prior=core_rel_prior,
                    mask_bool=mask_case,
                    sample_idx=b_idx,
                    freq_idx=m_idx,
                    omega_val=omega_val,
                )

            if args.log_every > 0 and (i_case % args.log_every == 0 or i_case == len(cases)):
                print(
                    f"[{i_case:04d}/{len(cases)}] sample={b_idx} freq={m_idx} "
                    f"prior={rmse_prior:.4e} dps={rmse_dps:.4e} "
                    f"num_ref={len(ref_ids)} "
                    f"core_rel_prior={core_rel_prior:.4e} core_rel_dps={core_rel_dps:.4e}"
                )

    if len(rows) == 0:
        raise RuntimeError("No valid evaluation rows were produced.")

    rows_sorted = sorted(rows, key=lambda x: (x["sample_idx"], x["freq_idx"]))
    csv_path = out_dir / "metrics_cases.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        header = (
            "sample_idx,freq_idx,omega,"
            "rmse_prior,rmse_dps,"
            "rmse_obs_prior,rmse_obs_dps,"
            "rmse_unobs_prior,rmse_unobs_dps,"
            "core_rel_err_prior,core_rel_err_dps,"
            "num_ref_cores\n"
        )
        f.write(header)
        for r_ in rows_sorted:
            f.write(
                f"{r_['sample_idx']},{r_['freq_idx']},{r_['omega']:.8g},"
                f"{r_['rmse_prior']:.8g},{r_['rmse_dps']:.8g},"
                f"{r_['rmse_obs_prior']:.8g},{r_['rmse_obs_dps']:.8g},"
                f"{r_['rmse_unobs_prior']:.8g},{r_['rmse_unobs_dps']:.8g},"
                f"{r_['core_rel_err_prior']:.8g},{r_['core_rel_err_dps']:.8g},"
                f"{r_['num_ref_cores']}\n"
            )

    unique_freqs = sorted({int(r_["freq_idx"]) for r_ in rows_sorted})
    freq_to_omega = {}
    for r_ in rows_sorted:
        fi = int(r_["freq_idx"])
        if fi not in freq_to_omega:
            freq_to_omega[fi] = float(r_["omega"])

    omega_curve = np.array([freq_to_omega[fi] for fi in unique_freqs], dtype=np.float32)
    prior_curve = np.array(
        [float(np.mean(freq_prior[fi])) if len(freq_prior[fi]) > 0 else np.nan for fi in unique_freqs],
        dtype=np.float32,
    )
    dps_curve = np.array(
        [float(np.mean(freq_dps[fi])) if len(freq_dps[fi]) > 0 else np.nan for fi in unique_freqs],
        dtype=np.float32,
    )

    _plot_freq_curve(
        out_path=out_dir / "freq_rmse_curve.png",
        omega=omega_curve,
        prior_vals=prior_curve,
        dps_vals=dps_curve,
    )

    mean_prior = float(np.mean([r_["rmse_prior"] for r_ in rows_sorted]))
    mean_dps = float(np.mean([r_["rmse_dps"] for r_ in rows_sorted]))
    mean_core_rel_prior = float(np.mean([r_["core_rel_err_prior"] for r_ in rows_sorted]))
    mean_core_rel_dps = float(np.mean([r_["core_rel_err_dps"] for r_ in rows_sorted]))
    improve = mean_prior - mean_dps
    improve_pct = 100.0 * improve / max(mean_prior, 1e-12)

    summary = {
        "diff_ckpt": str(args.diff_ckpt),
        "ftm_ckpt": str(args.ftm_ckpt),
        "data_h5": str(args.data_h5),
        "num_cases": int(len(rows_sorted)),
        "mean_rmse_prior": mean_prior,
        "mean_rmse_dps": mean_dps,
        "mean_core_rel_err_prior": mean_core_rel_prior,
        "mean_core_rel_err_dps": mean_core_rel_dps,
        "absolute_improvement": improve,
        "relative_improvement_percent": improve_pct,
        "dps_weight": float(args.dps_weight),
        "phys_weight": float(args.phys_weight),
        "ref_weight": float(args.ref_weight),
        "ref_num_each_side": int(args.ref_num_each_side),
        "ref_weight_mode": str(args.ref_weight_mode),
        "ref_weight_lambda": float(ref_weight_lambda),
        "ref_var_eps": float(args.ref_var_eps),
        "phys_use_source": bool(args.phys_use_source),
        "phys_interior_only": bool(args.phys_interior_only),
        "phys_scale_source_by_data": bool(args.phys_scale_source_by_data),
        "data_scale": float(data_scale),
        "guidance_scale": float(args.guidance_scale),
        "sample_steps": int(args.sample_steps),
        "output_dir": str(out_dir),
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nEvaluation finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved metrics: {csv_path}")
    print(f"Saved summary: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test conditional diffusion with DPS + local ref prior")

    p.add_argument("--diff_ckpt", type=str, default="ckp/diffusion_core_norm71.pt")
    p.add_argument("--ftm_ckpt", type=str, default="ckp/ftm_gpu_checkpoint.pt")
    p.add_argument("--data_h5", type=str, default="data_for_test/helmholtz_dataset_42_mask5_Extra.h5")
    p.add_argument("--out_dir", type=str, default="new_idea2.0/visual_data/diffusion_eval_mask_ratio5_Extra")

    p.add_argument("--freq_indices", type=str, default="")
    p.add_argument("--max_samples", type=int, default=1)
    p.add_argument("--max_cases", type=int, default=0)

    p.add_argument("--sample_steps", type=int, default=500)
    p.add_argument("--dps_weight", type=float, default=1.0)
    p.add_argument("--phys_weight", type=float, default=0.001)
    p.add_argument("--ref_weight", type=float, default=0)
    p.add_argument("--ref_num_each_side", type=int, default=1, choices=[1, 2, 3])
    p.add_argument(
        "--ref_weight_mode",
        type=str,
        default="two_sided_linear",
        choices=["distance_softmax", "uniform", "two_sided_linear"],
    )
    p.add_argument(
        "--ref_weight_lambda",
        type=float,
        default=-1.0,
        help="If <=0, auto set to 1/avg_frequency_spacing",
    )
    p.add_argument("--ref_var_eps", type=float, default=1e-4)
    p.add_argument("--guidance_scale", type=float, default=15)
    p.add_argument("--guidance_grad_clip", type=float, default=1e-3)

    p.add_argument("--phys_use_source", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--phys_scale_source_by_data", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--phys_interior_only", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--helmholtz_L", type=float, default=1.0)
    p.add_argument("--helmholtz_c", type=float, default=1.0)
    p.add_argument("--helmholtz_pml_width", type=float, default=0.12)
    p.add_argument("--helmholtz_sigma_max", type=float, default=50.0)
    p.add_argument("--source_sigma", type=float, default=0.025)

    p.add_argument("--num_visualize", type=int, default=70)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--log_guidance", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--core_lstsq_rcond", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
