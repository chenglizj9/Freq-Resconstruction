"""
test_diffusion.py  (3D Helmholtz edition)
------------------------------------------
Evaluate conditional diffusion + DPS for 3D frequency-domain field reconstruction.

Pipeline
--------
1) Load trained diffusion model p(G | omega) on 3D FTM latent cores.
2) Load 3D FTM basis networks (net_x, net_y, net_z) from FTM checkpoint.
3) Build Phi_3d (Nx*Ny*Nz, Rx*Ry*Rz) — fixed feature matrix.
4) For each test case (sample, frequency):
   - Prior sampling: run DDPM without guidance
   - DPS-guided sampling: closed-form gradient ∇_G ||M(Phi·G) - y||^2
5) Decode G → field and report metrics + 3D full-volume plots.

Key advantage: Phi is fixed → DPS gradient is a closed-form matrix product,
no backprop through decoder needed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import h5py
import matplotlib
import numpy as np
import torch

from train_diffusion import (
    UNet3DScoreNet, build_linear_schedule, build_cosine_schedule
)
from train_FTM_GPU import MLP1D, build_phi_3d, normalize_coords_to_unit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vis3d import save_3d_visuals


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_omega(omega: float, omega_min: float, omega_max: float) -> float:
    return float((omega - omega_min) / max(omega_max - omega_min, 1e-12))


def _rel_rmse(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.sum((pred - gt) ** 2) / max(np.sum(gt ** 2), eps)))


def _masked_rel_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> float:
    m = mask.astype(bool)
    if not np.any(m): return float("nan")
    return float(np.sqrt(np.sum((pred - gt) ** 2 * m) / max(np.sum(gt ** 2 * m), eps)))


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


def _decode_field_torch(z_phys: torch.Tensor, phi_t: torch.Tensor, Nx: int, Ny: int, Nz: int) -> torch.Tensor:
    """z_phys: (1, 2R) or (2R,), phi_t: (P, R) -> (Nx, Ny, Nz, 2) torch tensor."""
    if z_phys.ndim == 2:
        z_phys = z_phys[0]
    R = phi_t.shape[1]
    g_re = z_phys[:R]
    g_im = z_phys[R:]
    pred_re = (phi_t @ g_re).reshape(Nx, Ny, Nz)
    pred_im = (phi_t @ g_im).reshape(Nx, Ny, Nz)
    return torch.stack([pred_re, pred_im], dim=-1)


def _pde_residual_loss_3d(
    z_phys: torch.Tensor,
    phi_t: torch.Tensor,
    Nx: int,
    Ny: int,
    Nz: int,
    omega: float,
    L: float,
    c: float,
    pml_frac: float,
    sigma_max: float,
    data_scale: float,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Differentiable 3D Helmholtz residual on decoded field u.

    PDE: Δu + k^2 u = f, with simple PML-like damping term on the mass coefficient:
         Δu + (k^2 + i * omega * sigma(x)) u = f

    We use a fixed Gaussian source near the left side of the volume because the 3D dataset
    metadata stores only the number of sources, not their positions/amplitudes.
    This keeps the guidance physically meaningful and stable across mask-ratio runs.
    """
    device = z_phys.device
    dtype = z_phys.dtype
    pred = _decode_field_torch(z_phys, phi_t, Nx, Ny, Nz) * float(data_scale)
    u_re = pred[..., 0]
    u_im = pred[..., 1]

    dx = float(L) / max(Nx - 1, 1)
    dy = float(L) / max(Ny - 1, 1)
    dz = float(L) / max(Nz - 1, 1)

    u_re_5 = u_re.unsqueeze(0).unsqueeze(0)
    u_im_5 = u_im.unsqueeze(0).unsqueeze(0)

    kernel_x = torch.tensor([1.0, -2.0, 1.0], device=device, dtype=dtype).view(1, 1, 3, 1, 1) / (dx * dx)
    kernel_y = torch.tensor([1.0, -2.0, 1.0], device=device, dtype=dtype).view(1, 1, 1, 3, 1) / (dy * dy)
    kernel_z = torch.tensor([1.0, -2.0, 1.0], device=device, dtype=dtype).view(1, 1, 1, 1, 3) / (dz * dz)

    lap_re = (
        F.conv3d(F.pad(u_re_5, (0, 0, 0, 0, 1, 1), mode="replicate"), kernel_x)
        + F.conv3d(F.pad(u_re_5, (0, 0, 1, 1, 0, 0), mode="replicate"), kernel_y)
        + F.conv3d(F.pad(u_re_5, (1, 1, 0, 0, 0, 0), mode="replicate"), kernel_z)
    )[0, 0]
    lap_im = (
        F.conv3d(F.pad(u_im_5, (0, 0, 0, 0, 1, 1), mode="replicate"), kernel_x)
        + F.conv3d(F.pad(u_im_5, (0, 0, 1, 1, 0, 0), mode="replicate"), kernel_y)
        + F.conv3d(F.pad(u_im_5, (1, 1, 0, 0, 0, 0), mode="replicate"), kernel_z)
    )[0, 0]

    x = torch.linspace(0.0, float(L), Nx, device=device, dtype=dtype)
    y = torch.linspace(0.0, float(L), Ny, device=device, dtype=dtype)
    z = torch.linspace(0.0, float(L), Nz, device=device, dtype=dtype)
    pml_d = float(pml_frac) * float(L)

    def _sigma_1d(coord: torch.Tensor) -> torch.Tensor:
        sig = torch.zeros_like(coord)
        if pml_d <= 0:
            return sig
        left = coord < pml_d
        right = coord > (float(L) - pml_d)
        sig[left] = float(sigma_max) * ((pml_d - coord[left]) / pml_d) ** 2
        sig[right] = float(sigma_max) * ((coord[right] - (float(L) - pml_d)) / pml_d) ** 2
        return sig

    sx = _sigma_1d(x).view(Nx, 1, 1)
    sy = _sigma_1d(y).view(1, Ny, 1)
    sz = _sigma_1d(z).view(1, 1, Nz)
    sigma = sx + sy + sz

    k2 = (float(omega) / max(float(c), 1e-12)) ** 2

    src_x = float(L) * 0.30
    src_y = float(L) * 0.50
    src_z = float(L) * 0.50
    src_sigma = 0.05 * float(L)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    f_re = torch.exp(-((X - src_x) ** 2 + (Y - src_y) ** 2 + (Z - src_z) ** 2) / (2.0 * src_sigma * src_sigma))
    f_im = torch.zeros_like(f_re)

    res_re = lap_re + k2 * u_re - float(omega) * sigma * u_im - f_re
    res_im = lap_im + k2 * u_im + float(omega) * sigma * u_re - f_im

    interior = torch.ones((Nx, Ny, Nz), device=device, dtype=torch.bool)
    interior[[0, -1], :, :] = False
    interior[:, [0, -1], :] = False
    interior[:, :, [0, -1]] = False

    num = torch.mean((res_re[interior] ** 2 + res_im[interior] ** 2))
    den = torch.clamp(torch.mean(f_re[interior] ** 2 + f_im[interior] ** 2), min=eps)
    return num / den


# ─────────────────────────────────────────────────────────────────────────────
# Load checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def _load_diffusion(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model_cfg  = ckpt["model_config"]
    diff_cfg   = ckpt["diffusion_config"]
    core_stats = ckpt["core_stats"]

    # Strip metadata keys not accepted by the constructor; restore tuple type for channel_mults.
    unet_kwargs = {k: v for k, v in model_cfg.items() if k != "type"}
    unet_kwargs["channel_mults"] = tuple(unet_kwargs.get("channel_mults", (1, 2, 4, 8)))
    model = UNet3DScoreNet(**unet_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    sched_name = str(diff_cfg.get("schedule", "cosine")).lower()
    if sched_name == "linear":
        sched = build_linear_schedule(diff_cfg["num_steps"], diff_cfg["beta_start"],
                                      diff_cfg["beta_end"], device)
    else:
        sched = build_cosine_schedule(diff_cfg["num_steps"], device=device)

    mean = torch.from_numpy(_to_numpy(core_stats["mean"]).astype(np.float32)).to(device)
    std  = torch.from_numpy(_to_numpy(core_stats["std"]).astype(np.float32)).to(device)

    return {
        "model": model, "schedule": sched,
        "mean": mean, "std": std,
        "rx": int(core_stats["rx"]), "ry": int(core_stats["ry"]), "rz": int(core_stats["rz"]),
        "R": int(core_stats["R"]), "latent_dim": int(core_stats["latent_dim"]),
        "omega_min": float(ckpt["omega_stats"]["min"]),
        "omega_max": float(ckpt["omega_stats"]["max"]),
        "num_steps": int(diff_cfg["num_steps"]),
    }


def _load_ftm_basis(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"FTM checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt.get("config", {})
    for k in ("rank_x", "rank_y", "rank_z", "hidden_dim", "hidden_layers", "activation"):
        if k not in cfg:
            raise KeyError(f"FTM config missing key: {k}")

    def _net(rank: int) -> MLP1D:
        return MLP1D(int(rank), int(cfg["hidden_dim"]), int(cfg["hidden_layers"]),
                     str(cfg["activation"])).to(device)

    net_x = _net(cfg["rank_x"]); net_x.load_state_dict(ckpt["net_x_state"]); net_x.eval()
    net_y = _net(cfg["rank_y"]); net_y.load_state_dict(ckpt["net_y_state"]); net_y.eval()
    net_z = _net(cfg["rank_z"]); net_z.load_state_dict(ckpt["net_z_state"]); net_z.eval()
    return {
        "net_x": net_x, "net_y": net_y, "net_z": net_z,
        "normalize_coords": bool(cfg.get("normalize_coords", True)),
        "rank_x": int(cfg["rank_x"]), "rank_y": int(cfg["rank_y"]), "rank_z": int(cfg["rank_z"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DDPM sampling with closed-form DPS guidance
# ─────────────────────────────────────────────────────────────────────────────

def _build_timestep_seq(total: int, sample_steps: int) -> List[int]:
    if sample_steps <= 0 or sample_steps >= total:
        return list(range(total - 1, -1, -1))
    seq = np.linspace(total - 1, 0, sample_steps, dtype=np.int64)
    out, seen = [], set()
    for t in seq:
        if int(t) not in seen: out.append(int(t)); seen.add(int(t))
    return out


def sample_core(
    model: UNet3DScoreNet,
    sched: Dict[str, torch.Tensor],
    omega_norm_val: float,
    mean: torch.Tensor,    # (1, latent_dim)
    std:  torch.Tensor,    # (1, latent_dim)
    phi_obs_re: torch.Tensor,   # (n_obs, R) rows of phi for observed indices
    phi_obs_im: torch.Tensor,
    y_re: torch.Tensor,         # (n_obs,)
    y_im: torch.Tensor,
    timestep_seq: List[int],
    dps_weight: float,
    phys_weight: float,
    guidance_scale: float,
    init_noise: torch.Tensor,   # (1, latent_dim)
    phi_full: Optional[torch.Tensor] = None,
    physics_cfg: Optional[Dict[str, float]] = None,
    guidance_grad_clip: float = 1e-3,
    pde_loss_clip: float = 10.0,
    eps: float = 1e-4,
    log_guidance: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Sample one core vector (latent_dim,) = (2*R,).

    Observation term stays in latent/core space (closed-form).
    PDE term decodes Phi·G to the 3D field and applies a differentiable
    finite-difference 3D Helmholtz residual.
    """
    R = phi_obs_re.shape[1]
    x = init_noise.clone()  # (1, 2*R)
    omega_cond = torch.tensor([[omega_norm_val]], device=device, dtype=torch.float32)
    use_phys = phys_weight > 0.0 and phi_full is not None and physics_cfg is not None

    for i, t_idx in enumerate(timestep_seq):
        t = torch.full((1,), t_idx, device=device, dtype=torch.long)
        with torch.no_grad():
            eps_pred = model(x, t, omega_cond)
            abar_t   = sched["alpha_bars"][t_idx]
            x0_hat   = (x - torch.sqrt(torch.clamp(1.0 - abar_t, min=1e-12)) * eps_pred) / \
                       torch.sqrt(torch.clamp(abar_t, min=1e-12))

        safe_step = len(timestep_seq) * 0.4
        is_safe_step = i < len(timestep_seq) - safe_step
        more_safe_step = i < len(timestep_seq) - safe_step * 0.5
        use_dps = dps_weight > 0.0

        if use_dps or use_phys:
            x0_var = x0_hat.detach().requires_grad_(True)
            x0_phys = x0_var * std + mean                    # (1, 2R)
            g_re = x0_phys[0, :R]
            g_im = x0_phys[0, R:]

            total_loss = torch.zeros((), device=device, dtype=torch.float32)
            like_loss: Optional[torch.Tensor] = None
            pde_loss: Optional[torch.Tensor] = None

            if use_dps:
                pred_re_obs = phi_obs_re @ g_re
                pred_im_obs = phi_obs_im @ g_im
                rel_re = torch.mean((pred_re_obs - y_re) ** 2) / (torch.mean(y_re ** 2) + eps)
                rel_im = torch.mean((pred_im_obs - y_im) ** 2) / (torch.mean(y_im ** 2) + eps)
                like_loss = rel_re + rel_im
                if not more_safe_step:
                    like_loss = like_loss * 0.5
                total_loss = total_loss + dps_weight * like_loss

            if use_phys:
                pde_loss = _pde_residual_loss_3d(
                    z_phys=x0_phys,
                    phi_t=phi_full,
                    Nx=int(physics_cfg["Nx"]),
                    Ny=int(physics_cfg["Ny"]),
                    Nz=int(physics_cfg["Nz"]),
                    omega=float(physics_cfg["omega"]),
                    L=float(physics_cfg["L"]),
                    c=float(physics_cfg["c"]),
                    pml_frac=float(physics_cfg["pml_frac"]),
                    sigma_max=float(physics_cfg["sigma_max"]),
                    data_scale=float(physics_cfg["data_scale"]),
                    eps=eps,
                )
                pde_loss = torch.clamp(pde_loss, max=pde_loss_clip)
                if more_safe_step:
                    pde_loss = pde_loss * 3.0
                total_loss = total_loss + phys_weight * pde_loss

            grad_z = torch.autograd.grad(total_loss, x0_var, retain_graph=False, create_graph=False)[0]

            time_weight = 1.0 - (t_idx / max(timestep_seq[0], 1))
            if omega_norm_val < 0.2:
                grad_z = grad_z * 2.0
                time_weight = 1.0
            if omega_norm_val > 0.8:
                grad_z = grad_z * 1.3

            if not is_safe_step and guidance_grad_clip > 0.0:
                grad_z = torch.clamp(grad_z, -guidance_grad_clip, guidance_grad_clip)
            if not more_safe_step and guidance_grad_clip > 0.0:
                grad_z = torch.clamp(grad_z, -guidance_grad_clip * 0.5, guidance_grad_clip * 0.5)

            x0_hat = x0_hat - time_weight * guidance_scale * grad_z

            if log_guidance and i % 20 == 0:
                msg = f"  step={t_idx:4d}"
                if like_loss is not None:
                    msg += f" like={like_loss.item():.3e}"
                if pde_loss is not None:
                    msg += f" pde={pde_loss.item():.3e}"
                msg += f" grad_max={grad_z.abs().max().item():.3e}"
                print(msg)

        if i == len(timestep_seq) - 1:
            x = x0_hat
        else:
            t_prev   = timestep_seq[i + 1]
            abar_prev = sched["alpha_bars"][t_prev]
            with torch.no_grad():
                x = (torch.sqrt(torch.clamp(abar_prev, min=1e-12)) * x0_hat
                     + torch.sqrt(torch.clamp(1.0 - abar_prev, min=1e-12)) * eps_pred)

    return x.squeeze(0)   # (2*R,)


# ─────────────────────────────────────────────────────────────────────────────
# Decode & visualize
# ─────────────────────────────────────────────────────────────────────────────

def _decode_field(z_phys: np.ndarray, phi_np: np.ndarray, Nx: int, Ny: int, Nz: int) -> np.ndarray:
    """z_phys: (2*R,), phi_np: (P, R) → field (Nx, Ny, Nz, 2)."""
    R = phi_np.shape[1]
    g_re = z_phys[:R]
    g_im = z_phys[R:]
    pred_re = (phi_np @ g_re).reshape(Nx, Ny, Nz)
    pred_im = (phi_np @ g_im).reshape(Nx, Ny, Nz)
    return np.stack([pred_re, pred_im], axis=-1).astype(np.float32)




def _plot_freq_curve(out_path: Path, omega: np.ndarray,
                     prior_vals: np.ndarray, dps_vals: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(omega, prior_vals, label="Prior", lw=1.8)
    ax.plot(omega, dps_vals,   label="DPS",   lw=1.8)
    ax.set_xlabel("omega"); ax.set_ylabel("Mean relative RMSE")
    ax.set_title("Frequency-wise Reconstruction Error (3D Helmholtz)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _parse_freq_indices(text: str, num_freqs: int) -> List[int]:
    if text.strip() == "": return list(range(num_freqs))
    out = []
    for p in text.split(","):
        p = p.strip()
        if not p: continue
        idx = int(p)
        if not (0 <= idx < num_freqs): raise ValueError(f"freq idx out of range: {idx}")
        out.append(idx)
    return sorted(set(out))


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"
                          if args.device == "auto" else args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diff = _load_diffusion(Path(args.diff_ckpt), device)
    ftm  = _load_ftm_basis(Path(args.ftm_ckpt),  device)

    model = diff["model"]; sched = diff["schedule"]
    mean  = diff["mean"];  std   = diff["std"]
    omega_min = diff["omega_min"]; omega_max = diff["omega_max"]
    R = diff["R"]; latent_dim = diff["latent_dim"]

    timestep_seq = _build_timestep_seq(diff["num_steps"], args.sample_steps)

    data_h5_path = Path(args.data_h5)
    with h5py.File(data_h5_path, "r") as f:
        for k in ("data", "mask_tr", "omega"):
            if k not in f: raise KeyError(f"HDF5 missing '{k}'")

        omega       = f["omega"][...].astype(np.float32)
        data_ds     = f["data"]        # (B, M, Nx, Ny, Nz, 2)
        mask_ds     = f["mask_tr"]

        b_data, m_data, Nx, Ny, Nz, C = data_ds.shape
        assert C == 2, "Only C=2 (real+imag) supported."

        freq_ids = _parse_freq_indices(args.freq_indices, m_data)
        b_eval   = b_data if args.max_samples <= 0 else min(b_data, args.max_samples)

        all_cases = [(b, m) for b in range(b_eval) for m in freq_ids]
        if args.max_cases > 0 and len(all_cases) > args.max_cases:
            rng = np.random.default_rng(args.seed)
            all_cases = [all_cases[int(i)] for i in
                         rng.choice(len(all_cases), size=args.max_cases, replace=False)]

        # Build grids
        x_grid = (f["grid_x"][...].astype(np.float32) if "grid_x" in f
                   else np.linspace(0, 1, Nx, dtype=np.float32))
        y_grid = (f["grid_y"][...].astype(np.float32) if "grid_y" in f
                   else np.linspace(0, 1, Ny, dtype=np.float32))
        z_grid = (f["grid_z"][...].astype(np.float32) if "grid_z" in f
                   else np.linspace(0, 1, Nz, dtype=np.float32))

        if ftm["normalize_coords"]:
            x_np = normalize_coords_to_unit(x_grid.astype(np.float64)).astype(np.float32)
            y_np = normalize_coords_to_unit(y_grid.astype(np.float64)).astype(np.float32)
            z_np = normalize_coords_to_unit(z_grid.astype(np.float64)).astype(np.float32)
        else:
            x_np, y_np, z_np = x_grid, y_grid, z_grid

        x_t = torch.from_numpy(x_np).unsqueeze(-1).to(device)
        y_t = torch.from_numpy(y_np).unsqueeze(-1).to(device)
        z_t = torch.from_numpy(z_np).unsqueeze(-1).to(device)

        with torch.no_grad():
            phi_t = build_phi_3d(ftm["net_x"], ftm["net_y"], ftm["net_z"], x_t, y_t, z_t)
        phi_np = phi_t.cpu().numpy().astype(np.float32)  # (Nx*Ny*Nz, R)
        h5_meta = _load_h5_metadata_dict(f)
        data_scale = max(_load_data_scale(data_h5_path, f), 1e-12)

        pde_L = float(h5_meta.get("L", args.helmholtz_L))
        pde_c = float(h5_meta.get("c", args.helmholtz_c))
        pde_pml_frac = float(h5_meta.get("pml_frac", args.helmholtz_pml_frac))
        pde_sigma_max = float(h5_meta.get("sigma_max", args.helmholtz_sigma_max))

        if phi_np.shape[1] != R:
            raise ValueError(f"Phi rank mismatch: phi has R={phi_np.shape[1]}, expected R={R}")

        rows: List[Dict] = []
        vis_count = 0
        freq_prior: Dict[int, List[float]] = {m: [] for m in freq_ids}
        freq_dps:   Dict[int, List[float]] = {m: [] for m in freq_ids}
        pde_prior_vals: List[float] = []
        pde_dps_vals: List[float] = []

        for i_case, (b_idx, m_idx) in enumerate(all_cases, start=1):
            gt       = data_ds[b_idx, m_idx].astype(np.float32)   # (Nx, Ny, Nz, 2)
            omega_val = float(omega[m_idx])
            omega_cond_val = _normalize_omega(omega_val, omega_min, omega_max)

            # Mask
            if mask_ds.ndim == 5:      # (M, Nx, Ny, Nz, C) shared
                mask_case = (mask_ds[m_idx].astype(np.float32) > 0.5)
            elif mask_ds.ndim == 6:    # (B, M, Nx, Ny, Nz, C) per-sample
                mask_case = (mask_ds[b_idx, m_idx].astype(np.float32) > 0.5)
            else:
                raise ValueError(f"Unexpected mask dims: {mask_ds.ndim}")

            idx_re = np.flatnonzero(mask_case[..., 0].reshape(-1))
            idx_im = np.flatnonzero(mask_case[..., 1].reshape(-1))
            if idx_re.size == 0 or idx_im.size == 0:
                print(f"  [WARN] case ({b_idx},{m_idx}): empty mask, skip.")
                continue

            phi_obs_re_t = torch.from_numpy(phi_np[idx_re]).to(device)   # (n_obs, R)
            phi_obs_im_t = torch.from_numpy(phi_np[idx_im]).to(device)
            y_re_t = torch.from_numpy(gt[..., 0].reshape(-1)[idx_re]).to(device)
            y_im_t = torch.from_numpy(gt[..., 1].reshape(-1)[idx_im]).to(device)

            g = torch.Generator(device=device)
            g.manual_seed(args.seed + 17 * i_case)
            init_noise = torch.randn((1, latent_dim), generator=g, device=device)

            prior_z = sample_core(
                model=model, sched=sched, omega_norm_val=omega_cond_val,
                mean=mean, std=std,
                phi_obs_re=phi_obs_re_t, phi_obs_im=phi_obs_im_t,
                y_re=y_re_t, y_im=y_im_t,
                timestep_seq=timestep_seq, dps_weight=0.0, phys_weight=0.0,
                guidance_scale=args.guidance_scale,
                init_noise=init_noise, phi_full=None, physics_cfg=None,
                guidance_grad_clip=args.guidance_grad_clip,
                eps=args.eps, log_guidance=False, device=device,
            )
            dps_z = sample_core(
                model=model, sched=sched, omega_norm_val=omega_cond_val,
                mean=mean, std=std,
                phi_obs_re=phi_obs_re_t, phi_obs_im=phi_obs_im_t,
                y_re=y_re_t, y_im=y_im_t,
                timestep_seq=timestep_seq, dps_weight=args.dps_weight,
                phys_weight=args.phys_weight, guidance_scale=args.guidance_scale,
                init_noise=init_noise,
                phi_full=phi_t,
                physics_cfg={
                    "Nx": Nx, "Ny": Ny, "Nz": Nz,
                    "omega": omega_val,
                    "L": pde_L,
                    "c": pde_c,
                    "pml_frac": pde_pml_frac,
                    "sigma_max": pde_sigma_max,
                    "data_scale": data_scale,
                },
                guidance_grad_clip=args.guidance_grad_clip,
                pde_loss_clip=args.pde_loss_clip,
                eps=args.eps, log_guidance=args.log_guidance, device=device,
            )

            # Denormalize
            mean_np = mean.cpu().numpy()[0]
            std_np  = std.cpu().numpy()[0]
            prior_phys = prior_z.cpu().numpy() * std_np + mean_np   # (2*R,)
            dps_phys   = dps_z.cpu().numpy()   * std_np + mean_np

            pred_prior = _decode_field(prior_phys, phi_np, Nx, Ny, Nz)
            pred_dps   = _decode_field(dps_phys,   phi_np, Nx, Ny, Nz)

            rmse_prior = _rel_rmse(pred_prior, gt)
            rmse_dps   = _rel_rmse(pred_dps,   gt)

            with torch.no_grad():
                pde_prior = float(_pde_residual_loss_3d(
                    z_phys=torch.from_numpy(prior_phys).to(device=device, dtype=mean.dtype),
                    phi_t=phi_t,
                    Nx=Nx, Ny=Ny, Nz=Nz,
                    omega=omega_val,
                    L=pde_L, c=pde_c,
                    pml_frac=pde_pml_frac,
                    sigma_max=pde_sigma_max,
                    data_scale=data_scale,
                    eps=args.eps,
                ).item())
                pde_dps = float(_pde_residual_loss_3d(
                    z_phys=torch.from_numpy(dps_phys).to(device=device, dtype=mean.dtype),
                    phi_t=phi_t,
                    Nx=Nx, Ny=Ny, Nz=Nz,
                    omega=omega_val,
                    L=pde_L, c=pde_c,
                    pml_frac=pde_pml_frac,
                    sigma_max=pde_sigma_max,
                    data_scale=data_scale,
                    eps=args.eps,
                ).item())

            pde_prior_vals.append(pde_prior)
            pde_dps_vals.append(pde_dps)

            obs_mask_np  = mask_case.astype(np.float32)
            unobs_mask_np = 1.0 - obs_mask_np
            obs_rmse_prior  = _masked_rel_rmse(pred_prior, gt, obs_mask_np)
            obs_rmse_dps    = _masked_rel_rmse(pred_dps,   gt, obs_mask_np)
            unobs_rmse_prior = _masked_rel_rmse(pred_prior, gt, unobs_mask_np)
            unobs_rmse_dps   = _masked_rel_rmse(pred_dps,   gt, unobs_mask_np)

            rows.append({
                "sample_idx": b_idx, "freq_idx": m_idx, "omega": omega_val,
                "rmse_prior": rmse_prior, "rmse_dps": rmse_dps,
                "obs_rmse_prior": obs_rmse_prior, "obs_rmse_dps": obs_rmse_dps,
                "unobs_rmse_prior": unobs_rmse_prior, "unobs_rmse_dps": unobs_rmse_dps,
                "pde_res_prior": pde_prior, "pde_res_dps": pde_dps,
            })
            freq_prior[m_idx].append(rmse_prior)
            freq_dps[m_idx].append(rmse_dps)

            if vis_count < args.num_visualize:
                vis_count += 1
                case_stem = out_dir / f"case{vis_count:03d}_s{b_idx:03d}_f{m_idx:03d}"
                title = f"sample={b_idx} freq_idx={m_idx} ω={omega_val:.3f}"
                save_3d_visuals(
                    gt, pred_prior, mask_case[..., 0],
                    stem=case_stem.with_name(case_stem.name + "_prior"),
                    title=title + "  Prior",
                )
                save_3d_visuals(
                    gt, pred_dps, mask_case[..., 0],
                    stem=case_stem.with_name(case_stem.name + "_dps"),
                    title=title + "  DPS",
                )

            if args.log_every > 0 and (i_case % args.log_every == 0 or i_case == len(all_cases)):
                print(f"  [{i_case:04d}/{len(all_cases)}] s={b_idx} f={m_idx} "
                      f"prior={rmse_prior:.4e} dps={rmse_dps:.4e}")

    if not rows:
        raise RuntimeError("No evaluation rows produced.")

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = out_dir / "metrics_cases.csv"
    with open(csv_path, "w") as fcsv:
        fcsv.write("sample_idx,freq_idx,omega,rmse_prior,rmse_dps,"
                   "obs_rmse_prior,obs_rmse_dps,unobs_rmse_prior,unobs_rmse_dps,"
                   "pde_res_prior,pde_res_dps\n")
        for r in sorted(rows, key=lambda x: (x["sample_idx"], x["freq_idx"])):
            fcsv.write(f"{r['sample_idx']},{r['freq_idx']},{r['omega']:.8g},"
                       f"{r['rmse_prior']:.8g},{r['rmse_dps']:.8g},"
                       f"{r['obs_rmse_prior']:.8g},{r['obs_rmse_dps']:.8g},"
                       f"{r['unobs_rmse_prior']:.8g},{r['unobs_rmse_dps']:.8g},"
                       f"{r['pde_res_prior']:.8g},{r['pde_res_dps']:.8g}\n")

    # ── Frequency curve plot ──────────────────────────────────────────────
    unique_freqs = sorted({int(r["freq_idx"]) for r in rows})
    fi_to_om     = {int(r["freq_idx"]): float(r["omega"]) for r in rows}
    omega_curve  = np.array([fi_to_om[fi] for fi in unique_freqs])
    prior_curve  = np.array([np.mean(freq_prior[fi]) if freq_prior[fi] else np.nan for fi in unique_freqs])
    dps_curve    = np.array([np.mean(freq_dps[fi])   if freq_dps[fi]   else np.nan for fi in unique_freqs])
    _plot_freq_curve(out_dir / "freq_rmse_curve.png", omega_curve, prior_curve, dps_curve)

    mean_prior = float(np.mean([r["rmse_prior"] for r in rows]))
    mean_dps   = float(np.mean([r["rmse_dps"]   for r in rows]))
    improve_pct = 100.0 * (mean_prior - mean_dps) / max(mean_prior, 1e-12)

    summary = {
        "diff_ckpt": str(args.diff_ckpt), "ftm_ckpt": str(args.ftm_ckpt),
        "data_h5": str(args.data_h5), "num_cases": len(rows),
        "mean_rmse_prior": mean_prior, "mean_rmse_dps": mean_dps,
        "mean_pde_res_prior": float(np.mean(pde_prior_vals)) if pde_prior_vals else float("nan"),
        "mean_pde_res_dps": float(np.mean(pde_dps_vals)) if pde_dps_vals else float("nan"),
        "relative_improvement_percent": improve_pct,
        "dps_weight": float(args.dps_weight),
        "phys_weight": float(args.phys_weight),
        "guidance_scale": float(args.guidance_scale),
        "sample_steps": int(args.sample_steps),
        "output_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print("\nEvaluation finished.")
    print(json.dumps(summary, indent=2))
    print(f"CSV  : {csv_path}")
    print(f"Dir  : {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test 3D diffusion + DPS reconstruction")
    p.add_argument("--diff_ckpt",  type=str, default="ckp/diffusion3d.pt")
    p.add_argument("--ftm_ckpt",   type=str, default="ckp/ftm3d.pt")
    p.add_argument("--data_h5",    type=str, default="helmholtz3d_dataset_msk0.1.h5")
    p.add_argument("--out_dir",    type=str, default="visual_data/diffusion3d_eval_msk0.1")
    p.add_argument("--freq_indices", type=str, default="")
    p.add_argument("--max_samples",  type=int, default=5)
    p.add_argument("--max_cases",    type=int, default=0)
    p.add_argument("--sample_steps", type=int, default=500)
    p.add_argument("--dps_weight",   type=float, default=25.0)
    p.add_argument("--phys_weight",  type=float, default=0.30)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--guidance_grad_clip", type=float, default=1e-2)
    p.add_argument("--pde_loss_clip", type=float, default=10.0)
    p.add_argument("--helmholtz_L", type=float, default=1.0)
    p.add_argument("--helmholtz_c", type=float, default=1.0)
    p.add_argument("--helmholtz_pml_frac", type=float, default=0.15)
    p.add_argument("--helmholtz_sigma_max", type=float, default=40.0)
    p.add_argument("--num_visualize", type=int, default=10)
    p.add_argument("--log_every",     type=int, default=1)
    p.add_argument("--log_guidance",  action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--eps",   type=float, default=1e-4)
    p.add_argument("--seed",  type=int,   default=42)
    p.add_argument("--device",type=str,   default="auto")
    return p


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
