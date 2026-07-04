"""
test_diffusion.py  (elastic wave edition)
------------------------------------------
Evaluate conditional diffusion + DPS for elastic wave field reconstruction.

Pipeline
--------
1) Load trained diffusion model p(G | omega) on C-channel core images.
   C=4 for elastic wave: [ux_re, ux_im, uy_re, uy_im]
2) Load FTM basis networks (shared net_x / net_y) from FTM checkpoint.
3) For each test case (sample, frequency), run:
   - Prior sampling (no guidance)
   - DPS-guided sampling using sparse observations
4) Decode cores → fields and report metrics / plots.

Physics residual (elastic wave)
--------------------------------
The 2D frequency-domain elastic wave system is:
  [ A11  A12 ] [ ux ]   [ fx ]
  [ A21  A22 ] [ uy ] = [ fy ]

where A_ij are complex sparse N²×N² matrices depending on (λ, μ, ρ, ω).
The material fields (λ, μ, ρ) are loaded per sample from the HDF5 file.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib
import numpy as np
import scipy.sparse as sp
import torch

from train_diffusion import ConditionalUNet2D, build_cosine_schedule, build_linear_schedule
from train_FTM_GPU import MLP1D, build_phi, normalize_coords_to_unit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

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
    return float((omega - omega_min) / max(omega_max - omega_min, 1e-12))


def _parse_freq_indices(text: str, num_freqs: int) -> List[int]:
    if not text.strip():
        return list(range(num_freqs))
    out = []
    for p in text.split(","):
        p = p.strip()
        if p:
            idx = int(p)
            if not (0 <= idx < num_freqs):
                raise ValueError(f"freq index {idx} out of range [0, {num_freqs-1}]")
            out.append(idx)
    return sorted(set(out)) or list(range(num_freqs))


def _build_timestep_sequence(total_steps: int, sample_steps: int) -> List[int]:
    if sample_steps <= 0 or sample_steps >= total_steps:
        return list(range(total_steps - 1, -1, -1))
    seq = np.linspace(total_steps - 1, 0, sample_steps, dtype=np.int64).tolist()
    seen, out = set(), []
    for t in seq:
        if t not in seen:
            out.append(int(t)); seen.add(int(t))
    return out


def _load_h5_metadata_dict(f: h5py.File) -> Dict[str, Any]:
    if "metadata" not in f:
        return {}
    raw = f["metadata"][()]
    if isinstance(raw, (bytes, np.bytes_)):
        text = raw.decode("utf-8")
    elif isinstance(raw, np.ndarray) and raw.shape == ():
        s = raw.item()
        text = s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s)
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


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_diffusion(diff_ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    """Load diffusion model checkpoint.  Supports any C (C=2 or C=4)."""
    if not diff_ckpt_path.exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {diff_ckpt_path}")

    ckpt      = torch.load(diff_ckpt_path, map_location="cpu")
    model_cfg = ckpt["model_config"]
    diff_cfg  = ckpt["diffusion_config"]
    cs        = ckpt["core_stats"]

    model = ConditionalUNet2D(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    sched = diff_cfg.get("schedule", "linear").lower()
    if sched == "linear":
        schedule = build_linear_schedule(
            int(diff_cfg["num_steps"]), float(diff_cfg["beta_start"]),
            float(diff_cfg["beta_end"]), device,
        )
    elif sched == "cosine":
        schedule = build_cosine_schedule(int(diff_cfg["num_steps"]), device=device)
    else:
        raise ValueError(f"Unknown diffusion schedule: {sched}")

    mean = torch.from_numpy(_to_numpy(cs["mean"]).astype(np.float32)).to(device)
    std  = torch.from_numpy(_to_numpy(cs["std"]).astype(np.float32)).to(device)
    rx   = int(cs["rx"])
    ry   = int(cs["ry"])
    C    = int(cs.get("C", model_cfg["in_channels"]))

    # Normalise shapes: accept (C,1,1) or (C,Rx,Ry)
    assert mean.shape[0] == C and std.shape[0] == C, \
        f"mean/std channel mismatch: {mean.shape} vs C={C}"

    channel_names: List[str] = ckpt.get(
        "channel_names",
        ["ux_real", "ux_imag", "uy_real", "uy_imag"] if C == 4 else
        ["real", "imag"] if C == 2 else [f"ch{i}" for i in range(C)],
    )

    return {
        "model":         model,
        "schedule":      schedule,
        "mean":          mean,
        "std":           std,
        "C":             C,
        "rx":            rx,
        "ry":            ry,
        "channel_names": channel_names,
        "omega_min":     float(ckpt["omega_stats"]["min"]),
        "omega_max":     float(ckpt["omega_stats"]["max"]),
        "num_steps":     int(diff_cfg["num_steps"]),
    }


def _load_ftm_basis(ftm_ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    """Load FTM basis networks. Supports shared-basis (new + old) format."""
    if not ftm_ckpt_path.exists():
        raise FileNotFoundError(f"FTM checkpoint not found: {ftm_ckpt_path}")

    ckpt = torch.load(ftm_ckpt_path, map_location="cpu")
    cfg  = ckpt.get("config", {})

    for k in ("rank_x", "rank_y", "hidden_dim", "hidden_layers", "activation"):
        if k not in cfg:
            raise KeyError(f"FTM config missing '{k}'")

    def _make(rank_out):
        net = MLP1D(rank_out, int(cfg["hidden_dim"]),
                    int(cfg["hidden_layers"]), str(cfg["activation"])).to(device)
        net.eval()
        return net

    if "net_x_state_list" in ckpt:
        # split-basis format
        nets_x = [_make(cfg["rank_x"]) for _ in ckpt["net_x_state_list"]]
        nets_y = [_make(cfg["rank_y"]) for _ in ckpt["net_y_state_list"]]
        for net, sd in zip(nets_x, ckpt["net_x_state_list"]):
            net.load_state_dict(sd)
        for net, sd in zip(nets_y, ckpt["net_y_state_list"]):
            net.load_state_dict(sd)
        ch_to_group = ckpt.get("ch_to_group", list(range(len(nets_x))))
    else:
        # shared-basis format (old or new)
        net_x = _make(cfg["rank_x"]); net_x.load_state_dict(ckpt["net_x_state"])
        net_y = _make(cfg["rank_y"]); net_y.load_state_dict(ckpt["net_y_state"])
        nets_x, nets_y = [net_x], [net_y]
        ch_to_group = None   # resolved later when C is known

    return {
        "nets_x":           nets_x,
        "nets_y":           nets_y,
        "ch_to_group":      ch_to_group,
        "rank_x":           int(cfg["rank_x"]),
        "rank_y":           int(cfg["rank_y"]),
        "normalize_coords": bool(cfg.get("normalize_coords", True)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Elastic wave physics
# ─────────────────────────────────────────────────────────────────────────────

def _scipy_real_to_torch_sparse(mat_csr, device: torch.device) -> torch.Tensor:
    """Convert a real scipy CSR matrix to a torch sparse COO tensor."""
    coo  = mat_csr.tocoo()
    idx  = torch.from_numpy(
        np.vstack([coo.row, coo.col]).astype(np.int64)
    ).to(device)
    vals = torch.from_numpy(np.asarray(coo.data, dtype=np.float32)).to(device)
    return torch.sparse_coo_tensor(
        idx, vals, coo.shape, device=device, dtype=torch.float32
    ).coalesce()


def _build_elastic_physics(
    N: int,
    L: float,
    sigma_max: float,
    pml_width: int,
    lambda_field: np.ndarray,   # (N, N)
    mu_field:     np.ndarray,
    rho_field:    np.ndarray,
    omega:        float,
    phi_t:        torch.Tensor, # (P, R)
    device:       torch.device,
    data_scale:   float = 1.0,
) -> Dict[str, Any]:
    """
    Build all tensors needed for the elastic wave PDE residual.

    Returned dict keys
    ------------------
    A11_re/im, A12_re/im, A21_re/im, A22_re/im : (N², N²) torch sparse tensors
    fx, fy   : (N²,) RHS vectors (normalised by data_scale)
    phi_full : (P, R)
    residual_den : scalar tensor for normalisation
    """
    dx = L / N

    # ── 1D central-difference derivative ─────────────────────────────────
    D1 = sp.lil_matrix((N, N), dtype=np.float32)
    inv = 1.0 / dx;  h_inv = 0.5 / dx
    D1[0, 0]  = -inv; D1[0, 1]  = inv
    D1[-1,-2] = -inv; D1[-1,-1] = inv
    for i in range(1, N - 1):
        D1[i, i - 1] = -h_inv; D1[i, i + 1] = h_inv
    D1 = D1.tocsr()
    eye = sp.eye(N, format="csr")
    Dx = sp.kron(eye, D1, format="csr")
    Dy = sp.kron(D1, eye, format="csr")

    # ── PML damping mask ─────────────────────────────────────────────────
    pml = np.zeros((N, N), dtype=np.float32)
    for k in range(pml_width):
        v = ((pml_width - k) / pml_width) ** 2
        pml[k, :]      = np.maximum(pml[k, :],      v)
        pml[-(k+1), :] = np.maximum(pml[-(k+1), :], v)
        pml[:, k]      = np.maximum(pml[:, k],      v)
        pml[:, -(k+1)] = np.maximum(pml[:, -(k+1)], v)

    flat_lam  = lambda_field.reshape(-1).astype(np.complex64)
    flat_mu   = mu_field.reshape(-1).astype(np.complex64)
    flat_rho  = rho_field.reshape(-1).astype(np.complex64)
    flat_pml  = pml.reshape(-1)

    lam_d    = sp.diags(flat_lam,                  format="csr")
    mu_d     = sp.diags(flat_mu,                   format="csr")
    lp2mu_d  = sp.diags(flat_lam + 2.0 * flat_mu, format="csr")
    mass_d   = sp.diags(
        flat_rho * omega**2 + 1j * omega * sigma_max * flat_pml,
        format="csr"
    )

    A11 = Dx @ lp2mu_d @ Dx + Dy @ mu_d    @ Dy + mass_d
    A12 = Dx @ lam_d   @ Dy + Dy @ mu_d    @ Dx
    A21 = Dx @ mu_d    @ Dy + Dy @ lam_d   @ Dx
    A22 = Dx @ mu_d    @ Dx + Dy @ lp2mu_d @ Dy + mass_d

    def to_ts(mat):
        """complex scipy → (re, im) torch sparse pair."""
        return (
            _scipy_real_to_torch_sparse(sp.csr_matrix(mat.real), device),
            _scipy_real_to_torch_sparse(sp.csr_matrix(mat.imag), device),
        )

    A11_re, A11_im = to_ts(A11)
    A12_re, A12_im = to_ts(A12)
    A21_re, A21_im = to_ts(A21)
    A22_re, A22_im = to_ts(A22)

    # ── Source vector ─────────────────────────────────────────────────────
    N2      = N * N
    src_idx = (N // 2) * N + N // 6
    fx_np   = np.zeros(N2, dtype=np.float32)
    fx_np[src_idx] = 1000.0 / max(float(data_scale), 1e-12)

    fx = torch.from_numpy(fx_np).to(device)
    fy = torch.zeros(N2, device=device, dtype=torch.float32)

    residual_den = torch.clamp(torch.mean(fx ** 2 + fy ** 2), min=1.0)

    return {
        "A11_re": A11_re, "A11_im": A11_im,
        "A12_re": A12_re, "A12_im": A12_im,
        "A21_re": A21_re, "A21_im": A21_im,
        "A22_re": A22_re, "A22_im": A22_im,
        "fx":           fx,
        "fy":           fy,
        "phi_full":     phi_t,
        "residual_den": residual_den,
    }


def _spmv(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Sparse (P,P) × dense (P,) → dense (P,)."""
    return torch.sparse.mm(A, v.unsqueeze(1)).squeeze(1)


def _elastic_pde_residual_loss(
    x0_var:  torch.Tensor,   # (1, C, Rx, Ry)   requires_grad
    mean:    torch.Tensor,   # (C, 1, 1)
    std:     torch.Tensor,   # (C, 1, 1)
    physics: Dict[str, Any],
    eps:     float = 1e-4,
) -> torch.Tensor:
    """
    ||A[ux;uy] - f||² / residual_den  in normalised space.

    Channels are assumed to be [ux_re, ux_im, uy_re, uy_im].
    """
    # Denormalise → (1, C, Rx, Ry)
    cp = x0_var * std.unsqueeze(0) + mean.unsqueeze(0)
    phi = physics["phi_full"]    # (P, R)

    def field(c_idx):
        return phi @ cp[0, c_idx].reshape(-1)   # (P,)

    ux_re = field(0); ux_im = field(1)
    uy_re = field(2); uy_im = field(3)

    A11r, A11i = physics["A11_re"], physics["A11_im"]
    A12r, A12i = physics["A12_re"], physics["A12_im"]
    A21r, A21i = physics["A21_re"], physics["A21_im"]
    A22r, A22i = physics["A22_re"], physics["A22_im"]
    fx, fy     = physics["fx"],     physics["fy"]

    # ux equation
    res_ux_re = (_spmv(A11r, ux_re) - _spmv(A11i, ux_im)
                + _spmv(A12r, uy_re) - _spmv(A12i, uy_im) - fx)
    res_ux_im = (_spmv(A11r, ux_im) + _spmv(A11i, ux_re)
                + _spmv(A12r, uy_im) + _spmv(A12i, uy_re))    # fx_im = 0

    # uy equation
    res_uy_re = (_spmv(A21r, ux_re) - _spmv(A21i, ux_im)
                + _spmv(A22r, uy_re) - _spmv(A22i, uy_im) - fy)
    res_uy_im = (_spmv(A21r, ux_im) + _spmv(A21i, ux_re)
                + _spmv(A22r, uy_im) + _spmv(A22i, uy_re))    # fy_im = 0

    num = torch.mean(res_ux_re**2 + res_ux_im**2 + res_uy_re**2 + res_uy_im**2)
    return num / torch.clamp(physics["residual_den"], min=eps)


def _inline_physics_residual_numpy(
    pred: np.ndarray,           # (H, W, C)
    physics_np: Dict[str, Any], # A blocks as scipy sparse + numpy source
    data_scale: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """
    Compute elastic PDE residual in numpy (for logging, not DPS gradient).
    """
    H, W, C = pred.shape
    N2 = H * W

    ux_re = pred[:, :, 0].reshape(N2) * data_scale
    ux_im = pred[:, :, 1].reshape(N2) * data_scale
    uy_re = pred[:, :, 2].reshape(N2) * data_scale
    uy_im = pred[:, :, 3].reshape(N2) * data_scale

    A11 = physics_np["A11"]; A12 = physics_np["A12"]
    A21 = physics_np["A21"]; A22 = physics_np["A22"]
    fx  = physics_np["fx"];  fy  = physics_np["fy"]

    res_ux = A11 @ (ux_re + 1j * ux_im) + A12 @ (uy_re + 1j * uy_im) - (fx + 0j)
    res_uy = A21 @ (ux_re + 1j * ux_im) + A22 @ (uy_re + 1j * uy_im) - (fy + 0j)

    num = float(np.mean(np.abs(res_ux)**2 + np.abs(res_uy)**2))
    den = float(np.mean(np.abs(fx)**2 + np.abs(fy)**2))
    return float(np.sqrt(num / max(den, eps)))


# ─────────────────────────────────────────────────────────────────────────────
# Core / field operations  (C-channel generic)
# ─────────────────────────────────────────────────────────────────────────────

def _decode_cores_from_image(
    x_norm: torch.Tensor,   # (N, C, Rx, Ry)
    mean:   torch.Tensor,   # (C, 1, 1)
    std:    torch.Tensor,   # (C, 1, 1)
) -> List[torch.Tensor]:
    """Return list[C] of (N, R) denormalised core vectors."""
    cp = x_norm * std.unsqueeze(0) + mean.unsqueeze(0)  # (N, C, Rx, Ry)
    N, C, Rx, Ry = cp.shape
    return [cp[:, c].reshape(N, Rx * Ry) for c in range(C)]


def _decode_field_numpy(
    core_norm: np.ndarray,  # (C, Rx, Ry)
    phi:       np.ndarray,  # (P, R)
    mean:      np.ndarray,  # (C, 1, 1)
    std:       np.ndarray,  # (C, 1, 1)
    H: int, W: int,
) -> np.ndarray:            # (H, W, C)
    C = core_norm.shape[0]
    R = core_norm.shape[1] * core_norm.shape[2]
    cp = core_norm * std + mean                       # (C, Rx, Ry)
    out = np.stack([phi @ cp[c].reshape(R) for c in range(C)], axis=-1)
    return out.reshape(H, W, C).astype(np.float32)


def _lstsq_core_from_gt(
    gt:  np.ndarray,       # (H, W, C)
    phi: np.ndarray,       # (P, R)
    Rx: int, Ry: int,
    rcond: Optional[float] = None,
) -> np.ndarray:           # (C, Rx, Ry)
    H, W, C = gt.shape
    gt_flat = gt.reshape(H * W, C)
    out = np.zeros((C, Rx, Ry), dtype=np.float32)
    for c in range(C):
        core_c, *_ = np.linalg.lstsq(phi, gt_flat[:, c], rcond=rcond)
        out[c] = core_c.reshape(Rx, Ry)
    return out


def _rel_rmse(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.sum((pred - gt) ** 2) / max(np.sum(gt ** 2), eps)))


def _masked_rel_rmse(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps: float = 1e-12
) -> float:
    m = mask.astype(bool)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.sum(((pred - gt) ** 2)[m]) / max(np.sum((gt ** 2)[m]), eps)))


def _core_rel_err(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.sum((pred - gt) ** 2) / max(np.sum(gt ** 2), eps)))


# ─────────────────────────────────────────────────────────────────────────────
# DPS sampling  (C-channel generic)
# ─────────────────────────────────────────────────────────────────────────────

def sample_core(
    model:          ConditionalUNet2D,
    schedule:       Any,
    omega_norm:     float,
    mean:           torch.Tensor,          # (C, 1, 1)
    std:            torch.Tensor,          # (C, 1, 1)
    C:              int,
    Rx:             int,
    Ry:             int,
    obs_mats:       List[torch.Tensor],    # list[C] of (n_obs_c, R)
    obs_vals:       List[torch.Tensor],    # list[C] of (1, n_obs_c)
    timestep_seq:   Sequence[int],
    dps_weight:     float,
    phys_weight:    float,
    guidance_scale: float,
    init_noise:     torch.Tensor,          # (1, C, Rx, Ry)
    physics:        Optional[Dict] = None,
    guidance_grad_clip: float = 1e-3,
    log_guidance:   bool = False,
    eps:            float = 1e-4,
) -> torch.Tensor:                         # (C, Rx, Ry)
    """
    DDPM reverse process with optional DPS observation + elastic PDE guidance.
    Works for any C (C=2 Helmholtz, C=4 elastic wave).
    """
    device    = init_noise.device
    x         = init_noise.clone()
    omega_cond = torch.tensor([[omega_norm]], device=device, dtype=torch.float32)
    n_steps   = len(timestep_seq)

    for i, t_idx in enumerate(timestep_seq):
        t = torch.full((1,), int(t_idx), device=device, dtype=torch.long)

        with torch.no_grad():
            eps_pred = model(x, t, omega_cond)
            abar_t   = schedule.alpha_bars[t_idx]
            x0_hat   = (x - torch.sqrt(torch.clamp(1.0 - abar_t, min=1e-12)) * eps_pred) \
                       / torch.sqrt(torch.clamp(abar_t, min=1e-12))

        use_dps  = dps_weight  > 0.0
        use_phys = phys_weight > 0.0 and physics is not None

        if use_dps or use_phys:
            x0_var = x0_hat.detach().requires_grad_(True)

            # Denormalised core vectors: list[C] of (1, R)
            cores = _decode_cores_from_image(x0_var, mean, std)   # list[C] of (1,R)

            total_loss = x0_var.new_zeros(())
            like_loss  = None
            pde_loss   = None

            # ── DPS observation loss ──────────────────────────────────
            if use_dps:
                obs_loss = x0_var.new_zeros(())
                for c in range(C):
                    pred_obs = torch.matmul(cores[c], obs_mats[c].t())  # (1, n_obs_c)
                    y_c      = obs_vals[c]                               # (1, n_obs_c)
                    rel = torch.mean((pred_obs - y_c) ** 2) \
                          / (torch.mean(y_c ** 2) + eps)
                    obs_loss = obs_loss + rel
                like_loss  = obs_loss / C

                # Taper guidance strength toward late steps
                frac = i / max(n_steps - 1, 1)
                # if frac > 0.7:
                #     like_loss = like_loss * 0.5

                total_loss = total_loss + dps_weight * like_loss

            # ── Elastic PDE residual loss ─────────────────────────────
            if use_phys and physics is not None:
                if C == 4:
                    pde_loss = _elastic_pde_residual_loss(
                        x0_var, mean, std, physics, eps=eps
                    )
                else:
                    # Fallback: no physics guidance for other C values
                    pde_loss = x0_var.new_zeros(())

                if i < n_steps * 0.5:        # stronger physics early
                    pde_loss = pde_loss * 3.0
                total_loss = total_loss + phys_weight * pde_loss

            grad = torch.autograd.grad(
                total_loss, x0_var, retain_graph=False, create_graph=False
            )[0]

            # Time-based step-size taper
            # time_w = 1.0 - (t_idx / max(timestep_seq[0], 1))
            time_w = 1.0
            if omega_norm < 0.1:
                time_w = 1.0; grad = grad * 3.0

            if guidance_grad_clip > 0.0 and i > n_steps * 0.7:
                clip = guidance_grad_clip * (0.5 if i > n_steps * 0.9 else 1.0)
                grad = torch.clamp(grad, -clip, clip)

            x0_hat = x0_hat - time_w * guidance_scale * grad

            if log_guidance:
                msg = f"[t={t_idx:03d}] grad_max={grad.abs().max():.3e}"
                if like_loss is not None:
                    msg += f"  obs={float(like_loss):.3e}"
                if pde_loss is not None:
                    msg += f"  pde={float(pde_loss):.3e}"
                print(msg)

        # DDPM reverse step
        if i == n_steps - 1:
            x = x0_hat
        else:
            t_prev   = timestep_seq[i + 1]
            abar_prev = schedule.alpha_bars[t_prev]
            with torch.no_grad():
                x = (torch.sqrt(torch.clamp(abar_prev, min=1e-12)) * x0_hat
                     + torch.sqrt(torch.clamp(1.0 - abar_prev, min=1e-12)) * eps_pred)

    return x.squeeze(0)   # (C, Rx, Ry)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _plot_displacement_component(
    ax_row: List[plt.Axes],
    gt_re: np.ndarray, gt_im: np.ndarray,
    pr_re: np.ndarray, pr_im: np.ndarray,
    dp_re: np.ndarray, dp_im: np.ndarray,
    name: str,
    fig: plt.Figure,
) -> None:
    """Fill 3 rows (re / im / amp) × 4 cols (GT | Prior | DPS | Err-DPS)."""
    gt_amp = np.sqrt(gt_re**2 + gt_im**2)
    pr_amp = np.sqrt(pr_re**2 + pr_im**2)
    dp_amp = np.sqrt(dp_re**2 + dp_im**2)

    def _lim(a, b):
        return min(a.min(), b.min()), max(a.max(), b.max())

    rows = [
        (gt_re, pr_re, dp_re, np.abs(dp_re - gt_re), f"{name} Real"),
        (gt_im, pr_im, dp_im, np.abs(dp_im - gt_im), f"{name} Imag"),
        (gt_amp, pr_amp, dp_amp, np.abs(dp_amp - gt_amp), f"{name} Amp"),
    ]
    for axes, (g, p, d, e, label) in zip(ax_row, rows):
        lo, hi = _lim(g, d)
        for ax, (img, title, is_err) in zip(
            axes,
            [(g, f"GT {label}", False),
             (p, f"Prior {label}", False),
             (d, f"DPS {label}", False),
             (e, f"|Err| {label}", True)],
        ):
            kw = {"origin": "lower", "cmap": "magma" if is_err else "viridis"}
            if not is_err:
                kw.update({"vmin": lo, "vmax": hi})
            im = ax.imshow(img, **kw)
            ax.set_title(title, fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_case_elastic(
    out_path:   Path,
    gt:         np.ndarray,           # (H, W, 4)
    pred_prior: np.ndarray,
    pred_dps:   np.ndarray,
    core_gt_lstsq: np.ndarray,        # (4, Rx, Ry)
    core_dps:   np.ndarray,
    mask:       np.ndarray,           # (H, W, 4)
    metrics:    Dict[str, float],
    sample_idx: int,
    freq_idx:   int,
    omega_val:  float,
    channel_names: List[str],
) -> None:
    n_rows = 8    # 3 (ux) + 3 (uy) + 1 (masks) + 1 (cores)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))

    # ── ux displacement rows (0-2) ────────────────────────────────────────
    _plot_displacement_component(
        ax_row=[axes[0], axes[1], axes[2]],
        gt_re=gt[..., 0], gt_im=gt[..., 1],
        pr_re=pred_prior[..., 0], pr_im=pred_prior[..., 1],
        dp_re=pred_dps[..., 0],   dp_im=pred_dps[..., 1],
        name="ux", fig=fig,
    )

    # ── uy displacement rows (3-5) ────────────────────────────────────────
    _plot_displacement_component(
        ax_row=[axes[3], axes[4], axes[5]],
        gt_re=gt[..., 2], gt_im=gt[..., 3],
        pr_re=pred_prior[..., 2], pr_im=pred_prior[..., 3],
        dp_re=pred_dps[..., 2],   dp_im=pred_dps[..., 3],
        name="uy", fig=fig,
    )

    # ── Masks row (row 6) ────────────────────────────────────────────────
    mask_labels = [f"Mask {n}" for n in channel_names]
    for col, (ax, lab) in enumerate(zip(axes[6], mask_labels)):
        im = ax.imshow(mask[..., col], cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.set_title(lab, fontsize=7); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Core comparison row (row 7) ──────────────────────────────────────
    # Show only first 4 channels of cores (ux_re, ux_im, uy_re, uy_im)
    core_items = []
    for c_idx, cname in enumerate(channel_names[:4]):
        cg = core_gt_lstsq[c_idx]
        cd = core_dps[c_idx]
        core_items.append((cg, cd, np.abs(cd - cg), cname))

    # Pack 4 pairs into 4 cols: show |err| for first 4 channels
    for col, (cg, cd, err, cname) in enumerate(core_items[:4]):
        ax = axes[7, col]
        lo = min(cg.min(), cd.min()); hi = max(cg.max(), cd.max())
        im = ax.imshow(err, cmap="magma", origin="lower")
        ax.set_title(f"Core |Err| {cname}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"sample={sample_idx}  freq_idx={freq_idx}  ω={omega_val:.3f}\n"
        f"RMSE prior={metrics.get('rmse_prior',0):.3e}  "
        f"DPS={metrics.get('rmse_dps',0):.3e}  "
        f"core_rel_prior={metrics.get('core_rel_prior',0):.3e}  "
        f"core_rel_dps={metrics.get('core_rel_dps',0):.3e}",
        fontsize=9, y=1.001,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_case_generic(
    out_path:   Path,
    gt:         np.ndarray,   # (H, W, C)
    pred_prior: np.ndarray,
    pred_dps:   np.ndarray,
    mask:       np.ndarray,   # (H, W, C)
    metrics:    Dict[str, float],
    sample_idx: int,
    freq_idx:   int,
    omega_val:  float,
    channel_names: List[str],
) -> None:
    """Generic C-channel field visualisation (amplitude per channel)."""
    C = gt.shape[-1]
    fig, axes = plt.subplots(C, 4, figsize=(16, C * 3.5))
    if C == 1:
        axes = axes[None, :]
    for c, cname in enumerate(channel_names):
        g, p, d = gt[..., c], pred_prior[..., c], pred_dps[..., c]
        err = np.abs(d - g)
        lo, hi = min(g.min(), d.min()), max(g.max(), d.max())
        for col, (img, title, is_err) in enumerate([
            (g,   f"GT {cname}",    False),
            (p,   f"Prior {cname}", False),
            (d,   f"DPS {cname}",   False),
            (err, f"|Err| {cname}", True),
        ]):
            ax = axes[c, col]
            kw = {"origin": "lower", "cmap": "magma" if is_err else "viridis"}
            if not is_err:
                kw.update({"vmin": lo, "vmax": hi})
            im = ax.imshow(img, **kw)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"sample={sample_idx}  freq_idx={freq_idx}  ω={omega_val:.3f}  "
        f"RMSE prior={metrics.get('rmse_prior',0):.3e} DPS={metrics.get('rmse_dps',0):.3e}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_freq_curve(
    out_path: Path, omega: np.ndarray,
    prior_vals: np.ndarray, dps_vals: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(omega, prior_vals, lw=1.8, label="Prior")
    ax.plot(omega, dps_vals,   lw=1.8, label="DPS")
    ax.set_xlabel("ω"); ax.set_ylabel("Mean relative RMSE")
    ax.set_title("Frequency-wise Reconstruction Error")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )

    # ── Load checkpoints ──────────────────────────────────────────────────
    diff = _load_diffusion(Path(args.diff_ckpt), device)
    ftm  = _load_ftm_basis(Path(args.ftm_ckpt),  device)

    model         = diff["model"]
    schedule      = diff["schedule"]
    mean          = diff["mean"]           # (C, 1, 1)
    std           = diff["std"]
    C             = diff["C"]
    Rx, Ry        = diff["rx"], diff["ry"]
    R             = Rx * Ry
    channel_names = diff["channel_names"]
    omega_min     = diff["omega_min"]
    omega_max     = diff["omega_max"]

    # Resolve ch_to_group (default: all channels share group 0)
    ch_to_group = ftm["ch_to_group"] or [0] * C
    timestep_seq = _build_timestep_sequence(diff["num_steps"], args.sample_steps)

    print(f"\n{'─'*64}")
    print(f"  Elastic Wave Diffusion Evaluation  (C={C})")
    print(f"  diff_ckpt = {args.diff_ckpt}")
    print(f"  ftm_ckpt  = {args.ftm_ckpt}")
    print(f"  data_h5   = {args.data_h5}")
    print(f"  channels  = {channel_names}")
    print(f"  device    = {device}   sample_steps={args.sample_steps}")
    print(f"{'─'*64}\n")

    data_h5_path = Path(args.data_h5)

    with h5py.File(data_h5_path, "r") as fh:
        for key in ("data", "mask_tr", "omega"):
            if key not in fh:
                raise KeyError(f"HDF5 missing '{key}'")

        data_ds  = fh["data"]
        mask_ds  = fh["mask_tr"]
        omega_arr = fh["omega"][...].astype(np.float32)

        B_data, M_data, H, W, C_data = data_ds.shape
        if C_data != C:
            raise ValueError(f"Data channels {C_data} ≠ checkpoint channels {C}")

        h5_meta   = _load_h5_metadata_dict(fh)
        data_scale = max(_load_data_scale(data_h5_path, fh), 1e-12)

        freq_ids  = _parse_freq_indices(args.freq_indices, M_data)
        B_eval    = B_data if args.max_samples <= 0 else min(B_data, args.max_samples)
        all_cases = [(b, m) for b in range(B_eval) for m in freq_ids]

        if args.max_cases > 0 and len(all_cases) > args.max_cases:
            rng = np.random.default_rng(args.seed)
            sel = rng.choice(len(all_cases), size=args.max_cases, replace=False)
            cases = [all_cases[int(i)] for i in sel]
        else:
            cases = all_cases

        # ── Spatial coordinates & phi ─────────────────────────────────────
        grid_x = fh["grid_x"][...].astype(np.float32) if "grid_x" in fh \
                 else np.linspace(0, 1, H, dtype=np.float32)
        grid_y = fh["grid_y"][...].astype(np.float32) if "grid_y" in fh \
                 else np.linspace(0, 1, W, dtype=np.float32)

        x_np = normalize_coords_to_unit(grid_x.astype(np.float64)).astype(np.float32) \
               if ftm["normalize_coords"] else grid_x
        y_np = normalize_coords_to_unit(grid_y.astype(np.float64)).astype(np.float32) \
               if ftm["normalize_coords"] else grid_y

        x_t = torch.from_numpy(x_np).unsqueeze(-1).to(device)
        y_t = torch.from_numpy(y_np).unsqueeze(-1).to(device)

        # Build phi for each basis group
        with torch.no_grad():
            phis_t = [build_phi(ftm["nets_x"][g], ftm["nets_y"][g], x_t, y_t)
                      for g in range(len(ftm["nets_x"]))]
        phis_np = [p.cpu().numpy().astype(np.float32) for p in phis_t]

        # For each channel c, select its phi
        phi_for_ch = [phis_np[ch_to_group[c]] for c in range(C)]
        phi_t_for_ch = [phis_t[ch_to_group[c]] for c in range(C)]

        # Single shared phi for physics (all channels use group 0)
        phi_shared_np = phis_np[0]
        phi_shared_t  = phis_t[0]

        mean_np = mean.cpu().numpy().astype(np.float32)
        std_np  = std.cpu().numpy().astype(np.float32)

        # ── Physics setup ─────────────────────────────────────────────────
        use_phys = args.phys_weight > 0.0 and C == 4

        elastic_N      = int(h5_meta.get("grid", H))
        elastic_L      = float(h5_meta.get("L", args.elastic_L))
        elastic_sigma  = float(h5_meta.get("sigma_max", args.elastic_sigma_max))
        elastic_pml    = int(h5_meta.get("pml_width", args.elastic_pml_width))

        if use_phys:
            print(f"[INFO] Physics guidance enabled: N={elastic_N}, L={elastic_L}, "
                  f"σ_max={elastic_sigma}, pml_width={elastic_pml}, "
                  f"data_scale={data_scale:.4g}")

        # Cache: sample_idx → (lambda, mu, rho) as np arrays
        # Cache: (sample_idx, freq_idx) → physics dict (torch tensors)
        material_cache:  Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        physics_cache:   Dict[Tuple[int, int], Dict[str, Any]] = {}
        physics_np_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # ── Evaluation loop ───────────────────────────────────────────────
        rows: List[Dict[str, Any]] = []
        vis_count    = 0
        freq_prior   = {m: [] for m in freq_ids}
        freq_dps     = {m: [] for m in freq_ids}

        for i_case, (b_idx, m_idx) in enumerate(cases, start=1):
            omega_val  = float(omega_arr[m_idx])
            omega_norm = _normalize_omega(omega_val, omega_min, omega_max)

            # Ground-truth field and mask
            gt = data_ds[b_idx, m_idx].astype(np.float32)    # (H, W, C)
            if mask_ds.ndim == 4:
                mask_case = (mask_ds[m_idx].astype(np.float32) > 0.5)        # (H,W,C)
            else:
                mask_case = (mask_ds[b_idx, m_idx].astype(np.float32) > 0.5) # (H,W,C)

            # Build per-channel observation operators
            obs_mats: List[torch.Tensor] = []
            obs_vals: List[torch.Tensor] = []
            skip = False
            for c in range(C):
                idx_c = np.flatnonzero(mask_case[..., c].reshape(-1))
                if idx_c.size == 0:
                    skip = True; break
                obs_mats.append(
                    torch.from_numpy(phi_for_ch[c][idx_c]).to(device)  # (n_obs_c, R)
                )
                obs_vals.append(
                    torch.from_numpy(gt[..., c].reshape(-1)[idx_c]).to(device).view(1, -1)
                )
            if skip:
                print(f"  [skip] sample={b_idx} freq={m_idx}: empty mask channel")
                continue

            # ── Physics dict ─────────────────────────────────────────────
            physics_case: Optional[Dict] = None
            physics_np_case: Optional[Dict] = None

            if use_phys:
                key = (b_idx, m_idx)
                if key not in physics_cache:
                    # Load material fields (cached per sample)
                    if b_idx not in material_cache:
                        lam = fh["lambda_field"][b_idx].astype(np.float32)
                        mu  = fh["mu_field"][b_idx].astype(np.float32)
                        rho = fh["rho_field"][b_idx].astype(np.float32)
                        material_cache[b_idx] = (lam, mu, rho)
                    lam, mu, rho = material_cache[b_idx]

                    physics_cache[key] = _build_elastic_physics(
                        N=elastic_N, L=elastic_L,
                        sigma_max=elastic_sigma, pml_width=elastic_pml,
                        lambda_field=lam, mu_field=mu, rho_field=rho,
                        omega=omega_val, phi_t=phi_shared_t,
                        device=device, data_scale=data_scale,
                    )
                    # Numpy version for metrics
                    lam_c = lam.reshape(-1).astype(np.complex64)
                    mu_c  = mu.reshape(-1).astype(np.complex64)
                    rho_c = rho.reshape(-1).astype(np.complex64)
                    # Build scipy matrices for numpy residual
                    _p = physics_cache[key]
                    physics_np_cache[key] = {
                        "A11": None, "A12": None, "A21": None, "A22": None,
                        "fx": _p["fx"].cpu().numpy(),
                        "fy": _p["fy"].cpu().numpy(),
                    }  # Will compute lazily below if needed

                physics_case = physics_cache[key]

            # ── Shared init noise ─────────────────────────────────────────
            g_gen = torch.Generator(device=device)
            g_gen.manual_seed(args.seed + 17 * i_case)
            init_noise = torch.randn(
                (1, C, Rx, Ry), generator=g_gen, device=device, dtype=torch.float32
            )

            # ── Prior sample ──────────────────────────────────────────────
            prior_core = sample_core(
                model=model, schedule=schedule, omega_norm=omega_norm,
                mean=mean, std=std, C=C, Rx=Rx, Ry=Ry,
                obs_mats=obs_mats, obs_vals=obs_vals,
                timestep_seq=timestep_seq,
                dps_weight=0.0, phys_weight=0.0,
                guidance_scale=args.guidance_scale,
                init_noise=init_noise, physics=None,
                guidance_grad_clip=args.guidance_grad_clip,
                log_guidance=False, eps=args.eps,
            )

            # ── DPS sample ────────────────────────────────────────────────
            dps_core = sample_core(
                model=model, schedule=schedule, omega_norm=omega_norm,
                mean=mean, std=std, C=C, Rx=Rx, Ry=Ry,
                obs_mats=obs_mats, obs_vals=obs_vals,
                timestep_seq=timestep_seq,
                dps_weight=args.dps_weight, phys_weight=args.phys_weight,
                guidance_scale=args.guidance_scale,
                init_noise=init_noise, physics=physics_case,
                guidance_grad_clip=args.guidance_grad_clip,
                log_guidance=args.log_guidance, eps=args.eps,
            )

            prior_np = prior_core.cpu().numpy().astype(np.float32)   # (C, Rx, Ry)
            dps_np   = dps_core.cpu().numpy().astype(np.float32)

            # ── Decode to physical fields ─────────────────────────────────
            # Use per-channel phi for decoding (all same if shared basis)
            pred_prior = np.stack([
                phi_for_ch[c] @ (prior_np[c] * std_np[c] + mean_np[c]).reshape(R)
                for c in range(C)
            ], axis=-1).reshape(H, W, C)

            pred_dps = np.stack([
                phi_for_ch[c] @ (dps_np[c] * std_np[c] + mean_np[c]).reshape(R)
                for c in range(C)
            ], axis=-1).reshape(H, W, C)

            # ── Least-squares GT core ─────────────────────────────────────
            # Use shared phi (assumption: all channels same basis)
            core_gt = _lstsq_core_from_gt(
                gt, phi_shared_np, Rx, Ry, rcond=args.core_lstsq_rcond
            )  # (C, Rx, Ry)

            prior_phys = prior_np * std_np + mean_np
            dps_phys   = dps_np   * std_np + mean_np

            core_rel_prior = _core_rel_err(prior_phys, core_gt, eps=args.eps)
            core_rel_dps   = _core_rel_err(dps_phys,   core_gt, eps=args.eps)

            # ── Field metrics ─────────────────────────────────────────────
            rmse_prior = _rel_rmse(pred_prior, gt, eps=args.eps)
            rmse_dps   = _rel_rmse(pred_dps,   gt, eps=args.eps)
            obs_rmse_p = _masked_rel_rmse(pred_prior, gt, mask_case,       eps=args.eps)
            obs_rmse_d = _masked_rel_rmse(pred_dps,   gt, mask_case,       eps=args.eps)
            un_rmse_p  = _masked_rel_rmse(pred_prior, gt, ~mask_case.astype(bool), eps=args.eps)
            un_rmse_d  = _masked_rel_rmse(pred_dps,   gt, ~mask_case.astype(bool), eps=args.eps)

            # ── PDE residual (numpy, for logging) ─────────────────────────
            pde_prior, pde_dps = np.nan, np.nan
            if use_phys and C == 4:
                # Build scipy matrices if not yet done (lazy)
                pkey = (b_idx, m_idx)
                if physics_np_cache[pkey]["A11"] is None:
                    lam, mu, rho = material_cache[b_idx]
                    lam_c = lam.reshape(-1).astype(np.complex64)
                    mu_c  = mu.reshape(-1).astype(np.complex64)
                    rho_c = rho.reshape(-1).astype(np.complex64)
                    dx = elastic_L / elastic_N
                    D1 = sp.lil_matrix((elastic_N, elastic_N), dtype=np.float32)
                    inv = 1.0 / dx; h_inv = 0.5 / dx
                    D1[0, 0]  = -inv; D1[0, 1]  = inv
                    D1[-1,-2] = -inv; D1[-1,-1] = inv
                    for k in range(1, elastic_N - 1):
                        D1[k, k-1] = -h_inv; D1[k, k+1] = h_inv
                    D1 = D1.tocsr()
                    eye = sp.eye(elastic_N, format="csr")
                    Dx = sp.kron(eye, D1, format="csr")
                    Dy = sp.kron(D1, eye, format="csr")
                    pml = np.zeros((elastic_N, elastic_N), dtype=np.float32)
                    for k in range(elastic_pml):
                        v = ((elastic_pml - k) / elastic_pml) ** 2
                        pml[k, :] = np.maximum(pml[k, :], v)
                        pml[-(k+1),:] = np.maximum(pml[-(k+1),:], v)
                        pml[:, k] = np.maximum(pml[:, k], v)
                        pml[:,-(k+1)] = np.maximum(pml[:,-(k+1)], v)
                    flat_pml = pml.reshape(-1)
                    lam_d   = sp.diags(lam_c,                 format="csr")
                    mu_d    = sp.diags(mu_c,                  format="csr")
                    lp2mu_d = sp.diags(lam_c + 2.0 * mu_c,   format="csr")
                    mass_d  = sp.diags(rho_c * omega_val**2 + 1j * omega_val * elastic_sigma * flat_pml, format="csr")
                    A11 = Dx @ lp2mu_d @ Dx + Dy @ mu_d    @ Dy + mass_d
                    A12 = Dx @ lam_d   @ Dy + Dy @ mu_d    @ Dx
                    A21 = Dx @ mu_d    @ Dy + Dy @ lam_d   @ Dx
                    A22 = Dx @ mu_d    @ Dx + Dy @ lp2mu_d @ Dy + mass_d
                    N2  = elastic_N ** 2
                    src = (elastic_N // 2) * elastic_N + elastic_N // 6
                    fx_np = np.zeros(N2, dtype=np.complex64); fx_np[src] = 1000.0
                    fy_np = np.zeros(N2, dtype=np.complex64)
                    physics_np_cache[pkey] = {
                        "A11": A11, "A12": A12, "A21": A21, "A22": A22,
                        "fx": fx_np, "fy": fy_np,
                    }
                try:
                    pde_prior = _inline_physics_residual_numpy(
                        pred_prior, physics_np_cache[pkey], data_scale, args.eps
                    )
                    pde_dps = _inline_physics_residual_numpy(
                        pred_dps, physics_np_cache[pkey], data_scale, args.eps
                    )
                except Exception:
                    pass

            metrics = {
                "rmse_prior": rmse_prior, "rmse_dps": rmse_dps,
                "core_rel_prior": core_rel_prior, "core_rel_dps": core_rel_dps,
            }

            rows.append({
                "sample_idx": b_idx, "freq_idx": m_idx, "omega": omega_val,
                "rmse_prior": rmse_prior, "rmse_dps": rmse_dps,
                "rmse_obs_prior": obs_rmse_p, "rmse_obs_dps": obs_rmse_d,
                "rmse_unobs_prior": un_rmse_p, "rmse_unobs_dps": un_rmse_d,
                "core_rel_err_prior": core_rel_prior, "core_rel_err_dps": core_rel_dps,
                "pde_res_prior": pde_prior, "pde_res_dps": pde_dps,
            })
            freq_prior[m_idx].append(rmse_prior)
            freq_dps[m_idx].append(rmse_dps)

            # ── Visualise ─────────────────────────────────────────────────
            if vis_count < args.num_visualize:
                vis_count += 1
                fig_path = out_dir / (
                    f"compare_{vis_count:03d}_s{b_idx:03d}_f{m_idx:03d}.png"
                )
                if C == 4:
                    _plot_case_elastic(
                        out_path=fig_path, gt=gt,
                        pred_prior=pred_prior, pred_dps=pred_dps,
                        core_gt_lstsq=core_gt, core_dps=dps_phys,
                        mask=mask_case, metrics=metrics,
                        sample_idx=b_idx, freq_idx=m_idx,
                        omega_val=omega_val, channel_names=channel_names,
                    )
                else:
                    _plot_case_generic(
                        out_path=fig_path, gt=gt,
                        pred_prior=pred_prior, pred_dps=pred_dps,
                        mask=mask_case, metrics=metrics,
                        sample_idx=b_idx, freq_idx=m_idx,
                        omega_val=omega_val, channel_names=channel_names,
                    )

            if args.log_every > 0 and (i_case % args.log_every == 0 or i_case == len(cases)):
                print(
                    f"[{i_case:04d}/{len(cases)}] "
                    f"s={b_idx} f={m_idx} ω={omega_val:.1f}  "
                    f"prior={rmse_prior:.4e} dps={rmse_dps:.4e}  "
                    f"core_p={core_rel_prior:.4e} core_d={core_rel_dps:.4e}  "
                    f"pde_p={pde_prior:.4e} pde_d={pde_dps:.4e}"
                )

    if not rows:
        raise RuntimeError("No evaluation rows produced.")

    rows.sort(key=lambda r: (r["sample_idx"], r["freq_idx"]))

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = out_dir / "metrics_cases.csv"
    with open(csv_path, "w") as fcsv:
        fcsv.write(
            "sample_idx,freq_idx,omega,"
            "rmse_prior,rmse_dps,"
            "rmse_obs_prior,rmse_obs_dps,"
            "rmse_unobs_prior,rmse_unobs_dps,"
            "core_rel_err_prior,core_rel_err_dps,"
            "pde_res_prior,pde_res_dps\n"
        )
        for r in rows:
            fcsv.write(
                f"{r['sample_idx']},{r['freq_idx']},{r['omega']:.8g},"
                f"{r['rmse_prior']:.8g},{r['rmse_dps']:.8g},"
                f"{r['rmse_obs_prior']:.8g},{r['rmse_obs_dps']:.8g},"
                f"{r['rmse_unobs_prior']:.8g},{r['rmse_unobs_dps']:.8g},"
                f"{r['core_rel_err_prior']:.8g},{r['core_rel_err_dps']:.8g},"
                f"{r['pde_res_prior']:.8g},{r['pde_res_dps']:.8g}\n"
            )

    # ── Frequency curve ───────────────────────────────────────────────────
    unique_freqs = sorted({int(r["freq_idx"]) for r in rows})
    f2o  = {r["freq_idx"]: r["omega"] for r in rows}
    om   = np.array([f2o[f] for f in unique_freqs], dtype=np.float32)
    p_c  = np.array([np.mean(freq_prior[f]) if freq_prior[f] else np.nan
                     for f in unique_freqs], dtype=np.float32)
    d_c  = np.array([np.mean(freq_dps[f])   if freq_dps[f]   else np.nan
                     for f in unique_freqs], dtype=np.float32)
    _plot_freq_curve(out_dir / "freq_rmse_curve.png", om, p_c, d_c)

    # ── Summary ───────────────────────────────────────────────────────────
    def _nanmean(key):
        v = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.mean(v)) if v else float("nan")

    mean_p   = _nanmean("rmse_prior")
    mean_d   = _nanmean("rmse_dps")
    imp_abs  = mean_p - mean_d
    imp_pct  = 100.0 * imp_abs / max(mean_p, 1e-12)

    summary = {
        "diff_ckpt":               str(args.diff_ckpt),
        "ftm_ckpt":                str(args.ftm_ckpt),
        "data_h5":                 str(args.data_h5),
        "C":                       C,
        "channel_names":           channel_names,
        "num_cases":               len(rows),
        "mean_rmse_prior":         mean_p,
        "mean_rmse_dps":           mean_d,
        "mean_core_rel_err_prior": _nanmean("core_rel_err_prior"),
        "mean_core_rel_err_dps":   _nanmean("core_rel_err_dps"),
        "mean_pde_res_prior":      _nanmean("pde_res_prior"),
        "mean_pde_res_dps":        _nanmean("pde_res_dps"),
        "absolute_improvement":    imp_abs,
        "relative_improvement_%":  imp_pct,
        "dps_weight":              float(args.dps_weight),
        "phys_weight":             float(args.phys_weight),
        "guidance_scale":          float(args.guidance_scale),
        "sample_steps":            int(args.sample_steps),
        "data_scale":              float(data_scale),
        "output_dir":              str(out_dir),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as fsm:
        json.dump(summary, fsm, indent=2, ensure_ascii=False)

    print("\nEvaluation finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Metrics: {csv_path}")
    print(f"Summary: {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Test diffusion + DPS on elastic wave data (C=4)"
    )
    p.add_argument("--diff_ckpt",  type=str, default="ckp/diffusion_core.pt")
    p.add_argument("--ftm_ckpt",   type=str, default="ckp/ftm_gpu_checkpoint.pt")
    p.add_argument("--data_h5",    type=str, default="elastic_dataset_msk0.1.h5")
    p.add_argument("--out_dir",    type=str, default="visual_data/elastic_eval_msk0.1")

    p.add_argument("--freq_indices",    type=str, default="")
    p.add_argument("--max_samples",     type=int, default=3)
    p.add_argument("--max_cases",       type=int, default=0)

    p.add_argument("--sample_steps",    type=int,   default=500)
    p.add_argument("--dps_weight",      type=float, default=25.0)
    p.add_argument("--phys_weight",     type=float, default=0.00,
                   help="Elastic PDE residual guidance weight (0 to disable).")
    p.add_argument("--guidance_scale",  type=float, default=1.0)
    p.add_argument("--guidance_grad_clip", type=float, default=1e-2)

    # Elastic wave physics parameters (override from metadata if available)
    p.add_argument("--elastic_L",         type=float, default=1.0)
    p.add_argument("--elastic_sigma_max", type=float, default=60.0)
    p.add_argument("--elastic_pml_width", type=int,   default=12)

    p.add_argument("--num_visualize",   type=int, default=30)
    p.add_argument("--log_every",       type=int, default=1)
    p.add_argument("--log_guidance",    action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--eps",             type=float, default=1e-4)
    p.add_argument("--core_lstsq_rcond", type=float, default=None)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--device",          type=str,   default="auto")
    return p


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()