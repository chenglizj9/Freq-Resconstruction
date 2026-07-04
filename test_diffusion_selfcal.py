"""
test_diffusion_selfcal.py
--------------------------
Self-calibrating guidance for FTM + conditional diffusion reconstruction.

Extends test_diffusion.py with a new sampler `sample_core_selfcal` that
replaces the fixed (dps_weight, phys_weight, guidance_scale) with
per-step adaptive calibration:

DPS calibration
---------------
At each step we measure the normalised observation residual

    r_obs(t) = ‖Φ_obs g_t - y‖ / (‖y‖ + ε)

where Φ_obs is the row-subset of phi corresponding to observed points,
g_t = [core_re ; core_im] decoded from x̂₀.

The effective guidance step-size scales as

    scale_obs(t) = λ_dps × clamp(r_obs(t) / r_obs_ref, s_min, s_max)

• When r_obs >> r_obs_ref  (residual large, model far from data):
  scale is larger — push harder toward observations
• When r_obs << r_obs_ref  (residual small, nearly converged):
  scale is smaller — avoid over-shooting / oscillation

PDE calibration
---------------
Similarly:

    r_pde(t) = ‖A u_t - f‖ / (‖f‖ + ε)

    scale_pde(t) = λ_pde × clamp(r_pde(t) / r_pde_ref, s_min, s_max)

Balance between DPS and PDE
-----------------------------
We also renormalise so the two guidance signals contribute proportionally
to each other's residual magnitude:

    w_obs_balanced = scale_obs / (scale_obs + scale_pde + ε)
    w_pde_balanced = scale_pde / (scale_obs + scale_pde + ε)

    total_loss = (scale_obs + scale_pde) × (w_obs_balanced × like_loss
                                           + w_pde_balanced × pde_loss)

This ensures neither signal dominates when both are large/small.

Comparison mode
---------------
--mode compare  runs both the original `sample_core` and `sample_core_selfcal`
on the same init_noise for fair comparison.

Usage
-----
    python test_diffusion_selfcal.py --max_samples 3 --max_cases 0 \\
        --out_dir visual_data/selfcal_test
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
import torch

# Imports from original dependency modules (same as test_diffusion.py uses)
from train_diffusion import ConditionalUNet2D, build_cosine_schedule, build_linear_schedule
from train_FTM_GPU import MLP1D, build_phi, normalize_coords_to_unit
from Helmholtz_Solver import HelmholtzSolver
from physics_metric import evaluate_physics_residual

# Internal helpers defined in test_diffusion (not re-exported from other modules)
from test_diffusion import (
    _to_numpy,
    _normalize_omega,
    _parse_freq_indices,
    _load_diffusion,
    _load_ftm_basis,
    _build_timestep_sequence,
    _load_h5_metadata_dict,
    _load_data_scale,
    _csc_to_torch_sparse,
    _build_interior_index,
    _load_source_field_for_sample,
    _pde_residual_loss,
    _core_vectors_from_image,
    _decode_field_from_core_image,
    _rel_rmse,
    _masked_rel_rmse,
    _solve_lstsq_core_from_gt,
    _core_rel_err,
    _plot_case,
    _plot_freq_curve,
    sample_core,          # original sampler (for comparison)
    set_seed,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Self-calibrating sampler
# ---------------------------------------------------------------------------

def sample_core_selfcal(
    model: ConditionalUNet2D,
    schedule: Any,
    omega_norm: float,
    mean: torch.Tensor,       # (2,Rx,Ry)
    std: torch.Tensor,        # (2,Rx,Ry)
    a_re: torch.Tensor,       # (n_obs_re, R)  observation sub-matrix
    a_im: torch.Tensor,       # (n_obs_im, R)
    y_re: torch.Tensor,       # (1, n_obs_re)  sparse observations
    y_im: torch.Tensor,       # (1, n_obs_im)
    timestep_seq: Sequence[int],
    # --- self-cal base params ---
    lambda_dps: float = 1.0,       # base observation guidance scale
    lambda_pde: float = 0.02,      # base PDE guidance scale
    guidance_scale: float = 15.0,  # outer multiplier (same as original)
    # --- calibration knobs ---
    r_obs_ref: float = 0.3,    # reference obs residual (calibrate around this)
    r_pde_ref: float = 1.0,    # reference pde residual
    s_min: float = 0.2,        # minimum scale factor
    s_max: float = 3.0,        # maximum scale factor
    # --- original knobs kept ---
    init_noise: Optional[torch.Tensor] = None,  # (1,2,Rx,Ry)
    physics: Optional[Dict[str, torch.Tensor]] = None,
    guidance_grad_clip: float = 1e-3,
    log_guidance: bool = False,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Self-calibrating guidance sampler.

    Returns (core_image, step_log) where step_log is a list of per-step dicts
    with keys: t, r_obs, r_pde, scale_obs, scale_pde, total_guidance_scale.
    """
    device = init_noise.device if init_noise is not None else a_re.device
    Rx, Ry = mean.shape[-2], mean.shape[-1]
    x = init_noise.clone() if init_noise is not None \
        else torch.randn((1, 2, Rx, Ry), device=device, dtype=torch.float32)

    omega_cond = torch.tensor([[omega_norm]], device=device, dtype=torch.float32)
    step_log: List[Dict] = []

    use_dps  = lambda_dps > 0.0
    use_phys = lambda_pde > 0.0 and physics is not None

    for i, t_idx in enumerate(timestep_seq):
        t = torch.full((1,), int(t_idx), device=device, dtype=torch.long)

        with torch.no_grad():
            eps_pred = model(x, t, omega_cond)
            abar_t = schedule.alpha_bars[t_idx]
            x0_hat = (x - torch.sqrt(torch.clamp(1.0 - abar_t, min=1e-12)) * eps_pred) \
                   / torch.sqrt(torch.clamp(abar_t, min=1e-12))

        safe_step = len(timestep_seq) * 0.4
        is_safe_step = i < len(timestep_seq) - safe_step
        more_safe_step = i < len(timestep_seq) - safe_step * 0.5

        if use_dps or use_phys:
            x0_var = x0_hat.detach().requires_grad_(True)
            core_re, core_im = _core_vectors_from_image(x0_var, mean=mean, std=std)

            with torch.no_grad():
                r_obs_val = 0.0
                if use_dps:
                    pred_re_obs = torch.matmul(core_re.detach(), a_re.t())
                    pred_im_obs = torch.matmul(core_im.detach(), a_im.t())
                    res_re = pred_re_obs - y_re
                    res_im = pred_im_obs - y_im
                    denom_re = torch.mean(y_re ** 2).clamp(min=eps)
                    denom_im = torch.mean(y_im ** 2).clamp(min=eps)
                    r_obs_val = float(
                        (torch.mean(res_re ** 2) / denom_re + torch.mean(res_im ** 2) / denom_im)
                        .sqrt().item()
                    )

                r_pde_val = 0.0
                if use_phys and physics is not None:
                    pde_for_ref = _pde_residual_loss(
                        core_re=core_re.detach(), core_im=core_im.detach(),
                        physics=physics, eps=eps,
                    )
                    r_pde_val = float(pde_for_ref.item())

            # ── Self-calibration: keep dps weight fixed, only adapt phys weight ──
            # When obs residual is small (obs converging), increase PDE weight.
            # When obs residual is large (early denoising), PDE should not dominate.
            # Formula: phys_weight_t = lambda_pde * (r_obs_ref / (r_obs_val + eps))
            # → large obs residual → phys weight shrinks (don't confuse the sampler)
            # → small obs residual → phys weight grows (physics polishes the solution)
            # Clamped to [s_min × lambda_pde, s_max × lambda_pde]
            effective_dps = lambda_dps          # DPS weight unchanged from base
            if use_phys and r_obs_val > 0.0:
                phys_cal = float(np.clip(r_obs_ref / (r_obs_val + eps), s_min, s_max))
            else:
                phys_cal = 1.0
            effective_pde = lambda_pde * phys_cal

            total_loss = torch.zeros((), device=device, dtype=torch.float32)
            like_loss_val: Optional[float] = None
            pde_loss_val: Optional[float] = None

            if use_dps:
                pred_re_obs_g = torch.matmul(core_re, a_re.t())
                pred_im_obs_g = torch.matmul(core_im, a_im.t())
                rel_re = torch.mean((pred_re_obs_g - y_re) ** 2) / (torch.mean(y_re ** 2) + eps)
                rel_im = torch.mean((pred_im_obs_g - y_im) ** 2) / (torch.mean(y_im ** 2) + eps)
                like_loss = rel_re + rel_im
                if not more_safe_step:
                    like_loss = like_loss * 0.5
                like_loss_val = float(like_loss.item())
                total_loss = total_loss + effective_dps * like_loss

            if use_phys and physics is not None:
                pde_loss = _pde_residual_loss(core_re=core_re, core_im=core_im, physics=physics, eps=eps)
                if more_safe_step:
                    pde_loss = pde_loss * 3.0
                pde_loss_val = float(pde_loss.item())
                total_loss = total_loss + effective_pde * pde_loss

            grad = torch.autograd.grad(total_loss, x0_var)[0]

            time_weight = 1.0 - (t_idx / max(timestep_seq[0], 1))
            if omega_cond.item() < 0.1:
                grad = grad * 3.0
                time_weight = 1.0

            if not is_safe_step and guidance_grad_clip > 0.0:
                grad = torch.clamp(grad, -guidance_grad_clip, guidance_grad_clip)
            if not more_safe_step and guidance_grad_clip > 0.0:
                grad = torch.clamp(grad, -guidance_grad_clip * 0.5, guidance_grad_clip * 0.5)

            effective_scale = time_weight * guidance_scale
            x0_hat = x0_hat - effective_scale * grad

            if log_guidance and (i % max(len(timestep_seq) // 10, 1) == 0 or i == 0):
                step_log.append({
                    "t": t_idx, "i": i,
                    "r_obs": r_obs_val, "r_pde": r_pde_val,
                    "phys_cal": phys_cal,
                    "effective_dps": effective_dps, "effective_pde": effective_pde,
                    "effective_scale": float(effective_scale),
                    "like_loss": like_loss_val, "pde_loss": pde_loss_val,
                })
                print(f"  [t={t_idx:3d}] r_obs={r_obs_val:.3e} r_pde={r_pde_val:.3e} "
                      f"→ phys_cal={phys_cal:.3f} e_pde={effective_pde:.4f} eff_scale={effective_scale:.3f}")

        # DDIM-style update: x_{t_prev} = sqrt(ᾱ_prev) x̂₀ + sqrt(1-ᾱ_prev) ε
        if i == len(timestep_seq) - 1:
            x = x0_hat
        else:
            t_prev = timestep_seq[i + 1]
            abar_prev = schedule.alpha_bars[t_prev]
            with torch.no_grad():
                x = (torch.sqrt(torch.clamp(abar_prev, min=1e-12)) * x0_hat
                     + torch.sqrt(torch.clamp(1.0 - abar_prev, min=1e-12)) * eps_pred)

    return x.squeeze(0), step_log


# ---------------------------------------------------------------------------
# Comparison plot (original vs self-cal)
# ---------------------------------------------------------------------------

def _plot_compare(
    out_path: Path,
    gt: np.ndarray,
    pred_orig: np.ndarray,
    pred_sc: np.ndarray,
    mask_bool: np.ndarray,
    sample_idx: int,
    freq_idx: int,
    omega_val: float,
    rmse_orig: float,
    rmse_sc: float,
) -> None:
    gt_re, gt_im = gt[..., 0], gt[..., 1]
    or_re, or_im = pred_orig[..., 0], pred_orig[..., 1]
    sc_re, sc_im = pred_sc[..., 0], pred_sc[..., 1]
    mask_img = mask_bool[..., 0].astype(np.float32)

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    items = [
        (gt_re,  "GT Real",          "viridis"),
        (or_re,  "Orig DPS Real",    "viridis"),
        (sc_re,  "SelfCal Real",     "viridis"),
        (sc_re - or_re, "SelfCal - Orig Re", "RdBu_r"),

        (gt_im,  "GT Imag",          "viridis"),
        (or_im,  "Orig DPS Imag",    "viridis"),
        (sc_im,  "SelfCal Imag",     "viridis"),
        (sc_im - or_im, "SelfCal - Orig Im", "RdBu_r"),

        (np.abs(or_re - gt_re), "|Err| Orig Re",   "magma"),
        (np.abs(sc_re - gt_re), "|Err| SelfCal Re", "magma"),
        (np.abs(or_im - gt_im), "|Err| Orig Im",   "magma"),
        (mask_img, "Obs Mask", "gray"),
    ]
    for ax, (img, title, cmap) in zip(axes.flat, items):
        im = ax.imshow(img, origin="lower", cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"sample={sample_idx}  freq={freq_idx}  ω={omega_val:.4f}\n"
        f"RMSE orig={rmse_orig:.4e}  selfcal={rmse_sc:.4e}  "
        f"Δ={rmse_orig - rmse_sc:+.4e}  "
        f"({'better' if rmse_sc < rmse_orig else 'worse'})",
        y=0.998, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_selfcal(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
              if args.device == "auto" else torch.device(args.device)

    diff = _load_diffusion(Path(args.diff_ckpt), device=device)
    ftm  = _load_ftm_basis(Path(args.ftm_ckpt), device=device)

    model     = diff["model"]
    schedule  = diff["schedule"]
    mean, std = diff["mean"], diff["std"]
    omega_min, omega_max = diff["omega_min"], diff["omega_max"]
    rx, ry    = diff["rx"], diff["ry"]
    r         = rx * ry

    timestep_seq = _build_timestep_sequence(diff["num_steps"], args.sample_steps)

    data_h5_path = Path(args.data_h5)
    with h5py.File(data_h5_path, "r") as f:
        data_ds = f["data"]
        mask_ds = f["mask_tr"]
        omega   = f["omega"][...].astype(np.float32)
        b_data, m_data, h, w, c = data_ds.shape

        freq_ids = _parse_freq_indices(args.freq_indices, m_data)
        b_eval   = b_data if args.max_samples <= 0 else min(b_data, args.max_samples)
        cases    = [(b, m) for b in range(b_eval) for m in freq_ids]
        if args.max_cases > 0:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(cases), size=min(args.max_cases, len(cases)), replace=False)
            cases = [cases[int(i)] for i in idx]

        grid_x = f["grid_x"][...].astype(np.float32) if "grid_x" in f \
                 else np.linspace(0, 1, h, dtype=np.float32)
        grid_y = f["grid_y"][...].astype(np.float32) if "grid_y" in f \
                 else np.linspace(0, 1, w, dtype=np.float32)

        if ftm["normalize_coords"]:
            x_coords = normalize_coords_to_unit(grid_x.astype(np.float64)).astype(np.float32)
            y_coords = normalize_coords_to_unit(grid_y.astype(np.float64)).astype(np.float32)
        else:
            x_coords, y_coords = grid_x, grid_y

        x_t = torch.from_numpy(x_coords).unsqueeze(-1).to(device)
        y_t = torch.from_numpy(y_coords).unsqueeze(-1).to(device)
        with torch.no_grad():
            phi = build_phi(ftm["net_x"], ftm["net_y"], x_t, y_t)
        phi_np    = phi.detach().cpu().numpy().astype(np.float32)
        mean_np   = mean.detach().cpu().numpy().astype(np.float32)
        std_np    = std.detach().cpu().numpy().astype(np.float32)
        phi_t_full = torch.from_numpy(phi_np).to(device)

        h5_meta    = _load_h5_metadata_dict(f)
        data_scale = max(_load_data_scale(data_h5_path, f), 1e-12)

        # Physics setup
        use_phys = args.lambda_pde > 0.0
        solver   = None; a_cache = {}; source_cache = {}; interior_idx_t = None
        source_sigma = float(h5_meta.get("source_sigma", args.source_sigma))

        if use_phys and h == w:
            l_val = float(h5_meta.get("L", args.helmholtz_L))
            c_val = float(h5_meta.get("c", args.helmholtz_c))
            pml   = float(h5_meta.get("pml_width", args.helmholtz_pml_width))
            sig   = float(h5_meta.get("sigma_max", args.helmholtz_sigma_max))
            solver = HelmholtzSolver(N=h, L=l_val, c=c_val, pml_width=pml, sigma_max=sig)
            if args.phys_interior_only:
                interior_idx_t = _build_interior_index(h, w, device=device)
            print(f"[INFO] PDE guidance enabled: L={l_val} c={c_val} pml={pml} σ={sig}")

        rows_orig = []; rows_sc = []
        freq_orig = {m: [] for m in freq_ids}; freq_sc = {m: [] for m in freq_ids}
        vis_count = 0

        for i_case, (b_idx, m_idx) in enumerate(cases, start=1):
            gt         = data_ds[b_idx, m_idx].astype(np.float32)
            omega_val  = float(omega[m_idx])
            omega_cond = _normalize_omega(omega_val, omega_min, omega_max)

            if mask_ds.ndim == 4:
                mask_case = (mask_ds[m_idx].astype(np.float32) > 0.5)
            elif mask_ds.ndim == 5:
                mask_case = (mask_ds[b_idx, m_idx].astype(np.float32) > 0.5)
            else:
                raise ValueError(f"Invalid mask dims: {mask_ds.ndim}")

            idx_re = np.flatnonzero(mask_case[..., 0].reshape(-1))
            idx_im = np.flatnonzero(mask_case[..., 1].reshape(-1))
            if idx_re.size == 0 or idx_im.size == 0:
                continue

            a_re_t = torch.from_numpy(phi_np[idx_re]).to(device)
            a_im_t = torch.from_numpy(phi_np[idx_im]).to(device)
            y_re_t = torch.from_numpy(gt[..., 0].reshape(-1)[idx_re]).to(device).view(1, -1)
            y_im_t = torch.from_numpy(gt[..., 1].reshape(-1)[idx_im]).to(device).view(1, -1)

            # Physics per-case
            physics_case = None
            if use_phys and solver is not None:
                if m_idx not in a_cache:
                    A = solver._build_matrix(omega_val)
                    a_cache[m_idx] = (
                        _csc_to_torch_sparse(A.real.astype(np.float32), device),
                        _csc_to_torch_sparse(A.imag.astype(np.float32), device),
                    )
                a_re_op, a_im_op = a_cache[m_idx]

                if args.phys_use_source:
                    if b_idx not in source_cache:
                        source_cache[b_idx] = _load_source_field_for_sample(
                            f=f, sample_idx=b_idx, solver=solver, source_sigma=source_sigma)
                    src = source_cache[b_idx]
                    f_re_np = src.real.reshape(-1).astype(np.float32)
                    f_im_np = src.imag.reshape(-1).astype(np.float32)
                    if args.phys_scale_source_by_data:
                        f_re_np /= data_scale; f_im_np /= data_scale
                else:
                    f_re_np = np.zeros(h * w, dtype=np.float32)
                    f_im_np = np.zeros(h * w, dtype=np.float32)

                f_re_t = torch.from_numpy(f_re_np).to(device)
                f_im_t = torch.from_numpy(f_im_np).to(device)
                if interior_idx_t is not None:
                    fd_re = f_re_t[interior_idx_t]; fd_im = f_im_t[interior_idx_t]
                else:
                    fd_re = f_re_t; fd_im = f_im_t
                residual_den = torch.clamp(torch.mean(fd_re**2 + fd_im**2), min=1.0)

                physics_case = {
                    "phi_full": phi_t_full, "A_re": a_re_op, "A_im": a_im_op,
                    "f_re": f_re_t, "f_im": f_im_t,
                    "interior_idx": interior_idx_t, "residual_den": residual_den,
                }

            # Shared init noise
            g = torch.Generator(device=device)
            g.manual_seed(args.seed + 17 * i_case)
            init_noise = torch.randn((1, 2, rx, ry), generator=g, device=device)

            # ── Original DPS (from test_diffusion.py) ─────────────────────
            orig_core = sample_core(
                model=model, schedule=schedule, omega_norm=omega_cond,
                mean=mean, std=std,
                a_re=a_re_t, a_im=a_im_t, y_re=y_re_t, y_im=y_im_t,
                timestep_seq=timestep_seq,
                dps_weight=args.dps_weight, phys_weight=args.phys_weight,
                guidance_scale=args.guidance_scale,
                init_noise=init_noise, physics=physics_case,
                guidance_grad_clip=args.guidance_grad_clip, log_guidance=False,
                eps=args.eps,
            )

            # ── Self-calibrated DPS ─────────────────────────────────────────
            sc_core, step_log = sample_core_selfcal(
                model=model, schedule=schedule, omega_norm=omega_cond,
                mean=mean, std=std,
                a_re=a_re_t, a_im=a_im_t, y_re=y_re_t, y_im=y_im_t,
                timestep_seq=timestep_seq,
                lambda_dps=args.lambda_dps, lambda_pde=args.lambda_pde,
                guidance_scale=args.guidance_scale,
                r_obs_ref=args.r_obs_ref, r_pde_ref=args.r_pde_ref,
                s_min=args.s_min, s_max=args.s_max,
                init_noise=init_noise, physics=physics_case,
                guidance_grad_clip=args.guidance_grad_clip,
                log_guidance=args.log_guidance, eps=args.eps,
            )

            for label, core_np_norm, rows_list, freq_dict in [
                ("orig",   orig_core.detach().cpu().numpy(), rows_orig, freq_orig),
                ("selfcal",  sc_core.detach().cpu().numpy(), rows_sc,   freq_sc),
            ]:
                core_phys = (core_np_norm * std_np + mean_np).astype(np.float32)
                core_gt   = _solve_lstsq_core_from_gt(gt, phi_np, rx, ry)
                core_err  = _core_rel_err(core_phys, core_gt, eps=args.eps)
                pred      = _decode_field_from_core_image(core_np_norm, phi_np, mean_np, std_np, h, w)
                rmse      = _rel_rmse(pred, gt, eps=args.eps)
                obs_rmse  = _masked_rel_rmse(pred, gt, mask_case, eps=args.eps)
                unobs_rmse= _masked_rel_rmse(pred, gt, ~mask_case.astype(bool), eps=args.eps)
                try:
                    pde_res = evaluate_physics_residual(
                        pred=pred, h5_file=f, sample_idx=b_idx,
                        omega=omega_val, h5_meta=h5_meta)
                except Exception:
                    pde_res = np.nan
                rows_list.append({
                    "sample_idx": b_idx, "freq_idx": m_idx, "omega": omega_val,
                    "rmse": rmse, "obs_rmse": obs_rmse, "unobs_rmse": unobs_rmse,
                    "core_rel_err": core_err, "pde_res": pde_res,
                })
                if label == "selfcal":
                    freq_dict[m_idx].append(rmse)
                else:
                    freq_dict[m_idx].append(rmse)

            rmse_o = rows_orig[-1]["rmse"]; rmse_s = rows_sc[-1]["rmse"]
            delta = rmse_o - rmse_s
            print(f"[{i_case:03d}/{len(cases)}] s={b_idx} f={m_idx} ω={omega_val:.2f} | "
                  f"orig={rmse_o:.4e}  selfcal={rmse_s:.4e}  Δ={delta:+.4e} "
                  f"({'↑' if delta > 0 else '↓'})")

            if vis_count < args.num_visualize:
                vis_count += 1
                orig_np = _decode_field_from_core_image(
                    orig_core.detach().cpu().numpy(), phi_np, mean_np, std_np, h, w)
                sc_np = _decode_field_from_core_image(
                    sc_core.detach().cpu().numpy(), phi_np, mean_np, std_np, h, w)
                fig_path = out_dir / f"compare_{vis_count:03d}_s{b_idx:03d}_f{m_idx:03d}.png"
                _plot_compare(fig_path, gt, orig_np, sc_np, mask_case,
                              b_idx, m_idx, omega_val, rmse_o, rmse_s)

    # ── Summary ───────────────────────────────────────────────────────────────
    if not rows_orig:
        raise RuntimeError("No evaluation cases produced.")

    mean_orig  = float(np.mean([r["rmse"] for r in rows_orig]))
    mean_sc    = float(np.mean([r["rmse"] for r in rows_sc]))
    pde_orig   = float(np.nanmean([r["pde_res"] for r in rows_orig]))
    pde_sc     = float(np.nanmean([r["pde_res"] for r in rows_sc]))
    impr_pct   = 100.0 * (mean_orig - mean_sc) / max(mean_orig, 1e-12)

    summary = {
        "diff_ckpt": args.diff_ckpt, "ftm_ckpt": args.ftm_ckpt,
        "data_h5": args.data_h5, "num_cases": len(rows_orig),
        "mean_rmse_orig":    mean_orig,
        "mean_rmse_selfcal": mean_sc,
        "improvement_pct":   impr_pct,
        "mean_pde_res_orig":    pde_orig,
        "mean_pde_res_selfcal": pde_sc,
        "params": {
            "lambda_dps": args.lambda_dps, "lambda_pde": args.lambda_pde,
            "r_obs_ref": args.r_obs_ref, "r_pde_ref": args.r_pde_ref,
            "s_min": args.s_min, "s_max": args.s_max,
            "guidance_scale": args.guidance_scale,
            "orig_dps_weight": args.dps_weight, "orig_phys_weight": args.phys_weight,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Per-frequency curve
    unique_freqs = sorted(freq_ids)
    freq_to_omega = {}
    for row in rows_orig:
        fi = int(row["freq_idx"])
        if fi not in freq_to_omega: freq_to_omega[fi] = row["omega"]
    omega_arr = np.array([freq_to_omega[fi] for fi in unique_freqs])
    orig_arr  = np.array([np.mean(freq_orig[fi]) if freq_orig[fi] else np.nan for fi in unique_freqs])
    sc_arr    = np.array([np.mean(freq_sc[fi])   if freq_sc[fi]   else np.nan for fi in unique_freqs])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(omega_arr, orig_arr, label="Original DPS", lw=1.8)
    ax.plot(omega_arr, sc_arr,   label="Self-Cal DPS", lw=1.8, ls="--")
    ax.set_xlabel("ω"); ax.set_ylabel("Mean relative RMSE")
    ax.set_title(f"Freq-wise Error: orig={mean_orig:.4f}  selfcal={mean_sc:.4f}  Δ={impr_pct:+.2f}%")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "freq_rmse_curve.png", dpi=150)
    plt.close(fig)

    print("\n=== Self-calibrating guidance evaluation ===")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Self-calibrating DPS — FTM + diffusion (2D Helmholtz)")

    p.add_argument("--diff_ckpt", default="ckp/diffusion_core.pt")
    p.add_argument("--ftm_ckpt",  default="ckp/ftm_gpu_checkpoint.pt")
    p.add_argument("--data_h5",   default="data_for_test/helmholtz_dataset_42_for_test_mask2.h5")
    p.add_argument("--out_dir",   default="visual_data/selfcal_test")

    p.add_argument("--freq_indices",  type=str, default="")
    p.add_argument("--max_samples",   type=int, default=3)
    p.add_argument("--max_cases",     type=int, default=0)
    p.add_argument("--sample_steps",  type=int, default=500)
    p.add_argument("--num_visualize", type=int, default=20)

    # Self-cal params
    p.add_argument("--lambda_dps",  type=float, default=1.0,
                   help="Base observation guidance scale (replaces dps_weight in selfcal)")
    p.add_argument("--lambda_pde",  type=float, default=0.02,
                   help="Base PDE guidance scale (replaces phys_weight in selfcal)")
    p.add_argument("--r_obs_ref",   type=float, default=0.3,
                   help="Reference obs residual for calibration (tune to typical obs_res at step T/2)")
    p.add_argument("--r_pde_ref",   type=float, default=1.0,
                   help="Reference PDE residual for calibration")
    p.add_argument("--s_min",       type=float, default=0.2,
                   help="Minimum scale factor multiplier")
    p.add_argument("--s_max",       type=float, default=3.0,
                   help="Maximum scale factor multiplier")

    # Original params (for comparison baseline)
    p.add_argument("--dps_weight",        type=float, default=1.0)
    p.add_argument("--phys_weight",       type=float, default=0.02)
    p.add_argument("--guidance_scale",    type=float, default=15.0)
    p.add_argument("--guidance_grad_clip", type=float, default=1e-3)

    # Physics
    p.add_argument("--phys_use_source",           action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--phys_scale_source_by_data", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--phys_interior_only",        action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--helmholtz_L",          type=float, default=1.0)
    p.add_argument("--helmholtz_c",          type=float, default=1.0)
    p.add_argument("--helmholtz_pml_width",  type=float, default=0.12)
    p.add_argument("--helmholtz_sigma_max",  type=float, default=50.0)
    p.add_argument("--source_sigma",         type=float, default=0.025)

    p.add_argument("--eps",         type=float, default=1e-4)
    p.add_argument("--log_guidance", action="store_true", default=False)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      type=str, default="auto")

    return p


def main() -> None:
    evaluate_selfcal(build_parser().parse_args())


if __name__ == "__main__":
    main()
