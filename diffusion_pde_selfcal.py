"""
diffusion_pde_selfcal.py
------------------------
Self-calibrating DPS guidance for the 2D Helmholtz DiffusionPDE baseline.

Extends diffusion_pde_baseline.py with two calibration mechanisms:

1. DPS observation guidance with SELF-CALIBRATED ζ
   Instead of a fixed zeta, we adapt the guidance strength each step based on
   the ratio of the observation residual to the prior-step magnitude:

       ζ_t = λ_dps * ‖Δx_prior‖ / (‖g_t‖ + ε)        (same as fixed-ζ DPS)

   PLUS dynamic decay: if the residual is already small (< thresh_low) we
   reduce ζ to avoid over-shooting; if it is large (> thresh_high) we boost
   it up to λ_dps_max. This acts as a soft PID controller on the guidance.

2. PDE residual guidance with SELF-CALIBRATED weight
   At each step we evaluate the Helmholtz residual ‖Au - f‖ and compare
   it to the observation residual ‖M(x̂₀) - y‖.  The PDE weight is adapted
   so that neither signal dominates:

       w_pde_t = λ_pde * (obs_res / (pde_res + ε))

   This balances data-fidelity vs physics-consistency automatically.

Usage (eval only – reuse trained checkpoint from diffusion_pde_baseline.py):
    python diffusion_pde_selfcal.py --mode eval \\
        --ckpt ckp/dpde_baseline.pt \\
        --test_h5 data_for_test/helmholtz_dataset_42_for_test_mask2.h5 \\
        --max_samples 3 --out_dir visual_data/dpde_selfcal_test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse all model/training code from the baseline
from diffusion_pde_baseline import (
    HelmholtzDPDEData,
    PixelScoreUNet,
    DiffusionSchedule,
    _relative_rmse,
    _masked_relative_rmse,
    _plot_eval,
    _load_meta,
    _collate_eval,
    _safe_load,
    set_seed,
)

try:
    from physics_metric import evaluate_physics_residual
    _HAS_PHYS = True
except ImportError:
    _HAS_PHYS = False

try:
    from Helmholtz_Solver import HelmholtzSolver
    _HAS_SOLVER = True
except ImportError:
    _HAS_SOLVER = False

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helmholtz PDE residual helper (pixel-space, no FTM basis)
# ---------------------------------------------------------------------------

def _build_helmholtz_A(solver: Any, omega: float, device: torch.device):
    """Cache the sparse Helmholtz operator for a given frequency."""
    import scipy.sparse as sp
    A = solver._build_matrix(float(omega))  # complex CSC
    A_re = A.real.astype(np.float32).tocoo()
    A_im = A.imag.astype(np.float32).tocoo()

    def _to_sparse(coo):
        idx = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
        val = torch.from_numpy(coo.data)
        return torch.sparse_coo_tensor(idx, val, size=coo.shape, device=device).coalesce()

    return _to_sparse(A_re), _to_sparse(A_im)


def _pde_residual_pixel(
    field_re: torch.Tensor,   # (N,) flattened real part of predicted field
    field_im: torch.Tensor,   # (N,) flattened imag part
    A_re: torch.Tensor,       # sparse (N, N)
    A_im: torch.Tensor,       # sparse (N, N)
    f_re: torch.Tensor,       # (N,) source real
    f_im: torch.Tensor,       # (N,) source imag
    interior_idx: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return normalised Helmholtz residual ‖Au - f‖² / ‖f‖² for a single field."""
    u_re = field_re.unsqueeze(1)   # (N, 1)
    u_im = field_im.unsqueeze(1)

    au_re = torch.sparse.mm(A_re, u_re) - torch.sparse.mm(A_im, u_im)
    au_im = torch.sparse.mm(A_re, u_im) + torch.sparse.mm(A_im, u_re)

    res_re = (au_re.squeeze(1) + f_re)
    res_im = (au_im.squeeze(1) + f_im)

    if interior_idx is not None:
        res_re = res_re[interior_idx]
        res_im = res_im[interior_idx]
        f_re   = f_re[interior_idx]
        f_im   = f_im[interior_idx]

    num = (res_re ** 2 + res_im ** 2).mean()
    den = (f_re ** 2 + f_im ** 2).mean().clamp(min=eps)
    return num / den


# ---------------------------------------------------------------------------
# Self-calibrating DPS sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def dps_selfcal_sample(
    model: PixelScoreUNet,
    sched: DiffusionSchedule,
    omega: torch.Tensor,            # (B, 1) normalised ω
    y_obs: torch.Tensor,            # (B, C, H, W)  sparse obs (0 outside mask)
    mask: torch.Tensor,             # (B, 1, H, W)  binary mask
    T: int,
    # --- DPS calibration params ---
    lambda_dps: float = 0.3,        # base DPS scale (same role as fixed zeta)
    lambda_dps_max: float = 2.0,    # clip ceiling for adaptive boost factor
    thresh_low: float = 0.05,       # obs-residual below this → gently reduce guidance
    thresh_high: float = 0.5,       # obs-residual above this → clip back (model not ready)
    # --- PDE calibration params ---
    lambda_pde: float = 0.0,        # base PDE guidance weight (0 = disabled)
    pde_physics: Optional[Dict] = None,
    # --- misc ---
    device: torch.device = torch.device("cpu"),
    eps: float = 1e-8,
    log_steps: bool = False,
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Self-calibrating DPS sampler.

    Calibration strategy:
    - Scale-invariant base: ζ_t = λ * ‖Δx_prior‖ / ‖g_t‖  (same as fixed-ζ DPS)
    - Soft boost when residual is in a moderate range (model making progress):
        boost = sigmoid-smoothed ramp from 0.5 to lambda_dps_max over [thresh_low, thresh_high]
    - When obs_res > thresh_high (high freq, poor fit), the model signal is noisy →
      clip boost to 1.0 to avoid overshooting (conservative approach)
    - When obs_res < thresh_low (nearly converged) → gentle decay to 0.5 to avoid jitter

    The key insight: scale-invariant DPS already normalises for magnitude via
    prior_step_norm / g_norm. The boost only modulates whether the step is
    slightly larger or smaller, not arbitrarily large.

    Returns:
        x  : (B, C, H, W) final prediction
        log: list of per-step dicts
    """
    B, C, H, W = y_obs.shape
    x = torch.randn(B, C, H, W, device=device, dtype=torch.float32)
    step_log: List[Dict] = []

    for t_idx in reversed(range(T)):
        t_vec = torch.full((B,), t_idx, device=device, dtype=torch.long)

        # ── Score prediction (frozen) ──────────────────────────────────────
        eps_pred = model(x, t_vec, omega)

        ab_t  = sched.alpha_bars[t_idx]
        a_t   = sched.alphas[t_idx]
        b_t   = sched.betas[t_idx]

        # ── DDPM prior step ────────────────────────────────────────────────
        x0hat   = (x - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt().clamp(min=eps)
        noise   = torch.randn_like(x) if t_idx > 0 else torch.zeros_like(x)
        x_prior = (1.0 / a_t.sqrt()) * (
            x - (1.0 - a_t) / (1.0 - ab_t).sqrt() * eps_pred
        ) + b_t.sqrt() * noise

        prior_step_norm = (x_prior - x).pow(2).sum(dim=(1,2,3), keepdim=True).sqrt().clamp(min=eps)

        # ── DPS observation guidance ───────────────────────────────────────
        residual = mask * x0hat - y_obs          # (B, C, H, W)
        g_t      = mask * residual

        g_norm = g_t.pow(2).sum(dim=(1,2,3), keepdim=True).sqrt().clamp(min=eps)

        # Normalised obs residual (relative L2 at observed points)
        y_norm2 = y_obs.pow(2).sum(dim=(1,2,3)).clamp(min=eps)
        obs_res_norm = float(
            (residual.pow(2).sum(dim=(1,2,3)) / y_norm2).sqrt().mean().item()
        )

        # Self-calibrated boost factor — bounded in [0.5, lambda_dps_max]
        # Low residual (<thresh_low): already converged, gently reduce to avoid jitter
        # Mid residual [thresh_low, thresh_high]: linearly ramp up to lambda_dps_max
        # High residual (>thresh_high): model signal noisy, cap at 1.0 (conservative)
        if obs_res_norm < thresh_low:
            boost = 0.5 + 0.5 * (obs_res_norm / (thresh_low + eps))
        elif obs_res_norm <= thresh_high:
            t_frac = (obs_res_norm - thresh_low) / (thresh_high - thresh_low + eps)
            boost = 1.0 + (lambda_dps_max - 1.0) * t_frac
        else:
            # High residual: conservative — stick to base scale
            boost = 1.0

        # Clip boost to [0.1, lambda_dps_max] for safety
        boost = max(0.1, min(boost, lambda_dps_max))

        zeta_t = lambda_dps * boost * prior_step_norm / g_norm   # (B,1,1,1)

        x_new = x_prior - zeta_t * g_t

        # ── PDE residual guidance (optional) ──────────────────────────────
        pde_res_val = float("nan")
        pde_w_used  = 0.0
        if lambda_pde > 0.0 and pde_physics is not None:
            # Compute PDE residual on current x̂₀ estimate
            # Average over batch (usually B=1 for DPS eval)
            pde_losses = []
            for b in range(B):
                # Decode pixel field from x̂₀ for this sample
                f_re_b = x0hat[b, 0].reshape(-1)   # (H*W,)
                f_im_b = x0hat[b, 1].reshape(-1) if C > 1 else torch.zeros_like(f_re_b)
                pde_l  = _pde_residual_pixel(
                    f_re_b, f_im_b,
                    pde_physics["A_re"], pde_physics["A_im"],
                    pde_physics["f_re"], pde_physics["f_im"],
                    pde_physics.get("interior_idx"),
                    eps=eps,
                )
                pde_losses.append(pde_l)
            pde_loss_t = torch.stack(pde_losses).mean()
            pde_res_val = float(pde_loss_t.item())

            # Self-calibrate PDE weight: proportional to obs/pde residual ratio
            # If obs residual is small but PDE large → upweight PDE guidance
            pde_res_safe = max(pde_res_val, eps)
            adaptive_pde_w = lambda_pde * (obs_res_norm / pde_res_safe)
            # Clip to avoid runaway
            adaptive_pde_w = min(adaptive_pde_w, lambda_pde * 10.0)
            pde_w_used = adaptive_pde_w

            # PDE gradient: needs autograd — enable grad temporarily
            x0_var = x0hat.detach().requires_grad_(True)
            pde_losses_g = []
            for b in range(B):
                f_re_b = x0_var[b, 0].reshape(-1)
                f_im_b = x0_var[b, 1].reshape(-1) if C > 1 else torch.zeros_like(f_re_b)
                pde_losses_g.append(
                    _pde_residual_pixel(
                        f_re_b, f_im_b,
                        pde_physics["A_re"], pde_physics["A_im"],
                        pde_physics["f_re"], pde_physics["f_im"],
                        pde_physics.get("interior_idx"),
                        eps=eps,
                    )
                )
            pde_grad_loss = torch.stack(pde_losses_g).mean()
            pde_grad = torch.autograd.grad(pde_grad_loss, x0_var)[0].detach()

            # Normalise PDE gradient to match prior_step_norm scale
            pde_grad_norm = pde_grad.pow(2).sum(dim=(1,2,3), keepdim=True).sqrt().clamp(min=eps)
            pde_step = adaptive_pde_w * prior_step_norm / pde_grad_norm * pde_grad
            x_new = x_new - pde_step

        x = x_new

        if log_steps and (t_idx % max(T // 20, 1) == 0 or t_idx == 0):
            step_log.append({
                "t": t_idx, "obs_res": obs_res_norm,
                "zeta_scalar": float(zeta_t.mean().item()),
                "boost": boost, "pde_res": pde_res_val, "pde_w": pde_w_used,
            })

    return x, step_log


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _csc_to_torch_sparse(mat: Any, device: torch.device) -> torch.Tensor:
    coo = mat.tocoo()
    idx = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    val = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, size=coo.shape, device=device).coalesce()


def _build_interior_idx(H: int, W: int, device: torch.device) -> torch.Tensor:
    m = np.ones((H, W), dtype=bool)
    m[[0, -1], :] = False; m[:, [0, -1]] = False
    return torch.from_numpy(np.flatnonzero(m.reshape(-1)).astype(np.int64)).to(device)


def evaluate_selfcal(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    ckpt  = _safe_load(Path(args.ckpt))
    model = PixelScoreUNet(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dcfg  = ckpt.get("diffusion_config", {})
    T     = dcfg.get("T", args.T)
    sched = DiffusionSchedule.linear(
        T, dcfg.get("beta_start", args.beta_start),
        dcfg.get("beta_end", args.beta_end)
    ).to(device)

    data_path = Path(args.test_h5)
    meta = _load_meta(data_path)
    N    = int(meta["shape"][0])
    max_s = N if args.max_samples <= 0 else min(N, args.max_samples)
    ds   = HelmholtzDPDEData(data_path, list(range(max_s)), args.eval_freq_indices, return_obs=True)
    dl   = DataLoader(ds, args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate_eval)

    from torch.utils.data import DataLoader

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    h5_file = h5py.File(data_path, "r")
    h5_meta = meta["meta"]

    # ── Physics setup (optional) ───────────────────────────────────────────
    solver = None
    A_cache: Dict[float, Tuple] = {}
    use_pde = args.lambda_pde > 0.0 and _HAS_SOLVER

    if use_pde:
        L   = float(h5_meta.get("L", args.helmholtz_L))
        c   = float(h5_meta.get("c", args.helmholtz_c))
        pw  = float(h5_meta.get("pml_width", args.helmholtz_pml_width))
        sm  = float(h5_meta.get("sigma_max", args.helmholtz_sigma_max))
        solver = HelmholtzSolver(N=meta["shape"][2], L=L, c=c, pml_width=pw, sigma_max=sm)
        print(f"[PDE] Helmholtz solver ready: L={L} c={c} pml={pw} sigma={sm}")

    rows: List[Dict] = []; vis_cnt = 0

    for batch in dl:
        y       = batch["y"].to(device)
        omega   = batch["omega"].to(device)
        mask    = batch["mask"].to(device)
        obs     = batch["obs"].to(device)
        s_idx   = batch["sample_idx"].cpu().numpy()
        f_idx   = batch["freq_idx"].cpu().numpy()
        omega_r = batch["omega_raw"].cpu().numpy()

        B, C, H, W = y.shape

        # Process per-sample (physics is per-frequency, so share within batch if same freq)
        preds = []
        for i in range(B):
            omega_i = float(omega_r[i])

            physics_i: Optional[Dict] = None
            if use_pde and solver is not None:
                if omega_i not in A_cache:
                    A = solver._build_matrix(omega_i)
                    A_cache[omega_i] = (
                        _csc_to_torch_sparse(A.real.astype(np.float32), device),
                        _csc_to_torch_sparse(A.imag.astype(np.float32), device),
                    )
                A_re_op, A_im_op = A_cache[omega_i]
                interior_idx = _build_interior_idx(H, W, device)
                # Zero source (homogeneous approximation)
                physics_i = {
                    "A_re": A_re_op, "A_im": A_im_op,
                    "f_re": torch.zeros(H * W, device=device),
                    "f_im": torch.zeros(H * W, device=device),
                    "interior_idx": interior_idx,
                }

            pred_i, step_log = dps_selfcal_sample(
                model=model, sched=sched,
                omega=omega[i:i+1],
                y_obs=obs[i:i+1], mask=mask[i:i+1],
                T=T,
                lambda_dps=args.lambda_dps,
                lambda_dps_max=args.lambda_dps_max,
                thresh_low=args.thresh_low,
                thresh_high=args.thresh_high,
                lambda_pde=args.lambda_pde,
                pde_physics=physics_i,
                device=device,
                eps=args.eps,
                log_steps=args.log_guidance,
            )
            preds.append(pred_i)

            if args.log_guidance and step_log:
                print(f"  [s={int(s_idx[i])} f={int(f_idx[i])}] step-log sample:")
                for entry in step_log[:3]:
                    print(f"    t={entry['t']:3d}  obs_res={entry['obs_res']:.3e}  "
                          f"ζ={entry['zeta_scalar']:.3e}  boost={entry['boost']:.2f}  "
                          f"pde_res={entry['pde_res']:.3e}  pde_w={entry['pde_w']:.3e}")

        pred_t  = torch.cat(preds, dim=0)   # (B, C, H, W)
        pred_np = pred_t.cpu().numpy()
        gt_np   = y.cpu().numpy()
        mask_np = batch["mask"].numpy()

        for i in range(B):
            pred_i  = pred_np[i].transpose(1,2,0)
            gt_i    = gt_np[i].transpose(1,2,0)
            mask_i  = mask_np[i].transpose(1,2,0)
            rmse    = _relative_rmse(pred_i, gt_i)
            obs_r   = _masked_relative_rmse(pred_i, gt_i, mask_i)
            unobs_r = _masked_relative_rmse(pred_i, gt_i, 1.0 - mask_i)
            pde_res = float("nan")
            if _HAS_PHYS:
                try:
                    pde_res = evaluate_physics_residual(
                        pred=pred_i, h5_file=h5_file,
                        sample_idx=int(s_idx[i]), omega=float(omega_r[i]),
                        h5_meta=h5_meta)
                except Exception:
                    pass

            row = {"sample_idx": int(s_idx[i]), "freq_idx": int(f_idx[i]),
                   "omega": float(omega_r[i]), "rmse": rmse,
                   "obs_rmse": obs_r, "unobs_rmse": unobs_r, "pde_res": pde_res}
            rows.append(row)
            print(f"  sample={int(s_idx[i])} freq={int(f_idx[i])} ω={float(omega_r[i]):.3f} "
                  f"rmse={rmse:.4e} obs={obs_r:.4e} unobs={unobs_r:.4e} pde={pde_res:.3e}")

            if vis_cnt < args.num_visualize:
                vis_cnt += 1
                vp = out_dir / f"case{vis_cnt:03d}_s{int(s_idx[i]):03d}_f{int(f_idx[i]):03d}.png"
                _plot_eval(vp, gt_i, pred_i, mask_i,
                           int(s_idx[i]), int(f_idx[i]), float(omega_r[i]),
                           rmse, obs_r, unobs_r, pde_res, dpi=args.vis_dpi)
                rows[-1]["vis_path"] = str(vp)

    h5_file.close()
    if not rows:
        raise RuntimeError("No evaluation rows produced.")

    csv_path = out_dir / "metrics_cases.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, ["sample_idx","freq_idx","omega","rmse",
                               "obs_rmse","unobs_rmse","pde_res","vis_path"])
        w.writeheader()
        for r in rows: w.writerow(r)

    summary = {
        "ckpt": args.ckpt, "test_h5": str(data_path), "num_cases": len(rows),
        "method": "self-calibrated-DPS",
        "lambda_dps": args.lambda_dps,
        "lambda_dps_max": args.lambda_dps_max,
        "thresh_low": args.thresh_low, "thresh_high": args.thresh_high,
        "lambda_pde": args.lambda_pde,
        "mean_rmse":       float(np.mean([r["rmse"]      for r in rows])),
        "mean_obs_rmse":   float(np.mean([r["obs_rmse"]  for r in rows])),
        "mean_unobs_rmse": float(np.mean([r["unobs_rmse"]for r in rows])),
        "mean_pde_res":    float(np.nanmean([r["pde_res"]for r in rows])),
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Self-calibrating DPS evaluation finished ===")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Comparison: run fixed-ζ DPS and self-cal DPS side-by-side
# ---------------------------------------------------------------------------

def evaluate_compare(args: argparse.Namespace) -> None:
    """Run both fixed-ζ and self-cal and print side-by-side metrics."""
    from diffusion_pde_baseline import dps_sample as dps_fixed

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    ckpt  = _safe_load(Path(args.ckpt))
    model = PixelScoreUNet(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dcfg  = ckpt.get("diffusion_config", {})
    T     = dcfg.get("T", args.T)
    sched = DiffusionSchedule.linear(
        T, dcfg.get("beta_start", args.beta_start),
        dcfg.get("beta_end", args.beta_end)
    ).to(device)

    data_path = Path(args.test_h5)
    meta = _load_meta(data_path)
    N    = int(meta["shape"][0])
    max_s = N if args.max_samples <= 0 else min(N, args.max_samples)
    ds   = HelmholtzDPDEData(data_path, list(range(max_s)), args.eval_freq_indices, return_obs=True)

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, 1, shuffle=False, num_workers=0, collate_fn=_collate_eval)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    h5_file = h5py.File(data_path, "r")
    h5_meta = meta["meta"]

    rows_fixed  = []
    rows_selfcal = []

    for batch in dl:
        y       = batch["y"].to(device)
        omega   = batch["omega"].to(device)
        mask    = batch["mask"].to(device)
        obs     = batch["obs"].to(device)
        s_i     = int(batch["sample_idx"][0])
        f_i     = int(batch["freq_idx"][0])
        omega_v = float(batch["omega_raw"][0])

        # Fixed-ζ DPS
        pred_fixed = dps_fixed(model, sched, omega, obs, mask, T, args.zeta_fixed, device)

        # Self-calibrated DPS
        pred_sc, _ = dps_selfcal_sample(
            model, sched, omega, obs, mask, T,
            lambda_dps=args.lambda_dps,
            lambda_dps_max=args.lambda_dps_max,
            thresh_low=args.thresh_low,
            thresh_high=args.thresh_high,
            lambda_pde=args.lambda_pde,
            pde_physics=None,
            device=device, eps=args.eps,
            log_steps=False,
        )

        gt_np    = y[0].cpu().numpy().transpose(1,2,0)
        mask_np  = batch["mask"][0].numpy().transpose(1,2,0)
        pf_np    = pred_fixed[0].cpu().numpy().transpose(1,2,0)
        psc_np   = pred_sc[0].cpu().numpy().transpose(1,2,0)

        for label, pred_np, rows_list in [
            ("fixed",   pf_np,  rows_fixed),
            ("selfcal", psc_np, rows_selfcal),
        ]:
            rmse    = _relative_rmse(pred_np, gt_np)
            obs_r   = _masked_relative_rmse(pred_np, gt_np, mask_np)
            unobs_r = _masked_relative_rmse(pred_np, gt_np, 1.0 - mask_np)
            pde_res = float("nan")
            if _HAS_PHYS:
                try:
                    pde_res = evaluate_physics_residual(
                        pred=pred_np, h5_file=h5_file,
                        sample_idx=s_i, omega=omega_v, h5_meta=h5_meta)
                except Exception:
                    pass
            rows_list.append({"s": s_i, "f": f_i, "rmse": rmse,
                               "obs_rmse": obs_r, "unobs_rmse": unobs_r, "pde_res": pde_res})

        print(f"s={s_i} f={f_i} ω={omega_v:.3f} | "
              f"fixed rmse={rows_fixed[-1]['rmse']:.4e}  "
              f"selfcal rmse={rows_selfcal[-1]['rmse']:.4e}  "
              f"Δ={rows_fixed[-1]['rmse'] - rows_selfcal[-1]['rmse']:+.4e}")

        # Save comparison figure
        gt_re = gt_np[..., 0]; gt_im = gt_np[..., 1]
        pf_re = pf_np[..., 0]; psc_re = psc_np[..., 0]
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        for ax, (img, title, cmap) in zip(axes.flat, [
            (gt_re,  "GT Real", "viridis"),
            (pf_re,  "Fixed-ζ Real", "viridis"),
            (psc_re, "SelfCal Real", "viridis"),
            (np.abs(pf_re - gt_re) - np.abs(psc_re - gt_re), "Err(fixed)-Err(sc) Re", "RdBu_r"),
            (gt_im,  "GT Imag", "viridis"),
            (pf_np[...,1], "Fixed-ζ Imag", "viridis"),
            (psc_np[...,1],"SelfCal Imag", "viridis"),
            (np.abs(pf_np[...,1]-gt_im) - np.abs(psc_np[...,1]-gt_im), "Err(fixed)-Err(sc) Im", "RdBu_r"),
        ]):
            im = ax.imshow(img, origin="lower", cmap=cmap)
            ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"s={s_i} f={f_i} ω={omega_v:.3f}  |  "
            f"fixed RMSE={rows_fixed[-1]['rmse']:.4e}  "
            f"selfcal RMSE={rows_selfcal[-1]['rmse']:.4e}",
            y=0.995)
        fig.tight_layout()
        fig.savefig(out_dir / f"compare_s{s_i:03d}_f{f_i:03d}.png", dpi=150)
        plt.close(fig)

    h5_file.close()

    mean_rmse_fixed   = float(np.mean([r["rmse"] for r in rows_fixed]))
    mean_rmse_selfcal = float(np.mean([r["rmse"] for r in rows_selfcal]))
    mean_pde_fixed    = float(np.nanmean([r["pde_res"] for r in rows_fixed]))
    mean_pde_selfcal  = float(np.nanmean([r["pde_res"] for r in rows_selfcal]))

    summary = {
        "method": "compare_fixed_vs_selfcal",
        "ckpt": args.ckpt, "test_h5": str(data_path), "num_cases": len(rows_fixed),
        "fixed_zeta": args.zeta_fixed,
        "selfcal_lambda_dps": args.lambda_dps,
        "selfcal_lambda_pde": args.lambda_pde,
        "mean_rmse_fixed":      mean_rmse_fixed,
        "mean_rmse_selfcal":    mean_rmse_selfcal,
        "improvement_pct":      100.0 * (mean_rmse_fixed - mean_rmse_selfcal) / max(mean_rmse_fixed, 1e-12),
        "mean_pde_res_fixed":   mean_pde_fixed,
        "mean_pde_res_selfcal": mean_pde_selfcal,
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Comparison finished ===")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Self-calibrating DPS guidance — 2D Helmholtz")
    p.add_argument("--mode", choices=["eval", "compare"], default="compare",
                   help="eval: selfcal only; compare: fixed-ζ vs selfcal side-by-side")

    # Shared
    p.add_argument("--ckpt",    default="ckp/dpde_baseline.pt")
    p.add_argument("--test_h5", default="data_for_test/helmholtz_dataset_42_for_test_mask2.h5")
    p.add_argument("--out_dir", default="visual_data/dpde_selfcal")
    p.add_argument("--eval_freq_indices", type=str, default="")
    p.add_argument("--max_samples", type=int, default=3)
    p.add_argument("--batch_size",  type=int, default=1)
    p.add_argument("--num_visualize", type=int, default=20)
    p.add_argument("--vis_dpi",   type=int, default=150)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--device",    type=str, default="auto")
    p.add_argument("--log_guidance", action="store_true", default=False)

    # Diffusion schedule (fallback if not in ckpt)
    p.add_argument("--T",          type=int,   default=500)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end",   type=float, default=2e-2)
    p.add_argument("--eps",        type=float, default=1e-8)

    # Self-calibrated DPS params
    p.add_argument("--lambda_dps",      type=float, default=0.3,
                   help="Base DPS guidance scale (same role as fixed zeta)")
    p.add_argument("--lambda_dps_max",  type=float, default=2.0,
                   help="Max adaptive boost multiplier for DPS guidance")
    p.add_argument("--thresh_low",      type=float, default=0.05,
                   help="Obs-residual below this → reduce guidance")
    p.add_argument("--thresh_high",     type=float, default=0.5,
                   help="Obs-residual above this → boost guidance")

    # Self-calibrated PDE params
    p.add_argument("--lambda_pde",       type=float, default=0.0,
                   help="Base PDE guidance weight (0 = disabled)")
    p.add_argument("--helmholtz_L",      type=float, default=1.0)
    p.add_argument("--helmholtz_c",      type=float, default=1.0)
    p.add_argument("--helmholtz_pml_width",  type=float, default=0.12)
    p.add_argument("--helmholtz_sigma_max",  type=float, default=50.0)

    # Compare mode: fixed-ζ baseline
    p.add_argument("--zeta_fixed", type=float, default=0.3,
                   help="Fixed-ζ for the baseline DPS in compare mode")

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "eval":
        evaluate_selfcal(args)
    else:
        evaluate_compare(args)


if __name__ == "__main__":
    main()
