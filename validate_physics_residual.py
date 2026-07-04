"""
validate_physics_residual.py
----------------------------
Standalone check for the physics residual implementation used in test_diffusion.py.

What this script does
---------------------
1) Load a generated Helmholtz dataset (HDF5).
2) Build the same discrete Helmholtz operator A(omega) used by the solver.
3) Reuse _pde_residual_loss from test_diffusion.py.
4) Compare:
   - residual from _pde_residual_loss (Torch path)
   - direct residual computed with NumPy/Scipy using the same reconstructed field
5) Also report direct residual on the true field from dataset.

A close match between (Torch path) and (NumPy projected path) validates
that the residual function implementation is correct.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
import numpy as np
import torch

from Helmholtz_Solver import HelmholtzSolver
from test_diffusion import (
    _build_interior_index,
    _csc_to_torch_sparse,
    _load_data_scale,
    _load_h5_metadata_dict,
    _load_source_field_for_sample,
    _parse_freq_indices,
    _pde_residual_loss,
)
from train_FTM_GPU import MLP1D, build_phi, normalize_coords_to_unit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load_ftm_basis(ftm_ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    if not ftm_ckpt_path.exists():
        raise FileNotFoundError(f"FTM checkpoint not found: {ftm_ckpt_path}")

    ckpt = torch.load(ftm_ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})

    required = ["rank_x", "rank_y", "hidden_dim", "hidden_layers", "activation"]
    for key in required:
        if key not in cfg:
            raise KeyError(f"FTM checkpoint config missing key: {key}")

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
        "normalize_coords": bool(cfg.get("normalize_coords", True)),
    }


def _direct_residual_metric(
    a_mat: Any,
    u_re_vec: np.ndarray,
    u_im_vec: np.ndarray,
    f_re_vec: np.ndarray,
    f_im_vec: np.ndarray,
    interior_mask: Optional[np.ndarray],
    eps: float,
) -> float:
    u_complex = u_re_vec.astype(np.float64) + 1j * u_im_vec.astype(np.float64)
    f_complex = f_re_vec.astype(np.float64) + 1j * f_im_vec.astype(np.float64)

    res = a_mat.dot(u_complex) + f_complex
    res_re = res.real.astype(np.float64)
    res_im = res.imag.astype(np.float64)
    f_re = f_re_vec.astype(np.float64)
    f_im = f_im_vec.astype(np.float64)

    if interior_mask is not None:
        res_re = res_re[interior_mask]
        res_im = res_im[interior_mask]
        f_re = f_re[interior_mask]
        f_im = f_im[interior_mask]

    num = float(np.mean(res_re * res_re + res_im * res_im))
    den = float(np.mean(f_re * f_re + f_im * f_im))
    den = max(den, 1.0)
    return float(num / max(den, eps))


def _build_case_list(
    num_samples: int,
    num_freqs: int,
    freq_indices_text: str,
    max_samples: int,
    max_cases: int,
    seed: int,
) -> List[Tuple[int, int]]:
    freq_ids = _parse_freq_indices(freq_indices_text, num_freqs)
    b_eval = num_samples if max_samples <= 0 else min(num_samples, max_samples)
    sample_ids = list(range(b_eval))

    all_cases = [(b, m) for b in sample_ids for m in freq_ids]
    if not all_cases:
        raise ValueError("No evaluation cases.")

    if max_cases > 0 and len(all_cases) > max_cases:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(all_cases), size=max_cases, replace=False)
        return [all_cases[int(i)] for i in sel]

    return all_cases


def _plot_case_comparison(
    out_path: Path,
    u_re_true: np.ndarray,
    u_im_true: np.ndarray,
    u_re_proj: np.ndarray,
    u_im_proj: np.ndarray,
    sample_idx: int,
    freq_idx: int,
    omega_val: float,
    residual_func: float,
    residual_direct_proj: float,
    residual_direct_true: float,
    rel_diff: float,
    dpi: int,
) -> None:
    gt_amp = np.sqrt(u_re_true * u_re_true + u_im_true * u_im_true)
    pr_amp = np.sqrt(u_re_proj * u_re_proj + u_im_proj * u_im_proj)
    amp_err = np.abs(pr_amp - gt_amp)

    vmax_re = max(float(np.max(np.abs(u_re_true))), float(np.max(np.abs(u_re_proj))), 1e-8)
    vmax_im = max(float(np.max(np.abs(u_im_true))), float(np.max(np.abs(u_im_proj))), 1e-8)
    vmax_amp = max(float(np.max(gt_amp)), float(np.max(pr_amp)), 1e-8)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    items = [
        (u_re_true, "True Real", "seismic", -vmax_re, vmax_re),
        (u_re_proj, "Reconstructed Real", "seismic", -vmax_re, vmax_re),
        (np.abs(u_re_proj - u_re_true), "Abs Err Real", "magma", 0.0, None),
        (u_im_true, "True Imag", "seismic", -vmax_im, vmax_im),
        (u_im_proj, "Reconstructed Imag", "seismic", -vmax_im, vmax_im),
        (np.abs(u_im_proj - u_im_true), "Abs Err Imag", "magma", 0.0, None),
        (gt_amp, "True Amp", "viridis", 0.0, vmax_amp),
        (pr_amp, "Reconstructed Amp", "viridis", 0.0, vmax_amp),
        (amp_err, "Abs Err Amp", "magma", 0.0, None),
    ]

    for ax, (img, title, cmap, vmin, vmax) in zip(axes.flat, items):
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"sample={sample_idx}, freq={freq_idx}, omega={omega_val:.4f} | "
        f"func={residual_func:.3e}, direct_proj={residual_direct_proj:.3e}, "
        f"direct_true={residual_direct_true:.3e}, rel_diff={rel_diff:.3e}",
        y=0.99,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data_h5_path = Path(args.data_h5)
    ftm_ckpt_path = Path(args.ftm_ckpt)

    ftm = _load_ftm_basis(ftm_ckpt_path, device=device)

    with h5py.File(data_h5_path, "r") as f:
        if "omega" not in f:
            raise KeyError("HDF5 must contain omega")

        has_data = "data" in f
        has_fields = "fields_real" in f and "fields_imag" in f
        if not has_data and not has_fields:
            raise KeyError("HDF5 must contain either data or (fields_real, fields_imag)")

        if has_data:
            data_ds = f["data"]
            b_data, m_data, h, w, c = data_ds.shape
            if c != 2:
                raise ValueError("data must have 2 channels")
        else:
            re_ds = f["fields_real"]
            im_ds = f["fields_imag"]
            b_data, m_data, h, w = re_ds.shape
            if im_ds.shape != re_ds.shape:
                raise ValueError("fields_real/fields_imag shape mismatch")

        if h != w:
            raise ValueError("Only square grids are supported for this validator")

        omega = f["omega"][...].astype(np.float32)
        if omega.shape[0] != m_data:
            raise ValueError(f"omega shape mismatch: {omega.shape} vs M={m_data}")

        cases = _build_case_list(
            num_samples=b_data,
            num_freqs=m_data,
            freq_indices_text=args.freq_indices,
            max_samples=args.max_samples,
            max_cases=args.max_cases,
            seed=args.seed,
        )

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
            phi_t = build_phi(ftm["net_x"], ftm["net_y"], x_t, y_t)

        phi_np = phi_t.detach().cpu().numpy().astype(np.float32)
        phi_full_t = torch.from_numpy(phi_np).to(device)

        h5_meta = _load_h5_metadata_dict(f)
        data_scale = max(_load_data_scale(data_h5_path, f), 1e-12)

        l_val = float(h5_meta.get("L", args.helmholtz_L))
        c_val = float(h5_meta.get("c", args.helmholtz_c))
        pml_val = float(h5_meta.get("pml_width", args.helmholtz_pml_width))
        sigma_val = float(h5_meta.get("sigma_max", args.helmholtz_sigma_max))
        source_sigma = float(h5_meta.get("source_sigma", args.source_sigma))

        solver = HelmholtzSolver(
            N=h,
            L=l_val,
            c=c_val,
            pml_width=pml_val,
            sigma_max=sigma_val,
        )

        interior_idx_t: Optional[torch.Tensor] = None
        interior_mask_np: Optional[np.ndarray] = None
        if args.phys_interior_only:
            interior_idx_t = _build_interior_index(h, w, device=device)
            interior_mask_np = np.zeros(h * w, dtype=bool)
            interior_mask_np[interior_idx_t.detach().cpu().numpy().astype(np.int64)] = True

        op_cache: Dict[int, Any] = {}
        op_torch_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        source_cache: Dict[int, np.ndarray] = {}

        rows: List[Dict[str, Any]] = []
        vis_count = 0
        vis_dir = Path(args.vis_dir)
        if args.num_visualize > 0:
            vis_dir.mkdir(parents=True, exist_ok=True)

        for i_case, (b_idx, m_idx) in enumerate(cases, start=1):
            omega_val = float(omega[m_idx])

            if m_idx not in op_cache:
                op_cache[m_idx] = solver._build_matrix(omega_val)
                op_torch_cache[m_idx] = (
                    _csc_to_torch_sparse(op_cache[m_idx].real.astype(np.float32), device=device),
                    _csc_to_torch_sparse(op_cache[m_idx].imag.astype(np.float32), device=device),
                )

            if b_idx not in source_cache:
                source_cache[b_idx] = _load_source_field_for_sample(
                    f=f,
                    sample_idx=b_idx,
                    solver=solver,
                    source_sigma=source_sigma,
                )

            if has_data:
                u_case = data_ds[b_idx, m_idx].astype(np.float32)
                u_re_true = u_case[..., 0].reshape(-1)
                u_im_true = u_case[..., 1].reshape(-1)
            else:
                u_re_true = f["fields_real"][b_idx, m_idx].astype(np.float32).reshape(-1)
                u_im_true = f["fields_imag"][b_idx, m_idx].astype(np.float32).reshape(-1)

            src = source_cache[b_idx]
            f_re = src.real.astype(np.float32).reshape(-1)
            f_im = src.imag.astype(np.float32).reshape(-1)
            if args.phys_scale_source_by_data:
                f_re = f_re / data_scale
                f_im = f_im / data_scale

            core_re, *_ = np.linalg.lstsq(phi_np, u_re_true, rcond=args.lstsq_rcond)
            core_im, *_ = np.linalg.lstsq(phi_np, u_im_true, rcond=args.lstsq_rcond)

            core_re_t = torch.from_numpy(core_re.astype(np.float32)).view(1, -1).to(device)
            core_im_t = torch.from_numpy(core_im.astype(np.float32)).view(1, -1).to(device)

            f_re_t = torch.from_numpy(f_re).to(device)
            f_im_t = torch.from_numpy(f_im).to(device)
            if interior_idx_t is not None:
                f_re_den = f_re_t.index_select(0, interior_idx_t)
                f_im_den = f_im_t.index_select(0, interior_idx_t)
            else:
                f_re_den = f_re_t
                f_im_den = f_im_t

            residual_den = torch.mean(f_re_den * f_re_den + f_im_den * f_im_den)
            residual_den = torch.clamp(residual_den, min=1.0)

            a_re_t, a_im_t = op_torch_cache[m_idx]
            physics = {
                "phi_full": phi_full_t,
                "A_re": a_re_t,
                "A_im": a_im_t,
                "f_re": f_re_t,
                "f_im": f_im_t,
                "interior_idx": interior_idx_t,
                "residual_den": residual_den,
            }

            loss_func = float(_pde_residual_loss(core_re_t, core_im_t, physics, eps=args.eps).item())

            u_re_proj = (phi_np @ core_re.astype(np.float32)).astype(np.float32)
            u_im_proj = (phi_np @ core_im.astype(np.float32)).astype(np.float32)

            loss_direct_proj = _direct_residual_metric(
                a_mat=op_cache[m_idx],
                u_re_vec=u_re_proj,
                u_im_vec=u_im_proj,
                f_re_vec=f_re,
                f_im_vec=f_im,
                interior_mask=interior_mask_np,
                eps=args.eps,
            )

            loss_direct_true = _direct_residual_metric(
                a_mat=op_cache[m_idx],
                u_re_vec=u_re_true,
                u_im_vec=u_im_true,
                f_re_vec=f_re,
                f_im_vec=f_im,
                interior_mask=interior_mask_np,
                eps=args.eps,
            )

            abs_diff = abs(loss_func - loss_direct_proj)
            rel_diff = abs_diff / max(abs(loss_direct_proj), args.eps)

            row = {
                "sample_idx": int(b_idx),
                "freq_idx": int(m_idx),
                "omega": float(omega_val),
                "residual_func": float(loss_func),
                "residual_direct_proj": float(loss_direct_proj),
                "residual_direct_true": float(loss_direct_true),
                "abs_diff_func_vs_direct": float(abs_diff),
                "rel_diff_func_vs_direct": float(rel_diff),
            }

            if vis_count < args.num_visualize:
                vis_count += 1
                vis_path = vis_dir / f"case{vis_count:03d}_sample{b_idx:03d}_freq{m_idx:03d}.png"
                _plot_case_comparison(
                    out_path=vis_path,
                    u_re_true=u_re_true.reshape(h, w),
                    u_im_true=u_im_true.reshape(h, w),
                    u_re_proj=u_re_proj.reshape(h, w),
                    u_im_proj=u_im_proj.reshape(h, w),
                    sample_idx=int(b_idx),
                    freq_idx=int(m_idx),
                    omega_val=float(omega_val),
                    residual_func=float(loss_func),
                    residual_direct_proj=float(loss_direct_proj),
                    residual_direct_true=float(loss_direct_true),
                    rel_diff=float(rel_diff),
                    dpi=int(args.vis_dpi),
                )
                row["vis_path"] = str(vis_path)

            rows.append(row)

            if args.log_every > 0 and (i_case % args.log_every == 0 or i_case == len(cases)):
                print(
                    f"[{i_case:04d}/{len(cases)}] sample={b_idx} freq={m_idx} "
                    f"func={loss_func:.3e} direct_proj={loss_direct_proj:.3e} "
                    f"direct_true={loss_direct_true:.3e} rel_diff={rel_diff:.3e}"
                )

    if not rows:
        raise RuntimeError("No cases were evaluated")

    abs_diffs = np.array([r["abs_diff_func_vs_direct"] for r in rows], dtype=np.float64)
    rel_diffs = np.array([r["rel_diff_func_vs_direct"] for r in rows], dtype=np.float64)
    true_res = np.array([r["residual_direct_true"] for r in rows], dtype=np.float64)

    summary = {
        "data_h5": str(data_h5_path),
        "ftm_ckpt": str(ftm_ckpt_path),
        "num_cases": int(len(rows)),
        "num_visualized": int(vis_count),
        "vis_dir": str(vis_dir) if args.num_visualize > 0 else "",
        "max_abs_diff_func_vs_direct": float(np.max(abs_diffs)),
        "mean_abs_diff_func_vs_direct": float(np.mean(abs_diffs)),
        "max_rel_diff_func_vs_direct": float(np.max(rel_diffs)),
        "mean_rel_diff_func_vs_direct": float(np.mean(rel_diffs)),
        "mean_direct_true_residual": float(np.mean(true_res)),
        "median_direct_true_residual": float(np.median(true_res)),
        "phys_interior_only": bool(args.phys_interior_only),
        "phys_scale_source_by_data": bool(args.phys_scale_source_by_data),
    }

    out = {
        "summary": summary,
        "rows": rows,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)

    print("\nValidation finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved details: {out_path}")

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate physics residual implementation on generated data")

    parser.add_argument("--data_h5", type=str, default="helmholtz_dataset_42.h5")
    parser.add_argument("--ftm_ckpt", type=str, default="ckp/ftm_gpu_checkpoint.pt")
    parser.add_argument("--out_json", type=str, default="visual_data/physics_residual_validation.json")

    parser.add_argument("--freq_indices", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=2)
    parser.add_argument("--max_cases", type=int, default=8)
    parser.add_argument("--num_visualize", type=int, default=8)
    parser.add_argument("--vis_dir", type=str, default="visual_data/physics_residual_validation_cases")
    parser.add_argument("--vis_dpi", type=int, default=180)

    parser.add_argument("--phys_interior_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phys_scale_source_by_data", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--helmholtz_L", type=float, default=1.0)
    parser.add_argument("--helmholtz_c", type=float, default=1.0)
    parser.add_argument("--helmholtz_pml_width", type=float, default=0.12)
    parser.add_argument("--helmholtz_sigma_max", type=float, default=50.0)
    parser.add_argument("--source_sigma", type=float, default=0.025)

    parser.add_argument("--lstsq_rcond", type=float, default=None)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate(args)


if __name__ == "__main__":
    main()
