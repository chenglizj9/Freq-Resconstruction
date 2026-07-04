"""
test_FTM.py  (multi-channel edition)
-------------------------------------
Evaluate a trained FTM checkpoint on 2-channel (Helmholtz) or
4-channel (elastic wave: ux_re, ux_im, uy_re, uy_im) data.

Outputs
-------
1) Per-channel error curves (frequency RMSE)
2) Per-sample RMSE curve
3) Reconstruction vs ground-truth comparison figures
4) Metric CSV tables and summary JSON

Usage
-----
    # Helmholtz (C=2)
    python test_FTM.py --ckpt ckp/ftm.pt --data_h5 helmholtz.h5

    # Elastic wave (C=4)
    python test_FTM.py --ckpt ckp/ftm_elastic.pt --data_h5 elastic.h5
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

# Import network / utility from the (updated) training module
from train_FTM_GPU import MLP1D, build_phi, normalize_coords_to_unit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading  (supports both old C=2 format and new multi-channel)
# ─────────────────────────────────────────────────────────────────────────────

def _load_checkpoint(
    ckpt_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Returns a dict with:
      cfg            : training config dict
      nets_x / nets_y: list of MLP1D (one per basis group)
      ch_to_group    : list[int], mapping channel index → basis group
      cores          : list[C] of np.ndarray (B, M, Rx, Ry)
      channel_names  : list[str]
      omega_ckpt     : np.ndarray (M,)
      data_scale     : float
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt.get("config", {})

    for k in ("rank_x", "rank_y", "hidden_dim", "hidden_layers", "activation"):
        if k not in cfg:
            raise KeyError(f"Checkpoint config missing key: '{k}'")

    def _make_net(rank_out: int) -> MLP1D:
        net = MLP1D(
            out_dim=int(rank_out),
            hidden_dim=int(cfg["hidden_dim"]),
            num_hidden_layers=int(cfg["hidden_layers"]),
            activation=str(cfg["activation"]),
        ).to(device)
        net.eval()
        return net

    # ── Detect checkpoint format ───────────────────────────────────────────
    # New format: "cores" (list) + optional "net_x_state_list"
    # Old format: "cores_real"/"cores_imag" + "net_x_state"/"net_y_state"

    if "cores" in ckpt:
        # New multi-channel format
        cores_raw = ckpt["cores"]   # list of tensors
        cores = [_to_numpy(c).astype(np.float32) for c in cores_raw]
        channel_names = ckpt.get("channel_names", [f"ch{i}" for i in range(len(cores))])

        if "net_x_state_list" in ckpt:
            # split_basis: multiple basis pairs
            states_x = ckpt["net_x_state_list"]
            states_y = ckpt["net_y_state_list"]
            nets_x = [_make_net(cfg["rank_x"]) for _ in states_x]
            nets_y = [_make_net(cfg["rank_y"]) for _ in states_y]
            for net, sd in zip(nets_x, states_x):
                net.load_state_dict(sd)
            for net, sd in zip(nets_y, states_y):
                net.load_state_dict(sd)
            ch_to_group = ckpt.get("ch_to_group",
                                   [min(c // 2, len(nets_x) - 1) for c in range(len(cores))])
        else:
            # shared basis (single pair)
            net_x = _make_net(cfg["rank_x"])
            net_y = _make_net(cfg["rank_y"])
            net_x.load_state_dict(ckpt["net_x_state"])
            net_y.load_state_dict(ckpt["net_y_state"])
            nets_x = [net_x]
            nets_y = [net_y]
            ch_to_group = [0] * len(cores)

    else:
        # Old C=2 format (backward compatible)
        cores = [
            _to_numpy(ckpt["cores_real"]).astype(np.float32),
            _to_numpy(ckpt["cores_imag"]).astype(np.float32),
        ]
        channel_names = ["real", "imag"]
        net_x = _make_net(cfg["rank_x"])
        net_y = _make_net(cfg["rank_y"])
        net_x.load_state_dict(ckpt["net_x_state"])
        net_y.load_state_dict(ckpt["net_y_state"])
        nets_x = [net_x]
        nets_y = [net_y]
        ch_to_group = [0, 0]

    C = len(cores)
    shapes = [c.shape for c in cores]
    if len(set(shapes)) != 1:
        raise ValueError(f"All cores must have the same shape, got {shapes}")

    omega_ckpt = _to_numpy(ckpt.get("omega", np.array([], dtype=np.float32))).astype(np.float32)
    data_scale = float(ckpt.get("data_scale", 1.0))

    return {
        "cfg":           cfg,
        "nets_x":        nets_x,
        "nets_y":        nets_y,
        "ch_to_group":   ch_to_group,
        "cores":         cores,           # list[C] of (B, M, Rx, Ry)
        "channel_names": channel_names,
        "omega_ckpt":    omega_ckpt,
        "data_scale":    data_scale,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Decoding
# ─────────────────────────────────────────────────────────────────────────────

def _build_phis(
    nets_x: List[MLP1D],
    nets_y: List[MLP1D],
    x_t: torch.Tensor,
    y_t: torch.Tensor,
    device: torch.device,
) -> List[np.ndarray]:
    """Build one phi (P, R) per basis group."""
    phis = []
    with torch.no_grad():
        for nx, ny in zip(nets_x, nets_y):
            phi = build_phi(nx, ny, x_t, y_t)
            phis.append(phi.cpu().numpy().astype(np.float32))
    return phis


def _decode_channels(
    cores_list: List[np.ndarray],   # list[C] of (bsz, M, R)
    phi_list:   List[np.ndarray],   # list[n_groups] of (P, R)
    ch_to_group: List[int],
    H: int,
    W: int,
) -> np.ndarray:
    """
    Returns pred: (bsz, M, H, W, C)
    """
    bsz = cores_list[0].shape[0]
    M   = cores_list[0].shape[1]
    C   = len(cores_list)

    out = np.zeros((bsz, M, H, W, C), dtype=np.float32)
    for c, (core, grp) in enumerate(zip(cores_list, ch_to_group)):
        phi = phi_list[grp]   # (P, R)
        flat = np.einsum("bmr,pr->bmp", core, phi, optimize=True)  # (bsz, M, P)
        out[..., c] = flat.reshape(bsz, M, H, W)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Mask helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_mask_view(mask_ds: h5py.Dataset, B_eval: int) -> Tuple[Optional[np.ndarray], bool]:
    """
    Returns (shared_mask_or_None, per_sample_bool).
    shared_mask shape: (M, H*W*C) float32 {0,1}
    """
    if mask_ds.ndim == 4:          # (M, H, W, C) — shared
        arr = (mask_ds[...].astype(np.float32) > 0.5)
        M, H, W, C = arr.shape
        return arr.reshape(M, H * W * C), False
    if mask_ds.ndim == 5:          # (B, M, H, W, C) — per sample
        if mask_ds.shape[0] < B_eval:
            raise ValueError(
                f"mask_tr sample dim {mask_ds.shape[0]} < eval samples {B_eval}."
            )
        return None, True
    raise ValueError(f"mask_tr must have 4 or 5 dims, got {mask_ds.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _safe_rel(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """sqrt(num/den) where den>0, else nan."""
    out = np.full_like(num, np.nan, dtype=np.float64)
    ok  = den > 0
    out[ok] = np.sqrt(num[ok] / den[ok])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_error_curves(
    out_dir: Path,
    omega: np.ndarray,
    rmse_freq:       np.ndarray,   # (M,) or (M, C)
    rmse_freq_obs:   np.ndarray,
    rmse_freq_unobs: np.ndarray,
    rmse_sample:     np.ndarray,   # (B,)
    channel_names:   List[str],
) -> None:
    """
    Left panel : per-frequency RMSE (one curve per channel if C>1).
    Right panel: per-sample RMSE.
    """
    C = len(channel_names)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(C, 2)))

    if rmse_freq.ndim == 1:
        rmse_freq = rmse_freq[:, None]
        rmse_freq_obs   = rmse_freq_obs[:, None]
        rmse_freq_unobs = rmse_freq_unobs[:, None]

    for c, cname in enumerate(channel_names):
        col = colors[c]
        ax.plot(omega, rmse_freq[:, c],       lw=2.0,  color=col, label=f"{cname} full")
        if np.isfinite(rmse_freq_obs[:, c]).any():
            ax.plot(omega, rmse_freq_obs[:, c],   lw=1.5, color=col, ls="--",
                    label=f"{cname} obs")
        if np.isfinite(rmse_freq_unobs[:, c]).any():
            ax.plot(omega, rmse_freq_unobs[:, c], lw=1.5, color=col, ls=":",
                    label=f"{cname} unobs")

    ax.set_title("Relative RMSE vs Frequency (per channel)")
    ax.set_xlabel("ω")
    ax.set_ylabel("Relative RMSE")
    ax.legend(fontsize=8, ncol=max(1, C // 2))
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(np.arange(len(rmse_sample)), rmse_sample, lw=1.8)
    ax.set_title("Per-sample Mean Relative RMSE")
    ax.set_xlabel("sample index")
    ax.set_ylabel("Relative RMSE")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "error_curves.png", dpi=180)
    plt.close(fig)


def _plot_compare_2ch(
    out_file: Path,
    gt: np.ndarray,    # (H, W, 2)
    pred: np.ndarray,  # (H, W, 2)
    omega_value: float,
    sample_idx: int,
    freq_idx: int,
    ch_prefix: str = "",
) -> None:
    """Comparison grid for one complex field: re / im / amplitude × GT|Pred|Err."""
    gt_re, gt_im     = gt[..., 0],   gt[..., 1]
    pd_re, pd_im     = pred[..., 0], pred[..., 1]
    gt_amp           = np.sqrt(gt_re ** 2 + gt_im ** 2)
    pd_amp           = np.sqrt(pd_re ** 2 + pd_im ** 2)
    err_re, err_im   = np.abs(pd_re - gt_re), np.abs(pd_im - gt_im)
    err_amp          = np.abs(pd_amp - gt_amp)

    def _lim(a, b):
        return float(min(np.min(a), np.min(b))), float(max(np.max(a), np.max(b)))

    vmin_re, vmax_re   = _lim(gt_re,  pd_re)
    vmin_im, vmax_im   = _lim(gt_im,  pd_im)
    vmin_amp, vmax_amp = _lim(gt_amp, pd_amp)

    items = [
        (gt_re,  f"GT {ch_prefix}Real",        vmin_re,  vmax_re,  "viridis"),
        (pd_re,  f"Pred {ch_prefix}Real",       vmin_re,  vmax_re,  "viridis"),
        (err_re, f"Err {ch_prefix}Real",         None,     None,    "magma"),
        (gt_im,  f"GT {ch_prefix}Imag",         vmin_im,  vmax_im,  "viridis"),
        (pd_im,  f"Pred {ch_prefix}Imag",        vmin_im,  vmax_im,  "viridis"),
        (err_im, f"Err {ch_prefix}Imag",          None,     None,    "magma"),
        (gt_amp, f"GT {ch_prefix}Amplitude",    vmin_amp, vmax_amp, "viridis"),
        (pd_amp, f"Pred {ch_prefix}Amplitude",   vmin_amp, vmax_amp, "viridis"),
        (err_amp, f"Err {ch_prefix}Amplitude",    None,     None,    "magma"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, (img, title, vmin, vmax, cmap) in zip(axes.flat, items):
        kw = {"cmap": cmap, "origin": "lower"}
        if vmin is not None:
            kw.update({"vmin": vmin, "vmax": vmax})
        im = ax.imshow(img, **kw)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    title = f"sample={sample_idx}  freq_idx={freq_idx}  ω={omega_value:.4f}"
    if ch_prefix:
        title = f"[{ch_prefix.strip()}] " + title
    fig.suptitle(title, y=0.99)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def _plot_compare_elastic(
    out_file: Path,
    gt: np.ndarray,    # (H, W, 4)  [ux_re, ux_im, uy_re, uy_im]
    pred: np.ndarray,
    omega_value: float,
    sample_idx: int,
    freq_idx: int,
    channel_names: List[str],
) -> None:
    """
    For C=4 elastic data: show ux and uy side-by-side.
    Layout: 2 rows (ux / uy) × 3 cols (GT amp | Pred amp | Err amp),
    plus individual re/im panels below.
    """
    def _amp(arr, c_re, c_im):
        return np.sqrt(arr[..., c_re] ** 2 + arr[..., c_im] ** 2)

    def _lim(a, b):
        return float(min(np.min(a), np.min(b))), float(max(np.max(a), np.max(b)))

    # Determine which channel indices belong to ux and uy
    # Convention: first pair → ux, second pair → uy
    ux_re_i, ux_im_i = 0, 1
    uy_re_i, uy_im_i = 2, 3

    components = [
        ("ux", ux_re_i, ux_im_i),
        ("uy", uy_re_i, uy_im_i),
    ]

    # 6 rows × 3 cols: for each component: [GT_re, Pred_re, Err_re],
    #                                       [GT_im, Pred_im, Err_im],
    #                                       [GT_amp, Pred_amp, Err_amp]
    n_rows = len(components) * 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, n_rows * 3.5))

    row = 0
    for name, ri, ii in components:
        gt_re,  pd_re  = gt[..., ri],  pred[..., ri]
        gt_im,  pd_im  = gt[..., ii],  pred[..., ii]
        gt_amp          = np.sqrt(gt_re  ** 2 + gt_im  ** 2)
        pd_amp          = np.sqrt(pd_re  ** 2 + pd_im  ** 2)

        for (g, p, label, use_shared_scale) in [
            (gt_re,  pd_re,  f"{name} Real",      True),
            (gt_im,  pd_im,  f"{name} Imag",      True),
            (gt_amp, pd_amp, f"{name} Amplitude", True),
        ]:
            err = np.abs(p - g)
            if use_shared_scale:
                lo, hi = _lim(g, p)
            else:
                lo, hi = None, None

            for col_ax, (img, ttl, is_err) in enumerate(
                [(g, f"GT {label}", False),
                 (p, f"Pred {label}", False),
                 (err, f"Err {label}", True)]
            ):
                ax = axes[row, col_ax]
                kw = {"origin": "lower"}
                if is_err:
                    kw["cmap"] = "magma"
                else:
                    kw["cmap"] = "viridis"
                    if lo is not None:
                        kw.update({"vmin": lo, "vmax": hi})
                im = ax.imshow(img, **kw)
                ax.set_title(ttl, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            row += 1

    fig.suptitle(
        f"Elastic  sample={sample_idx}  freq_idx={freq_idx}  ω={omega_value:.4f}",
        y=1.001, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def _plot_compare(
    out_file: Path,
    gt: np.ndarray,    # (H, W, C)
    pred: np.ndarray,
    omega_value: float,
    sample_idx: int,
    freq_idx: int,
    channel_names: List[str],
) -> None:
    C = gt.shape[-1]
    if C == 2:
        _plot_compare_2ch(out_file, gt, pred, omega_value, sample_idx, freq_idx)
    elif C == 4:
        _plot_compare_elastic(out_file, gt, pred, omega_value, sample_idx, freq_idx, channel_names)
    else:
        # Generic: one sub-figure per channel
        fig, axes = plt.subplots(C, 3, figsize=(12, C * 4))
        for c, cname in enumerate(channel_names):
            g, p = gt[..., c], pred[..., c]
            err  = np.abs(p - g)
            lo   = float(min(np.min(g), np.min(p)))
            hi   = float(max(np.max(g), np.max(p)))
            for col_ax, (img, ttl, is_err) in enumerate([
                (g,  f"GT {cname}", False),
                (p,  f"Pred {cname}", False),
                (err, f"Err {cname}", True),
            ]):
                ax = axes[c, col_ax] if C > 1 else axes[col_ax]
                kw = {"origin": "lower", "cmap": "magma" if is_err else "viridis"}
                if not is_err:
                    kw.update({"vmin": lo, "vmax": hi})
                im = ax.imshow(img, **kw)
                ax.set_title(ttl, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"sample={sample_idx}  freq_idx={freq_idx}  ω={omega_value:.4f}")
        fig.tight_layout()
        fig.savefig(out_file, dpi=150)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation routine
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )

    # ── Checkpoint ────────────────────────────────────────────────────────
    info = _load_checkpoint(Path(args.ckpt), device)
    cfg           = info["cfg"]
    nets_x        = info["nets_x"]
    nets_y        = info["nets_y"]
    ch_to_group   = info["ch_to_group"]
    cores         = info["cores"]          # list[C] of (B_ckpt, M_ckpt, Rx, Ry)
    channel_names = info["channel_names"]
    data_scale_ckpt = info["data_scale"]

    C_ckpt = len(cores)
    B_ckpt, M_ckpt, Rx, Ry = cores[0].shape
    R = Rx * Ry

    # Flatten cores to (B, M, R) for decoding
    cores_flat = [c.reshape(B_ckpt, M_ckpt, R) for c in cores]

    # ── Open HDF5 ─────────────────────────────────────────────────────────
    with h5py.File(args.data_h5, "r") as f:
        for key in ("data", "mask_tr", "omega"):
            if key not in f:
                raise KeyError(f"HDF5 must contain '{key}'.")

        data_ds = f["data"]
        mask_ds = f["mask_tr"]
        omega   = f["omega"][...].astype(np.float32)

        B_data, M_data, H, W, C_data = data_ds.shape

        if C_data != C_ckpt:
            raise ValueError(
                f"Channel mismatch: data has C={C_data}, checkpoint has C={C_ckpt}. "
                f"Data channels: {channel_names}"
            )
        if M_data != M_ckpt:
            raise ValueError(
                f"Frequency length mismatch: data M={M_data}, checkpoint M={M_ckpt}."
            )

        B_eval = min(B_data, B_ckpt)
        if args.max_eval_samples > 0:
            B_eval = min(B_eval, args.max_eval_samples)
        if B_eval <= 0:
            raise ValueError("No samples to evaluate.")

        grid_x = (f["grid_x"][...].astype(np.float32) if "grid_x" in f
                   else np.linspace(0.0, 1.0, H, dtype=np.float32))
        grid_y = (f["grid_y"][...].astype(np.float32) if "grid_y" in f
                   else np.linspace(0.0, 1.0, W, dtype=np.float32))

        # ── Coordinates & basis ───────────────────────────────────────────
        use_norm = bool(cfg.get("normalize_coords", True))
        x_np = normalize_coords_to_unit(grid_x.astype(np.float64)).astype(np.float32) \
               if use_norm else grid_x
        y_np = normalize_coords_to_unit(grid_y.astype(np.float64)).astype(np.float32) \
               if use_norm else grid_y

        x_t = torch.from_numpy(x_np).unsqueeze(-1).to(device)
        y_t = torch.from_numpy(y_np).unsqueeze(-1).to(device)

        phi_list = _build_phis(nets_x, nets_y, x_t, y_t, device)

        # Verify rank consistency
        expected_R = phi_list[0].shape[1]
        if expected_R != R:
            raise ValueError(f"Rank mismatch: phi gives R={expected_R}, cores have R={R}.")

        # ── Mask ─────────────────────────────────────────────────────────
        shared_mask_flat, per_sample_mask = _build_mask_view(mask_ds, B_eval)
        # shared_mask_flat: (M, H*W*C) or None

        # ── Accumulate per-channel metrics ────────────────────────────────
        # Accumulators shape: (M, C)
        sum_err2_freq  = np.zeros((M_data, C_data), dtype=np.float64)
        sum_gt2_freq   = np.zeros((M_data, C_data), dtype=np.float64)
        sum_err2_obs   = np.zeros((M_data, C_data), dtype=np.float64)
        sum_gt2_obs    = np.zeros((M_data, C_data), dtype=np.float64)
        sum_err2_unobs = np.zeros((M_data, C_data), dtype=np.float64)
        sum_gt2_unobs  = np.zeros((M_data, C_data), dtype=np.float64)
        cnt_freq   = np.zeros((M_data, C_data), dtype=np.float64)
        cnt_obs    = np.zeros((M_data, C_data), dtype=np.float64)
        cnt_unobs  = np.zeros((M_data, C_data), dtype=np.float64)

        # Per-sample: mean relative RMSE over freq & channels
        rmse_sample  = np.zeros(B_eval, dtype=np.float64)
        # Per-case for worst-case selection: (B, M) averaged over C
        case_rel = np.zeros((B_eval, M_data), dtype=np.float64)

        data_scale = data_scale_ckpt if args.denormalize else 1.0

        for s in range(0, B_eval, args.batch_size):
            e   = min(B_eval, s + args.batch_size)
            bsz = e - s
            idx_slice = slice(s, e)

            # Predict
            cores_batch = [cf[idx_slice] for cf in cores_flat]   # list[C] of (bsz, M, R)
            pred = _decode_channels(cores_batch, phi_list, ch_to_group, H, W)
            # pred: (bsz, M, H, W, C)

            gt = data_ds[s:e].astype(np.float32)   # (bsz, M, H, W, C)

            if data_scale != 1.0:
                pred *= data_scale
                gt   *= data_scale

            # Flatten spatial dims: (bsz, M, P, C) where P = H*W
            err2 = (pred - gt) ** 2
            gt2  = gt ** 2
            err2_f = err2.reshape(bsz, M_data, H * W, C_data)
            gt2_f  = gt2.reshape(bsz, M_data, H * W, C_data)

            # Full-domain
            sum_err2_freq += np.sum(err2_f, axis=(0, 2))   # (M, C)
            sum_gt2_freq  += np.sum(gt2_f,  axis=(0, 2))
            cnt_freq      += bsz * H * W

            # Per-case relative error (M, C) → average over C for ranking
            case_err  = np.sum(err2_f, axis=2)   # (bsz, M, C)
            case_gt   = np.sum(gt2_f,  axis=2)
            case_rel_c = np.sqrt(case_err / np.maximum(case_gt, 1e-30))   # (bsz, M, C)
            case_rel[s:e] = case_rel_c.mean(axis=2)                       # (bsz, M)
            rmse_sample[s:e] = case_rel[s:e].mean(axis=1)                 # (bsz,)

            # Mask-split
            if shared_mask_flat is not None:
                # shared_mask_flat: (M, H*W*C) → reshape to (M, H*W, C)
                obs   = shared_mask_flat.reshape(M_data, H * W, C_data)   # {0,1}
                unobs = 1.0 - obs

                sum_err2_obs   += np.sum(err2_f * obs[None],   axis=(0, 2))
                sum_gt2_obs    += np.sum(gt2_f  * obs[None],   axis=(0, 2))
                sum_err2_unobs += np.sum(err2_f * unobs[None], axis=(0, 2))
                sum_gt2_unobs  += np.sum(gt2_f  * unobs[None], axis=(0, 2))
                cnt_obs        += bsz * np.sum(obs,   axis=1)
                cnt_unobs      += bsz * np.sum(unobs, axis=1)
            else:
                mask_b = (mask_ds[s:e].astype(np.float32) > 0.5)  # (bsz, M, H, W, C)
                obs    = mask_b.reshape(bsz, M_data, H * W, C_data).astype(np.float32)
                unobs  = 1.0 - obs

                sum_err2_obs   += np.sum(err2_f * obs,   axis=(0, 2))
                sum_gt2_obs    += np.sum(gt2_f  * obs,   axis=(0, 2))
                sum_err2_unobs += np.sum(err2_f * unobs, axis=(0, 2))
                sum_gt2_unobs  += np.sum(gt2_f  * unobs, axis=(0, 2))
                cnt_obs        += np.sum(obs,   axis=(0, 2))
                cnt_unobs      += np.sum(unobs, axis=(0, 2))

            print(f"  Eval {e}/{B_eval} samples …")

        # ── Compute RMSE arrays ──────────────────────────────────────────
        # shape: (M, C)
        rmse_freq       = _safe_rel(sum_err2_freq,  sum_gt2_freq)
        rmse_freq_obs   = _safe_rel(sum_err2_obs,   sum_gt2_obs)
        rmse_freq_unobs = _safe_rel(sum_err2_unobs, sum_gt2_unobs)

        # ── Plots ────────────────────────────────────────────────────────
        _plot_error_curves(
            out_dir, omega,
            rmse_freq, rmse_freq_obs, rmse_freq_unobs,
            rmse_sample, channel_names,
        )

        # ── CSV tables ───────────────────────────────────────────────────
        # Per-frequency: one row per omega, columns: omega, ch0_full, ch0_obs, …
        col_headers = ["omega"] + [
            f"{cn}_{suffix}"
            for cn in channel_names
            for suffix in ("full", "obs", "unobs")
        ]
        freq_cols = [omega]
        for c in range(C_data):
            freq_cols += [rmse_freq[:, c], rmse_freq_obs[:, c], rmse_freq_unobs[:, c]]
        freq_table = np.column_stack(freq_cols)
        np.savetxt(
            out_dir / "metrics_per_frequency.csv",
            freq_table,
            delimiter=",",
            header=",".join(col_headers),
            comments="",
        )

        sample_table = np.column_stack([np.arange(B_eval), rmse_sample])
        np.savetxt(
            out_dir / "metrics_per_sample.csv",
            sample_table,
            delimiter=",",
            header="sample_idx,rmse_mean",
            comments="",
        )

        # ── Visualise worst cases ─────────────────────────────────────────
        vis_cases: List[Tuple[int, int]] = []
        if args.sample_idx >= 0 and args.freq_idx >= 0:
            if not (0 <= args.sample_idx < B_eval):
                raise ValueError(f"sample_idx out of range [0,{B_eval-1}].")
            if not (0 <= args.freq_idx < M_data):
                raise ValueError(f"freq_idx out of range [0,{M_data-1}].")
            vis_cases.append((args.sample_idx, args.freq_idx))

        flat_case = case_rel.reshape(-1)
        for idx in np.argsort(flat_case)[::-1]:
            # si = int(idx // M_data)
            # fi = int(idx %  M_data)
            fi = int(idx // B_eval)   
            si = int(idx % B_eval)
            pair = (si, fi)
            if pair not in vis_cases:
                vis_cases.append(pair)
            if len(vis_cases) >= args.num_visualize:
                break

        for rank, (si, fi) in enumerate(vis_cases, start=1):
            gt_frame = data_ds[si, fi].astype(np.float32)   # (H, W, C)
            if data_scale != 1.0:
                gt_frame = gt_frame * data_scale

            cores_one = [cf[si:si+1, fi:fi+1, :] for cf in cores_flat]  # list[C] (1,1,R)
            pred_frame = _decode_channels(
                cores_one, phi_list, ch_to_group, H, W
            )[0, 0]   # (H, W, C)
            if data_scale != 1.0:
                pred_frame = pred_frame * data_scale

            out_file = out_dir / f"compare_rank{rank:02d}_s{si:03d}_f{fi:03d}.png"
            _plot_compare(out_file, gt_frame, pred_frame, float(omega[fi]),
                          si, fi, channel_names)

    # ── Summary ───────────────────────────────────────────────────────────
    global_rmse = float(np.mean(rmse_sample))
    rmse_per_ch = {
        cname: {
            "freq_mean": float(np.nanmean(rmse_freq[:, c])),
            "freq_std":  float(np.nanstd(rmse_freq[:, c])),
            "obs_mean":  float(np.nanmean(rmse_freq_obs[:, c])),
            "unobs_mean": float(np.nanmean(rmse_freq_unobs[:, c])),
        }
        for c, cname in enumerate(channel_names)
    }

    summary = {
        "checkpoint":         str(args.ckpt),
        "data_h5":            str(args.data_h5),
        "evaluated_samples":  int(B_eval),
        "channel_names":      channel_names,
        "global_rmse":        global_rmse,
        "rmse_per_channel":   rmse_per_ch,
        "data_scale_used":    float(data_scale),
        "output_dir":         str(out_dir),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print("\n" + "─" * 60)
    print("Evaluation finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Figures & metrics saved to: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate FTM checkpoint (Helmholtz C=2 or Elastic C=4)"
    )
    p.add_argument("--ckpt",              type=str, default="ckp/ftm_gpu_checkpoint.pt")
    p.add_argument("--data_h5",           type=str, default="elastic_dataset.h5")
    p.add_argument("--out_dir",           type=str, default="visual_data/ftm_eval")

    p.add_argument("--batch_size",        type=int, default=32)
    p.add_argument("--max_eval_samples",  type=int, default=3)
    p.add_argument("--num_visualize",     type=int, default=16,
                   help="Number of worst-case comparison plots to generate.")

    p.add_argument("--sample_idx",        type=int, default=-1,
                   help="Specific sample to always visualise (ignored if <0).")
    p.add_argument("--freq_idx",          type=int, default=1,
                   help="Specific frequency index to always visualise (ignored if <0).")

    p.add_argument("--denormalize",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Multiply predictions/GT by data_scale before computing metrics.")
    p.add_argument("--device",            type=str, default="auto")
    return p


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()