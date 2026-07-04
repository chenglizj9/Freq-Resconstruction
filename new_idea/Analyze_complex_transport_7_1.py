"""
Analyze_complex_transport_7_1.py
--------------------------------
Run section 7.1 interpretability validation for complex transport using solved
51-frequency cores and optional field-space comparison.

Implemented analyses
--------------------
A) Complex ratio statistics vs frequency gap.
B) Fixed-pair baseline comparison:
   identity / real-scalar / complex-scalar / diagonal-complex / full-linear.
C) Error-vs-gap curves (pair-averaged) for lightweight baselines.
D) Latent-space vs field-space diagonal complex transport comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load_core_npz(core_npz: Path) -> Dict[str, np.ndarray]:
    if not core_npz.exists():
        raise FileNotFoundError(f"core npz not found: {core_npz}")
    z = np.load(core_npz)
    needed = ["core_re", "core_im", "omega"]
    for k in needed:
        if k not in z:
            raise KeyError(f"Missing key in core npz: {k}")
    core_re = z["core_re"].astype(np.float32)
    core_im = z["core_im"].astype(np.float32)
    omega = z["omega"].astype(np.float32)
    if core_re.shape != core_im.shape:
        raise ValueError(f"core_re/core_im shape mismatch: {core_re.shape} vs {core_im.shape}")
    if core_re.ndim != 3:
        raise ValueError(f"Expected core shape (B,M,D), got {core_re.shape}")
    if omega.ndim != 1 or omega.shape[0] != core_re.shape[1]:
        raise ValueError(f"omega shape mismatch: omega={omega.shape}, M={core_re.shape[1]}")
    return {
        "core_re": core_re,
        "core_im": core_im,
        "omega": omega,
    }


def _load_complex_fields(data_h5: Path, b_need: int, m_need: int) -> np.ndarray:
    if not data_h5.exists():
        raise FileNotFoundError(f"data_h5 not found: {data_h5}")
    with h5py.File(data_h5, "r") as f:
        if "data" in f:
            ds = f["data"]
            if ds.ndim != 5 or ds.shape[-1] != 2:
                raise ValueError(f"Expected data shape (B,M,H,W,2), got {ds.shape}")
            b, m = ds.shape[0], ds.shape[1]
            if b < b_need or m < m_need:
                raise ValueError(f"data shape {ds.shape} smaller than needed B={b_need}, M={m_need}")
            data = ds[:b_need, :m_need].astype(np.float32)
            u = data[..., 0] + 1j * data[..., 1]
        elif "fields_real" in f and "fields_imag" in f:
            re = f["fields_real"]
            im = f["fields_imag"]
            if re.shape != im.shape or re.ndim != 4:
                raise ValueError(f"Invalid fields_real/fields_imag shape: {re.shape}, {im.shape}")
            b, m = re.shape[0], re.shape[1]
            if b < b_need or m < m_need:
                raise ValueError(f"fields shape {re.shape} smaller than needed B={b_need}, M={m_need}")
            u = re[:b_need, :m_need].astype(np.float32) + 1j * im[:b_need, :m_need].astype(np.float32)
        else:
            raise KeyError("HDF5 must contain 'data' or ('fields_real','fields_imag').")
    # (B,M,H,W) -> (B,M,P)
    return u.reshape(u.shape[0], u.shape[1], -1).astype(np.complex64)


def _split_indices(b: int, train_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.1 <= train_ratio <= 0.95):
        raise ValueError("train_ratio should be in [0.1, 0.95]")
    rng = np.random.default_rng(seed)
    ids = np.arange(b)
    rng.shuffle(ids)
    n_train = int(round(b * train_ratio))
    n_train = max(2, min(n_train, b - 2))
    tr = np.sort(ids[:n_train])
    te = np.sort(ids[n_train:])
    return tr, te


def _build_pairs(m: int) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for a in range(m - 1):
        for b in range(a + 1, m):
            out.append((a, b, b - a))
    return out


def _select_fixed_pairs_by_gap(m: int, gap_targets: Sequence[int]) -> List[Tuple[int, int, int]]:
    pairs: List[Tuple[int, int, int]] = []
    seen = set()
    for g in gap_targets:
        if g <= 0 or g >= m:
            continue
        a = (m - 1 - g) // 2
        b = a + g
        key = (a, b)
        if key not in seen:
            seen.add(key)
            pairs.append((a, b, g))
    return pairs


def _rel_err(pred: np.ndarray, gt: np.ndarray, eps: float) -> np.ndarray:
    # pred/gt: (N,D) complex
    num = np.linalg.norm(pred - gt, axis=1)
    den = np.linalg.norm(gt, axis=1)
    return (num / np.maximum(den, eps)).astype(np.float64)


def _fit_real_scalar(x: np.ndarray, y: np.ndarray, eps: float) -> float:
    # x,y: (N,D) complex
    num = float(np.real(np.sum(np.conj(x) * y)))
    den = float(np.sum(np.abs(x) ** 2))
    return num / max(den, eps)


def _fit_complex_scalar(x: np.ndarray, y: np.ndarray, eps: float) -> complex:
    num = np.sum(np.conj(x) * y)
    den = np.sum(np.abs(x) ** 2)
    return complex(num / max(float(np.real(den)), eps))


def _fit_diag_complex(x: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    # x,y: (N,D) complex -> alpha: (D,)
    num = np.sum(np.conj(x) * y, axis=0)
    den = np.sum(np.abs(x) ** 2, axis=0)
    return num / np.maximum(den, eps)


def _fit_full_linear(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    # x,y: (N,D) complex. Return W:(D,D), y ~= x @ W^T.
    # Equivalent matrix form with column samples:
    # X:(D,N), Y:(D,N), W = Y X^H (X X^H + lambda I)^-1
    x_col = x.T.astype(np.complex128, copy=False)
    y_col = y.T.astype(np.complex128, copy=False)
    d = x_col.shape[0]
    gram = x_col @ x_col.conj().T
    gram = gram + float(ridge) * np.eye(d, dtype=np.complex128)
    rhs = y_col @ x_col.conj().T
    w = rhs @ np.linalg.inv(gram)
    return w


def _circular_std(phase: np.ndarray) -> float:
    r = np.abs(np.mean(np.exp(1j * phase)))
    r = float(np.clip(r, 1e-12, 1.0))
    return float(np.sqrt(max(-2.0 * np.log(r), 0.0)))


def analyze_a_ratio_stats(
    g_core: np.ndarray,
    omega: np.ndarray,
    out_dir: Path,
    eps: float,
    hist_gaps: Sequence[int],
    hist_max_points: int,
    seed: int,
) -> Dict[str, float]:
    b, m, d = g_core.shape
    rows = []
    rng = np.random.default_rng(seed)

    for gap in range(1, m):
        x = g_core[:, : m - gap, :]
        y = g_core[:, gap:, :]
        ratio = y / (x + eps)
        amp = np.abs(ratio).reshape(-1)
        ph = np.angle(ratio).reshape(-1)

        amp_q10, amp_q50, amp_q90 = np.quantile(amp, [0.10, 0.50, 0.90])
        ph_std = _circular_std(ph)
        rows.append(
            {
                "gap": int(gap),
                "delta_omega_mean": float(np.mean(omega[gap:] - omega[:-gap])),
                "amp_q10": float(amp_q10),
                "amp_q50": float(amp_q50),
                "amp_q90": float(amp_q90),
                "phase_circ_std": float(ph_std),
                "num_points": int(amp.size),
            }
        )

    csv_path = out_dir / "A_ratio_gap_stats.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("gap,delta_omega_mean,amp_q10,amp_q50,amp_q90,phase_circ_std,num_points\n")
        for r in rows:
            f.write(
                f"{r['gap']},{r['delta_omega_mean']:.8g},{r['amp_q10']:.8g},"
                f"{r['amp_q50']:.8g},{r['amp_q90']:.8g},{r['phase_circ_std']:.8g},"
                f"{r['num_points']}\n"
            )

    gap_vals = np.array([r["gap"] for r in rows], dtype=np.int32)
    amp_q10 = np.array([r["amp_q10"] for r in rows], dtype=np.float64)
    amp_q50 = np.array([r["amp_q50"] for r in rows], dtype=np.float64)
    amp_q90 = np.array([r["amp_q90"] for r in rows], dtype=np.float64)
    ph_std = np.array([r["phase_circ_std"] for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    ax = axes[0]
    ax.fill_between(gap_vals, amp_q10, amp_q90, alpha=0.2, label="|R| q10-q90")
    ax.plot(gap_vals, amp_q50, lw=2.0, label="|R| median")
    ax.set_xlabel("frequency gap index")
    ax.set_ylabel("amplitude ratio stats")
    ax.set_title("A: Complex ratio amplitude vs gap")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(gap_vals, ph_std, lw=2.0, color="tab:orange")
    ax.set_xlabel("frequency gap index")
    ax.set_ylabel("circular std of phase")
    ax.set_title("A: Complex ratio phase spread vs gap")
    ax.grid(alpha=0.3)

    fig.savefig(out_dir / "A_ratio_stats_vs_gap.png", dpi=180)
    plt.close(fig)

    hist_gaps_valid = [g for g in hist_gaps if 1 <= g < m]
    if len(hist_gaps_valid) > 0:
        ncols = len(hist_gaps_valid)
        fig, axes = plt.subplots(2, ncols, figsize=(3.8 * ncols, 7.0), squeeze=False, constrained_layout=True)
        for i, gap in enumerate(hist_gaps_valid):
            x = g_core[:, : m - gap, :]
            y = g_core[:, gap:, :]
            ratio = y / (x + eps)
            amp = np.abs(ratio).reshape(-1)
            ph = np.angle(ratio).reshape(-1)

            if amp.size > hist_max_points > 0:
                sel = rng.choice(amp.size, size=hist_max_points, replace=False)
                amp = amp[sel]
                ph = ph[sel]

            a_lo, a_hi = np.quantile(amp, [0.01, 0.99])
            amp_clip = np.clip(amp, a_lo, a_hi)

            ax = axes[0, i]
            ax.hist(amp_clip, bins=80, density=True, color="tab:blue", alpha=0.85)
            ax.set_title(f"gap={gap} |R| hist")
            ax.grid(alpha=0.2)

            ax = axes[1, i]
            ax.hist(ph, bins=72, range=(-math.pi, math.pi), density=True, color="tab:green", alpha=0.85)
            ax.set_title(f"gap={gap} arg(R) hist")
            ax.grid(alpha=0.2)

        fig.savefig(out_dir / "A_ratio_hist_selected_gaps.png", dpi=180)
        plt.close(fig)

    return {
        "amp_median_gap1": float(amp_q50[0]) if len(amp_q50) > 0 else float("nan"),
        "amp_median_gap_max": float(amp_q50[-1]) if len(amp_q50) > 0 else float("nan"),
        "phase_std_gap1": float(ph_std[0]) if len(ph_std) > 0 else float("nan"),
        "phase_std_gap_max": float(ph_std[-1]) if len(ph_std) > 0 else float("nan"),
    }


def analyze_b_fixed_pairs(
    g_core: np.ndarray,
    omega: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    out_dir: Path,
    eps: float,
    full_linear_ridge: float,
    fixed_gap_targets: Sequence[int],
) -> Dict[str, float]:
    _, m, _ = g_core.shape
    fixed_pairs = _select_fixed_pairs_by_gap(m=m, gap_targets=fixed_gap_targets)
    if len(fixed_pairs) == 0:
        raise ValueError("No valid fixed pairs were selected for analysis B.")

    methods = ["identity", "real_scalar", "complex_scalar", "diag_complex", "full_linear"]
    rows = []

    for a, b, gap in fixed_pairs:
        x_tr = g_core[train_ids, a, :]
        y_tr = g_core[train_ids, b, :]
        x_te = g_core[test_ids, a, :]
        y_te = g_core[test_ids, b, :]

        pred_id = x_te
        c_real = _fit_real_scalar(x_tr, y_tr, eps=eps)
        pred_real = c_real * x_te
        c_cplx = _fit_complex_scalar(x_tr, y_tr, eps=eps)
        pred_cplx = c_cplx * x_te
        alpha = _fit_diag_complex(x_tr, y_tr, eps=eps)
        pred_diag = x_te * alpha[None, :]
        w_full = _fit_full_linear(x_tr, y_tr, ridge=full_linear_ridge)
        pred_full = (w_full @ x_te.T).T

        pred_map = {
            "identity": pred_id,
            "real_scalar": pred_real,
            "complex_scalar": pred_cplx,
            "diag_complex": pred_diag,
            "full_linear": pred_full,
        }

        for mth in methods:
            err = _rel_err(pred_map[mth], y_te, eps=eps)
            rows.append(
                {
                    "pair": f"{a}->{b}",
                    "freq_a": int(a),
                    "freq_b": int(b),
                    "omega_a": float(omega[a]),
                    "omega_b": float(omega[b]),
                    "gap": int(gap),
                    "method": mth,
                    "mean_rel_err": float(np.mean(err)),
                    "std_rel_err": float(np.std(err)),
                }
            )

    csv_path = out_dir / "B_fixed_pair_baselines.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("pair,freq_a,freq_b,omega_a,omega_b,gap,method,mean_rel_err,std_rel_err\n")
        for r in rows:
            f.write(
                f"{r['pair']},{r['freq_a']},{r['freq_b']},{r['omega_a']:.8g},{r['omega_b']:.8g},"
                f"{r['gap']},{r['method']},{r['mean_rel_err']:.8g},{r['std_rel_err']:.8g}\n"
            )

    # Plot grouped bars.
    pair_labels = [f"{a}->{b}\n(g={g})" for a, b, g in fixed_pairs]
    method_order = methods
    width = 0.16
    x0 = np.arange(len(fixed_pairs))

    fig, ax = plt.subplots(figsize=(12, 5.4), constrained_layout=True)
    for i, mth in enumerate(method_order):
        vals = []
        for a, b, _ in fixed_pairs:
            pair_name = f"{a}->{b}"
            v = [r["mean_rel_err"] for r in rows if r["pair"] == pair_name and r["method"] == mth]
            vals.append(v[0] if len(v) > 0 else np.nan)
        ax.bar(x0 + (i - 2) * width, vals, width=width, label=mth)

    ax.set_xticks(x0)
    ax.set_xticklabels(pair_labels)
    ax.set_ylabel("mean relative error")
    ax.set_title("B: Fixed-pair transport baseline comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=3)
    fig.savefig(out_dir / "B_fixed_pair_baselines.png", dpi=180)
    plt.close(fig)

    # Aggregate key values.
    diag_vals = [r["mean_rel_err"] for r in rows if r["method"] == "diag_complex"]
    id_vals = [r["mean_rel_err"] for r in rows if r["method"] == "identity"]
    full_vals = [r["mean_rel_err"] for r in rows if r["method"] == "full_linear"]
    return {
        "diag_mean_over_fixed_pairs": float(np.mean(diag_vals)),
        "identity_mean_over_fixed_pairs": float(np.mean(id_vals)),
        "full_linear_mean_over_fixed_pairs": float(np.mean(full_vals)),
    }


def analyze_c_error_vs_gap(
    g_core: np.ndarray,
    omega: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    out_dir: Path,
    eps: float,
    max_pairs: int,
    seed: int,
) -> Dict[str, float]:
    _, m, _ = g_core.shape
    pairs = _build_pairs(m)
    if max_pairs > 0 and max_pairs < len(pairs):
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(i)] for i in np.sort(sel)]

    methods = ["identity", "real_scalar", "complex_scalar", "diag_complex"]
    pair_rows = []

    for a, b, gap in pairs:
        x_tr = g_core[train_ids, a, :]
        y_tr = g_core[train_ids, b, :]
        x_te = g_core[test_ids, a, :]
        y_te = g_core[test_ids, b, :]

        c_real = _fit_real_scalar(x_tr, y_tr, eps=eps)
        c_cplx = _fit_complex_scalar(x_tr, y_tr, eps=eps)
        alpha = _fit_diag_complex(x_tr, y_tr, eps=eps)

        pred_map = {
            "identity": x_te,
            "real_scalar": c_real * x_te,
            "complex_scalar": c_cplx * x_te,
            "diag_complex": x_te * alpha[None, :],
        }

        for mth in methods:
            err = _rel_err(pred_map[mth], y_te, eps=eps)
            pair_rows.append(
                {
                    "freq_a": int(a),
                    "freq_b": int(b),
                    "gap": int(gap),
                    "omega_a": float(omega[a]),
                    "omega_b": float(omega[b]),
                    "method": mth,
                    "mean_rel_err": float(np.mean(err)),
                }
            )

    csv_pair = out_dir / "C_gap_error_pairs.csv"
    with open(csv_pair, "w", encoding="utf-8") as f:
        f.write("freq_a,freq_b,gap,omega_a,omega_b,method,mean_rel_err\n")
        for r in pair_rows:
            f.write(
                f"{r['freq_a']},{r['freq_b']},{r['gap']},{r['omega_a']:.8g},{r['omega_b']:.8g},"
                f"{r['method']},{r['mean_rel_err']:.8g}\n"
            )

    # Aggregate by gap and method.
    curve_rows = []
    for gap in sorted({r["gap"] for r in pair_rows}):
        for mth in methods:
            vals = [r["mean_rel_err"] for r in pair_rows if r["gap"] == gap and r["method"] == mth]
            if len(vals) == 0:
                continue
            curve_rows.append(
                {
                    "gap": int(gap),
                    "method": mth,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "count": int(len(vals)),
                }
            )

    csv_curve = out_dir / "C_gap_error_curve.csv"
    with open(csv_curve, "w", encoding="utf-8") as f:
        f.write("gap,method,mean,std,count\n")
        for r in curve_rows:
            f.write(f"{r['gap']},{r['method']},{r['mean']:.8g},{r['std']:.8g},{r['count']}\n")

    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    for mth in methods:
        rows_m = [r for r in curve_rows if r["method"] == mth]
        rows_m = sorted(rows_m, key=lambda x: x["gap"])
        x = [r["gap"] for r in rows_m]
        y = [r["mean"] for r in rows_m]
        ax.plot(x, y, lw=1.9, marker="o", ms=3, label=mth)
    ax.set_xlabel("frequency gap index")
    ax.set_ylabel("mean relative error")
    ax.set_title("C: Transport error vs frequency gap")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_yscale("log")
    fig.savefig(out_dir / "C_error_vs_gap.png", dpi=180)
    plt.close(fig)

    # Near/far summary for feasibility check.
    def _mean_for(method: str, cond) -> float:
        vals = [r["mean"] for r in curve_rows if r["method"] == method and cond(r["gap"])]
        return float(np.mean(vals)) if len(vals) > 0 else float("nan")

    near_cond = lambda g: g <= 5
    far_cond = lambda g: g >= 25
    return {
        "diag_near_gap_mean": _mean_for("diag_complex", near_cond),
        "identity_near_gap_mean": _mean_for("identity", near_cond),
        "diag_far_gap_mean": _mean_for("diag_complex", far_cond),
        "identity_far_gap_mean": _mean_for("identity", far_cond),
    }


def analyze_d_field_vs_latent(
    g_core: np.ndarray,
    u_field: np.ndarray,
    omega: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    out_dir: Path,
    eps: float,
    field_dim_subsample: int,
    seed: int,
) -> Dict[str, float]:
    b, m, _ = g_core.shape
    if u_field.shape[0] < b or u_field.shape[1] < m:
        raise ValueError(f"Field shape {u_field.shape} is incompatible with core shape {g_core.shape}")

    u = u_field[:b, :m, :]
    p = u.shape[2]

    # Optional dimension subsample to speed up field-space diagonal fitting.
    if field_dim_subsample > 0 and field_dim_subsample < p:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(p, size=field_dim_subsample, replace=False))
        u = u[:, :, idx]
        p = field_dim_subsample

    rows = []
    for gap in range(1, m):
        a = (m - 1 - gap) // 2
        b_idx = a + gap

        x_lat_tr = g_core[train_ids, a, :]
        y_lat_tr = g_core[train_ids, b_idx, :]
        x_lat_te = g_core[test_ids, a, :]
        y_lat_te = g_core[test_ids, b_idx, :]

        x_f_tr = u[train_ids, a, :]
        y_f_tr = u[train_ids, b_idx, :]
        x_f_te = u[test_ids, a, :]
        y_f_te = u[test_ids, b_idx, :]

        alpha_lat = _fit_diag_complex(x_lat_tr, y_lat_tr, eps=eps)
        alpha_f = _fit_diag_complex(x_f_tr, y_f_tr, eps=eps)

        pred_lat = x_lat_te * alpha_lat[None, :]
        pred_f = x_f_te * alpha_f[None, :]

        err_lat = float(np.mean(_rel_err(pred_lat, y_lat_te, eps=eps)))
        err_f = float(np.mean(_rel_err(pred_f, y_f_te, eps=eps)))

        rows.append(
            {
                "gap": int(gap),
                "freq_a": int(a),
                "freq_b": int(b_idx),
                "omega_a": float(omega[a]),
                "omega_b": float(omega[b_idx]),
                "latent_diag_err": err_lat,
                "field_diag_err": err_f,
            }
        )

    csv_path = out_dir / "D_field_vs_latent_diag.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("gap,freq_a,freq_b,omega_a,omega_b,latent_diag_err,field_diag_err\n")
        for r in rows:
            f.write(
                f"{r['gap']},{r['freq_a']},{r['freq_b']},{r['omega_a']:.8g},{r['omega_b']:.8g},"
                f"{r['latent_diag_err']:.8g},{r['field_diag_err']:.8g}\n"
            )

    x = np.array([r["gap"] for r in rows], dtype=np.int32)
    y_lat = np.array([r["latent_diag_err"] for r in rows], dtype=np.float64)
    y_f = np.array([r["field_diag_err"] for r in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    ax.plot(x, y_lat, lw=2.0, marker="o", ms=3, label="latent-space diag complex")
    ax.plot(x, y_f, lw=2.0, marker="o", ms=3, label="field-space diag complex")
    ax.set_xlabel("frequency gap index")
    ax.set_ylabel("mean relative error")
    ax.set_title("D: Field-space vs latent-space diagonal transport")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_yscale("log")
    fig.savefig(out_dir / "D_field_vs_latent_diag.png", dpi=180)
    plt.close(fig)

    improv = (y_f - y_lat) / np.maximum(y_f, eps)
    return {
        "latent_diag_mean": float(np.mean(y_lat)),
        "field_diag_mean": float(np.mean(y_f)),
        "relative_improvement_mean": float(np.mean(improv)),
        "relative_improvement_median": float(np.median(improv)),
        "field_dim_used": int(p),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Section 7.1 interpretability analysis for complex transport")
    p.add_argument(
        "--core_npz",
        type=str,
        default="new_idea/transport_predictor_experiment/solved_cores_all_samples.npz",
    )
    p.add_argument(
        "--data_h5",
        type=str,
        default="new_idea/helmholtz_dataset_42_new_idea_mask10.h5",
    )
    p.add_argument("--out_dir", type=str, default="new_idea/interpretability_7_1")

    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eps", type=float, default=1e-8)

    p.add_argument("--hist_gaps", type=str, default="1,5,10,20,30,40")
    p.add_argument("--hist_max_points", type=int, default=400000)

    p.add_argument("--fixed_gap_targets", type=str, default="1,5,10,20,40")
    p.add_argument("--full_linear_ridge", type=float, default=1e-3)

    p.add_argument(
        "--max_pairs_c",
        type=int,
        default=0,
        help="If >0, randomly sample this many frequency pairs for analysis C.",
    )
    p.add_argument(
        "--field_dim_subsample",
        type=int,
        default=0,
        help="If >0, randomly subsample field dimensions for analysis D.",
    )
    return p


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for chunk in text.split(","):
        c = chunk.strip()
        if c == "":
            continue
        out.append(int(c))
    return out


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    core_bundle = _load_core_npz(Path(args.core_npz))
    core_re = core_bundle["core_re"]
    core_im = core_bundle["core_im"]
    omega = core_bundle["omega"]

    g_core = (core_re + 1j * core_im).astype(np.complex64)
    b, m, d = g_core.shape

    train_ids, test_ids = _split_indices(b=b, train_ratio=args.train_ratio, seed=args.seed)
    hist_gaps = _parse_int_list(args.hist_gaps)
    fixed_gap_targets = _parse_int_list(args.fixed_gap_targets)

    # A/B/C run only on solved cores.
    a_summary = analyze_a_ratio_stats(
        g_core=g_core,
        omega=omega,
        out_dir=out_dir,
        eps=float(args.eps),
        hist_gaps=hist_gaps,
        hist_max_points=int(args.hist_max_points),
        seed=int(args.seed),
    )

    b_summary = analyze_b_fixed_pairs(
        g_core=g_core,
        omega=omega,
        train_ids=train_ids,
        test_ids=test_ids,
        out_dir=out_dir,
        eps=float(args.eps),
        full_linear_ridge=float(args.full_linear_ridge),
        fixed_gap_targets=fixed_gap_targets,
    )

    c_summary = analyze_c_error_vs_gap(
        g_core=g_core,
        omega=omega,
        train_ids=train_ids,
        test_ids=test_ids,
        out_dir=out_dir,
        eps=float(args.eps),
        max_pairs=int(args.max_pairs_c),
        seed=int(args.seed),
    )

    # D requires fields.
    u_field = _load_complex_fields(Path(args.data_h5), b_need=b, m_need=m)
    d_summary = analyze_d_field_vs_latent(
        g_core=g_core,
        u_field=u_field,
        omega=omega,
        train_ids=train_ids,
        test_ids=test_ids,
        out_dir=out_dir,
        eps=float(args.eps),
        field_dim_subsample=int(args.field_dim_subsample),
        seed=int(args.seed),
    )

    # Feasibility cues from A/B/C/D.
    diag_near = c_summary["diag_near_gap_mean"]
    id_near = c_summary["identity_near_gap_mean"]
    near_gain = (id_near - diag_near) / max(id_near, float(args.eps))

    diag_far = c_summary["diag_far_gap_mean"]
    id_far = c_summary["identity_far_gap_mean"]
    far_gain = (id_far - diag_far) / max(id_far, float(args.eps))

    summary = {
        "core_npz": str(args.core_npz),
        "data_h5": str(args.data_h5),
        "out_dir": str(out_dir),
        "num_samples": int(b),
        "num_freqs": int(m),
        "core_dim": int(d),
        "train_size": int(train_ids.size),
        "test_size": int(test_ids.size),
        "A_ratio_stats": a_summary,
        "B_fixed_pairs": b_summary,
        "C_gap_curve": c_summary,
        "D_field_vs_latent": d_summary,
        "feasibility_indicators": {
            "diag_vs_identity_gain_near_gap": float(near_gain),
            "diag_vs_identity_gain_far_gap": float(far_gain),
            "latent_vs_field_relative_improvement_mean": float(d_summary["relative_improvement_mean"]),
        },
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Finished section 7.1 interpretability analysis.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
