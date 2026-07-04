"""
evaluate_diffusion.py
---------------------
Evaluate diffusion model quality without DPS guidance.

For each target frequency, this script:
1) Samples core tensors from conditional diffusion p(G | omega).
2) Computes core-FID against real cores at the same frequency.
3) Decodes generated cores to fields through FTM basis and computes field-FID
   against real fields at the same frequency.

Notes
-----
- FID is computed in a projected feature space (random projection by default)
  for numerical stability and tractability.
- This evaluation is prior-only (no posterior guidance / DPS).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib
import numpy as np
import torch

from train_diffusion import ConditionalUNet2D, build_cosine_schedule, build_linear_schedule
from train_FTM_GPU import MLP1D, build_phi, normalize_coords_to_unit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def normalize_omega(omega: float, omega_min: float, omega_max: float) -> float:
    den = max(float(omega_max - omega_min), 1e-12)
    return float((omega - omega_min) / den)


def parse_freq_indices(text: str, num_freqs: int) -> List[int]:
    if text.strip() == "":
        return list(range(num_freqs))

    out: List[int] = []
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


def build_timestep_sequence(total_steps: int, sample_steps: int) -> Tuple[List[int], str]:
    if sample_steps <= 0 or sample_steps >= total_steps:
        return list(range(total_steps - 1, -1, -1)), "ddpm"

    seq = np.linspace(total_steps - 1, 0, sample_steps, dtype=np.int64).tolist()
    out: List[int] = []
    seen = set()
    for t in seq:
        if t not in seen:
            out.append(int(t))
            seen.add(int(t))
    return out, "ddim"


def build_random_projection(in_dim: int, out_dim: int, seed: int) -> Optional[np.ndarray]:
    if out_dim <= 0 or out_dim >= in_dim:
        return None
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    mat /= math.sqrt(float(out_dim))
    return mat


def project_features(x: np.ndarray, proj: Optional[np.ndarray]) -> np.ndarray:
    # x: (N, D)
    if proj is None:
        return x.astype(np.float64, copy=False)
    return (x.astype(np.float32, copy=False) @ proj).astype(np.float64, copy=False)


def compute_stats(features: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    # features: (N, D)
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")

    n, d = features.shape
    if n <= 0:
        raise ValueError("features has zero rows")

    mu = np.mean(features, axis=0)
    if n == 1:
        cov = np.eye(d, dtype=np.float64) * eps
    else:
        xc = features - mu[None, :]
        cov = (xc.T @ xc) / float(n - 1)
        cov = 0.5 * (cov + cov.T)
        cov += np.eye(d, dtype=np.float64) * eps

    return mu, cov


def sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    mat = 0.5 * (mat + mat.T)
    evals, evecs = np.linalg.eigh(mat)
    evals = np.clip(evals, a_min=0.0, a_max=None)
    return (evecs * np.sqrt(evals)[None, :]) @ evecs.T


def frechet_distance(mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray) -> float:
    diff = mu1 - mu2
    cov1_sqrt = sqrtm_psd(cov1)
    mid = cov1_sqrt @ cov2 @ cov1_sqrt
    mid_sqrt = sqrtm_psd(mid)

    fid = float(diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2.0 * np.trace(mid_sqrt))
    return max(fid, 0.0)


def fid_from_features(real_feat: np.ndarray, fake_feat: np.ndarray) -> float:
    mu_r, cov_r = compute_stats(real_feat)
    mu_f, cov_f = compute_stats(fake_feat)
    return frechet_distance(mu_r, cov_r, mu_f, cov_f)


def load_diffusion_bundle(diff_ckpt: Path, device: torch.device) -> Dict[str, Any]:
    if not diff_ckpt.exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {diff_ckpt}")

    ckpt = safe_torch_load(diff_ckpt)
    if "model_config" not in ckpt or "diffusion_config" not in ckpt:
        raise KeyError("Diffusion checkpoint missing model_config or diffusion_config")

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
        schedule = build_cosine_schedule(num_steps=int(diff_cfg["num_steps"]), device=device)
    else:
        raise ValueError(f"Unsupported diffusion schedule: {schedule_name}")

    core_stats = ckpt["core_stats"]
    mean = to_numpy(core_stats["mean"]).astype(np.float32)
    std = to_numpy(core_stats["std"]).astype(np.float32)
    rx = int(core_stats["rx"])
    ry = int(core_stats["ry"])

    # Accept possible shapes: (2,1,1), (2,Rx,Ry), (1,2,1,1), (1,2,Rx,Ry)
    if mean.ndim == 4 and mean.shape[0] == 1:
        mean = mean[0]
    if std.ndim == 4 and std.shape[0] == 1:
        std = std[0]
    if mean.ndim != 3 or std.ndim != 3:
        raise ValueError(f"Invalid mean/std shape: mean={mean.shape}, std={std.shape}")
    if mean.shape[0] != 2 or std.shape[0] != 2:
        raise ValueError(f"mean/std channel mismatch: mean={mean.shape}, std={std.shape}")

    omega_stats = ckpt["omega_stats"]
    omega_min = float(omega_stats["min"])
    omega_max = float(omega_stats["max"])

    return {
        "model": model,
        "schedule": schedule,
        "num_steps": int(diff_cfg["num_steps"]),
        "schedule_name": schedule_name,
        "mean": mean,
        "std": np.maximum(std, 1e-6).astype(np.float32),
        "rx": rx,
        "ry": ry,
        "omega_min": omega_min,
        "omega_max": omega_max,
        "raw_ckpt": ckpt,
    }


def load_real_cores(ftm_ckpt: Path) -> Dict[str, Any]:
    if not ftm_ckpt.exists():
        raise FileNotFoundError(f"FTM checkpoint not found: {ftm_ckpt}")

    ckpt = safe_torch_load(ftm_ckpt)
    if "cores_real" not in ckpt or "cores_imag" not in ckpt:
        raise KeyError("FTM checkpoint must contain cores_real and cores_imag")

    cores_real = to_numpy(ckpt["cores_real"]).astype(np.float32)
    cores_imag = to_numpy(ckpt["cores_imag"]).astype(np.float32)
    if cores_real.shape != cores_imag.shape:
        raise ValueError(f"core shape mismatch: {cores_real.shape} vs {cores_imag.shape}")

    if cores_real.ndim == 3:
        cfg = ckpt.get("config", {})
        if "rank_x" not in cfg or "rank_y" not in cfg:
            raise ValueError("Flattened cores require rank_x/rank_y in config")
        rx = int(cfg["rank_x"])
        ry = int(cfg["rank_y"])
        if cores_real.shape[-1] != rx * ry:
            raise ValueError("Flattened core length mismatch with rank_x*rank_y")
        cores_real = cores_real.reshape(cores_real.shape[0], cores_real.shape[1], rx, ry)
        cores_imag = cores_imag.reshape(cores_imag.shape[0], cores_imag.shape[1], rx, ry)
    elif cores_real.ndim != 4:
        raise ValueError(f"Unsupported core shape: {cores_real.shape}")

    omega = to_numpy(ckpt.get("omega", np.array([], dtype=np.float32))).astype(np.float32)
    return {
        "cores_real": cores_real,
        "cores_imag": cores_imag,
        "omega": omega,
    }


def load_phi_from_ftm(
    ftm_ckpt: Path,
    data_h5: Path,
    device: torch.device,
) -> Dict[str, Any]:
    ckpt = safe_torch_load(ftm_ckpt)
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

    with h5py.File(data_h5, "r") as f:
        h = int(f["data"].shape[2])
        w = int(f["data"].shape[3])
        grid_x = f["grid_x"][...].astype(np.float32) if "grid_x" in f else np.linspace(0, 1, h, dtype=np.float32)
        grid_y = f["grid_y"][...].astype(np.float32) if "grid_y" in f else np.linspace(0, 1, w, dtype=np.float32)

    use_norm_coords = bool(cfg.get("normalize_coords", True))
    if use_norm_coords:
        x_coords = normalize_coords_to_unit(grid_x.astype(np.float64)).astype(np.float32)
        y_coords = normalize_coords_to_unit(grid_y.astype(np.float64)).astype(np.float32)
    else:
        x_coords = grid_x
        y_coords = grid_y

    x_t = torch.from_numpy(x_coords).unsqueeze(-1).to(device)
    y_t = torch.from_numpy(y_coords).unsqueeze(-1).to(device)
    with torch.no_grad():
        phi = build_phi(net_x, net_y, x_t, y_t)
    phi_np = phi.detach().cpu().numpy().astype(np.float32)

    return {
        "phi": phi_np,
        "h": h,
        "w": w,
    }


def sample_cores_for_frequency(
    model: ConditionalUNet2D,
    schedule: Any,
    omega_norm: float,
    rx: int,
    ry: int,
    num_samples: int,
    batch_size: int,
    timestep_seq: Sequence[int],
    sample_mode: str,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    # Returns normalized cores: (N,2,Rx,Ry)
    out: List[np.ndarray] = []

    step_count = len(timestep_seq)
    if step_count <= 0:
        raise ValueError("Empty timestep sequence")

    for start in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - start)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + start)

        x = torch.randn((bs, 2, rx, ry), generator=gen, device=device, dtype=torch.float32)
        omega_cond = torch.full((bs, 1), float(omega_norm), device=device, dtype=torch.float32)

        with torch.no_grad():
            for i, t_idx in enumerate(timestep_seq):
                t = torch.full((bs,), int(t_idx), device=device, dtype=torch.long)
                eps_pred = model(x, t, omega_cond)

                abar_t = schedule.alpha_bars[t_idx]

                if sample_mode == "ddpm":
                    alpha_t = schedule.alphas[t_idx]
                    beta_t = schedule.betas[t_idx]

                    mean = (
                        x - (beta_t / torch.sqrt(torch.clamp(1.0 - abar_t, min=1e-12))) * eps_pred
                    ) / torch.sqrt(torch.clamp(alpha_t, min=1e-12))

                    if t_idx > 0:
                        abar_prev = schedule.alpha_bars[t_idx - 1]
                        beta_tilde = beta_t * (1.0 - abar_prev) / torch.clamp(1.0 - abar_t, min=1e-12)
                        noise = torch.randn((bs, 2, rx, ry), generator=gen, device=device, dtype=torch.float32)
                        x = mean + torch.sqrt(torch.clamp(beta_tilde, min=1e-20)) * noise
                    else:
                        x = mean
                else:
                    x0_hat = (
                        x - torch.sqrt(torch.clamp(1.0 - abar_t, min=1e-12)) * eps_pred
                    ) / torch.sqrt(torch.clamp(abar_t, min=1e-12))

                    if i == step_count - 1:
                        x = x0_hat
                    else:
                        t_prev = int(timestep_seq[i + 1])
                        abar_prev = schedule.alpha_bars[t_prev]
                        x = (
                            torch.sqrt(torch.clamp(abar_prev, min=1e-12)) * x0_hat
                            + torch.sqrt(torch.clamp(1.0 - abar_prev, min=1e-12)) * eps_pred
                        )

        out.append(x.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(out, axis=0)


def decode_fields_from_cores(
    cores: np.ndarray,  # (N,2,Rx,Ry)
    phi: np.ndarray,    # (P,R)
    h: int,
    w: int,
    batch_size: int,
) -> np.ndarray:
    n = cores.shape[0]
    out = np.zeros((n, h, w, 2), dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        c = cores[s:e]
        re = c[:, 0].reshape(e - s, -1)
        im = c[:, 1].reshape(e - s, -1)
        pred_re = np.einsum("nr,pr->np", re, phi, optimize=True)
        pred_im = np.einsum("nr,pr->np", im, phi, optimize=True)
        out[s:e, ..., 0] = pred_re.reshape(e - s, h, w)
        out[s:e, ..., 1] = pred_im.reshape(e - s, h, w)
    return out


def plot_fid_curves(
    out_file: Path,
    omega: np.ndarray,
    core_fid: np.ndarray,
    field_fid: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].plot(omega, core_fid, marker="o", lw=1.8)
    axes[0].set_title("Core FID vs Frequency")
    axes[0].set_xlabel("omega")
    axes[0].set_ylabel("FID")
    axes[0].grid(alpha=0.3)

    axes[1].plot(omega, field_fid, marker="o", lw=1.8)
    axes[1].set_title("Field FID vs Frequency")
    axes[1].set_xlabel("omega")
    axes[1].set_ylabel("FID")
    axes[1].grid(alpha=0.3)

    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diff = load_diffusion_bundle(Path(args.diff_ckpt), device=device)
    real_core_data = load_real_cores(Path(args.ftm_ckpt))
    phi_bundle = load_phi_from_ftm(Path(args.ftm_ckpt), Path(args.data_h5), device=device)

    model = diff["model"]
    schedule = diff["schedule"]
    rx = int(diff["rx"])
    ry = int(diff["ry"])
    mean = diff["mean"]
    std = diff["std"]

    cores_real = real_core_data["cores_real"]
    cores_imag = real_core_data["cores_imag"]
    b_core, m_core = cores_real.shape[0], cores_real.shape[1]

    with h5py.File(args.data_h5, "r") as f:
        if "data" not in f or "omega" not in f:
            raise KeyError("HDF5 must contain data and omega")

        data_ds = f["data"]
        omega_data = f["omega"][...].astype(np.float32)
        b_data, m_data, h, w, c = data_ds.shape
        if c != 2:
            raise ValueError(f"Expected data channel=2, got shape {data_ds.shape}")

        b_eval = min(b_core, b_data)
        if args.max_samples > 0:
            b_eval = min(b_eval, args.max_samples)
        if b_eval < 2:
            raise ValueError("Need at least 2 samples for FID evaluation")

        if args.random_subset and b_eval < min(b_core, b_data):
            rng = np.random.default_rng(args.seed)
            sample_ids = np.sort(rng.choice(min(b_core, b_data), size=b_eval, replace=False)).tolist()
        else:
            sample_ids = list(range(b_eval))

        m_eval = min(m_core, m_data)
        freq_ids = [fi for fi in parse_freq_indices(args.freq_indices, m_eval)]
        if args.max_freqs > 0:
            freq_ids = freq_ids[: args.max_freqs]
        if len(freq_ids) == 0:
            raise ValueError("No frequency index selected")

        timestep_seq, sample_mode = build_timestep_sequence(diff["num_steps"], args.sample_steps)

        core_dim_in = 2 * rx * ry
        field_dim_in = 2 * h * w

        core_proj = build_random_projection(core_dim_in, args.core_fid_dim, args.seed + 101)
        field_proj = build_random_projection(field_dim_in, args.field_fid_dim, args.seed + 202)

        core_dim_eff = core_dim_in if core_proj is None else core_proj.shape[1]
        field_dim_eff = field_dim_in if field_proj is None else field_proj.shape[1]

        rows: List[Dict[str, Any]] = []

        for i, f_idx in enumerate(freq_ids, start=1):
            omega_val = float(omega_data[f_idx])
            omega_norm = normalize_omega(omega_val, diff["omega_min"], diff["omega_max"])

            n_real = len(sample_ids)
            n_fake = n_real if args.num_gen_per_freq <= 0 else int(args.num_gen_per_freq)
            n_fake = max(n_fake, 2)

            real_core = np.stack(
                [
                    cores_real[sample_ids, f_idx],
                    cores_imag[sample_ids, f_idx],
                ],
                axis=1,
            ).astype(np.float32)

            fake_core_norm = sample_cores_for_frequency(
                model=model,
                schedule=schedule,
                omega_norm=omega_norm,
                rx=rx,
                ry=ry,
                num_samples=n_fake,
                batch_size=args.sample_batch_size,
                timestep_seq=timestep_seq,
                sample_mode=sample_mode,
                seed=args.seed + 1000 * f_idx,
                device=device,
            )

            fake_core = (fake_core_norm * std[None, ...] + mean[None, ...]).astype(np.float32)

            real_core_feat = project_features(real_core.reshape(n_real, -1), core_proj)
            fake_core_feat = project_features(fake_core.reshape(n_fake, -1), core_proj)
            core_fid = fid_from_features(real_core_feat, fake_core_feat)

            real_field = data_ds[sample_ids, f_idx].astype(np.float32)
            fake_field = decode_fields_from_cores(
                cores=fake_core,
                phi=phi_bundle["phi"],
                h=phi_bundle["h"],
                w=phi_bundle["w"],
                batch_size=args.decode_batch_size,
            )

            real_field_feat = project_features(real_field.reshape(n_real, -1), field_proj)
            fake_field_feat = project_features(fake_field.reshape(n_fake, -1), field_proj)
            field_fid = fid_from_features(real_field_feat, fake_field_feat)

            rows.append(
                {
                    "freq_idx": int(f_idx),
                    "omega": float(omega_val),
                    "num_real": int(n_real),
                    "num_fake": int(n_fake),
                    "core_fid": float(core_fid),
                    "field_fid": float(field_fid),
                }
            )

            if args.log_every > 0 and (i % args.log_every == 0 or i == len(freq_ids)):
                print(
                    f"[{i:03d}/{len(freq_ids)}] freq_idx={f_idx:03d} omega={omega_val:.4g} "
                    f"core_fid={core_fid:.6e} field_fid={field_fid:.6e}"
                )

    rows = sorted(rows, key=lambda x: x["freq_idx"])
    csv_path = out_dir / "metrics_per_frequency.csv"
    with open(csv_path, "w", encoding="utf-8") as fp:
        fp.write("freq_idx,omega,num_real,num_fake,core_fid,field_fid\n")
        for r in rows:
            fp.write(
                f"{r['freq_idx']},{r['omega']:.8g},{r['num_real']},{r['num_fake']},"
                f"{r['core_fid']:.8g},{r['field_fid']:.8g}\n"
            )

    omega_curve = np.array([r["omega"] for r in rows], dtype=np.float32)
    core_curve = np.array([r["core_fid"] for r in rows], dtype=np.float32)
    field_curve = np.array([r["field_fid"] for r in rows], dtype=np.float32)
    plot_fid_curves(out_dir / "fid_curves.png", omega=omega_curve, core_fid=core_curve, field_fid=field_curve)

    summary = {
        "diff_ckpt": str(args.diff_ckpt),
        "ftm_ckpt": str(args.ftm_ckpt),
        "data_h5": str(args.data_h5),
        "device": str(device),
        "sampling_mode": sample_mode,
        "diffusion_steps": int(diff["num_steps"]),
        "sample_steps": int(args.sample_steps if args.sample_steps > 0 else diff["num_steps"]),
        "evaluated_freq_count": int(len(rows)),
        "evaluated_sample_count": int(rows[0]["num_real"] if rows else 0),
        "num_gen_per_freq": int(rows[0]["num_fake"] if rows else 0),
        "core_fid_feature_dim": int(core_dim_eff),
        "field_fid_feature_dim": int(field_dim_eff),
        "mean_core_fid": float(np.mean(core_curve)),
        "std_core_fid": float(np.std(core_curve)),
        "mean_field_fid": float(np.mean(field_curve)),
        "std_field_fid": float(np.std(field_curve)),
        "output_dir": str(out_dir),
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print("\nDiffusion-only evaluation finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved frequency metrics: {csv_path}")
    print(f"Saved summary: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate conditional diffusion quality with core-FID and field-FID")

    p.add_argument("--diff_ckpt", type=str, default="ckp/diffusion_core.pt")
    p.add_argument("--ftm_ckpt", type=str, default="ckp/ftm_gpu_checkpoint.pt")
    p.add_argument("--data_h5", type=str, default="helmholtz_dataset_42.h5")
    p.add_argument("--out_dir", type=str, default="visual_data/diffusion_fid_eval")

    p.add_argument("--freq_indices", type=str, default="")
    p.add_argument("--max_freqs", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--random_subset", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--num_gen_per_freq", type=int, default=0)
    p.add_argument("--sample_steps", type=int, default=0)
    p.add_argument("--sample_batch_size", type=int, default=16)
    p.add_argument("--decode_batch_size", type=int, default=64)

    p.add_argument("--core_fid_dim", type=int, default=256)
    p.add_argument("--field_fid_dim", type=int, default=512)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--log_every", type=int, default=1)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
