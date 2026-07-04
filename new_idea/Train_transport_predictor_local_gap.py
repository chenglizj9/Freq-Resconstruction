"""
Train_transport_predictor_local_gap.py
-------------------------------------
Train a complex transport predictor using only local frequency pairs
|delta_omega| <= max_delta_omega.

Design follows Complex-transport-in-Tucker-Core.md:
1) Input: [delta_omega, omega_ref, omega_tar, z_ref]
    where z_ref can be either global stats or the full reference core
2) Output: generator-form transport params (gamma, beta)
3) Prediction:
    mode A: alpha = exp(delta_omega * gamma), G_hat = alpha @ G_ref + beta
    mode B: G_hat = gamma @ G_ref + beta
4) Loss:
      L = L_trans + lambda_id*L_id + lambda_comp*L_comp + lambda_smooth*L_smooth

This script can:
- build solved cores from HDF5 + FTM shared basis if core npz is missing
- train on local-gap pairs only
- evaluate on nearby test pairs
- save metrics, plots and checkpoint in new_idea
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train_FTM_GPU import MLP1D, build_phi, normalize_coords_to_unit


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_ftm_basis(ftm_ckpt: Path, device: torch.device) -> Dict[str, Any]:
    if not ftm_ckpt.exists():
        raise FileNotFoundError(f"FTM checkpoint not found: {ftm_ckpt}")

    ckpt = _safe_torch_load(ftm_ckpt)
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


def _load_complex_fields(data_h5: Path) -> Dict[str, np.ndarray]:
    if not data_h5.exists():
        raise FileNotFoundError(f"Dataset not found: {data_h5}")

    with h5py.File(data_h5, "r") as f:
        if "omega" not in f:
            raise KeyError("HDF5 must contain omega")
        omega = f["omega"][...].astype(np.float32)

        if "data" in f:
            data = f["data"][...].astype(np.float32)
            if data.ndim != 5 or data.shape[-1] != 2:
                raise ValueError(f"Expected data shape (B,M,H,W,2), got {data.shape}")
            fields_re = data[..., 0]
            fields_im = data[..., 1]
        elif "fields_real" in f and "fields_imag" in f:
            fields_re = f["fields_real"][...].astype(np.float32)
            fields_im = f["fields_imag"][...].astype(np.float32)
            if fields_re.shape != fields_im.shape:
                raise ValueError(
                    f"fields_real/fields_imag mismatch: {fields_re.shape} vs {fields_im.shape}"
                )
        else:
            raise KeyError("HDF5 must contain 'data' or ('fields_real','fields_imag').")

        b, m, h, w = fields_re.shape
        if omega.shape[0] != m:
            raise ValueError(f"omega size mismatch: omega={omega.shape[0]} vs M={m}")

        if "grid_x" in f:
            grid_x = f["grid_x"][...].astype(np.float32)
        else:
            grid_x = np.linspace(0.0, 1.0, h, dtype=np.float32)
        if "grid_y" in f:
            grid_y = f["grid_y"][...].astype(np.float32)
        else:
            grid_y = np.linspace(0.0, 1.0, w, dtype=np.float32)

    return {
        "fields_re": fields_re,
        "fields_im": fields_im,
        "omega": omega,
        "grid_x": grid_x,
        "grid_y": grid_y,
    }


def _build_phi_np(ftm_basis: Dict[str, Any], grid_x: np.ndarray, grid_y: np.ndarray, device: torch.device) -> np.ndarray:
    if ftm_basis["normalize_coords"]:
        x_coords = normalize_coords_to_unit(grid_x.astype(np.float64)).astype(np.float32)
        y_coords = normalize_coords_to_unit(grid_y.astype(np.float64)).astype(np.float32)
    else:
        x_coords = grid_x.astype(np.float32)
        y_coords = grid_y.astype(np.float32)

    x_t = torch.from_numpy(x_coords).unsqueeze(-1).to(device)
    y_t = torch.from_numpy(y_coords).unsqueeze(-1).to(device)
    with torch.no_grad():
        phi = build_phi(ftm_basis["net_x"], ftm_basis["net_y"], x_t, y_t)
    return phi.detach().cpu().numpy().astype(np.float64)


def solve_cores_all_samples(
    data_h5: Path,
    ftm_ckpt: Path,
    core_npz_out: Path,
    lstsq_rcond: float,
    chunk_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    fields = _load_complex_fields(data_h5)
    ftm_basis = _load_ftm_basis(ftm_ckpt, device=device)

    fields_re = fields["fields_re"]  # (B,M,H,W)
    fields_im = fields["fields_im"]
    omega = fields["omega"]
    grid_x = fields["grid_x"]
    grid_y = fields["grid_y"]

    b, m, h, w = fields_re.shape
    p = h * w
    r = ftm_basis["rank_x"] * ftm_basis["rank_y"]
    rank_x = int(ftm_basis["rank_x"])
    rank_y = int(ftm_basis["rank_y"])
    if rank_x * rank_y != r:
        raise ValueError(f"Invalid rank shape: rank_x={rank_x}, rank_y={rank_y}, r={r}")

    phi_np = _build_phi_np(ftm_basis=ftm_basis, grid_x=grid_x, grid_y=grid_y, device=device)  # (P,R)
    if phi_np.shape != (p, r):
        raise ValueError(f"Phi shape mismatch: got {phi_np.shape}, expected {(p, r)}")

    pinv = np.linalg.pinv(phi_np, rcond=lstsq_rcond)  # (R,P)

    core_re = np.zeros((b, m, rank_x, rank_y), dtype=np.float32)
    core_im = np.zeros((b, m, rank_x, rank_y), dtype=np.float32)
    rel_rmse = np.zeros((b, m), dtype=np.float64)

    for s in range(0, b, chunk_size):
        e = min(b, s + chunk_size)

        y_re = fields_re[s:e].reshape(e - s, m, p).astype(np.float64, copy=False)
        y_im = fields_im[s:e].reshape(e - s, m, p).astype(np.float64, copy=False)

        c_re_flat = np.einsum("rp,bmp->bmr", pinv, y_re, optimize=True)
        c_im_flat = np.einsum("rp,bmp->bmr", pinv, y_im, optimize=True)
        core_re[s:e] = c_re_flat.reshape(e - s, m, rank_x, rank_y).astype(np.float32)
        core_im[s:e] = c_im_flat.reshape(e - s, m, rank_x, rank_y).astype(np.float32)

        pred_re = np.einsum("pr,bmr->bmp", phi_np, c_re_flat, optimize=True)
        pred_im = np.einsum("pr,bmr->bmp", phi_np, c_im_flat, optimize=True)

        diff2 = (pred_re - y_re) ** 2 + (pred_im - y_im) ** 2
        gt2 = y_re ** 2 + y_im ** 2
        rel_rmse[s:e] = np.sqrt(np.sum(diff2, axis=2) / np.maximum(np.sum(gt2, axis=2), 1e-12))

    core_npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        core_npz_out,
        core_re=core_re,
        core_im=core_im,
        omega=omega,
        rel_rmse=rel_rmse,
    )

    return {
        "core_re": core_re,
        "core_im": core_im,
        "omega": omega,
        "rel_rmse": rel_rmse,
    }


def load_or_build_cores(
    core_npz: Path,
    data_h5: Path,
    ftm_ckpt: Path,
    rank_x: int,
    rank_y: int,
    force_rebuild: bool,
    lstsq_rcond: float,
    chunk_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    if core_npz.exists() and not force_rebuild:
        z = np.load(core_npz)
        need = ["core_re", "core_im", "omega"]
        for k in need:
            if k not in z:
                raise KeyError(f"Missing key in core npz: {k}")
        core_re = z["core_re"].astype(np.float32)
        core_im = z["core_im"].astype(np.float32)
        if core_re.shape != core_im.shape:
            raise ValueError(f"core_re/core_im mismatch: {core_re.shape} vs {core_im.shape}")

        # Backward compatibility: reshape old flat core (B,M,D) to (B,M,rx,ry).
        if core_re.ndim == 3:
            if rank_x * rank_y != core_re.shape[-1]:
                raise ValueError(
                    f"Cannot reshape flat core with D={core_re.shape[-1]} to ({rank_x},{rank_y})"
                )
            b, m, _ = core_re.shape
            core_re = core_re.reshape(b, m, rank_x, rank_y)
            core_im = core_im.reshape(b, m, rank_x, rank_y)
        elif core_re.ndim != 4:
            raise ValueError(f"Unexpected core ndim={core_re.ndim}, expected 3 or 4")

        out = {
            "core_re": core_re,
            "core_im": core_im,
            "omega": z["omega"].astype(np.float32),
        }
        if "rel_rmse" in z:
            out["rel_rmse"] = z["rel_rmse"].astype(np.float64)
        return out

    return solve_cores_all_samples(
        data_h5=data_h5,
        ftm_ckpt=ftm_ckpt,
        core_npz_out=core_npz,
        lstsq_rcond=lstsq_rcond,
        chunk_size=chunk_size,
        device=device,
    )


def split_samples(b: int, train_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.1 <= train_ratio <= 0.95):
        raise ValueError("train_ratio should be in [0.1, 0.95]")
    rng = np.random.default_rng(seed)
    ids = np.arange(b)
    rng.shuffle(ids)
    n_train = int(round(b * train_ratio))
    n_train = max(2, min(n_train, b - 2))
    return np.sort(ids[:n_train]), np.sort(ids[n_train:])


def build_local_pairs(
    omega: np.ndarray,
    sample_ids: np.ndarray,
    max_delta_omega: float,
) -> Dict[str, np.ndarray]:
    m = omega.shape[0]
    sid_list: List[int] = []
    i_ref_list: List[int] = []
    i_tar_list: List[int] = []
    delta_list: List[float] = []

    for s in sample_ids.tolist():
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                d = float(omega[j] - omega[i])
                # if abs(d) <= max_delta_omega + 1e-12:
                if d > 0.0 and d <= max_delta_omega:
                    sid_list.append(int(s))
                    i_ref_list.append(i)
                    i_tar_list.append(j)
                    delta_list.append(d)

    if len(sid_list) == 0:
        raise ValueError("No local pairs found. Increase max_delta_omega.")

    return {
        "sid": np.asarray(sid_list, dtype=np.int64),
        "i_ref": np.asarray(i_ref_list, dtype=np.int64),
        "i_tar": np.asarray(i_tar_list, dtype=np.int64),
        "delta": np.asarray(delta_list, dtype=np.float32),
    }


def build_composition_triplets(
    omega: np.ndarray,
    sample_ids: np.ndarray,
    max_delta_omega: float,
    max_comp_delta_omega: float,
) -> Dict[str, np.ndarray]:
    m = omega.shape[0]
    sid_list: List[int] = []
    i_list: List[int] = []
    j_list: List[int] = []
    k_list: List[int] = []

    for s in sample_ids.tolist():
        for i in range(m - 2):
            for j in range(i + 1, m - 1):
                d12 = float(omega[j] - omega[i])
                if d12 <= 0.0 or d12 > max_delta_omega:
                    continue
                for k in range(j + 1, m):
                    d23 = float(omega[k] - omega[j])
                    d13 = float(omega[k] - omega[i])
                    if d23 <= 0.0 or d23 > max_delta_omega:
                        continue
                    if d13 > max_comp_delta_omega:
                        continue
                    sid_list.append(int(s))
                    i_list.append(i)
                    j_list.append(j)
                    k_list.append(k)

    if len(sid_list) == 0:
        return {
            "sid": np.zeros((0,), dtype=np.int64),
            "i": np.zeros((0,), dtype=np.int64),
            "j": np.zeros((0,), dtype=np.int64),
            "k": np.zeros((0,), dtype=np.int64),
        }

    return {
        "sid": np.asarray(sid_list, dtype=np.int64),
        "i": np.asarray(i_list, dtype=np.int64),
        "j": np.asarray(j_list, dtype=np.int64),
        "k": np.asarray(k_list, dtype=np.int64),
    }


@dataclass
class PairBatch:
    sid: torch.Tensor
    i_ref: torch.Tensor
    i_tar: torch.Tensor
    delta: torch.Tensor


class LocalTransportPredictor(nn.Module):
    def __init__(self, in_dim: int, core_dim: int, hidden_dim: int = 256, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        layers: List[nn.Module] = []
        d_in = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.head_eta = nn.Linear(hidden_dim, core_dim)
        self.head_xi = nn.Linear(hidden_dim, core_dim)
        self.head_beta_re = nn.Linear(hidden_dim, core_dim)
        self.head_beta_im = nn.Linear(hidden_dim, core_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(cond)
        eta = self.head_eta(h)
        xi = self.head_xi(h)
        beta_re = self.head_beta_re(h)
        beta_im = self.head_beta_im(h)
        return eta, xi, beta_re, beta_im


def build_condition_inputs(
    delta: torch.Tensor,      # (B,)
    omega_ref: torch.Tensor,  # (B,)
    omega_tar: torch.Tensor,  # (B,)
    ref_re: torch.Tensor,     # (B,D)
    ref_im: torch.Tensor,     # (B,D)
    omega_min: float,
    omega_max: float,
    delta_scale: float,
    eps: float,
    ref_input_mode: str,
) -> torch.Tensor:
    # Normalize frequency coordinates to [0,1].
    den = max(float(omega_max - omega_min), eps)
    o_ref_n = ((omega_ref - omega_min) / den).unsqueeze(1)
    o_tar_n = ((omega_tar - omega_min) / den).unsqueeze(1)
    d_n = (delta / max(float(delta_scale), eps)).unsqueeze(1)
    ref_re_flat = ref_re.reshape(ref_re.shape[0], -1)
    ref_im_flat = ref_im.reshape(ref_im.shape[0], -1)

    if ref_input_mode == "stats":
        amp = torch.sqrt(ref_re_flat * ref_re_flat + ref_im_flat * ref_im_flat + eps)
        mean_amp = torch.mean(amp, dim=1, keepdim=True)
        std_amp = torch.std(amp, dim=1, keepdim=True, unbiased=False)
        mean_re = torch.mean(ref_re_flat, dim=1, keepdim=True)
        mean_im = torch.mean(ref_im_flat, dim=1, keepdim=True)
        rms_amp = torch.sqrt(torch.mean(amp * amp, dim=1, keepdim=True) + eps)
        cond = torch.cat([d_n, o_ref_n, o_tar_n, mean_amp, std_amp, mean_re, mean_im, rms_amp], dim=1)
    elif ref_input_mode == "full_core":
        cond = torch.cat([d_n, o_ref_n, o_tar_n, ref_re_flat, ref_im_flat], dim=1)
    else:
        raise ValueError(f"Unsupported ref_input_mode: {ref_input_mode}")

    return cond


def infer_condition_dim(core_dim: int, ref_input_mode: str) -> int:
    if ref_input_mode == "stats":
        return 8
    if ref_input_mode == "full_core":
        return 3 + 2 * int(core_dim)
    raise ValueError(f"Unsupported ref_input_mode: {ref_input_mode}")


def transport_predict(
    model: LocalTransportPredictor,
    cond: torch.Tensor,
    delta: torch.Tensor,
    ref_re: torch.Tensor,
    ref_im: torch.Tensor,
    transport_mode: str,
) -> Dict[str, torch.Tensor]:
    eta, xi, beta_re, beta_im = model(cond)
    if ref_re.ndim != 3 or ref_im.ndim != 3:
        raise ValueError(f"Expected ref core shape (B,rx,ry), got {ref_re.shape} / {ref_im.shape}")

    bsz, rx, ry = ref_re.shape
    if rx != ry:
        raise ValueError(
            f"alpha @ G_ref requires square core matrix for left multiply, got ({rx},{ry})."
        )

    eta = eta.reshape(bsz, rx, ry)
    xi = xi.reshape(bsz, rx, ry)
    beta_re = beta_re.reshape(bsz, rx, ry)
    beta_im = beta_im.reshape(bsz, rx, ry)

    if transport_mode == "exp_alpha":
        d = delta.view(-1, 1, 1)
        gamma_c = torch.complex(eta, xi)
        # Matrix exponential gives a valid transport matrix with alpha(0)=I.
        alpha_c = torch.matrix_exp(d * gamma_c)
        a_re = alpha_c.real
        a_im = alpha_c.imag
    elif transport_mode == "direct_gamma":
        # Directly use gamma=(eta+i*xi) as multiplicative transport coefficient.
        a_re = eta
        a_im = xi
    else:
        raise ValueError(f"Unsupported transport_mode: {transport_mode}")

    # Complex matrix product: (A_re + iA_im) @ (G_re + iG_im).
    hat_re = torch.matmul(a_re, ref_re) - torch.matmul(a_im, ref_im) + beta_re
    hat_im = torch.matmul(a_im, ref_re) + torch.matmul(a_re, ref_im) + beta_im

    return {
        "eta": eta,
        "xi": xi,
        "beta_re": beta_re,
        "beta_im": beta_im,
        "alpha_re": a_re,
        "alpha_im": a_im,
        "hat_re": hat_re,
        "hat_im": hat_im,
    }


def _sample_pair_batch(pair: Dict[str, np.ndarray], batch_ids: np.ndarray, device: torch.device) -> PairBatch:
    sid = torch.from_numpy(pair["sid"][batch_ids]).long().to(device)
    i_ref = torch.from_numpy(pair["i_ref"][batch_ids]).long().to(device)
    i_tar = torch.from_numpy(pair["i_tar"][batch_ids]).long().to(device)
    delta = torch.from_numpy(pair["delta"][batch_ids]).float().to(device)
    return PairBatch(sid=sid, i_ref=i_ref, i_tar=i_tar, delta=delta)


def _gather_core(
    core_re_t: torch.Tensor,
    core_im_t: torch.Tensor,
    sid: torch.Tensor,
    freq_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    re = core_re_t[sid, freq_idx, :]
    im = core_im_t[sid, freq_idx, :]
    return re, im


def compute_total_loss(
    model: LocalTransportPredictor,
    core_re_t: torch.Tensor,
    core_im_t: torch.Tensor,
    omega_t: torch.Tensor,
    batch: PairBatch,
    comp_triplets: Dict[str, np.ndarray],
    comp_batch_size: int,
    omega_min: float,
    omega_max: float,
    max_delta_omega: float,
    smooth_delta_omega: float,
    lambda_id: float,
    lambda_comp: float,
    lambda_smooth: float,
    eps: float,
    ref_input_mode: str,
    transport_mode: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ref_re, ref_im = _gather_core(core_re_t, core_im_t, batch.sid, batch.i_ref)
    tar_re, tar_im = _gather_core(core_re_t, core_im_t, batch.sid, batch.i_tar)
    o_ref = omega_t[batch.i_ref]
    o_tar = omega_t[batch.i_tar]

    cond = build_condition_inputs(
        delta=batch.delta,
        omega_ref=o_ref,
        omega_tar=o_tar,
        ref_re=ref_re,
        ref_im=ref_im,
        omega_min=omega_min,
        omega_max=omega_max,
        delta_scale=max_delta_omega,
        eps=eps,
        ref_input_mode=ref_input_mode,
    )
    pred = transport_predict(
        model=model,
        cond=cond,
        delta=batch.delta,
        ref_re=ref_re,
        ref_im=ref_im,
        transport_mode=transport_mode,
    )

    # trans = torch.mean((pred["hat_re"] - tar_re) ** 2 + (pred["hat_im"] - tar_im) ** 2)
    # trans = torch.sum(torch.abs(pred["hat_re"] - tar_re))/torch.sum(torch.abs(tar_re)) + torch.sum(torch.abs(pred["hat_im"] - tar_im))/torch.sum(torch.abs(tar_im))
    err_norm = torch.sqrt(
        torch.sum((pred["hat_re"] - tar_re) ** 2 + (pred["hat_im"] - tar_im) ** 2, dim=(1, 2))
    )
    tar_norm = torch.sqrt(
        torch.sum(tar_re ** 2 + tar_im ** 2, dim=(1, 2))
    )
    rel = err_norm / tar_norm
    trans = torch.mean(rel)
    # Identity consistency: delta=0, omega_tar=omega_ref.
    zero_delta = torch.zeros_like(batch.delta)
    cond_id = build_condition_inputs(
        delta=zero_delta,
        omega_ref=o_ref,
        omega_tar=o_ref,
        ref_re=ref_re,
        ref_im=ref_im,
        omega_min=omega_min,
        omega_max=omega_max,
        delta_scale=max_delta_omega,
        eps=eps,
        ref_input_mode=ref_input_mode,
    )
    pred_id = transport_predict(
        model=model,
        cond=cond_id,
        delta=zero_delta,
        ref_re=ref_re,
        ref_im=ref_im,
        transport_mode=transport_mode,
    )
    eye = torch.eye(ref_re.shape[1], ref_re.shape[2], device=ref_re.device, dtype=ref_re.dtype).unsqueeze(0)
    id_alpha = torch.mean((pred_id["alpha_re"] - eye) ** 2 + pred_id["alpha_im"] ** 2)
    id_beta = torch.mean(pred_id["beta_re"] ** 2 + pred_id["beta_im"] ** 2)
    lid = id_alpha + id_beta

    # Smoothness: encourage local alpha close to identity for very small delta.
    small_mask = torch.abs(batch.delta) <= smooth_delta_omega + 1e-12
    if torch.any(small_mask):
        a_re_s = pred["alpha_re"][small_mask]
        a_im_s = pred["alpha_im"][small_mask]
        eye_s = torch.eye(ref_re.shape[1], ref_re.shape[2], device=ref_re.device, dtype=ref_re.dtype).unsqueeze(0)
        lsmooth = torch.mean((a_re_s - eye_s) ** 2 + a_im_s ** 2)
    else:
        lsmooth = torch.zeros((), device=trans.device, dtype=trans.dtype)

    # Composition consistency.
    if comp_triplets["sid"].size == 0 or lambda_comp <= 0:
        lcomp = torch.zeros((), device=trans.device, dtype=trans.dtype)
    else:
        n_trip = comp_triplets["sid"].shape[0]
        choose = np.random.randint(0, n_trip, size=min(comp_batch_size, n_trip))

        sid = torch.from_numpy(comp_triplets["sid"][choose]).long().to(trans.device)
        i = torch.from_numpy(comp_triplets["i"][choose]).long().to(trans.device)
        j = torch.from_numpy(comp_triplets["j"][choose]).long().to(trans.device)
        k = torch.from_numpy(comp_triplets["k"][choose]).long().to(trans.device)

        # (i->j)
        re_i, im_i = _gather_core(core_re_t, core_im_t, sid, i)
        o_i = omega_t[i]
        o_j = omega_t[j]
        d12 = o_j - o_i
        cond12 = build_condition_inputs(
            delta=d12,
            omega_ref=o_i,
            omega_tar=o_j,
            ref_re=re_i,
            ref_im=im_i,
            omega_min=omega_min,
            omega_max=omega_max,
            delta_scale=max_delta_omega,
            eps=eps,
            ref_input_mode=ref_input_mode,
        )
        p12 = transport_predict(
            model=model,
            cond=cond12,
            delta=d12,
            ref_re=re_i,
            ref_im=im_i,
            transport_mode=transport_mode,
        )

        # (j->k)
        re_j, im_j = _gather_core(core_re_t, core_im_t, sid, j)
        o_k = omega_t[k]
        d23 = o_k - o_j
        cond23 = build_condition_inputs(
            delta=d23,
            omega_ref=o_j,
            omega_tar=o_k,
            ref_re=re_j,
            ref_im=im_j,
            omega_min=omega_min,
            omega_max=omega_max,
            delta_scale=max_delta_omega,
            eps=eps,
            ref_input_mode=ref_input_mode,
        )
        p23 = transport_predict(
            model=model,
            cond=cond23,
            delta=d23,
            ref_re=re_j,
            ref_im=im_j,
            transport_mode=transport_mode,
        )

        # (i->k)
        d13 = o_k - o_i
        cond13 = build_condition_inputs(
            delta=d13,
            omega_ref=o_i,
            omega_tar=o_k,
            ref_re=re_i,
            ref_im=im_i,
            omega_min=omega_min,
            omega_max=omega_max,
            delta_scale=max_delta_omega,
            eps=eps,
            ref_input_mode=ref_input_mode,
        )
        p13 = transport_predict(
            model=model,
            cond=cond13,
            delta=d13,
            ref_re=re_i,
            ref_im=im_i,
            transport_mode=transport_mode,
        )

        # alpha13 ~= alpha23 @ alpha12 (complex matrix product)
        a_comp_re = torch.matmul(p23["alpha_re"], p12["alpha_re"]) - torch.matmul(p23["alpha_im"], p12["alpha_im"])
        a_comp_im = torch.matmul(p23["alpha_im"], p12["alpha_re"]) + torch.matmul(p23["alpha_re"], p12["alpha_im"])
        l_alpha = torch.mean((p13["alpha_re"] - a_comp_re) ** 2 + (p13["alpha_im"] - a_comp_im) ** 2)

        # beta13 ~= alpha23 @ beta12 + beta23
        b_comp_re = torch.matmul(p23["alpha_re"], p12["beta_re"]) - torch.matmul(p23["alpha_im"], p12["beta_im"]) + p23["beta_re"]
        b_comp_im = torch.matmul(p23["alpha_im"], p12["beta_re"]) + torch.matmul(p23["alpha_re"], p12["beta_im"]) + p23["beta_im"]
        l_beta = torch.mean((p13["beta_re"] - b_comp_re) ** 2 + (p13["beta_im"] - b_comp_im) ** 2)
        lcomp = l_alpha + l_beta

    total = trans + lambda_id * lid + lambda_comp * lcomp + lambda_smooth * lsmooth

    logs = {
        "loss": float(total.detach().item()),
        "trans": float(trans.detach().item()),
        "id": float(lid.detach().item()),
        "comp": float(lcomp.detach().item()),
        "smooth": float(lsmooth.detach().item()),
    }
    return total, logs


def evaluate_predictor(
    model: LocalTransportPredictor,
    core_re_t: torch.Tensor,
    core_im_t: torch.Tensor,
    omega_t: torch.Tensor,
    pair: Dict[str, np.ndarray],
    batch_size: int,
    omega_min: float,
    omega_max: float,
    max_delta_omega: float,
    eps: float,
    ref_input_mode: str,
    transport_mode: str,
) -> Dict[str, np.ndarray | float]:
    model.eval()
    n = pair["sid"].shape[0]

    rel_err = np.zeros(n, dtype=np.float64)
    rel_err_identity = np.zeros(n, dtype=np.float64)
    amp_rel = np.zeros(n, dtype=np.float64)
    phase_mae = np.zeros(n, dtype=np.float64)

    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(n, s + batch_size)
            ids = np.arange(s, e, dtype=np.int64)
            b = _sample_pair_batch(pair=pair, batch_ids=ids, device=core_re_t.device)

            ref_re, ref_im = _gather_core(core_re_t, core_im_t, b.sid, b.i_ref)
            tar_re, tar_im = _gather_core(core_re_t, core_im_t, b.sid, b.i_tar)
            o_ref = omega_t[b.i_ref]
            o_tar = omega_t[b.i_tar]

            cond = build_condition_inputs(
                delta=b.delta,
                omega_ref=o_ref,
                omega_tar=o_tar,
                ref_re=ref_re,
                ref_im=ref_im,
                omega_min=omega_min,
                omega_max=omega_max,
                delta_scale=max_delta_omega,
                eps=eps,
                ref_input_mode=ref_input_mode,
            )
            pred = transport_predict(
                model=model,
                cond=cond,
                delta=b.delta,
                ref_re=ref_re,
                ref_im=ref_im,
                transport_mode=transport_mode,
            )

            hat_re = pred["hat_re"]
            hat_im = pred["hat_im"]

            # Relative vector error.
            num = torch.sqrt(torch.sum((hat_re - tar_re) ** 2 + (hat_im - tar_im) ** 2, dim=(1, 2)) + eps)
            den = torch.sqrt(torch.sum(tar_re ** 2 + tar_im ** 2, dim=(1, 2)) + eps)
            rel = num / den

            num_id = torch.sqrt(torch.sum((ref_re - tar_re) ** 2 + (ref_im - tar_im) ** 2, dim=(1, 2)) + eps)
            rel_id = num_id / den

            # Amplitude relative error and phase MAE.
            amp_hat = torch.sqrt(hat_re * hat_re + hat_im * hat_im + eps)
            amp_tar = torch.sqrt(tar_re * tar_re + tar_im * tar_im + eps)
            amp_re = torch.mean(torch.abs(amp_hat - amp_tar) / (amp_tar + eps), dim=(1, 2))

            phase_hat = torch.atan2(hat_im, hat_re)
            phase_tar = torch.atan2(tar_im, tar_re)
            dph = torch.atan2(torch.sin(phase_hat - phase_tar), torch.cos(phase_hat - phase_tar))
            ph_mae = torch.mean(torch.abs(dph), dim=(1, 2))

            rel_err[s:e] = rel.detach().cpu().numpy().astype(np.float64)
            rel_err_identity[s:e] = rel_id.detach().cpu().numpy().astype(np.float64)
            amp_rel[s:e] = amp_re.detach().cpu().numpy().astype(np.float64)
            phase_mae[s:e] = ph_mae.detach().cpu().numpy().astype(np.float64)

    return {
        "rel_err": rel_err,
        "rel_err_identity": rel_err_identity,
        "amp_rel": amp_rel,
        "phase_mae": phase_mae,
        "mean_rel_err": float(np.mean(rel_err)),
        "mean_rel_err_identity": float(np.mean(rel_err_identity)),
        "mean_amp_rel": float(np.mean(amp_rel)),
        "mean_phase_mae": float(np.mean(phase_mae)),
    }


def _infer_core_shape(core_dim: int, rank_x: int, rank_y: int) -> Tuple[int, int]:
    if rank_x > 0 and rank_y > 0 and rank_x * rank_y == core_dim:
        return rank_x, rank_y
    r = int(round(math.sqrt(core_dim)))
    if r * r == core_dim:
        return r, r
    raise ValueError(
        f"Cannot infer 2D core shape from core_dim={core_dim}. Set rank_x/rank_y explicitly."
    )


def plot_training_curve(history: Dict[str, List[float]], out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    ax.plot(history["epoch"], history["loss"], lw=1.8, label="train total loss")
    ax.plot(history["epoch"], history["trans"], lw=1.6, label="train trans")
    ax.plot(history["epoch"], history["id"], lw=1.2, label="train id")
    ax.plot(history["epoch"], history["comp"], lw=1.2, label="train comp")
    ax.plot(history["epoch"], history["smooth"], lw=1.2, label="train smooth")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Local-gap transport predictor training curve")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_error_vs_delta(pair: Dict[str, np.ndarray], eval_out: Dict[str, np.ndarray | float], out_file: Path) -> None:
    delta_abs = np.abs(pair["delta"]).astype(np.float64)
    rel = eval_out["rel_err"].astype(np.float64)
    rel_id = eval_out["rel_err_identity"].astype(np.float64)

    uniq = sorted(np.unique(delta_abs))
    x = []
    y = []
    y_id = []
    for d in uniq:
        m = np.isclose(delta_abs, d)
        if np.any(m):
            x.append(float(d))
            y.append(float(np.mean(rel[m])))
            y_id.append(float(np.mean(rel_id[m])))

    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    ax.plot(x, y, marker="o", lw=2.0, label="predictor")
    ax.plot(x, y_id, marker="o", lw=2.0, label="identity")
    ax.set_xlabel("|delta omega|")
    ax.set_ylabel("mean relative error")
    ax.set_title("Prediction error vs local delta omega")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_case_compare(
    core_ref_re: np.ndarray,
    core_ref_im: np.ndarray,
    core_tar_re: np.ndarray,
    core_tar_im: np.ndarray,
    core_pred_re: np.ndarray,
    core_pred_im: np.ndarray,
    omega_ref: float,
    omega_tar: float,
    out_file: Path,
) -> None:
    vmax_re = float(np.percentile(np.abs(np.concatenate([core_ref_re.ravel(), core_tar_re.ravel(), core_pred_re.ravel()])), 99.5))
    vmax_im = float(np.percentile(np.abs(np.concatenate([core_ref_im.ravel(), core_tar_im.ravel(), core_pred_im.ravel()])), 99.5))
    vmax_re = max(vmax_re, 1e-8)
    vmax_im = max(vmax_im, 1e-8)

    err_re = np.abs(core_pred_re - core_tar_re)
    err_im = np.abs(core_pred_im - core_tar_im)

    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.0), constrained_layout=True)
    items = [
        (core_ref_re, f"Ref Real\nomega={omega_ref:.4g}", "RdBu_r", -vmax_re, vmax_re),
        (core_tar_re, f"Target Real\nomega={omega_tar:.4g}", "RdBu_r", -vmax_re, vmax_re),
        (core_pred_re, "Pred Real", "RdBu_r", -vmax_re, vmax_re),
        (err_re, "|Pred-Target| Real", "magma", 0.0, float(np.percentile(err_re, 99.5))),
        (core_ref_im, f"Ref Imag\nomega={omega_ref:.4g}", "RdBu_r", -vmax_im, vmax_im),
        (core_tar_im, f"Target Imag\nomega={omega_tar:.4g}", "RdBu_r", -vmax_im, vmax_im),
        (core_pred_im, "Pred Imag", "RdBu_r", -vmax_im, vmax_im),
        (err_im, "|Pred-Target| Imag", "magma", 0.0, float(np.percentile(err_im, 99.5))),
    ]

    for ax, (img, title, cmap, vmin, vmax) in zip(axes.flat, items):
        vmax = max(float(vmax), float(vmin) + 1e-8)
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Local transport predictor: core comparison case", fontsize=13)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def save_pair_metrics_csv(pair: Dict[str, np.ndarray], eval_out: Dict[str, np.ndarray | float], omega: np.ndarray, out_csv: Path) -> None:
    rel = eval_out["rel_err"].astype(np.float64)
    rel_id = eval_out["rel_err_identity"].astype(np.float64)
    amp = eval_out["amp_rel"].astype(np.float64)
    ph = eval_out["phase_mae"].astype(np.float64)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(
            "sample_idx,freq_ref,freq_tar,omega_ref,omega_tar,delta_omega,"
            "rel_err_pred,rel_err_identity,amp_rel_err,phase_mae\n"
        )
        for i in range(pair["sid"].shape[0]):
            fr = int(pair["i_ref"][i])
            ft = int(pair["i_tar"][i])
            f.write(
                f"{int(pair['sid'][i])},{fr},{ft},{float(omega[fr]):.8g},{float(omega[ft]):.8g},"
                f"{float(pair['delta'][i]):.8g},{rel[i]:.8g},{rel_id[i]:.8g},{amp[i]:.8g},{ph[i]:.8g}\n"
            )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _select_device(args.device)

    core_npz = Path(args.core_npz)
    core_bundle = load_or_build_cores(
        core_npz=core_npz,
        data_h5=Path(args.data_h5),
        ftm_ckpt=Path(args.ftm_ckpt),
        rank_x=args.rank_x,
        rank_y=args.rank_y,
        force_rebuild=args.rebuild_cores,
        lstsq_rcond=args.lstsq_rcond,
        chunk_size=args.solve_chunk_size,
        device=device,
    )

    core_re = core_bundle["core_re"]  # (B,M,rx,ry)
    core_im = core_bundle["core_im"]
    omega = core_bundle["omega"]

    if core_re.ndim != 4:
        raise ValueError(f"Expected core shape (B,M,rx,ry), got {core_re.shape}")
    if core_im.shape != core_re.shape:
        raise ValueError(f"core_re/core_im mismatch: {core_re.shape} vs {core_im.shape}")

    b, m, rank_x, rank_y = core_re.shape
    d = rank_x * rank_y
    if args.rank_x > 0 and args.rank_y > 0 and (args.rank_x != rank_x or args.rank_y != rank_y):
        print(
            f"[warn] args rank=({args.rank_x},{args.rank_y}) != core rank=({rank_x},{rank_y}); use core rank."
        )

    train_ids, test_ids = split_samples(b=b, train_ratio=args.train_ratio, seed=args.seed)
    pair_train = build_local_pairs(omega=omega, sample_ids=train_ids, max_delta_omega=args.max_delta_omega)
    pair_test = build_local_pairs(omega=omega, sample_ids=test_ids, max_delta_omega=args.max_delta_omega)

    max_comp_delta = args.max_comp_delta_omega
    if max_comp_delta <= 0:
        max_comp_delta = args.max_delta_omega

    comp_triplets = build_composition_triplets(
        omega=omega,
        sample_ids=train_ids,
        max_delta_omega=args.max_delta_omega,
        max_comp_delta_omega=max_comp_delta,
    )

    core_re_t = torch.from_numpy(core_re).float().to(device)
    core_im_t = torch.from_numpy(core_im).float().to(device)
    omega_t = torch.from_numpy(omega).float().to(device)

    in_dim = infer_condition_dim(core_dim=d, ref_input_mode=args.ref_input_mode)

    model = LocalTransportPredictor(
        in_dim=in_dim,
        core_dim=d,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    omega_min = float(np.min(omega))
    omega_max = float(np.max(omega))

    history: Dict[str, List[float]] = {
        "epoch": [],
        "loss": [],
        "trans": [],
        "id": [],
        "comp": [],
        "smooth": [],
    }

    n_train = pair_train["sid"].shape[0]

    print("\n" + "-" * 72)
    print("Train local-gap complex transport predictor")
    print(f"device={device}")
    print(f"cores: B={b}, M={m}, D={d}, rank=({rank_x},{rank_y})")
    print(f"train samples={train_ids.size}, test samples={test_ids.size}")
    print(f"train pairs={pair_train['sid'].shape[0]}, test pairs={pair_test['sid'].shape[0]}")
    print(f"train triplets(comp)={comp_triplets['sid'].shape[0]}")
    print(f"max_delta_omega={args.max_delta_omega}, smooth_delta_omega={args.smooth_delta_omega}")
    print(f"ref_input_mode={args.ref_input_mode}, cond_dim={in_dim}")
    print(f"transport_mode={args.transport_mode}")
    print("-" * 72 + "\n")

    for epoch in range(1, args.epochs + 1):
        model.train()

        perm = np.random.permutation(n_train)
        running = {"loss": 0.0, "trans": 0.0, "id": 0.0, "comp": 0.0, "smooth": 0.0}
        n_batches = 0

        for s in range(0, n_train, args.batch_size):
            e = min(n_train, s + args.batch_size)
            ids = perm[s:e]
            batch = _sample_pair_batch(pair=pair_train, batch_ids=ids, device=device)

            optimizer.zero_grad(set_to_none=True)

            loss, logs = compute_total_loss(
                model=model,
                core_re_t=core_re_t,
                core_im_t=core_im_t,
                omega_t=omega_t,
                batch=batch,
                comp_triplets=comp_triplets,
                comp_batch_size=args.comp_batch_size,
                omega_min=omega_min,
                omega_max=omega_max,
                max_delta_omega=args.max_delta_omega,
                smooth_delta_omega=args.smooth_delta_omega,
                lambda_id=args.lambda_id,
                lambda_comp=args.lambda_comp,
                lambda_smooth=args.lambda_smooth,
                eps=args.eps,
                ref_input_mode=args.ref_input_mode,
                transport_mode=args.transport_mode,
            )
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            # scheduler.step()
            for k in running.keys():
                running[k] += logs[k]
            n_batches += 1

        scheduler.step()
        for k in running.keys():
            running[k] /= max(n_batches, 1)

        history["epoch"].append(epoch)
        history["loss"].append(running["loss"])
        history["trans"].append(running["trans"])
        history["id"].append(running["id"])
        history["comp"].append(running["comp"])
        history["smooth"].append(running["smooth"])

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"[epoch {epoch:04d}/{args.epochs}] "
                f"loss={running['loss']:.6e} trans={running['trans']:.6e} "
                f"id={running['id']:.6e} comp={running['comp']:.6e} smooth={running['smooth']:.6e}"
            )

    # Evaluate predictor on near-gap test pairs.
    eval_out = evaluate_predictor(
        model=model,
        core_re_t=core_re_t,
        core_im_t=core_im_t,
        omega_t=omega_t,
        pair=pair_test,
        batch_size=args.eval_batch_size,
        omega_min=omega_min,
        omega_max=omega_max,
        max_delta_omega=args.max_delta_omega,
        eps=args.eps,
        ref_input_mode=args.ref_input_mode,
        transport_mode=args.transport_mode,
    )

    # Save pair metrics.
    pair_csv = out_dir / "test_pair_metrics_local.csv"
    save_pair_metrics_csv(pair=pair_test, eval_out=eval_out, omega=omega, out_csv=pair_csv)

    # Save checkpoint.
    ckpt_path = out_dir / "transport_predictor_local.pt"
    ckpt = {
        "model_state": model.state_dict(),
        "model_config": {
            "in_dim": int(in_dim),
            "core_dim": int(d),
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "dropout": float(args.dropout),
            "ref_input_mode": str(args.ref_input_mode),
            "transport_mode": str(args.transport_mode),
        },
        "omega_stats": {
            "omega_min": omega_min,
            "omega_max": omega_max,
            "max_delta_omega": float(args.max_delta_omega),
        },
        "core_shape": {
            "rank_x": int(rank_x),
            "rank_y": int(rank_y),
            "core_dim": int(d),
        },
        "train_info": {
            "num_samples": int(b),
            "num_freqs": int(m),
            "train_samples": int(train_ids.size),
            "test_samples": int(test_ids.size),
            "train_pairs": int(pair_train["sid"].shape[0]),
            "test_pairs": int(pair_test["sid"].shape[0]),
            "train_triplets": int(comp_triplets["sid"].shape[0]),
        },
        "history": history,
        "config": vars(args),
    }
    torch.save(ckpt, ckpt_path)

    # Plots.
    train_curve = out_dir / "training_curve_local.png"
    plot_training_curve(history=history, out_file=train_curve)

    err_curve = out_dir / "error_vs_delta_local.png"
    plot_error_vs_delta(pair=pair_test, eval_out=eval_out, out_file=err_curve)

    # Demo case: choose a test pair with smallest |delta|.
    delta_abs = np.abs(pair_test["delta"])
    demo_idx = int(np.argmin(delta_abs))
    sid = int(pair_test["sid"][demo_idx])
    i_ref = int(pair_test["i_ref"][demo_idx])
    i_tar = int(pair_test["i_tar"][demo_idx])

    model.eval()
    with torch.no_grad():
        sid_t = torch.tensor([sid], device=device, dtype=torch.long)
        i_ref_t = torch.tensor([i_ref], device=device, dtype=torch.long)
        i_tar_t = torch.tensor([i_tar], device=device, dtype=torch.long)
        d_t = torch.tensor([float(omega[i_tar] - omega[i_ref])], device=device, dtype=torch.float32)

        ref_re_t, ref_im_t = _gather_core(core_re_t, core_im_t, sid_t, i_ref_t)
        tar_re_t, tar_im_t = _gather_core(core_re_t, core_im_t, sid_t, i_tar_t)

        cond = build_condition_inputs(
            delta=d_t,
            omega_ref=omega_t[i_ref_t],
            omega_tar=omega_t[i_tar_t],
            ref_re=ref_re_t,
            ref_im=ref_im_t,
            omega_min=omega_min,
            omega_max=omega_max,
            delta_scale=args.max_delta_omega,
            eps=args.eps,
            ref_input_mode=args.ref_input_mode,
        )
        pred = transport_predict(
            model=model,
            cond=cond,
            delta=d_t,
            ref_re=ref_re_t,
            ref_im=ref_im_t,
            transport_mode=args.transport_mode,
        )

        pred_re = pred["hat_re"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        pred_im = pred["hat_im"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        ref_re = ref_re_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        ref_im = ref_im_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        tar_re = tar_re_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        tar_im = tar_im_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

    demo_rel = float(
        np.linalg.norm((pred_re - tar_re) + 1j * (pred_im - tar_im))
        / max(np.linalg.norm(tar_re + 1j * tar_im), args.eps)
    )

    demo_plot = out_dir / "demo_case_compare_local.png"
    plot_case_compare(
        core_ref_re=ref_re.reshape(rank_x, rank_y),
        core_ref_im=ref_im.reshape(rank_x, rank_y),
        core_tar_re=tar_re.reshape(rank_x, rank_y),
        core_tar_im=tar_im.reshape(rank_x, rank_y),
        core_pred_re=pred_re.reshape(rank_x, rank_y),
        core_pred_im=pred_im.reshape(rank_x, rank_y),
        omega_ref=float(omega[i_ref]),
        omega_tar=float(omega[i_tar]),
        out_file=demo_plot,
    )

    summary = {
        "core_npz": str(core_npz),
        "ftm_ckpt": str(args.ftm_ckpt),
        "data_h5": str(args.data_h5),
        "device": str(device),
        "num_samples": int(b),
        "num_freqs": int(m),
        "core_dim": int(d),
        "rank_x": int(rank_x),
        "rank_y": int(rank_y),
        "max_delta_omega": float(args.max_delta_omega),
        "transport_mode": str(args.transport_mode),
        "train_pairs": int(pair_train["sid"].shape[0]),
        "test_pairs": int(pair_test["sid"].shape[0]),
        "train_triplets": int(comp_triplets["sid"].shape[0]),
        "mean_rel_err_predictor": float(eval_out["mean_rel_err"]),
        "mean_rel_err_identity": float(eval_out["mean_rel_err_identity"]),
        "mean_amp_rel_err": float(eval_out["mean_amp_rel"]),
        "mean_phase_mae": float(eval_out["mean_phase_mae"]),
        "demo_case": {
            "sample_idx": int(sid),
            "freq_ref": int(i_ref),
            "freq_tar": int(i_tar),
            "omega_ref": float(omega[i_ref]),
            "omega_tar": float(omega[i_tar]),
            "delta_omega": float(omega[i_tar] - omega[i_ref]),
            "demo_rel_err": float(demo_rel),
        },
        "outputs": {
            "checkpoint": str(ckpt_path),
            "pair_metrics_csv": str(pair_csv),
            "training_curve": str(train_curve),
            "error_vs_delta": str(err_curve),
            "demo_plot": str(demo_plot),
        },
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nTraining finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved summary: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train local-gap complex transport predictor")

    p.add_argument("--data_h5", type=str, default="helmholtz_dataset_42_new_idea_mask10.h5")
    p.add_argument("--ftm_ckpt", type=str, default="../ckp/ftm_gpu_checkpoint.pt")
    p.add_argument("--core_npz", type=str, default="local_transport_gap/solved_cores_all_samples.npz")
    p.add_argument("--rebuild_cores", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--lstsq_rcond", type=float, default=1e-6)
    p.add_argument("--solve_chunk_size", type=int, default=8)

    p.add_argument("--out_dir", type=str, default="local_transport_gap")

    p.add_argument("--max_delta_omega", type=float, default=1.0)
    p.add_argument("--smooth_delta_omega", type=float, default=1.0)
    p.add_argument("--max_comp_delta_omega", type=float, default=1.0)

    p.add_argument("--train_ratio", type=float, default=0.7)

    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--comp_batch_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=1024)

    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=0.0)

    p.add_argument("--lambda_id", type=float, default=0.1)
    p.add_argument("--lambda_comp", type=float, default=0.1)
    p.add_argument("--lambda_smooth", type=float, default=0.05)

    p.add_argument(
        "--ref_input_mode",
        type=str,
        default="full_core",
        choices=["stats", "full_core"],
        help="Conditioning input mode: global stats or full reference core",
    )

    p.add_argument(
        "--transport_mode",
        type=str,
        default="exp_alpha",
        choices=["direct_gamma", "exp_alpha"],
        help="Transport formulation: direct gamma multiplication or exp(alpha) formulation",
    )

    p.add_argument("--rank_x", type=int, default=24)
    p.add_argument("--rank_y", type=int, default=24)

    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--log_every", type=int, default=10)
    return p


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
