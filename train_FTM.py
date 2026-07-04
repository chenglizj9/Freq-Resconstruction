"""
train_FTM.py
------------
Train a frequency-axis Functional Tucker Model (FTM) for 2D complex fields.

This script follows the alternating optimization plan:
1) Fix latent functions and solve core sequences by linear least-squares
   with frequency-weighted TV regularization.
2) Fix core sequences and update latent functions with Adam on observed points.

Expected dataset format (from Generate_dataset.py)
---------------------------------------------------
HDF5 keys:
- data      : (B, M, H, W, 2) float32 (real/imag channels)
- mask_tr   : (M, H, W, 2) or (B, M, H, W, 2) in {0,1}
- omega     : (M,) float64
- grid_x    : (H,) float64 (optional)
- grid_y    : (W,) float64 (optional)

Output checkpoint
-----------------
A torch .pt checkpoint containing:
- latent function network weights (x/y)
- learned cores (real/imag): (B, M, Rx, Ry)
- omega/grid coordinates and training config
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import splu


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class MLP1D(nn.Module):
    def __init__(
        self,
        out_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        activation: str = "sine",
    ):
        super().__init__()
        if activation == "sine":
            act = Sine
        elif activation == "relu":
            act = nn.ReLU
        elif activation == "tanh":
            act = nn.Tanh
        else:
            raise ValueError("activation must be one of: sine, relu, tanh")

        layers: List[nn.Module] = []
        in_dim = 1
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights(activation)

    def _init_weights(self, activation: str) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if activation == "sine":
                    # Keep sine activations in a stable range initially.
                    nn.init.uniform_(m.weight, -0.5, 0.5)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class FTMData:
    data: np.ndarray          # (B, M, H, W, C)
    mask: np.ndarray          # (M, H, W, C) or (B, M, H, W, C)
    omega: np.ndarray         # (M,)
    grid_x: np.ndarray        # (H,)
    grid_y: np.ndarray        # (W,)
    data_scale: float


# -----------------------------------------------------------------------------
# Data I/O and preparation
# -----------------------------------------------------------------------------


def _try_load_scale_from_sidecar(h5_path: Path, metadata_npy: str) -> float:
    if metadata_npy:
        meta_path = Path(metadata_npy)
    else:
        meta_path = h5_path.with_name(f"{h5_path.stem}_metadata.npy")

    if not meta_path.exists():
        return 1.0

    try:
        meta_obj = np.load(meta_path, allow_pickle=True).item()
        return float(meta_obj.get("data", {}).get("data_scale", 1.0))
    except Exception:
        return 1.0


def load_ftm_data(h5_path: str, metadata_npy: str = "", max_samples: int = 0) -> FTMData:
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with h5py.File(path, "r") as f:
        if "data" not in f or "mask_tr" not in f:
            raise KeyError("HDF5 must contain datasets 'data' and 'mask_tr'.")

        data = f["data"][...].astype(np.float32)
        mask = f["mask_tr"][...].astype(np.float32)
        omega = f["omega"][...].astype(np.float64)

        if "grid_x" in f:
            grid_x = f["grid_x"][...].astype(np.float64)
        else:
            grid_x = np.linspace(0.0, 1.0, data.shape[2], dtype=np.float64)

        if "grid_y" in f:
            grid_y = f["grid_y"][...].astype(np.float64)
        else:
            grid_y = np.linspace(0.0, 1.0, data.shape[3], dtype=np.float64)

    if data.ndim != 5:
        raise ValueError(f"Expected data shape (B,M,H,W,C), got {data.shape}")

    if data.shape[-1] != 2:
        raise ValueError("This implementation expects 2 channels (real/imag).")

    if max_samples > 0:
        data = data[:max_samples]

    if mask.ndim == 4:
        if tuple(mask.shape) != tuple(data.shape[1:]):
            raise ValueError(
                "mask_tr shape must match data's (M,H,W,C) when mask is shared. "
                f"Got data={data.shape}, mask={mask.shape}."
            )
    elif mask.ndim == 5:
        if mask.shape[0] < data.shape[0]:
            raise ValueError(
                "mask_tr has fewer samples than data. "
                f"Got data B={data.shape[0]}, mask B={mask.shape[0]}."
            )
        # Align with max_samples slicing on data.
        mask = mask[: data.shape[0]]
        if tuple(mask.shape[1:]) != tuple(data.shape[1:]):
            raise ValueError(
                "mask_tr per-sample shape must match data's (B,M,H,W,C). "
                f"Got data={data.shape}, mask={mask.shape}."
            )
    else:
        raise ValueError(f"Expected mask shape (M,H,W,C) or (B,M,H,W,C), got {mask.shape}")

    if not np.all(np.diff(omega) > 0):
        raise ValueError("omega must be strictly increasing.")

    data_scale = _try_load_scale_from_sidecar(path, metadata_npy)

    return FTMData(
        data=data,
        mask=mask,
        omega=omega,
        grid_x=grid_x,
        grid_y=grid_y,
        data_scale=data_scale,
    )


def normalize_coords_to_unit(x: np.ndarray) -> np.ndarray:
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if math.isclose(x_max, x_min):
        return np.zeros_like(x, dtype=np.float64)
    return (x - x_min) / (x_max - x_min)


def build_observation_views(data: np.ndarray, mask: np.ndarray):
    """
    Build observation index lists and observed values.

    Returns
    -------
    obs_idx_by_channel : List[List[np.ndarray]]
        obs_idx_by_channel[c][m] is 1-D flat index array into H*W
    y_obs_by_channel : List[List[np.ndarray]]
        y_obs_by_channel[c][m] is shape (B, n_obs_m)
    """
    B, M, H, W, C = data.shape
    P = H * W

    mask_bool = mask > 0.5
    flat_data = data.reshape(B, M, P, C)

    obs_idx_by_channel: List[List[np.ndarray]] = []
    y_obs_by_channel: List[List[np.ndarray]] = []

    for c in range(C):
        c_idx: List[np.ndarray] = []
        c_vals: List[np.ndarray] = []
        for m in range(M):
            idx = np.flatnonzero(mask_bool[m, :, :, c].reshape(-1))
            if idx.size == 0:
                raise ValueError(f"No observations at channel={c}, freq_index={m}.")
            y = flat_data[:, m, idx, c]  # (B, n_obs)
            c_idx.append(idx.astype(np.int64))
            c_vals.append(y.astype(np.float32))
        obs_idx_by_channel.append(c_idx)
        y_obs_by_channel.append(c_vals)

    return obs_idx_by_channel, y_obs_by_channel


def build_observation_views_per_sample(data: np.ndarray, mask: np.ndarray):
    """
    Build per-sample observation index lists and observed values.

    Returns
    -------
    obs_idx_by_channel : List[List[List[np.ndarray]]]
        obs_idx_by_channel[c][b][m] is 1-D flat index array into H*W
    y_obs_by_channel : List[List[List[np.ndarray]]]
        y_obs_by_channel[c][b][m] is shape (n_obs_m,)
    """
    if mask.ndim != 5:
        raise ValueError("build_observation_views_per_sample expects mask with shape (B,M,H,W,C).")

    B, M, H, W, C = data.shape
    P = H * W
    mask_bool = mask > 0.5
    flat_data = data.reshape(B, M, P, C)

    obs_idx_by_channel: List[List[List[np.ndarray]]] = []
    y_obs_by_channel: List[List[List[np.ndarray]]] = []

    for c in range(C):
        c_idx: List[List[np.ndarray]] = []
        c_vals: List[List[np.ndarray]] = []
        for b in range(B):
            b_idx: List[np.ndarray] = []
            b_vals: List[np.ndarray] = []
            for m in range(M):
                idx = np.flatnonzero(mask_bool[b, m, :, :, c].reshape(-1))
                if idx.size == 0:
                    raise ValueError(f"No observations at sample={b}, channel={c}, freq_index={m}.")
                y = flat_data[b, m, idx, c]  # (n_obs,)
                b_idx.append(idx.astype(np.int64))
                b_vals.append(y.astype(np.float32))
            c_idx.append(b_idx)
            c_vals.append(b_vals)
        obs_idx_by_channel.append(c_idx)
        y_obs_by_channel.append(c_vals)

    return obs_idx_by_channel, y_obs_by_channel


# -----------------------------------------------------------------------------
# Core closed-form update
# -----------------------------------------------------------------------------


def build_phi(
    net_x: nn.Module,
    net_y: nn.Module,
    x_coords: torch.Tensor,   # (H,1)
    y_coords: torch.Tensor,   # (W,1)
) -> torch.Tensor:
    """Build flattened Kronecker features Phi(x,y), shape (H*W, Rx*Ry)."""
    fx = net_x(x_coords)  # (H, Rx)
    fy = net_y(y_coords)  # (W, Ry)
    phi = torch.einsum("ir,jq->ijrq", fx, fy).reshape(-1, fx.shape[1] * fy.shape[1])
    return phi


def _tv_weights(omega: np.ndarray, beta: float) -> np.ndarray:
    d_omega = np.diff(omega)
    if np.any(d_omega <= 0.0):
        raise ValueError("omega must be strictly increasing for TV regularization.")
    return beta / (d_omega ** 2)


def build_linear_system(
    ata_blocks: Sequence[np.ndarray],
    omega: np.ndarray,
    beta: float,
    ridge: float,
) -> csc_matrix:
    """
    Build block-tridiagonal normal matrix H for
      sum_m ||A_m w_m - y_m||^2 + beta * sum_m ||(w_m-w_{m-1})/d_omega||^2 + ridge*||w||^2.
    """
    M = len(ata_blocks)
    R = ata_blocks[0].shape[0]
    H = lil_matrix((M * R, M * R), dtype=np.float64)

    inv_d2 = _tv_weights(omega, beta)

    alpha = np.zeros(M, dtype=np.float64)
    if M > 1:
        alpha[0] += inv_d2[0]
        alpha[-1] += inv_d2[-1]
    if M > 2:
        alpha[1:-1] += inv_d2[:-1] + inv_d2[1:]

    I_R = np.eye(R, dtype=np.float64)

    for m in range(M):
        blk = ata_blocks[m] + (ridge + alpha[m]) * I_R
        rs = m * R
        re = (m + 1) * R
        H[rs:re, rs:re] = blk

    for m in range(M - 1):
        g = inv_d2[m]
        off = -g * I_R
        r0 = m * R
        r1 = (m + 1) * R
        r2 = (m + 2) * R
        H[r0:r1, r1:r2] = off
        H[r1:r2, r0:r1] = off

    return H.tocsc()


def solve_cores_for_channel(
    phi_np: np.ndarray,                # (P, R)
    obs_idx_by_m: Sequence[np.ndarray],
    y_obs_by_m: Sequence[np.ndarray],  # each (B, n_obs_m)
    omega: np.ndarray,
    beta: float,
    ridge: float,
) -> np.ndarray:
    """
    Solve all sample cores for one channel in one shot using shared H and multiple RHS.

    Returns
    -------
    cores : (B, M, R)
    """
    M = len(obs_idx_by_m)
    B = y_obs_by_m[0].shape[0]
    R = phi_np.shape[1]

    ata_blocks: List[np.ndarray] = []
    rhs = np.zeros((M * R, B), dtype=np.float64)

    for m in range(M):
        idx = obs_idx_by_m[m]
        A_m = phi_np[idx, :]  # (n_obs, R)

        ata = A_m.T @ A_m
        ata_blocks.append(ata)

        # y_obs_by_m[m]: (B, n_obs)
        y_m = y_obs_by_m[m].astype(np.float64, copy=False)
        rhs[m * R : (m + 1) * R, :] = A_m.T @ y_m.T

    H = build_linear_system(ata_blocks=ata_blocks, omega=omega, beta=beta, ridge=ridge)
    lu = splu(H)
    sol = lu.solve(rhs)  # (M*R, B)

    # from scipy.sparse.linalg import spsolve
    # sol = spsolve(H, rhs)
    cores = sol.T.reshape(B, M, R)
    return cores


def solve_cores_for_one_sample(
    phi_np: np.ndarray,                 # (P, R)
    obs_idx_by_m: Sequence[np.ndarray],
    y_obs_by_m: Sequence[np.ndarray],   # each (n_obs_m,)
    omega: np.ndarray,
    beta: float,
    ridge: float,
) -> np.ndarray:
    """Solve one sample's core sequence for one channel. Returns shape (M, R)."""
    M = len(obs_idx_by_m)
    R = phi_np.shape[1]

    ata_blocks: List[np.ndarray] = []
    rhs = np.zeros((M * R,), dtype=np.float64)

    for m in range(M):
        idx = obs_idx_by_m[m]
        A_m = phi_np[idx, :]  # (n_obs, R)
        y_m = y_obs_by_m[m].astype(np.float64, copy=False)

        ata_blocks.append(A_m.T @ A_m)
        rhs[m * R : (m + 1) * R] = A_m.T @ y_m

    H = build_linear_system(ata_blocks=ata_blocks, omega=omega, beta=beta, ridge=ridge)
    lu = splu(H)
    sol = lu.solve(rhs)  # (M*R,)
    return sol.reshape(M, R)


def solve_cores_for_channel_per_sample(
    phi_np: np.ndarray,                           # (P, R)
    obs_idx_by_bm: Sequence[Sequence[np.ndarray]],
    y_obs_by_bm: Sequence[Sequence[np.ndarray]],
    omega: np.ndarray,
    beta: float,
    ridge: float,
) -> np.ndarray:
    """
    Solve all sample cores for one channel with per-sample masks.

    Parameters
    ----------
    obs_idx_by_bm : obs_idx_by_bm[b][m] -> index array
    y_obs_by_bm   : y_obs_by_bm[b][m]   -> value array
    """
    B = len(obs_idx_by_bm)
    M = len(obs_idx_by_bm[0])
    R = phi_np.shape[1]

    cores = np.zeros((B, M, R), dtype=np.float64)
    for b in range(B):
        cores[b] = solve_cores_for_one_sample(
            phi_np=phi_np,
            obs_idx_by_m=obs_idx_by_bm[b],
            y_obs_by_m=y_obs_by_bm[b],
            omega=omega,
            beta=beta,
            ridge=ridge,
        )
    return cores


# -----------------------------------------------------------------------------
# Loss and training
# -----------------------------------------------------------------------------


def weighted_tv_value(cores: np.ndarray, omega: np.ndarray) -> float:
    """
    cores: (B, M, R)
    Returns average weighted TV over batch and frequencies.
    """
    if cores.shape[1] <= 1:
        return 0.0
    d_omega = np.diff(omega).reshape(1, -1, 1)
    diff = (cores[:, 1:, :] - cores[:, :-1, :]) / d_omega
    return float(np.mean(diff ** 2))


def reconstruction_loss_batch(
    phi: torch.Tensor,                    # (P, R)
    cores_re: torch.Tensor,               # (B, M, R)
    cores_im: torch.Tensor,               # (B, M, R)
    y_obs_re: Sequence[torch.Tensor],     # each (B, n_obs)
    y_obs_im: Sequence[torch.Tensor],     # each (B, n_obs)
    obs_idx_re: Sequence[torch.Tensor],   # each (n_obs,)
    obs_idx_im: Sequence[torch.Tensor],
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Observed-point MSE over one sample minibatch."""
    core_re_b = cores_re.index_select(0, batch_idx)
    core_im_b = cores_im.index_select(0, batch_idx)

    loss_sum = torch.zeros((), dtype=phi.dtype, device=phi.device)
    count = 0

    M = core_re_b.shape[1]
    for m in range(M):
        A_re = phi.index_select(0, obs_idx_re[m])
        pred_re = torch.matmul(core_re_b[:, m, :], A_re.t())
        tgt_re = y_obs_re[m].index_select(0, batch_idx)
        # diff_re = pred_re - tgt_re
        rel_loss = torch.sum((pred_re - tgt_re) ** 2) / torch.sum(tgt_re ** 2)
        # loss_sum = loss_sum + torch.sum(diff_re * diff_re)
        diff_re = torch.sqrt(rel_loss)
        loss_sum = loss_sum + diff_re
        count += diff_re.numel()

        A_im = phi.index_select(0, obs_idx_im[m])
        pred_im = torch.matmul(core_im_b[:, m, :], A_im.t())
        tgt_im = y_obs_im[m].index_select(0, batch_idx)
        # diff_im = pred_im - tgt_im
        rel_loss = torch.sum((pred_im - tgt_im) ** 2) / torch.sum(tgt_im ** 2)
        # loss_sum = loss_sum + torch.sum(diff_im * diff_im)
        diff_im = torch.sqrt(rel_loss)
        loss_sum = loss_sum + diff_im
        count += diff_im.numel()

    return loss_sum / max(count, 1)


def reconstruction_loss_batch_per_sample(
    phi: torch.Tensor,                                      # (P, R)
    cores_re: torch.Tensor,                                 # (B, M, R)
    cores_im: torch.Tensor,                                 # (B, M, R)
    y_obs_re: Sequence[Sequence[torch.Tensor]],             # [B][M], each (n_obs,)
    y_obs_im: Sequence[Sequence[torch.Tensor]],             # [B][M], each (n_obs,)
    obs_idx_re: Sequence[Sequence[torch.Tensor]],           # [B][M], each (n_obs,)
    obs_idx_im: Sequence[Sequence[torch.Tensor]],           # [B][M], each (n_obs,)
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Observed-point relative loss over one sample minibatch for per-sample masks."""
    loss_sum = torch.zeros((), dtype=phi.dtype, device=phi.device)
    count = 0

    for b in batch_idx.tolist():
        core_re_b = cores_re[b]  # (M, R)
        core_im_b = cores_im[b]  # (M, R)
        M = core_re_b.shape[0]

        for m in range(M):
            # Real channel
            A_re = phi.index_select(0, obs_idx_re[b][m])          # (n_obs, R)
            pred_re = torch.matmul(A_re, core_re_b[m])            # (n_obs,)
            tgt_re = y_obs_re[b][m]                               # (n_obs,)
            rel_loss_re = torch.sum((pred_re - tgt_re) ** 2) / (torch.sum(tgt_re ** 2) + 1e-12)
            loss_sum = loss_sum + torch.sqrt(rel_loss_re + 1e-12)
            count += 1

            # Imag channel
            A_im = phi.index_select(0, obs_idx_im[b][m])
            pred_im = torch.matmul(A_im, core_im_b[m])
            tgt_im = y_obs_im[b][m]
            rel_loss_im = torch.sum((pred_im - tgt_im) ** 2) / (torch.sum(tgt_im ** 2) + 1e-12)
            loss_sum = loss_sum + torch.sqrt(rel_loss_im + 1e-12)
            count += 1

    return loss_sum / max(count, 1)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ftm_data = load_ftm_data(
        h5_path=args.data_h5,
        metadata_npy=args.metadata_npy,
        max_samples=args.max_samples,
    )

    data = ftm_data.data
    mask = ftm_data.mask
    omega = ftm_data.omega
    grid_x = ftm_data.grid_x
    grid_y = ftm_data.grid_y

    B, M, H, W, C = data.shape
    if C != 2:
        raise ValueError("Only 2-channel complex data is supported.")

    if args.normalize_coords:
        x_coords_np = normalize_coords_to_unit(grid_x)
        y_coords_np = normalize_coords_to_unit(grid_y)
    else:
        x_coords_np = grid_x.astype(np.float64)
        y_coords_np = grid_y.astype(np.float64)

    per_sample_mask = (mask.ndim == 5)
    if per_sample_mask:
        obs_idx_by_channel, y_obs_by_channel = build_observation_views_per_sample(data, mask)
    else:
        obs_idx_by_channel, y_obs_by_channel = build_observation_views(data, mask)

    net_x = MLP1D(
        out_dim=args.rank_x,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.hidden_layers,
        activation=args.activation,
    ).to(device)
    net_y = MLP1D(
        out_dim=args.rank_y,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.hidden_layers,
        activation=args.activation,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(net_x.parameters()) + list(net_y.parameters()),
        lr=args.lr,
    )

    x_t = torch.from_numpy(x_coords_np.astype(np.float32)).unsqueeze(-1).to(device)
    y_t = torch.from_numpy(y_coords_np.astype(np.float32)).unsqueeze(-1).to(device)

    # Static observation tensors used in NN update.
    if per_sample_mask:
        obs_idx_re_t = [
            [torch.from_numpy(obs_idx_by_channel[0][b][m]).long().to(device) for m in range(M)]
            for b in range(B)
        ]
        obs_idx_im_t = [
            [torch.from_numpy(obs_idx_by_channel[1][b][m]).long().to(device) for m in range(M)]
            for b in range(B)
        ]
        y_obs_re_t = [
            [torch.from_numpy(y_obs_by_channel[0][b][m].astype(np.float32)).to(device) for m in range(M)]
            for b in range(B)
        ]
        y_obs_im_t = [
            [torch.from_numpy(y_obs_by_channel[1][b][m].astype(np.float32)).to(device) for m in range(M)]
            for b in range(B)
        ]
    else:
        obs_idx_re_t = [torch.from_numpy(idx).long().to(device) for idx in obs_idx_by_channel[0]]
        obs_idx_im_t = [torch.from_numpy(idx).long().to(device) for idx in obs_idx_by_channel[1]]
        y_obs_re_t = [torch.from_numpy(y.astype(np.float32)).to(device) for y in y_obs_by_channel[0]]
        y_obs_im_t = [torch.from_numpy(y.astype(np.float32)).to(device) for y in y_obs_by_channel[1]]

    R = args.rank_x * args.rank_y

    cores_re_np = np.zeros((B, M, R), dtype=np.float64)
    cores_im_np = np.zeros((B, M, R), dtype=np.float64)

    best_loss = float("inf")
    stale_rounds = 0

    print("\n" + "-" * 72)
    print("FTM Training (Frequency-axis, Alternating Optimization)")
    print(f"data={args.data_h5}")
    print(f"shape: B={B}, M={M}, H={H}, W={W}, C={C}")
    print(f"rank: Rx={args.rank_x}, Ry={args.rank_y}, R={R}")
    print(f"mask mode: {'per-sample' if per_sample_mask else 'shared'}")
    print(f"device={device}, outer_iters={args.outer_iters}, nn_steps={args.nn_steps}")
    print("-" * 72 + "\n")

    for outer in range(1, args.outer_iters + 1):
        # -------------------------------
        # Step A: solve cores (closed-form quadratic solve)
        # -------------------------------
        net_x.eval()
        net_y.eval()
        with torch.no_grad():
            phi = build_phi(net_x, net_y, x_t, y_t)
        phi_np = phi.detach().cpu().numpy().astype(np.float64)

        if per_sample_mask:
            cores_re_np = solve_cores_for_channel_per_sample(
                phi_np=phi_np,
                obs_idx_by_bm=obs_idx_by_channel[0],
                y_obs_by_bm=y_obs_by_channel[0],
                omega=omega,
                beta=args.beta,
                ridge=args.ridge,
            )
            cores_im_np = solve_cores_for_channel_per_sample(
                phi_np=phi_np,
                obs_idx_by_bm=obs_idx_by_channel[1],
                y_obs_by_bm=y_obs_by_channel[1],
                omega=omega,
                beta=args.beta,
                ridge=args.ridge,
            )
        else:
            cores_re_np = solve_cores_for_channel(
                phi_np=phi_np,
                obs_idx_by_m=obs_idx_by_channel[0],
                y_obs_by_m=y_obs_by_channel[0],
                omega=omega,
                beta=args.beta,
                ridge=args.ridge,
            )
            cores_im_np = solve_cores_for_channel(
                phi_np=phi_np,
                obs_idx_by_m=obs_idx_by_channel[1],
                y_obs_by_m=y_obs_by_channel[1],
                omega=omega,
                beta=args.beta,
                ridge=args.ridge,
            )

        cores_re_t = torch.from_numpy(cores_re_np.astype(np.float32)).to(device)
        cores_im_t = torch.from_numpy(cores_im_np.astype(np.float32)).to(device)

        # -------------------------------
        # Step B: update latent functions with fixed cores
        # -------------------------------
        net_x.train()
        net_y.train()

        running_loss = 0.0
        for step in range(1, args.nn_steps + 1):
            if args.batch_size >= B:
                batch_idx = torch.arange(B, device=device)
            else:
                perm = torch.randperm(B, device=device)
                batch_idx = perm[: args.batch_size]

            optimizer.zero_grad(set_to_none=True)

            phi_train = build_phi(net_x, net_y, x_t, y_t)
            if per_sample_mask:
                loss = reconstruction_loss_batch_per_sample(
                    phi=phi_train,
                    cores_re=cores_re_t,
                    cores_im=cores_im_t,
                    y_obs_re=y_obs_re_t,
                    y_obs_im=y_obs_im_t,
                    obs_idx_re=obs_idx_re_t,
                    obs_idx_im=obs_idx_im_t,
                    batch_idx=batch_idx,
                )
            else:
                loss = reconstruction_loss_batch(
                    phi=phi_train,
                    cores_re=cores_re_t,
                    cores_im=cores_im_t,
                    y_obs_re=y_obs_re_t,
                    y_obs_im=y_obs_im_t,
                    obs_idx_re=obs_idx_re_t,
                    obs_idx_im=obs_idx_im_t,
                    batch_idx=batch_idx,
                )
            loss.backward()

            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    list(net_x.parameters()) + list(net_y.parameters()),
                    max_norm=args.grad_clip,
                )

            optimizer.step()
            running_loss += float(loss.item())

            if args.log_every > 0 and step % args.log_every == 0:
                avg_so_far = running_loss / step
                print(f"[outer {outer:03d}] nn_step {step:04d}/{args.nn_steps}  loss={avg_so_far:.6e}")

        recon_loss = running_loss / max(args.nn_steps, 1)

        tv_re = weighted_tv_value(cores_re_np, omega)
        tv_im = weighted_tv_value(cores_im_np, omega)
        full_obj = recon_loss + args.beta * (tv_re + tv_im)

        print(
            f"[outer {outer:03d}] recon={recon_loss:.6e} "
            f"tv_re={tv_re:.6e} tv_im={tv_im:.6e} obj={full_obj:.6e}"
        )

        rel_improve = (best_loss - full_obj) / max(abs(best_loss), 1e-12)
        if full_obj < best_loss:
            best_loss = full_obj
            stale_rounds = 0
        else:
            stale_rounds += 1

        if rel_improve >= 0 and rel_improve < args.tol:
            stale_rounds += 1

        if args.patience > 0 and stale_rounds >= args.patience:
            print(
                f"Early stop at outer={outer}: no significant improvement for {stale_rounds} rounds."
            )
            break

    # Recompute cores once with the final latent functions for checkpoint consistency.
    net_x.eval()
    net_y.eval()
    with torch.no_grad():
        phi = build_phi(net_x, net_y, x_t, y_t)
    phi_np = phi.detach().cpu().numpy().astype(np.float64)

    if per_sample_mask:
        cores_re_np = solve_cores_for_channel_per_sample(
            phi_np=phi_np,
            obs_idx_by_bm=obs_idx_by_channel[0],
            y_obs_by_bm=y_obs_by_channel[0],
            omega=omega,
            beta=args.beta,
            ridge=args.ridge,
        )
        cores_im_np = solve_cores_for_channel_per_sample(
            phi_np=phi_np,
            obs_idx_by_bm=obs_idx_by_channel[1],
            y_obs_by_bm=y_obs_by_channel[1],
            omega=omega,
            beta=args.beta,
            ridge=args.ridge,
        )
    else:
        cores_re_np = solve_cores_for_channel(
            phi_np=phi_np,
            obs_idx_by_m=obs_idx_by_channel[0],
            y_obs_by_m=y_obs_by_channel[0],
            omega=omega,
            beta=args.beta,
            ridge=args.ridge,
        )
        cores_im_np = solve_cores_for_channel(
            phi_np=phi_np,
            obs_idx_by_m=obs_idx_by_channel[1],
            y_obs_by_m=y_obs_by_channel[1],
            omega=omega,
            beta=args.beta,
            ridge=args.ridge,
        )

    cores_re = cores_re_np.reshape(B, M, args.rank_x, args.rank_y).astype(np.float32)
    cores_im = cores_im_np.reshape(B, M, args.rank_x, args.rank_y).astype(np.float32)

    ckpt = {
        "net_x_state": net_x.state_dict(),
        "net_y_state": net_y.state_dict(),
        "cores_real": torch.from_numpy(cores_re),
        "cores_imag": torch.from_numpy(cores_im),
        "omega": torch.from_numpy(omega.astype(np.float32)),
        "grid_x": torch.from_numpy(grid_x.astype(np.float32)),
        "grid_y": torch.from_numpy(grid_y.astype(np.float32)),
        "data_scale": float(ftm_data.data_scale),
        "config": vars(args),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)

    summary = {
        "out": str(out_path),
        "B": int(B),
        "M": int(M),
        "H": int(H),
        "W": int(W),
        "rank_x": int(args.rank_x),
        "rank_y": int(args.rank_y),
        "data_scale": float(ftm_data.data_scale),
        "beta": float(args.beta),
        "ridge": float(args.ridge),
    }
    print("\nSaved checkpoint:", out_path)
    print("Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train frequency-axis Functional Tucker Model (FTM)")

    p.add_argument("--data_h5", type=str, default="new_idea/helmholtz_dataset_42_new_idea_mask10.h5")
    p.add_argument("--metadata_npy", type=str, default="")
    p.add_argument("--out", type=str, default="new_idea/ckp/ftm_checkpoint.pt")

    p.add_argument("--rank_x", type=int, default=18)
    p.add_argument("--rank_y", type=int, default=18)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--hidden_layers", type=int, default=4)
    p.add_argument("--activation", type=str, default="sine", choices=["sine", "relu", "tanh"])

    p.add_argument("--beta", type=float, default=1e-3)
    p.add_argument("--ridge", type=float, default=1e-6)

    p.add_argument("--outer_iters", type=int, default=240)
    p.add_argument("--nn_steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=0.0)

    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--patience", type=int, default=5)

    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--normalize_coords", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--log_every", type=int, default=5)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
