"""
train_FTM_GPU.py
----------------
Train a frequency-axis Functional Tucker Model (FTM) for 2D complex fields.

Compared with the original alternating strategy, this GPU-oriented script uses
single-stage joint optimization:
1) Core tensor sequences (real/imag) are trainable parameters.
2) Latent basis functions (x/y MLPs) are trained jointly with cores.
3) No closed-form linear solve and no outer/inner alternating loop.

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
from typing import List

import h5py
import numpy as np
import torch
import torch.nn as nn


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

# def core_matrix_smooth_loss(core: torch.Tensor, Rx: int, Ry: int) -> torch.Tensor:
#     # core: (B, M, Rx*Ry)
#     B, M, _ = core.shape
#     core_mat = core.view(B, M, Rx, Ry)

#     # 行平滑 (Rx 方向)
#     dx = torch.abs(core_mat[:, :, 1:, :] - core_mat[:, :, :-1, :])
#     # 列平滑 (Ry 方向)
#     dy = torch.abs(core_mat[:, :, :, 1:] - core_mat[:, :, :, :-1])

#     loss = torch.mean(dx**2) + torch.mean(dy**2)
#     return loss

def core_matrix_smooth_loss(
    core: torch.Tensor, 
    omega: torch.Tensor,  # (M,) 频率
    Rx: int, 
    Ry: int
) -> torch.Tensor:
    # core: (B, M, Rx*Ry)
    B, M, _ = core.shape
    core_mat = core.view(B, M, Rx, Ry)  # (B, M, Rx, Ry)

    # 行平滑 (Rx 方向)
    dx = core_mat[:, :, 1:, :] - core_mat[:, :, :-1, :]  # (B, M, Rx-1, Ry)
    # 列平滑 (Ry 方向)
    dy = core_mat[:, :, :, 1:] - core_mat[:, :, :, :-1]  # (B, M, Rx, Ry-1)

    # --------------------- 频率加权 ---------------------
    omega_norm = (omega - omega.min()) / (omega.max() - omega.min() + 1e-8)
    freq_weights = 1.0 + 14.0 * omega_norm  # [1, 15]
    freq_weights = freq_weights.view(1, M, 1, 1)  # (1, M, 1, 1)
    # ----------------------------------------------------

    # 分别计算加权损失，再相加（修复维度不匹配）
    loss_dx = torch.mean(freq_weights * (dx ** 2))
    loss_dy = torch.mean(freq_weights * (dy ** 2))
    loss = loss_dx + loss_dy

    return loss
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


def weighted_tv_loss_torch(cores: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """
    Weighted frequency-axis TV in squared form.

    Parameters
    ----------
    cores : (B, M, R)
    omega : (M,)
    """
    if cores.shape[1] <= 1:
        return torch.zeros((), dtype=cores.dtype, device=cores.device)

    d_omega = omega[1:] - omega[:-1]
    if torch.any(d_omega <= 0):
        raise ValueError("omega must be strictly increasing for TV regularization.")

    diff = (cores[:, 1:, :] - cores[:, :-1, :]) / d_omega.view(1, -1, 1)
    return torch.mean(diff * diff)


def _build_channel_views(
    data: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    data_re, data_im : (B, M, P)
    mask_re, mask_im : shared (M, P) or per-sample (B, M, P)
    """
    B, M, H, W, C = data.shape
    if C != 2:
        raise ValueError("Only 2-channel complex data is supported.")

    P = H * W
    data_re = data[..., 0].reshape(B, M, P)
    data_im = data[..., 1].reshape(B, M, P)

    if mask.ndim == 4:
        mask_re = (mask[..., 0].reshape(M, P) > 0.5).astype(np.float32)
        mask_im = (mask[..., 1].reshape(M, P) > 0.5).astype(np.float32)
    else:
        mask_re = (mask[..., 0].reshape(B, M, P) > 0.5).astype(np.float32)
        mask_im = (mask[..., 1].reshape(B, M, P) > 0.5).astype(np.float32)

    return data_re, data_im, mask_re, mask_im


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

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

    data_re, data_im, mask_re_np, mask_im_np = _build_channel_views(data, mask)
    per_sample_mask = mask.ndim == 5

    if per_sample_mask:
        obs_re_count = np.sum(mask_re_np, axis=2)
        obs_im_count = np.sum(mask_im_np, axis=2)

        bad_re = np.argwhere(obs_re_count <= 0)
        bad_im = np.argwhere(obs_im_count <= 0)
        if bad_re.size > 0:
            b0, m0 = bad_re[0]
            raise ValueError(
                f"No observed points in real channel for sample={int(b0)}, freq_index={int(m0)}."
            )
        if bad_im.size > 0:
            b0, m0 = bad_im[0]
            raise ValueError(
                f"No observed points in imag channel for sample={int(b0)}, freq_index={int(m0)}."
            )
    else:
        obs_re_count = np.sum(mask_re_np, axis=1)
        obs_im_count = np.sum(mask_im_np, axis=1)

        bad_re = np.argwhere(obs_re_count <= 0)
        bad_im = np.argwhere(obs_im_count <= 0)
        if bad_re.size > 0:
            m0 = bad_re[0, 0]
            raise ValueError(f"No observed points in real channel for freq_index={int(m0)}.")
        if bad_im.size > 0:
            m0 = bad_im[0, 0]
            raise ValueError(f"No observed points in imag channel for freq_index={int(m0)}.")

    if per_sample_mask:
        denom_re_np = np.sum((data_re ** 2) * mask_re_np, axis=2)
        denom_im_np = np.sum((data_im ** 2) * mask_im_np, axis=2)
    else:
        denom_re_np = np.sum((data_re ** 2) * mask_re_np[None, :, :], axis=2)
        denom_im_np = np.sum((data_im ** 2) * mask_im_np[None, :, :], axis=2)

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

    x_t = torch.from_numpy(x_coords_np.astype(np.float32)).unsqueeze(-1).to(device)
    y_t = torch.from_numpy(y_coords_np.astype(np.float32)).unsqueeze(-1).to(device)
    omega_t = torch.from_numpy(omega.astype(np.float32)).to(device)

    R = args.rank_x * args.rank_y
    core_init = float(args.core_init_scale)
    cores_re_param = nn.Parameter(core_init * torch.randn(B, M, R, device=device))
    cores_im_param = nn.Parameter(core_init * torch.randn(B, M, R, device=device))

    net_params = list(net_x.parameters()) + list(net_y.parameters())
    optimizer = torch.optim.Adam(
        [
            {"params": net_params, "lr": args.lr},
            {"params": [cores_re_param, cores_im_param], "lr": args.core_lr},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.99, eps=1e-12, min_lr=1e-7)
    if not per_sample_mask:
        mask_re_shared_t = torch.from_numpy(mask_re_np).to(device)
        mask_im_shared_t = torch.from_numpy(mask_im_np).to(device)
    else:
        mask_re_shared_t = None
        mask_im_shared_t = None

    best_obj = float("inf")
    stale_rounds = 0

    print("\n" + "-" * 72)
    print("FTM Training (Frequency-axis, Joint Optimization)")
    print(f"data={args.data_h5}")
    print(f"shape: B={B}, M={M}, H={H}, W={W}, C={C}")
    print(f"rank: Rx={args.rank_x}, Ry={args.rank_y}, R={R}")
    print(f"mask mode: {'per-sample' if per_sample_mask else 'shared'}")
    print(f"device={device}, iters={args.iters}, batch_size={args.batch_size}")
    print(
        f"lr_basis={args.lr:.2e}, lr_core={args.core_lr:.2e}, "
        f"beta={args.beta:.2e}, core_l2={args.core_l2:.2e}"
    )
    print("-" * 72 + "\n")

    for step in range(1, args.iters + 1):
        if args.batch_size >= B:
            batch_idx_np = np.arange(B, dtype=np.int64)
        else:
            batch_idx_np = np.random.permutation(B)[: args.batch_size].astype(np.int64)

        batch_idx_t = torch.from_numpy(batch_idx_np).long().to(device)
        bsz = batch_idx_np.shape[0]

        gt_re_t = torch.from_numpy(data_re[batch_idx_np]).to(device)
        gt_im_t = torch.from_numpy(data_im[batch_idx_np]).to(device)

        denom_re_t = torch.from_numpy(denom_re_np[batch_idx_np]).to(device)
        denom_im_t = torch.from_numpy(denom_im_np[batch_idx_np]).to(device)

        if per_sample_mask:
            mask_re_t = torch.from_numpy(mask_re_np[batch_idx_np]).to(device)
            mask_im_t = torch.from_numpy(mask_im_np[batch_idx_np]).to(device)
        else:
            mask_re_t = mask_re_shared_t.unsqueeze(0).expand(bsz, -1, -1)
            mask_im_t = mask_im_shared_t.unsqueeze(0).expand(bsz, -1, -1)

        optimizer.zero_grad(set_to_none=True)

        phi = build_phi(net_x, net_y, x_t, y_t)  # (P, R)

        core_re_b = cores_re_param.index_select(0, batch_idx_t)  # (bsz, M, R)
        core_im_b = cores_im_param.index_select(0, batch_idx_t)

        pred_re = torch.einsum("bmr,pr->bmp", core_re_b, phi)
        pred_im = torch.einsum("bmr,pr->bmp", core_im_b, phi)

        sq_err_re = (pred_re - gt_re_t) ** 2
        sq_err_im = (pred_im - gt_im_t) ** 2

        rel_re = torch.sqrt(
            torch.sum(sq_err_re * mask_re_t, dim=2) / (denom_re_t + args.eps) + args.eps
        )
        rel_im = torch.sqrt(
            torch.sum(sq_err_im * mask_im_t, dim=2) / (denom_im_t + args.eps) + args.eps
        )

        recon_loss = 0.5 * (torch.mean(rel_re) + torch.mean(rel_im))

        tv_re = weighted_tv_loss_torch(cores_re_param, omega_t)
        tv_im = weighted_tv_loss_torch(cores_im_param, omega_t)
        l2_reg = torch.mean(cores_re_param ** 2) + torch.mean(cores_im_param ** 2)

        core_smooth_re = core_matrix_smooth_loss(cores_re_param, omega_t, args.rank_x, args.rank_y)  
        core_smooth_im = core_matrix_smooth_loss(cores_im_param, omega_t, args.rank_x, args.rank_y)
        # obj = recon_loss + args.beta * (tv_re + tv_im) + args.core_l2 * l2_reg + args.core_smooth * (core_smooth_re + core_smooth_im)
        obj =recon_loss + args.core_smooth * (core_smooth_re + core_smooth_im)
        obj.backward()

        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(net_params, max_norm=args.grad_clip)
        if args.core_grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_([cores_re_param, cores_im_param], max_norm=args.core_grad_clip)

        optimizer.step()
        scheduler.step(obj)

        obj_value = float(obj.item())
        if obj_value < best_obj:
            rel_improve = (best_obj - obj_value) / max(abs(best_obj), 1e-12)
            best_obj = obj_value
            if rel_improve < args.tol:
                stale_rounds += 1
            else:
                stale_rounds = 0
        else:
            stale_rounds += 1

        if args.log_every > 0 and step % args.log_every == 0:
            print(
                f"[step {step:05d}/{args.iters}] "
                f"recon={float(recon_loss.item()):.6e} "
                f"tv={float((tv_re + tv_im).item()):.6e} "
                f"l2={float(l2_reg.item()):.6e} "
                f"core_smooth={float((core_smooth_re + core_smooth_im).item()):.6e} "
                f"obj={obj_value:.6e} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if args.patience > 0 and stale_rounds >= args.patience:
            print(
                f"Early stop at step={step}: no significant improvement for {stale_rounds} rounds."
            )
            break

    cores_re = (
        cores_re_param.detach()
        .cpu()
        .numpy()
        .reshape(B, M, args.rank_x, args.rank_y)
        .astype(np.float32)
    )
    cores_im = (
        cores_im_param.detach()
        .cpu()
        .numpy()
        .reshape(B, M, args.rank_x, args.rank_y)
        .astype(np.float32)
    )

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
        "core_l2": float(args.core_l2),
        "core_smooth": float(args.core_smooth),
    }
    print("\nSaved checkpoint:", out_path)
    print("Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train frequency-axis FTM with joint optimization")

    p.add_argument("--data_h5", type=str, default="new_idea/helmholtz_dataset_42_new_idea_mask10.h5")
    p.add_argument("--metadata_npy", type=str, default="")
    p.add_argument("--out", type=str, default="new_idea/ckp/ftm_gpu_checkpoint.pt")

    p.add_argument("--rank_x", type=int, default=24)
    p.add_argument("--rank_y", type=int, default=24)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--hidden_layers", type=int, default=4)
    p.add_argument("--activation", type=str, default="sine", choices=["sine", "relu", "tanh"])

    p.add_argument("--iters", type=int, default=25000)
    p.add_argument("--batch_size", type=int, default=64)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--core_lr", type=float, default=1e-4)
    p.add_argument("--core_init_scale", type=float, default=1e-3)

    p.add_argument("--beta", type=float, default=1e4)
    p.add_argument("--core_l2", type=float, default=5e2)
    p.add_argument("--core_smooth", type=float, default=1e5)

    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--core_grad_clip", type=float, default=0.0)

    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--eps", type=float, default=1e-12)

    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--normalize_coords", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--log_every", type=int, default=10)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
