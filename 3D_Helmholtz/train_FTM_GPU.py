"""
train_FTM_GPU.py  (3D Helmholtz edition)
-----------------------------------------
Train a 3-D Functional Tucker Model for C=2 complex fields (real, imag).

Field family: u_b(x,y,z,ω) ≈ Σ_{r,q,s} G^b_ω[r,q,s] · f_x[r](x)·f_y[q](y)·f_z[s](z)

Three shared coordinate-MLP basis networks: net_x, net_y, net_z.
Per-(sample,freq) Tucker core G: shape (Rx, Ry, Rz), flattened to R = Rx*Ry*Rz for training.
Real and imaginary channels get independent cores but share the spatial basis.

HDF5 expected
-------------
- data    : (B, M, N, N, N, C=2)   float32
- mask_tr : (M, N, N, N, C) or (B, M, N, N, N, C)  uint8/float32
- omega   : (M,)
- grid_x/y/z : (N,)   (optional, default linspace[0,1])

Checkpoint output
-----------------
- net_x_state / net_y_state / net_z_state
- cores  : list[C] of tensors (B, M, Rx, Ry, Rz)
- channel_names, omega, grid_x/y/z, data_scale, config

Usage
-----
    python train_FTM_GPU.py --data_h5 helmholtz3d_dataset.h5 --out ckp/ftm3d.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Utilities (shared with 2D version)
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class Sine(nn.Module):
    def forward(self, x): return torch.sin(x)


class MLP1D(nn.Module):
    """1-D coordinate → basis feature network (identical to 2D version)."""
    def __init__(self, out_dim, hidden_dim=256, num_hidden_layers=4, activation="sine"):
        super().__init__()
        act = {"sine": Sine, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        layers, in_dim = [], 1
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), act()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.4, 0.4) if activation == "sine" \
                    else nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x): return self.net(x)


def normalize_coords_to_unit(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if math.isclose(hi, lo) else (x - lo) / (hi - lo)


# ─────────────────────────────────────────────────────────────────────────────
# 3D Kronecker feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_phi_3d(
    net_x: nn.Module, net_y: nn.Module, net_z: nn.Module,
    x_t: torch.Tensor,   # (Nx, 1)
    y_t: torch.Tensor,   # (Ny, 1)
    z_t: torch.Tensor,   # (Nz, 1)
) -> torch.Tensor:
    """
    Kronecker feature matrix Φ(x,y,z), shape (Nx*Ny*Nz, Rx*Ry*Rz).

    Φ[ix*Ny*Nz + iy*Nz + iz, rx*Ry*Rz + ry*Rz + rz] =
        f_x[rx](x_ix) · f_y[ry](y_iy) · f_z[rz](z_iz)
    """
    fx = net_x(x_t)   # (Nx, Rx)
    fy = net_y(y_t)   # (Ny, Ry)
    fz = net_z(z_t)   # (Nz, Rz)
    # Build full 3D product
    # (Nx, Ny, Nz, Rx, Ry, Rz)
    fxyz = torch.einsum("ir,jq,ks->ijkrqs", fx, fy, fz)
    Nx, Ny, Nz = fxyz.shape[:3]
    Rx, Ry, Rz = fxyz.shape[3:]
    return fxyz.reshape(Nx * Ny * Nz, Rx * Ry * Rz)


# ─────────────────────────────────────────────────────────────────────────────
# Regularisation losses
# ─────────────────────────────────────────────────────────────────────────────

def weighted_tv_loss(cores: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Frequency-axis weighted TV², cores: (B, M, R)."""
    if cores.shape[1] <= 1:
        return cores.new_zeros(())
    d_omega = omega[1:] - omega[:-1]
    diff = (cores[:, 1:] - cores[:, :-1]) / d_omega.view(1, -1, 1)
    return torch.mean(diff * diff)


def core_smooth_loss_3d(
    cores: torch.Tensor,   # (B, M, Rx*Ry*Rz)
    omega: torch.Tensor,   # (M,)
    Rx: int, Ry: int, Rz: int,
) -> torch.Tensor:
    """Spatial smoothness of 3D core tensors, frequency-weighted."""
    B, M, _ = cores.shape
    c = cores.view(B, M, Rx, Ry, Rz)
    dx = c[:, :, 1:] - c[:, :, :-1]       # (B,M,Rx-1,Ry,Rz)
    dy = c[:, :, :, 1:] - c[:, :, :, :-1]
    dz = c[:, :, :, :, 1:] - c[:, :, :, :, :-1]
    w = (1.0 + 14.0 * (omega - omega.min()) / (omega.max() - omega.min() + 1e-8))
    w = w.view(1, M, 1, 1, 1)
    return torch.mean(w * dx**2) + torch.mean(w * dy**2) + torch.mean(w * dz**2)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading for 3D
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FTMData3D:
    data: np.ndarray           # (B, M, Nx, Ny, Nz, C)
    mask: np.ndarray           # (M, Nx, Ny, Nz, C) or (B, M, Nx, Ny, Nz, C)
    omega: np.ndarray          # (M,)
    grid_x: np.ndarray         # (Nx,)
    grid_y: np.ndarray         # (Ny,)
    grid_z: np.ndarray         # (Nz,)
    data_scale: float
    channel_names: List[str] = field(default_factory=list)


def load_ftm_data_3d(h5_path: str, metadata_npy: str = "",
                     max_samples: int = 0) -> FTMData3D:
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    with h5py.File(path, "r") as f:
        for k in ("data", "mask_tr"):
            if k not in f: raise KeyError(f"HDF5 missing '{k}'.")

        data  = f["data"][...].astype(np.float32)
        mask  = f["mask_tr"][...].astype(np.float32)
        omega = f["omega"][...].astype(np.float64)
        N     = data.shape[2]
        def _grid(key, fallback):
            return f[key][...].astype(np.float64) if key in f else fallback
        grid_x = _grid("grid_x", np.linspace(0, 1, N))
        grid_y = _grid("grid_y", np.linspace(0, 1, N))
        grid_z = _grid("grid_z", np.linspace(0, 1, N))
        channel_names: List[str] = []
        if "metadata" in f:
            try:
                ms = f["metadata"][()]
                ms = ms.decode() if isinstance(ms, bytes) else str(ms)
                channel_names = json.loads(ms).get("channels", [])
            except Exception:
                pass

    if data.ndim != 6:
        raise ValueError(f"Expected (B,M,Nx,Ny,Nz,C), got {data.shape}")
    C = data.shape[-1]
    if not channel_names:
        channel_names = ["real", "imag"] if C == 2 else [f"ch{c}" for c in range(C)]

    if max_samples > 0:
        data = data[:max_samples]
    B = data.shape[0]

    # Validate mask
    if mask.ndim == 5:    # shared (M, Nx, Ny, Nz, C)
        if tuple(mask.shape) != tuple(data.shape[1:]):
            raise ValueError(f"Mask {mask.shape} incompatible with data {data.shape}")
    elif mask.ndim == 6:  # per-sample (B, M, Nx, Ny, Nz, C)
        mask = mask[:B]
        if tuple(mask.shape[1:]) != tuple(data.shape[1:]):
            raise ValueError(f"Mask {mask.shape} incompatible with data {data.shape}")
    else:
        raise ValueError(f"mask_tr must have 5 or 6 dims, got {mask.ndim}")

    # data_scale from sidecar
    data_scale = 1.0
    meta_path = Path(metadata_npy) if metadata_npy else \
        path.with_name(f"{path.stem}_metadata.npy")
    if meta_path.exists():
        try:
            obj = np.load(meta_path, allow_pickle=True).item()
            data_scale = float(obj.get("data", {}).get("data_scale", 1.0))
        except Exception:
            pass

    return FTMData3D(data=data, mask=mask, omega=omega,
                     grid_x=grid_x, grid_y=grid_y, grid_z=grid_z,
                     data_scale=data_scale, channel_names=channel_names)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device)

    fd = load_ftm_data_3d(args.data_h5, args.metadata_npy, args.max_samples)
    data, mask, omega = fd.data, fd.mask, fd.omega
    grid_x, grid_y, grid_z = fd.grid_x, fd.grid_y, fd.grid_z
    channel_names = fd.channel_names

    B, M, Nx, Ny, Nz, C = data.shape
    per_sample_mask = mask.ndim == 6
    R = args.rank_x * args.rank_y * args.rank_z
    P = Nx * Ny * Nz

    # Coordinate tensors
    def _coord(g):
        g2 = normalize_coords_to_unit(g) if args.normalize_coords else g.copy()
        return torch.from_numpy(g2.astype(np.float32)).unsqueeze(-1).to(device)

    x_t, y_t, z_t = _coord(grid_x), _coord(grid_y), _coord(grid_z)
    omega_t = torch.from_numpy(omega.astype(np.float32)).to(device)

    # Channel views: (B, M, P)
    data_ch = [data[..., c].reshape(B, M, P) for c in range(C)]
    if per_sample_mask:
        mask_ch = [(mask[..., c].reshape(B, M, P) > 0.5).astype(np.float32)
                   for c in range(C)]
    else:
        mask_ch = [(mask[..., c].reshape(M, P) > 0.5).astype(np.float32)
                   for c in range(C)]

    # Denominators for relative loss
    def _denom(dc, mc):
        return np.sum(dc**2 * mc, axis=2) if per_sample_mask \
            else np.sum(dc**2 * mc[None], axis=2)
    denom_list = [_denom(dc, mc) for dc, mc in zip(data_ch, mask_ch)]

    # Basis networks (one triplet, shared across channels)
    def _mlp(rank):
        return MLP1D(rank, args.hidden_dim, args.hidden_layers, args.activation).to(device)

    net_x = _mlp(args.rank_x)
    net_y = _mlp(args.rank_y)
    net_z = _mlp(args.rank_z)

    # Core parameters: (B, M, R) per channel
    cores_params = nn.ParameterList([
        nn.Parameter(args.core_init_scale * torch.randn(B, M, R, device=device))
        for _ in range(C)
    ])

    basis_params = list(net_x.parameters()) + list(net_y.parameters()) + list(net_z.parameters())
    core_list    = list(cores_params.parameters())

    optimizer = torch.optim.Adam([
        {"params": basis_params, "lr": args.lr},
        {"params": core_list,    "lr": args.core_lr},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=20, factor=0.99, eps=1e-12, min_lr=4e-8)

    mask_shared = [torch.from_numpy(mc).to(device) if not per_sample_mask else None
                   for mc in mask_ch]

    print(f"\n{'─'*72}")
    print(f"3D FTM Training  C={C}  grid={Nx}³  B={B}  M={M}")
    print(f"rank=({args.rank_x},{args.rank_y},{args.rank_z})  R={R}  P={P}")
    print(f"device={device}  iters={args.iters}  batch={args.batch_size}")
    print(f"{'─'*72}\n")

    best_obj = float("inf")
    stale = 0

    for step in range(1, args.iters + 1):
        if args.batch_size >= B:
            idx_np = np.arange(B, dtype=np.int64)
        else:
            idx_np = np.random.permutation(B)[:args.batch_size].astype(np.int64)

        idx_t = torch.from_numpy(idx_np).long().to(device)
        bsz   = len(idx_np)

        optimizer.zero_grad(set_to_none=True)
        phi = build_phi_3d(net_x, net_y, net_z, x_t, y_t, z_t)  # (P, R)

        total_recon = total_tv = total_smooth = phi.new_zeros(())

        for c in range(C):
            core_b = cores_params[c].index_select(0, idx_t)   # (bsz, M, R)
            pred   = torch.einsum("bmr,pr->bmp", core_b, phi)  # (bsz, M, P)
            gt     = torch.from_numpy(data_ch[c][idx_np]).to(device)

            mk = (torch.from_numpy(mask_ch[c][idx_np]).to(device)
                  if per_sample_mask
                  else mask_shared[c].unsqueeze(0).expand(bsz, -1, -1))
            denom = torch.from_numpy(denom_list[c][idx_np]).to(device)

            sq_err = (pred - gt)**2
            rel = torch.sqrt(torch.sum(sq_err * mk, 2) / (denom + args.eps) + args.eps)
            total_recon = total_recon + torch.mean(rel)
            total_tv    = total_tv    + weighted_tv_loss(cores_params[c], omega_t)
            total_smooth = total_smooth + core_smooth_loss_3d(
                cores_params[c], omega_t, args.rank_x, args.rank_y, args.rank_z)

        total_recon  = total_recon  / C
        total_tv     = total_tv     / C
        total_smooth = total_smooth / C
        l2_reg = sum(torch.mean(p**2) for p in cores_params) / C

        obj = (total_recon
               + args.beta       * total_tv
               + args.core_l2    * l2_reg
               + args.core_smooth* total_smooth)
        obj.backward()

        if args.grad_clip      > 0: nn.utils.clip_grad_norm_(basis_params, args.grad_clip)
        if args.core_grad_clip > 0: nn.utils.clip_grad_norm_(core_list,    args.core_grad_clip)

        optimizer.step()
        scheduler.step(obj)

        obj_val = float(obj.item())
        if obj_val < best_obj:
            rel_imp = (best_obj - obj_val) / max(abs(best_obj), 1e-12)
            best_obj = obj_val
            stale = 0 if rel_imp >= args.tol else stale + 1
        else:
            stale += 1

        if args.log_every > 0 and step % args.log_every == 0:
            print(f"[{step:05d}/{args.iters}] "
                  f"recon={float(total_recon):.4e}  "
                  f"smooth={float(total_smooth):.4e}  "
                  f"obj={obj_val:.4e}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if args.patience > 0 and stale >= args.patience:
            print(f"Early stop step={step}"); break

    # ── Save checkpoint ───────────────────────────────────────────────────
    cores_saved = [
        cores_params[c].detach().cpu()
                       .reshape(B, M, args.rank_x, args.rank_y, args.rank_z)
                       .numpy().astype(np.float32)
        for c in range(C)
    ]

    ckpt = {
        "channel_names": channel_names,
        "cores":   [torch.from_numpy(cr) for cr in cores_saved],
        "omega":   torch.from_numpy(omega.astype(np.float32)),
        "grid_x":  torch.from_numpy(grid_x.astype(np.float32)),
        "grid_y":  torch.from_numpy(grid_y.astype(np.float32)),
        "grid_z":  torch.from_numpy(grid_z.astype(np.float32)),
        "net_x_state": net_x.state_dict(),
        "net_y_state": net_y.state_dict(),
        "net_z_state": net_z.state_dict(),
        "data_scale":  float(fd.data_scale),
        "config":      vars(args),
        "spatial_dims": 3,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)

    summary = {"out": str(out_path), "B": B, "M": M, "N": Nx, "C": C,
               "channel_names": channel_names,
               "rank_x": args.rank_x, "rank_y": args.rank_y, "rank_z": args.rank_z,
               "R": R, "data_scale": float(fd.data_scale)}
    print("\nSaved:", out_path)
    print(json.dumps(summary, indent=2))


def build_parser():
    p = argparse.ArgumentParser(description="Train 3D FTM")
    p.add_argument("--data_h5",      type=str,   default="helmholtz3d_dataset.h5")
    p.add_argument("--metadata_npy", type=str,   default="")
    p.add_argument("--out",          type=str,   default="ckp/ftm3d.pt")
    p.add_argument("--rank_x",       type=int,   default=14)
    p.add_argument("--rank_y",       type=int,   default=14)
    p.add_argument("--rank_z",       type=int,   default=14)
    p.add_argument("--hidden_dim",   type=int,   default=512)
    p.add_argument("--hidden_layers",type=int,   default=5)
    p.add_argument("--activation",   type=str,   default="sine",
                   choices=["sine", "relu", "tanh"])
    p.add_argument("--iters",        type=int,   default=30000)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--core_lr",      type=float, default=1e-4)
    p.add_argument("--core_init_scale", type=float, default=1e-3)
    p.add_argument("--beta",         type=float, default=0.0)
    p.add_argument("--core_l2",      type=float, default=1e2)
    p.add_argument("--core_smooth",  type=float, default=1e4)
    p.add_argument("--grad_clip",    type=float, default=0.0)
    p.add_argument("--core_grad_clip",type=float,default=0.0)
    p.add_argument("--tol",          type=float, default=1e-6)
    p.add_argument("--patience",     type=int,   default=0)
    p.add_argument("--eps",          type=float, default=1e-12)
    p.add_argument("--max_samples",  type=int,   default=0)
    p.add_argument("--normalize_coords", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       type=str,   default="auto")
    p.add_argument("--log_every",    type=int,   default=100)
    return p

if __name__ == "__main__":
    train(build_parser().parse_args())
