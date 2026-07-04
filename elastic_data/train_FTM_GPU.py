"""
train_FTM_GPU.py  (multi-channel edition)
-----------------------------------------
Train a frequency-axis Functional Tucker Model (FTM) for 2D complex fields
with an arbitrary number of channels C.

Default usage (Helmholtz, C=2):
    python train_FTM_GPU.py --data_h5 helmholtz_dataset.h5

Elastic-wave usage (C=4: ux_re, ux_im, uy_re, uy_im):
    python train_FTM_GPU.py --data_h5 elastic_dataset.h5 --split_basis

Key design choices
------------------
* Every channel gets its own core sequence G_c ∈ R^{B × M × Rx × Ry}.
* The spatial basis functions (net_x / net_y) are *shared* across all channels
  by default, which keeps parameter count low and exploits shared spatial
  structure.  Pass --split_basis to give each physical field (ux, uy) its
  own basis pair.
* Loss, TV regularisation, and core-smoothness regularisation are computed
  per channel and averaged.

HDF5 key expected
-----------------
- data    : (B, M, H, W, C)  float32
- mask_tr : (M, H, W, C) or (B, M, H, W, C)  uint8 / float32
- omega   : (M,)
- grid_x  : (H,)   (optional)
- grid_y  : (W,)   (optional)

Output checkpoint (torch .pt)
-----------------------------
- net_x_state / net_y_state  — or net_x_state_list / net_y_state_list if split
- cores  : list of C tensors, each (B, M, Rx, Ry)
- channel_names : list[str]
- omega / grid_x / grid_y
- data_scale / config
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
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

CHANNEL_NAMES_2 = ["real", "imag"]
CHANNEL_NAMES_4 = ["ux_real", "ux_imag", "uy_real", "uy_imag"]


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
    """1-D coordinate → basis feature network."""

    def __init__(
        self,
        out_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 4,
        activation: str = "sine",
    ):
        super().__init__()
        act_map = {"sine": Sine, "relu": nn.ReLU, "tanh": nn.Tanh}
        if activation not in act_map:
            raise ValueError(f"activation must be one of {list(act_map)}")
        act = act_map[activation]

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
                    nn.init.uniform_(m.weight, -0.4, 0.4)
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
    channel_names: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _try_load_scale_from_sidecar(h5_path: Path, metadata_npy: str) -> float:
    meta_path = Path(metadata_npy) if metadata_npy else \
        h5_path.with_name(f"{h5_path.stem}_metadata.npy")
    if not meta_path.exists():
        return 1.0
    try:
        obj = np.load(meta_path, allow_pickle=True).item()
        return float(obj.get("data", {}).get("data_scale", 1.0))
    except Exception:
        return 1.0


def load_ftm_data(
    h5_path: str,
    metadata_npy: str = "",
    max_samples: int = 0,
) -> FTMData:
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with h5py.File(path, "r") as f:
        if "data" not in f or "mask_tr" not in f:
            raise KeyError("HDF5 must contain 'data' and 'mask_tr'.")

        data  = f["data"][...].astype(np.float32)
        mask  = f["mask_tr"][...].astype(np.float32)
        omega = f["omega"][...].astype(np.float64)
        grid_x = f["grid_x"][...].astype(np.float64) if "grid_x" in f else \
            np.linspace(0.0, 1.0, data.shape[2])
        grid_y = f["grid_y"][...].astype(np.float64) if "grid_y" in f else \
            np.linspace(0.0, 1.0, data.shape[3])

        # Try to recover channel names from metadata JSON
        channel_names: List[str] = []
        if "metadata" in f:
            try:
                meta_str = f["metadata"][()].decode() if isinstance(
                    f["metadata"][()], bytes) else str(f["metadata"][()])
                meta_json = json.loads(meta_str)
                channel_names = meta_json.get("channels", [])
            except Exception:
                pass

    if data.ndim != 5:
        raise ValueError(f"Expected data shape (B,M,H,W,C), got {data.shape}")

    C = data.shape[-1]
    if not channel_names:
        channel_names = CHANNEL_NAMES_4[:C] if C == 4 else \
            CHANNEL_NAMES_2 if C == 2 else [f"ch{c}" for c in range(C)]

    if max_samples > 0:
        data = data[:max_samples]

    B = data.shape[0]

    # ── Validate mask shape ─────────────────────────────────────────────────
    if mask.ndim == 4:
        if tuple(mask.shape) != tuple(data.shape[1:]):
            raise ValueError(
                f"Shared mask shape {mask.shape} must equal data[1:] {data.shape[1:]}")
    elif mask.ndim == 5:
        mask = mask[:B]
        if tuple(mask.shape[1:]) != tuple(data.shape[1:]):
            raise ValueError(
                f"Per-sample mask shape {mask.shape} incompatible with data {data.shape}")
    else:
        raise ValueError(f"Expected mask (M,H,W,C) or (B,M,H,W,C), got {mask.shape}")

    if not np.all(np.diff(omega) > 0):
        raise ValueError("omega must be strictly increasing.")

    data_scale = _try_load_scale_from_sidecar(path, metadata_npy)

    return FTMData(
        data=data, mask=mask, omega=omega,
        grid_x=grid_x, grid_y=grid_y,
        data_scale=data_scale,
        channel_names=channel_names,
    )


def normalize_coords_to_unit(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if math.isclose(hi, lo) else (x - lo) / (hi - lo)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-channel view helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_channel_views(
    data: np.ndarray,   # (B, M, H, W, C)
    mask: np.ndarray,   # (M, H, W, C) or (B, M, H, W, C)
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Returns
    -------
    data_channels : list[C] of (B, M, P) float32
    mask_channels : list[C] of (M, P) or (B, M, P) float32 in {0, 1}
    """
    B, M, H, W, C = data.shape
    P = H * W
    per_sample = mask.ndim == 5

    data_channels = [data[..., c].reshape(B, M, P) for c in range(C)]
    mask_channels = [
        (mask[..., c].reshape(M, P if not per_sample else -1) > 0.5).astype(np.float32)
        if not per_sample
        else (mask[..., c].reshape(B, M, P) > 0.5).astype(np.float32)
        for c in range(C)
    ]
    return data_channels, mask_channels


def validate_mask_coverage(
    mask_channels: List[np.ndarray],
    channel_names: List[str],
    per_sample: bool,
) -> None:
    for c, (m, name) in enumerate(zip(mask_channels, channel_names)):
        count = np.sum(m, axis=-1)  # (M,) or (B, M)
        bad = np.argwhere(count <= 0)
        if bad.size > 0:
            loc = tuple(int(v) for v in bad[0])
            dim = "sample/freq" if per_sample else "freq"
            raise ValueError(
                f"Channel '{name}' (ch={c}): no observed points at {dim} index {loc}."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Model components
# ─────────────────────────────────────────────────────────────────────────────

def build_phi(
    net_x: nn.Module,
    net_y: nn.Module,
    x_coords: torch.Tensor,  # (H, 1)
    y_coords: torch.Tensor,  # (W, 1)
) -> torch.Tensor:
    """Kronecker feature matrix Φ(x, y), shape (H*W, Rx*Ry)."""
    fx = net_x(x_coords)   # (H, Rx)
    fy = net_y(y_coords)   # (W, Ry)
    return torch.einsum("ir,jq->ijrq", fx, fy).reshape(-1, fx.shape[1] * fy.shape[1])


def weighted_tv_loss(cores: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Frequency-axis weighted TV²,  cores: (B, M, R)."""
    if cores.shape[1] <= 1:
        return cores.new_zeros(())
    d_omega = omega[1:] - omega[:-1]
    diff = (cores[:, 1:, :] - cores[:, :-1, :]) / d_omega.view(1, -1, 1)
    return torch.mean(diff * diff)


def core_smooth_loss(
    cores: torch.Tensor,   # (B, M, Rx*Ry)
    omega: torch.Tensor,   # (M,)
    Rx: int, Ry: int,
) -> torch.Tensor:
    """Spatial smoothness of core matrices, frequency-weighted."""
    B, M, _ = cores.shape
    c = cores.view(B, M, Rx, Ry)
    dx = c[:, :, 1:, :] - c[:, :, :-1, :]
    dy = c[:, :, :, 1:] - c[:, :, :, :-1]
    w = (1.0 + 14.0 * (omega - omega.min()) / (omega.max() - omega.min() + 1e-8))
    w = w.view(1, M, 1, 1)
    return torch.mean(w * dx ** 2) + torch.mean(w * dy ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    # ── Load data ─────────────────────────────────────────────────────────
    ftm_data = load_ftm_data(args.data_h5, args.metadata_npy, args.max_samples)
    data, mask, omega = ftm_data.data, ftm_data.mask, ftm_data.omega
    grid_x, grid_y = ftm_data.grid_x, ftm_data.grid_y
    channel_names = ftm_data.channel_names

    B, M, H, W, C = data.shape
    per_sample_mask = mask.ndim == 5
    R = args.rank_x * args.rank_y

    # ── Coordinates ───────────────────────────────────────────────────────
    if args.normalize_coords:
        x_np = normalize_coords_to_unit(grid_x)
        y_np = normalize_coords_to_unit(grid_y)
    else:
        x_np, y_np = grid_x.copy(), grid_y.copy()

    x_t = torch.from_numpy(x_np.astype(np.float32)).unsqueeze(-1).to(device)
    y_t = torch.from_numpy(y_np.astype(np.float32)).unsqueeze(-1).to(device)
    omega_t = torch.from_numpy(omega.astype(np.float32)).to(device)

    # ── Channel-wise data / mask views ────────────────────────────────────
    data_channels, mask_channels = build_channel_views(data, mask)
    validate_mask_coverage(mask_channels, channel_names, per_sample_mask)

    # Denominators for relative loss
    def _denom(dc, mc):
        if per_sample_mask:
            return np.sum(dc ** 2 * mc, axis=2)   # (B, M)
        return np.sum(dc ** 2 * mc[None], axis=2)  # (B, M)

    denom_list = [_denom(dc, mc) for dc, mc in zip(data_channels, mask_channels)]

    # ── Spatial basis networks ────────────────────────────────────────────
    # split_basis: separate (net_x, net_y) per physical displacement component
    # For C=2  → 1 pair;  C=4 (ux_re, ux_im, uy_re, uy_im) → 2 pairs
    n_basis_groups = C // 2 if (args.split_basis and C % 2 == 0) else 1

    def _make_mlp(rank):
        return MLP1D(rank, args.hidden_dim, args.hidden_layers, args.activation).to(device)

    net_x_list = [_make_mlp(args.rank_x) for _ in range(n_basis_groups)]
    net_y_list = [_make_mlp(args.rank_y) for _ in range(n_basis_groups)]

    # mapping: channel c → basis group index
    ch_to_group = [min(c // 2, n_basis_groups - 1) for c in range(C)]

    # ── Core parameters: one per channel ──────────────────────────────────
    core_init = float(args.core_init_scale)
    cores_params = nn.ParameterList([
        nn.Parameter(core_init * torch.randn(B, M, R, device=device))
        for _ in range(C)
    ])

    # ── Optimiser ─────────────────────────────────────────────────────────
    basis_params = [p for net in net_x_list + net_y_list for p in net.parameters()]
    core_param_list = list(cores_params.parameters())

    optimizer = torch.optim.Adam([
        {"params": basis_params,    "lr": args.lr},
        {"params": core_param_list, "lr": args.core_lr},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=20, factor=0.99, eps=1e-12, min_lr=4e-8
    )

    # Pre-move shared masks
    mask_shared = [
        torch.from_numpy(mc).to(device) if not per_sample_mask else None
        for mc in mask_channels
    ]

    # ── Print header ──────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print(f"FTM Training  (C={C} channels, {'split' if args.split_basis else 'shared'} basis)")
    print(f"data = {args.data_h5}")
    print(f"shape: B={B}, M={M}, H={H}, W={W}, C={C}")
    print(f"channels: {channel_names}")
    print(f"rank: Rx={args.rank_x}, Ry={args.rank_y}, R={R}")
    print(f"basis groups: {n_basis_groups}  (ch→group: {ch_to_group})")
    print(f"mask mode: {'per-sample' if per_sample_mask else 'shared'}")
    print(f"device={device}  iters={args.iters}  batch={args.batch_size}")
    print(f"lr_basis={args.lr:.2e}  lr_core={args.core_lr:.2e}  "
          f"beta={args.beta:.2e}  core_smooth={args.core_smooth:.2e}")
    print("─" * 72 + "\n")

    best_obj = float("inf")
    stale_rounds = 0

    for step in range(1, args.iters + 1):
        if args.batch_size >= B:
            idx_np = np.arange(B, dtype=np.int64)
        else:
            idx_np = np.random.permutation(B)[: args.batch_size].astype(np.int64)

        idx_t = torch.from_numpy(idx_np).long().to(device)
        bsz = len(idx_np)

        optimizer.zero_grad(set_to_none=True)

        # Build one phi per basis group
        phis = [build_phi(net_x_list[g], net_y_list[g], x_t, y_t)
                for g in range(n_basis_groups)]   # each (P, R)

        total_recon = cores_params[0].new_zeros(())
        total_tv    = cores_params[0].new_zeros(())
        total_smooth = cores_params[0].new_zeros(())

        for c in range(C):
            phi = phis[ch_to_group[c]]          # (P, R)
            core_b = cores_params[c].index_select(0, idx_t)   # (bsz, M, R)

            pred = torch.einsum("bmr,pr->bmp", core_b, phi)   # (bsz, M, P)

            gt = torch.from_numpy(data_channels[c][idx_np]).to(device)

            if per_sample_mask:
                mk = torch.from_numpy(mask_channels[c][idx_np]).to(device)
            else:
                mk = mask_shared[c].unsqueeze(0).expand(bsz, -1, -1)

            denom = torch.from_numpy(denom_list[c][idx_np]).to(device)

            sq_err = (pred - gt) ** 2
            rel = torch.sqrt(
                torch.sum(sq_err * mk, dim=2) / (denom + args.eps) + args.eps
            )
            total_recon = total_recon + torch.mean(rel)

            total_tv = total_tv + weighted_tv_loss(cores_params[c], omega_t)
            total_smooth = total_smooth + core_smooth_loss(
                cores_params[c], omega_t, args.rank_x, args.rank_y
            )

        total_recon  = total_recon  / C
        total_tv     = total_tv     / C
        total_smooth = total_smooth / C
        l2_reg = sum(torch.mean(p ** 2) for p in cores_params) / C

        obj = (total_recon
               + args.beta * total_tv
               + args.core_l2 * l2_reg
               + args.core_smooth * total_smooth)
        obj.backward()

        if args.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(basis_params, args.grad_clip)
        if args.core_grad_clip > 0.0:
            nn.utils.clip_grad_norm_(core_param_list, args.core_grad_clip)

        optimizer.step()
        scheduler.step(obj)

        obj_val = float(obj.item())
        if obj_val < best_obj:
            rel_improve = (best_obj - obj_val) / max(abs(best_obj), 1e-12)
            best_obj = obj_val
            stale_rounds = 0 if rel_improve >= args.tol else stale_rounds + 1
        else:
            stale_rounds += 1

        if args.log_every > 0 and step % args.log_every == 0:
            print(
                f"[step {step:05d}/{args.iters}] "
                f"recon={float(total_recon):.4e}  "
                f"tv={float(total_tv):.4e}  "
                f"smooth={float(total_smooth):.4e}  "
                f"obj={obj_val:.4e}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if args.patience > 0 and stale_rounds >= args.patience:
            print(f"Early stop at step={step} (stale={stale_rounds})")
            break

    # ── Save checkpoint ───────────────────────────────────────────────────
    cores_saved = [
        cores_params[c].detach().cpu()
                       .reshape(B, M, args.rank_x, args.rank_y)
                       .numpy().astype(np.float32)
        for c in range(C)
    ]

    ckpt: dict = {
        "channel_names": channel_names,
        "cores": [torch.from_numpy(cr) for cr in cores_saved],
        "omega":  torch.from_numpy(omega.astype(np.float32)),
        "grid_x": torch.from_numpy(grid_x.astype(np.float32)),
        "grid_y": torch.from_numpy(grid_y.astype(np.float32)),
        "data_scale": float(ftm_data.data_scale),
        "config": vars(args),
    }

    if n_basis_groups == 1:
        ckpt["net_x_state"] = net_x_list[0].state_dict()
        ckpt["net_y_state"] = net_y_list[0].state_dict()
    else:
        ckpt["net_x_state_list"] = [n.state_dict() for n in net_x_list]
        ckpt["net_y_state_list"] = [n.state_dict() for n in net_y_list]
        ckpt["ch_to_group"] = ch_to_group

    # Keep backward-compat keys for C=2 callers
    if C == 2:
        ckpt["net_x_state"] = net_x_list[0].state_dict()
        ckpt["net_y_state"] = net_y_list[0].state_dict()
        ckpt["cores_real"] = torch.from_numpy(cores_saved[0])
        ckpt["cores_imag"] = torch.from_numpy(cores_saved[1])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)

    summary = {
        "out": str(out_path),
        "B": B, "M": M, "H": H, "W": W, "C": C,
        "channel_names": channel_names,
        "rank_x": args.rank_x, "rank_y": args.rank_y,
        "n_basis_groups": n_basis_groups,
        "data_scale": float(ftm_data.data_scale),
    }
    print("\nSaved checkpoint:", out_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train frequency-axis FTM (multi-channel, joint optimisation)"
    )
    p.add_argument("--data_h5",      type=str,   default="elastic_dataset.h5")
    p.add_argument("--metadata_npy", type=str,   default="")
    p.add_argument("--out",          type=str,   default="ckp/ftm_gpu_checkpoint.pt")

    p.add_argument("--rank_x",       type=int,   default=32)
    p.add_argument("--rank_y",       type=int,   default=32)
    p.add_argument("--hidden_dim",   type=int,   default=1024+512)
    p.add_argument("--hidden_layers",type=int,   default=6)
    p.add_argument("--activation",   type=str,   default="sine",
                   choices=["sine", "relu", "tanh"])

    # split_basis: give each physical component (ux, uy) its own spatial basis
    p.add_argument("--split_basis",  action="store_true",
                   help="Use separate net_x/net_y per physical field pair "
                        "(ux_re+ux_im share one pair, uy_re+uy_im share another). "
                        "Only meaningful when C is even (e.g. C=4).")

    p.add_argument("--iters",        type=int,   default=28000)
    p.add_argument("--batch_size",   type=int,   default=64)

    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--core_lr",      type=float, default=1e-4)
    p.add_argument("--core_init_scale", type=float, default=1e-3)

    p.add_argument("--beta",         type=float, default=0,
                   help="Frequency-axis TV regularisation weight.")
    p.add_argument("--core_l2",      type=float, default=5e2,
                   help="L2 regularisation on core tensors.")
    p.add_argument("--core_smooth",  type=float, default=4e5,
                   help="Spatial smoothness regularisation on core matrices.")

    p.add_argument("--grad_clip",    type=float, default=0.0)
    p.add_argument("--core_grad_clip", type=float, default=0.0)

    p.add_argument("--tol",          type=float, default=1e-6)
    p.add_argument("--patience",     type=int,   default=0)
    p.add_argument("--eps",          type=float, default=1e-12)

    p.add_argument("--max_samples",  type=int,   default=0)
    p.add_argument("--normalize_coords", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       type=str,   default="auto")
    p.add_argument("--log_every",    type=int,   default=10)

    return p


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()