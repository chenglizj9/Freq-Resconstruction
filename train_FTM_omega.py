"""
train_FTM_omega.py
------------------
Train a 2D Helmholtz Functional Tucker Model with shared basis plus
omega-conditioned FiLM modulation.

Aligned with train_FTM_GPU.py, this script uses joint optimization:
1) Core tensor sequences (real/imag) are trainable parameters.
2) Conditional x/y basis networks are trained jointly with cores.
3) No closed-form linear solve and no alternating optimization loop.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from train_FTM_GPU import (
    FTMData,
    _build_channel_views,
    load_ftm_data,
    normalize_coords_to_unit,
    set_seed,
)


class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class FourierOmegaEmbedding(nn.Module):
    def __init__(self, num_bands: int = 8):
        super().__init__()
        self.num_bands = num_bands
        self.out_dim = 1 + 2 * num_bands

    def forward(self, omega_norm: torch.Tensor) -> torch.Tensor:
        feats = [omega_norm]
        for k in range(self.num_bands):
            freq = (2 ** k) * math.pi
            feats.append(torch.sin(freq * omega_norm))
            feats.append(torch.cos(freq * omega_norm))
        return torch.cat(feats, dim=-1)


class OmegaConditionedMLP1D(nn.Module):
    def __init__(
        self,
        out_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        activation: str = "sine",
        omega_bands: int = 8,
        omega_hidden_dim: int = 128,
        film_scale: float = 0.1,
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

        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.film_scale = float(film_scale)

        self.in_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        in_dim = 1
        for _ in range(num_hidden_layers):
            self.in_layers.append(nn.Linear(in_dim, hidden_dim))
            self.activations.append(act())
            in_dim = hidden_dim
        self.out_layer = nn.Linear(in_dim, out_dim)

        self.omega_emb = FourierOmegaEmbedding(omega_bands)
        self.omega_mlp = nn.Sequential(
            nn.Linear(self.omega_emb.out_dim, omega_hidden_dim),
            nn.SiLU(),
            nn.Linear(omega_hidden_dim, 2 * num_hidden_layers * hidden_dim),
        )

        self._init_weights(activation)

    def _init_weights(self, activation: str) -> None:
        for m in self.in_layers:
            if activation == "sine":
                nn.init.uniform_(m.weight, -0.5, 0.5)
                nn.init.zeros_(m.bias)
            else:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        if activation == "sine":
            nn.init.uniform_(self.out_layer.weight, -0.5, 0.5)
            nn.init.zeros_(self.out_layer.bias)
        else:
            nn.init.xavier_uniform_(self.out_layer.weight)
            nn.init.zeros_(self.out_layer.bias)
        for m in self.omega_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def film_params(self, omega_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cond = self.omega_mlp(self.omega_emb(omega_norm))
        cond = cond.view(-1, self.num_hidden_layers, 2, self.hidden_dim)
        gamma = cond[:, :, 0, :]
        beta = cond[:, :, 1, :]
        return gamma, beta

    def forward(self, x: torch.Tensor, omega_norm: torch.Tensor) -> torch.Tensor:
        if omega_norm.ndim == 1:
            omega_norm = omega_norm.unsqueeze(-1)
        if omega_norm.shape[-1] != 1:
            raise ValueError(f"omega_norm must have shape (N,1), got {omega_norm.shape}")
        if x.shape[0] != omega_norm.shape[0]:
            raise ValueError(f"x batch {x.shape[0]} != omega batch {omega_norm.shape[0]}")

        gamma, beta = self.film_params(omega_norm)
        h = x
        for layer_idx, (linear, activation) in enumerate(zip(self.in_layers, self.activations)):
            h = linear(h)
            h = h * (1.0 + self.film_scale * gamma[:, layer_idx, :]) + self.film_scale * beta[:, layer_idx, :]
            h = activation(h)
        return self.out_layer(h)

    def modulation_penalty(self, omega_norm: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film_params(omega_norm)
        return torch.mean(gamma ** 2) + torch.mean(beta ** 2)


def normalize_omega(omega: torch.Tensor) -> torch.Tensor:
    omega_min = torch.min(omega)
    omega_max = torch.max(omega)
    return (omega - omega_min) / (omega_max - omega_min + 1e-8)


def build_phi_conditional(
    net_x: OmegaConditionedMLP1D,
    net_y: OmegaConditionedMLP1D,
    x_coords: torch.Tensor,
    y_coords: torch.Tensor,
    omega_norm: torch.Tensor,
) -> torch.Tensor:
    """Build conditional Phi for each frequency, shape (M, H*W, Rx*Ry)."""
    M = omega_norm.shape[0]
    H = x_coords.shape[0]
    W = y_coords.shape[0]

    x_rep = x_coords.unsqueeze(0).expand(M, -1, -1).reshape(M * H, 1)
    y_rep = y_coords.unsqueeze(0).expand(M, -1, -1).reshape(M * W, 1)
    omega_x = omega_norm.unsqueeze(1).expand(-1, H, -1).reshape(M * H, 1)
    omega_y = omega_norm.unsqueeze(1).expand(-1, W, -1).reshape(M * W, 1)

    fx = net_x(x_rep, omega_x).reshape(M, H, -1)
    fy = net_y(y_rep, omega_y).reshape(M, W, -1)
    phi = torch.einsum("mhr,mwq->mhwrq", fx, fy).reshape(M, H * W, -1)
    return phi


def basis_smoothness_loss(phi_by_m: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    if phi_by_m.shape[0] <= 1:
        return torch.zeros((), dtype=phi_by_m.dtype, device=phi_by_m.device)
    d_omega = omega[1:] - omega[:-1]
    if torch.any(d_omega <= 0):
        raise ValueError("omega must be strictly increasing for basis smoothness.")
    diff = (phi_by_m[1:] - phi_by_m[:-1]) / d_omega.view(-1, 1, 1)
    return torch.mean(diff * diff)


def core_matrix_smooth_loss(
    core: torch.Tensor,
    omega: torch.Tensor,
    rank_x: int,
    rank_y: int,
) -> torch.Tensor:
    B, M, _ = core.shape
    core_mat = core.view(B, M, rank_x, rank_y)
    dx = core_mat[:, :, 1:, :] - core_mat[:, :, :-1, :]
    dy = core_mat[:, :, :, 1:] - core_mat[:, :, :, :-1]

    omega_norm = (omega - omega.min()) / (omega.max() - omega.min() + 1e-8)
    freq_weights = (1.0 + 14.0 * omega_norm).view(1, M, 1, 1)
    loss_dx = torch.mean(freq_weights * (dx ** 2))
    loss_dy = torch.mean(freq_weights * (dy ** 2))
    return loss_dx + loss_dy


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    ftm_data: FTMData = load_ftm_data(
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
        x_coords_np = grid_x.astype(float)
        y_coords_np = grid_y.astype(float)

    data_re, data_im, mask_re_np, mask_im_np = _build_channel_views(data, mask)
    per_sample_mask = mask.ndim == 5

    if per_sample_mask:
        obs_re_count = mask_re_np.sum(axis=2)
        obs_im_count = mask_im_np.sum(axis=2)
        bad_re = (obs_re_count <= 0).nonzero()
        bad_im = (obs_im_count <= 0).nonzero()
        if len(bad_re[0]) > 0:
            raise ValueError(
                f"No observed points in real channel for sample={int(bad_re[0][0])}, freq_index={int(bad_re[1][0])}."
            )
        if len(bad_im[0]) > 0:
            raise ValueError(
                f"No observed points in imag channel for sample={int(bad_im[0][0])}, freq_index={int(bad_im[1][0])}."
            )
    else:
        obs_re_count = mask_re_np.sum(axis=1)
        obs_im_count = mask_im_np.sum(axis=1)
        bad_re = (obs_re_count <= 0).nonzero()[0]
        bad_im = (obs_im_count <= 0).nonzero()[0]
        if len(bad_re) > 0:
            raise ValueError(f"No observed points in real channel for freq_index={int(bad_re[0])}.")
        if len(bad_im) > 0:
            raise ValueError(f"No observed points in imag channel for freq_index={int(bad_im[0])}.")

    if per_sample_mask:
        denom_re_np = ((data_re ** 2) * mask_re_np).sum(axis=2)
        denom_im_np = ((data_im ** 2) * mask_im_np).sum(axis=2)
    else:
        denom_re_np = ((data_re ** 2) * mask_re_np[None, :, :]).sum(axis=2)
        denom_im_np = ((data_im ** 2) * mask_im_np[None, :, :]).sum(axis=2)

    net_x = OmegaConditionedMLP1D(
        out_dim=args.rank_x,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.hidden_layers,
        activation=args.activation,
        omega_bands=args.omega_bands,
        omega_hidden_dim=args.omega_hidden_dim,
        film_scale=args.film_scale,
    ).to(device)
    net_y = OmegaConditionedMLP1D(
        out_dim=args.rank_y,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.hidden_layers,
        activation=args.activation,
        omega_bands=args.omega_bands,
        omega_hidden_dim=args.omega_hidden_dim,
        film_scale=args.film_scale,
    ).to(device)

    x_t = torch.from_numpy(x_coords_np.astype("float32")).unsqueeze(-1).to(device)
    y_t = torch.from_numpy(y_coords_np.astype("float32")).unsqueeze(-1).to(device)
    omega_t = torch.from_numpy(omega.astype("float32")).to(device)
    omega_norm_t = normalize_omega(omega_t).unsqueeze(-1)

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=20,
        factor=0.99,
        eps=1e-12,
        min_lr=1e-7,
    )

    if not per_sample_mask:
        mask_re_shared_t = torch.from_numpy(mask_re_np).to(device)
        mask_im_shared_t = torch.from_numpy(mask_im_np).to(device)
    else:
        mask_re_shared_t = None
        mask_im_shared_t = None

    best_obj = float("inf")
    stale_rounds = 0

    print("\n" + "-" * 72)
    print("FTM Training (Frequency-axis, Joint Optimization, Omega-Conditioned Basis)")
    print(f"data={args.data_h5}")
    print(f"shape: B={B}, M={M}, H={H}, W={W}, C={C}")
    print(f"rank: Rx={args.rank_x}, Ry={args.rank_y}, R={R}")
    print(f"mask mode: {'per-sample' if per_sample_mask else 'shared'}")
    print(f"device={device}, iters={args.iters}, batch_size={args.batch_size}")
    print(
        f"lr_basis={args.lr:.2e}, lr_core={args.core_lr:.2e}, "
        f"beta={args.beta:.2e}, core_l2={args.core_l2:.2e}, "
        f"basis_smooth={args.basis_smoothness_weight:.2e}, mod={args.modulation_weight:.2e}"
    )
    print("-" * 72 + "\n")

    for step in range(1, args.iters + 1):
        if args.batch_size >= B:
            batch_idx_t = torch.arange(B, device=device)
            batch_idx_np = batch_idx_t.detach().cpu().numpy()
        else:
            batch_idx_t = torch.randperm(B, device=device)[: args.batch_size]
            batch_idx_np = batch_idx_t.detach().cpu().numpy()
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

        phi_by_m = build_phi_conditional(net_x, net_y, x_t, y_t, omega_norm_t)
        core_re_b = cores_re_param.index_select(0, batch_idx_t)
        core_im_b = cores_im_param.index_select(0, batch_idx_t)

        pred_re = torch.einsum("bmr,mpr->bmp", core_re_b, phi_by_m)
        pred_im = torch.einsum("bmr,mpr->bmp", core_im_b, phi_by_m)

        sq_err_re = (pred_re - gt_re_t) ** 2
        sq_err_im = (pred_im - gt_im_t) ** 2

        rel_re = torch.sqrt(
            torch.sum(sq_err_re * mask_re_t, dim=2) / (denom_re_t + args.eps) + args.eps
        )
        rel_im = torch.sqrt(
            torch.sum(sq_err_im * mask_im_t, dim=2) / (denom_im_t + args.eps) + args.eps
        )

        recon_loss = 0.5 * (torch.mean(rel_re) + torch.mean(rel_im))
        basis_reg = basis_smoothness_loss(phi_by_m, omega_t)
        mod_reg = net_x.modulation_penalty(omega_norm_t) + net_y.modulation_penalty(omega_norm_t)
        l2_reg = torch.mean(cores_re_param ** 2) + torch.mean(cores_im_param ** 2)
        core_smooth_re = core_matrix_smooth_loss(cores_re_param, omega_t, args.rank_x, args.rank_y)
        core_smooth_im = core_matrix_smooth_loss(cores_im_param, omega_t, args.rank_x, args.rank_y)

        tv_term = torch.zeros((), dtype=phi_by_m.dtype, device=device)
        if M > 1:
            d_omega = omega_t[1:] - omega_t[:-1]
            diff_re = (cores_re_param[:, 1:, :] - cores_re_param[:, :-1, :]) / d_omega.view(1, -1, 1)
            diff_im = (cores_im_param[:, 1:, :] - cores_im_param[:, :-1, :]) / d_omega.view(1, -1, 1)
            tv_term = torch.mean(diff_re ** 2) + torch.mean(diff_im ** 2)

        obj = (
            recon_loss
            + args.beta * tv_term
            + args.core_l2 * l2_reg
            + args.core_smooth * (core_smooth_re + core_smooth_im)
            + args.basis_smoothness_weight * basis_reg
            + args.modulation_weight * mod_reg
        )
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
                f"tv={float(tv_term.item()):.6e} "
                f"l2={float(l2_reg.item()):.6e} "
                f"core_smooth={float((core_smooth_re + core_smooth_im).item()):.6e} "
                f"basis_smooth={float(basis_reg.item()):.6e} "
                f"mod={float(mod_reg.item()):.6e} "
                f"obj={obj_value:.6e} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if args.patience > 0 and stale_rounds >= args.patience:
            print(
                f"Early stop at step={step}: no significant improvement for {stale_rounds} rounds."
            )
            break

    cores_re = (
        cores_re_param.detach().cpu().numpy().reshape(B, M, args.rank_x, args.rank_y).astype("float32")
    )
    cores_im = (
        cores_im_param.detach().cpu().numpy().reshape(B, M, args.rank_x, args.rank_y).astype("float32")
    )

    ckpt = {
        "net_x_state": net_x.state_dict(),
        "net_y_state": net_y.state_dict(),
        "cores_real": torch.from_numpy(cores_re),
        "cores_imag": torch.from_numpy(cores_im),
        "omega": torch.from_numpy(omega.astype("float32")),
        "omega_norm": omega_norm_t.detach().cpu(),
        "omega_min": float(omega.min()),
        "omega_max": float(omega.max()),
        "grid_x": torch.from_numpy(grid_x.astype("float32")),
        "grid_y": torch.from_numpy(grid_y.astype("float32")),
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
        "basis_smoothness_weight": float(args.basis_smoothness_weight),
        "modulation_weight": float(args.modulation_weight),
    }
    print("\nSaved checkpoint:", out_path)
    print("Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train omega-conditioned Functional Tucker Model (2D Helmholtz)")

    p.add_argument("--data_h5", type=str, default="helmholtz_dataset_42.h5")
    p.add_argument("--metadata_npy", type=str, default="")
    p.add_argument("--out", type=str, default="ckp/ftm_omega_checkpoint.pt")

    p.add_argument("--rank_x", type=int, default=24)
    p.add_argument("--rank_y", type=int, default=24)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--hidden_layers", type=int, default=5)
    p.add_argument("--activation", type=str, default="sine", choices=["sine", "relu", "tanh"])
    p.add_argument("--omega_bands", type=int, default=8)
    p.add_argument("--omega_hidden_dim", type=int, default=128)
    p.add_argument("--film_scale", type=float, default=0.1)

    p.add_argument("--iters", type=int, default=25000)
    p.add_argument("--batch_size", type=int, default=64)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--core_lr", type=float, default=1e-4)
    p.add_argument("--core_init_scale", type=float, default=1e-3)

    p.add_argument("--beta", type=float, default=1e4)
    p.add_argument("--core_l2", type=float, default=5e2)
    p.add_argument("--core_smooth", type=float, default=1e5)
    p.add_argument("--basis_smoothness_weight", type=float, default=1e-4)
    p.add_argument("--modulation_weight", type=float, default=1e-5)

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
