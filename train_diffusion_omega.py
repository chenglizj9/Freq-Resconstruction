"""
train_diffusion_omega.py
------------------------
Train a conditional diffusion model p(G | omega) on cores produced by
train_FTM_omega.py.

This entrypoint reuses the current diffusion training implementation and only
specializes the default checkpoint paths for the omega-conditioned FTM pipeline.
"""

from __future__ import annotations

import argparse

from train_diffusion import train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train conditional U-Net diffusion on omega-conditioned 2D FTM core tensors"
    )

    p.add_argument("--ftm_ckpt", type=str, default="ckp/ftm_omega_checkpoint.pt")
    p.add_argument("--out", type=str, default="ckp/diffusion_core_omega.pt")

    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--max_pairs", type=int, default=0)

    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=0.0)

    p.add_argument("--base_channels", type=int, default=128)
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--time_dim", type=int, default=128)
    p.add_argument("--omega_bands", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--mid_attn_heads", type=int, default=4)
    p.add_argument("--mid_attn_dropout", type=float, default=0.0)

    p.add_argument("--hidden_dim", type=int, default=0)
    p.add_argument("--depth", type=int, default=0)

    p.add_argument("--diffusion_steps", type=int, default=500)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)
    p.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"])
    p.add_argument("--freq_consistency_weight", type=float, default=0.05)

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
