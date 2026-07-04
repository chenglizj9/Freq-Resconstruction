"""
diagnose_rank.py
----------------
Training-free diagnostic for *why* FTM (shared separable basis + per-sample
cores) reconstructs unobserved points well on some datasets but not others.

For a dataset it measures, per frequency:

  (1) PER-SAMPLE separable rank — SVD of each field U(H,W); how many singular
      values capture 99% energy.  This is the field's OWN best separable rank
      (each sample free to use its own basis).  Lower = intrinsically low-rank.

  (2) SHARED-BASIS ceiling — build ONE depth basis Bx (H,Rx) and ONE range
      basis By (W,Ry) shared across ALL samples/channels (top singular vectors
      of the stacked depth-/range-unfoldings), then reconstruct each field by
      double projection  U_hat = Bx Bxᵀ U By Byᵀ.  This is the BEST a dense FTM
      can do at rank (Rx,Ry) — independent of any sparse mask or optimisation.

The gap between (1) and (2) tells you the cause:
  * (1) low AND (2) low   → field is shared-low-rank → FTM should work.
  * (1) low BUT (2) high  → each field is low-rank but in a DIFFERENT subspace
                            (varying medium/geometry) → no compact shared basis.
  * (1) high              → fields intrinsically high separable rank.

Usage
-----
    python diagnose_rank.py --data_h5 ocean_dataset.h5 --tag ocean
    python diagnose_rank.py --data_h5 ../elastic_data/elastic_dataset.h5 --tag elastic
"""

from __future__ import annotations
import argparse
import h5py
import numpy as np


def per_sample_rank(fields, energy=0.99):
    """fields: (S, H, W) real.  Return median #SV for `energy` per sample, and
    mean relative error at a few fixed ranks."""
    ranks = []
    for U in fields:
        s = np.linalg.svd(U, compute_uv=False)
        e = np.cumsum(s ** 2) / max(np.sum(s ** 2), 1e-30)
        ranks.append(int(np.searchsorted(e, energy) + 1))
    return float(np.median(ranks)), float(np.percentile(ranks, 90))


def shared_basis_error(fields_by_chan, R):
    """fields_by_chan: list over channels of (S, H, W) real arrays.
    Build shared depth basis (H,R) and range basis (W,R) from ALL samples &
    channels, reconstruct, return mean per-field relative Frobenius error."""
    H = fields_by_chan[0].shape[1]
    W = fields_by_chan[0].shape[2]
    # depth-unfolding: stack columns (each depth-profile) → (H, ncol)
    depth_cols = []
    range_cols = []
    for F in fields_by_chan:
        S = F.shape[0]
        # (H, W*S): every range/sample column is a depth profile
        depth_cols.append(F.transpose(1, 0, 2).reshape(H, S * W))
        range_cols.append(F.transpose(2, 0, 1).reshape(W, S * H))
    Xd = np.concatenate(depth_cols, axis=1)        # (H, .)
    Xr = np.concatenate(range_cols, axis=1)        # (W, .)
    Ud, _, _ = np.linalg.svd(Xd, full_matrices=False)
    Ur, _, _ = np.linalg.svd(Xr, full_matrices=False)
    Bx = Ud[:, :R]                                  # (H, R)
    By = Ur[:, :R]                                  # (W, R)

    errs = []
    for F in fields_by_chan:
        for U in F:
            Uhat = Bx @ (Bx.T @ U @ By) @ By.T
            num = np.linalg.norm(U - Uhat)
            den = np.linalg.norm(U) + 1e-30
            errs.append(num / den)
    return float(np.mean(errs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_h5", required=True)
    p.add_argument("--tag", default="dataset")
    p.add_argument("--n_samples", type=int, default=80)
    p.add_argument("--freq_idx", type=int, nargs="*", default=None,
                   help="freq indices to analyse (default: first, middle, last)")
    p.add_argument("--ranks", type=int, nargs="*", default=[8, 16, 32, 48])
    args = p.parse_args()

    with h5py.File(args.data_h5, "r") as f:
        N, M, H, W, C = f["data"].shape
        omega = f["omega"][...]
        S = min(args.n_samples, N)
        idx = np.linspace(0, N - 1, S).round().astype(int)
        data = f["data"][idx]            # (S, M, H, W, C)
    print(f"\n=== {args.tag}  ({args.data_h5}) ===")
    print(f"N={N} M={M} grid={H}x{W} C={C}  using S={S} samples")

    freqs = args.freq_idx if args.freq_idx else \
        sorted({0, M // 2, M - 1})

    print(f"\n{'freq':>8} | {'per-sample r99 (med/p90)':>26} | shared-basis rel-err @ rank")
    print(f"{'':8} | {'':26} | " + "  ".join(f"R={r:<3}" for r in args.ranks))
    print("-" * 90)
    for fi in freqs:
        chans = [data[:, fi, :, :, c].astype(np.float64) for c in range(C)]
        # per-sample rank uses channel-0 magnitude proxy: use the complex if C>=2
        if C >= 2:
            mag = np.sqrt(chans[0] ** 2 + chans[1] ** 2)
        else:
            mag = np.abs(chans[0])
        med, p90 = per_sample_rank(mag)
        errs = [shared_basis_error(chans, R) for R in args.ranks]
        err_str = "  ".join(f"{e:5.3f}" for e in errs)
        print(f"{omega[fi]:8.0f} | {med:10.0f} / {p90:<13.0f} | {err_str}")
    print()


if __name__ == "__main__":
    main()
