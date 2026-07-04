"""
visualize_dataset.py
--------------------
Visualization interface for the Helmholtz 2D dataset.

Public API
----------
    ds = HelmholtzDataset("helmholtz_dataset.h5")
    ds.plot_field(sample=0, freq_idx=10)          # single field (amplitude/phase/real)
    ds.plot_frequency_sweep(sample=0)             # full ω sweep for one sample
    ds.plot_sample_comparison(freq_idx=10, n=6)   # compare N samples at fixed ω
    ds.plot_source_config(sample=0)               # source positions overlay
    ds.plot_wavenumber_spectrum(sample=0, freq_idx=10)  # 2D FFT of field
    ds.plot_ftm_validation(rank=8)                # FTM low-rank approximation check
    ds.plot_observation_mask(freq_idx=10, sample=0)  # plot sparse observation mask
    ds.plot_mask_sweep(sample=0)                     # plot mask across frequencies

Run as script for a quick demo:
    python visualize_dataset.py helmholtz_dataset.h5
"""

import json
from pathlib import Path
from typing import Optional, Union

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import TwoSlopeNorm


# ──────────────────────────────────────────────────────────────────────────────
# Style helpers
# ──────────────────────────────────────────────────────────────────────────────

CMAP_FIELD = "RdBu_r"    # diverging — Re(u)
CMAP_AMP   = "inferno"   # sequential — |u|
CMAP_PHASE = "hsv"       # cyclic     — angle(u)
CMAP_MASK  = "Blues"

def _symm_norm(data):
    """Symmetric colormap norm centred on zero."""
    vmax = np.percentile(np.abs(data), 99)
    return TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

def _label_panel(ax, text, loc="upper left", fontsize=9):
    ax.annotate(text, xy=(0.03, 0.97) if "upper left" in loc else (0.97, 0.97),
                xycoords="axes fraction",
                va="top", ha="left" if "left" in loc else "right",
                fontsize=fontsize, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, lw=0))


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loader
# ──────────────────────────────────────────────────────────────────────────────

class HelmholtzDataset:
    """
    Lazy-loading wrapper around the HDF5 dataset produced by generate_dataset.py.

    Parameters
    ----------
    path : str or Path — path to the .h5 file
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        with h5py.File(self.path, "r") as f:
            self.meta  = json.loads(f["metadata"][()])
            self.omega = f["omega"][:]           # (M,)
            self.gx    = f["grid_x"][:]          # (Ng,)
            self.gy    = f["grid_y"][:]          # (Ng,)
            self.has_mask = "mask_tr" in f.keys()
            if self.has_mask:
                self.mask_shape = f["mask_tr"].shape
                self.mask_per_sample = (len(self.mask_shape) == 5)
            else:
                self.mask_shape = None
                self.mask_per_sample = False

            # shapes stored for reference — arrays loaded on demand
            self._N, self._M, self._Ng = (
                f["fields_real"].shape[0],
                f["fields_real"].shape[1],
                f["fields_real"].shape[2],
            )

        self.L = self.meta["L"]
        print(f"Dataset:  {self.path.name}")
        print(f"  Samples  N = {self._N}")
        print(f"  Freqs    M = {self._M}  ω ∈ [{self.omega[0]:.2f}, {self.omega[-1]:.2f}]")
        print(f"  Grid     Ng = {self._Ng} × {self._Ng}")
        if self.has_mask:
            mode = "per-sample" if self.mask_per_sample else "shared"
            print(f"  Mask     FOUND ✅  shape={self.mask_shape}  mode={mode}")
        else:
            print("  Mask     NOT FOUND ❌")

    # ── low-level loaders ──────────────────────────────────────────────────

    def get_field(self, sample: int, freq_idx: int) -> np.ndarray:
        """Return complex field u for (sample, freq_idx) → shape (Ng, Ng)."""
        with h5py.File(self.path, "r") as f:
            r = f["fields_real"][sample, freq_idx]
            im = f["fields_imag"][sample, freq_idx]
        return r.astype(np.float64) + 1j * im.astype(np.float64)

    def get_trajectory(self, sample: int) -> np.ndarray:
        """Return all M fields for one sample → shape (M, Ng, Ng) complex."""
        with h5py.File(self.path, "r") as f:
            r  = f["fields_real"][sample]   # (M, Ng, Ng)
            im = f["fields_imag"][sample]
        return r.astype(np.float64) + 1j * im.astype(np.float64)

    def get_sources(self, sample: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (positions (K,2), amplitudes (K,) complex) for a sample."""
        with h5py.File(self.path, "r") as f:
            pos = f["sources"][sample]          # (max_K, 2)
            amp = f["amplitudes"][sample]       # (max_K, 2)
        valid = ~np.isnan(pos[:, 0])
        pos = pos[valid]
        amp = (amp[valid, 0] + 1j * amp[valid, 1])
        return pos, amp

    def get_mask_at_freq(self, freq_idx: int, sample: int = 0) -> np.ndarray:
        """Get mask at (sample, freq_idx): returns (Ng, Ng) float32."""
        if not self.has_mask:
            return np.zeros((self._Ng, self._Ng))

        with h5py.File(self.path, "r") as f:
            mask_ds = f["mask_tr"]
            if mask_ds.ndim == 4:
                return mask_ds[freq_idx, :, :, 0].astype(np.float32)
            if mask_ds.ndim == 5:
                s = int(np.clip(sample, 0, self._N - 1))
                return mask_ds[s, freq_idx, :, :, 0].astype(np.float32)
            raise ValueError(f"Unsupported mask_tr shape: {mask_ds.shape}")

    # ── Plot 1: single field ───────────────────────────────────────────────

    def plot_field(
        self,
        sample: int = 0,
        freq_idx: int = 0,
        mode: str = "all",    # "real" | "amp" | "phase" | "all"
        figsize: tuple = (13, 4),
        show: bool = True,
        save: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot one field slice.

        mode='all'   → 3-panel: Re(u) | |u| | ∠u
        mode='real'  → just Re(u)
        mode='amp'   → just |u|
        mode='phase' → just ∠u
        """
        u = self.get_field(sample, freq_idx)
        omega = self.omega[freq_idx]
        X, Y = np.meshgrid(self.gx, self.gy, indexing="ij")
        mask = self.get_mask_at_freq(freq_idx, sample=sample)

        def _plot_one(ax, data, cmap, norm, title, cbar_label):
            im = ax.pcolormesh(X, Y, data, cmap=cmap, norm=norm, shading="auto")
            ax.set_aspect("equal")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(title, fontsize=10)
            plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
            # overlay source positions
            pos, _ = self.get_sources(sample)
            ax.scatter(pos[:, 0], pos[:, 1], c="yellow", s=40, zorder=5,
                       edgecolors="k", linewidths=0.5, label="source")

            # overlay sparse observation points (white dots)
            if self.has_mask:
                mx, my = np.where(mask > 0.5)
                ax.scatter(self.gx[my], self.gy[mx], c='white', s=4, alpha=0.8,
                           zorder=4, label='observed')
                ax.legend(loc='upper right', fontsize=8)

        if mode == "all":
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            _plot_one(axes[0], u.real,           CMAP_FIELD, _symm_norm(u.real),
                      "Re(u)",  "amplitude")
            _plot_one(axes[1], np.abs(u),         CMAP_AMP,  None,
                      "|u|",   "magnitude")
            _plot_one(axes[2], np.angle(u),       CMAP_PHASE, None,
                      "∠u",   "phase (rad)")
            fig.suptitle(f"Helmholtz field — sample {sample},  ω = {omega:.2f}", y=1.02)
        else:
            fig, ax = plt.subplots(figsize=(5, 4.5))
            if mode == "real":
                _plot_one(ax, u.real,     CMAP_FIELD, _symm_norm(u.real), "Re(u)", "amplitude")
            elif mode == "amp":
                _plot_one(ax, np.abs(u),  CMAP_AMP,   None,               "|u|",  "magnitude")
            elif mode == "phase":
                _plot_one(ax, np.angle(u),CMAP_PHASE, None,               "∠u",  "phase (rad)")
            ax.set_title(f"Sample {sample},  ω = {omega:.2f}", fontsize=10)

        plt.tight_layout()
        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ── Plot 2: frequency sweep for one sample ─────────────────────────────

    def plot_frequency_sweep(
        self,
        sample: int = 0,
        n_panels: int = 9,
        mode: str = "real",
        figsize: tuple = (15, 5),
        show: bool = True,
        save: Optional[str] = None,
    ) -> plt.Figure:
        """
        Show a grid of field slices at evenly spaced frequencies for one sample.
        Illustrates how the wave pattern evolves with ω.
        """
        traj = self.get_trajectory(sample)          # (M, Ng, Ng)
        pos, _ = self.get_sources(sample)
        X, Y = np.meshgrid(self.gx, self.gy, indexing="ij")

        indices = np.linspace(0, self._M - 1, n_panels, dtype=int)
        ncols = min(n_panels, 9)
        nrows = (n_panels + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.array(axes).ravel()

        for k, idx in enumerate(indices):
            u = traj[idx]
            data = u.real if mode == "real" else np.abs(u)
            norm = _symm_norm(data) if mode == "real" else None
            cmap = CMAP_FIELD if mode == "real" else CMAP_AMP
            ax = axes[k]
            ax.pcolormesh(X, Y, data, cmap=cmap, norm=norm,
                          shading="auto", rasterized=True)
            ax.scatter(pos[:, 0], pos[:, 1], c="yellow", s=15, zorder=5,
                       edgecolors="k", linewidths=0.3)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            _label_panel(ax, f"ω={self.omega[idx]:.1f}")

        for ax in axes[n_panels:]:
            ax.set_visible(False)

        fig.suptitle(f"Frequency sweep — sample {sample}  "
                     f"({mode}, ω ∈ [{self.omega[0]:.1f}, {self.omega[-1]:.1f}])",
                     fontsize=11)
        plt.tight_layout()
        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ── Plot 3: compare multiple samples at fixed ω ────────────────────────

    def plot_sample_comparison(
        self,
        freq_idx: int = 0,
        n: int = 6,
        mode: str = "real",
        figsize: tuple = (14, 4),
        show: bool = True,
        save: Optional[str] = None,
    ) -> plt.Figure:
        """
        Compare n different BC/IC samples at the same frequency.
        Highlights variability across source configurations.
        """
        X, Y = np.meshgrid(self.gx, self.gy, indexing="ij")
        n = min(n, self._N)
        ncols = n
        fig, axes = plt.subplots(1, ncols, figsize=figsize)

        for k in range(n):
            u = self.get_field(k, freq_idx)
            data = u.real if mode == "real" else np.abs(u)
            norm = _symm_norm(data) if mode == "real" else None
            cmap = CMAP_FIELD if mode == "real" else CMAP_AMP
            pos, _ = self.get_sources(k)
            ax = axes[k]
            ax.pcolormesh(X, Y, data, cmap=cmap, norm=norm,
                          shading="auto", rasterized=True)
            ax.scatter(pos[:, 0], pos[:, 1], c="yellow", s=20, zorder=5,
                       edgecolors="k", linewidths=0.4)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            # _label_panel(ax, f"s={k}")

        fig.suptitle(f"{n} samples at ω = {self.omega[freq_idx]:.2f}  ({mode})", fontsize=11)
        plt.tight_layout()
        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ── Plot 4: wavenumber spectrum (2-D FFT) ──────────────────────────────

    def plot_wavenumber_spectrum(
        self,
        sample: int = 0,
        freq_idx: int = 0,
        figsize: tuple = (10, 4),
        show: bool = True,
        save: Optional[str] = None,
    ) -> plt.Figure:
        """
        Show the 2-D spatial Fourier spectrum of a field.
        Useful for verifying the expected dominant wavenumber k = ω/c.
        """
        u = self.get_field(sample, freq_idx)
        U = np.fft.fftshift(np.fft.fft2(u))
        Ng = self._Ng
        dk = 2 * np.pi / self.L
        kx = np.fft.fftshift(np.fft.fftfreq(Ng, d=self.L / Ng)) * 2 * np.pi
        KX, KY = np.meshgrid(kx, kx, indexing="ij")

        omega = self.omega[freq_idx]
        k_expected = omega / self.meta["c"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Spatial field
        X, Y = np.meshgrid(self.gx, self.gy, indexing="ij")
        ax1.pcolormesh(X, Y, u.real, cmap=CMAP_FIELD, norm=_symm_norm(u.real),
                       shading="auto")
        ax1.set_aspect("equal")
        ax1.set_title(f"Re(u)  ω={omega:.2f}")
        ax1.set_xlabel("x");  ax1.set_ylabel("y")
        pos, _ = self.get_sources(sample)
        ax1.scatter(pos[:, 0], pos[:, 1], c="yellow", s=30, zorder=5,
                    edgecolors="k", linewidths=0.5)

        # Spectrum
        spec = np.log1p(np.abs(U))
        ax2.pcolormesh(KX, KY, spec, cmap=CMAP_AMP, shading="auto")
        # Expected wavenumber ring
        theta = np.linspace(0, 2 * np.pi, 360)
        ax2.plot(k_expected * np.cos(theta), k_expected * np.sin(theta),
                 "w--", lw=1.2, label=f"|k|=ω/c={k_expected:.1f}")
        ax2.set_aspect("equal")
        ax2.set_xlim(-4 * k_expected, 4 * k_expected)
        ax2.set_ylim(-4 * k_expected, 4 * k_expected)
        ax2.set_title("log(1+|FFT(u)|)")
        ax2.set_xlabel("kx");  ax2.set_ylabel("ky")
        ax2.legend(fontsize=8, loc="upper right")

        fig.suptitle(f"Wavenumber spectrum — sample {sample}, ω={omega:.2f}", fontsize=11)
        plt.tight_layout()
        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ── Plot 5: FTM validation ─────────────────────────────────────────────

    def plot_ftm_validation(
        self,
        n_samples: int = 20,
        rank: int = 8,
        freq_idx_check: Optional[int] = None,
        figsize: tuple = (13, 9),
        show: bool = True,
        save: Optional[str] = None,
    ) -> plt.Figure:
        """
        Validate the FTM decomposition assumption  u ≈ f(x) · G(ω).

        Loads n_samples trajectories, stacks them into a tensor, performs
        truncated SVD, and reports:
          - Singular value decay (energy compaction)
          - Reconstruction error vs rank
          - Visual comparison of original vs rank-K reconstruction at one ω
        """
        n_samples = min(n_samples, self._N)

        freq_mask = (self.omega >= self.omega[0]) & (self.omega <= self.omega[-1])
        freq_indices = np.where(freq_mask)[0]
        M_sub = len(freq_indices)
        selected_omegas = self.omega[freq_indices]

        # Stack tensor: (n_samples * M, Ng*Ng)
        print(f"  Loading {n_samples} trajectories for FTM analysis...")
        rows = []
        for i in range(n_samples):
            traj = self.get_trajectory(i)     # (M, Ng, Ng)
            traj_sub = traj[freq_indices]         # (M_sub, Ng, Ng)
            rows.append(traj_sub.reshape(M_sub, -1))   # (M_sub, Ng²)
        T = np.concatenate(rows, axis=0)  # (n_samples*M_sub, Ng²) complex

        # SVD on real part (FTM in practice uses real decomposition)
        T_real = T.real
        U_sv, S, Vt = np.linalg.svd(T_real, full_matrices=False)

        # Reconstruction errors across ranks
        ranks_test = [1, 2, 4, 8, 16, 32, min(64, len(S))]
        errors = []
        for r in ranks_test:
            T_approx = (U_sv[:, :r] * S[:r]) @ Vt[:r]
            err = np.linalg.norm(T_real - T_approx) / np.linalg.norm(T_real)
            errors.append(err)

        # Choose a freq index for visual check
        if freq_idx_check is None:
            freq_idx_check = M_sub // 2
        # Reconstruct at chosen rank
        T_approx_r = (U_sv[:, :rank] * S[:rank]) @ Vt[:rank]

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        # Panel A: singular value decay
        ax_sv = fig.add_subplot(gs[0, 0])
        n_sv = min(64, len(S))
        ax_sv.semilogy(np.arange(1, n_sv + 1), S[:n_sv] / S[0], "o-",
                       ms=3, lw=1.5, color="#4C72B0")
        ax_sv.axvline(rank, color="r", lw=1, ls="--", label=f"rank={rank}")
        ax_sv.set_xlabel("Singular value index")
        ax_sv.set_ylabel("Normalised σ_i / σ_1")
        ax_sv.set_title("Singular value decay")
        ax_sv.legend(fontsize=8)
        ax_sv.grid(True, alpha=0.3)

        # Panel B: reconstruction error vs rank
        ax_err = fig.add_subplot(gs[0, 1])
        ax_err.semilogy(ranks_test, errors, "s-", ms=5, lw=1.5, color="#DD8452")
        ax_err.axvline(rank, color="r", lw=1, ls="--")
        ax_err.set_xlabel("Rank")
        ax_err.set_ylabel("Relative Frobenius error")
        ax_err.set_title("FTM reconstruction error")
        ax_err.grid(True, alpha=0.3)

        # Panel C: cumulative energy
        ax_cum = fig.add_subplot(gs[0, 2])
        cum_energy = np.cumsum(S**2) / np.sum(S**2)
        ax_cum.plot(np.arange(1, n_sv + 1), cum_energy[:n_sv] * 100,
                    "-", lw=1.5, color="#55A868")
        ax_cum.axvline(rank, color="r", lw=1, ls="--", label=f"rank={rank}")
        ax_cum.set_xlabel("Rank")
        ax_cum.set_ylabel("Cumulative energy (%)")
        ax_cum.set_title("Energy compaction")
        ax_cum.legend(fontsize=8)
        ax_cum.grid(True, alpha=0.3)
        r_90 = np.searchsorted(cum_energy, 0.90) + 1
        ax_cum.axhline(90, color="gray", lw=0.8, ls=":")
        ax_cum.text(r_90 + 0.5, 90, f"90% @ r={r_90}", fontsize=7, color="gray")

        # Panels D–F: original vs reconstruction vs error at one ω
        sample_row = freq_idx_check   # first sample, chosen freq
        orig  = T_real[sample_row].reshape(self._Ng, self._Ng)
        recon = T_approx_r[sample_row].reshape(self._Ng, self._Ng)
        err_map = orig - recon
        X, Y = np.meshgrid(self.gx, self.gy, indexing="ij")

        for col, (data, title, cmap, norm) in enumerate([
            (orig,    "Original",          CMAP_FIELD, _symm_norm(orig)),
            (recon,   f"Rank-{rank} approx", CMAP_FIELD, _symm_norm(orig)),
            (err_map, "Error",             CMAP_FIELD, _symm_norm(err_map)),
        ]):
            ax = fig.add_subplot(gs[1, col])
            ax.pcolormesh(X, Y, data, cmap=cmap, norm=norm, shading="auto")
            ax.set_aspect("equal")
            ax.set_xticks([]);  ax.set_yticks([])
            ax.set_title(f"{title}  ω={self.omega[freq_idx_check]:.1f}", fontsize=9)

        fig.suptitle(f"FTM validation  (n={n_samples} samples, rank={rank})", fontsize=12)
        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ── Plot 6: sparse observation mask (NEW ✔️) ────────────────────────────────
    def plot_observation_mask(
        self,
        freq_idx: int = 0,
        sample: int = 0,
        figsize: tuple = (5, 4.5),
        show: bool = True,
        save: Optional[str] = None,
    ) -> plt.Figure:
        """Plot sparse observation mask at given frequency."""
        if not self.has_mask:
            print("No mask found in dataset.")
            return

        X, Y = np.meshgrid(self.gx, self.gy, indexing="ij")
        mask = self.get_mask_at_freq(freq_idx, sample=sample)
        omega = self.omega[freq_idx]

        fig, ax = plt.subplots(figsize=figsize)
        ax.pcolormesh(X, Y, mask, cmap=CMAP_MASK, vmin=0, vmax=1, shading="auto")
        ax.set_aspect("equal")
        ax.set_title(f"Sparse observations | sample={sample}, ω={omega:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.tight_layout()

        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ── Plot 7: mask sweep across frequencies (NEW ✔️) ──────────────────────────
    def plot_mask_sweep(
        self,
        n_panels: int = 9,
        sample: int = 0,
        figsize: tuple = (15, 5),
        show: bool = True,
        save: Optional[str] = None,
    ) -> plt.Figure:
        """Show observation masks across frequencies."""
        if not self.has_mask:
            print("No mask found.")
            return

        indices = np.linspace(0, self._M - 1, n_panels, dtype=int)
        ncols = min(n_panels,9)
        nrows = (n_panels + ncols-1)//ncols
        X, Y = np.meshgrid(self.gx, self.gy, indexing="ij")

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.ravel(axes)

        for k, idx in enumerate(indices):
            ax = axes[k]
            m = self.get_mask_at_freq(idx, sample=sample)
            ax.pcolormesh(X, Y, m, cmap=CMAP_MASK, vmin=0, vmax=1, shading="auto")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            _label_panel(ax, f"ω={self.omega[idx]:.1f}")

        for ax in axes[n_panels:]:
            ax.set_visible(False)

        fig.suptitle(f"Sparse observation mask sweep (sample={sample})")
        plt.tight_layout()
        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data_for_test/helmholtz_dataset_42_for_test_mask1.h5"
    ds = HelmholtzDataset(path)

    print("\n── Plot 1: single field (all panels) ──")
    ds.plot_field(sample=0, freq_idx=0, mode="all",
                  save="visual_data/vis_field_42.png", show=False)

    print("── Plot 2: frequency sweep ──")
    ds.plot_frequency_sweep(sample=0, n_panels=9,
                            save="visual_data/vis_sweep_42.png", show=False)

    print("── Plot 3: sample comparison ──")
    ds.plot_sample_comparison(freq_idx=len(ds.omega) // 2, n=6,
                              save="visual_data/vis_comparison_42.png", show=False)

    print("── Plot 4: wavenumber spectrum ──")
    ds.plot_wavenumber_spectrum(sample=0, freq_idx=len(ds.omega) // 2,
                                save="visual_data/vis_spectrum_42.png", show=False)

    print("── Plot 5: FTM validation ──")
    ds.plot_ftm_validation(n_samples=20, rank=40,
                           save="visual_data/vis_ftm_42.png", show=False)

    print("\n── Plot 6: observation mask (NEW) ──")
    ds.plot_observation_mask(freq_idx=len(ds.omega)//2,
                             save="visual_data/vis_mask_42.png", show=False)

    print("── Plot 7: mask sweep (NEW) ──")
    ds.plot_mask_sweep(n_panels=9, save="visual_data/vis_mask_sweep_42.png", show=False)

    print("\nAll plots saved.")