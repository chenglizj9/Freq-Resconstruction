"""
visualize_elastic.py
--------------------
Visualization interface for the 2D Elastic Wave dataset.

Public API
----------
    ds = ElasticDataset("elastic_dataset.h5")
    ds.plot_field(sample=0, freq_idx=10)          # ux + uy field (real/amp/phase)
    ds.plot_frequency_sweep(sample=0)             # full ω sweep for one sample
    ds.plot_sample_comparison(freq_idx=10, n=6)   # compare N samples at fixed ω
    ds.plot_obstacle_and_materials(sample=0)     # obstacle + λ/μ/ρ fields
    ds.plot_wavenumber_spectrum(sample=0, freq_idx=10)  # 2D FFT of ux
    ds.plot_ftm_validation(rank=8)                # FTM low-rank check
    ds.plot_observation_mask(freq_idx=10, sample=0) # sparse observation mask
    ds.plot_mask_sweep(sample=0)                  # mask across frequencies

Run as script for a quick demo:
    python visualize_elastic.py elastic_dataset.h5
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
CMAP_MAT   = "viridis"   # material fields (λ, μ, ρ)
CMAP_OBST  = "gray"      # obstacle mask

def _symm_norm(data):
    """Symmetric colormap norm centred on zero."""
    vmax = np.percentile(np.abs(data), 99)
    return TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

def _label_panel(ax, text, loc="upper left", fontsize=9):
    ax.annotate(text, xy=(0.03, 0.97) if "upper left" in loc else (0.97, 0.97),
                xycoords="axes fraction",
                va="top", ha="left" if "left" in loc else "right",
                fontsize=fontsize, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45))


# ──────────────────────────────────────────────────────────────────────────────
# Elastic Dataset loader
# ──────────────────────────────────────────────────────────────────────────────

class ElasticDataset:
    """
    Lazy-loading wrapper around HDF5 elastic wave 2D dataset.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        with h5py.File(self.path, "r") as f:
            self.meta = json.loads(f["metadata"][()])
            self.omega = f["omega"][:]
            self.gx = f["grid_x"][:]
            self.gy = f["grid_y"][:]

            # Core shapes
            self._N = f["ux_real"].shape[0]
            self._M = f["ux_real"].shape[1]
            self._Ng = f["ux_real"].shape[2]

            # Mask
            self.has_mask = "mask_tr" in f.keys()
            if self.has_mask:
                self.mask_shape = f["mask_tr"].shape
                self.mask_per_sample = (len(self.mask_shape) == 5)
            else:
                self.mask_shape = None
                self.mask_per_sample = False

        self.L = self.meta["L"]
        self.c = 1.0  # dummy for spectrum plot (elastic uses wave speeds)
        print(f"Elastic Dataset: {self.path.name}")
        print(f"  Samples  N = {self._N}")
        print(f"  Freqs    M = {self._M}  ω ∈ [{self.omega[0]:.2f}, {self.omega[-1]:.2f}]")
        print(f"  Grid     Ng = {self._Ng} × {self._Ng}")
        print(f"  Fields   ux, uy (2 displacement components)")
        if self.has_mask:
            mode = "per-sample" if self.mask_per_sample else "shared"
            print(f"  Mask     FOUND ✅  shape={self.mask_shape}  mode={mode}")
        else:
            print("  Mask     NOT FOUND ❌")

    # ── Low-level loaders ──────────────────────────────────────────────────

    def get_ux(self, sample: int, freq_idx: int) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            r = f["ux_real"][sample, freq_idx]
            im = f["ux_imag"][sample, freq_idx]
        return r + 1j * im

    def get_uy(self, sample: int, freq_idx: int) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            r = f["uy_real"][sample, freq_idx]
            im = f["uy_imag"][sample, freq_idx]
        return r + 1j * im

    def get_both_fields(self, sample: int, freq_idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.get_ux(sample, freq_idx), self.get_uy(sample, freq_idx)

    def get_trajectory_ux(self, sample: int) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            r = f["ux_real"][sample]
            im = f["ux_imag"][sample]
        return r + 1j * im

    def get_obstacle(self, sample: int) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            return f["obstacle_mask"][sample]

    def get_materials(self, sample: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(self.path, "r") as f:
            lam = f["lambda_field"][sample]
            mu = f["mu_field"][sample]
            rho = f["rho_field"][sample]
        return lam, mu, rho

    def get_mask_at_freq(self, freq_idx: int, sample: int = 0) -> np.ndarray:
        if not self.has_mask:
            return np.zeros((self._Ng, self._Ng))
        with h5py.File(self.path, "r") as f:
            mask_ds = f["mask_tr"]
            if mask_ds.ndim == 4:
                return mask_ds[freq_idx, :, :, 0].astype(np.float32)
            if mask_ds.ndim == 5:
                s = int(np.clip(sample, 0, self._N - 1))
                return mask_ds[s, freq_idx, :, :, 0].astype(np.float32)
            raise ValueError(f"Unsupported mask shape: {mask_ds.shape}")

    # ── Plot 1: ux + uy fields ─────────────────────────────────────────────
    def plot_field(
        self,
        sample: int = 0,
        freq_idx: int = 0,
        mode: str = "all",
        figsize: tuple = (15, 9),
        show: bool = True,
        save: Optional[str] = None,
    ) -> plt.Figure:
        ux, uy = self.get_both_fields(sample, freq_idx)
        omega = self.omega[freq_idx]
        X, Y = np.meshgrid(self.gx, self.gy, indexing="ij")
        mask = self.get_mask_at_freq(freq_idx, sample)
        obst = self.get_obstacle(sample)

        def plot_panel(ax, data, title, cmap, norm):
            im = ax.pcolormesh(X, Y, data, cmap=cmap, norm=norm, shading="auto")
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
            # Obstacle overlay
            ax.contour(X, Y, obst, levels=[0.5], colors="white")
            # Source
            ax.scatter(self.L/6, self.L/2, c="yellow", s=50, edgecolors="k", zorder=5)

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 3, figure=fig)

        # ux row
        plot_panel(fig.add_subplot(gs[0,0]), ux.real, "Re(ux)", CMAP_FIELD, _symm_norm(ux.real))
        plot_panel(fig.add_subplot(gs[0,1]), np.abs(ux), "|ux|", CMAP_AMP, None)
        plot_panel(fig.add_subplot(gs[0,2]), np.angle(ux), "∠(ux)", CMAP_PHASE, None)

        # uy row
        plot_panel(fig.add_subplot(gs[1,0]), uy.real, "Re(uy)", CMAP_FIELD, _symm_norm(uy.real))
        plot_panel(fig.add_subplot(gs[1,1]), np.abs(uy), "|uy|", CMAP_AMP, None)
        plot_panel(fig.add_subplot(gs[1,2]), np.angle(uy), "∠(uy)", CMAP_PHASE, None)

        fig.suptitle(f"Elastic Wave | sample {sample}, ω={omega:.2f}", y=0.98)
        plt.tight_layout()
        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ── Plot 2: frequency sweep ─────────────────────────────────────────────
    def plot_frequency_sweep(
        self, sample=0, n_panels=9, mode="real", figsize=(16,5), show=True, save=None
    ) -> plt.Figure:
        traj = self.get_trajectory_ux(sample)
        indices = np.linspace(0, self._M-1, n_panels, dtype=int)
        X, Y = np.meshgrid(self.gx, self.gy, indexing="ij")
        obst = self.get_obstacle(sample)

        ncols = min(n_panels,9)
        nrows = 1
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.ravel(axes)

        for k, idx in enumerate(indices):
            u = traj[idx]
            data = u.real if mode=="real" else np.abs(u)
            norm = _symm_norm(data) if mode=="real" else None
            ax = axes[k]
            ax.pcolormesh(X,Y,data, cmap=CMAP_FIELD if mode=="real" else CMAP_AMP, norm=norm)
            ax.contour(X,Y,obst, levels=[0.5], colors="w")
            ax.scatter(self.L/6, self.L/2, c="yellow", s=12, edgecolors="k")
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            _label_panel(ax, f"ω={self.omega[idx]:.1f}")

        for ax in axes[n_panels:]: ax.set_visible(False)
        fig.suptitle(f"Frequency sweep (ux) | sample {sample}")
        plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches="tight")
        if show: plt.show()
        return fig

    # ── Plot3: sample comparison ───────────────────────────────────────────
    def plot_sample_comparison(
        self, freq_idx=0, n=6, mode="real", figsize=(15,4), show=True, save=None
    ):
        X,Y = np.meshgrid(self.gx,self.gy,indexing="ij")
        n = min(n, self._N)
        fig, axes = plt.subplots(1,n,figsize=figsize)

        for i in range(n):
            ux = self.get_ux(i, freq_idx)
            data = ux.real if mode=="real" else np.abs(ux)
            norm = _symm_norm(data) if mode=="real" else None
            obst = self.get_obstacle(i)
            ax = axes[i]
            ax.pcolormesh(X,Y,data, cmap=CMAP_FIELD if mode=="real" else CMAP_AMP, norm=norm)
            ax.contour(X,Y,obst, levels=[0.5], colors="w")
            ax.scatter(self.L/6, self.L/2, c="yellow", s=15, edgecolors="k")
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(f"{n} samples at ω={self.omega[freq_idx]:.2f} (ux)")
        plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches="tight")
        if show: plt.show()
        return fig

    # ── Plot4: obstacle + materials ────────────────────────────────────────
    def plot_obstacle_and_materials(self, sample=0, figsize=(13,4), show=True, save=None):
        X,Y = np.meshgrid(self.gx,self.gy,indexing="ij")
        obst = self.get_obstacle(sample)
        lam, mu, rho = self.get_materials(sample)

        fig, axes = plt.subplots(1,4,figsize=figsize)
        axes[0].pcolormesh(X,Y,obst, cmap=CMAP_OBST, vmin=0,vmax=1)
        axes[0].set_title("Obstacle mask")
        axes[1].pcolormesh(X,Y,lam, cmap=CMAP_MAT); axes[1].set_title("λ")
        axes[2].pcolormesh(X,Y,mu, cmap=CMAP_MAT); axes[2].set_title("μ")
        axes[3].pcolormesh(X,Y,rho,cmap=CMAP_MAT); axes[3].set_title("ρ")

        for ax in axes:
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(plt.cm.ScalarMappable(cmap=CMAP_MAT), ax=ax, fraction=0.04)
        axes[0].scatter(self.L/6, self.L/2, c="yellow", s=50, edgecolors="k")

        fig.suptitle(f"Obstacle & Materials | sample {sample}")
        plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches="tight")
        if show: plt.show()
        return fig

    # ── Plot5: wavenumber spectrum ─────────────────────────────────────────
    def plot_wavenumber_spectrum(self, sample=0, freq_idx=0, figsize=(10,4), show=True, save=None):
        ux = self.get_ux(sample, freq_idx)
        omega = self.omega[freq_idx]
        X,Y = np.meshgrid(self.gx,self.gy,indexing="ij")

        U = np.fft.fftshift(np.fft.fft2(ux))
        spec = np.log1p(np.abs(U))
        kx = np.fft.fftshift(np.fft.fftfreq(self._Ng, self.L/self._Ng))*2*np.pi
        KX,KY = np.meshgrid(kx,kx,indexing="ij")

        fig, (ax1,ax2) = plt.subplots(1,2,figsize=figsize)
        ax1.pcolormesh(X,Y,ux.real, cmap=CMAP_FIELD, norm=_symm_norm(ux.real))
        ax1.contour(X,Y,self.get_obstacle(sample), levels=[0.5], colors="w")
        ax1.set_title(f"Re(ux) ω={omega:.1f}")
        ax1.set_aspect("equal")

        ax2.pcolormesh(KX,KY,spec, cmap=CMAP_AMP)
        ax2.set_title("log(1+|FFT(ux)|)")
        ax2.set_aspect("equal")
        ax2.set_xlim(-30,30); ax2.set_ylim(-30,30)

        plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches="tight")
        if show: plt.show()
        return fig

    # ── Plot6: FTM validation ──────────────────────────────────────────────
    def plot_ftm_validation(
        self, n_samples=20, rank=8, freq_idx_check=None, figsize=(13,9), show=True, save=None
    ):
        n_samples = min(n_samples, self._N)
        M = self._M
        Ng = self._Ng

        stack = []
        for i in range(n_samples):
            traj = self.get_trajectory_ux(i).reshape(M, -1).real
            stack.append(traj)
        T = np.concatenate(stack, axis=0)

        U_sv, S, Vt = np.linalg.svd(T, full_matrices=False)
        ranks = [1,2,4,8,16,32,64]
        errs = [np.linalg.norm(T-(U_sv[:,:r]*S[:r])@Vt[:r])/np.linalg.norm(T) for r in ranks]

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2,3,figure=fig)

        ax1 = fig.add_subplot(gs[0,0])
        ax1.semilogy(S[:64]/S[0],"o-"); ax1.axvline(rank,c="r",ls="--")
        ax1.set_title("Singular values")

        ax2 = fig.add_subplot(gs[0,1])
        ax2.semilogy(ranks, errs,"s-"); ax2.axvline(rank,c="r",ls="--")
        ax2.set_title("Reconstruction error")

        ax3 = fig.add_subplot(gs[0,2])
        ax3.plot(np.cumsum(S**2)/np.sum(S**2)*100); ax3.axvline(rank,c="r",ls="--")
        ax3.set_title("Cumulative energy")

        if freq_idx_check is None: freq_idx_check = M//2
        T_approx = (U_sv[:,:rank]*S[:rank])@Vt[:rank]
        orig = T[freq_idx_check].reshape(Ng,Ng)
        recon = T_approx[freq_idx_check].reshape(Ng,Ng)
        err = orig - recon
        X,Y = np.meshgrid(self.gx,self.gy,indexing="ij")

        for i, (d,t) in enumerate([(orig,"Original"),(recon,f"Rank-{rank}"),(err,"Error")]):
            ax = fig.add_subplot(gs[1,i])
            ax.pcolormesh(X,Y,d, cmap=CMAP_FIELD, norm=_symm_norm(d))
            ax.set_title(t); ax.set_aspect("equal")

        plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches="tight")
        if show: plt.show()
        return fig

    # ── Plot7: observation mask ────────────────────────────────────────────
    def plot_observation_mask(self, freq_idx=0, sample=0, figsize=(5,4.5), show=True, save=None):
        if not self.has_mask:
            print("No mask")
            return
        X,Y = np.meshgrid(self.gx,self.gy,indexing="ij")
        mask = self.get_mask_at_freq(freq_idx, sample)
        fig,ax = plt.subplots(figsize=figsize)
        ax.pcolormesh(X,Y,mask, cmap=CMAP_MASK, vmin=0,vmax=1)
        ax.set_aspect("equal")
        ax.set_title(f"Mask | sample {sample}, ω={self.omega[freq_idx]:.2f}")
        if save: fig.savefig(save,dpi=150,bbox_inches="tight")
        if show: plt.show()
        return fig

    # ── Plot8: mask sweep ──────────────────────────────────────────────────
    def plot_mask_sweep(self, n_panels=9, sample=0, figsize=(15,5), show=True, save=None):
        if not self.has_mask:
            print("No mask")
            return
        indices = np.linspace(0,self._M-1,n_panels,dtype=int)
        X,Y = np.meshgrid(self.gx,self.gy,indexing="ij")
        fig,axes = plt.subplots(1, min(n_panels,9), figsize=figsize)
        axes = np.ravel(axes)
        for k,idx in enumerate(indices):
            m = self.get_mask_at_freq(idx,sample)
            axes[k].pcolormesh(X,Y,m,cmap=CMAP_MASK,vmin=0,vmax=1)
            axes[k].set_aspect("equal"); axes[k].set_xticks([]); axes[k].set_yticks([])
            _label_panel(axes[k], f"ω={self.omega[idx]:.1f}")
        fig.suptitle(f"Mask sweep | sample {sample}")
        plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches="tight")
        if show: plt.show()
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "elastic_dataset_obstacle.h5"
    ds = ElasticDataset(path)

    print("\n── Plot 1: ux + uy full field")
    ds.plot_field(sample=0, freq_idx=0, save="elastic_vis_field.png", show=False)

    print("── Plot 2: frequency sweep")
    ds.plot_frequency_sweep(save="elastic_vis_sweep.png", show=False)

    print("── Plot3: sample comparison")
    ds.plot_sample_comparison(n=6, save="elastic_vis_compare.png", show=False)

    print("── Plot4: obstacle + materials")
    ds.plot_obstacle_and_materials(save="elastic_vis_materials.png", show=False)

    print("── Plot5: spectrum")
    ds.plot_wavenumber_spectrum(save="elastic_vis_spectrum.png", show=False)

    print("── Plot6: FTM")
    ds.plot_ftm_validation(rank=40, save="elastic_vis_ftm.png", show=False)

    print("── Plot7: mask")
    ds.plot_observation_mask(save="elastic_vis_mask.png", show=False)

    print("── Plot8: mask sweep")
    ds.plot_mask_sweep(save="elastic_vis_mask_sweep.png", show=False)

    print("\nAll elastic wave plots saved!")