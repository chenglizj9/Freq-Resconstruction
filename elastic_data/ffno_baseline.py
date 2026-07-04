"""
ffno_baseline.py  (elastic wave edition)
-----------------------------------------
Factorized FNO (F-FNO) baseline for 4-channel elastic field reconstruction.
[Tran et al., "Factorized Fourier Neural Operators", ICLR 2023]

Input channels (8):
  obs_ux_re, obs_ux_im, obs_uy_re, obs_uy_im, mask, omega_norm, x_coord, y_coord
Output (4): ux_re, ux_im, uy_re, uy_im

Usage
-----
Train:
    python ffno_baseline.py --mode train \\
        --train_h5 elastic_dataset.h5 --out ckp/ffno_elastic.pt
Eval:
    python ffno_baseline.py --mode eval \\
        --ckpt ckp/ffno_elastic.pt --test_h5 elastic_dataset.h5
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _normalize_omega(omega, omega_min, omega_max):
    return float((omega - omega_min) / max(omega_max - omega_min, 1e-12))


def _parse_indices(text, limit):
    if text.strip() == "": return list(range(limit))
    out = []
    for p in text.split(","):
        p = p.strip()
        if p:
            idx = int(p)
            if idx < 0 or idx >= limit: raise ValueError(f"index {idx} out of [0,{limit-1}]")
            out.append(idx)
    return sorted(set(out))


def _relative_rmse(pred, gt, eps=1e-12):
    return float(np.sqrt(np.sum((pred-gt)**2) / max(np.sum(gt**2), eps)))


def _masked_relative_rmse(pred, gt, mask, eps=1e-12):
    m = mask.astype(bool)
    if m.ndim == 3 and m.shape[-1] == 1: m = m[..., 0]
    if not np.any(m): return float("nan")
    diff = (pred-gt)**2; gt_sq = gt**2
    if diff.ndim == 3 and diff.shape[-1] > 1: diff = np.sum(diff,-1); gt_sq = np.sum(gt_sq,-1)
    return float(np.sqrt(np.sum(diff[m]) / max(np.sum(gt_sq[m]), eps)))


def _safe_load(path):
    try: return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ElasticFNOData(Dataset):
    def __init__(self, h5_path, sample_indices=None, freq_indices=""):
        self.path = Path(h5_path)
        if not self.path.exists(): raise FileNotFoundError(self.path)

        with h5py.File(self.path, "r") as f:
            self.omega = f["omega"][...].astype(np.float32)
            self.omega_min = float(np.min(self.omega)); self.omega_max = float(np.max(self.omega))
            self.data = f["data"][...].astype(np.float32)
            self.N, self.M, self.H, self.W, self.C = self.data.shape
            self.has_mask = "mask_tr" in f
            self.mask_data = f["mask_tr"][...].astype(np.float32) if self.has_mask else None
            self.gx = f["grid_x"][...].astype(np.float32) if "grid_x" in f \
                else np.linspace(0., 1., self.H, dtype=np.float32)
            self.gy = f["grid_y"][...].astype(np.float32) if "grid_y" in f \
                else np.linspace(0., 1., self.W, dtype=np.float32)

        self.sample_indices = list(range(self.N)) if sample_indices is None else [int(i) for i in sample_indices]
        self.freq_indices = _parse_indices(freq_indices, self.M)
        self.pairs = [(b,m) for b in self.sample_indices for m in self.freq_indices]

    def __len__(self): return len(self.pairs)

    def _get_mask(self, s, m):
        if not self.has_mask: return np.ones((self.H, self.W, 1), dtype=np.float32)
        md = self.mask_data
        if md.ndim == 4: md = md[m]
        elif md.ndim == 5: md = md[s, m]
        if md.shape[-1] > 1: md = md[..., :1]
        return md

    def __getitem__(self, idx):
        s, m  = self.pairs[idx]
        field = self.data[s, m]        # (H,W,4)
        mask  = self._get_mask(s, m)  # (H,W,1)
        omega = float(self.omega[m])
        omega_n = _normalize_omega(omega, self.omega_min, self.omega_max)
        obs = field * mask
        x = np.stack(
            [obs[..., c] for c in range(self.C)]
            + [mask[..., 0],
               np.full((self.H, self.W), omega_n, dtype=np.float32),
               np.broadcast_to(self.gx[:,None], (self.H,self.W)).copy(),
               np.broadcast_to(self.gy[None,:], (self.H,self.W)).copy()],
            axis=0).astype(np.float32)
        y = field.transpose(2,0,1).astype(np.float32)
        return {"x": x, "y": y, "mask": mask.transpose(2,0,1).astype(np.float32),
                "omega": np.array([omega_n],dtype=np.float32),
                "sample_idx": np.array(s,dtype=np.int64),
                "freq_idx": np.array(m,dtype=np.int64),
                "omega_raw": np.array(omega,dtype=np.float32)}


# ---------------------------------------------------------------------------
# Factorized Spectral Conv 2D
# ---------------------------------------------------------------------------

def _cmul1d(x_ft, w):
    xr, xi = x_ft.real, x_ft.imag
    wr, wi = w[..., 0], w[..., 1]
    return torch.complex(
        torch.einsum("bil,iol->bol", xr, wr) - torch.einsum("bil,iol->bol", xi, wi),
        torch.einsum("bil,iol->bol", xr, wi) + torch.einsum("bil,iol->bol", xi, wr))


class FactorizedSpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, m1, m2):
        super().__init__()
        self.in_ch = in_ch; self.out_ch = out_ch; self.m1 = m1; self.m2 = m2
        scale = 1.0 / (in_ch * out_ch)
        self.wx = nn.Parameter(scale * torch.randn(in_ch, out_ch, m1, 2))
        self.wy = nn.Parameter(scale * torch.randn(in_ch, out_ch, m2, 2))

    def forward(self, x):
        B, C, H, W = x.shape
        Fx  = torch.fft.rfft(x, dim=-2)
        mx  = min(self.m1, Fx.shape[-2])
        Fxbw = Fx.permute(0,3,1,2).reshape(B*W, C, Fx.shape[-2])
        out_x = torch.zeros(B*W, self.out_ch, Fx.shape[-2], dtype=torch.cfloat, device=x.device)
        out_x[:,:,:mx] = _cmul1d(Fxbw[:,:,:mx], self.wx[:,:,:mx,:])
        x_hat_x = torch.fft.irfft(out_x, n=H, dim=-1).reshape(B,W,self.out_ch,H).permute(0,2,3,1)

        Fy  = torch.fft.rfft(x, dim=-1)
        my  = min(self.m2, Fy.shape[-1])
        Fybh = Fy.permute(0,2,1,3).reshape(B*H, C, Fy.shape[-1])
        out_y = torch.zeros(B*H, self.out_ch, Fy.shape[-1], dtype=torch.cfloat, device=x.device)
        out_y[:,:,:my] = _cmul1d(Fybh[:,:,:my], self.wy[:,:,:my,:])
        x_hat_y = torch.fft.irfft(out_y, n=W, dim=-1).reshape(B,H,self.out_ch,W).permute(0,2,1,3)

        return x_hat_x + x_hat_y


class FFNOBlock(nn.Module):
    def __init__(self, width, m1, m2):
        super().__init__()
        self.sp = FactorizedSpectralConv2d(width, width, m1, m2)
        self.w  = nn.Conv2d(width, width, 1)
        self.n  = nn.GroupNorm(min(8, width), width)

    def forward(self, x): return F.gelu(self.n(self.sp(x) + self.w(x)))


class FFNO2d(nn.Module):
    def __init__(self, in_channels=8, out_channels=4, width=64, modes1=16, modes2=16,
                 n_layers=4, padding=4):
        super().__init__()
        self.padding = padding
        self.fc0 = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.ModuleList([FFNOBlock(width, modes1, modes2) for _ in range(n_layers)])
        self.fc1 = nn.Conv2d(width, width*2, 1)
        self.fc2 = nn.Conv2d(width*2, out_channels, 1)

    def forward(self, x):
        if self.padding > 0: x = F.pad(x, (0,self.padding,0,self.padding))
        x = self.fc0(x)
        for b in self.blocks: x = b(x)
        x = F.gelu(self.fc1(x)); x = self.fc2(x)
        if self.padding > 0: x = x[...,:-self.padding,:-self.padding]
        return x


# ---------------------------------------------------------------------------
# Helpers / train / eval
# ---------------------------------------------------------------------------

def _split(n, ratio, seed):
    rng = np.random.default_rng(seed); ids = rng.permutation(n)
    k = max(1, min(int(round(n*ratio)), n-1) if n>1 else n)
    return ids[:k].tolist(), (ids[k:].tolist() or ids[-1:].tolist())


def _collate(batch):
    keys = ["x","y","mask","omega","sample_idx","freq_idx","omega_raw"]
    out = {k: torch.from_numpy(np.stack([b[k] for b in batch])) for k in keys}
    out["sample_idx"] = out["sample_idx"].long(); out["freq_idx"] = out["freq_idx"].long()
    return out


def _load_meta(path):
    with h5py.File(path, "r") as f:
        meta = {}
        if "metadata" in f:
            raw = f["metadata"][()]
            try: meta = json.loads(raw if isinstance(raw,str) else raw.decode("utf-8"))
            except Exception: pass
        omega = f["omega"][...].astype(np.float32)
        shape = f["data"].shape if "data" in f else f["fields_real"].shape
    return {"meta": meta, "omega": omega, "shape": shape}


def train_model(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    meta = _load_meta(Path(args.train_h5)); N = int(meta["shape"][0])
    tr_ids, va_ids = _split(N, args.train_ratio, args.seed)
    tr_ds = ElasticFNOData(args.train_h5, tr_ids, args.train_freq_indices)
    va_ds = ElasticFNOData(args.train_h5, va_ids, args.eval_freq_indices)
    in_ch, out_ch = tr_ds.C + 4, tr_ds.C

    model = FFNO2d(in_ch, out_ch, args.width, args.modes1, args.modes2, args.n_layers, args.padding).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    tr_dl = DataLoader(tr_ds, args.batch_size, shuffle=True,  num_workers=4, collate_fn=_collate)
    va_dl = DataLoader(va_ds, args.batch_size, shuffle=False, num_workers=4, collate_fn=_collate)

    best_val = float("inf")
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nF-FNO (elastic) | in={in_ch} out={out_ch} width={args.width} device={device}")

    for epoch in range(1, args.epochs+1):
        model.train(); tl = 0.0; tn = 0
        for b in tr_dl:
            x, y = b["x"].to(device), b["y"].to(device)
            mask = b["mask"].to(device)
            loss = F.mse_loss(model(x) * mask, y * mask)
            opt.zero_grad(set_to_none=True); loss.backward()
            if args.grad_clip > 0: nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); tl += loss.item()*x.shape[0]; tn += x.shape[0]
        model.eval(); vl = 0.0; vn = 0
        with torch.no_grad():
            for b in va_dl:
                x, y = b["x"].to(device), b["y"].to(device)
                vl += F.mse_loss(model(x), y).item()*x.shape[0]; vn += x.shape[0]
        tl /= max(tn,1); vl /= max(vn,1); sched.step(vl)
        if vl < best_val:
            best_val = vl
            torch.save({"model_state": model.state_dict(),
                        "model_config": {"in_channels": in_ch,"out_channels": out_ch,
                                         "width": args.width,"modes1": args.modes1,"modes2": args.modes2,
                                         "n_layers": args.n_layers,"padding": args.padding},
                        "train_config": vars(args),"best_val_loss": best_val}, out_path)
        if args.log_every > 0 and (epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs):
            print(f"[{epoch:04d}/{args.epochs}] train={tl:.4e} val={vl:.4e} best={best_val:.4e}")
    print(f"Done. ckpt={out_path}")


def evaluate_model(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    ckpt  = _safe_load(Path(args.ckpt))
    model = FFNO2d(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    data_path = Path(args.test_h5); meta = _load_meta(data_path)
    N = int(meta["shape"][0]); max_s = N if args.max_samples <= 0 else min(N, args.max_samples)
    ds = ElasticFNOData(data_path, list(range(max_s)), args.eval_freq_indices)
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []; vis_cnt = 0
    with torch.no_grad():
        for batch in dl:
            x, y = batch["x"].to(device), batch["y"].to(device)
            mask = batch["mask"].cpu().numpy()
            s_idx = batch["sample_idx"].cpu().numpy(); f_idx = batch["freq_idx"].cpu().numpy()
            omega_r = batch["omega_raw"].cpu().numpy()
            pred = model(x).cpu().numpy(); gt = y.cpu().numpy()
            for i in range(pred.shape[0]):
                pred_i = pred[i].transpose(1,2,0); gt_i = gt[i].transpose(1,2,0)
                mask_i = mask[i].transpose(1,2,0)
                rmse = _relative_rmse(pred_i, gt_i)
                obs_r = _masked_relative_rmse(pred_i, gt_i, mask_i)
                unobs_r = _masked_relative_rmse(pred_i, gt_i, 1.0 - mask_i)
                rows.append({"sample_idx": int(s_idx[i]),"freq_idx": int(f_idx[i]),
                             "omega": float(omega_r[i]),"rmse": rmse,
                             "obs_rmse": obs_r,"unobs_rmse": unobs_r})
                if vis_cnt < args.num_visualize:
                    vis_cnt += 1
                    vp = out_dir / f"case{vis_cnt:03d}_s{int(s_idx[i]):03d}_f{int(f_idx[i]):03d}.png"
                    C = pred_i.shape[-1]; labels = ["ux_re","ux_im","uy_re","uy_im"][:C]
                    fig, axes = plt.subplots(C, 3, figsize=(15,5*C))
                    for ci in range(C):
                        row = axes[ci] if C>1 else axes
                        for ax,(img,title) in zip(row,[(gt_i[...,ci],f"GT {labels[ci]}"),(pred_i[...,ci],f"Pred {labels[ci]}"),(np.abs(pred_i[...,ci]-gt_i[...,ci]),f"Err {labels[ci]}")]):
                            im = ax.imshow(img,origin="lower",cmap="viridis" if "Err" not in title else "magma")
                            ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
                            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    fig.suptitle(f"s={int(s_idx[i])} f={int(f_idx[i])} ω={float(omega_r[i]):.3f} | rmse={rmse:.2e}",y=0.995)
                    fig.tight_layout(); fig.savefig(vp, dpi=args.vis_dpi); plt.close(fig)
                    rows[-1]["vis_path"] = str(vp)

    csv_path = out_dir / "metrics_cases.csv"
    with open(csv_path,"w",newline="") as f:
        w = csv.DictWriter(f,["sample_idx","freq_idx","omega","rmse","obs_rmse","unobs_rmse","vis_path"])
        w.writeheader()
        for r in rows: w.writerow(r)
    summary = {"ckpt": args.ckpt,"test_h5": str(data_path),"num_cases": len(rows),
               "mean_rmse": float(np.mean([r["rmse"] for r in rows])),
               "mean_obs_rmse": float(np.mean([r["obs_rmse"] for r in rows])),
               "mean_unobs_rmse": float(np.mean([r["unobs_rmse"] for r in rows]))}
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2))
    print("\nEvaluation finished."); print(json.dumps(summary,indent=2))


def build_parser():
    p = argparse.ArgumentParser("F-FNO baseline — elastic wave")
    p.add_argument("--mode", choices=["train","eval"], default="eval")
    p.add_argument("--train_h5", default="elastic_dataset.h5")
    p.add_argument("--test_h5",  default="elastic_dataset_msk0.01.h5")
    p.add_argument("--ckpt",     default="ckp/ffno_elastic.pt")
    p.add_argument("--out",      default="ckp/ffno_elastic.pt")
    p.add_argument("--out_dir",  default="visual_data/ffno_elastic_eval_msk0.01")
    p.add_argument("--train_ratio",type=float,default=0.8)
    p.add_argument("--train_freq_indices",type=str,default="")
    p.add_argument("--eval_freq_indices", type=str,default="")
    p.add_argument("--max_samples",type=int,default=-1)
    p.add_argument("--epochs",type=int,default=50); p.add_argument("--batch_size",type=int,default=32)
    p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--wd",type=float,default=1e-6)
    p.add_argument("--grad_clip",type=float,default=1.0); p.add_argument("--log_every",type=int,default=5)
    p.add_argument("--width",type=int,default=64); p.add_argument("--modes1",type=int,default=12)
    p.add_argument("--modes2",type=int,default=12); p.add_argument("--n_layers",type=int,default=4)
    p.add_argument("--padding",type=int,default=4); p.add_argument("--num_visualize",type=int,default=50)
    p.add_argument("--vis_dpi",type=int,default=150); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",type=str,default="auto")
    return p


def main():
    args = build_parser().parse_args()
    train_model(args) if args.mode == "train" else evaluate_model(args)


if __name__ == "__main__":
    main()
