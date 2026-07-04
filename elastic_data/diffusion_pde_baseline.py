"""
diffusion_pde_baseline.py  (elastic wave edition)
--------------------------------------------------
Pixel-space diffusion baseline for 4-channel elastic field reconstruction.
DiffusionPDE-style [Huang et al., NeurIPS 2024]:
  - Train: ω-conditional score network (NO obs/mask in model input)
  - Eval:  DDPM + scale-invariant DPS guidance (fixed λ)

Input to score net: noisy field (4 channels) + ω embedding
Output: noise prediction (4 channels)

Usage
-----
Train:
    python diffusion_pde_baseline.py --mode train \\
        --train_h5 elastic_dataset.h5 --out ckp/dpde_elastic.pt
Eval:
    python diffusion_pde_baseline.py --mode eval \\
        --ckpt ckp/dpde_elastic.pt --test_h5 elastic_dataset.h5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

def set_seed(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); (torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None)

def _normalize_omega(omega, omega_min, omega_max):
    return float((omega - omega_min) / max(omega_max - omega_min, 1e-12))

def _parse_indices(text, limit):
    if text.strip() == "": return list(range(limit))
    out = []
    for p in text.split(","):
        p = p.strip()
        if p: idx = int(p); out.append(idx) if 0 <= idx < limit else (_ for _ in ()).throw(ValueError(f"index {idx}"))
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

class ElasticDPDEData(Dataset):
    def __init__(self, h5_path, sample_indices=None, freq_indices="", return_obs=False):
        self.path = Path(h5_path)
        if not self.path.exists(): raise FileNotFoundError(self.path)
        self.return_obs = return_obs

        with h5py.File(self.path, "r") as f:
            self.omega = f["omega"][...].astype(np.float32)
            self.omega_min = float(np.min(self.omega)); self.omega_max = float(np.max(self.omega))
            self.data = f["data"][...].astype(np.float32)
            self.N, self.M, self.H, self.W, self.C = self.data.shape
            self.has_mask = "mask_tr" in f
            self.mask_data = f["mask_tr"][...].astype(np.float32) if self.has_mask else None

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
        s, m = self.pairs[idx]
        field = self.data[s, m]    # (H,W,4)
        mask  = self._get_mask(s, m)
        omega = float(self.omega[m])
        omega_n = _normalize_omega(omega, self.omega_min, self.omega_max)
        y = field.transpose(2,0,1).astype(np.float32)
        out = {"y": y,
               "mask": mask.transpose(2,0,1).astype(np.float32),  # (1, H, W)
               "omega": np.array([omega_n],dtype=np.float32),
               "sample_idx": np.array(s,dtype=np.int64),
               "freq_idx": np.array(m,dtype=np.int64),
               "omega_raw": np.array(omega,dtype=np.float32)}
        if self.return_obs:
            out["obs"] = (field * mask).transpose(2,0,1).astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# Diffusion schedule + Score network
# ---------------------------------------------------------------------------

class DiffusionSchedule:
    def __init__(self, betas):
        self.betas = betas; al = 1.0 - betas; self.alphas = al
        self.alpha_bars = torch.cumprod(al, 0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    @classmethod
    def linear(cls, T, b0=1e-4, b1=2e-2, device="cpu"):
        return cls(torch.linspace(b0, b1, T, device=device))

    def to(self, device):
        for a in ["betas","alphas","alpha_bars","sqrt_alpha_bars","sqrt_one_minus_alpha_bars"]:
            setattr(self, a, getattr(self, a).to(device))
        return self

    def gather(self, vals, t, x):
        out = vals.gather(0, t).to(dtype=x.dtype)
        return out.view(-1, *([1]*(x.ndim-1)))


def _gn(ch, mg=8):
    g = min(mg, ch)
    while g > 1 and ch % g != 0: g -= 1
    return g


class FourierOmegaEmb(nn.Module):
    def __init__(self, bands=8):
        super().__init__(); self.bands = bands; self.out_dim = 1+2*bands
    def forward(self, w):
        feats = [w]
        for k in range(self.bands):
            f = (2**k)*math.pi; feats += [torch.sin(f*w), torch.cos(f*w)]
        return torch.cat(feats,-1)


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim//2
        freqs = torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/max(half-1,1))
        args = t.float().unsqueeze(1)*freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args),torch.cos(args)],-1)
        return F.pad(emb,(0,1)) if self.dim%2==1 else emb


class ScoreRes(nn.Module):
    def __init__(self, ic, oc, cd):
        super().__init__()
        self.n1=nn.GroupNorm(_gn(ic),ic); self.c1=nn.Conv2d(ic,oc,3,padding=1)
        self.n2=nn.GroupNorm(_gn(oc),oc); self.c2=nn.Conv2d(oc,oc,3,padding=1)
        self.cp=nn.Linear(cd,2*oc)
        self.sk=nn.Conv2d(ic,oc,1) if ic!=oc else nn.Identity()
    def forward(self, x, cond):
        h=self.c1(F.silu(self.n1(x))); h=self.n2(h)
        sc=self.cp(cond); sc,sh=sc.chunk(2,-1)
        h=h*(1+sc.view(-1,sc.shape[-1],1,1))+sh.view(-1,sh.shape[-1],1,1)
        return self.c2(F.silu(h))+self.sk(x)


class PixelScoreUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_channels=64,
                 cond_dim=256, time_dim=128, omega_bands=8):
        super().__init__()
        self.te=SinusoidalTimeEmb(time_dim); self.oe=FourierOmegaEmb(omega_bands)
        self.cm=nn.Sequential(nn.Linear(time_dim+self.oe.out_dim,cond_dim),nn.SiLU(),nn.Linear(cond_dim,cond_dim))
        c1,c2,c3=base_channels,base_channels*2,base_channels*4
        self.st=nn.Conv2d(in_channels,c1,3,padding=1)
        self.e1=ScoreRes(c1,c1,cond_dim); self.d1=nn.Conv2d(c1,c2,3,stride=2,padding=1)
        self.e2=ScoreRes(c2,c2,cond_dim); self.d2=nn.Conv2d(c2,c3,3,stride=2,padding=1)
        self.mid=ScoreRes(c3,c3,cond_dim)
        self.u2=ScoreRes(c3+c2,c2,cond_dim); self.u1=ScoreRes(c2+c1,c1,cond_dim)
        self.on=nn.GroupNorm(_gn(c1),c1); self.oc=nn.Conv2d(c1,out_channels,3,padding=1)
    def forward(self, x, t, omega_norm):
        cond=self.cm(torch.cat([self.te(t),self.oe(omega_norm)],-1))
        e0=self.st(x); e1=self.e1(e0,cond)
        e2=self.e2(self.d1(e1),cond); m=self.mid(self.d2(e2),cond)
        u2=self.u2(torch.cat([F.interpolate(m,e2.shape[-2:],mode="nearest"),e2],1),cond)
        u1=self.u1(torch.cat([F.interpolate(u2,e1.shape[-2:],mode="nearest"),e1],1),cond)
        return self.oc(F.silu(self.on(u1)))


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def _split(n, ratio, seed):
    rng=np.random.default_rng(seed); ids=rng.permutation(n)
    k=max(1,min(int(round(n*ratio)),n-1) if n>1 else n)
    return ids[:k].tolist(), (ids[k:].tolist() or ids[-1:].tolist())

def _collate_train(batch):
    keys=["y","mask","omega","sample_idx","freq_idx","omega_raw"]
    out={k: torch.from_numpy(np.stack([b[k] for b in batch])) for k in keys}
    out["sample_idx"]=out["sample_idx"].long(); out["freq_idx"]=out["freq_idx"].long()
    return out

def _collate_eval(batch):
    keys=["y","omega","sample_idx","freq_idx","omega_raw","mask","obs"]
    out={k: torch.from_numpy(np.stack([b[k] for b in batch])) for k in keys}
    out["sample_idx"]=out["sample_idx"].long(); out["freq_idx"]=out["freq_idx"].long()
    return out

def _load_meta(path):
    with h5py.File(path,"r") as f:
        meta={}
        if "metadata" in f:
            raw=f["metadata"][()]
            try: meta=json.loads(raw if isinstance(raw,str) else raw.decode("utf-8"))
            except Exception: pass
        return {"meta":meta,"omega":f["omega"][...].astype(np.float32),"shape":f["data"].shape}


def train_model(args):
    set_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else torch.device(args.device)
    meta=_load_meta(Path(args.train_h5)); N=int(meta["shape"][0]); C=int(meta["shape"][-1])
    tr_ids,va_ids=_split(N,args.train_ratio,args.seed)
    tr_ds=ElasticDPDEData(args.train_h5,tr_ids,args.train_freq_indices,return_obs=False)
    va_ds=ElasticDPDEData(args.train_h5,va_ids,args.eval_freq_indices,return_obs=False)
    model=PixelScoreUNet(C,C,args.base_channels,args.cond_dim,args.time_dim,args.omega_bands).to(device)
    sched=DiffusionSchedule.linear(args.T,args.beta_start,args.beta_end,device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.wd)
    lr_sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=10,factor=0.5)
    tr_dl=DataLoader(tr_ds,args.batch_size,shuffle=True, num_workers=4,collate_fn=_collate_train)
    va_dl=DataLoader(va_ds,args.batch_size,shuffle=False,num_workers=4,collate_fn=_collate_train)
    best_val=float("inf"); out_path=Path(args.out); out_path.parent.mkdir(parents=True,exist_ok=True)
    print(f"\nDiffusionPDE (elastic) | C={C} base={args.base_channels} T={args.T} device={device}")
    for epoch in range(1,args.epochs+1):
        model.train(); tl=0.0; tn=0
        for batch in tr_dl:
            y=batch["y"].to(device); mask=batch["mask"].to(device)
            omega=batch["omega"].to(device); B=y.shape[0]
            t=torch.randint(0,args.T,(B,),device=device)
            noise=torch.randn_like(y)
            sa=sched.gather(sched.sqrt_alpha_bars,t,y)
            sb=sched.gather(sched.sqrt_one_minus_alpha_bars,t,y)
            eps_pred=model(sa*y+sb*noise,t,omega)
            diff=(eps_pred-noise)**2
            n_obs=mask.sum()*y.shape[1]
            loss=(diff*mask).sum()/n_obs.clamp(min=1)
            opt.zero_grad(set_to_none=True); loss.backward()
            if args.grad_clip>0: nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
            opt.step(); tl+=loss.item()*B; tn+=B
        model.eval(); vl=0.0; vn=0
        with torch.no_grad():
            for batch in va_dl:
                y=batch["y"].to(device); mask=batch["mask"].to(device)
                omega=batch["omega"].to(device); B=y.shape[0]
                t=torch.randint(0,args.T,(B,),device=device); noise=torch.randn_like(y)
                sa=sched.gather(sched.sqrt_alpha_bars,t,y); sb=sched.gather(sched.sqrt_one_minus_alpha_bars,t,y)
                eps_pred=model(sa*y+sb*noise,t,omega)
                diff=(eps_pred-noise)**2; n_obs=mask.sum()*y.shape[1]
                vl+=((diff*mask).sum()/n_obs.clamp(min=1)).item()*B; vn+=B
        tl/=max(tn,1); vl/=max(vn,1); lr_sched.step(vl)
        if vl<best_val:
            best_val=vl
            torch.save({"model_state":model.state_dict(),
                        "model_config":{"in_channels":C,"out_channels":C,"base_channels":args.base_channels,
                                        "cond_dim":args.cond_dim,"time_dim":args.time_dim,"omega_bands":args.omega_bands},
                        "diffusion_config":{"T":args.T,"beta_start":args.beta_start,"beta_end":args.beta_end},
                        "train_config":vars(args),"best_val_loss":best_val},out_path)
        if args.log_every>0 and (epoch==1 or epoch%args.log_every==0 or epoch==args.epochs):
            print(f"[{epoch:04d}/{args.epochs}] train={tl:.4e} val={vl:.4e} best={best_val:.4e}")
    print(f"Done. ckpt={out_path}")


@torch.no_grad()
def dps_sample(model, sched, omega, y_obs, mask, T, zeta, device):
    B=y_obs.shape[0]; x=torch.randn_like(y_obs)
    for t_idx in reversed(range(T)):
        t_vec=torch.full((B,),t_idx,device=device,dtype=torch.long)
        eps_pred=model(x,t_vec,omega)
        ab_t=sched.alpha_bars[t_idx]; a_t=sched.alphas[t_idx]; b_t=sched.betas[t_idx]
        x0hat=(x-(1-ab_t).sqrt()*eps_pred)/ab_t.sqrt().clamp(min=1e-8)
        noise=torch.randn_like(x) if t_idx>0 else torch.zeros_like(x)
        x_prior=(1.0/a_t.sqrt())*(x-((1.0-a_t)/(1.0-ab_t).sqrt())*eps_pred)+b_t.sqrt()*noise
        g_t=mask*(mask*x0hat-y_obs)
        prior_step_norm=(x_prior-x).pow(2).sum(dim=(1,2,3),keepdim=True).sqrt().clamp(min=1e-8)
        g_norm=g_t.pow(2).sum(dim=(1,2,3),keepdim=True).sqrt().clamp(min=1e-8)
        x=x_prior-zeta*prior_step_norm/g_norm*g_t
    return x


def evaluate_model(args):
    set_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else torch.device(args.device)
    ckpt=_safe_load(Path(args.ckpt))
    model=PixelScoreUNet(**ckpt["model_config"]).to(device); model.load_state_dict(ckpt["model_state"]); model.eval()
    dcfg=ckpt.get("diffusion_config",{}); T=dcfg.get("T",args.T)
    sched=DiffusionSchedule.linear(T,dcfg.get("beta_start",args.beta_start),dcfg.get("beta_end",args.beta_end)).to(device)
    data_path=Path(args.test_h5); meta=_load_meta(data_path)
    N=int(meta["shape"][0]); max_s=N if args.max_samples<=0 else min(N,args.max_samples)
    ds=ElasticDPDEData(data_path,list(range(max_s)),args.eval_freq_indices,return_obs=True)
    dl=DataLoader(ds,args.batch_size,shuffle=False,num_workers=0,collate_fn=_collate_eval)
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    rows=[]; vis_cnt=0
    for batch in dl:
        y=batch["y"].to(device); omega=batch["omega"].to(device)
        mask_t=batch["mask"].to(device); obs_t=batch["obs"].to(device)
        s_idx=batch["sample_idx"].cpu().numpy(); f_idx=batch["freq_idx"].cpu().numpy(); omega_r=batch["omega_raw"].cpu().numpy()
        pred_t=dps_sample(model,sched,omega,obs_t,mask_t,T,args.zeta,device)
        pred_np=pred_t.cpu().numpy(); gt_np=y.cpu().numpy(); mask_np=batch["mask"].numpy()
        for i in range(pred_np.shape[0]):
            pred_i=pred_np[i].transpose(1,2,0); gt_i=gt_np[i].transpose(1,2,0); mask_i=mask_np[i].transpose(1,2,0)
            rmse=_relative_rmse(pred_i,gt_i); obs_r=_masked_relative_rmse(pred_i,gt_i,mask_i)
            unobs_r=_masked_relative_rmse(pred_i,gt_i,1.0-mask_i)
            rows.append({"sample_idx":int(s_idx[i]),"freq_idx":int(f_idx[i]),"omega":float(omega_r[i]),
                         "rmse":rmse,"obs_rmse":obs_r,"unobs_rmse":unobs_r})
            if vis_cnt<args.num_visualize:
                vis_cnt+=1
                vp=out_dir/f"case{vis_cnt:03d}_s{int(s_idx[i]):03d}_f{int(f_idx[i]):03d}.png"
                C=pred_i.shape[-1]; lb=["ux_re","ux_im","uy_re","uy_im"][:C]
                fig,axes=plt.subplots(C,3,figsize=(15,5*C))
                for ci in range(C):
                    row=axes[ci] if C>1 else axes
                    for ax,(img,title) in zip(row,[(gt_i[...,ci],f"GT {lb[ci]}"),(pred_i[...,ci],f"Pred {lb[ci]}"),(np.abs(pred_i[...,ci]-gt_i[...,ci]),f"Err {lb[ci]}")]):
                        im=ax.imshow(img,origin="lower",cmap="viridis" if "Err" not in title else "magma")
                        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
                fig.suptitle(f"s={int(s_idx[i])} f={int(f_idx[i])} ω={float(omega_r[i]):.3f} | rmse={rmse:.2e}",y=0.995)
                fig.tight_layout(); fig.savefig(vp,dpi=args.vis_dpi); plt.close(fig)
                rows[-1]["vis_path"]=str(vp)
    csv_path=out_dir/"metrics_cases.csv"
    with open(csv_path,"w",newline="") as f:
        w=csv.DictWriter(f,["sample_idx","freq_idx","omega","rmse","obs_rmse","unobs_rmse","vis_path"])
        w.writeheader()
        for r in rows: w.writerow(r)
    summary={"ckpt":args.ckpt,"test_h5":str(data_path),"num_cases":len(rows),"dps_zeta":args.zeta,
             "mean_rmse":float(np.mean([r["rmse"] for r in rows])),
             "mean_obs_rmse":float(np.mean([r["obs_rmse"] for r in rows])),
             "mean_unobs_rmse":float(np.mean([r["unobs_rmse"] for r in rows]))}
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2))
    print("\nEvaluation finished."); print(json.dumps(summary,indent=2))


def build_parser():
    p=argparse.ArgumentParser("DiffusionPDE baseline — elastic wave")
    p.add_argument("--mode",choices=["train","eval"],default="eval")
    p.add_argument("--train_h5",default="elastic_dataset.h5"); p.add_argument("--test_h5",default="elastic_dataset_msk0.01.h5")
    p.add_argument("--ckpt",default="ckp/dpde_elastic.pt"); p.add_argument("--out",default="ckp/dpde_elastic.pt")
    p.add_argument("--out_dir",default="visual_data/dpde_elastic_eval_msk0.01")
    p.add_argument("--train_ratio",type=float,default=0.8); p.add_argument("--train_freq_indices",type=str,default="")
    p.add_argument("--eval_freq_indices",type=str,default=""); p.add_argument("--max_samples",type=int,default=-1)
    p.add_argument("--epochs",type=int,default=150); p.add_argument("--batch_size",type=int,default=32)
    p.add_argument("--lr",type=float,default=1e-4); p.add_argument("--wd",type=float,default=1e-6)
    p.add_argument("--grad_clip",type=float,default=1.0); p.add_argument("--log_every",type=int,default=10)
    p.add_argument("--base_channels",type=int,default=64); p.add_argument("--cond_dim",type=int,default=256)
    p.add_argument("--time_dim",type=int,default=128); p.add_argument("--omega_bands",type=int,default=8)
    p.add_argument("--T",type=int,default=500); p.add_argument("--beta_start",type=float,default=1e-4)
    p.add_argument("--beta_end",type=float,default=2e-2); p.add_argument("--zeta",type=float,default=0.3)
    p.add_argument("--num_visualize",type=int,default=50); p.add_argument("--vis_dpi",type=int,default=150)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--device",type=str,default="auto")
    return p


def main():
    args=build_parser().parse_args()
    train_model(args) if args.mode=="train" else evaluate_model(args)


if __name__ == "__main__":
    main()
