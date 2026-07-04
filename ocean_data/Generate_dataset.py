"""
Generate_dataset.py  (ocean acoustic version)
---------------------------------------------
Generate a 2-D frequency-domain *ocean acoustic* dataset by driving the
`ocean-acoustic-agent` (Python orchestration + UnderwaterAcoustics.jl /
AcousticsToolbox.jl) and packing the complex acoustic pressure field into the
SAME FTM-ready HDF5 layout consumed by the existing pipeline
(`train_FTM_GPU.py` / `train_diffusion.py` / `test_diffusion.py`).

Physical mapping (ocean acoustic  ->  Freq_Reconstruction problem)
------------------------------------------------------------------
    frequency sweep   f_hz            ->  omega axis  (M frequencies)
    different env.     (per sample)   ->  N samples   (BC/IC analogue)
    2-D plane (x,y)                   ->  (depth z, range r) grid  (H x W)
    complex pressure  p = Re + j Im   ->  channels C = 2  [real, imag]

The agent computes the complex pressure field p(z, r) on a (depth x range)
receiver grid via `output_requests=["pressure_field"]`.  We sweep frequency in
an outer loop and stack the per-frequency fields into (N, M, H, W, 2).

Per-sample random sampling (every factor has an independent ON/OFF toggle)
--------------------------------------------------------------------------
    --vary_water_depth     water_depth_m   ~ U[wd_min, wd_max]
    --vary_source_depth    source_depth_m  ~ U[sd_min, sd_max]
    --vary_ssp             sound-speed profile (isovelocity speed, or thermocline)
    --vary_seabed          seabed type     ~ choice(seabed pool)
    --vary_obstacle        range-dependent bathymetry (seamount)  [needs Bellhop]

When a factor is OFF the corresponding fixed default is used for every sample.

Propagation model is auto-selected per sample:
    * range-dependent bathymetry (obstacle) OR a thermocline SSP  -> "bellhop"
      (requires AcousticsToolbox.jl)
    * otherwise                                                   -> "pekeris_ray"
      (range-independent, isovelocity; fast)

HDF5 layout (identical keys to the Helmholtz / elastic datasets)
----------------------------------------------------------------
    /metadata     JSON str (generation params + per-sample env list)
    /omega        (M,)            float64   frequencies in Hz
    /grid_x       (H,)            float64   depth axis   (normalised [0,1])
    /grid_y       (W,)            float64   range axis   (normalised [0,1])
    /data         (N, M, H, W, 2) float32  [Re(p), Im(p)]   (per-freq normalised)
    /mask_tr      sparse obs mask (see mask_mode)
    /data_scale   ()              float32
A sidecar  <out>_metadata.npy  mirrors the Helmholtz/elastic convention.

Usage
-----
    # tiny smoke test (sequential)
    python Generate_dataset.py --N 4 --grid_h 24 --grid_w 32 \
        --freq_train 100 200 300 --freq_extrap 400 \
        --vary_ssp --vary_source_depth --out ocean_dataset_smoke.h5

    # vary only a couple of factors
    python Generate_dataset.py --N 200 --vary_water_depth --vary_seabed
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Agent imports are deferred to runtime so that `--help` works without Julia.
# ─────────────────────────────────────────────────────────────────────────────


def _import_agent():
    """Import the ocean-acoustic-agent lazily and return the symbols we need."""
    from ocean_acoustic_agent import SimulationTask, run_simulation
    from ocean_acoustic_agent.schemas.task import ReceiverGrid
    from ocean_acoustic_agent.schemas.environment import (
        EnvironmentSpec, SoundSpeedProfile, SoundSpeedPoint, SeabedType,
        BathymetryProfile, BathymetryPoint,
    )
    return dict(
        SimulationTask=SimulationTask, run_simulation=run_simulation,
        ReceiverGrid=ReceiverGrid, EnvironmentSpec=EnvironmentSpec,
        SoundSpeedProfile=SoundSpeedProfile, SoundSpeedPoint=SoundSpeedPoint,
        SeabedType=SeabedType, BathymetryProfile=BathymetryProfile,
        BathymetryPoint=BathymetryPoint,
    )


# Seabed pools (AT models reject rigid / pressure_release).
_SEABED_PEKERIS = ["rigid", "rock", "sand", "sandy_clay"]
_SEABED_AT      = ["sandy_clay", "sand", "rock"]


# ─────────────────────────────────────────────────────────────────────────────
# Sparse observation mask builder  (identical API to Helmholtz / elastic)
# ─────────────────────────────────────────────────────────────────────────────

def build_sparse_mask(
    rng: np.random.Generator,
    N: int, M: int, H: int, W: int,
    channels: int, obs_ratio: float, mask_mode: str,
) -> np.ndarray:
    """Return mask_tr; shared modes -> (M,H,W,C), per-sample -> (N,M,H,W,C)."""
    if not (0.0 < obs_ratio <= 1.0):
        raise ValueError("obs_ratio must be in (0, 1].")

    if mask_mode == "fixed":
        spatial = (rng.random((H, W)) < obs_ratio).astype(np.uint8)
        mask = np.repeat(spatial[None, :, :, None], M, axis=0)
    elif mask_mode == "per_freq":
        mask = (rng.random((M, H, W, 1)) < obs_ratio).astype(np.uint8)
    elif mask_mode == "per_sample_fixed":
        spatial = (rng.random((N, H, W, 1)) < obs_ratio).astype(np.uint8)
        mask = np.repeat(spatial[:, None, :, :, :], M, axis=1)
    elif mask_mode in {"per_sample", "per_sample_per_freq"}:
        mask = (rng.random((N, M, H, W, 1)) < obs_ratio).astype(np.uint8)
    else:
        raise ValueError("mask_mode must be fixed/per_freq/per_sample_fixed/per_sample_per_freq")

    return np.repeat(mask, channels, axis=-1).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample environment sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_environment(rng: np.random.Generator, cfg: dict, A: dict) -> dict:
    """Draw one random environment description (independent per-factor toggles).

    Returns a plain dict describing the environment so the worker can build the
    agent objects and we can serialise it into metadata for reproducibility.
    """
    # ── water depth ─────────────────────────────────────────────────────────
    if cfg["vary_water_depth"]:
        water_depth = float(rng.uniform(*cfg["wd_range"]))
    else:
        water_depth = float(cfg["wd_fixed"])

    # ── source depth (always kept strictly inside the water column) ──────────
    if cfg["vary_source_depth"]:
        sd_hi = min(cfg["sd_range"][1], water_depth - 2.0)
        sd_lo = min(cfg["sd_range"][0], sd_hi)
        source_depth = float(rng.uniform(sd_lo, sd_hi))
    else:
        source_depth = float(min(cfg["sd_fixed"], water_depth - 2.0))

    # ── sound-speed profile ──────────────────────────────────────────────────
    ssp_mode = "isovelocity"
    ssp_speed = float(cfg["c_fixed"])
    ssp_points: List[Tuple[float, float]] = []
    if cfg["vary_ssp"]:
        if cfg["ssp_profile"]:  # thermocline-style 1-D profile (needs Bellhop)
            ssp_mode = "thermocline"
            c_surf = float(rng.uniform(1500.0, 1515.0))
            c_bot = float(rng.uniform(1480.0, 1495.0))
            d_thermo = float(rng.uniform(0.25, 0.55)) * water_depth
            c_thermo = 0.5 * (c_surf + c_bot)
            ssp_points = [
                (0.0, c_surf),
                (round(d_thermo, 2), round(c_thermo, 2)),
                (round(water_depth, 2), c_bot),
            ]
        else:                    # vary the isovelocity speed
            ssp_speed = float(rng.uniform(*cfg["c_range"]))

    # ── obstacle / range-dependent bathymetry (seamount) ─────────────────────
    has_obstacle = False
    bathy_points: List[Tuple[float, float]] = []
    if cfg["vary_obstacle"]:
        has_obstacle = True
        r_end = cfg["range_end"]
        n_hills = int(rng.integers(1, 3))
        knots = {0.0: water_depth, r_end: water_depth}
        for _ in range(n_hills):
            r_c = float(rng.uniform(0.25, 0.75) * r_end)
            half = float(rng.uniform(0.06, 0.14) * r_end)
            rise = float(rng.uniform(0.30, 0.60) * water_depth)  # seamount height
            top_depth = max(water_depth - rise, source_depth + 5.0, 15.0)
            knots[max(r_c - half, 1.0)] = water_depth
            knots[r_c] = top_depth
            knots[min(r_c + half, r_end - 1.0)] = water_depth
        bathy_points = sorted(knots.items())

    # ── model selection ──────────────────────────────────────────────────────
    needs_at = has_obstacle or ssp_mode == "thermocline"
    model_name = "bellhop" if needs_at else cfg["pekeris_model"]

    # ── seabed (sampled from the model-appropriate pool) ─────────────────────
    pool = _SEABED_AT if needs_at else _SEABED_PEKERIS
    if cfg["vary_seabed"]:
        seabed = str(rng.choice(pool))
    else:
        seabed = cfg["seabed_fixed"]
        if needs_at and seabed in {"rigid", "pressure_release"}:
            seabed = "sandy_clay"  # remap incompatible default for AT models

    return dict(
        water_depth_m=water_depth,
        source_depth_m=source_depth,
        ssp_mode=ssp_mode,
        ssp_speed=ssp_speed,
        ssp_points=ssp_points,
        seabed=seabed,
        has_obstacle=has_obstacle,
        bathy_points=bathy_points,
        model_name=model_name,
    )


def _build_task(env: dict, freq_hz: float, cfg: dict, A: dict):
    """Translate an env dict + frequency into a SimulationTask."""
    water_depth = env["water_depth_m"]
    depth_end = max(water_depth - 1.0, 2.0)

    if env["ssp_mode"] == "thermocline":
        ssp = A["SoundSpeedProfile"](
            points=[A["SoundSpeedPoint"](depth_m=d, speed_mps=c) for d, c in env["ssp_points"]],
            interpolation="linear",
        )
    else:
        ssp = A["SoundSpeedProfile"].isovelocity(env["ssp_speed"])

    env_kwargs = dict(
        water_depth_m=water_depth,
        ssp=ssp,
        seabed=A["SeabedType"](kind=env["seabed"]),
    )
    if env["has_obstacle"] and env["bathy_points"]:
        env_kwargs["bathymetry"] = A["BathymetryProfile"](
            points=[A["BathymetryPoint"](range_m=r, depth_m=d) for r, d in env["bathy_points"]]
        )

    grid = A["ReceiverGrid"](
        range_start_m=cfg["range_start"], range_end_m=cfg["range_end"], range_steps=cfg["W"],
        depth_start_m=1.0, depth_end_m=depth_end, depth_steps=cfg["H"],
    )
    return A["SimulationTask"](
        frequency_hz=float(freq_hz),
        source_depth_m=env["source_depth_m"],
        receiver_depth_m=float(min(30.0, depth_end)),
        receiver_range_m=cfg["range_end"],
        model_name=env["model_name"],
        environment=A["EnvironmentSpec"](**env_kwargs),
        output_requests=["pressure_field"],
        receiver_grid=grid,
    )


def _run_and_extract(task, A, cfg) -> Tuple[np.ndarray, np.ndarray, str]:
    """Run one simulation, return (Re(H,W), Im(H,W), status).  Cleans case dir."""
    result = A["run_simulation"](task)
    case_dir = result.case_dir
    try:
        if result.complex_field is None:
            return None, None, f"no_complex_field: {result.error_message}"
        re = np.asarray(result.complex_field["pressure_real"], dtype=np.float32)
        im = np.asarray(result.complex_field["pressure_imag"], dtype=np.float32)
        if re.shape != (cfg["H"], cfg["W"]):
            return None, None, f"bad_shape: {re.shape}"
        return re, im, "ok"
    finally:
        if cfg["clean_runs"] and case_dir is not None:
            shutil.rmtree(case_dir, ignore_errors=True)


def compute_one_sample(sample_id: int, cfg: dict, freq_list: List[float], seed: int) -> dict:
    """Solve all M frequencies for one sample (one environment)."""
    A = _import_agent()
    rng = np.random.default_rng(seed + 1000 * (sample_id + 1))
    env = sample_environment(rng, cfg, A)

    H, W, M = cfg["H"], cfg["W"], len(freq_list)
    re_arr = np.zeros((M, H, W), dtype=np.float32)
    im_arr = np.zeros((M, H, W), dtype=np.float32)

    fails = []
    for j, f_hz in enumerate(freq_list):
        task = _build_task(env, f_hz, cfg, A)
        re, im, status = _run_and_extract(task, A, cfg)
        if status != "ok":
            # robust fallback: most-compatible config (pekeris + rigid + isovelocity)
            fb = dict(env, model_name="pekeris_ray", seabed="rigid",
                      ssp_mode="isovelocity", has_obstacle=False, bathy_points=[])
            re, im, status = _run_and_extract(_build_task(fb, f_hz, cfg, A), A, cfg)
            if status != "ok":
                raise RuntimeError(
                    f"sample {sample_id} freq {f_hz}Hz failed even after fallback: {status}")
            fails.append({"freq_hz": f_hz, "status": "fallback"})
        re_arr[j] = re
        im_arr[j] = im

    return dict(sample_id=sample_id, env=env, re=re_arr, im=im_arr, fails=fails)


# ─────────────────────────────────────────────────────────────────────────────
# Main generation routine
# ─────────────────────────────────────────────────────────────────────────────

def generate_ocean_dataset(
    num_samples: int,
    freq_train: Sequence[float],
    freq_extrap: Sequence[float],
    H: int, W: int,
    out: str,
    seed: int,
    cfg: dict,
    n_jobs: int = 1,
    obs_ratio: float = 0.1,
    mask_mode: str = "per_sample_fixed",
    normalize: bool = True,
    metadata_out: str = "",
) -> Path:
    freq_list = [float(x) for x in list(freq_train) + list(freq_extrap)]
    M = len(freq_list)
    cfg = dict(cfg, H=H, W=W)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'-'*64}")
    print(f"  Ocean Acoustic 2-D Dataset Generation")
    print(f"  N={num_samples} samples x M={M} freqs x grid={H}x{W} (depth x range)")
    print(f"  freq_train={list(freq_train)}  freq_extrap={list(freq_extrap)} (Hz)")
    print(f"  toggles: depth={cfg['vary_water_depth']} src={cfg['vary_source_depth']} "
          f"ssp={cfg['vary_ssp']}(profile={cfg['ssp_profile']}) "
          f"seabed={cfg['vary_seabed']} obstacle={cfg['vary_obstacle']}")
    print(f"  output -> {out_path}")
    print(f"{'-'*64}\n")

    t0 = time.time()

    if n_jobs == 1:
        results = []
        for sid in range(num_samples):
            r = compute_one_sample(sid, cfg, freq_list, seed)
            results.append(r)
            n_fb = sum(len(rr["fails"]) for rr in [r])
            print(f"  [{sid+1:>4}/{num_samples}] model={r['env']['model_name']:11s} "
                  f"wd={r['env']['water_depth_m']:6.1f} seabed={r['env']['seabed']:13s} "
                  f"{'(fallbacks:%d)' % n_fb if n_fb else ''}")
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(compute_one_sample)(sid, cfg, freq_list, seed)
            for sid in range(num_samples)
        )
        results.sort(key=lambda r: r["sample_id"])

    elapsed = time.time() - t0
    print(f"\n  Solved {num_samples * M} fields in {elapsed:.1f}s "
          f"({elapsed / max(num_samples * M, 1) * 1000:.0f} ms/field)")

    # ── assemble (N, M, H, W, 2) ────────────────────────────────────────────
    re_all = np.stack([r["re"] for r in results], axis=0)  # (N,M,H,W)
    im_all = np.stack([r["im"] for r in results], axis=0)
    data = np.stack([re_all, im_all], axis=-1).astype(np.float32)  # (N,M,H,W,2)

    # ── per-frequency normalisation (matches elastic) ───────────────────────
    scale = 1.0
    if normalize:
        max_per_freq = np.max(np.abs(data), axis=(0, 2, 3, 4), keepdims=True)  # (1,M,1,1,1)
        max_per_freq = np.maximum(max_per_freq, 1e-12)
        data = (data / max_per_freq).astype(np.float32)

    # ── shared normalised coordinate axes ───────────────────────────────────
    grid_x = np.linspace(0.0, 1.0, H, dtype=np.float64)  # depth axis  (len H)
    grid_y = np.linspace(0.0, 1.0, W, dtype=np.float64)  # range axis  (len W)

    # ── sparse observation mask ─────────────────────────────────────────────
    rng_mask = np.random.default_rng(seed + 999_983)
    mask_tr = build_sparse_mask(rng_mask, num_samples, M, H, W,
                                channels=2, obs_ratio=obs_ratio, mask_mode=mask_mode)

    # ── metadata ────────────────────────────────────────────────────────────
    samples_meta = [
        dict(sample_id=r["sample_id"],
             model_name=r["env"]["model_name"],
             water_depth_m=r["env"]["water_depth_m"],
             source_depth_m=r["env"]["source_depth_m"],
             seabed=r["env"]["seabed"],
             ssp_mode=r["env"]["ssp_mode"],
             ssp_speed=r["env"]["ssp_speed"],
             has_obstacle=r["env"]["has_obstacle"],
             n_fallbacks=len(r["fails"]))
        for r in results
    ]
    meta = dict(
        equation="frequency_domain_ocean_acoustic_2d",
        backend="ocean-acoustic-agent (UnderwaterAcoustics.jl / AcousticsToolbox.jl)",
        field="complex acoustic pressure p(z,r) = real + j*imag (dimensionless transfer fn)",
        channels=["real", "imag"],
        N=num_samples, M=M, H=H, W=W,
        grid_layout="(depth x range); grid_x=depth axis (len H), grid_y=range axis (len W)",
        freq_train=list(freq_train), freq_extrap=list(freq_extrap),
        range_m=[cfg["range_start"], cfg["range_end"]],
        toggles=dict(vary_water_depth=cfg["vary_water_depth"],
                     vary_source_depth=cfg["vary_source_depth"],
                     vary_ssp=cfg["vary_ssp"], ssp_profile=cfg["ssp_profile"],
                     vary_seabed=cfg["vary_seabed"], vary_obstacle=cfg["vary_obstacle"]),
        fixed=dict(wd_fixed=cfg["wd_fixed"], sd_fixed=cfg["sd_fixed"],
                   c_fixed=cfg["c_fixed"], seabed_fixed=cfg["seabed_fixed"]),
        obs_ratio=float(obs_ratio), mask_mode=mask_mode,
        normalize=bool(normalize), data_scale=float(scale),
        seed=seed, elapsed_seconds=round(elapsed, 2),
        samples=samples_meta,
    )

    # ── write HDF5 ──────────────────────────────────────────────────────────
    import h5py
    gz = {"compression": "gzip", "compression_opts": 4}
    with h5py.File(out_path, "w") as f:
        f.create_dataset("metadata", data=json.dumps(meta))
        f.create_dataset("omega", data=np.array(freq_list, dtype=np.float64))
        f.create_dataset("grid_x", data=grid_x)
        f.create_dataset("grid_y", data=grid_y)
        f.create_dataset("data", data=data, dtype=np.float32, **gz)
        f.create_dataset("mask_tr", data=mask_tr, dtype=np.uint8, **gz)
        f.create_dataset("data_scale", data=np.array(scale, dtype=np.float32))

    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved {out_path}  ({size_mb:.1f} MB)")

    # ── sidecar .npy (mirrors Helmholtz / elastic convention) ───────────────
    meta_ftm = {"data": {
        "u_ind_uni": grid_x.astype(np.float32),
        "v_ind_uni": grid_y.astype(np.float32),
        "w_ind_uni": np.arange(2, dtype=np.float32),
        "t_ind_uni": np.array(freq_list, dtype=np.float32),
        "data_scale": float(scale),
        "obs_ratio": float(obs_ratio),
        "mask_mode": mask_mode,
        "mask_tr_shape": tuple(int(v) for v in mask_tr.shape),
        "mask_shared_across_samples": bool(mask_tr.ndim == 4),
        "normalize": bool(normalize),
        "channels": ["real", "imag"],
    }}
    if mask_tr.ndim == 4:
        meta_ftm["data"]["mask_tr"] = mask_tr.astype(np.float32)
    meta_path = Path(metadata_out) if metadata_out else out_path.with_name(
        f"{out_path.stem}_metadata.npy")
    np.save(meta_path, meta_ftm, allow_pickle=True)
    print(f"  Saved {meta_path}\n")

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Generate 2-D ocean acoustic dataset (HDF5)")
    # scale / grid
    p.add_argument("--N", type=int, default=800)
    p.add_argument("--grid_h", type=int, default=96, help="depth_steps (H)")
    p.add_argument("--grid_w", type=int, default=128, help="range_steps (W)")
    p.add_argument("--freq_train", type=float, nargs="+",
                   default=[100, 150, 200, 250, 300, 350, 400])
    p.add_argument("--freq_extrap", type=float, nargs="*", default=[500, 600])
    p.add_argument("--range_start", type=float, default=50.0)
    p.add_argument("--range_end", type=float, default=5000.0)
    # toggles
    p.add_argument("--vary_water_depth", action="store_true")
    p.add_argument("--vary_source_depth", action="store_true")
    p.add_argument("--vary_ssp", action="store_true")
    p.add_argument("--ssp_profile", action="store_true",
                   help="when --vary_ssp: use thermocline profiles (Bellhop) instead of "
                        "varying the isovelocity speed")
    p.add_argument("--vary_seabed", action="store_true")
    p.add_argument("--vary_obstacle", action="store_true",
                   help="range-dependent seamount bathymetry (requires AcousticsToolbox.jl)")
    # fixed defaults (used when a factor is OFF)
    p.add_argument("--wd_fixed", type=float, default=100.0)
    p.add_argument("--sd_fixed", type=float, default=10.0)
    p.add_argument("--c_fixed", type=float, default=1500.0)
    p.add_argument("--seabed_fixed", type=str, default="rigid")
    p.add_argument("--pekeris_model", type=str, default="pekeris_ray",
                   choices=["pekeris_ray", "pekeris_mode"])
    # sampling ranges (used when a factor is ON)
    p.add_argument("--wd_range", type=float, nargs=2, default=[80.0, 180.0])
    p.add_argument("--sd_range", type=float, nargs=2, default=[5.0, 50.0])
    p.add_argument("--c_range", type=float, nargs=2, default=[1475.0, 1500.0])
    # mask / normalisation
    p.add_argument("--obs_ratio", type=float, default=0.2)
    p.add_argument("--mask_mode", type=str, default="per_sample_fixed",
                   choices=["fixed", "per_freq", "per_sample_fixed", "per_sample_per_freq"])
    p.add_argument("--disable_normalize", action="store_true")
    # io / runtime
    p.add_argument("--out", type=str, default="ocean_dataset.h5")
    p.add_argument("--metadata_out", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_jobs", type=int, default=1,
                   help="parallel samples; >1 spins up one Julia runtime per worker (heavy)")
    p.add_argument("--keep_runs", action="store_true",
                   help="keep per-call agent output dirs (default: delete to save disk)")
    p.add_argument("--output_dir", type=str, default="",
                   help="scratch dir for the agent's per-call outputs (default: a temp dir)")
    args = p.parse_args()

    # redirect the agent's auto-save dir to scratch (cleaned per call unless --keep_runs)
    scratch = args.output_dir or tempfile.mkdtemp(prefix="ocean_agent_runs_")
    os.environ["OUTPUT_DIR"] = scratch

    cfg = dict(
        vary_water_depth=args.vary_water_depth,
        vary_source_depth=args.vary_source_depth,
        vary_ssp=args.vary_ssp, ssp_profile=args.ssp_profile,
        vary_seabed=args.vary_seabed, vary_obstacle=args.vary_obstacle,
        wd_fixed=args.wd_fixed, sd_fixed=args.sd_fixed,
        c_fixed=args.c_fixed, seabed_fixed=args.seabed_fixed,
        pekeris_model=args.pekeris_model,
        wd_range=tuple(args.wd_range), sd_range=tuple(args.sd_range),
        c_range=tuple(args.c_range),
        range_start=args.range_start, range_end=args.range_end,
        clean_runs=not args.keep_runs,
    )

    try:
        generate_ocean_dataset(
            num_samples=args.N,
            freq_train=args.freq_train, freq_extrap=args.freq_extrap,
            H=args.grid_h, W=args.grid_w,
            out=args.out, seed=args.seed, cfg=cfg, n_jobs=args.n_jobs,
            obs_ratio=args.obs_ratio, mask_mode=args.mask_mode,
            normalize=not args.disable_normalize, metadata_out=args.metadata_out,
        )
    finally:
        if not args.output_dir and not args.keep_runs:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
