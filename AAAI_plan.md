# AAAI Plan — Frequency-Aware Latent Tensor Diffusion for Sparse Wave-Field Reconstruction

> Working title (pick one):
> - **"Learning the Frequency-Response Manifold: Latent Tensor Diffusion for Sparse Reconstruction of Wave Fields"**
> - "Frequency-Aware Latent Tucker Diffusion for Extreme-Sparse Physical-Field Reconstruction"
> - "FreqTD: Frequency-Conditioned Tensor-Core Diffusion for Wave-Field Inverse Problems"

---

## 0. Venue & timeline (read first)

- **AAAI-26 main track deadline (Aug 1–4, 2025) already passed** (conf. Jan 2026, Singapore). Realistic target = **AAAI-27** (full paper ≈ **early Aug 2026**), so ~6–8 weeks of runway from now (June 2026).
- Backup venues if AAAI-27 slips: ICLR 2027 (≈ Sept 2026), or a strong workshop (AAAI AI-for-Science) for early signal.
- **Implication for the plan:** experiments are split into **MUST (core acceptance)**, **SHOULD (rebuttal-proof)**, **COULD (polish)**. Aim to finish MUST + SHOULD before the deadline; COULD goes to camera-ready/appendix.

---

## 1. The one-paragraph pitch

Reconstructing a continuous frequency-domain physical field (acoustic/elastic/EM) from a handful of scattered sensors is a severely ill-posed inverse problem, made worse when the **target frequency was never seen in training** (frequency extrapolation) or when sensing is **extremely sparse (≤1%)**. Neural operators regress this map deterministically and break out-of-band; pixel-space generative solvers (DiffusionPDE, CoNFiLD) are resolution-bound, expensive, and frequency-agnostic; classical low-rank/tensor methods (LRTFR) fit one instance and cannot extrapolate. **Our insight:** although a wave field varies wildly and non-linearly with frequency ω, a *family* of such fields on a common spatial domain lives on a **low-dimensional manifold whose coordinates evolve smoothly and predictably with ω** — captured by a **shared continuous spatial basis** times **frequency-indexed low-rank Tucker cores** (Functional Tucker Model, FTM). (Empirically this holds even when the medium is strongly heterogeneous *and varies per sample* — e.g. our elastic data with per-sample random λ,μ,ρ fields.) We (i) extract this manifold with FTM, (ii) learn a **frequency-conditioned latent diffusion prior over the cores** `p(G | ω)` — the *frequency-response manifold* — and (iii) at inference **collapse the prior onto sparse evidence with multilinear Diffusion Posterior Sampling (DPS)**, whose likelihood gradient is a *closed-form matrix product* thanks to the multilinear decoder (no autodiff through a neural decoder). This yields state-of-the-art **frequency extrapolation**, **extreme-sparse reconstruction**, **order-of-magnitude faster guidance**, **native uncertainty**, and **clean scaling to 3D** (the basis factorizes per spatial axis).

---

## 2. Contributions (3 main, to dodge "incremental combination")

1. **A frequency-response-manifold formulation of wave-field reconstruction.** We recast a family of frequency-domain fields as a shared spatial basis + a frequency-trajectory of low-rank tensor cores, and learn a *frequency-conditioned generative prior* over that trajectory. This is the first generative PDE-field method to target **out-of-band frequency extrapolation** as a first-class capability.
2. **Multilinear latent DPS.** Because the FTM decoder is multilinear, the observation-likelihood gradient used to guide diffusion is a *closed-form* tensor contraction (matrix multiply), not autodiff through a CNN/INR decoder. This makes guidance exact and 1–2 orders of magnitude cheaper than pixel-space (DiffusionPDE) or neural-field (CoNFiLD) guidance, and enables arbitrary-resolution / 3D decoding.
3. **Self-calibrating guidance — parameter-free balancing of the observation and PDE-residual terms.** The two guidance coefficients (ζ_obs for the sparse-observation likelihood, ζ_pde for the physics residual) are brittle and depend on dataset / frequency / sparsity — too large diverges (we observed NaNs at high DPS weight), too small under-conditions, and tuning them per setting is a real, reviewer-visible weakness of guided diffusion. We replace them with an automatic, **scale-invariant, feedback-controlled** scheme (§4.3): normalize each guidance step relative to the denoising step (one bounded knob λ∈(0,1)), and a lightweight controller drives the *realized* observation- and PDE-residuals along target schedules over diffusion time — auto-balancing the two terms and the data→physics transition with **no per-setting tuning**. The closest prior art is DiffusionPDE's *hand-designed fixed* two-term transition; ours is principled and transfers across PDEs/sparsity/frequency. It is most naturally enabled by contribution 2: because the observation gradient is closed-form, measuring its magnitude/residual every step is essentially free, so adaptive guidance is a *consequence* of the design, not a bolt-on.

> **Decision gate for contribution 3:** present it as a main contribution; validate via E8 (matches oracle-tuned coefficients with a single transferable λ across all datasets, removes divergence). If the experiments under-deliver, demote it to a robustness ablation and elevate the §4 analysis instead.
>
> **A note on "scope".** Earlier we considered a "shared-low-rank diagnostic" as a contribution. We **drop it as a headline claim**: our own data show FTM's shared low-rank structure holds even under *per-sample-varying heterogeneous media* (elastic with random λ,μ,ρ → shared-basis rel-err 0.004 @ R=32), so a "fixed-medium scope" framing is both wrong and self-limiting. The diagnostic survives only as a *supporting analysis* (§4 / appendix) that (a) justifies *why* the latent is low-rank and learnable, and (b) explains the two genuine, mundane requirements — a **shared, consistently-parameterized spatial domain** across the family, and **rank ≳ #modes ∝ ω·L/c**. The ocean failure is attributable to violating (a) (per-sample depth renormalization with varying physical depth ⇒ inconsistent coordinates) plus extreme dynamic range — a data-generation artifact, not a fundamental limit.

---

## 3. Where we sit vs. the literature (related-work map)

Three competitor families; we straddle all three and beat each on its weak axis.

**(A) Sparse-sensor field reconstruction (deterministic).**
Shallow decoder [Erichson+ 2020]; **Voronoi-CNN** [Fukami+ Nat. Mach. Intell. 2021]; **Senseiver** (cross-attention + INR decoder); **RecFNO / FLRONet** (operator-based reconstruction from sparse sensors). → *Weakness:* point estimates, no UQ, fixed sensor layout assumptions, no frequency extrapolation. We add a generative prior + frequency axis.

**(B) Neural operators (forward/inverse maps).**
**FNO** [Li+ 2020], **F-FNO** [ICLR 2023], **DeepONet** [Lu+ 2021], **PINO**, **U-NO**, **LSM** [ICML 2023], **GNOT** [ICML 2023], **OFormer**, **Transolver** [ICML 2024], **LNO** [NeurIPS 2024]. → *Weakness:* deterministic regression; poor OOD-frequency and extreme-sparsity behavior; no uncertainty. Most are reproducible via THUML **Neural-Solver-Library** (cheap to add).

**(C) Generative / tensor priors.**
**DiffusionPDE** [Huang+ NeurIPS 2024] (pixel-space joint coeff+solution diffusion + guided sampling); **CoNFiLD** [Du+ Nat. Commun. 2024] (neural-field latent diffusion, Bayesian posterior sampling, turbulence/time axis); **FunDPS** [NeurIPS 2025] (function-space diffusion guidance); **LRTFR** [Luo+ 2022] and **functional TT/TR INR** (continuous low-rank tensor fits, incl. a 2026 EM-surrogate LRTFR study). → *Weakness:* pixel/neural-field diffusion is expensive and frequency-agnostic (DiffusionPDE, CoNFiLD); LRTFR is single-instance, no learned prior, no extrapolation. We use an *explicit low-rank tensor latent* (cheap multilinear DPS) *conditioned on frequency*. (DiffusionPDE also balances two guidance terms but with a *fixed hand-designed* iteration schedule — the closest prior art to our self-calibrating guidance, contribution 3.)

**Positioning sentence for the paper:** *"DiffusionPDE diffuses in pixel space, CoNFiLD in a neural-field latent, LRTFR has no prior at all; we diffuse in a multilinear tensor-core latent conditioned on a physical parameter, uniquely enabling closed-form guidance and frequency extrapolation."*

Citations to fetch (already verified): DiffusionPDE arXiv:2406.17763; CoNFiLD arXiv:2403.05940 / Nat. Commun. 15:10416; LRTFR arXiv:2212.00262; Voronoi-CNN arXiv:2101.00554; Transolver arXiv:2402.02366; FLRONet arXiv:2412.08009; FunDPS OpenReview oAgwvZay2U; DPS [Chung+ ICLR 2023]; Neural-Solver-Library (github thuml).

---

## 4. Method (formalization for §3 of the paper)

Field family `u(x, ω) ∈ C^C` on a **common spatial domain**, indexed by sample b (varying source/BC/medium/local geometry) and frequency ω. The medium may be heterogeneous and vary per sample; what the shared basis needs is a *consistently-parameterized* domain across the family (not per-sample coordinate renormalization).

- **FTM decomposition (latent extraction).**
  `u_b(x,ω) ≈ Σ_{r,q,...} G^b_ω[r,q,...] · f_x[r](x)·f_y[q](y)·(f_z[·](z))`, with **shared** coordinate-MLP bases `f_x,f_y,(f_z)` across the whole family and **per-(b,ω)** Tucker core `G^b_ω`. Real/imag (and vector components) are channels sharing the basis.
- **Frequency-conditioned latent diffusion (the prior).** Train a diffusion model on pairs `(G^b_ω, ω)` to learn `p(G|ω)` — Fourier-embed ω as conditioning. This is the *frequency-response manifold*.
- **Multilinear DPS (inference).** Given target ω* and sparse obs `y = M(u_{ω*}) + ε`, sample `G` from `p(G|ω*)` while guiding by `∇_G ‖ M(Φ·G) − y‖²`, where `Φ = f_x⊗f_y(⊗f_z)` is *fixed* → the gradient is a **closed-form contraction** (no decoder autodiff). Decode `û(x,ω*) = Φ(x)·G` at any resolution.

Emphasize the 3 levers that are novel *in combination*: (latent = low-rank tensor) × (conditioning = continuous physical frequency) × (guidance = closed-form multilinear).

### 4.3 Self-calibrating guidance (contribution 3)

The guided update at diffusion step t has the schematic form
`G_{t-1} = G_{t-1}^{prior} − ζ_obs·g_obs − ζ_pde·g_pde`,
with `g_obs = ∇_G ‖M(Φ·Ĝ0) − y‖²` (closed-form, multilinear) and `g_pde = ∇_G R_pde(Φ·Ĝ0)` (PDE residual of the decoded field). The coefficients `ζ_obs, ζ_pde` are the brittle, hard-to-tune knobs. We propose to set them automatically with three composable mechanisms:

1. **Scale-invariant normalization (removes absolute scale).** Set each guidance step so its norm is a fixed fraction of the prior denoising step:
   `ζ_• = λ_• · ‖ΔG^{prior}‖ / (‖g_•‖ + ε)`, with bounded trust ratios `λ_obs, λ_pde ∈ (0,1)`. This kills dependence on the operator scale, loss units, and noise level σ_t — turning unbounded coefficients into one or two interpretable, transferable knobs. (Also fixes the divergence/NaN failure mode we observed at large fixed ζ.)
2. **Feedback control on a residual schedule (auto-balances the two terms + the data→physics annealing).** Define target decay trajectories for the *realized* residuals r_obs(t)=‖M(Φ·Ĝ0)−y‖ and r_pde(t)=‖R_pde(·)‖ (e.g. geometric decay toward the measurement-noise floor / a physics tolerance). A multiplicative-weights / PI controller nudges `λ_obs, λ_pde` each step so realized residuals track their targets — early steps emphasize fitting observations, late steps emphasize physics, with the crossover discovered automatically rather than hand-scheduled.
3. **(Optional) cross-term equalization.** When both terms are active, equalize their effective gradient contributions (GradNorm- / homoscedastic-uncertainty-style) so neither silently dominates.

The result is a guidance procedure with **no per-dataset coefficient tuning** — the same `λ` works across PDEs, sparsity ratios, and in-/out-of-band frequencies. Crucially this is cheap *because* the observation gradient is closed-form (§4 contribution 2): measuring `‖g_obs‖` and `r_obs` every step costs a matrix product, so the controller adds negligible overhead. Contrast with DiffusionPDE, whose two-term balance uses a *fixed hand-designed* iteration-dependent transition.

---

## 5. Datasets — current + what to add (and *why each*)

**Design rule (corrected):** use a **shared, consistently-parameterized spatial domain** across all samples (fixed grid extent; **no per-sample coordinate renormalization**). *Within* that domain, vary freely — sources, BCs, local geometry/obstacles, **and the medium itself (heterogeneous, per-sample-varying λ,μ,ρ / sound speed)**: our elastic data already varies the medium per sample and FTM still reconstructs unobserved points to ~0.014 rel-L2. The only thing to avoid is what broke the ocean run — letting the *domain size/parameterization* change per sample (there, water depth changed the waveguide and the depth axis was renormalized per sample, so identical normalized coordinates meant different physics). This makes "robustness to heterogeneous, varying media" a **selling point**, not a limitation.

| Dataset | Status | C | Why it's in the paper |
|---|---|---|---|
| **2D Helmholtz** (point sources, fixed medium) | ✅ done | 2 | Canonical wave field; main frequency-extrapolation + sparsity story. |
| **2D Elastic wave** (vector field, **per-sample-varying λ,μ,ρ** + obstacles) | ✅ done | 4 | Multi-channel + **per-sample heterogeneous-medium variation** — the key evidence that FTM's shared basis survives varying media (sell this, don't hide it); harder modal structure. |
| **3D Helmholtz** (heterogeneous medium, point sources, fixed box) | **ADD (MUST)** | 2 | **Hero scalability claim.** Tucker basis factorizes `f_x⊗f_y⊗f_z`, so latent grows mildly while pixel-diffusion/operators blow up in memory → our efficiency + feasibility advantage is starkest here. Directly stresses DiffusionPDE/CoNFiLD. |
| **2D frequency-domain Maxwell / EM scattering** (per-sample scatterer/permittivity varied, swept ω) | **ADD (SHOULD)** | 2 | **Generality beyond acoustics**; complex-valued, same pipeline; lets us cite/borrow the EM-surrogate-LRTFR line. Shows the "frequency-response manifold" is a property of wave physics, not one equation. |
| **2D Helmholtz, strongly heterogeneous / multi-scatterer medium (per sample)** | **ADD (SHOULD)** | 2 | **Positive generality result**: vary the medium aggressively per sample on a fixed grid → show FTM + our pipeline still reconstruct (corroborates the elastic finding, pre-empts "only homogeneous"). |

**3D Helmholtz generation specifics (to implement):** reuse the Helmholtz solver structure → 3D FD Helmholtz with PML, `grid ~ 48³–64³`, M≈7–9 frequencies (train band + 2 extrapolation), N≈300–800 samples each with K random point sources (random position/phase) in a **fixed-size** box whose sound-speed field may be heterogeneous and vary per sample (keep the *grid extent/parameterization* fixed — no per-sample renormalization). Export the same `(N,M,Dx,Dy,Dz,2)` + sparse-mask HDF5 contract so `train_FTM_GPU` (already channel/axis-generic) extends with a 3rd basis net `f_z`. Storage: keep grid modest (48³) and M small to bound disk.

**Sparsity protocol (all datasets):** sensor ratios ρ ∈ {5%, 1%, 0.5%}, masks per-sample fixed (sensor array) and per-freq (scanning) — both already supported. Add an **off-grid / scattered-sensor** variant for the Senseiver/Voronoi comparison (our decoder is continuous, so off-grid is a strength to showcase).

---

## 6. Baselines — categorized, prioritized, with how to get them

Goal: ≥2 strong baselines per family so reviewers can't say "missing comparison X."

**(A) Sparse-sensor reconstruction**
- **Voronoi-CNN** [Fukami+ 2021] — MUST (the standard sparse-sensor baseline). Reimplement (small).
- **Senseiver** (attention + INR) — SHOULD (SOTA sparse, handles off-grid). Code available.
- (have) — none yet here; this family is currently *missing* and reviewers will expect it.

**(B) Neural operators** (via THUML Neural-Solver-Library → cheap)
- **FNO** — ✅ have.
- **F-FNO** or **U-NO** — MUST (stronger FNO variant).
- **Transolver** [ICML 2024] — SHOULD (current operator SOTA; geometry-general).
- **DeepONet** — COULD (classic; cheap; reviewers like seeing it).
- Input adapter: feed sparse obs as masked field / Voronoi tessellation + ω embedding → field.

**(C) Generative & tensor priors**
- **DiffusionPDE** [NeurIPS 2024] — **MUST** (the headline generative competitor; official code). Run its guided sampling on our fields; this is the comparison that sells the efficiency + frequency story.
- **CoNFiLD** — ✅ have (neural-field latent diffusion). Keep.
- **LRTFR** — ✅ have (tensor, no prior). Keep.
- **Functional TT / Tensor-Ring INR** — SHOULD (alternative low-rank format; shows our Tucker+prior beats prior-free TT/TR). Adapt from the EM-surrogate-LRTFR formulations.
- **FunDPS** (function-space diffusion) — COULD (recent, strong; may be heavy to port).

**Our own variants (ablation "baselines"):**
- **FTM + least-squares core (no diffusion prior)** — the deterministic tensor method; demonstrates the prior's value (and is literally the ocean-finding fit).
- **Pixel-space core-free diffusion** (= our pipeline minus FTM latent) — isolates the latent.
- **Unconditional core diffusion (no ω)** — isolates frequency conditioning.
- **DPS via autodiff vs closed-form** — isolates the efficiency claim.

---

## 7. Experiment matrix (claim → setup → metric → baselines)

Each row is a self-contained figure/table; together they form the "环环相扣" arc: *insight → it works → it extrapolates → it's cheap → it's calibrated → it's general → we know why.*

| # | Claim (the hook) | Setup | Metric | Key baselines |
|---|---|---|---|---|
| **E1** | Accurate full-field recon at extreme sparsity | ρ∈{5,1,0.5%}, in-band ω, all datasets | rel-L2 / VRMSE, PDE residual | Voronoi-CNN, Senseiver, FNO/F-FNO, DiffusionPDE, CoNFiLD, LRTFR |
| **E2** | **Frequency extrapolation** (the headline) | train on band [ωmin,ωmax], test on ω outside; sparsity fixed | rel-L2 vs Δω (in-band vs OOD curve) | all of above (operators expected to collapse OOD) |
| **E3** | **Efficiency / scalability**, esp. 3D | inference time + peak memory vs grid size (2D→3D) | sec/sample, GB, accuracy@budget | DiffusionPDE (pixel), CoNFiLD (neural-field), closed-form vs autodiff DPS |
| **E4** | **Uncertainty is calibrated** | posterior samples per recon | coverage / CRPS / error-vs-variance corr. | CoNFiLD (Bayesian), deep-ensemble FNO |
| **E5** | Generality across physics | 2D Helmholtz, elastic, 3D Helmholtz, (Maxwell) | rel-L2 table | best per-family baseline |
| **E6** | Robustness to sensor placement / off-grid | scattered/off-grid sensors, noise σ | rel-L2 vs noise; on/off-grid | Voronoi-CNN, Senseiver |
| **E7** (analysis, *supporting*) | **Why the latent is low-rank & learnable** | shared-basis ceiling vs rank; incl. **per-sample-varying-media** (elastic, new Helmholtz-hetero) staying low-error; required-rank ∝ ω·L/c | shared-basis ceiling curves | — (analysis, not a headline claim) |
| **E8** | **Self-calibrating guidance** (contrib. 3) | (a) sensitivity heatmap over (ζ_obs,ζ_pde); (b) adaptive vs best-tuned-manual (oracle); (c) one fixed λ transferred across all datasets/ρ/ω; (d) stability (no divergence) | rel-L2 gap to oracle-tuned, #tuning runs saved, divergence rate | manual-grid DPS, DiffusionPDE fixed-transition guidance |

**Ablations (A1–A6):** FTM rank; with/without ω-conditioning (→ E2 mechanism); with/without frequency-smoothness (TV) regularizer; latent-core vs pixel diffusion; DPS guidance weight & schedule; #training samples / #frequencies (data-efficiency). Tie A2 to E2 and A4 to E3.

**Headline figures to produce:**
1. Teaser: GT vs sparse obs vs our recon vs DiffusionPDE/FNO at ρ=1%, including an **OOD frequency** panel. (re/im/|·|/phase, like the visualizer already built.)
2. **Frequency-extrapolation curve** (E2): rel-L2 vs ω with the train band shaded — our line flat across the boundary, baselines spiking. *This is the money figure.*
3. **Efficiency frontier** (E3): accuracy vs compute, 2D and 3D, log-scale; our point in the bottom-left.
4. **Analysis** (E7): shared-basis ceiling vs rank — including curves for *per-sample-varying heterogeneous media* staying low — + required-rank∝ω overlay.
5. Per-dataset qualitative grids + UQ band figure.

---

## 8. Threats, limitations, reviewer-proofing

- **"Just FTM + diffusion + DPS combined."** → Counter with contributions 2 (closed-form multilinear guidance, an actual algorithmic property, with the E3 speed evidence) and 3 (self-calibrating guidance, with the E8 evidence). Show DiffusionPDE/CoNFiLD *cannot* do closed-form guidance or OOD frequency, and use hand-tuned/fixed guidance schedules.
- **"Only works on homogeneous / fixed media."** → False, and we show it: elastic + the strongly-heterogeneous Helmholtz set vary the medium per sample and still reconstruct unobserved points well (E5, E7). The only genuine requirement is a consistently-parameterized shared domain (§5) — a mild, standard assumption.
- **"Guided diffusion needs careful coefficient tuning."** → E8: our self-calibrating guidance matches oracle-tuned coefficients with one transferable λ and no per-setting search, and removes the divergence/NaN failure mode; contrast with DiffusionPDE's hand-designed fixed transition.
- **"Sensors on a grid only."** → E6 off-grid (our continuous decoder shines).
- **"Toy PDEs."** → 3D Helmholtz + Maxwell + elastic obstacles cover wave physics breadth; cite that DiffusionPDE itself uses Helmholtz/Darcy/Burgers/NS.
- **Physics consistency.** → report PDE residual (you have `validate_physics_residual.py` / `physics_metric.py`) so recon isn't just pixel-accurate.
- **Fair baselines.** → tune baselines via their official code/Neural-Solver-Library; report compute parity; put hyperparameters in appendix.

---

## 9. Paper outline (AAAI: 7 pages + refs)

1. **Intro** (1p): the frequency-extrapolation + extreme-sparsity gap; the manifold insight; 3 contributions; teaser fig.
2. **Related work** (0.5–0.75p): the 3 families (§3), one sentence each on why they fall short on our axes.
3. **Method** (2p): problem setup; FTM latent; frequency-conditioned diffusion; multilinear DPS (closed-form-gradient derivation — short, clean); **self-calibrating guidance** (contribution 3); arbitrary-resolution/3D decode. Short *supporting analysis* paragraph: why the latent is low-rank/learnable (holds across varying heterogeneous media), required-rank∝ω·L/c.
4. **Experiments** (2.5p): E1 sparsity, E2 extrapolation (hero), E3 efficiency/3D, E8 self-calibrating guidance, E4 UQ, E5 generality (incl. varying media), ablations; tables + the headline figures.
5. **Limitations & Conclusion** (0.25p): need for a consistently-parameterized shared domain; per-sample-domain-geometry variation (e.g. waveguide-size change) as future work (conditioned/aligned basis).
7. **Appendix** (unlimited): solver/dataset details, baseline configs, extra qualitative, full ablations, the off-grid/noise studies, proofs.

---

## 10. Prioritized work plan (~6–8 week runway)

**MUST (core — without these, no paper):**
- [ ] 3D Helmholtz dataset generator + FTM 3-axis basis extension; run full pipeline (E5, E3).
- [ ] Add baselines: **DiffusionPDE** (official code), **Voronoi-CNN**, one stronger operator (**F-FNO/U-NO**). (Family coverage A/B/C complete.)
- [ ] **E2 frequency-extrapolation study** on 2D Helmholtz + elastic + 3D Helmholtz (the hero result).
- [ ] **E1 sparsity sweep** (ρ=5/1/0.5%) with all baselines.
- [ ] **E3 efficiency/memory** table+frontier (closed-form vs autodiff DPS; 2D vs 3D vs DiffusionPDE).
- [ ] **Self-calibrating guidance (contribution 3)** — implement scale-invariant normalization + residual-schedule controller; run **E8** (sensitivity heatmap, adaptive-vs-oracle, cross-dataset transfer with one λ, stability). *Decision gate: if it matches oracle-tuned across datasets, keep as main contribution; else demote to a robustness ablation.*

**SHOULD (rebuttal-proof):**
- [ ] **E7 supporting analysis** figure (extend `diagnose_rank.py`: required-rank∝ω overlay; include per-sample-varying-media curves staying low to back the generality claim).
- [ ] Strongly-heterogeneous / multi-scatterer **varying-media Helmholtz** dataset (E5 generality, corroborates elastic).
- [ ] 2D Maxwell/EM-scattering dataset (E5 generality).
- [ ] **Senseiver** + **Transolver** baselines.
- [ ] **E4 UQ calibration** (coverage/CRPS) vs CoNFiLD/ensembles.
- [ ] Ablations A1–A5; PDE-residual reporting on all mains.

**COULD (camera-ready / appendix):**
- [ ] E6 off-grid + noise robustness.
- [ ] Per-sample *domain-geometry* variation study (e.g. changing waveguide size) — the genuine boundary; motivates aligned/conditioned-basis future work.
- [ ] FunDPS / functional-TT baselines; data-efficiency ablation A6.

**Suggested order (dependency-aware):** 3D Helmholtz data → FTM 3-axis → DiffusionPDE+Voronoi+F-FNO wired to a common eval harness → E1/E2 across datasets → E3 efficiency → diagnostic figure → SHOULD items → write. Lock the **E2 hero figure** first; if it's strong, the paper is real.

---

## 11. Notes / assets already in repo to reuse

- Pipeline (channel/axis-generic): `train_FTM_GPU.py`, `train_diffusion.py`, `test_diffusion.py` (DPS), `evaluate_diffusion.py`, `validate_physics_residual.py`, `physics_metric.py`.
- Existing baselines: `fno_baseline.py`, `lrtfr_baseline.py`, `confild_baseline.py` (2D + elastic).
- **Supporting-analysis tool (E7):** `ocean_data/diagnose_rank.py` (generalize to any dataset; prints shared-basis ceiling vs rank). Verified: elastic with *per-sample-varying* λ,μ,ρ → 0.004 @ R=32 (medium variation does NOT break the shared basis).
- Visualizer style for qualitative figures: `ocean_data/Visualize_dataset.py` (re/im/|·|/phase, dB) — reuse for paper-quality field panels.
- The **ocean experiment is a cautionary control**, correctly attributed to **per-sample coordinate renormalization + domain-size variation + extreme dynamic range** (NOT medium heterogeneity). Keep out of main results; it justifies only the mild "consistently-parameterized shared domain" assumption (§5).
