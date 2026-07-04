# Omega-Conditional Basis for 2D Helmholtz Reconstruction

## 1. Motivation

Current FTM + latent diffusion uses a shared spatial basis:

$$
U(x, y; \omega, s) \approx \Phi(x, y) \, g_{s,\omega},
$$

where:
- $U(x, y; \omega, s)$ is the physical field for sample $s$ at frequency $\omega$
- $\Phi(x, y) \in \mathbb{R}^{P \times R}$ is a frequency-independent basis
- $g_{s,\omega} \in \mathbb{R}^R$ is the latent core

This factorization works well when the main variation across frequencies can be absorbed into the coefficient trajectory $g_{s,\omega}$. However, for Helmholtz fields, frequency directly changes the spatial oscillation pattern:
- wavelength scales as $\lambda \propto 1 / \omega$
- nodal structure and interference pattern move with $\omega$
- high-frequency fields require finer spatial oscillations than low-frequency fields

Therefore, forcing **all frequencies to share the same basis** may place too much burden on the core tensor. In that regime:
- the latent core must encode both sample-specific structure and frequency-specific wave pattern
- the latent trajectory may become entangled and hard for diffusion to model
- explicit $\omega$ conditioning in latent diffusion may become weak or even harmful

This motivates a different factorization:

$$
U(x, y; \omega, s) \approx \Phi_{\omega}(x, y) \, g_s,
$$

or more generally,

$$
U(x, y; \omega, s) \approx \Phi(x, y; \omega) \, g_{s,\omega}^{\text{res}},
$$

where the basis itself is frequency-aware.

The key idea is:

> Move the frequency dependence from the diffusion prior on the latent core to the decoder/basis side, so that the basis carries the frequency-dependent spatial wave pattern and the latent code only needs to represent sample-specific content.

---

## 2. Core idea

### 2.1 From shared basis to conditional basis

Current model:

$$
\Phi(x, y) = \phi_x(x) \otimes \phi_y(y)
$$

with a fixed basis for all frequencies.

Proposed model:

$$
\Phi_{\omega}(x, y) = \phi_x(x, \omega) \otimes \phi_y(y, \omega)
$$

or a slightly more stable variant:

$$
\Phi_{\omega}(x, y) = \Phi_0(x, y) + \Delta \Phi(x, y; \omega),
$$

where:
- $\Phi_0$ is a shared base basis
- $\Delta \Phi(\cdot; \omega)$ is a frequency-conditioned correction

This allows the spatial basis to adapt its oscillatory pattern with frequency.

### 2.2 Desired effect

Instead of asking the core tensor to carry frequency-specific wave geometry, we want:
- basis to explain "how the field oscillates at this frequency"
- latent code to explain "which sample-specific combination of these basis atoms is active"

This should produce a latent representation that is:
- smoother across frequency
- lower-entropy
- easier to model with a simple diffusion prior
- potentially even weakly conditioned or unconditional

---

## 3. Three possible formulations

## 3.1 Formulation A: Fully omega-conditional basis

Use basis networks:

$$
\phi_x(x, \omega), \quad \phi_y(y, \omega)
$$

with outputs
- $\phi_x \in \mathbb{R}^{N_x \times R_x}$
- $\phi_y \in \mathbb{R}^{N_y \times R_y}$

and construct

$$
\Phi_{\omega} = \phi_y(\omega) \otimes \phi_x(\omega).
$$

Field reconstruction becomes:

$$
U_{s,\omega} \approx \Phi_{\omega} G_{s,\omega}.
$$

### Pros
- Most expressive
- Directly injects frequency into spatial representation
- Most likely to capture changing oscillation scale

### Cons
- Harder to train
- Basis may drift too much across frequency
- Core identity across frequencies becomes less stable
- May weaken the shared-basis advantage of FTM

---

## 3.2 Formulation B: Shared base basis + omega-conditioned modulation

Use a shared basis plus FiLM-style modulation:

$$
\phi_x(x, \omega) = \gamma_x(\omega) \odot \phi_x^{(0)}(x) + \beta_x(\omega)
$$

$$
\phi_y(y, \omega) = \gamma_y(\omega) \odot \phi_y^{(0)}(y) + \beta_y(\omega)
$$

where $\gamma, \beta$ are small MLP outputs from a frequency embedding.

Equivalent viewpoint:
- keep the original basis generator
- add frequency-dependent channel-wise scaling and shifting

### Pros
- Much more stable than fully conditional basis
- Retains shared global basis structure
- Lower implementation risk
- Easy to ablate against current FTM

### Cons
- May be too weak if frequency dependence is highly nonlinear

This is likely the **best first implementation**.

---

## 3.3 Formulation C: Shared basis + conditional residual basis branch

Use

$$
\Phi_{\omega}(x,y) = \Phi_0(x,y) + \alpha(\omega) \Phi_{\text{res}}(x,y;\omega)
$$

where:
- $\Phi_0$ captures frequency-invariant structure
- $\Phi_{\text{res}}$ captures frequency-dependent oscillatory corrections
- $\alpha(\omega)$ controls the correction strength

This explicitly decomposes field structure into:
1. common spatial scaffold
2. frequency-driven oscillatory residual

### Pros
- Most interpretable
- Easy to visualize whether the residual branch is actually frequency-sensitive
- Strong fit with your paper narrative

### Cons
- More components and hyperparameters
- Slightly more engineering than Formulation B

---

## 4. Recommended implementation path

I recommend the following staged route:

### Stage 1: Frequency-modulated basis (Formulation B)

This is the safest and most practical starting point.

#### Architecture

Replace current 1D basis nets
- `net_x(x) -> phi_x(x)`
- `net_y(y) -> phi_y(y)`

with conditional basis nets
- `net_x(x, omega) -> phi_x(x, omega)`
- `net_y(y, omega) -> phi_y(y, omega)`

Implementation options:

#### Option B1: Concatenate omega to input

For each coordinate:

$$
[x, \omega_{norm}] \mapsto \phi_x,
$$

$$
[y, \omega_{norm}] \mapsto \phi_y.
$$

Simple, but weak: omega only enters at input level.

#### Option B2: FiLM modulation inside each hidden layer

For each hidden block:

$$
h^{(l+1)} = \sigma\big(\gamma^{(l)}(\omega) \odot W^{(l)} h^{(l)} + \beta^{(l)}(\omega)\big)
$$

This is preferable because:
- frequency can reshape intermediate features
- modulation is lightweight
- easier to preserve the existing basis structure

#### Option B3: Hypernetwork for last layer only

Use a small omega-conditioned hypernetwork to generate the final projection weights from hidden features to basis coefficients.

This gives stronger conditionality while keeping most of the basis network shared.

### Recommendation

For the first version, use **FiLM modulation on hidden layers**.

---

## 5. How the full pipeline would change

## 5.1 FTM training stage

Current objective is roughly:

$$
\min_{\Phi, G} \sum_{s,\omega} \|U_{s,\omega} - \Phi G_{s,\omega}\|^2
$$

Proposed objective:

$$
\min_{\Phi_{\omega}, G} \sum_{s,\omega} \|U_{s,\omega} - \Phi_{\omega} G_{s,\omega}\|^2
$$

with additional regularization to prevent basis drift.

### Necessary regularizers

#### 1. Basis smoothness across frequency

We do not want adjacent frequencies to produce unrelated bases.

Add:

$$
\mathcal{L}_{\text{basis-smooth}} = \sum_{\omega_i,\omega_{i+1}} \|\Phi_{\omega_{i+1}} - \Phi_{\omega_i}\|_F^2
$$

or factorized version:

$$
\|\phi_x(\omega_{i+1}) - \phi_x(\omega_i)\|_F^2 +
\|\phi_y(\omega_{i+1}) - \phi_y(\omega_i)\|_F^2.
$$

#### 2. Modulation magnitude regularization

If using FiLM:

$$
\mathcal{L}_{\text{mod}} = \sum_l \left(\|\gamma^{(l)}(\omega)-1\|^2 + \|\beta^{(l)}(\omega)\|^2\right)
$$

This encourages the conditional basis to stay close to the shared basis unless frequency really needs to change it.

#### 3. Basis orthogonality / conditioning regularization

If the frequency-conditioned basis becomes ill-conditioned, least-squares fitting of cores becomes unstable.

Possible regularizer:

$$
\mathcal{L}_{\text{orth}} = \|\Phi_\omega^T \Phi_\omega - I\|_F^2
$$

or applied separately to `phi_x`, `phi_y`.

---

## 5.2 Latent diffusion stage

After FTM training, you will obtain cores under the new conditional basis.

Two possibilities should be tested:

### Case 1: Continue using omega-conditioned diffusion

Train:

$$
p_\theta(G_{s,\omega} \mid \omega)
$$

This is the direct analogue of the current setup.

### Case 2: Remove or weaken explicit omega conditioning

If the conditional basis already absorbs most frequency dependence, then core distribution may become nearly frequency-invariant. In that case train:

$$
p_\theta(G)
$$

or use a much weaker condition.

This is actually one of the most interesting hypotheses of Direction D:

> A good omega-conditional basis may make explicit omega-conditioning in latent diffusion unnecessary.

That would be a strong conceptual result.

---

## 5.3 Test-time reconstruction

At test frequency $\omega^*$:

1. build conditional basis $\Phi_{\omega^*}$
2. run diffusion in latent/core space to sample a prior core
3. perform DPS guidance using the observation model

The observation model becomes:

$$
y = M \Phi_{\omega^*} G + \epsilon
$$

Compared with the current method, the only change is that the forward operator depends on $\omega$ through the basis.

This is clean and should integrate naturally into your current test pipeline.

---

## 6. Concrete model design

## 6.1 Basis network parameterization

A practical implementation for `net_x` and `net_y`:

### Shared trunk

- input: coordinate (`x` or `y`)
- several MLP layers produce hidden feature `h`

### Frequency conditioning branch

- input: normalized frequency `omega_norm`
- pass through Fourier embedding or simple MLP
- output per-layer FiLM parameters `(gamma_l, beta_l)`

### Modulated hidden layers

For each layer in `net_x` / `net_y`:

$$
h_l' = \gamma_l(\omega) \odot h_l + \beta_l(\omega)
$$

Then continue through activation and projection.

This keeps the existing FTM code structure mostly intact.

---

## 6.2 Core fitting options

There are two possible strategies:

### Strategy 1: Per-frequency least-squares core fitting

For each $(s,\omega)$:

$$
G_{s,\omega} = \arg\min_G \|U_{s,\omega} - \Phi_\omega G\|^2
$$

This is the most straightforward extension.

### Strategy 2: Sample-wise shared core + frequency decoder

A stronger version is:

$$
U_{s,\omega} \approx \Phi_\omega z_s
$$

where $z_s$ is sample-specific but frequency-invariant.

This is much more ambitious. It would mean:
- one latent code per sample
- frequency dependence handled almost entirely by the basis

This is conceptually elegant, but likely too restrictive for the first implementation.

### Recommendation

Start with **Strategy 1**.

---

## 6.3 Rank handling

A subtle point: if high frequencies need richer oscillatory structure, the effective rank may increase with frequency.

Three options:

### Option 1: Fixed rank, conditional basis
- simplest
- clean comparison to current FTM

### Option 2: Fixed max rank + frequency-dependent gate

$$
\Phi_\omega = \Phi \cdot \text{diag}(a(\omega))
$$

where $a(\omega) \in [0,1]^R$ gates basis atoms.

This gives an interpretable notion of "active rank grows with frequency".

### Option 3: Dynamic rank
- more flexible
- much harder to implement and compare fairly

### Recommendation

Use **fixed rank + frequency-dependent gate** as a future extension, not in the first version.

---

## 7. Why this may help specifically for 2D Helmholtz

This idea is especially well-matched to 2D Helmholtz because:

1. **Frequency affects spatial wavelength directly**.
   The field geometry itself changes with frequency, not just amplitude.

2. **The governing PDE contains frequency in a structured way**.
   The effect is not arbitrary; it is tied to the operator $(\Delta + k^2)$.

3. **The current no-omega ablation suggests the core may not be the right place for frequency semantics**.
   If explicit omega conditioning hurts latent diffusion, then shifting frequency dependence to the basis is a natural remedy.

4. **The field remains smooth in physical space while oscillation scale changes**.
   A conditional basis can adapt oscillation scale explicitly, instead of forcing the core to encode it implicitly.

---

## 8. Main risks

## 8.1 Basis drift destroys comparability across frequency

If $\Phi_\omega$ changes too much, then cores at different frequencies live in different coordinate systems. Then:
- diffusion over core tensors becomes harder, not easier
- core trajectories may lose continuity

This is the biggest risk.

### Mitigation
- use modulation, not fully separate basis networks
- add frequency smoothness regularization
- keep modulation small initially

---

## 8.2 Overfitting frequency identity

The basis may memorize each training frequency and fail to interpolate to nearby frequencies.

### Mitigation
- use continuous frequency input, not learned discrete embeddings
- test interpolation/extrapolation across frequency
- regularize modulation branch strongly

---

## 8.3 Improved reconstruction but worse latent diffusion

It is possible that conditional basis improves FTM reconstruction error but makes latent diffusion harder because the latent coordinates are no longer aligned across frequencies.

### Mitigation
- measure not only reconstruction RMSE but also core smoothness and diffusion prior quality
- compare no-omega diffusion vs omega-conditioned diffusion under the new basis

---

## 9. Key diagnostics to evaluate the idea

To know whether Direction D is working, track the following:

## 9.1 Basis-level diagnostics

1. **Frequency smoothness of basis**
   - $\|\Phi_{\omega_{i+1}} - \Phi_{\omega_i}\|_F$
2. **Condition number / orthogonality of basis**
3. **Visualization of basis atoms across frequency**
   - do atoms change smoothly?
   - do higher frequencies show finer oscillation patterns?

## 9.2 Core-level diagnostics

1. **Core smoothness across frequency**
   - $\|G_{\omega_{i+1}} - G_{\omega_i}\|$
2. **Core rank / energy concentration**
3. **Whether no-omega diffusion improves after introducing conditional basis**

## 9.3 Reconstruction-level diagnostics

1. FTM-only reconstruction error
2. Diffusion prior RMSE before DPS
3. Final DPS RMSE
4. PDE residual
5. Frequency-wise RMSE curves

The most important question is not only whether final RMSE goes down, but whether:

> the prior samples become more frequency-aware and physically plausible before guidance.

---

## 10. Minimal experiment plan

## Experiment D1: Conditional basis vs shared basis in pure FTM

Compare:
- current shared basis FTM
- omega-modulated basis FTM

Metrics:
- reconstruction RMSE
- core smoothness across frequency
- basis smoothness across frequency

Goal:
- determine whether the basis actually captures frequency-dependent structure better

## Experiment D2: Latent diffusion under conditional basis

Train diffusion on cores from the new basis:
- with omega condition
- without omega condition

Metrics:
- prior RMSE
- final DPS RMSE
- visual smoothness of prior fields

Goal:
- determine whether the new basis makes latent diffusion easier

## Experiment D3: Frequency interpolation/extrapolation

Train conditional basis using a subset of frequencies, test on held-out frequencies.

Goal:
- determine whether the basis learns a continuous frequency law or just memorizes

## Experiment D4: Ablation on modulation strength

Compare:
- input concatenation only
- FiLM on hidden layers
- residual basis branch

Goal:
- find the lightest mechanism that already improves prior quality

---

## 11. Suggested coding strategy

A low-risk engineering sequence would be:

### Step 1
Duplicate current FTM training path into a new experimental file, e.g.:
- `train_FTM_omega_basis.py`

### Step 2
Implement conditional `net_x` and `net_y` with FiLM modulation.

### Step 3
Keep everything else unchanged:
- same rank
- same least-squares fitting logic
- same dataset
- same evaluation metrics

### Step 4
Only after FTM-only gains are confirmed, export the new cores and retrain diffusion.

This avoids mixing too many variables at once.

---

## 12. Strongest version of the paper insight

If Direction D works, the conceptual message is strong:

> In frequency-domain PDE reconstruction, explicit frequency conditioning is not always most effective in latent prior space. A better strategy is to place frequency dependence into the spatial basis itself, so that the latent variables become more frequency-invariant and easier to model generatively.

This would explain:
- why current omega-conditioned latent diffusion is not clearly helping
- why no-omega ablation can outperform omega-conditioned latent diffusion
- why physical-space methods sometimes show stronger frequency sensitivity than latent-space methods

This is a meaningful methodological insight, not just an engineering tweak.

---

## 13. Final recommendation

If this idea is pursued, the best first version is:

1. keep the Tucker/FTM structure
2. replace fixed basis with **shared basis + omega FiLM modulation**
3. regularize basis smoothness across frequency
4. first test whether FTM reconstruction and latent smoothness improve
5. then compare diffusion with and without explicit omega conditioning

In short:

> Do not immediately make the entire latent model more complex. First move the frequency semantics from the diffusion prior into the basis generator, and see whether the latent representation becomes cleaner.

That is the most principled implementation of Direction D.
