# Brighter-Fatter Effect: Physics-Based Forward Model and Correction

## Overview

MIRI ramp data contain two coupled instrumental systematics that distort
group-to-group gradients:

1. **Charge reset decay (RCD)**: after each detector reset, a non-astrophysical
   current leaks into the pixel and decays exponentially over ~1–2 groups,
   elevating early gradients above the true photon rate.

2. **Brighter-fatter effect (BFE)**: as charge accumulates in a pixel during an
   integration, the electric field inside the pixel grows and repels newly
   arriving photoelectrons into neighbouring pixels. The PSF core dims
   progressively through the ramp while a ring around it brightens.

This document describes the joint forward model, the correction pipeline, and
fitted parameters for three MIRI time-series targets.

---

## Physical Mechanism

### Charge Reset Decay

After each detector reset, residual charge leaks from the reset transistor into
the pixel. The gradient (DN/group) at each pixel follows:

$$\text{grad}(g) = C + A \cdot e^{-g/\tau} - \delta_{g,0} \cdot \Delta$$

where `C` is the true photon rate, `A` is the decay amplitude, `tau` is the
timescale in groups, and `Delta` is the first-frame offset. `tau` is global;
`C`, `A`, and `Delta` are fitted per pixel.

### Brighter-Fatter Effect

Accumulated charge Q(g) in a pixel repels incoming photoelectrons. The BFE
kernel is radially symmetric:

$$K(i,j) = -\frac{1}{r^\alpha}, \quad r > 0; \qquad K(0,0) = -\sum_{(i,j)\neq(0,0)} K(i,j)$$

The kernel sums to zero by construction (`ΣK = 0`), which is the flux-conservation
constraint. The exponent alpha ≈ 2.8 for MIRI (SNR-weighted average of Wolf-359
and EV Lac fits).

---

## Combined Forward Model

The observed gradient at group g is modelled as:

$$\text{grad}_\text{obs}(x,y;g) = \text{true\_grad}(x,y;g) - A_\text{BFE} \cdot \bigl(K \ast (Q \cdot \text{true\_grad})\bigr)(x,y;g)$$

where Q(g) is the accumulated charge up to group g. This formulation conserves
total image flux exactly: since `ΣK = 0`, summing over all pixels gives
`Σ(K⊛(Q·true_grad)) = 0`.

The BFE and RCD effects are separable in the median over integrations (Q_med is
the same for all integrations), so the median gradient is a clean template for both.

---

## Correction Pipeline (`correct_bfe_rcd`)

Three sequential steps applied to the raw gradients:

### Step 1 — Causal Iterative BFE Correction

For each group g in causal order, solve iteratively for `true_grad` using the
Born series:

$$\text{true\_grad}^{(n+1)} = \text{grad}_\text{obs} + A_\text{BFE} \cdot K \ast (Q_\text{med} \cdot \text{true\_grad}^{(n)})$$

Starting from `true_grad^(0) = grad_obs`, 3 iterations are used. The corrected
gradient for each integration is:

```python
grads_bfe[:, g] = grads_raw[:, g] + A_bfe * fftconvolve(Q_med * true_grad_est, K)
```

This is flux-conserving: `Σ(K⊛(Q·true_grad)) = 0` because K sums to zero.
The correction has negligible impact on background pixels (<0.05% per group for
Wolf-359 at r > 20 px).

### Step 2 — Parametric RCD Subtraction

Fit the global decay timescale tau from background pixels, then fit per-pixel
[C, A, Delta] via least squares. Subtract the fitted decay from every integration.

### Step 3 — Non-Parametric Residual Removal

Subtract the per-pixel per-group median over integrations, then add back the
flat rate from the last few groups. This removes residual group-correlated
structure not captured by the exponential model and is the dominant correction
for faint targets.

The corrected cube is reconstructed from corrected gradients:

```python
cube_cor[:, 1:] = cube[:, :1] + np.cumsum(grads_cor, axis=1)
```

### Lightcurve Improvement

| Target | Raw RMS | Corrected RMS | Improvement |
|---|---|---|---|
| EV Lac | 2.477% | 0.150% | 16× |
| TRAPPIST-1 | 0.827% | 0.314% | 2.6× |

---

## Automated BFE Parameter Fitting (`fit_bfe_params`)

### Source Detection

The detection image is the median over integrations and gradient indices
1 to n_grads-1. SEP (`sep.extract`) is run at 5σ. Sources are filtered to:
- More than 20 pixels from the image boundary
- Semi-major/minor axis ratio a/b < 3 (excludes elongated edge artifacts)
- Isophotal flux ≥ 50,000 DN (minimum for reliable BFE fitting)

If no source meets the brightness threshold, the BFE correction is skipped
(`A_bfe = 0`).

### Forward Model Fit

RCD parameters (tau, rate_map, Adec_map) are fitted from a background annulus.
The forward model runs on a cropped region (crop = cut + kh + 30 px). A_BFE
is fitted by minimising the noise-weighted chi-squared between simulated and
observed late−early normalised PSF difference:

$$\chi^2 = \sum_\text{pixels} \left(\frac{\text{sim\_diff} - \text{obs\_diff}}{\sigma_\text{diff}}\right)^2$$

The fitting radius `fit_r` is determined automatically from the SNR profile of
the observed PSF difference — the outermost radius where SNR > 2, with a
minimum of 5 pixels.

By default alpha is fixed at 2.797. Passing `alpha_bfe=None` fits both A_BFE
and alpha simultaneously using Powell minimisation.

### Usage

```python
from ramp_correction import correct_bfe_rcd, fit_bfe_params

# Auto-fit A_bfe (alpha fixed at 2.797), then correct
cube_cor = correct_bfe_rcd(cube, fit_bfe=True, sci_mask=sci_mask, verbose=True)

# Fit only A_bfe (alpha fixed)
A_bfe, sx, sy = fit_bfe_params(cube, sci_mask=sci_mask, verbose=True)

# Fit both A_bfe and alpha
A_bfe, alpha, sx, sy = fit_bfe_params(cube, alpha_bfe=None, sci_mask=sci_mask, verbose=True)
```

---

## Fitted Parameters

Alpha is consistent across Wolf-359 and EV Lac (~2.78–2.83), confirming the
kernel shape is a detector property. The default alpha (2.797) is a SNR-weighted
average of the two fits. A_BFE varies by detector position.

| Target | A_BFE | alpha | tau (groups) | Method |
|---|---|---|---|---|
| Wolf-359 | 1.035 × 10⁻⁶ | 2.783 | 1.498 | Free alpha, Powell |
| EV Lac | 3.11 × 10⁻⁷ | 2.826 | 1.251 | Free alpha, Powell |
| TRAPPIST-1 | 3.72 × 10⁻⁷ | 2.797 (fixed) | 1.819 | Alpha fixed, FIT_R=5 |

TRAPPIST-1 alpha is fixed because its weaker BFE signal cannot constrain the
kernel shape independently. Its A_BFE is better constrained by fitting only the
inner 5 pixels (FIT_R=5) where the SNR is highest.

---

## Transient Injection Test

A synthetic transient (bright spike + exponential decay, peak 5000 DN/group ≈
8% of TRAPPIST-1 stellar flux, τ = 5 integrations) was injected at the star
position using the MIRI F1500W PSF from WebbPSF. The BFE+RCD correction was
applied to both the original and injected cubes.

**Result**: correction impact on the transient peak = **0.011%** — negligible.
The transient contributes charge only within the integration where it occurs,
so it does not substantially change the accumulated Q that drives the BFE
correction.

---

## Key Scripts

| Script | Purpose |
|---|---|
| `ramp_correction.py` | `correct_bfe_rcd` (3-step hybrid correction) and `fit_bfe_params` (automated BFE fitting with SNR-based fit radius and brightness threshold) |
| `apply_nonparametric_correction.py` | Stage-by-stage comparison: raw → BFE only → joint parametric → hybrid |
| `fit_combined_model.py` | Wolf-359 forward model fit (free alpha, Powell) |
| `fit_combined_model_evlac.py` | EV Lac forward model fit (free alpha, Powell, noise-weighted) |
| `fit_combined_model_trappist.py` | TRAPPIST-1 forward model fit (alpha fixed, FIT_R=5) |
| `test_evlac.py` | Apply `correct_bfe_rcd` to EV Lac; aperture lightcurves before/after |
| `test_trappist.py` | Apply `correct_bfe_rcd` to TRAPPIST-1 with jurassic mask |
| `validate_fit_bfe_params.py` | Validate automated source detection and A_BFE fitting |
| `inject_transient.py` | Transient injection-recovery test using MIRI F1500W PSF |

---

## BFE Kernel

The kernel spans 41×41 pixels (kh=20), is radially symmetric, and sums to zero:

```python
kh = 20
ii, jj = np.mgrid[-kh:kh+1, -kh:kh+1].astype(float)
r = np.sqrt(ii**2 + jj**2)
K = np.where(r > 0, -1.0 / r**alpha, 0.0)
K[kh, kh] = -K.sum()   # enforces ΣK = 0
```

The off-diagonal values fall as 1/r^2.8 — by r=10 px they are ~10× smaller
than at r=1, and by r=20 px ~700× smaller.

---

## Status

The joint BFE+RCD correction is validated on three MIRI targets with a
flux-conserving iterative BFE inversion (Step 1). The correction has negligible
impact on background pixels (<0.05%) and on injected transients (<0.02%).

### Open Questions

1. **Spatial variation of A_BFE**: the factor of ~3 difference between Wolf-359
   and EV Lac/TRAPPIST-1 likely reflects spatial non-uniformity of the BFE
   coupling across the MIRI detector.

2. **Kernel symmetry**: the current kernel is radially symmetric. Real MIRI BFE
   may have preferred directions (readout, crystal axes) that a purely radial
   kernel cannot capture.

3. **Last-frame anomaly**: the final gradient in each integration is excluded.
   Its cause is unknown.
