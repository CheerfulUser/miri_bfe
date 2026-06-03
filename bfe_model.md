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
timescale in groups, and `Delta` is the first-frame offset (a separate anomaly
at group 0). `tau` is global (same for all pixels in a subarray); `C`, `A`, and
`Delta` are fitted per pixel.

### Brighter-Fatter Effect

Accumulated charge Q(g) in a pixel repels incoming photoelectrons, reducing the
effective collection area. For a charge sheet in 2D, the electrostatic force
falls off as 1/r, giving the BFE kernel:

$$K(i,j) = -\frac{1}{r^\alpha}, \quad r > 0; \qquad K(0,0) = -\sum_{(i,j)\neq(0,0)} K(i,j)$$

The positive centre (charge repels new electrons out of the pixel) and negative
off-diagonal elements (neighbouring pixels gain them) conserve total flux.
The exponent alpha absorbs the effective charge-spreading geometry; empirically
alpha ≈ 2.8 for MIRI.

---

## Combined Forward Model

The observed gradient at group g is:

$$\text{grad}_\text{obs}(g) = \underbrace{\left[C + A \cdot e^{-g/\tau} - \delta_{g,0}\Delta\right]}_{\text{true gradient}} \times \left[1 - A_\text{BFE} \cdot (K \ast Q)(g)\right]$$

where Q(g) is the accumulated charge up to group g:

$$Q(g) = \sum_{k=0}^{g-1} \text{grad}_\text{true}(k)$$

The BFE and RCD are separable in the median over integrations: the BFE factor
depends on the median Q (identical for all integrations), so the median gradient
is a clean template for both effects.

---

## Correction Pipeline (`correct_bfe_rcd`)

Three sequential steps applied to the raw gradients:

### Step 1 — Causal BFE Inversion

For each group in order (causal), divide out the BFE suppression factor:

$$\text{grad}_\text{BFE}(g) = \frac{\text{grad}_\text{obs}(g)}{1 - A_\text{BFE} \cdot (K \ast Q_\text{med})(g)}$$

where Q_med is built from the running sum of the median-over-integrations of
the BFE-corrected gradients. Since Q_med is the same for every integration, this
factor is identical across integrations — the BFE correction does not introduce
integration-to-integration noise.

### Step 2 — Parametric RCD Subtraction

From the BFE-corrected gradients, fit the global decay timescale tau from
background pixels (excluding the star), then fit per-pixel [C, A, Delta] via
least squares. Subtract the fitted decay from every integration.

### Step 3 — Non-Parametric Residual Removal

Subtract the per-pixel per-group median over integrations from the
parametrically corrected gradients, then add back the flat rate estimated from
the last few groups. This removes any residual group-correlated structure not
captured by the exponential model (detector non-idealities, column effects,
etc.) and is the dominant correction for faint targets.

The corrected cube is reconstructed by integrating the corrected gradients:

```python
cube_cor[:, 1:] = cube[:, :1] + np.cumsum(grads_cor, axis=1)
```

### Lightcurve Improvement

| Target | Raw RMS | Corrected RMS | Improvement |
|---|---|---|---|
| Wolf-359 | — | — | — |
| EV Lac | 2.477% | 0.152% | 16× |
| TRAPPIST-1 | 0.827% | 0.314% | 2.6× |

---

## Automated BFE Parameter Fitting (`fit_bfe_params`)

The function `fit_bfe_params` fits A_BFE directly from the data without
requiring prior knowledge of the star position or BFE parameters.

### Source Detection

A detection image is formed from the median over integrations and gradient
indices 1 to n_grads-1 (excluding the first-frame anomaly and last-frame
anomaly). SEP (`sep.extract`) is run with a 5-sigma threshold. Edge artifacts
are excluded by requiring the source to lie more than 20 pixels from the image
boundary and to have a semi-major/minor axis ratio a/b < 3. The brightest
remaining source is taken as the target star.

### Forward Model Fit

The reset-decay parameters (tau, rate_map, Adec_map, delta_map) are fitted from
the median gradient over a background annulus around the detected star. The
forward model is then run on a cropped region around the star (crop = cut + kh
+ 30 pixels, where kh = 20 is the kernel half-width). A_BFE is fitted by
minimising the noise-weighted chi-squared between the simulated and observed
late − early normalised PSF difference:

$$\chi^2 = \sum_\text{pixels} \left(\frac{\text{sim\_diff} - \text{obs\_diff}}{\sigma_\text{diff}}\right)^2$$

where sigma_diff is the standard error of the per-integration PSF differences
across all integrations. alpha is held fixed at 2.783. The minimisation uses
`scipy.optimize.minimize_scalar` with bounded search over log10(A_BFE) in
[-9, -4].

### Usage

```python
from ramp_correction import correct_bfe_rcd, fit_bfe_params

# Fit A_bfe automatically then correct
cube_cor = correct_bfe_rcd(cube, fit_bfe=True, sci_mask=sci_mask,
                           bg_mask=bg_mask, verbose=True)

# Or fit separately and inspect
A_bfe, star_x, star_y = fit_bfe_params(cube, sci_mask=sci_mask, verbose=True)
```

---

## Fitted Parameters

Alpha is consistent across all three targets (~2.8), confirming the BFE kernel
shape is a detector property. A_BFE varies spatially — Wolf-359, EV Lac, and
TRAPPIST-1 land on different detector regions.

| Target | A_BFE | alpha | tau (groups) |
|---|---|---|---|
| Wolf-359 | 1.035 × 10⁻⁶ | 2.783 | 1.498 |
| EV Lac | 2.93 × 10⁻⁷ | 2.800 (fixed) | 1.251 |
| TRAPPIST-1 | 1.20 × 10⁻⁷ | 2.783 (fixed) | 1.819 |

Wolf-359 parameters were fitted with alpha free (2D differential evolution +
Nelder-Mead). EV Lac and TRAPPIST-1 have alpha fixed to the Wolf-359 value
because their shorter ramps or weaker BFE signal do not constrain alpha
independently.

---

## Key Scripts

| Script | Purpose |
|---|---|
| `ramp_correction.py` | `correct_bfe_rcd` (3-step hybrid correction) and `fit_bfe_params` (automated BFE fitting) |
| `apply_nonparametric_correction.py` | Stage-by-stage comparison: raw → BFE only → joint parametric → hybrid |
| `fit_combined_model.py` | Original Wolf-359 forward model fit (free alpha, 2D optimisation) |
| `fit_combined_model_evlac.py` | EV Lac forward model fit (alpha fixed, 1D minimisation) |
| `fit_combined_model_trappist.py` | TRAPPIST-1 forward model fit (alpha fixed, noise-weighted) |
| `test_evlac.py` | Apply `correct_bfe_rcd` to EV Lac; aperture lightcurves before/after |
| `test_trappist.py` | Apply `correct_bfe_rcd` to TRAPPIST-1; aperture lightcurves before/after |
| `validate_fit_bfe_params.py` | Validate automated source detection and A_BFE fitting on EV Lac and TRAPPIST-1 |

---

## BFE Kernel

The kernel has half-width kh = 20 pixels and is normalised to conserve flux:

```python
kh = 20
ii, jj = np.mgrid[-kh:kh+1, -kh:kh+1].astype(float)
r = np.sqrt(ii**2 + jj**2)
K = np.where(r > 0, -1.0 / r**alpha, 0.0)
K[kh, kh] = -K.sum()
```

---

## Status

The joint BFE + RCD correction is working and validated on three MIRI targets.
The dominant correction for faint targets (TRAPPIST-1) comes from the
non-parametric median subtraction in Step 3. For bright targets (EV Lac, Wolf-359)
the BFE inversion in Step 1 is the primary improvement.

Automated parameter fitting (`fit_bfe_params`) recovers A_BFE to within ~2%
of the standalone forward model fits on both EV Lac and TRAPPIST-1.

### Open Questions

1. **Spatial variation of A_BFE**: the factor of ~3–8 difference between
   Wolf-359 and the other two targets likely reflects spatial non-uniformity
   of the BFE coupling across the MIRI detector, not a physical difference
   between sources.

2. **Short-ramp degeneracy**: with ≤5 gradient groups (EV Lac), tau and A_BFE
   are partially degenerate. Fixing alpha and using the non-parametric Step 3
   mitigates this but does not eliminate it.

3. **Last-frame anomaly**: the final gradient in each integration is always
   excluded. Its cause is unknown and it is not corrected.
