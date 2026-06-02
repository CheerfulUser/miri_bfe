"""
apply_joint_correction.py

Joint inversion of the combined reset-decay + BFE forward model.

The forward model is causal:
    grad_obs(g) = true_grad(g) * (1 - A * K ⊛ Q(g))
    true_grad(g) = rate(px) + A_decay(px) * exp(-g/tau)   [+ first-frame offset]
    Q(g)         = sum_{k<g} true_grad(k)

Inversion strategy (avoids sequential-correction interference):

  Step 1 — fit reset decay once on raw gradients:
      fit tau globally, then per-pixel [rate, A_decay, delta] via lstsq.
      Parameters are locked for the rest of the correction.

  Step 2 — causal BFE inversion (single forward pass, exact):
      Q(0) = 0  →  true_grad(0) = grad_obs(0)
      Q(g) = Q(g-1) + true_grad(g-1)   [accumulated corrected charge]
      true_grad(g) = grad_obs(g) / (1 - A * K ⊛ Q(g))

  Step 3 — subtract reset decay analytically using locked parameters:
      grad_final(g) = true_grad(g) - A_decay(px) * exp(-g/tau)
      [plus delta_map term at g=0]

No re-fitting occurs after BFE correction, so reset-decay and BFE
corrections cannot absorb each other.

Parameters from combined fit (fit_combined_model.py):
    A     = 1.035e-6  DN^-1
    alpha = 2.783

Outputs
-------
bfe_correction/wolf359_joint_corrected_psf.png      -- PSF comparison
bfe_correction/wolf359_joint_corrected_profiles.png -- gradient profiles
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from ramp_correction import correct_reset_decay

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
A = 1.035e-6
ALPHA = 2.783

STAR_Y, STAR_X = 89, 110
AP_RADIUS = 6
CUT = 20
EARLY_GROUPS = [1, 2, 3]
LATE_GROUPS = [6, 7, 8]

BASE = Path('/Users/rri38/Documents/work/code/jwst/ramps/wolf-359')
OUT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Build BFE kernel
# ---------------------------------------------------------------------------
def make_kernel(alpha, kh=20):
    ii, jj = np.mgrid[-kh:kh+1, -kh:kh+1].astype(float)
    r = np.sqrt(ii**2 + jj**2)
    K = np.where(r > 0, -1.0 / r**alpha, 0.0)
    K[kh, kh] = -K.sum()
    return K

K = make_kernel(ALPHA)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
uncal_path = BASE / 'uncal-fits/jw06122002001_02101_00001_mirimage_uncal.fits'
with fits.open(uncal_path) as h:
    cube = h['SCI'].data.astype(float)

n_int, n_groups, ny, nx = cube.shape
n_grads = n_groups - 1
n_grads_fit = n_grads - 1       # exclude last gradient (last-frame anomaly)
grads_raw = np.diff(cube, axis=1)   # (n_int, n_grads, ny, nx)
g_arr = np.arange(n_grads_fit, dtype=float)
print(f'Loaded: {cube.shape}')

# ---------------------------------------------------------------------------
# Step 1: fit reset decay parameters on raw gradients (locked)
#
# Global tau from spatial mean over background pixels, excluding g=0
# (first-frame anomaly suppresses g=0, making the profile non-monotonic).
# Per-pixel [rate, A_decay, delta] via lstsq with fixed tau.
# ---------------------------------------------------------------------------
med_grad = np.median(grads_raw, axis=0)   # (n_grads, ny, nx)
mean_profile = np.nanmean(med_grad[:n_grads_fit], axis=(1, 2))

def exp_model(g, C, A, t): return C + A * np.exp(-g / t)
popt, _ = curve_fit(exp_model, g_arr[1:], mean_profile[1:],
                    p0=[mean_profile[-1], mean_profile[0] - mean_profile[-1], 1.5])
tau = float(popt[2])
print(f'Reset decay tau = {tau:.3f} groups (fitted on raw, locked)')

exp_g = np.exp(-g_arr / tau)                      # (n_grads_fit,)
ff_col = np.zeros(n_grads_fit); ff_col[0] = -1.0
X = np.column_stack([np.ones(n_grads_fit), exp_g, ff_col])  # (n_grads_fit, 3)
params, _, _, _ = np.linalg.lstsq(
    X, med_grad[:n_grads_fit].reshape(n_grads_fit, -1), rcond=None)
rate_map  = params[0].reshape(ny, nx)
Adec_map  = params[1].reshape(ny, nx)
delta_map = params[2].reshape(ny, nx)
print(f'rate_map  mean: {rate_map.mean():.2f} DN/group')
print(f'Adec_map  mean: {Adec_map.mean():.2f} DN/group')

# ---------------------------------------------------------------------------
# Step 2: causal BFE inversion (single forward pass)
#
# Q(g) is built from the already-corrected (previous) gradients.
# This is exact because Q(g) depends only on k < g.
# The BFE field uses median Q over integrations (same as the fit).
# ---------------------------------------------------------------------------
grads_bfe = grads_raw.copy()
Q_med = np.zeros((ny, nx))

for g in range(n_grads):
    if g > 0:
        Q_med = Q_med + np.median(grads_bfe[:, g-1, :, :], axis=0)

    KQ = fftconvolve(Q_med, K, mode='same')
    factor = 1.0 - A * KQ
    factor = np.where(factor > 0.05, factor, 1.0)
    grads_bfe[:, g, :, :] = grads_raw[:, g, :, :] / factor[None, :, :]
    print(f'  BFE g={g}: factor [{factor.min():.4f}, {factor.max():.4f}]', end='\r')

print()
print('BFE inversion done.')

# ---------------------------------------------------------------------------
# Step 3: subtract reset decay using locked parameters (no re-fitting)
#
# Correction in gradient space:
#   g=0: grads_final = grads_bfe - Adec_map + delta_map
#   g>0: grads_final = grads_bfe - Adec_map * exp(-g/tau)
# Last gradient (g=n_grads-1) excluded (last-frame anomaly).
# ---------------------------------------------------------------------------
grads_joint = grads_bfe.copy()

for g in range(n_grads_fit):
    decay_g = Adec_map * np.exp(-g / tau)
    if g == 0:
        grads_joint[:, 0, :, :] = grads_bfe[:, 0, :, :] - decay_g[None] + delta_map[None]
    else:
        grads_joint[:, g, :, :] = grads_bfe[:, g, :, :] - decay_g[None]

print('Reset decay subtracted (locked parameters).')

# ---------------------------------------------------------------------------
# Baseline: reset-decay-only correction for comparison
# ---------------------------------------------------------------------------
cube_rd = correct_reset_decay(cube, method='median')
grads_rd = np.diff(cube_rd, axis=1)
print('Reset-decay-only baseline done.')

# ---------------------------------------------------------------------------
# PSF comparison: early vs late groups
# ---------------------------------------------------------------------------
sy, sx = STAR_Y, STAR_X

def group_psf(grads, group_list):
    stack = np.median(grads[:, group_list, :, :], axis=(0, 1))
    cut = stack[sy-CUT:sy+CUT+1, sx-CUT:sx+CUT+1]
    yy, xx = np.mgrid[:cut.shape[0], :cut.shape[1]]
    ap = np.sqrt((yy-CUT)**2 + (xx-CUT)**2) <= AP_RADIUS
    return cut / cut[ap].sum()

early_rd    = group_psf(grads_rd,    EARLY_GROUPS)
late_rd     = group_psf(grads_rd,    LATE_GROUPS)
early_joint = group_psf(grads_joint, EARLY_GROUPS)
late_joint  = group_psf(grads_joint, LATE_GROUPS)

diff_rd    = late_rd    - early_rd
diff_joint = late_joint - early_joint
vabs = np.nanpercentile(np.abs(diff_rd), 99)

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
ext = [-CUT-0.5, CUT+0.5, -CUT-0.5, CUT+0.5]

for row, (early, late, diff, label) in enumerate([
        (early_rd,    late_rd,    diff_rd,    'Reset-decay corrected only'),
        (early_joint, late_joint, diff_joint, 'Joint BFE + reset-decay correction'),
]):
    axes[row, 0].set_ylabel(label, fontsize=9)
    for col, (img, title) in enumerate([
            (early, f'Early groups {EARLY_GROUPS}'),
            (late,  f'Late groups {LATE_GROUPS}'),
            (diff,  'Late − Early'),
    ]):
        ax = axes[row, col]
        if col == 2:
            im = ax.imshow(img, origin='lower', cmap='RdBu_r',
                           vmin=-vabs, vmax=vabs, extent=ext)
        else:
            im = ax.imshow(img, origin='lower', cmap='viridis', extent=ext)
        fig.colorbar(im, ax=ax, label='Norm. flux')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Δx (px)')
        ax.set_ylabel('Δy (px)')

fig.suptitle('Wolf-359 PSF: joint BFE + reset-decay correction',
             fontsize=11, fontweight='bold')
fig.tight_layout()
out1 = OUT / 'wolf359_joint_corrected_psf.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out1}')

# ---------------------------------------------------------------------------
# Gradient profiles at PSF core
# ---------------------------------------------------------------------------
yy, xx = np.mgrid[:ny, :nx]
ap_mask = (yy-sy)**2 + (xx-sx)**2 <= AP_RADIUS**2
g_plot = np.arange(n_grads_fit)

prof_raw   = np.array([np.median(grads_raw[:,   g, ap_mask]) for g in g_plot])
prof_rd    = np.array([np.median(grads_rd[:,    g, ap_mask]) for g in g_plot])
prof_joint = np.array([np.median(grads_joint[:, g, ap_mask]) for g in g_plot])

norm_g = 5
prof_raw_n   = prof_raw   / prof_raw[norm_g]
prof_rd_n    = prof_rd    / prof_rd[norm_g]
prof_joint_n = prof_joint / prof_joint[norm_g]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(g_plot, prof_raw_n,   'o-', color='C3', lw=1.5, label='Raw')
ax.plot(g_plot, prof_rd_n,    's-', color='C1', lw=1.5, label='Reset-decay only')
ax.plot(g_plot, prof_joint_n, '^-', color='C0', lw=1.5, label='Joint correction')
ax.axhline(1.0, color='k', lw=0.7, ls='--', alpha=0.5)
ax.set_xlabel('Gradient index')
ax.set_ylabel(f'Normalised gradient (rel. to group {norm_g})')
ax.set_title('PSF-core aperture gradient profile')
ax.legend(fontsize=9)
ax.set_xticks(g_plot)

ax = axes[1]
ax.plot(g_plot[1:], (prof_rd_n    - prof_raw_n)[1:] * 100, 's-', color='C1',
        lw=1.5, label='Reset-decay only')
ax.plot(g_plot[1:], (prof_joint_n - prof_raw_n)[1:] * 100, '^-', color='C0',
        lw=1.5, label='Joint correction')
ax.axhline(0, color='k', lw=0.7, ls='--', alpha=0.5)
ax.set_xlabel('Gradient index')
ax.set_ylabel('Correction (% of raw gradient)')
ax.set_title('Correction magnitude vs gradient index')
ax.legend(fontsize=9)
ax.set_xticks(g_plot[1:])

fig.suptitle('Wolf-359: gradient profiles — raw vs corrections',
             fontsize=11, fontweight='bold')
fig.tight_layout()
out2 = OUT / 'wolf359_joint_corrected_profiles.png'
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out2}')

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
rms_rd    = np.sqrt(np.mean((prof_rd_n[1:]    - 1.0)**2))
rms_joint = np.sqrt(np.mean((prof_joint_n[1:] - 1.0)**2))
print(f'\nGradient profile flatness (RMS deviation from unity, groups 1-8):')
print(f'  Reset-decay only : {rms_rd:.5f}')
print(f'  Joint correction : {rms_joint:.5f}')
if rms_joint < rms_rd:
    print(f'  Improvement      : {rms_rd/rms_joint:.2f}x flatter')
else:
    print(f'  Joint is {rms_joint/rms_rd:.2f}x worse — BFE overcorrects profile at this aperture')

# PSF difference amplitude at core pixel
core_amp_rd    = np.abs(diff_rd   [CUT, CUT])
core_amp_joint = np.abs(diff_joint[CUT, CUT])
print(f'\nLate−Early PSF diff at core pixel:')
print(f'  Reset-decay only : {core_amp_rd:.5f}')
print(f'  Joint correction : {core_amp_joint:.5f}')
print(f'  Suppression      : {(1 - core_amp_joint/core_amp_rd)*100:.1f}%')
