"""
apply_bfe_correction.py

Apply the physics-driven BFE correction to Wolf-359 MIRI ramp data.

The BFE forward model at the gradient level:
    grad_obs(g) = grad_true(g) * (1 - A * K ⊛ Q(g))

where Q(g) = cube[:, g, :, :] is the accumulated charge at group g.

Inverted per gradient via fixed-point iteration:
    f_{n+1}(g) = grad_obs(g) + A * f_n(g) * (K ⊛ Q_obs(g))

until convergence. Q_obs(g) is fixed (from the raw ramp) and only the
gradient estimate is iterated.

Correction order: BFE first on raw gradients, then reset decay correction
on the BFE-corrected ramp.

Best-fit kernel parameters from PSF morphology fit:
    A     = 2.148e-7  DN^-1
    alpha = 2.319

Outputs
-------
bfe_correction/wolf359_bfe_corrected_psf.png      -- early/late PSF before/after
bfe_correction/wolf359_bfe_corrected_profiles.png -- gradient profiles before/after
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from scipy.signal import fftconvolve
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from ramp_correction import correct_reset_decay

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
A     = 2.148e-7
ALPHA = 2.319
BFE_TOL   = 1e-6
BFE_MAXIT = 30

STAR_Y, STAR_X = 89, 110
AP_RADIUS = 6
EARLY_GROUPS = [1, 2, 3]
LATE_GROUPS  = [6, 7, 8]

BASE = Path('/Users/rri38/Documents/work/code/jwst/ramps/wolf-359')
OUT  = Path(__file__).parent

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
# Iterative BFE corrector at the gradient level
# grad_obs(g) = grad_true(g) * (1 - A * K ⊛ Q(g))
# Fixed point: f = grad_obs + A * f * KQ,  KQ fixed from raw Q
# ---------------------------------------------------------------------------
def bfe_correct_grad(grad_obs, KQ, A, tol=BFE_TOL, max_iter=BFE_MAXIT):
    f = grad_obs.copy()
    for _ in range(max_iter):
        f_new = grad_obs + A * f * KQ
        if np.max(np.abs(f_new - f)) < tol:
            break
        f = f_new
    return f_new

# ---------------------------------------------------------------------------
# Load Wolf-359 obs 1
# ---------------------------------------------------------------------------
uncal_path = BASE / 'uncal-fits/jw06122002001_02101_00001_mirimage_uncal.fits'
with fits.open(uncal_path) as h:
    cube = h['SCI'].data.astype(float)   # (n_int, 11, ny, nx)

n_int, n_groups, ny, nx = cube.shape
n_grads = n_groups - 1
print(f'Loaded cube: {cube.shape}')

# Raw gradients
grads_raw = np.diff(cube, axis=1)   # (n_int, 10, ny, nx)

# ---------------------------------------------------------------------------
# Step 1: BFE correction applied to each gradient frame
# K ⊛ Q(g) uses the median accumulated charge at group g as the BFE field
# ---------------------------------------------------------------------------
grads_bfe = np.empty_like(grads_raw)
for g in range(n_grads):
    print(f'  BFE correcting gradient {g}/{n_grads-1}...', end='\r')
    Q_g = np.median(cube[:, g, :, :], axis=0)   # charge at start of collection
    KQ  = fftconvolve(Q_g, K, mode='same')       # BFE field (ny, nx)
    for i in range(n_int):
        grads_bfe[i, g] = bfe_correct_grad(grads_raw[i, g], KQ, A)
print()
print('BFE correction done.')

# ---------------------------------------------------------------------------
# Step 2: Reconstruct BFE-corrected ramp from corrected gradients,
# then apply reset decay correction
# ---------------------------------------------------------------------------
cube_bfe = np.empty_like(cube)
cube_bfe[:, 0, :, :] = cube[:, 0, :, :]   # group 0 unchanged
for g in range(n_grads):
    cube_bfe[:, g+1, :, :] = cube_bfe[:, g, :, :] + grads_bfe[:, g, :, :]

cube_rd     = correct_reset_decay(cube,     method='median')
cube_bfe_rd = correct_reset_decay(cube_bfe, method='median')
print('Reset decay corrected.')

grads_rd  = np.diff(cube_rd,     axis=1)
grads_bfe_rd = np.diff(cube_bfe_rd, axis=1)

# ---------------------------------------------------------------------------
# PSF comparison: early vs late, before and after BFE correction
# ---------------------------------------------------------------------------
CUT = 20
sy, sx = STAR_Y, STAR_X

def group_psf(grads, group_list):
    stack = np.median(grads[:, group_list, :, :], axis=(0, 1))
    cut = stack[sy-CUT:sy+CUT+1, sx-CUT:sx+CUT+1]
    yy, xx = np.mgrid[:cut.shape[0], :cut.shape[1]]
    ap = np.sqrt((yy-CUT)**2 + (xx-CUT)**2) <= AP_RADIUS
    return cut / cut[ap].sum()

early_rd  = group_psf(grads_rd,     EARLY_GROUPS)
late_rd   = group_psf(grads_rd,     LATE_GROUPS)
early_bfe = group_psf(grads_bfe_rd, EARLY_GROUPS)
late_bfe  = group_psf(grads_bfe_rd, LATE_GROUPS)

diff_rd  = late_rd  - early_rd
diff_bfe = late_bfe - early_bfe
vabs = np.nanpercentile(np.abs(diff_rd), 99)

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
ext = [-CUT-0.5, CUT+0.5, -CUT-0.5, CUT+0.5]

for row, (early, late, diff, label) in enumerate([
        (early_rd,  late_rd,  diff_rd,  'Reset-decay corrected (no BFE correction)'),
        (early_bfe, late_bfe, diff_bfe, 'After BFE correction'),
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
        ax.set_xlabel('Δx (px)'); ax.set_ylabel('Δy (px)')

fig.suptitle('Wolf-359 PSF: BFE correction effect on early vs late groups',
             fontsize=11, fontweight='bold')
fig.tight_layout()
out1 = OUT / 'wolf359_bfe_corrected_psf.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out1}')

# ---------------------------------------------------------------------------
# Gradient profiles at PSF core before and after BFE correction
# ---------------------------------------------------------------------------
yy, xx = np.mgrid[:ny, :nx]
ap_mask = (yy-sy)**2 + (xx-sx)**2 <= AP_RADIUS**2
g_arr = np.arange(n_grads)

prof_rd  = np.array([np.median(grads_rd[:,     g, ap_mask]) for g in g_arr])
prof_bfe = np.array([np.median(grads_bfe_rd[:, g, ap_mask]) for g in g_arr])

norm_g = 5
prof_rd_n  = prof_rd  / prof_rd[norm_g]
prof_bfe_n = prof_bfe / prof_bfe[norm_g]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.plot(g_arr, prof_rd_n,  'o-', color='C3', lw=1.5, label='Reset-decay corrected')
ax.plot(g_arr, prof_bfe_n, 's-', color='C0', lw=1.5, label='+ BFE corrected')
ax.axhline(1.0, color='k', lw=0.7, ls='--', alpha=0.5)
ax.set_xlabel('Gradient index')
ax.set_ylabel(f'Normalised gradient (rel. to group {norm_g})')
ax.set_title('Aperture-summed gradient profile at PSF core')
ax.legend(fontsize=9)
ax.set_xticks(g_arr)

ax = axes[1]
ax.plot(g_arr[1:-1], (prof_bfe_n - prof_rd_n)[1:-1] * 100, 'o-', color='C2', lw=1.5)
ax.axhline(0, color='k', lw=0.7, ls='--', alpha=0.5)
ax.set_xlabel('Gradient index')
ax.set_ylabel('BFE correction (% of gradient)')
ax.set_title('Change in normalised gradient from BFE correction')
ax.set_xticks(g_arr[1:-1])

fig.suptitle('Wolf-359 — gradient profile: reset-decay vs reset-decay + BFE correction',
             fontsize=11, fontweight='bold')
fig.tight_layout()
out2 = OUT / 'wolf359_bfe_corrected_profiles.png'
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out2}')
