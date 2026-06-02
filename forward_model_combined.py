"""
forward_model_combined.py

Forward model combining charge reset decay and BFE for MIRI ramp data.

The full model for the observed gradient at group g is:

    grad_obs(g) = [rate + D(g)] * (1 - A_bfe * K ⊛ Q(g))

where:
    rate       -- true photon rate per pixel (DN/group)
    D(g)       -- reset decay: A_decay * exp(-g/tau)  (additive, per pixel)
    Q(g)       -- accumulated charge including reset decay:
                  Q(g) = sum_{k=0}^{g} [rate + D(k)]
    K          -- BFE kernel (1/r^alpha, flux-conserving)
    A_bfe      -- BFE coupling coefficient (DN^-1)

The reset decay charge inflates Q at early groups, so the BFE field is
stronger early in the ramp than it would be from photon charge alone.

This script:
1. Fits reset decay parameters (tau, per-pixel A_decay, C) from obs 1
2. Simulates the combined forward model using best-fit BFE parameters
3. Compares simulated early/late PSF difference to observed

No existing files are overwritten.  All outputs go to bfe_correction/.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit

BASE = Path('/Users/rri38/Documents/work/code/jwst/ramps/wolf-359')
OUT  = Path(__file__).parent

# ---------------------------------------------------------------------------
# BFE parameters from PSF morphology fit
# ---------------------------------------------------------------------------
A_BFE = 2.148e-7
ALPHA = 2.319
KH    = 20

def make_kernel(alpha, kh=KH):
    ii, jj = np.mgrid[-kh:kh+1, -kh:kh+1].astype(float)
    r = np.sqrt(ii**2 + jj**2)
    K = np.where(r > 0, -1.0 / r**alpha, 0.0)
    K[kh, kh] = -K.sum()
    return K

K = make_kernel(ALPHA)

# ---------------------------------------------------------------------------
# Load Wolf-359 obs 1
# ---------------------------------------------------------------------------
with fits.open(BASE / 'uncal-fits/jw06122002001_02101_00001_mirimage_uncal.fits') as h:
    cube = h['SCI'].data.astype(float)   # (n_int, 11, ny, nx)

n_int, n_groups, ny, nx = cube.shape
n_grads = n_groups - 1
print(f'Loaded: {cube.shape}')

grads = np.diff(cube, axis=1)                     # (n_int, 10, ny, nx)
med_grad = np.median(grads, axis=0)               # (10, ny, nx)
g_arr = np.arange(n_grads, dtype=float)

# ---------------------------------------------------------------------------
# Fit reset decay: global tau, then per-pixel [C, A_decay, delta]
# Exclude gradient 9 (last-frame anomaly)
# ---------------------------------------------------------------------------
N_GRADS_FIT = n_grads - 1   # gradients 0-8

mean_profile = np.nanmean(med_grad[:N_GRADS_FIT, :, :], axis=(1, 2))
mean_profile_fit = mean_profile[1:]  # exclude g=0 (first-frame anomaly)

def exp_model(g, C, A, t):
    return C + A * np.exp(-g / t)

popt, _ = curve_fit(exp_model, g_arr[1:N_GRADS_FIT], mean_profile_fit,
                    p0=[mean_profile_fit[-1],
                        mean_profile_fit[0] - mean_profile_fit[-1], 1.5])
tau = float(popt[2])
print(f'Global tau = {tau:.3f} groups')

exp_g    = np.exp(-g_arr[:N_GRADS_FIT] / tau)
ff_col   = np.zeros(N_GRADS_FIT); ff_col[0] = -1.0
X        = np.column_stack([np.ones(N_GRADS_FIT), exp_g, ff_col])  # (9, 3)
params, _, _, _ = np.linalg.lstsq(
    X, med_grad[:N_GRADS_FIT].reshape(N_GRADS_FIT, -1), rcond=None)

rate_map  = params[0].reshape(ny, nx)     # true photon rate C
Adec_map  = params[1].reshape(ny, nx)     # reset decay amplitude A_decay
delta_map = params[2].reshape(ny, nx)     # first-frame offset

print(f'rate_map  : {rate_map.mean():.1f} DN/group (mean)')
print(f'Adec_map  : {Adec_map.mean():.1f} DN/group (mean)')

# ---------------------------------------------------------------------------
# Combined forward model
#
# For each group g, the true gradient (before BFE) is:
#   true_grad(g) = rate + A_decay * exp(-g/tau)   [+ first-frame term at g=0]
#
# Accumulated charge Q(g) = sum_{k=0}^{g} true_grad(k)
#                         = rate*(g+1) + A_decay*tau*(1 - exp(-(g+1)/tau))
#                           [plus first-frame offset at g=0]
#
# BFE-modified gradient:
#   grad_obs(g) = true_grad(g) * (1 - A_bfe * K ⊛ Q(g))
# ---------------------------------------------------------------------------
def simulate_combined(rate_map, Adec_map, delta_map, tau,
                      A_bfe, K, n_grads_out=None):
    """
    Simulate the combined reset-decay + BFE forward model.

    Returns grads_sim : (n_grads_out, ny, nx)
    """
    if n_grads_out is None:
        n_grads_out = n_grads - 1   # exclude last-frame anomaly

    Q = np.zeros((ny, nx))
    grads_sim = np.zeros((n_grads_out, ny, nx))

    for g in range(n_grads_out):
        # True gradient at group g
        true_grad = rate_map + Adec_map * np.exp(-g / tau)
        if g == 0:
            true_grad -= delta_map   # first-frame offset suppresses gradient 0

        # BFE field from charge accumulated before this group
        KQ = fftconvolve(Q, K, mode='same')

        # Observed gradient
        grads_sim[g] = true_grad * (1.0 - A_bfe * KQ)

        # Accumulate charge (use true gradient so Q is deterministic)
        Q = Q + true_grad

    return grads_sim

grads_sim = simulate_combined(rate_map, Adec_map, delta_map, tau, A_BFE, K)
print('Forward model simulated.')

# ---------------------------------------------------------------------------
# Compare simulated vs observed PSF: early and late groups
# ---------------------------------------------------------------------------
STAR_Y, STAR_X = 89, 110
AP_RADIUS = 6
CUT = 20
EARLY = [1, 2, 3]
LATE  = [6, 7, 8]

def cutout_psf(arr_3d, group_list):
    """Flux-normalised median PSF cutout from (n_grads, ny, nx) array."""
    stack = np.median(arr_3d[group_list, :, :], axis=0)
    cut = stack[STAR_Y-CUT:STAR_Y+CUT+1, STAR_X-CUT:STAR_X+CUT+1]
    yy, xx = np.mgrid[:cut.shape[0], :cut.shape[1]]
    ap = np.sqrt((yy-CUT)**2 + (xx-CUT)**2) <= AP_RADIUS
    return cut / cut[ap].sum()

# Observed
obs_early = cutout_psf(med_grad, EARLY)
obs_late  = cutout_psf(med_grad, LATE)
obs_diff  = obs_late - obs_early

# Simulated (combined model)
sim_early = cutout_psf(grads_sim, EARLY)
sim_late  = cutout_psf(grads_sim, LATE)
sim_diff  = sim_late - sim_early

# Simulated BFE-only (no reset decay in Q)
def simulate_bfe_only(rate_map, tau, A_bfe, K, n_grads_out=None):
    if n_grads_out is None:
        n_grads_out = n_grads - 1
    Q = np.zeros((ny, nx))
    grads_s = np.zeros((n_grads_out, ny, nx))
    for g in range(n_grads_out):
        KQ = fftconvolve(Q, K, mode='same')
        grads_s[g] = rate_map * (1.0 - A_bfe * KQ)
        Q += rate_map
    return grads_s

grads_bfe_only = simulate_bfe_only(rate_map, tau, A_BFE, K)
bfe_early = cutout_psf(grads_bfe_only, EARLY)
bfe_late  = cutout_psf(grads_bfe_only, LATE)
bfe_diff  = bfe_late - bfe_early

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
vabs = np.nanpercentile(np.abs(obs_diff), 99)
ext = [-CUT-0.5, CUT+0.5, -CUT-0.5, CUT+0.5]

fig, axes = plt.subplots(3, 3, figsize=(13, 11))

rows = [
    (obs_early,  obs_late,  obs_diff,  'Observed'),
    (sim_early,  sim_late,  sim_diff,  'Combined model (reset decay + BFE)'),
    (bfe_early,  bfe_late,  bfe_diff,  'BFE-only model'),
]

for row_idx, (early, late, diff, label) in enumerate(rows):
    axes[row_idx, 0].set_ylabel(label, fontsize=9)
    for col_idx, (img, title) in enumerate([
            (early, f'Early groups {EARLY}'),
            (late,  f'Late groups {LATE}'),
            (diff,  'Late − Early'),
    ]):
        ax = axes[row_idx, col_idx]
        if col_idx == 2:
            im = ax.imshow(img, origin='lower', cmap='RdBu_r',
                           vmin=-vabs, vmax=vabs, extent=ext)
        else:
            im = ax.imshow(img, origin='lower', cmap='viridis', extent=ext)
        fig.colorbar(im, ax=ax, label='Norm. flux')
        if row_idx == 0:
            ax.set_title(title, fontsize=9)
        ax.set_xlabel('Δx (px)')

# Radial profiles of difference images
fig2, ax2 = plt.subplots(figsize=(7, 4))
yy, xx = np.mgrid[:obs_diff.shape[0], :obs_diff.shape[1]]
r_map = np.sqrt((yy-CUT)**2 + (xx-CUT)**2)
r_int = np.arange(0, CUT)

def radial(img):
    return np.array([np.mean(img[np.round(r_map).astype(int)==r]) for r in r_int])

ax2.plot(r_int, radial(obs_diff),  'k-o',  ms=4, lw=1.5, label='Observed diff')
ax2.plot(r_int, radial(sim_diff),  'C3--s', ms=4, lw=1.5, label='Combined model diff')
ax2.plot(r_int, radial(bfe_diff),  'C0--^', ms=4, lw=1.5, label='BFE-only diff')
ax2.axhline(0, color='k', lw=0.7, ls=':', alpha=0.5)
ax2.set_xlabel('Radius (px)')
ax2.set_ylabel('Mean late−early (norm. flux)')
ax2.set_title('Radial profile of late−early PSF difference')
ax2.legend(fontsize=9)

fig.suptitle('Wolf-359: combined reset-decay + BFE forward model vs observed',
             fontsize=11, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'wolf359_combined_forward_model.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig2.tight_layout()
fig2.savefig(OUT / 'wolf359_combined_forward_model_radial.png', dpi=150, bbox_inches='tight')
plt.close(fig2)

print(f'Saved wolf359_combined_forward_model.png')
print(f'Saved wolf359_combined_forward_model_radial.png')

# Quantify improvement
obs_d_r  = radial(obs_diff)
sim_d_r  = radial(sim_diff)
bfe_d_r  = radial(bfe_diff)
rms_bfe  = np.sqrt(np.mean((bfe_d_r - obs_d_r)**2))
rms_comb = np.sqrt(np.mean((sim_d_r  - obs_d_r)**2))
print(f'\nRMS residual (BFE only):    {rms_bfe:.6f}')
print(f'RMS residual (combined):    {rms_comb:.6f}')
print(f'Improvement factor:         {rms_bfe/rms_comb:.2f}x')
