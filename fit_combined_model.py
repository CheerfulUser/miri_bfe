"""
fit_combined_model.py

Fit the combined reset-decay + BFE forward model to Wolf-359 PSF data.

The full forward model:
    grad_obs(g) = [rate + A_decay*exp(-g/tau)] * (1 - A_bfe * K ⊛ Q(g))
    Q(g)        = sum_{k<g} [rate + A_decay*exp(-k/tau)]  (accumulated charge)

Reset decay parameters (tau, rate_map, Adec_map) are fixed from a
per-pixel lstsq fit to the median gradient profile.

Free parameters for optimisation:
    log10(A_bfe), alpha

Objective: minimise pixel-wise residual between simulated and observed
late-early PSF difference image within r <= 12 px.

Outputs go to bfe_correction/. Nothing is overwritten.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from scipy.signal import fftconvolve
from scipy.optimize import differential_evolution, minimize
from scipy.optimize import curve_fit

BASE = Path('/Users/rri38/Documents/work/code/jwst/ramps/wolf-359')
OUT  = Path(__file__).parent

STAR_Y, STAR_X = 89, 110
AP_RADIUS = 6
CUT = 20
EARLY = [1, 2, 3]
LATE  = [6, 7, 8]
FIT_R = 12

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with fits.open(BASE / 'uncal-fits/jw06122002001_02101_00001_mirimage_uncal.fits') as h:
    cube = h['SCI'].data.astype(float)

n_int, n_groups, ny, nx = cube.shape
n_grads = n_groups - 1
g_arr = np.arange(n_grads, dtype=float)
grads = np.diff(cube, axis=1)
med_grad = np.median(grads, axis=0)
print(f'Loaded: {cube.shape}')

# ---------------------------------------------------------------------------
# Fit reset decay parameters
# ---------------------------------------------------------------------------
N_FIT = n_grads - 1   # exclude last-frame anomaly

# Mask star core for tau fit — include only background pixels
yy_full, xx_full = np.mgrid[:ny, :nx]
r_star = np.sqrt((yy_full - STAR_Y)**2 + (xx_full - STAR_X)**2)
sci_mask = (r_star > 15) & (r_star < 100)   # background annulus, away from star
mean_profile = np.nanmean(med_grad[:N_FIT, sci_mask], axis=1)

def exp_model(g, C, A, t): return C + A * np.exp(-g / t)
popt, _ = curve_fit(exp_model, g_arr[1:N_FIT], mean_profile[1:],
                    p0=[mean_profile[-1], mean_profile[0]-mean_profile[-1], 1.5])
tau = float(popt[2])
print(f'tau = {tau:.3f} groups')

exp_g  = np.exp(-g_arr[:N_FIT] / tau)
ff_col = np.zeros(N_FIT); ff_col[0] = -1.0
X = np.column_stack([np.ones(N_FIT), exp_g, ff_col])
params, _, _, _ = np.linalg.lstsq(X, med_grad[:N_FIT].reshape(N_FIT, -1), rcond=None)
rate_map  = params[0].reshape(ny, nx)
Adec_map  = params[1].reshape(ny, nx)
delta_map = params[2].reshape(ny, nx)

# ---------------------------------------------------------------------------
# Observed PSF difference
# ---------------------------------------------------------------------------
def cutout_psf(arr_3d, group_list):
    stack = np.median(arr_3d[np.array(group_list)], axis=0)
    cut = stack[STAR_Y-CUT:STAR_Y+CUT+1, STAR_X-CUT:STAR_X+CUT+1]
    yy, xx = np.mgrid[:cut.shape[0], :cut.shape[1]]
    ap = np.sqrt((yy-CUT)**2 + (xx-CUT)**2) <= AP_RADIUS
    return cut / cut[ap].sum()

obs_early = cutout_psf(med_grad, EARLY)
obs_late  = cutout_psf(med_grad, LATE)
obs_diff  = obs_late - obs_early

yy_c, xx_c = np.mgrid[:obs_diff.shape[0], :obs_diff.shape[1]]
fit_mask = np.sqrt((yy_c-CUT)**2 + (xx_c-CUT)**2) <= FIT_R

# ---------------------------------------------------------------------------
# Forward model function
# ---------------------------------------------------------------------------
def simulate(A_bfe, alpha, n_grads_out=None):
    if n_grads_out is None:
        n_grads_out = N_FIT
    kh = 20
    ii, jj = np.mgrid[-kh:kh+1, -kh:kh+1].astype(float)
    r = np.sqrt(ii**2 + jj**2)
    K = np.where(r > 0, -1.0 / r**alpha, 0.0)
    K[kh, kh] = -K.sum()

    Q = np.zeros((ny, nx))
    grads_s = np.zeros((n_grads_out, ny, nx))
    for g in range(n_grads_out):
        true_grad = rate_map + Adec_map * np.exp(-g / tau)
        if g == 0:
            true_grad = true_grad - delta_map
        KQ = fftconvolve(Q, K, mode='same')
        grads_s[g] = true_grad * (1.0 - A_bfe * KQ)
        Q += true_grad
    return grads_s

def objective(params):
    log_A, alpha = params
    A_bfe = 10**log_A
    grads_s = simulate(A_bfe, alpha)
    sim_early = cutout_psf(grads_s, EARLY)
    sim_late  = cutout_psf(grads_s, LATE)
    sim_diff  = sim_late - sim_early
    return np.sum(((sim_diff - obs_diff)[fit_mask])**2)

# ---------------------------------------------------------------------------
# Optimise
# ---------------------------------------------------------------------------
print('Running differential evolution...')
bounds = [(-9, -4), (0.5, 4.0)]
result = differential_evolution(objective, bounds, seed=42, maxiter=300,
                                tol=1e-10, workers=1, polish=False,
                                disp=True)
print(f'DE result: log_A={result.x[0]:.4f}, alpha={result.x[1]:.4f}, res={result.fun:.4e}')

print('Polishing with Nelder-Mead...')
result2 = minimize(objective, result.x, method='Nelder-Mead',
                   options={'xatol':1e-10, 'fatol':1e-14, 'maxiter':50000})
log_A_fit, alpha_fit = result2.x
A_bfe_fit = 10**log_A_fit
print(f'Final: A={A_bfe_fit:.4e}, alpha={alpha_fit:.4f}, res={result2.fun:.4e}')

# ---------------------------------------------------------------------------
# Compare old BFE-only fit vs combined fit
# ---------------------------------------------------------------------------
A_OLD, ALPHA_OLD = 2.148e-7, 2.319

def simulate_bfe_only(A_bfe, alpha, n_grads_out=None):
    if n_grads_out is None:
        n_grads_out = N_FIT
    kh = 20
    ii, jj = np.mgrid[-kh:kh+1, -kh:kh+1].astype(float)
    r = np.sqrt(ii**2 + jj**2)
    K = np.where(r > 0, -1.0 / r**alpha, 0.0)
    K[kh, kh] = -K.sum()
    Q = np.zeros((ny, nx))
    grads_s = np.zeros((n_grads_out, ny, nx))
    for g in range(n_grads_out):
        KQ = fftconvolve(Q, K, mode='same')
        grads_s[g] = rate_map * (1.0 - A_bfe * KQ)
        Q += rate_map
    return grads_s

grads_old  = simulate_bfe_only(A_OLD, ALPHA_OLD)
grads_new  = simulate(A_bfe_fit, alpha_fit)

old_diff = cutout_psf(grads_old, LATE) - cutout_psf(grads_old, EARLY)
new_diff = cutout_psf(grads_new, LATE) - cutout_psf(grads_new, EARLY)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
vabs = np.nanpercentile(np.abs(obs_diff), 99)
ext  = [-CUT-0.5, CUT+0.5, -CUT-0.5, CUT+0.5]

fig, axes = plt.subplots(3, 3, figsize=(13, 11))
rows = [
    (obs_diff,  obs_diff - obs_diff,  'Observed diff'),
    (old_diff,  old_diff - obs_diff,  f'BFE-only fit  (A={A_OLD:.2e}, α={ALPHA_OLD:.3f})'),
    (new_diff,  new_diff - obs_diff,  f'Combined fit  (A={A_bfe_fit:.2e}, α={alpha_fit:.3f})'),
]

for row_idx, (diff, resid, label) in enumerate(rows):
    axes[row_idx, 0].set_ylabel(label, fontsize=8)
    for col_idx, (img, title) in enumerate([
            (diff,  'Late − Early'),
            (resid, 'Residual (model − obs)'),
    ]):
        ax = axes[row_idx, col_idx]
        im = ax.imshow(img, origin='lower', cmap='RdBu_r',
                       vmin=-vabs, vmax=vabs, extent=ext)
        fig.colorbar(im, ax=ax, label='Norm. flux')
        if row_idx == 0:
            ax.set_title(title, fontsize=9)
        ax.set_xlabel('Δx (px)')
    # Radial of diff
    ax = axes[row_idx, 2]
    yy_c, xx_c = np.mgrid[:diff.shape[0], :diff.shape[1]]
    r_map = np.sqrt((yy_c-CUT)**2 + (xx_c-CUT)**2)
    r_int = np.arange(0, CUT)
    rp = np.array([np.mean(diff[np.round(r_map).astype(int)==r]) for r in r_int])
    rp_obs = np.array([np.mean(obs_diff[np.round(r_map).astype(int)==r]) for r in r_int])
    ax.plot(r_int, rp_obs, 'k-o', ms=3, lw=1.5, label='Observed')
    ax.plot(r_int, rp,     'C3--s', ms=3, lw=1.5, label='Model')
    ax.axhline(0, color='k', lw=0.6, ls=':')
    ax.set_xlabel('Radius (px)'); ax.set_ylabel('Mean diff')
    if row_idx == 0:
        ax.set_title('Radial profile', fontsize=9)
    ax.legend(fontsize=7)

fig.suptitle('Wolf-359: combined reset-decay + BFE fit vs BFE-only fit',
             fontsize=11, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'wolf359_combined_fit.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved wolf359_combined_fit.png')

np.savez(OUT / 'combined_fit_params.npz',
         A_bfe=np.array([A_bfe_fit]),
         alpha=np.array([alpha_fit]),
         tau=np.array([tau]))
print(f'Saved combined_fit_params.npz')
print(f'\nSummary:')
print(f'  BFE-only fit:   A={A_OLD:.3e}, alpha={ALPHA_OLD:.3f}')
print(f'  Combined fit:   A={A_bfe_fit:.3e}, alpha={alpha_fit:.3f}')
