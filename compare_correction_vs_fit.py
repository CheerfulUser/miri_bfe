"""
compare_correction_vs_fit.py

Compare the joint correction residual to the forward model fit residual.

Three quantities on the same colorscale:
  1. Observed late-early PSF diff        (the signal being corrected)
  2. Joint-corrected late-early diff     (what remains after correction)
  3. Forward model fit residual          (model diff - observed diff, noise floor)

If the correction is working as well as the fit predicts, (2) and (3)
should have comparable amplitude and spatial structure.
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
OUT = Path(__file__).parent

STAR_Y, STAR_X = 89, 110
AP_RADIUS = 6
CUT = 20
EARLY = [1, 2, 3]
LATE = [6, 7, 8]

# ---------------------------------------------------------------------------
# Load saved fit parameters
# ---------------------------------------------------------------------------
params = np.load(OUT / 'combined_fit_params.npz')
A_bfe = float(params['A_bfe'])
alpha = float(params['alpha'])
tau_fit = float(params['tau'])
print(f'Fit params: A={A_bfe:.4e}, alpha={alpha:.4f}, tau={tau_fit:.3f}')

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with fits.open(BASE / 'uncal-fits/jw06122002001_02101_00001_mirimage_uncal.fits') as h:
    cube = h['SCI'].data.astype(float)

n_int, n_groups, ny, nx = cube.shape
n_grads = n_groups - 1
n_grads_fit = n_grads - 1
grads_raw = np.diff(cube, axis=1)
g_arr = np.arange(n_grads_fit, dtype=float)
med_grad = np.median(grads_raw, axis=0)
print(f'Loaded: {cube.shape}')

# ---------------------------------------------------------------------------
# Build kernel
# ---------------------------------------------------------------------------
def make_kernel(alpha, kh=20):
    ii, jj = np.mgrid[-kh:kh+1, -kh:kh+1].astype(float)
    r = np.sqrt(ii**2 + jj**2)
    K = np.where(r > 0, -1.0 / r**alpha, 0.0)
    K[kh, kh] = -K.sum()
    return K

K = make_kernel(alpha)

# ---------------------------------------------------------------------------
# Fit reset decay (same as apply_joint_correction.py)
# ---------------------------------------------------------------------------
mean_profile = np.nanmean(med_grad[:n_grads_fit], axis=(1, 2))

def exp_model(g, C, A, t): return C + A * np.exp(-g / t)
popt, _ = curve_fit(exp_model, g_arr[1:], mean_profile[1:],
                    p0=[mean_profile[-1], mean_profile[0] - mean_profile[-1], 1.5])
tau = float(popt[2])
print(f'Reset decay tau = {tau:.3f} groups')

exp_g = np.exp(-g_arr / tau)
ff_col = np.zeros(n_grads_fit); ff_col[0] = -1.0
X = np.column_stack([np.ones(n_grads_fit), exp_g, ff_col])
params_rd, _, _, _ = np.linalg.lstsq(
    X, med_grad[:n_grads_fit].reshape(n_grads_fit, -1), rcond=None)
rate_map  = params_rd[0].reshape(ny, nx)
Adec_map  = params_rd[1].reshape(ny, nx)
delta_map = params_rd[2].reshape(ny, nx)

# ---------------------------------------------------------------------------
# PSF cutout helper
# ---------------------------------------------------------------------------
sy, sx = STAR_Y, STAR_X

def cutout_psf(grads_3d, group_list):
    stack = np.median(grads_3d[np.array(group_list)], axis=0)
    cut = stack[sy-CUT:sy+CUT+1, sx-CUT:sx+CUT+1]
    yy, xx = np.mgrid[:cut.shape[0], :cut.shape[1]]
    ap = np.sqrt((yy-CUT)**2 + (xx-CUT)**2) <= AP_RADIUS
    return cut / cut[ap].sum()

def group_psf_4d(grads_4d, group_list):
    return cutout_psf(np.median(grads_4d[:, group_list, :, :], axis=0), [0])

# Observed PSF diff
obs_early = cutout_psf(med_grad, EARLY)
obs_late  = cutout_psf(med_grad, LATE)
obs_diff  = obs_late - obs_early

# ---------------------------------------------------------------------------
# Forward model simulation (same as fit_combined_model.py)
# ---------------------------------------------------------------------------
def simulate(A_bfe, alpha, n_grads_out=None):
    if n_grads_out is None:
        n_grads_out = n_grads_fit
    K_s = make_kernel(alpha)
    Q = np.zeros((ny, nx))
    grads_s = np.zeros((n_grads_out, ny, nx))
    for g in range(n_grads_out):
        true_grad = rate_map + Adec_map * np.exp(-g / tau)
        if g == 0:
            true_grad = true_grad - delta_map
        KQ = fftconvolve(Q, K_s, mode='same')
        grads_s[g] = true_grad * (1.0 - A_bfe * KQ)
        Q += true_grad
    return grads_s

grads_sim = simulate(A_bfe, alpha)
sim_early = cutout_psf(grads_sim, EARLY)
sim_late  = cutout_psf(grads_sim, LATE)
sim_diff  = sim_late - sim_early

# Fit residual: what the model couldn't account for
fit_resid = sim_diff - obs_diff

# ---------------------------------------------------------------------------
# Joint correction (causal BFE inversion + locked reset decay subtraction)
# ---------------------------------------------------------------------------
grads_bfe = grads_raw.copy()
Q_med = np.zeros((ny, nx))

for g in range(n_grads):
    if g > 0:
        Q_med = Q_med + np.median(grads_bfe[:, g-1, :, :], axis=0)
    KQ = fftconvolve(Q_med, K, mode='same')
    factor = 1.0 - A_bfe * KQ
    factor = np.where(factor > 0.05, factor, 1.0)
    grads_bfe[:, g, :, :] = grads_raw[:, g, :, :] / factor[None, :, :]

grads_joint = grads_bfe.copy()
for g in range(n_grads_fit):
    decay_g = Adec_map * np.exp(-g / tau)
    if g == 0:
        grads_joint[:, 0, :, :] = grads_bfe[:, 0, :, :] - decay_g[None] + delta_map[None]
    else:
        grads_joint[:, g, :, :] = grads_bfe[:, g, :, :] - decay_g[None]

med_joint = np.median(grads_joint, axis=0)
cor_early = cutout_psf(med_joint, EARLY)
cor_late  = cutout_psf(med_joint, LATE)
cor_diff  = cor_late - cor_early

# Correction residual: what the correction left behind
cor_resid = cor_diff   # relative to zero (perfect correction → zero diff)

print('Corrections done.')

# ---------------------------------------------------------------------------
# Figure: four panels on the same colorscale
# ---------------------------------------------------------------------------
vabs = np.nanpercentile(np.abs(obs_diff), 99)
ext = [-CUT-0.5, CUT+0.5, -CUT-0.5, CUT+0.5]

# Radial profile
yy_c, xx_c = np.mgrid[:obs_diff.shape[0], :obs_diff.shape[1]]
r_map = np.sqrt((yy_c-CUT)**2 + (xx_c-CUT)**2)
r_int = np.arange(0, CUT)

def radial(img):
    return np.array([np.nanmean(img[np.round(r_map).astype(int) == r]) for r in r_int])

r_obs      = radial(obs_diff)
r_fit_res  = radial(fit_resid)
r_cor_diff = radial(cor_diff)

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.35)

panels = [
    (obs_diff,  'Observed late−early diff',              gs[0, 0]),
    (sim_diff,  'Model late−early diff (best fit)',      gs[0, 1]),
    (fit_resid, 'Fit residual (model − observed)',       gs[0, 2]),
    (cor_diff,  'Correction residual (corrected diff)',  gs[0, 3]),
]

for img, title, pos in panels:
    ax = fig.add_subplot(pos)
    im = ax.imshow(img, origin='lower', cmap='RdBu_r',
                   vmin=-vabs, vmax=vabs, extent=ext)
    fig.colorbar(im, ax=ax, label='Norm. flux')
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('Δx (px)')
    ax.set_ylabel('Δy (px)')

ax_r = fig.add_subplot(gs[1, :])
ax_r.plot(r_int, r_obs,      'k-o',   ms=4, lw=1.5, label='Observed diff')
ax_r.plot(r_int, r_cor_diff, 'C0-^',  ms=4, lw=1.5, label='Correction residual (corrected diff)')
ax_r.plot(r_int, r_fit_res,  'C3--s', ms=4, lw=1.5, label='Fit residual (model − observed)')
ax_r.axhline(0, color='k', lw=0.7, ls=':', alpha=0.5)
ax_r.set_xlabel('Radius (px)')
ax_r.set_ylabel('Mean late−early (norm. flux)')
ax_r.set_title('Radial profiles: observed vs correction residual vs fit residual')
ax_r.legend(fontsize=9)

rms_fit_res  = np.sqrt(np.nanmean(fit_resid[r_map <= 12]**2))
rms_cor_diff = np.sqrt(np.nanmean(cor_diff[ r_map <= 12]**2))
fig.suptitle(
    f'Joint correction vs fit residual  |  '
    f'RMS (r≤12px): fit residual={rms_fit_res:.2e}, correction residual={rms_cor_diff:.2e}',
    fontsize=10, fontweight='bold')

out = OUT / 'wolf359_correction_vs_fit_residual.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
print(f'\nRMS within r<=12px:')
print(f'  Fit residual        : {rms_fit_res:.4e}')
print(f'  Correction residual : {rms_cor_diff:.4e}')
print(f'  Ratio (cor/fit)     : {rms_cor_diff/rms_fit_res:.2f}x')
