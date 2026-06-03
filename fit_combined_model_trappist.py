"""
fit_combined_model_trappist.py

Fit the combined reset-decay + BFE forward model to TRAPPIST-1 MIRI data
and compare fitted parameters to Wolf-359.

Forward model:
    grad_obs(g) = [rate + A_decay*exp(-g/tau)] * (1 - A_bfe * K ⊛ Q(g))
    Q(g)        = sum_{k<g} [rate + A_decay*exp(-k/tau)]

Free parameters: log10(A_bfe), alpha

Runs on a cropped region around the star to keep the full-detector
fftconvolve tractable.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from scipy.signal import fftconvolve
from scipy.optimize import minimize_scalar, curve_fit

TRAP_DIR = Path('/Users/rri38/Documents/work/code/jwst/ramps/trappist')
MASK_PATH = Path('/Users/rri38/Documents/work/code/jwst/jurassic/full_MIRI_mask.npy')
SEG = 'jw01177007001_03101_00001-seg001_mirimage_ramp.fits'
OUT = Path(__file__).parent

STAR_Y, STAR_X = 515, 697
CROP = 70      # half-side of region fed to forward model
CUT = 20       # half-side of PSF cutout
AP_RADIUS = 5
EARLY = [2, 3, 4]
LATE = [9, 10, 11]
FIT_R = 10
N_GROUPS = 14

# Wolf-359 fitted parameters for comparison
W359_A_BFE = 1.035e-6
W359_ALPHA = 2.783
W359_TAU = 1.498

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
sci_mask_full = np.load(MASK_PATH)

def load_seg(path, n_groups=N_GROUPS, ny=1024, nx=1032):
    with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
        offset = hdul['SCI']._data_offset
        n_int_hdr = hdul['SCI'].header['NAXIS4']
    bytes_per_int = n_groups * ny * nx * 4
    available = path.stat().st_size - offset
    n_int = min(n_int_hdr, int(available // bytes_per_int))
    with open(path, 'rb') as fh:
        fh.seek(offset)
        raw = np.frombuffer(fh.read(n_int * bytes_per_int), dtype='>f4')
    if n_int < n_int_hdr:
        print(f'  truncated: {n_int}/{n_int_hdr} integrations kept')
    return raw.reshape(n_int, n_groups, ny, nx).astype(float)

cube_full = load_seg(TRAP_DIR / SEG)
n_int, n_groups, ny_full, nx_full = cube_full.shape

# Crop to region around star
y0 = STAR_Y - CROP
y1 = STAR_Y + CROP + 1
x0 = STAR_X - CROP
x1 = STAR_X + CROP + 1
cube = cube_full[:, :, y0:y1, x0:x1]
sci_mask = sci_mask_full[y0:y1, x0:x1]

# Star position in cropped frame
sy = CROP
sx = CROP

n_int, n_groups, ny, nx = cube.shape
n_grads = n_groups - 1
N_FIT = n_grads - 1   # exclude last-frame anomaly
g_arr = np.arange(N_FIT, dtype=float)
grads = np.diff(cube, axis=1)
med_grad = np.median(grads, axis=0)
print(f'Loaded: full {cube_full.shape} -> crop {cube.shape}  N_FIT={N_FIT}')

# ---------------------------------------------------------------------------
# Fit reset decay parameters (background annulus, excluding star)
# ---------------------------------------------------------------------------
yy_full, xx_full = np.mgrid[:ny, :nx]
r_star = np.sqrt((yy_full - sy)**2 + (xx_full - sx)**2)
sci_mask_bool = sci_mask.astype(bool)
bg_mask = (r_star > 15) & (r_star < (CROP - 5)) & sci_mask_bool
mean_profile = np.nanmean(med_grad[:N_FIT][:, bg_mask], axis=1)

def exp_model(g, C, A, t): return C + A * np.exp(-g / t)
popt, _ = curve_fit(exp_model, g_arr[1:], mean_profile[1:],
                    p0=[mean_profile[-1], mean_profile[0]-mean_profile[-1], 1.5])
tau = float(popt[2])
print(f'tau = {tau:.4f} groups  (Wolf-359: {W359_TAU:.4f})')

exp_g = np.exp(-g_arr / tau)
ff_col = np.zeros(N_FIT); ff_col[0] = -1.0
X = np.column_stack([np.ones(N_FIT), exp_g, ff_col])
params, _, _, _ = np.linalg.lstsq(X, med_grad[:N_FIT].reshape(N_FIT, -1), rcond=None)
rate_map = params[0].reshape(ny, nx)
Adec_map = params[1].reshape(ny, nx)
delta_map = params[2].reshape(ny, nx)

# ---------------------------------------------------------------------------
# Observed PSF difference + noise from scatter across integrations
# ---------------------------------------------------------------------------
_ap_yy, _ap_xx = np.mgrid[:2*CUT+1, :2*CUT+1]
_ap_mask = np.sqrt((_ap_yy-CUT)**2 + (_ap_xx-CUT)**2) <= AP_RADIUS

def cutout_psf(arr_3d, group_list):
    stack = np.median(arr_3d[np.array(group_list)], axis=0)
    cut = stack[sy-CUT:sy+CUT+1, sx-CUT:sx+CUT+1]
    return cut / cut[_ap_mask].sum()

def cutout_psf_perint(grads_4d, group_list):
    gl = np.array(group_list)
    cuts = []
    for i in range(grads_4d.shape[0]):
        stack = np.median(grads_4d[i, gl], axis=0)
        cut = stack[sy-CUT:sy+CUT+1, sx-CUT:sx+CUT+1]
        cuts.append(cut / cut[_ap_mask].sum())
    return np.array(cuts)

diff_perint = cutout_psf_perint(grads, LATE) - cutout_psf_perint(grads, EARLY)
obs_diff = np.median(diff_perint, axis=0)
noise_diff = np.std(diff_perint, axis=0) / np.sqrt(n_int)
noise_diff = np.clip(noise_diff, noise_diff[noise_diff > 0].min() * 0.1, None)

yy_c, xx_c = np.mgrid[:obs_diff.shape[0], :obs_diff.shape[1]]
fit_mask = np.sqrt((yy_c-CUT)**2 + (xx_c-CUT)**2) <= FIT_R

# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------
def simulate(A_bfe, alpha):
    kh = 20
    ii, jj = np.mgrid[-kh:kh+1, -kh:kh+1].astype(float)
    r = np.sqrt(ii**2 + jj**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        K = np.where(r > 0, -1.0 / r**alpha, 0.0)
    K[kh, kh] = -K.sum()
    Q = np.zeros((ny, nx))
    grads_s = np.zeros((N_FIT, ny, nx))
    for g in range(N_FIT):
        true_grad = rate_map + Adec_map * np.exp(-g / tau)
        if g == 0:
            true_grad = true_grad - delta_map
        KQ = fftconvolve(Q, K, mode='same')
        grads_s[g] = true_grad * (1.0 - A_bfe * KQ)
        Q += true_grad
    return grads_s

# Alpha is unconstrained for this faint source — fix to Wolf-359/EV Lac consensus
# (kernel shape is a detector property) and fit only A_bfe.
ALPHA_FIXED = 2.783

def objective(log_A_arr):
    grads_s = simulate(10**log_A_arr[0], ALPHA_FIXED)
    sim_diff = cutout_psf(grads_s, LATE) - cutout_psf(grads_s, EARLY)
    return np.sum((((sim_diff - obs_diff) / noise_diff)[fit_mask])**2)

# ---------------------------------------------------------------------------
# Optimise
# ---------------------------------------------------------------------------
print('Running bounded scalar minimisation (alpha fixed)...')
result = minimize_scalar(lambda la: objective([la]), bounds=(-9, -4), method='bounded')
log_A_fit = result.x
alpha_fit = ALPHA_FIXED
A_bfe_fit = 10**log_A_fit
print(f'Final: A={A_bfe_fit:.4e}, alpha={alpha_fit:.4f}')

# ---------------------------------------------------------------------------
# Comparison with Wolf-359 parameters applied to TRAPPIST-1 data
# ---------------------------------------------------------------------------
grads_w359 = simulate(W359_A_BFE, W359_ALPHA)
grads_fit = simulate(A_bfe_fit, alpha_fit)

diff_w359 = cutout_psf(grads_w359, LATE) - cutout_psf(grads_w359, EARLY)
diff_fit = cutout_psf(grads_fit, LATE) - cutout_psf(grads_fit, EARLY)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
vabs = np.nanpercentile(np.abs(obs_diff), 99)
ext = [-CUT-0.5, CUT+0.5, -CUT-0.5, CUT+0.5]

fig, axes = plt.subplots(3, 3, figsize=(13, 11))
rows = [
    (obs_diff, obs_diff - obs_diff, 'Observed diff'),
    (diff_w359, diff_w359 - obs_diff, f'Wolf-359 params (A={W359_A_BFE:.2e}, α={W359_ALPHA:.3f})'),
    (diff_fit, diff_fit - obs_diff, f'TRAPPIST-1 fit (A={A_bfe_fit:.2e}, α={alpha_fit:.3f})'),
]

for row_idx, (diff, resid, label) in enumerate(rows):
    axes[row_idx, 0].set_ylabel(label, fontsize=8)
    for col_idx, (img, title) in enumerate([
            (diff, 'Late − Early'),
            (resid, 'Residual (model − obs)'),
    ]):
        ax = axes[row_idx, col_idx]
        im = ax.imshow(img, origin='lower', cmap='RdBu_r',
                       vmin=-vabs, vmax=vabs, extent=ext)
        fig.colorbar(im, ax=ax, label='Norm. flux')
        if row_idx == 0:
            ax.set_title(title, fontsize=9)
        ax.set_xlabel('Δx (px)')
    ax = axes[row_idx, 2]
    r_map = np.sqrt((yy_c-CUT)**2 + (xx_c-CUT)**2)
    r_int = np.arange(0, CUT)
    rp = np.array([np.mean(diff[np.round(r_map).astype(int)==r]) for r in r_int])
    rp_obs = np.array([np.mean(obs_diff[np.round(r_map).astype(int)==r]) for r in r_int])
    ax.plot(r_int, rp_obs, 'k-o', ms=3, lw=1.5, label='Observed')
    ax.plot(r_int, rp, 'C3--s', ms=3, lw=1.5, label='Model')
    ax.axhline(0, color='k', lw=0.6, ls=':')
    ax.set_xlabel('Radius (px)')
    ax.set_ylabel('Mean diff')
    if row_idx == 0:
        ax.set_title('Radial profile', fontsize=9)
    ax.legend(fontsize=7)

fig.suptitle('TRAPPIST-1: combined reset-decay + BFE fit vs Wolf-359 parameters',
             fontsize=11, fontweight='bold')
fig.tight_layout()
out = OUT / 'trappist_combined_fit.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')

np.savez(OUT / 'trappist_combined_fit_params.npz',
         A_bfe=np.array([A_bfe_fit]), alpha=np.array([alpha_fit]), tau=np.array([tau]))

print(f'\nParameter comparison:')
print(f'  {"":12s}  {"A_bfe":>12s}  {"alpha":>8s}  {"tau":>8s}')
print(f'  {"Wolf-359":12s}  {W359_A_BFE:>12.3e}  {W359_ALPHA:>8.4f}  {W359_TAU:>8.4f}')
print(f'  {"TRAPPIST-1":12s}  {A_bfe_fit:>12.3e}  {alpha_fit:>8.4f}  {tau:>8.4f}')
