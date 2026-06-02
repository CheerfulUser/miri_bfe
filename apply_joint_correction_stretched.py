"""
apply_joint_correction_stretched.py

Joint BFE + reset-decay correction using a stretched exponential decay model.

The single exponential fitted globally (tau~1.5) is a poor description at
the star core (and for the detector as a whole, median beta~0.62). This
script uses a stretched exponential:

    true_grad(g) = C + A * exp(-(g/tau)^beta)

Inversion strategy:
  1. Causal BFE inversion (single pass, exact)
  2. Fit global tau and beta from spatial mean of BFE-corrected gradients
  3. Per-pixel [C, A, delta] via lstsq with fixed tau, beta
  4. Subtract the stretched decay from BFE-corrected gradients

Outputs
-------
bfe_correction/wolf359_joint_stretched_psf.png
bfe_correction/wolf359_joint_stretched_profiles.png
bfe_correction/wolf359_joint_stretched_group_check.png
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
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))
from ramp_correction import correct_reset_decay

A_BFE = 1.035e-6
ALPHA_BFE = 2.783
STAR_Y, STAR_X = 89, 110
AP_RADIUS = 6
BG_RMIN, BG_RMAX = 20, 60
CUT = 20
EARLY_GROUPS = [1, 2, 3]
LATE_GROUPS = [6, 7, 8]

BASE = Path('/Users/rri38/Documents/work/code/jwst/ramps/wolf-359')
OUT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
with fits.open(BASE / 'uncal-fits/jw06122002001_02101_00001_mirimage_uncal.fits') as h:
    cube = h['SCI'].data.astype(float)

n_int, n_groups, ny, nx = cube.shape
n_grads = n_groups - 1
n_grads_fit = n_grads - 1
grads_raw = np.diff(cube, axis=1)
g_arr = np.arange(n_grads_fit, dtype=float)
print(f'Loaded: {cube.shape}')

yy, xx = np.mgrid[:ny, :nx]
r_star = np.sqrt((yy - STAR_Y)**2 + (xx - STAR_X)**2)
ap_mask = r_star <= AP_RADIUS
bg_mask = (r_star >= BG_RMIN) & (r_star <= BG_RMAX)

# ---------------------------------------------------------------------------
# BFE kernel
# ---------------------------------------------------------------------------
def make_kernel(alpha, kh=20):
    ii, jj = np.mgrid[-kh:kh+1, -kh:kh+1].astype(float)
    r = np.sqrt(ii**2 + jj**2)
    K = np.where(r > 0, -1.0 / r**alpha, 0.0)
    K[kh, kh] = -K.sum()
    return K

K = make_kernel(ALPHA_BFE)

# ---------------------------------------------------------------------------
# Causal BFE inversion
# ---------------------------------------------------------------------------
grads_bfe = grads_raw.copy()
Q_med = np.zeros((ny, nx))
for g in range(n_grads):
    if g > 0:
        Q_med = Q_med + np.median(grads_bfe[:, g-1], axis=0)
    KQ = fftconvolve(Q_med, K, mode='same')
    factor = np.where(1.0 - A_BFE * KQ > 0.05, 1.0 - A_BFE * KQ, 1.0)
    grads_bfe[:, g] = grads_raw[:, g] / factor[None]
    print(f'  BFE g={g}', end='\r')
print()
print('BFE inversion done.')

med_bfe = np.median(grads_bfe, axis=0)   # (n_grads, ny, nx)

# ---------------------------------------------------------------------------
# Fit global tau and beta from spatial mean of BFE-corrected background pixels
# Exclude g=0 (first-frame anomaly) from the global fit
# ---------------------------------------------------------------------------
g_fit_bg = g_arr[1:]   # g=1..8
mean_bg = np.nanmean(med_bfe[1:n_grads_fit, bg_mask], axis=1)

def stretched_exp(g, C, A, tau, beta):
    return C + A * np.exp(-(g / tau)**beta)

p0 = [mean_bg[-1], mean_bg[0] - mean_bg[-1], 1.5, 0.8]
popt_global, _ = curve_fit(stretched_exp, g_fit_bg, mean_bg, p0=p0,
                            bounds=([0, 0, 0.1, 0.1], [np.inf, np.inf, 20, 3]),
                            maxfev=20000)
tau_g, beta_g = float(popt_global[2]), float(popt_global[3])
print(f'Global stretched exp: tau={tau_g:.4f}, beta={beta_g:.4f}')

# Also report single-exp tau for comparison
def exp1(g, C, A, tau): return C + A * np.exp(-g / tau)
popt_e, _ = curve_fit(exp1, g_fit_bg, mean_bg,
                       p0=[mean_bg[-1], mean_bg[0]-mean_bg[-1], 1.5])
print(f'Global single exp:    tau={popt_e[2]:.4f}')

# ---------------------------------------------------------------------------
# Per-pixel [C, A, delta] via lstsq with fixed tau, beta
# Design matrix: [1, exp(-(g/tau)^beta), -1_{g=0}]
# ---------------------------------------------------------------------------
str_g  = np.exp(-(g_arr / tau_g)**beta_g)   # stretched exp at each group
ff_col = np.zeros(n_grads_fit); ff_col[0] = -1.0
X = np.column_stack([np.ones(n_grads_fit), str_g, ff_col])   # (n_grads_fit, 3)

params_str, _, _, _ = np.linalg.lstsq(
    X, med_bfe[:n_grads_fit].reshape(n_grads_fit, -1), rcond=None)
C_map_str     = params_str[0].reshape(ny, nx)
Adec_map_str  = params_str[1].reshape(ny, nx)
delta_map_str = params_str[2].reshape(ny, nx)

print(f'Adec_map mean (bg):   {Adec_map_str[bg_mask].mean():.2f} DN/group')
print(f'Adec_map mean (star): {Adec_map_str[ap_mask].mean():.2f} DN/group')

# ---------------------------------------------------------------------------
# Subtract stretched decay from BFE-corrected gradients
# ---------------------------------------------------------------------------
grads_str = grads_bfe.copy()
for g in range(n_grads_fit):
    decay_g = Adec_map_str * np.exp(-(g / tau_g)**beta_g)
    if g == 0:
        grads_str[:, 0] = grads_bfe[:, 0] - decay_g[None] + delta_map_str[None]
    else:
        grads_str[:, g] = grads_bfe[:, g] - decay_g[None]

# ---------------------------------------------------------------------------
# Baseline: single-exp joint correction (from apply_joint_correction.py)
# ---------------------------------------------------------------------------
exp_g_s = np.exp(-g_arr / popt_e[2])
ff_col2 = np.zeros(n_grads_fit); ff_col2[0] = -1.0
X2 = np.column_stack([np.ones(n_grads_fit), exp_g_s, ff_col2])
params_e, _, _, _ = np.linalg.lstsq(
    X2, med_bfe[:n_grads_fit].reshape(n_grads_fit, -1), rcond=None)
Adec_map_e  = params_e[1].reshape(ny, nx)
delta_map_e = params_e[2].reshape(ny, nx)

grads_exp = grads_bfe.copy()
for g in range(n_grads_fit):
    decay_g = Adec_map_e * np.exp(-g / popt_e[2])
    if g == 0:
        grads_exp[:, 0] = grads_bfe[:, 0] - decay_g[None] + delta_map_e[None]
    else:
        grads_exp[:, g] = grads_bfe[:, g] - decay_g[None]

# Reset-decay-only baseline
cube_rd = correct_reset_decay(cube, method='median')
grads_rd = np.diff(cube_rd, axis=1)

print('All corrections done.')

# ---------------------------------------------------------------------------
# PSF comparison
# ---------------------------------------------------------------------------
sy, sx = STAR_Y, STAR_X

def group_psf(grads, group_list):
    stack = np.median(grads[:, group_list, :, :], axis=(0, 1))
    cut = stack[sy-CUT:sy+CUT+1, sx-CUT:sx+CUT+1]
    yy2, xx2 = np.mgrid[:cut.shape[0], :cut.shape[1]]
    ap = np.sqrt((yy2-CUT)**2 + (xx2-CUT)**2) <= AP_RADIUS
    return cut / cut[ap].sum()

early_rd  = group_psf(grads_rd,  EARLY_GROUPS)
late_rd   = group_psf(grads_rd,  LATE_GROUPS)
early_exp = group_psf(grads_exp, EARLY_GROUPS)
late_exp  = group_psf(grads_exp, LATE_GROUPS)
early_str = group_psf(grads_str, EARLY_GROUPS)
late_str  = group_psf(grads_str, LATE_GROUPS)

diff_rd  = late_rd  - early_rd
diff_exp = late_exp - early_exp
diff_str = late_str - early_str
vabs = np.nanpercentile(np.abs(diff_rd), 99)
ext = [-CUT-0.5, CUT+0.5, -CUT-0.5, CUT+0.5]

fig, axes = plt.subplots(3, 3, figsize=(13, 11))
for row, (early, late, diff, label) in enumerate([
        (early_rd,  late_rd,  diff_rd,  'Reset-decay only'),
        (early_exp, late_exp, diff_exp, f'Joint: single exp (τ={popt_e[2]:.3f})'),
        (early_str, late_str, diff_str, f'Joint: stretched exp (τ={tau_g:.3f}, β={beta_g:.3f})'),
]):
    axes[row, 0].set_ylabel(label, fontsize=8)
    for col, (img, title) in enumerate([
            (early, f'Early groups {EARLY_GROUPS}'),
            (late,  f'Late groups {LATE_GROUPS}'),
            (diff,  'Late − Early'),
    ]):
        ax = axes[row, col]
        kw = dict(vmin=-vabs, vmax=vabs, cmap='RdBu_r') if col == 2 else dict(cmap='viridis')
        im = ax.imshow(img, origin='lower', extent=ext, **kw)
        fig.colorbar(im, ax=ax, label='Norm. flux')
        if row == 0:
            ax.set_title(title, fontsize=9)
        ax.set_xlabel('Δx (px)')
        ax.set_ylabel('Δy (px)')

fig.suptitle('Wolf-359 PSF: single exp vs stretched exp joint correction',
             fontsize=10, fontweight='bold')
fig.tight_layout()
out1 = OUT / 'wolf359_joint_stretched_psf.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out1}')

# ---------------------------------------------------------------------------
# Per-group median check: star aperture and background
# ---------------------------------------------------------------------------
g_show = np.arange(1, n_grads_fit)

def group_medians(grads, mask):
    med = np.median(grads, axis=0)
    return np.array([med[g, mask].mean() for g in range(n_grads)])

ap_rd  = group_medians(grads_rd,  ap_mask)
ap_exp = group_medians(grads_exp, ap_mask)
ap_str = group_medians(grads_str, ap_mask)
bg_rd  = group_medians(grads_rd,  bg_mask)
bg_exp = group_medians(grads_exp, bg_mask)
bg_str = group_medians(grads_str, bg_mask)

def norm(arr): return arr / np.median(arr[g_show])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, ap, ex, st, title in [
        (axes[0], ap_rd, ap_exp, ap_str, f'Star aperture (r≤{AP_RADIUS}px)'),
        (axes[1], bg_rd, bg_exp, bg_str, f'Background (r={BG_RMIN}–{BG_RMAX}px)'),
]:
    ax.plot(g_show, norm(ap)[g_show], 'o-', color='C3', lw=1.5, label='Reset-decay only')
    ax.plot(g_show, norm(ex)[g_show], 's-', color='C1', lw=1.5,
            label=f'Joint single exp (τ={popt_e[2]:.2f})')
    ax.plot(g_show, norm(st)[g_show], '^-', color='C0', lw=1.5,
            label=f'Joint stretched exp (τ={tau_g:.2f}, β={beta_g:.2f})')
    ax.axhline(1.0, color='k', lw=0.7, ls='--', alpha=0.4)
    ax.set_xlabel('Gradient index')
    ax.set_ylabel('Normalised mean gradient')
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xticks(g_show)

rms_rd  = np.std(norm(ap_rd)[g_show])
rms_exp = np.std(norm(ap_exp)[g_show])
rms_str = np.std(norm(ap_str)[g_show])
print(f'\nStar aperture profile flatness RMS (groups 1-8):')
print(f'  Reset-decay only   : {rms_rd*100:.3f}%')
print(f'  Joint single exp   : {rms_exp*100:.3f}%')
print(f'  Joint stretched exp: {rms_str*100:.3f}%')

core_amp_rd  = np.abs(diff_rd [CUT, CUT])
core_amp_exp = np.abs(diff_exp[CUT, CUT])
core_amp_str = np.abs(diff_str[CUT, CUT])
print(f'\nPSF core late-early diff:')
print(f'  Reset-decay only   : {core_amp_rd:.5f}')
print(f'  Joint single exp   : {core_amp_exp:.5f}  ({(1-core_amp_exp/core_amp_rd)*100:.1f}% suppression)')
print(f'  Joint stretched exp: {core_amp_str:.5f}  ({(1-core_amp_str/core_amp_rd)*100:.1f}% suppression)')

fig.suptitle('Per-group profile: single exp vs stretched exp joint correction',
             fontsize=10, fontweight='bold')
fig.tight_layout()
out2 = OUT / 'wolf359_joint_stretched_profiles.png'
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out2}')
