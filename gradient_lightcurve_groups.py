"""
gradient_lightcurve_groups.py

Joint-corrected gradient lightcurve of Wolf-359, colored by group index.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from astropy.io import fits
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

A = 1.035e-6
ALPHA = 2.783
STAR_Y, STAR_X = 89, 110
AP_RADIUS = 6

BASE = Path('/Users/rri38/Documents/work/code/jwst/ramps/wolf-359')
OUT = Path(__file__).parent

with fits.open(BASE / 'uncal-fits/jw06122002001_02101_00001_mirimage_uncal.fits') as h:
    cube = h['SCI'].data.astype(float)

n_int, n_groups, ny, nx = cube.shape
n_grads = n_groups - 1
n_grads_fit = n_grads - 1
grads_raw = np.diff(cube, axis=1)
g_arr = np.arange(n_grads_fit, dtype=float)
print(f'Loaded: {cube.shape}')

yy, xx = np.mgrid[:ny, :nx]
ap_mask = (yy - STAR_Y)**2 + (xx - STAR_X)**2 <= AP_RADIUS**2

def make_kernel(alpha, kh=20):
    ii, jj = np.mgrid[-kh:kh+1, -kh:kh+1].astype(float)
    r = np.sqrt(ii**2 + jj**2)
    K = np.where(r > 0, -1.0 / r**alpha, 0.0)
    K[kh, kh] = -K.sum()
    return K

K = make_kernel(ALPHA)

med_grad = np.median(grads_raw, axis=0)
mean_profile = np.nanmean(med_grad[:n_grads_fit], axis=(1, 2))

def exp_model(g, C, A, t): return C + A * np.exp(-g / t)
popt, _ = curve_fit(exp_model, g_arr[1:], mean_profile[1:],
                    p0=[mean_profile[-1], mean_profile[0] - mean_profile[-1], 1.5])
tau = float(popt[2])

exp_g = np.exp(-g_arr / tau)
ff_col = np.zeros(n_grads_fit); ff_col[0] = -1.0
X = np.column_stack([np.ones(n_grads_fit), exp_g, ff_col])
params_rd, _, _, _ = np.linalg.lstsq(
    X, med_grad[:n_grads_fit].reshape(n_grads_fit, -1), rcond=None)
Adec_map  = params_rd[1].reshape(ny, nx)
delta_map = params_rd[2].reshape(ny, nx)

grads_bfe = grads_raw.copy()
Q_med = np.zeros((ny, nx))
for g in range(n_grads):
    if g > 0:
        Q_med = Q_med + np.median(grads_bfe[:, g-1, :, :], axis=0)
    KQ = fftconvolve(Q_med, K, mode='same')
    factor = 1.0 - A * KQ
    factor = np.where(factor > 0.05, factor, 1.0)
    grads_bfe[:, g, :, :] = grads_raw[:, g, :, :] / factor[None, :, :]
    print(f'  BFE g={g}', end='\r')
print()

grads_joint = grads_bfe.copy()
for g in range(n_grads_fit):
    decay_g = Adec_map * np.exp(-g / tau)
    if g == 0:
        grads_joint[:, 0] = grads_bfe[:, 0] - decay_g[None] + delta_map[None]
    else:
        grads_joint[:, g] = grads_bfe[:, g] - decay_g[None]

print('Correction done.')

# (n_int, n_grads) aperture sums
lc_joint = grads_joint[:, :, ap_mask].sum(axis=2)
norm = np.median(lc_joint[:, 1:n_grads_fit])
lc_joint_n = lc_joint / norm

integ = np.arange(n_int)

# ---------------------------------------------------------------------------
# Figure: one scatter per group, colored by group index
# ---------------------------------------------------------------------------
cmap = cm.get_cmap('plasma', n_grads)
colors = [cmap(g) for g in range(n_grads)]

fig, ax = plt.subplots(figsize=(14, 5))

for g in range(n_grads):
    if g in (0, n_grads - 1):
        continue
    ax.scatter(integ, lc_joint_n[:, g], color=colors[g], s=4,
               alpha=0.7, label=f'g={g}', zorder=g+1)

ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.4)
ax.set_xlabel('Integration index')
ax.set_ylabel('Normalised aperture flux')
ax.set_title('Wolf-359 joint-corrected gradient lightcurve — colored by group index')

sm = plt.cm.ScalarMappable(cmap=cmap,
                            norm=plt.Normalize(vmin=0, vmax=n_grads-1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.01)
cbar.set_label('Group index')
cbar.set_ticks(np.arange(n_grads))

fig.tight_layout()
out = OUT / 'wolf359_joint_lc_by_group.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
