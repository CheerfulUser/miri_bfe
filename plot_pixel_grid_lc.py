"""
plot_pixel_grid_lc.py

2D pixel grid image of the Wolf-359 PSF with per-pixel gradient
lightcurves after joint BFE + reset-decay correction.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit

A = 1.035e-6
ALPHA = 2.783
STAR_Y, STAR_X = 89, 110
HALF = 4   # show (2*HALF+1) x (2*HALF+1) = 9x9 pixel grid

BASE = Path('/Users/rri38/Documents/work/code/jwst/ramps/wolf-359')
OUT = Path(__file__).parent

with fits.open(BASE / 'uncal-fits/jw06122002001_02101_00001_mirimage_uncal.fits') as h:
    cube = h['SCI'].data.astype(float)

n_int, n_groups, ny, nx = cube.shape
n_grads = n_groups - 1
n_grads_fit = n_grads - 1
grads_raw = np.diff(cube, axis=1)
g_arr = np.arange(n_grads_fit, dtype=float)

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
params, _, _, _ = np.linalg.lstsq(X, med_grad[:n_grads_fit].reshape(n_grads_fit, -1), rcond=None)
Adec_map  = params[1].reshape(ny, nx)
delta_map = params[2].reshape(ny, nx)

grads_bfe = grads_raw.copy()
Q_med = np.zeros((ny, nx))
for g in range(n_grads):
    if g > 0:
        Q_med = Q_med + np.median(grads_bfe[:, g-1, :, :], axis=0)
    KQ = fftconvolve(Q_med, K, mode='same')
    factor = np.where(1.0 - A * KQ > 0.05, 1.0 - A * KQ, 1.0)
    grads_bfe[:, g] = grads_raw[:, g] / factor[None]
    print(f'  BFE g={g}', end='\r')
print()

grads_joint = grads_bfe.copy()
for g in range(n_grads_fit):
    decay_g = Adec_map * np.exp(-g / tau)
    if g == 0:
        grads_joint[:, 0] = grads_bfe[:, 0] - decay_g[None] + delta_map[None]
    else:
        grads_joint[:, g] = grads_bfe[:, g] - decay_g[None]

# Median over integrations for the cutout region
sy, sx = STAR_Y, STAR_X
med_joint = np.median(grads_joint, axis=0)   # (n_grads, ny, nx)

# Mean image across good groups for the pixel grid color
img_mean = np.mean(med_joint[1:n_grads_fit], axis=0)
cutout = img_mean[sy-HALF:sy+HALF+1, sx-HALF:sx+HALF+1]

# Per-pixel profiles for cutout pixels: good groups 1-8
g_good = np.arange(1, n_grads_fit)
n_side = 2 * HALF + 1

# Assign a unique color to each pixel from a colormap
cmap_pix = plt.cm.turbo
pixel_colors = {}
for iy in range(n_side):
    for ix in range(n_side):
        idx = iy * n_side + ix
        pixel_colors[(iy, ix)] = cmap_pix(idx / (n_side**2 - 1))

# ---------------------------------------------------------------------------
# Figure: image on left, matching grid of lightcurve subplots on right
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 9))

# Left: pixel image
ax_img = fig.add_axes([0.03, 0.08, 0.28, 0.84])
vmin = np.nanpercentile(cutout, 1)
vmax = np.nanpercentile(cutout, 99)
ax_img.imshow(cutout, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax,
              extent=[-HALF-0.5, HALF+0.5, -HALF-0.5, HALF+0.5],
              interpolation='nearest')
for i in np.arange(-HALF, HALF+2) - 0.5:
    ax_img.axhline(i, color='white', lw=0.5, alpha=0.5)
    ax_img.axvline(i, color='white', lw=0.5, alpha=0.5)
ax_img.set_xlabel('Δx (px)', fontsize=8)
ax_img.set_ylabel('Δy (px)', fontsize=8)
ax_img.set_title('Mean joint-corrected\ngradient (DN/group)', fontsize=8)
ax_img.set_xticks(np.arange(-HALF, HALF+1))
ax_img.set_yticks(np.arange(-HALF, HALF+1))
ax_img.tick_params(labelsize=6)

# Right: n_side x n_side grid of lightcurve subplots
# Row 0 in subplot grid = top of image = iy = n_side-1
lc_left   = 0.35
lc_bottom = 0.05
lc_width  = 0.62
lc_height = 0.90
cell_w = lc_width  / n_side
cell_h = lc_height / n_side

for iy in range(n_side):
    for ix in range(n_side):
        py = sy - HALF + iy
        px = sx - HALF + ix
        profile = med_joint[g_good, py, px]

        left   = lc_left   + ix * cell_w
        bottom = lc_bottom + iy * cell_h
        ax = fig.add_axes([left, bottom, cell_w * 0.92, cell_h * 0.88])

        ax.plot(g_good, profile, '-', color='C0', lw=1.0)
        ax.plot(g_good, profile, '.', color='C0', ms=3)
        pad = (profile.max() - profile.min()) * 0.15 or 1.0
        ax.set_ylim(profile.min() - pad, profile.max() + pad)
        ax.set_xlim(g_good[0] - 0.5, g_good[-1] + 0.5)

        # Only show tick labels on edge subplots
        if iy == 0:
            ax.set_xlabel('g', fontsize=5)
            ax.tick_params(axis='x', labelsize=4)
        else:
            ax.set_xticklabels([])
        if ix == 0:
            ax.tick_params(axis='y', labelsize=4)
        else:
            ax.set_yticklabels([])

        # Label with pixel offset
        dy = iy - HALF
        dx = ix - HALF
        ax.set_title(f'({dx:+d},{dy:+d})', fontsize=4.5, pad=1)
        ax.tick_params(length=2)

fig.suptitle('Wolf-359: joint-corrected per-pixel gradient profiles (groups 1–8)',
             fontsize=10, fontweight='bold', x=0.65)

out = OUT / 'wolf359_pixel_grid_lc.png'
fig.savefig(out, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
