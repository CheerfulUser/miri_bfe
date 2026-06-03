"""
test_evlac.py

Apply correct_bfe_rcd to EV Lac MIRI uncal ramp data.
Use SEP to locate the star, then produce aperture lightcurves.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from astropy.io import fits
import sep
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from ramp_correction import correct_bfe_rcd

DATA = Path('/Users/rri38/Documents/work/code/jwst/ramps/evlac/MAST_2026-05-21T23_53_57.816Z/JWST/jw06122010001_02101_00001_mirimage_uncal.fits')
OUT = Path(__file__).parent

AP_RADIUS = 5
BG_INNER = 20
BG_OUTER = 80

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
with fits.open(DATA) as h:
    cube = h['SCI'].data.astype(float)

n_int, n_groups, ny, nx = cube.shape
print(f'Loaded: {cube.shape}')

# ---------------------------------------------------------------------------
# Find EV Lac with SEP
# ---------------------------------------------------------------------------
grads_raw = np.diff(cube, axis=1)
n_grads = n_groups - 2   # exclude last-frame anomaly
detect_img = np.median(grads_raw[:, 1:n_grads], axis=(0, 1)).astype(np.float64)

bkg = sep.Background(detect_img)
img_sub = (detect_img - bkg.back()).astype(np.float64)
objects = sep.extract(img_sub, thresh=5.0, err=bkg.globalrms)
objects.sort(order='flux')

star = objects[-1]
sy, sx = int(round(star['y'])), int(round(star['x']))
print(f'EV Lac found at x={sx}, y={sy}  flux={star["flux"]:.0f}')

# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------
yy, xx = np.mgrid[:ny, :nx]
r_star = np.sqrt((yy - sy)**2 + (xx - sx)**2)
ap_mask = r_star <= AP_RADIUS
bg_mask = (r_star >= BG_INNER) & (r_star <= BG_OUTER)
print(f'Aperture pixels: {ap_mask.sum()}  Background pixels: {bg_mask.sum()}')

# ---------------------------------------------------------------------------
# Apply correction
# ---------------------------------------------------------------------------
print('Running correct_bfe_rcd ...')
cube_cor = correct_bfe_rcd(cube, bg_mask=bg_mask, verbose=True)
print('Correction done.')

grads_cor = np.diff(cube_cor, axis=1)

# ---------------------------------------------------------------------------
# Aperture lightcurves
# ---------------------------------------------------------------------------
g_good = np.arange(1, n_grads)

lc_raw = grads_raw[:, g_good][:, :, ap_mask].sum(axis=2)
lc_cor = grads_cor[:, g_good][:, :, ap_mask].sum(axis=2)

lc_raw_n = lc_raw / np.median(lc_raw)
lc_cor_n = lc_cor / np.median(lc_cor)

integ = np.arange(n_int)

rms_raw = np.std(lc_raw_n) * 100
rms_cor = np.std(lc_cor_n) * 100
print(f'Aperture LC RMS (groups 1-{n_grads-1}):')
print(f'  Raw       : {rms_raw:.3f}%')
print(f'  Corrected : {rms_cor:.3f}%')

# ---------------------------------------------------------------------------
# Figure 1: LC colored by group
# ---------------------------------------------------------------------------
cmap_g = cm.get_cmap('plasma', n_groups)
colors = [cmap_g(g) for g in range(n_groups)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, lc_n, title, rms in [
        (axes[0], lc_raw_n, 'Raw', rms_raw),
        (axes[1], lc_cor_n, 'BFE + RCD corrected', rms_cor),
]:
    for i, g in enumerate(g_good):
        ax.scatter(integ, lc_n[:, i], color=colors[g], s=4, alpha=0.7, zorder=g+1)
    ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.4)
    ax.set_xlabel('Integration index')
    ax.set_ylabel('Normalised aperture flux')
    ax.set_title(f'{title}  (RMS={rms:.3f}%)', fontsize=9)

sm = plt.cm.ScalarMappable(cmap=cmap_g, norm=plt.Normalize(vmin=0, vmax=n_groups-1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[-1], pad=0.01)
cbar.set_label('Group index')
cbar.set_ticks(np.arange(n_groups))

fig.suptitle(f'EV Lac  (x={sx}, y={sy})  AP_RADIUS={AP_RADIUS}px',
             fontsize=10, fontweight='bold')
fig.tight_layout()
out1 = OUT / 'evlac_corrected_lc.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out1}')

# ---------------------------------------------------------------------------
# Figure 2: detection image
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
vmin, vmax = np.nanpercentile(img_sub, [1, 99])
ax.imshow(img_sub, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(sx + AP_RADIUS*np.cos(theta), sy + AP_RADIUS*np.sin(theta),
        'r-', lw=1.5, label=f'Aperture r={AP_RADIUS}px')
ax.plot(sx + BG_INNER*np.cos(theta), sy + BG_INNER*np.sin(theta),
        'w--', lw=1, label=f'BG r={BG_INNER}px')
ax.plot(sx + BG_OUTER*np.cos(theta), sy + BG_OUTER*np.sin(theta),
        'w--', lw=1, label=f'BG r={BG_OUTER}px')
ax.plot(sx, sy, '+', color='red', ms=12, lw=2)
ax.set_xlabel('x (px)')
ax.set_ylabel('y (px)')
ax.set_title('EV Lac detection image (median gradient)', fontsize=9)
ax.legend(fontsize=8)
fig.colorbar(plt.cm.ScalarMappable(
    norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap='viridis'), ax=ax, label='DN/group')
fig.tight_layout()
out2 = OUT / 'evlac_detection.png'
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out2}')
