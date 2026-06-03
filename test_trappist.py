"""
test_trappist.py

Apply correct_bfe_rcd to TRAPPIST-1 MIRI ramp data (seg001).
Use SEP with the jurassic full_MIRI_mask to locate the star correctly,
then produce aperture lightcurves before and after correction.
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

TRAP_DIR = Path('/Users/rri38/Documents/work/code/jwst/ramps/trappist')
MASK_PATH = Path('/Users/rri38/Documents/work/code/jwst/jurassic/full_MIRI_mask.npy')
OUT = Path(__file__).parent
SEG = 'jw01177007001_03101_00001-seg001_mirimage_ramp.fits'

AP_RADIUS = 5
BG_INNER = 20
BG_OUTER = 60
N_GROUPS = 14
NY, NX = 1024, 1032

# ---------------------------------------------------------------------------
# Load mask (True = science pixel)
# ---------------------------------------------------------------------------
sci_mask = np.load(MASK_PATH)   # (1024, 1032), True = good science pixel
print(f'Mask loaded: {sci_mask.sum()} science pixels / {sci_mask.size} total')

# ---------------------------------------------------------------------------
# Load ramp segment
# ---------------------------------------------------------------------------
def load_seg(path, n_groups=N_GROUPS, ny=NY, nx=NX):
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

cube = load_seg(TRAP_DIR / SEG)
n_int, n_groups, ny, nx = cube.shape
print(f'Loaded: {cube.shape}')

# ---------------------------------------------------------------------------
# Find TRAPPIST-1 with SEP, using mask to exclude bad pixels
# ---------------------------------------------------------------------------
grads_raw = np.diff(cube, axis=1)
detect_img = np.median(grads_raw[:, 1:n_groups-2], axis=(0, 1)).astype(np.float64)

# SEP convention: mask=True means pixel is masked/ignored
sep_mask = ~sci_mask

bkg = sep.Background(detect_img, mask=sep_mask)
img_sub = (detect_img - bkg.back()).astype(np.float64)
objects = sep.extract(img_sub, thresh=5.0, err=bkg.globalrms, mask=sep_mask)

# Pick brightest source near known approximate position
APPROX_X, APPROX_Y = 700, 500
SEARCH_R = 100
dist = np.sqrt((objects['x'] - APPROX_X)**2 + (objects['y'] - APPROX_Y)**2)
nearby = objects[dist < SEARCH_R]
nearby.sort(order='flux')
star = nearby[-1]
sy, sx = int(round(star['y'])), int(round(star['x']))
print(f'TRAPPIST-1 found at x={sx}, y={sy}  flux={star["flux"]:.0f}')

# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------
yy, xx = np.mgrid[:ny, :nx]
r_star = np.sqrt((yy - sy)**2 + (xx - sx)**2)
ap_mask = r_star <= AP_RADIUS
bg_mask = (r_star >= BG_INNER) & (r_star <= BG_OUTER) & sci_mask
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
g_good = np.arange(1, n_groups - 2)

lc_raw = grads_raw[:, g_good][:, :, ap_mask].sum(axis=2)
lc_cor = grads_cor[:, g_good][:, :, ap_mask].sum(axis=2)

lc_raw_n = lc_raw / np.median(lc_raw)
lc_cor_n = lc_cor / np.median(lc_cor)

integ = np.arange(n_int)

rms_raw = np.std(lc_raw_n) * 100
rms_cor = np.std(lc_cor_n) * 100
print(f'Aperture LC RMS:')
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
        ax.scatter(integ, lc_n[:, i], color=colors[g], s=6, alpha=0.7, zorder=g+1)
    ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.4)
    ax.set_xlabel('Integration index')
    ax.set_ylabel('Normalised aperture flux')
    ax.set_title(f'{title}  (RMS={rms:.3f}%)', fontsize=9)

sm = plt.cm.ScalarMappable(cmap=cmap_g, norm=plt.Normalize(vmin=0, vmax=n_groups-1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[-1], pad=0.01)
cbar.set_label('Group index')
cbar.set_ticks(np.arange(n_groups))

fig.suptitle(f'TRAPPIST-1  (x={sx}, y={sy})  AP_RADIUS={AP_RADIUS}px  seg001',
             fontsize=10, fontweight='bold')
fig.tight_layout()
out1 = OUT / 'trappist_corrected_lc.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out1}')

# ---------------------------------------------------------------------------
# Figure 2: detection image with star and apertures marked
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
cut = 80
region = img_sub[sy-cut:sy+cut, sx-cut:sx+cut].copy()
region[~sci_mask[sy-cut:sy+cut, sx-cut:sx+cut]] = np.nan
vmin, vmax = np.nanpercentile(region, [1, 99])
ax.imshow(region, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax,
          extent=[sx-cut-0.5, sx+cut+0.5, sy-cut-0.5, sy+cut+0.5])
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(sx + AP_RADIUS*np.cos(theta), sy + AP_RADIUS*np.sin(theta),
        'r-', lw=1.5, label=f'Aperture r={AP_RADIUS}px')
ax.plot(sx + BG_INNER*np.cos(theta), sy + BG_INNER*np.sin(theta),
        'w--', lw=1, label=f'BG inner r={BG_INNER}px')
ax.plot(sx + BG_OUTER*np.cos(theta), sy + BG_OUTER*np.sin(theta),
        'w--', lw=1, label=f'BG outer r={BG_OUTER}px')
ax.plot(sx, sy, '+', color='red', ms=12, lw=2)
ax.set_xlabel('x (px)')
ax.set_ylabel('y (px)')
ax.set_title(f'TRAPPIST-1 detection  (masked bad pixels shown as NaN)', fontsize=9)
ax.legend(fontsize=8)
fig.colorbar(plt.cm.ScalarMappable(
    norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap='viridis'), ax=ax, label='DN/group')
fig.tight_layout()
out2 = OUT / 'trappist_detection.png'
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out2}')
