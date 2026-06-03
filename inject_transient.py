"""
inject_transient.py

Inject a synthetic transient (bright spike + exponential decay) at the
TRAPPIST-1 position using the MIRI F1500W PSF, apply the BFE+RCD correction,
and quantify how much the transient signal is affected.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
import sys
sys.path.insert(0, str(Path(__file__).parent))
from ramp_correction import correct_bfe_rcd

OUT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Load TRAPPIST-1 ramp
# ---------------------------------------------------------------------------
TRAP_DIR = Path('/Users/rri38/Documents/work/code/jwst/ramps/trappist')
MASK_PATH = Path('/Users/rri38/Documents/work/code/jwst/jurassic/full_MIRI_mask.npy')
SEG = 'jw01177007001_03101_00001-seg001_mirimage_ramp.fits'
N_GROUPS = 14
SY, SX = 515, 697
AP_RADIUS = 5

sci_mask = np.load(MASK_PATH)

with fits.open(TRAP_DIR / SEG, memmap=False, ignore_missing_end=True) as hdul:
    offset = hdul['SCI']._data_offset
    n_int_hdr = hdul['SCI'].header['NAXIS4']
bytes_per_int = N_GROUPS * 1024 * 1032 * 4
available = (TRAP_DIR / SEG).stat().st_size - offset
n_int = min(n_int_hdr, int(available // bytes_per_int))
with open(TRAP_DIR / SEG, 'rb') as fh:
    fh.seek(offset)
    raw = np.frombuffer(fh.read(n_int * bytes_per_int), dtype='>f4')
cube = raw.reshape(n_int, N_GROUPS, 1024, 1032).astype(float)
n_int, n_groups, ny, nx = cube.shape
print(f'Loaded: {cube.shape}')

# ---------------------------------------------------------------------------
# Load PSF and place at star position
# ---------------------------------------------------------------------------
psf = np.load(OUT / 'miri_f1500w_psf.npy')   # (41, 41), normalised
ph = psf.shape[0] // 2                          # half-size = 20

psf_full = np.zeros((ny, nx))
y0, y1 = SY - ph, SY + ph + 1
x0, x1 = SX - ph, SX + ph + 1
psf_full[y0:y1, x0:x1] = psf

# ---------------------------------------------------------------------------
# Transient model: spike at integration i0, exponential decay
# ---------------------------------------------------------------------------
I0 = 10                        # integration index of peak
PEAK_DN_PER_GROUP = 5000.0     # peak flux in DN/group (aperture-summed)
TAU_TRANSIENT = 5.0            # e-folding decay in integrations

integ = np.arange(n_int)
transient_lc = np.zeros(n_int)
transient_lc[I0:] = PEAK_DN_PER_GROUP * np.exp(-(integ[I0:] - I0) / TAU_TRANSIENT)

# Inject into cube: add transient_lc[i] * psf_full to every group in integration i
# The PSF flux is per group (constant within each integration)
cube_inj = cube.copy()
for i in range(n_int):
    if transient_lc[i] > 0:
        for g in range(n_groups):
            cube_inj[i, g] += transient_lc[i] * psf_full * (g + 1)

print('Transient injected.')
print(f'  Peak integration: {I0}, peak flux: {PEAK_DN_PER_GROUP:.0f} DN/group')
print(f'  Decay tau: {TAU_TRANSIENT:.1f} integrations')

# ---------------------------------------------------------------------------
# Apply BFE+RCD correction to both
# ---------------------------------------------------------------------------
yy, xx = np.mgrid[:ny, :nx]
r_star = np.sqrt((yy - SY)**2 + (xx - SX)**2)
bg_mask = (r_star > 20) & (r_star < 60) & sci_mask.astype(bool)

print('Correcting original cube...')
cube_cor = correct_bfe_rcd(cube, bg_mask=bg_mask, verbose=False)

print('Correcting injected cube...')
cube_inj_cor = correct_bfe_rcd(cube_inj, bg_mask=bg_mask, verbose=False)

print('Done.')

# ---------------------------------------------------------------------------
# Extract aperture lightcurves
# ---------------------------------------------------------------------------
ap_mask = r_star <= AP_RADIUS
g_good = np.arange(1, n_groups - 2)

def aperture_lc(c):
    grads = np.diff(c, axis=1)
    return grads[:, g_good][:, :, ap_mask].sum(axis=2)   # (n_int, n_good_groups)

lc_raw = aperture_lc(cube)
lc_raw_inj = aperture_lc(cube_inj)
lc_cor = aperture_lc(cube_cor)
lc_cor_inj = aperture_lc(cube_inj_cor)

# Median over good groups for each integration
lc_raw_med = np.median(lc_raw, axis=1)
lc_raw_inj_med = np.median(lc_raw_inj, axis=1)
lc_cor_med = np.median(lc_cor, axis=1)
lc_cor_inj_med = np.median(lc_cor_inj, axis=1)

# Extracted transient signal = injected - original
sig_raw = lc_raw_inj_med - lc_raw_med
sig_cor = lc_cor_inj_med - lc_cor_med

# Injected model (aperture sum of PSF × transient_lc)
ap_psf_sum = psf_full[ap_mask].sum()
sig_model = transient_lc * ap_psf_sum

# ---------------------------------------------------------------------------
# Quantify impact
# ---------------------------------------------------------------------------
peak_raw = sig_raw[I0]
peak_cor = sig_cor[I0]
peak_model = sig_model[I0]

print(f'\nPeak transient recovery:')
print(f'  Injected model:    {peak_model:.1f} DN/group')
print(f'  Raw extracted:     {peak_raw:.1f} DN/group  ({100*peak_raw/peak_model:.2f}%)')
print(f'  Corrected extracted: {peak_cor:.1f} DN/group  ({100*peak_cor/peak_model:.2f}%)')
print(f'  Correction impact: {100*(peak_cor-peak_raw)/peak_model:.3f}%')

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

ax = axes[0]
ax.plot(integ, sig_model / peak_model, 'k-', lw=2, label='Injected model')
ax.plot(integ, sig_raw / peak_model, 'C0-o', ms=4, lw=1.5, label='Raw extracted')
ax.plot(integ, sig_cor / peak_model, 'C3--s', ms=4, lw=1.5, label='BFE+RCD corrected')
ax.axvline(I0, color='k', lw=0.8, ls=':', alpha=0.5)
ax.set_xlabel('Integration index')
ax.set_ylabel('Normalised transient flux')
ax.set_title('Transient lightcurve', fontsize=9)
ax.legend(fontsize=8)

ax = axes[1]
ratio_raw = sig_raw / sig_model
ratio_cor = sig_cor / sig_model
valid = sig_model > 0.01 * peak_model
ax.plot(integ[valid], ratio_raw[valid], 'C0-o', ms=4, lw=1.5, label='Raw / model')
ax.plot(integ[valid], ratio_cor[valid], 'C3--s', ms=4, lw=1.5, label='Corrected / model')
ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Integration index')
ax.set_ylabel('Extracted / injected')
ax.set_title('Recovery fraction', fontsize=9)
ax.legend(fontsize=8)
ax.set_ylim(0.9, 1.1)

ax = axes[2]
impact_pct = 100 * (sig_cor - sig_raw) / np.where(sig_model > 0.01 * peak_model, sig_model, np.nan)
ax.plot(integ[valid], impact_pct[valid], 'k-o', ms=4, lw=1.5)
ax.axhline(0, color='k', lw=0.6, ls=':')
ax.set_xlabel('Integration index')
ax.set_ylabel('Correction impact (%)')
ax.set_title('BFE+RCD correction impact on transient', fontsize=9)

fig.suptitle(f'TRAPPIST-1: transient injection-recovery  (peak={PEAK_DN_PER_GROUP:.0f} DN/group, τ={TAU_TRANSIENT} int)',
             fontsize=10, fontweight='bold')
fig.tight_layout()
out = OUT / 'transient_injection_recovery.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
