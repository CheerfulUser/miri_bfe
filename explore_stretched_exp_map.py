"""
explore_stretched_exp_map.py

Fit the stretched exponential reset decay model to every pixel across
the detector and map the fitted parameters spatially.

Model: C + A * exp(-(g/tau)^beta)

If beta < 1 universally, the stretched exponential is the correct functional
form for all pixels, not just the star. If beta varies with charge level,
the decay timescale distribution depends on pixel flux.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

A_BFE = 1.035e-6
ALPHA_BFE = 2.783
STAR_Y, STAR_X = 89, 110

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

# ---------------------------------------------------------------------------
# BFE inversion
# ---------------------------------------------------------------------------
def make_kernel(alpha, kh=20):
    ii, jj = np.mgrid[-kh:kh+1, -kh:kh+1].astype(float)
    r = np.sqrt(ii**2 + jj**2)
    K = np.where(r > 0, -1.0 / r**alpha, 0.0)
    K[kh, kh] = -K.sum()
    return K

K = make_kernel(ALPHA_BFE)
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

med_bfe = np.median(grads_bfe, axis=0)   # (n_grads, ny, nx)

# Good groups 1-8
g_fit = np.arange(1, n_grads_fit, dtype=float)
n_gfit = len(g_fit)
profiles = med_bfe[g_fit.astype(int)]   # (8, ny, nx)

# Mean flux per pixel (used for charge-level correlation)
mean_flux = profiles.mean(axis=0)   # (ny, nx)

# ---------------------------------------------------------------------------
# Fit stretched exponential to every pixel
# ---------------------------------------------------------------------------
def f_stretch(g, C, A, tau, beta):
    return C + A * np.exp(-(g / tau)**beta)

def f_exp1(g, C, A, tau):
    return C + A * np.exp(-g / tau)

C_map    = np.full((ny, nx), np.nan)
A_map    = np.full((ny, nx), np.nan)
tau_map  = np.full((ny, nx), np.nan)
beta_map = np.full((ny, nx), np.nan)
rms_str  = np.full((ny, nx), np.nan)
tau_exp1 = np.full((ny, nx), np.nan)
rms_exp1 = np.full((ny, nx), np.nan)

total = ny * nx
done = 0
for iy in range(ny):
    for ix in range(nx):
        prof = profiles[:, iy, ix]
        if not np.all(np.isfinite(prof)) or prof.std() < 0.01:
            done += 1
            continue
        p0_s = [prof[-1], prof[0]-prof[-1], 1.5, 0.8]
        p0_e = [prof[-1], prof[0]-prof[-1], 1.5]
        try:
            ps, _ = curve_fit(f_stretch, g_fit, prof, p0=p0_s,
                              bounds=([0, -np.inf, 0.05, 0.05],
                                      [np.inf, np.inf, 30, 5]),
                              maxfev=5000)
            C_map[iy, ix]    = ps[0]
            A_map[iy, ix]    = ps[1]
            tau_map[iy, ix]  = ps[2]
            beta_map[iy, ix] = ps[3]
            rms_str[iy, ix]  = np.std(prof - f_stretch(g_fit, *ps))
        except Exception:
            pass
        try:
            pe, _ = curve_fit(f_exp1, g_fit, prof, p0=p0_e, maxfev=5000)
            tau_exp1[iy, ix] = pe[2]
            rms_exp1[iy, ix] = np.std(prof - f_exp1(g_fit, *pe))
        except Exception:
            pass
        done += 1
    if iy % 16 == 0:
        print(f'  {done}/{total} pixels fitted ({100*done/total:.0f}%)', end='\r')
print(f'\nFitting done.')

# RMS improvement: stretched vs single exp
rms_improvement = rms_exp1 - rms_str   # positive = stretched is better

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
valid = np.isfinite(beta_map) & np.isfinite(mean_flux)
print(f'\nBeta statistics across detector:')
print(f'  Median beta : {np.nanmedian(beta_map):.3f}')
print(f'  Mean beta   : {np.nanmean(beta_map):.3f}')
print(f'  10th pct    : {np.nanpercentile(beta_map, 10):.3f}')
print(f'  90th pct    : {np.nanpercentile(beta_map, 90):.3f}')
print(f'\nTau (single exp) statistics:')
print(f'  Median tau  : {np.nanmedian(tau_exp1):.3f}')
print(f'  Star region tau (r<6px): {np.nanmedian(tau_exp1[np.sqrt((np.mgrid[:ny,:nx][0]-STAR_Y)**2+(np.mgrid[:ny,:nx][1]-STAR_X)**2) < 6]):.3f}')

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

vmin_beta, vmax_beta = 0.3, 1.5

ax = axes[0, 0]
im = ax.imshow(beta_map, origin='lower', cmap='RdBu_r',
               vmin=vmin_beta, vmax=vmax_beta)
fig.colorbar(im, ax=ax, label='beta')
ax.set_title('Stretched exp beta (β<1 = sub-exp)', fontsize=9)
ax.scatter([STAR_X], [STAR_Y], marker='+', color='k', s=60, lw=1.5)
ax.set_xlabel('x'); ax.set_ylabel('y')

ax = axes[0, 1]
im = ax.imshow(tau_exp1, origin='lower', cmap='viridis',
               vmin=0, vmax=np.nanpercentile(tau_exp1, 98))
fig.colorbar(im, ax=ax, label='tau (groups)')
ax.set_title('Single-exp tau map', fontsize=9)
ax.scatter([STAR_X], [STAR_Y], marker='+', color='w', s=60, lw=1.5)
ax.set_xlabel('x'); ax.set_ylabel('y')

ax = axes[0, 2]
im = ax.imshow(rms_improvement, origin='lower', cmap='viridis',
               vmin=0, vmax=np.nanpercentile(rms_improvement, 99))
fig.colorbar(im, ax=ax, label='ΔRMS (DN/group)')
ax.set_title('RMS improvement: stretched vs single exp', fontsize=9)
ax.scatter([STAR_X], [STAR_Y], marker='+', color='w', s=60, lw=1.5)
ax.set_xlabel('x'); ax.set_ylabel('y')

# Beta vs mean flux
ax = axes[1, 0]
flux_vals = mean_flux[valid].ravel()
beta_vals = beta_map[valid].ravel()
# Bin by flux
bins = np.percentile(flux_vals, np.linspace(0, 100, 30))
bin_cents = 0.5 * (bins[:-1] + bins[1:])
beta_binned = [np.median(beta_vals[(flux_vals >= bins[i]) & (flux_vals < bins[i+1])])
               for i in range(len(bins)-1)]
ax.scatter(flux_vals, beta_vals, s=0.3, alpha=0.1, color='C0', rasterized=True)
ax.plot(bin_cents, beta_binned, 'o-', color='C3', ms=4, lw=1.5, label='Binned median')
ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.5, label='Pure exponential')
ax.set_xlabel('Mean gradient (DN/group)')
ax.set_ylabel('beta')
ax.set_title('beta vs pixel flux level', fontsize=9)
ax.legend(fontsize=8)
ax.set_ylim(0, 3)

# Tau (stretched) vs mean flux
ax = axes[1, 1]
tau_vals = tau_map[valid].ravel()
tau_binned = [np.median(tau_vals[(flux_vals >= bins[i]) & (flux_vals < bins[i+1])])
              for i in range(len(bins)-1)]
ax.scatter(flux_vals, tau_vals, s=0.3, alpha=0.1, color='C0', rasterized=True)
ax.plot(bin_cents, tau_binned, 'o-', color='C3', ms=4, lw=1.5, label='Binned median')
ax.set_xlabel('Mean gradient (DN/group)')
ax.set_ylabel('tau (groups)')
ax.set_title('Stretched exp tau vs pixel flux', fontsize=9)
ax.legend(fontsize=8)
ax.set_ylim(0, np.nanpercentile(tau_vals, 95))

# Beta histogram
ax = axes[1, 2]
ax.hist(beta_map[valid].ravel(), bins=60, color='C0', alpha=0.7, range=(0.1, 3.0))
ax.axvline(1.0, color='C3', lw=1.5, ls='--', label='Pure exp (β=1)')
ax.axvline(np.nanmedian(beta_map), color='k', lw=1.5, ls='-',
           label=f'Median β={np.nanmedian(beta_map):.2f}')
ax.set_xlabel('beta')
ax.set_ylabel('Number of pixels')
ax.set_title('Beta distribution across detector', fontsize=9)
ax.legend(fontsize=8)

fig.suptitle('Wolf-359: stretched exponential reset decay fit across detector',
             fontsize=11, fontweight='bold')
fig.tight_layout()
out = OUT / 'wolf359_stretched_exp_map.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
