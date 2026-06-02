"""
explore_rcd_functional_form.py

Explore whether the reset charge decay at the Wolf-359 star core follows
a different functional form than the simple exponential fitted to background.

After BFE inversion, the gradient profile at each pixel should contain only
the reset decay (plus noise). We fit multiple functional forms to the
BFE-corrected median gradient at the star core and compare.

Functional forms tested:
  1. Single exponential:   C + A * exp(-g/tau)
  2. Stretched exponential: C + A * exp(-(g/tau)^beta)
  3. Two-component exp:    C + A1*exp(-g/tau1) + A2*exp(-g/tau2)
  4. Power law:            C + A * (g+1)^(-alpha)
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

A_BFE = 1.035e-6
ALPHA_BFE = 2.783
STAR_Y, STAR_X = 89, 110
AP_RADIUS = 3    # tight core aperture for cleanest signal
BG_RMIN, BG_RMAX = 20, 60

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
r_star = np.sqrt((yy - STAR_Y)**2 + (xx - STAR_X)**2)
ap_mask = r_star <= AP_RADIUS
bg_mask = (r_star >= BG_RMIN) & (r_star <= BG_RMAX)

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

# Aperture-mean profile for star and background (good groups only)
g_fit = np.arange(1, n_grads_fit, dtype=float)   # groups 1-8, exclude g=0 anomaly
prof_star = np.array([med_bfe[g, ap_mask].mean() for g in g_fit.astype(int)])
prof_bg   = np.array([med_bfe[g, bg_mask].mean() for g in g_fit.astype(int)])

print(f'Star core profile (r<={AP_RADIUS}px, BFE-corrected, groups 1-8):')
for g, v in zip(g_fit.astype(int), prof_star):
    print(f'  g={g}: {v:.1f} DN/group')

# ---------------------------------------------------------------------------
# Functional forms
# ---------------------------------------------------------------------------
def f_exp1(g, C, A, tau):
    return C + A * np.exp(-g / tau)

def f_stretch(g, C, A, tau, beta):
    return C + A * np.exp(-(g / tau)**beta)

def f_two_exp(g, C, A1, tau1, A2, tau2):
    return C + A1 * np.exp(-g / tau1) + A2 * np.exp(-g / tau2)

def f_power(g, C, A, alpha):
    return C + A * (g + 1)**(-alpha)

# ---------------------------------------------------------------------------
# Fit all forms to star and background profiles
# ---------------------------------------------------------------------------
def fit_and_report(g, prof, label):
    results = {}
    g = g.copy()

    # 1. Single exponential
    try:
        p0 = [prof[-1], prof[0]-prof[-1], 1.5]
        popt, _ = curve_fit(f_exp1, g, prof, p0=p0, maxfev=10000)
        resid = prof - f_exp1(g, *popt)
        results['exp1'] = dict(popt=popt, rms=np.std(resid),
                               labels=['C', 'A', 'tau'], fn=f_exp1)
    except Exception as e:
        print(f'  exp1 failed: {e}')

    # 2. Stretched exponential
    try:
        p0 = [prof[-1], prof[0]-prof[-1], 1.5, 1.0]
        popt, _ = curve_fit(f_stretch, g, prof, p0=p0,
                            bounds=([0, 0, 0.1, 0.1], [np.inf, np.inf, 20, 5]),
                            maxfev=20000)
        resid = prof - f_stretch(g, *popt)
        results['stretch'] = dict(popt=popt, rms=np.std(resid),
                                  labels=['C', 'A', 'tau', 'beta'], fn=f_stretch)
    except Exception as e:
        print(f'  stretch failed: {e}')

    # 3. Two-component exponential
    try:
        p0 = [prof[-1], prof[0]-prof[-1], 0.5, (prof[0]-prof[-1])*0.2, 3.0]
        popt, _ = curve_fit(f_two_exp, g, prof, p0=p0,
                            bounds=([0, 0, 0.1, 0, 0.5], [np.inf, np.inf, 3, np.inf, 20]),
                            maxfev=20000)
        resid = prof - f_two_exp(g, *popt)
        results['two_exp'] = dict(popt=popt, rms=np.std(resid),
                                  labels=['C', 'A1', 'tau1', 'A2', 'tau2'], fn=f_two_exp)
    except Exception as e:
        print(f'  two_exp failed: {e}')

    # 4. Power law
    try:
        p0 = [prof[-1], prof[0]-prof[-1], 1.0]
        popt, _ = curve_fit(f_power, g, prof, p0=p0, maxfev=10000)
        resid = prof - f_power(g, *popt)
        results['power'] = dict(popt=popt, rms=np.std(resid),
                                labels=['C', 'A', 'alpha'], fn=f_power)
    except Exception as e:
        print(f'  power failed: {e}')

    print(f'\n{label}:')
    for name, r in results.items():
        params_str = ', '.join(f'{l}={v:.4f}' for l, v in zip(r['labels'], r['popt']))
        print(f'  {name:10s}  RMS={r["rms"]:.3f}  {params_str}')

    return results

res_star = fit_and_report(g_fit, prof_star, 'Star core (BFE-corrected)')
res_bg   = fit_and_report(g_fit, prof_bg,   'Background')

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
g_fine = np.linspace(g_fit[0], g_fit[-1], 200)

colors = {'exp1': 'C3', 'stretch': 'C0', 'two_exp': 'C2', 'power': 'C4'}
labels_nice = {'exp1': 'Single exp', 'stretch': 'Stretched exp',
               'two_exp': 'Two-component exp', 'power': 'Power law'}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for col, (prof, res, title) in enumerate([
        (prof_star, res_star, f'Star core (r≤{AP_RADIUS}px, BFE-corrected)'),
        (prof_bg,   res_bg,   f'Background (r={BG_RMIN}–{BG_RMAX}px, BFE-corrected)'),
]):
    # Profile + fits
    ax = axes[0, col]
    ax.plot(g_fit, prof, 'ko', ms=6, zorder=5, label='Data')
    for name, r in res.items():
        y = r['fn'](g_fine, *r['popt'])
        ax.plot(g_fine, y, '-', color=colors[name], lw=1.5,
                label=f"{labels_nice[name]}  (RMS={r['rms']:.2f})")
    ax.set_xlabel('Gradient index')
    ax.set_ylabel('Mean gradient (DN/group)')
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.set_xticks(g_fit.astype(int))

    # Residuals
    ax = axes[1, col]
    for name, r in res.items():
        resid = prof - r['fn'](g_fit, *r['popt'])
        ax.plot(g_fit, resid, 'o-', color=colors[name], lw=1.2, ms=4,
                label=f"{labels_nice[name]}  (RMS={r['rms']:.2f})")
    ax.axhline(0, color='k', lw=0.7, ls='--', alpha=0.4)
    ax.set_xlabel('Gradient index')
    ax.set_ylabel('Residual (DN/group)')
    ax.set_title('Fit residuals', fontsize=9)
    ax.legend(fontsize=7)
    ax.set_xticks(g_fit.astype(int))

fig.suptitle('Wolf-359: reset decay functional form — star core vs background',
             fontsize=11, fontweight='bold')
fig.tight_layout()
out = OUT / 'wolf359_rcd_functional_form.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved {out}')
