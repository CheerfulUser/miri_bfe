"""
fit_rcd_params_perpixel.py

Fit stretched exponential reset decay independently to every pixel on the
BFE-corrected median gradients. Show the spatial maps and distributions of
all four parameters: C (rate), A (amplitude), tau, beta.
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
BORDER = 10

BASE = Path('/Users/rri38/Documents/work/code/jwst/ramps/wolf-359')
OUT = Path(__file__).parent

NPZ = OUT / 'rcd_perpixel_params.npz'
if NPZ.exists():
    data = np.load(NPZ)
    C_map = data['C']
    A_map = data['A']
    tau_map = data['tau']
    beta_map = data['beta']
    rms_map = data['rms']
    ny, nx = C_map.shape
    print(f'Loaded params from {NPZ}  ({ny}x{nx})')
else:
    with fits.open(BASE / 'uncal-fits/jw06122002001_02101_00001_mirimage_uncal.fits') as h:
        cube = h['SCI'].data.astype(float)

    n_int, n_groups, ny, nx = cube.shape
    n_grads = n_groups - 1
    n_grads_fit = n_grads - 1
    grads_raw = np.diff(cube, axis=1)
    print(f'Loaded: {cube.shape}')

    # -----------------------------------------------------------------------
    # BFE inversion
    # -----------------------------------------------------------------------
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

    med_bfe = np.median(grads_bfe, axis=0)

    # Fit on groups 1-8 (exclude g=0 first-frame anomaly)
    g_fit = np.arange(1, n_grads_fit, dtype=float)
    profiles = med_bfe[g_fit.astype(int)]   # (8, ny, nx)

    # -----------------------------------------------------------------------
    # Per-pixel stretched exponential fit
    # -----------------------------------------------------------------------
    def f_stretch(g, C, A, tau, beta):
        return C + A * np.exp(-(g / tau)**beta)

    C_map    = np.full((ny, nx), np.nan)
    A_map    = np.full((ny, nx), np.nan)
    tau_map  = np.full((ny, nx), np.nan)
    beta_map = np.full((ny, nx), np.nan)
    rms_map  = np.full((ny, nx), np.nan)

    total = ny * nx
    done = 0
    for iy in range(ny):
        for ix in range(nx):
            prof = profiles[:, iy, ix]
            if not np.all(np.isfinite(prof)) or prof.std() < 0.01:
                done += 1
                continue
            amp = prof[0] - prof[-1]
            p0 = [prof[-1], max(amp, 1.0), 1.5, 0.8]
            try:
                ps, _ = curve_fit(f_stretch, g_fit, prof, p0=p0,
                                  bounds=([0, 0, 0.05, 0.05],
                                          [np.inf, np.inf, 30, 5]),
                                  maxfev=5000)
                C_map[iy, ix] = ps[0]
                A_map[iy, ix] = ps[1]
                tau_map[iy, ix] = ps[2]
                beta_map[iy, ix] = ps[3]
                rms_map[iy, ix] = np.std(prof - f_stretch(g_fit, *ps))
            except Exception:
                pass
            done += 1
        if iy % 16 == 0:
            print(f'  {done}/{total} ({100*done/total:.0f}%)', end='\r')
    print(f'\nDone.')

    np.savez(NPZ, C=C_map, A=A_map, tau=tau_map, beta=beta_map, rms=rms_map)
    print(f'Saved {NPZ}')

# ---------------------------------------------------------------------------
# Parameter statistics
# ---------------------------------------------------------------------------
interior = np.zeros((ny, nx), dtype=bool)
interior[BORDER:-BORDER, BORDER:-BORDER] = True
valid = np.isfinite(beta_map) & np.isfinite(tau_map) & interior
for name, arr in [('C (rate)', C_map), ('A (amplitude)', A_map),
                  ('tau', tau_map), ('beta', beta_map), ('rms', rms_map)]:
    v = arr[valid]
    print(f'{name:20s}  median={np.median(v):.3f}  '
          f'p10={np.percentile(v,10):.3f}  p90={np.percentile(v,90):.3f}')

# ---------------------------------------------------------------------------
# Figure: spatial maps + histograms
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

params = [
    (C_map,    'C — rate (DN/group)',            'viridis', None, None),
    (A_map,    'A — decay amplitude (DN/group)', 'viridis', None, None),
    (tau_map,  'tau — timescale (groups)',        'plasma',  0,    None),
    (beta_map, 'beta — stretch exponent',         'RdBu_r',  0.1,  2.5),
]

for col, (arr, title, cmap, vmin, vmax) in enumerate(params):
    # Map — no clipping so star is visible
    ax = axes[0, col]
    kw = {}
    if vmin is not None: kw['vmin'] = vmin
    if vmax is not None: kw['vmax'] = vmax
    im = ax.imshow(arr, origin='lower', cmap=cmap, **kw)
    fig.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('x'); ax.set_ylabel('y')

    # Histogram — full range including star
    ax = axes[1, col]
    v = arr[valid].ravel()
    lo = np.nanpercentile(v, 0.5)
    hi = np.nanpercentile(v, 99.5)
    v_clip = v[(v >= lo) & (v <= hi)]
    ax.hist(v_clip, bins=80, color='C0', alpha=0.7)
    ax.axvline(np.median(v_clip), color='C3', lw=1.5, ls='--',
               label=f'median={np.median(v_clip):.2f}')
    if title.startswith('beta'):
        ax.axvline(1.0, color='k', lw=1.0, ls=':', label='β=1 (pure exp)')
    ax.set_xlabel(title.split(' — ')[0])
    ax.set_ylabel('N pixels')
    ax.legend(fontsize=7)

fig.suptitle('Wolf-359: per-pixel stretched exponential reset decay parameters',
             fontsize=11, fontweight='bold')
fig.tight_layout()
out = OUT / 'wolf359_rcd_perpixel_params.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
