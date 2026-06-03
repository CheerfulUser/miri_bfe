"""
validate_fit_bfe_params.py

Prove that fit_bfe_params correctly identifies the star and fits A_bfe
by comparing the forward-model PSF difference against the observed one
for EV Lac and TRAPPIST-1.
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
sys.path.insert(0, str(Path(__file__).parent))
from ramp_correction import fit_bfe_params

OUT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
datasets = [
    dict(
        name='EV Lac',
        loader='uncal',
        path=Path('/Users/rri38/Documents/work/code/jwst/ramps/evlac/'
                  'MAST_2026-05-21T23_53_57.816Z/JWST/'
                  'jw06122010001_02101_00001_mirimage_uncal.fits'),
        sci_mask=None,
        n_groups=7,
    ),
    dict(
        name='TRAPPIST-1',
        loader='ramp',
        path=Path('/Users/rri38/Documents/work/code/jwst/ramps/trappist/'
                  'jw01177007001_03101_00001-seg001_mirimage_ramp.fits'),
        sci_mask=np.load('/Users/rri38/Documents/work/code/jwst/jurassic/full_MIRI_mask.npy'),
        n_groups=14,
        ny=1024, nx=1032,
    ),
]


def load_cube(ds):
    if ds['loader'] == 'uncal':
        with fits.open(ds['path']) as h:
            return h['SCI'].data.astype(float)
    ng, ny, nx = ds['n_groups'], ds['ny'], ds['nx']
    with fits.open(ds['path'], memmap=False, ignore_missing_end=True) as hdul:
        offset = hdul['SCI']._data_offset
        n_int_hdr = hdul['SCI'].header['NAXIS4']
    bytes_per_int = ng * ny * nx * 4
    available = ds['path'].stat().st_size - offset
    n_int = min(n_int_hdr, int(available // bytes_per_int))
    with open(ds['path'], 'rb') as fh:
        fh.seek(offset)
        raw = np.frombuffer(fh.read(n_int * bytes_per_int), dtype='>f4')
    return raw.reshape(n_int, ng, ny, nx).astype(float)


# ---------------------------------------------------------------------------
# Forward model helper (runs on cropped region around star)
# ---------------------------------------------------------------------------
def forward_model_diff(cube, A_bfe, alpha_bfe, sy, sx,
                       early_groups, late_groups, cut=20):
    n_int, n_groups, ny, nx = cube.shape
    n_grads = n_groups - 2
    g_arr = np.arange(n_grads, dtype=float)

    grads = np.diff(cube, axis=1)[:, :n_grads]
    med_grad = np.median(grads, axis=0)

    # tau and per-pixel maps from background annulus
    yy, xx = np.mgrid[:ny, :nx]
    r_star = np.sqrt((yy - sy)**2 + (xx - sx)**2)
    bg = (r_star > 15) & (r_star < min(ny, nx) // 3)
    mean_bg = np.nanmean(med_grad[1:, bg], axis=1)
    def _exp1(g, C, A, t): return C + A * np.exp(-g / t)
    popt, _ = curve_fit(_exp1, g_arr[1:], mean_bg,
                        p0=[mean_bg[-1], mean_bg[0]-mean_bg[-1], 1.5])
    tau = float(popt[2])

    exp_g = np.exp(-g_arr / tau)
    ff_col = np.zeros(n_grads); ff_col[0] = -1.0
    X = np.column_stack([np.ones(n_grads), exp_g, ff_col])
    params, _, _, _ = np.linalg.lstsq(X, med_grad.reshape(n_grads, -1), rcond=None)
    rate_map = params[0].reshape(ny, nx)
    Adec_map = params[1].reshape(ny, nx)
    delta_map = params[2].reshape(ny, nx)

    # Crop around star
    kh = 20
    crop = cut + kh + 30
    y0, y1 = max(0, sy-crop), min(ny, sy+crop+1)
    x0, x1 = max(0, sx-crop), min(nx, sx+crop+1)
    rate_c = rate_map[y0:y1, x0:x1]
    Adec_c = Adec_map[y0:y1, x0:x1]
    delta_c = delta_map[y0:y1, x0:x1]
    cy, cx = sy-y0, sx-x0
    nyc, nxc = rate_c.shape

    ii, jj = np.mgrid[-kh:kh+1, -kh:kh+1].astype(float)
    r = np.sqrt(ii**2 + jj**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        K = np.where(r > 0, -1.0 / r**alpha_bfe, 0.0)
    K[kh, kh] = -K.sum()

    Q = np.zeros((nyc, nxc))
    grads_s = np.zeros((n_grads, nyc, nxc))
    for g in range(n_grads):
        tg = rate_c + Adec_c * np.exp(-g / tau)
        if g == 0:
            tg = tg - delta_c
        KQ = fftconvolve(Q, K, mode='same')
        grads_s[g] = tg * (1.0 - A_bfe * KQ)
        Q += tg

    _ap_yy, _ap_xx = np.mgrid[:2*cut+1, :2*cut+1]
    _ap = np.sqrt((_ap_yy-cut)**2 + (_ap_xx-cut)**2) <= 5

    def _cutout(arr_3d, glist):
        stack = np.median(arr_3d[np.array(glist)], axis=0)
        c = stack[cy-cut:cy+cut+1, cx-cut:cx+cut+1]
        return c / c[_ap].sum()

    def _cutout_obs(arr_4d, glist):
        c_med = np.median(arr_4d[:, np.array(glist)], axis=(0, 1))
        c = c_med[cy-cut:cy+cut+1, cx-cut:cx+cut+1]
        return c / c[_ap].sum()

    grads_crop = grads[:, :, y0:y1, x0:x1]
    obs_early = _cutout_obs(grads_crop, early_groups)
    obs_late = _cutout_obs(grads_crop, late_groups)
    obs_diff = obs_late - obs_early

    sim_diff = _cutout(grads_s, late_groups) - _cutout(grads_s, early_groups)

    return obs_diff, sim_diff, tau


# ---------------------------------------------------------------------------
# Run fit and build figure
# ---------------------------------------------------------------------------
CUT = 20
fig, axes = plt.subplots(2, 3, figsize=(13, 8))

for row, ds in enumerate(datasets):
    print(f'\n=== {ds["name"]} ===')
    cube = load_cube(ds)
    n_int, n_groups, ny, nx = cube.shape
    n_grads = n_groups - 2

    A_bfe, sx, sy = fit_bfe_params(cube, sci_mask=ds['sci_mask'], verbose=True)

    n_e = max(2, min(3, n_grads // 4))
    start = 1 if n_grads < 8 else 2
    early_groups = list(range(start, start + n_e))
    late_groups = list(range(n_grads - n_e, n_grads))
    print(f'early={early_groups}  late={late_groups}')

    obs_diff, sim_diff, tau = forward_model_diff(
        cube, A_bfe, alpha_bfe=2.783, sy=sy, sx=sx,
        early_groups=early_groups, late_groups=late_groups, cut=CUT)

    residual = sim_diff - obs_diff
    vabs = np.nanpercentile(np.abs(obs_diff), 99)
    ext = [-CUT-0.5, CUT+0.5, -CUT-0.5, CUT+0.5]

    for col, (img, title) in enumerate([
        (obs_diff, 'Observed Late−Early'),
        (sim_diff, f'Model  A={A_bfe:.2e}'),
        (residual, 'Residual (model−obs)'),
    ]):
        ax = axes[row, col]
        vr = np.nanpercentile(np.abs(residual), 99) if col == 2 else vabs
        im = ax.imshow(img, origin='lower', cmap='RdBu_r',
                       vmin=-vr, vmax=vr, extent=ext)
        fig.colorbar(im, ax=ax, label='Norm. flux')
        if row == 0:
            ax.set_title(title, fontsize=9)
        ax.set_xlabel('Δx (px)')
        if col == 0:
            ax.set_ylabel(f'{ds["name"]}\nΔy (px)', fontsize=8)

    # Radial profile overlay on residual panel
    ax = axes[row, 2]
    yy_c, xx_c = np.mgrid[:2*CUT+1, :2*CUT+1]
    r_map = np.sqrt((yy_c-CUT)**2 + (xx_c-CUT)**2)
    r_int = np.arange(0, CUT)
    rp_obs = np.array([np.mean(obs_diff[np.round(r_map).astype(int)==r]) for r in r_int])
    rp_sim = np.array([np.mean(sim_diff[np.round(r_map).astype(int)==r]) for r in r_int])
    ax2 = ax.inset_axes([0.55, 0.55, 0.43, 0.43])
    ax2.plot(r_int, rp_obs, 'k-', lw=1.5, label='Obs')
    ax2.plot(r_int, rp_sim, 'C3--', lw=1.5, label='Model')
    ax2.axhline(0, color='k', lw=0.5, ls=':')
    ax2.set_xlabel('r (px)', fontsize=6)
    ax2.tick_params(labelsize=6)
    ax2.legend(fontsize=6)

fig.suptitle('fit_bfe_params validation: observed vs forward-model PSF difference',
             fontsize=11, fontweight='bold')
fig.tight_layout()
out = OUT / 'validate_fit_bfe_params.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved {out}')
