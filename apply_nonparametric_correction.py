"""
apply_nonparametric_correction.py

Joint BFE + reset-decay correction with no functional form assumption.

Steps:
  1. Causal BFE inversion (single pass, exact)
  2. Compute per-pixel per-group median over integrations -> C(y,x) + D(g,y,x)
  3. Estimate flat rate C_hat(y,x) from late groups (6-8) of the median profile
  4. Subtract median profile and add back C_hat to each integration

The RCD subtraction requires no model for the decay shape.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from astropy.io import fits
from scipy.signal import fftconvolve
import warnings
warnings.filterwarnings('ignore')

A_BFE = 1.035e-6
ALPHA_BFE = 2.783
STAR_Y, STAR_X = 89, 110
AP_RADIUS = 5
LATE_GROUPS = [6, 7, 8]

BASE = Path('/Users/rri38/Documents/work/code/jwst/ramps/wolf-359')
OUT = Path(__file__).parent

with fits.open(BASE / 'uncal-fits/jw06122002001_02101_00001_mirimage_uncal.fits') as h:
    cube = h['SCI'].data.astype(float)

n_int, n_groups, ny, nx = cube.shape
n_grads = n_groups - 1
n_grads_fit = n_grads - 1
grads_raw = np.diff(cube, axis=1)
print(f'Loaded: {cube.shape}')

yy, xx = np.mgrid[:ny, :nx]
ap_mask = (yy - STAR_Y)**2 + (xx - STAR_X)**2 <= AP_RADIUS**2

# ---------------------------------------------------------------------------
# Causal BFE inversion
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
print('BFE inversion done.')

# ---------------------------------------------------------------------------
# Non-parametric RCD subtraction
# ---------------------------------------------------------------------------
# Per-pixel per-group median over integrations: shape (n_grads, ny, nx)
med_bfe = np.median(grads_bfe, axis=0)

# Flat rate estimate: mean over late groups where decay is negligible
C_hat = np.mean(med_bfe[LATE_GROUPS], axis=0)   # (ny, nx)

# Subtract median profile, add back flat rate
grads_np = grads_bfe.copy()
for g in range(n_grads_fit):
    grads_np[:, g] = grads_bfe[:, g] - med_bfe[g][None] + C_hat[None]

print('Non-parametric correction done.')

# ---------------------------------------------------------------------------
# Joint parametric correction (single exp, tau fitted from BFE-corrected bg)
# ---------------------------------------------------------------------------
from scipy.optimize import curve_fit

BG_RMIN, BG_RMAX = 20, 60
bg_mask = (np.sqrt((yy - STAR_Y)**2 + (xx - STAR_X)**2) >= BG_RMIN) & \
          (np.sqrt((yy - STAR_Y)**2 + (xx - STAR_X)**2) <= BG_RMAX)

g_arr = np.arange(n_grads_fit, dtype=float)
g_fit_bg = g_arr[1:]
mean_bg = np.nanmean(med_bfe[1:n_grads_fit, bg_mask], axis=1)

def exp1(g, C, A, tau): return C + A * np.exp(-g / tau)
popt_e, _ = curve_fit(exp1, g_fit_bg, mean_bg,
                       p0=[mean_bg[-1], mean_bg[0]-mean_bg[-1], 1.5])
tau_e = float(popt_e[2])
print(f'Joint parametric tau={tau_e:.4f} groups')

exp_g = np.exp(-g_arr / tau_e)
ff_col = np.zeros(n_grads_fit); ff_col[0] = -1.0
X = np.column_stack([np.ones(n_grads_fit), exp_g, ff_col])
params_e, _, _, _ = np.linalg.lstsq(
    X, med_bfe[:n_grads_fit].reshape(n_grads_fit, -1), rcond=None)
Adec_map_e = params_e[1].reshape(ny, nx)
delta_map_e = params_e[2].reshape(ny, nx)

grads_joint = grads_bfe.copy()
for g in range(n_grads_fit):
    decay_g = Adec_map_e * np.exp(-g / tau_e)
    if g == 0:
        grads_joint[:, 0] = grads_bfe[:, 0] - decay_g[None] + delta_map_e[None]
    else:
        grads_joint[:, g] = grads_bfe[:, g] - decay_g[None]

print('Joint parametric correction done.')

# ---------------------------------------------------------------------------
# Hybrid: subtract median ramp of joint-corrected gradients per pixel
# ---------------------------------------------------------------------------
med_joint = np.median(grads_joint, axis=0)   # (n_grads, ny, nx)
C_hat_joint = np.mean(med_joint[LATE_GROUPS], axis=0)   # (ny, nx)

grads_hybrid = grads_joint.copy()
for g in range(n_grads_fit):
    grads_hybrid[:, g] = grads_joint[:, g] - med_joint[g][None] + C_hat_joint[None]

print('Hybrid correction done.')

# ---------------------------------------------------------------------------
# Aperture lightcurves
# ---------------------------------------------------------------------------
lc_raw = grads_raw[:, :, ap_mask].sum(axis=2)   # (n_int, n_grads)
lc_bfe = grads_bfe[:, :, ap_mask].sum(axis=2)
lc_joint = grads_joint[:, :, ap_mask].sum(axis=2)
lc_hybrid = grads_hybrid[:, :, ap_mask].sum(axis=2)

g_good = np.arange(1, n_grads_fit)
lc_raw_n = lc_raw / np.median(lc_raw[:, g_good])
lc_bfe_n = lc_bfe / np.median(lc_bfe[:, g_good])
lc_joint_n = lc_joint / np.median(lc_joint[:, g_good])
lc_hybrid_n = lc_hybrid / np.median(lc_hybrid[:, g_good])

integ = np.arange(n_int)

print(f'Aperture LC RMS (groups 1-8):')
for label, lc_n in [('Raw', lc_raw_n), ('BFE only', lc_bfe_n),
                     ('Joint parametric', lc_joint_n), ('Hybrid', lc_hybrid_n)]:
    print(f'  {label:22s}: {np.std(lc_n[:, g_good])*100:.3f}%')

# ---------------------------------------------------------------------------
# Figure 1: LC colored by group
# ---------------------------------------------------------------------------
cmap_g = cm.get_cmap('plasma', n_grads)
colors = [cmap_g(g) for g in range(n_grads)]

fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=False)

for ax, lc_n, title in [
        (axes[0], lc_raw_n, 'Raw'),
        (axes[1], lc_bfe_n, 'BFE only'),
        (axes[2], lc_joint_n, 'Joint parametric RCD'),
        (axes[3], lc_hybrid_n, 'Joint + median subtraction'),
]:
    for g in g_good:
        ax.scatter(integ, lc_n[:, g], color=colors[g], s=4, alpha=0.7,
                   label=f'g={g}', zorder=g+1)
    ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.4)
    ax.set_xlabel('Integration index')
    ax.set_ylabel('Normalised aperture flux')
    ax.set_title(f'{title}  (RMS={np.std(lc_n[:, g_good])*100:.3f}%)', fontsize=9)

sm = plt.cm.ScalarMappable(cmap=cmap_g, norm=plt.Normalize(vmin=0, vmax=n_grads-1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[-1], pad=0.01)
cbar.set_label('Group index')
cbar.set_ticks(np.arange(n_grads))

fig.suptitle('Wolf-359: gradient LC by group — correction comparison',
             fontsize=10, fontweight='bold')
fig.tight_layout()
out1 = OUT / 'wolf359_nonparam_lc_by_group.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out1}')

# ---------------------------------------------------------------------------
# Figure 2: per-group median profile (star aperture)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(g_good, np.median(lc_raw_n[:, g_good], axis=0), 'o-', color='C3', lw=1.5, label='Raw')
ax.plot(g_good, np.median(lc_bfe_n[:, g_good], axis=0), 's-', color='C1', lw=1.5, label='BFE only')
ax.plot(g_good, np.median(lc_joint_n[:, g_good], axis=0), 'D-', color='C2', lw=1.5, label='Joint parametric RCD')
ax.plot(g_good, np.median(lc_hybrid_n[:, g_good], axis=0), '^-', color='C0', lw=1.5, label='Joint + median subtraction')
ax.axhline(1.0, color='k', lw=0.7, ls='--', alpha=0.4)
ax.set_xlabel('Gradient index')
ax.set_ylabel('Normalised mean aperture flux')
ax.set_title(f'Wolf-359 star aperture (r≤{AP_RADIUS}px): per-group median profile', fontsize=9)
ax.set_xticks(g_good)
ax.legend(fontsize=8)

fig.tight_layout()
out2 = OUT / 'wolf359_nonparam_profile.png'
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out2}')

# ---------------------------------------------------------------------------
# Figure 3: early / late group ratio images (raw, BFE, non-parametric)
# ---------------------------------------------------------------------------
EARLY = [1, 2, 3]
LATE = [6, 7, 8]
CUT = 30

sy, sx = STAR_Y, STAR_X

def ratio_cutout(grads):
    med = np.median(grads, axis=0)   # (n_grads, ny, nx)
    early_img = np.mean(med[EARLY], axis=0)
    late_img = np.mean(med[LATE], axis=0)
    ratio = early_img / np.where(late_img != 0, late_img, np.nan)
    return ratio[sy-CUT:sy+CUT+1, sx-CUT:sx+CUT+1]

ratio_raw = ratio_cutout(grads_raw)
ratio_bfe = ratio_cutout(grads_bfe)
ratio_joint = ratio_cutout(grads_joint)
ratio_hybrid = ratio_cutout(grads_hybrid)

ext = [-CUT-0.5, CUT+0.5, -CUT-0.5, CUT+0.5]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, ratio, title, pct in [
        (axes[0], ratio_raw, 'Raw', 99),
        (axes[1], ratio_bfe, 'BFE only', 99),
        (axes[2], ratio_joint, 'Joint parametric RCD', 99),
        (axes[3], ratio_hybrid, 'Joint + median subtraction', 95),
]:
    vdev = np.nanpercentile(np.abs(ratio - 1), pct)
    im = ax.imshow(ratio, origin='lower', extent=ext, cmap='RdBu_r',
                   vmin=1-vdev, vmax=1+vdev)
    fig.colorbar(im, ax=ax, label='Early / Late')
    ax.set_title(f'{title}\n(scale ±{vdev:.4f})', fontsize=8)
    ax.set_xlabel('Δx (px)')
    ax.set_ylabel('Δy (px)')

fig.suptitle(f'Wolf-359: early groups {EARLY} / late groups {LATE} ratio',
             fontsize=10, fontweight='bold')
fig.tight_layout()
out3 = OUT / 'wolf359_nonparam_early_late_ratio.png'
fig.savefig(out3, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out3}')

# ---------------------------------------------------------------------------
# Figure 4: per-integration early/late ratio, median over integrations
# (median of ratios, not ratio of medians — so the construction doesn't force = 1)
# ---------------------------------------------------------------------------
def per_int_ratio_cutout(grads):
    early = grads[:, EARLY, sy-CUT:sy+CUT+1, sx-CUT:sx+CUT+1].mean(axis=1)
    late = grads[:, LATE, sy-CUT:sy+CUT+1, sx-CUT:sx+CUT+1].mean(axis=1)
    return np.median(early / np.where(late != 0, late, np.nan), axis=0)

pi_ratio_raw = per_int_ratio_cutout(grads_raw)
pi_ratio_joint = per_int_ratio_cutout(grads_joint)
pi_ratio_hybrid = per_int_ratio_cutout(grads_hybrid)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, ratio, title in [
        (axes[0], pi_ratio_raw, 'Raw'),
        (axes[1], pi_ratio_joint, 'Joint parametric RCD'),
        (axes[2], pi_ratio_hybrid, 'Joint + median subtraction'),
]:
    vdev = np.nanpercentile(np.abs(ratio - 1), 99)
    im = ax.imshow(ratio, origin='lower', extent=ext, cmap='RdBu_r',
                   vmin=1-vdev, vmax=1+vdev)
    fig.colorbar(im, ax=ax, label='Early / Late')
    ax.set_title(f'{title}\n(scale ±{vdev:.4f})', fontsize=8)
    ax.set_xlabel('Δx (px)')
    ax.set_ylabel('Δy (px)')

fig.suptitle(f'Wolf-359: median over integrations of per-integration early/late ratio',
             fontsize=10, fontweight='bold')
fig.tight_layout()
out4 = OUT / 'wolf359_hybrid_ratio_zoom.png'
fig.savefig(out4, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out4}')

# ---------------------------------------------------------------------------
# Figure 5: background pixel lightcurve at each correction stage
# ---------------------------------------------------------------------------
# Pick a background pixel: first pixel in bg_mask at r~35 from star
bg_ys, bg_xs = np.where(bg_mask)
mid = len(bg_ys) // 2
py, px = int(bg_ys[mid]), int(bg_xs[mid])
print(f'Background pixel: ({py}, {px}), r={np.sqrt((py-STAR_Y)**2+(px-STAR_X)**2):.1f} px')

fig, axes = plt.subplots(1, 4, figsize=(22, 4))
for ax, grads, title in [
        (axes[0], grads_raw, 'Raw'),
        (axes[1], grads_bfe, 'BFE only'),
        (axes[2], grads_joint, 'Joint parametric RCD'),
        (axes[3], grads_hybrid, 'Joint + median subtraction'),
]:
    for g in g_good:
        ax.scatter(integ, grads[:, g, py, px], color=colors[g], s=4, alpha=0.7, zorder=g+1)
    ax.set_xlabel('Integration index')
    ax.set_ylabel('Gradient (DN/group)')
    ax.set_title(f'{title}', fontsize=9)

sm = plt.cm.ScalarMappable(cmap=cmap_g, norm=plt.Normalize(vmin=0, vmax=n_grads-1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[-1], pad=0.01)
cbar.set_label('Group index')
cbar.set_ticks(np.arange(n_grads))

fig.suptitle(f'Background pixel ({py},{px}), r={np.sqrt((py-STAR_Y)**2+(px-STAR_X)**2):.1f} px from star',
             fontsize=10, fontweight='bold')
fig.tight_layout()
out5 = OUT / 'wolf359_bg_pixel_lc.png'
fig.savefig(out5, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out5}')
