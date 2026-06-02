import numpy as np
from pathlib import Path
from scipy.interpolate import griddata
from scipy.optimize import curve_fit


def build_correction_map(cube, mask=None):
    """
    Derive the per-pixel group correction map from a reference ramp cube.
    Use a quiescent observation (no flares/transients) to build this.

    Parameters
    ----------
    cube : ndarray, shape (n_int, n_groups, ny, nx)
        Stage-1 corrected ramp cube (raw group values).
    mask : ndarray, shape (ny, nx), bool, optional
        True = pixels to interpolate over (bad pixels, saturated core, etc.)

    Returns
    -------
    C_map : ndarray, shape (n_groups-1, ny, nx)
        Multiplicative correction factor per group per pixel.
        C_map[0] = 1 everywhere (group 0 is the reference).
    """
    grads = np.diff(cube, axis=1).astype(float)
    med_grad = np.median(grads, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        C_map = np.where(med_grad != 0, med_grad[-2:-1] / med_grad, np.nan)

    if mask is not None:
        ny, nx = mask.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        good = ~mask & np.isfinite(C_map[0])
        good_yx = np.column_stack([yy[good], xx[good]])

        for g in range(C_map.shape[0]):
            needs_fill = mask | ~np.isfinite(C_map[g])
            if not needs_fill.any():
                continue
            fill_yx = np.column_stack([yy[needs_fill], xx[needs_fill]])
            good_vals = C_map[g][good]
            filled = griddata(good_yx, good_vals, fill_yx, method='linear')
            still_nan = ~np.isfinite(filled)
            if still_nan.any():
                filled[still_nan] = griddata(
                    good_yx, good_vals, fill_yx[still_nan], method='nearest'
                )
            C_map[g][needs_fill] = filled

    return C_map


def correct_reset_decay(cube, method='median', mask=None, mask_dilation=0,
                        edge_margin=10, dq=None, sat_bit=2,
                        diagnostics=False, save_path=None):
    """
    Correct charge reset decay in MIRI ramp data.

    tau is fitted globally from the spatial mean gradient profile and is the
    same for all pixels. The last group-to-group gradient is always excluded
    (last-frame anomaly).

    Three methods:

    'median' (default)
        Fits C + A*exp(-g/tau) to the per-pixel median gradient profile.
        A and C are per-pixel via linear regression with tau fixed.
        A is constant across integrations.

    'per_int'
        Fits [C, A, delta] independently for each integration and each pixel
        using linear regression with tau fixed. Removes residual offsets caused
        by integration-to-integration variation in A (e.g. from charge-dependent
        decay amplitude). Noisier than 'median' for individual pixels but
        produces unbiased aperture-summed lightcurves.

    'stretched_exp'
        Fits per-pixel A from the median gradient profile (same first step),
        then fits A(Q) = scale * exp(beta * Q^c) across pixels. For each ramp,
        A is evaluated from the charge Q at the last good group, giving a
        per-integration per-pixel amplitude while tau remains global.

    Parameters
    ----------
    cube : ndarray (n_int, n_groups, ny, nx), float
        Raw SCI data from uncal.fits.
    method : {'median', 'per_int', 'stretched_exp'}
    mask : ndarray (ny, nx) bool, optional
        True = non-science pixel. Masked pixels are excluded from the tau
        spatial mean fit and the A(Q) fit. Does not affect per-pixel A fitting
        or the correction itself.
    mask_dilation : int
        Dilate the mask by this many pixels (circular) before applying to
        fitting statistics. Excludes pixels near masked regions.
    edge_margin : int
        Border pixels excluded from the A(Q) fit in 'stretched_exp'.
    dq : ndarray (n_int, n_groups, ny, nx) uint8, optional
        GROUPDQ array. Used in 'stretched_exp' to find the last unsaturated
        group per ramp for Q estimation.
    sat_bit : int
        GROUPDQ bit value for SATURATED (default 2).
    diagnostics : bool
        If True, produce diagnostic figures.
    save_path : str or Path, optional
        File path to save the diagnostic figure. Only used when diagnostics=True.

    Returns
    -------
    cube_cor : ndarray (n_int, n_groups, ny, nx)
        Corrected SCI cube. Groups 1 through n_groups-2 have the cumulative
        decay subtracted; group 0 is corrected for the first-frame offset.
    """
    cube = np.asarray(cube, dtype=float)
    n_int, n_groups, ny, nx = cube.shape
    n_grads = n_groups - 2  # drop last gradient (last-frame anomaly)

    grads = np.diff(cube, axis=1)[:, :n_grads]        # (n_int, n_grads, ny, nx)
    med_grad = np.median(grads, axis=0)                # (n_grads, ny, nx)
    g_arr = np.arange(n_grads, dtype=float)

    if mask is not None and mask_dilation > 0:
        from scipy.ndimage import binary_dilation
        r = mask_dilation
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        struct = (yy**2 + xx**2) <= r**2
        mask = binary_dilation(mask, structure=struct)
    sci = ~mask if mask is not None else np.ones((ny, nx), dtype=bool)

    # Global tau from spatial mean over science pixels, excluding gradient 0.
    # Gradient 0 is suppressed by the first-frame anomaly (group 0 has extra
    # reset charge), which breaks the monotonic-decay assumption at g=0.
    mean_profile = np.nanmean(med_grad[:, sci], axis=1)
    mean_profile_fit = mean_profile[1:]
    def _exp_model(g, C, A, t):
        return C + A * np.exp(-g / t)
    popt, _ = curve_fit(_exp_model, g_arr[1:], mean_profile_fit,
                        p0=[mean_profile_fit[-1],
                            mean_profile_fit[0] - mean_profile_fit[-1],
                            1.5])
    tau = float(popt[2])

    # Per-pixel fit: [C, A, delta] where delta is the first-frame offset.
    # The design matrix has a -1 in the delta column only for g=0, accounting
    # for the suppression of gradient 0 by the first-frame anomaly.
    exp_g = np.exp(-g_arr / tau)                       # (n_grads,)
    ff_col = np.zeros(n_grads); ff_col[0] = -1.0
    X = np.column_stack([np.ones(n_grads), exp_g, ff_col])  # (n_grads, 3)
    params, _, _, _ = np.linalg.lstsq(
        X, med_grad.reshape(n_grads, -1), rcond=None)
    A_map = params[1].reshape(ny, nx)                  # (ny, nx)
    delta_map = params[2].reshape(ny, nx)              # first-frame offset (ny, nx)

    if method == 'median':
        decay_cumsum = np.cumsum(A_map * exp_g[:, None, None], axis=0)  # (n_grads, ny, nx)
        cube_cor = cube.copy()
        cube_cor[:, 1:n_grads + 1] -= decay_cumsum[None]
        cube_cor[:, 0] -= delta_map[None]

    elif method == 'per_int':
        # Fit [C_i, A_i, delta_i] independently per integration per pixel.
        # tau is still global. This removes residual offsets from integration-
        # to-integration variation in A (charge-dependent decay amplitude).
        grads_flat = grads.reshape(n_int, n_grads, -1)    # (n_int, n_grads, ny*nx)
        A_int = np.empty((n_int, ny * nx))
        delta_int = np.empty((n_int, ny * nx))
        for i in range(n_int):
            p, _, _, _ = np.linalg.lstsq(X, grads_flat[i], rcond=None)
            A_int[i] = p[1]
            delta_int[i] = p[2]
        A_int = A_int.reshape(n_int, ny, nx)
        delta_int = delta_int.reshape(n_int, ny, nx)

        decay_cumsum = np.cumsum(
            A_int[:, None, :, :] * exp_g[None, :, None, None], axis=1)
        cube_cor = cube.copy()
        cube_cor[:, 1:n_grads + 1] -= decay_cumsum
        cube_cor[:, 0] -= delta_int

    else:
        # --- method == 'stretched_exp' ---
        edge_mask = np.zeros((ny, nx), dtype=bool)
        edge_mask[:edge_margin] = True
        edge_mask[-edge_margin:] = True
        edge_mask[:, :edge_margin] = True
        edge_mask[:, -edge_margin:] = True

        Q_med = np.median(cube[:, n_grads, :, :], axis=0)  # (ny, nx)
        fit_mask = ~edge_mask & sci & np.isfinite(A_map) & (Q_med > 0)

        def _stretched(Q, scale, beta, c):
            return scale * np.exp(beta * Q**c)
        Q_fit, A_fit = Q_med[fit_mask], A_map[fit_mask]
        popt_s, _ = curve_fit(_stretched, Q_fit, A_fit,
                              p0=[np.percentile(A_fit, 10), 1e-3, 0.6],
                              maxfev=50000)
        scale, beta, c = popt_s

        # Per-ramp Q from last unsaturated group
        if dq is not None:
            bad = (dq[:, :n_grads + 1] & sat_bit) > 0
            not_bad_rev = ~bad[:, ::-1]
            last_rev = np.argmax(not_bad_rev, axis=1)
            last_good = np.clip(n_grads - last_rev, 0, n_grads)
            ii = np.arange(n_int)[:, None, None]
            yy = np.arange(ny)[None, :, None]
            xx = np.arange(nx)[None, None, :]
            Q_int = cube[ii, last_good, yy, xx]        # (n_int, ny, nx)
        else:
            Q_int = cube[:, n_grads, :, :]

        Q_int = np.clip(Q_int, 1.0, None)
        A_int = scale * np.exp(beta * Q_int**c)        # (n_int, ny, nx)

        decay_cumsum = np.cumsum(
            A_int[:, None, :, :] * exp_g[None, :, None, None], axis=1)
        cube_cor = cube.copy()
        cube_cor[:, 1:n_grads + 1] -= decay_cumsum
        cube_cor[:, 0] -= delta_map[None]

    if diagnostics:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_panels = 3 if method == 'stretched_exp' else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))

        ax = axes[0]
        g_fine = np.linspace(1, n_grads - 1, 200)
        C_fit, A_fit_mean = float(popt[0]), float(popt[1])
        ax.plot(g_arr[0], mean_profile[0], 'o', color='gray', ms=5, label='g=0 (excluded from fit)')
        ax.plot(g_arr[1:], mean_profile[1:], 'o', color='k', ms=5, label='Spatial mean')
        ax.plot(g_fine, C_fit + A_fit_mean * np.exp(-g_fine / tau),
                '--', color='C3', lw=1.5, label=f'Fit  τ={tau:.2f} grp')
        ax.set_xlabel('Gradient index')
        ax.set_ylabel('Mean gradient (DN/group)')
        ax.set_title('Global τ fit')
        ax.legend(fontsize=8)
        ax.set_xticks(g_arr.astype(int))

        ax = axes[1]
        vmax = np.nanpercentile(A_map, 99)
        im = ax.imshow(A_map, origin='lower', vmin=0, vmax=vmax, cmap='viridis')
        fig.colorbar(im, ax=ax, label='DN/group')
        ax.set_title('Decay amplitude A')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        if method == 'stretched_exp':
            ax = axes[2]
            ax.scatter(Q_med[fit_mask], A_map[fit_mask], s=1, alpha=0.1,
                       color='C0', rasterized=True)
            q_line = np.linspace(np.nanpercentile(Q_med[fit_mask], 1),
                                 np.nanpercentile(Q_med[fit_mask], 99), 300)
            ax.plot(q_line, scale * np.exp(beta * q_line**c), '-', color='C3',
                    lw=1.5, label=f'scale={scale:.2f}, β={beta:.3e}, c={c:.3f}')
            ax.set_xlabel('Q at last group (DN)')
            ax.set_ylabel('A (DN/group)')
            ax.set_title('A(Q) stretched exponential fit')
            ax.legend(fontsize=8)

        fig.suptitle(f'Reset decay correction diagnostics  (method={method})',
                     fontsize=11, fontweight='bold')
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(Path(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)

    return cube_cor


def correct_bfe_rcd(cube, A_bfe=1.035e-6, alpha_bfe=2.783,
                    bg_mask=None, late_groups=None, verbose=False):
    """
    Joint BFE + reset-decay correction for MIRI ramp data.

    Three sequential steps applied to gradients:
      1. Causal BFE inversion: each gradient is divided by (1 - A_bfe * K⊛Q)
         where Q is the accumulated charge from all previous groups.
      2. Parametric RCD subtraction: fit C + A*exp(-g/tau) with tau global
         (from background pixels) and [A, C, delta] per pixel via lstsq.
         Subtract the fitted decay from every integration.
      3. Non-parametric residual removal: subtract the per-pixel per-group
         median over integrations, then add back the flat rate estimated from
         late groups. Removes any residual group-correlated structure not
         captured by the exponential model.

    The last gradient (last-frame anomaly) is BFE-corrected but excluded from
    the RCD and median subtraction steps, matching the convention in
    correct_reset_decay.

    Parameters
    ----------
    cube : ndarray (n_int, n_groups, ny, nx), float
        Raw SCI data from uncal.fits.
    A_bfe : float
        BFE kernel amplitude (default 1.035e-6).
    alpha_bfe : float
        BFE kernel power-law index (default 2.783).
    bg_mask : ndarray (ny, nx) bool, optional
        True = background pixels used to fit the global RCD timescale tau.
        If None, all pixels are used.
    late_groups : list of int, optional
        Gradient indices used to estimate the flat rate for median subtraction.
        Defaults to the last three good gradients.
    verbose : bool
        Print BFE inversion progress.

    Returns
    -------
    cube_cor : ndarray (n_int, n_groups, ny, nx)
        Corrected SCI cube reconstructed from corrected gradients.
        Group 0 is unchanged (reset level reference).
    """
    from scipy.signal import fftconvolve

    cube = np.asarray(cube, dtype=float)
    n_int, n_groups, ny, nx = cube.shape
    n_grads_all = n_groups - 1        # all gradients
    n_grads = n_groups - 2            # gradients to correct (exclude last-frame anomaly)

    grads_raw = np.diff(cube, axis=1)   # (n_int, n_grads_all, ny, nx)
    g_arr = np.arange(n_grads, dtype=float)

    if late_groups is None:
        late_groups = list(range(n_grads - 3, n_grads))

    # Step 1: causal BFE inversion over all gradients
    kh = 20
    ii, jj = np.mgrid[-kh:kh+1, -kh:kh+1].astype(float)
    r = np.sqrt(ii**2 + jj**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        K = np.where(r > 0, -1.0 / r**alpha_bfe, 0.0)
    K[kh, kh] = -K.sum()

    grads_bfe = grads_raw.copy()
    Q_med = np.zeros((ny, nx))
    for g in range(n_grads_all):
        if g > 0:
            Q_med = Q_med + np.median(grads_bfe[:, g-1], axis=0)
        KQ = fftconvolve(Q_med, K, mode='same')
        factor = np.where(1.0 - A_bfe * KQ > 0.05, 1.0 - A_bfe * KQ, 1.0)
        grads_bfe[:, g] = grads_raw[:, g] / factor[None]
        if verbose:
            print(f'  BFE g={g}', end='\r')
    if verbose:
        print()

    # Step 2: fit global tau from BFE-corrected background, excluding g=0
    med_bfe = np.median(grads_bfe[:, :n_grads], axis=0)   # (n_grads, ny, nx)
    g_fit = g_arr[1:]
    if bg_mask is not None:
        mean_bg = np.nanmean(med_bfe[1:, bg_mask], axis=1)
    else:
        mean_bg = np.nanmean(med_bfe[1:].reshape(n_grads-1, -1), axis=1)

    def _exp1(g, C, A, tau): return C + A * np.exp(-g / tau)
    popt, _ = curve_fit(_exp1, g_fit, mean_bg,
                        p0=[mean_bg[-1], mean_bg[0] - mean_bg[-1], 1.5])
    tau = float(popt[2])

    exp_g = np.exp(-g_arr / tau)
    ff_col = np.zeros(n_grads); ff_col[0] = -1.0
    X = np.column_stack([np.ones(n_grads), exp_g, ff_col])
    params, _, _, _ = np.linalg.lstsq(
        X, med_bfe.reshape(n_grads, -1), rcond=None)
    Adec_map = params[1].reshape(ny, nx)
    delta_map = params[2].reshape(ny, nx)

    grads_joint = grads_bfe.copy()
    for g in range(n_grads):
        decay_g = Adec_map * np.exp(-g / tau)
        if g == 0:
            grads_joint[:, 0] = grads_bfe[:, 0] - decay_g[None] + delta_map[None]
        else:
            grads_joint[:, g] = grads_bfe[:, g] - decay_g[None]

    # Step 3: non-parametric median subtraction
    med_joint = np.median(grads_joint[:, :n_grads], axis=0)   # (n_grads, ny, nx)
    C_hat = np.mean(med_joint[late_groups], axis=0)            # (ny, nx)

    grads_cor = grads_joint.copy()
    for g in range(n_grads):
        grads_cor[:, g] = grads_joint[:, g] - med_joint[g][None] + C_hat[None]

    # Reconstruct corrected cube: group 0 unchanged, integrate corrected gradients
    cube_cor = cube.copy()
    cube_cor[:, 1:] = cube[:, :1] + np.cumsum(grads_cor, axis=1)

    return cube_cor


def correct_ramp(cube, C_map):
    """
    Apply the per-pixel correction to a ramp cube, returning corrected
    group-gradients ready for difference imaging.

    Parameters
    ----------
    cube  : ndarray, shape (n_int, n_groups, ny, nx)
        Stage-1 corrected ramp cube (raw group values).
    C_map : ndarray, shape (n_groups-1, ny, nx)
        Correction map from build_correction_map.

    Returns
    -------
    grads_corrected : ndarray, shape (n_int, n_groups-1, ny, nx)
        All group-gradients are on a common photometric scale.
        Subtract any two frames directly for difference imaging.
    """
    grads = np.diff(cube, axis=1).astype(float)
    return grads * C_map[None]
