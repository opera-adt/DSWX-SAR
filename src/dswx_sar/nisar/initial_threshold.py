import copy
import logging
import mimetypes
import os
import time

import numpy as np
from joblib import Parallel, delayed
from osgeo import gdal
from scipy import interpolate, ndimage
from scipy.interpolate import Rbf
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import norm
from skimage.filters import threshold_multiotsu, threshold_otsu

from dswx_sar.common import (
    _dswx_sar_util,
    _generate_log,
    _refine_with_bimodality,
    _region_growing)
from dswx_sar.nisar.dswx_ni_runconfig import (
    DSWX_NI_POL_DICT,
    _get_parser)
from dswx_sar.common import (_generate_log,
                             _initial_threshold)


logger = logging.getLogger('dswx_sar')


def convert_db2pow(db):
    """Convert decibels to power (linear)"""
    return 10 ** (db / 10.0)


def maybe_pow2db(x, scale):
    return 10 * np.log10(x) if scale == 'db' else x



def _solve_gaussian_intersections(m1, s1, A1, m2, s2, A2):
    if s1 <= 0 or s2 <= 0 or A1 <= 0 or A2 <= 0:
        return []
    inv2s1 = 1.0 / (2.0 * s1 * s1)
    inv2s2 = 1.0 / (2.0 * s2 * s2)
    lnA = np.log(A2 / A1)
    a = inv2s1 - inv2s2
    b = -2.0 * (m1 * inv2s1 - m2 * inv2s2)
    c = (m1 * m1 * inv2s1 - m2 * m2 * inv2s2) - lnA
    if np.isclose(a, 0.0):
        if np.isclose(b, 0.0): return []
        return [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < 0: return []
    sqrt_disc = np.sqrt(disc)
    return sorted([(-b - sqrt_disc) / (2.0 * a),
                   (-b + sqrt_disc) / (2.0 * a)])

def _threshold_from_bimodal_fit(first_mode, second_mode, bounds=None, p_low=0.98, p_high=0.02):
    m1, s1, A1 = first_mode
    m2, s2, A2 = second_mode
    if m1 > m2:
        (m1, s1, A1), (m2, s2, A2) = (m2, s2, A2), (m1, s1, A1)
    roots = _solve_gaussian_intersections(m1, s1, A1, m2, s2, A2)
    intersection = None
    for x in roots:
        if m1 <= x <= m2:
            intersection = x
            break
    if intersection is None:
        x_low  = norm.ppf(p_low,     m1, s1)      # near right tail of low mode
        x_high = norm.ppf(1-p_high,  m2, s2)      # near left tail of high mode
        cand = 0.5 * (np.clip(x_low, m1, m2) + np.clip(x_high, m1, m2))
        intersection = np.clip(cand, m1, m2)
    if bounds is not None:
        intersection = float(np.clip(intersection, bounds[0], bounds[1]))
    return intersection, m1


def _solve_gaussian_intersections_(m1, s1, A1, m2, s2, A2):
    # Returns sorted real roots where A1*N(m1,s1) == A2*N(m2,s2)
    if s1 <= 0 or s2 <= 0 or A1 <= 0 or A2 <= 0:
        return []
    inv2s1 = 1.0 / (2.0 * s1 * s1)
    inv2s2 = 1.0 / (2.0 * s2 * s2)
    lnA = np.log(A2 / A1)
    a = inv2s1 - inv2s2
    b = -2.0 * (m1 * inv2s1 - m2 * inv2s2)
    c = (m1 * m1 * inv2s1 - m2 * m2 * inv2s2) - lnA
    if np.isclose(a, 0.0):
        if np.isclose(b, 0.0):
            return []
        return [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return []
    sqrt_disc = np.sqrt(disc)
    return sorted([(-b - sqrt_disc) / (2.0 * a),
                   (-b + sqrt_disc) / (2.0 * a)])


def _tile_relaxed_threshold_from_boundary(
        intensity_tile_db,  # 2D (dB)
        tau_strict,         # scalar (dB)
        max_band_px=3,
        core_iter=1):
    """
    Data-driven relaxed threshold for a single tile:
      - Create strict mask at tau_strict
      - Build confident cores (erode water & land)
      - Fit Gaussians to cores -> intersection t_relaxed
      - Check boundary band looks like water; if yes, allow a limited lift
    Returns updated tau (>= tau_strict), or tau_strict if not enough evidence.
    """
    from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

    strict = intensity_tile_db < tau_strict
    if np.sum(strict) == 0 or np.sum(~strict) == 0:
        return tau_strict

    # Cores
    w_core = binary_erosion(strict, iterations=core_iter)
    l_core = binary_erosion(~strict, iterations=core_iter)
    if np.sum(w_core) < 10 or np.sum(l_core) < 10:
        return tau_strict

    w_vals = intensity_tile_db[w_core]; w_vals = w_vals[np.isfinite(w_vals)]
    l_vals = intensity_tile_db[l_core]; l_vals = l_vals[np.isfinite(l_vals)]
    if w_vals.size < 20 or l_vals.size < 20:
        return tau_strict

    def _robust_std(x):
        x = x[np.isfinite(x)]
        if x.size < 2: return 1e-3
        mad = np.median(np.abs(x - np.median(x)))
        return max(np.std(x), 1.4826 * mad, 1e-3)

    mu_w = float(np.median(w_vals)); sd_w = float(_robust_std(w_vals))
    mu_l = float(np.median(l_vals)); sd_l = float(_robust_std(l_vals))
    Aw   = max(len(w_vals), 1);       Al   = max(len(l_vals), 1)

    roots = _solve_gaussian_intersections_(mu_w, sd_w, Aw, mu_l, sd_l, Al)
    if mu_w > mu_l:
        mu_w, mu_l = mu_l, mu_w
    if roots:
        between = [r for r in roots if mu_w <= r <= mu_l]
        t_relaxed = between[0] if between else 0.5*(mu_w + mu_l)
    else:
        t_relaxed = 0.5*(mu_w + mu_l)

    # Boundary band (thin shell around current boundary)
    band = binary_dilation(strict, iterations=max_band_px) & \
           binary_dilation(~strict, iterations=max_band_px)
    b_vals = intensity_tile_db[band]; b_vals = b_vals[np.isfinite(b_vals)]
    if b_vals.size < 50:
        return tau_strict

    # If boundary looks more like water than land, allow a lift.
    b_mean = float(np.mean(b_vals))
    looks_water = abs(b_mean - mu_w) < abs(b_mean - mu_l)
    if not looks_water:
        return tau_strict

    # Cap lift to a fraction of separation for stability
    sep = max(0.0, (mu_l - mu_w))
    cap = 0.5 * sep  # <= half the class gap
    tau_new = np.clip(t_relaxed, tau_strict, tau_strict + cap)
    return float(tau_new)


def determine_threshold(
        intensity,
        candidate_tile_coords,
        min_intensity_histogram=-32,
        max_intensity_histogram=0,
        step_histogram=0.1,
        bounds=None,
        method='ki',
        mutli_threshold=True,
        adjust_if_nonoverlap=True,
        adjust_thresh_low_dist_percent=None,
        adjust_thresh_high_dist_percent=None,
        ):
    """Compute the thresholds and peak values for left Gaussian
    from intensity image for given candidate coordinates.
    The three methods are supported:
        1) Kittler and Illingworth’s algorithm
        2) Otsu thresholding algorithm
        3) Region-growing based thresholding algorithm

    Parameters
    ----------
    intensity : numpy.ndarray
        intensity raster in decibel scale
    candidate_tile_coords : numpy.ndarray
            The x and y coordinates that pass the tile selection test.
            Each row has index, y_start, y_end, x_start, and x_end.
    winsize : float
        size of searching window used for tile selection
    min_intensity_histogram : float
        minimum decibel value for histogram.
        If min_intensity_histogram == -1000,
        min_intensity_histogram will be calculated directly from image.
    max_intensity_histogram : float
        maximum decibel value for histogram.
        If max_intensity_histogram == -1000,
        max_intensity_histogram will be calculated directly from image.
    step_histogram : float
        step value for histogram
    bounds : list
        bounds for the threshold
    method: str
        Thresholding algorithm ('ki', 'otsu', 'rg')
    multi_threshold : bool
        Flag indicating whether tri-mode Gaussian distribution
        is assumed or not.
    adjust_if_nonoverlap : bool
        Flag enabling the adjustment of the threshold
        If True, the threshold goes up to the point where the lower
        distribution ends
    adjust_thresh_low_dist_percent: float
        Percentile threshold for the lower distribution. When
        'adjust_if_nonoverlap' is enabled, this parameter defines the
        threshold as the value at the specified percentile of the lower
        distribution. This adjustment ensures that the threshold aligns
        with the desired percentile position within the distribution.
    adjust_thresh_high_dist_percent: float
        Percentile threshold for the higher distribution.
        Percentile threshold for the higher distribution. When
        'adjust_if_nonoverlap' is enabled, this parameter defines the
        threshold as the value at the specified percentile of the higher
        distribution. This adjustment ensures that the threshold aligns
        with the desired percentile position within the distribution.

    Returns
    -------
    global_threshold : float
        thresholds calculated from KI algorithm
    glob_mode_thres : float
        mode value of gaussian distribution of water body
    """
    TILE_RELAX = True   # set True to enable tile-wise relaxed threshold

    if bounds is None:
        bounds = [-20, -13]
    if max_intensity_histogram == -1000:
        max_intensity_histogram = np.nanpercentile(intensity, 90)
    if min_intensity_histogram == -1000:
        min_intensity_histogram = np.nanpercentile(intensity, 10)
    numstep = int((max_intensity_histogram - min_intensity_histogram) /
                  step_histogram)
    if numstep < 100:
        step_histogram0 = step_histogram
        step_histogram = ((max_intensity_histogram - min_intensity_histogram) /
                          1000)
        logger.info(f'Histogram bin step changes from {step_histogram0} '
                    f'to {step_histogram} in threshold computation.')
    USE_MODEL_INTERSECTION = True
    threshold_array = []
    threshold_idx_array = []
    mode_array = []
    negligible_value = _dswx_sar_util.Constants.negligible_value
    min_threshold, max_threshold = bounds[0], bounds[1]

    for coord in candidate_tile_coords:

        # assume that coord consists of 5 elements
        ystart, yend, xstart, xend = coord[1:]

        intensity_sub = intensity[ystart:yend,
                                  xstart:xend]

        # generate histogram with intensity higher than -35 dB
        intensity_sub = intensity_sub[intensity_sub > -35]
        intensity_sub = _initial_threshold.remove_invalid(intensity_sub)

        if intensity_sub.size == 0:
            return np.nan, np.nan

        if (not np.isfinite(min_intensity_histogram)) or (not np.isfinite(max_intensity_histogram)) \
        or (max_intensity_histogram <= min_intensity_histogram):
            # fallback to local (subtile) percentiles
            min_intensity_histogram = np.nanpercentile(intensity_sub, 5)
            max_intensity_histogram = np.nanpercentile(intensity_sub, 95)
            if not np.isfinite(min_intensity_histogram) or not np.isfinite(max_intensity_histogram) \
            or max_intensity_histogram <= min_intensity_histogram:
                # final epsilon expansion
                min_intensity_histogram = np.nanmin(intensity_sub)
                max_intensity_histogram = min_intensity_histogram + 1e-6

        # Ensure at least 2 edges
        bins = np.linspace(min_intensity_histogram,
                        max_intensity_histogram,
                        max(2, numstep + 1))

        intensity_counts, bins = np.histogram(intensity_sub, bins=bins, density=True)

        # If density=True produced NaNs (zero bin width / empty), retry with density=False
        if not np.isfinite(intensity_counts).any():
            intensity_counts, bins = np.histogram(intensity_sub, bins=bins, density=False)

        # Sanitize any remaining non-finites
        intensity_counts = np.nan_to_num(intensity_counts, nan=0.0, posinf=0.0, neginf=0.0)
        intensity_bins = bins[:-1]

        if method == 'ki':
            threshold, idx_threshold = _initial_threshold.compute_ki_threshold(
                intensity_sub,
                min_intensity_histogram,
                max_intensity_histogram,
                step_histogram)

        elif method in ['otsu', 'rg']:
            threshold = threshold_otsu(intensity_sub)

        # get index of threshold from histogram.
        idx_threshold = np.searchsorted(intensity_bins, threshold)
        idx_threshold = int(np.clip(idx_threshold, 0, max(0, len(intensity_bins) - 1)))

        # if estimated threshold is higher than bounds,
        # re-estimate threshold assuming tri-mode distribution
        if threshold > bounds[1] and mutli_threshold:
            try:
                thresholds = threshold_multiotsu(intensity_sub)

                if thresholds[0] < threshold:
                    threshold = thresholds[0]
                    idx_threshold = np.searchsorted(intensity_bins, threshold)
            except ValueError:
                logger.info('Unable to find multi threshold')

        # Make sure idx_threshold is within bounds
        idx_threshold = int(np.clip(idx_threshold, 0, max(0, len(intensity_bins) - 1)))

        # Low-side slice (<= threshold)
        low_slice = intensity_counts[:idx_threshold + 1]
        low_slice = np.nan_to_num(low_slice, nan=0.0)

        if low_slice.size == 0 or not np.isfinite(low_slice).any() or np.all(low_slice == 0):
            lowmaxind = max(0, idx_threshold)   # safe fallback
        else:
            lowmaxind_cands, _ = find_peaks(low_slice, distance=5)
            if lowmaxind_cands.size == 0:
                lowmaxind = int(np.argmax(low_slice))
            else:
                lowmaxind = int(lowmaxind_cands[np.argmax(low_slice[lowmaxind_cands])])

        # High-side slice (>= threshold)
        high_slice = intensity_counts[idx_threshold:]
        high_slice = np.nan_to_num(high_slice, nan=0.0)

        if high_slice.size == 0 or not np.isfinite(high_slice).any() or np.all(high_slice == 0):
            highmaxind = idx_threshold
        else:
            highmaxind_cands, _ = find_peaks(high_slice, distance=5)
            if highmaxind_cands.size == 0:
                highmaxind_rel = int(np.argmax(high_slice))
            else:
                highmaxind_rel = int(highmaxind_cands[np.argmax(high_slice[highmaxind_cands])])
            highmaxind = idx_threshold + highmaxind_rel

        # Clamp indices
        lowmaxind  = int(np.clip(lowmaxind,  0, len(intensity_bins) - 1))
        highmaxind = int(np.clip(highmaxind, 0, len(intensity_bins) - 1))

        # mode values
        tau_mode_left  = intensity_bins[lowmaxind]
        tau_mode_right = intensity_bins[highmaxind]

        # mode values
        tau_mode_left = intensity_bins[lowmaxind]
        tau_mode_right = intensity_bins[highmaxind]

        tau_amp_left = intensity_counts[lowmaxind]
        tau_amp_right = intensity_counts[highmaxind]

        try:
            expected = (tau_mode_left, .5, tau_amp_left,
                        tau_mode_right, .5, tau_amp_right)

            params, _ = curve_fit(
                _initial_threshold.bimodal,
                intensity_bins,
                intensity_counts,
                expected,
                bounds=((-30, 0.001, 0.01,
                        -30, 0.001, 0.01),
                        (5, 5, 0.95,
                        5, 5, 0.95)))
            if params[0] > params[3]:
                second_mode = params[:3]
                first_mode = params[3:]
            else:
                first_mode = params[:3]
                second_mode = params[3:]

            if USE_MODEL_INTERSECTION:
                try:
                    threshold_model, mode_lower = _threshold_from_bimodal_fit(
                        first_mode, second_mode, bounds=bounds, p_low=0.98, p_high=0.02
                    )
                    threshold = threshold_model
                    modevalue = mode_lower
                    idx_threshold = np.searchsorted(intensity_bins, threshold)
                    idx_threshold = int(np.clip(idx_threshold, 0, max(0, len(intensity_bins) - 1)))
                except Exception as e:
                    logger.info(f'Model-intersection override skipped (error: {e}); keeping KI/Otsu.')

            lock_mode_from_model = USE_MODEL_INTERSECTION

            simul_first = _initial_threshold.gauss(intensity_bins, *first_mode)
            simul_second = _initial_threshold.gauss(intensity_bins, *second_mode)
            simul_second_sum = np.nansum(simul_second)
            if simul_second_sum == 0:
                simul_second_sum = negligible_value
            converge_ind = np.where((intensity_bins < tau_mode_right)
                                    & (intensity_bins > tau_mode_left)
                                    & (intensity_bins < threshold)
                                    & (np.cumsum(simul_second) /
                                       simul_second_sum < 0.03))
            if not lock_mode_from_model:
                if len(converge_ind[0]):
                    modevalue = intensity_bins[converge_ind[0][-1]]
                else:
                    modevalue = tau_mode_left
            # if len(converge_ind[0]):
            #     modevalue = intensity_bins[converge_ind[0][-1]]
            # else:
            #     modevalue = tau_mode_left

            optimization = True

        except:
            optimization = False
            logger.info(
                'Bimodal curve Fitting fails in threshold computation.')
            modevalue = tau_mode_left
        try:
            dividers = threshold_multiotsu(intensity_sub)

            expected = (dividers[0], .5, tau_amp_left,
                        dividers[1], .5, tau_amp_right,
                        (dividers[0]+dividers[1])/2, .5, 0.1)
            # curve_fit fits the trimodal distributions
            # All distributions are assumed to be in the bound
            # -35 to 5 dB, with standard deviation of 0 - 10[dB]
            # and amplitudes of 0.01 to 0.95.
            params, _ = curve_fit(_initial_threshold.trimodal,
                                  intensity_bins,
                                  intensity_counts,
                                  expected,
                                  bounds=((-35, 0.001, 0.01,
                                           -35, 0.001, 0.01,
                                           -35, 0.001, 0.01),
                                          (5, 10, 0.95,
                                           5, 10, 0.95,
                                           5, 10, 0.95)))

            # re-sort the order of estimated modes using amplitudes
            first_setind = 0
            second_setind = 3
            third_setind = 6

            if params[first_setind] > params[second_setind]:
                first_setind, second_setind = second_setind, first_setind
            if params[second_setind] > params[third_setind]:
                second_setind, third_setind = third_setind, second_setind
            if params[first_setind] > params[second_setind]:
                first_setind, second_setind = second_setind, first_setind

            tri_first_mode = params[first_setind:first_setind+3]
            tri_second_mode = params[second_setind:second_setind+3]
            tri_third_mode = params[third_setind:third_setind+3]

            simul_second_sum = np.sum(simul_second)
            if simul_second_sum == 0:
                simul_second_sum = negligible_value
            converge_ind = np.where((intensity_bins < tau_mode_right)
                                    & (intensity_bins > tau_mode_left)
                                    & (intensity_bins < threshold)
                                    & (np.cumsum(simul_second) /
                                       simul_second_sum < 0.03))

            if len(converge_ind[0]):
                modevalue = intensity_bins[converge_ind[0][-1]]

            else:
                modevalue = tau_mode_left

            large_amp = np.max([tri_first_mode[2],
                                tri_second_mode[2],
                                tri_third_mode[2]])

            tri_ratio_bool = tri_first_mode[2] / large_amp > 0.08 and \
                tri_second_mode[2] / large_amp > 0.08 and \
                tri_third_mode[2] / large_amp > 0.08

            if (np.abs(tri_first_mode[0] - tri_second_mode[0]) > 1) or \
               (np.abs(tri_first_mode[1] - tri_second_mode[1]) > 1.5):
                first_second_dist_bool = True
            else:
                first_second_dist_bool = False

            if (np.abs(tri_third_mode[0] - tri_second_mode[0]) > 1) or \
               (np.abs(tri_third_mode[1] - tri_second_mode[1]) > 1.5):
                third_second_dist_bool = True

            else:
                third_second_dist_bool = False

            intensity_sub_min = np.nanmin(intensity_sub)
            if tri_ratio_bool and third_second_dist_bool and \
               first_second_dist_bool and tri_first_mode[0] > -32 and \
               intensity_sub_min < tri_second_mode[0]:
                tri_optimization = True
            else:
                tri_optimization = False

        except:
            tri_optimization = False
            logger.info(
                'Trimodal curve Fitting fails in threshold computation.')

        if tri_optimization:
            intensity_sub2 = intensity_sub[intensity_sub < tri_second_mode[0]]

            if intensity_sub2.size > 0:
                tau_bound_gauss = threshold_otsu(intensity_sub2)

                if threshold > tau_bound_gauss:
                    threshold = tau_bound_gauss
                    idx_tau_bound_gauss = np.searchsorted(intensity_bins,
                                                          threshold)
                    tri_lowmaxind_cands, _ = find_peaks(
                        intensity_counts[0:idx_tau_bound_gauss+1],
                        distance=5)

                    if not tri_lowmaxind_cands.any():
                        tri_lowmaxind_cands = np.array(
                            [np.nanargmax(
                             intensity_counts[: idx_tau_bound_gauss+1])])
                    intensity_counts_cand = intensity_counts[
                        tri_lowmaxind_cands]
                    tri_lowmaxind = tri_lowmaxind_cands[
                        np.nanargmax(intensity_counts_cand)]
                    tri_lowmaxind = np.squeeze(tri_lowmaxind)

                    if tri_lowmaxind.size > 1:
                        tri_lowmaxind = tri_lowmaxind[tri_lowmaxind <=
                                                      idx_tau_bound_gauss]
                        tri_lowmaxind = tri_lowmaxind[0]
                    modevalue = intensity_bins[tri_lowmaxind]

        if method == 'rg':
            intensity_countspp, _ = np.histogram(
                intensity_sub,
                bins=np.linspace(min_intensity_histogram,
                                 max_intensity_histogram,
                                 numstep + 1),
                density=False)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = np.where(intensity_counts != 0,
                                  intensity_countspp / intensity_counts,
                                  np.nan)
            ratio = np.nanmean(ratios)

            if optimization:
                diff_dist = intensity_counts - simul_first
                diff_dist[idx_threshold:] = np.nan

                if len(lowmaxind_cands) > 1:
                    lowmaxind_cands = lowmaxind_cands[-1]
                diff_dist[:int(lowmaxind_cands)] = np.nan
                diff_dist_ind = np.where(diff_dist > 0.05)

                if len(diff_dist_ind[0]) > 0:
                    diverse_ind = diff_dist_ind[0][0]
                    modevalue = (modevalue + intensity_bins[diverse_ind]) / 2

                rms_sss = []
                rms_xx = []

                for hist_bin in bins:
                    if hist_bin > threshold:
                        rg_layer = _region_growing.region_growing(
                            intensity_sub,
                            initial_threshold=threshold,
                            relaxed_threshold=hist_bin,
                            maxiter=200,
                            mode='ascending',
                            verbose=False)

                        rg_target_area = intensity_sub[rg_layer]
                        intensity_counts_rg, _ = np.histogram(
                            rg_target_area,
                            bins=np.linspace(
                                min_intensity_histogram,
                                max_intensity_histogram,
                                numstep+1),
                            density=False)

                        intensity_counts_rg = intensity_counts_rg / ratio
                        compare_index1 = (np.abs(bins - threshold)).argmin()
                        compare_index2 = (np.abs(bins - hist_bin)).argmin()
                        # to avoid empty array
                        if compare_index1 != compare_index2:
                            rms = np.sqrt(np.nanmean(
                                (intensity_counts_rg[compare_index1:
                                                     compare_index2] -
                                 simul_first[compare_index1:
                                             compare_index2]) ** 2))
                            rms_sss.append(rms)
                            rms_xx.append(hist_bin)
                        else:
                            rms_sss.append(np.nan)
                            rms_sss.append(hist_bin)

                min_rms_ind = np.where(rms_sss)
                rg_tolerance = rms_xx[min_rms_ind[0][0]]
                threshold = (rg_tolerance + threshold) / 2

        if adjust_if_nonoverlap:
            old_threshold = threshold

            if optimization:
                mean1, std1 = first_mode[0:2]
                mean2, std2 = second_mode[0:2]

                threshold = _initial_threshold.optimize_inter_distribution_threshold(
                    old_threshold,
                    mean1=mean1,
                    std1=std1,
                    mean2=mean2,
                    std2=std2,
                    step_fraction=0.05,
                    max_iterations=100,
                    thresh_low_dist_percent=adjust_thresh_low_dist_percent,
                    thresh_high_dist_percent=adjust_thresh_high_dist_percent)

        # --- TILE-WISE RELAXED THRESHOLD (data-driven; optional) ---
        if TILE_RELAX:
            try:
                # use the tile's intensities (already in dB)
                tau_relaxed = _tile_relaxed_threshold_from_boundary(
                    intensity_tile_db=intensity_sub,
                    tau_strict=float(threshold),
                    max_band_px=3,
                    core_iter=1)
                threshold = float(tau_relaxed)
            except Exception:
                pass

        # add final threshold to threshold list
        threshold_array.append(threshold)
        threshold_idx_array.append(idx_threshold)
        mode_array.append(modevalue)


        idx_threshold = np.searchsorted(intensity_bins, threshold)
        idx_threshold = int(np.clip(idx_threshold, 0, max(0, len(intensity_bins) - 1)))

        # Recompute the lower slice up to the final threshold
        low_slice_final = np.nan_to_num(intensity_counts[:idx_threshold + 1], nan=0.0)

        if optimization:
            # left Gaussian mean
            if lock_mode_from_model:
                # keep the lower-Gaussian mean from the model
                modevalue_final = first_mode[0]
            else:
                # legacy behavior
                modevalue_final = min(first_mode[0], intensity_bins[idx_threshold])
            # modevalue_final = min(first_mode[0], intensity_bins[idx_threshold])
        else:
            # fallback: histogram peak on low side
            if low_slice.size == 0 or np.all(low_slice == 0):
                modevalue_final = intensity_bins[max(0, idx_threshold)]
            else:
                lowmaxind_cands_final, _ = find_peaks(low_slice_final, distance=5)
                if lowmaxind_cands_final.size == 0:
                    lowmaxind_final = int(np.argmax(low_slice_final))
                else:
                    lowmaxind_final = int(lowmaxind_cands_final[
                        np.argmax(low_slice_final[lowmaxind_cands_final])])
                lowmaxind_final = int(np.clip(lowmaxind_final, 0, idx_threshold))
                modevalue_final = intensity_bins[lowmaxind_final]
        margin = 0.05
        if np.isfinite(modevalue_final) and np.isfinite(threshold) and (modevalue_final >= threshold):
            threshold = modevalue_final + margin
            # keep idx consistent after guard tweak
            idx_threshold = np.searchsorted(intensity_bins, threshold)
            idx_threshold = int(np.clip(idx_threshold, 0, max(0, len(intensity_bins) - 1)))
        modevalue = modevalue_final
 
        # add final threshold to threshold list
        threshold_array.append(threshold)
        threshold_idx_array.append(idx_threshold)
        mode_array.append(modevalue)

    threshold_array = np.array(threshold_array)
    mode_array = np.array(mode_array)

    threshold_array[threshold_array > max_threshold] = np.nan
    threshold_array[threshold_array < min_threshold] = np.nan
    mode_array[mode_array > max_threshold] = np.nan

    global_threshold = threshold_array
    glob_mode_thres = mode_array

    return global_threshold, glob_mode_thres


def run_sub_block(intensity,
                  water_body_subset,
                  cfg,
                  winsize=200,
                  thres_max=None):
    """
    Process sub-blocks of SAR intensity data for water detection based on
    the specified configuration.

    Parameters
    ----------
    intensity : np.array
        SAR intensity data, can be 2D or 3D (polarizations).
    water_body_subset : np.array
        Water body subset data for masking.
    cfg : object
        Configuration settings for processing.
    win_size : int, optional
        Window size for processing tiles (default is 200).
    threshold_max : list, optional
        Maximum thresholds for different polarizations

    Returns
    -------
    tuple of lists
        A tuple containing lists of thresholds, mode values, and candidate tile
        coordinates for each polarization.
    """
    if intensity.ndim == 3:
        _, height, width = np.shape(intensity)
    else:
        height, width = np.shape(intensity)
        intensity = np.reshape(intensity, [1, height, width])

    processing_cfg = cfg.groups.processing

    pol_list = processing_cfg.polarizations
    dswx_workflow = processing_cfg.dswx_workflow

    # initial threshold cfg
    threshold_cfg = processing_cfg.initial_threshold
    tile_selection_method = threshold_cfg.selection_method
    threshold_method = threshold_cfg.threshold_method
    mutli_threshold_flag = threshold_cfg.multi_threshold
    threshold_bounds_co_pol = threshold_cfg.threshold_bounds.co_pol
    threshold_bounds_cross_pol = threshold_cfg.threshold_bounds.cross_pol

    tile_selection_twele = threshold_cfg.tile_selection_twele
    tile_selection_bimodality = threshold_cfg.tile_selection_bimodality

    threshold_scale = getattr(threshold_cfg, 'threshold_scale', 'db').lower()

    adjust_threshold_flag = threshold_cfg.adjust_if_nonoverlap
    low_dist_percentile = threshold_cfg.low_dist_percentile
    high_dist_percentile = threshold_cfg.high_dist_percentile

    # water cfg
    water_cfg = processing_cfg.reference_water

    if dswx_workflow.lower() == 'twele':
        tile_selection_method = 'twele'

    # Tile Selection (w/o water body)
    if (height < winsize) | (width < winsize):
        logger.info('winsize is smaller than image size')
        winsize = np.min([winsize, height, width])

    number_y_window = np.int16(height / winsize)
    number_x_window = np.int16(width / winsize)

    number_y_window = number_y_window + \
        (1 if np.mod(height, winsize) > 0 else 0)
    number_x_window = number_x_window + \
        (1 if np.mod(width, winsize) > 0 else 0)

    threshold_tau_set = []
    mode_tau_set = []
    candidate_tile_coords_set = []

    tile_selection_object = _initial_threshold.TileSelection(
        ref_water_max=water_cfg.max_value,
        no_data=water_cfg.no_data_value)
    tile_selection_object.threshold_twele = tile_selection_twele
    tile_selection_object.threshold_bimodality = \
        tile_selection_bimodality

    # Tile Selection (with water body)
    for polind, pol in enumerate(pol_list):

        candidate_tile_coords = tile_selection_object.tile_selection_wbd(
                           intensity=intensity[polind, :, :],
                           water_mask=water_body_subset,
                           win_size=winsize,
                           selection_methods=tile_selection_method)

        if len(candidate_tile_coords) > 0:
            if pol in ['VV', 'VH', 'HH', 'HV', 'span', 'ratio']:
                target_im = _initial_threshold.convert_pow2db(intensity[polind]) \
                    if threshold_scale == 'db' else intensity[polind]
            else:
                target_im = intensity[polind]
            (min_intensity_histogram,
             max_intensity_histogram,
             step_histogram) = _get_histogram_params(pol, threshold_scale)

            wbdsub_norm = water_body_subset / \
                tile_selection_object.wbd_max_value

            if wbdsub_norm.shape[0]*wbdsub_norm.shape[1] == 0:
                water_variation = 0
            else:
                water_variation = len(wbdsub_norm[
                                     (wbdsub_norm < 0.8) &
                                     (wbdsub_norm > 0.2)])\
                                     / (wbdsub_norm.shape[0] *
                                        wbdsub_norm.shape[1])
            # When water bodies are not enough to compute the bound values,
            # use the pre-defined values.
            if pol in ['VV', 'HH', 'span']:
                bound_min_db, bound_max_db = threshold_bounds_co_pol[0], threshold_bounds_co_pol[1]
            else:
                bound_min_db, bound_max_db = threshold_bounds_cross_pol[0], threshold_bounds_cross_pol[1]

            if threshold_scale == 'db':
                threshold_temp_min, threshold_temp_max_cfg = bound_min_db, bound_max_db
            else:
                threshold_temp_min, threshold_temp_max_cfg = convert_db2pow(bound_min_db), convert_db2pow(bound_max_db)

            if water_variation > 0.1:
                threshold_temp_max = thres_max[polind] if threshold_scale == 'db' else convert_db2pow(thres_max[polind])
            else:
                threshold_temp_max = threshold_temp_max_cfg

            intensity_threshold, mode_tau = determine_threshold(
                intensity=target_im,
                candidate_tile_coords=candidate_tile_coords,
                min_intensity_histogram=min_intensity_histogram,
                max_intensity_histogram=max_intensity_histogram,
                step_histogram=step_histogram,
                bounds=[threshold_temp_min, threshold_temp_max],
                method=threshold_method,
                mutli_threshold=mutli_threshold_flag,
                adjust_if_nonoverlap=adjust_threshold_flag,
                adjust_thresh_low_dist_percent=low_dist_percentile,
                adjust_thresh_high_dist_percent=high_dist_percentile,
                )

            if threshold_scale == 'linear':
                # intensity_threshold and mode_tau are arrays (or NaNs) in linear power
                with np.errstate(divide='ignore', invalid='ignore'):
                    if isinstance(intensity_threshold, np.ndarray):
                        intensity_threshold = np.where(np.isfinite(intensity_threshold),
                                                    10.0 * np.log10(intensity_threshold),
                                                    np.nan)
                    elif np.isfinite(intensity_threshold):
                        intensity_threshold = 10.0 * np.log10(intensity_threshold)

                    if isinstance(mode_tau, np.ndarray):
                        mode_tau = np.where(np.isfinite(mode_tau),
                                            10.0 * np.log10(mode_tau),
                                            np.nan)
                    elif np.isfinite(mode_tau):
                        mode_tau = 10.0 * np.log10(mode_tau)

            logger.info(f'method {threshold_method} for {pol}')
            logger.info('global threshold and bound : '
                        f'{intensity_threshold} {threshold_temp_max}')
            logger.info(f'global mode thresholding : {mode_tau}')

        else:
            intensity_threshold = np.nan
            mode_tau = np.nan
        threshold_tau_set.append(intensity_threshold)
        mode_tau_set.append(mode_tau)
        candidate_tile_coords_set.append(candidate_tile_coords)

    return threshold_tau_set, mode_tau_set, candidate_tile_coords_set


def _get_histogram_params(polarization, scale):
    """Define Mininum, maximum, and step for histogram
    based on polarization.
    """
    if scale == 'db':
        if polarization in ['VV', 'VH', 'HH', 'HV', 'span']:
            return -35, 10, 0.1
        elif polarization == 'ratio':
            return -11, 0, 0.1
        else:
            return np.nan, np.nan, np.nan
    else:  # linear
        # Use auto bounds via percentiles in determine_threshold
        # (min/max == -1000 triggers auto)
        return -1000, -1000, 0.1


def compute_water_spatial_coverage(
        water_body_data_path,
        no_data_path,
        water_threshold,
        water_body_max,
        lines_per_block):
    """
    Calculates the percentage of water coverage in a given area.

    Parameters
    ----------
    water_body_data_path : str
        Path to the raster file containing water body data.
    no_data_path : str
        Path to the raster file indicating no data areas.
    water_threshold : float
        Threshold value used to classify water presence.
    water_body_max : int
        Maximum valid value in the water body data for normalization.
    lines_per_block : int
        Number of lines per block for processing the data in chunks.

    Returns
    -------
    water_percentage : float
        The percentage of water coverage in the area.
    """
    im_meta = _dswx_sar_util.get_meta_from_tif(water_body_data_path)
    data_shape = [im_meta['length'], im_meta['width']]
    valid_pixel_number = 0
    water_pixel_number = 0
    pad_shape = (0, 0)
    block_params = _dswx_sar_util.block_param_generator(
        lines_per_block,
        data_shape,
        pad_shape)
    for block_param in block_params:
        water_body_data = _dswx_sar_util.get_raster_block(
            water_body_data_path, block_param)
        no_data_area = _dswx_sar_util.get_raster_block(
            no_data_path, block_param)
        water_binary = water_body_data / water_body_max > water_threshold

        valid_area = no_data_area == 0
        water_pixel_number += np.sum(water_binary[valid_area])
        valid_pixel_number += np.sum(valid_area)

    water_percentage = water_pixel_number / valid_pixel_number
    return water_percentage


def fill_threshold_and_mode_decoupled_with_gdal(threshold_dict,
                                                mode_dict,
                                                rows,
                                                cols,
                                                filename_threshold,   # e.g. 'intensity_threshold_filled'
                                                filename_mode,        # e.g. 'mode_tau_filled'
                                                outputdir,
                                                pol_list,
                                                margin=0.05,
                                                no_data=-50,
                                                average_tile=True):
    """
    Interpolate threshold and (threshold-mode) jointly to enforce mode <= threshold - margin.
    Steps:
      1) build delta samples = max(threshold - mode, margin) at sample points
      2) gdal_grid threshold -> raster T
      3) gdal_grid delta     -> raster D
      4) mode raster M = T - max(D, margin)
    """
    # Build a "delta" dict from the threshold/mode dicts (matching structure)
    def _build_delta_dict(th_dict, md_dict):
        dd = {k: None for k in th_dict.keys()}
        if average_tile:
            # th: [ny, nx, nb], md: [ny, nx, nb]
            th = np.array(th_dict['array'], dtype=float)
            md = np.array(md_dict['array'], dtype=float)
            th[th == no_data] = np.nan
            md[md == no_data] = np.nan
            delta = th - md
            if np.isscalar(margin):
                delta = np.where(np.isfinite(delta), np.maximum(delta, margin), np.nan)
            dd['array'] = delta
            dd['block_row'] = th_dict['block_row']
            dd['block_col'] = th_dict['block_col']
        else:
            # scattered samples per pol
            dd['array'] = []
            dd['block_row'] = []
            dd['block_col'] = []
            dd['subtile_coord'] = th_dict.get('subtile_coord', None)
            for p in range(len(pol_list)):
                th = np.asarray(th_dict['array'][p], dtype=float)
                md = np.asarray(md_dict['array'][p], dtype=float)
                th[th == no_data] = np.nan
                md[md == no_data] = np.nan
                delta = th - md
                delta = np.where(np.isfinite(delta), np.maximum(delta, margin), np.nan)
                dd['array'].append(delta)
                dd['block_row'].append(np.asarray(th_dict['block_row'][p], dtype=int))
                dd['block_col'].append(np.asarray(th_dict['block_col'][p], dtype=int))
        return dd

    delta_dict = _build_delta_dict(threshold_dict, mode_dict)

    # 1) Interpolate threshold -> T
    _initial_threshold.fill_threshold_with_gdal(threshold_array=threshold_dict,
                             rows=rows, cols=cols,
                             filename=filename_threshold,
                             outputdir=outputdir,
                             pol_list=pol_list,
                             filled_value=None,
                             no_data=no_data,
                             average_tile=average_tile)

    # 2) Interpolate delta -> D  (temporary filename)
    tmp_delta_name = filename_threshold + "_delta"
    _initial_threshold.fill_threshold_with_gdal(threshold_array=delta_dict,
                             rows=rows, cols=cols,
                             filename=tmp_delta_name,
                             outputdir=outputdir,
                             pol_list=pol_list,
                             filled_value=None,
                             no_data=no_data,
                             average_tile=average_tile)

    # 3) Compose mode = threshold - max(delta, margin) and write out
    for pol in pol_list:
        t_path = os.path.join(outputdir, f"{filename_threshold}_{pol}.tif")
        d_path = os.path.join(outputdir, f"{tmp_delta_name}_{pol}.tif")
        m_path = os.path.join(outputdir, f"{filename_mode}_{pol}.tif")

        t_ds = gdal.Open(t_path, gdal.GA_ReadOnly)
        d_ds = gdal.Open(d_path, gdal.GA_ReadOnly)
        if t_ds is None or d_ds is None:
            logger.info(f"Missing interpolated rasters for {pol}; skipping.")
            continue

        T = t_ds.ReadAsArray().astype(np.float32)
        D = d_ds.ReadAsArray().astype(np.float32)
        t_gt, t_prj = t_ds.GetGeoTransform(), t_ds.GetProjection()

        # Enforce positive margin; handle NaNs
        D = np.where(np.isfinite(D), np.maximum(D, margin), margin)
        M = T - D

        drv = gdal.GetDriverByName('GTiff')
        out = drv.Create(m_path, T.shape[1], T.shape[0], 1, gdal.GDT_Float32)
        out.SetGeoTransform(t_gt)
        out.SetProjection(t_prj)
        out.GetRasterBand(1).WriteArray(M)
        out.FlushCache()
        out = None
        t_ds = None
        d_ds = None

        _dswx_sar_util._save_as_cog(m_path, outputdir, logger,
                                   compression='DEFLATE', nbits=16)


def process_block(ii, jj,
                  n_rows_block, n_cols_block,
                  m_rows_block, m_cols_block,
                  block_row, block_col, width,
                  filt_im_str, wbd_im_str,
                  cfg, thres_max,
                  average_tile_flag=False):
    """
    Processes a specific block of an image.

    Parameters
    ----------
    ii : int
        Current row index of the block.
    jj : int
        Current column index of the block.
    n_rows_block : int
        Number of row blocks.
    n_cols_block : int
        Number of column blocks.
    m_rows_block : int
        Remaining rows in the block.
    m_cols_block : int
        Remaining columns in the block.
    block_row : int
        Rows per block.
    block_col : int
        Columns per block.
    width : int
        Width of the image.
    filt_im_str : str
        Filepath to the filtered image.
    wbd_im_str : str
        Filepath to the water body data image.
    cfg:
        Configuration object for processing.
    thres_max:
        Maximum threshold for processing.
    average_tile_flag (bool, optional):
        Flag to determine if tile averaging is applied. Defaults to False.

    Returns:
    --------
        tuple: Contains processed block information,
               including row index, column index,
               threshold list, mode list, and candidate coordinates.
    """
    x_size = m_cols_block \
        if (jj == n_cols_block - 1) and m_cols_block > 0 else block_col
    y_size = m_rows_block \
        if (ii == n_rows_block - 1) and m_rows_block > 0 else block_row

    logger.info(f"block_processing: {ii + 1}/{n_rows_block} "
                f"_ {jj + 1}/{n_cols_block}"
                f" - {ii * n_cols_block + jj + 1}/"
                f"{n_rows_block * n_cols_block}")

    filt_raster_tif = gdal.Open(filt_im_str)
    image_sub = filt_raster_tif.ReadAsArray(jj * block_col,
                                            ii * block_row,
                                            x_size,
                                            y_size)
    filt_raster_tif = None

    wbd_gdal = gdal.Open(wbd_im_str)
    wbd_sub = wbd_gdal.ReadAsArray(jj * block_col,
                                   ii * block_row,
                                   x_size,
                                   y_size)
    wbd_gdal = None
    threshold_tau_block, mode_tau_block, candidate_tile_coords = \
        run_sub_block(
            image_sub,
            wbd_sub,
            cfg,
            thres_max=thres_max)

    if average_tile_flag:
        threshold_list = [np.nanmedian(test_threshold)
                          if not np.all(np.isnan(test_threshold)) else np.nan
                          for test_threshold in threshold_tau_block]
        mode_list = [np.nanmedian(test_mode)
                     if not np.all(np.isnan(test_mode)) else np.nan
                     for test_mode in mode_tau_block]
    else:
        threshold_list = [np.nan_to_num(ind_list, nan=-50).tolist()
                          for ind_list in threshold_tau_block]
        mode_list = [np.nan_to_num(ind_list, nan=-50).tolist()
                     for ind_list in mode_tau_block]

    return ii, jj, threshold_list, mode_list, candidate_tile_coords


def run(cfg):
    """
    Run inital threshold with parameters in cfg dictionary
    """
    t_all = time.time()
    logger.info('Start Initial Threshold')

    processing_cfg = cfg.groups.processing
    pol_list = copy.deepcopy(processing_cfg.polarizations)
    pol_options = processing_cfg.polarimetric_option

    if pol_options is not None:
        pol_list += pol_options

    pol_all_str = '_'.join(pol_list)

    outputdir = cfg.groups.product_path_group.scratch_path

    # options for initial threshold
    init_threshold_cfg = processing_cfg.initial_threshold
    tile_selection_method = init_threshold_cfg.selection_method
    average_threshold_flag = init_threshold_cfg.tile_average
    threshold_extending_method = init_threshold_cfg.extending_method
    lines_per_block = init_threshold_cfg.line_per_block
    threshold_scale = getattr(init_threshold_cfg, 'threshold_scale', 'db').lower()

    logger.info(f'Tile selection method: {tile_selection_method}')
    logger.info(f'Average_threshold_flag: {average_threshold_flag}')

    number_workers = init_threshold_cfg.number_cpu

    # options for reference water
    ref_water_cfg = processing_cfg.reference_water
    drought_erosion_pixel = ref_water_cfg.drought_erosion_pixel
    flood_dilation_pixel = ref_water_cfg.flood_dilation_pixel
    permanent_water_value = ref_water_cfg.permanent_water_value
    ref_water_max = processing_cfg.reference_water.max_value

    # Filtered RTC image
    filt_im_str = os.path.join(
        outputdir, f"filtered_image_{pol_all_str}.tif")
    no_data_geotiff_path = os.path.join(
        outputdir, f"no_data_area_{pol_all_str}.tif")
    # Relocated reference water
    wbd_im_str = os.path.join(outputdir, 'interpolated_wbd.tif')

    # Read metadata from intensity image (projection, geotransform)
    water_meta = _dswx_sar_util.get_meta_from_tif(filt_im_str)
    band_number, height, width = [water_meta[attr_name]
                                  for attr_name in
                                  ["band_number", "length", "width"]]

    # create water masks for normal,
    # flood and drought using dilation and erosion
    water_mask_tif_name = f"water_mask_{pol_all_str}.tif"
    water_mask_tif_str = os.path.join(
        outputdir, f"{water_mask_tif_name}")
    _initial_threshold.create_three_water_masks(
        wbd_im_str,
        water_mask_tif_name,
        outputdir,
        water_threshold=permanent_water_value,
        no_data=processing_cfg.reference_water.no_data_value,
        wbd_max_value=ref_water_max,
        drought_erosion_pixel=drought_erosion_pixel,
        flood_dilation_pixel=flood_dilation_pixel)

    water_portion = compute_water_spatial_coverage(
        wbd_im_str,
        no_data_path=no_data_geotiff_path,
        water_threshold=permanent_water_value,
        water_body_max=ref_water_max,
        lines_per_block=lines_per_block)

    logger.info(f'water spatial coverage : {water_portion} ')

    thres_max = np.empty([band_number])

    if water_portion == 1:
        # If the areas cover only water,
        # then use the very high threshold to classify all pixels as water.
        for band_ind in range(band_number):
            pol_str = pol_list[band_ind]
            thresh_file_str = os.path.join(
                outputdir, f"intensity_threshold_filled_{pol_str}.tif")
            thresh_peak_str = os.path.join(
                outputdir, f"mode_tau_filled_{pol_str}.tif")
            for filled_file_path in [thresh_file_str, thresh_peak_str]:
                _dswx_sar_util.create_geotiff_with_one_value(
                    filled_file_path,
                    shape=[height, width],
                    filled_value=30)
    else:
        # Here we compute the bounds of the backscattering of water objects

        thres_max, intensity_sub_mean, intensity_sub_std, is_bimodal = \
            _initial_threshold.compute_threshold_max_bound(
                intensity_path=filt_im_str,
                reference_water_path=wbd_im_str,
                water_max_value=ref_water_max,
                water_threshold=permanent_water_value,
                no_data_path=no_data_geotiff_path,
                lines_per_block=lines_per_block)
        if threshold_scale == 'linear':
            thres_max = np.array([convert_db2pow(x) if np.isfinite(x) else x for x in thres_max])

        for band_ind in range(band_number):
            if pol_list[band_ind] == 'span':
                thres_max[band_ind] = 30
            else:
                logger.info(
                    'mean  intensity [dB] over water '
                    f'{pol_list[band_ind]}:'
                    f' {intensity_sub_mean[band_ind]:.2f}, {is_bimodal}')
                logger.info(
                    'std   intensity [dB] over water '
                    f'{pol_list[band_ind]}:'
                    f' {intensity_sub_std[band_ind]:.2f}, {is_bimodal}')
                logger.info(
                    'max bound intensity [dB] over water '
                    f'{pol_list[band_ind]}:'
                    f' {thres_max[band_ind]:.2f}, {is_bimodal}')

        block_row = init_threshold_cfg.maximum_tile_size.y
        block_col = init_threshold_cfg.maximum_tile_size.x

        # number_y_window
        n_rows_block = height // block_row
        # number_x_window
        n_cols_block = width // block_col
        m_rows_block = height % block_row
        m_cols_block = width % block_col

        n_rows_block = n_rows_block + (1 if m_rows_block > 0 else 0)
        n_cols_block = n_cols_block + (1 if m_cols_block > 0 else 0)

        threshold_tau_set = np.zeros([n_rows_block, n_cols_block, band_number])
        mode_tau_set = np.zeros([n_rows_block, n_cols_block, band_number])

        threshold_tau_dict = {}
        mode_tau_dict = {}

        # Parallel processing
        # -1 means using all processors
        n_jobs = number_workers

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_block)(
                ii, jj,
                n_rows_block, n_cols_block,
                m_rows_block, m_cols_block,
                block_row, block_col,
                width, filt_im_str,
                water_mask_tif_str, cfg,
                thres_max, average_threshold_flag)
            for ii in range(0, n_rows_block)
            for jj in range(0, n_cols_block)
            )

        # If average_threshold_flag is True, all thresholds within
        # individual tile are averaged and assigned to the tile.
        # If not, the thresholds remain as they are and are used for
        # interpolation.
        if average_threshold_flag:

            threshold_tau_set = np.zeros([n_rows_block,
                                          n_cols_block,
                                          band_number])
            mode_tau_set = np.zeros([n_rows_block,
                                    n_cols_block,
                                    band_number])

            for ii, jj, threshold_tau_block, mode_tau_block, window_coord \
                    in results:
                threshold_tau_set[ii, jj, :] = threshold_tau_block
                mode_tau_set[ii, jj, :] = mode_tau_block

            threshold_tau_dict = _initial_threshold.save_threshold_dict(
                threshold_tau_set,
                block_row,
                block_col)
            mode_tau_dict = _initial_threshold.save_threshold_dict(
                mode_tau_set,
                block_row,
                block_col)

        else:
            threshold_tau_set = [[], [], []]
            mode_tau_set = [[], [], []]
            coord_row_list = [[], [], []]
            coord_col_list = [[], [], []]
            window_coord_list = [[], [], []]

            for ii, jj, threshold_tau_blocks, mode_tau_blocks, window_coords \
                    in results:
                # extract threshold for tiles
                pol_index = 0

                # individual polarizations
                for threshold_tau_block, mode_tau_block, window_coord in zip(
                     threshold_tau_blocks, mode_tau_blocks, window_coords):
                    # -50 represent the no-data value.
                    threshold_tau_subset = list(filter(lambda x: x != -50,
                                                [threshold_tau_block]))
                    mode_tau_subset = list(filter(lambda x: x != -50,
                                                  [mode_tau_block]))

                    if threshold_tau_subset:
                        # window center for row and col
                        window_center_row_list = [
                            int(ii * block_row +
                                (sub_window[1] + sub_window[2]) / 2)
                            for sub_window in window_coord]

                        window_center_col_list = [
                            int(jj * block_col +
                                (sub_window[3] + sub_window[4]) / 2)
                            for sub_window in window_coord]

                        # window coordinates
                        absolute_window_coord = [
                            [ii * block_row + sub_window[1],
                             ii * block_row + sub_window[2],
                             jj * block_col + sub_window[3],
                             jj * block_col + sub_window[4]]
                            for sub_window in window_coord]

                        coord_row_list[pol_index] = \
                            coord_row_list[pol_index] + window_center_row_list
                        coord_col_list[pol_index] = \
                            coord_col_list[pol_index] + window_center_col_list

                        threshold_tau_set[pol_index].extend(
                            threshold_tau_subset[0])
                        mode_tau_set[pol_index].extend(mode_tau_subset[0])
                        window_coord_list[pol_index].extend(
                            absolute_window_coord)
                    pol_index += 1

            threshold_tau_dict['block_row'] = coord_row_list
            threshold_tau_dict['block_col'] = coord_col_list

            mode_tau_dict['block_row'] = coord_row_list
            mode_tau_dict['block_col'] = coord_col_list

            threshold_tau_dict['array'] = threshold_tau_set
            threshold_tau_dict['subtile_coord'] = window_coord_list
            mode_tau_dict['array'] = mode_tau_set

        if not threshold_tau_dict:
            logger.info('No threshold_tau')
        # Currently, only 'gdal_grid' method is supported.
        if threshold_extending_method == 'gdal_grid':
            dict_threshold_list = [threshold_tau_dict, mode_tau_dict]
            interp_thres_str_list = ['intensity_threshold_filled',
                                     'mode_tau_filled']
            for dict_thres, thres_str in zip(dict_threshold_list,
                                             interp_thres_str_list):
                fill_threshold_and_mode_decoupled_with_gdal(
                    threshold_dict=threshold_tau_dict,
                    mode_dict=mode_tau_dict,
                    rows=height,
                    cols=width,
                    filename_threshold='intensity_threshold_filled',
                    filename_mode='mode_tau_filled',
                    outputdir=outputdir,
                    pol_list=pol_list,
                    margin=0.05,
                    no_data=-50,
                    average_tile=average_threshold_flag
                )
                # fill_threshold_with_gdal(
                #     threshold_array=dict_thres,
                #     rows=height,
                #     cols=width,
                #     filename=thres_str,
                #     outputdir=outputdir,
                #     pol_list=pol_list,
                #     filled_value=thres_max,
                #     no_data=-50,
                #     average_tile=average_threshold_flag)

    if processing_cfg.debug_mode:

        intensity_whole = _dswx_sar_util.read_geotiff(filt_im_str)
        if intensity_whole.ndim == 2:
            intensity_whole = np.expand_dims(intensity_whole,
                                             axis=0)
        if not average_threshold_flag:
            _dswx_sar_util.block_threshold_visualization_rg(
                intensity_whole,
                threshold_tau_dict,
                outputdir=outputdir,
                figname='int_threshold_visualization_')
        else:
            for band_ind2 in range(band_number):
                _dswx_sar_util.block_threshold_visualization(
                    np.squeeze(intensity_whole[band_ind2, :, :]),
                    block_row,
                    block_col,
                    threshold_tau_set[:, :, band_ind2],
                    outputdir,
                    f'int_threshold_visualization_{pol_list[band_ind2]}')

        data_shape = (height, width)
        pad_shape = (0, 0)

        for polind, pol in enumerate(pol_list):

            pad_shape = (0, 0)
            block_params = _dswx_sar_util.block_param_generator(
                lines_per_block,
                data_shape,
                pad_shape)

            thresh_file_path = os.path.join(
                outputdir, f"intensity_threshold_filled_{pol}.tif")
            initial_water_tif_path = os.path.join(
                outputdir, f"initial_water_{pol}.tif")
            threshold_geotiff = os.path.join(
                outputdir, f"intensity_threshold_filled_{pol}_georef.tif")

            for block_param in block_params:
                threshold_block = _dswx_sar_util.get_raster_block(
                    thresh_file_path, block_param=block_param)
                intensity_block = _dswx_sar_util.get_raster_block(
                    filt_im_str, block_param=block_param)
                if intensity_block.ndim == 2:
                    intensity_block = np.expand_dims(intensity_block,
                                                     axis=0)
                # if threshold_scale == 'db':
                #     left = convert_pow2db(np.squeeze(intensity_block[polind, :, :]))
                # else:
                left_db = _initial_threshold.convert_pow2db(
                    np.squeeze(intensity_block[polind, :, :]))

                initial_water_binary = left_db < threshold_block
                # Mask nodata in threshold
                nodata_thr = -50.0  # keep consistent with fill_threshold_with_gdal no_data
                thr_valid = np.isfinite(threshold_block) & (threshold_block != nodata_thr)

                initial_water_binary = np.zeros_like(threshold_block, dtype=np.uint8)
                initial_water_binary[thr_valid] = (left_db[thr_valid] < threshold_block[thr_valid]).astype(np.uint8)
                if initial_water_binary.ndim == 1:
                    initial_water_binary = initial_water_binary[np.newaxis, :]
                    threshold_block = threshold_block[np.newaxis, :]

                _dswx_sar_util.write_raster_block(
                    out_raster=initial_water_tif_path,
                    data=initial_water_binary,
                    block_param=block_param,
                    geotransform=water_meta['geotransform'],
                    projection=water_meta['projection'],
                    datatype='byte',
                    cog_flag=True,
                    scratch_dir=outputdir)

                _dswx_sar_util.write_raster_block(
                    out_raster=threshold_geotiff,
                    data=threshold_block,
                    block_param=block_param,
                    geotransform=water_meta['geotransform'],
                    projection=water_meta['projection'],
                    datatype='float32',
                    cog_flag=True,
                    scratch_dir=outputdir)

    t_all_elapsed = time.time() - t_all
    logger.info(f"successfully ran computing initial threshold in "
                f"{t_all_elapsed:.3f} seconds")


def main():

    parser = _get_parser()

    args = parser.parse_args()

    _generate_log.configure_log_file(args.log_file)

    mimetypes.add_type("text/yaml", ".yaml", strict=True)
    flag_first_file_is_text = 'text' in mimetypes.guess_type(
        args.input_yaml[0])[0]

    if len(args.input_yaml) > 1 and flag_first_file_is_text:
        logger.info('ERROR only one runconfig file is allowed')
        return

    if flag_first_file_is_text:

        from dswx_sar.nisar.dswx_ni_runconfig import RunConfig
        workflow = 'dswx_ni'
        cfg = RunConfig.load_from_yaml(args.input_yaml[0],
                                       workflow, args)

    processing_cfg = cfg.groups.processing
    pol_mode = processing_cfg.polarization_mode
    pol_list = processing_cfg.polarizations
    if pol_mode == 'MIX_DUAL_POL':
        proc_pol_set = [DSWX_NI_POL_DICT['DV_POL'],
                        DSWX_NI_POL_DICT['DH_POL']]
    elif pol_mode == 'MIX_SINGLE_POL':
        proc_pol_set = [DSWX_NI_POL_DICT['SV_POL'],
                        DSWX_NI_POL_DICT['SH_POL']]
    else:
        proc_pol_set = [pol_list]
    for pol_set in proc_pol_set:
        processing_cfg.polarizations = pol_set
        run(cfg)


if __name__ == '__main__':
    main()
