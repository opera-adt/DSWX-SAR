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
from scipy.ndimage import gaussian_filter1d

from dswx_sar.common import (
    _dswx_sar_util,
    _generate_log,
    _initial_threshold,
    _refine_with_bimodality,
    _region_growing)

from dswx_sar.nisar.dswx_ni_runconfig import (
    DSWX_NI_POL_DICT,
    _get_parser,
    RunConfig)


logger = logging.getLogger('dswx_sar')

import json
from pathlib import Path


def _safe_float(x):
    try:
        x = float(x)
        if np.isfinite(x):
            return x
        return None
    except Exception:
        return None


def _array_summary(a):
    a = np.asarray(a)
    finite = np.isfinite(a)

    out = {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "size": int(a.size),
        "n_finite": int(finite.sum()),
        "n_nan": int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.number) else None,
        "n_inf": int(np.isinf(a).sum()) if np.issubdtype(a.dtype, np.number) else None,
    }

    if finite.any():
        af = a[finite].astype(np.float64)
        out.update({
            "min": _safe_float(np.min(af)),
            "max": _safe_float(np.max(af)),
            "mean": _safe_float(np.mean(af)),
            "std": _safe_float(np.std(af)),
            "p01": _safe_float(np.percentile(af, 1)),
            "p50": _safe_float(np.percentile(af, 50)),
            "p99": _safe_float(np.percentile(af, 99)),
        })
    else:
        out.update({
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p01": None,
            "p50": None,
            "p99": None,
        })

    return out

import hashlib


def _array_hash(a, round_decimals=3, sort_values=True):
    """
    Create a stable hash for numerical arrays.

    This is for debugging Intel/Mac reproducibility.
    Rounding removes tiny numerical noise, and sorting makes the hash
    insensitive to ordering if we only care about value distribution.
    """
    a = np.asarray(a)
    a = a[np.isfinite(a)]

    if a.size == 0:
        return None

    a = a.astype(np.float64)

    if round_decimals is not None:
        a = np.round(a, round_decimals)

    if sort_values:
        a = np.sort(a)

    a = a.astype(np.float32)
    return hashlib.sha256(a.tobytes()).hexdigest()

def _pick_stable_peak_index(
        counts,
        candidate_indices=None,
        prefer="first",
        rel_tol=0.002,
        abs_tol=1e-4):
    """
    Pick a histogram peak deterministically using a tolerance band.

    Instead of selecting the exact maximum, this selects all peaks whose
    heights are close to the maximum, then applies a fixed tie-break rule.
    This avoids Intel/Mac flips when several histogram peaks are nearly equal.
    """
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)

    if counts.size == 0:
        return 0

    if candidate_indices is None or len(candidate_indices) == 0:
        candidate_indices = np.arange(counts.size, dtype=int)
    else:
        candidate_indices = np.asarray(candidate_indices, dtype=int)

    candidate_indices = candidate_indices[
        (candidate_indices >= 0) & (candidate_indices < counts.size)
    ]

    if candidate_indices.size == 0:
        candidate_indices = np.arange(counts.size, dtype=int)

    vals = counts[candidate_indices]
    max_val = np.max(vals)

    tol = max(abs_tol, abs(max_val) * rel_tol)

    # Include near-maximum peaks, not only the exact maximum.
    tied = candidate_indices[vals >= max_val - tol]

    if tied.size == 0:
        return int(candidate_indices[np.argmax(vals)])

    if prefer == "last":
        return int(tied[-1])

    if prefer == "center":
        center = 0.5 * (counts.size - 1)
        return int(tied[np.argmin(np.abs(tied - center))])

    # default: stable low-index choice
    return int(tied[0])


def _pick_dominant_hist_mode(
        bins,
        counts,
        candidate_slice=None,
        rel_tol=0.002,
        abs_tol=1e-4,
        merge_gap=5,
        smooth_sigma=1.0):
    """
    Pick a histogram mode deterministically.

    Use a smoothed histogram only for selecting the dominant mode location.
    The original counts are still returned for the selected bin amplitude.
    """
    bins = np.asarray(bins, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)

    if candidate_slice is None:
        offset = 0
        sub_bins = bins
        sub_counts = counts
    else:
        start = 0 if candidate_slice.start is None else candidate_slice.start
        stop = len(counts) if candidate_slice.stop is None else candidate_slice.stop
        offset = start
        sub_bins = bins[start:stop]
        sub_counts = counts[start:stop]

    if sub_counts.size == 0 or np.all(sub_counts <= 0):
        idx = min(max(offset, 0), len(bins) - 1)
        return idx, float(bins[idx]), float(counts[idx]), {
            "near_indices": [],
            "groups": [],
            "selected_group": None,
            "centroid": None,
            "smooth_sigma": smooth_sigma,
        }

    if smooth_sigma is not None and smooth_sigma > 0:
        score_counts = gaussian_filter1d(sub_counts, sigma=smooth_sigma)
    else:
        score_counts = sub_counts.copy()

    max_val = float(np.max(score_counts))
    tol = max(abs_tol, abs(max_val) * rel_tol)
    near_rel = np.where(score_counts >= max_val - tol)[0]

    if near_rel.size == 0:
        rel_idx = int(np.argmax(score_counts))
        idx = offset + rel_idx
        return idx, float(bins[idx]), float(counts[idx]), {
            "near_indices": [],
            "groups": [],
            "selected_group": None,
            "centroid": float(bins[idx]),
            "smooth_sigma": smooth_sigma,
        }

    groups = []
    current = [int(near_rel[0])]

    for ii in near_rel[1:]:
        ii = int(ii)
        if ii - current[-1] <= merge_gap:
            current.append(ii)
        else:
            groups.append(current)
            current = [ii]
    groups.append(current)

    group_scores = []
    for g in groups:
        lo = max(0, min(g) - 1)
        hi = min(sub_counts.size, max(g) + 2)

        # Use smoothed counts for mass/centroid stability.
        w = score_counts[lo:hi]
        x = sub_bins[lo:hi]
        mass = float(np.sum(w))

        if mass > 0:
            centroid = float(np.sum(x * w) / mass)
        else:
            centroid = float(sub_bins[g[0]])

        group_scores.append((mass, centroid, lo, hi, g))

    # Highest mass; deterministic tie by earlier centroid/bin
    best = max(group_scores, key=lambda z: (z[0], -z[2]))
    _, centroid, lo, hi, best_group = best

    rel_idx = int(np.argmin(np.abs(sub_bins - centroid)))
    idx = offset + rel_idx
    idx = int(np.clip(idx, 0, len(bins) - 1))

    debug = {
        "near_indices": [int(offset + i) for i in near_rel],
        "near_bins": [float(bins[offset + i]) for i in near_rel],
        "groups": [[int(offset + j) for j in g] for g in groups],
        "selected_group": [int(offset + j) for j in best_group],
        "centroid": centroid,
        "selected_index": int(idx),
        "selected_bin": float(bins[idx]),
        "selected_count": float(counts[idx]),
        "smooth_sigma": smooth_sigma,
    }

    return idx, float(bins[idx]), float(counts[idx]), debug


def _write_stage_record(out_dir, record):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "stage_manifest.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

import json
from pathlib import Path


def _json_safe(v):
    """Convert numpy scalars/arrays to JSON-safe objects."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _json_safe(val) for k, val in v.items()}
    return v


def save_curve_fit_case(
        out_dir,
        case_id,
        *,
        model_name,
        pol=None,
        block_ij=None,
        coord=None,
        absolute_coord=None,
        intensity_sub=None,
        intensity_bins=None,
        intensity_counts=None,
        expected=None,
        bounds=None,
        threshold_before_fit=None,
        idx_threshold=None,
        tau_mode_left=None,
        tau_mode_right=None,
        tau_amp_left=None,
        tau_amp_right=None,
        method=None,
        threshold_scale=None):
    """
    Save one exact curve_fit test case.

    Files:
      curvefit_case_xxxxxx.npz  : numerical arrays
      curvefit_case_xxxxxx.json : metadata
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"curvefit_case_{case_id:06d}_{model_name}.npz"
    json_path = out_dir / f"curvefit_case_{case_id:06d}_{model_name}.json"

    # Keep intensity_sub optional because it can be large.
    arrays = {
        "intensity_bins": np.asarray(intensity_bins, dtype=np.float64),
        "intensity_counts": np.asarray(intensity_counts, dtype=np.float64),
        "expected": np.asarray(expected, dtype=np.float64),
        "lower_bounds": np.asarray(bounds[0], dtype=np.float64),
        "upper_bounds": np.asarray(bounds[1], dtype=np.float64),
    }

    if intensity_sub is not None:
        arrays["intensity_sub"] = np.asarray(intensity_sub, dtype=np.float32)

    np.savez_compressed(npz_path, **arrays)

    meta = {
        "case_id": case_id,
        "model_name": model_name,
        "npz_path": str(npz_path),
        "pol": pol,
        "block_ij": block_ij,
        "coord_local": coord,
        "coord_absolute": absolute_coord,
        "method": method,
        "threshold_scale": threshold_scale,
        "threshold_before_fit": threshold_before_fit,
        "idx_threshold": idx_threshold,
        "tau_mode_left": tau_mode_left,
        "tau_mode_right": tau_mode_right,
        "tau_amp_left": tau_amp_left,
        "tau_amp_right": tau_amp_right,
        "n_intensity_sub": None if intensity_sub is None else int(np.asarray(intensity_sub).size),
        "n_bins": None if intensity_bins is None else int(np.asarray(intensity_bins).size),
        "finite_counts": None if intensity_counts is None else int(np.isfinite(intensity_counts).sum()),
        "counts_min": None if intensity_counts is None else float(np.nanmin(intensity_counts)),
        "counts_max": None if intensity_counts is None else float(np.nanmax(intensity_counts)),
    }

    with open(json_path, "w") as f:
        json.dump(_json_safe(meta), f, indent=2)

    return str(npz_path), str(json_path)



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


def _multiotsu_from_hist_deterministic(intensity_bins, intensity_counts):
    """
    Deterministic 3-class Otsu threshold from an existing histogram.

    Returns two thresholds corresponding to skimage threshold_multiotsu(..., classes=3),
    but uses canonical histogram bins/counts instead of rebuilding histogram
    from samples.
    """
    x = np.asarray(intensity_bins, dtype=np.float64)
    h = np.asarray(intensity_counts, dtype=np.float64)

    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    h = np.round(h, 12)

    valid = np.isfinite(x) & np.isfinite(h) & (h >= 0)
    x = x[valid]
    h = h[valid]

    if x.size < 3 or np.sum(h) <= 0:
        return None

    # Normalize, but keep deterministic float64 path.
    p = h / np.sum(h)

    P = np.cumsum(p)
    S = np.cumsum(p * x)
    S2 = np.cumsum(p * x * x)

    total_mean = S[-1]

    best_score = -np.inf
    best_i = None
    best_j = None

    n = x.size

    # classes: [0:i], [i+1:j], [j+1:end]
    for i in range(0, n - 2):
        w0 = P[i]
        if w0 <= 0:
            continue

        m0 = S[i] / w0

        for j in range(i + 1, n - 1):
            w1 = P[j] - P[i]
            w2 = 1.0 - P[j]

            if w1 <= 0 or w2 <= 0:
                continue

            m1 = (S[j] - S[i]) / w1
            m2 = (S[-1] - S[j]) / w2

            score = (
                w0 * (m0 - total_mean) ** 2
                + w1 * (m1 - total_mean) ** 2
                + w2 * (m2 - total_mean) ** 2
            )

            # Deterministic tie rule:
            # if scores are extremely close, choose the lower thresholds.
            if score > best_score + 1e-15:
                best_score = score
                best_i = i
                best_j = j

    if best_i is None or best_j is None:
        return None

    return np.array([x[best_i], x[best_j]], dtype=np.float64)


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


def _otsu_threshold_from_hist(intensity_bins, intensity_counts):
    x = np.asarray(intensity_bins, dtype=np.float64)
    w = np.asarray(intensity_counts, dtype=np.float64)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[valid]
    w = w[valid]

    if x.size < 2 or np.sum(w) <= 0:
        return np.nan

    cw = np.cumsum(w)
    cx = np.cumsum(w * x)

    total_w = cw[-1]
    total_x = cx[-1]

    w0 = cw
    w1 = total_w - cw

    valid_split = (w0 > 0) & (w1 > 0)
    if not np.any(valid_split):
        return np.nan

    mu0 = np.zeros_like(x)
    mu1 = np.zeros_like(x)

    mu0[valid_split] = cx[valid_split] / w0[valid_split]
    mu1[valid_split] = (total_x - cx[valid_split]) / w1[valid_split]

    between_var = np.full_like(x, -np.inf)
    between_var[valid_split] = (
        w0[valid_split]
        * w1[valid_split]
        * (mu0[valid_split] - mu1[valid_split]) ** 2
    )

    idx = int(np.argmax(between_var))
    return float(x[idx])


def _valid_bimodal_fit_for_threshold(first_mode, second_mode):
    try:
        f = np.asarray(first_mode, dtype=np.float64).ravel()
        s = np.asarray(second_mode, dtype=np.float64).ravel()

        if f.size < 3 or s.size < 3:
            return False

        m1, sig1, amp1 = f[:3]
        m2, sig2, amp2 = s[:3]

        sep = abs(m2 - m1)

        return (
            np.all(np.isfinite([m1, sig1, amp1, m2, sig2, amp2]))
            and 0.05 <= sig1 <= 3.5
            and 0.05 <= sig2 <= 3.5
            and 0.01 <= amp1 <= 0.50
            and 0.01 <= amp2 <= 0.50
            and 0.5 <= sep <= 8.0
        )
    except Exception:
        return False


def determine_threshold(
        intensity,
        candidate_tile_coords,
        min_intensity_histogram=-32,
        max_intensity_histogram=0,
        step_histogram=0.1,
        bounds=None,
        method='ki',
        multi_threshold=True,
        adjust_if_nonoverlap=True,
        adjust_thresh_low_dist_percent=None,
        adjust_thresh_high_dist_percent=None,
        extract_curvefit=False,
        curvefit_out_dir=None,
        pol=None,
        block_ij=None,
        block_origin=None,
        threshold_scale=None,
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
    curvefit_case_id = 0
    for coord in candidate_tile_coords:

        # assume that coord consists of 5 elements
        ystart, yend, xstart, xend = coord[1:]

        # Debug trace for threshold changes after stage_3
        threshold_initial = None
        threshold_after_ki_or_otsu = None
        threshold_after_multiotsu = None
        threshold_after_bimodal_fit = None
        threshold_after_model_intersection = None
        threshold_after_trimodal_fit = None
        threshold_after_trimodal_override = None
        threshold_after_rg = None
        threshold_after_nonoverlap = None
        threshold_after_tile_relax = None
        threshold_after_margin_guard = None

        mode_after_bimodal_fit = None
        mode_after_model_intersection = None
        mode_after_trimodal_fit = None
        mode_after_margin_guard = None

        bimodal_params_debug = None
        trimodal_params_debug = None
        first_mode_debug = None
        second_mode_debug = None
        tri_first_mode_debug = None
        tri_second_mode_debug = None
        tri_third_mode_debug = None

        model_intersection_used = False
        trimodal_override_used = False
        tile_relax_used = False
        margin_guard_used = False
        hist_trimodal_override_used = False
        threshold_before_margin_guard = None
        intensity_sub = intensity[ystart:yend,
                                  xstart:xend]
        intensity_sub = np.asarray(intensity_sub, dtype=np.float64)
        # generate histogram with intensity higher than -35 dB
        if threshold_scale == "db":
            intensity_sub = intensity_sub[intensity_sub > -35]
        else:
            intensity_sub = intensity_sub[intensity_sub > 0]
        intensity_sub = _initial_threshold.remove_invalid(intensity_sub)
        intensity_sub = np.asarray(intensity_sub, dtype=np.float64)
        intensity_sub = intensity_sub[np.isfinite(intensity_sub)]

        if threshold_scale == "db":
            # All threshold fitting in dB uses 0.001 dB precision.
            intensity_sub = np.round(intensity_sub, 3).astype(np.float64)
        else:
            # Linear case keeps more precision.
            intensity_sub = np.round(intensity_sub, 8).astype(np.float64)

        if extract_curvefit and curvefit_out_dir is not None:

            _write_stage_record(curvefit_out_dir, {
                "stage": "stage_2_tile_prepare",
                "model_context": "before_histogram",
                "pol": pol,
                "block_ij": list(block_ij) if block_ij is not None else None,
                "coord_local": [int(ystart), int(yend), int(xstart), int(xend)],
                "coord_absolute": [
                    int(block_origin[0] + ystart),
                    int(block_origin[0] + yend),
                    int(block_origin[1] + xstart),
                    int(block_origin[1] + xend),
                ] if block_origin is not None else None,
                "n_after_filter": int(intensity_sub.size),
                "intensity_sub": _array_summary(intensity_sub),
                "intensity_sub_hash_round3": _array_hash(intensity_sub, 3),
                "intensity_sub_hash_round4": _array_hash(intensity_sub, 4),
                "min_intensity_histogram": _safe_float(min_intensity_histogram),
                "max_intensity_histogram": _safe_float(max_intensity_histogram),
                "step_histogram": _safe_float(step_histogram),
                "method": method,
                "threshold_scale": threshold_scale,
            })
        if intensity_sub.size == 0:
            threshold_array.append(np.nan)
            threshold_idx_array.append(-1)
            mode_array.append(np.nan)
            continue

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
        intensity_counts = np.asarray(intensity_counts, dtype=np.float64)
        intensity_bins = np.asarray(bins[:-1], dtype=np.float64)
        # If density=True produced NaNs (zero bin width / empty), retry with density=False
        if not np.isfinite(intensity_counts).any():
            intensity_counts, bins = np.histogram(intensity_sub, bins=bins, density=False)

        # Sanitize any remaining non-finites
        intensity_counts = np.nan_to_num(intensity_counts, nan=0.0, posinf=0.0, neginf=0.0)
        intensity_bins = bins[:-1]

        intensity_bins = np.asarray(intensity_bins, dtype=np.float64)
        intensity_counts = np.asarray(intensity_counts, dtype=np.float64)

        # Canonicalize histogram for reproducible peak search and curve_fit.
        intensity_bins = np.round(intensity_bins, 6).astype(np.float64)
        intensity_counts = np.round(intensity_counts, 12).astype(np.float64)
        if method == 'ki':
            threshold, idx_threshold, ki_prob_array = _initial_threshold.compute_ki_threshold_from_hist(
                intensity_bins,
                intensity_counts
            )

        elif method in ['otsu', 'rg']:
            threshold = threshold_otsu(intensity_sub)
        threshold_after_ki_or_otsu = _safe_float(threshold)
        threshold_initial = _safe_float(threshold)
        # get index of threshold from histogram.
        idx_threshold = np.searchsorted(intensity_bins, threshold)
        idx_threshold = int(np.clip(idx_threshold, 0, max(0, len(intensity_bins) - 1)))

        # if estimated threshold is higher than bounds,
        # re-estimate threshold assuming tri-mode distribution
        if threshold > bounds[1] and multi_threshold:
            thresholds = _multiotsu_from_hist_deterministic(
                intensity_bins,
                intensity_counts,
            )

            if thresholds is not None and thresholds[0] < threshold:
                threshold = float(thresholds[0])
                idx_threshold = np.searchsorted(intensity_bins, threshold)

        threshold_after_multiotsu = _safe_float(threshold)
        # Make sure idx_threshold is within bounds
        idx_threshold = int(np.clip(idx_threshold, 0, max(0, len(intensity_bins) - 1)))
        lowmaxind, tau_mode_left, tau_amp_left, low_peak_debug = \
            _pick_dominant_hist_mode(
                intensity_bins,
                intensity_counts,
                candidate_slice=slice(0, idx_threshold + 1),
                rel_tol=0.002,
                abs_tol=1e-4,
                merge_gap=5,
                smooth_sigma=1.0,
            )

        highmaxind, tau_mode_right, tau_amp_right, high_peak_debug = \
            _pick_dominant_hist_mode(
                intensity_bins,
                intensity_counts,
                candidate_slice=slice(idx_threshold, len(intensity_counts)),
                rel_tol=0.002,
                abs_tol=1e-4,
                merge_gap=5,
                smooth_sigma=1.0,
            )
        # # Low-side slice (<= threshold)
        # low_slice = intensity_counts[:idx_threshold + 1]
        # low_slice = np.nan_to_num(low_slice, nan=0.0, posinf=0.0, neginf=0.0)

        # if low_slice.size == 0 or not np.isfinite(low_slice).any() or np.all(low_slice == 0):
        #     lowmaxind = max(0, idx_threshold)
        #     lowmaxind_cands = np.array([], dtype=int)
        # else:
        #     lowmaxind_cands, _ = find_peaks(low_slice, distance=5)

        #     lowmaxind = _pick_stable_peak_index(
        #         low_slice,
        #         lowmaxind_cands,
        #         prefer="first",
        #         rel_tol=0.002,
        #         abs_tol=1e-4,
        #     )
        # # High-side slice (>= threshold)
        # high_slice = intensity_counts[idx_threshold:]
        # high_slice = np.nan_to_num(high_slice, nan=0.0, posinf=0.0, neginf=0.0)

        # if high_slice.size == 0 or not np.isfinite(high_slice).any() or np.all(high_slice == 0):
        #     highmaxind = idx_threshold
        #     highmaxind_cands = np.array([], dtype=int)
        # else:
        #     highmaxind_cands, _ = find_peaks(high_slice, distance=5)

        #     highmaxind_rel = _pick_stable_peak_index(
        #         high_slice,
        #         highmaxind_cands,
        #         prefer="first",
        #         rel_tol=0.002,
        #         abs_tol=1e-4,
        #     )
        #     highmaxind = idx_threshold + highmaxind_rel

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
        modevalue = float(tau_mode_left)
        optimization = False
        tri_optimization = False
        lock_mode_from_model = False
        try:

            expected = np.array(
                [tau_mode_left, .5, tau_amp_left,
                tau_mode_right, .5, tau_amp_right],
                dtype=np.float64
            )
            expected = np.round(expected, 12)

            fit_bounds = ((-30, 0.001, 0.01,
                        -30, 0.001, 0.01),
                        (5, 5, 0.95,
                        5, 5, 0.95))

            if extract_curvefit and curvefit_out_dir is not None:
                if block_origin is not None:
                    abs_coord = [
                        int(block_origin[0] + ystart),
                        int(block_origin[0] + yend),
                        int(block_origin[1] + xstart),
                        int(block_origin[1] + xend),
                    ]
                else:
                    abs_coord = None
                _write_stage_record(curvefit_out_dir, {
                    "stage": "stage_3_fit_ready",
                    "model_name": "bimodal",
                    "pol": pol,
                    "block_ij": list(block_ij) if block_ij is not None else None,
                    "coord_local": [int(ystart), int(yend), int(xstart), int(xend)],
                    "coord_absolute": [
                        int(block_origin[0] + ystart),
                        int(block_origin[0] + yend),
                        int(block_origin[1] + xstart),
                        int(block_origin[1] + xend),
                    ] if block_origin is not None else None,
                    "threshold_before_fit": _safe_float(threshold),
                    "idx_threshold": int(idx_threshold),
                    "tau_mode_left": _safe_float(tau_mode_left),
                    "tau_mode_right": _safe_float(tau_mode_right),
                    "tau_amp_left": _safe_float(tau_amp_left),
                    "tau_amp_right": _safe_float(tau_amp_right),
                    "intensity_bins": _array_summary(intensity_bins),
                    "intensity_bins_hash_round6": _array_hash(intensity_bins, 6),
                    "intensity_counts_hash_round12": _array_hash(intensity_counts, 12),
                    "intensity_counts": _array_summary(intensity_counts),
                    "expected": [float(x) for x in expected],
                    "lowmaxind": int(lowmaxind),
                    "highmaxind": int(highmaxind),
                    "low_peak_debug": low_peak_debug,
                    "high_peak_debug": high_peak_debug,
                    "ki_threshold_from_hist": _safe_float(threshold),
                    "ki_idx_from_hist": int(idx_threshold),

                })
                save_curve_fit_case(
                    curvefit_out_dir,
                    curvefit_case_id,
                    model_name="bimodal",
                    pol=pol,
                    block_ij=block_ij,
                    coord=[int(ystart), int(yend), int(xstart), int(xend)],
                    absolute_coord=abs_coord,
                    intensity_sub=intensity_sub,
                    intensity_bins=intensity_bins,
                    intensity_counts=intensity_counts,
                    expected=expected,
                    bounds=fit_bounds,
                    threshold_before_fit=float(threshold),
                    idx_threshold=int(idx_threshold),
                    tau_mode_left=float(tau_mode_left),
                    tau_mode_right=float(tau_mode_right),
                    tau_amp_left=float(tau_amp_left),
                    tau_amp_right=float(tau_amp_right),
                    method=method,
                    threshold_scale=threshold_scale,
                )
                curvefit_case_id += 1
            x_fit = np.asarray(intensity_bins, dtype=np.float64)
            y_fit = np.asarray(intensity_counts, dtype=np.float64)
            p0_fit = np.asarray(expected, dtype=np.float64)
            params, _ = curve_fit(
                _initial_threshold.bimodal,
                x_fit,
                y_fit,
                p0=p0_fit,
                bounds=fit_bounds
            )

            if params[0] > params[3]:
                second_mode = params[:3]
                first_mode = params[3:]
            else:
                first_mode = params[:3]
                second_mode = params[3:]

            bimodal_params_debug = [float(x) for x in np.asarray(params).ravel()]
            first_mode_debug = [float(x) for x in np.asarray(first_mode).ravel()]
            second_mode_debug = [float(x) for x in np.asarray(second_mode).ravel()]

            bimodal_fit_valid = _valid_bimodal_fit_for_threshold(
                first_mode,
                second_mode,
            )
            threshold_after_bimodal_fit = _safe_float(threshold)
            mode_after_bimodal_fit = _safe_float(tau_mode_left)
            min_sigma_for_model_intersection = 0.05

            use_model_intersection_this_tile = (
                USE_MODEL_INTERSECTION
                and bimodal_fit_valid
            )

            if use_model_intersection_this_tile:
                try:
                    threshold_model, mode_lower = _threshold_from_bimodal_fit(
                        first_mode, second_mode, bounds=bounds, p_low=0.98, p_high=0.02
                    )

                    model_intersection_used = True

                    MODEL_INTERSECTION_MAX_SHIFT_DB = 0.25
                    model_shift = abs(threshold_model - threshold)

                    if model_shift <= MODEL_INTERSECTION_MAX_SHIFT_DB:
                        threshold = threshold_model
                        modevalue = mode_lower
                        model_intersection_used = True
                    else:
                        model_intersection_used = False
                        threshold = threshold
                        modevalue = float(tau_mode_left)
                    idx_threshold = np.searchsorted(intensity_bins, threshold)
                    idx_threshold = int(np.clip(idx_threshold, 0, max(0, len(intensity_bins) - 1)))
                    threshold_after_model_intersection = _safe_float(threshold)
                    mode_after_model_intersection = _safe_float(modevalue)
                except Exception as e:
                    modevalue = float(tau_mode_left)
                    threshold_after_model_intersection = _safe_float(threshold)
                    mode_after_model_intersection = _safe_float(modevalue)
                    logger.info(f'Model-intersection override skipped (error: {e}); keeping KI/Otsu.')
            else:
                threshold_after_model_intersection = _safe_float(threshold)
                mode_after_model_intersection = _safe_float(modevalue if 'modevalue' in locals() else tau_mode_left)
            lock_mode_from_model = model_intersection_used

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

        except Exception as e:
            optimization = False
            logger.info(
                f'Bimodal curve Fitting fails in threshold computation: {e}')
            modevalue = tau_mode_left

            threshold_after_bimodal_fit = _safe_float(threshold)
            mode_after_bimodal_fit = _safe_float(modevalue)
            threshold_after_model_intersection = _safe_float(threshold)
            mode_after_model_intersection = _safe_float(modevalue)
        try:
            dividers = threshold_multiotsu(intensity_sub)

            expected = (dividers[0], .5, tau_amp_left,
                        dividers[1], .5, tau_amp_right,
                        (dividers[0]+dividers[1])/2, .5, 0.1)
            expected = np.array(
                [dividers[0], .5, tau_amp_left,
                dividers[1], .5, tau_amp_right,
                (dividers[0] + dividers[1]) / 2, .5, 0.1],
                dtype=np.float64
            )
            expected = np.round(expected, 12)
            fit_bounds = ((-35, 0.001, 0.01,
                        -35, 0.001, 0.01,
                        -35, 0.001, 0.01),
                        (5, 10, 0.95,
                        5, 10, 0.95,
                        5, 10, 0.95))

            if extract_curvefit and curvefit_out_dir is not None:
                if block_origin is not None:
                    abs_coord = [
                        int(block_origin[0] + ystart),
                        int(block_origin[0] + yend),
                        int(block_origin[1] + xstart),
                        int(block_origin[1] + xend),
                    ]
                else:
                    abs_coord = None

                save_curve_fit_case(
                    curvefit_out_dir,
                    curvefit_case_id,
                    model_name="trimodal",
                    pol=pol,
                    block_ij=block_ij,
                    coord=[int(ystart), int(yend), int(xstart), int(xend)],
                    absolute_coord=abs_coord,
                    intensity_sub=intensity_sub,
                    intensity_bins=intensity_bins,
                    intensity_counts=intensity_counts,
                    expected=expected,
                    bounds=fit_bounds,
                    threshold_before_fit=float(threshold),
                    idx_threshold=int(idx_threshold),
                    tau_mode_left=float(tau_mode_left),
                    tau_mode_right=float(tau_mode_right),
                    tau_amp_left=float(tau_amp_left),
                    tau_amp_right=float(tau_amp_right),
                    method=method,
                    threshold_scale=threshold_scale,
                )
                curvefit_case_id += 1

            x_fit = np.asarray(intensity_bins, dtype=np.float64)
            y_fit = np.asarray(intensity_counts, dtype=np.float64)
            p0_fit = np.asarray(expected, dtype=np.float64)
            params, _ = curve_fit(
                _initial_threshold.trimodal,
                    x_fit,
                    y_fit,
                    p0=p0_fit,
                    bounds=fit_bounds
                )

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
            tri_modes = [tri_first_mode, tri_second_mode, tri_third_mode]

            tri_sigmas = np.array([m[1] for m in tri_modes], dtype=np.float64)
            tri_amps = np.array([m[2] for m in tri_modes], dtype=np.float64)
            tri_means = np.array([m[0] for m in tri_modes], dtype=np.float64)

            trimodal_params_debug = [float(x) for x in np.asarray(params).ravel()]
            tri_first_mode_debug = [float(x) for x in np.asarray(tri_first_mode).ravel()]
            tri_second_mode_debug = [float(x) for x in np.asarray(tri_second_mode).ravel()]
            tri_third_mode_debug = [float(x) for x in np.asarray(tri_third_mode).ravel()]

            threshold_after_trimodal_fit = _safe_float(threshold)
            mode_after_trimodal_fit = _safe_float(modevalue)

            # simul_second_sum = np.sum(simul_second)
            # if simul_second_sum == 0:
            #     simul_second_sum = negligible_value
            # converge_ind = np.where((intensity_bins < tau_mode_right)
            #                         & (intensity_bins > tau_mode_left)
            #                         & (intensity_bins < threshold)
            #                         & (np.cumsum(simul_second) /
            #                            simul_second_sum < 0.03))

            # if len(converge_ind[0]):
            #     modevalue = intensity_bins[converge_ind[0][-1]]

            # else:
            #     modevalue = tau_mode_left

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
            tri_means = np.array([
                tri_first_mode[0],
                tri_second_mode[0],
                tri_third_mode[0],
            ], dtype=np.float64)

            tri_sigmas = np.array([
                tri_first_mode[1],
                tri_second_mode[1],
                tri_third_mode[1],
            ], dtype=np.float64)

            tri_amps = np.array([
                tri_first_mode[2],
                tri_second_mode[2],
                tri_third_mode[2],
            ], dtype=np.float64)

            tri_fit_valid = (
                np.all(np.isfinite(tri_means))
                and np.all(np.isfinite(tri_sigmas))
                and np.all(np.isfinite(tri_amps))
                and np.all(tri_sigmas >= 0.05)
                and np.all(tri_sigmas <= 3.5)
                and np.all(tri_amps >= 0.01)
                and np.all(tri_amps <= 0.50)
                            )

            tri_modes_separated = (
                abs(tri_second_mode[0] - tri_first_mode[0]) >= 1.0
                and abs(tri_third_mode[0] - tri_second_mode[0]) >= 1.0
            )

            tri_middle_reasonable = (
                tri_second_mode[1] <= 3.0
                and tri_second_mode[2] >= 0.02
            )

            degenerate_trimodal = not (
                tri_fit_valid
                and tri_modes_separated
                and tri_middle_reasonable
            )
            intensity_sub_min = np.nanmin(intensity_sub)
            if (
                (not degenerate_trimodal)
                and tri_ratio_bool
                and third_second_dist_bool
                and first_second_dist_bool
                and tri_first_mode[0] > -32
                and intensity_sub_min < tri_second_mode[0]
            ):
                tri_optimization = True

        except Exception as e:
            logger.info(
                "Trimodal curve fitting failed in threshold computation: %s",
                e,
            )
            tri_optimization = False
            trimodal_params_debug = None
            tri_first_mode_debug = None
            tri_second_mode_debug = None
            tri_third_mode_debug = None
            logger.info(
                'Trimodal curve Fitting fails in threshold computation.')
        threshold_after_trimodal_fit = _safe_float(threshold)
        mode_after_trimodal_fit = _safe_float(modevalue)

        ENABLE_HIST_TRIMODAL_OVERRIDE = True

        hist_trimodal_override_used = False
        threshold_before_hist_trimodal = float(threshold)

        if ENABLE_HIST_TRIMODAL_OVERRIDE and multi_threshold:
            try:
                # Use deterministic histogram-based multi-Otsu.
                dividers = _multiotsu_from_hist_deterministic(
                    intensity_bins,
                    intensity_counts,
                )

                if dividers is not None and len(dividers) >= 2:
                    low_mid_divider = float(dividers[0])
                    mid_high_divider = float(dividers[1])

                    hist_mask = intensity_bins < mid_high_divider

                    if (
                        np.count_nonzero(hist_mask) > 2
                        and np.sum(intensity_counts[hist_mask]) > 0
                    ):
                        tau_bound_hist = _otsu_threshold_from_hist(
                            intensity_bins[hist_mask],
                            intensity_counts[hist_mask],
                        )

                        TRI_Q = 0.01

                        threshold_before_q = (
                            np.round(threshold_before_hist_trimodal / TRI_Q) * TRI_Q
                        )
                        proposed_q = (
                            np.round(tau_bound_hist / TRI_Q) * TRI_Q
                        )

                        shift_db = abs(proposed_q - threshold_before_q)

                        # Conservative: do not accept one-bin / borderline corrections.
                        min_trimodal_shift_db = 0.11
                        max_trimodal_shift_db = 0.25

                        accept_hist_trimodal = (
                            np.isfinite(proposed_q)
                            and threshold_before_q > proposed_q
                            and min_trimodal_shift_db <= shift_db <= max_trimodal_shift_db
                            and proposed_q >= bounds[0]
                            and proposed_q <= bounds[1]
                        )

                        if accept_hist_trimodal:
                            threshold = float(proposed_q)
                            trimodal_override_used = True
                            hist_trimodal_override_used = True
                        else:
                            threshold = float(threshold_before_q)
                            trimodal_override_used = False
                            hist_trimodal_override_used = False
                    else:
                        threshold = float(
                            np.round(threshold_before_hist_trimodal / 0.01) * 0.01
                        )
                        trimodal_override_used = False
                        hist_trimodal_override_used = False

            except Exception as e:
                threshold = float(
                    np.round(threshold_before_hist_trimodal / 0.01) * 0.01
                )
                trimodal_override_used = False
                hist_trimodal_override_used = False
                logger.info(f'Histogram trimodal override skipped: {e}')
        threshold_after_trimodal_override = _safe_float(threshold)
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

                diff_dist[:int(lowmaxind)] = np.nan
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
                            rms_xx.append(hist_bin)
                valid = np.isfinite(rms_sss) & np.isfinite(rms_xx)
                if np.any(valid):
                    valid_indices = np.flatnonzero(valid)
                    best_index = valid_indices[np.argmin(rms_sss[valid])]
                    rg_tolerance = rms_xx[best_index]
                    threshold = 0.5 * (rg_tolerance + threshold)

        threshold_after_rg = _safe_float(threshold)
        threshold_before_nonoverlap = None
        if adjust_if_nonoverlap:
            old_threshold = threshold
            threshold_before_nonoverlap = None
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
            threshold_before_nonoverlap = _safe_float(old_threshold)
        threshold_after_nonoverlap = _safe_float(threshold)
        # --- TILE-WISE RELAXED THRESHOLD (data-driven; optional) ---
        threshold_before_tile_relax = None
        threshold_before_tile_relax = _safe_float(threshold)

        if TILE_RELAX:
            try:
                threshold_before_relax = float(threshold)

                tau_relaxed = _tile_relaxed_threshold_from_boundary(
                    intensity_tile_db=intensity_sub,
                    tau_strict=threshold_before_relax,
                    max_band_px=3,
                    core_iter=1,
                )

                if np.isfinite(tau_relaxed):
                    tau_relaxed = float(tau_relaxed)

                    max_tile_relax_shift_db = 0.5
                    relax_shift_db = abs(tau_relaxed - threshold_before_relax)

                    if relax_shift_db <= max_tile_relax_shift_db:
                        if relax_shift_db > 1e-12:
                            tile_relax_used = True
                        threshold = tau_relaxed
                    else:
                        tile_relax_used = False
                        threshold = threshold_before_relax

            except Exception as e:
                logger.info(f'Tile relaxed threshold skipped: {e}')

        threshold_after_tile_relax = _safe_float(threshold)

        idx_threshold = np.searchsorted(intensity_bins, threshold)
        idx_threshold = int(np.clip(idx_threshold, 0, max(0, len(intensity_bins) - 1)))

        # Recompute the lower slice up to the final threshold
        low_slice_final = np.nan_to_num(intensity_counts[:idx_threshold + 1], nan=0.0)

        if low_slice_final.size == 0 or np.all(low_slice_final == 0):
            hist_mode_final = intensity_bins[max(0, idx_threshold)]
        else:
            lowmaxind_final, hist_mode_final, _, low_final_debug = \
                _pick_dominant_hist_mode(
                    intensity_bins,
                    intensity_counts,
                    candidate_slice=slice(0, idx_threshold + 1),
                    rel_tol=0.002,
                    abs_tol=1e-4,
                    merge_gap=5,
                    smooth_sigma=1.0,
                )

        # Critical reproducibility rule:
        # Do not use first_mode[0] for final mode unless the fitted model was
        # actually accepted to modify the threshold.
        if model_intersection_used and optimization and bimodal_fit_valid:
            modevalue_final = float(tau_mode_left)
        else:
            modevalue_final = float(hist_mode_final)
        FINAL_QUANTIZE_DB = 0.01

        if threshold_scale == "db":
            if np.isfinite(threshold):
                threshold = np.round(threshold / FINAL_QUANTIZE_DB) * FINAL_QUANTIZE_DB

            if np.isfinite(modevalue_final):
                modevalue_final = np.round(modevalue_final / FINAL_QUANTIZE_DB) * FINAL_QUANTIZE_DB

        threshold_before_margin_guard = _safe_float(threshold)
        margin = 0.05
        if np.isfinite(modevalue_final) and np.isfinite(threshold) and (modevalue_final >= threshold):
            margin_guard_used = True
            threshold = modevalue_final + margin
            idx_threshold = np.searchsorted(intensity_bins, threshold)
            idx_threshold = int(np.clip(idx_threshold, 0, max(0, len(intensity_bins) - 1)))
        else:
            margin_guard_used = False
        modevalue = modevalue_final

        threshold_after_margin_guard = _safe_float(threshold)
        mode_after_margin_guard = _safe_float(modevalue)


        # Quantize final outputs
        if threshold_scale == "db":
            threshold = np.round(threshold / FINAL_QUANTIZE_DB) * FINAL_QUANTIZE_DB if np.isfinite(threshold) else threshold
            modevalue = np.round(modevalue / FINAL_QUANTIZE_DB) * FINAL_QUANTIZE_DB if np.isfinite(modevalue) else modevalue
        if extract_curvefit and curvefit_out_dir is not None:
            _write_stage_record(curvefit_out_dir, {
                "stage": "stage_4_threshold_final",
                "pol": pol,
                "block_ij": list(block_ij) if block_ij is not None else None,
                "coord_local": [int(ystart), int(yend), int(xstart), int(xend)],
                "coord_absolute": [
                    int(block_origin[0] + ystart),
                    int(block_origin[0] + yend),
                    int(block_origin[1] + xstart),
                    int(block_origin[1] + xend),
                ] if block_origin is not None else None,

                "threshold_final": _safe_float(threshold),
                "idx_threshold_final": int(idx_threshold),
                "mode_final": _safe_float(modevalue),

                # Main threshold trace
                "threshold_initial": threshold_initial,
                "threshold_after_ki_or_otsu": threshold_after_ki_or_otsu,
                "threshold_after_multiotsu": threshold_after_multiotsu,
                "threshold_after_bimodal_fit": threshold_after_bimodal_fit,
                "threshold_after_model_intersection": threshold_after_model_intersection,
                "threshold_after_trimodal_fit": threshold_after_trimodal_fit,
                "threshold_after_trimodal_override": threshold_after_trimodal_override,
                "threshold_after_rg": threshold_after_rg,
                "threshold_before_nonoverlap": threshold_before_nonoverlap,
                "threshold_after_nonoverlap": threshold_after_nonoverlap,
                "threshold_before_tile_relax": threshold_before_tile_relax,
                "threshold_after_tile_relax": threshold_after_tile_relax,
                "threshold_before_margin_guard": threshold_before_margin_guard,
                "threshold_after_margin_guard": threshold_after_margin_guard,
                "hist_trimodal_override_used": bool(hist_trimodal_override_used),
                # Mode trace
                "mode_after_bimodal_fit": mode_after_bimodal_fit,
                "mode_after_model_intersection": mode_after_model_intersection,
                "mode_after_trimodal_fit": mode_after_trimodal_fit,
                "mode_after_margin_guard": mode_after_margin_guard,

                # Flags
                "optimization": bool(optimization),
                "tri_optimization": bool(tri_optimization),
                "model_intersection_used": bool(model_intersection_used),
                "trimodal_override_used": bool(trimodal_override_used),
                "tile_relax_used": bool(tile_relax_used),
                "margin_guard_used": bool(margin_guard_used),
                "threshold_scale": threshold_scale,

                # Fit params
                "bimodal_params": bimodal_params_debug,
                "first_mode": first_mode_debug,
                "second_mode": second_mode_debug,
                "trimodal_params": trimodal_params_debug,
                "tri_first_mode": tri_first_mode_debug,
                "tri_second_mode": tri_second_mode_debug,
                "tri_third_mode": tri_third_mode_debug,
            })

        if threshold_scale == "db":
            if np.isfinite(threshold):
                threshold = np.round(threshold / FINAL_QUANTIZE_DB) * FINAL_QUANTIZE_DB

            if np.isfinite(modevalue):
                modevalue = np.round(modevalue / FINAL_QUANTIZE_DB) * FINAL_QUANTIZE_DB
        else:
            # Linear case: keep more precision, or quantize in dB then convert back if needed.
            if np.isfinite(threshold):
                threshold = np.round(threshold, 8)

            if np.isfinite(modevalue):
                modevalue = np.round(modevalue, 8)

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
                  thres_max=None,
                  extract_curvefit=False,
                  curvefit_out_dir=None,
                  block_ij=None,
                  block_origin=None):
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
    multi_threshold_flag = threshold_cfg.multi_threshold
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
        src_im = np.asarray(intensity[polind], dtype=np.float32)
        src_im = np.round(src_im, 8).astype(np.float32)
        tile_selection_im = src_im.copy()
        tile_selection_im[~np.isfinite(tile_selection_im)] = np.nan
        tile_selection_im = np.round(tile_selection_im, 8).astype(np.float32)

        if pol in ['VV', 'VH', 'HH', 'HV', 'span', 'ratio']:
            if threshold_scale == 'db':
                with np.errstate(divide="ignore", invalid="ignore"):
                    target_im = _initial_threshold.convert_pow2db(src_im)

            else:
                target_im = src_im
        else:
            target_im = src_im

        target_im = np.asarray(target_im, dtype=np.float32)
        tile_im = np.asarray(target_im, dtype=np.float32).copy()

        # Remove non-finite values before tile-selection metrics
        tile_im[~np.isfinite(tile_im)] = np.nan
        if threshold_scale == "db":
            tile_im = np.round(tile_im, 3).astype(np.float32)
        else:
            tile_im = np.round(tile_im, 8).astype(np.float32)
        debug_dir = os.path.join(
            cfg.groups.product_path_group.scratch_path,
            "curvefit_debug"
        )
        debug_curvefit = False
        if debug_curvefit:
            _write_stage_record(debug_dir, {
                "stage": "stage_1_before_tile_selection",
                "pol": pol,
                "polind": int(polind),
                "block_ij": list(block_ij) if block_ij is not None else None,
                "input_to_tile_selection": _array_summary(tile_im),
                "tile_selection_input_linear": _array_summary(tile_selection_im),
                "threshold_input_db": _array_summary(tile_im),
                "water_mask": _array_summary(water_body_subset),
                "winsize": int(winsize),
                "selection_methods": tile_selection_method,
            })

        tile_selection_object.debug_tile_metric = False
        tile_selection_object.debug_metric_dir = debug_dir
        tile_selection_object.debug_context = {
            "stage": "stage_1_tile_metric",
            "pol": pol,
            "polind": int(polind),
            "block_ij": list(block_ij) if block_ij is not None else None,
            "block_origin": list(block_origin) if block_origin is not None else None,
            "threshold_scale": threshold_scale,
            "selection_methods": tile_selection_method,
        }
        candidate_tile_coords = tile_selection_object.tile_selection_wbd(
                        intensity=tile_selection_im,
                        water_mask=water_body_subset,
                        win_size=winsize,
                        selection_methods=tile_selection_method)
        candidate_tile_coords_arr = np.asarray(candidate_tile_coords)

        if debug_curvefit:

            _write_stage_record(debug_dir, {
                "stage": "stage_1_after_tile_selection",
                "pol": pol,
                "polind": int(polind),
                "block_ij": list(block_ij) if block_ij is not None else None,
                "n_candidate_tiles": int(len(candidate_tile_coords)),
                "candidate_tile_coords": candidate_tile_coords_arr.tolist(),
            })
        if len(candidate_tile_coords) > 0:
            target_im = tile_im
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
                multi_threshold=multi_threshold_flag,
                adjust_if_nonoverlap=adjust_threshold_flag,
                adjust_thresh_low_dist_percent=low_dist_percentile,
                adjust_thresh_high_dist_percent=high_dist_percentile,
                extract_curvefit=extract_curvefit,
                curvefit_out_dir=curvefit_out_dir,
                pol=pol,
                block_ij=block_ij,
                block_origin=block_origin,
                threshold_scale=threshold_scale,
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
    if valid_pixel_number == 0:
        logger.warning("No valid pixels found for water spatial coverage.")
        return 0.0
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
    _initial_threshold.fill_threshold_with_gdal(
        threshold_array=threshold_dict,
        rows=rows,
        cols=cols,
        filename=filename_threshold,
        outputdir=outputdir,
        pol_list=pol_list,
        filled_value=None,
        no_data=no_data,
        average_tile=average_tile,
        smooth_sigma=0.75,
        resample_alg="bilinear",
        save_cog=True,
    )

    # 2) Interpolate delta -> D  (temporary filename)
    tmp_delta_name = filename_threshold + "_delta"
    _initial_threshold.fill_threshold_with_gdal(
        threshold_array=delta_dict,
        rows=rows,
        cols=cols,
        filename=tmp_delta_name,
        outputdir=outputdir,
        pol_list=pol_list,
        filled_value=None,
        no_data=no_data,
        average_tile=average_tile,
        smooth_sigma=0.75,
        resample_alg="bilinear",
        save_cog=False,
    )

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
                                   compression='DEFLATE', nbits=None)


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
                                            y_size,
                                            buf_type=gdal.GDT_Float32
                                            )
    image_sub = np.asarray(image_sub, dtype=np.float32)
    if getattr(cfg.groups.processing.initial_threshold, "force_float16_debug", False):
        image_sub = image_sub.astype(np.float16).astype(np.float32)
    filt_raster_tif = None

    wbd_gdal = gdal.Open(wbd_im_str)
    wbd_sub = wbd_gdal.ReadAsArray(jj * block_col,
                                   ii * block_row,
                                   x_size,
                                   y_size,
                                   buf_type=gdal.GDT_Byte
                                )
    wbd_gdal = None
    wbd_sub = np.asarray(wbd_sub, dtype=np.uint8)
    debug_dir = os.path.join(
        cfg.groups.product_path_group.scratch_path,
        "curvefit_debug"
    )
    debug_curvefit = False
    if debug_curvefit:
        _write_stage_record(debug_dir, {
            "stage": "stage_0_block_read",
            "block_ij": [int(ii), int(jj)],
            "block_origin": [int(ii * block_row), int(jj * block_col)],
            "x_size": int(x_size),
            "y_size": int(y_size),
            "image": _array_summary(image_sub),
            "wbd": _array_summary(wbd_sub),
        })
    block_origin = (ii * block_row, jj * block_col)
    block_ij = (ii, jj)
    curvefit_out_dir = os.path.join(
        cfg.groups.product_path_group.scratch_path,
        "curvefit_cases"
    )

    threshold_tau_block, mode_tau_block, candidate_tile_coords = \
        run_sub_block(
            image_sub,
            wbd_sub,
            cfg,
            thres_max=thres_max,
            extract_curvefit=False,
            curvefit_out_dir=curvefit_out_dir,
            block_ij=block_ij,
            block_origin=block_origin)

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
            threshold_tau_set = [[] for _ in range(band_number)]
            mode_tau_set = [[] for _ in range(band_number)]
            coord_row_list = [[] for _ in range(band_number)]
            coord_col_list = [[] for _ in range(band_number)]
            window_coord_list = [[] for _ in range(band_number)]

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
        intensity_whole = np.asarray(intensity_whole, dtype=np.float32)
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


    cfg = RunConfig.load_from_yaml(args.input_yaml[0],
                                    'dswx_ni', args)

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
