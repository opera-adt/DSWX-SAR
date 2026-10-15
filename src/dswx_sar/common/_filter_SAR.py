"""SAR filters with explicit validity and scale handling.

Inputs are 2-D positive LINEAR intensities or polarization ratios, not dB.
NaN, infinity, zero (including block padding), and negative values are invalid.
All filter outputs stay on the input grid, preserve its positive finite pixel
footprint, and leave the input array unchanged. Failed estimates fall back to
the original value at valid pixels.

Lee uses a masked arithmetic mean. Chambolle retains the original dB domain.
Bregman retains the original log10 domain so existing positive lambda_value
settings keep their scale. Guided filtering retains logarithmic guidance with
a linear source; eps therefore remains in squared dB units.

TV and guided filters temporarily extend nearest valid values into missing
areas and restore missing pixels afterward. This avoids a synthetic -30 edge;
it is still an extension, not a fully masked TV or guided optimization.

Example integration in the IV script:
    from dswx_sar.common import _filter_SAR_corrected as _filter_SAR
"""

import numpy as np
from scipy import ndimage, signal

K_DEFAULT = 1.0
CU_DEFAULT = 0.523
CMAX_DEFAULT = 1.73


def _linear_image(data):
    image = np.asarray(data, dtype=np.float64)
    if image.ndim != 2 or 0 in image.shape:
        raise ValueError("Expected a nonempty 2-D linear SAR image or ratio")
    valid = np.isfinite(image) & (image > 0)
    return image, valid


def _positive_parameter(value, name, allow_zero=False):
    value = float(value)
    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        bound = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {bound}")
    return value


def _window_size(value):
    size = int(value)
    if size < 1 or size != value:
        raise ValueError("window_size must be a positive integer")
    return size


def _restore_valid_pixels(filtered, original, valid):
    result = np.array(filtered, dtype=np.float64, copy=True)
    if result.shape != original.shape:
        raise ValueError("The filter changed the input array shape")
    failed = valid & (~np.isfinite(result) | (result <= 0))
    result[failed] = original[failed]
    result[~valid] = np.nan
    return result


def _nearest_valid_extension(values, valid):
    """Extend solver inputs only; callers restore the original missing mask."""
    if valid.all():
        return np.array(values, copy=True)
    indices = ndimage.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    return np.asarray(values)[tuple(indices)].copy()


def masked_convolve2d(array, window, *args, **kwargs):
    """Masked averaging for a nonnegative, normalized 2-D averaging kernel.

    Retains the original call signature. Available kernel support is used for
    normalization, including at array boundaries. Windows with more than 20%
    missing support return NaN; this helper computes statistics, not a map.
    """
    image, valid = _linear_image(array)
    kernel = np.asarray(window, dtype=np.float64)
    if (kernel.ndim != 2 or not np.isfinite(kernel).all()
            or np.any(kernel < 0) or kernel.sum() <= 0):
        raise ValueError("window must be a finite, nonnegative averaging kernel")
    kernel = kernel / kernel.sum()
    numerator = signal.convolve(np.where(valid, image, 0.0), kernel,
                                *args, **kwargs)
    denominator = signal.convolve(valid.astype(np.float64), kernel,
                                  *args, **kwargs)
    available = signal.convolve(np.ones(image.shape), kernel,
                                *args, **kwargs)
    mean = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=mean, where=denominator > 0)
    fraction = np.zeros(numerator.shape)
    np.divide(denominator, available, out=fraction, where=available > 0)
    mean[fraction < 0.8 - 1e-12] = np.nan
    return mean.astype(np.float32)


def compute_window_mean_std_fast(arr, winsize, nan_frac_thresh=0.2):
    """Mean/std over valid pixels, with unavailable border support excluded.

    The mean and second moment are divided by the actual valid support rather
    than by the full window area. Insufficient support produces NaN statistics;
    the Lee filter explicitly falls back to a valid original center value.
    """
    image, valid = _linear_image(arr)
    size = _window_size(winsize)
    threshold = float(nan_frac_thresh)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("nan_frac_thresh must be between 0 and 1")
    options = dict(size=size, mode="constant", cval=0.0)
    finite_image = np.where(valid, image, 0.0)
    sum_average = ndimage.uniform_filter(finite_image, **options)
    square_average = ndimage.uniform_filter(finite_image * finite_image, **options)
    valid_support = ndimage.uniform_filter(valid.astype(np.float64), **options)
    available_support = ndimage.uniform_filter(np.ones(image.shape), **options)

    # A tiny floating-point residual is not a supported window.
    has_support = valid_support > np.finfo(np.float64).eps
    mean = np.full(image.shape, np.nan)
    second_moment = np.full(image.shape, np.nan)
    np.divide(sum_average, valid_support, out=mean, where=has_support)
    np.divide(square_average, valid_support, out=second_moment,
              where=has_support)
    variance = np.maximum(second_moment - mean * mean, 0.0)
    std = np.sqrt(variance)

    valid_fraction = np.zeros(image.shape)
    np.divide(valid_support, available_support, out=valid_fraction,
              where=available_support > 0)
    insufficient = (~has_support) | (valid_fraction < 1.0 - threshold - 1e-12)
    mean[insufficient] = np.nan
    std[insufficient] = np.nan
    return mean, std


def compute_window_mean_std(arr, winsize):
    """Compatibility wrapper for the masked local statistics."""
    return compute_window_mean_std_fast(arr, winsize)


def weightingarr(im, winsize, k=K_DEFAULT, cu=CU_DEFAULT, cmax=CMAX_DEFAULT,
                 nan_frac_thresh=0.2):
    """Enhanced Lee weights; failed window estimates get zero weight."""
    image, _ = _linear_image(im)
    k = _positive_parameter(k, "k", allow_zero=True)
    cu = _positive_parameter(cu, "cu", allow_zero=True)
    cmax = _positive_parameter(cmax, "cmax")
    if cmax <= cu:
        raise ValueError("cmax must exceed cu")
    mean, std = compute_window_mean_std_fast(
        image, winsize, nan_frac_thresh=nan_frac_thresh
    )
    ci = np.full(image.shape, np.inf)
    usable = np.isfinite(mean) & np.isfinite(std) & (mean > 0)
    np.divide(std, mean, out=ci, where=usable)
    weights = np.zeros(image.shape)
    weights[usable & (ci <= cu)] = 1.0
    transition = usable & (ci > cu) & (ci < cmax)
    ci_transition = ci[transition]
    weights[transition] = np.exp(
        -k * (ci_transition - cu) / (cmax - ci_transition)
    )
    return weights, mean, std


def lee_enhanced_filter(img, k=K_DEFAULT, cu=CU_DEFAULT, cmax=CMAX_DEFAULT,
                        **kwargs):
    """Enhanced Lee in linear units, preserving valid center pixels.

    window_size defaults to 3; nan_frac_thresh defaults to 0.2. Original Lee
    weight constants are retained. Low coefficient-of-variation windows still
    use the local mean, so these fixes do not disable ordinary smoothing.
    """
    image, valid = _linear_image(img)
    weights, mean, _ = weightingarr(
        image, kwargs.get("window_size", 3), k, cu, cmax,
        nan_frac_thresh=kwargs.get("nan_frac_thresh", 0.2),
    )
    result = image.copy()
    update = valid & np.isfinite(mean) & (weights > 0)
    result[update] = (mean[update] * weights[update]
                      + image[update] * (1.0 - weights[update]))
    return _restore_valid_pixels(result, image, valid)


def fill_nan_value(data):
    """Legacy explicit -30 fill, now returning a copy instead of mutating data.

    The filters below use nearest-valid extension instead of this sentinel.
    """
    result = np.array(data, copy=True)
    result[~np.isfinite(result)] = -30
    return result


def anisotropic_diffusion(img, **kwargs):
    """Legacy method name for Chambolle TV in dB, returning linear units.

    weight defaults to 1.0 in the original dB domain. Larger weights produce
    more smoothing. weight=0 bypasses denoising; negative weights are rejected.
    """
    image, valid = _linear_image(img)
    weight = _positive_parameter(kwargs.get("weight", 1.0), "weight",
                                  allow_zero=True)
    if not valid.any() or weight == 0:
        return _restore_valid_pixels(image, image, valid)
    from skimage.restoration import denoise_tv_chambolle

    db = np.full(image.shape, np.nan)
    db[valid] = 10.0 * np.log10(image[valid])
    extended = _nearest_valid_extension(db, valid)
    filtered_db = denoise_tv_chambolle(
        extended, weight=weight,
        eps=kwargs.get("eps", 2e-4),
        max_num_iter=kwargs.get("max_num_iter", 200),
        channel_axis=None,
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        filtered = np.power(10.0, filtered_db / 10.0)
    return _restore_valid_pixels(filtered, image, valid)


def guided_filter(img, **kwargs):
    """Logarithmic guidance with a linear source, returning linear units.

    radius=1 and eps=3.0 retain the original parameter conventions. The guide
    and source are float32 copies for OpenCV; invalid input is never overwritten.
    eps is in squared dB units because the guidance image is in dB.
    """
    image, valid = _linear_image(img)
    if not valid.any():
        return _restore_valid_pixels(image, image, valid)
    radius = int(kwargs.get("radius", 1))
    if radius < 0 or radius != kwargs.get("radius", 1):
        raise ValueError("radius must be a nonnegative integer")
    eps = _positive_parameter(kwargs.get("eps", 3.0), "eps")
    depth = kwargs.get("ddepth", -1)
    import cv2

    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "guidedFilter"):
        raise ImportError("guided_filter requires OpenCV with ximgproc")
    source = _nearest_valid_extension(image, valid).astype(np.float32)
    guide = (10.0 * np.log10(source)).astype(np.float32)
    filtered = cv2.ximgproc.guidedFilter(
        guide=guide, src=source, radius=radius, eps=eps, dDepth=depth
    )
    return _restore_valid_pixels(filtered, image, valid)


def tv_bregman(X, **kwargs):
    """Bregman TV in the original log10 domain, returning linear units.

    lambda_value must be positive and defaults to the documented 20.0 instead
    of the previous invalid -1. Larger lambda_value means LESS smoothing.
    Keeping log10 rather than 10*log10 preserves existing positive settings;
    Bregman and Chambolle weights are not interchangeable.
    """
    image, valid = _linear_image(X)
    weight = _positive_parameter(kwargs.get("lambda_value", 20.0),
                                 "lambda_value")
    if not valid.any():
        return _restore_valid_pixels(image, image, valid)
    from skimage.restoration import denoise_tv_bregman

    logged = np.full(image.shape, np.nan)
    logged[valid] = np.log10(image[valid])
    extended = _nearest_valid_extension(logged, valid)
    filtered_log = denoise_tv_bregman(
        extended, weight=weight,
        eps=kwargs.get("eps", 1e-3),
        max_num_iter=kwargs.get("max_num_iter", 100),
        isotropic=kwargs.get("isotropic", True),
        channel_axis=None,
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        filtered = np.power(10.0, filtered_log)
    return _restore_valid_pixels(filtered, image, valid)
