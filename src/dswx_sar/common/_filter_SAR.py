import cv2
import numpy as np
from scipy import ndimage, signal
from scipy.interpolate import griddata
from skimage.restoration import denoise_tv_chambolle, denoise_tv_bregman

from dswx_sar.common._dswx_sar_util import Constants

K_DEFAULT = 1.0
CU_DEFAULT = 0.523
CMAX_DEFAULT = 1.73


def masked_convolve2d(array, window, *args, **kwargs):
    '''Perform convolution without spreading nan value to the neighbor pixels

    Parameters
    ----------
    array: numpy.ndarray
        2 dimensional array
    window: integer
        2 dimensional window
    '''
    frames_complex = np.zeros_like(array, dtype=np.complex64)
    frames_complex[np.isnan(array)] = np.array((1j))
    frames_complex[np.bitwise_not(np.isnan(array))] = \
        array[np.bitwise_not(np.isnan(array))]

    convolved_array = signal.convolve(frames_complex, window, *args, **kwargs)
    convolved_array[np.imag(convolved_array) > 0.2] = np.nan
    convolved_array = convolved_array.real.astype(np.float32)

    return convolved_array


def compute_window_mean_std(arr, winsize):
    '''
    Compute mean and standard deviation within window size
    by moving the window from 2 dimensional array.

    Parameters
    ----------
    arr: numpy.ndarray
        2 dimensional array
    winsize: integer
        window size to compute the mean and std.

    Returns
    -------
    mean: numpy.ndarray
        mean array
    std: numpy.ndarray
        std array
    '''
    window = np.ones([winsize, winsize]) / (winsize * winsize)
    arr_masked = np.ma.masked_equal(arr, np.nan)
    mean = masked_convolve2d(arr_masked, window, mode='same')
    c2 = masked_convolve2d(arr_masked*arr_masked, window, mode='same')

    var = c2 - mean * mean

    # The negative number in sqrt is replaced
    # with the negligibly small number to avoid numpy warning message.
    var = np.where(var < 0, Constants.negligible_value, var)
    std = var ** .5

    return mean, std


def weightingarr(im, winsize, k=K_DEFAULT,
                 cu=CU_DEFAULT, cmax=CMAX_DEFAULT):
    """
    Computes the weighthing function for Lee filter using cu as the noise
    coefficient.

    Parameters
    ----------
    im: numpy.ndarray
        2 dimensional array
    winsize: integer
        window size to compute the mean and std.

    Returns
    -------
    w_t_arr: numpy.ndarray
    window_mean: numpy.ndarray
        mean array
    window_std: numpy.ndarray
        std array
    """
    # cu is the noise variation coefficient
    window_mean, window_std = compute_window_mean_std(im, winsize=winsize)

    # ci is the variation coefficient in the window
    ci = window_std / window_mean
    w_t_arr = np.zeros(im.shape)
    w_t_arr[ci <= cu] = 1
    w_t_arr[(ci > cu) & (ci < cmax)] =\
        np.exp((-k * (ci[(ci > cu) & (ci < cmax)] - cu))
               / (cmax - ci[(ci > cu) & (ci < cmax)]))
    w_t_arr[ci >= cmax] = 0

    return w_t_arr, window_mean, window_std


def lee_enhanced_filter(img, k=K_DEFAULT, cu=CU_DEFAULT,
                        cmax=CMAX_DEFAULT, **kwargs):
    """
    Enhanced Lee filter for SAR image

    Zhu, J., Wen, J., & Zhang, Y. (2013, December). A new algorithm for SAR
    image despeckling using an enhanced Lee filter and median filter.
    In 2013 6th International congress on image and signal processing
    (CISP) (Vol. 1, pp. 224-228). IEEE. 10.1109/CISP.2013.6743991

    Parameters
    ----------
    img: numpy.ndarray
        2 dimensional array which has real intensity values
    win_size: integer
        window size to apply filter

    Returns
    -------
    filter_im: numpy.ndarray
        filtered intensity image.
    """
    print('>> lee_enhanced_filter', kwargs)
    win_size = kwargs.get('window_size', 3)

    # we process the entire img as float64 to avoid type overflow error
    img = np.float64(img)
    w_t, mean, _ = weightingarr(img, win_size, k, cu, cmax)
    filter_im = (mean * w_t) + (img * (1 - w_t))

    return filter_im


def fill_nan_value(data):
    mask = np.isnan(data) | np.isinf(data)
    data[mask] = -30
    return data


def anisotropic_diffusion(img, **kwargs):
    """
    Anisotropic Diffusion

    Parameters
    ----------
    img: numpy.ndarray
        2 dimensional array which has real intensity values
    weight: float
        window size to apply filter

    Returns
    -------
    filter_im: numpy.ndarray
        filtered intensity image.

    Notes
    -----
    A. Chambolle, An algorithm for total variation minimization
    and applications, Journal of Mathematical Imaging and Vision,
    Springer, 2004, 20, 89-97.
    https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_tv_chambolle
    """
    print('anisotropic_diffusion', kwargs)
    weight = kwargs.get('weight', 1)
    mask = np.isnan(img)

    img_db = 10 * np.log10(img)
    img_db_filled = fill_nan_value(img_db)
    filtered_img = denoise_tv_chambolle(img_db_filled,
                                        weight=weight,
                                        )
    # Vectorize conditional replacements using masks
    filtered_img[mask] = np.nan  # Preserve original NaN positions
    # zero_or_negative_mask = filtered_img <= 0
    infinite_mask = np.isinf(filtered_img)

    # Directly use img values where filtered_img fails conditions
    # filtered_img[zero_or_negative_mask] = img[zero_or_negative_mask]
    filtered_img[infinite_mask] = img[infinite_mask]
    return 10 ** (filtered_img / 10)


def guided_filter(img, **kwargs):
    """
    Apply a Guided Filter to an image to enhance and smooth it while
    preserving edges.

    This function applies a Guided Filter to the input image using OpenCV's
    guidedFilter implementation. NaN values are preserved as they are in the
    input image. The filter is applied to the logarithmic scale (10 * log10)
    of the image after handling NaNs.

    Parameters:
    ----------
    img : np.ndarray
        A 2D or 3D array representing the input image. For a 3D array, the
        operation is applied to each channel independently.
    **kwargs
        Additional keyword arguments:
        'radius' : int
            Radius of the kernel used in the Guided Filter.
            Default is 1.
        'eps' : float
            Regularization parameter in Guided Filter to smooth within
            a radius. Default is 3.
        'ddepth' : int
            The depth of the output image.
            Default is -1 (use same depth as the source).

    Returns:
    -------
    np.ndarray
        A 2D or 3D array of the same shape as 'img', containing the filtered
        image. NaN values and zero or negative values are handled
        specifically as described in the notes below.

   Notes
   -----
   He, Kaiming, Jian Sun, and Xiaoou Tang. "Guided image filtering."
    IEEE transactions on pattern analysis and machine intelligence 35.6 (2012)
    : 1397-1409.
    https://docs.opencv.org/4.x/de/d73/classcv_1_1ximgproc_1_1GuidedFilter.html
    """
    radius = kwargs.get('radius', 1)
    eps = kwargs.get('eps', 3)
    ddepth = kwargs.get('ddepth', -1)
    # Apply Guided Filter
    mask = np.isnan(img)
    img_filled = fill_nan_value(img)

    img_db = 10 * np.log10(img_filled)
    img_db_filled = fill_nan_value(img_db)

    filtered_img = cv2.ximgproc.guidedFilter(
        guide=img_db_filled,
        src=img_filled,
        radius=radius,
        eps=eps,
        dDepth=ddepth)

    # Vectorize conditional replacements using masks
    filtered_img[mask] = np.nan  # Preserve original NaN positions
    zero_or_negative_mask = filtered_img <= 0
    infinite_mask = np.isinf(filtered_img)

    # Directly use img values where filtered_img fails conditions
    filtered_img[zero_or_negative_mask] = img[zero_or_negative_mask]
    filtered_img[infinite_mask] = img[infinite_mask]

    return filtered_img


def tv_bregman(X: np.ndarray, **kwargs) -> np.ndarray:
    """
    Denoise an input array X using Total Variation Bregman denoising.

    This function applies Total Variation (TV) Bregman denoising to the
    logarithm (base 10) of the input array, then converts it back to the
    original domain. Values of the input array that are NaN are treated
    specially: they are masked out during the denoising process and set
    to NaN in the output array.

    Parameters:
    ----------
    X : np.ndarray
        A 2D array of input values. It is assumed that X contains
        non-negative values since the logarithm is taken.

    lamb : float, optional
        The regularization parameter for TV Bregman denoising. It controls
        the amount of smoothing. Higher values produce more smoothed results.
        Default is 20.

    Returns:
    -------
    np.ndarray
        A 2D array of the same shape as X, containing the denoised results.

    Notes
    -----
    Tom Goldstein and Stanley Osher, “The Split Bregman Method For L1
    Regularized Problems”, https://ww3.math.ucla.edu/camreport/cam08-29.pdf
    https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_tv_bregman
    """
    lamb = kwargs.get('lambda_value', -1)
    print('bregman', kwargs)
    X_db = np.log10(X, out=np.full(X.shape, np.nan), where=(~np.isnan(X)))
    X_db[np.isnan(X)] = -30
    X_db_dspkl = denoise_tv_bregman(X_db, weight=lamb)
    X_dspkl = np.power(10, X_db_dspkl)
    X_dspkl[np.isnan(X)] = np.nan

    return X_dspkl
