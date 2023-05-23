import numpy as np 
from scipy import signal
from scipy.ndimage import generic_filter

K_DEFAULT = 1.0
CU_DEFAULT = 0.523
CMAX_DEFAULT = 1.73

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
    window = np.ones([winsize, winsize])/(winsize*winsize)
    mean = signal.convolve2d(arr, window, mode='same')
    c2 = signal.convolve2d(arr*arr, window, mode='same')
    std = ((c2 - mean*mean)**.5)

    return mean, std

# def window_stdev_nan(arr, winsize):

#     c1 = generic_filter(arr, np.nanmean, mode='constant', cval=np.nan, size=winsize*2)
#     c2 = generic_filter(arr*arr, np.nanmean, size=winsize*2, mode='constant')#, origin=-winsize)
#     return ((c2 - c1*c1)**.5)[:-winsize*2+1,:-winsize*2+1]

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
    window_mean, window_std = compute_window_mean_std(im, winsize= winsize) 

    # ci is the variation coefficient in the window
    ci = window_std / window_mean
    w_t_arr = np.zeros(im.shape)
    w_t_arr[ci <= cu] = 1
    w_t_arr[ ( ci>cu) & (ci < cmax) ] = np.exp((-k * (ci[( ci>cu) & (ci < cmax)] - cu)) / (cmax - ci[( ci>cu) & (ci < cmax)]))
    w_t_arr[ci >= cmax] = 0

    return w_t_arr, window_mean, window_std

def lee_enhanced_filter(img, win_size=3, k=K_DEFAULT, cu=CU_DEFAULT,
                        cmax=CMAX_DEFAULT):
    """
    Apply Enhanced Lee filter to a numpy matrix containing the image, with a
    window of win_size x win_size.
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
    print('>> lee_enhanced_filter')
    # we process the entire img as float64 to avoid type overflow error
    img = np.float64(img)
    
    w_t, mean, std = weightingarr(img, win_size, k, cu, cmax)
    filter_im = (mean * w_t) + (img * (1 - w_t))

    return filter_im


