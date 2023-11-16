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
from scipy.spatial import cKDTree
from skimage.filters import threshold_multiotsu, threshold_otsu

from dswx_sar import (dswx_sar_util,
                      generate_log,
                      refine_with_bimodality,
                      region_growing)
from dswx_sar.dswx_runconfig import _get_parser, RunConfig


logger = logging.getLogger('dswx_s1')


def convert_pow2db(intensity):
    """Convert power to decibels

    Parameters
    ----------
    _ : numpy.ndarray
        intensity or power values
    """
    return 10 * np.log10(intensity)


class TileSelection:
    '''
    Select tile candidates that have boundary between water and non-water
    '''
    def __init__(self, ref_water_max, no_data):
        # no data value for reference_water
        self.no_data = no_data
        # value for pixels where water exists in reference_water
        self.wbd_max_value = ref_water_max
        self.threshold_twele = [0.09, 0.8, 0.97]
        self.threshold_bimodality = 0.7


    def rescale_value(self,
                      raster,
                      min_value,
                      max_value):
        """Rescales the entries of an array to the interval
        [min_value, max_value].

        Parameters
        ----------
        raster : numpy.ndarray
            Array containing values to be scaled
        min_value : float
            Minimum value that values in raster can be scaled to
        max_value : float
            Maximum value that values in raster can be scaled to

        Returns
        -------
        rescaled_value : numpy.ndarray
            rescaled image.
        """
        rescaled_value = (raster - min_value) / (max_value - min_value)

        return rescaled_value


    def select_tile_bimodality(self,
                               intensity,
                               threshold=0.7,
                               min_intensity_histogram=-30,
                               max_intensity_histogram=5,
                               numstep=100):
        ''' Select tiles with bimodal distribution

        Parameters
        ----------
        intensity : numpy.ndarray
            intensity array in linear scale
        threshold : float
            threshold value to detect bimodality
        min_intensity_histogram : float
            minimum range for histogram
        max_intensity_histogram : float
            maximum range for histogram
        numstep : integer
            number of histogram bins

        Returns
        -------
        max_sigma : float
            maximum value for estimated bimodality
        sigma : numpy.ndarray
            bimodality array
        select_flag : bool
            True/False if bimodality exists
        '''
        intensity_db = convert_pow2db(intensity)

        if min_intensity_histogram is None:
            min_intensity_histogram = np.nanpercentile(intensity_db, 2)
        if max_intensity_histogram is None:
            max_intensity_histogram = np.nanpercentile(intensity_db, 98)

        intensity_counts, intensity_bins = np.histogram(
            intensity_db,
            bins=np.linspace(min_intensity_histogram,
                             max_intensity_histogram,
                             numstep+1),
            density=True)

        bincenter = ((intensity_bins[:-1] + intensity_bins[1:]) / 2)
        int_db_variance = np.nanstd(intensity_db)**2

        intensity_bins_step = intensity_bins[2] - intensity_bins[1]

        sigma = np.zeros_like(bincenter)
        for bin_index, value in enumerate(bincenter):

            intensity_db_left = intensity_db[intensity_db <= value]
            intensity_db_right = intensity_db[intensity_db >= value]

            if len(intensity_db_left) & len(intensity_db_right):
                cand1 = bincenter <= value
                cand2 = bincenter >= value

                meanp1 = np.nanmean(intensity_db_left)
                meanp2 = np.nanmean(intensity_db_right)

                probp1 = np.nansum(intensity_counts[cand1]) * intensity_bins_step
                probp2 = np.nansum(intensity_counts[cand2]) * intensity_bins_step

                sigma[bin_index] = probp1 * probp2 * (
                    (meanp1 - meanp2) ** 2) / int_db_variance

        max_sigma = np.nanmax(sigma)
        select_flag = max_sigma > threshold

        return max_sigma, sigma, select_flag


    def select_tile_twele(self,
                          intensity_gray,
                          mean_int_glob,
                          thresholds=None):
        """Select tiles based on Twele's method

        Parameters
        ----------
        intensity_gray : np.ndarray
            tile of intensity image [gray scale, dB]
        mean_int_glob : float
            global mean intensity image
        thresholds : list
            three thresholds to select tile

        Returns
        -------
        select_flag : bool
            Indicates whether the tile is selected based on Twele's method.
            Returns True if the tile is selected, and False otherwise.
        cvx : float
            The coefficient of variation for the tile, calculated as the ratio of
            the standard deviation to the mean intensity of the tile.
        rx : float
            The ratio of the mean intensity of the tile to the
            global mean intensity.

        Reference
        ---------
        André Twele, Wenxi Cao, Simon Plank & Sandro Martinis (2016)
        Sentinel-1-based flood mapping: a fully automated processing chain,
        International Journal of Remote Sensing, 37:13, 2990-3004,
        DOI: [10.1080/01431161.2016.1192304]
        (https://doi.org/10.1080/01431161.2016.1192304)

        """

        if thresholds is None:
            thresholds = self.threshold_twele

        mean_int = np.nanmean(intensity_gray)
        sig_int = np.nanstd(intensity_gray)
        cvx = sig_int / mean_int
        rx = mean_int / mean_int_glob

        select_flag = (cvx >= thresholds[0]) & \
                      (rx >= thresholds[1]) & \
                      (rx <= thresholds[2])

        return select_flag, cvx, rx


    def select_tile_chini(self,
                          intensity):
        """Select tiles based on Chini's method
        Chini, M., Hostache, R., Giustarini, L., & Matgen, P. (2017).
        A hierarchical split-based approach for parametric thresholding
        of SAR images: Flood inundation as a test case. IEEE Transactions
        on Geoscience and Remote Sensing, 55(12), 6975-6988.

        Parameters
        ----------
        intensity : np.ndarray
            A 2D array representing a tile of the intensity image in dB.

        Returns
        -------
        select_flag : bool
            Indicates whether the tile is selected based on Chini's method.
            Returns True if the tile's intensity distribution is bimodal,
            and False otherwise.
        """
        metric_obj = refine_with_bimodality.BimodalityMetrics(
                    intensity)
        select_flag = metric_obj.compute_metric()

        return select_flag


    def get_water_portion_mask(self, water_mask):
        """
        Check if a water mask image contains both water and non-water areas.

        The function first computes the ratio of water (True in the mask)
        for each pixel in the image, and then checks if there are any areas
        that are between 0 and 1,
        i.e., both water and non-water areas are present.

        Parameters
        ----------
        water_mask : np.ndarray
            Layers of masks.
            Each mask layer contains water and non-water pixels.
        Returns
        -------
        water_area_flag : bool
            Indicates whether both water and
            non-water areas are present in mask.
        """
        _, water_nrow, water_ncol = water_mask.shape
        water_mask_sample = water_nrow * water_ncol

        if water_mask_sample == 0:
            water_spatial_portion = 0
        else:
            water_spatial_portion = np.nansum(water_mask, axis=(1, 2)) / \
                                              water_mask_sample

        # If the spatial coverage of water is equal to 1 (all water) or
        # is equal to 0 (all lands), then we don't attempt to search the
        # tile showing bimodal distribution.
        water_spatial_portion_bool = (water_spatial_portion > 0) & \
                                     (water_spatial_portion < 1)
        water_area_flag = np.nansum(water_spatial_portion_bool) > 0

        return water_area_flag


    def tile_selection_wbd(self,
                           intensity,
                           water_mask,
                           win_size=200,
                           selection_method='combined',
                           mininum_tile=20,
                           minimum_pixel_number=40):
        '''Select the tile candidates containing water and non-water
        from aid of water body layer based on the selection method
        {twele, chini, bimodality, combined}.

        Parameters
        ----------
        intensity : numpy.ndarray
            tile of intensity image
        wbd : numpy.ndarray
            water body layer [0 ~ 1]
        thresholds : list
            threshold values to determine the tiles
        win_size : integer
            size of window to search the water body that should be smaller
            than size of intensity
        selection_method : str
            window type {twele, chini, bimodality, combined}
            'combined' method runs all methods

        Returns
        -------
        candidate_tile_coords : numpy.ndarray
            The x and y coordinates that pass the tile selection test.
            Each row has index, y_start, y_end, x_start, and x_end.
        '''

        if np.all(np.isnan(intensity)) or np.all(intensity==0):
            candidate_tile_coords = []
            logger.info(f'No valid intensity values found.')

        else:
            height, width = intensity.shape
            n_water_masks = water_mask.ndim
            assert n_water_masks >= 2, "water mask must have at least 2 dimensions."

            # create 3 dimensional water mask
            if n_water_masks == 2:
                water_mask = np.expand_dims(water_mask, axis=0)
            _, water_nrow, water_ncol = water_mask.shape

            # If image size is smaller than window size,
            # window size become half of image size
            if (height <= win_size) | (width <= win_size):
                win_size0 = win_size
                win_size = np.min([height, width])
                logger.info(f'tile size changed {win_size0} -> {win_size}')

            if (height != water_nrow) and (width != water_ncol):
                raise ValueError("reference water image size differ from intensity image")

            if win_size < minimum_pixel_number:
                logger.info('tile & image sizes are too small')
                coordinate = []
                return coordinate

            # Check if number of pixel is enough
            num_pixel_max = win_size * win_size / 3

            number_y_window = np.ceil(height / win_size)
            number_x_window = np.ceil(width / win_size)

            number_y_window = number_y_window + \
                (1 if np.mod(height, win_size) > 0 else 0)
            number_x_window = number_x_window + \
                (1 if np.mod(width, win_size) > 0 else 0)

            ind_subtile = 0

            bimodality_max_set = []
            coordinate = []

            selected_tile_twele = []
            selected_tile_chini = []
            selected_tile_bimodality = []

            # convert linear to dB scale
            intensity_db = convert_pow2db(intensity)

            # covert gray scale
            intensity_gray = self.rescale_value(
                intensity_db,
                min_value=-35,
                max_value=10)

            water_area_flag = self.get_water_portion_mask(water_mask)

            mean_int_global = np.nanmean(intensity_gray)

            box_count = 0
            num_detected_box = 0
            num_detected_box_sum = 0
            subrun = 0

            # Initially check if the area under the searching window contains
            # both water bodies and lands from the reference water map.
            # if 0.0 < water_coverage < 1 and 0.0 < land_coverage < 1:
            if water_area_flag:
                while (num_detected_box_sum <= mininum_tile) and \
                    (win_size >= minimum_pixel_number):

                    subrun += 1
                    number_y_window = np.int16(height / win_size)
                    number_x_window = np.int16(width / win_size)

                    # Define step sizes
                    x_step = win_size // 2
                    y_step = win_size // 2

                    # Loop through the rectangle array with the sliding window
                    for x_coord in range(0, width - win_size + 1, x_step):
                        for y_coord in range(0, height - win_size + 1, y_step):
                            # Grab the small area using the window
                            x_start = x_coord
                            x_end = x_coord + win_size
                            y_start = y_coord
                            y_end = y_coord + win_size

                            x_end = min(x_end, height)
                            y_end = min(y_end, width)
                            box_count = box_count + 1

                            intensity_sub = intensity[x_start:x_end,
                                                    y_start:y_end]

                            intensity_sub_gray = intensity_gray[x_start:x_end,
                                                                y_start:y_end]

                            validnum = np.count_nonzero(~np.isnan(intensity_sub))

                            # create subset from water mask
                            water_mask_sublock = water_mask[:, x_start:x_end,
                                                            y_start:y_end]

                            # compute valid pixels from subset
                            water_number_sample_sub = np.count_nonzero(
                                    ~np.isnan(water_mask_sublock[0, :, :]))

                            # compute spatial portion for each polarization
                            water_area_sublock_flag = self.get_water_portion_mask(
                                water_mask_sublock)

                            if water_number_sample_sub > 0 \
                                and water_area_sublock_flag \
                                and validnum > num_pixel_max:

                                if selection_method in ['twele', 'combined']:
                                    tile_selected_flag, _, _ = \
                                        self.select_tile_twele(
                                            intensity_sub_gray,
                                            mean_int_global)
                                    if tile_selected_flag:
                                        selected_tile_twele.append(True)
                                    else:
                                        selected_tile_twele.append(False)

                                if selection_method in ['chini', 'combined']:
                                    # Once 'combined' method is selected,
                                    # the chini test is carried when the
                                    # 'twele' method passed.
                                    if (selection_method in ['combined']) & \
                                        (not tile_selected_flag):

                                        selected_tile_chini.append(False)

                                    else:
                                        tile_selected_flag = \
                                            self.select_tile_chini(intensity_sub)

                                        if tile_selected_flag:
                                            selected_tile_chini.append(True)
                                        else:
                                            selected_tile_chini.append(False)

                                if selection_method in ['bimodality', 'combined']:

                                    if (selection_method == 'combined') and \
                                        not tile_selected_flag:

                                        selected_tile_bimodality.append(False)
                                    else:
                                        _, _, tile_bimode_flag = \
                                            self.select_tile_bimodality(
                                                intensity_sub,
                                                threshold=self.threshold_bimodality)

                                        if tile_bimode_flag:
                                            selected_tile_bimodality.append(True)
                                        else:
                                            selected_tile_bimodality.append(False)

                            else:
                                bimodality_max_set.append(np.nan)
                                selected_tile_twele.append(False)
                                selected_tile_chini.append(False)
                                selected_tile_bimodality.append(False)

                            # keep coordiates for the searching window.
                            coordinate.append([ind_subtile,
                                            x_start,
                                            x_end,
                                            y_start,
                                            y_end])
                            ind_subtile += 1

                    if selection_method in ['twele']:
                        num_detected_box = np.sum(selected_tile_twele)

                    if selection_method in ['chini']:
                        num_detected_box = np.sum(selected_tile_chini)

                    if selection_method in ['bimodality']:
                        num_detected_box = np.sum(selected_tile_bimodality)

                    if selection_method in ['combined']:
                        selected_tile_merged = np.logical_and(
                            selected_tile_twele,
                            selected_tile_chini)
                        selected_tile_merged = np.logical_and(
                            selected_tile_merged,
                            selected_tile_bimodality)
                        num_detected_box = np.sum(selected_tile_merged)

                    if num_detected_box_sum <= mininum_tile:
                        # try tile-selection with smaller win size
                        win_size = int(win_size * 0.8)
                        num_pixel_max = win_size * win_size / 3

                    num_detected_box_sum += num_detected_box

            else:
                logger.info('No water body found')

            if selection_method in ['twele']:
                selected_tile = selected_tile_twele
            if selection_method in ['chini']:
                selected_tile = selected_tile_chini
            if selection_method in ['bimodality']:
                selected_tile = selected_tile_bimodality
            if selection_method == 'combined':
                selected_tile = np.logical_and(selected_tile_twele,
                                            selected_tile_chini)
                selected_tile = np.logical_and(selected_tile,
                                            selected_tile_bimodality)

            coordinate = np.array(coordinate)
            candidate_tile_coords = coordinate[selected_tile]

        return candidate_tile_coords


def create_three_water_masks(
        wbd,
        water_threshold,
        no_data,
        wbd_max_value,
        flood_dilation_pixel=16,
        drought_erosion_pixel=10):
    """Creates a water mask for each of the following three settings:
        normal, flood, and drought.

    Parameters
    ----------
    wbd : np.ndarray
        The input raster with water presence information.
    water_threshold : float
        The threshold used for water classification.
        The value must range between 0 and 1.
    no_data : int
        The no data value in the raster.
        All cells with this value will be treated as no data areas.
    wbd_max : int
        The maximum valid value in the raster.
        The raster values will be normalized by this value.
    flood_dilation_pixel : int
        Dilation interation number
    drought_erosion_pixel : int
        Erosion interation number

    Returns
    -------
    water_mask_set : np.ndarray
        The water mask with three settings: normal, flood, and drought.
    """
    wbd = np.asarray(wbd, dtype='float32')
    wbd[wbd == no_data] = wbd_max_value
    water_global_norm = wbd / wbd_max_value

    water_normal_mask = water_global_norm > water_threshold

    # Assuming flood water (+480m)
    flood_water_mask = ndimage.binary_dilation(
        water_normal_mask,
        iterations=flood_dilation_pixel)

    # Assuming drought (-300m)
    drought_water_mask = ndimage.binary_erosion(
        water_normal_mask,
        iterations=drought_erosion_pixel)
    water_mask_set = np.zeros([3, wbd.shape[0], wbd.shape[1]])

    water_mask_set[0, :, :] = water_normal_mask
    water_mask_set[1, :, :] = flood_water_mask
    water_mask_set[2, :, :] = drought_water_mask

    return water_mask_set


def remove_invalid(sample_array, no_data=0):
    """ Removes invalid values from a numpy array.

    Parameters
    ----------
    sample_array : np.ndarray
        The array that contains invalid values.
    no_data : int
        The value that is considered invalid. Default is 0.

    Returns
    -------
    sample_array : np.ndarray
        The array that only contains valid values.
    """
    mask = (np.isnan(sample_array) | (sample_array == no_data))
    sample_array = sample_array[np.invert(mask)]
    return sample_array


def gauss(array, mu, sigma, amplitude):
    """Generate gaussian distribution with given mean and std
    """
    return amplitude * np.exp(-(array - mu)**2 / 2 / sigma**2)


def bimodal(array, mu1, sigma1, amplitude1,
            mu2, sigma2, amplitude2):
    """Generate bimodal gaussian distribution with given means and stds
    """
    return gauss(array, mu1, sigma1, amplitude1) + \
        gauss(array, mu2, sigma2, amplitude2)


def trimodal(array, mu1, sigma1, amplitude1,
             mu2, sigma2, amplitude2,
              mu3, sigma3, amplitude3):
    """Generate trimodal gaussian distribution with given means and stds
    """
    return gauss(array, mu1, sigma1, amplitude1) + \
        gauss(array, mu2, sigma2, amplitude2) + \
        gauss(array, mu3, sigma3, amplitude3)


def compute_ki_threshold(
        intensity,
        min_intensity_histogram,
        max_intensity_histogram,
        step_histogram):
    """ Computes the threshold using Kittler-Illingworth algorithm

    Parameters
    ----------
    intensity: np.ndarray
        The image intensities.
    min_intensity_histogram: int
        The minimum intensity.
    max_intensity_histogram: int
        The maximum intensity.
    step_histogram :int
        The step intensity to build the histogram.

    Returns
    -------
    threshold : float
        The computed threshold.
    index_ki_threshold : int
        index for ki threshold
    """
    numstep = int((max_intensity_histogram - min_intensity_histogram) / step_histogram)

    if numstep < 100:
        step_histogram = ((max_intensity_histogram
                           - min_intensity_histogram)
                           / 1000)
        numstep = 1000

    intensity_counts, intensity_bins = np.histogram(
        intensity,
        bins=np.linspace(min_intensity_histogram,
                         max_intensity_histogram,
                         numstep + 1),
        density=True)
    intensity_bins = intensity_bins[:-1]

    intensity_counts = intensity_counts.astype(np.float64)
    intensity_bins = intensity_bins.astype(np.float64)
    # A small constant value
    eps = dswx_sar_util.Constants.negligible_value

    intensity_cumsum = np.cumsum(intensity_counts)
    # Replace zeros and negative numbers with 'eps'
    intensity_cumsum = np.where(intensity_cumsum <= 0, 
                                eps, intensity_cumsum)

    intensity_cumsum[intensity_cumsum==0] = np.nan

    intensity_area = np.cumsum(intensity_counts * intensity_bins)
    intenisty_s = np.cumsum(intensity_counts * intensity_bins ** 2)

    var_f = intenisty_s / intensity_cumsum - \
        (intensity_area / intensity_cumsum) ** 2
    var_f = np.where(var_f <= 0, eps, var_f)
    sigma_f = np.sqrt(var_f)

    cb = intensity_cumsum[-1] - intensity_cumsum
    cb = np.where(cb <= 0, eps, cb)

    mb = intensity_area[-1] - intensity_area
    sb = intenisty_s[-1] - intenisty_s
    var_b = sb / cb - (mb / cb) ** 2
    var_b = np.where(var_b <= 0, eps, var_b)
    sigma_b = np.sqrt(var_b)

    normalized_int_cumsum =  intensity_cumsum / intensity_cumsum[-1]
    normalized_int_cumsum = np.where(normalized_int_cumsum >= 1, 
                                1 - eps, normalized_int_cumsum)
    normalized_int_cumsum = np.where(normalized_int_cumsum <= 0, 
                                     eps, normalized_int_cumsum)
    prob_array = \
        normalized_int_cumsum * np.log(sigma_f) + \
        (1 - normalized_int_cumsum) * np.log(sigma_b) - \
        normalized_int_cumsum * np.log(normalized_int_cumsum) - \
        (1 - normalized_int_cumsum) * np.log(1 - normalized_int_cumsum)
    prob_array[~np.isfinite(prob_array)] = np.inf

    index_ki_threshold = np.argmin(prob_array)
    threshold = intensity_bins[index_ki_threshold]

    return threshold, index_ki_threshold


def determine_threshold(
        intensity,
        candidate_tile_coords,
        min_intensity_histogram=-32,
        max_intensity_histogram=0,
        step_histogram=0.1,
        bounds=[-20, -13],
        method='ki',
        mutli_threshold=True):
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
        Flag indicating whether tri-mode Gaussian distribution is assumed or not.

    Returns
    -------
    global_threshold : float
        thresholds calculated from KI algorithm
    glob_mode_thres : float
        mode value of gaussian distribution of water body
    """
    if max_intensity_histogram == -1000:
        max_intensity_histogram = np.nanpercentile(intensity, 90)
    if min_intensity_histogram == -1000:
        min_intensity_histogram = np.nanpercentile(intensity, 10)
    numstep = int((max_intensity_histogram - min_intensity_histogram) / step_histogram)
    if numstep < 100:
        step_histogram0 = step_histogram
        step_histogram = ((max_intensity_histogram - min_intensity_histogram) / 1000)
        logger.info(f'Histogram bin step changes from {step_histogram0} to {step_histogram} '
                    'in threshold computation.')

    threshold_array = []
    threshold_idx_array = []
    mode_array = []
    eps = dswx_sar_util.Constants.negligible_value
    min_threshold, max_threshold = bounds[0], bounds[1]

    for coord in candidate_tile_coords:

        # assume that coord consists of 5 elements
        ystart, yend, xstart, xend = coord[1:]

        intensity_sub = intensity[ystart:yend,
                                  xstart:xend]

        # generate histogram with intensity higher than -35 dB
        intensity_sub = intensity_sub[intensity_sub > -35]
        intensity_sub = remove_invalid(intensity_sub)

        if intensity_sub.size == 0:
            return global_threshold, glob_mode_thres
        
        intensity_counts, bins = np.histogram(
            intensity_sub,
            bins=np.linspace(min_intensity_histogram,
                             max_intensity_histogram,
                             numstep + 1),
            density=True)
        intensity_bins = bins[:-1]

        if method == 'ki':
            threshold, idx_threshold = compute_ki_threshold(
                intensity_sub,
                min_intensity_histogram,
                max_intensity_histogram,
                step_histogram)

        elif method in ['otsu', 'rg']:
            threshold = threshold_otsu(intensity_sub)

        # get index of threshold from histogram.
        idx_threshold = np.searchsorted(intensity_bins, threshold)

        # if estimated threshold is higher than bounds,
        # re-estimate threshold assuming tri-mode distribution
        if threshold > bounds[1] and mutli_threshold:
            thresholds = threshold_multiotsu(intensity_sub)

            if thresholds[0] < threshold:
                threshold = thresholds[0]
                idx_threshold = np.searchsorted(intensity_bins, threshold)

        # Search the local peaks from histogram for initial values for fitting
        # peak for lower distribution
        lowmaxind_cands, _ = find_peaks(intensity_counts[0:idx_threshold+1], distance=5)
        if not lowmaxind_cands.any():
            lowmaxind_cands = np.array([np.nanargmax(intensity_counts[: idx_threshold+1])])

        intensity_counts_cand = intensity_counts[lowmaxind_cands]
        lowmaxind = lowmaxind_cands[np.nanargmax(intensity_counts_cand)]

        # peak for higher distribution
        highmaxind_cands, _ = find_peaks(intensity_counts[idx_threshold:], distance=5)

        # if highmaxind_cands is empty
        if not highmaxind_cands.any():
            highmaxind_cands = np.array([np.nanargmax(intensity_counts[idx_threshold:])])

        intensity_counts_cand = intensity_counts[idx_threshold + highmaxind_cands]
        highmaxind = idx_threshold + \
            highmaxind_cands[np.nanargmax(intensity_counts_cand)]

        lowmaxind = np.squeeze(lowmaxind)
        highmaxind = np.squeeze(highmaxind)

        if highmaxind.size > 1:
            highmaxind = highmaxind[highmaxind >= idx_threshold]
            highmaxind = highmaxind[0]

        if lowmaxind.size > 1:
            lowmaxind = lowmaxind[lowmaxind <= idx_threshold]
            lowmaxind = lowmaxind[0]

        # mode values
        tau_mode_left = intensity_bins[lowmaxind]
        tau_mode_right = intensity_bins[highmaxind]

        tau_amp_left = intensity_counts[lowmaxind]
        tau_amp_right = intensity_counts[highmaxind]

        try:
            expected = (tau_mode_left, .5, tau_amp_left,
                        tau_mode_right, .5, tau_amp_right)

            params, _ = curve_fit(bimodal,
                                  intensity_bins,
                                  intensity_counts,
                                  expected,
                                  bounds=((-30, 0, 0.01,
                                           -30, 0, 0.01),
                                          (5, 5, 0.95,
                                           5, 5, 0.95)))
            if params[0] > params[3]:
                second_mode = params[:3]
                first_mode = params[3:]
            else:
                first_mode = params[:3]
                second_mode = params[3:]

            simul_first = gauss(intensity_bins, *first_mode)
            simul_second = gauss(intensity_bins, *second_mode)
            if simul_second_sum == 0:
                simul_second_sum = eps
            converge_ind = np.where((intensity_bins < tau_mode_right)
                                    & (intensity_bins > tau_mode_left)
                                    & (intensity_bins < threshold)
                                    & (np.cumsum(simul_second) /
                                       simul_second_sum < 0.03))

            if len(converge_ind[0]):
                modevalue = intensity_bins[converge_ind[0][-1]]
            else:
                modevalue = tau_mode_left

            optimization = True

        except:
            optimization = False
            logger.info(f'Bimodal curve Fitting fails in threshold computation.')
            modevalue = tau_mode_left
        try:
            expected = (tau_mode_left, .5, tau_amp_left,
                        tau_mode_right, .5, tau_amp_right,
                        (tau_mode_left+tau_mode_right)/2, .5, 0.1)
            # curve_fit fits the trimodal distributions
            # All distributions are assumed to be in the bound
            # -35 to 5 dB, with standard deviation of 0 - 5[dB]
            # and amplitudes of 0.01 to 0.95.
            params, _ = curve_fit(trimodal,
                                  intensity_bins,
                                  intensity_counts,
                                  expected,
                                  bounds=((-35, 0, 0.01,
                                           -35, 0, 0.01,
                                           -35, 0, 0.01),
                                           (5, 5, 0.95,
                                            5, 5, 0.95,
                                            5, 5, 0.95)))

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
                simul_second_sum = eps
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

            if tri_ratio_bool and third_second_dist_bool and \
               first_second_dist_bool and tri_first_mode[0] > -32:
                tri_optimization = True

            else:
                tri_optimization = False

        except:
            tri_optimization = False
            logger.info(f'Trimodal curve Fitting fails in threshold computation.')
        if tri_optimization:
            intensity_sub2 = intensity_sub[intensity_sub < tri_second_mode[0]]
            tau_bound_gauss = threshold_otsu(intensity_sub2)

            if threshold > tau_bound_gauss:
                threshold = tau_bound_gauss
                modevalue = tri_first_mode[0]

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
                        rg_layer = region_growing.region_growing(
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
                                (intensity_counts_rg[compare_index1:compare_index2] -
                                simul_first[compare_index1:compare_index2])**2))
                            rms_sss.append(rms)
                            rms_xx.append(hist_bin)
                        else:
                            rms_sss.append(np.nan)
                            rms_sss.append(hist_bin)

                min_rms_ind = np.where(rms_sss)
                rg_tolerance = rms_xx[min_rms_ind[0][0]]
                threshold = (rg_tolerance + threshold) / 2

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


def fill_threshold_with_gdal(threshold_array,
                             rows,
                             cols,
                             filename,
                             outputdir,
                             pol_list,
                             filled_value,
                             no_data=-50,
                             average_tile=True):
    """Interpolate thresholds over a 2-D grid.

    Parameters
    ----------
    threshold_array : dict[numpy.ndarray]
        The i and j coordinates that pass the tile selection test.
        Each row has index, x_start, x_end, y_start, and y_end.
    rows : integer
        number of rows
    cols : integer
        number of columns
    filename : str
        output file name
    outputdir : str
        output dir path
    pol_list: list
        polarization list
    no_data : float
        no_data value
    average_tile : bool
        flag to average the thresholds within each tile.
        If true, the single threshold will be assigned to each tile.
        If false, the thresholds are stored with their positions.
    Returns
    -------
    threshold_raster : numpy.ndarray
        interpolated raster
    """

    if average_tile:
        tau_row, tau_col, _ = threshold_array['array'].shape
        y_tau = threshold_array['block_row'] * np.arange(0, tau_row) + \
            threshold_array['block_row'] / 2
        x_tau = threshold_array['block_col'] * np.arange(0, tau_col) + \
            threshold_array['block_col'] / 2
        x_arr_tau, y_arr_tau = np.meshgrid(x_tau, y_tau)

    else:
        x_coarse_grid = np.arange(0, cols + 1, 400)
        y_coarse_grid = np.arange(0, rows + 1, 400)

    threshold_raster = []

    for polind, pol in enumerate(pol_list):
        if average_tile:
            z_tau = threshold_array['array'][:, :, polind]
            z_tau[z_tau == no_data] = np.nan

            nan_mask = np.invert(np.isnan(z_tau))

            x_arr_tau_valid = x_arr_tau[nan_mask]
            y_arr_tau_valid = y_arr_tau[nan_mask]

        else:
            z_tau = np.array(threshold_array['array'][polind],
                             dtype='float32')
            x_tau = np.array(threshold_array['block_col'][polind],
                             dtype='int16')
            y_tau = np.array(threshold_array['block_row'][polind],
                             dtype='int16')
            z_tau[z_tau == no_data] = np.nan
            x_arr_tau, y_arr_tau = np.meshgrid(x_coarse_grid, y_coarse_grid)

            nan_mask = np.invert(np.isnan(z_tau))

            x_arr_tau_valid = x_tau[nan_mask]
            y_arr_tau_valid = y_tau[nan_mask]

        z_arr_tau_valid = z_tau[nan_mask]
        tif_file_str = os.path.join(outputdir, f"{filename}_{pol}.tif")

        if len(z_arr_tau_valid) > 1:
            # try bilinear interpolation for the evenly spaced grid
            try:
                interp_tau = interpolate.griddata((x_arr_tau_valid,
                                                   y_arr_tau_valid),
                                                  z_arr_tau_valid,
                                                  (x_arr_tau.flatten(),
                                                   y_arr_tau.flatten()),
                                                  method='linear')

                if average_tile:
                    interp_tau = interp_tau.reshape([tau_row, tau_col])

                else:
                    interp_tau = interp_tau.reshape([len(y_coarse_grid),
                                                    len(x_coarse_grid)])

                # fill in for points outside of
                # the convex hull using nearest interpolator
                nan_mask = np.invert(np.isnan(interp_tau))
                x_arr_tau_valid = x_arr_tau[nan_mask]
                y_arr_tau_valid = y_arr_tau[nan_mask]
                z_arr_tau_valid = interp_tau[nan_mask]
            except:
                logger.info('Linear interpolation failed.')
                pass
            # nearest extrapolation for the evenly spaced grid
            interp_tau = interpolate.griddata((x_arr_tau_valid,
                                               y_arr_tau_valid),
                                              z_arr_tau_valid,
                                              (x_arr_tau.flatten(),
                                               y_arr_tau.flatten()),
                                              method='nearest')

            if average_tile:
                data_valid = np.vstack([x_arr_tau.flatten(),
                                        y_arr_tau.flatten(),
                                        interp_tau])
            else:
                data_valid = np.vstack([np.hstack([x_arr_tau.flatten(),
                                                   x_arr_tau_valid]),
                                        np.hstack([y_arr_tau.flatten(),
                                                   y_arr_tau_valid]),
                                        np.hstack([interp_tau,
                                                   z_arr_tau_valid])])

            csv_file_str = os.path.join(outputdir,
                                        f"data_thres_{pol}_{filename}.csv")
            np.savetxt(csv_file_str, data_valid.transpose(), delimiter=",")

            vrt_file = os.path.join(outputdir,
                                    f"data_thres_{pol}_{filename}.vrt")

            with open(vrt_file, 'w') as fpar:
                fpar.write('<OGRVRTDataSource>\n')
                fpar.write(f'<OGRVRTLayer name="data_thres_{pol}_{filename}">\n')
                fpar.write(f"    <SrcDataSource>{csv_file_str}</SrcDataSource>\n")
                fpar.write('    <GeometryType>wkbPoint</GeometryType>\n')
                fpar.write('    <GeometryField encoding="PointFromColumns" x="field_1" y="field_2" z="field_3"/>\n')
                fpar.write('</OGRVRTLayer>\n')
                fpar.write('</OGRVRTDataSource>')

            if average_tile:
                gdal_grid_str = \
                    "gdal_grid -zfield field_3 -l " \
                    f"data_thres_{pol}_{filename} {vrt_file}  {tif_file_str}" \
                    f" -txe 0 {cols} -tye 0 {rows} -a invdist:power=0.500:" \
                    "smoothing=1.0:radius1=" \
                    f"{threshold_array['block_row']*2}:radius2=" \
                    f"{threshold_array['block_col']*2}:angle=0.000000:" \
                    "max_points=0:min_points=1:nodata=0.000000 " \
                    f" -outsize {cols} {rows} -ot Float32"
            else:
                gdal_grid_str = \
                    "gdal_grid -zfield field_3 " \
                    f"-l data_thres_{pol}_{filename} {vrt_file}  {tif_file_str} " \
                    f" -txe 0 {cols} -tye 0 {rows} " \
                    f" -a invdist:power=0.5000000:smoothing=1.000000:radius1={400*2}" \
                    f":radius2={400*2}:angle=0.000000:max_points=0:min_points=1:" \
                    f"nodata=0.000 -outsize {cols} {rows} -ot Float32"

            os.system(gdal_grid_str)
            threshold_raster.append(dswx_sar_util.read_geotiff(tif_file_str))
        else:
            logger.info('threshold array is empty')

            # Define the GeoTransform (you may need to adjust this based on your spatial reference)
            geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)  # Change as needed

            # Define the projection (you may need to specify the correct projection)
            projection = "EPSG:4326"  # Change as needed

            # Create an empty NumPy array filled with zeros
            empty_data = np.ones((rows, cols)) * no_data

            # Create a new GeoTIFF dataset
            driver = gdal.GetDriverByName('GTiff')
            ds = driver.Create(tif_file_str, cols, rows, 1, gdal.GDT_Float32)

            # Set the GeoTransform and Projection
            ds.SetGeoTransform(geotransform)
            ds.SetProjection(projection)

            # Write the empty data to the dataset
            ds.GetRasterBand(1).WriteArray(empty_data)

            # Close the dataset
            ds = None
            threshold_raster.append(
                np.ones([rows, cols], dtype=np.float64) * -50)

    return np.array(threshold_raster)


def fill_threshold_with_distance(threshold_array,
                                 rows,
                                 cols,
                                 filename,
                                 outputdir,
                                 pol_list,
                                 no_data=-50):

    """Interpolate thresholds over a 2-D grid.

    Parameters
    ----------
    threshold_array : dict[numpy.ndarray]
        The i and j coordinates that pass the tile selection test.
        Each row has index, x_start, x_end, y_start, and y_end.
    rows : integer
        number of rows
    cols : integer
        number of columns
    filename : str
        output file name
    outputdir : str
        output dir path
    pol_list: list
        polarization list
    no_data : float
        no_data value

    Returns
    -------
    threshold_raster : numpy.ndarray
        interpolated raster
    """
    tau_row, tau_col, _ = threshold_array['array'].shape

    y_tau = threshold_array['block_row'] * np.arange(0, tau_row) + \
        threshold_array['block_row'] / 2
    x_tau = threshold_array['block_col'] * np.arange(0, tau_col) + \
        threshold_array['block_col'] / 2
    threshold_raster = []

    for polind, pol in enumerate(pol_list):
        z_tau = threshold_array['array'][:, :, polind]
        z_tau[z_tau == no_data] = np.nan
        x_arr_tau, y_arr_tau = np.meshgrid(x_tau, y_tau)

        nan_mask = np.invert(np.isnan(z_tau))
        x_arr_tau_valid = x_arr_tau[nan_mask]
        y_arr_tau_valid = y_arr_tau[nan_mask]
        z_arr_tau_valid = z_tau[nan_mask]

        x_arr_tau, y_arr_tau = np.meshgrid(x_tau, y_tau)

        interp_tau = interpolate.griddata((x_arr_tau_valid,
                                           y_arr_tau_valid),
                                          z_arr_tau_valid,
                                          (x_arr_tau.flatten(),
                                          y_arr_tau.flatten()),
                                          method='cubic')

        interp_tau = interp_tau.reshape([tau_row, tau_col])
        # fill in for points outside of the convex hull
        # using nearest interpolator
        nan_mask = np.invert(np.isnan(interp_tau))

        x_arr_tau_valid = x_arr_tau[nan_mask]
        y_arr_tau_valid = y_arr_tau[nan_mask]
        z_arr_tau_valid = interp_tau[nan_mask]
        interp_tau = interpolate.griddata((x_arr_tau_valid,
                                           y_arr_tau_valid),
                                          z_arr_tau_valid,
                                          (x_arr_tau.flatten(),
                                           y_arr_tau.flatten()),
                                          method='nearest')

        z_arr_tau_valid = interp_tau
        x_arr_tau_valid = x_arr_tau.flatten()
        y_arr_tau_valid = y_arr_tau.flatten()

        # Let's assume we have some 2D points and corresponding values
        points = np.zeros([len(z_arr_tau_valid), 2])
        points[:, 0] = x_arr_tau_valid
        points[:, 1] = y_arr_tau_valid
        values = z_arr_tau_valid

        # Now we create a KD-tree
        tree = cKDTree(points)
        if len(values) < 10:
            number_point = len(values)
        else:
            number_point = 10
        # Let's assume we have a grid and we want to interpolate the values on this grid
        grid_y, grid_x = np.mgrid[0:rows, 0:cols]  # a grid in 2D

        rbf = Rbf(points[:,0], points[:,1], values,
                  function='multiquadric', smooth=0.2)
        # grid_values = rbf(grid_x, grid_y)
        grid_values = np.zeros(grid_x.shape)

        lines_per_block = 10

        data_length = rows
        lines_per_block = min(data_length, lines_per_block)

        nblocks = int(np.ceil(data_length / lines_per_block))

        for block in range(0, nblocks):
            row_start = block * lines_per_block

            if (row_start + lines_per_block > rows):
                block_rows = rows - row_start
            else:
                block_rows = lines_per_block

            grid_values[row_start:row_start+block_rows, :] = rbf(
                grid_x[row_start:row_start+block_rows, :],
                grid_y[row_start:row_start+block_rows, :])

        tif_file_str = os.path.join(outputdir, f"{filename}_{pol}.tif")
        driver = gdal.GetDriverByName('GTIFF')
        dst_ds = driver.Create(tif_file_str,
                               cols,
                               rows,
                               1,
                               gdal.GDT_Float32)
        dst_ds.GetRasterBand(1).WriteArray(grid_values)
        dst_ds = None
        del dst_ds
        threshold_raster.append(grid_values)

    return np.array(threshold_raster)


def save_threshold_dict(threshold, block_row, block_col):
    '''Save threshold array and shape to pickle

    Parameters
    ----------
    threshold : numpy.ndarray
        The i and j coordinates that pass the tile selection test.
        Each row has index, x_start, x_end, y_start, and y_end.
    block_row : integer
        block index in row
    block_col : integer
        block index in column

    Returns
    -------
    threshold_dict : dict
        Dict mapping block index and threshold array
    '''
    threshold_dict = {}
    threshold_dict['block_row'] = block_row
    threshold_dict['block_col'] = block_col
    threshold_dict['array'] = threshold

    return threshold_dict


def run_sub_block(intensity, wbdsub, cfg, winsize=200, thres_max=[-15, -22]):

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
    tile_selection_twele = threshold_cfg.tile_selection_twele
    tile_selection_bimodality = threshold_cfg.tile_selection_bimodality

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

    tile_selection_object = TileSelection(ref_water_max=water_cfg.max_value,
                                          no_data=water_cfg.no_data_value)
    tile_selection_object.threshold_twele = tile_selection_twele
    tile_selection_object.threshold_bimodality = \
        tile_selection_bimodality

    ## Tile Selection (with water body)
    for polind, pol in enumerate(pol_list):

        candidate_tile_coords = tile_selection_object.tile_selection_wbd(
                           intensity=intensity[polind, :, :],
                           water_mask=wbdsub,
                           win_size=winsize,
                           selection_method=tile_selection_method)

        if pol in ['VV', 'VH', 'HH', 'HV', 'span']:
            min_intensity_histogram, max_intensity_histogram, step_histogram = -35, 10, 0.1
            target_im = np.squeeze(10*np.log10(intensity[polind, :, :]))
        elif pol == 'ratio':
            min_intensity_histogram, max_intensity_histogram, step_histogram = -11, 0, 0.1
            target_im = np.squeeze(10*np.log10(intensity[polind, :, :]))
        else:
            min_intensity_histogram, max_intensity_histogram, step_histogram = -1000, -1000, 0.1
            target_im = np.squeeze(intensity[polind, :, :])

        if len(candidate_tile_coords) > 0 :
            wbdsub_norm = wbdsub/tile_selection_object.wbd_max_value
            if wbdsub_norm.shape[0]*wbdsub_norm.shape[1] == 0:
                water_variation = 0
            else:
                water_variation = len(wbdsub_norm[
                                     (wbdsub_norm < 0.8) &
                                     (wbdsub_norm > 0.2)])\
                                     / (wbdsub_norm.shape[0] *
                                        wbdsub_norm.shape[1])

            if water_variation > 0.1:
                threshold_temp_max = thres_max[polind]
            else:
                if pol in ['VV', 'HH', 'span']:
                    threshold_temp_max = -13
                else:
                    threshold_temp_max = -20

            int_threshold, mode_tau = determine_threshold(
                intensity=target_im,
                candidate_tile_coords=candidate_tile_coords,
                min_intensity_histogram=min_intensity_histogram,
                max_intensity_histogram=max_intensity_histogram,
                step_histogram=step_histogram,
                bounds=[-28, threshold_temp_max],
                method=threshold_method,
                mutli_threshold=mutli_threshold_flag)

            logger.info(f'method {threshold_method} for {pol}')
            logger.info(f'global threshold and bound : {int_threshold} {threshold_temp_max}')
            logger.info(f'global mode thresholding : {mode_tau}')

        else:
            int_threshold = np.nan
            mode_tau = np.nan
        threshold_tau_set.append(int_threshold)
        mode_tau_set.append(mode_tau)
        candidate_tile_coords_set.append(candidate_tile_coords)

    return threshold_tau_set, mode_tau_set, candidate_tile_coords_set


def process_block(ii, jj, n_rows_block, n_cols_block, m_rows_block, m_cols_block,
                  block_row, block_col, width, filt_im_str, wbd_im_str,
                  cfg, thres_max, average_tile_flag=False):
    """
    Processes a specific block of an image.

    Parameters:
    -----------
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
        tuple: Contains processed block information, including row index, column index,
               threshold list, mode list, and candidate coordinates.
    """
    x_size = m_cols_block if (jj == n_cols_block - 1) and m_cols_block > 0 else block_col
    y_size = m_rows_block if (ii == n_rows_block - 1) and m_rows_block > 0 else block_row

    logger.info(f"block_processing: {ii + 1}/{n_rows_block} _ {jj + 1}/{n_cols_block}"
                f" - {ii * n_cols_block + jj + 1}/{n_rows_block * n_cols_block}")

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
        threshold_list = [np.nanmean(test_threshold)
                          if not np.all(np.isnan(test_threshold)) else np.nan
                          for test_threshold in threshold_tau_block]
        mode_list = [np.nanmean(test_mode)
                     if not np.all(np.isnan(test_mode)) else np.nan 
                     for test_mode in mode_tau_block]
    else:
        threshold_list = [np.nan_to_num(ind_list, nan=-50).tolist()
                        for ind_list in threshold_tau_block]
        mode_list = [np.nan_to_num(ind_list, nan=-50).tolist()
                    for ind_list in mode_tau_block]
        
    return ii, jj, threshold_list, mode_list, candidate_tile_coords


def compute_threshold_max_bound(intensity_array,
                                is_bimodal,
                                metric_obj=None):
    int_array_db = convert_pow2db(intensity_array)
    if is_bimodal:
        if metric_obj.optimization:
            int_sub_mean, int_sub_std, _ = metric_obj.first_mode
        else:
            thres_temp = threshold_otsu(int_array_db)
            valid_samples = int_array_db[int_array_db > thres_temp]
            int_sub_mean = np.nanmean(valid_samples)
            int_sub_std = np.nanstd(valid_samples)
    else:
        int_sub_mean = np.nanmean(int_array_db)
        int_sub_std = np.nanstd(int_array_db)

    return np.nanmax([np.percentile(int_array_db, 99.5) + int_sub_std * 2,
                      int_sub_mean + int_sub_std * 2]), int_sub_mean, int_sub_std


def create_geotiff_with_one_value(outpath, shape, filled_value):
    """
    Create a new GeoTIFF file filled with a specified value.

    Parameters:
    ----------
    outpath: str
        The file path where the new GeoTIFF will be saved.
    shape: tuple
        A tuple (height, width) representing the dimensions of the GeoTIFF.
    filled_value: float
        The value with which the GeoTIFF will be filled.
    """
    # Set up the new file's spatial properties
    height, width= shape

    # Create the file with a single band, Float32 type
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(outpath, width, height, 1, gdal.GDT_Float32)

    # Write zeros to the raster band
    band = ds.GetRasterBand(1)
    band.WriteArray(np.full((height, width), filled_value, dtype=np.float32))
    band.FlushCache()

    ds = None  # Close the file


def run(cfg):
    """
    Run inital threshold with parameters in cfg dictionary
    """
    t_all = time.time()
    logger.info('Start Initial Threshold')

    processing_cfg = cfg.groups.processing
    pol_list = processing_cfg.polarizations
    pol_all_str = '_'.join(pol_list)

    outputdir = cfg.groups.product_path_group.scratch_path

    # options for initial threshold
    init_threshold_cfg = processing_cfg.initial_threshold
    number_iterations = init_threshold_cfg.number_iterations
    number_workers = init_threshold_cfg.number_cpu
    tile_selection_method = init_threshold_cfg.selection_method
    average_threshold_flag = init_threshold_cfg.tile_average
    threshold_extending_method = init_threshold_cfg.extending_method
    logger.info(f'Tile selection method: {tile_selection_method}')
    logger.info(f'Average_threshold_flag: {average_threshold_flag}')

    # options for reference water
    ref_water_cfg = processing_cfg.reference_water
    drought_erosion_pixel = ref_water_cfg.drought_erosion_pixel
    flood_dilation_pixel = ref_water_cfg.flood_dilation_pixel
    permanent_water_value = ref_water_cfg.permanent_water_value

    # Read filtered RTC image
    filt_im_str = os.path.join(outputdir,
                               f"filtered_image_{pol_all_str}.tif")
    filt_raster_tif = gdal.Open(filt_im_str)

    #read metadata from intensity image (projection, geotransform)
    water_meta = dswx_sar_util.get_meta_from_tif(filt_im_str)

    band_number = filt_raster_tif.RasterCount
    height, width = filt_raster_tif.RasterYSize, filt_raster_tif.RasterXSize

    # READ relocated reference water
    wbd_im_str = os.path.join(outputdir, 'interpolated_wbd.tif')
    wbd_gdal = gdal.Open(wbd_im_str)

    # compute the statistics of intensity.
    ref_water_max = processing_cfg.reference_water.max_value
    wbd_whole = wbd_gdal.ReadAsArray()
    wbd_whole_norm = wbd_whole / ref_water_max

    # create water masks for normal,
    # flood and drought using dilation and erosion
    water_mask_set = create_three_water_masks(
        wbd_whole,
        water_threshold=0.9,
        no_data=processing_cfg.reference_water.no_data_value,
        wbd_max_value=ref_water_max,
        drought_erosion_pixel=drought_erosion_pixel,
        flood_dilation_pixel=flood_dilation_pixel)

    water_mask_tif_str = os.path.join(outputdir,
                                      f"water_mask_{pol_all_str}.tif")
    dswx_sar_util.save_raster_gdal(water_mask_set,
                                   water_mask_tif_str,
                                   geotransform=water_meta['geotransform'],
                                   projection=water_meta['projection'],
                                   scratch_dir=outputdir,
                                   datatype='uint8')

    intensity_whole = filt_raster_tif.ReadAsArray()

    if band_number == 1:
        intensity_whole = intensity_whole[np.newaxis, :, :]

    valid_area = np.squeeze(intensity_whole[0, :, :] > 0)
    water_pixel_number = np.sum(water_mask_set[0, valid_area])
    valid_pixel_number = np.sum(intensity_whole[0, :, :] > 0)

    water_portion = water_pixel_number / valid_pixel_number
    logger.info(f'water spatial coverage : {water_portion} ')  

    thres_max = np.empty([band_number])
    threshold_iteration = number_iterations
    initial_water_set = np.ones([height, width, len(pol_list)])

    if water_portion == 1:
        # If the areas cover only water,
        # then use the very high threshold to classify all pixels as water. 
        for band_ind in range(band_number):
            pol_str = pol_list[band_ind]
            thresh_file_str = os.path.join(outputdir,
                                           f"int_threshold_filled_{pol_str}.tif")
            thresh_peak_str = os.path.join(outputdir, f"mode_tau_filled_{pol_str}.tif")
            for filled_file_path in [thresh_file_str, thresh_peak_str]:
                create_geotiff_with_one_value(filled_file_path, shape=[height, width],
                                              filled_value=30)
    else:
        # Here we compute the bounds of the backscattering of water objects
        for iter_ind in range(threshold_iteration):
            logger.info(f'iterations : {iter_ind + 1} of {number_iterations}')

            for band_ind in range(band_number):
                if pol_list[band_ind] == 'span':
                    thres_max[band_ind] = 30
                else:
                    int_water_array = intensity_whole[
                        band_ind, (wbd_whole_norm > permanent_water_value) &
                        (np.nansum(initial_water_set, axis=2) > 1)]
                    int_water_array = remove_invalid(int_water_array)

                    if len(int_water_array):
                        metric_obj = refine_with_bimodality.BimodalityMetrics(int_water_array)
                        is_bimodal = metric_obj.compute_metric()

                        thres_max[band_ind], int_sub_mean, int_sub_std = compute_threshold_max_bound(
                            int_water_array, is_bimodal, metric_obj)
                    else:
                        int_sub_mean, int_sub_std, thres_max[band_ind] = np.nan, np.nan, np.nan
                        is_bimodal = False

                    logger.info(f'mean  intensity [dB] over water {pol_list[band_ind]}:'
                                f' {int_sub_mean:.2f}, {is_bimodal}')
                    logger.info(f'std   intensity [dB] over water {pol_list[band_ind]}:'
                                f' {int_sub_std:.2f}, {is_bimodal}')
                    logger.info(f'max bound intensity [dB] over water {pol_list[band_ind]}:'
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
                delayed(process_block)(ii, jj,
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

                threshold_tau_dict = save_threshold_dict(
                    threshold_tau_set,
                    block_row,
                    block_col)
                mode_tau_dict = save_threshold_dict(
                    mode_tau_set,
                    block_row,
                    block_col)
                testtest = threshold_tau_set[:, :, 0]
                testtest[testtest == -50] = np.nan
                testtest = threshold_tau_set[:, :, 1]
                testtest[testtest == -50] = np.nan

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
                            window_center_row_list = [int(ii * block_row +
                                                        (sub_window[1] +
                                                        sub_window[2])/2)
                                                    for sub_window in
                                                    window_coord]

                            window_center_col_list = [int(jj * block_col +
                                                        (sub_window[3] +
                                                        sub_window[4])/2)
                                                    for sub_window in
                                                    window_coord]
                            # window coordinates
                            absolute_window_coord = [[ii * block_row +
                                                    sub_window[1],
                                                    ii * block_row +
                                                    sub_window[2],
                                                    jj * block_col +
                                                    sub_window[3],
                                                    jj * block_col +
                                                    sub_window[4]]
                                                    for sub_window in
                                                    window_coord]
                            coord_row_list[pol_index] = coord_row_list[pol_index] + \
                                window_center_row_list
                            coord_col_list[pol_index] = coord_col_list[pol_index] + \
                                window_center_col_list
                            threshold_tau_set[pol_index].extend(threshold_tau_subset[0])
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
                interp_thres_str_list = ['int_threshold_filled', 'mode_tau_filled']
                for dict_thres, thres_str in zip(dict_threshold_list,
                                                interp_thres_str_list):
                    fill_threshold_with_gdal(
                        threshold_array=dict_thres,
                        rows=height,
                        cols=width,
                        filename=thres_str,
                        outputdir=outputdir,
                        pol_list=pol_list,
                        filled_value=thres_max,
                        no_data=-50,
                        average_tile=average_threshold_flag)

        # create initial map for iteration method
        if processing_cfg.debug_mode:
            initial_water_set = np.zeros([height, width, len(pol_list)])

            for polind, pol in enumerate(pol_list):
                if threshold_extending_method == 'gdal_grid':
                    thresh_file_str = os.path.join(outputdir,
                                                f"int_threshold_filled_{pol}.tif")
                    threshold_grid = dswx_sar_util.read_geotiff(thresh_file_str)

                initial_water_binary = convert_pow2db(np.squeeze(
                    intensity_whole[polind, :, :])) < threshold_grid
                no_data_raster = np.squeeze(intensity_whole[polind, :, :]) == 0

                initial_water_set[:, :, polind] = initial_water_binary

                water_tif_str = os.path.join(outputdir,
                                            f"initial_water_{pol}_{iter_ind}.tif")

                dswx_sar_util.save_dswx_product(
                    initial_water_binary,
                    water_tif_str,
                    geotransform=water_meta['geotransform'],
                    projection=water_meta['projection'],
                    description='Water classification (WTR)',
                    scratch_dir=outputdir,
                    no_data=no_data_raster)

                dswx_sar_util.save_raster_gdal(
                    data=threshold_grid,
                    output_file=os.path.join(outputdir,
                                            f"int_threshold_filled_{pol}_georef.tif"),
                    geotransform=water_meta['geotransform'],
                    projection=water_meta['projection'],
                    scratch_dir=outputdir)

            if not average_threshold_flag:
                dswx_sar_util.block_threshold_visulaization_rg(
                    intensity_whole,
                    threshold_tau_dict,
                    outputdir=outputdir,
                    figname=f'int_threshold_{iter_ind}iter_')
            else:
                for band_ind2 in range(band_number):
                    dswx_sar_util.block_threshold_visulaization(
                        np.squeeze(intensity_whole[band_ind2, :, :]),
                        block_row,
                        block_col,
                        threshold_tau_set[:, :, band_ind2],
                        outputdir,
                        f'int_threshold_{pol_list[band_ind2]}_{iter_ind}')

    filt_raster_tif.FlushCache()
    filt_raster_tif = None
    wbd_gdal.FlushCache()
    wbd_gdal = None
    del intensity_whole

    t_all_elapsed = time.time() - t_all
    logger.info(f"successfully ran computing initial threshold in "
                f"{t_all_elapsed:.3f} seconds")


def main():

    parser = _get_parser()

    args = parser.parse_args()

    generate_log.configure_log_file(args.log_file)

    mimetypes.add_type("text/yaml", ".yaml", strict=True)
    flag_first_file_is_text = 'text' in mimetypes.guess_type(
        args.input_yaml[0])[0]

    if len(args.input_yaml) > 1 and flag_first_file_is_text:
        logger.info('ERROR only one runconfig file is allowed')
        return

    if flag_first_file_is_text:
        cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_s1', args)

    run(cfg)


if __name__ == '__main__':
    main()
