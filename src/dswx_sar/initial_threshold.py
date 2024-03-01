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

from dswx_sar import (dswx_sar_util,
                      generate_log,
                      refine_with_bimodality,
                      region_growing)
from dswx_sar.dswx_runconfig import (DSWX_S1_POL_DICT,
                                     _get_parser,
                                     RunConfig)


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
        intensity_db_variance = np.nanstd(intensity_db)**2

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

                probp1 = np.nansum(intensity_counts[cand1]) * \
                    intensity_bins_step
                probp2 = np.nansum(intensity_counts[cand2]) * \
                    intensity_bins_step

                sigma[bin_index] = probp1 * probp2 * (
                    (meanp1 - meanp2) ** 2) / intensity_db_variance

        max_sigma = np.nanmax(sigma)
        select_flag = max_sigma > threshold

        return max_sigma, sigma, select_flag

    def select_tile_twele(self,
                          intensity_gray,
                          mean_intensity_global,
                          thresholds=None):
        """Select tiles based on Twele's method

        Parameters
        ----------
        intensity_gray : np.ndarray
            tile of intensity image [gray scale, dB]
        mean_intensity_global : float
            global mean intensity image
        thresholds : list
            three thresholds to select tile

        Returns
        -------
        select_flag : bool
            Indicates whether the tile is selected based on Twele's method.
            Returns True if the tile is selected, and False otherwise.
        cvx : float
            The coefficient of variation for the tile, calculated as the ratio
            of the standard deviation to the mean intensity of the tile.
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
        rx = mean_int / mean_intensity_global

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
                           selection_methods=['combined'],
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
        selection_methods : list
            window type {twele, chini, bimodality, combined}
            'combined' method runs all methods

        Returns
        -------
        candidate_tile_coords : numpy.ndarray
            The x and y coordinates that pass the tile selection test.
            Each row has index, y_start, y_end, x_start, and x_end.
        '''

        if np.all(np.isnan(intensity)) or np.all(intensity == 0):
            candidate_tile_coords = []
            logger.info('No valid intensity values found.')

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
                raise ValueError("reference water image size differ from "
                                 "intensity image")

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
            selected_tile = []
            detected_box_array = []
            # convert linear to dB scale
            intensity_db = convert_pow2db(intensity)

            # covert gray scale
            intensity_gray = self.rescale_value(
                intensity_db,
                min_value=-35,
                max_value=10)

            water_area_flag = self.get_water_portion_mask(water_mask)

            mean_intensity_global = np.nanmean(intensity_gray)

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

                            validnum = np.count_nonzero(
                                ~np.isnan(intensity_sub))

                            # create subset from water mask
                            water_mask_sublock = water_mask[:, x_start:x_end,
                                                            y_start:y_end]

                            # compute valid pixels from subset
                            water_number_sample_sub = np.count_nonzero(
                                    ~np.isnan(water_mask_sublock[0, :, :]))

                            # compute spatial portion for each polarization
                            water_area_sublock_flag = \
                                self.get_water_portion_mask(water_mask_sublock)

                            if water_number_sample_sub > 0 \
                                    and water_area_sublock_flag \
                                    and validnum > num_pixel_max:

                                # Initially set flag as True
                                tile_selected_flag = True

                                if {'twele', 'combined'}.intersection(
                                        set(selection_methods)):
                                    tile_selected_flag, _, _ = \
                                        self.select_tile_twele(
                                            intensity_sub_gray,
                                            mean_intensity_global)
                                    if tile_selected_flag:
                                        selected_tile_twele.append(True)
                                    else:
                                        selected_tile_twele.append(False)
                                else:
                                    selected_tile_twele.append(False)

                                if {'chini', 'combined'}.intersection(
                                        set(selection_methods)):
                                    # Once 'combined' method is selected,
                                    # the chini test is carried when the
                                    # 'twele' method passed.
                                    if (selection_methods in ['combined']) & \
                                       (not tile_selected_flag):

                                        selected_tile_chini.append(False)

                                    else:
                                        tile_selected_flag = \
                                            self.select_tile_chini(
                                                intensity_sub
                                                )

                                        if tile_selected_flag:
                                            selected_tile_chini.append(True)
                                        else:
                                            selected_tile_chini.append(False)

                                else:
                                    selected_tile_chini.append(False)

                                if {'bimodality', 'combined'}.intersection(
                                        set(selection_methods)):

                                    if (selection_methods == 'combined') and \
                                        not tile_selected_flag:

                                        selected_tile_bimodality.append(False)
                                    else:
                                        _, _, tile_bimode_flag = \
                                            self.select_tile_bimodality(
                                                intensity_sub,
                                                threshold=self.threshold_bimodality
                                                )

                                        if tile_bimode_flag:
                                            selected_tile_bimodality.append(
                                                True
                                                )
                                        else:
                                            selected_tile_bimodality.append(
                                                False
                                                )
                                else:
                                    selected_tile_bimodality.append(False)

                            else:
                                bimodality_max_set.append(np.nan)
                                selected_tile_twele.append(False)
                                selected_tile_chini.append(False)
                                selected_tile_bimodality.append(False)

                            # keep coordiates for the searching window.
                            coordinate.append(
                                [ind_subtile,
                                 x_start, x_end,
                                 y_start, y_end])
                            ind_subtile += 1

                    if 'combined' in selection_methods:
                        selected_tile_merged = np.logical_and(
                            selected_tile_twele,
                            selected_tile_chini)
                        selected_tile_merged = np.logical_and(
                            selected_tile_merged,
                            selected_tile_bimodality)
                        num_detected_box = np.sum(selected_tile_merged)
                    else:
                        # Initialize the result array with False values
                        detected_box_array = np.ones_like(selected_tile_twele,
                                                          dtype=bool)
                        # Check and aggregate the results
                        # based on the selection_methods
                        if 'twele' in selection_methods:
                            detected_box_array = np.logical_and(
                                detected_box_array,
                                selected_tile_twele)
                        if 'chini' in selection_methods:
                            detected_box_array = np.logical_and(
                                detected_box_array,
                                selected_tile_chini)
                        if 'bimodality' in selection_methods:
                            detected_box_array = np.logical_and(
                                detected_box_array,
                                selected_tile_bimodality)
                        num_detected_box = np.sum(detected_box_array)

                    if num_detected_box_sum <= mininum_tile:
                        # try tile-selection with smaller win size
                        win_size = int(win_size * 0.5)
                        num_pixel_max = win_size * win_size / 3

                    num_detected_box_sum += num_detected_box

            else:
                logger.info('No water body found')

            if 'combined' in selection_methods:
                selected_tile = np.logical_and(
                    selected_tile_twele,
                    selected_tile_chini)
                selected_tile = np.logical_and(
                    selected_tile,
                    selected_tile_bimodality)
            else:
                selected_tile = detected_box_array
            coordinate = np.array(coordinate)
            candidate_tile_coords = coordinate[selected_tile]

        return candidate_tile_coords


def create_three_water_masks(
        wbd_im_str,
        water_set_str,
        scratch_dir,
        water_threshold,
        no_data,
        wbd_max_value,
        flood_dilation_pixel=16,
        drought_erosion_pixel=10):
    """
    Creates three water masks indicating normal, flood, and drought conditions.

    Parameters
    ----------
    wbd_im_str : str
        Path to the input raster with water presence information.
    water_set_str : str
        Path for the output water mask set.
    scratch_dir : str
        Directory path for intermediate processing and outputs.
    water_threshold : float
        Threshold for water classification (0 to 1).
    no_data : int
        Value representing 'no data' in the raster.
    wbd_max_value : int
        Maximum valid value in the raster for normalization.
    flood_dilation_pixel : int, optional
        Number of iterations for dilation to simulate flood conditions.
    drought_erosion_pixel : int, optional
        Number of iterations for erosion to simulate drought conditions.

    Returns
    -------
    water_mask_set : np.ndarray
        Array containing water masks for normal, flood, and drought conditions.
    """

    wbd_gdal = gdal.Open(wbd_im_str)
    wbd = wbd_gdal.ReadAsArray()
    del wbd_gdal
    water_meta = dswx_sar_util.get_meta_from_tif(wbd_im_str)

    wbd = np.asarray(wbd, dtype='float32')
    wbd[wbd == no_data] = wbd_max_value
    water_global_norm = wbd / wbd_max_value

    # Creating normal water mask
    water_normal_mask = water_global_norm > water_threshold

    # Assuming flood water (default: +480m)
    flood_water_mask = ndimage.binary_dilation(
        water_normal_mask,
        iterations=flood_dilation_pixel)

    # Assuming drought (default: -300m)
    drought_water_mask = ndimage.binary_erosion(
        water_normal_mask,
        iterations=drought_erosion_pixel)

    water_mask_set = np.stack([water_normal_mask,
                               flood_water_mask,
                               drought_water_mask], axis=0)

    water_mask_tif_str = os.path.join(
        scratch_dir, f"{water_set_str}")
    dswx_sar_util.save_raster_gdal(
        water_mask_set,
        water_mask_tif_str,
        geotransform=water_meta['geotransform'],
        projection=water_meta['projection'],
        scratch_dir=scratch_dir,
        datatype='uint8')


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
    numstep = int((max_intensity_histogram -
                   min_intensity_histogram) /
                  step_histogram)

    if numstep < 100:
        step_histogram = ((max_intensity_histogram -
                           min_intensity_histogram) /
                          1000)
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
    negligible_value = dswx_sar_util.Constants.negligible_value

    intensity_cumsum = np.cumsum(intensity_counts)
    # Replace zeros and negative numbers with 'negligible_value'
    intensity_cumsum = np.where(intensity_cumsum <= 0,
                                negligible_value,
                                intensity_cumsum)

    intensity_cumsum[intensity_cumsum == 0] = np.nan

    intensity_area = np.cumsum(intensity_counts * intensity_bins)
    intenisty_s = np.cumsum(intensity_counts * intensity_bins ** 2)

    var_f = intenisty_s / intensity_cumsum - \
        (intensity_area / intensity_cumsum) ** 2
    var_f = np.where(var_f <= 0, negligible_value, var_f)
    sigma_f = np.sqrt(var_f)

    cb = intensity_cumsum[-1] - intensity_cumsum
    cb = np.where(cb <= 0, negligible_value, cb)

    mb = intensity_area[-1] - intensity_area
    sb = intenisty_s[-1] - intenisty_s
    var_b = sb / cb - (mb / cb) ** 2
    var_b = np.where(var_b <= 0, negligible_value, var_b)
    sigma_b = np.sqrt(var_b)

    normalized_intensity_cumsum = intensity_cumsum / intensity_cumsum[-1]
    normalized_intensity_cumsum = \
        np.where(normalized_intensity_cumsum >= 1,
                 1 - negligible_value,
                 normalized_intensity_cumsum)
    normalized_intensity_cumsum = \
        np.where(normalized_intensity_cumsum <= 0,
                 negligible_value,
                 normalized_intensity_cumsum)

    minus_cumsum = 1 - normalized_intensity_cumsum

    prob_array = \
        normalized_intensity_cumsum * np.log(sigma_f) + \
        minus_cumsum * np.log(sigma_b) - \
        normalized_intensity_cumsum * np.log(normalized_intensity_cumsum) - \
        minus_cumsum * np.log(minus_cumsum)
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

    threshold_array = []
    threshold_idx_array = []
    mode_array = []
    negligible_value = dswx_sar_util.Constants.negligible_value
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
            return np.nan, np.nan

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
            try:
                thresholds = threshold_multiotsu(intensity_sub)

                if thresholds[0] < threshold:
                    threshold = thresholds[0]
                    idx_threshold = np.searchsorted(intensity_bins, threshold)
            except ValueError:
                logger.info('Unable to find multi threshold')
        # Search the local peaks from histogram for initial values for fitting
        # peak for lower distribution
        lowmaxind_cands, _ = find_peaks(intensity_counts[0:idx_threshold+1],
                                        distance=5)
        if not lowmaxind_cands.any():
            lowmaxind_cands = np.array(
                [np.nanargmax(intensity_counts[: idx_threshold+1])])

        intensity_counts_cand = intensity_counts[lowmaxind_cands]
        lowmaxind = lowmaxind_cands[np.nanargmax(intensity_counts_cand)]

        # peak for higher distribution
        highmaxind_cands, _ = find_peaks(intensity_counts[idx_threshold:],
                                         distance=5)

        # if highmaxind_cands is empty
        if not highmaxind_cands.any():
            highmaxind_cands = np.array([
                np.nanargmax(intensity_counts[idx_threshold:])])
        intensity_counts_cand = intensity_counts[idx_threshold +
                                                 highmaxind_cands]
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
            simul_second_sum = np.nansum(simul_second)
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

            if optimization:
                threshold = optimize_inter_distribution_threshold(
                    old_threshold,
                    mean1=mean1,
                    std1=std1,
                    mean2=mean2,
                    std2=std2,
                    step_fraction=0.05,
                    max_iterations=100,
                    thresh_low_dist_percent=adjust_thresh_low_dist_percent,
                    thresh_high_dist_percent=adjust_thresh_high_dist_percent)

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


def optimize_inter_distribution_threshold(
        threshold,
        mean1,
        std1,
        mean2,
        std2,
        step_fraction=0.05,
        max_iterations=100,
        thresh_low_dist_percent=None,
        thresh_high_dist_percent=None):
    """
    Adjust the threshold between two non-overlapping Gaussian distributions.

    This function iteratively adjusts the threshold towards the mean of the
    second distribution until overlap criteria are met or the maximum number
    of iterations is reached.

    Parameters
    ----------
    threshold : float
        Initial threshold value.
    mean1 : float
        Mean of the first Gaussian distribution.
    std1 : float
        Standard deviation of the first Gaussian distribution.
    mean2 : float
        Mean of the second Gaussian distribution.
    std2 : float
        Standard deviation of the second Gaussian distribution.
    step_fraction : float, optional
        Fraction of distance to move the threshold towards the mean of the
        second distribution at each iteration (default is 0.05).
    max_iterations : int, optional
        Maximum number of iterations to adjust the threshold (default is 100).
    thresh_low_dist_percent : float, optional
        Lower distribution threshold percentage criterion for optimization.
    thresh_high_dist_percent : float, optional
        Higher distribution threshold percentage criterion for optimization.

    Returns
    -------
    threshold : float
        Optimized threshold value for separating the two distributions.

    Notes
    -----
    The function uses a while loop to iteratively adjust the threshold.
    The loop terminates when either the overlap criterion (based on cumulative
    distribution functions) is met, or the maximum number of iterations is
    reached.
    """
    if thresh_low_dist_percent is None:
        thresh_low_dist_percent = 0.98
    if thresh_high_dist_percent is None:
        thresh_high_dist_percent = 0.02
    # Adjust the threshold if the distributions do not overlap
    iteration = 0
    while (norm.cdf(threshold, mean1, std1) < thresh_low_dist_percent) and \
          (norm.cdf(threshold, mean2, std2) < thresh_high_dist_percent) and \
            iteration < max_iterations:

        # Move threshold towards the mean of the second distribution
        threshold += (mean2 - threshold) * step_fraction
        iteration += 1

    return threshold


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
    """

    if average_tile:
        tau_row, tau_col, _ = threshold_array['array'].shape
        y_tau = threshold_array['block_row'] * np.arange(0, tau_row) + \
            threshold_array['block_row'] / 2
        # y axis should be flipped
        y_tau = np.ones_like(y_tau) * rows - y_tau
        x_tau = threshold_array['block_col'] * np.arange(0, tau_col) + \
            threshold_array['block_col'] / 2
        x_arr_tau, y_arr_tau = np.meshgrid(x_tau, y_tau)

    else:
        x_coarse_grid = np.arange(0, cols + 1, 400)
        y_coarse_grid = np.arange(0, rows + 1, 400)

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
                    f" -txe 0 {cols} -tye  0 {rows} -a invdist:power=0.500:" \
                    "smoothing=1.0:radius1=" \
                    f"{threshold_array['block_row']*2}:radius2=" \
                    f"{threshold_array['block_col']*2}:angle=0.000000:" \
                    "max_points=0:min_points=1:nodata=0.000000 " \
                    f" -outsize {cols} {rows} -ot Float32"
            else:
                gdal_grid_str = \
                    "gdal_grid -zfield field_3 " \
                    f"-l data_thres_{pol}_{filename} {vrt_file} " \
                    f" {tif_file_str} " \
                    f" -txe 0 {cols} -tye 0 {rows} " \
                    f" -a invdist:power=0.5000000:smoothing=1.000000:" \
                    f"radius1={400*2}:radius2={400*2}:angle=0.000000:" \
                    "max_points=0:min_points=1:nodata=0.000 "\
                    f"-outsize {cols} {rows} -ot Float32"

            os.system(gdal_grid_str)
        else:
            logger.info('threshold array is empty')

            # Define the GeoTransform
            geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)

            # Define the projection
            projection = "EPSG:4326"

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

        dswx_sar_util._save_as_cog(
            tif_file_str,
            outputdir,
            logger,
            compression='DEFLATE',
            nbits=16)


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

        # Let's assume we have a grid and
        # we want to interpolate the values on this grid
        grid_y, grid_x = np.mgrid[0:rows, 0:cols]  # a grid in 2D

        rbf = Rbf(points[:, 0], points[:, 1], values,
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


def _get_histogram_params(polarization):
    """Define Mininum, maximum, and step for histogram
    based on polarization.
    """
    if polarization in ['VV', 'VH', 'HH', 'HV', 'span']:
        return -35, 10, 0.1
    elif polarization == 'ratio':
        return -11, 0, 0.1
    else:
        return np.nan, np.nan, np.nan


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

    tile_selection_object = TileSelection(ref_water_max=water_cfg.max_value,
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
                target_im = 10 * np.log10(intensity[polind])
            else:
                target_im = intensity[polind]
            (min_intensity_histogram,
             max_intensity_histogram,
             step_histogram) = _get_histogram_params(pol)

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
            if water_variation > 0.1:
                threshold_temp_max = thres_max[polind]
            else:
                if pol in ['VV', 'HH', 'span']:
                    threshold_temp_max = threshold_bounds_co_pol[1]
                else:
                    threshold_temp_max = threshold_bounds_cross_pol[1]

            if pol in ['VV', 'HH', 'span']:
                threshold_temp_min = threshold_bounds_co_pol[0]
            else:
                threshold_temp_min = threshold_bounds_cross_pol[0]

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


def compute_threshold_max_bound(intensity_path,
                                reference_water_path,
                                water_max_value,
                                water_threshold,
                                no_data_path,
                                lines_per_block):
    """
    Compute the threshold maximum bound and related statistics for intensity
    values in water regions.

    Parameters
    ----------
    intensity_path : str
        File path to the GeoTIFF representing the intensity image.
    reference_water_path : str
        File path to the GeoTIFF representing reference water data.
    water_max_value : float
        Maximum value used for normalizing the water body data.
    water_threshold : float
        Threshold value to determine water regions in the normalized
        water body data.
    no_data_path : str
        File path to the GeoTIFF representing no-data areas.
    lines_per_block : int
        The number of lines per block for processing the raster data.

    Returns:
    --------
    tuple:
        A tuple containing four elements:
        - List of threshold maximum bounds for each band.
        - List of mean intensity values for valid water regions
          for each band.
        - List of standard deviation of intensity values
          for valid water regions for each band.
        - Boolean indicating if the intensity distribution is bimodal.
    """
    im_meta = dswx_sar_util.get_meta_from_tif(intensity_path)
    data_shape = [im_meta['length'], im_meta['width']]
    pad_shape = (0, 0)

    thres_max_set, intensity_sub_mean_set, intensity_sub_std_set = [], [], []
    last_block = np.round(im_meta['length'] / lines_per_block)

    for band_ind in range(im_meta['band_number']):
        intensity_array_db_set = np.array([])
        intensity_water_array_set = np.array([])

        block_params = dswx_sar_util.block_param_generator(
            lines_per_block,
            data_shape,
            pad_shape)

        for block_ind, block_param in enumerate(block_params):

            intensity_array, water_body_data, no_data_area = [
                dswx_sar_util.get_raster_block(path, block_param)
                for path in [intensity_path,
                             reference_water_path,
                             no_data_path]]

            if im_meta['band_number'] == 1:
                intensity_array = intensity_array[np.newaxis, :, :]

            valid_mask = \
                (water_body_data / water_max_value > water_threshold) & \
                (no_data_area == 0)

            if valid_mask.ndim == 1:
                valid_mask = valid_mask[np.newaxis, :]
            # Get intensity array over valid regions
            # for band_ind in range(im_meta['band_number']):
            intensity_water_array = intensity_array[band_ind][valid_mask]
            valid_intensity_water_array = remove_invalid(intensity_water_array)

            if len(valid_intensity_water_array):
                intensity_water_array_set = np.append(
                    intensity_water_array_set,
                    valid_intensity_water_array)
                intensity_array_db_set = np.append(
                    intensity_array_db_set,
                    convert_pow2db(valid_intensity_water_array))

            if block_ind + 1 == last_block:
                if len(intensity_water_array_set):
                    metric_obj = refine_with_bimodality.BimodalityMetrics(
                        intensity_water_array_set)
                    is_bimodal = metric_obj.compute_metric()

                    if is_bimodal:
                        if metric_obj.optimization:
                            (intensity_sub_mean,
                             intensity_sub_std,
                             _) = metric_obj.first_mode
                        else:
                            thres_temp = threshold_otsu(intensity_array_db_set)
                            valid_samples = intensity_array_db_set[
                                intensity_array_db_set > thres_temp]
                            intensity_sub_mean = np.nanmean(valid_samples)
                            intensity_sub_std = np.nanstd(valid_samples)
                    else:
                        intensity_sub_mean = np.nanmean(intensity_array_db_set)
                        intensity_sub_std = np.nanstd(intensity_array_db_set)
                    thres_max_set.append(
                        np.nanmax([np.percentile(intensity_array_db_set,
                                                 99.5) +
                                   intensity_sub_std * 2,
                                   intensity_sub_mean +
                                   intensity_sub_std * 2]))
                    intensity_sub_mean_set.append(intensity_sub_mean)
                    intensity_sub_std_set.append(intensity_sub_std)

                else:
                    intensity_sub_mean_set.append(np.nan)
                    intensity_sub_std_set.append(np.nan)
                    thres_max_set.append(np.nan)
                    is_bimodal = False

    return (thres_max_set, intensity_sub_mean_set,
            intensity_sub_std_set, is_bimodal)


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
    im_meta = dswx_sar_util.get_meta_from_tif(water_body_data_path)
    data_shape = [im_meta['length'], im_meta['width']]
    valid_pixel_number = 0
    water_pixel_number = 0
    pad_shape = (0, 0)
    block_params = dswx_sar_util.block_param_generator(
        lines_per_block,
        data_shape,
        pad_shape)
    for block_param in block_params:
        water_body_data = dswx_sar_util.get_raster_block(
            water_body_data_path, block_param)
        no_data_area = dswx_sar_util.get_raster_block(
            no_data_path, block_param)
        water_binary = water_body_data / water_body_max > water_threshold

        valid_area = no_data_area == 0
        water_pixel_number += np.sum(water_binary[valid_area])
        valid_pixel_number += np.sum(valid_area)

    water_percentage = water_pixel_number / valid_pixel_number
    return water_percentage


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
    water_meta = dswx_sar_util.get_meta_from_tif(filt_im_str)
    band_number, height, width = [water_meta[attr_name]
                                  for attr_name in
                                  ["band_number", "length", "width"]]

    # create water masks for normal,
    # flood and drought using dilation and erosion
    water_mask_tif_name = f"water_mask_{pol_all_str}.tif"
    water_mask_tif_str = os.path.join(
        outputdir, f"{water_mask_tif_name}")
    create_three_water_masks(
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
                dswx_sar_util.create_geotiff_with_one_value(
                    filled_file_path,
                    shape=[height, width],
                    filled_value=30)
    else:
        # Here we compute the bounds of the backscattering of water objects

        thres_max, intensity_sub_mean, intensity_sub_std, is_bimodal = \
            compute_threshold_max_bound(
                intensity_path=filt_im_str,
                reference_water_path=wbd_im_str,
                water_max_value=ref_water_max,
                water_threshold=permanent_water_value,
                no_data_path=no_data_geotiff_path,
                lines_per_block=lines_per_block)

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

            threshold_tau_dict = save_threshold_dict(
                threshold_tau_set,
                block_row,
                block_col)
            mode_tau_dict = save_threshold_dict(
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

    if processing_cfg.debug_mode:

        intensity_whole = dswx_sar_util.read_geotiff(filt_im_str)
        if not average_threshold_flag:
            dswx_sar_util.block_threshold_visualization_rg(
                intensity_whole,
                threshold_tau_dict,
                outputdir=outputdir,
                figname='int_threshold_visualization_')
        else:
            for band_ind2 in range(band_number):
                dswx_sar_util.block_threshold_visualization(
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
            block_params = dswx_sar_util.block_param_generator(
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
                threshold_block = dswx_sar_util.get_raster_block(
                    thresh_file_path, block_param=block_param)
                intensity_block = dswx_sar_util.get_raster_block(
                    filt_im_str, block_param=block_param)

                initial_water_binary = convert_pow2db(np.squeeze(
                    intensity_block[polind, :, :])) < threshold_block
                if initial_water_binary.ndim == 1:
                    initial_water_binary = initial_water_binary[np.newaxis, :]
                    threshold_block = threshold_block[np.newaxis, :]

                dswx_sar_util.write_raster_block(
                    out_raster=initial_water_tif_path,
                    data=initial_water_binary,
                    block_param=block_param,
                    geotransform=water_meta['geotransform'],
                    projection=water_meta['projection'],
                    datatype='byte',
                    cog_flag=True,
                    scratch_dir=outputdir)

                dswx_sar_util.write_raster_block(
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

    generate_log.configure_log_file(args.log_file)

    mimetypes.add_type("text/yaml", ".yaml", strict=True)
    flag_first_file_is_text = 'text' in mimetypes.guess_type(
        args.input_yaml[0])[0]

    if len(args.input_yaml) > 1 and flag_first_file_is_text:
        logger.info('ERROR only one runconfig file is allowed')
        return

    if flag_first_file_is_text:
        cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_s1', args)

    processing_cfg = cfg.groups.processing
    pol_mode = processing_cfg.polarization_mode
    pol_list = processing_cfg.polarizations
    if pol_mode == 'MIX_DUAL_POL':
        proc_pol_set = [DSWX_S1_POL_DICT['DV_POL'],
                        DSWX_S1_POL_DICT['DH_POL']]
    elif pol_mode == 'MIX_SINGLE_POL':
        proc_pol_set = [DSWX_S1_POL_DICT['SV_POL'],
                        DSWX_S1_POL_DICT['SH_POL']]
    else:
        proc_pol_set = [pol_list]
    for pol_set in proc_pol_set:
        processing_cfg.polarizations = pol_set
        run(cfg)


if __name__ == '__main__':
    main()
