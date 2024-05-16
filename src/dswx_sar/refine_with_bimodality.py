import copy
import logging
import mimetypes
import os
import time

import cv2
from joblib import Parallel, delayed
import numpy as np
import scipy
from scipy import ndimage, stats
from scipy.optimize import curve_fit
from skimage.filters import (threshold_otsu,
                             threshold_multiotsu)
import gc

from dswx_sar import (dswx_sar_util,
                      generate_log,
                      masking_with_ancillary)
from dswx_sar.dswx_runconfig import (_get_parser,
                                     RunConfig,
                                     DSWX_S1_POL_DICT)

logger = logging.getLogger('dswx_sar')


class BimodalityMetrics:
    '''Estimate metrics for bimodality'''

    def __init__(self,
                 intensity_array,
                 hist_min=-32,
                 hist_max=-5,
                 hist_num=200,
                 gauss_dist_thres_bound=None):
        """Initialized BimodalityMetrics Class with intensity array

        Parameters
        ----------
        intensity_array : np.ndarray
            intensity array in linear scale
        hist_min : float
            minimum value to build histogram in dB
        hist_max : float
            maximum value to build histogram in dB
        hist_num : float
            number of histogram bins
        gauss_dist_thres_bound: list
            bound for threshold to fit the bimodal
            distribution
        """
        if gauss_dist_thres_bound is None:
            gauss_dist_thres_bound = [-18, 0]
        self.intensity_array = intensity_array.flatten()
        int_db = 10 * np.log10(self.intensity_array)
        self.int_db = int_db

        bins_hist = np.linspace(hist_min,
                                hist_max,
                                hist_num + 1)

        self.counts, bins = np.histogram(int_db,
                                         bins=bins_hist,
                                         density=True)
        self.bincenter = (bins[:-1] + bins[1:]) / 2
        self.binstep = bins[2] - bins[1]

        # remove invalid values
        mask = (np.isnan(int_db)) | (np.isinf(int_db)) | (np.isinf(-int_db))
        int_db = int_db[np.invert(mask)]
        if len(int_db) >= 3:
            self.enough_number = True
            # Threshold for two Gaussian fitting
            self.threshold_global_otsu = threshold_otsu(int_db)

            # If the threshold is too low,
            # then apply multi-threshold algorithm
            if self.threshold_global_otsu < gauss_dist_thres_bound[0]:
                try:
                    multithreshold_global_otsu = \
                        threshold_multiotsu(int_db)

                    if any(multithreshold_global_otsu >=
                           gauss_dist_thres_bound[0]):
                        self.threshold_global_otsu_ind = \
                            np.where((multithreshold_global_otsu >
                                      gauss_dist_thres_bound[0]) &
                                     (multithreshold_global_otsu <
                                      gauss_dist_thres_bound[1]))
                        if self.threshold_global_otsu_ind[0]:
                            self.threshold_global_otsu = \
                                multithreshold_global_otsu[
                                    self.threshold_global_otsu_ind[0][0]]

                except ValueError:
                    logger.info('multiotsu method failed.')

            self.prob = self.counts * self.binstep

            # Initial values for curve fitting using threshold computed above
            left_sample = int_db[int_db < self.threshold_global_otsu]
            right_sample = int_db[int_db > self.threshold_global_otsu]
            if len(left_sample) > 0 and len(right_sample) > 0:
                mean_lt = np.nanmean(left_sample)
                mean_gt = np.nanmean(right_sample)
                std_lt = np.std(left_sample)
                std_gt = np.std(right_sample)
            else:
                mean_lt = self.threshold_global_otsu - 1
                mean_gt = self.threshold_global_otsu + 1
                std_lt, std_gt = 1, 1

            amp_lt_ind = np.abs(self.bincenter - mean_lt).argmin()
            amp_lt = self.prob[amp_lt_ind]
            amp_gt_ind = np.abs(self.bincenter - mean_gt).argmin()
            amp_gt = self.prob[amp_gt_ind]

            try:
                # starting value for curve_fit
                # mean, std, amplitude, mean, std, amplitude
                expected = (mean_lt, std_lt, amp_lt,
                            mean_gt, std_gt, amp_gt)
                params, _ = curve_fit(self.bimodal,
                                      self.bincenter,
                                      self.prob,
                                      expected,
                                      bounds=(
                                        (-30, 1e-10, 0,
                                         -30, 1e-10, 0),
                                        (5, 5, 1,
                                         5, 5, 1)))
                if params[0] > params[3]:
                    self.second_mode = params[:3]
                    self.first_mode = params[3:]
                else:
                    self.first_mode = params[:3]
                    self.second_mode = params[3:]
                # Left Gaussian
                self.simul_first = self.gauss(self.bincenter,
                                              *self.first_mode)
                self.simul_second = self.gauss(self.bincenter,
                                               *self.second_mode)
                self.simul_all = self.simul_first + self.simul_second
                self.optimization = True

            except ValueError:
                logger.info('ValueError: Bimodal curve Fitting fails in '
                            'BimodalityMetrics.')
                self.optimization = False
            except RuntimeError:
                logger.info('RuntimeError: Bimodal curve Fitting fails in '
                            'BimodalityMetrics.')
                self.optimization = False
        else:
            self.optimization = False
            self.enough_number = False

    def gauss(self, array, mu, sigma, amplitude):
        """ Calculate the value of a Gaussian (normal) function.

        Parameters
        ----------
        array : float or array-like
            The value(s) at which to evaluate the Gaussian.
        mu: float
            The mean of the Gaussian.
        sigma: float
            The standard deviation of the Gaussian.
        amplitude: float
            Amplitude of the Gaussian.

        Returns
        ----------
        float or array-like
            The value(s) of the Gaussian function at x.
        """
        return amplitude * np.exp(-(array - mu)**2 / 2 / sigma ** 2)

    def bimodal(self, array, mu1, sigma1, amplitud1,
                mu2, sigma2, amplitud2):
        """ Calculate the value of a bimodal Gaussian (normal) function.

        Parameters
        ----------
        array : float or array-like
            The value(s) at which to evaluate the Gaussian.
        mu1: float
            The mean of the first Gaussian.
        sigma1: float
            The standard deviation of the first Gaussian.
        amplitud1: float
            Amplitude of the first Gaussian.
        mu2: float
            The mean of the second Gaussian.
        sigma2: float
            The standard deviation of the second Gaussian.
        amplitud2: float
            Amplitude of the second Gaussian.

        Returns
        ----------
        float or array-like
            The value(s) of the Gaussian function at x.
        """
        return self.gauss(array, mu1, sigma1, amplitud1) + \
            self.gauss(array, mu2, sigma2, amplitud2)

    def compute_ashman(self):
        """Compute the Ashman coefficient for bimodality estimation.

        This method calculates the Ashman coefficient,
        which is a measure of bimodality,
        based on the first and second distributions of the input histogram.

        Returns:
        ----------
            float: The computed Ashman coefficient.

        """
        numerator = np.sqrt(2) * np.abs(self.first_mode[0]
                                        - self.second_mode[0])
        denominator = np.sqrt(self.first_mode[1] ** 2
                              + self.second_mode[1] ** 2)
        ashman_coeff = numerator / denominator
        return ashman_coeff

    def compute_bhc(self):
        """
        Compute the Bhattacharyya coefficient for bimodality estimation.

        This method calculates the Bhattacharyya coefficient, which is
        a measure of bimodality, based on
        the normalized histograms of simulated and observed data.

        Returns:
        ----------
            float: The computed Bhattacharyya coefficient.
        """
        simul_all_norm = self.simul_all / np.sum(self.simul_all)
        counts_norm = self.counts / np.sum(self.counts)
        bhc_coeff = np.sum(np.sqrt(simul_all_norm * counts_norm))
        return bhc_coeff

    def compute_surface_ratio(self):
        """Compute the Surface Ratio coefficient
        for bimodality estimation.

        This method calculates the Surface Ratio coefficient,
        which is a measure of bimodality, based on
        the areas under the first and second modes of
        the simulated data histograms.

        Returns:
        ----------
            float: The computed Surface Ratio coefficient.
        """
        area_first = np.sum(self.simul_first)
        area_second = np.sum(self.simul_second)
        surface_ratio_coeff = np.nanmin([area_first, area_second]) / \
            np.max([area_first, area_second])
        return surface_ratio_coeff

    def compute_bc_coefficient(self):
        """Compute the BC coefficient for bimodality estimation.

        This method calculates the BC coefficient,
        which is a measure of bimodality,
        based on the skewness and kurtosis of the input data.

        Returns:
        ----------
            float: The computed BC coefficient.
      """
        sample_size = len(self.int_db)
        skewness_sq = stats.skew(self.int_db,
                                 nan_policy='omit',
                                 bias=False) ** 2
        kurtosis = stats.kurtosis(self.int_db,
                                  nan_policy='omit',
                                  bias=False)
        adjustment = 3 * ((sample_size - 1) ** 2) / (
            sample_size - 2) / (sample_size - 3)
        bc_coeff = (skewness_sq + 1) / (kurtosis + adjustment)
        return bc_coeff

    def compute_bimodality(self):
        """
        Compute the bimodality coefficient based on histogram analysis.

        This method estimates the bimodality coefficient,
        which measures the degree of bimodality in the
        input data. It uses histogram analysis to identify
        the bin corresponding to the valley between the
        two modes of the data. Then, it calculates the bimodality
        coefficient based on the mean values and
        probabilities of the two modes.

        Returns:
        ----------
            float: The computed bimodality coefficient.
        """
        try:
            local_left_ind = np.argmax(self.simul_first)
            local_right_ind = np.argmax(self.simul_second)

            if self.simul_first.ndim == 1:
                start_ind = max(0, local_left_ind - 1)
                end_ind = min(local_right_ind, len(self.simul_all))
            else:
                start_ind = max(0, local_left_ind[0] - 1)
                end_ind = min(local_right_ind[0], len(self.simul_all))

            local_min_ind = np.argmin(self.simul_all[start_ind:end_ind]) + \
                start_ind

            value = self.bincenter[local_min_ind]
            cand_lte = self.bincenter <= value
            cand_gte = self.bincenter >= value

            # if values are one of the bounds, int_db[slice] yields
            # empty array. we cannot compute sigma_b in this way.
            if self.int_db[self.int_db <= value].size > 0 and \
               self.int_db[self.int_db >= value].size > 0:
                meanp_lte = np.nanmean(self.int_db[self.int_db <= value])
                meanp_gte = np.nanmean(self.int_db[self.int_db >= value])
                probp_lte = np.nansum(self.counts[cand_lte]) * self.binstep
                probp_gte = np.nansum(self.counts[cand_gte]) * self.binstep

                var_all = np.nanvar(self.int_db)
                sigma_b = probp_lte * probp_gte * \
                    ((meanp_lte - meanp_gte) ** 2) / var_all
            else:
                sigma_b, _ = estimate_bimodality(self.int_db)
        except ValueError:
            sigma_b, _ = estimate_bimodality(self.int_db)

        return sigma_b

    def compute_metric(self,
                       ashman_flag=True,
                       bm_flag=True,
                       surface_ratio_flag=True,
                       bc_flag=True,
                       thresholds=None):
        """
        Compute the bimodality metric for the given data.

        This method computes the bimodality metric based on different
        criteria and thresholds. It takes several flags to control
        the computation of specific metrics. If optimization is enabled,
        it uses pre-computed metrics stored in the object's attributes.
        Otherwise, it calculates the bimodality metric using the
        `estimate_bimodality` function.

        Parameters
        ----------
        ashman_flag : bool
            Flag to indicate whether to consider
            the Ashman coefficient in the metric.
        bm_flag : bool
            Flag to indicate whether to consider
            the bimodality coefficient in the metric.
        surface_ratio_flag : bool
            Flag to indicate whether to consider
            the surface ratio in the metric.
        bc_flag : bool
            Flag to indicate whether to consider
            the BC coefficient in the metric.
        thresholds : list
            A list containing threshold values
            for different metrics. # ashman, bhc, surface_ratio, bm

        Returns
        -------
        bimodality_flag : bool
            A boolean value indicating whether the data satisfies
            the bimodality condition based on the
            selected metrics and thresholds.
        """
        if thresholds is None:
            thresholds = [1.5, 0.97, 0.1, 0.7]
        ashman, surface_ratio, bm_coeff, bc_coeff = \
            (None, None, None, None)
        bimodality_flag = False

        if self.enough_number:
            if self.optimization and len(self.int_db) > 4:
                (ashman, bhc, surface_ratio,
                 bm_coeff, bc_coeff) = self.get_metric()

                # Check if the data satisfies the conditions for bimodality
                bm_coeff_bool = bm_flag and \
                    (bm_coeff is None or bm_coeff > thresholds[3])
                ashman_bool = ashman_flag and \
                    (ashman is None or ashman > thresholds[0])
                surface_ratio_bool = surface_ratio_flag and \
                    (surface_ratio is None or surface_ratio > thresholds[2])
                bc_coeff_bool = bc_flag and \
                    (bc_coeff is None or bc_coeff > 5/9)
                bimodal_metrics = (int(ashman_bool) +
                                   int(bm_coeff_bool) +
                                   int(bc_coeff_bool))
                # If more than two metric satisficed are higher than threshold
                # or ashman coefficient is higher than 3
                # and surface ratio is higher than threshold
                bool_set = [(bimodal_metrics >= 2) or
                            (ashman > 3),
                            surface_ratio_bool]

                if all(element for element in bool_set):
                    bimodality_flag = True
            else:
                # If optimization fails, then apply alternative way
                # Compute bimodality using the estimate_bimodality function
                bt_max, ad_max = estimate_bimodality(self.int_db)
                if (bt_max > thresholds[3]) & (ad_max > 1.5):
                    bimodality_flag = True

        return bimodality_flag

    def get_metric(self):
        """
        Calculate bimodality metrics based on the optimization flag.

        Returns:
            list: A list containing bimodality metrics.
            The list has the following elements:
                - ashman (float): The Ashman coefficient.
                - bhc (float): The Bhattacharyya coefficient.
                - surface_ratio (float): The surface ratio.
                - bm_coeff (float): The bimodality coefficient.
                - bc_coeff (float): The bc coefficient.
        """
        if self.optimization:
            ashman = self.compute_ashman()
            bhc = self.compute_bhc()
            surface_ratio = self.compute_surface_ratio()
            bm_coeff = self.compute_bimodality()
            bc_coeff = self.compute_bc_coefficient()
        else:
            ashman = 0
            bhc = 0
            surface_ratio = 0
            bm_coeff, _ = estimate_bimodality(self.int_db)
            bc_coeff = self.compute_bc_coefficient()

        return ashman, bhc, surface_ratio, bm_coeff, bc_coeff


def estimate_bimodality(array,
                        min_im=-30,
                        max_im=5,
                        numstep=100):
    ''' Quantify bimodal distribution from the histogram

    Parameters
    ----------
    array : numpy.ndarray
        intensity array
    min_im : float
        minimum range for histogram
    max_im : float
        maximum range for histogram
    numstep : integer
        number of histogram bins

    Returns
    -------
    sigma_max : float
        maximum value for estimated bimodality
    ad_max : numpy.ndarray
        maximum value for estimated bimodality
    '''
    array = array[np.invert(np.isnan(array)) & np.invert(np.isinf(array))]
    hist_bin = np.linspace(min_im, max_im, numstep + 1)
    counts, bins = np.histogram(array,
                                bins=hist_bin,
                                density=False)

    bincenter = (bins[:-1] + bins[1:]) / 2

    # smooth histogram by appling gaussian filter
    counts_smooth = scipy.signal.convolve(counts,
                                          [0.2261, 0.5478, 0.2261],
                                          'same')
    sigma_max = np.nan
    ad_max = np.nan

    if len(array) > 2:

        std_int = np.nanstd(array, ddof=1)**2

        sigma_b = np.zeros_like(bincenter)
        ads = np.zeros_like(bincenter)
        countsum = np.nansum(counts_smooth)

        for bin_idx, value in enumerate(bincenter):
            cand_left = bincenter <= value
            cand_right = bincenter >= value

            if np.any(cand_left):
                left_sum = np.nansum(counts_smooth[cand_left])
            else:
                left_sum = hist_bin[0]

            if np.any(cand_right):
                right_sum = np.nansum(counts_smooth[cand_right])
            else:
                right_sum = hist_bin[-1]

            # when number of histogram bin is not zero
            if left_sum > 0 and right_sum > 0:
                if np.any(cand_left):
                    meanp_left = \
                        np.nansum(counts_smooth[cand_left] *
                                  bincenter[cand_left]) \
                        / left_sum
                    stdp_left = np.sqrt(np.nansum(
                        ((counts_smooth[cand_left] * bincenter[cand_left])
                         - meanp_left)**2)) / left_sum
                    probp_left = np.nansum(counts_smooth[cand_left]) / countsum

                else:
                    meanp_left, stdp_left, probp_left = 0, 0, 0

                if np.any(cand_right):
                    meanp_right = \
                        np.nansum(counts_smooth[cand_right] *
                                  bincenter[cand_right]) \
                        / right_sum
                    stdp_right = np.sqrt(np.nansum(
                        ((counts_smooth[cand_right] *
                          bincenter[cand_right]) -
                         meanp_right)**2)) / right_sum
                    probp_right = \
                        np.nansum(counts_smooth[cand_right]) / countsum
                else:
                    meanp_right, stdp_right, probp_right = 0, 0, 0

                sigma_b[bin_idx] = probp_left * probp_right * (
                    (meanp_left - meanp_right)**2) / std_int
                ads[bin_idx] = np.sqrt(2) * (
                    np.abs(meanp_left - meanp_right)) / np.sqrt(
                    (stdp_left ** 2 + stdp_right ** 2))

        sigma_max = np.nanmax(sigma_b)
        ad_max = np.nanmax(ads)

    return sigma_max, ad_max


def process_dark_land_component(args):
    """
    Process a dark land component and compute bimodality metric.

    This function takes a set of input arguments and processes
    a dark land component using various raster datasets such as
    landcover, intensity bands, binary water mask, land areas,
    and labeled water elements. It computes the bimodality metric
    for the component based on certain conditions and returns the
    results.

    Parameters
    ----------
    args : tuple
        A tuple containing the following elements:
            - i: int
                The index of the dark land component.
            - sizes: int
                The size of the dark land component in pixels.
            - bounds: tuple
                The bounding box of the component
                (row, col, width, height).
            - ref_land_block: np.ndarray
                numpy array to the raster dataset representing land areas.
            - pol_ind: int
                Index of the polarization band to process.
            - bands_block: np.ndarray
                numpy array of the intensity bands raster dataset.
            - water_label_block: np.ndarray
                numpy array of the labeled water elements raster dataset.
            - thresholds: list
                List of the thresholds to determine bimiodality.
            - minimum_pixel (int): minimum number of pixels to accept
              as water bodies.
            - debug_mode: boolean
                Flag indicating whether to enable debug mode.
            - startline: int
                start line of the block
            - blocksize: int
                block size of the data

    Returns
    -------
    tuple:
        A tuple containing the results for the dark land component:
            - i (int): The index of the dark land component.
            - bimodality_array_i (bool): True if bimodality metric
              is computed successfully,
              False otherwise.
            - ref_land_portion (float): The portion of the land within
              binary water elements.
            - metric_output_i (numpy.ndarray): An array of size 5
              containing metric output
              values if debug_mode is True, otherwise, it contains zeros.
    """
    (i, sizes, bounds, ref_land_block,
     pol_ind, bands_block, water_label_block, thresholds,
     minimum_pixel, debug_mode, startline, blocksize) = args

    bounds[3] = min(bounds[3], blocksize+startline)
    bounds[2] = max(bounds[2], startline)
    # bounding box covering the water
    x_off, _, y_off, _ = bounds
    width = bounds[1] - bounds[0]
    height = bounds[3] - bounds[2]

    # Assign the datasets to their respective variables
    # landcover, bands, ref_land, water_label = raster_datasets
    if bands_block.ndim == 2:
        bands_block = np.expand_dims(bands_block, axis=0)

    bands = bands_block[pol_ind,
                        y_off-startline:y_off+height-startline,
                        x_off:x_off+width]
    ref_land = ref_land_block[y_off-startline:y_off+height-startline,
                              x_off:x_off+width]
    water_label = water_label_block[
        y_off-startline:y_off+height-startline,
        x_off:x_off+width]

    if bands.ndim == 2:
        bands = bands[np.newaxis, :, :]
    # Identify out of boundary areas.
    out_boundary = (np.isnan(np.sum(bands, axis=0)) == 0) & (water_label == 0)

    # Prepare array for 5 metrics
    metric_output_i = np.zeros(5)
    watermask = water_label == i + 1

    # water mask == 1 represents areas where water is located from
    # previous step. landcover == 0  is the no-data area
    # (landcover != 0)) may need to be added
    mask = np.array((watermask == 1))

    # ref_land consists of 0 and 1 values
    ref_land_masked = ref_land[mask]
    # compute the portion of the land within binary water elements
    ref_land_portion = np.nanmean(ref_land_masked)

    # estimate bimodality only when pixel size is larger than 4
    # and ref_land_portion
    if sizes >= minimum_pixel and not np.isnan(ref_land_portion):

        # process only when land is dominant
        if ref_land_portion > 0.8:
            margin = int((np.sqrt(2) - 1.2) * np.sqrt(sizes))
            if margin == 0:
                margin = 1
            # apply dilation to binary image
            mask_buffer = ndimage.binary_dilation(watermask == 1,
                                                  iterations=margin,
                                                  mask=out_boundary)
            single_band = bands[pol_ind, ...]

            # compute median value for polygons
            intensity_center = np.nanmedian(single_band[watermask == 1])
            intensity_adjacent_area = single_band[(watermask == 0) &
                                                  (mask_buffer == 1)]

            intensity_adjacent_low = np.nanpercentile(intensity_adjacent_area,
                                                      15)

            # If polygon is brighter than adjancet pixels
            # the polygon does not belong to the dark land.
            # we don't compute bimodality
            if intensity_center > intensity_adjacent_low:
                bimodality_array_i = False
            else:
                intensity_array = single_band[mask_buffer]
                int_mask = (np.isnan(intensity_array)) | \
                           (intensity_array == 0)
                intensity_array = intensity_array[np.invert(int_mask)]
                # BimodalityMetrics requires at least 4 pixels
                if len(intensity_array) > 4:
                    metric_obj = BimodalityMetrics(intensity_array)
                    bimodality_array_i = metric_obj.compute_metric(
                        thresholds=thresholds)

                    if debug_mode:
                        metric_output_i = metric_obj.get_metric()
                else:
                    # if intensity array is empty
                    bimodality_array_i = False
        else:
            # if the water body candiates covers the water,
            # we don't compute bimodality
            bimodality_array_i = True
    else:
        # if the water body candiates are too small,
        # we compare the ref_land_portion, which is binary image of the land.
        # If target area is only land, the area is deleted.
        if ref_land_portion == 1:
            bimodality_array_i = False
        else:
            bimodality_array_i = True

    return i, bimodality_array_i, ref_land_portion, metric_output_i


def process_bright_water_component(args):
    """
    Process a bright water component and estimate bimodality metrics.
    This function takes a set of input arguments and processes a bright
    water component using various raster datasets such as landcover,
    intensity bands, and labeled water elements. It estimates bimodality
    metrics for the component based on certain conditions and thresholds.

    Parameters
    ----------
    args : tuple
    A tuple containing the following elements:
        - ind_bright_water: int
            The index of the bright water component.
        - sizes: int
            The size of the bright water component in pixels.
        - bounds: tuple
            The bounding box of the component
            (row, col, width, height).
        - output_water_block: np.ndarray
            numpy array of the labeled water elements raster dataset.
        - landcover_block: np.ndarray
            numpy array of the landcover raster dataset.
        - bands_block: np.ndarray
            numpy array of the intensity bands raster dataset.
        - ref_land_block: np.ndarray
            numpy array to the raster dataset representing land areas.
        - pol_ind: int
            Index of the polarization band to process.
        - threshold: tuple
            A tuple containing two threshold values for Bt
            and Ad metrics.
        - startline: int
            start line of the block
        - blocksize: int
            block size of the data

    Returns
    -------
    tuple
        A tuple containing the following results
        for the bright water component
            - Btmax (float): The estimated Bt metric value.
            - ADmax (float): The estimated AD metric value.
            - ind_bright_water (int): The index of the bright water component.
    """
    (ind_bright_water, sizes, bounds, output_water_block,
     landcover_block, bands_block, ref_land_block, pol_ind,
     threshold, startline, blocksize) = args

    # bounding box covering the bright waters
    bounds[3] = min(bounds[3], blocksize + startline)
    bounds[2] = max(bounds[2], startline)

    width = bounds[1] - bounds[0]
    height = bounds[3] - bounds[2]

    if bands_block.ndim == 2:
        bands_block = np.expand_dims(bands_block, axis=0)

    # Assume that blocks in block processing are split along
    # the y-axis only.
    slice_x = slice(bounds[0], bounds[0] + width)
    slice_y = slice(bounds[2] - startline, bounds[2] - startline + height)

    bands = bands_block[pol_ind, slice_y, slice_x]
    ref_land = ref_land_block[slice_y, slice_x]
    landcover = landcover_block[slice_y, slice_x]
    output_water = output_water_block[slice_y, slice_x]

    # Find areas where is within entire image boundary and
    # water areas (output_water == 0)
    adjacent_areas = (np.isnan(np.sum(bands, axis=0)) == 0) & \
                     (output_water == 0)
    # Fine water areas from ref_land and landcover
    landcover_water = (ref_land == 0) | (landcover == 0)

    mask_water = output_water == ind_bright_water + 1

    landcover_water_target = landcover_water[mask_water]
    landcover_portion = np.mean(landcover_water_target) \
        if landcover_water_target.size > 0 else 0

    # Initially, value is set to be higher than threshold
    ad_value = threshold[1] + 0.5
    bt_value = threshold[0] + 0.5

    # if most areas are covered by water in landcover map,
    # compute bimodality
    if landcover_portion > 0.99:
        margin = int((np.sqrt(2) - 1.2) * np.sqrt(sizes))
        margin = max(margin, 5)
        mask_buffer = ndimage.binary_dilation(
                        mask_water,
                        iterations=margin,
                        mask=adjacent_areas)

        intensity_array = bands[mask_buffer]
        bt_value, ad_value = estimate_bimodality(
            10 * np.log10(intensity_array))

    return bt_value, ad_value, ind_bright_water


def remove_false_water_bimodality_parallel(water_mask_path,
                                           pol_list,
                                           thresholds,
                                           outputdir,
                                           meta_info=None,
                                           input_dict=None,
                                           minimum_pixel=4,
                                           debug_mode=False,
                                           number_workers=1,
                                           lines_per_block=500):
    """
    Remove falsely detected water bimodality from an image in parallel.
    This function identifies and processes areas of water and adjacent lands
    to verify and refine the accuracy of water detection in an image.
    This function should be used only for ['VV', 'VH', 'HH', 'HV'].
    In the case that the other polarization is given, then return input
    as it is.

    Parameters
    ----------
    water_mask_path: str
        Path of binary mask file indicating water areas in the image.
    pol_list: list
        List of polarizations (e.g., ['VV', 'VH', 'HH', 'HV']).
    thresholds : list
        A list containing threshold values
        for different metrics. # ashman, bhc, surface_ratio, bm
    outputdir: str
        Directory where the output images and metrics are saved.
    meta_info: dict
        Contains metadata of the input image such as
        'geotransform' and 'projection'.
    input_dict: dict
        Additional inputs required for processing.
        Must contain keys like 'ref_land',
        'landcover', 'intensity', etc. if provided.
    minimum_pixel: int (default=4)
        Minimum number of pixels for a water body
        to be considered for processing.
    debug_mode: bool
        If True, additional output metrics and
        images are saved for debugging purposes.
    lines_per_block: int
        lines of the block processing

    Returns
    -------
    bimodality_total: numpy.ndarray
        An image indicating the bimodality values across the entire scene.
    """
    rows, cols = meta_info['length'], meta_info['width']

    input_lines_per_block = lines_per_block
    pol_str = "_".join(pol_list)

    # To minimize memory usage, the bimodality test will be
    # carried out with the block first, and entire image will be
    # used for the remained componenets.
    lines_per_block_set = [input_lines_per_block,
                           input_lines_per_block*3,
                           input_lines_per_block*20,
                           rows]

    data_shape = [rows, cols]
    pad_shape = (0, 0)

    remove_false_path_prefix = 'remove_false_water_temp'
    remove_false_water_path_set = []

    for block_iter, lines_per_block in enumerate(lines_per_block_set):
        block_params = dswx_sar_util.block_param_generator(
            lines_per_block,
            data_shape,
            pad_shape)
        removed_false_water_path = os.path.join(
            outputdir,
            f'{remove_false_path_prefix}_{pol_str}_{block_iter}.tif')
        remove_false_water_path_set.append(removed_false_water_path)

        for block_ind, block_param in enumerate(block_params):

            logger.info(
                'remove_false_water_bimodality_parallel '
                f'block #{block_ind} from '
                f'{block_param.read_start_line} to '
                f'{block_param.read_start_line + block_param.read_length}')

            water_mask = dswx_sar_util.get_raster_block(
                water_mask_path, block_param)

            # computes the connected components labeled image of boolean image
            # and also produces a statistics output for each label
            nb_components_water, output_water, stats_water, _ = \
                cv2.connectedComponentsWithStats(water_mask.astype(np.uint8),
                                                 connectivity=8)

            nb_components_water = nb_components_water - 1
            logger.info(f'detected component number : {nb_components_water}')

            # save the water label into file
            water_label_str = os.path.join(
                outputdir, f'false_water_label_{pol_str}_{block_iter}.tif')
            dswx_sar_util.write_raster_block(
                water_label_str,
                output_water,
                block_param,
                geotransform=meta_info['geotransform'],
                projection=meta_info['projection'],
                datatype='int32',
                cog_flag=True,
                scratch_dir=outputdir)

            intensity_block = dswx_sar_util.get_raster_block(
                input_dict['intensity'], block_param)
            refland_block = dswx_sar_util.get_raster_block(
                input_dict['ref_land'], block_param)

            bimodality_set = []

            # sizes are last column and
            # bounding boxes are first to forth column.
            sizes = stats_water[1:, -1]
            bounding_boxes = stats_water[1:, :4]
            index_set = []
            component_data = {}

            for ind in range(0, nb_components_water):
                bbox_x, bbox_y, bbox_w, bbox_h = bounding_boxes[ind, :]
                size = sizes[ind]

                # Check if the component touches the boundary
                if bbox_y != 0 and \
                   (bbox_y + bbox_h) != block_param.block_length:
                    margin = int((np.sqrt(2) - 1.2) * np.sqrt(size))
                    margin = max(margin, 1)

                    sub_x_start = bbox_x - margin
                    sub_x_end = bbox_x + bbox_w + margin
                    sub_y_start = bbox_y - margin + block_param.read_start_line
                    sub_y_end = bbox_y + bbox_h + margin + \
                        block_param.read_start_line

                    # Adjust the bounds to be within the valid range
                    sub_x_start = np.maximum(sub_x_start, 0)
                    sub_y_start = np.maximum(sub_y_start, 0)
                    sub_x_end = np.minimum(sub_x_end, cols)
                    sub_y_end = np.minimum(sub_y_end, rows)

                    index_set.append(ind)
                    component_data[ind] = (ind, size,
                                           [sub_x_start, sub_x_end,
                                            sub_y_start, sub_y_end])

            for pol_ind, pol in enumerate(pol_list):
                if pol in ['VV', 'VH', 'HH', 'HV']:
                    logger.info('removing false water using bimodality '
                                f'for {pol}')
                    # 1 dimensional array for bimodality values
                    bimodality_output = np.zeros(nb_components_water,
                                                 dtype='byte')
                    check_output = np.ones(len(sizes), dtype='byte')

                    if debug_mode:
                        metric_output = np.zeros([nb_components_water, 5])
                        ref_land_portion_output = \
                            np.zeros(nb_components_water)

                    args_list = [(component_data[i][0],
                                  component_data[i][1],
                                  component_data[i][2],
                                  refland_block,
                                  pol_ind,
                                  intensity_block,
                                  output_water,
                                  thresholds,
                                  minimum_pixel,
                                  debug_mode,
                                  block_param.read_start_line,
                                  block_param.block_length)
                                 for i in component_data.keys()]
                    with Parallel(n_jobs=number_workers) as parallel:
                        results = parallel(
                            delayed(process_dark_land_component)(args)
                            for args in args_list)

                    # Assign results computed in parallel into variables
                    for result in results:
                        bimodal_ind, bimodality_array_i, \
                            ref_land_portion_output_i, metric_output_i = result
                        bimodality_output[bimodal_ind] = bimodality_array_i
                        check_output[bimodal_ind] = 0
                        if debug_mode:
                            ref_land_portion_output[bimodal_ind] = \
                                ref_land_portion_output_i
                            metric_output[bimodal_ind, :] = metric_output_i
                    if nb_components_water > 65535:
                        output_water_type = 'uint32'
                    else:
                        output_water_type = 'uint16'
                    output_water = output_water.astype(output_water_type)
                    old_val = np.arange(1, nb_components_water + 1) - .1
                    index_array_to_image = np.searchsorted(
                        old_val, output_water).astype(dtype=output_water_type)
                    bimodality_output_add = np.insert(bimodality_output, 0, 0,
                                                      axis=0)
                    bimodality_image = bimodality_output_add[
                        index_array_to_image].astype('byte')
                    del bimodality_output

                    # Skip saving the checking file in last iteration
                    if block_iter < len(lines_per_block_set)-1:
                        check_output = np.insert(check_output, 0, 0, axis=0)
                        check_image = check_output[
                            index_array_to_image].astype('byte')

                    bimodality_set.append(bimodality_image)

                    if debug_mode:
                        ref_land_portion_output = np.insert(
                            ref_land_portion_output, 0, -1, axis=0)
                        ref_land_portion_image = ref_land_portion_output[
                            index_array_to_image]
                        dswx_sar_util.write_raster_block(
                            os.path.join(
                                outputdir,
                                f'land_portion_{pol}_{block_iter}.tif'),
                            ref_land_portion_image,
                            block_param,
                            geotransform=meta_info['geotransform'],
                            projection=meta_info['projection'],
                            datatype='float32',
                            cog_flag=True,
                            scratch_dir=outputdir)

                        metric_detail_name = [
                            f'binary_ahman_{pol}_{block_iter}.tif',
                            f'binary_bhc_{pol}_{block_iter}.tif',
                            f'binary_asurface_ratio_{pol}_{block_iter}.tif',
                            f'binary_bm_coeff_{pol}_{block_iter}.tif',
                            f'binary_bc_coeff_{pol}_{block_iter}.tif']

                        metric_output = np.insert(metric_output,
                                                  0,
                                                  np.zeros([1, 5]),
                                                  axis=0)
                        for metric_ind, metric_name in enumerate(
                             metric_detail_name):

                            metric_image0 = metric_output[index_array_to_image,
                                                          metric_ind]
                            dswx_sar_util.write_raster_block(
                                os.path.join(outputdir, metric_name),
                                metric_image0,
                                block_param,
                                geotransform=meta_info['geotransform'],
                                projection=meta_info['projection'],
                                datatype='float32',
                                cog_flag=True,
                                scratch_dir=outputdir)
                    del results, bimodality_output_add
                    del bimodality_image, check_output
                    gc.collect()

            if {'HH', 'HV', 'VV', 'VH'}.intersection(set(pol_list)):
                bimodality_total = np.array(
                    np.nansum(bimodality_set, axis=0) > 0,
                    dtype='byte')
                # 0 value in output_water indicates the non-water
                bimodality_total[output_water == 0] = False
            else:
                # If the polarization is not in the list
                # ['VV', 'VH', 'HH', 'HV'],
                # Return input as it is without further modification.
                bimodality_total = water_mask
            del output_water, bimodality_set

            dswx_sar_util.write_raster_block(
                removed_false_water_path,
                bimodality_total,
                block_param,
                geotransform=meta_info['geotransform'],
                projection=meta_info['projection'],
                datatype='byte',
                cog_flag=True,
                scratch_dir=outputdir)

            # Skip saving the checking file in last iteration
            if block_iter < len(lines_per_block_set) - 1:
                # 'check_remove_false_water' has 1 value
                # for unprocessed components
                # due to its touching the boundaries and
                # has 0 value for processed components.
                check_remove_false_water_path = os.path.join(
                    outputdir,
                    f'check_remove_false_water_{pol_str}_{block_iter}.tif')
                dswx_sar_util.write_raster_block(
                    check_remove_false_water_path,
                    check_image,
                    block_param,
                    geotransform=meta_info['geotransform'],
                    projection=meta_info['projection'],
                    datatype='byte',
                    cog_flag=True,
                    scratch_dir=outputdir)
                # In last block, the input water change to entire image.
                # When dealing with the entire image, only remaining components
                # will be checked.
                if block_param.block_length + \
                   block_param.read_start_line >= rows:

                    water_mask_path = check_remove_false_water_path
                del check_image
            del bimodality_total
            gc.collect()

    if len(remove_false_water_path_set) >= 2:
        # Merge two results processed with block and entire image
        merged_removed_false_water_path = os.path.join(
            outputdir, f'merged_removed_false_water_{pol_str}.tif'
        )
        dswx_sar_util.merge_binary_layers(
            layer_list=remove_false_water_path_set,
            value_list=[1] * len(lines_per_block_set),
            merged_layer_path=merged_removed_false_water_path,
            lines_per_block=input_lines_per_block,
            mode='or',
            cog_flag=True,
            scratch_dir=outputdir)
    else:
        merged_removed_false_water_path = remove_false_water_path_set[0]

    return merged_removed_false_water_path


def fill_gap_water_bimodality_parallel(
        bright_water_path,
        pol_list,
        threshold=None,
        meta_info=None,
        outputdir=None,
        input_dict=None,
        number_workers=1,
        lines_per_block=500):
    """Fill gaps in water bodies using bimodality and
    normalized separation metrics in parallel.

    This function fills gaps in water bodies using bimodality and
    normalized separation metrics in parallel. It estimates bimodality
    and normalized separation for bright water components,
    and combines the metrics to create a binary image indicating water gaps.

    Parameters
    ----------
    bright_water_path : str
        Path indicating the binary water mask as a 2D NumPy array.
    pol_list : list
        List of polarization bands.
    threshold : list
        A list containing two threshold values for Bt and Ad metrics.
    meta_info : dict
        Metadata information such as geotransform and projection.
    outputdir : str
        Directory path for saving intermediate raster outputs.
    input_dict : dict
        A dictionary containing file paths for landcover, intensity bands,
        binary water mask, and raster dataset representing land areas.
    lines_per_block: int
        Number of lines to be used for the block processing

    Returns
    -------
    bimodal_ad_binary : numpy.ndarray
        A binary 2D NumPy array indicating the water gaps.
    """
    if threshold is None:
        threshold = [0.7, 1.5]

    rows, cols = meta_info['length'], meta_info['width']
    input_lines_per_block = lines_per_block
    pol_str = "_".join(pol_list)

    # To minimize memory usage, the bimodality test will be
    # carried out with the block first, and entire image will be
    # used for the remained componenets.
    lines_per_block_set = [lines_per_block,
                           lines_per_block * 20,
                           rows]
    data_shape = [rows, cols]
    pad_shape = (0, 0)

    fill_gap_path_prefix = 'fill_gap_water_temp'
    remove_bright_water_path_set = []

    for block_iter, lines_per_block in enumerate(lines_per_block_set):

        block_params = dswx_sar_util.block_param_generator(
            lines_per_block,
            data_shape,
            pad_shape)
        removed_bright_water_path = os.path.join(
            outputdir,
            f'{fill_gap_path_prefix}_{pol_str}_{block_iter}.tif')

        remove_bright_water_path_set.append(removed_bright_water_path)

        for block_ind, block_param in enumerate(block_params):
            logger.info(
                f'fill_gap_water_bimodality_parallel block #{block_ind} '
                f'from {block_param.read_start_line} to '
                f'{block_param.read_start_line + block_param.read_length}')
            water_mask = dswx_sar_util.get_raster_block(
                bright_water_path, block_param)
            out_boundary = dswx_sar_util.get_raster_block(
                input_dict['no_data'], block_param)
            water_mask[out_boundary == 1] = 0
            intensity_block = dswx_sar_util.get_raster_block(
                input_dict['intensity'], block_param)
            landcover_block = dswx_sar_util.get_raster_block(
                input_dict['landcover'], block_param)
            refland_block = dswx_sar_util.get_raster_block(
                input_dict['ref_land'], block_param)

            # computes the connected components labeled image of boolean image
            # and also produces a statistics output for each label
            nb_components_water, output_water, stats_water, _ = \
                cv2.connectedComponentsWithStats(water_mask.astype(np.uint8),
                                                 connectivity=8)
            if nb_components_water > 65535:
                output_water_type = 'uint32'
            else:
                output_water_type = 'uint16'
            del out_boundary, water_mask

            nb_components_water = nb_components_water - 1
            logger.info(f'detected component number : {nb_components_water}')

            water_label_str = os.path.join(
                outputdir, f'water_label_bright_water_{pol_str}.tif')
            dswx_sar_util.write_raster_block(
                water_label_str,
                output_water,
                block_param,
                geotransform=meta_info['geotransform'],
                projection=meta_info['projection'],
                datatype='int32',
                cog_flag=True,
                scratch_dir=outputdir)

            bimodality_set = np.zeros([block_param.block_length, cols],
                                      dtype='byte')

            sizes = stats_water[1:, -1]
            bounding_boxes = stats_water[1:, :4]
            index_set = []
            component_data = {}
            for ind in range(0, nb_components_water):
                bbox_x, bbox_y, bbox_w, bbox_h = bounding_boxes[ind, :]

                # Check if the component touches the boundary
                if bbox_y != 0 and \
                   (bbox_y + bbox_h) != block_param.block_length:

                    margin = int((np.sqrt(2) - 1.2) * np.sqrt(sizes[ind]))
                    sub_x_start = bbox_x - margin
                    sub_x_end = bbox_x + bbox_w + margin + 1
                    sub_y_start = bbox_y - margin + block_param.read_start_line
                    sub_y_end = bbox_y + bbox_h + \
                        margin + 1 + block_param.read_start_line

                    # Adjust the bounds to be within the valid range
                    sub_x_start = np.maximum(sub_x_start, 0)
                    sub_y_start = np.maximum(sub_y_start, 0)
                    sub_x_end = np.minimum(sub_x_end, cols)
                    sub_y_end = np.minimum(sub_y_end, rows)

                    size = sizes[ind]
                    index_set.append(ind)
                    component_data[ind] = (ind,
                                           size,
                                           [sub_x_start,
                                            sub_x_end,
                                            sub_y_start,
                                            sub_y_end])

            for pol_ind, pol in enumerate(pol_list):
                if pol in ['VV', 'VH', 'HH', 'HV']:
                    logger.info(
                        'filling bright water bodies with bimodality '
                        f'using {pol}')
                    bimodality_output = np.zeros(len(sizes),
                                                 dtype='byte')
                    check_output = np.ones(len(sizes),
                                           dtype='byte')

                    args_list = [(
                        component_data[i][0],
                        component_data[i][1],
                        component_data[i][2],
                        output_water,
                        landcover_block,
                        intensity_block,
                        refland_block,
                        pol_ind,
                        threshold,
                        block_param.read_start_line,
                        block_param.block_length)
                        for i in component_data.keys()]

                with Parallel(n_jobs=number_workers) as parallel:
                    results = parallel(
                        delayed(process_bright_water_component)(args)
                        for args in args_list)

                for res in results:
                    bt_value, ad_value, result_ind = res
                    bimodality_bright_water = \
                        (bt_value < threshold[0]) | \
                        (ad_value < threshold[1])
                    bimodality_output[result_ind] = bimodality_bright_water

                    # To avoid the duplicated jobs,
                    # the checked compoenents is recorded.
                    check_output[result_ind] = 0

                output_water = output_water.astype(output_water_type)

                old_val = np.arange(1, nb_components_water + 1) - .1
                index_array_to_image = np.searchsorted(
                    old_val, output_water).astype(output_water_type)
                bimodality_output = np.insert(bimodality_output, 0, 0, axis=0)
                check_output = np.insert(check_output, 0, 0, axis=0)

                bimodality_image = bimodality_output[
                    index_array_to_image].astype(dtype='byte')
                check_image = check_output[
                    index_array_to_image].astype(dtype='byte')
                bimodality_set += bimodality_image

            bimodal_ad_binary = bimodality_set > 0
            # 0 value in output_water indicates the non-water
            bimodal_ad_binary[output_water == 0] = False
            del bimodality_set
            gc.collect()
            dswx_sar_util.write_raster_block(
                removed_bright_water_path,
                bimodal_ad_binary,
                block_param,
                geotransform=meta_info['geotransform'],
                projection=meta_info['projection'],
                datatype='byte',
                cog_flag=True,
                scratch_dir=outputdir)

            check_fill_gap_path = os.path.join(
                outputdir, f'check_fill_gap_{pol_str}_{block_iter}.tif')

            dswx_sar_util.write_raster_block(
                check_fill_gap_path,
                check_image,
                block_param,
                geotransform=meta_info['geotransform'],
                projection=meta_info['projection'],
                datatype='byte',
                cog_flag=True,
                scratch_dir=outputdir)

        # In last block, the input water change to entire image.
        # When dealing with the entire image, only remaining components
        # will be checked.
        if block_param.block_length + block_param.read_start_line >= rows:
            bright_water_path = check_fill_gap_path

    # Merge two results processed with block and entire image
    meregd_fill_gap_layer_path = os.path.join(
        outputdir, f'merged_fill_gap_{pol_str}.tif'
    )
    dswx_sar_util.merge_binary_layers(
        layer_list=remove_bright_water_path_set,
        value_list=[1] * len(lines_per_block_set),
        merged_layer_path=meregd_fill_gap_layer_path,
        lines_per_block=input_lines_per_block,
        mode='or',
        cog_flag=True,
        scratch_dir=outputdir)

    return meregd_fill_gap_layer_path


def run(cfg):

    t_all = time.time()
    logger.info("Starting the refinement based on bimodality")

    outputdir = cfg.groups.product_path_group.scratch_path
    processing_cfg = cfg.groups.processing
    pol_list = copy.deepcopy(processing_cfg.polarizations)
    pol_options = processing_cfg.polarimetric_option

    if pol_options is not None:
        pol_list += pol_options

    pol_str = '_'.join(pol_list)
    co_pol = list(set(processing_cfg.copol) & set(pol_list))

    bimodality_cfg = processing_cfg.refine_with_bimodality
    minimum_pixel = bimodality_cfg.minimum_pixel
    threshold_set = bimodality_cfg.thresholds
    ashman_threshold = threshold_set.ashman
    bhc_threshold = threshold_set.Bhattacharyya_coefficient
    bm_threshold = threshold_set.bm_coefficient
    surface_ratio_threshold = threshold_set.surface_ratio
    number_workers = bimodality_cfg.number_cpu
    lines_per_block = bimodality_cfg.lines_per_block

    filt_im_str = os.path.join(outputdir, f"filtered_image_{pol_str}.tif")
    no_data_geotiff_path = os.path.join(
        outputdir, f"no_data_area_{pol_str}.tif")
    im_meta = dswx_sar_util.get_meta_from_tif(filt_im_str)

    # read the result of landcover masindex_array_to_imageg
    water_map_tif_str = os.path.join(
        outputdir, f'refine_landcover_binary_{pol_str}.tif')

    # read landcover map
    landcover_map_tif_str = os.path.join(
        outputdir, 'interpolated_landcover.tif')
    landcover_map = dswx_sar_util.read_geotiff(landcover_map_tif_str)
    landcover_label = masking_with_ancillary.get_label_landcover_esa_10()

    reference_water_gdal_str = os.path.join(outputdir, 'interpolated_wbd.tif')

    # Identify the non-water area from Landcover map
    if 'openSea' in landcover_label:
        landcover_not_water = (landcover_map != landcover_label['openSea']) &\
             (landcover_map != landcover_label['Permanent water bodies'])
    else:
        landcover_not_water = \
            (landcover_map != landcover_label['Permanent water bodies']) &\
            (landcover_map != landcover_label['No_data'])

    ref_land_str = os.path.join(outputdir,
                                f'landcover_not_water_{pol_str}.tif')
    dswx_sar_util.save_raster_gdal(
                    data=landcover_not_water,
                    output_file=ref_land_str,
                    geotransform=im_meta['geotransform'],
                    projection=im_meta['projection'],
                    scratch_dir=outputdir)
    del landcover_not_water, landcover_map

    # If the landcover is non-water,
    # compute the bimnodality one more time
    # and remove the water body if test fails.
    input_file_dict = {'intensity': filt_im_str,
                       'landcover': landcover_map_tif_str,
                       'reference_water': reference_water_gdal_str,
                       'water_mask': water_map_tif_str,
                       'ref_land': ref_land_str,
                       'no_data': no_data_geotiff_path}

    # Identify waters that have not existed and
    # remove if bimodality does not exist
    bimodal_binary_path = \
        remove_false_water_bimodality_parallel(
            water_map_tif_str,
            pol_list=co_pol,
            thresholds=[ashman_threshold,
                        bhc_threshold,
                        surface_ratio_threshold,
                        bm_threshold],
            outputdir=outputdir,
            meta_info=im_meta,
            input_dict=input_file_dict,
            minimum_pixel=minimum_pixel,
            debug_mode=processing_cfg.debug_mode,
            number_workers=number_workers,
            lines_per_block=lines_per_block)

    # Identify gaps within the water bodies and fill the gaps
    # if bimodality exists
    bright_water_path = os.path.join(
        outputdir, f"bimodality_bright_water_{pol_str}.tif")

    dswx_sar_util.merge_binary_layers(
        layer_list=[bimodal_binary_path],
        value_list=[0],
        merged_layer_path=bright_water_path,
        lines_per_block=lines_per_block,
        mode='or', cog_flag=True,
        scratch_dir=outputdir)

    fill_gap_bindary_path = \
        fill_gap_water_bimodality_parallel(
            bright_water_path,
            pol_list,
            threshold=[bm_threshold,
                       ashman_threshold],
            meta_info=im_meta,
            outputdir=outputdir,
            input_dict=input_file_dict,
            number_workers=number_workers,
            lines_per_block=lines_per_block)

    water_tif_str = os.path.join(
        outputdir, f"bimodality_output_binary_{pol_str}.tif")
    dswx_sar_util.merge_binary_layers(
        layer_list=[bimodal_binary_path, fill_gap_bindary_path],
        value_list=[1, 1],
        merged_layer_path=water_tif_str,
        lines_per_block=lines_per_block,
        mode='or',
        cog_flag=True,
        scratch_dir=outputdir)

    t_time_end = time.time()
    t_all_elapsed = t_time_end - t_all
    logger.info("successfully ran bimodality test in "
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
