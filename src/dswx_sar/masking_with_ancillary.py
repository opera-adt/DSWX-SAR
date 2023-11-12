import logging
import mimetypes
import os
import time

import cv2
import numpy as np
import rasterio
from joblib import Parallel, delayed
from rasterio.windows import Window
from scipy import ndimage
from skimage.filters import threshold_multiotsu
from typing import List, Tuple

from dswx_sar import (dswx_sar_util,
                      generate_log,
                      refine_with_bimodality)
from dswx_sar.dswx_runconfig import _get_parser, RunConfig


logger = logging.getLogger('dswx_s1')

def get_label_landcover_esa_10():
    '''Get integer number information what they represent
    ESA 10 m
    https://viewer.esa-worldcover.org/worldcover/
    '''
    label = dict()

    label['Tree Cover'] = 10
    label['Shrubs'] = 20
    label['Grassland'] =30
    label['Crop'] = 40
    label['Urban'] = 50
    label['Bare sparse vegetation'] = 60
    label['Snow and Ice'] = 70
    label['Permanent water bodies'] = 80
    label['Herbaceous wetland'] =90
    label['Mangrove'] = 95
    label['Moss and lichen'] = 100
    label['No_data'] = 0

    return label


class FillMaskLandCover:
    def __init__(self, landcover_file_path):
        '''Initialize FillMaskLandCover

        Parameters
        ----------
        landcover_file_path : str
            path for the landcover file
        '''
        self.landcover_file_path = landcover_file_path
        if not os.path.isfile(self.landcover_file_path):
            raise OSError(f"{self.landcover_file_path} is not found")


    def open_landcover(self):
        '''Open landcover map

        Returns
        ----------
        landcover_map : numpy.ndarray
            2 dimensional land cover map
        '''
        landcover_map = dswx_sar_util.read_geotiff(self.landcover_file_path)
        return landcover_map


    def get_mask(self, mask_label):
        '''Obtain areas corresponding to the givin labels from landcover map

        Parameters
        ----------
        mask_label : list[str]
            list of the label

        Returns
        -------
        landcover_binary : numpy.ndarray
            binary layers
        '''
        landcover = self.open_landcover()
        landcover_label = get_label_landcover_esa_10()
        landcover_binary = np.zeros(landcover.shape, dtype=bool)
        for label_name in mask_label:
            logger.info(f'Land Cover: {label_name} is extracted.')
            temp = landcover == landcover_label[f"{label_name}"]
            landcover_binary[temp] = True

        return landcover_binary


def extract_bbox_with_buffer(
        binary: np.ndarray,
        buffer: int,
        ) -> Tuple[List[List[int]], np.ndarray, np.ndarray]:
    """Extract bounding boxes with buffer from binary image and
    save the labeled image.

    Parameters:
    ----------
    binary : np.ndarray
        Binary image containing connected components.
    buffer : int
        Buffer size to expand the bounding boxes.

    Returns:
    -------
    coord_list : Tuple[List[List[int]]
        A tuple containing the list of coordinates for each bounding
        box and the sizes of the connected components.
    sizes :  np.ndarray
        Sizes of the connected components.
    label_image : np.ndarray
        2 dimensional lebel array for each binary object
    """
    rows, cols = binary.shape

    # computes the connected components labeled image of boolean image
    # and also produces a statistics output for each label
    nb_components_water, label_image, stats_water, _ = \
        cv2.connectedComponentsWithStats(binary.astype(np.uint8),
                                         connectivity=8)
    nb_components_water -= 1

    sizes = stats_water[1:, -1]
    bboxes = stats_water[1:, :4]

    coord_list = []
    for i, (x, y, w, h) in enumerate(bboxes):
        # additional buffer areas should be balanced with the object area.
        extra_buffer = int((np.sqrt(2) - 1.2) * np.sqrt(sizes[i]))
        extra_buffer = max(extra_buffer, 1)
        buffer_all = extra_buffer + buffer

        sub_x_start = max(0, x - buffer_all)
        sub_y_start = max(0, y - buffer_all)
        sub_x_end = min(cols, x + w + buffer_all)
        sub_y_end = min(rows, y + h + buffer_all)

        coord_list.append([sub_x_start,
                           sub_x_end,
                           sub_y_start,
                           sub_y_end])

    return coord_list, sizes, label_image


def check_water_land_mixture(args):
    """Check for water-land mixture in a given region
    using bimodality metrics.
    This function processes a subset of a water map and intensity
    from a provided bounding box. It identifies regions with a bimodal
    distribution of intensity values, implying a mix of water and land.
    The function further refines these regions using local metrics.

    Parameters:
    ----------
    args : tuple
    Contains the following elements:
        * i (int): Index of the water component.
        * size (int): Size of the water component.
        * minimum_pixel (int): Minimum pixel threshold for processing.
        * bounds (list): List containing bounding box coordinates.
        * int_linear_str (str): Path to the linear intensity raster file.
        * water_label_str (str): Path to the water label raster file.
        * water_mask_str (str): Path to the water mask raster file.
        * pol_ind (int): Polarization index.

    Returns:
    -------
    i : int
        Index of the water component.
    change_flag : bool
        Flag indicating whether any change was made to the water mask.
    water_mask : numpy.ndarray
        Processed water mask.
    """
    i, size, minimum_pixel, bounds, int_linear_str, water_label_str, \
        water_mask_str, pol_ind = args

    row = bounds[0]
    col = bounds[2]
    width = bounds[1] - bounds[0]
    height = bounds[3] - bounds[2]
    window = Window(row, col, width, height)

    # read subset of water map and intensity from given bounding box
    with rasterio.open(int_linear_str) as src:
        int_linear = src.read(window=window)
        int_linear = np.squeeze(int_linear[pol_ind, :,:])

    with rasterio.open(water_label_str) as src:
        water_label = src.read(window=window)
        water_label = np.squeeze(water_label)

    with rasterio.open(water_mask_str) as src:
        water_mask = src.read(window=window)
        water_mask = np.squeeze(water_mask)

    change_flag = False
    # label for water object computed from cv2 start from 1.
    target_water = water_label == i + 1
    intensity_array = int_linear[target_water]

    if (size > minimum_pixel) and (intensity_array.size > 0):

        invalid_mask = (np.isnan(intensity_array) | (intensity_array==0))

        # check if the area has bimodality
        metric_obj = refine_with_bimodality.BimodalityMetrics(intensity_array)
        bimodality_flag = metric_obj.compute_metric()

        # extract the area out of image boundary
        out_boundary = (np.isnan(int_linear) == 0) & (water_label == 0)

        if bimodality_flag:
            logger.info(f'landcover: found bimodality {bounds}')
            out_boundary_sub = np.copy(out_boundary)

            # dilation will not be cover the out_boundary_sub area.
            out_boundary_sub[target_water] = 1

            # intensity value only for potential water areas
            int_db_sub = 10 * np.log10(intensity_array[np.invert(invalid_mask)])
            # intensity value for water and adjacent areas within bbox.
            int_db = 10 * np.log10(int_linear)

            # compute multi-thresholds
            threshold_local_otsu = threshold_multiotsu(int_db_sub, nbins=100)
            for t_ind, threshold in enumerate(threshold_local_otsu):
                # assume that the test dark area has trimodal distribution
                if t_ind == 0:
                    # assumes that area is lower than threshold and
                    # belongs to target water
                    dark_water = (int_db < threshold) & (target_water)
                else:
                    dark_water = (int_db < threshold) & \
                                 (int_db >= threshold_local_otsu[t_ind - 1]) & \
                                 (target_water)

                n_dark_water_pixels = np.count_nonzero(dark_water)
                # compute additional iteration number using size.
                # it does not need to be precise.
                add_iter = int((np.sqrt(2) - 1.2) * np.sqrt(n_dark_water_pixels))
                dark_mask_buffer = ndimage.binary_dilation(
                    dark_water,
                    iterations=add_iter,
                    mask=out_boundary_sub)
                dark_water_linear = int_linear[dark_mask_buffer]
                hist_min = np.nanpercentile(10 * np.log10(dark_water_linear), 1)
                hist_max = np.nanpercentile(10 * np.log10(dark_water_linear), 99)

                # Check if the candidates of 'dark water' has distinct backscattering
                # compared to the adjacent pixels using bimodality
                metric_obj_local = refine_with_bimodality.BimodalityMetrics(
                    dark_water_linear,
                    hist_min=hist_min,
                    hist_max=hist_max)

                if not metric_obj_local.compute_metric(
                    ashman_flag=True,
                    bhc_flag=True,
                    bm_flag=True,
                    surface_ratio_flag=True,
                    bc_flag=False):

                    water_mask[dark_water] = 0
                    change_flag = True

                if t_ind == len(threshold_local_otsu) - 1:
                    bright_water_pixels = (int_db >= threshold) & \
                                          (target_water)
                else:
                    bright_water_pixels = \
                        (int_db >= threshold) & \
                        (int_db < threshold_local_otsu[t_ind+1]) & \
                        (target_water)

                n_bright_water_pixels = np.count_nonzero(bright_water_pixels)
                add_iter = int((np.sqrt(2) - 1.2) * np.sqrt(n_bright_water_pixels))
                bright_mask_buffer = ndimage.binary_dilation(
                    bright_water_pixels,
                    iterations=add_iter,
                    mask=out_boundary_sub)
                bright_water_linear = int_linear[bright_mask_buffer]
                bright_water_linear[bright_water_linear == 0] = np.nan
                hist_min = np.nanpercentile(10 * np.log10(bright_water_linear), 2)
                hist_max = np.nanpercentile(10 * np.log10(bright_water_linear), 98)
                metric_obj_local = refine_with_bimodality.BimodalityMetrics(
                    bright_water_linear,
                    hist_min=hist_min,
                    hist_max=hist_max)

                if not metric_obj_local.compute_metric(ashman_flag=True,
                    bhc_flag=True,
                    bm_flag=True,
                    surface_ratio_flag=True,
                    bc_flag=False):
                    water_mask[bright_water_pixels] = 0
                    change_flag = True

    return i, change_flag, water_mask


def split_extended_water_parallel(
        water_mask: np.ndarray,
        pol_ind: int,
        outputdir: str,
        meta_info: dict,
        input_dict: dict,
        number_workers : int) -> np.ndarray:
    """Split extended water areas into smaller subsets based on
    bounding boxes with buffer.

    Parameters:
    -----------
    water_mask : np.ndarray
        Binary mask representing the water areas.
    pol_ind : int
        Index of the polarization.
    outputdir : str
        Directory to save the resulting water mask.
    meta_info : dict
        Metadata information including geotransform and projection.
    input_dict : dict
        Dictionary containing input data including
        intensity and water mask.

    Returns:
    -------
    water_mask : np.ndarray
        The updated water mask with extended water areas split
        into smaller subsets.
    """

    coord_list, sizes, label_image = extract_bbox_with_buffer(
        binary=water_mask,
        buffer=10)

    water_label_filename =  'water_label_landcover.tif'
    water_label_str = os.path.join(outputdir, water_label_filename)
    dswx_sar_util.save_raster_gdal(
                    data=label_image,
                    output_file=water_label_str,
                    geotransform=meta_info['geotransform'],
                    projection=meta_info['projection'],
                    scratch_dir=outputdir)
    # check if the individual water body candidates has bimodality
    # bright water vs dark water
    # water vs land
    minimum_pixel = 5000
    args_list = [(i, sizes[i], minimum_pixel, coord_list[i],
                  input_dict['intensity'],
                  water_label_str,
                  input_dict['water_mask'],
                  pol_ind) for i in range(0, len(sizes))]

    # check if the objects have heterogeneous characteristics (bimodality)
    # If so, split the objects using multi-otsu thresholds and check bimodality.
    results = Parallel(n_jobs=number_workers)(delayed(check_water_land_mixture)(args)
                                  for args in args_list)

    # If water need to be refined (change_flat=True), then update the water mask.
    for i, change_flag, water_mask_subset in results:
        if change_flag:
            water_mask[coord_list[i][2] : coord_list[i][3],
                       coord_list[i][0] : coord_list[i][1]] = water_mask_subset

    return water_mask


def compute_spatial_coverage_from_ancillary_parallel(
        water_mask: np.ndarray,
        ref_water: np.ndarray,
        mask_landcover: np.ndarray,
        outputdir: str,
        meta_info: dict,
        spatial_coverage_threshold: float = 0.5,
        number_workers: int = -1,
    ) -> np.ndarray:
    """Compute spatial coverage of water areas based on ancillary information.

    Parameters:
        water_mask : np.ndarray
            Binary mask representing the water areas.
        ref_water : np.ndarray
            Reference water information.
        mask_landcover : np.ndarray
            Mask representing landcover areas.
        spatial_coverage_threshold : float
            Threshold for spatial coverage of land.
            If the spatial coverage is higher than threshold,
            it is reclassified as land.
        outputdir : str
            Directory to save the resulting water mask.
        meta_info : dict
            Metadata information including geotransform and projection.

    Returns:
        np.ndarray: The computed mask representing the portion of water areas.
    """

    coord_list, sizes, label_image = extract_bbox_with_buffer(
        binary=water_mask,
        buffer=10)

    # Save label image into geotiff file. The labels are assigned to
    # dark land candidates. This Geotiff file will be used in parallel
    # processing to read sub-areas instead of reading entire image.
    water_label_filename = 'water_label_landcover_spatial_coverage.tif'
    water_label_str = os.path.join(outputdir, water_label_filename)
    dswx_sar_util.save_raster_gdal(
        data=label_image,
        output_file=water_label_str,
        geotransform=meta_info['geotransform'],
        projection=meta_info['projection'],
        scratch_dir=outputdir)

    nb_components_water = len(sizes)

    test_output = np.zeros(len(sizes))
    # From reference water map (0: non-water, 1: permanent water) and
    # landcover map (ex. bare/sparse vegetation), extract the areas
    # which are likely mis-classified as water.
    dry_darkland = np.logical_and(mask_landcover,
                                  ref_water < 0.05)

    ref_land_tif_str = os.path.join(outputdir,
                                    "maskedout_ancillary_2_landcover.tif")
    dswx_sar_util.save_dswx_product(
        dry_darkland.astype(np.uint8),
        ref_land_tif_str,
        geotransform=meta_info['geotransform'],
        projection=meta_info['projection'],
        description='Water classification (WTR)',
        scratch_dir=outputdir)

    # Arrange arguments to be used for parallel processing
    args_list = [(i, sizes[i], spatial_coverage_threshold, coord_list[i],
                  water_label_str, ref_land_tif_str)
                 for i in range(len(sizes))]

    # Output consists of index and 2D image consisting of True/Flase.
    # True represents the land and False represents not-land.
    results = Parallel(n_jobs=number_workers)(delayed(compute_spatial_coverage)(args)
                                  for args in args_list)
    for i, test_output_i in results:
        test_output[i] = test_output_i

    # Convert 1D array result to 2D image.
    old_val = np.arange(1, nb_components_water + 1) - 0.1
    kin = np.searchsorted(old_val, label_image)
    test_output = np.insert(test_output, 0, 0, axis=0)
    mask_water_image = test_output[kin]

    return mask_water_image


def compute_spatial_coverage(args):
    """Compute the spatial coverage of non-water areas
    based on given arguments.

    Parameters:
    -----------
    args (tuple): Tuple containing the following arguments:
        i (int): Index of the current iteration.
        size (int): Size of the current component.
        threshold (float): Threshold value for comparison.
        bounds (list): List containing the bounds of the subset.
        water_label_str (str): Path to the water label file.
        ref_land_tif_str (str): Path to the reference land file.

    Returns:
    --------
        tuple: Tuple containing the index (i) and the test output (test_output_i).
    """
    i, size, threshold, bounds, water_label_str, ref_land_tif_str = args
    row, col, width, height = bounds
    window = Window(row, col, width, height)

    # read subset of water map and intensity from given bounding box
    with rasterio.open(ref_land_tif_str) as src:
        mask_excluded_ancillary = src.read(window=window)
        mask_excluded_ancillary = np.squeeze(mask_excluded_ancillary)

    with rasterio.open(water_label_str) as src:
        water_label = src.read(window=window)
        water_label = np.squeeze(water_label)

    mask_water = water_label == i + 1

    ref_land_portion = np.nanmean(mask_excluded_ancillary[mask_water])
    test_output_i = ref_land_portion > threshold

    return i, test_output_i


def run(cfg):
    '''
    Remove the false water which have low backscattering based on the
    occurrence map and landcover map.
    '''
    logger.info('Starting DSWx-S1 masking with ancillary data')

    t_all = time.time()
    outputdir = cfg.groups.product_path_group.scratch_path
    processing_cfg = cfg.groups.processing
    pol_list = processing_cfg.polarizations
    pol_str = '_'.join(pol_list)
    co_pol_ind = 0

    water_cfg = processing_cfg.reference_water
    ref_water_max = water_cfg.max_value
    masking_ancillary_cfg = processing_cfg.masking_ancillary
    number_workers = masking_ancillary_cfg.number_cpu

    landcover_cfg = processing_cfg.masking_ancillary

    # Binary water map extracted from region growing
    water_map_tif_str = os.path.join(
        outputdir, f'region_growing_output_binary_{pol_str}.tif')
    water_map = dswx_sar_util.read_geotiff(water_map_tif_str)
    water_meta = dswx_sar_util.get_meta_from_tif(water_map_tif_str)

    # Filtered SAR intensity image
    filt_im_str = os.path.join(outputdir, f"filtered_image_{pol_str}.tif")
    band_set = dswx_sar_util.read_geotiff(filt_im_str)

    # Reference water map
    interp_wbd_str = os.path.join(outputdir, 'interpolated_wbd')
    interp_wbd = dswx_sar_util.read_geotiff(interp_wbd_str)
    interp_wbd = np.array(interp_wbd, dtype='float32')

    # Worldcover map
    landcover_path_str = os.path.join(outputdir, 'interpolated_landcover')

    # Identify target areas from landcover
    mask_obj = FillMaskLandCover(landcover_path_str)
    mask_excluded_landcover = mask_obj.get_mask(mask_label=['Bare sparse vegetation',
                                                  'Urban',
                                                  'Moss and lichen'])
    input_file_dict = {'intensity': filt_im_str,
                       'landcover': landcover_path_str,
                       'reference_water': interp_wbd_str,
                       'water_mask': water_map_tif_str}

    if band_set.ndim == 3:
        num_band, cols, rows = np.shape(band_set)
    else:
        cols, rows = np.shape(band_set)
        num_band = 1
        band_set = np.reshape(band_set, [num_band, cols, rows])

    # 1) Identify low-backscattering areas
    low_backscatter_cand = np.ones([cols, rows], dtype=bool)
    for pol_ind, pol in enumerate(pol_list):
        if pol in ['VV', 'HH']:
            pol_threshold = landcover_cfg.co_pol_threshold
        elif pol in ['VH', 'HV']:
            pol_threshold = landcover_cfg.cross_pol_threshold
        else:
            continue  # Skip unknown polarizations
        low_backscatter = 10 * np.log10(band_set[pol_ind, :, :]) < pol_threshold
        low_backscatter_cand = np.logical_and(low_backscatter,
                                              low_backscatter_cand)

    # 2) Create intial mask binary
    # mask_excluded indicates the areas satisfying all conditions which are
    # 1: 5 % of water occurrence over 37 year (Pekel)
    # 2: specified landcovers (bare ground, sparse vegetation, urban, moss...)
    # 3: low backscattering area
    # mask = intersect of landcover mask + no water + dark land
    mask_excluded = (mask_excluded_landcover) & \
                    (interp_wbd / ref_water_max < 0.05) & \
                    (low_backscatter_cand)

    # 3) The water candidates extracted in the previous step (region growing)
    # can have dark land and water in one singe polygon where dark land and water
    # are spatially connected. Here 'split_extended_water' checks
    # if the backscatter splits into smaller pieces that are distinguishable.
    # So 'split_extended_water' checks for bimodality for each polygon.
    # If so, the code calculates a threshold and checks the bimodality
    # for each split area.
    input_map = np.where(water_map == 1, 1, 0)
    split_mask_water_raster = split_extended_water_parallel(
        input_map,
        co_pol_ind,
        outputdir,
        water_meta,
        input_file_dict,
        number_workers)

    # 4) re-define false water candidate estimated from 'split_extended_water_parallel'
    # by considering the spatial coverage with the ancillary files
    false_water_cand = (split_mask_water_raster == 0) & (water_map==1)
    mask_water_image = compute_spatial_coverage_from_ancillary_parallel(
                    false_water_cand,
                    ref_water=interp_wbd/ref_water_max,
                    mask_landcover=mask_excluded_landcover,
                    outputdir=outputdir,
                    meta_info=water_meta,
                    number_workers=number_workers)

    dark_land = (mask_water_image == 1) | mask_excluded

    water_map[dark_land] = 0

    water_tif_str = os.path.join(outputdir,
                                 f"refine_landcover_binary_{pol_str}.tif")
    dswx_sar_util.save_dswx_product(
        water_map,
        water_tif_str,
        geotransform=water_meta['geotransform'],
        projection=water_meta['projection'],
        description='Water classification (WTR)',
        scratch_dir=outputdir,
        dark_land=dark_land)

    if processing_cfg.debug_mode:

        darkland_tif_str = os.path.join(outputdir,
                                        f"landcover_dark_land_binary_{pol_str}.tif")
        dswx_sar_util.save_dswx_product(
            mask_excluded,
            darkland_tif_str,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Water classification (WTR)',
            scratch_dir=outputdir
            )

        test_tif_str = os.path.join(outputdir,
                                    f"test_land_binary_{pol_str}.tif")
        dswx_sar_util.save_dswx_product(
            np.array(false_water_cand, dtype=np.uint8),
            test_tif_str,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Water classification (WTR)',
            scratch_dir=outputdir
            )

        darkland_str = os.path.join(outputdir,
                                    f"split_dark_land_candidate_{pol_str}.tif")
        dswx_sar_util.save_dswx_product(
            split_mask_water_raster,
            darkland_str,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Darkland candidates',
            scratch_dir=outputdir,
            dark_land=dark_land)

        darkland_str = os.path.join(outputdir,
                                    f"dark_land_50_candidate_{pol_str}.tif")
        dswx_sar_util.save_dswx_product(
            mask_water_image,
            darkland_str,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Darkland candidates',
            scratch_dir=outputdir,
            dark_land=dark_land)

        test_tif_str = os.path.join(outputdir,
                                    f"test_land_binary_{pol_str}.tif")
        dswx_sar_util.save_dswx_product(
                    np.array(false_water_cand,
                             dtype=np.uint8),
                    test_tif_str,
                    geotransform=water_meta['geotransform'],
                    projection=water_meta['projection'],
                    description='Water classification (WTR)',
                    scratch_dir=outputdir
                    )
    t_all_elapsed = time.time() - t_all
    logger.info(f"successfully ran landcover masking in {t_all_elapsed:.3f} seconds")


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
