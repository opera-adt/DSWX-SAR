import copy
import logging
import mimetypes
import os
import time

import cv2
import numpy as np
import rasterio
from osgeo import gdal
from joblib import Parallel, delayed
from rasterio.windows import Window
from scipy import ndimage
from skimage.filters import threshold_multiotsu
from typing import List, Tuple

from dswx_sar import (dswx_sar_util,
                      generate_log,
                      refine_with_bimodality,
                      region_growing)
from dswx_sar.dswx_runconfig import (DSWX_S1_POL_DICT,
                                     _get_parser,
                                     RunConfig)


logger = logging.getLogger('dswx_s1')


def get_label_landcover_esa_10():
    '''Get integer number information what they represent
    ESA 10 m
    https://viewer.esa-worldcover.org/worldcover/
    '''
    label = dict()

    label['Tree Cover'] = 10
    label['Shrubs'] = 20
    label['Grassland'] = 30
    label['Crop'] = 40
    label['Urban'] = 50
    label['Bare sparse vegetation'] = 60
    label['Snow and Ice'] = 70
    label['Permanent water bodies'] = 80
    label['Herbaceous wetland'] = 90
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

    def get_mask(self, mask_label, block_param=None):
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
        if block_param is None:
            landcover = self.open_landcover()
        else:
            landcover = dswx_sar_util.get_raster_block(
                self.landcover_file_path,
                block_param)

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

    Parameters
    ----------
    binary : np.ndarray
        Binary image containing connected components.
    buffer : int
        Buffer size to expand the bounding boxes.

    Returns
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

        sub_x_start = int(max(0, x - buffer_all))
        sub_y_start = int(max(0, y - buffer_all))
        sub_x_end = int(min(cols, x + w + buffer_all))
        sub_y_end = int(min(rows, y + h + buffer_all))

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

    Parameters
    ----------
    args : tuple
    Contains the following elements:
        * i (int): Index of the water component.
        * size (int): Size of the water component.
        * minimum_pixel (int): Minimum pixel threshold for processing.
        * bounds (list): List containing bounding box coordinates.
        * int_linear_block (str):
            Numpy array of the linear intensity raster file.
        * water_label_block (str):
            Numpy array of the water label raster file.
        * water_mask_block (str): Numpy array of the water mask raster file.
        * pol_ind (int): Polarization index.

    Returns
    -------
    i : int
        Index of the water component.
    change_flag : bool
        Flag indicating whether any change was made to the water mask.
    water_mask : numpy.ndarray
        Processed water mask.
    """
    i, size, minimum_pixel, bounds, int_linear_block, water_label_block, \
        water_mask_block, pol_ind = args

    if int_linear_block.ndim == 2:
        int_linear_block = np.expand_dims(int_linear_block,
                                          axis=0)

    int_linear = int_linear_block[pol_ind,
                                  bounds[2]:bounds[3],
                                  bounds[0]:bounds[1]]
    water_label = water_label_block[bounds[2]:bounds[3],
                                    bounds[0]:bounds[1]]
    water_mask = water_mask_block[bounds[2]:bounds[3],
                                  bounds[0]:bounds[1]].copy()

    change_flag = False
    # label for water object computed from cv2 start from 1.
    target_water = water_label == i + 1
    intensity_array = int_linear[target_water]

    if (size > minimum_pixel) and (intensity_array.size > 0):

        invalid_mask = (np.isnan(intensity_array) | (intensity_array == 0))

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
            int_db_sub = 10 * np.log10(
                intensity_array[np.invert(invalid_mask)])
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
                                 (int_db >= threshold_local_otsu[t_ind - 1]) &\
                                 (target_water)

                n_dark_water_pixels = np.count_nonzero(dark_water)
                # compute additional iteration number using size.
                # it does not need to be precise.
                add_iter = int((np.sqrt(2) - 1.2) *
                               np.sqrt(n_dark_water_pixels))
                dark_mask_buffer = ndimage.binary_dilation(
                    dark_water,
                    iterations=add_iter,
                    mask=out_boundary_sub)
                dark_water_linear = int_linear[dark_mask_buffer]
                hist_min = np.nanpercentile(
                    10 * np.log10(dark_water_linear), 1)
                hist_max = np.nanpercentile(
                    10 * np.log10(dark_water_linear), 99)

                # Check if the candidates of 'dark water' has distinct
                # backscattering compared to the adjacent pixels
                # using bimodality
                metric_obj_local = refine_with_bimodality.BimodalityMetrics(
                    dark_water_linear,
                    hist_min=hist_min,
                    hist_max=hist_max)

                if not metric_obj_local.compute_metric(
                        ashman_flag=True,
                        bm_flag=True,
                        surface_ratio_flag=True,
                        bc_flag=False):
                    water_mask.setflags(write=1)
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
                add_iter = int((np.sqrt(2) - 1.2) *
                               np.sqrt(n_bright_water_pixels))
                bright_mask_buffer = ndimage.binary_dilation(
                    bright_water_pixels,
                    iterations=add_iter,
                    mask=out_boundary_sub)

                bright_water_linear = int_linear[bright_mask_buffer].copy()
                bright_water_linear[bright_water_linear == 0] = np.nan

                hist_min = np.nanpercentile(
                    10 * np.log10(bright_water_linear), 2)
                hist_max = np.nanpercentile(
                    10 * np.log10(bright_water_linear), 98)
                metric_obj_local = refine_with_bimodality.BimodalityMetrics(
                    bright_water_linear,
                    hist_min=hist_min,
                    hist_max=hist_max)

                if not metric_obj_local.compute_metric(
                        ashman_flag=True,
                        bm_flag=True,
                        surface_ratio_flag=True,
                        bc_flag=False):

                    water_mask.setflags(write=1)
                    water_mask[bright_water_pixels] = 0
                    change_flag = True

    return i, change_flag, water_mask


def split_extended_water_parallel(
        water_mask_path: str,
        output_path: str,
        pol_ind: int,
        outputdir: str,
        input_dict: dict,
        number_workers: int,
        input_lines_per_block: int):
    """
    Splits extended water areas into smaller subsets based on bounding boxes
    with buffer, processing in parallel for efficiency.

    Parameters
    ----------
    water_mask_path : str
        Path to the binary water mask GeoTIFF file.
    output_path : str
        Path to save the updated water mask GeoTIFF file.
    pol_ind : int
        Index of the polarization used in analysis.
    outputdir : str
        Directory to save intermediate and resulting files.
    input_dict : dict
        Dictionary containing input data including intensity and water mask.
    number_workers : int
        Number of parallel workers for processing.
    input_lines_per_block: int
        Lines per block processing

    Returns
    -------
    None
        Saves the updated water mask with extended areas split into smaller
        subsets in the specified output path.
    """
    meta_info = dswx_sar_util.get_meta_from_tif(water_mask_path)
    # exceed the size of the block. To prevent discontinuity and incorrect
    # decisions, the block size increases within the for loop.
    # Initially, the procedure is run for the entire area using a small block
    # size (input_lines_per_block), and then the size is increased by a
    # multiple of 3 and 20.
    lines_per_block_set = [input_lines_per_block,
                           input_lines_per_block * 3,
                           input_lines_per_block * 20,
                           meta_info['length']]
    pad_shape = (0, 0)

    temp_prefix = 'split_extended_water_parallel_temp'
    temp_path_set = []
    minimum_pixel = 5000

    for block_iter, lines_per_block in enumerate(lines_per_block_set):
        block_params = dswx_sar_util.block_param_generator(
            lines_per_block,
            [meta_info['length'], meta_info['width']],
            pad_shape)
        if len(lines_per_block_set) > 1:
            removed_false_water_path = os.path.join(
                outputdir,
                f'{temp_prefix}_{block_iter}.tif')
        else:
            removed_false_water_path = output_path

        temp_path_set.append(removed_false_water_path)

        for block_ind, block_param in enumerate(block_params):
            logger.info(
                f'split_extended_water_parallel block #{block_ind} '
                f'from {block_param.read_start_line} to '
                f'{block_param.read_start_line + block_param.read_length}')
            water_mask = dswx_sar_util.get_raster_block(
                water_mask_path, block_param)
            intensity_block = dswx_sar_util.get_raster_block(
                input_dict['intensity'], block_param)

            # Extract bounding boxes with buffer
            coord_list, sizes, label_image = extract_bbox_with_buffer(
                binary=water_mask, buffer=10)

            filtered_sizes = []
            filtered_coord_list = []
            filtered_index = []
            check_output = np.ones(len(sizes), dtype='byte')
            old_val = np.arange(1, len(sizes) + 1) - .1
            index_array_to_image = np.array(
                np.searchsorted(old_val, label_image),
                dtype='uint32')

            for ind, (coords, size) in enumerate(zip(coord_list, sizes)):
                (bbox_x_start,
                 bbox_x_end,
                 bbox_y_start,
                 bbox_y_end) = coords

                # Check if the component touches the boundary
                if bbox_y_start != 0 and \
                   bbox_x_end != block_param.block_length and \
                   size >= minimum_pixel:

                    filtered_index.append(ind)
                    filtered_coord_list.append([bbox_x_start, bbox_x_end,
                                                bbox_y_start, bbox_y_end])
                    filtered_sizes.append(size)

            # check if the individual water body candidates has bimodality
            # bright water vs dark water
            # water vs land
            # Prepare arguments for parallel processing
            args_list = [(filtered_index[i], filtered_sizes[i],
                          minimum_pixel, filtered_coord_list[i],
                          intensity_block,
                          label_image,
                          water_mask,
                          pol_ind) for i in range(len(filtered_sizes))]

            # Check if the objects have heterogeneous characteristics
            # If so, split the objects using multi-otsu thresholds
            # and check bimodality.
            results = Parallel(n_jobs=number_workers)(
                delayed(check_water_land_mixture)(args)
                for args in args_list)

            # If water need to be refined (change_flat=True),
            # then update the water mask.
            for i, change_flag, water_mask_subset in results:
                if change_flag:
                    water_mask[coord_list[i][2]: coord_list[i][3],
                               coord_list[i][0]: coord_list[i][1]] = \
                                water_mask_subset
                check_output[i] = 0

            del results
            dswx_sar_util.write_raster_block(
                removed_false_water_path,
                water_mask,
                block_param,
                geotransform=meta_info['geotransform'],
                projection=meta_info['projection'],
                datatype='byte',
                cog_flag=True,
                scratch_dir=outputdir)

            if block_iter < len(lines_per_block_set) - 1:
                check_output = np.insert(check_output, 0, 0, axis=0)
                check_image = np.array(
                    check_output[index_array_to_image], dtype='byte')

                # 'check_remove_false_water' has 1 value for unprocessed
                # components when binary area touches the boundaries.
                # processed components have 0 values.
                check_remove_false_water_path = os.path.join(
                    outputdir, f'check_water_land_mixture_{block_iter}.tif')
                dswx_sar_util.write_raster_block(
                    check_remove_false_water_path,
                    check_image,
                    block_param,
                    geotransform=meta_info['geotransform'],
                    projection=meta_info['projection'],
                    datatype='byte',
                    cog_flag=True,
                    scratch_dir=outputdir)
                del check_image

            if block_param.block_length + block_param.read_start_line >= \
               meta_info['length']:
                water_mask_path = check_remove_false_water_path

    if len(temp_path_set) >= 2:
        # Merge multiple results processed with different block sizes
        merged_removed_false_water_path = output_path

        dswx_sar_util.merge_binary_layers(
            layer_list=temp_path_set,
            value_list=[1] * len(lines_per_block_set),
            merged_layer_path=merged_removed_false_water_path,
            lines_per_block=input_lines_per_block,
            mode='or',
            cog_flag=True,
            scratch_dir=outputdir)
    else:
        merged_removed_false_water_path = temp_path_set[0]

    logger.info(f'split_extended_water_parallel output: '
                f'{merged_removed_false_water_path}')


def compute_spatial_coverage_from_ancillary_parallel(
        flase_water_binary_path: str,
        reference_water_path: str,
        mask_landcover_path: str,
        output_file_path: str,
        outputdir: str,
        water_max_value: float,
        spatial_coverage_threshold: float = 0.5,
        number_workers: int = -1,
        lines_per_block: int = 400):
    """
    Computes spatial coverage of water areas using ancillary data,
    processing in parallel for efficiency.

    Parameters
    ----------
    flase_water_binary_path : str
        Path to the binary water mask GeoTIFF file.
    reference_water_path : str
        Path to the reference water GeoTIFF file.
    mask_landcover_path : str
        Path to the landcover mask GeoTIFF file.
    output_file_path : str
        Path to save the output water mask GeoTIFF file.
    output_dir : str
        Directory to save intermediate and resulting files.
    water_max_value : float
        Maximum valid value in the water data for normalization.
    spatial_coverage_threshold : float, optional
        Threshold for spatial coverage of land.
    number_workers : int, optional
        Number of parallel workers for processing.
    lines_per_block : int, optional
        Number of lines per block for processing.

    Returns
    -------
    None
        Saves the computed water mask indicating water areas in the
        specified output file path.
    """
    water_mask = dswx_sar_util.read_geotiff(flase_water_binary_path)
    meta_info = dswx_sar_util.get_meta_from_tif(flase_water_binary_path)

    # Extract bounding boxes with buffer
    coord_list, sizes, label_image = extract_bbox_with_buffer(
        binary=water_mask, buffer=10)

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

    # From reference water map (0: non-water, 1: permanent water) and
    # landcover map (ex. bare/sparse vegetation), extract the areas
    # which are likely mis-classified as water.
    data_shape = [meta_info['length'], meta_info['width']]

    pad_shape = (0, 0)
    block_params = dswx_sar_util.block_param_generator(
        lines_per_block,
        data_shape,
        pad_shape)

    ref_land_tif_str = os.path.join(
        outputdir, "landcover_darklandcandidate_spatial_coverage.tif")
    for block_param in block_params:
        mask_excluded = dswx_sar_util.get_raster_block(
            mask_landcover_path, block_param)
        water_block = dswx_sar_util.get_raster_block(
            reference_water_path, block_param)

        dry_darkland = np.logical_and(
            mask_excluded, water_block/water_max_value < 0.05)

        dswx_sar_util.write_raster_block(
            out_raster=ref_land_tif_str,
            data=dry_darkland.astype(np.uint8),
            block_param=block_param,
            geotransform=meta_info['geotransform'],
            projection=meta_info['projection'],
            datatype='byte')

    # Arrange arguments to be used for parallel processing
    args_list = [(i, sizes[i], spatial_coverage_threshold, coord_list[i],
                  water_label_str, ref_land_tif_str)
                 for i in range(len(sizes))]

    # Output consists of index and 2D image consisting of True/Flase.
    # True represents the land and False represents not-land.
    results = Parallel(n_jobs=number_workers)(
        delayed(compute_spatial_coverage)(args)
        for args in args_list)

    test_output = np.zeros(len(sizes))
    for i, test_output_i in results:
        test_output[i] = test_output_i

    # Convert 1D array result to 2D image.
    old_val = np.arange(1, nb_components_water + 1) - 0.1
    kin = np.searchsorted(old_val, label_image)
    test_output = np.insert(test_output, 0, 0, axis=0)
    mask_water_image = test_output[kin]

    dswx_sar_util.save_dswx_product(
        mask_water_image,
        output_file_path,
        geotransform=meta_info['geotransform'],
        projection=meta_info['projection'],
        scratch_dir=outputdir
        )


def compute_spatial_coverage(args):
    """Compute the spatial coverage of non-water areas
    based on given arguments.

    Parameters
    ----------
    args (tuple): Tuple containing the following arguments:
        i (int): Index of the current iteration.
        size (int): Size of the current component.
        threshold (float): Threshold value for comparison.
        bounds (list): List containing the bounds of the subset.
        water_label_str (str): Path to the water label file.
        ref_land_tif_str (str): Path to the reference land file.

    Returns
    -------
        tuple: Tuple containing the index (i) and the test output
               (test_output_i).
    """
    i, _, threshold, bounds, water_label_str, ref_land_tif_str = args

    x_off = bounds[0]
    y_off = bounds[2]
    width = bounds[1] - bounds[0]
    height = bounds[3] - bounds[2]
    window = Window(x_off, y_off, width, height)

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


def remove_small_components(image, min_size=3):
    """
    Remove small connected components from a binary image.

    Parameters:
    -----------
    image : ndarray
        2D binary image where the components to be removed are True or 1.
    min_size : int, optional
        Minimum size of components to keep. Components with fewer pixels
        than this will be removed. Default is 3.

    Returns:
    --------
    cleaned_image : ndarray
        2D binary image with small components removed.
    """
    nb_components_water, output_water, stats_water, _ = \
        cv2.connectedComponentsWithStats(image.astype(np.uint8),
                                         connectivity=8)
    sizes = stats_water[1:, -1]
    nb_components_water = nb_components_water - 1

    if nb_components_water > 65535:
        output_water_type = 'uint32'
    else:
        output_water_type = 'uint16'

    old_val = np.arange(1, nb_components_water + 1) - .1
    index_array_to_image = np.searchsorted(
        old_val, output_water).astype(dtype=output_water_type)
    size_output_add = np.insert(sizes, 0, 0, axis=0)
    size_image = size_output_add[index_array_to_image]

    # Use the mask to remove small components
    cleaned_image = size_image > min_size

    return cleaned_image


def extend_land_cover(landcover_path,
                      reference_landcover_binary,
                      target_landcover,
                      water_landcover,
                      exclude_area_rg,
                      minimum_pixel,
                      water_buffer,
                      metainfo,
                      scratch_dir):
    """
    Extends the specified type of land cover within a geographical dataset.

    Parameters
    ----------
    landcover_path (str):
        Path to the land cover data file.
    reference_landcover_binary (numpy.ndarray):
        Binary array representing the initial land cover.
    target_landcover (str):
        Label of the land cover type to be extended.
    water_landcover (str):
        Label of the water land cover type.
    exclude_area_rg (numpy.ndarray):
        Array representing areas to be excluded from processing.
    minimum_pixel (int):
        Minimum number of the seed pixels for region growing
    water_buffer (int):
        Buffer size to apply the water. Then, the area near the water
        bodies are excluded from extended landcover masks.
    metainfo (dict):
        Metadata including geotransform and projection information.
    scratch_dir (str):
        Directory for saving intermediate and output files.

    Returns
    -------
    numpy.ndarray:
        Updated binary land cover array with the extended target land cover.
    """
    logger.info('Extending land cover....')
    mask_obj = FillMaskLandCover(landcover_path)

    mask_excluded_candidate = mask_obj.get_mask(
        mask_label=target_landcover)

    new_landcover = np.zeros(
        reference_landcover_binary.shape,
        dtype='float32')
    if minimum_pixel > 0:
        logger.info(f'Removing small components less than {minimum_pixel}')

        reference_landcover_binary = remove_small_components(
            reference_landcover_binary,
            minimum_pixel)

    new_landcover[reference_landcover_binary] = 1
    new_landcover[mask_excluded_candidate] = 0.75

    if water_buffer > 0:
        logger.info('Excluding buffered area along th water.')
        water_landcover_binary = mask_obj.get_mask(
            mask_label=water_landcover)
        water_landcover_binary = ndimage.binary_dilation(
            water_landcover_binary,
            iterations=water_buffer)
        new_landcover[water_landcover_binary] = 0

    excluded_rg_path = os.path.join(
        scratch_dir, "landcover_excluded_rg.tif")
    dswx_sar_util.save_dswx_product(
        exclude_area_rg,
        excluded_rg_path,
        geotransform=metainfo['geotransform'],
        projection=metainfo['projection'],
        scratch_dir=scratch_dir
        )

    darkland_tif_str = os.path.join(
        scratch_dir, "landcover_target.tif")
    dswx_sar_util.save_dswx_product(
        reference_landcover_binary,
        darkland_tif_str,
        geotransform=metainfo['geotransform'],
        projection=metainfo['projection'],
        scratch_dir=scratch_dir
        )

    darkland_cand_tif_str = os.path.join(
        scratch_dir, "landcover_target_candidate.tif")
    dswx_sar_util.save_raster_gdal(
        np.array(new_landcover, dtype='float32'),
        darkland_cand_tif_str,
        geotransform=metainfo['geotransform'],
        projection=metainfo['projection'],
        scratch_dir=scratch_dir,
        datatype='float32')

    temp_rg_tif_path = os.path.join(
        scratch_dir, 'landcover_temp_transition.tif')

    region_growing.run_parallel_region_growing(
        darkland_cand_tif_str,
        temp_rg_tif_path,
        exclude_area_path=excluded_rg_path,
        lines_per_block=1000,
        initial_threshold=0.9,
        relaxed_threshold=0.7,
        maxiter=0)

    fuzzy_map = dswx_sar_util.read_geotiff(darkland_cand_tif_str)
    temp_rg = dswx_sar_util.read_geotiff(temp_rg_tif_path)

    # replace the fuzzy values with 1 for the pixels
    # where the region-growing already applied
    fuzzy_map[temp_rg == 1] = 1

    # Run region-growing again for entire image
    region_grow_map = region_growing.region_growing(
        fuzzy_map,
        initial_threshold=0.9,
        relaxed_threshold=0.7,
        maxiter=0)
    reference_landcover_binary[region_grow_map] = 1

    return reference_landcover_binary


def extract_boundary(binary_data):
    """Extracts the boundary of a binary image."""
    # Dilate the binary data and then subtract the original data.
    erosion = ndimage.binary_erosion(binary_data)
    return np.bitwise_xor(binary_data, erosion)


def extract_values_using_boundary(boundary_data, float_data):
    """Extracts values from float_data where boundary_data is 1."""
    data_array = float_data[boundary_data == 1]
    float_data[boundary_data == 0] = 0

    return data_array, float_data


def hand_filter_along_boundary(
        target_area_path,
        height_std_threshold,
        hand_path,
        output_path,
        debug_mode,
        metainfo,
        scratch_dir):
    """
    Filters geographic data along boundaries based on HAND model and
    standard deviation thresholds.

    Parameters
    ----------
    target_area_path : str
        Path containing array representing the target area for filtering.
    height_std_threshold : float
        Standard deviation threshold for HAND values.
    hand_path : str
        Path to the HAND data file.
    output_path : str
        Path to save output binary image
    debug_mode : bool
        Flag to activate debug mode for additional outputs.
    metainfo : dict
        Metadata including geotransform and projection information.
    scratch_dir : str
        Directory for saving debug outputs.

    Returns
    -------
    numpy.ndarray
        Binary array representing the filtered HAND data.
    """
    target_area = dswx_sar_util.read_geotiff(target_area_path)
    hand_obj = gdal.Open(hand_path)

    coord_lists, sizes, output_water = \
        extract_bbox_with_buffer(target_area, 10)
    nb_components_water = len(sizes)

    hand_filtered_binary = np.zeros(target_area.shape, dtype='byte')
    hand_std_image = np.zeros(target_area.shape, dtype='float32')

    if debug_mode:
        height_array = np.zeros(nb_components_water)

    for ind, coord_list in enumerate(coord_lists):
        sub_x_start, sub_x_end, sub_y_start, sub_y_end = coord_list
        sub_win_x = int(sub_x_end - sub_x_start)
        sub_win_y = int(sub_y_end - sub_y_start)
        sub_water_label = output_water[sub_y_start:sub_y_end,
                                       sub_x_start:sub_x_end]
        sub_hand = hand_obj.ReadAsArray(sub_x_start,
                                        sub_y_start,
                                        sub_win_x,
                                        sub_win_y)
        initial_area = sub_water_label == ind + 1

        water_boundary = extract_boundary(
            np.array(sub_water_label == ind + 1, dtype='byte'))
        hand_line_data, hand_image_data = \
            extract_values_using_boundary(water_boundary, sub_hand)
        hand_std = np.nanstd(hand_line_data)

        if debug_mode:
            height_array[ind] = np.nanstd(hand_line_data)

        hand_std_image[sub_y_start:sub_y_end,
                       sub_x_start:sub_x_end] += hand_image_data

        if hand_std > height_std_threshold:
            final_binary = np.zeros(sub_hand.shape, dtype='byte')
            area_median = np.median(hand_line_data)
            hand_threshold_erosion = area_median + hand_std * 1
            hand_image_mask = hand_image_data > hand_threshold_erosion

            bad_hand_count = 1
            iter_count = 0
            while bad_hand_count > 0:
                new_binary = ndimage.binary_erosion(
                    initial_area,
                    mask=hand_image_mask)
                new_bound = extract_boundary(new_binary)
                hand_line_data, hand_image_data = \
                    extract_values_using_boundary(new_bound, sub_hand)
                hand_image_mask = hand_image_data > hand_threshold_erosion
                bad_hand_count = np.sum(hand_image_mask)
                initial_area = new_binary
                iter_count += 1
            final_binary[new_binary == 1] = 1
            hand_filtered_binary[sub_y_start:sub_y_end,
                                 sub_x_start:sub_x_end] += final_binary
        else:
            hand_filtered_binary[sub_y_start:sub_y_end,
                                 sub_x_start:sub_x_end] += initial_area

    output_water = np.array(output_water)
    old_val = np.arange(1, nb_components_water + 1) - .1
    index_array_to_image = np.searchsorted(old_val, output_water)

    if debug_mode:
        height_array = np.insert(height_array, 0, 0, axis=0)
        height_std_raster = np.array(height_array[index_array_to_image],
                                     dtype='float32')

    target_area[hand_filtered_binary == 0] = 0

    dswx_sar_util.save_dswx_product(
        target_area,
        output_path,
        geotransform=metainfo['geotransform'],
        projection=metainfo['projection'],
        scratch_dir=scratch_dir
        )

    if debug_mode:
        hand_std_path = os.path.join(
            scratch_dir, "landcover_hand_std.tif")
        dswx_sar_util.save_raster_gdal(
            np.array(height_std_raster, dtype='float32'),
            hand_std_path,
            geotransform=metainfo['geotransform'],
            projection=metainfo['projection'],
            scratch_dir=scratch_dir,
            datatype='float32')
        hand_std_path = os.path.join(
            scratch_dir, "landcover_hand_std_image.tif")
        dswx_sar_util.save_raster_gdal(
            np.array(hand_std_image, dtype='float32'),
            hand_std_path,
            geotransform=metainfo['geotransform'],
            projection=metainfo['projection'],
            scratch_dir=scratch_dir,
            datatype='float32')
        hand_binary_path = os.path.join(
            scratch_dir, "landcover_hand_bindary.tif")
        dswx_sar_util.save_dswx_product(
            hand_filtered_binary,
            hand_binary_path,
            geotransform=metainfo['geotransform'],
            projection=metainfo['projection'],
            scratch_dir=scratch_dir
            )
    del hand_obj


def get_darkland_from_intensity_ancillary(
        intensity_path,
        landcover_path,
        reference_water_path,
        darkland_candidate_path,
        lines_per_block,
        pol_list,
        co_pol_threshold,
        cross_pol_threshold,
        ref_water_max,
        dry_water_area_threshold):
    """
    Identifies low backscatter areas from SAR intensity data, considering
    landcover and reference water information.

    Parameters
    ----------
    intensity_path : str
        Path to the SAR intensity data file.
    landcover_path : str
        Path to the landcover data file.
    reference_water_path : str
        Path to the reference water data file.
    darkland_candidate_path : str
        Path to save the output darkland candidate data.
    lines_per_block : int
        Number of lines per block for processing.
    pol_list : list of str
        List of polarizations (e.g., ['VV', 'HH', 'VH', 'HV']).
    co_pol_threshold : float
        Threshold for co-polarized channels (e.g., 'VV', 'HH').
    cross_pol_threshold : float
        Threshold for cross-polarized channels (e.g., 'VH', 'HV').
    ref_water_max : float
        Maximum valid value in the reference water data for normalization.
    dry_water_area_threshold : float
        Threshold for identifying dry water areas.

    Returns
    -------
    None
        The function saves the identified low backscatter areas as a binary
        raster file at `darkland_candidate_path`.
    """
    band_meta = dswx_sar_util.get_meta_from_tif(intensity_path)
    data_shape = [band_meta['length'], band_meta['width']]

    pad_shape = (0, 0)
    block_params = dswx_sar_util.block_param_generator(
        lines_per_block, data_shape, pad_shape)

    for block_param in block_params:
        intensity_block = dswx_sar_util.get_raster_block(
            intensity_path, block_param)
        mask_excluded_landcover = dswx_sar_util.get_raster_block(
            landcover_path, block_param)
        water_block = dswx_sar_util.get_raster_block(
            reference_water_path, block_param)

        # Reshaping the intensity block if necessary
        if intensity_block.ndim < 3:
            intensity_block = np.expand_dims(intensity_block, axis=0)

        low_backscatter_cand = np.ones(intensity_block.shape[1:], dtype=bool)

        for pol_ind, pol in enumerate(pol_list):
            pol_threshold = co_pol_threshold \
                if pol in ['VV', 'HH'] else cross_pol_threshold
            low_backscatter = \
                10 * np.log10(intensity_block[pol_ind, :, :]) < pol_threshold
            low_backscatter_cand = np.logical_and(low_backscatter,
                                                  low_backscatter_cand)

        mask_excluded = np.logical_and(
            mask_excluded_landcover == 1,
            np.logical_and(
                water_block / ref_water_max < dry_water_area_threshold,
                low_backscatter_cand))

        dswx_sar_util.write_raster_block(
            out_raster=darkland_candidate_path,
            data=mask_excluded,
            block_param=block_param,
            geotransform=band_meta['geotransform'],
            projection=band_meta['projection'],
            datatype='byte')


def run(cfg):
    '''
    Remove the false water which have low backscattering based on the
    occurrence map and landcover map.
    '''
    logger.info('Starting DSWx-S1 masking with ancillary data')

    t_all = time.time()
    outputdir = cfg.groups.product_path_group.scratch_path
    processing_cfg = cfg.groups.processing
    pol_list = copy.deepcopy(processing_cfg.polarizations)
    pol_options = processing_cfg.polarimetric_option

    if pol_options is not None:
        pol_list += pol_options

    pol_str = '_'.join(pol_list)
    co_pol_ind = 0

    water_cfg = processing_cfg.reference_water
    ref_water_max = water_cfg.max_value

    # options for masking with ancillary data
    masking_ancillary_cfg = processing_cfg.masking_ancillary
    number_workers = masking_ancillary_cfg.number_cpu

    dry_water_area_threshold = masking_ancillary_cfg.water_threshold
    extended_landcover_flag = masking_ancillary_cfg.extended_darkland
    extend_minimum = masking_ancillary_cfg.extended_darkland_minimum_pixel
    water_buffer = masking_ancillary_cfg.extended_darkland_water_buffer
    hand_variation_mask = masking_ancillary_cfg.hand_variation_mask
    hand_variation_threshold = masking_ancillary_cfg.hand_variation_threshold
    landcover_masking_list = masking_ancillary_cfg.land_cover_darkland_list
    landcover_masking_extension_list = \
        masking_ancillary_cfg.land_cover_darkland_extension_list
    landcover_water_label = masking_ancillary_cfg.land_cover_water_label
    lines_per_block = masking_ancillary_cfg.line_per_block

    # Binary water map extracted from region growing
    water_map_tif_str = os.path.join(
        outputdir, f'region_growing_output_binary_{pol_str}.tif')
    water_map = dswx_sar_util.read_geotiff(water_map_tif_str)
    water_meta = dswx_sar_util.get_meta_from_tif(water_map_tif_str)

    # Filtered SAR intensity image
    filt_im_str = os.path.join(outputdir, f"filtered_image_{pol_str}.tif")

    # Reference water map
    interp_wbd_str = os.path.join(outputdir, 'interpolated_wbd.tif')
    interp_wbd = dswx_sar_util.read_geotiff(interp_wbd_str)
    interp_wbd = np.array(interp_wbd, dtype='float32')

    # HAND
    hand_path_str = os.path.join(outputdir, 'interpolated_hand.tif')

    # Worldcover map
    landcover_path = os.path.join(outputdir, 'interpolated_landcover.tif')

    # Identify dark land candidate areas from landcover
    mask_obj = FillMaskLandCover(landcover_path)
    mask_excluded_landcover = mask_obj.get_mask(
        mask_label=landcover_masking_list)
    del mask_obj

    # If `extended_landcover_flag`` is enabled, additional landcovers
    # spatially connected with `mask_excluded_landcover` are added.
    if extended_landcover_flag:
        logger.info('Extending landcover enabled.')
        rg_excluded_area = (interp_wbd > dry_water_area_threshold)

        mask_excluded_landcover = extend_land_cover(
            landcover_path=landcover_path,
            reference_landcover_binary=mask_excluded_landcover,
            target_landcover=landcover_masking_extension_list,
            water_landcover=landcover_water_label,
            exclude_area_rg=rg_excluded_area,
            water_buffer=water_buffer,
            minimum_pixel=extend_minimum,
            metainfo=water_meta,
            scratch_dir=outputdir)
        logger.info('Landcover extension completed.')

    mask_excluded_landcover_path = os.path.join(
        outputdir, 'mask_excluded_landcover_ancillary.tif')
    dswx_sar_util.save_dswx_product(
        mask_excluded_landcover,
        mask_excluded_landcover_path,
        geotransform=water_meta['geotransform'],
        projection=water_meta['projection'],
        scratch_dir=outputdir)

    # 2) Create intial mask binary
    # mask_excluded indicates the areas satisfying all conditions which are
    # 1: `dry_water_area_threshold` of water occurrence over 37 year (Pekel)
    # 2: specified landcovers (bare ground, sparse vegetation, urban, moss...)
    # 3: low backscattering area
    # mask = intersect of landcover mask + no water + dark land

    darkland_cand_path = \
        os.path.join(outputdir,
                     'darkland_candidate_from_landcover_water_backscatter.tif')

    get_darkland_from_intensity_ancillary(
        intensity_path=filt_im_str,
        landcover_path=mask_excluded_landcover_path,
        reference_water_path=interp_wbd_str,
        darkland_candidate_path=darkland_cand_path,
        lines_per_block=lines_per_block,
        pol_list=pol_list,
        co_pol_threshold=masking_ancillary_cfg.co_pol_threshold,
        cross_pol_threshold=masking_ancillary_cfg.cross_pol_threshold,
        ref_water_max=ref_water_max,
        dry_water_area_threshold=dry_water_area_threshold)

    # 3) The water candidates extracted in the previous step (region growing)
    # can have dark land and water in one singe polygon where dark land and
    # water are spatially connected. Here 'split_extended_water' checks
    # if the backscatter splits into smaller pieces that are distinguishable.
    # So 'split_extended_water' checks for bimodality for each polygon.
    # If so, the code calculates a threshold and checks the bimodality
    # for each split area.
    input_map = np.where(water_map == 1, 1, 0)
    del water_map

    dswx_sar_util.save_dswx_product(
        input_map,
        os.path.join(outputdir, 'input_image_test.tif'),
        geotransform=water_meta['geotransform'],
        projection=water_meta['projection'],
        scratch_dir=outputdir)
    del input_map

    input_file_dict = {'intensity': filt_im_str,
                       'landcover': landcover_path,
                       'reference_water': interp_wbd_str,
                       'water_mask': water_map_tif_str}

    split_mask_water_path = \
        os.path.join(outputdir, 'split_mask_water_masking.tif')
    split_extended_water_parallel(
        water_mask_path=os.path.join(outputdir, 'input_image_test.tif'),
        output_path=split_mask_water_path,
        pol_ind=co_pol_ind,
        outputdir=outputdir,
        input_dict=input_file_dict,
        number_workers=number_workers,
        input_lines_per_block=lines_per_block)

    # 4) re-define false water candidate estimated from
    # 'split_extended_water_parallel'
    # by considering the spatial coverage with the ancillary files
    false_water_candidate_path = os.path.join(
        outputdir, 'intermediate_false_water_candidate.tif')
    dswx_sar_util.merge_binary_layers(
        layer_list=[split_mask_water_path,
                    water_map_tif_str],
        value_list=[0, 1],
        merged_layer_path=false_water_candidate_path,
        lines_per_block=lines_per_block,
        mode='and',
        cog_flag=True,
        scratch_dir=outputdir)

    adjacent_false_positive_bindary_path = \
        os.path.join(outputdir, 'false_positive_connected_water.tif')

    compute_spatial_coverage_from_ancillary_parallel(
            flase_water_binary_path=false_water_candidate_path,
            reference_water_path=interp_wbd_str,
            mask_landcover_path=mask_excluded_landcover_path,
            output_file_path=adjacent_false_positive_bindary_path,
            outputdir=outputdir,
            water_max_value=ref_water_max,
            number_workers=number_workers,
            lines_per_block=lines_per_block)

    if hand_variation_mask:
        darkland_removed_path = \
            os.path.join(outputdir, 'masking_ancillary_darkland_removed.tif')
        water_tif_str = \
            os.path.join(outputdir, f"refine_landcover_binary_{pol_str}.tif")
    else:
        darkland_removed_path = \
             os.path.join(outputdir, f"refine_landcover_binary_{pol_str}.tif")

    dswx_sar_util.merge_binary_layers(
        layer_list=[adjacent_false_positive_bindary_path,
                    darkland_cand_path,
                    water_map_tif_str],
        value_list=[0, 0, 1],
        merged_layer_path=darkland_removed_path,
        lines_per_block=lines_per_block,
        mode='and',
        cog_flag=True,
        scratch_dir=outputdir)

    if hand_variation_mask:
        hand_filter_along_boundary(
            target_area_path=darkland_removed_path,
            height_std_threshold=hand_variation_threshold,
            hand_path=hand_path_str,
            output_path=water_tif_str,
            debug_mode=processing_cfg.debug_mode,
            metainfo=water_meta,
            scratch_dir=outputdir)

    if processing_cfg.debug_mode:
        excluded_path = os.path.join(
            outputdir, f"landcover_exclude_{pol_str}.tif")
        dswx_sar_util.save_dswx_product(
            mask_excluded_landcover,
            excluded_path,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Water classification (WTR)',
            scratch_dir=outputdir
            )

    t_all_elapsed = time.time() - t_all
    logger.info("successfully ran landcover masking in "
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
