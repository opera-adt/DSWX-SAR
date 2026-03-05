import copy
import logging
import mimetypes
import os
import time

import cv2
from dswx_sar.common import _dswx_sar_util, _refine_with_bimodality
import numpy as np
import rasterio
from osgeo import gdal
from joblib import Parallel, delayed
from rasterio.windows import Window
from scipy import ndimage
from skimage.filters import threshold_multiotsu
from typing import List, Tuple

from dswx_sar.common import _region_growing
from dswx_sar.common._dswx_sar_util import (
    iter_windows,
    create_gtiff_1band
)

logger = logging.getLogger('dswx_sar')


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
    def __init__(self, landcover_file_path, type):
        '''Initialize FillMaskLandCover

        Parameters
        ----------
        landcover_file_path : str
            path for the landcover file
        '''
        self.landcover_file_path = landcover_file_path
        if not os.path.isfile(self.landcover_file_path):
            raise OSError(f"{self.landcover_file_path} is not found")
        self.type = type

    def open_landcover(self):
        '''Open landcover map

        Returns
        ----------
        landcover_map : numpy.ndarray
            2 dimensional land cover map
        '''
        landcover_map = _dswx_sar_util.read_geotiff(self.landcover_file_path)
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
            landcover = _dswx_sar_util.get_raster_block(
                self.landcover_file_path,
                block_param)
        if self.type == 'WorldCover':
            landcover_label = get_label_landcover_esa_10()
            landcover_binary = np.zeros(landcover.shape, dtype=bool)
            for label_name in mask_label:
                logger.info(f'Land Cover: {label_name} is extracted.')
                temp = landcover == landcover_label[f"{label_name}"]
                landcover_binary[temp] = True
        elif self.type == 'GLAD':
            landcover_binary = np.zeros(landcover.shape, dtype=bool)
            for label_name in mask_label:
                temp = landcover == label_name
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
        2 dimensional label array for each binary object
    """
    rows, cols = binary.shape

    # computes the connected components labeled image of boolean image
    # and also produces a statistics output for each label
    nb_components_water, label_image, stats_water, _ = \
        cv2.connectedComponentsWithStats(binary.astype(np.uint8),
                                         connectivity=8)
    # label_image = label_image.astype(np.int64, copy=False)

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
        metric_obj = _refine_with_bimodality.BimodalityMetrics(intensity_array)
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
                metric_obj_local = _refine_with_bimodality.BimodalityMetrics(
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
                metric_obj_local = _refine_with_bimodality.BimodalityMetrics(
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


def check_water_land_mixture_v2(args):
    """
    Check for water-land mixture in a given region using bimodality metrics.

    Parameters
    ----------
    args : tuple
        (i: int,
         size: int,
         minimum_pixel: int,
         bounds: list[int, int, int, int],
         int_linear_block: np.ndarray,   # (P,H,W) or (H,W) linear intensity
         water_label_block: np.ndarray,  # (H,W), 1-based labels for components
         water_mask_block: np.ndarray,   # (H,W), uint8 {0,1}
         pol_ind: int)

    Returns
    -------
    (i: int, change_flag: bool, water_mask: np.ndarray)
    """

    (i, size, minimum_pixel, bounds,
     int_linear_block, water_label_block, water_mask_block, pol_ind) = args

    # Normalize intensity block to (P,H,W)
    if int_linear_block.ndim == 2:
        int_linear_block = np.expand_dims(int_linear_block, axis=0)

    P, H, W = int_linear_block.shape
    if not (0 <= pol_ind < P):
        # No safe way to proceed if polarization index is invalid
        return i, False, water_mask_block[bounds[2]:bounds[3], bounds[0]:bounds[1]].copy()

    # Slice subsets
    xs, xe, ys, ye = bounds[0], bounds[1], bounds[2], bounds[3]
    int_linear = int_linear_block[pol_ind, ys:ye, xs:xe]
    water_label = water_label_block[ys:ye, xs:xe]
    water_mask  = water_mask_block[ys:ye, xs:xe].copy()

    change_flag = False

    # Target object mask (labels are 1-based)
    target_water = (water_label == (i + 1))

    # Validity masks
    valid_lin = np.isfinite(int_linear) & (int_linear > 0)
    # Precompute dB (finite only)
    int_db = np.full(int_linear.shape, np.nan, dtype=np.float32)
    int_db[valid_lin] = 10.0 * np.log10(int_linear[valid_lin])

    # Quick exit if not enough pixels to justify work
    if (size <= minimum_pixel) or (np.count_nonzero(target_water) == 0):
        return i, False, water_mask

    # Edge band (operate near boundaries only)
    K = 2  # TODO: expose as parameter
    er = ndimage.binary_erosion(target_water, iterations=1, border_value=0)
    dl = ndimage.binary_dilation(target_water, iterations=1, border_value=0)
    edge = ndimage.binary_dilation(dl ^ er, iterations=K, border_value=0)

    # Global bimodality check within the component
    samples_lin = int_linear[target_water]
    # Require some finite samples to avoid spurious metrics
    finite_comp = np.isfinite(samples_lin) & (samples_lin > 0)
    if np.count_nonzero(finite_comp) < 64:
        return i, False, water_mask

    metric_obj = _refine_with_bimodality.BimodalityMetrics(samples_lin[finite_comp])
    bimodality_flag = metric_obj.compute_metric()
    if not bimodality_flag:
        return i, False, water_mask

    logger.info(f'landcover: found bimodality at bounds {bounds}')

    # Limit where dilation-based neighborhoods are allowed:
    # allow only where there is valid intensity AND we are *outside* the label
    # so we probe just beyond the boundary without bleeding into no-data.
    allow = np.isfinite(int_linear) & (water_label == 0)

    # Build finite dB samples inside the component for Otsu
    finite_comp_db = np.isfinite(int_db) & target_water
    if np.count_nonzero(finite_comp_db) < 64:
        return i, False, water_mask
    db_comp = int_db[finite_comp_db]

    # Try multi-Otsu; fall back gracefully if data is degenerate
    try:
        thresholds = threshold_multiotsu(db_comp, nbins=64)
    except Exception:
        return i, False, water_mask

    # Helper to compute buffered band and run local bimodality check
    def _process_band(band_mask):
        # Restrict edits to edge band
        band_mask &= edge
        n = int(np.count_nonzero(band_mask))
        if n == 0:
            return False
        add_iter = max(1, int((np.sqrt(2.0) - 1.2) * np.sqrt(n)))
        # Grow in places we're allowed to inspect (valid intensity & unlabeled)
        buffered = ndimage.binary_dilation(band_mask, iterations=add_iter, mask=allow)
        # Pull linear values in the buffered region; sanitize zeros to NaN
        vals_lin = int_linear[buffered].astype(np.float32)
        vals_lin[~np.isfinite(vals_lin) | (vals_lin <= 0)] = np.nan
        finite = np.isfinite(vals_lin)
        if np.count_nonzero(finite) < 64:
            return False
        vals_db = 10.0 * np.log10(vals_lin[finite])

        # Robust histogram range
        hmin = np.nanpercentile(vals_db, 2.0)
        hmax = np.nanpercentile(vals_db, 98.0)

        metric_local = _refine_with_bimodality.BimodalityMetrics(
            vals_lin[finite], hist_min=hmin, hist_max=hmax
        )
        return metric_local.compute_metric(
            ashman_flag=True, bm_flag=True, surface_ratio_flag=True, bc_flag=False
        )

    # Iterate threshold intervals: “dark” then “bright”
    for t_idx, thr in enumerate(thresholds):
        if t_idx == 0:
            dark = (int_db < thr) & target_water
        else:
            dark = (int_db < thr) & (int_db >= thresholds[t_idx - 1]) & target_water

        if _process_band(dark.copy()):
            water_mask[dark] = 0
            change_flag = True

        if t_idx == len(thresholds) - 1:
            bright = (int_db >= thr) & target_water
        else:
            bright = (int_db >= thr) & (int_db < thresholds[t_idx + 1]) & target_water

        if _process_band(bright.copy()):
            water_mask[bright] = 0
            change_flag = True

    return i, change_flag, water_mask


def split_extended_water_parallel_v2(
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
    meta_info = _dswx_sar_util.get_meta_from_tif(water_mask_path)
    # exceed the size of the block. To prevent discontinuity and incorrect
    # decisions, the block size increases within the for loop.
    # Initially, the procedure is run for the entire area using a small block
    # size (input_lines_per_block), and then the size is increased by a
    # multiple of 3 and 20.
    lines_per_block_set = [input_lines_per_block,
                           input_lines_per_block * 3,
                           input_lines_per_block * 20,
                           meta_info['length']]

    # delete last iteration if rows is equal to 20 * input_lines_per_block
    lines_per_block_set = list(dict.fromkeys(
        [x for x in lines_per_block_set if x <= meta_info['length']]))

    pad_shape = (0, 0)

    temp_prefix = 'split_extended_water_parallel_temp'
    temp_path_set = []
    minimum_pixel = 5000
    process_mask_path = os.path.join(outputdir,
                                     f'{temp_prefix}_process_mask.tif')
    _dswx_sar_util.save_dswx_product(
        np.ones((meta_info['length'], meta_info['width']), dtype='uint8'),
        process_mask_path,
        geotransform=meta_info['geotransform'],
        projection=meta_info['projection'],
        scratch_dir=outputdir)

    for block_iter, lines_per_block in enumerate(lines_per_block_set):
        block_params = _dswx_sar_util.block_param_generator(
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
            water_mask = _dswx_sar_util.get_raster_block(
                water_mask_path, block_param)
            # water_mask = np.where(water_map == 1, 1, 0)

            intensity_block = _dswx_sar_util.get_raster_block(
                input_dict['intensity'], block_param)
            # Read the current process-mask to skip components already done
            process_mask_block = _dswx_sar_util.get_raster_block(
                process_mask_path, block_param)

            # Extract bounding boxes with buffer
            water_mask_for_label = (water_mask != 0) & (process_mask_block != 0)

            coord_list, sizes, label_image = extract_bbox_with_buffer(
                binary=water_mask_for_label, buffer=10)

            filtered_sizes = []
            filtered_coord_list = []
            filtered_index = []
            check_output = np.ones(len(sizes), dtype='byte')
            # old_val = np.arange(1, len(sizes) + 1) - .1
            # index_array_to_image = np.array(
            #     np.searchsorted(old_val, label_image),
            #     dtype='uint32')


            for ind, (coords, size) in enumerate(zip(coord_list, sizes)):
                (bbox_x_start,
                 bbox_x_end,
                 bbox_y_start,
                 bbox_y_end) = coords

                # skip polygons touching this block’s top/bottom, and too small
                if bbox_y_start == 0 or \
                   bbox_y_end == block_param.block_length or \
                   size < minimum_pixel:
                    continue

                # only process if this polygon intersects the process-mask
                sub_proc = process_mask_block[bbox_y_start:bbox_y_end,
                                              bbox_x_start:bbox_x_end]
                sub_lab  = label_image[bbox_y_start:bbox_y_end,
                                       bbox_x_start:bbox_x_end] == (ind + 1)
                if np.count_nonzero(sub_proc & sub_lab) == 0:
                    continue

                filtered_index.append(ind)
                filtered_coord_list.append([bbox_x_start,
                                            bbox_x_end,
                                            bbox_y_start,
                                            bbox_y_end])
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
            results = Parallel(n_jobs=number_workers, prefer='threads')(
                delayed(check_water_land_mixture_v2)(args)
                for args in args_list)

            # If water need to be refined (change_flat=True),
            # then update the water mask.
            for i, change_flag, water_mask_subset in results:
                if change_flag:
                    water_mask[coord_list[i][2]: coord_list[i][3],
                               coord_list[i][0]: coord_list[i][1]] = \
                                water_mask_subset
                check_output[i] = 0

            _dswx_sar_util.write_raster_block(
                removed_false_water_path,
                water_mask,
                block_param,
                geotransform=meta_info['geotransform'],
                projection=meta_info['projection'],
                datatype='byte',
                cog_flag=True,
                scratch_dir=outputdir)

            # Build/update the next pass’s process-mask
            # (only if there IS a next pass)
            if block_iter < len(lines_per_block_set) - 1:
                max_label = int(label_image.max())
                if max_label != len(sizes):
                    logger.warning(
                        f"label max {max_label} != number of sizes {len(sizes)} (unexpected)"
                    )

                lut = np.ones(max_label + 1, dtype=np.uint8)
                lut[0] = 1
                lut[1:1+len(check_output)] = check_output
                check_img = lut[label_image].astype(np.uint8, copy=False)
                check_img = (check_img & process_mask_block).astype(np.uint8, copy=False)
                _dswx_sar_util.write_raster_block(
                    process_mask_path,
                    check_img,
                    block_param,
                    geotransform=meta_info['geotransform'],
                    projection=meta_info['projection'],
                    datatype='byte',
                    cog_flag=True,
                    scratch_dir=outputdir)

    merged_removed_false_water_path = (temp_path_set[0] if len(temp_path_set) == 1
                                       else output_path)

    if len(temp_path_set) >= 2:
        _dswx_sar_util.merge_binary_layers(
            layer_list=temp_path_set,
            value_list=[1] * len(lines_per_block_set),
            merged_layer_path=merged_removed_false_water_path,
            lines_per_block=input_lines_per_block,
            mode='or',
            cog_flag=True,
            scratch_dir=outputdir)

    logger.info(f'split_extended_water_parallel output: '
                f'{merged_removed_false_water_path}')


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
    meta_info = _dswx_sar_util.get_meta_from_tif(water_mask_path)
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
        block_params = _dswx_sar_util.block_param_generator(
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
            water_mask = _dswx_sar_util.get_raster_block(
                water_mask_path, block_param)
            intensity_block = _dswx_sar_util.get_raster_block(
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
                   bbox_y_end != block_param.block_length and \
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
            _dswx_sar_util.write_raster_block(
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
                _dswx_sar_util.write_raster_block(
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

        _dswx_sar_util.merge_binary_layers(
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
        false_water_binary_path: str,
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
    false_water_binary_path : str
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
    water_mask = _dswx_sar_util.read_geotiff(false_water_binary_path)
    meta_info = _dswx_sar_util.get_meta_from_tif(false_water_binary_path)

    # Extract bounding boxes with buffer
    coord_list, sizes, label_image = extract_bbox_with_buffer(
        binary=water_mask, buffer=10)

    # Save label image into geotiff file. The labels are assigned to
    # dark land candidates. This Geotiff file will be used in parallel
    # processing to read sub-areas instead of reading entire image.
    water_label_filename = 'water_label_landcover_spatial_coverage.tif'
    water_label_str = os.path.join(outputdir, water_label_filename)
    _dswx_sar_util.save_raster_gdal(
        data=label_image,
        output_file=water_label_str,
        geotransform=meta_info['geotransform'],
        projection=meta_info['projection'],
        scratch_dir=outputdir)

    nb_components_water = len(sizes)
    print("number of pixels ", np.sum(label_image > 0))
    print("number of pixels ", np.sum(label_image ==0 ))
    if nb_components_water == 0:
        logger.warning("compute_spatial_coverage_from_ancillary_parallel")
        logger.warning("    -- No water components found — writing zero mask and exiting.")
        # dswx_sar_util.save_dswx_product(
        #     np.zeros_like(water_mask, dtype='uint8'),
        #     output_file_path,
        #     geotransform=meta_info['geotransform'],
        #     projection=meta_info['projection'],
        #     scratch_dir=outputdir
        # )
        mask_temp = np.zeros_like(water_mask, dtype='uint8')
        mask_temp[1, 1] = 1
        _dswx_sar_util.save_raster_gdal(
            data=mask_temp,
            output_file=output_file_path,
            geotransform=meta_info['geotransform'],
            projection=meta_info['projection'],
            scratch_dir=outputdir,
            datatype='byte'  # ensure uint8
            # critically: do not call SetNoDataValue
        )
        return
    # From reference water map (0: non-water, 1: permanent water) and
    # landcover map (ex. bare/sparse vegetation), extract the areas
    # which are likely mis-classified as water.
    data_shape = [meta_info['length'], meta_info['width']]

    pad_shape = (0, 0)
    block_params = _dswx_sar_util.block_param_generator(
        lines_per_block,
        data_shape,
        pad_shape)

    ref_land_tif_str = os.path.join(
        outputdir, "landcover_darklandcandidate_spatial_coverage.tif")
    for block_param in block_params:
        mask_excluded = _dswx_sar_util.get_raster_block(
            mask_landcover_path, block_param)
        water_block = _dswx_sar_util.get_raster_block(
            reference_water_path, block_param)

        dry_darkland = np.logical_and(
            mask_excluded, water_block/water_max_value < 0.05)

        # Protect flood expansions near persistent water (occurrence >= protect_occ)
        protect_occ = 0.10  # make configurable
        protect_buffer = 3   # in pixels; make configurable
        near_persistent = (water_block / water_max_value) >= protect_occ
        near_persistent = ndimage.binary_dilation(near_persistent,
                                                  iterations=protect_buffer)

        dry_darkland = np.logical_and(dry_darkland, np.logical_not(near_persistent))

        _dswx_sar_util.write_raster_block(
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

    # Output consists of index and 2D image consisting of True/False.
    # True represents the land and False represents not-land.
    results = Parallel(
        n_jobs=number_workers,
        prefer="threads")(
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

    _dswx_sar_util.save_dswx_product(
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

def extend_land_cover_v2(
        landcover_path,
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
    mask_obj = FillMaskLandCover(landcover_path, 'WorldCover')

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
    _dswx_sar_util.save_dswx_product(
        exclude_area_rg,
        excluded_rg_path,
        geotransform=metainfo['geotransform'],
        projection=metainfo['projection'],
        scratch_dir=scratch_dir
        )

    darkland_tif_str = os.path.join(
        scratch_dir, "landcover_target.tif")
    _dswx_sar_util.save_dswx_product(
        reference_landcover_binary,
        darkland_tif_str,
        geotransform=metainfo['geotransform'],
        projection=metainfo['projection'],
        scratch_dir=scratch_dir
        )

    darkland_cand_tif_str = os.path.join(
        scratch_dir, "landcover_target_candidate.tif")
    _dswx_sar_util.save_raster_gdal(
        np.array(new_landcover, dtype='float32'),
        darkland_cand_tif_str,
        geotransform=metainfo['geotransform'],
        projection=metainfo['projection'],
        scratch_dir=scratch_dir,
        datatype='float32')

    temp_rg_tif_path = os.path.join(
        scratch_dir, 'landcover_temp_transition.tif')

    _region_growing.run_parallel_region_growing(
        darkland_cand_tif_str,
        temp_rg_tif_path,
        exclude_area_path=excluded_rg_path,
        lines_per_block=1000,
        initial_threshold=0.9,
        relaxed_threshold=0.7,
        maxiter=0,
        rg_method='fast')

    fuzzy_map = _dswx_sar_util.read_geotiff(darkland_cand_tif_str)
    temp_rg = _dswx_sar_util.read_geotiff(temp_rg_tif_path)

    # replace the fuzzy values with 1 for the pixels
    # where the region-growing already applied
    fuzzy_map[temp_rg == 1] = 1

    # Run region-growing again for entire image
    region_grow_map = _region_growing.region_growing_fast(
        likelihood_image = fuzzy_map,
        initial_threshold=0.9,
        relaxed_threshold=0.7,
        exclude_area=exclude_area_rg,
        maxiter=0)
    reference_landcover_binary[region_grow_map] = 1

    return reference_landcover_binary

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
    mask_obj = FillMaskLandCover(landcover_path, 'WorldCover')

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
    _dswx_sar_util.save_dswx_product(
        exclude_area_rg,
        excluded_rg_path,
        geotransform=metainfo['geotransform'],
        projection=metainfo['projection'],
        scratch_dir=scratch_dir
        )

    darkland_tif_str = os.path.join(
        scratch_dir, "landcover_target.tif")
    _dswx_sar_util.save_dswx_product(
        reference_landcover_binary,
        darkland_tif_str,
        geotransform=metainfo['geotransform'],
        projection=metainfo['projection'],
        scratch_dir=scratch_dir
        )

    darkland_cand_tif_str = os.path.join(
        scratch_dir, "landcover_target_candidate.tif")
    _dswx_sar_util.save_raster_gdal(
        np.array(new_landcover, dtype='float32'),
        darkland_cand_tif_str,
        geotransform=metainfo['geotransform'],
        projection=metainfo['projection'],
        scratch_dir=scratch_dir,
        datatype='float32')

    temp_rg_tif_path = os.path.join(
        scratch_dir, 'landcover_temp_transition.tif')

    _region_growing.run_parallel_region_growing(
        darkland_cand_tif_str,
        temp_rg_tif_path,
        exclude_area_path=excluded_rg_path,
        lines_per_block=1000,
        initial_threshold=0.9,
        relaxed_threshold=0.7,
        maxiter=0)

    fuzzy_map = _dswx_sar_util.read_geotiff(darkland_cand_tif_str)
    temp_rg = _dswx_sar_util.read_geotiff(temp_rg_tif_path)

    # replace the fuzzy values with 1 for the pixels
    # where the region-growing already applied
    fuzzy_map[temp_rg == 1] = 1

    # Run region-growing again for entire image
    region_grow_map = _region_growing.region_growing(
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
    target_area = _dswx_sar_util.read_geotiff(target_area_path)
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

    _dswx_sar_util.save_dswx_product(
        target_area,
        output_path,
        geotransform=metainfo['geotransform'],
        projection=metainfo['projection'],
        scratch_dir=scratch_dir
        )

    if debug_mode:
        hand_std_path = os.path.join(
            scratch_dir, "landcover_hand_std.tif")
        _dswx_sar_util.save_raster_gdal(
            np.array(height_std_raster, dtype='float32'),
            hand_std_path,
            geotransform=metainfo['geotransform'],
            projection=metainfo['projection'],
            scratch_dir=scratch_dir,
            datatype='float32')
        hand_std_path = os.path.join(
            scratch_dir, "landcover_hand_std_image.tif")
        _dswx_sar_util.save_raster_gdal(
            np.array(hand_std_image, dtype='float32'),
            hand_std_path,
            geotransform=metainfo['geotransform'],
            projection=metainfo['projection'],
            scratch_dir=scratch_dir,
            datatype='float32')
        hand_binary_path = os.path.join(
            scratch_dir, "landcover_hand_binary.tif")
        _dswx_sar_util.save_dswx_product(
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
    band_meta = _dswx_sar_util.get_meta_from_tif(intensity_path)
    data_shape = [band_meta['length'], band_meta['width']]

    pad_shape = (0, 0)
    block_params = _dswx_sar_util.block_param_generator(
        lines_per_block, data_shape, pad_shape)

    for block_param in block_params:
        intensity_block = _dswx_sar_util.get_raster_block(
            intensity_path, block_param)
        mask_excluded_landcover = _dswx_sar_util.get_raster_block(
            landcover_path, block_param)
        water_block = _dswx_sar_util.get_raster_block(
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

        _dswx_sar_util.write_raster_block(
            out_raster=darkland_candidate_path,
            data=mask_excluded,
            block_param=block_param,
            geotransform=band_meta['geotransform'],
            projection=band_meta['projection'],
            datatype='byte')


# Updated (Feb 10, 2026). This function is applied only to DSWx-NI.
def write_or_masks(mask_a_path, mask_b_path, out_path, metainfo, block_x=2048, block_y=2048):


    a_ds = gdal.Open(mask_a_path, gdal.GA_ReadOnly)
    b_ds = gdal.Open(mask_b_path, gdal.GA_ReadOnly)
    if a_ds is None or b_ds is None:
        raise RuntimeError("Failed to open input masks")

    xsize, ysize = a_ds.RasterXSize, a_ds.RasterYSize
    if b_ds.RasterXSize != xsize or b_ds.RasterYSize != ysize:
        raise ValueError("Mask size mismatch")

    a_b = a_ds.GetRasterBand(1)
    b_b = b_ds.GetRasterBand(1)

    driver = gdal.GetDriverByName("GTiff")
    opts = [
        "TILED=YES","BLOCKXSIZE=512","BLOCKYSIZE=512",
        "COMPRESS=DEFLATE","PREDICTOR=2","ZLEVEL=6","BIGTIFF=IF_SAFER","NBITS=8"
    ]
    out_ds = driver.Create(out_path, xsize, ysize, 1, gdal.GDT_Byte, options=opts)
    out_ds.SetGeoTransform(metainfo["geotransform"])
    out_ds.SetProjection(metainfo["projection"])
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(0)

    for yoff in range(0, ysize, block_y):
        ywin = min(block_y, ysize - yoff)
        for xoff in range(0, xsize, block_x):
            xwin = min(block_x, xsize - xoff)
            a = a_b.ReadAsArray(xoff, yoff, xwin, ywin).astype(np.uint8, copy=False)
            b = b_b.ReadAsArray(xoff, yoff, xwin, ywin).astype(np.uint8, copy=False)
            out = ((a != 0) | (b != 0)).astype(np.uint8, copy=False)
            out_band.WriteArray(out, xoff, yoff)

    out_band.FlushCache()
    out_ds.FlushCache()
    out_band = None; out_ds = None
    a_b = None; b_b = None
    a_ds = None; b_ds = None


def make_lut_u8(values):
    lut = np.zeros(256, dtype=np.uint8)
    lut[np.asarray(values, dtype=np.uint16)] = 1
    return lut


def write_or_mask_landcover_glad_streaming(
    landcover_path: str,
    glad_path: str,
    out_path: str,
    landcover_masking_list,   # WorldCover names (strings)
    glad_excluded_values,               # list[int] (e.g. parse_ranges(['1-24']))
    geotransform,
    projection,
    block_x: int = 2048,
    block_y: int = 2048,
    logger=None,
):
    lc_ds = gdal.Open(landcover_path, gdal.GA_ReadOnly)
    gl_ds = gdal.Open(glad_path, gdal.GA_ReadOnly)
    if lc_ds is None:
        raise RuntimeError(f"Failed to open: {landcover_path}")
    if gl_ds is None:
        raise RuntimeError(f"Failed to open: {glad_path}")

    xsize, ysize = lc_ds.RasterXSize, lc_ds.RasterYSize
    if gl_ds.RasterXSize != xsize or gl_ds.RasterYSize != ysize:
        raise ValueError("Landcover/GLAD raster size mismatch")

    lc_band = lc_ds.GetRasterBand(1)
    gl_band = gl_ds.GetRasterBand(1)

    # precompute codes once
    wc_codes = worldcover_names_to_codes(landcover_masking_list)
    lut_wc = make_lut_u8(wc_codes)
    glad_vals = np.asarray(list(glad_excluded_values), dtype=np.int32)
    gl_max = int(max(glad_vals)) if len(glad_vals) else 0
    lut_gl = make_lut_u8(glad_vals)  # or read max from data once

    glad_codes = np.asarray(list(glad_excluded_values), dtype=np.int32)

    # create output (1-band uint8 mask)
    driver = gdal.GetDriverByName("GTiff")
    create_opts = [
        "TILED=YES",
        "BLOCKXSIZE=512",
        "BLOCKYSIZE=512",
        "COMPRESS=DEFLATE",
        "PREDICTOR=2",
        "ZLEVEL=6",
        "BIGTIFF=IF_SAFER",
        "NBITS=8",
    ]
    out_ds = driver.Create(out_path, xsize, ysize, 1, gdal.GDT_Byte, options=create_opts)
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(0)

    # window loop
    for yoff in range(0, ysize, block_y):
        ywin = min(block_y, ysize - yoff)
        for xoff in range(0, xsize, block_x):
            xwin = min(block_x, xsize - xoff)

            lc = lc_band.ReadAsArray(xoff, yoff, xwin, ywin)
            gl = gl_band.ReadAsArray(xoff, yoff, xwin, ywin)

            # membership masks (vectorized)
            # m_lc = np.isin(lc, wc_codes)
            # m_gl = np.isin(gl, glad_codes)
            # m_lc = lut_wc[lc] != 0
            out = ((lut_wc[lc] | lut_gl[gl]) != 0).astype(np.uint8, copy=False)

            # out = (m_lc | m_gl).astype(np.uint8, copy=False)
            out_band.WriteArray(out, xoff, yoff)

    out_band.FlushCache()
    out_ds.FlushCache()

    # close
    out_band = None
    out_ds = None
    lc_band = None
    gl_band = None
    lc_ds = None
    gl_ds = None

    if logger:
        logger.info(f"Wrote: {out_path}")


def write_rg_excluded_area_from_wbd_and_glad_streaming(
    interp_wbd_str, glad_path, out_path,
    thr, glad_vals, geotransform, projection,
    block_x=2048, block_y=2048
    ):
    wbd_ds = gdal.Open(interp_wbd_str, gdal.GA_ReadOnly)
    gl_ds  = gdal.Open(glad_path, gdal.GA_ReadOnly)
    if wbd_ds is None:
        raise RuntimeError(f"Failed to open: {interp_wbd_str}")
    if gl_ds is None:
        raise RuntimeError(f"Failed to open: {glad_path}")

    xsize, ysize = wbd_ds.RasterXSize, wbd_ds.RasterYSize
    if gl_ds.RasterXSize != xsize or gl_ds.RasterYSize != ysize:
        raise ValueError("WBD and GLAD-mask size mismatch")
    wbd_band = wbd_ds.GetRasterBand(1)
    gl_band  = gl_ds.GetRasterBand(1)

    driver = gdal.GetDriverByName("GTiff")
    create_opts = [
        "TILED=YES","BLOCKXSIZE=512","BLOCKYSIZE=512",
        "COMPRESS=DEFLATE","PREDICTOR=2","ZLEVEL=6","BIGTIFF=IF_SAFER","NBITS=8",
    ]
    out_ds = driver.Create(out_path, xsize, ysize, 1, gdal.GDT_Byte, options=create_opts)
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(0)

    lut_gl = make_lut_u8(glad_vals)
    for yoff in range(0, ysize, block_y):
        ywin = min(block_y, ysize - yoff)
        for xoff in range(0, xsize, block_x):
            xwin = min(block_x, xsize - xoff)

            wbd = wbd_band.ReadAsArray(xoff, yoff, xwin, ywin).astype(np.float32, copy=False)
            gl  = gl_band.ReadAsArray(xoff, yoff, xwin, ywin).astype(np.uint8, copy=False)

            m = ((wbd > thr) | (lut_gl[gl] != 0)).astype(np.uint8, copy=False)
            out_band.WriteArray(m, xoff, yoff)
    out_band.FlushCache()
    out_ds.FlushCache()
    out_band = None
    out_ds = None
    wbd_band = None
    gl_band = None
    wbd_ds = None
    gl_ds = None


#
def _make_kernel(radius: int):
    k = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))



def worldcover_names_to_codes(mask_label_names):
    """Convert WorldCover label names -> integer codes."""
    lut = get_label_landcover_esa_10()
    return np.asarray([lut[name] for name in mask_label_names], dtype=np.int16)


def write_worldcover_mask_streaming(landcover_path, out_path, label_names, metainfo,
                                   block_x=2048, block_y=2048):
    ds = gdal.Open(landcover_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Failed to open {landcover_path}")
    band = ds.GetRasterBand(1)
    xsize, ysize = ds.RasterXSize, ds.RasterYSize

    codes = worldcover_names_to_codes(label_names)

    out_ds, out_band = create_gtiff_1band(
        out_path, xsize, ysize, gdal.GDT_Byte,
        metainfo["geotransform"], metainfo["projection"],
        nodata=0, nbits=8
    )

    for xoff, yoff, xwin, ywin in iter_windows(xsize, ysize, block_x, block_y):
        lc = band.ReadAsArray(xoff, yoff, xwin, ywin)
        m = np.isin(lc, codes).astype(np.uint8, copy=False)
        out_band.WriteArray(m, xoff, yoff)

    out_band.FlushCache(); out_ds.FlushCache()
    out_band = None; out_ds = None
    band = None; ds = None


def write_buffered_binary_mask(in_path, out_path, radius, metainfo,
                               block_x=2048, block_y=2048):
    """
    Dilate a binary mask by 'radius' pixels using tile+halo.
    Input must be 0/1 uint8.
    """
    ds = gdal.Open(in_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Failed to open {in_path}")
    band = ds.GetRasterBand(1)
    xsize, ysize = ds.RasterXSize, ds.RasterYSize

    out_ds, out_band = create_gtiff_1band(
        out_path, xsize, ysize, gdal.GDT_Byte,
        metainfo["geotransform"], metainfo["projection"],
        nodata=0, nbits=8
    )

    halo = int(radius)
    st = ndimage.generate_binary_structure(2, 1)  # 4-neighborhood; change if you want 8

    for xoff, yoff, xwin, ywin in iter_windows(xsize, ysize, block_x, block_y):
        # expanded read window
        x0 = max(0, xoff - halo)
        y0 = max(0, yoff - halo)
        x1 = min(xsize, xoff + xwin + halo)
        y1 = min(ysize, yoff + ywin + halo)

        big = band.ReadAsArray(x0, y0, x1 - x0, y1 - y0).astype(bool, copy=False)
        dil = ndimage.binary_dilation(big, structure=st, iterations=halo)

        # crop back to core
        cx0 = xoff - x0
        cy0 = yoff - y0
        core = dil[cy0:cy0 + ywin, cx0:cx0 + xwin].astype(np.uint8, copy=False)
        out_band.WriteArray(core, xoff, yoff)

    out_band.FlushCache(); out_ds.FlushCache()
    out_band = None; out_ds = None
    band = None; ds = None



def write_fuzzy_likelihood(reference_binary_path, candidate_mask_path, water_buffer_mask_path_or_none,
                           out_path, metainfo, block_x=2048, block_y=2048):
    ref_ds = gdal.Open(reference_binary_path, gdal.GA_ReadOnly)
    cand_ds = gdal.Open(candidate_mask_path, gdal.GA_ReadOnly)
    if ref_ds is None or cand_ds is None:
        raise RuntimeError("Failed to open ref/candidate inputs")

    ref_b = ref_ds.GetRasterBand(1)
    cand_b = cand_ds.GetRasterBand(1)
    xsize, ysize = ref_ds.RasterXSize, ref_ds.RasterYSize

    if cand_ds.RasterXSize != xsize or cand_ds.RasterYSize != ysize:
        raise ValueError("reference_binary and candidate_mask size mismatch")

    wb_ds = None
    wb_b = None
    if water_buffer_mask_path_or_none is not None:
        wb_ds = gdal.Open(water_buffer_mask_path_or_none, gdal.GA_ReadOnly)
        if wb_ds is None:
            raise RuntimeError(f"Failed to open {water_buffer_mask_path_or_none}")
        if wb_ds.RasterXSize != xsize or wb_ds.RasterYSize != ysize:
            raise ValueError("water_buffer_mask size mismatch")
        wb_b = wb_ds.GetRasterBand(1)

    out_ds, out_band = create_gtiff_1band(
        out_path, xsize, ysize, gdal.GDT_Float32,
        metainfo["geotransform"], metainfo["projection"],
        nodata=0.0
    )

    for xoff, yoff, xwin, ywin in iter_windows(xsize, ysize, block_x, block_y):
        ref = ref_b.ReadAsArray(xoff, yoff, xwin, ywin).astype(np.uint8, copy=False)
        cand = cand_b.ReadAsArray(xoff, yoff, xwin, ywin).astype(np.uint8, copy=False)

        # start at 0
        fuzzy = np.zeros((ywin, xwin), dtype=np.float32)
        fuzzy[ref != 0] = 1.0
        fuzzy[(ref == 0) & (cand != 0)] = 0.75

        if wb_b is not None:
            wb = wb_b.ReadAsArray(xoff, yoff, xwin, ywin).astype(np.uint8, copy=False)
            fuzzy[wb != 0] = 0.0

        out_band.WriteArray(fuzzy, xoff, yoff)

    out_band.FlushCache(); out_ds.FlushCache()
    out_band = None; out_ds = None
    ref_b = None; cand_b = None
    ref_ds = None; cand_ds = None
    if wb_b is not None:
        wb_b = None; wb_ds = None


def write_fuzzy_updated_with_rg(fuzzy_path, rg_path, out_path, metainfo,
                               block_x=2048, block_y=2048):
    f_ds = gdal.Open(fuzzy_path, gdal.GA_ReadOnly)
    r_ds = gdal.Open(rg_path, gdal.GA_ReadOnly)
    if f_ds is None or r_ds is None:
        raise RuntimeError("Failed to open fuzzy/rg")

    f_b = f_ds.GetRasterBand(1)
    r_b = r_ds.GetRasterBand(1)
    xsize, ysize = f_ds.RasterXSize, f_ds.RasterYSize

    out_ds, out_band = create_gtiff_1band(
        out_path, xsize, ysize, gdal.GDT_Float32,
        metainfo["geotransform"], metainfo["projection"],
        nodata=0.0
    )

    for xoff, yoff, xwin, ywin in iter_windows(xsize, ysize, block_x, block_y):
        fuzzy = f_b.ReadAsArray(xoff, yoff, xwin, ywin).astype(np.float32, copy=False)
        rg = r_b.ReadAsArray(xoff, yoff, xwin, ywin).astype(np.uint8, copy=False)
        # where rg==1, set fuzzy to 1.0
        out = fuzzy.copy()
        out[rg != 0] = 1.0
        out_band.WriteArray(out, xoff, yoff)

    out_band.FlushCache(); out_ds.FlushCache()
    out_band = None; out_ds = None
    f_b = None; r_b = None
    f_ds = None; r_ds = None



def write_or_binary(ref_path, rg_path, out_path, metainfo, block_x=2048, block_y=2048):
    a_ds = gdal.Open(ref_path, gdal.GA_ReadOnly)
    b_ds = gdal.Open(rg_path, gdal.GA_ReadOnly)
    if a_ds is None or b_ds is None:
        raise RuntimeError("Failed to open ref/rg")
    a_b = a_ds.GetRasterBand(1)
    b_b = b_ds.GetRasterBand(1)
    xsize, ysize = a_ds.RasterXSize, a_ds.RasterYSize

    out_ds, out_band = create_gtiff_1band(
        out_path, xsize, ysize, gdal.GDT_Byte,
        metainfo["geotransform"], metainfo["projection"],
        nodata=0, nbits=8
    )

    for xoff, yoff, xwin, ywin in iter_windows(xsize, ysize, block_x, block_y):
        a = a_b.ReadAsArray(xoff, yoff, xwin, ywin).astype(np.uint8, copy=False)
        b = b_b.ReadAsArray(xoff, yoff, xwin, ywin).astype(np.uint8, copy=False)
        out = ((a != 0) | (b != 0)).astype(np.uint8, copy=False)
        out_band.WriteArray(out, xoff, yoff)

    out_band.FlushCache(); out_ds.FlushCache()
    out_band = None; out_ds = None
    a_b = None; b_b = None
    a_ds = None; b_ds = None



def extend_land_cover_v3(
    landcover_path,
    reference_landcover_binary_path,
    target_landcover,
    water_landcover,
    exclude_area_rg_path,
    minimum_pixel,
    water_buffer,
    metainfo,
    scratch_dir,
    output_binary_path,
    lines_per_block=2048,
):
    """
    Streaming + path-based version:
    - no full-scene numpy arrays
    - uses run_parallel_region_growing twice (block-based)
    """

    logger.info("Extending land cover (streaming v3)....")

    os.makedirs(scratch_dir, exist_ok=True)

    # 1) Candidate mask from WorldCover
    candidate_mask_path = os.path.join(scratch_dir, "lc_target_candidate_mask.tif")
    write_worldcover_mask_streaming(
        landcover_path=landcover_path,
        out_path=candidate_mask_path,
        label_names=target_landcover,
        metainfo=metainfo,
        block_x=2048,
        block_y=2048,
    )

    # 2) Optional: buffered water mask
    water_buffer_mask_path = None
    if water_buffer > 0:
        logger.info(f"Excluding buffered area along water (buffer={water_buffer})")
        water_mask_path = os.path.join(scratch_dir, "lc_water_mask.tif")
        write_worldcover_mask_streaming(
            landcover_path=landcover_path,
            out_path=water_mask_path,
            label_names=water_landcover,
            metainfo=metainfo,
            block_x=2048,
            block_y=2048,
        )
        water_buffer_mask_path = os.path.join(scratch_dir, "lc_water_mask_buffered.tif")
        write_buffered_binary_mask(
            in_path=water_mask_path,
            out_path=water_buffer_mask_path,
            radius=water_buffer,
            metainfo=metainfo,
            block_x=2048,
            block_y=2048,
        )

    # 3) Optional: remove small components from seeds without full RAM
    # NOTE: your previous remove_small_components() likely loads full arrays.
    # Best scalable replacement is GDAL sieve; do it here if you want.
    seed_path = reference_landcover_binary_path
    if minimum_pixel > 0:
        logger.info(f"Removing small components less than {minimum_pixel} pixels (recommend GDAL SieveFilter)")
        # Placeholder: if you already have a raster-based sieve util, call it here.
        # Otherwise, keep minimum_pixel=0 until you implement sieve safely.
        # seed_path = sieve_binary_raster(seed_path, out_path=..., threshold=minimum_pixel, metainfo=metainfo)

    # 4) Build fuzzy likelihood raster fuzzy0
    fuzzy0_path = os.path.join(scratch_dir, "lc_fuzzy0.tif")
    write_fuzzy_likelihood(
        reference_binary_path=seed_path,
        candidate_mask_path=candidate_mask_path,
        water_buffer_mask_path_or_none=water_buffer_mask_path,
        out_path=fuzzy0_path,
        metainfo=metainfo,
        block_x=2048,
        block_y=2048,
    )

    # 5) First block RG (uses your existing block-based RG implementation)
    rg0_path = os.path.join(scratch_dir, "lc_rg0.tif")
    _region_growing.run_parallel_region_growing(
        fuzzy0_path,
        rg0_path,
        exclude_area_path=exclude_area_rg_path,
        lines_per_block=lines_per_block,
        initial_threshold=0.9,
        relaxed_threshold=0.7,
        maxiter=0,
        rg_method="fast",
    )

    # 6) Update fuzzy where rg0==1
    fuzzy1_path = os.path.join(scratch_dir, "lc_fuzzy1.tif")
    write_fuzzy_updated_with_rg(
        fuzzy_path=fuzzy0_path,
        rg_path=rg0_path,
        out_path=fuzzy1_path,
        metainfo=metainfo,
        block_x=2048,
        block_y=2048,
    )

    # 7) Second block RG (replaces region_growing_fast on full arrays)
    rg1_path = os.path.join(scratch_dir, "lc_rg1.tif")
    _region_growing.run_parallel_region_growing(
        fuzzy1_path,
        rg1_path,
        exclude_area_path=exclude_area_rg_path,
        lines_per_block=lines_per_block,
        initial_threshold=0.9,
        relaxed_threshold=0.7,
        maxiter=0,
        rg_method="fast",
    )

    # 8) Final extended binary = seed OR rg1
    write_or_binary(
        ref_path=seed_path,
        rg_path=rg1_path,
        out_path=output_binary_path,
        metainfo=metainfo,
        block_x=2048,
        block_y=2048,
    )

    logger.info("Landcover extension completed (v3).")
    return output_binary_path



# Updated HAND filter method (Feb 20, 2026)
def _create_u8_gtiff(path, xsize, ysize, gt, proj, nodata=0,
                     blockx=512, blocky=512, zlevel=6, compress="DEFLATE"):
    drv = gdal.GetDriverByName("GTiff")
    opts = [
        "TILED=YES",
        f"BLOCKXSIZE={blockx}",
        f"BLOCKYSIZE={blocky}",
        f"COMPRESS={compress}",
        "PREDICTOR=2",
        f"ZLEVEL={zlevel}",
        "BIGTIFF=IF_SAFER",
        "NBITS=8",
    ]
    ds = drv.Create(path, xsize, ysize, 1, gdal.GDT_Byte, options=opts)
    if ds is None:
        raise RuntimeError(f"Failed to create: {path}")
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    return ds, band


def _or_write_mask_window(mask_band, x0, y0, win_w, win_h, add_mask_u8):
    """
    removed_union_mask.tif 에서 bbox window만 OR 누적.
    add_mask_u8: uint8 (0/1)
    """
    prev = mask_band.ReadAsArray(x0, y0, win_w, win_h)
    if prev is None:
        prev = np.zeros((win_h, win_w), dtype=np.uint8)
    out = (prev | add_mask_u8).astype(np.uint8, copy=False)
    mask_band.WriteArray(out, x0, y0)


def _extract_boundary_u8(mask_u8: np.ndarray, ker1: np.ndarray) -> np.ndarray:
    """
    boundary = mask & ~erode(mask)  (8-connectivity-ish depending on kernel)
    mask_u8: 0/1 uint8
    """
    er = cv2.erode(mask_u8, ker1, iterations=1)
    return (mask_u8 & (1 - er)).astype(np.uint8, copy=False)


def _hand_vals_on_boundary(hand_f32: np.ndarray,
                           boundary_u8: np.ndarray,
                           valid_mask: np.ndarray) -> np.ndarray:
    """
    Return 1D array of HAND values on boundary pixels, excluding nodata/NaN.
    """
    b = (boundary_u8 != 0) & valid_mask
    vals = hand_f32[b]
    # guard: remove NaN just in case
    if vals.size == 0:
        return vals
    return vals[np.isfinite(vals)]


def hand_filter_along_boundary_componentwise(
    target_area_path: str,
    height_std_threshold: float,
    hand_path: str,
    output_path: str,
    metainfo: dict,
    scratch_dir: str,
    buffer_pixels: int = 10,
    max_iters: int = 64,
    debug_mode: bool = False,
    block_y_finalize: int = 4096,
    connectivity: int = 8,
):
    """
    Original logic preserved (matches your reference function):
      - component bbox (+buffer)
      - compute HAND std from INITIAL boundary once:
            if std <= threshold -> no erosion
      - else iterative masked erosion:
            thr = median(boundary_vals) + std(boundary_vals)*1
            hand_image_mask = (hand > thr)
            new = binary_erosion(initial_area, mask=hand_image_mask)
            repeat until bad_hand_count==0 or empty

    Output writing (Option A):
      - OR accumulate removed pixels into removed_union_mask
      - finalize output = target & (~removed_union_mask) in stripes

    Debug:
      - landcover_hand_std.tif: per-component constant std via LUT mapping from labels
    """

    def _create_f32_gtiff(path, xsize, ysize, gt, proj,
                        blockx=512, blocky=512, zlevel=6, compress="DEFLATE",
                        nodata=np.nan):
        drv = gdal.GetDriverByName("GTiff")
        opts = [
            "TILED=YES",
            f"BLOCKXSIZE={blockx}",
            f"BLOCKYSIZE={blocky}",
            f"COMPRESS={compress}",
            "PREDICTOR=2",
            f"ZLEVEL={zlevel}",
            "BIGTIFF=IF_SAFER",
        ]
        ds = drv.Create(path, xsize, ysize, 1, gdal.GDT_Float32, options=opts)
        if ds is None:
            raise RuntimeError(f"Failed to create: {path}")
        ds.SetGeoTransform(gt)
        ds.SetProjection(proj)
        band = ds.GetRasterBand(1)
        if nodata is not None and np.isfinite(nodata):
            band.SetNoDataValue(float(nodata))
        # if nodata is NaN, GDAL nodata handling varies; OK to skip SetNoDataValue
        return ds, band
    gt = metainfo["geotransform"]
    proj = metainfo["projection"]
    os.makedirs(scratch_dir, exist_ok=True)

    # Open inputs
    tgt_ds = gdal.Open(target_area_path, gdal.GA_ReadOnly)
    if tgt_ds is None:
        raise RuntimeError(f"Failed to open: {target_area_path}")
    tgt_band = tgt_ds.GetRasterBand(1)
    xsize, ysize = tgt_ds.RasterXSize, tgt_ds.RasterYSize

    hand_ds = gdal.Open(hand_path, gdal.GA_ReadOnly)
    if hand_ds is None:
        raise RuntimeError(f"Failed to open: {hand_path}")
    hand_band = hand_ds.GetRasterBand(1)
    if hand_ds.RasterXSize != xsize or hand_ds.RasterYSize != ysize:
        raise ValueError("HAND/target size mismatch")
    hand_nodata = hand_band.GetNoDataValue()

    # removed_union_mask
    removed_mask_path = os.path.join(scratch_dir, "removed_union_mask.tif")
    rm_ds, rm_band = _create_u8_gtiff(removed_mask_path, xsize, ysize, gt, proj, nodata=0)

    # Load target as u8 for component labeling (this is the one big RAM object)
    tgt_full = tgt_band.ReadAsArray(0, 0, xsize, ysize)
    if tgt_full is None:
        raise RuntimeError("target ReadAsArray returned None")
    tgt_u8_full = (tgt_full != 0).astype(np.uint8, copy=False)
    del tgt_full

    # Connected components on full scene (same as your cv2 approach)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(
        tgt_u8_full, connectivity=connectivity
    )
    # free if you want: tgt_u8_full not needed anymore after labels/stats
    # (labels still RAM heavy)
    # del tgt_u8_full

    # Debug std LUT (index=label id)
    hand_std_values = None
    if debug_mode:
        hand_std_values = np.full(nlab, np.nan, dtype=np.float32)

    ker1 = _make_kernel(1)

    # Loop components
    for lab in range(1, nlab):
        x, y, w, h, area = stats[lab]
        if area == 0:
            continue

        # bbox + buffer
        x0 = max(0, x - buffer_pixels)
        y0 = max(0, y - buffer_pixels)
        x1 = min(xsize, x + w + buffer_pixels)
        y1 = min(ysize, y + h + buffer_pixels)
        win_w = x1 - x0
        win_h = y1 - y0

        # Component mask in window
        sub_label = labels[y0:y1, x0:x1]
        comp_u8 = (sub_label == lab).astype(np.uint8)

        if comp_u8.sum() == 0:
            continue

        # Boundary of initial component
        boundary_u8 = _extract_boundary_u8(comp_u8, ker1)
        if boundary_u8.sum() == 0:
            # no boundary -> nothing to erode
            if debug_mode:
                hand_std_values[lab] = 0.0
            continue

        # Read HAND in window
        hand = hand_band.ReadAsArray(x0, y0, win_w, win_h)
        if hand is None:
            continue
        hand = hand.astype(np.float32, copy=False)

        # valid HAND mask
        if hand_nodata is None:
            valid = np.isfinite(hand)
        else:
            valid = (hand != float(hand_nodata)) & np.isfinite(hand)

        # INITIAL boundary stats -> decide whether to run erosion at all
        vals0 = _hand_vals_on_boundary(hand, boundary_u8, valid)
        if vals0.size == 0:
            if debug_mode:
                hand_std_values[lab] = np.nan
            continue

        hand_std0 = float(np.nanstd(vals0))
        if debug_mode:
            hand_std_values[lab] = hand_std0 if np.isfinite(hand_std0) else np.nan

        if (not np.isfinite(hand_std0)) or (hand_std0 <= height_std_threshold):
            # keep component (no removals)
            continue

        # --- ORIGINAL ITERATION LOGIC ---
        initial_area = comp_u8.astype(bool, copy=False)

        # build SciPy structure matching "default" 3x3 neighborhood
        # ndimage.binary_erosion uses structure; None => full connectivity.
        # Use 3x3 ones for typical erosion behavior.
        structure = np.ones((3, 3), dtype=bool)

        bad_hand_count = 1
        it = 0

        while bad_hand_count > 0 and it < max_iters:
            # boundary on current area
            cur_u8 = initial_area.astype(np.uint8, copy=False)
            cur_boundary = _extract_boundary_u8(cur_u8, ker1)
            if cur_boundary.sum() == 0:
                break

            vals = _hand_vals_on_boundary(hand, cur_boundary, valid)
            if vals.size == 0:
                break

            hand_std = float(np.nanstd(vals))
            if not np.isfinite(hand_std):
                break

            area_median = float(np.nanmedian(vals))
            if not np.isfinite(area_median):
                break

            hand_threshold_erosion = area_median + hand_std * 1.0

            # hand_image_mask (window-sized), exactly like original
            hand_image_mask = (hand > hand_threshold_erosion) & valid

            # masked erosion (original semantics)
            new_binary = ndimage.binary_erosion(
                initial_area,
                structure=structure,
                mask=hand_image_mask
            )

            # stop if fully gone
            if not new_binary.any():
                initial_area = new_binary
                break

            # recompute "bad pixels count" using boundary-derived mask (original does sum(hand_image_mask))
            # original uses hand_image_data > thr and then sum; effectively counts masked pixels (not only boundary).
            bad_hand_count = int(np.sum(hand_image_mask))

            initial_area = new_binary
            it += 1

        # compute removed pixels (only where comp was 1 but final is 0)
        final_u8 = initial_area.astype(np.uint8, copy=False)
        removed_u8 = ((comp_u8 != 0) & (final_u8 == 0)).astype(np.uint8, copy=False)

        if removed_u8.any():
            _or_write_mask_window(rm_band, x0, y0, win_w, win_h, removed_u8)

    # Finalize output: output = target & (~removed_union_mask)
    out_ds, out_band = _create_u8_gtiff(output_path, xsize, ysize, gt, proj, nodata=0)

    for y0 in range(0, ysize, block_y_finalize):
        ywin = min(block_y_finalize, ysize - y0)
        tgt_blk = tgt_band.ReadAsArray(0, y0, xsize, ywin)
        rm_blk  = rm_band.ReadAsArray(0, y0, xsize, ywin)
        if tgt_blk is None or rm_blk is None:
            raise RuntimeError("ReadAsArray returned None in finalize")

        tgt_u8 = (tgt_blk != 0).astype(np.uint8, copy=False)
        keep = (rm_blk == 0)
        out = (tgt_u8 & keep).astype(np.uint8, copy=False)
        out_band.WriteArray(out, 0, y0)

    out_band.FlushCache(); out_ds.FlushCache()
    rm_band.FlushCache(); rm_ds.FlushCache()

    out_band = None; out_ds = None
    rm_band = None; rm_ds = None

    # Debug std raster (LUT over labels -> no rectangle overwrites)
    if debug_mode:
        std_path = os.path.join(scratch_dir, "landcover_hand_std.tif")
        std_ds, std_band = _create_f32_gtiff(std_path, xsize, ysize, gt, proj, nodata=np.nan)

        for y0 in range(0, ysize, block_y_finalize):
            ywin = min(block_y_finalize, ysize - y0)
            lab_blk = labels[y0:y0 + ywin, :]  # int32 in RAM
            std_blk = hand_std_values[lab_blk].astype(np.float32, copy=False)
            std_band.WriteArray(std_blk, 0, y0)

        std_band.FlushCache(); std_ds.FlushCache()
        std_band = None; std_ds = None

    # Close inputs
    tgt_band = None; tgt_ds = None
    hand_band = None; hand_ds = None