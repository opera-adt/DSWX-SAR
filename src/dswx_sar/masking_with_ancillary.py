import os
import numpy as np
import logging
import cv2
import rasterio
import time
import mimetypes

from scipy import ndimage
from skimage.filters import threshold_multiotsu
from rasterio.windows import Window
from joblib import Parallel, delayed
from typing import List, Tuple

from dswx_sar import dswx_sar_util
from dswx_sar import generate_log
from dswx_sar import refine_with_bimodality
from dswx_sar.dswx_runconfig import _get_parser, RunConfig


logger = logging.getLogger('dswx_S1')

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

class FillMaskLandcover:
    def __init__(self, landcover_file_path):

        self.landcover_file_str = landcover_file_path
        if not os.path.isfile(self.landcover_file_str):
            raise OSError(f"{self.landcover_file_str} is not found")

    def open_landcover(self):
        '''Open landcover map
        '''
        landcover_map = dswx_sar_util.read_geotiff(self.landcover_file_str)
        return landcover_map

    def get_mask(self, mask_label):
        '''Obtain areas corresponding to the givin labels from landcover map

        Parameters
        ----------
        mask_label : list[str]
            list of the label

        Returns
        -------
        mask : numpy.ndarray
            binary layers
        '''
        landcover = self.open_landcover()
        landcover_label = get_label_landcover_esa_10()
        mask = np.zeros(landcover.shape, dtype=bool)
        for label_name in mask_label:
            print(label_name)
            temp = landcover == landcover_label[f"{label_name}"]
            mask[temp] = True

        return mask


def extract_bbox_with_buffer(binary: np.ndarray,
                             buffer: int,
                             labelname: str,
                             outputdir: str,
                             meta_info: dict) -> Tuple[List[List[int]], np.ndarray]:
    """Extract bounding boxes with buffer from binary image and save the labeled image.

    Parameters:
        binary : np.ndarray
            Binary image containing connected components.
        buffer : int
            Buffer size to expand the bounding boxes.
        labelname : str
            Filename for saving the labeled image.
        outputdir : str
            Directory to save the labeled image.
        meta_info : dict
            Metadata information including geotransform and projection.

    Returns:
        coord_list : Tuple[List[List[int]]
            A tuple containing the list of coordinates for each bounding box and the sizes of the connected components.
        sizes :  np.ndarray
            Sizes of the connected components.
    """
    rows, cols = binary.shape

    nb_components_water, output_water, stats_water, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    nb_components_water -= 1

    water_label_str = os.path.join(outputdir, labelname)
    dswx_sar_util.save_raster_gdal(
                    data=output_water,
                    output_file=water_label_str,
                    geotransform=meta_info['geotransform'],
                    projection=meta_info['projection'],
                    scratch_dir=outputdir)

    sizes = stats_water[1:, -1]
    bboxes = stats_water[1:, :4]

    coord_list = []
    for i, (x, y, w, h) in enumerate(bboxes):
        add_iter = int((np.sqrt(2) - 1.2) * np.sqrt(sizes[i]))
        if add_iter == 0:
            add_iter = 1

        sub_x_start = max(0, x - add_iter - buffer)
        sub_y_start = max(0, y - add_iter - buffer)
        sub_x_end = min(cols, x + w + add_iter + buffer)
        sub_y_end = min(rows, y + h + add_iter + buffer)

        coord_list.append([sub_x_start, sub_x_end, sub_y_start, sub_y_end])

    return coord_list, sizes


def check_water_land_mixture_parallel(args):
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

    if size > minimum_pixel:
        target_water = water_label == i+1

        intensity_array = int_linear[target_water]
        invalid_mask = (np.isnan(intensity_array) | (intensity_array==0))

        # check if the area has bimodality
        metric_obj = refine_with_bimodality.BimodalityMetrics(intensity_array)
        test_output = metric_obj.compute_metric()

        out_boundary = (np.isnan(int_linear) == 0) & (water_label == 0)

        if test_output:
            print('landcover: found bimodality', bounds)
            out_boundary_sub = np.copy(out_boundary)
            out_boundary_sub[target_water] = 1

            int_db_sub = 10 * np.log10(intensity_array[np.invert(invalid_mask)])
            int_db = 10 * np.log10(int_linear)
            threshold_local_otsu = threshold_multiotsu(int_db_sub, nbins=100)
            for t_ind, threshold in enumerate(threshold_local_otsu):

                # assume that the test dark area has trimodal distribution
                if t_ind == 0:
                    # area is lower than threshold and belongs to target water
                    dark_water = (int_db < threshold) & (target_water)
                else:
                    dark_water = (int_db < threshold) & \
                                 (int_db >= threshold_local_otsu[t_ind-1]) & \
                                 (target_water)

                newsize = np.count_nonzero(dark_water)
                add_iter = int((np.sqrt(2)-1.2)*np.sqrt(newsize))
                dark_mask_buffer = ndimage.binary_dilation(dark_water,
                                                        iterations=add_iter,
                                                        mask=out_boundary_sub)
                dark_water_linear = int_linear[dark_mask_buffer]
                hist_min = np.percentile(10*np.log10(dark_water_linear), 1)
                hist_max = np.percentile(10*np.log10(dark_water_linear), 99)

                metric_obj_local = refine_with_bimodality.BimodalityMetrics(
                    dark_water_linear,
                    hist_min=hist_min,
                    hist_max=hist_max)

                if not metric_obj_local.compute_metric(ashman_flag=True,
                                                        bhc_flag=True,
                                                        bm_flag=True,
                                                        surface_ratio_flag=True,
                                                        bc_flag=False):
                    water_mask[dark_water] = 0
                    change_flag = True

                if t_ind == len(threshold_local_otsu)-1:
                    bright_water = (int_db >= threshold) & (target_water)
                else:
                    bright_water = (int_db >= threshold) & \
                                   (int_db < threshold_local_otsu[t_ind+1]) & \
                                   (target_water)

                newsize = np.count_nonzero(bright_water)
                add_iter = int((np.sqrt(2)-1.2)*np.sqrt(newsize))
                bright_mask_buffer = ndimage.binary_dilation(bright_water,
                                                        iterations=add_iter,
                                                        mask=out_boundary_sub)
                bright_water_linear = int_linear[bright_mask_buffer]
                bright_water_linear[bright_water_linear==0] = np.nan
                hist_min = np.percentile(10*np.log10(bright_water_linear), 2)
                hist_max = np.percentile(10*np.log10(bright_water_linear), 98)
                metric_obj_local = refine_with_bimodality.BimodalityMetrics(
                    bright_water_linear,
                    hist_min=hist_min,
                    hist_max=hist_max)

                if not metric_obj_local.compute_metric(ashman_flag=True,
                    bhc_flag=True,
                    bm_flag=True,
                    surface_ratio_flag=True,
                    bc_flag=False):
                    water_mask[bright_water] = 0
                    change_flag = True

    return i, change_flag, water_mask


def split_extended_water(water_mask: np.ndarray,
                         pol_ind: int,
                         outputdir: str,
                         meta_info: dict,
                         input_dict: dict) -> np.ndarray:
    """Split extended water areas into smaller subsets based on bounding boxes with buffer.

    Parameters:
        water_mask (np.ndarray): Binary mask representing the water areas.
        pol_ind (int): Index of the polarization.
        outputdir (str): Directory to save the resulting water mask.
        meta_info (dict): Metadata information including geotransform and projection.
        input_dict (dict): Dictionary containing input data including intensity and water mask.

    Returns:
        np.ndarray: The updated water mask with extended water areas split into smaller subsets.
    """


    water_label_filename =  'water_label_landcover.tif'
    water_label_str = os.path.join(outputdir, water_label_filename)

    coord_list, sizes = extract_bbox_with_buffer(binary=water_mask,
                                                 buffer=10,
                                                 labelname=water_label_filename,
                                                 outputdir=outputdir,
                                                 meta_info=meta_info)

    # check if the individual water body candidates has bimodality
    # bright water vs dark water
    # water vs land
    minimum_pixel = 5000
    args_list = [(i, sizes[i], minimum_pixel, coord_list[i],
                  input_dict['intensity'],
                  water_label_str,
                  input_dict['water_mask'],
                  pol_ind) for i in range(0, len(sizes))]

    results = Parallel(n_jobs=10)(delayed(check_water_land_mixture_parallel)(args) for args in args_list)

    # If water need to be refined (change_flat=True), then update the water mask.
    for i, change_flag, water_mask_subset in results:
        if change_flag:
            water_mask[coord_list[i][2] : coord_list[i][3],
                       coord_list[i][0] : coord_list[i][1]] = water_mask_subset

    return water_mask

def compute_portion_from_ancillary(water_mask: np.ndarray,
                                   ref_water: np.ndarray,
                                   mask_landcover: np.ndarray,
                                   outputdir: str,
                                   meta_info: dict) -> np.ndarray:
    """Compute portion of water areas based on ancillary information.

    Parameters:
        water_mask (np.ndarray): Binary mask representing the water areas.
        ref_water (np.ndarray): Reference water information.
        mask_landcover (np.ndarray): Mask representing landcover areas.
        outputdir (str): Directory to save the resulting water mask.
        meta_info (dict): Metadata information including geotransform and projection.

    Returns:
        np.ndarray: The computed mask representing the portion of water areas.
    """

    water_label_filename =  'water_label_landcover.tif'
    water_label_str = os.path.join(outputdir, water_label_filename)

    coord_list, sizes = extract_bbox_with_buffer(binary=water_mask,
                                                 buffer=10,
                                                 labelname=water_label_filename,
                                                 outputdir=outputdir,
                                                 meta_info=meta_info)

    nb_components_water, output_water, _, _ = \
        cv2.connectedComponentsWithStats(water_mask.astype(np.uint8), connectivity=8)

    nb_components_water -= 1
    test_output = np.zeros(len(sizes))

    mask_excluded_ancillary = np.logical_and(mask_landcover, ref_water < 0.05)

    ref_land_tif_str = os.path.join(outputdir, "maskedout_ancillary_2_landcover.tif")
    dswx_sar_util.save_dswx_product(mask_excluded_ancillary.astype(np.uint8),
                                    ref_land_tif_str,
                                    geotransform=meta_info['geotransform'],
                                    projection=meta_info['projection'],
                                    description='Water classification (WTR)',
                                    scratch_dir=outputdir)
    threshold = 0.5
    args_list = [(i, sizes[i], threshold, coord_list[i], water_label_str, ref_land_tif_str)
                 for i in range(len(sizes))]

    results = Parallel(n_jobs=10)(delayed(compute_portion)(args) for args in args_list)
    for i, test_output_i in results:
        test_output[i] = test_output_i

    old_val = np.arange(1, nb_components_water + 1) - 0.1
    kin = np.searchsorted(old_val, output_water)
    test_output = np.insert(test_output, 0, 0, axis=0)
    mask_water_image = test_output[kin]

    return mask_water_image

def compute_portion(args):
    """Compute the portion of water areas based on given arguments.

    Parameters:
        args (tuple): Tuple containing the following arguments:
            i (int): Index of the current iteration.
            size (int): Size of the current component.
            threshold (float): Threshold value for comparison.
            bounds (list): List containing the bounds of the subset.
            water_label_str (str): Path to the water label file.
            ref_land_tif_str (str): Path to the reference land file.

    Returns:
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
    ref_no_data = water_cfg.no_data_value

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
    mask_obj = FillMaskLandcover(landcover_path_str)
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
            pol_threshold = landcover_cfg.vv_threshold
        elif pol in ['VH', 'HV']:
            pol_threshold = landcover_cfg.vh_threshold
        else:
            continue  # Skip unknown polarizations
        low_backscatter = 10 * np.log10(band_set[pol_ind, :, :]) < pol_threshold
        low_backscatter_cand = np.logical_and(low_backscatter, low_backscatter_cand)

    # 2) Create intial mask binary
    # mask = intersect of landcover mask + no water + dark land
    mask_excluded = (mask_excluded_landcover) & \
                    (interp_wbd / ref_water_max < 0.05) & \
                    (low_backscatter_cand)

    # 3) split dark areas into pieces andif the distribution is close to the bimodal
    input_map = np.where(water_map == 1, 1, 0)
    split_mask_water_raster = split_extended_water(
                                input_map,
                                co_pol_ind,
                                outputdir,
                                water_meta,
                                input_file_dict)

    # re-define false water candidate from intersecting areas
    # between original bindary water and dark water bodies
    false_water_cand = (split_mask_water_raster == 0) & (water_map==1)

    mask_water_image = compute_portion_from_ancillary(
                    false_water_cand,
                    ref_water=interp_wbd/ref_water_max,
                    mask_landcover=mask_excluded_landcover,
                    outputdir=outputdir,
                    meta_info=water_meta)

    ## The landcover only captures standing water and does not have fuzziess on
    # backscattering, so there may be mismatched areas compared to the water map.
    # mask_water_image indicates the

    # mask_excluded indicates the areas satisfying all conditions which are
    # 1: 5 % of water occurrence over 37 year (Pekel)
    # 2: specified landcovers (bare ground, sparse vegetation, urban, moss...)
    # 3: low backscattering area
    # dark_land = (split_mask_water_raster == 0) | mask_excluded
    dark_land = (mask_water_image == 1) | mask_excluded

    water_map[dark_land] = 0

    # areas where pixeles are "no_data" in both reference water map and landcover
    landcover_nodata = mask_obj.get_mask(mask_label=['No_data'])
    no_data_area = (interp_wbd == ref_no_data) & (landcover_nodata)

    water_tif_str = os.path.join(outputdir, f"refine_landcover_binary_{pol_str}.tif")
    dswx_sar_util.save_dswx_product(water_map,
                  water_tif_str,
                  geotransform=water_meta['geotransform'],
                  projection=water_meta['projection'],
                  description='Water classification (WTR)',
                  scratch_dir=outputdir,
                  dark_land=dark_land,
                  no_data=no_data_area)

    if processing_cfg.debug_mode:

        darkland_tif_str = os.path.join(outputdir, f"landcover_dark_land_binary_{pol_str}.tif")
        dswx_sar_util.save_dswx_product(mask_excluded,
                  darkland_tif_str,
                  geotransform=water_meta['geotransform'],
                  projection=water_meta['projection'],
                  description='Water classification (WTR)',
                  scratch_dir=outputdir
                  )

        test_tif_str = os.path.join(outputdir, f"test_land_binary_{pol_str}.tif")
        dswx_sar_util.save_dswx_product(np.array(false_water_cand, dtype=np.uint8),
                    test_tif_str,
                    geotransform=water_meta['geotransform'],
                    projection=water_meta['projection'],
                    description='Water classification (WTR)',
                    scratch_dir=outputdir
                    )

        dswx_sar_util.binary_display(water_map, outputdir, 'landcover_mask_out')

        output_str = os.path.join(
            outputdir, 'refine_landcover_comparison_{}.tif'.format(pol_str))
        dswx_sar_util.WaterBinary_comparison_ConvertTiff(
            water_map,
            interp_wbd/ref_water_max>=0.95,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            output_tiff_str=output_str)

        darkland_str = os.path.join(outputdir, f"split_dark_land_candidate_{pol_str}.tif")
        dswx_sar_util.save_dswx_product(split_mask_water_raster,
                  darkland_str,
                  geotransform=water_meta['geotransform'],
                  projection=water_meta['projection'],
                  description='Darkland candidates',
                  scratch_dir=outputdir,
                  dark_land=dark_land)

        darkland_str = os.path.join(outputdir, f"dark_land_50_candidate_{pol_str}.tif")
        dswx_sar_util.save_dswx_product(mask_water_image,
                  darkland_str,
                  geotransform=water_meta['geotransform'],
                  projection=water_meta['projection'],
                  description='Darkland candidates',
                  scratch_dir=outputdir,
                  dark_land=dark_land)

        test_tif_str = os.path.join(outputdir, f"test_land_binary_{pol_str}.tif")
        dswx_sar_util.save_dswx_product(np.array(false_water_cand, dtype=np.uint8),
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

