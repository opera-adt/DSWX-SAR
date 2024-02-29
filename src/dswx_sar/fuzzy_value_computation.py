import copy
import logging
import mimetypes
import os
import time

import cv2
import numpy as np

from dswx_sar import (dswx_sar_util,
                      generate_log,
                      masking_with_ancillary)
from dswx_sar.dswx_runconfig import (DSWX_S1_POL_DICT,
                                     _get_parser,
                                     RunConfig)

logger = logging.getLogger('dswx_s1')

# Define constants
SOBEL_KERNEL_SIZE = 3
PIXEL_RESOLUTION_X = 30  # Replace with appropriate value
PIXEL_RESOLUTION_Y = 30  # Replace with appropriate value
RAD_TO_DEG = 180 / np.pi


def compute_slope_dem(dem):
    '''Calculate slope angle from DEM

    Parameters
    ----------
    dem : numpy.ndarray
        Dem raster image

    Returns
    -------
    sl : numpy.ndarray
        slope angle raster
    '''
    sobelx = cv2.Sobel(dem, cv2.CV_64F, 1, 0, ksize=SOBEL_KERNEL_SIZE)  # x
    sobely = cv2.Sobel(dem, cv2.CV_64F, 0, 1, ksize=SOBEL_KERNEL_SIZE)  # y

    # Compute slope
    slope_angle = np.arctan(np.sqrt(
        (sobelx / SOBEL_KERNEL_SIZE / PIXEL_RESOLUTION_X) ** 2 +
        (sobely / SOBEL_KERNEL_SIZE / PIXEL_RESOLUTION_Y) ** 2)) * RAD_TO_DEG

    return slope_angle


def create_slope_angle_geotiff(dem_path,
                               slope_path,
                               lines_per_block):
    """create Geotiff File for slope angle
    Parameters
    ----------
    dem_path: str
        GeoTiff path for filename to read input dem
    slope_path: str
        full path for filename to save the slope angle
    lines_per_block: int
        lines per block processing
    """
    pad_shape = (SOBEL_KERNEL_SIZE, 0)
    im_meta = dswx_sar_util.get_meta_from_tif(dem_path)
    scratch_dir = os.path.dirname(slope_path)

    block_params = dswx_sar_util.block_param_generator(
        lines_per_block=lines_per_block,
        data_shape=(im_meta['length'],
                    im_meta['width']),
        pad_shape=pad_shape)

    for block_param in block_params:
        dem = dswx_sar_util.get_raster_block(
            dem_path,
            block_param)
        slope = compute_slope_dem(dem)

        dswx_sar_util.write_raster_block(
            out_raster=slope_path,
            data=slope,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='float32',
            cog_flag=True,
            scratch_dir=scratch_dir)


def smf(values, minv, maxv):
    ''' Generate S-shape function for the given values

    Parameters
    ----------
    values : numpy.ndarray
        input value to be used for membership function
    minv : float
        minimum value for membership function
    maxv : float
        maximum value for membership function

    Returns
    -------
    output : numpy.ndarray
        rescaled value from s-shape membership function
    '''
    center_value = (minv + maxv) / 2
    output = np.zeros(np.shape(values), dtype='float32')

    # When using numpy arrays for min and max values in
    # a membership function, identical elements in these
    # arrays are replaced with a slightly higher number
    # to avoid zero-division warnings.
    if isinstance(minv, np.ndarray):
        diff = maxv - minv
        number_strange = np.nansum(diff <= 0)
        if number_strange > 0:
            logger.info(f'{number_strange} pixels have minimum values '
                        'larger than maximum values')
            minv[diff <= 0] = maxv[diff <= 0] - \
                dswx_sar_util.Constants.negligible_value

    membership_left = 2 * ((values - minv) / (maxv - minv))**2
    output[(values >= minv) & (values <= center_value)] = \
        membership_left[(values >= minv) & (values <= center_value)]
    membership_right = 1 - 2 * ((values - maxv) / (maxv-minv))**2
    output[(values >= center_value) & (values <= maxv)] = \
        membership_right[(values >= center_value) & (values <= maxv)]

    output[values <= minv] = 0
    output[values >= maxv] = 1

    return output


def zmf(values, minv, maxv):
    ''' Generate Z-shape function for the given values

    Parameters
    ----------
    values : numpy.ndarray
        input value to be used for membership function
    minv : float
        minimum value for membership function
    maxv : float
        maximum value for membership function

    Returns
    -------
    output : numpy.ndarray
        rescaled value from z-shape membership function
    '''
    output = np.zeros(np.shape(values))

    # When using numpy arrays for min and max values in
    # a membership function, identical elements in these
    # arrays are replaced with a slightly higher number
    # to avoid zero-division warnings.
    if isinstance(minv, np.ndarray):
        diff = maxv - minv
        number_strange = np.nansum(diff < 0)
        if number_strange > 0:
            logger.info(f'{number_strange} pixels have minimum values '
                        'larger than maximum values')
            minv[diff <= 0] = maxv[diff <= 0] - \
                dswx_sar_util.Constants.negligible_value

    center_value = (minv + maxv) / 2

    membership_left = 1 - 2 * ((values - minv) / (maxv - minv))**2
    mask_left = (values >= minv) & (values <= center_value)
    output[mask_left] = membership_left[mask_left]

    membership_right = 2 * ((values - maxv) / (maxv - minv))**2
    mask_right = (values >= center_value) & (values <= maxv)
    output[mask_right] = membership_right[mask_right]

    output[values >= maxv] = 0
    output[values <= minv] = 1
    output[np.isnan(values)] = np.nan

    return output


def calculate_water_area(binary_raster):
    '''Estimate areas of polygons extracted from binary_raster

    Parameters
    ----------
    binary_raster : numpy.ndarray
        binary raster containing only 0 and 1

    Returns
    -------
    size_raster : numpy.ndarray
        Each component in input binary raster is replaced with the size value
        of connected components
    '''
    nb_components, output, stats, _ \
        = cv2.connectedComponentsWithStats(
            np.array(binary_raster, dtype=np.uint8),
            connectivity=8)
    # Identify the labels of the areas not to provide the number of the pixels
    excluded_area_ind = np.unique(output[binary_raster == 0])
    sizes = stats[:, -1]
    sizes = np.delete(sizes, excluded_area_ind)
    nb_components = nb_components - 1

    old_val = np.arange(1, nb_components + 1) - 0.1
    kin = np.searchsorted(old_val, output)
    sizes = np.insert(sizes, 0, 0, axis=0)
    size_raster = sizes[kin]

    return size_raster


def compute_fuzzy_value(intensity,
                        slope,
                        hand,
                        landcover,
                        landcover_label,
                        reference_water,
                        pol_list,
                        outputdir,
                        workflow,
                        fuzzy_option,
                        block_param):
    '''Compute fuzzy values from intensity, dem,
    reference water, and HAND

    Parameters
    ----------
    intensity : numpy.ndarray
        The intensity or ratio layers.
        A number of layers could be arbitrary.
    slope : numpy.ndarray
        The slope angle raster
    hand : numpy.ndarray
        The Height Above Nearest Drainage (HAND) raster
    landcover : numpy.ndarray
        landcover raster
    landcover_label : dict
        dict consisting of landcover labels
    reference_water : numpy.ndarray
        reference water map.
        The values are expected from 0 to 1.
    pol_list : list
        list of the input polarizations
    outputdir : string
        output directory
    workflow : string
        workflows i.e.twele or opera_dswx_s1
    fuzzy_option : dict
        fuzzy options to compute the fuzzy values containing
        fllowing key:value parameters
            'hand_threshold': HAND value to mask out
            'slope_min': minimum value for z membership function for slope
            'slope_max': maximum value for z membership function for slope
            'area_min': minimum value for s membership function for area
            'area_max': maximum value for s membership function for area
            'reference_water_min':
                minimum value for s membership function for reference water
            'reference_water_max':
                maximum value for s membership function for reference water
            'dark_area_water':
                backscattering value in dB to define dark water
            'high_frequent_water_min':
                minimum value for the reference water to define
                area where water extent changes frequently
            'high_frequent_water_max':
                maximum value for the reference water to define
                area where water extent changes frequently
    block_param: BlockParam
        Object specifying size of block and where to read from raster,
        and amount of padding for the read array

    Returns
    -------
    avgvalue : numpy.ndarray
        fuzzy value layer considering
        the intensity and ancillary layers
    intensity_z_set : numpy.ndarray
        z-membership for intensity layer
    hand_z : numpy.ndarray
        z-membership for dem
    slope_z : numpy.ndarray
        z-membership for slope
    area_s : numpy.ndarray
        s-membership for area
    reference_water_s : numpy.ndarray
        s-membership for reference water
    copol_only : numpy.ndarray
       binary image showing the area where
       ony co-polization is used.
    '''
    _, rows, cols = intensity.shape

    # fuzzy value for intensity for each polarization
    intensity_z_set = []
    initial_map = np.ones([rows, cols], dtype='byte')
    low_backscatter_cand = np.ones([rows, cols], dtype=bool)
    dark_water_cand = np.ones([rows, cols], dtype=bool)

    for int_id, pol in enumerate(pol_list):

        thresh_valley_str = os.path.join(
            outputdir, f"intensity_threshold_filled_{pol}.tif")
        thresh_peak_str = os.path.join(outputdir, f"mode_tau_filled_{pol}.tif")

        valley_threshold_raster = dswx_sar_util.get_raster_block(
            thresh_valley_str, block_param)
        peak_threshold_raster = dswx_sar_util.get_raster_block(
            thresh_peak_str, block_param)

        intensity_band = intensity[int_id, :, :]

        # Fuzzy membership computation from intensity
        # lower intensity is more likely to be water -> zmf
        temp = zmf(intensity_band,
                   peak_threshold_raster,
                   valley_threshold_raster)
        intensity_z_set.append(temp)

        intensity_mask_peak = intensity_band < peak_threshold_raster
        initial_map[intensity_mask_peak == 0] = 0

        if pol in ['VH', 'HV']:
            pol_threshold = fuzzy_option['dark_area_land']
            water_threshold = fuzzy_option['dark_area_water']
            low_backscatter = (intensity[int_id, :, :] < pol_threshold)
            # Low backscattering candidates
            low_backscatter_cand &= low_backscatter
            dark_water_cand &= intensity[int_id, :, :] < water_threshold

    intensity_z_set = np.array(intensity_z_set)

    # Co-polarization is effective to detect the dry/flat surface from water
    # but water and dark lands are not distinguishable in cross-polarization
    # Here, we identify dry/flat area using slope/landcover/backscattering
    # and use only co-polarization instead of dual polarization.

    # Darkland candidate from landcover
    landcover_flat_area_cand = \
        (landcover == landcover_label['Bare sparse vegetation']) | \
        (landcover == landcover_label['Shrubs']) | \
        (landcover == landcover_label['Grassland']) | \
        (landcover == landcover_label['Herbaceous wetland'])

    landcover_flat_area = (landcover_flat_area_cand) & \
                          (slope < 5) & \
                          (low_backscatter_cand)
    high_frequent_water = \
        (reference_water > fuzzy_option['high_frequent_water_min']) & \
        (reference_water < fuzzy_option['high_frequent_water_max']) & \
        (low_backscatter_cand)
    dark_water = (dark_water_cand) & \
                 (reference_water >= fuzzy_option['high_frequent_water_max'])

    co_pol_ind = []
    cross_pol_ind = []

    # When dual-polarizations are available, cross-polarization intensity
    # is replaced by the co-polarization over the challenging areas.
    if ('HH' in pol_list and 'HV' in pol_list) or \
       ('VV' in pol_list and 'VH' in pol_list):
        for polindex, pol in enumerate(pol_list):
            if (pol == 'VV') | (pol == 'HH'):
                co_pol_ind = polindex
            elif (pol == 'VH') | (pol == 'HV'):
                cross_pol_ind = polindex
            elif pol == 'span':
                span_ind = polindex

        if 'span' in pol_list:
            change_ind = span_ind
        else:
            change_ind = co_pol_ind

        # Cross-polarization intensity is replaced with co- (or span-) pol
        # where water varation is high and areas are dark/flat.
        intensity_z_set[cross_pol_ind][high_frequent_water] = \
            intensity_z_set[change_ind][high_frequent_water]
        intensity_z_set[cross_pol_ind][landcover_flat_area] = \
            intensity_z_set[change_ind][landcover_flat_area]

        # Co-polarization intensity is replaced with cross polarizations
        # where very dark water exists.
        intensity_z_set[change_ind][dark_water] = \
            intensity_z_set[cross_pol_ind][dark_water]

    copol_only = (high_frequent_water == 1) | \
                 (landcover_flat_area == 1)

    # Compute mean of intensities.
    nanmean_intensity_z_set = np.nanmean(intensity_z_set, axis=0)

    # Compute HAND membership
    hand[np.isnan(hand)] = 0
    hand_z = zmf(hand, fuzzy_option['hand_min'], fuzzy_option['hand_max'])

    # compute slope membership
    slope_z = zmf(slope,
                  fuzzy_option['slope_min'],
                  fuzzy_option['slope_max'])

    # Compute area membership
    handem = hand < fuzzy_option['hand_threshold']
    wbsmask = (initial_map == 1) & (handem)
    watermap = calculate_water_area(wbsmask)
    area_s = smf(watermap,
                 fuzzy_option['area_min'],
                 fuzzy_option['area_max'])

    # Reference water map membership
    reference_water_s = smf(reference_water,
                            fuzzy_option['reference_water_min'],
                            fuzzy_option['reference_water_max'])

    # Compute fuzzy-logic-based value
    # The Opera dswx s1 algorithm calculates fuzzy-logic-based values based on
    # input parameters, including intensity, hand, slope, and reference water.
    # The Twele algorithm, on the other hand, computes these values using
    # intensity, hand, slope, and areas.
    # It's important to note that half of the fuzzy values (0.5) are derived
    # from the intensity values, while the remaining half (0.5) comes from
    # ancillary data. To achieve this, the intensity membership is divided
    # by the number of bands and then multiplied by 0.5. The maximum value
    # for intensity membership is capped at 0.5. Similarly, the membership
    # of the ancillary data contributes 0.5 to the final result.
    method_dict = {
        'opera_dswx_s1': lambda: (nanmean_intensity_z_set * 0.5
                                  + (hand_z + slope_z + reference_water_s)
                                  / 3 * 0.5),
        'twele': lambda: (nanmean_intensity_z_set * 0.5 +
                          (hand_z + slope_z + area_s) / 3 * 0.5)
    }
    avgvalue = method_dict[workflow]()

    return avgvalue, intensity_z_set, hand_z, \
        slope_z, area_s, reference_water_s, copol_only


def run(cfg):
    '''
    Run fuzzy logic calculation with parameters in cfg dictionary
    '''
    t_all = time.time()

    outputdir = cfg.groups.product_path_group.scratch_path

    processing_cfg = cfg.groups.processing
    pol_list = copy.deepcopy(processing_cfg.polarizations)
    pol_options = processing_cfg.polarimetric_option

    if pol_options is not None:
        pol_list += pol_options

    pol_all_str = '_'.join(pol_list)

    # reference water cfg
    water_cfg = processing_cfg.reference_water
    ref_water_max = water_cfg.max_value
    ref_no_data = water_cfg.no_data_value

    # fuzzy cfg
    fuzzy_cfg = processing_cfg.fuzzy_value
    lines_per_block = fuzzy_cfg.line_per_block
    option_dict = {'hand_threshold': processing_cfg.hand.mask_value,
                   'hand_min': fuzzy_cfg.hand.member_min,
                   'hand_max': fuzzy_cfg.hand.member_max,
                   'slope_min': fuzzy_cfg.slope.member_min,
                   'slope_max': fuzzy_cfg.slope.member_max,
                   'area_min': fuzzy_cfg.area.member_min,
                   'area_max': fuzzy_cfg.area.member_max,
                   'reference_water_min': fuzzy_cfg.reference_water.member_min,
                   'reference_water_max': fuzzy_cfg.reference_water.member_max,
                   'dark_area_land': fuzzy_cfg.dark_area.cross_land,
                   'dark_area_water': fuzzy_cfg.dark_area.cross_water,
                   'high_frequent_water_min':
                   fuzzy_cfg.high_frequent_water.water_min_value,
                   'high_frequent_water_max':
                   fuzzy_cfg.high_frequent_water.water_max_value
                   }

    workflow = processing_cfg.dswx_workflow

    logger.info('compute slope z membership')
    logger.info(f"      {option_dict['slope_min']} {option_dict['slope_max']}"
                " are used to compute slope membership")

    logger.info('reference s membership')
    logger.info(f"      {option_dict['reference_water_min']} "
                f"{option_dict['reference_water_max']}"
                " are used to compute reference water membership")

    logger.info('compute hand z membership')
    logger.info(f"     {option_dict['hand_min']} {option_dict['hand_max']}"
                " are used to compute HAND membership")

    logger.info('area s membership')
    logger.info(f"      {option_dict['area_min']} {option_dict['area_max']}"
                " are used to compute area membership")

    filt_im_str = os.path.join(outputdir, f"filtered_image_{pol_all_str}.tif")

    dem_gdal_str = os.path.join(outputdir, 'interpolated_DEM.tif')
    hand_gdal_str = os.path.join(outputdir, 'interpolated_hand.tif')
    landcover_gdal_str = os.path.join(outputdir, 'interpolated_landcover.tif')
    reference_water_gdal_str = os.path.join(outputdir, 'interpolated_wbd.tif')
    slope_gdal_str = os.path.join(outputdir, 'slope.tif')
    no_data_raster_path = os.path.join(
        outputdir,
        f"no_data_area_{pol_all_str}.tif")

    # Output of Fuzzy_computation
    fuzzy_output_str = os.path.join(
        outputdir,
        f"fuzzy_image_{pol_all_str}.tif")

    # read metadata including geotransform, projection, size
    im_meta = dswx_sar_util.get_meta_from_tif(filt_im_str)

    create_slope_angle_geotiff(
        dem_gdal_str,
        slope_gdal_str,
        lines_per_block=lines_per_block)

    landcover_label = masking_with_ancillary.get_label_landcover_esa_10()

    data_shape = [im_meta['length'], im_meta['width']]
    pad_shape = (0, 0)
    block_params = dswx_sar_util.block_param_generator(
        lines_per_block,
        data_shape,
        pad_shape)

    for block_ind, block_param in enumerate(block_params):
        logger.info(f'fuzzy logic computation block {block_ind}')
        intensity = dswx_sar_util.get_raster_block(
            filt_im_str, block_param)
        if im_meta['band_number'] == 1:
            intensity = intensity[np.newaxis, :, :]

        no_data_raster = dswx_sar_util.get_raster_block(
            no_data_raster_path, block_param)

        # Read Ancillary files
        interphand = dswx_sar_util.get_raster_block(
            hand_gdal_str, block_param)

        landcover_map = dswx_sar_util.get_raster_block(
            landcover_gdal_str, block_param)

        wbd = dswx_sar_util.get_raster_block(
            reference_water_gdal_str, block_param)
        wbd = np.array(wbd, dtype='float32')
        wbd[wbd == ref_no_data] = np.nan
        # normalize water occurrence/seasonality value
        wbd = wbd / ref_water_max

        # Compute slope angle from DEM
        slope = dswx_sar_util.get_raster_block(
            slope_gdal_str, block_param)

        # compute fuzzy value
        (fuzzy_avgvalue, intensity_z, hand_z,
         slope_z, area_s, ref_water, copol_only) = \
            compute_fuzzy_value(
                intensity=10*np.log10(intensity),
                slope=slope,
                hand=interphand,
                landcover=landcover_map,
                landcover_label=landcover_label,
                reference_water=wbd,
                pol_list=pol_list,
                outputdir=outputdir,
                fuzzy_option=option_dict,
                workflow=workflow,
                block_param=block_param)

        fuzzy_avgvalue[interphand > option_dict['hand_threshold']] = 0
        fuzzy_avgvalue[no_data_raster == 1] = -1

        dswx_sar_util.write_raster_block(
            out_raster=fuzzy_output_str,
            data=fuzzy_avgvalue,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='float32',
            cog_flag=True,
            scratch_dir=outputdir)

        if processing_cfg.debug_mode:

            rasters_to_save = [
                ('hand_z', hand_z),
                ('slope_z', slope_z),
                ('area_s', area_s),
                ('ref_water', ref_water),
                ('copol_only', copol_only)]

            for raster_name, raster in rasters_to_save:
                output_file_name = os.path.join(
                    outputdir, f"fuzzy_{raster_name}_{pol_all_str}.tif")
                dswx_sar_util.write_raster_block(
                    out_raster=output_file_name,
                    data=raster,
                    block_param=block_param,
                    geotransform=im_meta['geotransform'],
                    projection=im_meta['projection'],
                    datatype='float32',
                    cog_flag=True,
                    scratch_dir=outputdir)

            for polind, pol in enumerate(pol_list):
                dswx_sar_util.write_raster_block(
                    out_raster=os.path.join(
                        outputdir, f"fuzzy_intensity_{pol}.tif"),
                    data=np.reshape(intensity_z[polind, :, :],
                                    [intensity_z.shape[1],
                                     intensity_z.shape[2]]),
                    block_param=block_param,
                    geotransform=im_meta['geotransform'],
                    projection=im_meta['projection'],
                    datatype='float32',
                    cog_flag=True,
                    scratch_dir=outputdir)

    if processing_cfg.debug_mode:

        for raster_name, _ in rasters_to_save:
            filename = os.path.join(
                outputdir,
                f"fuzzy_{raster_name}_{pol_all_str}.tif")
            if not filename.endswith('.tif'):
                continue
            logger.info(f'    processing file: {filename}')
            dswx_sar_util._save_as_cog(
                filename,
                outputdir,
                logger,
                compression='DEFLATE',
                nbits=16)

        for pol in pol_list:
            filename = os.path.join(
                outputdir, f"fuzzy_intensity_{pol}.tif")
            dswx_sar_util._save_as_cog(
                filename,
                outputdir,
                logger,
                compression='DEFLATE',
                nbits=16)

    t_all_elapsed = time.time() - t_all
    logger.info("successfully ran fuzzy processing in "
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
