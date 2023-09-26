import logging
import mimetypes
import os
import time

import cv2
import numpy as np

from dswx_sar import (dswx_sar_util,
                      generate_log,
                      masking_with_ancillary)
from dswx_sar.dswx_runconfig import _get_parser, RunConfig

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
    output= np.zeros(np.shape(values), dtype='float32')

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
    center_value = (minv + maxv) / 2
    output = np.zeros(np.shape(values))

    membership_left = 1 - 2 * ((values - minv) / (maxv - minv))**2
    mask_left = (values >= minv) & (values <= center_value)
    output[mask_left] = membership_left[mask_left]

    membership_right = 2 * ((values - maxv) / (maxv - minv))**2
    mask_right = (values >= center_value) & (values <= maxv)
    output[mask_right] = membership_right[mask_right]

    output[values >= maxv] = 0
    output[values <= minv] = 1

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
    excluded_area_ind = np.unique(output[binary_raster==0])
    sizes = stats[:, -1]
    sizes = np.delete(sizes, excluded_area_ind)
    nb_components = nb_components - 1

    old_val = np.arange(1, nb_components + 1) - 0.1
    kin = np.searchsorted(old_val, output)
    sizes =  np.insert(sizes, 0, 0, axis=0)
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
                        fuzzy_option):
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
    band_number, rows, cols = intensity.shape

    # fuzzy value for intensity for each polarization
    intensity_z_set = []
    initial_map = np.ones([rows, cols], dtype='byte')
    low_backscatter_cand = np.ones([rows, cols], dtype=bool)
    dark_water_cand = np.ones([rows, cols], dtype=bool)

    for int_id, pol  in enumerate(pol_list):

        thresh_valley_str = os.path.join(outputdir, f"int_threshold_filled_{pol}.tif")
        thresh_peak_str = os.path.join(outputdir, f"mode_tau_filled_{pol}.tif")

        valley_threshold_raster = dswx_sar_util.read_geotiff(thresh_valley_str)
        peak_threshold_raster = dswx_sar_util.read_geotiff(thresh_peak_str)

        intensity_band = np.squeeze(intensity[int_id, :, :])

        # Fuzzy membership computation from intensity
        # lower intensity is more likely to be water -> zmf
        temp = zmf(intensity_band,
                   peak_threshold_raster,
                   valley_threshold_raster)
        intensity_z_set.append(temp)

        intensity_mask_peak= intensity_band < peak_threshold_raster
        initial_map[intensity_mask_peak==0] = 0

        if pol in ['VH', 'HV']:
            pol_threshold = fuzzy_option['dark_area_land']
            water_threshold = fuzzy_option['dark_area_water']
            low_backscatter = (intensity[int_id, :, :] < pol_threshold) & \
                              (intensity[int_id, :, :] > water_threshold)
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
            (landcover == landcover_label['Shrubs'])| \
            (landcover == landcover_label['Grassland'])| \
            (landcover == landcover_label['Herbaceous wetland'])

    landcover_flat_area = (landcover_flat_area_cand) & \
                          (slope < 5) & \
                          (low_backscatter_cand)
    high_frequent_water = \
            (reference_water > fuzzy_option['high_frequent_water_min']) & \
            (reference_water < fuzzy_option['high_frequent_water_max']) & \
            (low_backscatter_cand)

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

        logger.info('Cross pol is replaced with co-pol and span '
                    'over the controversial area')
        # Cross-polarization intensity is replaced with co- (or span-) polarizations
        # where water varation is high and areas are dark/flat.
        intensity_z_set[cross_pol_ind][high_frequent_water] = \
            intensity_z_set[change_ind][high_frequent_water]
        intensity_z_set[cross_pol_ind][landcover_flat_area] = \
            intensity_z_set[change_ind][landcover_flat_area]

        # Co-polarization intensity is replaced with cross polarizations
        # where very dark water exists.
        intensity_z_set[change_ind][dark_water_cand] = \
            intensity_z_set[cross_pol_ind][dark_water_cand]

    copol_only = (high_frequent_water == 1) | \
                 (landcover_flat_area==1)

    # Compute sum of intensities.
    nansum_intensity_z_set = np.squeeze(np.nansum(intensity_z_set, axis=0))

    # Compute HAND membership
    logger.info('compute hand z membership')
    mu_h = np.nanmean(hand[initial_map == 1])
    std_h = np.nanstd(hand[initial_map == 1])
    min_dem = mu_h
    max_dem = mu_h + (std_h + 2.5) * std_h
    hand[np.isnan(hand)] = 0

    if np.isnan(min_dem):
        min_dem = fuzzy_option['hand_min']
    if np.isnan(max_dem):
        max_dem = fuzzy_option['hand_max']
    logger.info(f'     {min_dem} {max_dem} are used to compute HAND membership')

    hand_z = zmf(hand, min_dem, max_dem)

    # compute slope membership
    logger.info('compute slope z membership')
    logger.info(f"      {fuzzy_option['slope_min']} {fuzzy_option['slope_max']}"
                " are used to compute slope membership")
    slope_z = zmf(slope,
                  fuzzy_option['slope_min'],
                  fuzzy_option['slope_max'])

    # Compute area membership
    logger.info('area s membership')
    logger.info(f"      {fuzzy_option['area_min']} {fuzzy_option['area_max']}"
                " are used to compute area membership")
    handem = hand < fuzzy_option['hand_threshold']
    wbsmask = (initial_map == 1) & (handem)
    watermap = calculate_water_area(wbsmask)
    area_s = smf(watermap,
                fuzzy_option['area_min'],
                fuzzy_option['area_max'])

    # Reference water map membership
    logger.info('reference s membership')
    logger.info(f"      {fuzzy_option['reference_water_min']} {fuzzy_option['reference_water_max']}"
                " are used to compute reference water membership")
    reference_water_s = smf(reference_water,
                            fuzzy_option['reference_water_min'],
                            fuzzy_option['reference_water_max'])

    # Compute fuzzy-logic-based value
    # The Opera dswx s1 algorithm calculates fuzzy-logic-based values based on
    # input parameters, including intensity, hand, slope, and reference water.
    # The Twele algorithm, on the other hand, computes these values using
    # intensity, hand, slope, and areas.
    # It's important to note that half of the fuzzy values (0.5) are derived from
    # the intensity values, while the remaining half (0.5) comes from ancillary data.
    # To achieve this, the intensity membership is divided by the number of bands
    # and then multiplied by 0.5. The maximum value for intensity membership is
    # capped at 0.5. Similarly, the membership of the ancillary data contributes
    # 0.5 to the final result.
    method_dict = {
        'opera_dswx_s1': lambda: (nansum_intensity_z_set / (band_number) * 0.5 + \
                   (hand_z + slope_z + reference_water_s)  / 3 * 0.5),
        'twele': lambda: (nansum_intensity_z_set / (band_number + 1) * 0.5 + \
                   (hand_z + slope_z + area_s)  / 3 * 0.5)
    }
    avgvalue = method_dict[workflow]()
    mask = np.squeeze(np.nansum(intensity, axis=0)) == 0
    avgvalue[mask] = 0

    return avgvalue, intensity_z_set, hand_z, \
        slope_z, area_s, reference_water_s, copol_only


def run(cfg):
    '''
    Run fuzzy logic calculation with parameters in cfg dictionary
    '''
    t_all = time.time()

    outputdir = cfg.groups.product_path_group.scratch_path

    processing_cfg = cfg.groups.processing
    pol_list = processing_cfg.polarizations
    pol_all_str = '_'.join(pol_list)

    # reference water cfg
    water_cfg = processing_cfg.reference_water
    ref_water_max = water_cfg.max_value
    ref_no_data = water_cfg.no_data_value

    # fuzzy cfg
    fuzzy_cfg = processing_cfg.fuzzy_value
    option_dict = {'hand_threshold': fuzzy_cfg.hand.excluded_mask,
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
                   'high_frequent_water_min': fuzzy_cfg.high_frequent_water.water_min_value,
                   'high_frequent_water_max': fuzzy_cfg.high_frequent_water.water_max_value
    }

    workflow = processing_cfg.dswx_workflow

    # Read filtered RTC image
    filt_im_str = os.path.join(outputdir, f"filtered_image_{pol_all_str}.tif")
    im_meta = dswx_sar_util.get_meta_from_tif(filt_im_str)
    intensity = dswx_sar_util.read_geotiff(filt_im_str)
    if im_meta['band_number'] == 1:
        intensity = intensity[np.newaxis, :, :]

    mean_intensity = np.nanmean(intensity, axis=0)
    no_data_raster = np.isnan(mean_intensity)

    # Read Ancillary files
    dem_gdal_str = os.path.join(outputdir, 'interpolated_DEM')
    interpdem = dswx_sar_util.read_geotiff(dem_gdal_str)

    hand_gdal_str = os.path.join(outputdir, 'interpolated_hand')
    interphand = dswx_sar_util.read_geotiff(hand_gdal_str)

    landcover_gdal_str = os.path.join(outputdir, 'interpolated_landcover')
    landcover_map = dswx_sar_util.read_geotiff(landcover_gdal_str)
    landcover_label = masking_with_ancillary.get_label_landcover_esa_10()

    reference_water_gdal_str = os.path.join(outputdir, 'interpolated_wbd')
    wbd = dswx_sar_util.read_geotiff(reference_water_gdal_str)
    wbd = np.array(wbd, dtype='float32')
    wbd[wbd == ref_no_data] = np.nan
    # normalize water occurrence/seasonality value
    wbd = wbd / ref_water_max

    # Compute slope angle from DEM
    slope = compute_slope_dem(interpdem)

    # compute fuzzy value
    fuzzy_avgvalue, intensity_z, hand_z, slope_z, area_s, ref_water, copol_only = \
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
            workflow=workflow)

    fuzzy_avgvalue[interphand > option_dict['hand_threshold']] = 0
    fuzzy_avgvalue[no_data_raster] = -1

    # Output of Fuzzy_computation
    fuzzy_output_str = os.path.join(outputdir,
                                    f"fuzzy_image_{pol_all_str}.tif")

    dswx_sar_util.save_raster_gdal(
        data=fuzzy_avgvalue,
        output_file=fuzzy_output_str,
        geotransform=im_meta['geotransform'],
        projection=im_meta['projection'],
        scratch_dir=outputdir)

    if processing_cfg.debug_mode:

        rasters_to_save = [('hand_z', hand_z),
                           ('slope_z', slope_z),
                           ('area_s', area_s),
                           ('ref_water', ref_water),
                           ('copol_only', copol_only)]

        for raster_name, raster in rasters_to_save:
            output_file_name = os.path.join(outputdir, f"fuzzy_{raster_name}_{pol_all_str}.tif")
            dswx_sar_util.save_raster_gdal(
                data=raster,
                output_file=output_file_name,
                geotransform=im_meta['geotransform'],
                projection=im_meta['projection'],
                scratch_dir=outputdir)

        for polind, pol in enumerate(pol_list):
            dswx_sar_util.save_raster_gdal(
                data=np.squeeze(intensity_z[polind,:,:]),
                output_file=\
                    os.path.join(outputdir, f"fuzzy_intensity_{pol}.tif"),
                geotransform=im_meta['geotransform'],
                projection=im_meta['projection'],
                scratch_dir=outputdir)

    t_all_elapsed = time.time() - t_all
    logger.info(f"successfully ran fuzzy processing in {t_all_elapsed:.3f} seconds")


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
