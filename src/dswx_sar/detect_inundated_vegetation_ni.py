import copy
import logging
import mimetypes
import os
import time

import numpy as np

from dswx_sar import dswx_sar_util, filter_SAR, generate_log
from dswx_sar.dswx_ni_runconfig import DSWX_NI_POL_DICT, _get_parser, RunConfig
from dswx_sar.pre_processing import pol_ratio
from dswx_sar.masking_with_ancillary import FillMaskLandCover

logger = logging.getLogger('dswx_sar')


def parse_ranges(ranges):
    """
    Parse a list of ranges in the format "start-end" and single numbers.

    Parameters
    ----------
    ranges : list of str
        A list of strings where each string is either a single number or a
        range in the format "start-end".

    Returns
    -------
    result : list of int
        A list of integers that includes all numbers in the specified ranges
        and single numbers.

    Examples
    --------
    >>> parse_ranges(["1-3", "5", "7-9"])
    [1, 2, 3, 5, 7, 8, 9]

    >>> parse_ranges(["10", "12-15", "18"])
    [10, 12, 13, 14, 15, 18]
    """
    result = []
    for item in ranges:
        if '-' in item:
            start, end = map(int, item.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(item))
    return result

from scipy.optimize import least_squares

def residuals(params, H, obs):
    sigma_model = water_cloud_model(H, *params)
    return sigma_model - obs  # normal residuals

def huber_loss(residual, c=1.345):
    # c is the Huber parameter
    abs_r = np.abs(residual)
    is_small = abs_r <= c
    # Piecewise definition
    huber = np.where(is_small, 0.5 * residual**2, c * (abs_r - 0.5 * c))
    return huber

def cost_function(params, H, obs):
    res = residuals(params, H, obs)
    return huber_loss(res)

def cost_function_vectorized(params, H, obs):
    # We need to return a vector, so the sum or average won't help directly.
    res = residuals(params, H, obs)
    # Return sqrt of the pointwise robust cost, so least_squares can sum squares
    return np.sqrt(huber_loss(res))


def water_cloud_model_dB(H, sigma0_g_dB, sigma0_v_dB, k, a, b):
    # Convert dB inputs to linear
    sigma0_g_lin = 10**(sigma0_g_dB / 10.0)
    sigma0_v_lin = 10**(sigma0_v_dB / 10.0)
    V = a * (H**b)
    sigma0_lin = (sigma0_g_lin * np.exp(-2*k*V) +
                  sigma0_v_lin * (1 - np.exp(-2*k*V)))
    # Convert output back to dB
    return 10 * np.log10(sigma0_lin)


def water_cloud_model(H, sigma0_g, sigma0_v, k, a, b):
    # a simple example with an allometric relationship: V = a * H^b
    V = a * (H**b)
    return sigma0_g * np.exp(-2 * k * V) + sigma0_v * (1 - np.exp(-2 * k * V))


def pow2db(value):
    return 10 * np.log10(value)

def db2pow(value):
    return 10 ** (value / 10)


def estimate_water_cloud_parameter(observation,
                                   forest_parameter,
                                   initial_guess,
                                   lower_bounds,
                                   upper_bounds):
    from scipy.optimize import curve_fit

    mask = ~ np.isnan(observation)
    observation = observation[mask]
    forest_parameter = np.array(forest_parameter)
    forest_parameter = forest_parameter[mask]
    # popt, pcov = curve_fit(water_cloud_model,
    #                        forest_parameter,
    #                        observation,
    #                        p0=initial_guess)

    res = least_squares(
        cost_function_vectorized,
        x0=initial_guess,
        args=(forest_parameter, observation),
        bounds=(lower_bounds, upper_bounds)
        )
    print(res)
    popt_robust = res.x
    return popt_robust
    # return popt

def detect_iv_with_water_cloud(
        co_pol_image,
        cross_pol_image,
        vegetation_param_image,
        dry_forest_mask,
        wet_forest_mask,
        initial_guess_co_pol,
        initial_guess_cross_pol,
        lower_bounds_co,
        upper_bounds_co,
        lower_bounds_cross,
        upper_bounds_cross,
        ratio_threshold
):
    valid_mask = np.invert(np.isnan(co_pol_image))
    vegetation_param_image[np.isnan(co_pol_image)] = np.nan
    ratio = pow2db(co_pol_image / cross_pol_image)
    print(np.sum(dry_forest_mask))
    print(np.sum(wet_forest_mask))
    print(pow2db(np.nanmin(co_pol_image[dry_forest_mask])))
    print(pow2db(np.nanmax(co_pol_image[dry_forest_mask])))
    print(pow2db(np.nanmin(co_pol_image[dry_forest_mask])))
    print(pow2db(np.nanmax(co_pol_image[dry_forest_mask])))

    print(np.nanmin(co_pol_image[wet_forest_mask]))
    print(np.nanmax(co_pol_image[wet_forest_mask]))
    lowest_dry_forest = np.nanmin(vegetation_param_image[dry_forest_mask])
    highest_dry_forest = np.nanmax(vegetation_param_image[dry_forest_mask])

    print(pow2db(initial_guess_co_pol))
    # initial_guess_co_pol[0] = np.nanmean(co_pol_image[
    #     dry_forest_mask &
    #     (vegetation_param_image <= lowest_dry_forest + 1)])
    # initial_guess_co_pol[1] = np.nanmean(co_pol_image[
    #     dry_forest_mask &
    #     (vegetation_param_image >= highest_dry_forest - 1)])
    print(pow2db(initial_guess_co_pol))
    print(pow2db(lower_bounds_co))
    print(pow2db(upper_bounds_co))

    co_pol_parameter = estimate_water_cloud_parameter(
        co_pol_image[dry_forest_mask],
        vegetation_param_image[dry_forest_mask],
        initial_guess=initial_guess_co_pol,
        lower_bounds=lower_bounds_co,
        upper_bounds=upper_bounds_co,
        )

    cross_pol_parameter = estimate_water_cloud_parameter(
        cross_pol_image[dry_forest_mask],
        vegetation_param_image[dry_forest_mask],
        initial_guess = initial_guess_cross_pol,
        lower_bounds=lower_bounds_cross,
        upper_bounds=upper_bounds_cross,
        )

    simul_co_pol = water_cloud_model(
        vegetation_param_image,
        co_pol_parameter[0],
        co_pol_parameter[1],
        co_pol_parameter[2],
        co_pol_parameter[3],
        co_pol_parameter[4])

    simul_x_pol = water_cloud_model(
        vegetation_param_image,
        cross_pol_parameter[0],
        cross_pol_parameter[1],
        cross_pol_parameter[2],
        cross_pol_parameter[3],
        cross_pol_parameter[4])

    inundated_vegetation = ratio - \
        pow2db(simul_co_pol / simul_x_pol) > ratio_threshold
    inundated_vegetation[np.invert(wet_forest_mask)] = 0

    return inundated_vegetation, simul_co_pol, simul_x_pol, pow2db(simul_co_pol / simul_x_pol)


def run(cfg):

    logger.info('Start inundated vegetation mapping')

    t_all = time.time()

    processing_cfg = cfg.groups.processing
    scratch_dir = cfg.groups.product_path_group.scratch_path
    pol_list = copy.deepcopy(processing_cfg.polarizations)
    pol_options = processing_cfg.polarimetric_option

    if pol_options is not None:
        pol_list += pol_options

    pol_all_str = '_'.join(pol_list)

    inundated_vege_cfg = processing_cfg.inundated_vegetation
    inundated_vege_cross_pol_min = inundated_vege_cfg.cross_pol_min
    inundated_vege_method = inundated_vege_cfg.inundated_vegetation_method

    target_file_type = inundated_vege_cfg.target_area_file_type
    target_worldcover_class = inundated_vege_cfg.target_worldcover_class
    target_glad_class = inundated_vege_cfg.target_glad_class

    line_per_block = inundated_vege_cfg.line_per_block
    filter_options = inundated_vege_cfg.filter
    filter_method = inundated_vege_cfg.filter.method

    if inundated_vege_method == 'dual_pol_ratio':
        dual_pol_options_cfg = inundated_vege_cfg.dual_pol_ratio_options
        inundated_vege_ratio_max = dual_pol_options_cfg.dual_pol_ratio_max
        inundated_vege_ratio_min = dual_pol_options_cfg.dual_pol_ratio_min
        inundated_vege_ratio_threshold = \
            dual_pol_options_cfg.dual_pol_ratio_threshold
    elif inundated_vege_method == 'water_cloud':
        water_cloud_options_cfg = inundated_vege_cfg.water_cloud_model_options
        ratio_diff_threshold = water_cloud_options_cfg.dual_pol_ratio_difference_threshold
        height_data_type = water_cloud_options_cfg.height_data_type
    else:
        err_str = f"Inundated Vegetation Method {inundated_vege_method}' is not supported"
        raise ValueError(err_str)

    interp_glad_path_str = os.path.join(
        scratch_dir,
        'interpolated_glad.tif')
    interp_worldcover_path_str = os.path.join(
        scratch_dir,
        'interpolated_landcover.tif')
    interp_eth_path_str = os.path.join(
        scratch_dir,
        'interpolated_eth.tif')

    print(processing_cfg.debug_mode)
    if target_file_type == 'auto':
        if os.path.exists(interp_glad_path_str):
            target_file_type = 'GLAD'
        else:
            target_file_type = 'WorldCover'
    logger.info(f'Vegetation area is extracted from {target_file_type}.')

    # Currently, inundated vegetation for C-band is available for
    # Potential wetland area from Land cover maps
    if target_file_type == 'WorldCover':
        landcover_path_str = interp_worldcover_path_str
    else:
        landcover_path_str = interp_glad_path_str
        sup_mask_obj = FillMaskLandCover(interp_worldcover_path_str,
                                         'WorldCover')
    mask_obj = FillMaskLandCover(landcover_path_str, target_file_type)
    inundated_vege_path = \
        f"{scratch_dir}/temp_inundated_vegetation_{pol_all_str}.tif"
    target_area_path = \
        f"{scratch_dir}/temp_target_area_{pol_all_str}.tif"
    high_ratio_path = \
        f"{scratch_dir}/temp_high_dualpol_ratio_{pol_all_str}.tif"

    dual_pol_flag = False
    if (('HH' in pol_list) and ('HV' in pol_list)) or \
       (('VV' in pol_list) and ('VH' in pol_list)):
        dual_pol_flag = True

    if inundated_vege_cfg.enabled == 'auto':
        if dual_pol_flag:
            inundated_vege_cfg_flag = True
        else:
            inundated_vege_cfg_flag = False
    else:
        inundated_vege_cfg_flag = inundated_vege_cfg.enabled

    if inundated_vege_cfg_flag and not dual_pol_flag:
        err_str = 'Daul polarizations are required for inundated vegetation'
        raise ValueError(err_str)

    for polind, pol in enumerate(pol_list):
        if pol in ['HH', 'VV']:
            copol_ind = polind
        elif pol in ['HV', 'VH']:
            crosspol_ind = polind

    rtc_dual_path = f"{scratch_dir}/filtered_image_{pol_all_str}.tif"
    if not os.path.isfile(rtc_dual_path):
        err_str = f'{rtc_dual_path} is not found.'
        raise FileExistsError(err_str)

    if inundated_vege_method == 'dual_pol_ratio':
       if (inundated_vege_ratio_min > inundated_vege_ratio_threshold) or \
          (inundated_vege_ratio_max < inundated_vege_ratio_threshold):
            err_str = f'{inundated_vege_ratio_threshold} is not valid.'
            raise ValueError(err_str)

    im_meta = dswx_sar_util.get_meta_from_tif(rtc_dual_path)

    pad_shape = (filter_options.block_pad, 0)
    block_params = dswx_sar_util.block_param_generator(
        lines_per_block=line_per_block,
        data_shape=(im_meta['length'],
                    im_meta['width']),
        pad_shape=pad_shape)

    for block_param in block_params:

        rtc_dual = dswx_sar_util.get_raster_block(
            rtc_dual_path,
            block_param)
        co_pol_image = np.squeeze(rtc_dual[copol_ind, :, :])
        cross_pol_image = np.squeeze(rtc_dual[crosspol_ind, :, :])

        rtc_ratio = pol_ratio(
            co_pol_image,
            cross_pol_image)

        if filter_method == 'lee':
            filtering_method = filter_SAR.lee_enhanced_filter
            filter_option = vars(filter_options.lee_filter)

        elif filter_method == 'anisotropic_diffusion':
            filtering_method = filter_SAR.anisotropic_diffusion
            filter_option = vars(filter_options.anisotropic_diffusion)

        elif filter_method == 'guided_filter':
            filtering_method = filter_SAR.guided_filter
            filter_option = vars(filter_options.guided_filter)

        elif filter_method == 'bregman':
            filtering_method = filter_SAR.tv_bregman
            filter_option = vars(filter_options.bregman)

        filt_ratio = filtering_method(
                        rtc_ratio, **filter_option)
        filt_ratio_db = pow2db(
            filt_ratio + dswx_sar_util.Constants.negligible_value)

        filt_cross = filtering_method(
                np.squeeze(cross_pol_image), **filter_option)
        filtered_cross_db = pow2db(
            filt_cross + dswx_sar_util.Constants.negligible_value)

        if inundated_vege_method == 'water_cloud':
            filt_co = filtering_method(
                np.squeeze(co_pol_image), **filter_option)
            filtered_co_db =pow2db(
                filt_co + dswx_sar_util.Constants.negligible_value)

        output_data = np.zeros(filt_ratio.shape, dtype='uint8')

        target_cross_pol = filtered_cross_db > inundated_vege_cross_pol_min
        if target_file_type == 'WorldCover':
            target_inundated_vege_class = mask_obj.get_mask(
                mask_label=target_worldcover_class,
                block_param=block_param)
        elif target_file_type == 'GLAD':
            inundated_vege_target = parse_ranges(target_glad_class)
            target_inundated_vege_class = mask_obj.get_mask(
                mask_label=inundated_vege_target,
                block_param=block_param)

            # GLAD has no-data values for small island and polar regions
            # such as Greenland. The WorldCover will be alternatively used
            # for the no-data areas.
            glad_no_data = mask_obj.get_mask(
                mask_label=[255],
                block_param=block_param)
           
            logger.info(f'GLAD has {np.sum(glad_no_data)} no data')
            target_replace_class = sup_mask_obj.get_mask(
                mask_label=target_worldcover_class,
                block_param=block_param)
            target_inundated_vege_class = np.array(target_inundated_vege_class,
                                                   dtype='int8')
            target_inundated_vege_class[
                glad_no_data & target_replace_class] = 2
                # target_replace_class[glad_no_data]

        no_data = np.isnan(filt_ratio)
        target_inundated_vege_class[no_data] = 0
        print(target_file_type)
        if 'dual_pol_ratio' in inundated_vege_method:
            all_inundated_cand = \
                (filt_ratio_db > inundated_vege_ratio_threshold) & \
                target_cross_pol

        elif 'water_cloud' in inundated_vege_method:
            # Methods for forest and short vegetation are different.
            # For the forest, we apply the water-cloud model for the dry forest,
            # and simulate the backscattering for forest height.
            # We compute the expect dual-pol ratio and compare it with observed ratio.
            # If forest show higher than the expected ratio value, then those areas are
            # defined as inundated forest

            glad_height = mask_obj.get_class(block_param=block_param)

            # Inundated Forest
            if height_data_type == 'GLAD':
                vege_param_image = np.zeros(glad_height.shape)
                wet_forest_mask = np.logical_and(glad_height >= 125,
                                                    glad_height <= 148)
                dry_forest_mask = np.logical_and(glad_height >= 25,
                                                    glad_height <= 48)

                # class 125 represents the 3 m tree
                vege_param_image[wet_forest_mask] = \
                    glad_height[wet_forest_mask] \
                    - 125 + 3
                vege_param_image[dry_forest_mask] = \
                    glad_height[dry_forest_mask] - 22 + 3

            elif height_data_type == 'ETH_Global_Height':
                wet_forest_mask = np.logical_and(
                    glad_height >= 100,
                    glad_height <= 148)
                dry_forest_mask = np.logical_and(
                    glad_height >= 0,
                    glad_height <= 48)
                vege_param_image = dswx_sar_util.get_raster_block(
                    interp_eth_path_str,
                    block_param)
 
            (all_inundated_cand_forest,
            simul_forest_co_pol,
            simul_forest_x_pol,
            simul_forest_ratio) = detect_iv_with_water_cloud(
                co_pol_image=co_pol_image,
                cross_pol_image=cross_pol_image,
                vegetation_param_image=vege_param_image,
                dry_forest_mask=dry_forest_mask,
                wet_forest_mask=wet_forest_mask,
                initial_guess_co_pol=[0.141, 0.5, 0.2, 0.05, 1.0],
                initial_guess_cross_pol=[0.01, 0.1, 0.2, 0.05, 1.0],
                lower_bounds_co=[db2pow(-15), db2pow(-10), 0, 0, 0],
                lower_bounds_cross=[db2pow(-20), db2pow(-15), 0, 0, 0],
                upper_bounds_co=[db2pow(-2), db2pow(0), 2, 2, 15],
                upper_bounds_cross=[db2pow(-4), db2pow(0), 2, 2, 15],
                ratio_threshold=ratio_diff_threshold
                )

            if processing_cfg.debug_mode:
                out_rasters = [
                    os.path.join(scratch_dir, 'simul_forest_co_pol.tif'),
                    os.path.join(scratch_dir, 'simul_forest_cross_pol.tif'),
                    os.path.join(scratch_dir, 'simul_forest_ratio_db.tif')
                ]

                data_list = [simul_forest_co_pol, simul_forest_x_pol, simul_forest_ratio]

                for out_raster, data in zip(out_rasters, data_list):
                    dswx_sar_util.write_raster_block(
                        out_raster=out_raster,
                        data=data,
                        block_param=block_param,
                        geotransform=im_meta['geotransform'],
                        projection=im_meta['projection'],
                        datatype='float32',
                        cog_flag=True,
                        scratch_dir=scratch_dir
                    )

            if height_data_type == 'GLAD':
                # Inundated short vegetation
                # Since glad product has underestimated and not-precise height for short vegetation,
                # we cannot simulate the water cloud mode. Thus, we use the dual thresholds for co/cross-ratio.
                glad_height = mask_obj.get_class(block_param=block_param)
                short_vege_param_image = np.zeros(glad_height.shape)
                wet_short_vege_mask = np.logical_and(
                    glad_height >= 100,
                    glad_height <= 124
                    )
                dry_short_vege_mask = np.logical_and(
                    glad_height >= 0,
                    glad_height <= 24)
                occasional_open_water = np.logical_and(
                    glad_height >= 200,
                    glad_height <= 207)
                target_short_vege_area = wet_short_vege_mask | occasional_open_water
                short_vege_param_image[wet_short_vege_mask] = \
                    glad_height[wet_short_vege_mask] \

                short_vege_param_image[dry_short_vege_mask] = \
                    glad_height[dry_short_vege_mask]

            # (all_inundated_cand_short,
            # simul_short_co_pol,
            # simul_short_x_pol,
            # simul_short_ratio) = detect_iv_with_water_cloud(
            #     co_pol_image=co_pol_image,
            #     cross_pol_image=cross_pol_image,
            #     vegetation_param_image=short_vege_param_image,
            #     dry_forest_mask=dry_short_vege_mask,
            #     wet_forest_mask=wet_short_vege_mask,
            #     initial_guess_co_pol=[0.141, 0.5, 0.2, 0.05, 1.0],
            #     initial_guess_cross_pol=[0.01, 0.1, 0.2, 0.05, 1.0],
            #     lower_bounds_co=[db2pow(-15), db2pow(-10), 0, 0, 0],
            #     lower_bounds_cross=[db2pow(-20), db2pow(-15), 0, 0, 0],
            #     upper_bounds_co=[db2pow(-2), db2pow(0), 2, 2, 15],
            #     upper_bounds_cross=[db2pow(-4), db2pow(0), 2, 2, 15],
            #     ratio_threshold=inundated_vege_ratio_threshold)

            # For the forest, we apply the water-cloud model for the dry forest,
            # and simulate the backscattering for forest height.
            # We compute the expect dual-pol ratio and compare it with observed ratio.
            # If forest show higher than the expected ratio value, then those areas are
            # defined as inundated forest
                short_vege_co_threshold = -16
                short_vege_cross_threshold = -18
                short_vege_ratio_threshold  = 10.6

                dark_iv = np.logical_and(filtered_co_db < short_vege_co_threshold,
                    filtered_cross_db < short_vege_cross_threshold)
                bright_iv = filt_ratio_db > short_vege_ratio_threshold

                rtc_short_iv = dark_iv | bright_iv
                all_inundated_cand_short = rtc_short_iv & target_short_vege_area


                if processing_cfg.debug_mode:
                    out_rasters = [
                        os.path.join(scratch_dir, 'dark_iv.tif'),
                        os.path.join(scratch_dir, 'bright_iv.tif'),
                        os.path.join(scratch_dir, 'all_inundated_cand_short.tif')
                    ]

                    data_list = [dark_iv, bright_iv, all_inundated_cand_short]

                    for out_raster, data in zip(out_rasters, data_list):
                        print(out_raster)
                        dswx_sar_util.write_raster_block(
                            out_raster=out_raster,
                            data=data,
                            block_param=block_param,
                            geotransform=im_meta['geotransform'],
                            projection=im_meta['projection'],
                            datatype='float32',
                            cog_flag=True,
                            scratch_dir=scratch_dir)

                all_inundated_cand = all_inundated_cand_short | all_inundated_cand_forest

            elif height_data_type == 'ETH_Global_Height':
                all_inundated_cand = all_inundated_cand_forest


        inundated_vegetation = all_inundated_cand & \
            (target_inundated_vege_class > 0)
        output_data[inundated_vegetation] = 2

        dswx_sar_util.write_raster_block(
            out_raster=inundated_vege_path,
            data=output_data,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='byte',
            cog_flag=True,
            scratch_dir=scratch_dir)

        dswx_sar_util.write_raster_block(
            out_raster=target_area_path,
            data=target_inundated_vege_class,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='byte',
            cog_flag=True,
            scratch_dir=scratch_dir)

        dswx_sar_util.write_raster_block(
            out_raster=high_ratio_path,
            data=all_inundated_cand,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='byte',
            cog_flag=True,
            scratch_dir=scratch_dir)

        if processing_cfg.debug_mode:
            dswx_sar_util.write_raster_block(
                out_raster=os.path.join(
                    scratch_dir, f'intensity_db_ratio_{pol_all_str}.tif'),
                data=filt_ratio_db,
                block_param=block_param,
                geotransform=im_meta['geotransform'],
                projection=im_meta['projection'],
                datatype='float32',
                cog_flag=True,
                scratch_dir=scratch_dir)

    dswx_sar_util._save_as_cog(inundated_vege_path, scratch_dir)

    t_time_end = time.time()

    logger.info(
        f'total inundated vegetation mapping time: {t_time_end - t_all} sec')


def main():

    parser = _get_parser()

    args = parser.parse_args()

    generate_log.configure_log_file(args.log_file)

    cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_ni', args)
    print(args)
    processing_cfg = cfg.groups.processing
    print(processing_cfg.debug_mode, 'debug')
    pol_mode = processing_cfg.polarization_mode
    pol_list = processing_cfg.polarizations

    run(cfg)

if __name__ == '__main__':
    main()
