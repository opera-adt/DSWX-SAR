import copy
import logging
import mimetypes
import os
import time

from dswx_sar.common import _filter_SAR, _generate_log
import numpy as np

from dswx_sar.common import _dswx_sar_util
from dswx_sar.nisar.dswx_ni_runconfig import DSWX_NI_POL_DICT, _get_parser, RunConfig
from dswx_sar.common import _detect_inundated_vegetation
from dswx_sar.common._pre_processing import pol_ratio
from dswx_sar.common._masking_with_ancillary import FillMaskLandCover
from dswx_sar.common._fuzzy_value_computation import (
    create_slope_angle_geotiff,
    smf,
    zmf,
)

from dswx_sar.common._region_growing import (
    run_parallel_region_growing,
    region_growing_fast,
)


logger = logging.getLogger('dswx_sar')


def _compute_iv_fuzzy_value(
        ratio_db,
        hand,
        slope,
        target_area,
        no_data,
        ratio_min,
        ratio_max,
        hand_min,
        hand_max,
        slope_min,
        slope_max,
        hand_threshold,
        fuzzy_mode='weighted_sum'):
    """Compute fuzzy score for inundated vegetation refinement.

    This fuzzy value intentionally excludes reference water.

    Parameters
    ----------
    ratio_db : np.ndarray
        Filtered co-pol / cross-pol ratio in dB.
    hand : np.ndarray
        HAND raster block.
    slope : np.ndarray
        Slope raster block.
    target_area : np.ndarray
        Potential vegetation target mask. Positive values are valid.
    no_data : np.ndarray
        Boolean no-data mask.
    ratio_min : float
        Ratio value where IV membership starts increasing.
    ratio_max : float
        Ratio value where IV membership reaches high confidence.
    hand_min, hand_max : float
        HAND z-membership range. Lower HAND is more likely IV.
    slope_min, slope_max : float
        Slope z-membership range. Lower slope is more likely IV.
    hand_threshold : float
        Hard HAND cutoff.
    fuzzy_mode : str
        'weighted_sum' or 'product'.

    Returns
    -------
    iv_fuzzy : np.ndarray
        Float32 fuzzy score.
    """

    ratio_db = np.asarray(ratio_db, dtype=np.float32)
    hand = np.asarray(hand, dtype=np.float32)
    slope = np.asarray(slope, dtype=np.float32)

    # Higher dual-pol ratio is more likely inundated vegetation.
    ratio_s = smf(ratio_db, ratio_min, ratio_max)

    # Lower HAND and lower slope are more likely valid wet/inundated areas.
    hand_z = zmf(hand, hand_min, hand_max)
    slope_z = zmf(slope, slope_min, slope_max)

    if fuzzy_mode == 'product':
        iv_fuzzy = ratio_s * hand_z * slope_z
    else:
        iv_fuzzy = (
            ratio_s * 0.5 +
            hand_z * 0.25 +
            slope_z * 0.25
        )

    valid_area = (
        (target_area > 0) &
        (hand <= hand_threshold) &
        (~no_data)
    )

    iv_fuzzy[~valid_area] = 0
    iv_fuzzy[np.isnan(iv_fuzzy)] = 0

    return iv_fuzzy.astype(np.float32)


def _refine_inundated_vegetation_with_fuzzy_region_growing(
        cfg,
        pol_all_str,
        ratio_db_path,
        inundated_vege_path,
        target_area_path,
        im_meta):
    """Refine inundated vegetation using IV fuzzy logic and region growing."""

    processing_cfg = cfg.groups.processing
    scratch_dir = cfg.groups.product_path_group.scratch_path

    inundated_vege_cfg = processing_cfg.inundated_vegetation
    fuzzy_cfg = processing_cfg.fuzzy_value
    region_growing_cfg = processing_cfg.region_growing

    # Optional IV-specific config.
    # If these fields do not exist yet, fall back to existing fuzzy/RG params.
    iv_fuzzy_cfg = getattr(inundated_vege_cfg, 'fuzzy_refinement', None)

    if iv_fuzzy_cfg is None:
        logger.info('IV fuzzy refinement is not configured.')
        return

    if not iv_fuzzy_cfg.enabled:
        logger.info('IV fuzzy refinement is disabled.')
        return

    logger.info('Start IV fuzzy refinement with region growing.')

    dem_path = os.path.join(scratch_dir, 'interpolated_DEM.tif')
    hand_path = os.path.join(scratch_dir, 'interpolated_hand.tif')
    slope_path = os.path.join(scratch_dir, 'slope.tif')
    no_data_raster_path = os.path.join(
        scratch_dir,
        f'no_data_area_{pol_all_str}.tif'
    )
    if not os.path.exists(slope_path):
        create_slope_angle_geotiff(
            dem_path,
            slope_path,
            lines_per_block=lines_per_block
        )
    iv_fuzzy_path = os.path.join(
        scratch_dir,
        f'fuzzy_inundated_vegetation_{pol_all_str}.tif'
    )
    temp_rg_path = os.path.join(
        scratch_dir,
        f'temp_region_growing_inundated_vegetation_{pol_all_str}.tif'
    )
    rg_path = os.path.join(
        scratch_dir,
        f'region_growing_inundated_vegetation_{pol_all_str}.tif'
    )

    lines_per_block = getattr(
        iv_fuzzy_cfg,
        'line_per_block',
        inundated_vege_cfg.line_per_block
    )

    ratio_min = getattr(
        iv_fuzzy_cfg,
        'ratio_member_min',
        inundated_vege_cfg.dual_pol_ratio_min
    )
    ratio_max = getattr(
        iv_fuzzy_cfg,
        'ratio_member_max',
        inundated_vege_cfg.dual_pol_ratio_max
    )

    hand_min = getattr(
        iv_fuzzy_cfg,
        'hand_member_min',
        fuzzy_cfg.hand.member_min
    )
    hand_max = getattr(
        iv_fuzzy_cfg,
        'hand_member_max',
        fuzzy_cfg.hand.member_max
    )
    slope_min = getattr(
        iv_fuzzy_cfg,
        'slope_member_min',
        fuzzy_cfg.slope.member_min
    )
    slope_max = getattr(
        iv_fuzzy_cfg,
        'slope_member_max',
        fuzzy_cfg.slope.member_max
    )

    hand_threshold = getattr(
        iv_fuzzy_cfg,
        'hand_threshold',
        processing_cfg.hand.mask_value
    )

    fuzzy_mode = getattr(
        iv_fuzzy_cfg,
        'mode',
        'weighted_sum'
    )

    block_params = _dswx_sar_util.block_param_generator(
        lines_per_block=lines_per_block,
        data_shape=(im_meta['length'], im_meta['width']),
        pad_shape=(0, 0)
    )

    for block_ind, block_param in enumerate(block_params):
        logger.info(f'IV fuzzy refinement block {block_ind}')

        ratio_db = _dswx_sar_util.get_raster_block(
            ratio_db_path,
            block_param
        )

        hand = _dswx_sar_util.get_raster_block(
            hand_path,
            block_param
        )

        slope = _dswx_sar_util.get_raster_block(
            slope_path,
            block_param
        )

        target_area = _dswx_sar_util.get_raster_block(
            target_area_path,
            block_param
        )

        no_data = _dswx_sar_util.get_raster_block(
            no_data_raster_path,
            block_param
        ) == 1

        iv_fuzzy = _compute_iv_fuzzy_value(
            ratio_db=ratio_db,
            hand=hand,
            slope=slope,
            target_area=target_area,
            no_data=no_data,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            hand_min=hand_min,
            hand_max=hand_max,
            slope_min=slope_min,
            slope_max=slope_max,
            hand_threshold=hand_threshold,
            fuzzy_mode=fuzzy_mode
        )

        _dswx_sar_util.write_raster_block(
            out_raster=iv_fuzzy_path,
            data=iv_fuzzy,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='float32',
            cog_flag=True,
            scratch_dir=scratch_dir
        )

    # First block-wise RG for memory safety.
    run_parallel_region_growing(
        iv_fuzzy_path,
        temp_rg_path,
        lines_per_block=region_growing_cfg.line_per_block,
        initial_threshold=region_growing_cfg.initial_threshold,
        relaxed_threshold=region_growing_cfg.relaxed_threshold,
        maxiter=0,
        rg_method='fast'
    )

    # Then whole-image RG to connect across block boundaries.
    iv_fuzzy = _dswx_sar_util.read_geotiff(iv_fuzzy_path)
    temp_rg = _dswx_sar_util.read_geotiff(temp_rg_path)

    iv_fuzzy[temp_rg == 1] = 1
    del temp_rg

    rg_map = region_growing_fast(
        iv_fuzzy,
        initial_threshold=region_growing_cfg.initial_threshold,
        relaxed_threshold=region_growing_cfg.relaxed_threshold,
        maxiter=0
    )

    del iv_fuzzy

    _dswx_sar_util.save_dswx_product(
        rg_map,
        rg_path,
        geotransform=im_meta['geotransform'],
        projection=im_meta['projection'],
        scratch_dir=scratch_dir
    )

    # Combine with original IV map.
    original_iv = _dswx_sar_util.read_geotiff(inundated_vege_path)
    target_area = _dswx_sar_util.read_geotiff(target_area_path)

    refined_iv = np.zeros_like(original_iv, dtype=np.uint8)
    combine_mode = getattr(iv_fuzzy_cfg, 'combine_mode', 'intersection')

    if combine_mode == 'union':
        refined_mask = (
            ((original_iv == 2) | (rg_map == 1)) &
            (target_area > 0)
        )
    elif combine_mode == 'intersection':
        refined_mask = (
            (original_iv == 2) &
            (rg_map == 1) &
            (target_area > 0)
        )
    else:
        raise ValueError(
            f"Invalid combine_mode for IV fuzzy refinement: {combine_mode}. "
            "Expected 'union' or 'intersection'."
        )
    refined_iv[refined_mask] = 2

    _dswx_sar_util.save_dswx_product(
        refined_iv,
        inundated_vege_path,
        geotransform=im_meta['geotransform'],
        projection=im_meta['projection'],
        scratch_dir=scratch_dir
    )

    if processing_cfg.debug_mode:
        debug_refined_path = os.path.join(
            scratch_dir,
            f'temp_inundated_vegetation_refined_{pol_all_str}.tif'
        )
        _dswx_sar_util.save_dswx_product(
            refined_iv,
            debug_refined_path,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            scratch_dir=scratch_dir
        )

    logger.info('Finished IV fuzzy refinement with region growing.')


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
    inundated_vege_ratio_max = inundated_vege_cfg.dual_pol_ratio_max
    inundated_vege_ratio_min = inundated_vege_cfg.dual_pol_ratio_min
    inundated_vege_ratio_threshold = \
        inundated_vege_cfg.dual_pol_ratio_threshold
    inundated_vege_cross_pol_min = inundated_vege_cfg.cross_pol_min
    inundated_vege_copol_threshold = inundated_vege_cfg.copol_threshold
    target_file_type = inundated_vege_cfg.target_area_file_type
    target_worldcover_class = inundated_vege_cfg.target_worldcover_class
    target_glad_class = inundated_vege_cfg.target_glad_class

    line_per_block = inundated_vege_cfg.line_per_block
    filter_options = inundated_vege_cfg.filter
    filter_method = inundated_vege_cfg.filter.method

    iv_fuzzy_cfg = getattr(inundated_vege_cfg, 'fuzzy_refinement', None)
    fuzzy_logic_enabled = (
        iv_fuzzy_cfg is not None and
        getattr(iv_fuzzy_cfg, 'enabled', False)
    )
    interp_glad_path_str = os.path.join(scratch_dir, 'interpolated_glad.tif')
    interp_worldcover_path_str = os.path.join(scratch_dir,
                                              'interpolated_landcover.tif')

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
    ratio_db_path = os.path.join(
        scratch_dir,
        f'temp_intensity_db_ratio_{pol_all_str}.tif'
    )

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

    if (inundated_vege_ratio_min > inundated_vege_ratio_threshold) or \
       (inundated_vege_ratio_max < inundated_vege_ratio_threshold):
        err_str = f'{inundated_vege_ratio_threshold} is not valid.'
        raise ValueError(err_str)

    im_meta = _dswx_sar_util.get_meta_from_tif(rtc_dual_path)

    pad_shape = (filter_options.block_pad, 0)
    block_params = _dswx_sar_util.block_param_generator(
        lines_per_block=line_per_block,
        data_shape=(im_meta['length'],
                    im_meta['width']),
        pad_shape=pad_shape)

    for block_param in block_params:

        rtc_dual = _dswx_sar_util.get_raster_block(
            rtc_dual_path,
            block_param)

        rtc_ratio = pol_ratio(
            np.squeeze(rtc_dual[copol_ind, :, :]),
            np.squeeze(rtc_dual[crosspol_ind, :, :]))

        if filter_method == 'lee':
            filtering_method = _filter_SAR.lee_enhanced_filter
            filter_option = vars(filter_options.lee_filter)

        elif filter_method == 'anisotropic_diffusion':
            filtering_method = _filter_SAR.anisotropic_diffusion
            filter_option = vars(filter_options.anisotropic_diffusion)

        elif filter_method == 'guided_filter':
            filtering_method = _filter_SAR.guided_filter
            filter_option = vars(filter_options.guided_filter)

        elif filter_method == 'bregman':
            filtering_method = _filter_SAR.tv_bregman
            filter_option = vars(filter_options.bregman)

        filt_ratio = filtering_method(
                        rtc_ratio, **filter_option)
        filt_ratio_db = 10 * np.log10(filt_ratio +
                                      _dswx_sar_util.Constants.negligible_value)
        cross_db = 10 * np.log10(
            np.squeeze(rtc_dual[crosspol_ind, :, :]) +
            _dswx_sar_util.Constants.negligible_value)
        co_db = 10 * np.log10(
            np.squeeze(rtc_dual[copol_ind, :, :]) +
            _dswx_sar_util.Constants.negligible_value)


        _dswx_sar_util.write_raster_block(
            out_raster=ratio_db_path,
            data=filt_ratio_db,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='float32',
            cog_flag=True,
            scratch_dir=scratch_dir
        )

        output_data = np.zeros(filt_ratio.shape, dtype='uint8')

        target_cross_pol = cross_db > inundated_vege_cross_pol_min
        target_co_pol = co_db > inundated_vege_copol_threshold
        if target_file_type == 'WorldCover':
            target_inundated_vege_class = mask_obj.get_mask(
                mask_label=target_worldcover_class,
                block_param=block_param)
        elif target_file_type == 'GLAD':
            inundated_vege_target = _detect_inundated_vegetation.parse_ranges(target_glad_class)
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

        all_inundated_cand = \
            (filt_ratio_db > inundated_vege_ratio_threshold) & \
            target_cross_pol & target_co_pol 
        inundated_vegetation = all_inundated_cand & \
            (target_inundated_vege_class > 0)
        output_data[inundated_vegetation] = 2

        _dswx_sar_util.write_raster_block(
            out_raster=inundated_vege_path,
            data=output_data,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='byte',
            cog_flag=True,
            scratch_dir=scratch_dir)

        _dswx_sar_util.write_raster_block(
            out_raster=target_area_path,
            data=target_inundated_vege_class,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='byte',
            cog_flag=True,
            scratch_dir=scratch_dir)

        _dswx_sar_util.write_raster_block(
            out_raster=high_ratio_path,
            data=all_inundated_cand,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='byte',
            cog_flag=True,
            scratch_dir=scratch_dir)

        if processing_cfg.debug_mode:

            _dswx_sar_util.write_raster_block(
                out_raster=os.path.join(
                    scratch_dir, f'intensity_db_ratio_{pol_all_str}.tif'),
                data=filt_ratio_db,
                block_param=block_param,
                geotransform=im_meta['geotransform'],
                projection=im_meta['projection'],
                datatype='float32',
                cog_flag=True,
                scratch_dir=scratch_dir)
    if fuzzy_logic_enabled:

        _refine_inundated_vegetation_with_fuzzy_region_growing(
            cfg=cfg,
            pol_all_str=pol_all_str,
            ratio_db_path=ratio_db_path,
            inundated_vege_path=inundated_vege_path,
            target_area_path=target_area_path,
            im_meta=im_meta
        )
    t_time_end = time.time()

    logger.info(
        f'total inundated vegetation mapping time: {t_time_end - t_all} sec')


def main():

    parser = _get_parser()

    args = parser.parse_args()

    _generate_log.configure_log_file(args.log_file)

    mimetypes.add_type("text/yaml", ".yaml", strict=True)
    flag_first_file_is_text = 'text' in mimetypes.guess_type(
        args.input_yaml[0])[0]

    if len(args.input_yaml) > 1 and flag_first_file_is_text:
        logger.info('ERROR only one runconfig file is allowed')
        return

    if flag_first_file_is_text:
        cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_ni', args)

    processing_cfg = cfg.groups.processing
    pol_mode = processing_cfg.polarization_mode
    pol_list = processing_cfg.polarizations
    if pol_mode == 'MIX_DUAL_POL':
        proc_pol_set = [DSWX_NI_POL_DICT['DV_POL'],
                        DSWX_NI_POL_DICT['DH_POL']]
    elif pol_mode == 'MIX_SINGLE_POL':
        proc_pol_set = [DSWX_NI_POL_DICT['SV_POL'],
                        DSWX_NI_POL_DICT['SH_POL']]
    else:
        proc_pol_set = [pol_list]

    for pol_set in proc_pol_set:
        processing_cfg.polarizations = pol_set
        run(cfg)


if __name__ == '__main__':
    main()
