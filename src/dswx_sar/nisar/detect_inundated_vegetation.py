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

logger = logging.getLogger('dswx_sar')


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

    target_file_type = inundated_vege_cfg.target_area_file_type
    target_worldcover_class = inundated_vege_cfg.target_worldcover_class
    target_glad_class = inundated_vege_cfg.target_glad_class

    line_per_block = inundated_vege_cfg.line_per_block
    filter_options = inundated_vege_cfg.filter
    filter_method = inundated_vege_cfg.filter.method

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

        output_data = np.zeros(filt_ratio.shape, dtype='uint8')

        target_cross_pol = cross_db > inundated_vege_cross_pol_min
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
            target_cross_pol
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
