import copy
import logging
import mimetypes
import os
import time

import numpy as np

from dswx_sar import dswx_sar_util, filter_SAR, generate_log
from dswx_sar.dswx_runconfig import DSWX_S1_POL_DICT, _get_parser, RunConfig
from dswx_sar.pre_processing import pol_ratio
from dswx_sar.masking_with_ancillary import FillMaskLandCover

logger = logging.getLogger('dswx_s1')


def run(cfg):

    logger.info('Start inundated vegetation mapping')

    t_all = time.time()

    processing_cfg = cfg.groups.processing
    outputdir = cfg.groups.product_path_group.scratch_path
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
    inundated_vege_target = inundated_vege_cfg.target_land_cover

    filter_size = processing_cfg.filter.window_size
    line_per_block = inundated_vege_cfg.line_per_block

    # Currently, inundated vegetation for C-band is available for
    # Herbanceous wetland area
    landcover_path_str = os.path.join(outputdir, 'interpolated_landcover.tif')
    mask_obj = FillMaskLandCover(landcover_path_str)
    inundated_vege_path = \
        f"{outputdir}/temp_inundated_vegetation_{pol_all_str}.tif"

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

    rtc_dual_path = f"{outputdir}/filtered_image_{pol_all_str}.tif"
    if not os.path.isfile(rtc_dual_path):
        err_str = f'{rtc_dual_path} is not found.'
        raise FileExistsError(err_str)

    if (inundated_vege_ratio_min > inundated_vege_ratio_threshold) or \
       (inundated_vege_ratio_max < inundated_vege_ratio_threshold):
        err_str = f'{inundated_vege_ratio_threshold} is not valid.'
        raise ValueError(err_str)

    im_meta = dswx_sar_util.get_meta_from_tif(rtc_dual_path)

    pad_shape = (filter_size, 0)
    block_params = dswx_sar_util.block_param_generator(
        lines_per_block=line_per_block,
        data_shape=(im_meta['length'],
                    im_meta['width']),
        pad_shape=pad_shape)

    for block_param in block_params:

        rtc_dual = dswx_sar_util.get_raster_block(
            rtc_dual_path,
            block_param)

        rtc_ratio = pol_ratio(
            np.squeeze(rtc_dual[copol_ind, :, :]),
            np.squeeze(rtc_dual[crosspol_ind, :, :]))

        filt_ratio = filter_SAR.lee_enhanced_filter(
            rtc_ratio,
            win_size=filter_size)

        filt_ratio_db = 10 * np.log10(filt_ratio +
                                      dswx_sar_util.Constants.negligible_value)
        cross_db = 10 * np.log10(
            np.squeeze(rtc_dual[crosspol_ind, :, :]) +
            dswx_sar_util.Constants.negligible_value)

        output_data = np.zeros(filt_ratio.shape, dtype='uint8')

        target_cross_pol = cross_db > inundated_vege_cross_pol_min

        target_inundated_vege_class = mask_obj.get_mask(
            mask_label=inundated_vege_target,
            block_param=block_param)

        inundated_vegetation = (
            filt_ratio_db > inundated_vege_ratio_threshold) & \
            target_cross_pol & \
            target_inundated_vege_class

        output_data[inundated_vegetation] = 2

        dswx_sar_util.write_raster_block(
            out_raster=inundated_vege_path,
            data=output_data,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='byte',
            cog_flag=True,
            scratch_dir=outputdir)

        if processing_cfg.debug_mode:
            dswx_sar_util.write_raster_block(
                out_raster=os.path.join(
                    outputdir, f'intensity_db_ratio_{pol_all_str}.tif'),
                data=filt_ratio_db,
                block_param=block_param,
                geotransform=im_meta['geotransform'],
                projection=im_meta['projection'],
                datatype='float32',
                cog_flag=True,
                scratch_dir=outputdir)

    dswx_sar_util._save_as_cog(inundated_vege_path, outputdir)

    t_time_end = time.time()

    logger.info(
        f'total inundated vegetation mapping time: {t_time_end - t_all} sec')


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
