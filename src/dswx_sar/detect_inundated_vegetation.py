import logging
import mimetypes
import os
import time

import numpy as np
import osgeo.gdal as gdal

from dswx_sar import dswx_sar_util, filter_SAR, generate_log
from dswx_sar.dswx_runconfig import _get_parser, RunConfig
from dswx_sar.pre_processing import pol_ratio
from dswx_sar.masking_with_ancillary import FillMaskLandCover

logger = logging.getLogger('dswx_s1')


def run(cfg):

    logger.info('Start inundated vegetation mapping')

    t_all = time.time()

    processing_cfg = cfg.groups.processing
    pol_list = processing_cfg.polarizations
    pol_all_str = '_'.join(pol_list)
    outputdir = cfg.groups.product_path_group.scratch_path

    pol_list = processing_cfg.polarizations
    pol_all_str = '_'.join(pol_list)
    inundated_vege_cfg = processing_cfg.inundated_vegetation
    filter_size = processing_cfg.filter.window_size
    inundated_vege_ratio_max = inundated_vege_cfg.dual_pol_ratio_max
    inundated_vege_ratio_min = inundated_vege_cfg.dual_pol_ratio_min
    inundated_vege_ratio_threshold = \
        inundated_vege_cfg.dual_pol_ratio_threshold
    inundated_vege_cross_pol_min = inundated_vege_cfg.cross_pol_min

    dual_pol_flag = False
    if (('HH' in pol_list) and ('HV' in pol_list)) or \
       (('VV' in pol_list) and ('VH' in pol_list)):
        dual_pol_flag = True

    if inundated_vege_cfg.enabled and not dual_pol_flag:
        err_str = 'Daul polarizations are required for inundated vegetation'
        raise ValueError(err_str)

    rtc_dual_path = f"{outputdir}/filtered_image_{pol_all_str}.tif"
    if not os.path.isfile(rtc_dual_path):
        err_str = f'{rtc_dual_path} is not found.'
        raise FileExistsError(err_str)

    if (inundated_vege_ratio_min > inundated_vege_ratio_threshold) or \
       (inundated_vege_ratio_max < inundated_vege_ratio_threshold):
        err_str = f'{inundated_vege_ratio_threshold} is not valid.'
        raise ValueError(err_str)

    im_meta = dswx_sar_util.get_meta_from_tif(rtc_dual_path)

    rtc_dual_obj = gdal.Open(rtc_dual_path)
    rtc_dual = rtc_dual_obj.ReadAsArray()
    rtc_ratio = pol_ratio(np.squeeze(rtc_dual[0, :, :]),
                          np.squeeze(rtc_dual[1, :, :]))

    filt_ratio = filter_SAR.lee_enhanced_filter(
                rtc_ratio,
                win_size=filter_size)

    filt_ratio_db = 10 * np.log10(filt_ratio)
    cross_db = 10 * np.log10(np.squeeze(rtc_dual[1, :, :]))

    output_data = np.zeros(filt_ratio.shape, dtype='uint8')

    # Currently, inundated vegetation for C-band is available for
    # Herbanceous wetland area
    landcover_path_str = os.path.join(outputdir, 'interpolated_landcover')
    mask_obj = FillMaskLandCover(landcover_path_str)
    target_inundated_vege_class = mask_obj.get_mask(
        mask_label=['Herbaceous wetland'])

    target_cross_pol = cross_db > inundated_vege_cross_pol_min

    inundated_vegetation = (
        filt_ratio_db > inundated_vege_ratio_threshold) & \
        target_cross_pol & \
        target_inundated_vege_class

    mask_excluded = mask_obj.get_mask(mask_label=[
        'Bare sparse vegetation',
        'Urban'])

    output_data[inundated_vegetation] = 2
    output_data[mask_excluded] = 0

    inundated_vege_path = f"{outputdir}/temp_inundated_vegetation.tif"
    dswx_sar_util.save_dswx_product(
        output_data,
        inundated_vege_path,
        geotransform=im_meta['geotransform'],
        projection=im_meta['projection'],
        description='Water classification (WTR)',
        scratch_dir=outputdir)

    if processing_cfg.debug_mode:
        dswx_sar_util.save_raster_gdal(
            data=filt_ratio_db,
            output_file=os.path.join(outputdir, 'intensity_db_ratio.tif'),
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            scratch_dir=outputdir)

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

    run(cfg)


if __name__ == '__main__':
    main()
