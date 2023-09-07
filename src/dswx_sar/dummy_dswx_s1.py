'''
ATTENTION
This code will be removed and replaced with the new code in future release. 
'''
import os
import time
import numpy as np
import logging
import mimetypes

from dswx_sar import dswx_sar_util
from dswx_sar.dswx_runconfig import _get_parser, RunConfig
from dswx_sar import generate_log


logger = logging.getLogger('dswx_s1')

def run(cfg):

    t_all = time.time()
    outputdir = cfg.groups.product_path_group.scratch_path
    processing_cfg = cfg.groups.processing
    pol_list = processing_cfg.polarizations
    pol_str = '_'.join(pol_list)
    input_list = cfg.groups.input_file_group.input_file_path
    dswx_workflow = processing_cfg.dswx_workflow
    product_prefix = processing_cfg.mosaic.mosaic_prefix

    input_rtc = os.path.join(outputdir,f'{product_prefix}_{pol_list[0]}.tif')
    water_meta = dswx_sar_util.get_meta_from_tif(input_rtc)
    rtc_data = dswx_sar_util.read_geotiff(input_rtc)
    no_data_raster = np.isnan(rtc_data)

    # Create dummy results 
    rtc_data_db = 10 * np.log10(rtc_data)
    dummy_conf = (rtc_data_db + 17) / (-14 + 17)
    dummy_conf[dummy_conf>1] = 1
    dummy_conf[dummy_conf<0] = 0
    dummy_conf = np.abs(dummy_conf-1)*100
    dummy_conf_output = np.ones(dummy_conf.shape)*255

    dummy_conf_output[dummy_conf<=100] = dummy_conf[dummy_conf<=100]

    water_tif_strs = [f'{outputdir}/bimodality_output_binary_{pol_str}.tif', 
                      f'{outputdir}/region_growing_output_binary_{pol_str}.tif', 
                      f'{outputdir}/refine_landcover_binary_{pol_str}.tif']
    thresholds = [-17, -14, -17.5,]

    for path, threshold in zip(water_tif_strs, thresholds):

        dswx_sar_util.save_dswx_product(
                    rtc_data_db < threshold,
                    path,
                    geotransform=water_meta['geotransform'],
                    projection=water_meta['projection'],
                    description='Water classification (WTR)',
                    scratch_dir=outputdir,
                    no_data=no_data_raster)

    dswx_sar_util.save_raster_gdal(
                dummy_conf_output,
                f'{outputdir}/fuzzy_image_{pol_str}.tif',
                geotransform=water_meta['geotransform'],
                projection=water_meta['projection'],
                scratch_dir=outputdir)

def main():

    parser = _get_parser()

    args = parser.parse_args()

    mimetypes.add_type("text/yaml", ".yaml", strict=True)
    flag_first_file_is_text = 'text' in mimetypes.guess_type(
        args.input_yaml[0])[0]

    if len(args.input_yaml) > 1 and flag_first_file_is_text:
        logger.info('ERROR only one runconfig file is allowed')
        return

    if flag_first_file_is_text:
        cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_s1', args)

    generate_log.configure_log_file(cfg.groups.log_file)

    run(cfg)

if __name__ == '__main__':
    main()