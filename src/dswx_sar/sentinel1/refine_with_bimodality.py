import copy
import logging
import mimetypes
import os
import time

from dswx_sar.common import _dswx_sar_util, _generate_log
import gc

from dswx_sar.common import _masking_with_ancillary
from dswx_sar.common._refine_with_bimodality import (
    remove_false_water_bimodality_parallel,
    fill_gap_water_bimodality_parallel
)
from dswx_sar.sentinel1.dswx_runconfig import (_get_parser,
                                     RunConfig,
                                     DSWX_S1_POL_DICT)

logger = logging.getLogger('dswx_sar')

def run(cfg):

    t_all = time.time()
    logger.info("Starting the refinement based on bimodality")

    outputdir = cfg.groups.product_path_group.scratch_path
    processing_cfg = cfg.groups.processing
    pol_list = copy.deepcopy(processing_cfg.polarizations)
    pol_options = processing_cfg.polarimetric_option

    if pol_options is not None:
        pol_list += pol_options

    pol_str = '_'.join(pol_list)
    co_pol = list(set(processing_cfg.copol) & set(pol_list))

    bimodality_cfg = processing_cfg.refine_with_bimodality
    minimum_pixel = bimodality_cfg.minimum_pixel
    threshold_set = bimodality_cfg.thresholds
    ashman_threshold = threshold_set.ashman
    bhc_threshold = threshold_set.Bhattacharyya_coefficient
    bm_threshold = threshold_set.bm_coefficient
    surface_ratio_threshold = threshold_set.surface_ratio
    number_workers = bimodality_cfg.number_cpu
    lines_per_block = bimodality_cfg.lines_per_block

    filt_im_str = os.path.join(outputdir, f"filtered_image_{pol_str}.tif")
    no_data_geotiff_path = os.path.join(
        outputdir, f"no_data_area_{pol_str}.tif")
    im_meta = _dswx_sar_util.get_meta_from_tif(filt_im_str)

    # read the result of landcover masindex_array_to_imageg
    water_map_tif_str = os.path.join(
        outputdir, f'refine_landcover_binary_{pol_str}.tif')

    # read landcover map
    landcover_map_tif_str = os.path.join(
        outputdir, 'interpolated_landcover.tif')
    landcover_map = _dswx_sar_util.read_geotiff(landcover_map_tif_str)
    landcover_label = _masking_with_ancillary.get_label_landcover_esa_10()

    reference_water_gdal_str = os.path.join(outputdir, 'interpolated_wbd.tif')

    # Identify the non-water area from Landcover map
    if 'openSea' in landcover_label:
        landcover_not_water = (landcover_map != landcover_label['openSea']) &\
             (landcover_map != landcover_label['Permanent water bodies'])
    else:
        landcover_not_water = \
            (landcover_map != landcover_label['Permanent water bodies']) &\
            (landcover_map != landcover_label['No_data'])

    ref_land_str = os.path.join(outputdir,
                                f'landcover_not_water_{pol_str}.tif')
    _dswx_sar_util.save_raster_gdal(
                    data=landcover_not_water,
                    output_file=ref_land_str,
                    geotransform=im_meta['geotransform'],
                    projection=im_meta['projection'],
                    scratch_dir=outputdir)
    del landcover_not_water, landcover_map

    # If the landcover is non-water,
    # compute the bimnodality one more time
    # and remove the water body if test fails.
    input_file_dict = {'intensity': filt_im_str,
                       'landcover': landcover_map_tif_str,
                       'reference_water': reference_water_gdal_str,
                       'water_mask': water_map_tif_str,
                       'ref_land': ref_land_str,
                       'no_data': no_data_geotiff_path}

    # Identify waters that have not existed and
    # remove if bimodality does not exist
    bimodal_binary_path = \
        remove_false_water_bimodality_parallel(
            water_map_tif_str,
            pol_list=co_pol,
            thresholds=[ashman_threshold,
                        bhc_threshold,
                        surface_ratio_threshold,
                        bm_threshold],
            outputdir=outputdir,
            meta_info=im_meta,
            input_dict=input_file_dict,
            minimum_pixel=minimum_pixel,
            debug_mode=processing_cfg.debug_mode,
            number_workers=number_workers,
            lines_per_block=lines_per_block)

    # Identify gaps within the water bodies and fill the gaps
    # if bimodality exists
    bright_water_path = os.path.join(
        outputdir, f"bimodality_bright_water_{pol_str}.tif")

    _dswx_sar_util.merge_binary_layers(
        layer_list=[bimodal_binary_path],
        value_list=[0],
        merged_layer_path=bright_water_path,
        lines_per_block=lines_per_block,
        mode='or', cog_flag=True,
        scratch_dir=outputdir)

    fill_gap_bindary_path = \
        fill_gap_water_bimodality_parallel(
            bright_water_path,
            pol_list,
            threshold=[bm_threshold,
                       ashman_threshold],
            meta_info=im_meta,
            outputdir=outputdir,
            input_dict=input_file_dict,
            number_workers=number_workers,
            lines_per_block=lines_per_block)

    water_tif_str = os.path.join(
        outputdir, f"bimodality_output_binary_{pol_str}.tif")
    _dswx_sar_util.merge_binary_layers(
        layer_list=[bimodal_binary_path, fill_gap_bindary_path],
        value_list=[1, 1],
        merged_layer_path=water_tif_str,
        lines_per_block=lines_per_block,
        mode='or',
        cog_flag=True,
        scratch_dir=outputdir)

    t_time_end = time.time()
    t_all_elapsed = t_time_end - t_all
    logger.info("successfully ran bimodality test in "
                f"{t_all_elapsed:.3f} seconds")


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
