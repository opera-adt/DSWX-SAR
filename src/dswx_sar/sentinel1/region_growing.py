import copy
import gc
import mimetypes
import logging
import os
import time

from dswx_sar.common import _generate_log
from dswx_sar.common import (_dswx_sar_util)
from dswx_sar.common._region_growing import (run_parallel_region_growing,
                                             region_growing)
from dswx_sar.sentinel1.dswx_runconfig import (DSWX_S1_POL_DICT,
                                     RunConfig,
                                     _get_parser)

logger = logging.getLogger('dswx_sar')


def run(cfg):
    '''
    Run region growing with parameters in cfg dictionary
    '''
    logger.info('Starting DSWx-S1 Region-Growing')

    t_all = time.time()

    processing_cfg = cfg.groups.processing
    outputdir = cfg.groups.product_path_group.scratch_path
    pol_list = copy.deepcopy(processing_cfg.polarizations)
    pol_options = processing_cfg.polarimetric_option

    if pol_options is not None:
        pol_list += pol_options

    pol_str = '_'.join(pol_list)

    # Region growing cfg
    region_growing_cfg = processing_cfg.region_growing
    region_growing_seed = region_growing_cfg.initial_threshold
    region_growing_relaxed_threshold = region_growing_cfg.relaxed_threshold
    region_growing_line_per_block = region_growing_cfg.line_per_block

    logger.info(f'Region Growing Seed: {region_growing_seed}')
    logger.info('Region Growing relaxed threshold: '
                f'{region_growing_relaxed_threshold}')

    fuzzy_tif_path = os.path.join(
        outputdir, f'fuzzy_image_{pol_str}.tif')
    feature_meta = _dswx_sar_util.get_meta_from_tif(fuzzy_tif_path)
    feature_tif_path = os.path.join(
        outputdir, f"region_growing_output_binary_{pol_str}.tif")
    temp_rg_tif_path = os.path.join(
        outputdir, f'temp_region_growing_{pol_str}.tif')

    # First, run region-growing algorithm for blocks
    # to avoid repeated run with large image.
    run_parallel_region_growing(
        fuzzy_tif_path,
        temp_rg_tif_path,
        lines_per_block=region_growing_line_per_block,
        initial_threshold=region_growing_seed,
        relaxed_threshold=region_growing_relaxed_threshold,
        maxiter=0)

    fuzzy_map = _dswx_sar_util.read_geotiff(fuzzy_tif_path)
    temp_rg = _dswx_sar_util.read_geotiff(temp_rg_tif_path)

    # replace the fuzzy values with 1 for the pixels
    # where the region-growing already applied
    fuzzy_map[temp_rg == 1] = 1
    del temp_rg

    # Run region-growing again for entire image
    region_grow_map = region_growing(
        fuzzy_map,
        initial_threshold=region_growing_seed,
        relaxed_threshold=region_growing_relaxed_threshold,
        maxiter=0)

    _dswx_sar_util.save_dswx_product(
        region_grow_map,
        feature_tif_path,
        geotransform=feature_meta['geotransform'],
        projection=feature_meta['projection'],
        scratch_dir=outputdir)

    t_all_elapsed = time.time() - t_all
    logger.info(
        f"successfully ran region growing in {t_all_elapsed:.3f} seconds")


def main():
    'Main Function to run region growing'
    parser = _get_parser()

    args = parser.parse_args()

    mimetypes.add_type("text/yaml", ".yaml", strict=True)
    flag_first_file_is_text = 'text' in mimetypes.guess_type(
        args.input_yaml[0])[0]

    if not flag_first_file_is_text:
        raise ValueError('input yaml file is not text')

    if len(args.input_yaml) > 1 and flag_first_file_is_text:
        logger.info('ERROR only one runconfig file is allowed')
        return

    cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_s1', args)
    _generate_log.configure_log_file(args.log_file)

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
