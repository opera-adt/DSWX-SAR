#!/usr/bin/env python

import logging
import time

from dswx_sar import (fuzzy_value_computation,
                      initial_threshold,
                      masking_with_ancillary,
                      mosaic_rtc_burst,
                      pre_processing,
                      save_mgrs_tiles,
                      refine_with_bimodality,
                      region_growing,)
from dswx_sar.dswx_runconfig import _get_parser, RunConfig
from dswx_sar import generate_log

logger = logging.getLogger('dswx_s1')

def dswx_s1_workflow(cfg):

    t_all = time.time()
    processing_cfg = cfg.groups.processing
    pol_list = processing_cfg.polarizations
    input_list = cfg.groups.input_file_group.input_file_path
    dswx_workflow = processing_cfg.dswx_workflow

    logger.info("")
    logger.info("Starting DSWx-S1 algorithm")
    logger.info(f"Number of RTC products: {len(input_list)}")
    logger.info(f"Polarizations : {pol_list}")

    # Create mosaic burst RTCs
    mosaic_rtc_burst.run(cfg)

    # preprocessing (relocating ancillary data and filtering)
    pre_processing.run(cfg)

    # Estimate threshold for given polarizations
    initial_threshold.run(cfg)

    # Fuzzy value computation
    fuzzy_value_computation.run(cfg)

    # Region Growing
    region_growing.run(cfg)

    if dswx_workflow == 'opera_dswx_s1':
        # Land use map
        masking_with_ancillary.run(cfg)

        ### Refinement
        refine_with_bimodality.run(cfg)

        if processing_cfg.inundated_vegetation.enabled:
            #[TODO]detect_inundated_vegetation.run(cfg)
            pass

    # save product as mgrs tiles.
    save_mgrs_tiles.run(cfg)

    t_time_end = time.time()
    logger.info(f'total processing time: {t_time_end - t_all} sec')


def main():

    parser = _get_parser()
    args = parser.parse_args()
    cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_s1', args)
    generate_log.configure_log_file(cfg.groups.log_file)

    dswx_s1_workflow(cfg)

if __name__ == '__main__':
    '''run dswx_s1 from command line'''
    main()