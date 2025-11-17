#!/usr/bin/env python

import logging
import time

from dswx_sar.common import (detect_inundated_vegetation,)

from dswx_sar.nisar.dswx_ni_runconfig import (_get_parser,
                                        RunConfig,
                                        DSWX_NI_POL_DICT)

from dswx_sar.common import _initial_threshold as initial_threshold
from dswx_sar.common import _fuzzy_value_computation as fuzzy_value_computation

from dswx_sar.common import _generate_log
from dswx_sar.nisar import (mosaic_gcov_frame, 
                            pre_processing, 
                            save_mgrs_tiles_ni, 
                            refine_with_bimodality,
                            region_growing)
from dswx_sar.nisar import masking_with_ancillary

logger = logging.getLogger('dswx_sar')


def dswx_ni_workflow(cfg):

    t_all = time.time()
    processing_cfg = cfg.groups.processing
    pol_list = processing_cfg.polarizations
    pol_mode = processing_cfg.polarization_mode
    nisar_uni_mode = processing_cfg.nisar_uni_mode
    input_list = cfg.groups.input_file_group.input_file_path
    dswx_workflow = processing_cfg.dswx_workflow
    inundated_veg_cfg = processing_cfg.inundated_vegetation

    logger.info("")
    logger.info("Starting DSWx-NI algorithm")
    logger.info(f"Number of RTC products: {len(input_list)}")
    logger.info(f"Polarizations : {pol_list}")
    logger.info(f"Polarization Mode : {pol_mode}")
    logger.info(f"NISAR Processing Mode : {nisar_uni_mode}")

    # Create mosaic burst RTCs
    # mosaic_gcov_frame.run(cfg)

    if pol_mode == 'MIX_DUAL_POL':
        proc_pol_set = [DSWX_NI_POL_DICT['DV_POL'],
                        DSWX_NI_POL_DICT['DH_POL']]
    elif pol_mode == 'MIX_SINGLE_POL':
        proc_pol_set = [DSWX_NI_POL_DICT['SV_POL'],
                        DSWX_NI_POL_DICT['SH_POL']]
    elif pol_mode == 'MIX_DUAL_H_SINGLE_V_POL':
        proc_pol_set = [DSWX_NI_POL_DICT['DH_POL'],
                        DSWX_NI_POL_DICT['SV_POL']]
    elif pol_mode == 'MIX_DUAL_V_SINGLE_H_POL':
        proc_pol_set = [DSWX_NI_POL_DICT['DV_POL'],
                        DSWX_NI_POL_DICT['SH_POL']]
    # Add Full polarization mode
    # Add more scenarios to the above options
    # I need to verify frame1 : [HH, HV], frame2: [VH, VV]
    # Need to run for loop twice to generate 2 intermediate water products
    else:
        proc_pol_set = [pol_list]

    for pol_set in proc_pol_set:
        processing_cfg.polarizations = pol_set
        # preprocessing (relocating ancillary data and filtering)
        pre_processing.run(cfg)

        # Estimate threshold for given polarizations
        initial_threshold.run(cfg)

        # Fuzzy value computation
        fuzzy_value_computation.run(cfg)

        # Region Growing
        region_growing.run(cfg)

        if dswx_workflow == 'opera_dswx_ni':
            # Land use map
            masking_with_ancillary.run(cfg)

            # Refinement
            refine_with_bimodality.run(cfg)

            if ((inundated_veg_cfg.enabled == 'auto') and
               len(pol_set) >= 2) or \
               inundated_veg_cfg.enabled is True:

                detect_inundated_vegetation.run(cfg)

    processing_cfg.polarizations = pol_list
    # save product as mgrs tiles.
    save_mgrs_tiles_ni.run(cfg)

    t_time_end = time.time()
    logger.info(f'total processing time: {t_time_end - t_all} sec')


def main():

    parser = _get_parser()
    args = parser.parse_args()
    cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_ni', args)
    _generate_log.configure_log_file(cfg.groups.log_file)

    dswx_ni_workflow(cfg)


if __name__ == '__main__':
    '''run dswx_ni from command line'''
    main()
