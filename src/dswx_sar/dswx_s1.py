import logging
import time

from dswx_sar import mosaic_rtc_burst, save_mgrs_tiles, dummy_dswx_s1,\
                     pre_processing, region_growing
from dswx_sar.dswx_runconfig import _get_parser, RunConfig

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

    # create dummy water map.
    dummy_dswx_s1.run(cfg)

    # apply region-growing algorithm
    region_growing.run(cfg)

    # save product as mgrs tiles.
    save_mgrs_tiles.run(cfg)

    t_time_end = time.time()
    logger.info(f'total processing time: {t_time_end - t_all} sec')


def main():

    parser = _get_parser()
    args = parser.parse_args()
    cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_s1', args)

    dswx_s1_workflow(cfg)

if __name__ == '__main__':
    '''run dswx_s1 from command line'''
    main()