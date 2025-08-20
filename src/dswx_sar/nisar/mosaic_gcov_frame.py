import os
import logging

from collections.abc import Iterator
import mimetypes

from dswx_sar.common.gcov_reader import RTCReader
from dswx_sar.nisar.dswx_ni_runconfig import (
    _get_parser,
    RunConfig)


logger = logging.getLogger('dswx_sar')


def run(cfg):
    """Generate mosaic workflow with user-defined args stored
    in dictionary runconfig 'cfg'

    Parameters:
    -----------
    cfg: RunConfig
        RunConfig object with user runconfig options
    """

    # Mosaicking parameters
    processing_cfg = cfg.groups.processing

    input_list = cfg.groups.input_file_group.input_file_path

    mosaic_cfg = processing_cfg.mosaic
    mosaic_mode = mosaic_cfg.mosaic_mode
    mosaic_prefix = mosaic_cfg.mosaic_prefix
    mosaic_posting_thresh = mosaic_cfg.mosaic_posting_thresh
    nisar_uni_mode = processing_cfg.nisar_uni_mode

    # Determine if resampling is required
    if nisar_uni_mode:
        resamp_required = False
    else:
        resamp_required = True

    resamp_method = mosaic_cfg.resamp_method
    resamp_out_res = mosaic_cfg.resamp_out_res

    scratch_dir = cfg.groups.product_path_group.scratch_path
    os.makedirs(scratch_dir, exist_ok=True)

    row_blk_size = mosaic_cfg.read_row_blk_size
    col_blk_size = mosaic_cfg.read_col_blk_size

    # Create reader object
    reader = RTCReader(
        row_blk_size=row_blk_size,
        col_blk_size=col_blk_size,
    )

    # Mosaic input RTC into output Geotiff
    reader.process_rtc_hdf5(
        input_list,
        scratch_dir,
        mosaic_mode,
        mosaic_prefix,
        mosaic_posting_thresh,
        resamp_method,
        resamp_out_res,
        resamp_required,
    )

if __name__ == "__main__":
    '''Run mosaic rtc products from command line'''
    # load arguments from command line
    parser = _get_parser()

    # parse arguments
    args = parser.parse_args()

    mimetypes.add_type("text/yaml", ".yaml", strict=True)

    cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_ni', args)

    # Run Mosaic RTC workflow
    run(cfg)
