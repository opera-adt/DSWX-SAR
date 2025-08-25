import logging
import mimetypes

import numpy as np

from dswx_sar.common import (_generate_log,
                             _initial_threshold)
from dswx_sar.sentinel1.dswx_runconfig import (
    DSWX_S1_POL_DICT,
    _get_parser)


logger = logging.getLogger('dswx_sar')


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
        try:
            from dswx_sar.sentinel1.dswx_runconfig import RunConfig
            workflow = 'dswx_s1'
        except:
            from dswx_sar.nisar.dswx_ni_runconfig import RunConfig
            workflow = 'dswx_ni'
        cfg = RunConfig.load_from_yaml(args.input_yaml[0],
                                       workflow, args)

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
        _initial_threshold.run(cfg)


if __name__ == '__main__':
    main()
