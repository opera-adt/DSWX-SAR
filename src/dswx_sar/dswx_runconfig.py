import argparse
from dataclasses import dataclass
from functools import singledispatch
import os
import sys
import logging
from types import SimpleNamespace

import yamale
from ruamel.yaml import YAML

import dswx_sar

logger = logging.getLogger('dswx-s1')

WORKFLOW_SCRIPTS_DIR = os.path.dirname(dswx_sar.__file__)


CO_POL_LIST = ['HH', 'VV']
CROSS_POL_LIST = ['VH', 'HV']

def _get_parser():
    parser = argparse.ArgumentParser(description='',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Input
    parser.add_argument('input_yaml',
                        type=str,
                        nargs='+',
                        help='Input YAML run configuration file')

    parser.add_argument('--debug_mode', action='store_true', default=False,
                        help='Print figures. Off/False by default.')

    parser.add_argument('--log',
                        '--log-file',
                        dest='log_file',
                        type=str,
                        help='Log file')

    return parser

def _deep_update(original, update):
    """Update default runconfig dict with user-supplied dict.
    Parameters
    ----------
    original : dict
        Dict with default options to be updated
    update: dict
        Dict with user-defined options used to update original/default
    Returns
    -------
    original: dict
        Default dictionary updated with user-defined options
    References
    ----------
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for key, val in update.items():
        if isinstance(val, dict):
            original[key] = _deep_update(original.get(key, {}), val)
        else:
            if val is not None:
                original[key] = val

    # return updated original
    return original

def load_validate_yaml(yaml_path: str, workflow_name: str) -> dict:
    """Initialize RunConfig class with options from given yaml file.
    Parameters
    ----------
    yaml_path : str
        Path to yaml file containing the options to load
    workflow_name: str
        Name of the workflow for which uploading default options
    """
    try:
        # Load schema corresponding to 'workflow_name' and to validate against
        schema_name = workflow_name
        schema = yamale.make_schema(
            f'{WORKFLOW_SCRIPTS_DIR}/schemas/{schema_name}.yaml',
            parser='ruamel')
    except:
        err_str = f'unable to load schema for workflow {workflow_name}.'
        logger.error(err_str)
        raise ValueError(err_str)

    # load yaml file or string from command line
    if os.path.isfile(yaml_path):
        try:
            data = yamale.make_data(yaml_path, parser='ruamel')
        except yamale.YamaleError as yamale_err:
            err_str = f'Yamale unable to load {workflow_name} ' \
                      'runconfig yaml {yaml_path} for validation.'
            logger.error(err_str)
            raise yamale.YamaleError(err_str) from yamale_err
    else:
        raise FileNotFoundError

    # validate yaml file taken from command line
    try:
        yamale.validate(schema, data)
    except yamale.YamaleError as yamale_err:
        err_str = f'Validation fail for {workflow_name} runconfig yaml {yaml_path}.'
        logger.error(err_str)
        raise yamale.YamaleError(err_str) from yamale_err

    # load default runconfig
    parser = YAML(typ='safe')
    default_cfg_path = f'{WORKFLOW_SCRIPTS_DIR}/defaults/{schema_name}.yaml'
    with open(default_cfg_path, 'r') as f_default:
        default_cfg = parser.load(f_default)

    with open(yaml_path, 'r') as f_yaml:
        user_cfg = parser.load(f_yaml)

    # Copy user-supplied configuration options into default runconfig
    _deep_update(default_cfg, user_cfg)
    # Validate YAML values under groups dict
    if 'groups' in default_cfg['runconfig'].keys():
        validate_group_dict(default_cfg['runconfig']['groups'])

    return default_cfg


def check_write_dir(dst_path: str):
    """Check if given directory is writeable; else raise error.
    Parameters
    ----------
    dst_path : str
        File path to directory for which to check writing permission
    """
    if not dst_path:
        dst_path = '.'

    # check if scratch path exists
    dst_path_ok = os.path.isdir(dst_path)

    if not dst_path_ok:
        try:
            os.makedirs(dst_path, exist_ok=True)
        except OSError:
            err_str = f"Unable to create {dst_path}"
            logger.error(err_str)
            raise OSError(err_str)

    # check if path writeable
    write_ok = os.access(dst_path, os.W_OK)
    if not write_ok:
        err_str = f"{dst_path} scratch directory lacks write permission."
        logger.error(err_str)
        raise PermissionError(err_str)

def check_file_path(file_path: str) -> None:
    """Check if file_path exist else raise an error.
    Parameters
    ----------
    file_path : str
        Path to file to be checked
    """
    if not os.path.exists(file_path):
        err_str = f'{file_path} not found'
        logger.error(err_str)
        raise FileNotFoundError(err_str)

def _find_polarization_from_data(input_dir_list):
    """
    This function walks through each directory in the given list of input directories,
    searches for specific file names that match the OPERA L2 RTC standard,
    and extracts the polarization part from these filenames.
    It then returns a list of unique polarizations found in these files.

    Parameters
    ----------
    input_dir_list : list
        List of the input directories containing RTC
        GeoTIFF files. The function will search through
        all subdirectories in these input directories.

    Returns
    -------
    list
        A list of unique polarization types extracted from the file names.
        This list contains the polarization identifiers
        (like 'HH', 'VV', etc.) found in the filenames.
    """
    extracted_strings = []
    for input_dir in input_dir_list:
        for _, _, filenames in os.walk(input_dir):
            for filename in filenames:
                if filename.startswith('OPERA_L2_RTC-') and \
                    filename.endswith('.tif'):
                    # Splitting the filename to extract the required part
                    parts = filename.split('_')
                    extracted_string = parts[-1].split('.')[0]
                    extracted_strings.append(extracted_string)
    found_pol = list(set(extracted_strings))
    return found_pol


def check_polarizations(pol_list, input_dir_list):
    """Sort polarizations so that co-pols are preceded.
    Parameters
    ----------
    pol_list : list
        List of polarizations.
    input_dir_list : list
        List of the input directories with RTC GeoTIFF files.
    """
    actual_pol = _find_polarization_from_data(input_dir_list)

    if 'dual-pol' in pol_list:
        proc_pol_list = ['VV', 'VH', 'HH', 'HV']
    elif 'co-pol' in pol_list:
        proc_pol_list = ['VV', 'HH']
    elif 'cross-pol' in pol_list:
        proc_pol_list = ['VH', 'HV']
    else:
        proc_pol_list = pol_list
    # Find the common polarizations between requests and files.
    proc_pol_list = list(set(proc_pol_list) & set(actual_pol))
    if not proc_pol_list:
        err_str = f'Polarizations {pol_list} are requestest ' \
                 'but Input RTC dirs do not seem to have them.'
        logger.error(err_str)
        raise yamale.YamaleError(err_str)

    co_pol_list = []
    cross_pol_list = []
    def custom_sort(pol):
        if pol in CO_POL_LIST:
            return (0, pol)  # Sort 'VV' and 'HH' before others
        return (1, pol)

    sorted_pol_list = sorted(proc_pol_list, key=custom_sort)

    for pol in sorted_pol_list:
        if pol in CO_POL_LIST:
            co_pol_list.append(pol)
        else:
            cross_pol_list.append(pol)

    return co_pol_list, cross_pol_list, sorted_pol_list


def validate_group_dict(group_cfg: dict) -> None:
    """Check and validate runconfig entries.
    Parameters
    ----------
    group_cfg : dict
        Dictionary storing runconfig options to validate
    """
    # Check 'dynamic_ancillary_file_groups' section of runconfig
    # Check that ancillary file exist and is GDAL-compatible
    dem_path = group_cfg['dynamic_ancillary_file_group']['dem_file']
    landcover_path = group_cfg[
                    'dynamic_ancillary_file_group']['worldcover_file']
    ref_water_path = group_cfg[
                    'dynamic_ancillary_file_group']['reference_water_file']
    hand_path = group_cfg[
                    'dynamic_ancillary_file_group']['hand_file']
    ancillary_file_paths = [dem_path, landcover_path,
                            ref_water_path, hand_path]

    for path in ancillary_file_paths:
        check_file_path(path)

    # Check 'product_group' section of runconfig.
    # Check that directories herein have writing permissions
    product_group = group_cfg['product_path_group']
    check_write_dir(product_group['sas_output_path'])
    check_write_dir(product_group['scratch_path'])

    static_ancillary_flag = group_cfg[
        'static_ancillary_file_group']['static_ancillary_inputs_flag']

    if static_ancillary_flag:
        mgrs_database_file = group_cfg[
            'static_ancillary_file_group']['mgrs_database_file']
        mgrs_collection_database_file = group_cfg[
            'static_ancillary_file_group']['mgrs_collection_database_file']

        if mgrs_database_file is None or mgrs_collection_database_file is None:
            err_str = f'Static ancillary data flag is {static_ancillary_flag} ' \
                      f'but mgrs_database_file {mgrs_database_file} and ' \
                      f'mgrs_collection_database_file {mgrs_collection_database_file}'
            logger.error(err_str)
            raise ValueError(err_str)

        check_file_path(mgrs_collection_database_file)
        check_file_path(mgrs_database_file)


@singledispatch
def wrap_namespace(ob):
    return ob

@wrap_namespace.register(dict)
def _wrap_dict(ob):
    return SimpleNamespace(**{key: wrap_namespace(val)
                              for key, val in ob.items()})

@wrap_namespace.register(list)
def _wrap_list(ob):
    return [wrap_namespace(val) for val in ob]

def unwrap_to_dict(sns: SimpleNamespace) -> dict:
    sns_as_dict = {}
    for key, val in sns.__dict__.items():
        if isinstance(val, SimpleNamespace):
            sns_as_dict[key] = unwrap_to_dict(val)
        else:
            sns_as_dict[key] = val

    return sns_as_dict

@dataclass(frozen=True)
class RunConfig:
    '''dataclass containing RTC runconfig'''
    # workflow name
    name: str
    # runconfig options converted from dict
    groups: SimpleNamespace
    # run config path
    run_config_path: str

    @classmethod
    def load_from_yaml(cls, yaml_path: str, workflow_name: str, args):
        """Initialize RunConfig class with options from given yaml file.
        Parameters
        ----------
        yaml_path : str
            Path to yaml file containing the options to load
        workflow_name: str
            Name of the workflow for which uploading default options
        """
        cfg = load_validate_yaml(yaml_path, workflow_name)

        groups_cfg = cfg['runconfig']['groups']

        # Convert runconfig dict to SimpleNamespace
        sns = wrap_namespace(groups_cfg)
        product = sns.primary_executable.product_type
        sensor = product.split('_')[-1]
        ancillary = sns.dynamic_ancillary_file_group

        algorithm_cfg = load_validate_yaml(ancillary.algorithm_parameters,
                                           f'algorithm_parameter_{sensor.lower()}')

        # Check if input files have the requested polarizations and
        # sort the order of the polarizations.
        input_dir_list = \
            cfg['runconfig']['groups']['input_file_group']['input_file_path']
        requested_pol = algorithm_cfg['runconfig']['processing']['polarizations']
        co_pol, cross_pol, pol_list = check_polarizations(
            requested_pol, input_dir_list)

        # update the polarizations
        algorithm_cfg['runconfig']['processing']['polarizations'] = pol_list
        algorithm_cfg['runconfig']['processing']['copol'] = co_pol[0] if co_pol else None
        algorithm_cfg['runconfig']['processing']['crosspol'] = cross_pol[0] if cross_pol else None

        algorithm_sns = wrap_namespace(algorithm_cfg['runconfig']['processing'])

        # copy runconfig parameters from dictionary
        sns.processing = algorithm_sns
        processing_group = sns.processing

        debug_mode = processing_group.debug_mode

        if args.debug_mode and not debug_mode:
            logger.warning(f'command line visualization "{args.debug_mode}"'
                f' has precedence over runconfig visualization "{debug_mode}"')
            sns.processing.debug_mode = args.debug_mode

        log_file = sns.log_file
        if args.log_file is not None:
            logger.warning(f'command line log file "{args.log_file}"'
                f' has precedence over runconfig log file "{log_file}"')
            sns.log_file = args.log_file

        return cls(cfg['runconfig']['name'], sns, yaml_path)

    @property
    def input_file_path(self):
        return self.groups.input_file_group.input_file_path

    @property
    def dem(self) -> str:
        return self.groups.dynamic_ancillary_file_group.dem_file

    @property
    def dem_description(self) -> str:
        return self.groups.dynamic_ancillary_file_group.dem_description

    @property
    def polarizations(self): #-> list[str]:
        return self.groups.processing.polarizations

    @property
    def product_path(self):
        return self.groups.product_group.product_path

    @property
    def product_id(self):
        return self.groups.product_group.product_id

    @property
    def scratch_path(self):
        return self.groups.product_group.scratch_path

    def as_dict(self):
        ''' Convert self to dict for write to YAML/JSON
        Unable to dataclasses.asdict() because isce3 objects can not be pickled
        '''
        self_as_dict = {}
        for key, val in self.__dict__.items():
            if key == 'groups':
                val = unwrap_to_dict(val)

            self_as_dict[key] = val
        return self_as_dict

    def to_yaml(self):
        self_as_dict = self.as_dict()
        yaml = YAML(typ='safe')
        yaml.dump(self_as_dict, sys.stdout, indent=4)