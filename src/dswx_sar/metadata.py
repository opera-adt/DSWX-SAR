import os
import h5py
import glob
from datetime import datetime
from collections import OrderedDict
from dswx_sar.version import VERSION as SOFTWARE_VERSION

def _copy_meta_data_from_rtc(h5path_list, dswx_metadata_dict):
    """Copy metadata dictionary from RTC metadata
    Parameters
    ----------
    h5path_list : list
            List of metadata HDF5 file
    dswx_metadata_dict : collections.OrderedDict
        Metadata dictionary
    """

    metadata_dict = dict()

    metadata_rtc_str_list = [
        'orbitPassDirection',
        'burstID',
        'productVersion', 
        'zeroDopplerStartTime']
    metadata_rtc_int_list = [
        'trackNumber',
        'absoluteOrbitNumber']

    metadata_rtc_list = metadata_rtc_str_list + metadata_rtc_int_list
    dswx_metadata_list = ['ORBIT_PASS_DIRECTION',
                          'BURST_ID',
                          'RTC_PRODUCT_VERSION',
                          'SENSING_TIME',
                          'TRACK_NUMBER',
                          'ABSOLUTE_ORBIT_NUMBER']

    for metadata in metadata_rtc_list:
        metadata_dict[metadata] = []

    for h5_meta_path in h5path_list:
        with h5py.File(h5_meta_path) as src:

            spacecraft_name = list(src['/science/'].keys())[0]
            identification_path = f'/science/{spacecraft_name}/identification/'

            for metadata_field in metadata_rtc_str_list:
                metadata_dict[metadata_field].append(
                    str(src[f'{identification_path}/{metadata_field}'].asstr()[...]))

            for metadata_field in metadata_rtc_int_list:
                metadata_dict[metadata_field].append(
                    str(src[f'{identification_path}/{metadata_field}'][...]))

    for metadata, dswx_meta_field in zip(metadata_rtc_list, dswx_metadata_list):

        if dswx_meta_field == 'SENSING_TIME':
            input_date_format = "%Y-%m-%dT%H:%M:%S"
            output_date_format = "%Y-%m-%dT%H:%M:%SZ"

            date_list = [datetime.strptime(date_single[0: 19], input_date_format) 
                                for date_single in metadata_dict[metadata]]
            dswx_metadata_dict['SENSING_START'] = datetime.strftime(min(date_list), 
                                                    output_date_format)
            dswx_metadata_dict['SENSING_END'] = datetime.strftime(max(date_list), 
                                                    output_date_format)
        else:
            unique_set = set(metadata_dict[metadata])
            if len(unique_set) == 1:
                dswx_metadata_dict[dswx_meta_field] = list(unique_set)[0]
            else:
                dswx_metadata_dict[dswx_meta_field] = list(unique_set)

def _populate_ancillary_metadata_datasets(dswx_metadata_dict,
                                     dem_file=None,
                                     dem_file_description=None,
                                     worldcover_file=None,
                                     worldcover_file_description=None,
                                     reference_water_file=None,
                                     reference_water_file_description=None,
                                     hand_file=None,
                                     hand_file_description=None,
                                     shoreline_shapefile=None,
                                     shoreline_shapefile_description=None):
    """Populate metadata dictionary with input files
       Parameters
       ----------
       dswx_metadata_dict : collections.OrderedDict
              Metadata dictionary
       hls_dataset: str
              HLS dataset name
       dem_file: str
              DEM filename
       dem_file_description: str
              DEM description
       worldcover_file: str
              Worldcover filename
       worldcover_file_description: str
              Worldcover description
       reference_water_file: str
              Reference water filename
       reference_water_file_description: str
              Reference water filename description
       hand_file: str
              Height above nearest drainage filename
       hand_file_description: str
              Height above nearest drainage filename description
       shoreline_shapefile: str
              NOAA GSHHS shapefile
       shoreline_shapefile_description: str
              NOAA GSHHS shapefile description
    """

    if dem_file_description:
        dswx_metadata_dict['DEM_SOURCE'] = dem_file_description
    elif dem_file:
        dswx_metadata_dict['DEM_SOURCE'] = \
            os.path.basename(dem_file)
    else:
        dswx_metadata_dict['DEM_SOURCE'] = 'NOT_PROVIDED'

    if hand_file_description:
        dswx_metadata_dict['HAND_SOURCE'] = hand_file_description
    elif dem_file:
        dswx_metadata_dict['HAND_SOURCE'] = \
            os.path.basename(hand_file)
    else:
        dswx_metadata_dict['HAND_SOURCE'] = 'NOT_PROVIDED'

    if worldcover_file_description:
        dswx_metadata_dict['WORLDCOVER_SOURCE'] = worldcover_file_description
    elif worldcover_file:
        dswx_metadata_dict['WORLDCOVER_SOURCE'] = \
            os.path.basename(worldcover_file)
    else:
        dswx_metadata_dict['WORLDCOVER_SOURCE'] = 'NOT_PROVIDED'

    if shoreline_shapefile_description:
        dswx_metadata_dict['SHORELINE_SOURCE'] = \
            shoreline_shapefile_description
    elif shoreline_shapefile:
        dswx_metadata_dict['SHORELINE_SOURCE'] = \
            os.path.basename(shoreline_shapefile)
    else:
        dswx_metadata_dict['SHORELINE_SOURCE'] = 'NOT_PROVIDED_OR_NOT_USED'

    if reference_water_file_description:
        dswx_metadata_dict['REFERENCE_WATER_SOURCE'] = \
            reference_water_file_description
    elif reference_water_file:
        dswx_metadata_dict['REFERENCE_WATER_SOURCE'] = \
            os.path.basename(reference_water_file)
    else:
        dswx_metadata_dict['REFERENCE_WATER_SOURCE'] = 'NOT_PROVIDED_OR_NOT_USED'

def _populate_processing_metadata_datasets(dswx_metadata_dict,
                                           cfg):

    processing_cfg = cfg.groups.processing
    dswx_metadata_dict['POLARIZATION'] = processing_cfg.polarizations
    dswx_metadata_dict['FILTER'] = 'Enhanced Lee filter'

    threshold_method = processing_cfg.initial_threshold.threshold_method
    if threshold_method == 'otsu':
        dswx_metadata_dict['THRESHOLDING'] = 'OTSU'
    elif threshold_method == 'ki':
        dswx_metadata_dict['THRESHOLDING'] = 'Kittler-Illingworth'
    dswx_metadata_dict['TILE_SELECTION'] = processing_cfg.initial_threshold.selection_method
    dswx_metadata_dict['MULTI_THRESHOLD'] = processing_cfg.initial_threshold.multi_threshold
    dswx_metadata_dict['FUZZY_SEED'] = processing_cfg.region_growing.seed
    dswx_metadata_dict['FUZZY_TOLERANCE'] = processing_cfg.region_growing.tolerance
    dswx_metadata_dict['INUNDATED_VEGETATION'] = processing_cfg.inundated_vegetation.enabled
    
    # [TODO]
    dswx_metadata_dict['SPATIAL_COVERAGE'] = None
    dswx_metadata_dict['LAYOVER_SHADOW_COVERAGE'] = None

def _get_dswx_metadata_dict(cfg, product_version=None):
    """Generate metadata field for dswx products
    Parameters
    ----------
    cfg: RunConfig
        Input runconfig
    Returns
    -------
    dswx_metadata_dict : collections.OrderedDict
        Metadata dictionary
    """
    dswx_metadata_dict = OrderedDict()
    product_type = cfg.groups.primary_executable.product_type

    # identification
    if product_version is not None:
        dswx_metadata_dict['DSWX_PRODUCT_VERSION'] = product_version
    else:
        dswx_metadata_dict['DSWX_PRODUCT_VERSION'] = SOFTWARE_VERSION

    dswx_metadata_dict['SOFTWARE_VERSION'] = SOFTWARE_VERSION
    dswx_metadata_dict['PROJECT'] = 'OPERA'
    dswx_metadata_dict['PRODUCT_LEVEL'] = '3'

    if product_type == 'dswx_s1':
        dswx_metadata_dict['PRODUCT_TYPE'] = 'DSWx-S1'
        dswx_metadata_dict['PRODUCT_SOURCE'] = 'OPERA RTC S1'
        dswx_metadata_dict['SPACECRAFT_NAME'] = 'Sentinel-1A/B'
        dswx_metadata_dict['SENSOR'] = 'IW'

    else:
        dswx_metadata_dict['PRODUCT_TYPE'] = 'UNKNOWN'
        dswx_metadata_dict['PRODUCT_SOURCE'] = 'UNKNOWN'
        dswx_metadata_dict['SPACECRAFT_NAME'] = 'UNKNOWN'
        dswx_metadata_dict['SENSOR'] = 'UNKNOWN'

    # save datetime 'YYYY-MM-DD HH:MM:SS'
    dswx_metadata_dict['PROCESSING_DATETIME'] = \
        datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    return dswx_metadata_dict

def create_dswx_sar_metadata(cfg):
    """Create dictionary containing metadat
    Parameters
    ----------
    cfg: RunConfig
        Input runconfig
    Returns
    -------
    dswx_metadata_dict : collections.OrderedDict
        Metadata dictionary
    """
    dswx_metadata_dict = _get_dswx_metadata_dict(cfg, product_version=0.1)
    _populate_ancillary_metadata_datasets(dswx_metadata_dict,
                                        dem_file=None,
                                        dem_file_description=None,
                                        worldcover_file=None,
                                        worldcover_file_description=None,
                                        reference_water_file=None,
                                        reference_water_file_description=None,
                                        hand_file=None,
                                        hand_file_description=None,
                                        shoreline_shapefile=None,
                                        shoreline_shapefile_description=None)
    ancillary_cfg = cfg.groups.dynamic_ancillary_file_group
    input_list = cfg.groups.input_file_group.input_file_path
    h5path_list = []
    if len(input_list) > 1:
        for ind, input_dir in enumerate(input_list):
            h5path_list.append(glob.glob(f'{input_dir}/*h5')[0])
    else:
        h5path_list = glob.glob({input_list})
    _copy_meta_data_from_rtc(h5path_list, dswx_metadata_dict)

    _populate_ancillary_metadata_datasets(dswx_metadata_dict,
                                     dem_file=ancillary_cfg.dem_file,
                                     dem_file_description=ancillary_cfg.dem_file_description,
                                     worldcover_file=ancillary_cfg.worldcover_file,
                                     worldcover_file_description=ancillary_cfg.worldcover_file_description,
                                     reference_water_file=ancillary_cfg.reference_water_file,
                                     reference_water_file_description=ancillary_cfg.reference_water_file_description,
                                     hand_file=ancillary_cfg.hand_file,
                                     hand_file_description=ancillary_cfg.hand_file_description,
                                     shoreline_shapefile=ancillary_cfg.shoreline_shapefile,
                                     shoreline_shapefile_description=ancillary_cfg.shoreline_shapefile_description)

    _get_dswx_metadata_dict(cfg, product_version=None)
    _populate_processing_metadata_datasets(dswx_metadata_dict,
                                           cfg)

    return dswx_metadata_dict
