from collections import defaultdict
from datetime import datetime
import glob
import os

import numpy as np
import rasterio

from dswx_sar.dswx_runconfig import DSWX_S1_POL_DICT
from dswx_sar.dswx_sar_util import (band_assign_value_dict,
                                    read_geotiff)
from dswx_sar.version import VERSION as SOFTWARE_VERSION

# Constants
UNKNOWN = 'UNKNOWN'
PRODUCT_VERSION = UNKNOWN
DEFAULT_METADATA = {
    'DSWX_PRODUCT_VERSION': PRODUCT_VERSION,
    'SOFTWARE_VERSION': SOFTWARE_VERSION,
    'PROJECT': 'OPERA',
    'INSTITUTION': 'NASA JPL',
    'CONTACT_INFORMATION': 'operasds@jpl.nasa.gov',
    'PRODUCT_LEVEL': '3',
    'PRODUCT_TYPE': UNKNOWN,
    'PRODUCT_SOURCE': UNKNOWN,
    'SPACECRAFT_NAME': UNKNOWN,
    'SENSOR': UNKNOWN
}


def _copy_meta_data_from_rtc(metapath_list, dswx_metadata_dict):
    """Copy metadata dictionary from RTC metadata.

    Parameters
    ----------
    metapath_list : list
        List of metadata GeoTIFF file paths.
    dswx_metadata_dict : collections.OrderedDict
        Metadata dictionary to populate.
    """
    metadata_dict = defaultdict(list)

    dswx_meta_mapping = {
        'ORBIT_PASS_DIRECTION': 'RTC_ORBIT_PASS_DIRECTION',
        'BURST_ID': 'RTC_BURST_ID',
        'INPUT_L1_SLC_GRANULES': 'RTC_INPUT_L1_SLC_GRANULES',
        'PRODUCT_VERSION': 'RTC_PRODUCT_VERSION',
        'ZERO_DOPPLER_START_TIME': 'RTC_SENSING_START_TIME',
        'ZERO_DOPPLER_END_TIME': 'RTC_SENSING_END_TIME',
        'TRACK_NUMBER': 'RTC_TRACK_NUMBER',
        'ABSOLUTE_ORBIT_NUMBER': 'RTC_ABSOLUTE_ORBIT_NUMBER',
        'QA_RFI_INFO_AVAILABLE': 'RTC_QA_RFI_INFO_AVAILABLE',
    }
    # Collect metadata from overlapped bursts
    dswx_metadata_dict['RTC_INPUT_LIST'] = [
        os.path.basename(meta_path) for meta_path in metapath_list]

    for meta_path in metapath_list:

        with rasterio.open(meta_path) as src:
            # Accessing tags (additional metadata) of specific band
            # (e.g., band 1)
            tags = src.tags(0)
            for rtc_field, dswx_field in dswx_meta_mapping.items():
                rtc_meta_content = tags[rtc_field]
                # 'INPUT_L1_SLC_GRANULES' in RTC GeoTIFF has [' '].
                if rtc_field == 'INPUT_L1_SLC_GRANULES':
                    rtc_meta_content = rtc_meta_content[2:-2]
                metadata_dict[rtc_field].append(rtc_meta_content)

    for rtc_field, dswx_field in dswx_meta_mapping.items():
        values = metadata_dict[rtc_field]

        if rtc_field in ['ZERO_DOPPLER_START_TIME', 'ZERO_DOPPLER_END_TIME']:
            mode = 'min' if rtc_field == 'ZERO_DOPPLER_START_TIME' else 'max'
            sensing_time = _get_date_range(values, mode=mode)
            dswx_metadata_dict[dswx_field] = sensing_time

        elif rtc_field == 'QA_RFI_INFO_AVAILABLE':
            bool_list = [item.lower() == 'true' for item in values]
            dswx_metadata_dict[dswx_field] = any(bool_list)
            dswx_metadata_dict['RTC_QA_RFI_NUMBER_OF_BURSTS'] = \
                np.sum(bool_list)

        else:
            dswx_contents = sorted(set(values))
            dswx_metadata_dict[dswx_field] = \
                values[0] if len(dswx_contents) == 1 \
                else ', '.join(dswx_contents)


def _get_date_range(dates, mode='min'):
    """
    Converts a list of date strings to datetime objects and
    returns either the minimum or maximum date.

    Parameters
    ----------
    dates: list of str
        A list of date strings in the format "%Y-%m-%dT%H:%M:%S".
    mode: str, optional
        Determines whether to return the minimum or maximum date.
        Accepts 'min' or 'max'. Defaults to 'min'.

    Returns
    -------
    str:
        The minimum or maximum date in the format "%Y-%m-%dT%H:%M:%SZ",
        depending on the mode.
    """
    input_date_format = "%Y-%m-%dT%H:%M:%S"
    output_date_format = "%Y-%m-%dT%H:%M:%SZ"
    date_objects = [datetime.strptime(date[:19], input_date_format)
                    for date in dates]

    mode_functions = {'min': min, 'max': max}

    if mode in mode_functions:
        return datetime.strftime(mode_functions[mode](date_objects),
                                 output_date_format)
    else:
        raise ValueError('Invalid mode. Only "min" and "max" are supported.')


def _populate_ancillary_metadata_datasets(dswx_metadata_dict, ancillary_cfg):
    """Populate metadata dictionary with input files.

    Parameters
    ----------
    dswx_metadata_dict : collections.OrderedDict
        Metadata dictionary.
    ancillary_cfg: obj
        Configuration object containing all ancillary data sources
        and their descriptions.
    """
    # Dictionary mapping of source type to its file and description attributes
    # in ancillary_cfg
    source_map = {
        'INPUT_DEM_SOURCE': ('dem_file', 'dem_file_description'),
        'INPUT_HAND_SOURCE': ('hand_file', 'hand_file_description'),
        'INPUT_WORLDCOVER_SOURCE': ('worldcover_file',
                                    'worldcover_file_description'),
        'INPUT_SHORELINE_SOURCE': ('shoreline_shapefile',
                                   'shoreline_shapefile_description'),
        'INPUT_REFERENCE_WATER_SOURCE': ('reference_water_file',
                                         'reference_water_file_description')
    }

    for meta_key, (file_attr, desc_attr) in source_map.items():
        description = getattr(ancillary_cfg, desc_attr, None)
        file_path = getattr(ancillary_cfg, file_attr, None)
        if description:
            dswx_metadata_dict[meta_key] = description
        elif file_path:
            dswx_metadata_dict[meta_key] = os.path.basename(file_path)
        else:
            dswx_metadata_dict[meta_key] = 'NOT_PROVIDED_OR_NOT_USED'


def _populate_processing_metadata_datasets(dswx_metadata_dict, cfg):
    """
    Populate the metadata dictionary with processing information.
    Parameters
    ----------
    dswx_metadata_dict: dict
        Dictionary to be updated with processing metadata.
    cfg: object
        Configuration object containing processing metadata.
    """
    processing_cfg = cfg.groups.processing
    try:
        # Mapping for simple key-value assignments
        threshold_mapping = {
            'otsu': 'OTSU',
            'ki': 'Kittler-Illingworth',
            'rg': 'Region-growning based threshold'
        }
        initial_threshold_cfg = processing_cfg.initial_threshold
        masking_ancillary_cfg = processing_cfg.masking_ancillary
        fuzzy_value_cfg = processing_cfg.fuzzy_value
        inundated_vegetation_cfg = processing_cfg.inundated_vegetation
        refine_with_bimodality_cfg = processing_cfg.refine_with_bimodality

        dswx_metadata_dict.update({
            'PROCESSING_INFORMATION_POLARIZATION':
                processing_cfg.polarizations,
            'PROCESSING_INFORMATION_FILTER': 'Enhanced Lee filter',
            'PROCESSING_INFORMATION_FILTER_ENABLED':
                processing_cfg.filter.enabled,
            'PROCESSING_INFORMATION_FILTER_WINDOW_SIZE':
                processing_cfg.filter.window_size,

            'PROCESSING_INFORMATION_THRESHOLDING':
                threshold_mapping.get(initial_threshold_cfg.threshold_method),
            'PROCESSING_INFORMATION_THRESHOLD_TILE_SELECTION':
                initial_threshold_cfg.selection_method,
            'PROCESSING_INFORMATION_THRESHOLD_TILE_AVERAGE':
                initial_threshold_cfg.tile_average,
            'PROCESSING_INFORMATION_THRESHOLD_MULTI_THRESHOLD':
                initial_threshold_cfg.multi_threshold,
            'PROCESSING_INFORMATION_THRESHOLD_BIMODALITY':
                initial_threshold_cfg.tile_selection_bimodality,
            'PROCESSING_INFORMATION_THRESHOLD_TWELE':
                initial_threshold_cfg.tile_selection_twele,

            'PROCESSING_INFORMATION_REGION_GROWING_INITIAL_SEED':
                processing_cfg.region_growing.initial_threshold,
            'PROCESSING_INFORMATION_REGION_GROWING_RELAXED_THRESHOLD':
                processing_cfg.region_growing.relaxed_threshold,

            'PROCESSING_INFORMATION_MASKING_ANCILLARY_CO_POL_THRESHOLD':
                masking_ancillary_cfg.co_pol_threshold,
            'PROCESSING_INFORMATION_MASKING_ANCILLARY_CROSS_POL_THRESHOLD':
                masking_ancillary_cfg.cross_pol_threshold,
            'PROCESSING_INFORMATION_MASKING_ANCILLARY_WATER_THRESHOLD':
                masking_ancillary_cfg.water_threshold,

            'PROCESSING_INFORMATION_FUZZY_VALUE_HAND':
                [fuzzy_value_cfg.hand.member_min,
                 fuzzy_value_cfg.hand.member_max],
            'PROCESSING_INFORMATION_FUZZY_VALUE_SLOPE':
                [fuzzy_value_cfg.slope.member_min,
                 fuzzy_value_cfg.slope.member_max],
            'PROCESSING_INFORMATION_FUZZY_VALUE_REFERENCE_WATER':
                [fuzzy_value_cfg.reference_water.member_min,
                 fuzzy_value_cfg.reference_water.member_max],
            'PROCESSING_INFORMATION_FUZZY_VALUE_AREA':
                [fuzzy_value_cfg.area.member_min,
                 fuzzy_value_cfg.area.member_max],
            'PROCESSING_INFORMATION_FUZZY_VALUE_DARK_AREA':
                [fuzzy_value_cfg.dark_area.cross_land,
                 fuzzy_value_cfg.dark_area.cross_water],
            'PROCESSING_INFORMATION_FUZZY_VALUE_HIGH_FREQUENT_AREA':
                [fuzzy_value_cfg.high_frequent_water.water_min_value,
                 fuzzy_value_cfg.high_frequent_water.water_max_value],

            'PROCESSING_INFORMATION_REFINE_BIMODALITY_MINIMUM_PIXEL':
                refine_with_bimodality_cfg.minimum_pixel,
            'PROCESSING_INFORMATION_REFINE_BIMODALITY_THRESHOLD':
                [refine_with_bimodality_cfg.thresholds.ashman,
                 refine_with_bimodality_cfg.thresholds.Bhattacharyya_coefficient,
                 refine_with_bimodality_cfg.thresholds.bm_coefficient,
                 refine_with_bimodality_cfg.thresholds.surface_ratio,],

            'PROCESSING_INFORMATION_INUNDATED_VEGETATION':
                inundated_vegetation_cfg.enabled,
            'PROCESSING_INFORMATION_INUNDATED_VEGETATION_DUAL_POL_RATIO_MAX':
                inundated_vegetation_cfg.dual_pol_ratio_max,
            'PROCESSING_INFORMATION_INUNDATED_VEGETATION_DUAL_POL_RATIO_MIN':
                inundated_vegetation_cfg.dual_pol_ratio_min,
            'PROCESSING_INFORMATION_INUNDATED_VEGETATION_DUAL_POL_RATIO_THRESHOLD':
                inundated_vegetation_cfg.dual_pol_ratio_threshold,
            'PROCESSING_INFORMATION_INUNDATED_VEGETATION_CROSS_POL_MIN':
                inundated_vegetation_cfg.cross_pol_min

        })
    except AttributeError as e:
        print(f"Attribute error occurred: {e}")
    except KeyError as e:
        print(f"Key error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def compute_spatial_coverage(data_array):
    """
    Compute the spatial coverage.

    Parameters
    ----------
    data_array : np.array
        The 2D numpy array representation of the GeoTIFF.

    Returns
    -------
    float
        Spatial coverage as a percentage.
    """
    total_pixels = data_array.size
    invalid_pixels = np.sum(data_array == band_assign_value_dict['no_data'])
    valid_pixels = total_pixels - invalid_pixels

    return round(valid_pixels / total_pixels * 100, 4)


def compute_layover_shadow_coverage(data_array, spatial_coverage):
    """
    Compute the layover-shadow coverage.

    Parameters
    ----------
    data_array : np.array
        The 2D numpy array representation of the GeoTIFF.
    spatial_coverage : float
        Spatial coverage as a percentage.

    Returns
    -------
    float
        Layover-shadow coverage as a percentage.
    """
    layover_shadow_pixels = np.sum(
        data_array == band_assign_value_dict['layover_shadow_mask'])

    if spatial_coverage > 0:
        return round(layover_shadow_pixels /
                     (spatial_coverage * data_array.size), 4)
    else:
        return np.nan


def _populate_statics_metadata_datasets(dswx_metadata_dict, dswx_geotiff):
    """
    Populate the metadata dictionary with spatial
    and layover shadow coverages.

    Parameters
    ----------
    dswx_metadata_dict : dict
        Dictionary to be updated with coverages.
    dswx_geotiff : str
        Path to the GeoTIFF file.
    """
    try:
        dswx_data = read_geotiff(dswx_geotiff, verbose=False)

        spatial_cov = compute_spatial_coverage(dswx_data)
        layover_shadow_cov = compute_layover_shadow_coverage(
            dswx_data, spatial_cov)

        dswx_metadata_dict['SPATIAL_COVERAGE'] = spatial_cov
        dswx_metadata_dict['LAYOVER_SHADOW_COVERAGE'] = layover_shadow_cov

    except Exception as e:
        print(f"An error occurred while processing the GeoTIFF: {e}")


def set_dswx_s1_metadata(metadata_dict):
    """Update the dictionary with DSWx-S1 specific metadata."""
    metadata_dict.update({
        'PRODUCT_TYPE': 'DSWx-S1',
        'PRODUCT_SOURCE': 'OPERA_RTC_S1',
        'SPACECRAFT_NAME': 'Sentinel-1A/B',
        'SENSOR': 'IW'
    })


def _get_general_dswx_metadata_dict(cfg, product_version=None):
    """
    Generate metadata field for dswx products.

    Parameters
    ----------
    cfg: RunConfig
        Input runconfig.
    product_version: str, optional
        Version of the DSWx product. Defaults to None.

    Returns
    -------
    dict
        Metadata dictionary.
    """
    dswx_metadata_dict = DEFAULT_METADATA.copy()

    product_type = getattr(cfg.groups.primary_executable, 'product_type', None)

    if product_version:
        dswx_metadata_dict['DSWX_PRODUCT_VERSION'] = product_version

    if product_type == 'dswx_s1':
        set_dswx_s1_metadata(dswx_metadata_dict)

    # save datetime 'YYYY-MM-DDTHH:MM:SSZ'
    dswx_metadata_dict['PROCESSING_DATETIME'] = \
        datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    return dswx_metadata_dict


def gather_rtc_files(rtc_dirs, pols):
    """
    Given directories containing RTC files,
    gather all TIF files of the polarization `pols`.
    Parameters
    ----------
    rtc_dirs : list
        List of directories containing RTC files.
    pols : list
        The polarizations for which to find burst
        (e.g., 'HH', 'VV', 'HV', 'VH').

    Returns
    -------
    list
        A list of the RTC files
        for the specified polarization.
    """
    tif_files = []
    for pol in pols:  # Loop through each polarization
        for rtc_input_dir in rtc_dirs:
            # Find all matching files for the current polarization
            rtc_tif_files = glob.glob(
                os.path.join(rtc_input_dir, f'*{pol.upper()}*.tif'))
            # Extend the list with the found files
            tif_files.extend(rtc_tif_files)

    return tif_files


def collect_burst_id(rtc_dirs, pol):
    """
    Collect burst IDs from RTC files for a specific polarization.

    Parameters
    ----------
    rtc_dirs : list
        List of directories containing RTC files.
    pol : str
        The polarization for which to collect burst IDs
        (e.g., 'HH', 'VV', 'HV', 'VH').

    Returns
    -------
    list
        A list of unique burst IDs found in the RTC files
        for the specified polarization.
    """
    rtc_list = gather_rtc_files(rtc_dirs, pol)
    burst_id_list = []
    for rtc_file in rtc_list:
        with rasterio.open(rtc_file) as src:
            # Accessing tags (additional metadata) of specific band
            # (e.g., band 1)
            tags = src.tags(0)
            burst_id_list.append(tags['BURST_ID'])

    return list(set(burst_id_list))


def create_dswx_sar_metadata(cfg,
                             rtc_dirs,
                             product_version=None,
                             extra_meta_data=None):
    """
    Create dictionary containing metadata.

    Parameters
    ----------
    cfg: RunConfig
        Input runconfig.
    rtc_dirs: list
        List of directories containing RTC files.
    product_version: str, optional
        Version of the DSWx product. Defaults to None.
    extra_meta_data: dict, optional
        Additional metadata to merge with dswx_metadata_dict.
        Defaults to None.

    Returns
    -------
    dict
        Metadata dictionary.
    """
    # Get general DSWx-S1 metadata
    dswx_metadata_dict = _get_general_dswx_metadata_dict(
        cfg,
        product_version=product_version)
    # Add metadata related to ancillary data
    ancillary_cfg = cfg.groups.dynamic_ancillary_file_group

    h5path_list = gather_rtc_files(rtc_dirs, DSWX_S1_POL_DICT['CO_POL'])
    _copy_meta_data_from_rtc(h5path_list, dswx_metadata_dict)

    _populate_ancillary_metadata_datasets(dswx_metadata_dict, ancillary_cfg)
    _populate_processing_metadata_datasets(dswx_metadata_dict, cfg)
    # Merge extra_meta_data with dswx_metadata_dict if provided
    if extra_meta_data is not None:
        dswx_metadata_dict.update(extra_meta_data)
    return dswx_metadata_dict
