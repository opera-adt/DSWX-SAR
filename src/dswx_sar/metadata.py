from collections import defaultdict
from datetime import datetime
import glob
import os

import h5py
import numpy as np

from dswx_sar.version import VERSION as SOFTWARE_VERSION
from dswx_sar.dswx_sar_util import (band_assign_value_dict,
                                    read_geotiff)

# Constants
UNKNOWN = 'UNKNOWN'
PRODUCT_VERSION = UNKNOWN
DEFAULT_METADATA = {
    'DSWX_PRODUCT_VERSION': PRODUCT_VERSION,
    'SOFTWARE_VERSION': SOFTWARE_VERSION,
    'PROJECT': 'OPERA',
    'PRODUCT_LEVEL': '3',
    'PRODUCT_TYPE': UNKNOWN,
    'PRODUCT_SOURCE': UNKNOWN,
    'SPACECRAFT_NAME': UNKNOWN,
    'SENSOR': UNKNOWN
}


def _copy_meta_data_from_rtc(h5path_list, dswx_metadata_dict):
    """Copy metadata dictionary from RTC metadata.

    Parameters
    ----------
    h5path_list : list
        List of metadata HDF5 file paths.
    dswx_metadata_dict : collections.OrderedDict
        Metadata dictionary to populate.
    """
    metadata_dict = defaultdict(list)

    metadata_rtc_fields = {
        'orbitPassDirection': 'str',
        'burstID': 'str',
        'productVersion': 'str',
        'zeroDopplerStartTime': 'str',
        'trackNumber': 'int',
        'absoluteOrbitNumber': 'int'
    }

    # Mapping from RTC metadata to DSWx metadata
    dswx_meta_mapping = {
        'orbitPassDirection': 'ORBIT_PASS_DIRECTION',
        'burstID': 'BURST_ID',
        'productVersion': 'RTC_PRODUCT_VERSION',
        'zeroDopplerStartTime': 'SENSING_TIME',
        'trackNumber': 'TRACK_NUMBER',
        'absoluteOrbitNumber': 'ABSOLUTE_ORBIT_NUMBER'
    }

    for h5_meta_path in h5path_list:
        with h5py.File(h5_meta_path) as src:
            for field, dtype in metadata_rtc_fields.items():
                if dtype == 'str':
                    value = str(src[f'/identification/{field}'].asstr()[...])
                else:
                    value = int(src[f'/identification/{field}'][...])
                metadata_dict[field].append(value)

    for rtc_field, dswx_field in dswx_meta_mapping.items():
        values = metadata_dict[rtc_field]
        if dswx_field == 'SENSING_TIME':
            start, end = _get_date_range(values)
            dswx_metadata_dict['SENSING_START'] = start
            dswx_metadata_dict['SENSING_END'] = end
        else:
            dswx_metadata_dict[dswx_field] = values[0] if len(set(values)) == 1 else values


def _get_date_range(dates):
    """Converts and returns the min and max date from a list of date strings."""
    input_date_format = "%Y-%m-%dT%H:%M:%S"
    output_date_format = "%Y-%m-%dT%H:%M:%SZ"
    date_objects = [datetime.strptime(date[:19], input_date_format) for date in dates]
    return datetime.strftime(min(date_objects), output_date_format), \
           datetime.strftime(max(date_objects), output_date_format)


def _populate_ancillary_metadata_datasets(dswx_metadata_dict, ancillary_cfg):
    """Populate metadata dictionary with input files.

    Parameters
    ----------
    dswx_metadata_dict : collections.OrderedDict
        Metadata dictionary.
    ancillary_cfg: obj
        Configuration object containing all ancillary data sources and their descriptions.
    """
    # Dictionary mapping of source type to its file and description attributes in ancillary_cfg
    source_map = {
        'DEM_SOURCE': ('dem_file', 'dem_file_description'),
        'HAND_SOURCE': ('hand_file', 'hand_file_description'),
        'WORLDCOVER_SOURCE': ('worldcover_file', 'worldcover_file_description'),
        'SHORELINE_SOURCE': ('shoreline_shapefile', 'shoreline_shapefile_description'),
        'REFERENCE_WATER_SOURCE': ('reference_water_file', 'reference_water_file_description')
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
    try:
        processing_cfg = cfg.groups.processing
        # Mapping for simple key-value assignments
        threshold_mapping = {
            'otsu': 'OTSU',
            'ki': 'Kittler-Illingworth'
        }
        dswx_metadata_dict.update({
            'POLARIZATION': processing_cfg.polarizations,
            'FILTER': 'Enhanced Lee filter',
            'THRESHOLDING': threshold_mapping.get(processing_cfg.initial_threshold.threshold_method),
            'TILE_SELECTION': processing_cfg.initial_threshold.selection_method,
            'MULTI_THRESHOLD': processing_cfg.initial_threshold.multi_threshold,
            'FUZZY_SEED': processing_cfg.region_growing.initial_threshold,
            'FUZZY_TOLERANCE': processing_cfg.region_growing.relaxed_threshold,
            'INUNDATED_VEGETATION': processing_cfg.inundated_vegetation.enabled
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
    invalid_pixels = np.sum(data_array == 255)
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
    layover_shadow_pixels = np.sum(data_array == band_assign_value_dict['layover_shadow_mask'])

    if spatial_coverage > 0:
        return round(layover_shadow_pixels / (spatial_coverage / 100 * data_array.size) * 100, 4)
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
        layover_shadow_cov = compute_layover_shadow_coverage(dswx_data, spatial_cov)

        dswx_metadata_dict['SPATIAL_COVERAGE'] = spatial_cov
        dswx_metadata_dict['LAYOVER_SHADOW_COVERAGE'] = layover_shadow_cov

    except Exception as e:
        print(f"An error occurred while processing the GeoTIFF: {e}")


def set_dswx_s1_metadata(metadata_dict):
    """Update the dictionary with DSWx-S1 specific metadata."""
    metadata_dict.update({
        'PRODUCT_TYPE': 'DSWx-S1',
        'PRODUCT_SOURCE': 'OPERA RTC S1',
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
    dswx_metadata_dict['PROCESSING_DATETIME'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    return dswx_metadata_dict


def gather_rtc_files(rtc_dirs):
    """
    Given directories containing RTC files, gather all h5 files.
    """
    h5_files = []
    for rtc_input_dir in rtc_dirs:
        rtc_h5_file = glob.glob(os.path.join(rtc_input_dir, '*.h5'))
        if rtc_h5_file:
            h5_files.append(rtc_h5_file[0])
    return h5_files


def create_dswx_sar_metadata(cfg, rtc_dirs, product_version=None):
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

    h5path_list = gather_rtc_files(rtc_dirs)
    _copy_meta_data_from_rtc(h5path_list, dswx_metadata_dict)

    _populate_ancillary_metadata_datasets(dswx_metadata_dict, ancillary_cfg)
    _populate_processing_metadata_datasets(dswx_metadata_dict, cfg)

    return dswx_metadata_dict
