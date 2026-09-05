from __future__ import annotations

import ast
import copy
import datetime
import glob
import logging
import mimetypes
import os
import time

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict

from shapely.ops import transform as shp_transform

import geopandas as gpd
from dswx_sar.common import _dswx_sar_util, _generate_log
import h5py
import numpy as np
from osgeo import gdal
from pyproj import Transformer
import rasterio
from rasterio.warp import transform_bounds
from shapely import wkt
from shapely.geometry import Polygon
from shapely.ops import transform
from shapely.prepared import prep

from dswx_sar.nisar import (mosaic_gcov_frame)
from dswx_sar.common._dswx_sar_util import (band_assign_value_dict,
                                    _create_ocean_mask)
from dswx_sar.nisar.dswx_ni_runconfig import (RunConfig,
                                        _get_parser,
                                        get_pol_rtc_hdf5,
                                        DSWX_NI_POL_DICT)
from dswx_sar.common._metadata import (create_dswx_ni_metadata,
                               collect_frame_id,
                               _populate_statics_metadata_datasets)
from dswx_sar.common._save_mgrs_tiles import (
    get_bounding_box_from_mgrs_tile,
    get_bounding_box_from_mgrs_tile_db,
    get_intersecting_mgrs_tiles_list,
    merge_pol_layers)

logger = logging.getLogger('dswx_sar')



@dataclass(frozen=True)
class RTCFootprint:
    frame_name: str
    path: str
    polygon_4326: object  # shapely geometry (Polygon/MultiPolygon)


def _read_h5_string(ds) -> str:
    """Read h5py dataset and return a Python str."""
    val = ds[()]
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8")
    # sometimes val is numpy scalar string-like
    return str(val)


def load_rtc_footprints(
    input_rtc_files: Iterable[str],
    polygon_path: str = "/science/LSAR/identification/boundingPolygon",
    frame_name_path: Optional[str] = None,
) -> List[RTCFootprint]:
    """
    Read RTC bounding polygons (assumed EPSG:4326) and frame names once.

    - polygon_path: HDF5 dataset containing WKT polygon (EPSG:4326)
    - frame_name_path: optional HDF5 dataset for frame id/name.
      If None or missing, fallback to filename stem.
    """
    footprints: List[RTCFootprint] = []

    for path in input_rtc_files:
        frame_name = os.path.splitext(os.path.basename(path))[0]

        with h5py.File(path, "r") as src:
            # polygon
            if polygon_path not in src:
                raise KeyError(f"{polygon_path} not found in {path}")
            poly_wkt = _read_h5_string(src[polygon_path])
            poly = wkt.loads(poly_wkt)

            # frame name (optional)
            if frame_name_path is not None and frame_name_path in src:
                frame_name = _read_h5_string(src[frame_name_path])

        footprints.append(RTCFootprint(frame_name=frame_name, path=path, polygon_4326=poly))

    return footprints


def bbox_to_polygon(bbox: List[float]) -> Polygon:
    minx, miny, maxx, maxy = bbox
    return Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])


def transform_polygon(poly: Polygon, src_epsg: int, dst_epsg: int) -> Polygon:
    if src_epsg == dst_epsg:
        return poly
    transformer = Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)
    # manual transform to avoid shapely.ops.transform import overhead
    x, y = poly.exterior.coords.xy
    xy = list(zip(x, y))
    xy_t = [transformer.transform(xx, yy) for xx, yy in xy]
    return Polygon(xy_t)


def find_overlapping_rtc_frames(
    ref_bbox: List[float],
    ref_epsg: int,
    rtc_footprints_4326: Iterable[RTCFootprint],
    return_paths: bool = False,
) -> Optional[List[str]]:
    """
    Return RTC frame names overlapped with the tile bbox.

    - ref_bbox is in ref_epsg (often UTM)
    - rtc polygons are stored in EPSG:4326
    - We transform ref_bbox polygon to EPSG:4326, then intersect.
    """
    ref_poly = bbox_to_polygon(ref_bbox)
    ref_poly_4326 = transform_polygon(ref_poly, src_epsg=ref_epsg, dst_epsg=4326)

    # Prepared geometry speeds up many intersections
    ref_prepped = prep(ref_poly_4326)

    overlapped = []
    for fp in rtc_footprints_4326:
        if ref_prepped.intersects(fp.polygon_4326):
            overlapped.append(fp.path if return_paths else fp.frame_name)

    return overlapped if overlapped else None


def crop_and_save_mgrs_tile_spacing(
        source_tif_path,
        output_dir_path,
        output_tif_name,
        output_bbox,
        output_epsg,
        output_spacing,
        output_format,
        metadata,
        cog_compression,
        cog_nbits,
        interpolation_method='nearest',
        num_threads=2,
        warp_memory_limit_mb=512,
        gdal_cachemax_mb=512):
    """Crop the product along the MGRS tile grid and
    save it as a Cloud-Optimized GeoTIFF (COG).

    Parameters
    ----------
    source_tif_path : str
        Path to the original TIFF file.
    output_dir_path : str
        Path to the directory to save the output file.
    output_tif_name : str
        Filename for the cropped GeoTIFF.
    output_bbox : list
        List of bounding box
        i.e. [x_min, y_min, x_max, y_max]
    output_epsg : int
        EPSG for output GeoTIFF
    output_format : str
        Output file format (i.e., COG, GeoTIFF)
    metadata : dict
        Dictionary for metadata
    cog_compression: str
        Compression method for COG
    cog_nbits: int
        Compression nbits
    interpolation_method : str
        Interpolation method for cropping, by default 'nearest'.
    """
    os.makedirs(output_dir_path, exist_ok=True)
    output_tif_file_path = os.path.join(
        output_dir_path,
        output_tif_name
    )
    input_tif_obj = gdal.Open(source_tif_path)
    band = input_tif_obj.GetRasterBand(1)
    no_data_value = band.GetNoDataValue()
    output_type = band.DataType

    # Limit GDAL global cache.
    # This prevents GDAL from using a very large block cache.
    old_cachemax = gdal.GetCacheMax()
    gdal.SetCacheMax(int(gdal_cachemax_mb) * 1024 * 1024)

    # Use controlled threading.
    # Do not use ALL_CPUS unless memory is known to be sufficient.
    num_threads_str = str(num_threads)

    create_options = [
        "TILED=YES",
        "BLOCKXSIZE=512",
        "BLOCKYSIZE=512",
        "COMPRESS=DEFLATE",
        "PREDICTOR=2",
        "ZLEVEL=6",
        "BIGTIFF=IF_SAFER",
    ]

    if output_type == gdal.GDT_Byte:
        create_options.append("NBITS=8")
    elif output_type == gdal.GDT_UInt16:
        create_options.append("NBITS=16")

    try:
        warp_options = gdal.WarpOptions(
            dstSRS=f'EPSG:{output_epsg}',
            outputType=output_type,
            xRes=output_spacing,
            yRes=output_spacing,
            outputBounds=output_bbox,
            outputBoundsSRS=f'EPSG:{output_epsg}',
            resampleAlg=interpolation_method,
            srcNodata=no_data_value,
            dstNodata=no_data_value,
            format='GTiff',
            multithread=(num_threads_str != '1'),
            warpMemoryLimit=warp_memory_limit_mb,
            creationOptions=create_options,
            warpOptions=[
                f"NUM_THREADS={num_threads_str}",
                "INIT_DEST=NO_DATA",
            ],
        )

        out_ds = gdal.Warp(
            output_tif_file_path,
            source_tif_path,
            options=warp_options
        )

        if out_ds is None:
            raise RuntimeError(
                f'gdal.Warp failed for {source_tif_path} -> '
                f'{output_tif_file_path}'
            )

        out_ds.FlushCache()
        out_ds = None

    finally:
        # Restore GDAL cache setting.
        gdal.SetCacheMax(old_cachemax)

        band = None
        input_tif_obj = None

    # Populate metadata after crop.
    _populate_statics_metadata_datasets(
        metadata,
        output_tif_file_path
    )

    with rasterio.open(output_tif_file_path, 'r+') as src:
        src.update_tags(**metadata)
    effective_cog_nbits = cog_nbits

    if output_type == gdal.GDT_Byte and effective_cog_nbits is not None:
        effective_cog_nbits = min(int(effective_cog_nbits), 8)

    elif output_type == gdal.GDT_UInt16 and effective_cog_nbits is not None:
        effective_cog_nbits = min(int(effective_cog_nbits), 16)

    else:
        effective_cog_nbits = None

    if output_format == 'COG':
        _dswx_sar_util._save_as_cog(
            output_tif_file_path,
            output_dir_path,
            logger,
            compression=cog_compression,
            nbits=effective_cog_nbits
        )


def get_intersecting_mgrs_tiles_list_from_db(
        image_tif,
        mgrs_collection_file,
        track_number=None):
    """Find and return a list of MGRS tiles
    that intersect a reference GeoTIFF file
    By searching in database

    Parameters
    ----------
    image_tif: str
        Path to the input GeoTIFF file.
    mgrs_collection_file : str
        Path to the MGRS tile collection.
    track_number : int, optional
        Track number (or relative orbit number) to specify
        MGRS tile collection

    Returns
    ----------
    mgrs_list: list
        List of intersecting MGRS tiles.
    most_overlapped : GeoSeries
        The record of the MGRS tile with the maximum overlap area.
    """
    # Load the raster data
    with rasterio.open(image_tif) as src:
        epsg_code = src.crs.to_epsg() or 4326
        # Get bounds of the raster data
        left, bottom, right, top = src.bounds

        # Reproject to EPSG 4326 if the current EPSG is not 4326
        if epsg_code != 4326:
            left, bottom, right, top = transform_bounds(
                                                src.crs,
                                                'EPSG:4326',
                                                left,
                                                bottom,
                                                right,
                                                top)

    antimeridian_crossing_flag = False
    if left > 0 and right < 0:
        antimeridian_crossing_flag = True
        logger.info('The mosaic image crosses the antimeridian.')
    # Create a GeoDataFrame from the raster polygon
    if antimeridian_crossing_flag:
        # Create a Polygon from the bounds
        raster_polygon_left = Polygon(
            [(left, bottom),
             (left, top),
             (180, top),
             (180, bottom)])
        raster_polygon_right = Polygon(
            [(-180, bottom),
             (-180, top),
             (right, top),
             (right, bottom)])
        raster_gdf = gpd.GeoDataFrame([1, 2],
                                      geometry=[raster_polygon_left,
                                                raster_polygon_right],
                                      crs=4326)
    else:
        # Create a Polygon from the bounds
        raster_polygon = Polygon(
            [(left, bottom),
             (left, top),
             (right, top),
             (right, bottom)])
        raster_gdf = gpd.GeoDataFrame([1],
                                      geometry=[raster_polygon],
                                      crs=4326)

    # Load the vector data
    vector_gdf = gpd.read_file(mgrs_collection_file)

    # If track number is given, then search MGRS tile collection with
    # track number
    if track_number is not None and track_number != 0 and track_number != 1:
        vector_gdf = vector_gdf[
            vector_gdf['track_number'] ==
            track_number].to_crs("EPSG:4326")
    else:
        vector_gdf = vector_gdf.to_crs("EPSG:4326")

    # Calculate the intersection
    intersection = gpd.overlay(raster_gdf,
                               vector_gdf,
                               how='intersection')

    # Add a new column with the intersection area
    intersection['Area'] = intersection.to_crs(epsg=epsg_code).geometry.area

    # Find the polygon with the maximum intersection area
    most_overlapped = intersection.loc[intersection['Area'].idxmax()]

    mgrs_list = ast.literal_eval(most_overlapped['mgrs_tiles'])

    return list(set(mgrs_list)), most_overlapped


def get_mgrs_tiles_list_from_db(mgrs_collection_file,
                                mgrs_tile_collection_id):
    """Retrieve a list of MGRS tiles from a specified MGRS tile collection.

    Parameters
    ----------
    mgrs_collection_file : str
        Path to the file containing the MGRS tile collection.
        This file should be readable by GeoPandas.
    mgrs_tile_collection_id : str
        The ID of the MGRS tile collection from which to retrieve the MGRS tiles.

    Returns
    -------
    mgrs_list : list
        List of MGRS tile identifiers from the specified collection.
    """
    vector_gdf = gpd.read_file(mgrs_collection_file)
    most_overlapped = vector_gdf[
        vector_gdf['mgrs_set_id'] == mgrs_tile_collection_id].iloc[0]
    mgrs_list = ast.literal_eval(most_overlapped['mgrs_tiles'])

    return list(set(mgrs_list)), most_overlapped


def run(cfg):
    '''
    Run save mgrs tiles with parameters in cfg dictionary
    '''
    logger.info('Starting DSWx-NI save_mgrs_tiles')

    t_all = time.time()
    product_path_group_cfg = cfg.groups.product_path_group
    scratch_dir = product_path_group_cfg.scratch_path
    sas_outputdir = product_path_group_cfg.sas_output_path
    product_version = product_path_group_cfg.product_version

    # Output image format
    output_imagery_format = product_path_group_cfg.output_imagery_format
    output_imagery_compression = \
        product_path_group_cfg.output_imagery_compression
    output_imagery_nbits = product_path_group_cfg.output_imagery_nbits
    output_spacing = product_path_group_cfg.output_spacing

    # Processing parameters
    processing_cfg = cfg.groups.processing
    pol_list = copy.deepcopy(processing_cfg.polarizations)
    pol_options = processing_cfg.polarimetric_option
    if pol_options is not None:
        pol_list += pol_options
        pol_option_str = '_'.join(pol_options)
    pol_str = '_'.join(pol_list)
    pol_mode = processing_cfg.polarization_mode
    co_pol = processing_cfg.copol
    cross_pol = processing_cfg.crosspol

    input_list = cfg.groups.input_file_group.input_file_path
    input_mgrs_collection_id = \
        cfg.groups.input_file_group.input_mgrs_collection_id
    dswx_workflow = processing_cfg.dswx_workflow
    hand_mask_value = processing_cfg.hand.mask_value

    # Static ancillary database
    static_ancillary_file_group_cfg = cfg.groups.static_ancillary_file_group
    mgrs_db_path = static_ancillary_file_group_cfg.mgrs_database_file
    mgrs_collection_db_path = \
        static_ancillary_file_group_cfg.mgrs_collection_database_file

    # Inundated vegetation
    inundated_vege_cfg = processing_cfg.inundated_vegetation

    # Browse image options
    browser_image_cfg = cfg.groups.browse_image_group
    browse_image_flag = browser_image_cfg.save_browse
    browse_image_height = browser_image_cfg.browse_image_height
    browse_image_width = browser_image_cfg.browse_image_width

    flag_collapse_wtr_classes = browser_image_cfg.flag_collapse_wtr_classes
    exclude_inundated_vegetation = \
        browser_image_cfg.exclude_inundated_vegetation
    set_not_water_to_nodata = browser_image_cfg.set_not_water_to_nodata
    set_hand_mask_to_nodata = browser_image_cfg.set_hand_mask_to_nodata
    set_layover_shadow_to_nodata = \
        browser_image_cfg.set_layover_shadow_to_nodata
    set_ocean_masked_to_nodata = browser_image_cfg.set_ocean_masked_to_nodata
    save_tif_to_output = browser_image_cfg.save_tif_to_output

    shapefile = cfg.groups.dynamic_ancillary_file_group.shoreline_shapefile
    ocean_mask_enabled = processing_cfg.ocean_mask.mask_enabled
    margin_km = processing_cfg.ocean_mask.mask_margin_km
    polygon_water = processing_cfg.ocean_mask.mask_polygon_water

    partial_water_flag = processing_cfg.partial_surface_water.enabled
    partial_water_threshold = processing_cfg.partial_surface_water.threshold

    if product_version is None:
        logger.warning('WARNING: product version was not provided.')

    if mgrs_db_path is not None and mgrs_collection_db_path is not None:
        logger.info('Both the MGRS tile or the MGRS collection database '
                    'were provided.')
        database_bool = True
    else:
        logger.warning('WARNING: Either the MGRS tile or '
                       'the MGRS collection database was not provided.')
        database_bool = False

    os.makedirs(sas_outputdir, exist_ok=True)

    num_input_path = len(input_list)

    logger.info(f'Number of frames to process: {num_input_path}')
    date_str_list = []
    rtc_reader = mosaic_gcov_frame.RTCReader(row_blk_size=200,
                                             col_blk_size=200)
    for input_h5 in input_list:
        # Find HDF5 metadata

        rtc_metadata = rtc_reader.read_metadata_hdf5(input_h5)

        platform = 'LSAR'
        track_number = rtc_metadata['TRACK_NUMBER']
        resolution = int(output_spacing)
        date_str_list.append(rtc_metadata['ZERO_DOPPLER_START_TIME'])

        logger.info('')
        logger.info(f'GCOV track number : {track_number}')
        logger.info(f'GCOV resolution : {resolution}')
        logger.info('')

    input_date_format = "%Y-%m-%dT%H:%M:%S"
    output_date_format = "%Y%m%dT%H%M%SZ"

    date_str_id_temp = date_str_list[0][:19]
    date_str_id = datetime.datetime.strptime(
        date_str_id_temp, input_date_format).strftime(
            output_date_format)
    platform_str = platform
    resolution_str = str(int(resolution))

    if inundated_vege_cfg.enabled == 'auto':
        if cross_pol and co_pol:
            total_inundated_vege_flag = True
        else:
            total_inundated_vege_flag = False
        # update the IV flag
        inundated_vege_cfg.enabled = total_inundated_vege_flag

    else:
        total_inundated_vege_flag = \
            inundated_vege_cfg.enabled

    inundated_vege_mosaic_flag = False
    # Set merge_layer_flag and merge_pol_list based on pol_mode
    merge_layer_flag = pol_mode.startswith('MIX')
    if merge_layer_flag:
        if pol_mode == 'MIX_QD_DUAL_H_POL':
            pol_type1 = 'QP_POL'
            pol_type2 = 'DH_POL'
        elif pol_mode == 'MIX_QD_DUAL_V_POL':
            pol_type1 = 'QP_POL'
            pol_type2 = 'DV_POL'
        elif pol_mode == 'MIX_SINGLE_POL':
            pol_type1 = 'SV_POL'
            pol_type2 = 'SH_POL'
        else:
            logger.info('There is no need to mosaic different polarizations.')
            merge_layer_flag = False
        if merge_layer_flag:
            pol_set1 = DSWX_NI_POL_DICT[pol_type1]
            pol_set2 = DSWX_NI_POL_DICT[pol_type2]
            merge_pol_list = ['_'.join(pol_set1),
                              '_'.join(pol_set2)]
            logger.info(
                f'Merging products from {pol_type1} and {pol_type2}.'
            )
    else:
        pol_set1 = pol_list
        pol_set2 = []
        count_pols = []
        for input_dir in input_list:
            pol_types = DSWX_NI_POL_DICT['DV_POL'] + \
                        DSWX_NI_POL_DICT['DH_POL']
            # Initialize count
            count_pol = 0
            # Count files for each polarization type
            for target_pol in pol_types:
                pol_files = glob.glob(
                    os.path.join(input_dir, f'*{target_pol}*.tif'))
                count_pol += len(pol_files)
            count_pols.append(count_pol)
        all_match_first = all(count == count_pols[0]
                              for count in count_pols)

        # count_pols is list of number of the available pols.
        # count_pols[0] is always copol because
        # Co-pol proceed before cross-pol.
        if len(pol_list) > 1 and not all_match_first:
            # get first character from polarization (V or H)
            pol_id = pol_list[0][0]
            pol_mode = f'MIX_DUAL_{pol_id}_SINGLE_{pol_id}_POL'

    logger.info(f'Products are made from {pol_mode} scenario.')

    # If polarimetric methods such as ratio, span are used,
    # it is added to the name.
    if pol_options is not None and merge_layer_flag:

        merge_pol_list = [item + '_' + pol_option_str
                          for item in merge_pol_list]

    # Depending on the workflow, the final product are different.
    prefix_dict = {
        'final_water': 'bimodality_output_binary'
        if dswx_workflow == 'opera_dswx_ni'
        else 'region_growing_output_binary',
        'landcover_mask': 'refine_landcover_binary',
        'no_data_area': 'no_data_area',
        'region_growing': 'region_growing_output_binary',
        'fuzzy_value': 'fuzzy_image'
        }

    if total_inundated_vege_flag:
        if len(pol_set1) >= 2 and len(pol_set2) >= 2:
            inundated_vege_mosaic_flag = True

        prefix_dict['inundated_veg'] = 'temp_inundated_vegetation'
        prefix_dict['inundated_veg_target'] = 'temp_target_area'
        prefix_dict['inundated_veg_high_ratio'] = 'temp_high_dualpol_ratio'

    paths = {}
    for key, prefix in prefix_dict.items():
        file_path = f'{prefix}_{pol_str}.tif'
        paths[key] = os.path.join(scratch_dir, file_path)
        if merge_layer_flag:
            list_layers = [os.path.join(scratch_dir,
                                        f'{prefix}_{pol_cand_str}.tif')
                           for pol_cand_str in merge_pol_list]
            if key == 'no_data_area':
                extra_args = {'nodata_value': 1}
            elif key == 'fuzzy_value':
                extra_args = {'nodata_value': -1}
            else:
                extra_args = {'nodata_value': 0}
            if not inundated_vege_mosaic_flag and \
                key in ['inundated_veg', 'inundated_veg_target',
                        'inundated_veg_high_ratio']:
                dual_pol_vege_string = '_'.join(pol_set1)
                paths[key] = os.path.join(
                    scratch_dir, f'{prefix}_{dual_pol_vege_string}.tif')
            else:
                merge_pol_layers(list_layers,
                                 os.path.join(scratch_dir, file_path),
                                 **extra_args)

    # metadata for final product
    # e.g. geotransform, projection, length, width, utmzone, epsg
    water_meta = _dswx_sar_util.get_meta_from_tif(paths['final_water'])

    # repackage the water map
    # 1) water map
    water_is_1 = _dswx_sar_util._make_block_source(
        paths['final_water'],
        operation='eq',
        value=1)
    # not water
    water_is_0 = _dswx_sar_util._make_block_source(
        paths['final_water'],
        operation='eq',
        value=0)

    water_is_nodata = _dswx_sar_util._make_block_source(
        paths['final_water'],
        operation='eq',
        value=band_assign_value_dict['no_data']
    )

    nodata_from_file = _dswx_sar_util._make_block_source(
        paths['no_data_area'],
        operation='gt',
        value=0
    )

    no_data_raster = _dswx_sar_util._make_combined_mask(
        'or',
        nodata_from_file,
        water_is_nodata
    )

    # 2) layover/shadow
    layover_shadow_mask_path = os.path.join(
        scratch_dir,
        'mosaic_layovershadow_mask.tif'
    )

    if os.path.exists(layover_shadow_mask_path):
        static_positive = _dswx_sar_util._make_block_source(
            layover_shadow_mask_path,
            operation='gt',
            value=0
        )
        static_nodata = _dswx_sar_util._make_block_source(
            layover_shadow_mask_path,
            operation='eq',
            value=255
        )

        layover_shadow_mask = _dswx_sar_util._make_combined_mask(
            'and_not',
            static_positive,
            static_nodata
        )
        logger.info('Layover/shadow mask found')
    else:
        layover_shadow_mask = None
        logger.warning('No layover/shadow mask found')

    # 3) hand excluded
    hand_path = os.path.join(scratch_dir, 'interpolated_hand.tif')

    hand_mask = _dswx_sar_util._make_block_source(
        hand_path,
        operation='gt',
        value=hand_mask_value
    )

    full_wtr_water_set_path = \
        os.path.join(scratch_dir, 'full_water_binary_WTR_set.tif')
    full_bwtr_water_set_path = \
        os.path.join(scratch_dir, 'full_water_binary_BWTR_set.tif')
    full_conf_water_set_path = \
        os.path.join(scratch_dir, 'full_water_binary_CONF_set.tif')
    full_diag_water_set_path = \
        os.path.join(scratch_dir, 'full_water_binary_DIAG_set.tif')

    # 4) inundated_vegetation
    if total_inundated_vege_flag:
        inundated_vegetation = _dswx_sar_util._make_block_source(
            paths['inundated_veg'],
            operation='eq',
            value=2
        )

        inundated_vege_target_area = _dswx_sar_util._make_block_source(
            paths['inundated_veg_target'],
            operation='eq',
            value=1
        )

        inundated_vege_high_ratio = _dswx_sar_util._make_block_source(
            paths['inundated_veg_high_ratio'],
            operation='eq',
            value=1
        )

        logger.info('Inundated vegetation file was found.')

        iv_target_file_type = inundated_vege_cfg.target_area_file_type

        if iv_target_file_type == 'auto':
            interp_glad_path_str = os.path.join(
                scratch_dir,
                'interpolated_glad.tif'
            )

            if os.path.exists(interp_glad_path_str):
                inundated_vege_cfg.target_area_file_type = 'GLAD'

                # Count target-area values block-wise instead of reading
                # the whole target-area raster.
                worldcover_valid = _dswx_sar_util._count_raster_value_blockwise(
                    paths['inundated_veg_target'],
                    target_value=2,
                    lines_per_block=512
                )

                glad_valid = _dswx_sar_util._count_raster_value_blockwise(
                    paths['inundated_veg_target'],
                    target_value=1,
                    lines_per_block=512
                )

                # If some pixels are extracted from WorldCover,
                # IV source is GLAD/WorldCover.
                if worldcover_valid > 0 and glad_valid > 0:
                    inundated_vege_cfg.target_area_file_type = \
                        'GLAD/WorldCover'

                # If GLAD is provided but all pixels come from WorldCover
                # due to no-data of GLAD, IV source is WorldCover.
                elif worldcover_valid > 0 and glad_valid == 0:
                    inundated_vege_cfg.target_area_file_type = 'WorldCover'

            else:
                inundated_vege_cfg.target_area_file_type = 'WorldCover'

        logger.info(
            'Inundated vegetation areas are defined from  '
            f'{inundated_vege_cfg.target_area_file_type}.'
        )

    else:
        inundated_vegetation = None
        inundated_vege_target_area = None
        inundated_vege_high_ratio = None
        inundated_vege_cfg.target_area_file_type = None

        logger.info('Inundated vegetation file was disabled.')

    # 5) create ocean mask
    if ocean_mask_enabled:
        logger.info('Ocean mask enabled')
        ocean_mask = _create_ocean_mask(
            shapefile, margin_km, scratch_dir,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            length=water_meta['length'],
            width=water_meta['width'],
            polygon_water=polygon_water,
            temp_files_list=None)
    else:
        logger.info('Ocean mask disabled')
        ocean_mask = None

    if dswx_workflow == 'opera_dswx_ni':

        # WTR product
        _dswx_sar_util.save_dswx_product_blockwise(
            water_is_1,
            full_wtr_water_set_path,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Water classification (WTR)',
            scratch_dir=scratch_dir,
            logger=logger,
            layover_shadow_mask=layover_shadow_mask,
            hand_mask=hand_mask,
            inundated_vegetation=inundated_vegetation,
            no_data=no_data_raster,
            ocean_mask=ocean_mask,
            is_wtr=True,
        )

        # BWTR product
        # Water includes open water and inundated vegetation.
        if inundated_vegetation is not None:
            bwtr_water_mask = _dswx_sar_util._make_combined_mask(
                'or',
                water_is_1,
                inundated_vegetation
            )
            print(bwtr_water_mask)
        else:
            bwtr_water_mask = water_is_1

        _dswx_sar_util.save_dswx_product_blockwise(
            bwtr_water_mask,
            full_bwtr_water_set_path,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Binary Water classification (BWTR)',
            scratch_dir=scratch_dir,
            logger=logger,
            layover_shadow_mask=layover_shadow_mask,
            hand_mask=hand_mask,
            ocean_mask=ocean_mask,
            no_data=no_data_raster
        )

        # CONF product masks
        region_grow_is_1 = _dswx_sar_util._make_block_source(
            paths['region_growing'],
            operation='eq',
            value=1
        )

        landcover_is_1 = _dswx_sar_util._make_block_source(
            paths['landcover_mask'],
            operation='eq',
            value=1
        )

        landcover_is_0 = _dswx_sar_util._make_block_source(
            paths['landcover_mask'],
            operation='eq',
            value=0
        )

        landcover_not_1 = _dswx_sar_util._make_block_source(
            paths['landcover_mask'],
            operation='ne',
            value=1
        )

        landcover_mask = _dswx_sar_util._make_combined_mask(
            'and',
            region_grow_is_1,
            landcover_not_1
        )

        dark_land_mask = _dswx_sar_util._make_combined_mask(
            'and',
            landcover_is_1,
            water_is_0
        )

        bright_water_mask = _dswx_sar_util._make_combined_mask(
            'and',
            landcover_is_0,
            water_is_1
        )

        wetland = inundated_vege_target_area

        if wetland is not None:
            wetland_nonwater = _dswx_sar_util._make_combined_mask(
                'and',
                water_is_0,
                wetland
            )

            wetland_water = _dswx_sar_util._make_combined_mask(
                'and',
                water_is_1,
                wetland
            )

            wetland_bright_water_fill = _dswx_sar_util._make_combined_mask(
                'and',
                bright_water_mask,
                wetland
            )

            wetland_inundated_veg = _dswx_sar_util._make_combined_mask(
                'and',
                inundated_vegetation,
                wetland
            )

            wetland_dark_land_mask = _dswx_sar_util._make_combined_mask(
                'and',
                dark_land_mask,
                wetland
            )

            wetland_landcover_mask = _dswx_sar_util._make_combined_mask(
                'and',
                landcover_mask,
                wetland
            )

            high_ratio_and_nonwater = _dswx_sar_util._make_combined_mask(
                'and',
                inundated_vege_high_ratio,
                water_is_0
            )

            inundated_vegetation_conf = _dswx_sar_util._make_combined_mask(
                'and_not',
                high_ratio_and_nonwater,
                wetland
            )

        else:
            wetland_nonwater = None
            wetland_water = None
            wetland_bright_water_fill = None
            wetland_inundated_veg = None
            wetland_dark_land_mask = None
            wetland_landcover_mask = None
            inundated_vegetation_conf = None

        # CONF product
        _dswx_sar_util.save_dswx_product_blockwise(
            water_is_1,
            full_conf_water_set_path,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Confidence values (CONF)',
            scratch_dir=scratch_dir,
            logger=logger,
            landcover_mask=landcover_mask,
            bright_water_fill=bright_water_mask,
            dark_land_mask=dark_land_mask,
            inundated_vegetation_conf=inundated_vegetation_conf,
            wetland_nonwater=wetland_nonwater,
            wetland_water=wetland_water,
            wetland_bright_water_fill=wetland_bright_water_fill,
            wetland_inundated_veg=wetland_inundated_veg,
            wetland_dark_land_mask=wetland_dark_land_mask,
            wetland_landcover_mask=wetland_landcover_mask,
            layover_shadow_mask=layover_shadow_mask,
            hand_mask=hand_mask,
            ocean_mask=ocean_mask,
            no_data=no_data_raster,
            is_conf=True
        )

        # DIAG product
        fuzzy_value = _dswx_sar_util._make_block_source(
            paths['fuzzy_value'],
            operation='scale_round_clip',
            value=100,
            scale=100.0,
            output_dtype=np.uint8,
        )

        _dswx_sar_util.save_dswx_product_blockwise(
            fuzzy_value,
            full_diag_water_set_path,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Diagnostic layer (DIAG)',
            is_diag=True,
            scratch_dir=scratch_dir,
            datatype='uint8',
            logger=logger,
            layover_shadow_mask=layover_shadow_mask,
            hand_mask=hand_mask,
            ocean_mask=ocean_mask,
            no_data=no_data_raster
        )

    else:
        # Non-OPERA DSWx-NI workflow
        #
        # In Twele's workflow, bright water, dark land, and inundated
        # vegetation are not saved.
        _dswx_sar_util.save_dswx_product_blockwise(
            water_is_1,
            full_wtr_water_set_path,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Water classification (WTR)',
            scratch_dir=scratch_dir,
            logger=logger,
            layover_shadow_mask=layover_shadow_mask,
            hand_mask=hand_mask,
            no_data=no_data_raster
        )

    if partial_water_flag:
        new_water_meta = water_meta.copy()

        # Change 20 m to 30 m when counting partial surface water.
        new_geotransform = list(new_water_meta['geotransform'])
        new_geotransform[1] = output_spacing
        new_geotransform[5] = -1 * output_spacing

        new_water_meta['geotransform'] = tuple(new_geotransform)

        temp_full_wtr_water_set_path = os.path.join(
            scratch_dir,
            'full_water_binary_WTR_set_temp.tif'
        )

        os.rename(
            full_wtr_water_set_path,
            temp_full_wtr_water_set_path
        )

        _dswx_sar_util.partial_water_product_blockwise(
            input_file=temp_full_wtr_water_set_path,
            output_spacing=output_spacing,
            scratch_dir=scratch_dir,
            target_label=1,
            threshold=partial_water_threshold,
            output_file=full_wtr_water_set_path,
            logger=logger,
            lines_per_block=512,
            num_threads=1,
            warp_memory_limit_mb=256,
            keep_temp=False
        )

    # Get list of MGRS tiles overlapped with mosaic RTC image
    mgrs_meta_dict = {}

    if database_bool:
        actual_frame_id = collect_frame_id(input_list)
        # In the case that mgrs_tile_collection_id is given
        # from input, then extract the MGRS list from database
        if input_mgrs_collection_id is not None:
            logger.info(f'input mgrs collection id {input_mgrs_collection_id} is provided.')
            mgrs_tile_list, most_overlapped = \
                get_mgrs_tiles_list_from_db(
                    mgrs_collection_file=mgrs_collection_db_path,
                    mgrs_tile_collection_id=input_mgrs_collection_id)
        # In the case that mgrs_tile_collection_id is not given
        # from input, then extract the MGRS list from database
        # using track number and area intersecting with image_tif
        else:
            logger.info(f'Searching MGRS tiles using bounding box.')
            mgrs_tile_list, most_overlapped = \
                get_intersecting_mgrs_tiles_list_from_db(
                    mgrs_collection_file=mgrs_collection_db_path,
                    image_tif=paths['final_water'],
                    track_number=track_number
                    )
        maximum_frame = most_overlapped['number_of_frames']
        # convert string to list
        expected_frame_list = ast.literal_eval(most_overlapped['frames'])
        logger.info(f"Input RTCs are within {most_overlapped['mgrs_set_id']}")
        number_frame = len(actual_frame_id)
        mgrs_meta_dict['MGRS_SET_ID'] = most_overlapped['mgrs_set_id']
        mgrs_meta_dict['MGRS_COLLECTION_EXPECTED_NUMBER_OF_FRAMES'] = \
            maximum_frame
        mgrs_meta_dict['MGRS_COLLECTION_ACTUAL_NUMBER_OF_FRAMES'] = \
            number_frame
        missing_frame = len(list(set(expected_frame_list) -
                                 set(actual_frame_id)))
        mgrs_meta_dict['MGRS_COLLECTION_MISSING_NUMBER_OF_FRAMES'] = \
            missing_frame
        mgrs_meta_dict['MGRS_POL_MODE'] = pol_mode
        mgrs_meta_dict['INPUT_LIST'] = \
            [os.path.splitext(os.path.basename(path))[0] for path in input_list]
    else:
        mgrs_tile_list = get_intersecting_mgrs_tiles_list(
            image_tif=paths['final_water'])

    unique_mgrs_tile_list = list(set(mgrs_tile_list))
    logger.info(f'MGRS tiles: {unique_mgrs_tile_list}')

    rtc_footprints = load_rtc_footprints(
        input_list,
        polygon_path="/science/LSAR/identification/boundingPolygon",
        frame_name_path=None,  # set if you know the dataset path
    )

    processing_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ")
    if dswx_workflow == 'opera_dswx_ni':

        for mgrs_num_id, mgrs_tile_id in enumerate(unique_mgrs_tile_list):

            logger.info(f'MGRS tile {mgrs_num_id + 1}: {mgrs_tile_id}')

            if mgrs_db_path is None:
                (x_value_min, x_value_max,
                 y_value_min, y_value_max, epsg_output) = \
                    get_bounding_box_from_mgrs_tile(mgrs_tile_id)
            else:
                (x_value_min, x_value_max,
                 y_value_min, y_value_max, epsg_output) = \
                    get_bounding_box_from_mgrs_tile_db(mgrs_tile_id,
                                                       mgrs_db_path)
                if any(v is None for v in (x_value_min,
                                           x_value_max,
                                           y_value_min,
                                           y_value_max)):
                    continue
            mgrs_bbox = [x_value_min, y_value_min, x_value_max, y_value_max]

            overlapped_frame = find_overlapping_rtc_frames(
                ref_bbox=mgrs_bbox,
                ref_epsg=epsg_output,
                rtc_footprints_4326=rtc_footprints,
                return_paths=True,   # if create_dswx_ni_metadata expects file paths
            )
            logger.info(f'overlapped_bursts: {overlapped_frame}')

            # Metadata
            if overlapped_frame:
                dswx_metadata_dict = create_dswx_ni_metadata(
                     cfg,
                     overlapped_frame,
                     pol_list,
                     product_version=product_version,
                     extra_meta_data=mgrs_meta_dict)
                dswx_name_format_prefix = (f'OPERA_L3_DSWx-NI_T{mgrs_tile_id}_'
                                           f'{date_str_id}_{processing_time}_'
                                           f'{platform_str}_{resolution_str}_'
                                           f'v{product_version}')

                logger.info('Saving the file:')
                logger.info(f'      {dswx_name_format_prefix}')

                # Output File names
                output_mgrs_wtr = f'{dswx_name_format_prefix}_B01_WTR.tif'
                output_mgrs_bwtr = f'{dswx_name_format_prefix}_B02_BWTR.tif'
                output_mgrs_conf = f'{dswx_name_format_prefix}_B03_CONF.tif'
                output_mgrs_diag = f'{dswx_name_format_prefix}_B04_DIAG.tif'
                output_mgrs_browse = f'{dswx_name_format_prefix}_BROWSE'

                # Crop full size of BWTR, WTR, CONF file
                # and save them into MGRS tile grid
                full_input_file_paths = [full_bwtr_water_set_path,
                                         full_wtr_water_set_path,
                                         full_conf_water_set_path,
                                         full_diag_water_set_path]

                output_file_paths = [output_mgrs_bwtr,
                                     output_mgrs_wtr,
                                     output_mgrs_conf,
                                     output_mgrs_diag]

                for full_input_file_path, output_file_path in zip(
                    full_input_file_paths, output_file_paths
                ):
                    crop_and_save_mgrs_tile_spacing(
                        source_tif_path=full_input_file_path,
                        output_dir_path=sas_outputdir,
                        output_tif_name=output_file_path,
                        output_bbox=mgrs_bbox,
                        output_epsg=epsg_output,
                        output_spacing=output_spacing,
                        output_format=output_imagery_format,
                        metadata=dswx_metadata_dict,
                        cog_compression=output_imagery_compression,
                        cog_nbits=output_imagery_nbits,
                        interpolation_method='nearest')

                if browse_image_flag:
                    _dswx_sar_util.create_browse_image(
                        water_geotiff_filename=os.path.join(
                            sas_outputdir, output_mgrs_wtr),
                        output_dir_path=sas_outputdir,
                        browser_filename=output_mgrs_browse,
                        browse_image_height=browse_image_height,
                        browse_image_width=browse_image_width,
                        scratch_dir=scratch_dir,
                        flag_collapse_wtr_classes=flag_collapse_wtr_classes,
                        exclude_inundated_vegetation=exclude_inundated_vegetation,
                        set_not_water_to_nodata=set_not_water_to_nodata,
                        set_hand_mask_to_nodata=set_hand_mask_to_nodata,
                        set_layover_shadow_to_nodata=set_layover_shadow_to_nodata,
                        set_ocean_masked_to_nodata=set_ocean_masked_to_nodata,
                        save_tif_to_output_dir=save_tif_to_output)

    t_all_elapsed = time.time() - t_all
    logger.info("successfully ran save_mgrs_tiles in "
                f"{t_all_elapsed:.3f} seconds")


def main():

    parser = _get_parser()

    args = parser.parse_args()
    mimetypes.add_type("text/yaml", ".yaml", strict=True)
    flag_first_file_is_text = 'text' in mimetypes.guess_type(
        args.input_yaml[0])[0]

    if len(args.input_yaml) > 1 and flag_first_file_is_text:
        logger.info('ERROR only one runconfig file is allowed')
        return

    if flag_first_file_is_text:
        cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_ni', args)

    _generate_log.configure_log_file(cfg.groups.log_file)

    run(cfg)


if __name__ == '__main__':
    main()
