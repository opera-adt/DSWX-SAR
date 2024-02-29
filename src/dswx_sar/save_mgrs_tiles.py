import ast
import copy
import datetime
import glob
import logging
import mimetypes
import os
import time

import geopandas as gpd
import mgrs
import numpy as np
from osgeo import gdal, osr
from pyproj import CRS, Transformer
import rasterio
from rasterio.warp import transform_bounds
from rasterio.merge import merge
from shapely import wkt
from shapely.geometry import Polygon
from shapely.ops import transform

from dswx_sar import (dswx_sar_util,
                      generate_log)
from dswx_sar.dswx_sar_util import (band_assign_value_dict,
                                    _create_ocean_mask)
from dswx_sar.dswx_runconfig import RunConfig, _get_parser, DSWX_S1_POL_DICT
from dswx_sar.metadata import (create_dswx_sar_metadata,
                               collect_burst_id,
                               _populate_statics_metadata_datasets)

logger = logging.getLogger('dswx_s1')


def merge_pol_layers(list_layers,
                     output_file,
                     nodata_value=None,
                     scratch_dir='.'):
    """
    Merge multiple GeoTIFF files into a single file using rasterio.
    This function is used to merge the GeoTIFF with different polarizaitons.

    Parameters
    ----------
    list_layers : list
        List of GeoTIFF file paths to be merged.
    output_file : str
        Path for the output merged GeoTIFF file.
    nodata_value : float
        The no-data value to be considered in the merge.
    """
    # Open all the source files
    src_files_to_mosaic = []
    for file in list_layers:
        src = rasterio.open(file)
        src_files_to_mosaic.append(src)

    # Merge function
    if nodata_value is not None:
        mosaic, out_trans = merge(src_files_to_mosaic, nodata=nodata_value)
    else:
        mosaic, out_trans = merge(src_files_to_mosaic)

    # Get metadata from last file as kwargs for rasterio writing out the mosaic
    kwargs = src.meta

    # Update the metadata
    kwargs.update({"driver": "GTiff",
                   "height": mosaic.shape[1],
                   "width": mosaic.shape[2],
                   "transform": out_trans})

    # Write the mosaic raster to disk
    with rasterio.open(output_file, "w", **kwargs) as dest:
        dest.write(mosaic)

    # Close the source files
    for src in src_files_to_mosaic:
        src.close()

    dswx_sar_util._save_as_cog(output_file, scratch_dir)


def get_bounding_box_from_mgrs_tile_db(
        mgrs_tile_name,
        mgrs_db_path):
    """Get UTM bounding box from a given MGRS tile name
    from MGRS database

    Parameters
    ----------
    mgrs_tile_name: str
        Name of the MGRS tile (ex. 18LWQ)
    mgrs_db_path : str
        Path to the MGRS database file

    Returns
    -------
    minx: float
        Minimum x cooridate (UTM) for the given MGRS tile
    maxx: float
        Maximum x cooridate (UTM) for the given MGRS tile
    miny: float
        Minimum y cooridate (UTM) for the given MGRS tile
    maxy: float
        Maximum y cooridate (UTM) for the given MGRS tile
    epsg: int
        EPSG code
    """
    # Load the database from the MGRS db file.
    vector_gdf = gpd.read_file(mgrs_db_path)

    # Filter the MGRS database using the provided "mgrs_tile_name"
    filtered_gdf = vector_gdf[vector_gdf['mgrs_tile'] ==
                              mgrs_tile_name]

    # Get the bounding box coordinates and
    # EPSG code from the filtered data
    minx = filtered_gdf['xmin'].values[0]
    maxx = filtered_gdf['xmax'].values[0]
    miny = filtered_gdf['ymin'].values[0]
    maxy = filtered_gdf['ymax'].values[0]
    epsg = filtered_gdf['epsg'].values[0]

    return minx, maxx, miny, maxy, epsg


def get_bounding_box_from_mgrs_tile(mgrs_tile_name):
    """Get UTM bounding box a given MGRS tile.

    Parameters
    ----------
    mgrs_tile_name: str
        Name of the MGRS tile (ex. 18LWQ)

    Returns
    -------
    minx: float
        Minimum x cooridate (UTM) for the given MGRS tile
    maxx: float
        Maximum x cooridate (UTM) for the given MGRS tile
    miny: float
        Minimum y cooridate (UTM) for the given MGRS tile
    maxy: float
        Maximum y cooridate (UTM) for the given MGRS tile
    epsg: int
        EPSG code
    """
    # Convert MGRS tile to UTM coordinate
    mgrs_obj = mgrs.MGRS()
    lower_left_utm_coordinate = mgrs_obj.MGRSToUTM(mgrs_tile_name)

    x_min = lower_left_utm_coordinate[2]
    y_min = lower_left_utm_coordinate[3]

    is_southern = lower_left_utm_coordinate[1] == 'S'

    crs = CRS.from_dict({
        'proj': 'utm',
        'zone': lower_left_utm_coordinate[0],
        'south': is_southern
        })

    epsg = crs.to_authority()[1]

    # Compute bounding box
    x_list = []
    y_list = []
    for offset_x_multiplier in range(2):
        for offset_y_multiplier in range(2):
            # Additional coordinates to compute min/max of x/y
            # DSWX products have 9.8 buffer along top and right side.
            y_more = y_min - 9.8 * 1000 + offset_y_multiplier * 109.8 * 1000
            x_more = x_min + offset_x_multiplier * 109.8 * 1000
            x_list.append(x_more)
            y_list.append(y_more)

    minx = np.min(x_list)
    maxx = np.max(x_list)
    miny = np.min(y_list)
    maxy = np.max(y_list)

    return minx, maxx, miny, maxy, epsg


def find_intersecting_burst_with_bbox(ref_bbox,
                                      ref_epsg,
                                      input_rtc_dirs):
    """Find bursts overlapped with the reference bbox.

    Parameters
    ----------
    ref_bbox: list
        Bounding box, minx, miny, maxx, maxy
    ref_epsg: int
        reference EPSG code.
    input_rtc_dirs: list
        List of rtc directories

    Returns
    -------
    overlapped_rtc_dir_list: list
        List of rtc bursts overlapped with given bbox
    """
    minx, miny, maxx, maxy = ref_bbox
    ref_polygon = Polygon([(minx, miny),
                           (minx, maxy),
                           (maxx, maxy),
                           (maxx, miny)])

    overlapped_rtc_dir_list = []
    for input_dir in input_rtc_dirs:

        copol_file_list = [f for f in glob.glob(f'{input_dir}/*.tif')
                           if any(pol in f
                                  for pol in DSWX_S1_POL_DICT['CO_POL'])]
        if copol_file_list:
            with rasterio.open(copol_file_list[0]) as src:
                tags = src.tags(0)
                rtc_polygon_str = tags['BOUNDING_POLYGON']
                epsg_code = int(tags['BOUNDING_POLYGON_EPSG_CODE'])
                rtc_polygon = wkt.loads(rtc_polygon_str)

                # Reproject to EPSG 4326 if the current EPSG is not 4326
                if epsg_code != ref_epsg:
                    # # Create a transformer
                    transformer = Transformer.from_crs(f'EPSG:{epsg_code}',
                                                       f'EPSG:{ref_epsg}',
                                                       always_xy=True)

                    # Transform the polygon
                    rtc_polygon = transform(transformer.transform, rtc_polygon)

            # Check if bursts intersect the reference polygon
            if ref_polygon.intersects(rtc_polygon) or \
               ref_polygon.overlaps(rtc_polygon):
                overlapped_rtc_dir_list.append(input_dir)

    if not overlapped_rtc_dir_list:
        logger.warning('fail to find the overlapped rtc')
        overlapped_rtc_dir_list = None

    return overlapped_rtc_dir_list


def crop_and_save_mgrs_tile(
        source_tif_path,
        output_dir_path,
        output_tif_name,
        output_bbox,
        output_epsg,
        output_format,
        metadata,
        cog_compression,
        cog_nbits,
        interpolation_method='nearest'):
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
        Dictionry for metadata
    cog_compression: str
        Compression method for COG
    cog_nbits: int
        Compression nbits
    interpolation_method : str
        Interpolation method for cropping, by default 'nearest'.
    """
    input_tif_obj = gdal.Open(source_tif_path)
    band = input_tif_obj.GetRasterBand(1)
    no_data_value = band.GetNoDataValue()

    # Retrieve spatial resolution from input TIFF
    xspacing = input_tif_obj.GetGeoTransform()[1]
    yspacing = input_tif_obj.GetGeoTransform()[5]

    # Create output file path
    output_tif_file_path = os.path.join(output_dir_path,
                                        output_tif_name)

    # Define GDAL Warp options
    warp_options = gdal.WarpOptions(
        dstSRS=f'EPSG:{output_epsg}',
        outputType=band.DataType,
        xRes=xspacing,
        yRes=yspacing,
        outputBounds=output_bbox,
        resampleAlg=interpolation_method,
        dstNodata=no_data_value,
        format='GTIFF')

    gdal.Warp(output_tif_file_path,
              source_tif_path,
              options=warp_options)

    _populate_statics_metadata_datasets(metadata,
                                        output_tif_file_path)

    with rasterio.open(output_tif_file_path, 'r+') as src:
        src.update_tags(**metadata)

    input_tif_obj = None

    if output_format == 'COG':
        dswx_sar_util._save_as_cog(
            output_tif_file_path,
            output_dir_path,
            logger,
            compression=cog_compression,
            nbits=cog_nbits)


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
    if track_number is not None:
        vector_gdf = vector_gdf[
            vector_gdf['relative_orbit_number'] ==
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


def get_intersecting_mgrs_tiles_list(image_tif: str):
    """Find and return a list of MGRS tiles
    that intersect a reference GeoTIFF file.

    Parameters
    ----------
    image_tif: str
        Path to the input GeoTIFF file.

    Returns
    ----------
    mgrs_list: list
        List of intersecting MGRS tiles.
    """
    water_meta = dswx_sar_util.get_meta_from_tif(image_tif)

    # extract bounding for images.
    if water_meta['epsg'] == 4326:
        # Convert lat/lon coordinates to UTM

        # create UTM spatial reference
        utm_coordinate_system = osr.SpatialReference()
        utm_coordinate_system.SetUTM(
            water_meta['utmzone'],
            is_northern=water_meta['geotransform'][3] > 0)

        # create geographic (lat/lon) spatial reference
        wgs84_coordinate_system = osr.SpatialReference()
        wgs84_coordinate_system.SetWellKnownGeogCS("WGS84")

        # create transformation of coordinates
        # from UTM to geographic (lat/lon)
        transformation = osr.CoordinateTransformation(
                                wgs84_coordinate_system,
                                utm_coordinate_system)
        x_ul, y_ul, _ = transformation.TransformPoint(
                                water_meta['geotransform'][0],
                                water_meta['geotransform'][3],
                                0)
        x_ur, y_ur, _ = transformation.TransformPoint(
                                water_meta['geotransform'][0] +
                                water_meta['width'] *
                                water_meta['geotransform'][1],
                                water_meta['geotransform'][3], 0)
        x_ll, y_ll, _ = transformation.TransformPoint(
                                water_meta['geotransform'][0],
                                water_meta['geotransform'][3] +
                                water_meta['length'] *
                                water_meta['geotransform'][5], 0)

        x_lr, y_lr, _ = transformation.TransformPoint(
                                water_meta['geotransform'][0] +
                                water_meta['width'] *
                                water_meta['geotransform'][1],
                                water_meta['geotransform'][3] +
                                water_meta['length'] *
                                water_meta['geotransform'][5],
                                0)
        x_extent = [x_ul, x_ur, x_ll, x_lr]
        y_extent = [y_ul, y_ur, y_ll, y_lr]

    else:
        x_extent = [water_meta['geotransform'][0],
                    water_meta['geotransform'][0] +
                    water_meta['width'] *
                    water_meta['geotransform'][1]]
        y_extent = [water_meta['geotransform'][3],
                    water_meta['geotransform'][3] +
                    water_meta['length'] *
                    water_meta['geotransform'][5]]

        # Figure out northern or southern hemisphere
        srs = osr.SpatialReference()            # establish encoding
        srs.ImportFromEPSG(int(water_meta['epsg']))
        dst = osr.SpatialReference()            # establish encoding
        dst.ImportFromEPSG(4326)
        transformation_utm_to_ll = osr.CoordinateTransformation(srs, dst)

    # Get MGRS tile list
    mgrs_list = []
    mgrs_obj = mgrs.MGRS()

    # Search MGRS tiles within bounding box with 5000 m grids.
    for x_cand in range(int(x_extent[0]),
                        int(x_extent[1]),
                        5000):
        for y_cand in range(int(y_extent[0]),
                            int(y_extent[1]),
                            -5000):
            # extract MGRS tile
            lat, lon, _ = transformation_utm_to_ll.TransformPoint(x_cand,
                                                                  y_cand,
                                                                  0)
            mgrs_tile = mgrs_obj.toMGRS(lat, lon)
            mgrs_list.append(mgrs_tile[0:5])

    return list(set(mgrs_list))


def run(cfg):
    '''
    Run save mgrs tiles with parameters in cfg dictionary
    '''
    logger.info('Starting DSWx-S1 save_mgrs_tiles')

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
    dswx_workflow = processing_cfg.dswx_workflow
    hand_mask_value = processing_cfg.hand.mask_value

    # Static ancillary database
    static_ancillary_file_group_cfg = cfg.groups.static_ancillary_file_group
    mgrs_db_path = static_ancillary_file_group_cfg.mgrs_database_file
    mgrs_collection_db_path = \
        static_ancillary_file_group_cfg.mgrs_collection_database_file

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

    shapefile = cfg.groups.dynamic_ancillary_file_group.shoreline_shapefile
    ocean_mask_enabled = processing_cfg.ocean_mask.mask_enabled
    margin_km = processing_cfg.ocean_mask.mask_margin_km
    polygon_water = processing_cfg.ocean_mask.mask_polygon_water

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

    logger.info(f'Number of bursts to process: {num_input_path}')
    date_str_list = []
    for input_dir in input_list:
        # Find HDF5 metadata
        metadata_path_iter = glob.iglob(f'{input_dir}/*{co_pol}*.tif')
        metadata_path = next(metadata_path_iter)
        with rasterio.open(metadata_path) as src:
            tags = src.tags(0)
            date_str = tags['ZERO_DOPPLER_START_TIME']
            platform = tags['PLATFORM']
            track_number = int(tags['TRACK_NUMBER'])
            resolution = src.transform[0]
            date_str_list.append(date_str)

    input_date_format = "%Y-%m-%dT%H:%M:%S"
    output_date_format = "%Y%m%dT%H%M%SZ"

    date_str_id_temp = date_str_list[0][:19]
    date_str_id = datetime.datetime.strptime(
        date_str_id_temp, input_date_format).strftime(
            output_date_format)
    platform_str = platform[0] + platform.split('-')[-1]
    resolution_str = str(int(resolution))

    if processing_cfg.inundated_vegetation.enabled == 'auto':
        if cross_pol and co_pol:
            total_inundated_vege_flag = True
        else:
            total_inundated_vege_flag = False
    else:
        total_inundated_vege_flag = \
            processing_cfg.inundated_vegetation.enabled

    inundated_vege_mosaic_flag = False
    # Set merge_layer_flag and merge_pol_list based on pol_mode
    merge_layer_flag = pol_mode.startswith('MIX')
    if merge_layer_flag:
        if pol_mode == 'MIX_DUAL_POL':
            pol_type1 = 'DV_POL'
            pol_type2 = 'DH_POL'
        elif pol_mode in 'MIX_DUAL_H_SINGLE_V_POL':
            pol_type1 = 'DH_POL'
            pol_type2 = 'SV_POL'
        elif pol_mode in 'MIX_DUAL_V_SINGLE_H_POL':
            pol_type1 = 'DV_POL'
            pol_type2 = 'SH_POL'
        elif pol_mode in 'MIX_SINGLE_POL':
            pol_type1 = 'SV_POL'
            pol_type2 = 'SH_POL'
        else:
            logger.info('There is no need to mosaic different polarizations.')
        pol_set1 = DSWX_S1_POL_DICT[pol_type1]
        pol_set2 = DSWX_S1_POL_DICT[pol_type2]
        merge_pol_list = ['_'.join(pol_set1),
                          '_'.join(pol_set2)]
    else:
        pol_set1 = pol_list
        pol_set2 = []
    # If polarimetric methods such as ratio, span are used,
    # it is added to the name.
    if pol_options is not None and merge_layer_flag:

        merge_pol_list = [item + '_' + pol_option_str
                          for item in merge_pol_list]

    # Depending on the workflow, the final product are different.
    prefix_dict = {
        'final_water': 'bimodality_output_binary'
        if dswx_workflow == 'opera_dswx_s1'
        else 'region_growing_output_binary',
        'landcover_mask': 'refine_landcover_binary',
        'no_data_area': 'no_data_area',
        'region_growing': 'region_growing_output_binary',
        'fuzzy_value': 'fuzzy_image'
        }

    if total_inundated_vege_flag:
        if len(pol_set1) == 2 and len(pol_set2) == 2:
            inundated_vege_mosaic_flag = True

    if total_inundated_vege_flag:
        prefix_dict['inundated_veg'] = 'temp_inundated_vegetation'

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
            if not inundated_vege_mosaic_flag and key == 'inundated_veg':
                dual_pol_vege_string = '_'.join(pol_set1)
                paths[key] = os.path.join(
                    scratch_dir, f'{prefix}_{dual_pol_vege_string}.tif')
            else:
                merge_pol_layers(list_layers,
                                 os.path.join(scratch_dir, file_path),
                                 **extra_args)

    # metadata for final product
    # e.g. geotransform, projection, length, width, utmzon, epsg
    water_meta = dswx_sar_util.get_meta_from_tif(paths['final_water'])

    # repackage the water map
    # 1) water map
    water_map = dswx_sar_util.read_geotiff(paths['final_water'])
    no_data_raster = dswx_sar_util.read_geotiff(paths['no_data_area'])
    no_data_raster = (no_data_raster > 0) | \
        (water_map == band_assign_value_dict['no_data'])

    # 2) layover/shadow
    layover_shadow_mask_path = \
        os.path.join(scratch_dir, 'mosaic_layovershadow_mask.tif')

    if os.path.exists(layover_shadow_mask_path):
        layover_shadow_mask = \
            dswx_sar_util.read_geotiff(layover_shadow_mask_path)
        logger.info('Layover/shadow mask found')
    else:
        layover_shadow_mask = np.zeros(np.shape(water_map), dtype='byte')
        logger.warning('No layover/shadow mask found')

    # 3) hand excluded
    hand = dswx_sar_util.read_geotiff(
        os.path.join(scratch_dir, 'interpolated_hand.tif'))
    hand_mask = hand > hand_mask_value

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
        inundated_vegetation = dswx_sar_util.read_geotiff(
            paths['inundated_veg'])
        inundated_vegetation_mask = (inundated_vegetation == 2) & \
                                    (water_map == 1)
        inundated_vegetation[inundated_vegetation_mask] = 1
        logger.info('Inudated vegetation file was found.')
    else:
        inundated_vegetation = None
        logger.warning('Inudated vegetation file was disabled.')

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

    if dswx_workflow == 'opera_dswx_s1':
        logger.info('BWTR and WTR Files are created from pre-computed files.')

        region_grow_map = \
            dswx_sar_util.read_geotiff(paths['region_growing'])
        landcover_map =\
            dswx_sar_util.read_geotiff(paths['landcover_mask'])

        landcover_mask = (region_grow_map == 1) & (landcover_map != 1)
        dark_land_mask = (landcover_map == 1) & (water_map == 0)
        bright_water_mask = (landcover_map == 0) & (water_map == 1)

        # Open water/inundated vegetation
        # layover shadow mask/hand mask/no_data
        # will be saved in WTR product
        dswx_sar_util.save_dswx_product(
            water_map == 1,
            full_wtr_water_set_path,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Water classification (WTR)',
            scratch_dir=scratch_dir,
            logger=logger,
            layover_shadow_mask=layover_shadow_mask > 0,
            hand_mask=hand_mask,
            inundated_vegetation=inundated_vegetation == 2,
            no_data=no_data_raster,
            ocean_mask=ocean_mask
            )

        # water/ No-water
        # layover shadow mask/hand mask/no_data
        # will be saved in BWTR product
        # Water includes open water and inundated vegetation.
        dswx_sar_util.save_dswx_product(
            np.logical_or(water_map == 1, inundated_vegetation == 2),
            full_bwtr_water_set_path,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Binary Water classification (BWTR)',
            scratch_dir=scratch_dir,
            logger=logger,
            layover_shadow_mask=layover_shadow_mask > 0,
            hand_mask=hand_mask,
            ocean_mask=ocean_mask,
            no_data=no_data_raster)

        # Open water/landcover mask/bright water/dark land
        # layover shadow mask/hand mask/inundated vegetation
        # will be saved in CONF product
        dswx_sar_util.save_dswx_product(
            water_map == 1,
            full_conf_water_set_path,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Confidence values (CONF)',
            scratch_dir=scratch_dir,
            logger=logger,
            landcover_mask=landcover_mask,
            bright_water_fill=bright_water_mask,
            dark_land_mask=dark_land_mask,
            layover_shadow_mask=layover_shadow_mask > 0,
            hand_mask=hand_mask,
            ocean_mask=ocean_mask,
            inundated_vegetation=inundated_vegetation == 2,
            no_data=no_data_raster,
            )

        # Values ranging from 0 to 100 are used to represent the likelihood
        # or possibility of the presence of water. A higher value within
        # this range signifies a higher likelihood of water being present.
        fuzzy_value = dswx_sar_util.read_geotiff(paths['fuzzy_value'])
        fuzzy_value = np.round(fuzzy_value * 100)
        dswx_sar_util.save_dswx_product(
            fuzzy_value,
            full_diag_water_set_path,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Diagnostic layer (DIAG)',
            is_diag=True,
            scratch_dir=scratch_dir,
            datatype='uint8',
            logger=logger,
            layover_shadow_mask=layover_shadow_mask > 0,
            hand_mask=hand_mask,
            ocean_mask=ocean_mask,
            no_data=no_data_raster)
    else:

        # In Twele's workflow, bright water/dark land/inundated vegetation
        # is not saved.
        dswx_sar_util.save_dswx_product(
                water_map == 1,
                full_wtr_water_set_path,
                geotransform=water_meta['geotransform'],
                projection=water_meta['projection'],
                description='Water classification (WTR)',
                scratch_dir=scratch_dir,
                layover_shadow_mask=layover_shadow_mask > 0,
                hand_mask=hand_mask,
                no_data=no_data_raster)

    # Get list of MGRS tiles overlapped with mosaic RTC image
    mgrs_meta_dict = {}

    if database_bool:
        mgrs_tile_list, most_overlapped = \
            get_intersecting_mgrs_tiles_list_from_db(
                mgrs_collection_file=mgrs_collection_db_path,
                image_tif=paths['final_water'],
                track_number=track_number
                )
        maximum_burst = most_overlapped['number_of_bursts']
        # convert string to list
        expected_burst_list = ast.literal_eval(most_overlapped['bursts'])
        logger.info(f"Input RTCs are within {most_overlapped['mgrs_set_id']}")
        actual_burst_id = collect_burst_id(input_list,
                                           DSWX_S1_POL_DICT['CO_POL'])
        number_burst = len(actual_burst_id)
        mgrs_meta_dict['MGRS_COLLECTION_EXPECTED_NUMBER_OF_BURSTS'] = \
            maximum_burst
        mgrs_meta_dict['MGRS_COLLECTION_ACTUAL_NUMBER_OF_BURSTS'] = \
            number_burst
        missing_burst = len(list(set(expected_burst_list) -
                                 set(actual_burst_id)))
        mgrs_meta_dict['MGRS_COLLECTION_MISSING_NUMBER_OF_BURSTS'] = \
            missing_burst

    else:
        mgrs_tile_list = get_intersecting_mgrs_tiles_list(
            image_tif=paths['final_water'])

    unique_mgrs_tile_list = list(set(mgrs_tile_list))
    logger.info(f'MGRS tiles: {unique_mgrs_tile_list}')

    processing_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ")
    if dswx_workflow == 'opera_dswx_s1':

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
            mgrs_bbox = [x_value_min, y_value_min, x_value_max, y_value_max]
            overlapped_burst = find_intersecting_burst_with_bbox(
                ref_bbox=mgrs_bbox,
                ref_epsg=epsg_output,
                input_rtc_dirs=input_list)
            logger.info(f'overlapped_bursts: {overlapped_burst}')

            # Metadata
            if overlapped_burst:
                dswx_metadata_dict = create_dswx_sar_metadata(
                    cfg,
                    overlapped_burst,
                    product_version=product_version,
                    extra_meta_data=mgrs_meta_dict)

                dswx_name_format_prefix = (f'OPERA_L3_DSWx-S1_T{mgrs_tile_id}_'
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
                output_mgrs_browse = f'{dswx_name_format_prefix}_BROWSE.png'

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
                    crop_and_save_mgrs_tile(
                        source_tif_path=full_input_file_path,
                        output_dir_path=sas_outputdir,
                        output_tif_name=output_file_path,
                        output_bbox=mgrs_bbox,
                        output_epsg=epsg_output,
                        output_format=output_imagery_format,
                        metadata=dswx_metadata_dict,
                        cog_compression=output_imagery_compression,
                        cog_nbits=output_imagery_nbits,
                        interpolation_method='nearest')

                if browse_image_flag:
                    dswx_sar_util.create_browse_image(
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
                        set_ocean_masked_to_nodata=set_ocean_masked_to_nodata)

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
        cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_s1', args)

    generate_log.configure_log_file(cfg.groups.log_file)

    run(cfg)


if __name__ == '__main__':
    main()
