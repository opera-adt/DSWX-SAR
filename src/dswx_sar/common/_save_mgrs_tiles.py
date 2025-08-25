import ast
import logging

import geopandas as gpd

import mgrs
import numpy as np

import rasterio
from rasterio.warp import transform_bounds
from rasterio.merge import merge

from shapely.geometry import Polygon
from shapely.ops import transform
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

from dswx_sar.common import _dswx_sar_util

logger = logging.getLogger('dswx_sar')


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
        Minimum x coordinate (UTM) for the given MGRS tile
    maxx: float
        Maximum x coordinate (UTM) for the given MGRS tile
    miny: float
        Minimum y coordinate (UTM) for the given MGRS tile
    maxy: float
        Maximum y coordinate (UTM) for the given MGRS tile
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
        Minimum x coordinate (UTM) for the given MGRS tile
    maxx: float
        Maximum x coordinate (UTM) for the given MGRS tile
    miny: float
        Minimum y coordinate (UTM) for the given MGRS tile
    maxy: float
        Maximum y coordinate (UTM) for the given MGRS tile
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


def get_intersecting_mgrs_tiles_list_from_db(
        image_tif,
        mgrs_collection_file,
        track_number=None,
        burst_list=None):
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
    burst_list : list, optional
        List of burst IDs to filter the MGRS tiles.

    Returns
    -------
    mgrs_list: list
        List of intersecting MGRS tiles.
    most_overlapped : GeoSeries
        The record of the MGRS tile with the maximum overlap area.
    """
    vector_gdf = gpd.read_file(mgrs_collection_file)
    # Step 1: Filter by burst_list if provided
    if burst_list is not None:
        def burst_overlap(row):
            row_bursts = ast.literal_eval(row['bursts'])
            return len(set(burst_list).intersection(set(row_bursts)))

        vector_gdf['burst_overlap_count'] = vector_gdf.apply(burst_overlap, axis=1)
        max_burst_overlap = vector_gdf['burst_overlap_count'].max()
        vector_gdf = vector_gdf[vector_gdf['burst_overlap_count'] == max_burst_overlap]

        # If only one record matches, return it immediately
        if len(vector_gdf) == 1:
            logger.info(f"MGRS collection ID found from burst_list {vector_gdf}")
            mgrs_list = ast.literal_eval(vector_gdf.iloc[0]['mgrs_tiles'])
            return list(set(mgrs_list)), vector_gdf.iloc[0]

    # Step 2: Filter by track_number, track_number + 1, and track_number - 1 if provided
    if track_number is not None:
        valid_track_numbers = [track_number, track_number + 1, track_number - 1]
        vector_gdf = vector_gdf[
            vector_gdf['relative_orbit_number'].isin(valid_track_numbers)
        ].to_crs("EPSG:4326")

        # If only one record matches, return it immediately
        if len(vector_gdf) == 1:
            mgrs_list = ast.literal_eval(vector_gdf.iloc[0]['mgrs_tiles'])
            return list(set(mgrs_list)), vector_gdf.iloc[0]
    else:
        vector_gdf = vector_gdf.to_crs("EPSG:4326")

    # Load the raster data
    with rasterio.open(image_tif) as src:
        epsg_code = src.crs.to_epsg() or 4326
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
        raster_polygon = Polygon(
            [(left, bottom),
             (left, top),
             (right, top),
             (right, bottom)])
        raster_gdf = gpd.GeoDataFrame([1],
                                      geometry=[raster_polygon],
                                      crs=4326)

    # Calculate the intersection
    intersection = gpd.overlay(raster_gdf,
                               vector_gdf,
                               how='intersection')

    # Add a new column with the intersection area
    intersection['Area'] = intersection.to_crs(epsg=epsg_code).geometry.area
    sorted_intersection = intersection.sort_values(by='Area', ascending=False)

    most_overlapped = sorted_intersection.iloc[0] if len(sorted_intersection) > 0 else None
    mgrs_list = ast.literal_eval(most_overlapped['mgrs_tiles']) if most_overlapped is not None else []

    return list(set(mgrs_list)), most_overlapped

def merge_pol_layers(list_layers,
                     output_file,
                     nodata_value=None,
                     scratch_dir='.'):
    """
    Merge multiple GeoTIFF files into a single file using rasterio.
    This function is used to merge the GeoTIFF with different polarizations.

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

    _dswx_sar_util._save_as_cog(output_file, scratch_dir)

def get_intersecting_mgrs_tiles_list(image_tif: str):
    """Find and return a list of MGRS tiles
    that intersect a reference GeoTIFF file.

    Parameters
    ----------
    image_tif: str
        Path to the input GeoTIFF file.

    Returns
    -------
    mgrs_list: list
        List of intersecting MGRS tiles.
    """
    water_meta = _dswx_sar_util.get_meta_from_tif(image_tif)

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
