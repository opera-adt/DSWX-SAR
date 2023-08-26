import os
import time
import glob
import logging
import mimetypes
import ast
import datetime

import h5py
import mgrs
import rasterio
import numpy as np
import geopandas as gpd
from osgeo import gdal, osr
from pyproj import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import Polygon

from dswx_sar import dswx_sar_util
from dswx_sar.dswx_runconfig import _get_parser, RunConfig
from dswx_sar import generate_log

logger = logging.getLogger('dswx_s1')

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


def get_bounding_box_from_mgrs_tile(
        mgrs_tile_name):
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

def crop_and_save_mgrs_tile(
        mgrs_db_path,
        source_tif_path,
        output_dir_path,
        output_tif_name,
        output_mgrs_id,
        output_format,
        cog_compression,
        cog_nbits,
        interpolation_method='nearest',
        ):
    """Crop the product along the MGRS tile grid and
    save it as a Cloud-Optimized GeoTIFF (COG).

    Parameters
    ----------
    mgrs_db_path: str
        Path to MGRS database.
    source_tif_path : str
        Path to the original TIFF file.
    output_dir_path : str
        Path to the directory to save the output file.
    output_tif_name : str
        Filename for the cropped TIFF.
    output_mgrs_id : str
        MGRS tile grid ID (e.g., "10SEG").
    interpolation_method : str
        Interpolation method for cropping, by default 'nearest'.
    """
    input_tif_obj = gdal.Open(source_tif_path)
    band = input_tif_obj.GetRasterBand(1)

    # Retrieve spatial resolution from input TIFF
    xspacing = input_tif_obj.GetGeoTransform()[1]
    yspacing = input_tif_obj.GetGeoTransform()[5]

    # compute EPSG for given mgrs_tile_id
    # Get bounding extents for the given MGRS tile
    if mgrs_db_path is None:
        x_value_min, x_value_max, y_value_min, y_value_max, epsg_output = \
            get_bounding_box_from_mgrs_tile(output_mgrs_id)
    else:
        x_value_min, x_value_max, y_value_min, y_value_max, epsg_output = \
            get_bounding_box_from_mgrs_tile_db(output_mgrs_id,
                                            mgrs_db_path,)

    bbox = [x_value_min, y_value_min, x_value_max, y_value_max]
    # Create output file path
    output_tif_file_path = os.path.join(output_dir_path,
                                        output_tif_name)

    # Define GDAL Warp options
    warp_options = gdal.WarpOptions(
        dstSRS=f'EPSG:{epsg_output}',
        outputType=band.DataType,
        xRes=xspacing,
        yRes=yspacing,
        outputBounds=bbox,
        resampleAlg=interpolation_method,
        format='GTIFF')

    gdal.Warp(output_tif_file_path,
              source_tif_path,
              options=warp_options)
    input_tif_obj = None
    if  output_format == 'COG':
        dswx_sar_util._save_as_cog(
            output_tif_file_path,
            output_dir_path,
            logger,
            compression=cog_compression,
            nbits=cog_nbits)


def get_intersecting_mgrs_tiles_list_from_db(
        image_tif,
        mgrs_collection_file):
    """Find and return a list of MGRS tiles that intersect a reference GeoTIFF file

    Parameters
    ----------
    image_tif: str
        Path to the input GeoTIFF file.
    mgrs_collection_file : str
        Path to the MGRS tile collection.

    Returns
    ----------
    mgrs_list: list
        List of intersecting MGRS tiles.
    """
    # Load the raster data
    with rasterio.open(image_tif) as src:
        epsg_code = int(src.crs.data['init'].split(':')[1])

        # Get bounds of the raster data
        left, bottom, right, top = src.bounds

        # Reproject to EPSG 4326 if the current EPSG is not 4326
        if epsg_code != 4326:
            left, bottom, right, top = transform_bounds(
                                                src.crs,
                                                {'init': 'EPSG:4326'},
                                                left,
                                                bottom,
                                                right,
                                                top)

    # Create a Polygon from the bounds
    raster_polygon = Polygon([(left, bottom),
                                (left, top),
                                (right, top),
                                (right, bottom)])

    # Create a GeoDataFrame from the raster polygon
    raster_gdf = gpd.GeoDataFrame([1],
                                    geometry=[raster_polygon],
                                    crs={'init': 'EPSG:4326'})

    # Load the vector data
    vector_gdf = gpd.read_file(mgrs_collection_file)

    # Calculate the intersection
    intersection = gpd.overlay(raster_gdf,
                                vector_gdf,
                                how='intersection')

    # Add a new column with the intersection area
    intersection['Area'] = intersection.geometry.area

    # Find the polygon with the maximum intersection area
    most_overlapped = intersection.loc[intersection['Area'].idxmax()]

    mgrs_list = ast.literal_eval(most_overlapped['mgrs_tiles'])

    return list(set(mgrs_list))


def get_intersecting_mgrs_tiles_list(image_tif: str):
    """Find and return a list of MGRS tiles that intersect a reference GeoTIFF file

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
            is_northern=water_meta['geotransform'][3]>0)

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
            lat, lon, _ = transformation_utm_to_ll.TransformPoint(x_cand, y_cand, 0)
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
    outputdir = product_path_group_cfg.scratch_path
    sas_outputdir = product_path_group_cfg.sas_output_path
    product_version = product_path_group_cfg.product_version
    output_imagery_format = product_path_group_cfg.output_imagery_format
    output_imagery_compression = product_path_group_cfg.output_imagery_compression
    output_imagery_nbits = product_path_group_cfg.output_imagery_nbits

    processing_cfg = cfg.groups.processing
    pol_str = '_'.join(processing_cfg.polarizations)
    input_list = cfg.groups.input_file_group.input_file_path
    dswx_workflow = processing_cfg.dswx_workflow

    static_ancillary_file_group_cfg = cfg.groups.static_ancillary_file_group
    mgrs_db_path = static_ancillary_file_group_cfg.mgrs_database_file
    mgrs_collection_db_path = \
        static_ancillary_file_group_cfg.mgrs_collection_database_file

    if product_version is None:
        logger.warning('WARNING: product version was not provided.')

    if mgrs_db_path is not None and mgrs_collection_db_path is not None:
        logger.info('Both the MGRS tile or the MGRS collection database were provided.')
        database_bool = True
    else:
        logger.warning('WARNING: Either the MGRS tile or the MGRS collection database was not provided.')
        database_bool = False

    os.makedirs(sas_outputdir, exist_ok=True)

    num_input_path = len(input_list)
    mosaic_flag = num_input_path > 1

    #[TODO] In future, acquisition date and time will be calcuated
    # from the intersecting MGRS tile and bursts.
    if mosaic_flag:
        print('Number of bursts to process:', num_input_path)
        id_path = '/identification/'

        date_str_list = []
        for input_dir in input_list:
            # find HDF5 metadata
            metadata_path = glob.glob(f'{input_dir}/*h5')[0]
            with h5py.File(metadata_path) as meta_h5:
                date_str = meta_h5[f'{id_path}/zeroDopplerStartTime'][()]
            date_str_list.append(date_str)
        date_str_list = [x.decode() for x in date_str_list]

        input_date_format = "%Y-%m-%dT%H:%M:%S"
        output_date_format = "%Y%m%dT%H%M%SZ"

        date_str_id_temp = date_str_list[0][:19]
        date_str_id = datetime.datetime.strptime(
            date_str_id_temp, input_date_format).strftime(
                output_date_format)

    else:
        date_str_id = 'unknown'

    # [TODO] Final product has different name depending on the workflow
    if dswx_workflow == 'opera_dswx_s1':
        final_water_path = \
            os.path.join(outputdir,
                         f'bimodality_output_binary_{pol_str}.tif')
    else:
        # twele's workflow
        final_water_path = \
            os.path.join(outputdir,
                         f'region_growing_output_binary_{pol_str}.tif')

    # metadata for final product
    # e.g. geotransform, projection, length, width, utmzon, epsg
    water_meta = dswx_sar_util.get_meta_from_tif(final_water_path)

    # repackage the water map
    # 1) water map
    water_map = dswx_sar_util.read_geotiff(final_water_path)
    no_data_raster = water_map == 255

    # 2) layover/shadow
    layover_shadow_mask_path = \
        os.path.join(outputdir, 'mosaic_layovershadow_mask.tif')

    if os.path.exists(layover_shadow_mask_path):
        layover_shadow_mask = \
            dswx_sar_util.read_geotiff(layover_shadow_mask_path)
        logger.info('Layover/shadow mask found')
    else:
        layover_shadow_mask = np.zeros(np.shape(water_map), dtype='byte')
        logger.warning('No layover/shadow mask found')

    # 3) hand 200 m : need to populate the value from runconfig.
    hand = dswx_sar_util.read_geotiff(
        os.path.join(outputdir, 'interpolated_hand'))
    hand_mask = hand > 200

    full_wtr_water_set_path = \
        os.path.join(outputdir, 'full_water_binary_WTR_set.tif')
    full_bwtr_water_set_path = \
        os.path.join(outputdir, 'full_water_binary_BWTR_set.tif')
    full_conf_water_set_path = \
        os.path.join(outputdir, f'fuzzy_image_{pol_str}.tif')

    # 4) inundated_vegetation
    if processing_cfg.inundated_vegetation.enabled:
        inundated_vegetation = dswx_sar_util.read_geotiff(
            os.path.join(outputdir, "temp_inundated_vegetation.tif"))
        inundated_vegetation_mask = (inundated_vegetation == 2) & \
                                    (water_map == 1)
        inundated_vegetation[inundated_vegetation_mask] = 1
        logger.info('Inudated vegetation file was found.')
    else:
        inundated_vegetation = None
        logger.warning('Inudated vegetation file was disabled.')

    if dswx_workflow == 'opera_dswx_s1':
        logger.info('BWTR and WTR Files are created from pre-computed files.')

        region_grow_map = \
            dswx_sar_util.read_geotiff(
                os.path.join(outputdir,
                             f'region_growing_output_binary_{pol_str}.tif'))
        landcover_map =\
            dswx_sar_util.read_geotiff(
                os.path.join(outputdir,
                             f'refine_landcover_binary_{pol_str}.tif'))
        landcover_mask = (region_grow_map == 1) & (landcover_map != 1)
        dark_land_mask = (landcover_map == 1) & (water_map == 0)
        bright_water_mask = (landcover_map == 0) & (water_map == 1)
        # Open water/landcover mask/bright water/dark land
        # layover shadow mask/hand mask/inundated vegetation
        # will be saved in WTR product
        dswx_sar_util.save_dswx_product(
            water_map == 1,
            full_wtr_water_set_path,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Water classification (WTR)',
            scratch_dir=outputdir,
            landcover_mask=landcover_mask,
            bright_water_fill=bright_water_mask,
            dark_land_mask=dark_land_mask,
            layover_shadow_mask=layover_shadow_mask > 0,
            hand_mask=hand_mask,
            inundated_vegetation=inundated_vegetation == 2,
            no_data=no_data_raster,
            )
        # Open water/layover shadow mask/hand mask/
        # inundated vegetation will be saved in WTR product
        dswx_sar_util.save_dswx_product(
            water_map == 1,
            full_bwtr_water_set_path,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Binary Water classification (BWTR)',
            scratch_dir=outputdir,
            inundated_vegetation=inundated_vegetation == 2,
            layover_shadow_mask=layover_shadow_mask > 0,
            hand_mask=hand_mask,
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
                scratch_dir=outputdir,
                layover_shadow_mask=layover_shadow_mask > 0,
                hand_mask=hand_mask,
                no_data=no_data_raster)

    # Get MGRS tile list
    if database_bool:
        mgrs_tile_list = get_intersecting_mgrs_tiles_list_from_db(
            mgrs_collection_file=mgrs_collection_db_path,
            image_tif=final_water_path)
    else:
        mgrs_tile_list = get_intersecting_mgrs_tiles_list(
            image_tif=final_water_path)

    unique_mgrs_tile_list = list(set(mgrs_tile_list))
    logger.info(f'MGRS tiles: {unique_mgrs_tile_list}')

    processing_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ")
    if dswx_workflow == 'opera_dswx_s1':

        for mgrs_tile_id in unique_mgrs_tile_list:
            # [TODO] specify file name
            logger.info(f'mgrs tile : {mgrs_tile_id}')

            dswx_name_format_prefix = \
                f'OPERA_L3_DSWx-S1_T{mgrs_tile_id}_{date_str_id}_{processing_time}_v{product_version}'
            logger.info('Saving the file:')
            logger.info(f'      {dswx_name_format_prefix}')
            # Output File names
            output_mgrs_bwtr = f'{dswx_name_format_prefix}_B01_BWTR.tif'
            output_mgrs_wtr = f'{dswx_name_format_prefix}_B02_WTR.tif'
            output_mgrs_conf = f'{dswx_name_format_prefix}_B03_CONF.tif'

            # Crop full size of BWTR, WTR, CONF file
            # and save them into MGRS tile grid
            full_input_file_paths = [full_bwtr_water_set_path,
                                     full_wtr_water_set_path,
                                     full_conf_water_set_path]

            full_output_file_paths = [output_mgrs_bwtr,
                                     output_mgrs_wtr,
                                     output_mgrs_conf]

            for full_input_file_path, full_output_file_path in zip(
                full_input_file_paths, full_output_file_paths
            ):
                crop_and_save_mgrs_tile(
                    mgrs_db_path=mgrs_db_path,
                    source_tif_path=full_input_file_path,
                    output_dir_path=sas_outputdir,
                    output_tif_name=full_output_file_path,
                    output_mgrs_id=mgrs_tile_id,
                    output_format=output_imagery_format,
                    cog_compression=output_imagery_compression,
                    cog_nbits=output_imagery_nbits,
                    interpolation_method='nearest')

    t_all_elapsed = time.time() - t_all
    logger.info(f"successfully ran save_mgrs_tiles in {t_all_elapsed:.3f} seconds")

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