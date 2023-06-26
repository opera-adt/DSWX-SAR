import os
import time
import mgrs
import logging
import mimetypes
import h5py
import glob
import ast
import rasterio
import numpy as np
import geopandas as gpd
import datetime

from osgeo import gdal, osr
from pyproj import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import Polygon

from dswx_sar.version import VERSION as software_version
from dswx_sar import dswx_sar_util
from dswx_sar.dswx_runconfig import _get_parser, RunConfig


logger = logging.getLogger('dswx_S1')

def get_geographic_boundaries_from_mgrs_tile(mgrs_tile_name,
                                             mgrs_db_path=None):
    """Get geographic extents from given MGRS tile
    Parameters
    ----------
    mgrs_tile_name: str
        Name of MGRS tile (ex. 18LWQ)
    mgrs_db_path : str
        path for MGRS database

    Returns
    -------
    minx: float
        minimum x cooridate (UTM) for given MGRS tile
    maxx: float
        maximum x cooridate (UTM) for given MGRS tile
    miny: float
        minimum y cooridate (UTM) for given MGRS tile
    maxy: float
        maximum y cooridate (UTM) for given MGRS tile
    epsg: int
        epsg code
    """
    # if mgrs tile database is not given,
    # return the coordinates based on python mgrs
    if mgrs_db_path is None:
        mgrs_obj = mgrs.MGRS()
        lower_left_utm_coordinate = mgrs_obj.MGRSToUTM(mgrs_tile_name)

        x_min = lower_left_utm_coordinate[2]
        y_min = lower_left_utm_coordinate[3]

        if lower_left_utm_coordinate[1] == 'N':
            is_southern = False
        else:
            is_southern = True

        crs = CRS.from_dict({'proj': 'utm',
                             'zone': lower_left_utm_coordinate[0],
                             'south': is_southern})

        epsg = crs.to_authority()[1]

        # compute boundaries
        x_list = []
        y_list = []
        for offset_x_multiplier in range(2):
            for offset_y_multiplier in range(2):
                # additional coordinates to compute min/max of x/y
                x_more = x_min + offset_x_multiplier * 109.8 * 1000
                y_more = y_min + offset_y_multiplier * 109.8 * 1000
                x_list.append(x_more)
                y_list.append(y_more)

        minx = np.min(x_list)
        maxx = np.max(x_list)
        miny = np.min(y_list)
        maxy = np.max(y_list)

    else:
        # Load the vector data
        vector_gdf = gpd.read_file(mgrs_db_path)

        # filtered mgrs
        filtered_gdf = vector_gdf[vector_gdf['mgrs_tile'] == mgrs_tile_name]

        # Get the bounds for each polygon
        minx = filtered_gdf['xmin'].values[0]
        maxx = filtered_gdf['xmax'].values[0]
        miny = filtered_gdf['ymin'].values[0]
        maxy = filtered_gdf['ymax'].values[0]
        epsg = filtered_gdf['epsg'].values[0]

    return minx, maxx, miny, maxy, epsg

def save_mgrs_tile_db(mgrs_db_path,
                      source_tif_path,
                      output_dir_path,
                      output_tif_name,
                      output_mgrs_id,
                      method='nearest'):
    """Crop image along the MGRS tile grid and save as COG
    Parameters
    ----------
    mgrs_db_path: str
        path of MGRS database
    source_tif_path : str
        path for original tif file
    output_dir_path : str
        path for directory to save the file
    output_tif_name : str
        cropped tif filename
    output_mgrs_id : str
        MGRS tile grid ID (e.g. 10SEG)
    method : str
        interpolation method.
    """
    input_tif_obj = gdal.Open(source_tif_path)
    band = input_tif_obj.GetRasterBand(1)

    # spatial resolution for input and output are same.
    xspacing = input_tif_obj.GetGeoTransform()[1]
    yspacing = input_tif_obj.GetGeoTransform()[5]

    # compute EPSG for given mgrs_tile_id
    # Compute bounding extents for given MGRS tile

    x_value_min, x_value_max, y_value_min, y_value_max, epsg_output = \
        get_geographic_boundaries_from_mgrs_tile(output_mgrs_id,
                                                 mgrs_db_path,)

    bbox = [x_value_min, y_value_min, x_value_max, y_value_max]

    # create MGRS tiles for products.
    output_tif_file_path = f'{output_dir_path}/{output_tif_name}'
    opt = gdal.WarpOptions(dstSRS=f'EPSG:{epsg_output}',
                     outputType=band.DataType,
                     xRes=xspacing,
                     yRes=yspacing,
                     outputBounds=bbox,
                     resampleAlg=method,
                     format='GTIFF')
    gdal.Warp(output_tif_file_path, source_tif_path, options=opt)
    input_tif_obj = None
    dswx_sar_util._save_as_cog(output_tif_file_path, output_dir_path, logger,
                compression='ZSTD',
                nbits=16)

def find_mgrs_collection(image_tif,
                         mgrs_collection_file=None):
    """Find MGRS tile collection intersecting Geotiff
    Parameters
    ----------
    image_tif: str
        path of input geotiff
    mgrs_collection_file : str
        path for MGRS tile collection

    Returns
    ----------
    mgrs_list: list
        list of MGRS tiles
    """
    if mgrs_collection_file is not None:
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

    else:
        water_meta = dswx_sar_util.get_meta_from_tif(image_tif)

        # extract bounding for images.
        if water_meta['epsg'] == 4326:
            # For lat/lon coordinate, convert them to UTM.

            # create UTM spatial reference
            utm_coordinate_system = osr.SpatialReference()
            if water_meta['geotransform'][3]>0:
                utm_coordinate_system.SetUTM(water_meta['utmzone'],
                                             is_northern=True)
            else:
                utm_coordinate_system.SetUTM(water_meta['utmzone'],
                                             is_northern=False)

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
                                    water_meta['geotransform'][0] + \
                                    water_meta['width'] * \
                                    water_meta['geotransform'][1],
                                    water_meta['geotransform'][3], 0)
            x_ll, y_ll, _ = transformation.TransformPoint(
                                    water_meta['geotransform'][0],
                                    water_meta['geotransform'][3] + \
                                    water_meta['length'] * \
                                    water_meta['geotransform'][5], 0)

            x_lr, y_lr, _ = transformation.TransformPoint(
                                    water_meta['geotransform'][0] + \
                                    water_meta['width'] * \
                                    water_meta['geotransform'][1],
                                    water_meta['geotransform'][3] + \
                                    water_meta['length'] * \
                                    water_meta['geotransform'][5],
                                    0)
            x_extent = [x_ul, x_ur, x_ll, x_lr]
            y_extent = [y_ul, y_ur, y_ll, y_lr]

        else:
            x_extent = [water_meta['geotransform'][0],
                        water_meta['geotransform'][0] + \
                        water_meta['width'] * \
                        water_meta['geotransform'][1]]
            y_extent = [water_meta['geotransform'][3],
                        water_meta['geotransform'][3] + \
                        water_meta['length'] * \
                        water_meta['geotransform'][5]]

        # figure out norther or southern hemisphere
        if water_meta['epsg'] != 4326:
            srs = osr.SpatialReference()            # establish encoding
            srs.ImportFromEPSG(int(water_meta['epsg']))
            dst = osr.SpatialReference()            # establish encoding
            dst.ImportFromEPSG(4326)
            transformation = osr.CoordinateTransformation(srs, dst)
            lat, lon, _ = transformation.TransformPoint(x_extent[0],
                                                        y_extent[0], 0)

        # get MGRS tile list
        mgrs_list = []
        mgrs_obj = mgrs.MGRS()

        for x_cand in range(int(x_extent[0]), int(x_extent[1]), int(109800/4)):
            for y_cand in range(int(y_extent[0]), int(y_extent[1]), -int(109800/4)):
                # extract MGRS tile
                lat, lon, _ = transformation.TransformPoint(x_cand, y_cand, 0)
                mgrs_tile = mgrs_obj.toMGRS(lat, lon)
                mgrs_list.append(mgrs_tile[0:5])

    return  mgrs_list

def run(cfg):
    '''
    Run save mgrs tiles with parameters in cfg dictionary
    '''
    logger.info('Starting DSWx-S1 save_mgrs_tiles')

    t_all = time.time()
    outputdir = cfg.groups.product_path_group.scratch_path
    sas_outputdir = cfg.groups.product_path_group.sas_output_path
    processing_cfg = cfg.groups.processing
    pol_list = processing_cfg.polarizations
    pol_str = '_'.join(pol_list)
    input_list = cfg.groups.input_file_group.input_file_path
    dswx_workflow = processing_cfg.dswx_workflow
    product_version = cfg.groups.product_path_group.product_version
    static_layer_cfg = cfg.groups.static_ancillary_file_group
    mgrs_db_path = static_layer_cfg.mgrs_database_file
    mgrs_collection_db_path = static_layer_cfg.mgrs_collection_database_file

    if mgrs_db_path is None or mgrs_collection_db_path is None:
        # if one of databases does not exist,
        # disable the other database.
        logger.info('One of database is not available')
        mgrs_db_path = None
        mgrs_collection_db_path = None

    os.makedirs(sas_outputdir, exist_ok=True)

    num_input_path = len(input_list)
    if os.path.isdir(input_list[0]):
        if num_input_path > 1:
            mosaic_flag = True
        else:
            mosaic_flag = False
    else:
        if num_input_path == 1:
            mosaic_flag = False
        else:
            err_str = f'unable to process more than 1 images.'
            logger.error(err_str)
            raise ValueError(err_str)

    if mosaic_flag:
        print('Number of bursts to process:', num_input_path)

        id_path  = '/identification/'

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
        date_str_id = datetime.datetime.strptime(date_str_id_temp, input_date_format).strftime(output_date_format)

    else:
        date_str_id = 'unknown'

    # [TODO] Final product has different name depending on the workflow
    if dswx_workflow == 'opera_dswx_s1':
        final_water_path = \
            f'{outputdir}/bimodality_output_binary_{pol_str}.tif'
    else:
        final_water_path = \
            f'{outputdir}/region_growing_output_binary_{pol_str}.tif'

    # metadata for final product
    # e.g. geotransform, projection, length, width, utmzon, epsg
    water_meta = dswx_sar_util.get_meta_from_tif(final_water_path)

    # repackage the water map
    # 1) water map
    water_map = dswx_sar_util.read_geotiff(final_water_path)
    no_data_raster = water_map == 255

    # 2) layover/shadow
    layover_shadow_mask_path = f'{outputdir}/mosaic_layovershadow_mask.tif'
    if os.path.exists(layover_shadow_mask_path):
        layover_shadow_mask = \
            dswx_sar_util.read_geotiff(layover_shadow_mask_path)
    else:
        layover_shadow_mask = np.zeros(np.shape(water_map), dtype='byte')

    # 3) hand
    hand = dswx_sar_util.read_geotiff(os.path.join(outputdir, 'interpolated_hand'))
    hand_mask = hand >200

    full_wtr1_water_set_path = f'{outputdir}/full_water_binary_WTR1_set.tif'
    full_wtr_water_set_path = f'{outputdir}/full_water_binary_WTR_set.tif'
    full_conf_water_set_path = f'{outputdir}/fuzzy_image_{pol_str}.tif'

    # 4) inundated_vegetation
    if processing_cfg.inundated_vegetation.enabled:
        inundated_vegetation = dswx_sar_util.read_geotiff(
            f"{outputdir}/temp_inundated_vegetation.tif")
        inundated_vegetation_mask = (inundated_vegetation == 2) & \
                                    (water_map==1)
        inundated_vegetation[inundated_vegetation_mask] = 1
    else:
        inundated_vegetation = None

    if dswx_workflow == 'opera_dswx_s1':
        region_grow_map = \
            dswx_sar_util.read_geotiff(f'{outputdir}/region_growing_output_binary_{pol_str}.tif')
        landcover_map =\
            dswx_sar_util.read_geotiff(f'{outputdir}/refine_landcover_binary_{pol_str}.tif')
        landcover_mask = (region_grow_map == 1) & (landcover_map!=1)
        dark_land_mask = (landcover_map == 1) & (water_map==0)
        bright_water_mask = (landcover_map == 0) & (water_map==1)

        dswx_sar_util.save_dswx_product(water_map==1,
            full_wtr1_water_set_path,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Water classification (WTR)',
            scratch_dir=outputdir,
            landcover_mask=landcover_mask,
            bright_water_fill=bright_water_mask,
            dark_land_mask=dark_land_mask,
            layover_shadow_mask=layover_shadow_mask>0,
            hand_mask=hand_mask,
            inundated_vegetation=inundated_vegetation==2,
            no_data=no_data_raster,
            )

        dswx_sar_util.save_dswx_product(water_map==1,
            full_wtr_water_set_path,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Water classification (WTR)',
            scratch_dir=outputdir,
            inundated_vegetation=inundated_vegetation==2,
            layover_shadow_mask=layover_shadow_mask>0,
            hand_mask=hand_mask,
            no_data=no_data_raster)
    else:
        dswx_sar_util.save_dswx_product(water_map==1,
                    full_wtr1_water_set_path,
                    geotransform=water_meta['geotransform'],
                    projection=water_meta['projection'],
                    description='Water classification (WTR)',
                    scratch_dir=outputdir,
                    layover_shadow_mask=layover_shadow_mask>0,
                    hand_mask=hand_mask,
                    no_data=no_data_raster)

    # get MGRS tile list
    mgrs_tile_list = find_mgrs_collection(mgrs_collection_file=mgrs_collection_db_path,
                                          image_tif=final_water_path)
    unique_mgrs_tile_list = list(set(mgrs_tile_list))
    logger.info('MGRS tiles:', unique_mgrs_tile_list)

    processing_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ")
    if dswx_workflow == 'opera_dswx_s1':

        for mgrs_tile_id in unique_mgrs_tile_list:
            #[TODO] specify file name
            logger.info('mgrs tile', mgrs_tile_id)
            dswx_name_format_prefix = \
                f'OPERA_L3_DSWx-S1_T{mgrs_tile_id}_{date_str_id}_{processing_time}_v{product_version}'

            # bbox and epsg extract from MGRS tile
            output_mgrs_wtr = f'{dswx_name_format_prefix}_B01_WTR.tif'
            save_mgrs_tile_db(
                mgrs_db_path=mgrs_db_path,
                source_tif_path=full_wtr1_water_set_path,
                output_dir_path=sas_outputdir,
                output_tif_name=output_mgrs_wtr,
                output_mgrs_id=mgrs_tile_id,
                method='nearest')

            output_mgrs_bwtr = f'{dswx_name_format_prefix}_B02_BWTR.tif'
            save_mgrs_tile_db(
                mgrs_db_path=mgrs_db_path,
                source_tif_path=full_wtr_water_set_path,
                output_dir_path=sas_outputdir,
                output_tif_name=output_mgrs_bwtr,
                output_mgrs_id=mgrs_tile_id,
                method='nearest')

            output_mgrs_conf = f'{dswx_name_format_prefix}_B03_CONF.tif'
            save_mgrs_tile_db(
                mgrs_db_path=mgrs_db_path,
                source_tif_path=full_conf_water_set_path,
                output_dir_path=sas_outputdir,
                output_tif_name=output_mgrs_conf,
                output_mgrs_id=mgrs_tile_id,
                method='nearest')

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

    run(cfg)

if __name__ == '__main__':
    main()