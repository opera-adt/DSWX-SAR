import ast
import copy
import datetime
import glob
import logging
import mimetypes
import os
import time

from collections import Counter
import geopandas as gpd

import numpy as np
from osgeo import gdal
from pyproj import Transformer
import rasterio
from shapely import wkt
from shapely.geometry import Polygon
from shapely.ops import transform

from dswx_sar.common import _generate_log
from dswx_sar.common import (_dswx_sar_util)
from dswx_sar.common._dswx_sar_util import (band_assign_value_dict,
                                    _create_ocean_mask)
from dswx_sar.sentinel1.dswx_runconfig import RunConfig, _get_parser, DSWX_S1_POL_DICT
from dswx_sar.common._metadata import (create_dswx_s1_metadata,
                               collect_burst_id,
                               _populate_statics_metadata_datasets)
from dswx_sar.common._save_mgrs_tiles import (
    get_bounding_box_from_mgrs_tile,
    get_bounding_box_from_mgrs_tile_db,
    get_intersecting_mgrs_tiles_list,
    get_intersecting_mgrs_tiles_list_from_db,
    merge_pol_layers)

logger = logging.getLogger('dswx_sar')


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
        _dswx_sar_util._save_as_cog(
            output_tif_file_path,
            output_dir_path,
            logger,
            compression=cog_compression,
            nbits=cog_nbits)


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
    track_number_list = []
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
            track_number_list.append(track_number)
        counter = Counter(np.array(track_number_list))
        most_common = counter.most_common()
        track_number = most_common[0][0]

    input_date_format = "%Y-%m-%dT%H:%M:%S"
    output_date_format = "%Y%m%dT%H%M%SZ"

    date_str_id_temp = date_str_list[0][:19]
    date_str_id = datetime.datetime.strptime(
        date_str_id_temp, input_date_format).strftime(
            output_date_format)
    platform_str = platform[0] + platform.split('-')[-1]
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
            merge_layer_flag = False
        if merge_layer_flag:
            pol_set1 = DSWX_S1_POL_DICT[pol_type1]
            pol_set2 = DSWX_S1_POL_DICT[pol_type2]
            merge_pol_list = ['_'.join(pol_set1),
                              '_'.join(pol_set2)]
    else:
        pol_set1 = pol_list
        pol_set2 = []
        count_pols = []
        for input_dir in input_list:
            pol_types = DSWX_S1_POL_DICT['DV_POL'] + \
                        DSWX_S1_POL_DICT['DH_POL']
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
    water_map = _dswx_sar_util.read_geotiff(paths['final_water'])
    no_data_raster = _dswx_sar_util.read_geotiff(paths['no_data_area'])
    no_data_raster = (no_data_raster > 0) | \
        (water_map == band_assign_value_dict['no_data'])

    # 2) layover/shadow
    layover_shadow_mask_path = \
        os.path.join(scratch_dir, 'mosaic_layovershadow_mask.tif')

    if os.path.exists(layover_shadow_mask_path):
        layover_shadow_mask = \
            _dswx_sar_util.read_geotiff(layover_shadow_mask_path)
        logger.info('Layover/shadow mask found')
    else:
        layover_shadow_mask = np.zeros(np.shape(water_map), dtype='byte')
        logger.warning('No layover/shadow mask found')

    # 3) hand excluded
    hand = _dswx_sar_util.read_geotiff(
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
        inundated_vegetation = _dswx_sar_util.read_geotiff(
            paths['inundated_veg'])
        inundated_vege_target_area = _dswx_sar_util.read_geotiff(
            paths['inundated_veg_target'])
        inundated_vege_high_ratio = _dswx_sar_util.read_geotiff(
            paths['inundated_veg_high_ratio'])
        inundated_vegetation_mask = (inundated_vegetation == 2) & \
                                    (water_map == 1)
        inundated_vegetation[inundated_vegetation_mask] = 1
        logger.info('Inundated vegetation file was found.')
        iv_target_file_type = inundated_vege_cfg.target_area_file_type
        if iv_target_file_type == 'auto':
            # if target_file_type is auto and GLAD is provided,
            # GLAD is the source of inundated vegetation mapping
            interp_glad_path_str = os.path.join(
                scratch_dir,'interpolated_glad.tif')

            if os.path.exists(interp_glad_path_str):
                inundated_vege_cfg.target_area_file_type = 'GLAD'

                worldcover_valid = np.nansum(inundated_vege_target_area == 2)
                glad_valid = np.nansum(inundated_vege_target_area == 1)
                # If some pixels are extracted from WorldCover,
                # IV source is GLAD/WorldCover
                if worldcover_valid > 0 and glad_valid > 0:
                    inundated_vege_cfg.target_area_file_type = 'GLAD/WorldCover'
                # If the GLAD is provided but all pixels come from 'WorldCover'
                # due to the no-data of GLAD,
                # IV source is WorldCover
                elif worldcover_valid > 0 and glad_valid == 0:
                    inundated_vege_cfg.target_area_file_type = 'WorldCover'
            else:
                inundated_vege_cfg.target_area_file_type = 'WorldCover'
        logger.info('Inundated vegetation areas are defined from  '
                    f'{inundated_vege_cfg.target_area_file_type}.')

    else:
        inundated_vegetation = None
        inundated_vege_target_area = None
        inundated_vege_high_ratio = None
        inundated_vegetation_mask = None
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

    if dswx_workflow == 'opera_dswx_s1':
        logger.info('BWTR and WTR Files are created from pre-computed files.')

        region_grow_map = \
            _dswx_sar_util.read_geotiff(paths['region_growing'])
        landcover_map =\
            _dswx_sar_util.read_geotiff(paths['landcover_mask'])

        landcover_mask = (region_grow_map == 1) & (landcover_map != 1)
        dark_land_mask = (landcover_map == 1) & (water_map == 0)
        bright_water_mask = (landcover_map == 0) & (water_map == 1)
        wetland = inundated_vege_target_area == 1
        # Open water/inundated vegetation
        # layover shadow mask/hand mask/no_data
        # will be saved in WTR product
        _dswx_sar_util.save_dswx_product(
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
        _dswx_sar_util.save_dswx_product(
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
        _dswx_sar_util.save_dswx_product(
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
            inundated_vegetation_conf=(inundated_vege_high_ratio == 1) &
                (wetland == 0) & (water_map == 0),
            wetland_nonwater=(water_map == 0) & wetland,
            wetland_water=(water_map == 1) & wetland,
            wetland_bright_water_fill=bright_water_mask & wetland,
            wetland_inundated_veg=(inundated_vegetation == 2) & wetland,
            wetland_dark_land_mask=dark_land_mask & wetland,
            wetland_landcover_mask=landcover_mask & wetland,
            layover_shadow_mask=layover_shadow_mask > 0,
            hand_mask=hand_mask,
            ocean_mask=ocean_mask,
            no_data=no_data_raster,
            )

        # Values ranging from 0 to 100 are used to represent the likelihood
        # or possibility of the presence of water. A higher value within
        # this range signifies a higher likelihood of water being present.
        fuzzy_value = _dswx_sar_util.read_geotiff(paths['fuzzy_value'])
        fuzzy_value = np.round(fuzzy_value * 100)
        _dswx_sar_util.save_dswx_product(
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
        _dswx_sar_util.save_dswx_product(
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
        actual_burst_id = collect_burst_id(input_list,
                                           DSWX_S1_POL_DICT['CO_POL'])
        # In the case that mgrs_tile_collection_id is given
        # from input, then extract the MGRS list from database
        if input_mgrs_collection_id is not None:
            mgrs_tile_list, most_overlapped = \
                get_mgrs_tiles_list_from_db(
                    mgrs_collection_file=mgrs_collection_db_path,
                    mgrs_tile_collection_id=input_mgrs_collection_id)
        # In the case that mgrs_tile_collection_id is not given
        # from input, then extract the MGRS list from database
        # using track number and area intersecting with image_tif
        else:
            mgrs_tile_list, most_overlapped = \
                get_intersecting_mgrs_tiles_list_from_db(
                    mgrs_collection_file=mgrs_collection_db_path,
                    image_tif=paths['final_water'],
                    track_number=track_number,
                    burst_list=actual_burst_id
                    )
        track_number = most_overlapped['relative_orbit_number']
        maximum_burst = most_overlapped['number_of_bursts']
        # convert string to list
        expected_burst_list = ast.literal_eval(most_overlapped['bursts'])
        logger.info(f"Input RTCs are within {most_overlapped['mgrs_set_id']}")
        number_burst = len(actual_burst_id)
        mgrs_meta_dict['MGRS_COLLECTION_EXPECTED_NUMBER_OF_BURSTS'] = \
            maximum_burst
        mgrs_meta_dict['MGRS_COLLECTION_ACTUAL_NUMBER_OF_BURSTS'] = \
            number_burst
        missing_burst = len(list(set(expected_burst_list) -
                                 set(actual_burst_id)))
        mgrs_meta_dict['MGRS_COLLECTION_MISSING_NUMBER_OF_BURSTS'] = \
            missing_burst
        mgrs_meta_dict['MGRS_POL_MODE'] = pol_mode
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
                dswx_metadata_dict = create_dswx_s1_metadata(
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
        cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_s1', args)

    _generate_log.configure_log_file(cfg.groups.log_file)

    run(cfg)


if __name__ == '__main__':
    main()
