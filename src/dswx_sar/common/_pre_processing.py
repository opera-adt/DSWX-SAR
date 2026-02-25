import copy
import glob
import logging
import mimetypes
import numpy as np
import os
import pathlib
import shutil
import tempfile
import time

from osgeo import gdal, osr, ogr
from pathlib import Path

from dswx_sar.common import _dswx_sar_util
from dswx_sar.common import _filter_SAR

from dswx_sar.common._dswx_sar_util import check_gdal_raster_s3
from dswx_sar.common._masking_with_ancillary import get_label_landcover_esa_10


logger = logging.getLogger('dswx_sar')


def pol_ratio(array1, array2):
    '''
    Compute ratio between two arrays
    '''
    array1 = array1.astype('float32')
    array2 = array2.astype('float32')
    result = np.divide(array1, array2,
                       out=np.zeros_like(array1),
                       where=array2 != 0)

    return result


def validate_gtiff(geotiff_path, value_list):
    """
    Check the validity of a GeoTIFF file based on
    the mean value of its pixels.

    This function reads a GeoTIFF file, calculates its mean pixel
    value, and performs two checks:
    1. If the mean value is NaN, it indicates that some pixels in
       the file have NaN values.
    2. If the mean value equals any value in the provided 'value_list',
       it suggests that all pixels in the file might have the same value.

    Parameters
    ----------
    geotiff_path : str
        The file path to the GeoTIFF file to be checked.
    value_list : List[float]
        A list of values against which the mean pixel value of the GeoTIFF
        file is compared.

    Returns
    -------
    validation_result: str
        If any of the checks fail, return a string that indicates potential
        issues with the GeoTIFF file.
        - Return 'nan_value' if some pixels in the file have NaN values.
        - Return 'invalid_only' if the mean pixel value matches any value
          in 'value_list', suggesting uniform pixel values across the file.
    """
    image = _dswx_sar_util.read_geotiff(geotiff_path)
    mean_value = np.nanmean(image)

    validation_result = 'okay'

    for invalid_value in value_list:
        if np.any(image == invalid_value):
            validation_result = 'invalid_found'
            logger.warning(f'Some pixels in {geotiff_path} are within '
                           'the specified value list.')

    if np.isnan(mean_value):
        validation_result = 'nan_value'
        logger.warning(f'NaN pixels found in {geotiff_path}.')
    elif np.isin(image, value_list).all():
        validation_result = 'invalid_only'
        logger.warning(f'All pixels in {geotiff_path} are within '
                       'the specified invalid value list.')

    return validation_result


class AncillaryRelocation:
    """
    Relocate the ancillary file to have same geographic extents and
    spacing as the given rtc file.
    """

    def __init__(self, rtc_file_name, scratch_dir):
        """Initialize AncillaryRelocation class with rtc_file_name

        Parameters
        ----------
        rtc_file_name : str
            file name of RTC input GeoTIFF (or other GDAL-readable raster)
        scratch_dir : str
            scratch directory to save the relocated file
        """
        self.rtc_file_name = rtc_file_name
        self.scratch_dir = scratch_dir

        reftif = gdal.Open(rtc_file_name, gdal.GA_ReadOnly)
        if reftif is None:
            raise RuntimeError(f"Could not open RTC file: {rtc_file_name}")

        # Cache projection & geotransform
        self.projection_wkt = reftif.GetProjection()
        self.geotransform = reftif.GetGeoTransform()
        self.xsize = reftif.RasterXSize
        self.ysize = reftif.RasterYSize

        # Pixel spacing (cache)
        self.dx = self.geotransform[1]
        self.dy = self.geotransform[5]

        # Spatial reference of the RTC file
        proj = osr.SpatialReference()
        proj.ImportFromWkt(self.projection_wkt)
        self.epsg = proj.GetAttrValue('AUTHORITY', 1)

        # Cache SRS objects and proj4 string once
        # get_projection_proj4 should take a WKT and return a proj4 string
        self.tile_srs = proj
        self.tile_srs_str = get_projection_proj4(self.projection_wkt)

    def relocate(
        self,
        ancillary_file_name,
        relocated_file_str,
        method='near',
        no_data=np.nan,
        return_array=False,
    ):
        """
        Resample ancillary image to RTC grid.

        Parameters
        ----------
        ancillary_file_name : str
            file name of ancillary data
        relocated_file_str : str
            output file name (under scratch_dir)
        method : str
            interpolation method ("near", "bilinear", etc.)
        no_data : float
            output nodata value
        return_array : bool, optional
            If True, also return relocated array as numpy.ndarray.
            If False (default), only write the GeoTIFF and return None.
        """
        relocated_path = os.path.join(self.scratch_dir, relocated_file_str)

        relocated_array = self._warp(
            ancillary_file_name,
            self.geotransform,
            self.projection_wkt,
            self.ysize,
            self.xsize,
            scratch_dir=self.scratch_dir,
            resample_algorithm=method,
            relocated_file=relocated_path,
            margin_in_pixels=0,
            temp_files_list=None,
            no_data=no_data,
            return_array=return_array,
        )

        # COG conversion (if you already use this util)
        from dswx_sar.common import _dswx_sar_util
        _dswx_sar_util._save_as_cog(relocated_path, self.scratch_dir)

        if return_array:
            return relocated_array

    def _get_tile_srs_bbox(
        self,
        tile_min_y_utm, tile_max_y_utm,
        tile_min_x_utm, tile_max_x_utm,
        tile_srs, polygon_srs
    ):
        """
        Get tile bounding box for a given spatial reference system (SRS).
        (Same as your original function, only minor formatting changes.)
        """
        # forces returned values from TransformPoint() to be (x, y, z)
        # rather than (y, x, z) for geographic SRS
        if polygon_srs.IsGeographic():
            try:
                polygon_srs.SetAxisMappingStrategy(
                    osr.OAMS_TRADITIONAL_GIS_ORDER
                )
            except AttributeError:
                logger.warning(
                    'WARNING Could not set the ancillary input SRS '
                    'axis mapping strategy (SetAxisMappingStrategy())'
                    ' to osr.OAMS_TRADITIONAL_GIS_ORDER'
                )

        transformation = osr.CoordinateTransformation(tile_srs, polygon_srs)

        elevation = 0.0
        tile_x_array = np.zeros(4, dtype=float)
        tile_y_array = np.zeros(4, dtype=float)

        x_arrs = [tile_min_x_utm, tile_max_x_utm,
                  tile_max_x_utm, tile_min_x_utm]
        y_arrs = [tile_max_y_utm, tile_max_y_utm,
                  tile_min_y_utm, tile_min_y_utm]

        for i, (x_arr, y_arr) in enumerate(zip(x_arrs, y_arrs)):
            tile_x_array[i], tile_y_array[i], _ = transformation.TransformPoint(
                x_arr, y_arr, elevation
            )

        tile_min_y = np.min(tile_y_array)
        tile_max_y = np.max(tile_y_array)
        tile_min_x = np.min(tile_x_array)
        tile_max_x = np.max(tile_x_array)

        # handles antimeridian: tile_max_x around +180 and
        # tile_min_x around -180
        if tile_max_x > tile_min_x + 340:
            tile_min_x = np.max(tile_x_array[tile_x_array < -150])
            tile_max_x = np.min(tile_x_array[tile_x_array > 150])
            tile_min_x, tile_max_x = tile_max_x, tile_min_x + 360

        x_pts = [tile_min_x, tile_max_x, tile_max_x, tile_min_x, tile_min_x]
        y_pts = [tile_max_y, tile_max_y, tile_min_y, tile_min_y, tile_max_y]

        tile_ring = ogr.Geometry(ogr.wkbLinearRing)
        for x_pt, y_pt in zip(x_pts, y_pts):
            tile_ring.AddPoint(x_pt, y_pt)

        tile_polygon = ogr.Geometry(ogr.wkbPolygon)
        tile_polygon.AddGeometry(tile_ring)
        tile_polygon.AssignSpatialReference(polygon_srs)

        return tile_polygon, tile_min_y, tile_max_y, tile_min_x, tile_max_x

    def _warp(
        self,
        input_file,
        geotransform,
        projection_wkt,
        length,
        width,
        scratch_dir='.',
        resample_algorithm='nearest',
        relocated_file=None,
        margin_in_pixels=0,
        temp_files_list=None,
        no_data=np.nan,
        return_array=False,
    ):
        """
        Relocate/reproject a file based on geotransform, output size, and SRS.
        """

        # Pixel spacing (we already cache these; but geotransform is passed in
        # so we recompute for robustness).
        dy = geotransform[5]
        dx = geotransform[1]

        # Output Y-coordinate start (North) position with margin
        tile_max_y_utm = geotransform[3] - margin_in_pixels * dy

        # Output X-coordinate start (West) position with margin
        tile_min_x_utm = geotransform[0] - margin_in_pixels * dx

        # Output Y-coordinate end (South) position with margin
        tile_min_y_utm = tile_max_y_utm + (length + 2 * margin_in_pixels) * dy

        # Output X-coordinate end (East) position with margin
        tile_max_x_utm = tile_min_x_utm + (width + 2 * margin_in_pixels) * dx

        # Use cached tile_srs / proj4 string instead of rebuilding each call
        tile_srs = self.tile_srs
        tile_srs_str = self.tile_srs_str

        if relocated_file is None:
            relocated_file = tempfile.NamedTemporaryFile(
                dir=scratch_dir, suffix='.tif'
            ).name
            if temp_files_list is not None:
                temp_files_list.append(relocated_file)

        os.makedirs(scratch_dir, exist_ok=True)

        # Open ancillary input once
        gdal_ds = gdal.Open(input_file, gdal.GA_ReadOnly)
        if gdal_ds is None:
            raise RuntimeError(f"Could not open ancillary file: {input_file}")

        file_projection_wkt = gdal_ds.GetProjection()
        file_geotransform = gdal_ds.GetGeoTransform()
        file_min_x, file_dx, _, file_max_y, _, file_dy = file_geotransform
        file_width = gdal_ds.GetRasterBand(1).XSize
        del gdal_ds

        file_srs = osr.SpatialReference()
        file_srs.ImportFromWkt(file_projection_wkt)

        # margin in meters, for antimeridian bbox
        margin_m = 5000

        tile_polygon, tile_min_y, tile_max_y, tile_min_x, tile_max_x = \
            self._get_tile_srs_bbox(
                tile_min_y_utm - margin_m,
                tile_max_y_utm + margin_m,
                tile_min_x_utm - margin_m,
                tile_max_x_utm + margin_m,
                tile_srs, file_srs
            )

        if not self._antimeridian_crossing_requires_special_handling(
            file_srs, file_min_x, tile_min_x, tile_max_x
        ):
            # Simple case: no antimeridian special handling
            logger.info(
                f'    relocating file: {input_file} to file: {relocated_file}'
            )

            gdal.Warp(
                relocated_file,
                input_file,
                format='GTiff',
                dstSRS=tile_srs_str,
                outputBounds=[tile_min_x_utm, tile_min_y_utm,
                              tile_max_x_utm, tile_max_y_utm],
                multithread=True,
                xRes=dx,
                yRes=abs(dy),
                resampleAlg=resample_algorithm,
                errorThreshold=0,
                dstNodata=no_data,
                warpOptions=["NUM_THREADS=ALL_CPUS"],
            )

            if not return_array:
                return None

            gdal_ds = gdal.Open(relocated_file, gdal.GA_ReadOnly)
            relocated_array = gdal_ds.ReadAsArray()
            del gdal_ds
            return relocated_array

        # Antimeridian handling
        logger.info('    tile crosses the antimeridian')

        file_max_x = file_min_x + file_width * file_dx

        proj_win_antimeridian_left = [
            tile_min_x,
            tile_max_y,
            file_max_x,
            tile_min_y,
        ]
        cropped_input_antimeridian_left_temp = tempfile.NamedTemporaryFile(
            dir=scratch_dir, suffix='.tif'
        ).name
        logger.info(
            f'    cropping antimeridian-left side: {input_file} to '
            f'{cropped_input_antimeridian_left_temp} with indexes '
            f'(ulx uly lrx lry): {proj_win_antimeridian_left}'
        )

        gdal.Translate(
            cropped_input_antimeridian_left_temp,
            input_file,
            projWin=proj_win_antimeridian_left,
            outputSRS=file_srs,
            noData=no_data,
        )

        proj_win_antimeridian_right = [
            file_min_x,
            tile_max_y,
            tile_max_x - 360,
            tile_min_y,
        ]
        cropped_input_antimeridian_right_temp = tempfile.NamedTemporaryFile(
            dir=scratch_dir, suffix='.tif'
        ).name
        logger.info(
            f'    cropping antimeridian-right side: {input_file} to '
            f'{cropped_input_antimeridian_right_temp} with indexes '
            f'(ulx uly lrx lry): {proj_win_antimeridian_right}'
        )

        gdal.Translate(
            cropped_input_antimeridian_right_temp,
            input_file,
            projWin=proj_win_antimeridian_right,
            outputSRS=file_srs,
            noData=no_data,
        )

        if temp_files_list is not None:
            temp_files_list.append(cropped_input_antimeridian_left_temp)
            temp_files_list.append(cropped_input_antimeridian_right_temp)

        gdalwarp_input_file_list = [
            cropped_input_antimeridian_left_temp,
            cropped_input_antimeridian_right_temp,
        ]

        logger.info(
            f'    relocating file: {input_file} to file: {relocated_file}'
            f': {tile_min_x_utm} {tile_max_x_utm}'
            f': {tile_min_y_utm} {tile_max_y_utm}'
        )

        gdal.Warp(
            relocated_file,
            gdalwarp_input_file_list,
            format='GTiff',
            dstSRS=tile_srs_str,
            outputBounds=[tile_min_x_utm, tile_min_y_utm,
                          tile_max_x_utm, tile_max_y_utm],
            multithread=True,
            xRes=dx,
            yRes=abs(dy),
            resampleAlg=resample_algorithm,
            errorThreshold=0,
            dstNodata=no_data,
            warpOptions=["NUM_THREADS=ALL_CPUS"],
        )

        if not return_array:
            return None

        gdal_ds = gdal.Open(relocated_file, gdal.GA_ReadOnly)
        relocated_array = gdal_ds.ReadAsArray()
        del gdal_ds
        return relocated_array

    def _antimeridian_crossing_requires_special_handling(
        self, file_srs, file_min_x, tile_min_x, tile_max_x
    ):
        """
        Check if ancillary input requires special handling due to
        the antimeridian crossing.
        (Same logic as your original method.)
        """
        flag_tile_crosses_antimeridian = (
            tile_min_x < 180 and tile_max_x >= 180
        )

        flag_input_geographic_and_longitude_not_0_360 = (
            file_srs.IsGeographic() and file_min_x < -170
        )

        flag_requires_special_handling = (
            flag_tile_crosses_antimeridian
            and flag_input_geographic_and_longitude_not_0_360
        )

        return flag_requires_special_handling


def get_projection_proj4(projection):
    """Return projection in proj4 format

       projection : str
              Projection

       Returns
       -------
       projection_proj4 : str
              Projection in proj4 format
    """
    srs = osr.SpatialReference()
    srs.ImportFromProj4(projection)
    projection_proj4 = srs.ExportToProj4()
    projection_proj4 = projection_proj4.strip()
    return projection_proj4


def replace_reference_water_nodata_from_ancillary(
        reference_water_path: str,
        landcover_path: str,
        hand_path: str,
        reference_water_max: float,
        reference_water_no_data: float,
        line_per_block: int):
    """
    Replace no-data areas in a reference water raster
    with maximum values based on landcover and HAND data.

    Parameters
    ----------
    reference_water_path : str
        Path to the reference water raster file.
    landcover_path : str
        Path to the landcover raster file.
    hand_path : str
        Path to the HAND raster file.
    reference_water_max : float
        Maximum value to replace no-data areas in the reference water raster.
    reference_water_no_data : float
        No-data value in the reference water raster.
    lines_per_block : int
        Number of lines per block for processing the raster data.
    """
    scratch_dir = os.path.dirname(reference_water_path)
    temp_water_path = os.path.join(scratch_dir, 'temp_ref_water.tif')
    shutil.copy2(reference_water_path, temp_water_path)

    pad_shape = (0, 0)
    im_meta = _dswx_sar_util.get_meta_from_tif(reference_water_path)
    block_params = _dswx_sar_util.block_param_generator(
        lines_per_block=line_per_block,
        data_shape=(im_meta['length'],
                    im_meta['width']),
        pad_shape=pad_shape)

    landcover_label = get_label_landcover_esa_10()

    for block_param in block_params:
        ref_water_block, landcover_block, hand_block = [
            _dswx_sar_util.get_raster_block(path, block_param)
            for path in [temp_water_path,
                         landcover_path,
                         hand_path]]

        # Both invalid and no-water areas have zero value
        # in reference water. The area with zero values are
        # replaced with maximum values where the permanent water
        # area in landcover.

        replaced_area = np.logical_and(
            ref_water_block == reference_water_no_data,
            landcover_block == landcover_label['Permanent water bodies'])

        no_data_area = np.logical_and(
            ref_water_block == reference_water_no_data,
            hand_block <= 0.002)
        ref_water_block[no_data_area | replaced_area] = reference_water_max

        replaced_area = np.logical_and(
            ref_water_block == reference_water_no_data,
            landcover_block != landcover_label['Permanent water bodies'])
        ref_water_block[replaced_area] = 0

        # write updated reference water
        _dswx_sar_util.write_raster_block(
            out_raster=reference_water_path,
            data=ref_water_block,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='float32',
            cog_flag=True,
            scratch_dir=scratch_dir)


def run(cfg):

    logger.info("")
    logger.info('Starting DSWx-S1 Preprocessing')

    t_all = time.time()
    processing_cfg = cfg.groups.processing
    dynamic_data_cfg = cfg.groups.dynamic_ancillary_file_group

    input_list = cfg.groups.input_file_group.input_file_path
    scratch_dir = cfg.groups.product_path_group.scratch_path

    wbd_file = dynamic_data_cfg.reference_water_file
    landcover_file = dynamic_data_cfg.worldcover_file
    dem_file = dynamic_data_cfg.dem_file
    hand_file = dynamic_data_cfg.hand_file
    glad_file = dynamic_data_cfg.glad_classification_file

    ref_water_max = processing_cfg.reference_water.max_value
    ref_water_no_data = processing_cfg.reference_water.no_data_value
    pol_list = copy.deepcopy(processing_cfg.polarizations)
    pol_options = processing_cfg.polarimetric_option

    if pol_options is not None:
        pol_list += pol_options

    pol_all_str = '_'.join(pol_list)
    co_pol = processing_cfg.copol
    cross_pol = processing_cfg.crosspol

    filter_options = processing_cfg.filter
    filter_flag = processing_cfg.filter.enabled
    filter_method = processing_cfg.filter.method
    line_per_block = processing_cfg.filter.line_per_block

    mosaic_prefix = processing_cfg.mosaic.mosaic_prefix
    if mosaic_prefix is None:
        mosaic_prefix = 'mosaic'

    # configure if input is single/multi directory/file
    num_input_path = len(input_list)
    if os.path.isdir(input_list[0]):
        if num_input_path > 1:
            logger.info('Multiple input directories are found.')
            mosaic_flag = True
            ref_filename = f'{scratch_dir}/{mosaic_prefix}_{pol_list[0]}.tif'
        else:
            logger.info('Single input directories is found.')
            mosaic_flag = True
            ref_filename = glob.glob(
                f'{input_list[0]}/*_{pol_list[0]}*.tif')[0]
    else:
        if num_input_path == 1:
            logger.info('Single input RTC is found.')
            mosaic_flag = False
            ref_filename = input_list
            ref_filename = f'{ref_filename[:]}'

        else:
            err_str = 'unable to process more than 1 images.'
            logger.error(err_str)
            raise ValueError(err_str)

    logger.info(f'Ancillary data is reprojected using {ref_filename}')

    pathlib.Path(scratch_dir).mkdir(parents=True, exist_ok=True)
    filtered_images_str = f"filtered_image_{pol_all_str}.tif"
    filtered_image_path = os.path.join(
        scratch_dir, filtered_images_str)

    # read metadata from Geotiff File.
    im_meta = _dswx_sar_util.get_meta_from_tif(ref_filename)

    # create instance to relocate ancillary data
    ancillary_reloc = AncillaryRelocation(ref_filename, scratch_dir)

    # Note : landcover should precede before reference water.
    relocated_ancillary_filename_set = {
        'dem': 'interpolated_DEM.tif',
        'hand': 'interpolated_hand.tif',
        'landcover': 'interpolated_landcover.tif',
        'reference_water': 'interpolated_wbd.tif',
        'glad_classification': 'interpolated_glad.tif',
    }

    input_ancillary_filename_set = {
        'dem': dem_file,
        'hand': hand_file,
        'landcover': landcover_file,
        'reference_water': wbd_file,
        'glad_classification': glad_file,
    }

    landcover_label = get_label_landcover_esa_10()

    for anc_type, anc_filename in relocated_ancillary_filename_set.items():
        input_anc_path = input_ancillary_filename_set[anc_type]
        ancillary_path = Path(
            os.path.join(scratch_dir, anc_filename))

        # GLAD classification map is optional.
        if input_anc_path is None and anc_type == 'glad_classification':
            logger.info(f'{anc_type} file is not provided.')
            continue

        # Check if input ancillary data is valid.
        if not os.path.isfile(input_anc_path) and \
           not check_gdal_raster_s3(input_anc_path, raise_error=False):
            err_msg = f'Input {anc_type} file not found'
            raise FileNotFoundError(err_msg)

        if anc_type in ['reference_water']:
            no_data = ref_water_no_data
        elif anc_type in ['landcover']:
            no_data = landcover_label['No_data']
        elif anc_type in ['glad_classification']:
            no_data = 255
        else:
            no_data = np.nan

        # crop or relocate ancillary images to fit the reference
        # intensity (RTC) image.
        if not ancillary_path.is_file():
            logger.info(f'Relocated {anc_type} file will be created from '
                        f'{input_anc_path}.')
            ancillary_reloc.relocate(
                input_anc_path,
                anc_filename,
                method='near',
                no_data=no_data)

        # check if relocated ancillaries are filled with invalid values
        validation_result = validate_gtiff(
            os.path.join(scratch_dir, anc_filename),
            [no_data, np.inf])

        if validation_result not in ['okay']:
            if (anc_type in ['dem', 'hand']) and \
               (validation_result in ['nan_value', 'invalid_only']):
                err_msg = f'Unable to get valid {anc_type}'
                raise ValueError(err_msg)

            elif anc_type in ['landcover']:
                logger.warning(f'{anc_type} has invalid values.')

            elif anc_type in ['reference_water']:
                logger.warning(f'{anc_type} has invalid values.')
                # update reference water if any values of reference water
                # are invalid.
                replace_reference_water_nodata_from_ancillary(
                    os.path.join(scratch_dir,
                                 relocated_ancillary_filename_set[
                                     'reference_water']),
                    os.path.join(scratch_dir,
                                 relocated_ancillary_filename_set[
                                     'landcover']),
                    os.path.join(scratch_dir,
                                 relocated_ancillary_filename_set[
                                     'hand']),
                    reference_water_max=ref_water_max,
                    reference_water_no_data=ref_water_no_data,
                    line_per_block=line_per_block
                )

    # apply SAR filtering
    pad_shape = (filter_options.block_pad, 0)
    block_params = _dswx_sar_util.block_param_generator(
        lines_per_block=line_per_block,
        data_shape=(im_meta['length'],
                    im_meta['width']),
        pad_shape=pad_shape)

    for block_ind, block_param in enumerate(block_params):
        output_image_set = []
        for polind, pol in enumerate(pol_list):
            logger.info(f'  block processing {block_ind} - {pol}')

            if pol in ['ratio', 'span']:

                # If ratio/span is in the list,
                # then compute the ratio from VVVV and VHVH
                temp_pol_list = co_pol + cross_pol
                logger.info(f'  >> computing {pol} {temp_pol_list}')

                temp_raster_set = []
                for temp_pol in temp_pol_list:
                    filename = \
                        f'{scratch_dir}/{mosaic_prefix}_{temp_pol}.tif'

                    block_data = _dswx_sar_util.get_raster_block(
                        filename,
                        block_param)

                    temp_raster_set.append(block_data)

                temp_raster_set = np.array(temp_raster_set)
                if pol in ['ratio']:
                    ratio = pol_ratio(np.squeeze(temp_raster_set[0, :, :]),
                                      np.squeeze(temp_raster_set[1, :, :]))
                    output_image_set.append(ratio)
                    logger.info(f'  computing ratio {co_pol}/{cross_pol}')

                if pol in ['span']:
                    span = np.squeeze(temp_raster_set[0, :, :] +
                                      2 * temp_raster_set[1, :, :])
                    output_image_set.append(span)
            else:
                if mosaic_flag:
                    intensity_path = \
                        f'{scratch_dir}/{mosaic_prefix}_{pol}.tif'

                    intensity = _dswx_sar_util.get_raster_block(
                        intensity_path, block_param)
                else:
                    intensity = _dswx_sar_util.read_geotiff(
                            ref_filename, band_ind=polind)
                # need to replace 0 value in padded area to NaN.
                intensity[intensity == 0] = np.nan
                if filter_flag:
                    if filter_method == 'lee':
                        filtering_method = _filter_SAR.lee_enhanced_filter
                        filter_option = vars(filter_options.lee_filter)

                    elif filter_method == 'anisotropic_diffusion':
                        filtering_method = _filter_SAR.anisotropic_diffusion
                        filter_option = vars(
                            filter_options.anisotropic_diffusion)

                    elif filter_method == 'guided_filter':
                        filtering_method = _filter_SAR.guided_filter
                        filter_option = vars(filter_options.guided_filter)

                    elif filter_method == 'bregman':
                        filtering_method = _filter_SAR.tv_bregman
                        filter_option = vars(filter_options.bregman)
                    filtered_intensity = filtering_method(
                                                intensity, **filter_option)
                else:
                    filtered_intensity = intensity

                output_image_set.append(filtered_intensity)

        output_image_set = np.array(output_image_set, dtype='float32')
        output_image_set[output_image_set == 0] = np.nan

        _dswx_sar_util.write_raster_block(
            out_raster=filtered_image_path,
            data=output_image_set,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='float32')

    _dswx_sar_util._save_as_cog(filtered_image_path, scratch_dir)

    no_data_geotiff_path = os.path.join(
        scratch_dir, f"no_data_area_{pol_all_str}.tif")

    _dswx_sar_util.get_invalid_area(
        os.path.join(scratch_dir, filtered_images_str),
        no_data_geotiff_path,
        scratch_dir=scratch_dir,
        lines_per_block=line_per_block)

    if processing_cfg.debug_mode:
        filtered_intensity = _dswx_sar_util.read_geotiff(filtered_image_path)

        for pol_ind, pol in enumerate(pol_list):
            if pol == 'ratio':
                immin, immax = None, None
            else:
                immin, immax = -30, 0
            single_intensity = np.squeeze(filtered_intensity[pol_ind, :, :])
            _dswx_sar_util.intensity_display(
                single_intensity,
                scratch_dir,
                pol,
                immin,
                immax)

            if pol in ['ratio']:
                _dswx_sar_util.save_raster_gdal(
                    data=single_intensity,
                    output_file=os.path.join(
                        scratch_dir, f'intensity_{pol}.tif'),
                    geotransform=im_meta['geotransform'],
                    projection=im_meta['projection'],
                    scratch_dir=scratch_dir)
            else:
                _dswx_sar_util.save_raster_gdal(
                    data=10 * np.log10(single_intensity),
                    output_file=os.path.join(
                        scratch_dir, f'intensity_{pol}_db.tif'),
                    geotransform=im_meta['geotransform'],
                    projection=im_meta['projection'],
                    scratch_dir=scratch_dir)

    t_all_elapsed = time.time() - t_all
    logger.info("successfully ran pre-processing in "
                f"{t_all_elapsed:.3f} seconds")
