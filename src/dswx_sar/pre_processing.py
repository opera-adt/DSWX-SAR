import glob
import logging
import mimetypes
import numpy as np
import os
import pathlib
import shutil
import time

from osgeo import gdal, osr
from pathlib import Path

from dswx_sar import dswx_sar_util, filter_SAR, generate_log
from dswx_sar.dswx_runconfig import (DSWX_S1_POL_DICT,
                                     _get_parser,
                                     RunConfig)
from dswx_sar.masking_with_ancillary import get_label_landcover_esa_10

logger = logging.getLogger('dswx_s1')

def pol_ratio(array1, array2):
    '''
    Compute ratio between two arrays
    '''
    array1 = np.asarray(array1, dtype='float32')
    array2 = np.asarray(array2, dtype='float32')

    return array1 / array2


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

    Parameters:
    -----------
    geotiff_path : str
        The file path to the GeoTIFF file to be checked.
    value_list : List[float]
        A list of values against which the mean pixel value of the GeoTIFF
        file is compared.

    Return:
    -------
    validation_result: bool
        If any of the checks fail, return bool indicating potential issues
        with the GeoTIFF file.
        - Return 'nan_value' if some pixels in the file have NaN values.
        - Return 'no_data_only' if the mean pixel value matches any value
          in 'value_list', suggesting uniform pixel values across the file.
    """
    image = dswx_sar_util.read_geotiff(geotiff_path)    
    mean_value = np.nanmean(image)
    print(f'checking {geotiff_path}')
    validation_result = 'okay'

    for invalid_value in value_list:
        if np.any(image == invalid_value):
            validation_result = 'invalid_found'
            logger.warning(f'Some pixels in {geotiff_path} are within '\
                           'the specified value list.')

    if np.isnan(mean_value):
        validation_result = 'nan_value'
        logger.warning(f'NaN pixels found in {geotiff_path}.')
    elif np.isin(image, value_list).all():
        validation_result = 'invalid_only'
        logger.warning(f'All pixels in {geotiff_path} are within '\
                       'the specified invalid value list.')

    return validation_result

class AncillaryRelocation:
    '''
    Relocate the ancillary file to have same geographic extents and
    spacing as the given rtc file.
    '''
    def __init__(self, rtc_file_name, scratch_dir):
        """Initialized AncillaryRelocation Class with rtc_file_name

        Parameters
        ----------
        rtc_file_name : str
            file name of RTC input HDF5
        scratch_dir : str
            scratch directory to save the relocated file
        """

        self.rtc_file_name = rtc_file_name

        self.ycoord_rtc, self.xcoord_rtc = \
            self._read_x_y_array_geotiff(rtc_file_name)
        reftif = gdal.Open(rtc_file_name)
        proj = osr.SpatialReference(wkt=reftif.GetProjection())
        self.epsg = proj.GetAttrValue('AUTHORITY',1)
        self.scratch_dir = scratch_dir

    def relocate(self,
                 ancillary_file_name,
                 relocated_file_str,
                 method='near',
                 no_data=np.nan):

        """ resample image

        Parameters
        ----------
        ancillary_file_name : str
            file name of ancilary data
        relocated_file_str : str
            file name of output
        method : str
            interpolation method
        """
        self._interpolate_gdal(str(self.rtc_file_name),
                              ancillary_file_name,
                              os.path.join(self.scratch_dir,
                                           relocated_file_str),
                              method,
                              no_data=no_data)
        dswx_sar_util._save_as_cog(
            os.path.join(self.scratch_dir, relocated_file_str),
                         self.scratch_dir)

    def _interpolate_gdal(self, ref_file,
                          input_tif_str,
                          output_tif_str,
                          method,
                          no_data):

        """Interpolate the input image to have same projection and resolution
        as the reference file.

        Parameters
        ----------
        ref_file : str
            file name of RTC input HDF5
        input_tif_str : str
            path for the input Geotiff File
        output_tif_str : str
            Path for output Geotiff file
        method : str
            interpolation method
        """
        print(f"> gdalwarp {input_tif_str} -> {output_tif_str}")

        # get reference coordinate and projection
        reftif = gdal.Open(ref_file, gdal.GA_ReadOnly)
        lat0, lon0 = self._read_x_y_array_geotiff(ref_file)
        xsize = reftif.RasterXSize
        ysize = reftif.RasterYSize
        geotransform = reftif.GetGeoTransform()
        proj = osr.SpatialReference(wkt=reftif.GetProjection())
        epsg_output = proj.GetAttrValue('AUTHORITY',1)
        xspacing = geotransform[1]
        yspacing = geotransform[5]

        reftif = None

        if (len(lat0) != ysize) and (len(lon0) != xsize):

            N, S, W, E = [np.max(lat0) + yspacing / 2,
                          np.min(lat0) - yspacing / 2,
                          np.min(lon0) - xspacing / 2,
                          np.max(lon0) + xspacing / 2]

            print('Note: latitude shape is not same as image shape')

        else:
            N, S, W, E = [np.max(lat0),
                          np.min(lat0),
                          np.min(lon0),
                          np.max(lon0)]

        print('bounding box', N, S , W, E)

        # Crop (gdalwarp)image based on geo infomation of reference image
        if yspacing < 0:
            yspacing = -1 * yspacing

        opt = gdal.WarpOptions(dstSRS=f'EPSG:{epsg_output}',
                               xRes=xspacing,
                               yRes=yspacing,
                               outputBounds=[W, S, E, N],
                               resampleAlg=method,
                               format='GTIFF',
                               dstNodata=no_data)

        gdal.Warp(output_tif_str, input_tif_str, options=opt)


    def _read_x_y_array_geotiff(self, intput_tif_str):
        """Read X and Y coordinates from given Geotiff image

        Parameters
        ----------

        input_tif_str : str
            path for the input Geotiff File
        """
        ds = gdal.Open(intput_tif_str, gdal.GA_ReadOnly)

        #get the point to transform, pixel (0,0) in this case
        width = ds.RasterXSize
        height = ds.RasterYSize
        gt = ds.GetGeoTransform()
        minx = gt[0]
        miny = gt[3] + height * gt[5]
        maxx = gt[0] + width * gt[1]
        maxy = gt[3]

        #get the coordinates in lat long
        ycoord = np.linspace(maxy, miny, height)
        xcoord = np.linspace(minx, maxx, width)

        ds = None
        del ds  # close the dataset (Python object and pointers)

        return ycoord, xcoord


def replace_reference_water_nodata_from_ancillary(
        reference_water_path : str,
        landcover_path : str,
        hand_path : str,
        reference_water_max : float,
        reference_water_no_data : float,
        line_per_block : int):
    """
    Replace no-data areas in a reference water raster 
    with maximum values based on landcover and HAND data.

    Parameters:
    -----------
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
    im_meta = dswx_sar_util.get_meta_from_tif(reference_water_path)
    block_params = dswx_sar_util.block_param_generator(
        lines_per_block=line_per_block,
        data_shape=(im_meta['length'],
                    im_meta['width']),
        pad_shape=pad_shape)

    landcover_label = get_label_landcover_esa_10()

    for block_param in block_params:
        ref_water_block, landcover_block, hand_block = [
            dswx_sar_util.get_raster_block(path, block_param)
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

        # write updated reference water 
        dswx_sar_util.write_raster_block(
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

    ref_water_max = processing_cfg.reference_water.max_value
    ref_water_no_data = processing_cfg.reference_water.no_data_value
    pol_list = processing_cfg.polarizations
    pol_options = processing_cfg.polarimetric_option
    
    if pol_options is not None:
        pol_list += pol_options

    pol_all_str = '_'.join(pol_list)
    co_pol = processing_cfg.copol
    cross_pol = processing_cfg.crosspol

    filter_size = processing_cfg.filter.window_size
    filter_flag = processing_cfg.filter.enabled
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
            ref_filename = glob.glob(f'{input_list[0]}/*_{pol_list[0]}*.tif')[0]
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
    im_meta = dswx_sar_util.get_meta_from_tif(ref_filename)

    # create instance to relocate ancillary data
    ancillary_reloc = AncillaryRelocation(ref_filename, scratch_dir)

    # Note : landcover should precede before reference water.
    relocated_ancillary_filename_set = {
        'dem': 'interpolated_DEM.tif',
        'hand': 'interpolated_hand.tif',
        'landcover': 'interpolated_landcover.tif',
        'reference_water': 'interpolated_wbd.tif',
    }

    input_ancillary_filename_set = {
        'dem': dem_file,
        'hand': hand_file,
        'landcover': landcover_file,
        'reference_water': wbd_file,
    }

    landcover_label = get_label_landcover_esa_10()

    for anc_type, anc_filename in relocated_ancillary_filename_set.items():
        input_anc_path = input_ancillary_filename_set[anc_type]
        ancillary_path = Path(
            os.path.join(scratch_dir, anc_filename))

        # Check if input ancillary data is valid. 
        if not os.path.isfile(dem_file):
            err_msg = f'Input {anc_type} file not found'
            raise FileNotFoundError(err_msg)

        if anc_type in ['reference_water']:
            no_data = ref_water_no_data
        elif anc_type in ['landcover']:
            no_data = landcover_label['No_data']
        else:
            no_data = np.nan

        # crop or relocate ancillary images to fit the reference 
        # intensity (RTC) image. 
        if not ancillary_path.is_file():
            logger.info(f'Relocated {anc_type} file will be created from ' \
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

        if validation_result not in 'okay':
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
                                 relocated_ancillary_filename_set['reference_water']),
                    os.path.join(scratch_dir,
                                 relocated_ancillary_filename_set['landcover']),
                    os.path.join(scratch_dir,
                                 relocated_ancillary_filename_set['hand']),
                    reference_water_max=ref_water_max,
                    reference_water_no_data=ref_water_no_data,
                    line_per_block=line_per_block
                )

    # apply SAR filtering
    pad_shape = (filter_size, 0)
    block_params = dswx_sar_util.block_param_generator(
        lines_per_block=line_per_block,
        data_shape=(im_meta['length'],
                    im_meta['width']),
        pad_shape=pad_shape)

    for block_ind, block_param in enumerate(block_params):
        output_image_set = []
        for polind, pol in enumerate(pol_list):
            logger.info(f'block processing {block_ind} - {pol}')

            if pol in ['ratio', 'span']:

                # If ratio/span is in the list,
                # then compute the ratio from VVVV and VHVH
                temp_pol_list = co_pol + cross_pol
                logger.info(f'>> computing {pol} {temp_pol_list}')

                temp_raster_set = []
                for temp_pol in temp_pol_list:
                    filename = \
                        f'{scratch_dir}/{mosaic_prefix}_{temp_pol}.tif'

                    block_data = dswx_sar_util.get_raster_block(
                        filename,
                        block_param)

                    temp_raster_set.append(block_data)

                temp_raster_set = np.array(temp_raster_set)
                if pol in ['ratio']:
                    ratio = pol_ratio(np.squeeze(temp_raster_set[0, :, :]),
                                      np.squeeze(temp_raster_set[1, :, :]))
                    output_image_set.append(ratio)
                    logger.info(f'computing ratio {co_pol}/{cross_pol}')

                if pol in ['span']:
                    span = np.squeeze(temp_raster_set[0, :, :] +
                                      2 * temp_raster_set[1, :, :])
                    output_image_set.append(span)
            else:
                if mosaic_flag:
                    intensity_path = \
                        f'{scratch_dir}/{mosaic_prefix}_{pol}.tif'

                    intensity = dswx_sar_util.get_raster_block(
                        intensity_path, block_param)
                else:
                    intensity = dswx_sar_util.read_geotiff(
                            ref_filename, band_ind=polind)
                # need to replace 0 value in padded area to NaN.
                intensity[intensity==0] = np.nan
                if filter_flag:
                    filtered_intensity = filter_SAR.lee_enhanced_filter(
                                    intensity,
                                    win_size=filter_size)
                else:
                    filtered_intensity = intensity

                output_image_set.append(filtered_intensity)

        output_image_set = np.array(output_image_set, dtype='float32')
        output_image_set[output_image_set == 0] = np.nan

        dswx_sar_util.write_raster_block(
            out_raster=filtered_image_path,
            data=output_image_set,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='float32')

    dswx_sar_util._save_as_cog(filtered_image_path, scratch_dir)

    no_data_geotiff_path = os.path.join(
        scratch_dir, f"no_data_area_{pol_all_str}.tif")

    dswx_sar_util.get_invalid_area(
        os.path.join(scratch_dir, filtered_images_str),
        no_data_geotiff_path,
        scratch_dir=scratch_dir,
        lines_per_block=line_per_block)

    if processing_cfg.debug_mode:
        filtered_intensity = dswx_sar_util.read_geotiff(filtered_image_path)

        for pol_ind, pol in enumerate(pol_list):
            if pol == 'ratio':
                immin, immax = None, None
            else:
                immin, immax = -30, 0
            single_intensity = np.squeeze(filtered_intensity[pol_ind, :, :])
            dswx_sar_util.intensity_display(
                single_intensity,
                scratch_dir,
                pol,
                immin,
                immax)

            if pol in ['ratio']:
                dswx_sar_util.save_raster_gdal(
                    data=single_intensity,
                    output_file=os.path.join(
                        scratch_dir, 'intensity_{}.tif'.format(pol)),
                    geotransform=im_meta['geotransform'],
                    projection=im_meta['projection'],
                    scratch_dir=scratch_dir)
            else:
                dswx_sar_util.save_raster_gdal(
                    data = 10 * np.log10(single_intensity),
                    output_file=os.path.join(
                        scratch_dir, 'intensity_{}_db.tif'.format(pol)),
                    geotransform=im_meta['geotransform'],
                    projection=im_meta['projection'],
                    scratch_dir=scratch_dir)

    t_all_elapsed = time.time() - t_all
    logger.info(f"successfully ran pre-processing in {t_all_elapsed:.3f} seconds")


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

    processing_cfg = cfg.groups.processing
    pol_mode = processing_cfg.polarization_mode
    pol_list = processing_cfg.polarizations
    if pol_mode == 'MIX_DUAL_POL':
        proc_pol_set = [DSWX_S1_POL_DICT['DV_POL'],
                        DSWX_S1_POL_DICT['DH_POL']]
    elif pol_mode == 'MIX_SINGLE_POL':
        proc_pol_set = [DSWX_S1_POL_DICT['SV_POL'],
                        DSWX_S1_POL_DICT['SH_POL']]
    else:
        proc_pol_set = [pol_list]

    for pol_set in proc_pol_set:
        processing_cfg.polarizations = pol_set
        run(cfg)

if __name__ == '__main__':
    main()
