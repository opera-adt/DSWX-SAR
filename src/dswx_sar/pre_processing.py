import os
import pathlib
import time
import numpy as np
import logging
import mimetypes

from osgeo import osr, gdal
from pathlib import Path

from dswx_sar import filter_SAR
from dswx_sar import dswx_sar_util
from dswx_sar.dswx_runconfig import _get_parser, RunConfig
from dswx_sar import generate_log

logger = logging.getLogger('dswx_s1')

def pol_ratio(array1, array2):
    '''
    Compute ratio between two arrays
    '''
    array1 = np.asarray(array1, dtype='float32')
    array2 = np.asarray(array2, dtype='float32')

    return array1 / array2

def pol_coherence(array1, array2, array3):
    '''
    Compute polarimetric coherence from two co-pol and one cross-pol
    '''
    array1 = np.asarray(array1, dtype='float32') # VVVV
    array2 = np.asarray(array2, dtype='float32') # VHVH
    array3 = np.asarray(array3, dtype='complex') # VVVH

    return np.abs(array3 / np.sqrt(array1 * array2))


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
        reftif = None
        self.scratch_dir = scratch_dir

    def relocate(self,
                 ancillary_file_name,
                 relocated_file_str,
                 method='near'):

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
                              method)

    def _interpolate_gdal(self, ref_file,
                          input_tif_str,
                          output_tif_str,
                          method):

        """Interpolate the input image to have same projection and resolution
        as the refernce file.

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
        del reftif

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
                               format='ENVI')

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

    pol_list = processing_cfg.polarizations
    pol_all_str = '_'.join(pol_list)
    co_pol = processing_cfg.copol
    cross_pol = processing_cfg.crosspol
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
            mosaic_flag = False

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

    logger.info(f'ancillary data is reprojected using {ref_filename}')

    pathlib.Path(scratch_dir).mkdir(parents=True, exist_ok=True)
    filtered_images_str = f"filtered_image_{pol_all_str}.tif"

    # read metadata from Geotiff File.
    im_meta = dswx_sar_util.get_meta_from_tif(ref_filename)

    # create instance to relocate ancillary data
    ancillary_reloc = AncillaryRelocation(ref_filename, scratch_dir)

    # Check if the interpolated water body file exists
    wbd_interpolated_path = Path(os.path.join(scratch_dir,
                                              'interpolated_wbd'))
    if not wbd_interpolated_path.is_file():
        logger.info('interpolated wbd file was not found')
        ancillary_reloc.relocate(wbd_file, 'interpolated_wbd', method='near')

    # Check if the interpolated DEM exists
    dem_interpolated_path = Path(os.path.join(scratch_dir,
                                              'interpolated_DEM'))
    dem_reprocessing_flag = False
    if not dem_interpolated_path.is_file():
        logger.info('interpolated dem : not found ')

        if os.path.isfile(dem_file):
            logger.info('interpolated dem file was not found')
            ancillary_reloc.relocate(dem_file,
                                      'interpolated_DEM',
                                      method='near')
        else:
            raise FileNotFoundError

    # check if interpolated DEM has valid values
    if not dem_reprocessing_flag:
        dem_subset = dswx_sar_util.read_geotiff(
                        os.path.join(scratch_dir,
                                     'interpolated_DEM'))
        dem_mean = np.nanmean(dem_subset)
        if (dem_mean == 0) | np.isnan(dem_mean):
            raise ValueError

    # check if the interpolated landcover exists ###
    landcover_interpolated_path = Path(os.path.join(scratch_dir,
                                                    'interpolated_landcover'))
    if not landcover_interpolated_path.is_file():
        ancillary_reloc.relocate(landcover_file,
                                  'interpolated_landcover',
                                  method='near')

    # Check if the interpolated HAND exists
    hand_interpolated_path = os.path.join(scratch_dir,
                                          'interpolated_hand')
    if not os.path.isfile(hand_interpolated_path):

        # Check if compuated HAND exists
        if hand_file is None:
            logger.info('>> HAND file is not found, so will be computed.')
            # hand_calc.hand(dem_interpolated_path, args.scratch_dir)
            # args.hand_file = os.path.join(args.scratch_dir, 'temp_hand.tif')

        ancillary_reloc.relocate(hand_file,
                                  'interpolated_hand',
                                  method='near')

    intensity = []
    for polind, pol in enumerate(pol_list):
        if pol in ['ratio', 'coherence', 'span']:

            # If ratio/span is in the list,
            # then compute the ratio from VVVV and VHVH
            if pol in ['ratio', 'span']:
                temp_pol_list = [co_pol, cross_pol]
                logger.info(f'>> computing {pol} {temp_pol_list}')

            # If coherence is in the list,
            # then compute the coherence from VVVV, VHVH, VVVH
            if pol in ['coherence']:
                temp_pol_list = ['VV', 'VH', 'VVVH']
                logger.info(f'>> computing coherence {temp_pol_list}')

            temp_raster_set = []
            for temp_pol in temp_pol_list:
                filename = \
                    f'{scratch_dir}/{mosaic_prefix}_{temp_pol}.tif'
                temp_raster_set.append(dswx_sar_util.read_geotiff(filename))

            if pol in ['ratio']:
                ratio = pol_ratio(np.squeeze(temp_raster_set[0, :, :]),
                                  np.squeeze(temp_raster_set[1, :, :]))
                intensity.append(ratio)
                logger.info(f'computing ratio {co_pol}/{cross_pol}')

            if pol in ['coherence']:
                coherence = pol_coherence(
                    np.squeeze(temp_raster_set[0, :, :]),
                    np.squeeze(temp_raster_set[1, :, :]),
                    np.squeeze(temp_raster_set[2, :, :]))
                intensity.append(coherence)
                logger.info('computing polarimetric coherence')

            if pol in ['span']:
                span = np.squeeze(temp_raster_set[0, :, :] +
                        2 * np.squeeze(temp_raster_set[1, :, :]))
                intensity.append(span)

        else:
            logger.info(f'opening {pol}')
            if mosaic_flag:
                filename = \
                    f'{scratch_dir}/{mosaic_prefix}_{pol}.tif'

                temp_raster = dswx_sar_util.read_geotiff(
                        filename)
            else:
                temp_raster = dswx_sar_util.read_geotiff(
                        ref_filename, band_ind=polind)
            intensity.append(np.abs(temp_raster))

    intensity = np.asarray(intensity)

    # apply SAR filtering
    filter_size = processing_cfg.filter.window_size
    intensity_filt = []

    for ii, pol in enumerate(pol_list):

        temp_filt = filter_SAR.lee_enhanced_filter(
                        np.squeeze(intensity[ii, :, :]),
                        win_size=filter_size)
        temp_filt[temp_filt == 0] = np.nan
        intensity_filt.append(temp_filt)

        if processing_cfg.debug_mode:

            if pol == 'ratio':
                immin, immax = None, None
            if pol == 'coherence':
                immin, immax = 0, 0.4
            else:
                immin, immax = -30, 0
            dswx_sar_util.intensity_display(temp_filt, scratch_dir, pol, immin, immax)

            if pol in ['ratio', 'coherence']:

                dswx_sar_util.save_raster_gdal(
                    data=temp_filt,
                    output_file=os.path.join(scratch_dir,
                                             'intensity_{}.tif'.format(pol)),
                    geotransform=im_meta['geotransform'],
                    projection=im_meta['projection'],
                    scratch_dir=scratch_dir)
            else:

                dswx_sar_util.save_raster_gdal(
                    data=10*np.log10(temp_filt),
                    output_file=os.path.join(scratch_dir,
                                             'intensity_{}_db.tif'.format(pol)),
                    geotransform=im_meta['geotransform'],
                    projection=im_meta['projection'],
                    scratch_dir=scratch_dir)

    intensity_filt = np.array(intensity_filt)

    # Save filtered image to geotiff
    filtered_images_str = f"filtered_image_{pol_all_str}.tif"

    dswx_sar_util.save_raster_gdal(
        data=intensity_filt,
        output_file=os.path.join(scratch_dir,filtered_images_str),
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

    run(cfg)


if __name__ == '__main__':
    main()
