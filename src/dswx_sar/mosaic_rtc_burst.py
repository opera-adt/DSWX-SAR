'''
A module to mosaic Sentinel-1 geobursts from RTC workflow
'''
import copy
import glob
import logging
import mimetypes
import os
import tempfile
import time

from collections import Counter
import h5py
import numpy as np
from osgeo import osr, gdal
from scipy import ndimage

from dswx_sar.dswx_runconfig import _get_parser, RunConfig
from dswx_sar import (dswx_geogrid,
                      dswx_sar_util,
                      generate_log)

logger = logging.getLogger('dswx_s1')


def majority_element(num_list):
    """
    Determine the majority element in a list
    Parameters
    ----------
    num_list : List[int]
        A list of integers where the majority element needs to be determined.

    Returns
    -------
    int:
        The majority element in the list. If no majority exists,
        it may return any element from the list.
    """

    counter = Counter(np.array(num_list))
    most_common = counter.most_common()
    most_freq_elem = most_common[0][0]

    return most_freq_elem


def read_metadata_epsg(h5_meta_path):
    '''
    Extract metadata regarding coordinate spacing and projection
    from an HDF5 file.

    parameters:
    -----------
    h5_meta_path: str
        The path to the HDF5 file containing the metadata.

    returns:
    -------
    meta_dict : dict
        A dictionary containing:
        - 'xspacing': Spacing along the x-coordinate.
        - 'yspacing': Spacing along the y-coordinate.
        - 'epsg': The EPSG code for the data projection.

    '''
    freqA_path = '/data/'
    with h5py.File(h5_meta_path, 'r') as src_h5:
        xres = np.array(src_h5[f'{freqA_path}/xCoordinateSpacing'])
        yres = np.array(src_h5[f'{freqA_path}/yCoordinateSpacing'])
        epsg = np.array(src_h5[f'{freqA_path}/projection'])

    meta_dict = {}
    meta_dict['xspacing'] = xres
    meta_dict['yspacing'] = yres
    meta_dict['epsg'] = epsg
    return meta_dict


def save_h5_metadata_to_tif(h5_meta_path,
                            data_path,
                            output_tif_path,
                            epsg_output):
    '''
    extract data from RTC metadata and store it as geotiff
    paremeters:
    -----------
    h5_meta_path: str
        the path to the rtc metadata
    data_path: str
        the hdf5 path to the data to be extracted
    output_tif_path: str
        output tif path
    '''
    freqA_path = '/data/'

    with h5py.File(h5_meta_path, 'r') as src_h5:
        data = np.array(src_h5[data_path])
        xcoord = np.array(src_h5[f'{freqA_path}/xCoordinates'])
        ycoord = np.array(src_h5[f'{freqA_path}/yCoordinates'])
        xres = np.array(src_h5[f'{freqA_path}/xCoordinateSpacing'])
        yres = np.array(src_h5[f'{freqA_path}/yCoordinateSpacing'])
        epsg = np.array(src_h5[f'{freqA_path}/projection'])

    dtype = data.dtype
    geotransform = [xcoord[0] - xres / 2, float(xres), 0,
                    ycoord[0] - yres / 2, 0, float(yres)]
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(int(epsg))               # WGS84 lat/long
    projection = srs.ExportToWkt()
    output_dirname = os.path.dirname(output_tif_path)

    if epsg == epsg_output:
        dswx_sar_util.save_raster_gdal(
            data,
            output_tif_path,
            geotransform,
            projection,
            scratch_dir=output_dirname,
            datatype=dtype)
    else:
        output_tif_temp_dir_path = os.path.dirname(output_tif_path)
        output_tif_temp_base_path = \
            f'{os.path.basename(output_tif_path)}_temp.tif'
        output_tif_temp_path = os.path.join(output_tif_temp_dir_path,
                                            output_tif_temp_base_path)
        dswx_sar_util.save_raster_gdal(
            data,
            output_tif_temp_path,
            geotransform,
            projection,
            scratch_dir=output_tif_temp_dir_path,
            datatype=dtype)

        opt = gdal.WarpOptions(
            dstSRS=f'EPSG:{epsg_output}',
            xRes=xres,
            yRes=abs(yres),
            resampleAlg='nearest',
            format='GTIFF',
            creationOptions=['COMPRESS=DEFLATE',
                             'PREDICTOR=2'])
        gdal.Warp(output_tif_path, output_tif_temp_path, options=opt)
        os.remove(output_tif_temp_path)


def requires_reprojection(geogrid_mosaic,
                          rtc_image: str,
                          nlooks_image: str = None) -> bool:
    '''
    Check if the reprojection is required to mosaic input raster

    Parameters
    -----------
    geogrid_mosaic: isce3.product.GeoGridParameters
        Mosaic geogrid
    rtc_image: str
        Path to the geocoded RTC image
    nlooks_image: str (optional)
        Path to the nlooks raster

    Returns
    flag_requires_reprojection: bool
        True if reprojection is necessary to mosaic inputs
        False if the images are aligned, so that no reprojection is necessary.
    '''

    # Accepted error in the coordinates as floating number
    maxerr_coord = 1.0e-6

    raster_rtc_image = gdal.Open(rtc_image, gdal.GA_ReadOnly)
    if nlooks_image is not None:
        raster_nlooks = gdal.Open(nlooks_image, gdal.GA_ReadOnly)

    # Compare geotransforms of RTC image and nlooks (if provided)
    if (nlooks_image is not None and
            raster_rtc_image.GetGeoTransform() !=
            raster_nlooks.GetGeoTransform()):
        error_str = (f'ERROR geolocations of {raster_rtc_image} and'
                     f' {raster_nlooks} do not match')
        raise ValueError(error_str)

    # Compare dimension - between RTC imagery and corresponding nlooks
    if (nlooks_image is not None and
        (raster_rtc_image.RasterXSize != raster_nlooks.RasterXSize or
            raster_rtc_image.RasterYSize != raster_nlooks.RasterYSize)):
        error_str = (f'ERROR dimensions of {raster_rtc_image} and'
                     f' {raster_nlooks} do not match')
        raise ValueError(error_str)

    rasters_to_check = [raster_rtc_image]
    if nlooks_image is not None:
        rasters_to_check += [raster_nlooks]

    srs_mosaic = osr.SpatialReference()
    srs_mosaic.ImportFromEPSG(geogrid_mosaic.epsg)

    proj_mosaic = osr.SpatialReference(wkt=srs_mosaic.ExportToWkt())
    epsg_mosaic = proj_mosaic.GetAttrValue('AUTHORITY', 1)

    for raster in rasters_to_check:
        x0, dx, _, y0, _, dy = raster.GetGeoTransform()
        projection = raster.GetProjection()

        # check spacing
        if dx != geogrid_mosaic.spacing_x:
            flag_requires_reprojection = True
            return flag_requires_reprojection

        if dy != geogrid_mosaic.spacing_y:
            flag_requires_reprojection = True
            return flag_requires_reprojection

        # check projection
        if projection != srs_mosaic.ExportToWkt():
            proj_raster = osr.SpatialReference(wkt=projection)
            epsg_raster = proj_raster.GetAttrValue('AUTHORITY', 1)

            if epsg_raster != epsg_mosaic:
                flag_requires_reprojection = True
                return flag_requires_reprojection

        # check the coordinates
        if (abs((x0 - geogrid_mosaic.start_x) % geogrid_mosaic.spacing_x) >
                maxerr_coord):
            flag_requires_reprojection = True
            return flag_requires_reprojection

        if (abs((y0 - geogrid_mosaic.start_y) % geogrid_mosaic.spacing_y) >
                maxerr_coord):
            flag_requires_reprojection = True
            return flag_requires_reprojection

    flag_requires_reprojection = False
    return flag_requires_reprojection


def _compute_distance_to_burst_center(image, geotransform):
    '''
    Compute distance from burst center

    Parameters
    -----------
       image: np.ndarray
           Input image
       geotransform: list(float)
           Data geotransform

    Returns
        distance_image: np.ndarray
            Distance image
    '''

    length, width = image.shape
    center_of_mass = ndimage.center_of_mass(np.isfinite(image))

    x_vector = np.arange(width, dtype=np.float32)
    y_vector = np.arange(length, dtype=np.float32)

    _, dx, _, _, _, dy = geotransform

    x_distance_image, y_distance_image = np.meshgrid(x_vector, y_vector)
    distance = np.sqrt((dy * (y_distance_image - center_of_mass[0])) ** 2 +
                       (dx * (x_distance_image - center_of_mass[1])) ** 2)

    return distance


def compute_mosaic_array(list_rtc_images,
                         list_nlooks,
                         mosaic_mode,
                         scratch_dir='',
                         geogrid_in=None,
                         temp_files_list=None,
                         no_data_value=np.isnan,
                         verbose=True):
    '''
    Mosaic S-1 geobursts and return the mosaic as dictionary

    Parameters
    -----------
       list_rtc: list
           List of the path to the rtc geobursts
       list_nlooks: list
           List of the nlooks raster that corresponds to list_rtc
       mosaic_mode: str
            Mosaic mode. Choices: "average", "first", and "bursts_center"
       scratch_dir: str (optional)
            Directory for temporary files
       geogrid_in: isce3.product.GeoGridParameters, default: None
            Geogrid information to determine the output mosaic's shape and
            projection. The geogrid of the output mosaic will automatically
            determined when it is None
       temp_files_list: list (optional)
            Mutable list of temporary files. If provided,
            paths to the temporary files generated will be
            appended to this list
       verbose: flag (optional)
            Flag to enable (True) or disable (False) the verbose mode
    Returns
        mosaic_dict: dict
            Mosaic dictionary
    '''

    mosaic_mode_choices_list = ['average', 'first', 'bursts_center']
    if mosaic_mode.lower() not in mosaic_mode_choices_list:
        raise ValueError(f'ERROR invalid mosaic mode: {mosaic_mode}.'
                         f' Choices: {", ".join(mosaic_mode_choices_list)}')

    num_raster = len(list_rtc_images)
    description_list = []
    num_bands = None
    posting_x = None
    posting_y = None

    list_geo_transform = np.zeros((num_raster, 6))
    list_dimension = np.zeros((num_raster, 2), dtype=np.int32)

    for i, path_rtc in enumerate(list_rtc_images):

        raster_in = gdal.Open(path_rtc, gdal.GA_ReadOnly)
        list_geo_transform[i, :] = raster_in.GetGeoTransform()
        list_dimension[i, :] = (raster_in.RasterYSize, raster_in.RasterXSize)

        # Check if the number of bands are consistent over the
        # input RTC rasters
        if num_bands is None:
            num_bands = raster_in.RasterCount

        elif num_bands != raster_in.RasterCount:
            raise ValueError(f'ERROR: the file "{os.path.basename(path_rtc)}"'
                             f' has {raster_in.RasterCount} bands. Expected:'
                             f' {num_bands}.')

        if len(description_list) == 0:
            for i_band in range(num_bands):
                description_list.append(
                    raster_in.GetRasterBand(i_band+1).GetDescription())

        # Close GDAL dataset
        raster_in = None

    if geogrid_in is None:
        # determine GeoTransformation, posting, dimension, and projection from
        # the input raster
        for i in range(num_raster):
            if list_geo_transform[:, 1].max() == list_geo_transform[:, 1].min():
                posting_x = list_geo_transform[0, 1]

            if list_geo_transform[:, 5].max() == list_geo_transform[:, 5].min():
                posting_y = list_geo_transform[0, 5]

        # determine the dimension and the upper left corner of the output
        # mosaic
        xmin_mosaic = list_geo_transform[:, 0].min()
        ymax_mosaic = list_geo_transform[:, 3].max()
        xmax_mosaic = (list_geo_transform[:, 0] +
                       list_geo_transform[:, 1]*list_dimension[:, 1]).max()
        ymin_mosaic = (list_geo_transform[:, 3] +
                       list_geo_transform[:, 5]*list_dimension[:, 0]).min()

        dim_mosaic = (int(np.ceil((ymin_mosaic - ymax_mosaic) / posting_y)),
                      int(np.ceil((xmax_mosaic - xmin_mosaic) / posting_x)))

        gdal_ds_raster_in = gdal.Open(list_rtc_images[0], gdal.GA_ReadOnly)
        wkt_projection = gdal_ds_raster_in.GetProjectionRef()
        del gdal_ds_raster_in

    else:
        # Directly bring the geogrid information from the input parameter
        xmin_mosaic = geogrid_in.start_x
        ymax_mosaic = geogrid_in.start_y
        posting_x = geogrid_in.spacing_x
        posting_y = geogrid_in.spacing_y

        dim_mosaic = (geogrid_in.length, geogrid_in.width)

        xmax_mosaic = xmin_mosaic + posting_x * dim_mosaic[1]
        ymin_mosaic = ymax_mosaic + posting_y * dim_mosaic[0]

        srs_mosaic = osr.SpatialReference()
        srs_mosaic.ImportFromEPSG(geogrid_in.epsg)
        wkt_projection = srs_mosaic.ExportToWkt()

    if verbose:
        print('    mosaic geogrid:')
        print('        start X:', xmin_mosaic)
        print('        end X:', xmax_mosaic)
        print('        start Y:', ymax_mosaic)
        print('        end Y:', ymin_mosaic)
        print('        spacing X:', posting_x)
        print('        spacing Y:', posting_y)
        print('        width:', dim_mosaic[1])
        print('        length:', dim_mosaic[0])
        print('        projection:', wkt_projection)
        print('        number of bands: {num_bands}')

    if mosaic_mode.lower() == 'average':
        arr_numerator = np.zeros((num_bands, dim_mosaic[0], dim_mosaic[1]),
                                 dtype=float)
        arr_denominator = np.zeros(dim_mosaic, dtype=float)
    else:
        arr_numerator = np.full((num_bands, dim_mosaic[0], dim_mosaic[1]),
                                np.nan, dtype=float)
        if mosaic_mode.lower() == 'bursts_center':
            arr_distance = np.full(dim_mosaic, np.nan, dtype=float)

    for i, path_rtc in enumerate(list_rtc_images):
        if i < len(list_nlooks):
            path_nlooks = list_nlooks[i]
        else:
            path_nlooks = None

        if verbose:
            print(f'    mosaicking ({i+1}/{num_raster}): '
                  f'{os.path.basename(path_rtc)}')
        if geogrid_in is not None and requires_reprojection(
                geogrid_in, path_rtc, path_nlooks):
            if verbose:
                print('        the image requires reprojection/relocation')

                relocated_file = tempfile.NamedTemporaryFile(
                    dir=scratch_dir, suffix='.tif').name

                print('        reprojecting image to temporary file:',
                      relocated_file)

            if temp_files_list is not None:
                temp_files_list.append(relocated_file)

            warp_creation_options = gdal.WarpOptions(
                creationOptions=['COMPRESS=DEFLATE',
                                 'PREDICTOR=2'])

            gdal.Warp(
                relocated_file, path_rtc,
                format='GTiff',
                dstSRS=wkt_projection,
                outputBounds=[
                    geogrid_in.start_x,
                    geogrid_in.start_y +
                    geogrid_in.length * geogrid_in.spacing_y,
                    geogrid_in.start_x +
                    geogrid_in.width * geogrid_in.spacing_x,
                    geogrid_in.start_y],
                multithread=True,
                xRes=geogrid_in.spacing_x,
                yRes=abs(geogrid_in.spacing_y),
                resampleAlg='average',
                errorThreshold=0,
                dstNodata=np.nan,
                options=warp_creation_options
                )
            path_rtc = relocated_file

            if path_nlooks is not None:
                relocated_file_nlooks = tempfile.NamedTemporaryFile(
                    dir=scratch_dir, suffix='.tif').name

                print('        reprojecting number of looks layer to temporary'
                      ' file:', relocated_file_nlooks)

                if temp_files_list is not None:
                    temp_files_list.append(relocated_file_nlooks)

                gdal.Warp(
                    relocated_file_nlooks, path_nlooks,
                    format='GTiff',
                    dstSRS=wkt_projection,
                    outputBounds=[
                        geogrid_in.start_x,
                        geogrid_in.start_y +
                        geogrid_in.length * geogrid_in.spacing_y,
                        geogrid_in.start_x +
                        geogrid_in.width * geogrid_in.spacing_x,
                        geogrid_in.start_y],
                    multithread=True,
                    xRes=geogrid_in.spacing_x,
                    yRes=abs(geogrid_in.spacing_y),
                    resampleAlg='cubic',
                    errorThreshold=0,
                    dstNodata=np.nan)
                path_nlooks = relocated_file_nlooks

            offset_imgx = 0
            offset_imgy = 0
        else:

            # calculate the burst RTC's offset wrt. the output mosaic in
            # the image coordinate
            offset_imgx = int((list_geo_transform[i, 0] - xmin_mosaic) /
                              posting_x + 0.5)
            offset_imgy = int((list_geo_transform[i, 3] - ymax_mosaic) /
                              posting_y + 0.5)

        if verbose:
            print('        image offset (x, y): '
                  f'({offset_imgx}, {offset_imgy})')

        if path_nlooks is not None:
            nlooks_gdal_ds = gdal.Open(path_nlooks, gdal.GA_ReadOnly)
            arr_nlooks = nlooks_gdal_ds.ReadAsArray()
            invalid_ind = np.isnan(arr_nlooks)
            arr_nlooks[invalid_ind] = 0.0
        else:
            arr_nlooks = 1

        rtc_image_gdal_ds = gdal.Open(path_rtc, gdal.GA_ReadOnly)

        for i_band in range(num_bands):

            band_ds = rtc_image_gdal_ds.GetRasterBand(i_band + 1)
            arr_rtc = band_ds.ReadAsArray()

            if i_band == 0:
                length = min(arr_rtc.shape[0], dim_mosaic[0] - offset_imgy)
                width = min(arr_rtc.shape[1], dim_mosaic[1] - offset_imgx)

            if (length != arr_rtc.shape[0] or
                    width != arr_rtc.shape[1]):
                # Image needs to be cropped to fit in the mosaic
                arr_rtc = arr_rtc[0:length, 0:width]

            if mosaic_mode.lower() == 'average':
                # Replace NaN values with 0
                arr_rtc[np.isnan(arr_rtc)] = 0.0

                arr_numerator[i_band,
                              offset_imgy: offset_imgy + length,
                              offset_imgx: offset_imgx + width] += \
                    arr_rtc * arr_nlooks

                if path_nlooks is not None:
                    arr_denominator[
                        offset_imgy: offset_imgy + length,
                        offset_imgx: offset_imgx + width] += arr_nlooks
                else:
                    arr_denominator[
                        offset_imgy: offset_imgy + length,
                        offset_imgx: offset_imgx + width] += np.asarray(
                        arr_rtc > 0, dtype=np.byte)

                continue

            arr_temp = arr_numerator[i_band, offset_imgy: offset_imgy + length,
                                     offset_imgx: offset_imgx + width].copy()
            if not np.isnan(no_data_value):
                arr_temp[arr_temp == no_data_value] = np.nan

            if i_band == 0 and mosaic_mode.lower() == 'first':
                ind = np.isnan(arr_temp)
            elif i_band == 0 and mosaic_mode.lower() == 'bursts_center':
                geotransform = rtc_image_gdal_ds.GetGeoTransform()

                arr_new_distance = _compute_distance_to_burst_center(
                    arr_rtc, geotransform)

                arr_distance_temp = arr_distance[
                    offset_imgy: offset_imgy + length,
                    offset_imgx: offset_imgx + width]
                ind = np.logical_or(np.isnan(arr_distance_temp),
                                    arr_new_distance <= arr_distance_temp)

                arr_distance_temp[ind] = arr_new_distance[ind]
                arr_distance[
                    offset_imgy: offset_imgy + length,
                    offset_imgx: offset_imgx + width] = arr_distance_temp

                del arr_distance_temp

            arr_temp[ind] = arr_rtc[ind]
            arr_numerator[i_band,
                          offset_imgy: offset_imgy + length,
                          offset_imgx: offset_imgx + width] = arr_temp

        rtc_image_gdal_ds = None
        nlooks_gdal_ds = None

    if mosaic_mode.lower() == 'average':
        # Mode: average
        # `arr_numerator` holds the accumulated sum. Now, we divide it
        # by `arr_denominator` to get the average value
        for i_band in range(num_bands):
            valid_ind = arr_denominator > 0
            arr_numerator[i_band][valid_ind] = \
                arr_numerator[i_band][valid_ind] / arr_denominator[valid_ind]

            arr_numerator[i_band][arr_denominator == 0] = np.nan

    mosaic_dict = {
        'mosaic_array': arr_numerator,
        'description_list': description_list,
        'length': dim_mosaic[0],
        'width': dim_mosaic[1],
        'num_bands': num_bands,
        'wkt_projection': wkt_projection,
        'xmin_mosaic': xmin_mosaic,
        'ymax_mosaic': ymax_mosaic,
        'posting_x': posting_x,
        'posting_y': posting_y
    }
    return mosaic_dict


def mosaic_single_output_file(list_rtc_images, list_nlooks, mosaic_filename,
                              mosaic_mode, scratch_dir='', geogrid_in=None,
                              temp_files_list=None, no_data_value=np.nan,
                              verbose=True):
    '''
    Mosaic RTC images saving the output into a single multi-band file

    Parameters
    -----------
        list_rtc: list
            List of the path to the rtc geobursts
        list_nlooks: list
            List of the nlooks raster that corresponds to list_rtc
        mosaic_filename: str
            Path to the output mosaic
        scratch_dir: str (optional)
            Directory for temporary files
        geogrid_in: isce3.product.GeoGridParameters, default: None
            Geogrid information to determine the output mosaic's shape and
            projection. The geogrid of the output mosaic will automatically
            determined when it is None
        temp_files_list: list (optional)
            Mutable list of temporary files. If provided,
            paths to the temporary files generated will be
            appended to this list
        verbose : bool
            Flag to enable/disable the verbose mode
    '''
    mosaic_dict = compute_mosaic_array(
        list_rtc_images, list_nlooks, mosaic_mode, scratch_dir=scratch_dir,
        geogrid_in=geogrid_in, temp_files_list=temp_files_list,
        verbose=verbose, no_data_value=no_data_value)

    arr_numerator = mosaic_dict['mosaic_array']
    description_list = mosaic_dict['description_list']
    length = mosaic_dict['length']
    width = mosaic_dict['width']
    num_bands = mosaic_dict['num_bands']
    wkt_projection = mosaic_dict['wkt_projection']
    xmin_mosaic = mosaic_dict['xmin_mosaic']
    ymax_mosaic = mosaic_dict['ymax_mosaic']
    posting_x = mosaic_dict['posting_x']
    posting_y = mosaic_dict['posting_y']

    # Retrieve the datatype information from the first input image
    reference_raster = gdal.Open(list_rtc_images[0], gdal.GA_ReadOnly)
    datatype_mosaic = reference_raster.GetRasterBand(1).DataType
    reference_raster = None

    # Write out the array
    drv_out = gdal.GetDriverByName('Gtiff')
    raster_out = drv_out.Create(mosaic_filename,
                                width, length, num_bands,
                                datatype_mosaic)

    raster_out.SetGeoTransform((xmin_mosaic, posting_x, 0,
                                ymax_mosaic, 0, posting_y))
    raster_out.SetProjection(wkt_projection)

    for i_band in range(num_bands):
        gdal_band = raster_out.GetRasterBand(i_band+1)
        gdal_band.WriteArray(arr_numerator[i_band])
        gdal_band.SetDescription(description_list[i_band])


def mosaic_multiple_output_files(
        list_rtc_images, list_nlooks, output_file_list, mosaic_mode,
        scratch_dir='', geogrid_in=None, temp_files_list=None, verbose=True):
    '''
    Mosaic RTC images saving each mosaicked band into a separate file

    Paremeters:
    -----------
        list_rtc_images: list
            List of the path to the rtc geobursts
        list_nlooks: list
            List of the nlooks raster that corresponds to list_rtc
        output_file_list: list
            Output file list
        mosaic_mode: str
            Mosaic mode. Choices: "average", "first", and "bursts_center"
        scratch_dir: str (optional)
            Directory for temporary files
        geogrid_in: isce3.product.GeoGridParameters, default: None
            Geogrid information to determine the output mosaic's shape and
            projection. The geogrid of the output mosaic will automatically
            determined when it is None
        temp_files_list: list (optional)
            Mutable list of temporary files. If provided,
            paths to the temporary files generated will be
            appended to this list
        verbose : bool
            Flag to enable/disable the verbose mode

    '''
    mosaic_dict = compute_mosaic_array(
        list_rtc_images, list_nlooks, mosaic_mode, scratch_dir=scratch_dir,
        geogrid_in=geogrid_in, temp_files_list=temp_files_list,
        verbose=verbose)

    arr_numerator = mosaic_dict['mosaic_array']
    length = mosaic_dict['length']
    width = mosaic_dict['width']
    num_bands = mosaic_dict['num_bands']
    wkt_projection = mosaic_dict['wkt_projection']
    xmin_mosaic = mosaic_dict['xmin_mosaic']
    ymax_mosaic = mosaic_dict['ymax_mosaic']
    posting_x = mosaic_dict['posting_x']
    posting_y = mosaic_dict['posting_y']

    if num_bands != len(output_file_list):
        error_str = (f'ERROR number of output files ({len(output_file_list)})'
                     ' does not match with the number'
                     f' of input bursts` bands ({num_bands})')
        raise ValueError(error_str)

    for i_band, output_file in enumerate(output_file_list):

        # Retrieve the datatype information from the first input image
        reference_raster = gdal.Open(list_rtc_images[0], gdal.GA_ReadOnly)
        datatype_mosaic = reference_raster.GetRasterBand(1).DataType
        reference_raster = None

        # Write out the array
        drv_out = gdal.GetDriverByName('Gtiff')
        nbands = 1
        raster_out = drv_out.Create(output_file,
                                    width, length, nbands,
                                    datatype_mosaic)

        raster_out.SetGeoTransform((xmin_mosaic, posting_x, 0,
                                    ymax_mosaic, 0, posting_y))

        raster_out.SetProjection(wkt_projection)

        # for i_band in range(num_bands):
        raster_out.GetRasterBand(1).WriteArray(arr_numerator[i_band])


def run(cfg):
    '''
    Run mosaic burst workflow with user-defined
    args stored in dictionary runconfig `cfg`
    Parameters
    ---------
    cfg: RunConfig
        RunConfig object with user runconfig options
    '''
    # Start tracking processing time
    t_start = time.time()
    logger.info("Starting the mosaic burst RTC products")

    processing_cfg = cfg.groups.processing

    scratch_path = cfg.groups.product_path_group.scratch_path
    input_list = cfg.groups.input_file_group.input_file_path
    mosaic_cfg = cfg.groups.processing.mosaic

    mosaic_mode = mosaic_cfg.mosaic_mode
    product_prefix = processing_cfg.mosaic.mosaic_prefix
    pol_list = copy.deepcopy(processing_cfg.polarizations)

    imagery_extension = 'tif'
    os.makedirs(scratch_path, exist_ok=True)

    # number of input directories and files
    num_input_path = len(input_list)
    if os.path.isdir(input_list[0]):
        if num_input_path > 1:
            logger.info('Multiple input directories are found.')
            logger.info('Mosaic is enabled for burst RTCs ')
            mosaic_flag = True
        else:
            logger.info('Singple input directories is found.')
            logger.info('Mosaic is diabled for single burst RTC ')
            mosaic_flag = True
    else:
        if num_input_path == 1:
            logger.info('Single input RTC is found.')
            logger.info('Mosaic is disabled for input RTC')
            mosaic_flag = False
        else:
            err_str = 'unable to process more than 1 images.'
            logger.error(err_str)
            raise ValueError(err_str)

    if mosaic_flag:
        print('Number of bursts to process:', num_input_path)

        freqA_path = '/data/'

        output_file_list = []
        nlooks_list = []
        mask_list = []
        epsg_list = []

        for ind, input_dir in enumerate(input_list):

            first_rtc_path_iter = glob.iglob(f'{input_dir}/*.tif')
            first_rtc_path = next(first_rtc_path_iter)
            if first_rtc_path:
                rtc_meta = dswx_sar_util.get_meta_from_tif(first_rtc_path)
                epsg_list.append(rtc_meta['epsg'])
            else:
                err_msg = 'RTC files not found.'
                raise FileExistsError(err_msg)

        epsg_output = majority_element(epsg_list)
        geogrid_in = dswx_geogrid.DSWXGeogrid()

        logger.info('All RTC bursts and associated masks will be mosaicked '
                    f'using the ESPG projection designated by {epsg_output}.')
        # for each directory, find metadata, and RTC files.
        for ind, input_dir in enumerate(input_list):

            # find HDF5 metadata
            layover_path = glob.glob(f'{input_dir}/*mask.tif')
            temp_mask_path = f'{scratch_path}/layover_{ind}.tif'
            # If both `*_mask.tif` and `*.h5` exists in RTC-S1 burst product 
            # directory:
            # The metadata in `*_mask.tif` has priority over HDF5 file.
            if len(layover_path) > 0:
                logger.info('Layover/shadow GeoTIFF is found.')
                if epsg_output != epsg_list[ind]:
                    logger.info(f'{layover_path[0]}, '
                                f'{epsg_list[ind]} -> {epsg_output}')
                    dswx_sar_util.change_epsg_tif(
                        input_tif=layover_path,
                        output_tif=temp_mask_path,
                        epsg_output=epsg_output,
                        output_nodata=255)
                    mask_list.append(temp_mask_path)

                    # update geogrid using new GeoTiff file.
                    geogrid_in.update_geogrid(temp_mask_path)

                else:
                    mask_list.append(layover_path[0])
                    geogrid_in.update_geogrid(layover_path[0])

            else:
                # If mask GeoTiff is not available,
                # layover/shadow mask may be saved in hdf5 metadata.
                metadata_path = glob.glob(f'{input_dir}/*h5')[0]
                logger.info('Metadata HDF5 is found.')

                with h5py.File(metadata_path) as meta_src:
                    if 'mask' in meta_src[freqA_path]:
                        mask_name = 'mask'
                    elif 'layoverShadowMask' in meta_src[freqA_path]:
                        mask_name = 'layoverShadowMask'
                    else:
                        mask_name = None
                if mask_name is not None:
                    save_h5_metadata_to_tif(
                        metadata_path,
                        data_path=f'{freqA_path}/{mask_name}',
                        output_tif_path=temp_mask_path,
                        epsg_output=epsg_output)
                    mask_list.append(temp_mask_path)
            if not mask_list:
                logger.warning('mask layer is not found!')

        # Check if metadata have common values on
        # poliarzation /track number/ direction fields
        output_dir_mosaic_raster = scratch_path

        # Mosaic sub-bursts imagery
        logger.info('mosaicking files:')
        rtc_burst_imagery_list = []

        for pol in pol_list:
            if pol in ['VV', 'VH', 'HV', 'HH']:
                rtc_burst_imagery_list = []

                for input_ind, input_dir in enumerate(input_list):
                    rtc_path_inputs = glob.glob(
                        f'{input_dir}/*_{pol}.tif')
                    if rtc_path_inputs:
                        rtc_path_input = rtc_path_inputs[0]

                        if epsg_output != epsg_list[input_ind]:
                            rtc_path_temp = \
                                f'{scratch_path}/temp_{pol}_{input_ind}.tif'
                            dswx_sar_util.change_epsg_tif(rtc_path_input,
                                                          rtc_path_temp,
                                                          epsg_output)
                            rtc_burst_imagery_list.append(rtc_path_temp)

                            # update geogrid using new GeoTiff file.
                            geogrid_in.update_geogrid(rtc_path_temp)
                        else:
                            rtc_burst_imagery_list.append(rtc_path_input)
                            geogrid_in.update_geogrid(rtc_path_input)
                    else:
                        print(f'polarzation {pol} is not found in {input_dir}')
                nlooks_list = []

                if len(rtc_burst_imagery_list) > 0:
                    geo_pol_filename = \
                        (f'{output_dir_mosaic_raster}/{product_prefix}_{pol}.'
                         f'{imagery_extension}')
                    logger.info(f'    {geo_pol_filename}')
                    output_file_list.append(geo_pol_filename)

                    mosaic_single_output_file(
                        rtc_burst_imagery_list, nlooks_list, geo_pol_filename,
                        mosaic_mode, scratch_dir=scratch_path,
                        geogrid_in=geogrid_in, temp_files_list=None)

        if mask_list:
            geo_mask_filename = \
                (f'{output_dir_mosaic_raster}/{product_prefix}_layovershadow_mask.'
                 f'{imagery_extension}')
            logger.info(f'    {geo_mask_filename}')
            output_file_list.append(geo_mask_filename)
            mosaic_single_output_file(
                mask_list, nlooks_list, geo_mask_filename,
                mosaic_mode, scratch_dir=scratch_path,
                geogrid_in=geogrid_in, temp_files_list=None,
                no_data_value=255)

        # save files as COG format.
        if processing_cfg.mosaic.mosaic_cog_enable:
            logger.info('Saving files as Cloud-Optimized GeoTIFFs (COGs)')
            for filename in output_file_list:
                if not filename.endswith('.tif'):
                    continue
                logger.info(f'    processing file: {filename}')
                dswx_sar_util._save_as_cog(
                    filename, scratch_path, logger,
                    compression='DEFLATE',
                    nbits=16)

            nlook_files = glob.glob(f'{scratch_path}/*nLooks_*.tif')
            for rmfile in nlook_files:
                os.remove(rmfile)

    t_time_end = time.time()
    logger.info(f'total processing time: {t_time_end - t_start} sec')


if __name__ == "__main__":
    '''Run mosaic rtc products from command line'''
    # load arguments from command line
    parser = _get_parser()

    # parse arguments
    args = parser.parse_args()

    mimetypes.add_type("text/yaml", ".yaml", strict=True)
    flag_first_file_is_text = 'text' in mimetypes.guess_type(
        args.input_yaml[0])[0]

    if flag_first_file_is_text:
        cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_s1', args)

    generate_log.configure_log_file(cfg.groups.log_file)

    # Run mosaic burst RTC workflow
    run(cfg)
