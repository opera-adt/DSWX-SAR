'''
A module to mosaic Sentinel-1 geobursts from RTC workflow
'''

import os
import time
import glob
import numpy as np
import logging
import h5py
from osgeo import osr, gdal
import mimetypes

from dswx_sar.dswx_runconfig import _get_parser, RunConfig
from dswx_sar import dswx_sar_util

logger = logging.getLogger('dswx_s1')

def check_reprojection(geogrid_mosaic,
                       rtc_image: str,
                       nlooks_image: str = None) -> bool:
    '''
    Check if the reprojection is required to mosaic input raster
    Parameters:
    -----------
    geogrid_mosaic: isce3.product.GeoGridParameters
        Mosaic geogrid
    rtc_image: str
        Path to the geocoded RTC image
    nlooks_image: str (optional)
        Path to the nlooks raster
    Returns:
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
    if raster_nlooks is not None:
        rasters_to_check += [raster_nlooks]

    for raster in rasters_to_check:
        geotransform = raster.GetGeoTransform()
        projection = raster.GetProjection()

        x0 = geotransform[0]
        dx = geotransform[1]
        y0 = geotransform[3]
        dy = geotransform[5]

        # check spacing
        if dx != geogrid_mosaic.spacing_x:
            flag_requires_reprojection = True
            return flag_requires_reprojection

        if dy != geogrid_mosaic.spacing_y:
            flag_requires_reprojection = True
            return flag_requires_reprojection

        # check projection
        srs_mosaic = osr.SpatialReference()
        srs_mosaic.ImportFromEPSG(geogrid_mosaic.epsg)

        if projection != srs_mosaic.ExportToWkt():
            srs_1 = osr.SpatialReference()
            srs_1.SetWellKnownGeogCS(projection)

            srs_2 = osr.SpatialReference()
            srs_2.SetWellKnownGeogCS(projection)

            if not srs_1.IsSame(srs_2):
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

def check_consistency_metadata(list_metadata_hdf5, cfg):
    '''
    Check RTC metadata if they have same polarizations
    paremeters:
    -----------
        list_metadata_hdf5: list
            List of the path to the rtc metadata
        cfg: RunConfig
            Input runconfig
    '''
    input_pol_list = cfg.groups.processing.polarizations
    # [TODO] currently, input polarizations are 4 characters (e.g. VHVH)
    re_input_pol_list = []
    for input_pol in input_pol_list:
        if len(input_pol) > 3:
            if input_pol[0:2] == input_pol[2:4]:
                re_input_pol_list.append(input_pol[0:2])

    common_parent_path = '/science/SENTINEL1/'
    grid_path = f'{common_parent_path}/RTC/grids/'
    freqA_path = f'{grid_path}/frequencyA'
    id_path  = f'{common_parent_path}/identification/'

    track_number_list = []
    orbit_directs = []
    number_pol = []

    for h5_file_path in list_metadata_hdf5:

        with h5py.File(h5_file_path, 'r') as src_h5:
            temp_pols = list(src_h5[f'{freqA_path}/listOfPolarizations'])
            burst_pols = [x.decode() for x in temp_pols]
            number_pol.append(len(burst_pols))

            for input_pol in re_input_pol_list:
                if input_pol not in burst_pols:
                    raise ValueError(f'User-given polarizations are not found in metadata'
                             f' file: {os.path.basename(h5_file_path)}')

            orbit_direction = src_h5[f'{id_path}/orbitPassDirection'][()]
            orbit_directs.append(orbit_direction.decode())

            track_number =  src_h5[f'{id_path}/trackNumber'][()]
            track_number_list.append(track_number)

    if len(set(orbit_directs)) > 1:
        # for track_ind, orbit_cand in enumerate(orbit_directs):
        raise ValueError(f'different orbit directions are found in input metadata')

    if len(set(track_number_list)) > 1:
        # for track_ind, track_cand in enumerate(track_number_list):
        raise ValueError(f'different track numbers are found in input metadata')

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
    print(f'{data_path} - > {output_tif_path}')
    freqA_path = '/science/SENTINEL1/RTC/grids/frequencyA'
    
    with h5py.File(h5_meta_path, 'r') as src_h5:
        data = np.array(src_h5[data_path])
        xcoord = np.array(src_h5[f'{freqA_path}/xCoordinates'])
        ycoord = np.array(src_h5[f'{freqA_path}/yCoordinates'])
        xres = np.array(src_h5[f'{freqA_path}/xCoordinateSpacing'])
        yres = np.array(src_h5[f'{freqA_path}/yCoordinateSpacing'])
        epsg = np.array(src_h5[f'{freqA_path}/projection'])
    
    dtype = data.dtype
    gdal_dtype = dswx_sar_util.np2gdal_conversion[str(dtype)]
    
    geotransform = [xcoord[0], float(xres), 0, 
                    ycoord[0], 0, float(yres)]
    print(geotransform)
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(int(epsg))               # WGS84 lat/long
    projection = srs.ExportToWkt()
    output_filename = os.path.basename(output_tif_path)
    output_dirname = os.path.dirname(output_tif_path)

    if epsg == epsg_output:
        
        dswx_sar_util.save_raster_gdal(data, 
                    output_tif_path, 
                    geotransform,
                    projection, 
                    scratch_dir=output_dirname,
                    DataType=dtype)
    else:
        output_tif_temp_dir_path = os.path.dirname(output_tif_path)
        output_tif_temp_base_path = f'{os.path.basename(output_tif_path)}_temp.tif'
        output_tif_temp_path = os.path.join(output_tif_temp_dir_path, 
                                            output_tif_temp_base_path)
        dswx_sar_util.save_raster_gdal(data, 
            output_tif_temp_path, 
            geotransform,
            projection, 
            scratch_dir=output_tif_temp_dir_path,
            DataType=dtype)

        opt = gdal.WarpOptions(dstSRS=f'EPSG:{epsg_output}',
                     xRes=xres, 
                     yRes=xres,
                     resampleAlg='nearest',
                     format='GTIFF')
        ds = gdal.Warp(output_tif_path, output_tif_temp_path, options=opt)   
        ds = None 

        os.remove(output_tif_temp_path)

def compute_weighted_mosaic_array(list_rtc_images, list_nlooks,
                     geogrid_in=None, verbose = True):
    '''
    Mosaic S-1 geobursts and return the mosaic as dictionary
    paremeters:
    -----------
        list_rtc: list
            List of the path to the rtc geobursts
        list_nlooks: list
            List of the nlooks raster that corresponds to list_rtc
        geogrid_in: isce3.product.GeoGridParameters, default: None
            Geogrid information to determine the output mosaic's shape and projection
            The geogrid of the output mosaic will automatically determined when it is None
    Returns:
        mosaic_dict: dict
            Mosaic dictionary
    '''

    num_raster = len(list_rtc_images)

    num_bands = None
    posting_x = None
    posting_y = None

    list_geo_transform = np.zeros((num_raster, 6))
    list_dimension = np.zeros((num_raster, 2), dtype=np.int32)
    list_epsg = []
    for i, path_rtc in enumerate(list_rtc_images):

        if verbose:
            print(f'loading geocoding info: {i+1} of {num_raster}')

        raster_in = gdal.Open(path_rtc, gdal.GA_ReadOnly)
        list_geo_transform[i, :] = raster_in.GetGeoTransform()
        list_dimension[i, :] = (raster_in.RasterYSize, raster_in.RasterXSize)
        list_epsg.append(raster_in.GetProjectionRef())
        # Check if the number of bands are consistent over the input RTC rasters
        if num_bands is None:
            num_bands = raster_in.RasterCount
            continue
        elif num_bands != raster_in.RasterCount:
            raise ValueError(f'Anomaly detected on # of bands from source'
                             f' file: {os.path.basename(path_rtc)}')

        raster_in = None


    if geogrid_in is None:
        # determine GeoTransformation, posting, dimension, and projection from the input raster
        for i in range(num_raster):
            if list_geo_transform[:, 1].max() == list_geo_transform[:, 1].min():
                posting_x = list_geo_transform[0,1]
            else:
                print(list_geo_transform[:, 1].max(), list_geo_transform[:, 1].min())
            if list_geo_transform[:, 5].max() == list_geo_transform[:, 5].min():
                posting_y = list_geo_transform[0,5]

        # determine the dimension and the upper left corner of the output mosaic
        xmin_mosaic = list_geo_transform[:, 0].min()
        ymax_mosaic = list_geo_transform[:, 3].max()
        xmax_mosaic = (list_geo_transform[:, 0] + list_geo_transform[:, 1]*list_dimension[:, 1]).max()
        ymin_mosaic = (list_geo_transform[:, 3] + list_geo_transform[:, 5]*list_dimension[:, 0]).min()

        dim_mosaic = (int(np.ceil((ymin_mosaic - ymax_mosaic) / posting_y)),
                      int(np.ceil((xmax_mosaic - xmin_mosaic) / posting_x)))

        wkt_obj = gdal.Open(list_rtc_images[0], gdal.GA_ReadOnly)
        wkt_projection = wkt_obj.GetProjectionRef()
        wkt_obj = None
        del wkt_obj

    else:
        # Directly bring the geogrid information from the input parameter
        xmin_mosaic = geogrid_in.start_x
        ymax_mosaic = geogrid_in.start_y
        posting_x = geogrid_in.spacing_x
        posting_y = geogrid_in.spacing_y

        dim_mosaic = (geogrid_in.length, geogrid_in.width)

        xmax_mosaic = xmin_mosaic + posting_x*dim_mosaic[1]
        ymin_mosaic = ymax_mosaic + posting_y*dim_mosaic[0]
        dim_mosaic = (geogrid_in.length, geogrid_in.width)

        srs_mosaic = osr.SpatialReference()
        srs_mosaic.ImportFromEPSG(geogrid_in.epsg)
        wkt_projection = srs_mosaic.ExportToWkt()

    if verbose:
        print(f'mosaic dimension: {dim_mosaic}, #bands: {num_bands}')

    arr_numerator = np.zeros((num_bands, dim_mosaic[0], dim_mosaic[1]))
    arr_denominator = np.zeros(dim_mosaic)

    for i, path_rtc in enumerate(list_rtc_images):
        path_nlooks = list_nlooks[i]

        if not path_nlooks:
            path_nlooks = None

        if verbose:
            print(f'mosaicking: {i+1} of {num_raster}: {os.path.basename(path_rtc)}')

        if geogrid_in is not None and check_reprojection(
                geogrid_in, path_rtc, path_nlooks):
            # reprojection not implemented
            raise NotImplementedError

        # TODO: if geogrid_in is None, check reprojection

        # calculate the burst RTC's offset wrt. the output mosaic in the image coordinate
        offset_imgx = int((list_geo_transform[i,0] - xmin_mosaic) / posting_x + 0.5)
        offset_imgy = int((list_geo_transform[i,3] - ymax_mosaic) / posting_y + 0.5)

        if verbose:
            print(f'image offset [x, y] = [{offset_imgx}, {offset_imgy}]')
        raster_rtc = gdal.Open(path_rtc,0)
        arr_rtc = raster_rtc.ReadAsArray()

        #reshape arr_rtc when it is a singleband raster: to make it compatible in the for loop below
        if num_bands==1:
            arr_rtc=arr_rtc.reshape((1, arr_rtc.shape[0], arr_rtc.shape[1]))

        # Replace NaN values with 0
        arr_rtc[np.isnan(arr_rtc)] = 0.0
        if not path_nlooks or path_nlooks == None:
            arr_nlooks = np.ones([arr_rtc.shape[1], arr_rtc.shape[2]])
            arr_nlooks[arr_rtc[0,:,:]==0] = 0
        else:
            raster_nlooks = gdal.Open(path_nlooks, 0)
            arr_nlooks = raster_nlooks.ReadAsArray()

        invalid_ind = np.isnan(arr_nlooks)
        arr_nlooks[invalid_ind] = 0.0

        for i_band in range(num_bands):
            arr_numerator[i_band,
                          offset_imgy:offset_imgy+list_dimension[i, 0],
                          offset_imgx:offset_imgx+list_dimension[i, 1]] += \
                            arr_rtc[i_band, :, :] * arr_nlooks

        arr_denominator[offset_imgy:offset_imgy + list_dimension[i, 0],
                        offset_imgx:offset_imgx + list_dimension[i, 1]] += \
                            arr_nlooks

        raster_rtc = None
        raster_nlooks = None

    for i_band in range(num_bands):
        valid_ind = np.where(arr_denominator > 0)
        arr_numerator[i_band][valid_ind] = \
            arr_numerator[i_band][valid_ind] / arr_denominator[valid_ind]

        invalid_ind = np.where(arr_denominator == 0)
        arr_numerator[i_band][invalid_ind] = np.nan

    mosaic_dict = {
        'mosaic_array': arr_numerator,
        'length': dim_mosaic[0],
        'width': dim_mosaic[1],
        'num_bands': num_bands,
        'wkt_projection': wkt_projection,
        'xmin_mosaic': xmin_mosaic,
        'ymax_mosaic': ymax_mosaic,
        'posting_x': posting_x,
        'posting_y': posting_y,
        'epsg_set': list_epsg,
    }
    return mosaic_dict


def get_point_epsg(lat, lon):
    '''
    Get EPSG code based on latitude and longitude
    coordinates of a point
    Parameters
    ----------
    lat: float
        Latitude coordinate of the point
    lon: float
        Longitude coordinate of the point
    Returns
    -------
    epsg: int
        UTM zone, Polar stereographic (North / South)
    '''

    # "Warp" longitude value into [-180.0, 180.0]
    if (lon >= 180.0) or (lon <= -180.0):
        lon = (lon + 180.0) % 360.0 - 180.0

    if lat >= 75.0:
        return 3413
    elif lat <= -75.0:
        return 3031
    elif lat > 0:
        return 32601 + int(np.round((lon + 177) / 6.0))
    elif lat < 0:
        return 32701 + int(np.round((lon + 177) / 6.0))
    else:
        err_str = "'Could not determine EPSG for {0}, {1}'.format(lon, lat))"
        logger.error(err_str)
        raise ValueError(err_str)

def majority_element(num_list):
        idx, ctr = 0, 1
        
        for i in range(1, len(num_list)):
            if num_list[idx] == num_list[i]:
                ctr += 1
            else:
                ctr -= 1
                if ctr == 0:
                    idx = i
                    ctr = 1
        
        return num_list[idx]

def generate_geogrids(bursts, geo_dict, dem):
    '''
    Compute frame and bursts geogrids
    Parameters
    ----------
    bursts: list[s1reader.s1_burst_slc.Sentinel1BurstSlc]
        List of S-1 burst SLC objects
    geo_dict: dict
        Dictionary containing runconfig processing.geocoding
        parameters
    dem_file: str
        Dem file
    Returns
    -------
    geogrid_all: isce3.product.GeoGridParameters
        Frame geogrid
    geogrids_dict: dict
        Dict containing bursts geogrids indexed by burst_id
    '''

    # TODO use `dem_raster` to update `epsg`, if not provided in the runconfig
    # dem_raster = isce3.io.Raster(dem)

    # Unpack values from geocoding dictionary
    # epsg = geo_dict['output_epsg']
    # xmin = geo_dict['top_left']['x']
    # ymax = geo_dict['top_left']['y']
    # x_spacing = geo_dict['x_posting']
    # y_spacing_positive = geo_dict['y_posting']
    # xmax = geo_dict['bottom_right']['x']
    # ymin = geo_dict['bottom_right']['y']
    # x_snap = geo_dict['x_snap']
    # y_snap = geo_dict['y_snap']

    xmin_all_bursts = np.inf
    ymax_all_bursts = -np.inf
    xmax_all_bursts = -np.inf
    ymin_all_bursts = np.inf

    # Compute burst EPSG
    y_list = []
    x_list = []
    for burst_pol in bursts.values():
        pol_list = list(burst_pol.keys())
        burst = burst_pol[pol_list[0]]
        y_list.append(burst.center.y)
        x_list.append(burst.center.x)
    y_mean = np.nanmean(y_list)
    x_mean = np.nanmean(x_list)
    epsg = get_point_epsg(y_mean, x_mean)
    assert 1024 <= epsg <= 32767

    # Check spacing in X direction
    if x_spacing is not None and x_spacing <= 0:
        err_str = 'Pixel spacing in X/longitude direction needs to be positive'
        err_str += f' (x_spacing: {x_spacing})'
        logger.error(err_str)
        raise ValueError(err_str)
    elif x_spacing is None and epsg == 4326:
        x_spacing = -0.00027
    elif x_spacing is None:
        x_spacing = -30

    # Check spacing in Y direction
    if y_spacing_positive is not None and y_spacing_positive <= 0:
        err_str = 'Pixel spacing in Y/latitude direction needs to be positive'
        err_str += f'(y_spacing: {y_spacing_positive})'
        logger.error(err_str)
        raise ValueError(err_str)
    elif y_spacing_positive is None and epsg == 4326:
        y_spacing = -0.00027
    elif y_spacing_positive is None:
        y_spacing = -30
    else:
        y_spacing = - y_spacing_positive

    geogrids_dict = {}
    for burst_id, burst_pol in bursts.items():
        pol_list = list(burst_pol.keys())
        burst = burst_pol[pol_list[0]]
        if burst_id in geogrids_dict.keys():
            continue

        radar_grid = burst.as_isce3_radargrid()
        orbit = burst.orbit

        geogrid_burst = None
        if len(bursts) > 1 or None in [xmin, ymax, xmax, ymin]:
            # Initialize geogrid with estimated boundaries
            geogrid_burst = isce3.product.bbox_to_geogrid(
                radar_grid, orbit, isce3.core.LUT2d(), x_spacing, y_spacing,
                epsg)

            if len(bursts) == 1 and None in [xmin, ymax, xmax, ymin]:
                # Check and further initialize geogrid
                geogrid_burst = assign_check_geogrid(
                    geogrid_burst, xmin, ymax, xmax, ymin)
            geogrid_burst = intersect_geogrid(
                geogrid_burst, xmin, ymax, xmax, ymin)
        else:
            # If all the start/end coordinates have been assigned,
            # initialize the geogrid with them
            width = _grid_size(xmax, xmin, x_spacing)
            length = _grid_size(ymin, ymax, y_spacing)
            geogrid_burst = isce3.product.GeoGridParameters(
                xmin, ymax, x_spacing, y_spacing, width, length,
                epsg)

        # Check snap values
        check_snap_values(x_snap, y_snap, x_spacing, y_spacing)
        # Snap coordinates
        geogrid = snap_geogrid(geogrid_burst, x_snap, y_snap)

        xmin_all_bursts = min([xmin_all_bursts, geogrid.start_x])
        ymax_all_bursts = max([ymax_all_bursts, geogrid.start_y])
        xmax_all_bursts = max([xmax_all_bursts,
                                geogrid.start_x + geogrid.spacing_x *
                                geogrid.width])
        ymin_all_bursts = min([ymin_all_bursts,
                                geogrid.start_y + geogrid.spacing_y *
                                geogrid.length])

        geogrids_dict[burst_id] = geogrid

    if xmin is None:
        xmin = xmin_all_bursts
    if ymax is None:
        ymax = ymax_all_bursts
    if xmax is None:
        xmax = xmax_all_bursts
    if ymin is None:
        ymin = ymin_all_bursts

    width = _grid_size(xmax, xmin, x_spacing)
    length = _grid_size(ymin, ymax, y_spacing)
    geogrid_all = isce3.product.GeoGridParameters(
        xmin, ymax, x_spacing, y_spacing, width, length, epsg)

    # Check snap values
    check_snap_values(x_snap, y_snap, x_spacing, y_spacing)

    # Snap coordinates
    geogrid_all = snap_geogrid(geogrid_all, x_snap, y_snap)
    return geogrid_all, geogrids_dict


def read_metadata_epsg(h5_meta_path):
    freqA_path = '/science/SENTINEL1/RTC/grids/frequencyA'
    with h5py.File(h5_meta_path, 'r') as src_h5:
        xcoord = np.array(src_h5[f'{freqA_path}/xCoordinates'])
        ycoord = np.array(src_h5[f'{freqA_path}/yCoordinates'])
        xres = np.array(src_h5[f'{freqA_path}/xCoordinateSpacing'])
        yres = np.array(src_h5[f'{freqA_path}/yCoordinateSpacing'])
        epsg = np.array(src_h5[f'{freqA_path}/projection'])

    meta_dict = dict()
    meta_dict['xspacing'] = xres
    meta_dict['yspacing'] = yres
    meta_dict['epsg'] = epsg
    return meta_dict


def compute_weighted_mosaic_raster(list_rtc_images, list_nlooks, geo_filename,
                    geogrid_in=None, verbose = True):
    '''
    Mosaic the snapped S1 geobursts
    paremeters:
    -----------
        list_rtc: list
            List of the path to the rtc geobursts
        list_nlooks: list
            List of the nlooks raster that corresponds to list_rtc
        geo_filename: str
            Path to the output mosaic
        geogrid_in: isce3.product.GeoGridParameters, default: None
            Geogrid information to determine the output mosaic's shape and projection
            The geogrid of the output mosaic will automatically determined when it is None
    '''
    mosaic_dict = compute_weighted_mosaic_array(list_rtc_images, list_nlooks,
                                   geogrid_in=geogrid_in, verbose = verbose)

    arr_numerator = mosaic_dict['mosaic_array']
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
    raster_out = drv_out.Create(geo_filename,
                                width, length, num_bands,
                                datatype_mosaic)

    raster_out.SetGeoTransform((xmin_mosaic, posting_x, 0, ymax_mosaic, 0, posting_y))
    
    # srs = osr.SpatialReference()            # establish encoding
    # srs.ImportFromEPSG(int(epsg))               # WGS84 lat/long
    raster_out.SetProjection(wkt_projection)
    # raster_out.SetProjection(wkt_projection_input)

    for i_band in range(num_bands):
        raster_out.GetRasterBand(i_band+1).WriteArray(arr_numerator[i_band])


def compute_weighted_mosaic_raster_single_band(list_rtc_images, list_nlooks,
                                output_file_list,
                                geogrid_in=None, verbose = True):
    '''
    Mosaic the snapped S1 geobursts
    paremeters:
    -----------
        list_rtc: list
            List of the path to the rtc geobursts
        list_nlooks: list
            List of the nlooks raster that corresponds to list_rtc
        output_file_list: list
            Output file list
        geogrid_in: isce3.product.GeoGridParameters, default: None
            Geogrid information to determine the output mosaic's shape and projection
            The geogrid of the output mosaic will automatically determined when it is None
    '''
    mosaic_dict = compute_weighted_mosaic_array(list_rtc_images, list_nlooks,
                                     geogrid_in=geogrid_in, verbose = verbose)

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
                    f' does not match with the number'
                     f' of input bursts` bands ({num_bands})')
        raise ValueError(error_str)

    for i_band, output_file in enumerate(output_file_list):

        # Retrieve the datatype information from the first input image
        reference_raster = gdal.Open(list_rtc_images[0], gdal.GA_ReadOnly)
        datatype_mosaic = reference_raster.GetRasterBand(1).DataType
        reference_raster = None

        # Write out the array
        drv_out = gdal.GetDriverByName('Gtiff')
        raster_out = drv_out.Create(output_file,
                                    width, length, num_bands,
                                    datatype_mosaic)

        raster_out.SetGeoTransform((xmin_mosaic, posting_x, 0, ymax_mosaic, 0, posting_y))

        raster_out.SetProjection(wkt_projection)

        for i_band in range(num_bands):
            raster_out.GetRasterBand(i_band+1).WriteArray(arr_numerator[i_band])

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
    inundated_vege_cfg = processing_cfg.inundated_vegetation

    scratch_path = cfg.groups.product_path_group.scratch_path
    input_list = cfg.groups.input_file_group.input_file_path

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
            mosaic_flag = False
    else:
        if num_input_path == 1:
            logger.info('Single input RTC is found.')
            logger.info('Mosaic is disabled for input RTC')
            mosaic_flag = False
        else:
            err_str = f'unable to process more than 1 images.'
            logger.error(err_str)
            raise ValueError(err_str)

    if mosaic_flag:
        print('Number of bursts to process:', num_input_path)

        pol_list = cfg.groups.processing.polarizations

        common_parent_path = '/science/SENTINEL1/'
        grid_path = f'{common_parent_path}/RTC/grids/'
        freqA_path = f'{grid_path}/frequencyA'
        id_path  = f'{common_parent_path}/identification/'

        output_file_list = []   
        metadata_list = []
        nlooks_list = []
        mask_list = []
        epsg_list = []

        for ind, input_dir in enumerate(input_list):
            # find HDF5 metadata
            metadata_path = glob.glob(f'{input_dir}/*h5')[0]
            epsg_list.append(read_metadata_epsg(metadata_path)['epsg'])
            epsg_output = majority_element(epsg_list)

        # for each directory, find metadata, and RTC files. 
        for ind, input_dir in enumerate(input_list):
    
            # find HDF5 metadata
            metadata_path = glob.glob(f'{input_dir}/*h5')[0]
            metadata_list.append(metadata_path)

            temp_nlook_path = glob.glob(f'{input_dir}/*_nlooks.tif')
            # if nlook file is not found, nlook in metadata is saveed.
            if not temp_nlook_path:
                temp_nlook_path = f'{scratch_path}/nLooks_{ind}.tif'
                try:
                    save_h5_metadata_to_tif(metadata_path,
                                            data_path=f'{freqA_path}/numberOfLooks',
                                            output_tif_path=temp_nlook_path, 
                                            epsg_output=epsg_output)
                    nlooks_list.append(temp_nlook_path)
                except:
                    nlooks_list.append(None)
            else:
                nlooks_list.append(temp_nlook_path[0])

            # layover/shadow mask is saveed from hdf5 metadata.
            temp_mask_path = f'{scratch_path}/layover_{ind}.tif'
            save_h5_metadata_to_tif(metadata_path,
                                        data_path=f'{freqA_path}/layoverShadowMask',
                                        output_tif_path=temp_mask_path, 
                                        epsg_output=epsg_output)
            mask_list.append(temp_mask_path)

        # Check if metadata have common values on
        # poliarzation /track number/ direction fields
        check_consistency_metadata(metadata_list, cfg)

        output_dir_mosaic_raster = scratch_path
        product_prefix = processing_cfg.mosaic.mosaic_prefix
        imagery_extension = 'tif'
        output_metadata_dict = {}
        mosaic_geogrid_dict = {}
        logger.info(f'mosaicking files:')

        # mosaic dual polarization RTCs separately. 
        for pol in pol_list:

            # Mosaic output file name that will be stored
            geo_pol_filename = \
                (f'{output_dir_mosaic_raster}/{product_prefix}_{pol}.'
                 f'{imagery_extension}')
            logger.info(f'    {geo_pol_filename}')
            output_file_list.append(geo_pol_filename)

            # list the rtc files
            rtc_burst_imagery_list = []
            for input_ind, input_dir in enumerate(input_list):
                rtc_path_input = glob.glob(f'{input_dir}/*_{pol}.tif')[0]
                if epsg_output != epsg_list[input_ind]:
                    rtc_path_temp = f'{scratch_path}/temp_{pol}_{input_ind}.tif'
                    dswx_sar_util.change_epsg_tif(rtc_path_input, rtc_path_temp, epsg_output)
                    rtc_burst_imagery_list.append(rtc_path_temp)
                else:
                    rtc_burst_imagery_list.append(rtc_path_input)

            # mosaic burst RTCs 
            compute_weighted_mosaic_raster_single_band(
                rtc_burst_imagery_list, nlooks_list,
                list([geo_pol_filename]), None, verbose=False)

        # mosaic layover/shadow mask
        geo_mask_filename = \
            (f'{output_dir_mosaic_raster}/{product_prefix}_layovershadow_mask.'
                f'{imagery_extension}')
        logger.info(f'    {geo_mask_filename}')
        output_file_list.append(geo_mask_filename)
        compute_weighted_mosaic_raster(
            mask_list, nlooks_list,
            geo_mask_filename, None, verbose=False)

        if inundated_vege_cfg.enabled and inundated_vege_cfg.mode =='time_series':

            # mosaic dual polarization RTCs separately. 
            # [TODO]
            for pol in ['VV']:
                print(f'{rtc_stack_dir}/rtc_*')
                rtc_scenes_dir = glob.glob(f'{rtc_stack_dir}/rtc_*')
                for rtc_scene_dir in rtc_scenes_dir:
                    print('rtc_scene', rtc_scene_dir)
                    rtc_scene_base = os.path.basename(rtc_scene_dir)

                    # Mosaic output file name that will be stored
                    geo_pol_filename = \
                        (f'{output_dir_mosaic_raster}/{rtc_scene_base}_{pol}.'
                        f'{imagery_extension}')
                    logger.info(f'    {geo_pol_filename} hahah')
                    output_file_list.append(geo_pol_filename)

                    # list the rtc 
                    rtc_bursts_dir = glob.glob(f'{rtc_scene_dir}/t*iw*')

                    epsg_list = []
                    for ind, input_dir in enumerate(rtc_bursts_dir):
                        # find HDF5 metadata
                        metadata_path = glob.glob(f'{input_dir}/*h5')[0]
                        epsg_list.append(read_metadata_epsg(metadata_path)['epsg'])
                        epsg_output = majority_element(epsg_list)

                    rtc_burst_imagery_list = []
                    nlooks_list = []
                    if os.path.exists(f'{scratch_path}/rtc_stack_nLooks_*.tif'):
                        os.remove(f'{scratch_path}/rtc_stack_nLooks_*.tif')

                    for input_ind, rtc_burst_dir in enumerate(rtc_bursts_dir):
                        rtc_path_input = glob.glob(f'{rtc_burst_dir}/*_{pol}.tif')[0]
                        rtc_basename = os.path.basename(rtc_burst_dir)
                        if epsg_output != epsg_list[input_ind]:
                            os.makedirs(f'{scratch_path}/reprojected/', exist_ok=True)
                            rtc_path_temp = f'{scratch_path}/reprojected/{rtc_basename}_reprojected_{pol}_{input_ind}.tif'
                            dswx_sar_util.change_epsg_tif(rtc_path_input, rtc_path_temp, epsg_output)
                            rtc_burst_imagery_list.append(rtc_path_temp)
                        else:
                            rtc_burst_imagery_list.append(rtc_path_input)

                        # find HDF5 metadata
                        metadata_path = glob.glob(f'{rtc_burst_dir}/*h5')[0]

                        temp_nlook_path = glob.glob(f'{rtc_burst_dir}/*_nlooks.tif')
                        # if nlook file is not found, nlook in metadata is saveed.
                        if not temp_nlook_path:
                            temp_nlook_path = f'{scratch_path}/rtc_stack_nLooks_{input_ind}.tif'
                            # try:
                            save_h5_metadata_to_tif(metadata_path,
                                                    data_path=f'{freqA_path}/numberOfLooks',
                                                    output_tif_path=temp_nlook_path, 
                                                    epsg_output=epsg_output)
                            nlooks_list.append(temp_nlook_path)
                            # except:
                            #     nlooks_list.append(None)
                        else:
                            nlooks_list.append(temp_nlook_path[0])
                    print(len(nlooks_list), len(rtc_burst_imagery_list))
                    # mosaic burst RTCs 
                    compute_weighted_mosaic_raster_single_band(
                        rtc_burst_imagery_list, nlooks_list,
                        list([geo_pol_filename]), None, verbose=False)


        # save files as COG format. 
        if processing_cfg.mosaic.mosaic_cog_enable:
            logger.info(f'Saving files as Cloud-Optimized GeoTIFFs (COGs)')
            for filename in output_file_list:
                if not filename.endswith('.tif'):
                    continue
                logger.info(f'    processing file: {filename}')
                dswx_sar_util._save_as_cog(filename, scratch_path, logger,
                            compression='ZSTD',
                            nbits=16)
            for rmfile in mask_list:
                os.remove(rmfile)
            nlook_files = glob.glob(f'{scratch_path}/*nLooks_*.tif')
            for rmfile in nlook_files:
                os.remove(rmfile)

    t_time_end =time.time()
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

    # Run mosaic burst RTC workflow
    run(cfg)