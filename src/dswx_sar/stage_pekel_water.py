#!/usr/bin/env python3

# Pekel's Surface Water Map staging
# https://global-surface-water.appspot.com/download
# Pekel, JF., Cottam, A., Gorelick, N. et al. 
# High-resolution mapping of global surface water and its long-term changes. 
# Nature 540, 418–422 (2016). https://doi.org/10.1038/nature20584

import argparse
import os
import backoff
import logging

# import boto3
import urllib.request

import numpy as np
from osgeo import gdal, osr
from shapely.geometry import LinearRing, Polygon, box
import shapely.ops
import shapely.wkt


logger = logging.getLogger('stage peke water map')

# Enable exceptions
gdal.UseExceptions()

# [TODO] need to update the bucket
"""Name of the default S3 bucket containing the full JRC Pikel's water map to crop from.
"""
S3_REF_WATER_BUCKET = "opera-reference-water"
PEKEL_GOOGLE_SOURCE = "http://storage.googleapis.com/global-surface-water/downloads2021/"

EARTH_APPROX_CIRCUMFERENCE = 40075017.
EARTH_RADIUS = EARTH_APPROX_CIRCUMFERENCE / (2 * np.pi)


def get_parser():
    """Returns the command line parser for stage_pekel_water.py"""
    parser = argparse.ArgumentParser(
        description="Stage and verify reference water map for processing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-p', '--product', type=str, action='store',
                        help='Input reference RTC HDF5 product')
    parser.add_argument('-o', '--output', type=str, action='store',
                        default='reference_water.vrt', dest='outfile',
                        help='Output Reference water filepath (VRT format).')
    parser.add_argument('--s3-bucket', type=str, action='store',
                        default=S3_REF_WATER_BUCKET, dest='s3_bucket',
                        help='Name of the S3 bucket containing the full JRC '
                             'Global water map to extract from.')
    parser.add_argument('-s', '--source', type=str, action='store',
                        default="googleapi", dest='data_source',
                        help='Data source containing the full Pekel water '
                             'map to extract from.')
    parser.add_argument('-w', '--watermap_type', type=str, action='store',
                        default='seasonality', dest='watermap_type',
                        help='Water map type used as reference water map.')
    parser.add_argument('-b', '--bbox', type=float, action='store',
                        dest='bbox', default=None, nargs='+',
                        help='Spatial bounding box in '
                             'latitude/longitude (WSEN, decimal degrees)')
    parser.add_argument('-m', '--margin', type=int, action='store',
                        default=5, help='Margin for bounding box in km.')
    parser.add_argument('--log', '--log-file', dest='log_file',
                        type=str, help='Log file')

    return parser


def get_meta_from_tif(tif_file_name):
    """Read metadata from geotiff

    Parameters
    ----------
    input_tif_str: str
        geotiff file path to read the band
    Returns
    -------
    meta_dict: dict
        dictionary containing geotransform, projection, image size,
        utmzone, and epsg code.
    """
    tif_gdal = gdal.Open(tif_file_name)
    meta_dict = {}
    meta_dict['geotransform'] = tif_gdal.GetGeoTransform()
    meta_dict['projection'] = tif_gdal.GetProjection()
    meta_dict['length'] = tif_gdal.RasterYSize
    meta_dict['width'] = tif_gdal.RasterXSize
    proj = osr.SpatialReference(wkt=meta_dict['projection'])
    meta_dict['utmzone'] = proj.GetUTMZone()
    output_epsg = proj.GetAttrValue('AUTHORITY',1)
    meta_dict['epsg'] = output_epsg
    tif_gdal = None

    return meta_dict


def transform_polygon_coords_to_lonlat(polys, epsgs):
    """Transform coordinates of polys (list of polygons)
       to target epsgs (list of EPSG codes)

    Parameters:
    ----------
    polys: shapely.Geometry.Polygon
        List of shapely polygons
    epsgs: list, str
        List of EPSG codes corresponding to
        elements in polys
    """
    # Transform each point of the perimeter in target EPSG coordinates
    llh = osr.SpatialReference()
    llh.ImportFromEPSG(4326)
    tgt = osr.SpatialReference()

    xmin, ymin, xmax, ymax = [], [], [], []
    tgt_x, tgt_y = [], []
    if not isinstance(polys, list):
        polys = [polys]
        epsgs = [epsgs]

    for poly, epsg in zip(polys, epsgs):
        x, y = poly.exterior.coords.xy
        tgt.ImportFromEPSG(int(epsg))
        trans = osr.CoordinateTransformation(tgt, llh)
        for lx, ly in zip(x, y):
            dummy_lat, dummy_lon, _ = trans.TransformPoint(lx, ly, 0)
            tgt_x.append(dummy_lon)
            tgt_y.append(dummy_lat)
        xmin.append(min(tgt_x))
        ymin.append(min(tgt_y))
        xmax.append(max(tgt_x))
        ymax.append(max(tgt_y))
    # return a polygon
    poly = Polygon([(min(xmin), min(ymin)), (min(xmin), max(ymax)),
                    (max(xmax), max(ymax)), (max(xmax), min(ymin))])

    return poly


def get_geo_polygon(ref_tif):
    """Get polygon from GeoTiff image

    Parameters:
    -----------
    ref_tif: str
        Path to RTC product to stage the pekel's water map

    Returns:
    -------
    poly: shapely.Geometry.Polygon
        Bounding polygon corresponding to RTC
    """
    # Prepare SLC dataset input
    meta_dict = get_meta_from_tif(ref_tif)
    geo_transform = meta_dict['geotransform']
    xmin = geo_transform[0]
    ymax = geo_transform[3]
    xmax = geo_transform[0] + geo_transform[1] * meta_dict['width']
    ymin = geo_transform[3] + geo_transform[5] * meta_dict['length']

    poly = box(xmin, ymin, xmax, ymax)
    if meta_dict['epsg'] == 4326:
        return Polygon(poly)
    else:
        poly = Polygon([(xmin, ymin), (xmin, ymax),
                        (xmax, ymax), (xmax, ymin)])
        epsg_input = meta_dict['epsg']
        polyout = transform_polygon_coords_to_lonlat(poly, epsg_input)
        return polyout


def apply_margin_polygon(polygon, margin_in_km=5):
    '''Convert margin from km to degrees and apply to polygon

    Parameters
    ----------
    polygon: shapely.Geometry.Polygon
        Bounding polygon covering the area on the
        ground over which download the DEM
    margin_in_km: np.float
        Buffer in km to add to polygon

    Returns
    ------
    poly_with_margin: shapely.Geometry.box
        Bounding box with margin applied
    '''
    lon_min, lat_min, lon_max, lat_max = polygon.bounds
    lat_worst_case = max([lat_min, lat_max])

    # Convert margin from km to degrees
    lat_margin = margin_km_to_deg(margin_in_km)
    lon_margin = margin_km_to_longitude_deg(margin_in_km, lat=lat_worst_case)

    poly_with_margin = box(lon_min - lon_margin, max([lat_min - lat_margin, -90]),
                           lon_max + lon_margin, min([lat_max + lat_margin, 90]))
    return poly_with_margin


def margin_km_to_deg(margin_in_km):
    '''
    Converts a margin value from km to degrees

    Parameters
    ----------
    margin_in_km: np.float
        Margin in km

    Returns
    -------
    margin_in_deg: np.float
        Margin in degrees
    '''
    km_to_deg_at_equator = 1000. / (EARTH_APPROX_CIRCUMFERENCE / 360.)
    margin_in_deg = margin_in_km * km_to_deg_at_equator

    return margin_in_deg


def margin_km_to_longitude_deg(margin_in_km, lat=0):
    '''Converts margin from km to degrees as a function of
    latitude

    Parameters
    ----------
    margin_in_km: np.float
        Margin in km
    lat: np.float
        Latitude to use for the conversion

    Returns
    ------
    delta_lon: np.float
        Longitude margin as a result of the conversion
    '''
    delta_lon = (180 * 1000 * margin_in_km /
                (np.pi * EARTH_RADIUS * np.cos(np.pi * lat / 180)))
    return delta_lon


def determine_polygon(ref_rtc, bbox=None, margin_in_km=50):
    """
    Determine bounding polygon using MGRS tile code or user-defined bounding box.

    Parameters
    ----------
    tile_code: str
        MGRS tile code corresponding to the polygon to derive.
    bbox: list, optional
        Bounding box with lat/lon coordinates (decimal degrees) in the form of
        [West, South, East, North]. If provided, takes precedence over the tile
        code.
    margin_in_km: float, optional
        Margin in kilometers to be added to MGRS bounding box obtained from the
        MGRS `tile_code`. This margin is not added to the bounding box
        defined from the input parameter `bbox`.

    Returns
    -------
    poly: shapely.Geometry.Polygon
        Bounding polygon corresponding to the MGRS tile code or bbox shape on
        the ground.

    """
    if bbox:
        logger.info('Determining polygon from bounding box')
        poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
    else:
        logger.info('Determine polygon from RTC radar grid and orbit')
        poly = get_geo_polygon(ref_rtc)

    poly = apply_margin_polygon(poly, margin_in_km)

    logger.debug(f'Derived polygon {str(poly)}')

    return poly


def check_dateline(poly):
    """Split `poly` if it crosses the dateline.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        Input polygon.

    Returns
    -------
    polys : list of shapely.geometry.Polygon
         A list containing: the input polygon if it didn't cross
        the dateline, or two polygons otherwise (one on either
        side of the dateline).
    """
    xmin, _, xmax, _ = poly.bounds
    dateline_crossing = False
    # Check dateline crossing
    if (xmax - xmin) > 180.0:
        dateline = shapely.wkt.loads('LINESTRING( 180.0 -90.0, 180.0 90.0)')

        # build new polygon with all longitudes between 0 and 360
        x, y = poly.exterior.coords.xy
        new_x = (k + (k <= 0.) * 360 for k in x)
        new_ring = LinearRing(zip(new_x, y))

        # Split input polygon
        # (https://gis.stackexchange.com/questions/232771/splitting-polygon-by-linestring-in-geodjango_)
        merged_lines = shapely.ops.linemerge([dateline, new_ring])
        border_lines = shapely.ops.unary_union(merged_lines)
        decomp = shapely.ops.polygonize(border_lines)

        polys = list(decomp)

        # The Copernicus DEM used for NISAR processing has a longitude
        # range [-180, +180]. The current version of gdal.Translate
        # does not allow to perform dateline wrapping. Therefore, coordinates
        # above 180 need to be wrapped down to -180 to match the Copernicus
        # DEM longitude range
        for polygon_count in range(2):
            x, y = polys[polygon_count].exterior.coords.xy
            if not any([k > 180 for k in x]):
                continue

            # Otherwise, wrap longitude values down to 360 deg
            x_wrapped_minus_360 = np.asarray(x) - 360
            polys[polygon_count] = Polygon(zip(x_wrapped_minus_360, y))

        assert (len(polys) == 2)
        dateline_crossing = True
    else:
        # If dateline is not crossed, treat input poly as list
        polys = [poly]

    return polys, dateline_crossing


@backoff.on_exception(backoff.expo, Exception, max_tries=8, max_value=32)
def translate_pekel_water(vrt_filename, output_path, x_min, x_max, y_min, y_max):
    """
    Translate a Worldcover map from the esa-worldcover bucket.

    Notes
    -----
    This function is decorated to perform retries using exponential backoff to
    make the remote call resilient to transient issues stemming from network
    access, authorization and AWS throttling (see "Query throttling" section at
    https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html).

    Parameters
    ----------
    vrt_filename: str
        Path to the input VRT file
    output_path: str
        Path to the translated output GTiff file
    x_min: float
        Minimum longitude bound of the sub-window
    x_max: float
        Maximum longitude bound of the sub-window
    y_min: float
        Minimum latitude bound of the sub-window
    y_max: float
        Maximum latitude bound of the sub-window

    """
    logger.info("Translating JRC Global water map for projection window "
                f"{str([x_min, y_max, x_max, y_min])} "
                f"to {output_path}")
    ds = gdal.Open(vrt_filename, gdal.GA_ReadOnly)

    # update cropping coordinates to not exceed the input DEM bounding box
    input_x_min, xres, _, input_y_max, _, yres = ds.GetGeoTransform()
    length = ds.GetRasterBand(1).YSize
    width = ds.GetRasterBand(1).XSize
    input_y_min = input_y_max + (length * yres)
    input_x_max = input_x_min + (width * xres)

    x_min = max(x_min, input_x_min)
    x_max = min(x_max, input_x_max)
    y_min = max(y_min, input_y_min)
    y_max = min(y_max, input_y_max)

    gdal.Translate(
        output_path, ds, format='GTiff', projWin=[x_min, y_max, x_max, y_min]
    )


def download_reference_water_map(polys,
                                 watermap_source,
                                 watermap_bucket,
                                 watermap_type,
                                 outfile):
    """
    Download a Pekel's map from the s3 bucket.

    Parameters:
    ----------
    polys: list of shapely.geometry.Polygon
        List of shapely polygons.
    watermap_source : str
        Name of the S3 bucket containing the full Worldcover map to download from.
    watermap_type : str
        Type of Pekel's water map (i.e., seasonality, occurrence)
    outfile:
        Path to the where the output reference water file is to be staged.
    """
    # Download Pekel's water map for each polygon/epsg
    file_prefix = os.path.splitext(outfile)[0]
    wc_list = []
    counter = 0
    for idx, poly in enumerate(polys):
        x_min, y_min, x_max, y_max = poly.bounds

        if watermap_source == 'googleapi':
            x_min_10 = int((x_min // 10) * 10)
            y_min_10 = int((y_min // 10) * 10)
            x_max_10 = int(((x_max // 10) + 1) * 10)
            y_max_10 = int(((y_max // 10) + 1) * 10)
            # Pekel's water map is only available from 50S to 90N.
            x_min_10 = np.max([x_min_10, -180])
            x_max_10 = np.min([x_max_10, 180])
            y_min_10 = np.max([y_min_10, -50])
            y_max_10 = np.min([y_max_10, 90])

            # Pekel's water map is stored as 10 x 10 deg.
            for x_10 in range(x_min_10, x_max_10, 10):
                for y_10 in range(y_min_10, y_max_10, 10):
                    w_or_e = 'W' if x_10 < 0 else 'E'
                    s_or_n = 'S' if y_10 < 0 else 'N'

                    x_min_10_str = int(np.abs(x_10))
                    y_min_10_str = int(np.abs(y_10))
                    google_api_filename = f'{watermap_type}_{x_min_10_str}{w_or_e}_{y_min_10_str}{s_or_n}v1_4_2021.tif'
                    
                    url = os.path.join(PEKEL_GOOGLE_SOURCE, watermap_type, google_api_filename)
                    code = urllib.request.urlopen(url).getcode()
                    if (code != 404):
                        logger.info(f"Downloading {url} ({str(counter)})")
                        urllib.request.urlretrieve(url, google_api_filename)
                        wc_list.append(google_api_filename)
                    else:
                        logger.info(f"{url} not found")
                    counter += 1
        else:
            # [TODO] need to update the path
            vrt_filename = (
                f'/vsis3/{watermap_bucket}/{worldcover_ver}/{worldcover_year}/'
                f'ESA_WorldCover_10m_{worldcover_year}_{worldcover_ver}_Map_AWS.vrt'
            )

            output_path = f'{file_prefix}_{idx}.tif'
            wc_list.append(output_path)
            translate_pekel_water(vrt_filename, output_path, x_min, x_max, y_min, y_max)

    # Build vrt with downloaded maps
    gdal.BuildVRT(outfile, wc_list)


def check_aws_connection(worldcover_bucket):
    """Check connection to the provided S3 bucket.

    Parameters
    ----------
    worldcover_bucket : str
        Name of the bucket to use with the connection test.

    Raises
    ------
    RuntimeError
       If no connection can be established.
    """
    s3 = boto3.resource('s3')
    obj = s3.Object(worldcover_bucket, 'readme.html')

    try:
        logger.info(f'Attempting test read of s3://{obj.bucket_name}/{obj.key}')
        obj.get()['Body'].read()
        logger.info('Connection test successful.')
    except Exception:
        errmsg = (f'No access to the {worldcover_bucket} s3 bucket. '
                  f'Check your AWS credentials and re-run the code.')
        raise RuntimeError(errmsg)


def main(opts):
    """
    Main script to execute Worldcover map staging.

    Parameters:
    ----------
    opts : argparse.Namespace
        Arguments parsed from the command-line.

    """
    # Check if MGRS tile code or bbox are provided
    if opts.product is None and opts.bbox is None:
        errmsg = ("Need to provide reference RTC Geotiff image, MGRS tile code or bounding box. "
                  "Cannot download JRC Global water map.")
        raise ValueError(errmsg)

    # Make sure that output file has VRT extension
    if not opts.outfile.lower().endswith('.vrt'):
        err_msg = "JRC Global water output filename extension is not .vrt"
        raise ValueError(err_msg)

    # Determine polygon based on MGRS grid reference with a margin, or bbox
    poly = determine_polygon(opts.product,
                             opts.bbox,
                             opts.margin)

    # Check connection to the S3 bucket
    logger.info(f'Checking connection to AWS S3 {opts.s3_bucket} bucket.')
    if opts.data_source != 'googleapi':
        check_aws_connection(opts.s3_bucket)

    # Check dateline crossing. Returns list of polygons
    polys, _ = check_dateline(poly)

    # Download JRC Global water map(s)
    download_reference_water_map(polys,
                                 opts.data_source,
                                 opts.s3_bucket,
                                 opts.watermap_type,
                                 opts.outfile)

    logger.info(f"Done, Pekel's water map stored locally to {opts.outfile}")


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(filename=args.log_file,
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG)
    main(args)