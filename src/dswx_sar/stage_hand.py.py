#!/usr/bin/env python3

# Hand staging

import argparse
import os

import numpy as np
import shapely.ops
import shapely.wkt
from osgeo import gdal, osr
from shapely.geometry import LinearRing, Point, Polygon, box
from pystac_client import Client

# Enable exceptions
gdal.UseExceptions()

EARTH_APPROX_CIRCUMFERENCE = 40075017.
EARTH_RADIUS = EARTH_APPROX_CIRCUMFERENCE / (2 * np.pi)


def cmdLineParse():
    """
     Command line parser
    """
    parser = argparse.ArgumentParser(description="""
                                     Stage and verify Hand for processing. """,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p', '--product', type=str, action='store',
                        help='Input reference RTC HDF5 product')
    parser.add_argument('-o', '--output', type=str, action='store',
                        default='hand.vrt', dest='outfile',
                        help='Output Hand filepath (VRT format).')
    parser.add_argument('-f', '--path', type=str, action='store',
                        dest='filepath', default='file',
                        help='Filepath to user Hand.')
    parser.add_argument('-m', '--margin', type=int, action='store',
                        default=5, help='Margin for Hand bounding box (km)')
    parser.add_argument('-b', '--bbox', type=float, action='store',
                        dest='bbox', default=None, nargs='+',
                        help='Spatial bounding box in latitude/longitude (WSEN, decimal degrees)')
    return parser.parse_args()

def get_meta_from_tif(tif_file_name):
    
    tif_gdal = gdal.Open(tif_file_name)
    meta_dict = dict()
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
    else:
        # If dateline is not crossed, treat input poly as list
        polys = [poly]

    return polys


def determine_polygon(ref_slc, bbox=None):
    """Determine bounding polygon using RTC radar grid/orbit
    or user-defined bounding box

    Parameters:
    ----------
    ref_slc: str
        Filepath to reference RTC product
    bbox: list, float
        Bounding box with lat/lon coordinates (decimal degrees)
        in the form of [West, South, East, North]

    Returns:
    -------
    poly: shapely.Geometry.Polygon
        Bounding polygon corresponding to RTC perimeter
        or bbox shape on the ground
    """
    if bbox is not None:
        print('Determine polygon from bounding box')
        poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
    else:
        print('Determine polygon from RTC radar grid and orbit')
        poly = get_geo_polygon(ref_slc)

    return poly


def point2epsg(lon, lat):
    """Return EPSG code based on point lat/lon

    Parameters:
    ----------
    lat: float
        Latitude coordinate of the point
    lon: float
        Longitude coordinate of the point

    Returns:
    -------
    epsg code corresponding to the point lat/lon coordinates
    """
    if lon >= 180.0:
        lon = lon - 360.0
    if lat >= 75.0:
        return 3413
    elif lat <= -75.0:
        return 3031
    elif lat > 0:
        return 32601 + int(np.round((lon + 177) / 6.0))
    elif lat < 0:
        return 32701 + int(np.round((lon + 177) / 6.0))
    else:
        raise ValueError(
            'Could not determine projection for {0},{1}'.format(lat, lon))


def get_geo_polygon(ref_tif, pts_per_edge=5):
    """Create polygon (EPSG:4326) using RTC radar grid and orbits

    Parameters:
    -----------
    ref_slc: str
        Path to RTC product to stage the Hand for
    min_height: float
        Global minimum height (in m) for Hand interpolator
    max_height: float
        Global maximum height (in m) for Hand interpolator
    pts_per_edge: float
        Number of points per edge for min/max bounding box computation

    Returns:
    -------
    poly: shapely.Geometry.Polygon
        Bounding polygon corresponding to RTC perimeter on the ground
    """

    # Prepare SLC dataset input
    meta_dict = get_meta_from_tif(ref_tif)
    geo_transform = meta_dict['geotransform'] 
    print(geo_transform)
    xmin = geo_transform[0]
    ymax = geo_transform[3]
    xmax = geo_transform[0] + geo_transform[1]*meta_dict['width']
    ymin = geo_transform[3] + geo_transform[5]*meta_dict['length']

    poly = box(xmin, ymin, xmax, ymax)
    if meta_dict['epsg'] == 4326:
        return Polygon(poly)
    else:
        poly = [(xmin, ymin), (xmin, ymax),
                (xmax, ymax), (xmax, ymin)]
        epsg_input = meta_dict['epsg']
        polyout = transform_polygon_coords_to_lonlat(poly, epsg_input)
        return polyout


def determine_projection(polys):
    """Determine EPSG code for each polygon in polys.
    EPSG is computed for a regular list of points. EPSG
    is assigned based on a majority criteria.

    Parameters:
    -----------
    polys: shapely.Geometry.Polygon
        List of shapely Polygons
    Returns:
    --------
    epsg:
        List of EPSG codes corresponding to elements in polys
    """

    epsg = []

    # Make a regular grid based on polys min/max latitude longitude
    for p in polys:
        xmin, ymin, xmax, ymax = p.bounds
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 250),
                             np.linspace(ymin, ymax, 250))
        x = xx.flatten()
        y = yy.flatten()

        # Query to determine the zone
        zones = []
        for lx, ly in zip(x, y):
            # Create a point with grid coordinates
            pp = Point(lx, ly)
            # If Point is in polys, compute EPSG
            if pp.within(p):
                zones.append(point2epsg(lx, ly))

        # Count different EPSGs
        vals, counts = np.unique(zones, return_counts=True)
        # Get the ESPG for Polys
        epsg.append(vals[np.argmax(counts)])

    return epsg

def download_hand(polys, margin, outfile):
    """Download Hand from ASF bucket

    Parameters:
    ----------
    polys: shapely.geometry.Polygon
        List of shapely polygons
    epsg: str, list
        List of EPSG codes corresponding to polys
    margin: float
        Buffer margin (in km) applied for Hand download
    outfile:
        Path to the output Hand file to be staged
    """
    client = Client.open('https://stac.asf.alaska.edu/')
    # convert margin to degree (approx formula)
    margin = margin / 40000 * 360

    # Download Hand for each polygon/epsg
    file_prefix = os.path.splitext(outfile)[0]
    hand_list = []
   
    poly2 = polys.buffer(margin)
    print(poly2.bounds)    
    ymin = np.floor(poly2.bounds[1])
    ymax = np.ceil(poly2.bounds[3]-1)
    xmin = np.floor(poly2.bounds[0]+1)
    xmax = np.ceil(poly2.bounds[2])
    
    print(xmin, ymin, xmax, ymax)
    search_results = client.search(
        collections=['glo-30-hand'],
        bbox=[xmin, ymin, xmax, ymax],
        )
    print('search', search_results,[xmin, ymin, xmax, ymax] )
    for n, item in enumerate(search_results.items()):
        aws_path = item.assets['data'].href
        # hand_list.append(aws_path)
        outpath = f'{file_prefix}_{n}.tiff'
        command_line = f'wget {aws_path} -O {outpath}'
        print(command_line)
        os.system(command_line)
        hand_list.append(outpath)

    # Build vrt with downloaded Hands
    gdal.BuildVRT(outfile, hand_list)


def transform_polygon_coords_to_lonlat(polys, epsg):
    """Transform coordinates of polys (list of polygons)
       to target epsgs (list of EPSG codes)

    Parameters:
    ----------
    polys: shapely.Geometry.Polygon
        List of shapely polygons
    epsg: list, str
        List of EPSG codes corresponding to
        elements in polys
    """

    # Transform each point of the perimeter in target EPSG coordinates
    llh = osr.SpatialReference()
    llh.ImportFromEPSG(4326)
    tgt = osr.SpatialReference()

    xmin, ymin, xmax, ymax = [], [], [], []
    tgt_x, tgt_y = [], []
    for poly, epsg in zip(polys, epsgs):
        x, y = poly.exterior.coords.xy
        tgt.ImportFromEPSG(int(epsg))
        trans = osr.CoordinateTransformation(llh, tgt)
        for lx, ly in zip(x, y):
            dummy_x, dummy_y, dummy_z = trans.TransformPoint(ly, lx, 0)
            tgt_x.append(dummy_x)
            tgt_y.append(dummy_y)
        xmin.append(min(tgt_x))
        ymin.append(min(tgt_y))
        xmax.append(max(tgt_x))
        ymax.append(max(tgt_y))
    # return a polygon
    poly = [Polygon([(min(xmin), min(ymin)), (min(xmin), max(ymax)),
                     (max(xmax), max(ymax)), (max(xmax), min(ymin))])]

    return poly


def check_Hand_overlap(HandFilepath, polys):
    """Evaluate overlap between user-provided Hand
       and Hand that stage_Hand.py would download
       based on RTC or bbox provided information

    Parameters:
    ----------
    HandFilepath: str
        Filepath to the user-provided Hand
    polys: shapely.geometry.Polygon
        List of polygons computed from RTC or bbox

    Returns:
    -------
    perc_area: float
        Area (in percentage) covered by the intersection between the
        user-provided Hand and the one downloadable by stage_Hand.py
    """

    # Get local Hand edge coordinates
    Hand_meta = get_meta_from_tif(HandFilepath)
    ulx, xres, xskew, uly, yskew, yres = Hand_meta['geotransform']
    lrx = ulx + (Hand_meta['width'] * xres)
    lry = uly + (Hand_meta['length'] * yres)
    poly_Hand = Polygon([(ulx, uly), (ulx, lry), (lrx, lry), (lrx, uly)])

    # Initialize epsg
    epsg = Hand_meta['epsg']

    if epsg != 4326:
        polys = transform_polygon_coords_to_lonlat(polys, epsg)

    perc_area = 0
    for poly in polys:
        perc_area += (poly.intersection(poly_Hand).area / poly.area) * 100
    return perc_area


def check_aws_connection():
    """Check connection to AWS s3://nisar-Hand bucket
       Throw exception if no connection is established
    """
    import boto3
    s3 = boto3.resource('s3')
    obj = s3.Object('nisar-Hand', 'EPSG3031/EPSG3031.vrt')
    try:
        obj.get()['Body'].read()
    except Exception:
        errmsg = 'No access to nisar-Hand s3 bucket. Check your AWS credentials' \
                 'and re-run the code'
        raise ValueError(errmsg)


def check_aws_asf_connection():
    """Check connection to AWS s3://nisar-Hand bucket
       Throw exception if no connection is established
    """
    client = Client.open('https://stac.asf.alaska.edu/')    
    s3 = boto3.resource('s3')
    obj = s3.Object('nisar-Hand', 'EPSG3031/EPSG3031.vrt')
    try:
        obj.get()['Body'].read()
    except Exception:
        errmsg = 'No access to nisar-Hand s3 bucket. Check your AWS credentials' \
                 'and re-run the code'
        raise ValueError(errmsg)



def apply_margin_polygon(polygon, margin_in_km=5):
    '''
    Convert margin from km to degrees and
    apply to polygon

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
    '''
    Converts margin from km to degrees as a function of
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


def main(opts):
    """Main script to execute Hand staging

    Parameters:
    ----------
    opts : argparse.ArgumentParser
        Argument parser
    """

    # Check if RTC or bbox are provided
    if (opts.product is None) & (opts.bbox is None):
        errmsg = "Need to provide reference reference geotiff or bounding box. " \
                 "Cannot download Hand"
        raise ValueError(errmsg)

    # Make sure that output file has VRT extension
    if not opts.outfile.lower().endswith('.vrt'):
        err_msg = "Hand output filename extension is not .vrt"
        raise ValueError(err_msg)

    # Determine polygon based on RTC info or bbox
    poly = determine_polygon(opts.product, opts.bbox)
    print('POLYPOLY', poly)
    poly = apply_margin_polygon(poly, opts.margin)

    # Check dateline crossing. Returns list of polygons
    polys = check_dateline(poly)

    if os.path.isfile(opts.filepath):
        print('Check overlap with user-provided Hand')
        overlap = check_Hand_overlap(opts.filepath, polys)
        if overlap < 75.:
            print('Insufficient Hand coverage. Errors might occur')
        print(f'Hand coverage is {overlap} %')
    else:
        try:
            check_aws_connection()
        except ImportError:
            import warnings
            warnings.warn('boto3 is required to verify AWS connection'
                          'proceeding without verifying connection')

        # Download Hand
        print('polygons', polys)
        # download_hand(polys, opts.margin, opts.outfile)
        print('Done, Hand store locally')


if __name__ == '__main__':
    opts = cmdLineParse()
    main(opts)
