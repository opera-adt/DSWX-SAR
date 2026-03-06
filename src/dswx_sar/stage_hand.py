#!/usr/bin/env python3

# Hand staging
import argparse
import backoff
import math
import os
import time

import numpy as np
from osgeo import gdal, osr
from pystac_client import Client
import shapely.ops
import shapely.wkt
from shapely.geometry import LinearRing, Point, Polygon, box


# Enable exceptions
gdal.UseExceptions()

EARTH_APPROX_CIRCUMFERENCE = 40075017.
EARTH_RADIUS = EARTH_APPROX_CIRCUMFERENCE / (2 * np.pi)


def cmdLineParse():
    """
    Command line parser
    """
    parser = argparse.ArgumentParser(
        description="""
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
    parser.add_argument('--fill', dest='fill_missing', action='store',
                        default=True, help='Fill missing Hand tiles')
    parser.add_argument('-b', '--bbox', type=float, action='store',
                        dest='bbox', default=None, nargs='+',
                        help='Spatial bounding box in latitude/longitude (WSEN, decimal degrees)')
    return parser.parse_args()


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
            f'Could not determine projection for {lat}, {lon}')


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


def format_coords(lon, lat):
    """Convert longitude and latitude in float to
    string

    Parameters:
    ----------
    lon: float
        Longitude coordinate
    lat: float
        Latitude coordinate

    Returns:
    ----------
    lon_str: str
        Longitude string in 3 digit with hemisphere
    lat_str: str
        Latitude string in 2 digit with hemisphere

    """
    # Determine the hemisphere for latitude
    if lat >= 0:
        lat_hemi = 'N'
    else:
        lat_hemi = 'S'

    # Determine the hemisphere for longitude
    if lon >= 0:
        lon_hemi = 'E'
    else:
        lon_hemi = 'W'

    # Convert to string without the decimal portion
    lat_str = f"{lat_hemi}{int(abs(lat)):02}"
    lon_str = f"{lon_hemi}{int(abs(lon)):03}"

    return lon_str, lat_str


def get_geotiff_tile(geotiff_path):
    """Retrieve the 1-degree by 1-degree tile indices
    from the given GeoTIFF.

    Parameters:
    ----------
    geotiff_path: str
        Path to the GeoTIFF file.

    Returns: tuple
    --------
        Tuple of (x, y) indices representing the tile.
    """
    dataset = gdal.Open(geotiff_path)
    gt = dataset.GetGeoTransform()

    # Get the top-left corner of the GeoTIFF
    ulx, uly = int(np.round(gt[0])), int(np.round(gt[3]))
    return (ulx, uly)

def tiles_from_bbox(bbox):
    """Generate all 1-degree by 1-degree tiles covered by 
    the given bounding box.

    Parameters:
    ----------
    bbox: tuple
        A tuple containing (left, bottom, right, top) coordinates
        of the bounding box.

    Returns:
    --------
    tiles: tuple
        A set of (x, y) tile indices.
    """
    left, bottom, right, top = bbox
    tiles = set()
    # Find the upperleft coordiate from bounding box
    for x in range(math.floor(left), math.ceil(right)):
        for y in range(math.floor(bottom)+1, math.ceil(top)+1):
            tiles.add((x, y))

    return tiles

def create_zero_geotiff(tile, outpath, size=(1, 1), resolution=1):
    """
    Create a new GeoTIFF filled with zeros for the specified tile.

    Parameters:
    ----------
    tile: tuple
        A tuple (x, y) representing the top-left corner of the tile.
    outpath:
        The path to save the new GeoTIFF.
    size: tuple
        A tuple representing the width and height of the tile in degrees.
    resolution: float
        The resolution of the raster (i.e., pixel size).
    """
    # Set up the new file's spatial properties
    ulx, uly = tile
    width = int(size[0] / resolution)
    height = int(size[1] / resolution)

    # Create the file with a single band, Float32 type
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(outpath, width, height, 1, gdal.GDT_Float32)

    # Define the spatial properties for the GeoTIFF
    ds.SetGeoTransform((ulx, resolution, 0,
                        uly, 0, -resolution))

    # Set the spatial reference system
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS84
    ds.SetProjection(srs.ExportToWkt())

    # Write zeros to the raster band
    band = ds.GetRasterBand(1)
    band.WriteArray(np.zeros((height, width), dtype=np.float32))
    band.FlushCache()

    ds = None  # Close the file

def fill_missing_tile(geotiff_list, bbox):
    """Identify tiles that are missing based on the given bounding
    box and a list of GeoTIFF files.

    Parameters:
    ----------
    geotiff_list: list
        A list of paths to existing GeoTIFF files.
    bbox: tuple
        A tuple containing (left, bottom, right, top)
        coordinates of the bounding box.

    Returns:
    --------
    missing_tiles: set
        A set of tuples where each tuple represents
        the top-left corner (x, y) of a missing
        1-degree by 1-degree tile.
    """
    expected_tiles = tiles_from_bbox(bbox)
    existing_tiles = set()

    for file in geotiff_list:
        tile = get_geotiff_tile(file)
        existing_tiles.add(tile)

    missing_tiles = expected_tiles - existing_tiles

    return missing_tiles


def download_hand(polys, outfile, fill_flag):
    """Download Hand from ASF bucket

    Parameters:
    ----------
    polys: shapely.geometry.Polygon
        List of shapely polygons
    outfile:
        Path to the output Hand file to be staged
    """
    client = Client.open(
        'https://stac.asf.alaska.edu/')
    aws_base_path = \
        'https://glo-30-hand.s3.us-west-2.amazonaws.com/v1/2021'
    # Download Hand for each polygon/epsg
    file_prefix = os.path.splitext(outfile)[0]
    hand_list = []

    for n2, poly in enumerate(polys):

        xmin = poly.bounds[0]
        ymin = poly.bounds[1]
        xmax = poly.bounds[2]
        ymax = poly.bounds[3]
        # ASF search does not correctly work for tiles
        # smaller than 1 x 1 deg
        if (xmax - xmin < 1) and (ymax - ymin < 1):
            ymin = np.floor(ymin)
            ymax = np.ceil(ymax)
            lon_str, lat_str = format_coords(xmin, ymin)
            aws_paths = [
                f'{aws_base_path}/Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_HAND.tif']
        else:
            search_results = client.search(
                collections=['glo-30-hand'],
                bbox=[xmin, ymin, xmax, ymax],
                )
            aws_paths = []
            for item in search_results.items():
                aws_path = item.assets['data'].href
                aws_paths.append(aws_path)

        for n, download_path in enumerate(aws_paths):
            outpath = f'{file_prefix}_{n}_{n2}.tif'
            command_line = f'wget {download_path} -O {outpath}'
            os.system(command_line)
            hand_list.append(outpath)

        # The HAND tiles for ocean area do not exit. For those areas,
        # manually create the GeoTiffs with 0 values.
        if fill_flag:
            missing_tiles = fill_missing_tile(
                hand_list,
                bbox=[xmin, ymin, xmax, ymax])
            if missing_tiles:
                print('Filling is enabled.')
                print('Missing', missing_tiles)

                for tileid, tile in enumerate(missing_tiles):
                    outpath = f'{file_prefix}_filled_{tileid}_{n2}.tif'
                    create_zero_geotiff(tile, outpath,
                                        size=(1, 1),
                                        resolution=0.01)
                    hand_list.append(outpath)
    if hand_list:
        gdal.BuildVRT(outfile, hand_list)
    else:
        err_str = 'No Geotiff file found'
        raise FileNotFoundError(err_str)

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
            dummy_lat, dummy_lon, dummy_z = trans.TransformPoint(lx, ly, 0)
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


def check_hand_overlap(HandFilepath, polys):
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
    hand_meta = get_meta_from_tif(HandFilepath)
    ulx, xres, _, uly, _, yres = hand_meta['geotransform']
    lrx = ulx + (hand_meta['width'] * xres)
    lry = uly + (hand_meta['length'] * yres)
    poly_Hand = Polygon([(ulx, uly),
                         (ulx, lry),
                         (lrx, lry),
                         (lrx, uly)])

    # Initialize epsg
    epsg = hand_meta['epsg']
    if epsg != 4326:
        polys = transform_polygon_coords_to_lonlat(poly_Hand, epsg)

    perc_area = 0
    for poly in polys:
        perc_area += (poly.intersection(poly_Hand).area / poly.area) * 100
    return perc_area

def check_aws_asf_connection():
    """Check connection to ASF API.
       Throw exception if no connection is established
    """
    try:
        Client.open('https://stac.asf.alaska.edu/')
    except Exception:
        errmsg = 'No access to ASF HANDs'
        raise ValueError(errmsg)

def apply_margin_polygon(polygon, margin_in_km=5):
    '''Convert margin from km to degrees and
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
    poly = apply_margin_polygon(poly, opts.margin)

    # Check dateline crossing. Returns list of polygons
    polys, _ = check_dateline(poly)
    if os.path.isfile(opts.filepath):
        print('Check overlap with user-provided Hand')
        overlap = check_hand_overlap(opts.filepath, polys)
        if overlap < 75.:
            print('Insufficient Hand coverage. Errors might occur')
        print(f'Hand coverage is {overlap} %')
    else:
        check_aws_asf_connection()

        # Download Hand
        download_hand(polys,
                      opts.outfile,
                      opts.fill_missing)
        print('Done, Hand store locally')

if __name__ == '__main__':
    opts = cmdLineParse()
    main(opts)
