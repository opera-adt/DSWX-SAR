#!/usr/bin/env python
import argparse
import h5py
import numpy as np
from scipy.spatial import ConvexHull
from osgeo import osr, gdal
from shapely.geometry import Polygon
import geopandas as gpd

def _get_parser():
    parser = argparse.ArgumentParser(
        description='Update Track/Frame of NISAR format ALOS data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Inputs
    parser.add_argument('input_file',
                        type=str,
                        help='Input images')
    parser.add_argument('track_frame',
                        type=str,
                        help='Input images')
    return parser


def assign_frame_track(file, track, frame):
    with h5py.File(file, 'a') as file:

        # Delete the existing dataset
        del file['/science/LSAR/identification/trackNumber']
        del file['/science/LSAR/identification/frameNumber']

        # Create a new dataset with the same name but with your new data
        file.create_dataset('/science/LSAR/identification/trackNumber', data=track)
        file.create_dataset('/science/LSAR/identification/frameNumber', data=frame)

def read_frame_track(file):
    with h5py.File(file) as src:
        print(src['/science/LSAR/identification/trackNumber'][()])
        print(src['/science/LSAR/identification/frameNumber'][()])
        print(src['/science/LSAR/GCOV/grids/frequencyA/listOfPolarizations'][()])


def utm_to_ll(x, y, epsg):
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(int(epsg))
    dst = osr.SpatialReference()            # establish encoding
    dst.ImportFromEPSG(4326)
    transformation_utm_to_ll = osr.CoordinateTransformation(srs, dst)

    # Search MGRS tiles within bounding box with 5000 m grids.

    lat, lon, _ = transformation_utm_to_ll.TransformPoint(x, y, 0)
    return lat, lon


def read_polygon_h5(file):

    with h5py.File(file, 'r') as file:
        # Load data
        x_coordinates = file['/science/LSAR/GCOV/grids/frequencyA/xCoordinates'][:]
        y_coordinates = file['/science/LSAR/GCOV/grids/frequencyA/yCoordinates'][:]
        image = file['/science/LSAR/GCOV/grids/frequencyA/HHHH'][:]
        epsg = file['/science/LSAR/GCOV/grids/frequencyA/projection'][()]
    # Determine valid pixels (this criteria might need adjustment)
    valid_mask = image > 0  # Assuming valid pixels are non-zero

    # Find the indices of valid pixels
    valid_indices = np.argwhere(valid_mask)

    # Extract the convex hull of the valid pixel indices
    if valid_indices.size > 0:
        hull = ConvexHull(valid_indices)
        coordinates = []
        hull_points = valid_indices[hull.vertices]

        for yind, xind in hull_points:

            lat_single, lon_single = utm_to_ll(x_coordinates[xind],
                                               y_coordinates[yind],
                                               epsg)
            coordinates.append((lon_single, lat_single))

        # Create a Polygon
        polygon = Polygon(coordinates)

    else:
        print("No valid pixels found to create a polygon.")
        polygon = None
    return polygon


def run(args):

    track_frame_db = args.track_frame
    h5file = args.input_file
    print(track_frame_db)
    print(h5file)
    with h5py.File(h5file) as src:
        des_asc = src['/science/LSAR/identification/orbitPassDirection'][()].decode()
    print(des_asc)
    gdf = gpd.read_file(track_frame_db)

    # Filter by 'passDirection'
    filtered_gdf = gdf[gdf['passDirection'] == f'{des_asc}ing']
    print(len(filtered_gdf))

    input_polygon = read_polygon_h5(h5file)
    # Calculate the intersection area with the input polygon and find the max
    filtered_gdf['intersection_area'] = filtered_gdf['geometry'].apply(lambda x: x.intersection(input_polygon).area)

    # Find the geometry with the largest overlap
    largest_overlap = filtered_gdf.loc[filtered_gdf['intersection_area'].idxmax()]
    print(largest_overlap.track)
    print(largest_overlap['track'])

    assign_frame_track(h5file, largest_overlap.track, largest_overlap.frame)
    read_frame_track(h5file)


def main():
    parser = _get_parser()

    args = parser.parse_args()

    run(args)

if __name__ == '__main__':

    main()
