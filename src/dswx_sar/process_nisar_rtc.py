import numpy as np
from osgeo import gdal
import h5py
import rasterio
from rasterio.transform import from_origin
from rasterio.transform import Affine
from rasterio.windows import Window
import os
from collections.abc import Iterator

def slice_gen(total_size: int, batch_size: int, combine_rem: bool=True) -> Iterator[slice]:
    """Generate slices with size defined by batch_size.

    Parameters
    ----------
    total_size: int
        size of data to be manipulated by slice_gen
    batch_size: int
        designated data chunk size in which data is sliced into.
    combine_rem: bool
        Combine the remaining values with the last complete block if 'True'.
        If False, ignore the remaining values
        Default = 'True'

    Yields
    ------
    slice: slice obj
        Iterable slices of data with specified input batch size, bounded by start_idx
        and stop_idx.
    """

    num_complete_blks = total_size // batch_size
    num_total_complete = num_complete_blks * batch_size
    num_rem = total_size - num_total_complete

    if combine_rem and num_rem > 0:
        for start_idx in range(0, num_total_complete - batch_size, batch_size):
            stop_idx = start_idx + batch_size
            yield slice(start_idx, stop_idx)

        last_blk_start = num_total_complete - batch_size
        last_blk_stop = total_size
        yield slice(last_blk_start, last_blk_stop)
    else:
        for start_idx in range(0, num_total_complete, batch_size):
            stop_idx = start_idx + batch_size
            yield slice(start_idx, stop_idx)


def read_write_gcov(
    input_file, 
    output_gtiff,
    group,
    ds_name, 
    row_blk_size, 
    col_blk_size,
    geodata
):
    """Read an level-2 RTC prodcut in HDF5 format and writ it out in 
    GeoTiff format in data blocks defined by row_blk_size and col_blk_size.

    Parameters
    ------------
    input_file: str
        input HDF5 RTC product file
    output_gtiff: str
        Output Geotiff file
    group: str
        HDF5 group identifier for RTC product
    ds_name: str
        HDF5 dataset name (identifier)
    row_blk_size: int
        The number of rows to read each time from the dataset.
    col_blk_size: int
        The number of columns to read each time from the dataset
    geodata: dictionary
        This dictionary contains Geo information fields in the input
        RTC HDF5 file.
    """

    # Check to see if the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The file '{input_file}' does not exist.")

    # Read Geo information
    geo_dict = read_rtc_geo(
        input_file, 
        group,
        geodata 
    )

    # Read Level-2 RTC product
    ds_path = f'HDF5:{input_file}:/{group}{ds_name}'
    h5_ds = gdal.Open(ds_path, gdal.GA_ReadOnly)

    num_cols = h5_ds.RasterXSize
    num_rows = h5_ds.RasterYSize
  
    # Geotransformation
    transform = from_origin(
        geo_dict['xmin'],
        geo_dict['ymin'],
        geo_dict['xres'],
        geo_dict['yres'],
    ) 
       
    # Geotransformation
    transform = Affine.translation(
        geo_dict['xmin'] - geo_dict['xres']/2, geo_dict['ymin'] - geo_dict['yres']/2) * \
        Affine.scale(geo_dict['xres'],geo_dict['yres'])

    # Coordinate Reference System
    crs = f'EPSG:{geo_dict['epsg']}'

    # Read input RTC product in blocks and write it out in geotiff format in blocks
    # using rasterio
    with rasterio.open(
       output_gtiff,
       'w',
       driver='GTiff',
       height=num_rows,
       width=num_cols,
       count=1,
       dtype='float32',
       crs=crs,
       transform=transform,
    ) as dst:
        for idx_y, slice_row in enumerate(slice_gen(num_rows, row_blk_size)):
            row_slice_size = slice_row.stop - slice_row.start
            for idx_x, slice_col in enumerate(slice_gen(num_cols, col_blk_size)):
                col_slice_size = slice_col.stop - slice_col.start

                ds_blk = h5_ds.ReadAsArray(
                    slice_col.start, 
                    slice_row.start, 
                    col_slice_size, 
                    row_slice_size
                )

                dst.write(
                    ds_blk,
                    1,
                    window=Window(
                        slice_col.start, 
                        slice_row.start,
                        col_slice_size, 
                        row_slice_size
                   )
                )
         
    del h5_ds

    # Close the data set
    raster_out = None


def read_rtc_geo(
    input_file,
    group,
    geodata,
):
    """extract data from RTC Geo information and store it as a dictionary

    paremeters:
    -----------
    input_file: str
        input HDF5 file
    group: str
        HDF5 identifier for RTC product
    geodata: dictionary
        This dictionary contains Geo information fields in the input
        RTC HDF5 file.

    Returns
    -------
    geo_dict: dictionary
        This is the dictionary which stores Geo information extracted
        from RTC HDF5 product.
    """

    with h5py.File(input_file, 'r') as src_h5:
        xmin = src_h5[f'{group}/{geodata['xcoord']}'][:][0]
        ymin = src_h5[f'{group}/{geodata['ycoord']}'][:][0]
        xres = np.array(src_h5[f'{group}/{geodata['xposting']}'])
        yres = np.array(src_h5[f'{group}/{geodata['yposting']}'])
        epsg = np.array(src_h5[f'{group}{geodata['proj']}'])
 
    epsg = epsg.item()
    
    geo_dict = {
        'xmin': xmin,
        'ymin': ymin,
        'xres': xres,
        'yres': yres,
        'epsg': epsg
    }

    return geo_dict


def process_nisar_gcov(
    input_file, 
    output_gtiff,
    group, 
    ds_name, 
    row_blk_size,
    col_blk_size,
    ):

    """This wrapper reads the input HDF5 RTC product and writes it out
    in Geotiff format.

    paremeters:
    -----------
    input_file: str
        input HDF5 RTC product file
    output_gtiff: str
        Output Geotiff file
    group: str
        HDF5 group identifier for RTC product
    ds_name: str
        HDF5 dataset name (identifier)
    row_blk_size: int
        The number of rows to read each time from the dataset.
    col_blk_size: int
        The number of columns to read each time from the dataset

    Returns
    -------
    geo_dict: dictionary
        This is the dictionary which stores Geo information extracted
        from RTC HDF5 product.
    """


    # Geo information
    geodata = {
        'xcoord': 'xCoordinates',
        'ycoord': 'yCoordinates',
        'xposting': 'xCoordinateSpacing',
        'yposting': 'yCoordinateSpacing',
        'proj': 'projection'
    }

    # Read Level-2 GCOV
    ds_array = read_write_gcov(
        input_file, 
        output_gtiff,
        group,
        ds_name, 
        row_blk_size, 
        col_blk_size,
        geodata
    )


if __name__ == "__main__":
    # This is the workflow to generate Geotiff output using NISAR GCOV as input
    input_file = '/u/aurora-r0/bohuang/opera/DSWX-SAR/input/gcov.h5'
    group = '/science/LSAR/GCOV/grids/frequencyA/'
    row_blk_size = 1000
    col_blk_size = 1100


    # Process HHHH data
    ds_name = 'HHHH'
    output_gtiff = f'/u/aurora-r0/bohuang/opera/DSWX-SAR/output/gcov_{ds_name}.tif'
    process_nisar_gcov(
        input_file, 
        output_gtiff,
        group,
        ds_name, 
        row_blk_size, 
        col_blk_size,
    )

    # Process HVHV data
    ds_name = 'HVHV'
    output_gtiff = f'/u/aurora-r0/bohuang/opera/DSWX-SAR/output/gcov_{ds_name}.tif'
    process_nisar_gcov(
        input_file, 
        output_gtiff,
        group,
        ds_name, 
        row_blk_size, 
        col_blk_size,
    )

    ds_name = 'layoverShadowMask'
    output_gtiff = f'/u/aurora-r0/bohuang/opera/DSWX-SAR/output/gcov_{ds_name}.tif'
    process_nisar_gcov(
        input_file, 
        output_gtiff,
        group,
        ds_name, 
        row_blk_size, 
        col_blk_size,
    )

