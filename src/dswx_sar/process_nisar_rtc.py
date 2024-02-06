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

def read_meta_data(input_file):
    dswx_meta_mapping = {
        'RTC_ORBIT_PASS_DIRECTION': '/science/LSAR/identification/orbitPassDirection',
        'RTC_LOOK_DIRECTION': '/science/LSAR/identification/lookDirection',
        'RTC_INPUT_L1_SLC_GRANULES': '/science/LSAR/GCOV/metadata/processingInformation/inputs/l1SlcGranules',
        'RTC_PRODUCT_VERSION': '/science/LSAR/identification/productVersion',
        'RTC_SENSING_START_TIME': '/science/LSAR/identification/zeroDopplerStartTime',
        'RTC_SENSING_END_TIME': '/science/LSAR/identification/zeroDopplerEndTime',
        'RTC_FRAME_NUMBER': '/science/LSAR/identification/frameNumber',
        'RTC_TRACK_NUMBER': '/science/LSAR/identification/trackNumber',
        'RTC_ABSOLUTE_ORBIT_NUMBER': '/science/LSAR/identification/absoluteOrbitNumber',
        'RTC_APPLIED': '/science/LSAR/GCOV/metadata/processingInformation/parameters/radiometricTerrainCorrectionApplied',
        'RTC_QA_RFI_INFO_AVAILABLE': '/science/LSAR/GCOV/metadata/processingInformation/parameters/rfiCorrectionApplied',
    }

    with h5py.File(input_file, 'r') as src_h5:
        orbit_pass_dir = src_h5[dswx_meta_mapping['RTC_ORBIT_PASS_DIRECTION']][()].decode()
        look_dir = src_h5[dswx_meta_mapping['RTC_LOOK_DIRECTION']][()].decode()
        prod_ver = src_h5[dswx_meta_mapping['RTC_PRODUCT_VERSION']][()].decode()
        zero_dopp_start = src_h5[dswx_meta_mapping['RTC_SENSING_START_TIME']][()].decode()
        zero_dopp_end = src_h5[dswx_meta_mapping['RTC_SENSING_END_TIME']][()].decode()
        frame_number = src_h5[dswx_meta_mapping['RTC_FRAME_NUMBER']][()]
        track_number = src_h5[dswx_meta_mapping['RTC_TRACK_NUMBER']][()]
        abs_orbit_number = src_h5[dswx_meta_mapping['RTC_ABSOLUTE_ORBIT_NUMBER']][()]
        input_slc_granules = src_h5[dswx_meta_mapping['RTC_INPUT_L1_SLC_GRANULES']][(0)].decode()
        rtc_applied = src_h5[dswx_meta_mapping['RTC_APPLIED']][()].decode()
        rfi_applied = src_h5[dswx_meta_mapping['RTC_QA_RFI_INFO_AVAILABLE']][()].decode()

    dswx_metadata_dict = {
        'ORBIT_PASS_DIRECTION': orbit_pass_dir,
        'LOOK_DIRECTION': look_dir,
        'INPUT_L1_SLC_GRANULES': prod_ver,
        'PRODUCT_VERSION': prod_ver,
        'ZERO_DOPPLER_START_TIME': zero_dopp_start,
        'ZERO_DOPPLER_END_TIME': zero_dopp_end,
        'FRAME_NUMBER': track_number,
        'TRACK_NUMBER': track_number,
        'ABSOLUTE_ORBIT_NUMBER': abs_orbit_number,
        'RTC_APPLIED': rtc_applied,
        'QA_RFI_INFO_AVAILABLE': rfi_applied,
    }

    src_h5.close

    return dswx_metadata_dict

def read_write_gcov(
    input_file, 
    output_gtiff,
    group,
    ds_name, 
    row_blk_size, 
    col_blk_size,
    geodata,
    dswx_metadata_dict,
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
             
        dst.update_tags(**dswx_metadata_dict)

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

    src_h5.close

    return geo_dict


def process_nisar_gcov(
    input_file, 
    output_gtiff,
    group, 
    ds_name, 
    row_blk_size,
    col_blk_size,
    dswx_metadata_dict,
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
        geodata,
        dswx_metadata_dict,
    )


if __name__ == "__main__":
    # This is the workflow to generate Geotiff output using NISAR GCOV as input
    input_file = '/u/aurora-r0/bohuang/opera/DSWX-SAR/input/gcov.h5'
    group = '/science/LSAR/GCOV/grids/frequencyA/'
    row_blk_size = 1000
    col_blk_size = 1100

    # Read RTC input metadata
    dswx_metadata_dict = read_meta_data(input_file)

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
        dswx_metadata_dict,
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
        dswx_metadata_dict,
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
        dswx_metadata_dict,
    )
