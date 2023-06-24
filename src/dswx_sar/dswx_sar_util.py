from osgeo import gdal
from osgeo import osr
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import tempfile
import logging
from dataclasses import dataclass

np2gdal_conversion = {
  "uint8": 1,
  "int8": 1,
  "uint16": 2,
  "int16": 3,
  "uint32": 4,
  "int32": 5,
  "float32": 6,
  "float64": 7,
  "complex64": 10,
  "complex128": 11,
}

band_assign_value_dict = {
    'no_water': 0 ,
    'water': 1, # water body
    'bright_water_fill': 2,
    'dark_land_mask': 3,
    'landcover_mask': 4,
    'hand_mask': 5,
    'layover_shadow_mask': 6,
    'inundated_vegetation': 7,
    'no_data': 255
}

def get_interpreted_dswx_s1_ctable():
    """Get colortable for DSWx-S1 products
    Returns
    -------
    dswx_ctable: gdal.ColorTable
        colortable for dswx-s1 product
    """
    # create color table
    dswx_ctable = gdal.ColorTable()

    # set color for each value
    dswx_ctable.SetColorEntry(0, (255, 255, 255))  # White - Not water
    dswx_ctable.SetColorEntry(1, (0, 0, 255))  # Blue - Water (high confidence)
    dswx_ctable.SetColorEntry(2, (120, 120,  240 ))  # baby blue - bright water
    dswx_ctable.SetColorEntry(3, (240, 20,  20 ))  # Red - dark land
    dswx_ctable.SetColorEntry(4, (128, 255, 128))  # Light green - Landcover mask
    dswx_ctable.SetColorEntry(5, (200, 200, 200))  # light gray - Hand mask
    dswx_ctable.SetColorEntry(6, (128, 128, 128))  # Gray - Layover/shadow mask
    dswx_ctable.SetColorEntry(7, (200, 200, 50))  # Gray - Inundated vegetation

    dswx_ctable.SetColorEntry(255, (0, 0, 0, 255))  # Black - Not observed (out of Boundary)

    return dswx_ctable

def read_geotiff(input_tif_str, band_ind=None):
    """Read band from geotiff
    Parameters
    ----------
    input_tif_str: str
        geotiff file path to read the band
    Returns
    -------
    tifdata: numpy.ndarray
        image from geotiff
    """
    tif = gdal.Open(input_tif_str)
    if band_ind is None:
        tifdata = tif.ReadAsArray()

    else:
        tifdata = tif.GetRasterBand(band_ind+1).ReadAsArray()

    tif.FlushCache()
    tif = None
    del tif
    print(f"Reading {input_tif_str} ... {tifdata.shape}")
    return tifdata

def save_raster_gdal(data, output_file, geotransform,
                     projection, scratch_dir='.',
                     DataType='float32'):
    """Save images using Gdal
    Parameters
    ----------
    data: numpy.ndarray
        Data to save into the file
    output_file: str
        full path for filename to save the DSWx-S1 file
    geotransform: gdal
        gdaltransform information
    projection: gdal
        projection object
    scratch_dir: str
        temporary file path to process COG file.
    DataType: str
        Data types to save the file.
    """
    Gdal_type = np2gdal_conversion[str(DataType)]
    image_size = data.shape
    #  Set the Pixel Data (Create some boxes)
    # set geotransform
    if len(image_size) == 3:
        nim = image_size[0]
        ny = image_size[1]
        nx = image_size[2]
    elif len(image_size) == 2:
        ny = image_size[0]
        nx = image_size[1]
        nim = 1

    driver = gdal.GetDriverByName("GTiff")
    output_file_path = os.path.join(output_file)
    gdal_ds = driver.Create(output_file_path, nx, ny, nim, Gdal_type)
    gdal_ds.SetGeoTransform(geotransform)
    gdal_ds.SetProjection(projection)

    if nim == 1:
        gdal_ds.GetRasterBand(1).WriteArray(data)
    else:
        for im_ind in range(0, nim):
            gdal_ds.GetRasterBand(im_ind+1).WriteArray(
                np.squeeze(data[im_ind, :, :]))

    gdal_ds.FlushCache()
    gdal_ds = None
    del gdal_ds  # close the dataset (Python object and pointers)

    _save_as_cog(output_file, scratch_dir)

def save_dswx_product(wtr, output_file, geotransform,
                      projection, scratch_dir='.',
                      description = None, **dswx_processed_bands):
    """Save DSWx product for assigned classes with colortable
    Parameters
    ----------
    wtr: numpy.ndarray
        classified image for DSWx-S1 product
    output_file: str
        full path for filename to save the DSWx-S1 file
    geotransform: gdal
        gdaltransform information
    projection: gdal
        projection object
    scratch_dir: str
        temporary file path to process COG file.
    description: str
        description for DSWx-S1
    dswx_processed_bands
        classes to save to output
    """
    shape = wtr.shape
    driver = gdal.GetDriverByName("GTiff")
    wtr = np.asarray(wtr, dtype=np.byte)
    dswx_processed_bands_keys = dswx_processed_bands.keys()

    for band_key in band_assign_value_dict.keys():
        if band_key.lower() in dswx_processed_bands_keys:
            dswx_product_value = band_assign_value_dict[band_key]
            wtr[dswx_processed_bands[band_key.lower()]==1] = dswx_product_value
            print(band_key.lower(), 'found',dswx_product_value)

    gdal_ds = driver.Create(output_file, shape[1], shape[0], 1, gdal.GDT_Byte)
    gdal_ds.SetGeoTransform(geotransform)
    gdal_ds.SetProjection(projection)

    gdal_band = gdal_ds.GetRasterBand(1)
    gdal_band.WriteArray(wtr)
    gdal_band.SetNoDataValue(255)

    # set color table and color interpretation
    dswx_ctable = get_interpreted_dswx_s1_ctable()
    gdal_band.SetRasterColorTable(dswx_ctable)
    gdal_band.SetRasterColorInterpretation(
        gdal.GCI_PaletteIndex)

    if description is not None:
        gdal_band.SetDescription(description)
    else:
        gdal_band.SetDescription(description_from_dict)

    gdal_band.FlushCache()
    gdal_band = None

    gdal_ds.FlushCache()
    gdal_ds = None
    del gdal_ds  # close the dataset (Python object and pointers)

    _save_as_cog(output_file, scratch_dir)

def _save_as_cog(filename, scratch_dir = '.', logger = None,
                flag_compress=True, ovr_resamp_algorithm=None,
                compression='DEFLATE', nbits=None):
    """Save (overwrite) a GeoTIFF file as a cloud-optimized GeoTIFF.
       Parameters
       ----------
       filename: str
              GeoTIFF to be saved as a cloud-optimized GeoTIFF
       scratch_dir: str (optional)
              Temporary Directory
       ovr_resamp_algorithm: str (optional)
              Resampling algorithm for overviews.
              Options: "AVERAGE", "AVERAGE_MAGPHASE", "RMS", "BILINEAR",
              "CUBIC", "CUBICSPLINE", "GAUSS", "LANCZOS", "MODE",
              "NEAREST", or "NONE". Defaults to "NEAREST", if integer, and
              "CUBICSPLINE", otherwise.
        compression: str (optional)
              Compression type.
              Optional: "NONE", "LZW", "JPEG", "DEFLATE", "ZSTD", "WEBP",
              "LERC", "LERC_DEFLATE", "LERC_ZSTD", "LZMA"
    """
    if logger is None:
        logger = logging.getLogger('proteus')

    logger.info('        COG step 1: add overviews')
    gdal_ds = gdal.Open(filename, gdal.GA_Update)
    gdal_dtype = gdal_ds.GetRasterBand(1).DataType
    dtype_name = gdal.GetDataTypeName(gdal_dtype).lower()

    overviews_list = [4, 16, 64, 128]

    is_integer = 'byte' in dtype_name  or 'int' in dtype_name
    if ovr_resamp_algorithm is None and is_integer:
        ovr_resamp_algorithm = 'NEAREST'
    elif ovr_resamp_algorithm is None:
        ovr_resamp_algorithm = 'CUBICSPLINE'

    gdal_ds.BuildOverviews(ovr_resamp_algorithm, overviews_list,
                           gdal.TermProgress_nocb)

    del gdal_ds  # close the dataset (Python object and pointers)
    external_overview_file = filename + '.ovr'
    if os.path.isfile(external_overview_file):
        os.remove(external_overview_file)

    logger.info('        COG step 2: save as COG')
    temp_file = tempfile.NamedTemporaryFile(
                    dir=scratch_dir, suffix='.tif').name

    # Blocks of 512 x 512 => 256 KiB (UInt8) or 1MiB (Float32)
    tile_size = 512
    gdal_translate_options = ['BIGTIFF=IF_SAFER',
                              'MAX_Z_ERROR=0',
                              'TILED=YES',
                              f'BLOCKXSIZE={tile_size}',
                              f'BLOCKYSIZE={tile_size}',
                              'COPY_SRC_OVERVIEWS=YES']

    if compression:
        gdal_translate_options += [f'COMPRESS={compression}']

    if is_integer:
        gdal_translate_options += ['PREDICTOR=2']
    else:
        gdal_translate_options += ['PREDICTOR=3']

    if nbits is not None:
        gdal_translate_options += [f'NBITS={nbits}']

        # suppress type casting errors
        gdal.SetConfigOption('CPL_LOG', '/dev/null')

    gdal.Translate(temp_file, filename,
                   creationOptions=gdal_translate_options)

    shutil.move(temp_file, filename)

    logger.info('        COG step 3: validate')
    try:
        from rtc.extern.validate_cloud_optimized_geotiff import main as validate_cog
    except ModuleNotFoundError:
        logger.info('WARNING could not import module validate_cloud_optimized_geotiff')
        return

    argv = ['--full-check=yes', filename]
    validate_cog_ret = validate_cog(argv)
    if validate_cog_ret == 0:
        logger.info(f'        file "{filename}" is a valid cloud optimized'
                    ' GeoTIFF')
    else:
        logger.warning(f'        file "{filename}" is NOT a valid cloud'
                       f' optimized GeoTIFF!')


def change_epsg_tif(input_tif, output_tif, epsg_output):
    """Resample the input geotiff image to new EPSG code.
    Parameters
    ----------
    input_tif: str
        geotiff file path to be changed
    output_tif: str
        geotiff file path to be saved
    epsg_output: int
        new EPSG code
    """
    metadata = get_meta_from_tif(input_tif)
    opt = gdal.WarpOptions(dstSRS=f'EPSG:{epsg_output}',
                     resampleAlg='nearest',
                     dstNodata='Nan',
                     xRes=metadata['geotransform'][1],
                     yRes=metadata['geotransform'][5],
                     format='GTIFF')
    ds = gdal.Warp(output_tif, input_tif, options=opt)
    del ds

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
    if type(tif_file_name) is list:
        tif_name = tif_file_name[0]
    else:
        tif_name = tif_file_name
    tif_gdal = gdal.Open(tif_name)
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

def get_raster_block(raster, block_param):
    ''' Get a block of data from raster.
        Raster can be a HDF5 file or a GDAL-friendly raster

    Parameters
    ----------
    raster: str
        Raster where a block is to be read from. String value represents a
        filepath for GDAL rasters.
    block_param: BlockParam
        Object specifying size of block and where to read from raster,
        and amount of padding for the read array

    Returns
    -------
    data_block: np.ndarray
        Block read from raster with shape specified in block_param.
    '''

    # Open input data using GDAL to get raster length
    ds_data = gdal.Open(raster, gdal.GA_Update)
    data_block = ds_data.GetRasterBand(1).ReadAsArray(0,
                                            block_param.read_start_line,
                                            block_param.data_width,
                                            block_param.read_length)

    # Pad igram_block with zeros according to pad_length/pad_width
    data_block = np.pad(data_block, block_param.block_pad,
                         mode='constant', constant_values=0)

    return data_block

def write_raster_block(out_raster, data, 
                        block_param, geotransform, projection,
                        DataType='byte'):
    ''' Write processed block to out_raster.

    Parameters
    ----------
    out_raster: h5py.Dataset or str
        Raster where data (i.e., filtered data) needs to be written.
        String value represents filepath for GDAL rasters.
    data: np.ndarray
        Filtered data to write to out_raster.
    block_param: BlockParam
        Object specifying where and how much to write to out_raster.
    '''

    if DataType == 'float32':
        Gdal_type = gdal.GDT_Float32
    elif DataType == 'uint16':
        Gdal_type = gdal.GDT_UInt16
    elif DataType == 'byte':
        Gdal_type = gdal.GDT_Byte
    elif DataType == 'int16':
        Gdal_type = gdal.GDT_Int16
    elif DataType == 'int32':
        Gdal_type = gdal.GDT_Int32

    if block_param.write_start_line == 0:
        driver = gdal.GetDriverByName('GTiff')
        ds_data = driver.Create(out_raster, 
                                block_param.data_width,
                                block_param.data_length,
                                1, Gdal_type)
        ds_data.SetGeoTransform(geotransform)
        ds_data.SetProjection(projection)
        ds_data.WriteArray(data, xoff=0, yoff=0)
    else:
        ds_data = gdal.Open(out_raster, gdal.GA_Update)
        ds_data.GetRasterBand(1).WriteArray(
                data, 
                xoff=0, 
                yoff=block_param.write_start_line)

def block_param_generator(lines_per_block, data_shape, pad_shape):
    ''' Generator for block specific parameter class.

    Parameters
    ----------
    lines_per_block: int
        Lines to be processed per block (in batch).
    data_shape: tuple(int, int)
        Length and width of input raster.
    pad_shape: tuple(int, int)
        Padding for the length and width of block to be filtered.

    Returns
    -------
    _: BlockParam
        BlockParam object for current block
    '''
    data_length, data_width = data_shape
    pad_length, pad_width = pad_shape

    # Calculate number of blocks to break raster into
    num_blocks = int(np.ceil(data_length / lines_per_block))

    for block in range(num_blocks):
        start_line = block * lines_per_block

        # Discriminate between first, last, and middle blocks
        first_block = block == 0
        last_block = block == num_blocks - 1 or num_blocks == 1
        middle_block = not first_block and not last_block

        # Determine block size; Last block uses leftover lines
        block_length = data_length - start_line if last_block else lines_per_block

        # Determine padding along length. Full padding for middle blocks
        # Half padding for start and end blocks
        read_length_pad = pad_length if middle_block else pad_length // 2

        # Determine 1st line of output
        write_start_line = block * lines_per_block

        # Determine 1st dataset line to read. Subtract half padding length
        # to account for additional lines to be read.
        read_start_line = block * lines_per_block - pad_length // 2

        # If applicable, save negative start line as deficit to account for later
        read_start_line, start_line_deficit = (
            0, read_start_line) if read_start_line < 0 else (
            read_start_line, 0)

        # Initial guess at number lines to read; accounting for negative start at the end
        read_length = block_length + read_length_pad
        if not first_block:
            read_length -= abs(start_line_deficit)

        # Check for over-reading and adjust lines read as needed
        end_line_deficit = min(
            data_length - read_start_line - read_length, 0)
        read_length -= abs(end_line_deficit)

        # Determine block padding in length
        if first_block:
            # Only the top part of the block should be padded. If end_deficit_line=0
            # we have a sufficient number of lines to be read in the subsequent block
            top_pad = pad_length // 2
            bottom_pad = abs(end_line_deficit)
        elif last_block:
            # Only the bottom part of the block should be padded
            top_pad = abs(
                start_line_deficit) if start_line_deficit < 0 else 0
            bottom_pad = pad_length // 2
        else:
            # Top and bottom should be added taking into account line deficit
            top_pad = abs(
                start_line_deficit) if start_line_deficit < 0 else 0
            bottom_pad = abs(end_line_deficit)

        block_pad = ((top_pad, bottom_pad),
                     (pad_width // 2, pad_width // 2))

        yield BlockParam(block_length, write_start_line, read_start_line, read_length, block_pad, data_width, data_length)

    return


@dataclass
class BlockParam:
    '''
    Class for block specific parameters
    Facilitate block parameters exchange between functions
    '''
    # Length of current block to filter; padding not included
    block_length: int

    # First line to write to for current block
    write_start_line: int

    # First line to read from dataset for current block
    read_start_line: int

    # Number of lines to read from dataset for current block
    read_length: int

    # Padding to be applied to read in current block. First tuple is padding to
    # be applied to top/bottom (along length). Second tuple is padding to be
    # applied to left/right (along width). Values in second tuple do not change;
    # included in class so one less value is passed between functions.
    block_pad: tuple

    # Width of current block. Value does not change per block; included to
    # in class so one less value is to be passed between functions.
    data_width: int

    data_length: int

def intensity_display(intensity, outputdir, pol, immin=-30, immax=0):
    """save intensity images into png file

    Parameters
    ----------
    intensity: numpy.ndarray
        2 dimensional array containing linear intensity
    outputdir: str 
        path for output directory 
    pol: str
        specific polarization added to the file name
    immin: float
        mininum dB value for displaying intensity
    immax: float
        maximum dB value for displaying intensity        
    """
    fig = plt.figure(figsize=(20, 20))
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    im = ax.imshow(10*np.log10(intensity), cmap = plt.get_cmap('gray'),
                   vmin=immin,vmax=immax)
    plt.title('RTC')
    plt.savefig(os.path.join(outputdir, 'RTC_intensity_{}'.format(pol)))
