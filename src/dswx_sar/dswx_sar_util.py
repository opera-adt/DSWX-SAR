import logging
import os
import shutil
import tempfile
import rasterio

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, osr

gdal.DontUseExceptions()


np2gdal_conversion = {
  "byte": 1,
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
    'no_water': 0,
    'water': 1,  # water body
    'bright_water_fill': 2,
    'dark_land_mask': 3,
    'landcover_mask': 4,
    'hand_mask': 5,
    'layover_shadow_mask': 6,
    'inundated_vegetation': 7,
    'ocean_mask': 254,
    'no_data': 120,
}

band_assign_value_conf_dict = {
    'hand_mask': 252,
    'layover_shadow_mask': 253,
    'ocean_mask': 254,
    'no_data': 120,
}

'''
Internally, DSWx-S1 has 2 water classes; 
    1. low-backscattering water
    2. high-backscattering water
There classes are collapesed into water class when 
WTR layers converts to BWTR. 

Low-backscattering land (dark land) is captured from  
    1. masking_with_ancillary
    2. refine_with_bimodality
These classes are collapsed into no-water class 
'''

collapse_wtr_classes_dict = {
    band_assign_value_dict['no_water'] : band_assign_value_dict['no_water'],
    band_assign_value_dict['water']: band_assign_value_dict['water'],
    band_assign_value_dict['bright_water_fill']: band_assign_value_dict['water'],
    band_assign_value_dict['dark_land_mask']: \
        band_assign_value_dict['no_water'],
    band_assign_value_dict['landcover_mask']: \
        band_assign_value_dict['no_water'],
    band_assign_value_dict['hand_mask']: \
        band_assign_value_dict['hand_mask'],
    band_assign_value_dict['layover_shadow_mask']: \
        band_assign_value_dict['layover_shadow_mask'],    
    band_assign_value_dict['inundated_vegetation']: band_assign_value_dict['inundated_vegetation'],
    band_assign_value_dict['no_data']: band_assign_value_dict['no_data'],
    band_assign_value_dict['ocean_mask']: band_assign_value_dict['ocean_mask'],
    }

logger = logging.getLogger('dswx-s1')

@dataclass
class Constants:
    # negligible number to avoid the zero-division warning.
    negligible_value : float = 1e-5


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
    dswx_ctable.SetColorEntry(3, (240, 20,  20 ))  # Red - dark land (bimodality test)
    dswx_ctable.SetColorEntry(4, (200, 200, 50))   # Yellow - dark land (Landcover mask)
    dswx_ctable.SetColorEntry(5, (200, 200, 200))  # light gray - Hand mask
    dswx_ctable.SetColorEntry(6, (128, 128, 128))  # Gray - Layover/shadow mask
    dswx_ctable.SetColorEntry(7, (0, 255, 0))  # Green - Inundated vegetation

    dswx_ctable.SetColorEntry(120, (0, 0, 0, 255))  # Black - Not observed (out of Boundary)

    return dswx_ctable


def read_geotiff(input_tif_str, band_ind=None, verbose=True):
    """Read band from geotiff

    Parameters
    ----------
    input_tif_str: str
        geotiff file path to read the band
    band_ind: int
        Index of the band to read, starts from 0

    Returns
    -------
    tifdata: numpy.ndarray
        image from geotiff
    """
    tif = gdal.Open(input_tif_str)
    if band_ind is None:
        tifdata = tif.ReadAsArray()
    else:
        tifdata = tif.GetRasterBand(band_ind + 1).ReadAsArray()

    tif.FlushCache()
    tif = None
    del tif
    if verbose:
        print(f" -- Reading {input_tif_str} ... {tifdata.shape}")
    return tifdata


def save_raster_gdal(data, output_file, geotransform,
                     projection, scratch_dir='.',
                     datatype='float32'):
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
    datatype: str
        Data types to save the file.
    """
    gdal_type = np2gdal_conversion[str(datatype)]
    image_size = data.shape
    #  Set the Pixel Data (Create some boxes)
    # set geotransform
    if data.ndim == 3:
        ndim = image_size[0]
        ny = image_size[1]
        nx = image_size[2]
    elif data.ndim == 2:
        ny = image_size[0]
        nx = image_size[1]
        ndim = 1

    driver = gdal.GetDriverByName("GTiff")
    output_file_path = os.path.join(output_file)
    gdal_ds = driver.Create(output_file_path,
                            nx, ny,
                            ndim, gdal_type)
    gdal_ds.SetGeoTransform(geotransform)
    gdal_ds.SetProjection(projection)

    if data.ndim == 2:
        gdal_ds.GetRasterBand(1).WriteArray(data)
    else:
        for im_ind in range(0, ndim):
            gdal_ds.GetRasterBand(im_ind+1).WriteArray(
                np.squeeze(data[im_ind, :, :]))

    gdal_ds.FlushCache()
    gdal_ds = None
    del gdal_ds  # close the dataset (Python object and pointers)

    _save_as_cog(output_file, scratch_dir)


def save_dswx_product(wtr, output_file, geotransform,
                      projection, scratch_dir='.',
                      description=None, metadata=None,
                      is_conf=False, datatype='uint8',
                      **dswx_processed_bands):
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
    wtr = np.asarray(wtr, dtype=datatype)
    dswx_processed_bands_keys = dswx_processed_bands.keys()
    print(f'Saving dswx product : {output_file} ')

    if is_conf:
        band_value_dict = band_assign_value_conf_dict
    else:
        band_value_dict = band_assign_value_dict

    for band_key in band_value_dict.keys():
        if band_key.lower() in dswx_processed_bands_keys:
            dswx_product_value = band_value_dict[band_key]
            wtr[dswx_processed_bands[band_key.lower()]==1] = dswx_product_value
            print(f'    {band_key.lower()} found {dswx_product_value}')
    gdal_type = np2gdal_conversion[str(datatype)]

    gdal_ds = driver.Create(output_file,
                            shape[1], shape[0], 1, gdal_type)
    gdal_ds.SetGeoTransform(geotransform)
    gdal_ds.SetProjection(projection)

    gdal_band = gdal_ds.GetRasterBand(1)
    gdal_band.WriteArray(wtr)
    gdal_band.SetNoDataValue(band_value_dict['no_data'])
    gdal_band.SetMetadata(metadata)
    # set color table and color interpretation
    if not is_conf:
        dswx_ctable = get_interpreted_dswx_s1_ctable()
        gdal_band.SetRasterColorTable(dswx_ctable)
        gdal_band.SetRasterColorInterpretation(
            gdal.GCI_PaletteIndex)

    if description is not None:
        gdal_band.SetDescription(description)

    gdal_band.FlushCache()
    gdal_band = None

    gdal_ds.FlushCache()
    gdal_ds = None
    del gdal_ds  # close the dataset (Python object and pointers)

    _save_as_cog(output_file, scratch_dir)


def _save_as_cog(filename,
                 scratch_dir='.',
                 logger=None,
                 flag_compress=True,
                 ovr_resamp_algorithm=None,
                 compression='DEFLATE',
                 nbits=16):
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

    is_integer = 'byte' in dtype_name or 'int' in dtype_name
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


def change_epsg_tif(input_tif, output_tif, epsg_output,
                    resample_method='nearest',
                    output_nodata='NaN'):
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
    opt = gdal.WarpOptions(
        dstSRS=f'EPSG:{epsg_output}',
        resampleAlg=resample_method,
        dstNodata=output_nodata,
        xRes=metadata['geotransform'][1],
        yRes=metadata['geotransform'][5],
        format='GTIFF')

    gdal.Warp(output_tif, input_tif, options=opt)


def get_invalid_area(geotiff_path,
                     output_path=None,
                     scratch_dir=None,
                     lines_per_block=None):
    """get invalid areas (NaN) from GeoTiff and save it
    to new GeoTiff

    Parameters
    ----------
    geotiff_path: str
        full path for filename to get invalid area
    output_file: str
        full path for filename to save invalid area
    scratch_dir: str
        temporary file path to process COG file.
    """
    im_meta = get_meta_from_tif(geotiff_path)

    pad_shape = (0, 0)
    block_params = block_param_generator(
        lines_per_block=lines_per_block,
        data_shape=(im_meta['length'],
                    im_meta['width']),
        pad_shape=pad_shape)

    for block_param in block_params:
        image = get_raster_block(
            geotiff_path,
            block_param)

        if image.ndim == 3:
            no_data_raster = np.isnan(
                np.squeeze(image[0, :, :]))
        else:
            no_data_raster = np.isnan(image)

        write_raster_block(
            out_raster=output_path,
            data=no_data_raster,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='byte')

    if scratch_dir is not None:
        _save_as_cog(output_path, scratch_dir)


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
    meta_dict = {}
    meta_dict['band_number'] = tif_gdal.RasterCount
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


def create_geotiff_with_one_value(outpath, shape, filled_value):
    """
    Create a new GeoTIFF file filled with a specified value.

    Parameters
    ----------
    outpath: str
        The file path where the new GeoTIFF will be saved.
    shape: tuple
        A tuple (height, width) representing the dimensions of the GeoTIFF.
    filled_value: float
        The value with which the GeoTIFF will be filled.
    """
    # Set up the new file's spatial properties
    height, width= shape

    # Create the file with a single band, Float32 type
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(outpath, width, height, 1, gdal.GDT_Float32)

    # Write zeros to the raster band
    band = ds.GetRasterBand(1)
    band.WriteArray(np.full((height, width), filled_value, dtype=np.float32))
    band.FlushCache()

    ds = None  # Close the file


def get_raster_block(raster_path, block_param):
    ''' Get a block of data from raster.
        Raster can be a HDF5 file or a GDAL-friendly raster

    Parameters
    ----------
    raster_path: str
        raster path where a block is to be read from. String value represents a
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
    ds_data = gdal.Open(raster_path, gdal.GA_Update)

    # Number of bands in the raster
    num_bands = ds_data.RasterCount
    # List to store blocks from each band
    data_blocks = []
    for i in range(num_bands):
        band = ds_data.GetRasterBand(i+1)
        data_block = band.ReadAsArray(
            0,
            block_param.read_start_line,
            block_param.data_width,
            block_param.read_length)

        # Pad data_block with zeros according to pad_length/pad_width
        data_block = np.pad(data_block, block_param.block_pad,
                            mode='constant', constant_values=0)

        if data_block.ndim == 1:
            data_block = data_block[np.newaxis, :]
        data_blocks.append(data_block)
    data_blocks = np.array(data_blocks)

    if num_bands == 1:
        data_blocks = np.reshape(data_blocks,
                                 [data_blocks.shape[1],
                                  data_blocks.shape[2]])
    return data_blocks


def write_raster_block(out_raster, data,
                       block_param, geotransform, projection,
                       datatype='byte',
                       cog_flag=False, 
                       scratch_dir='.'):
    """
    Write processed data block to the specified raster file.

    Parameters
    ----------
    out_raster : h5py.Dataset or str
        Raster where data needs to be written. String value represents
        filepath for GDAL rasters.
    data : np.ndarray
        Data to be written to the raster.
    block_param : BlockParam
        Specifications for the data block to be written.
    geotransform : tuple
        GeoTransform parameters for the raster.
    projection : str
        Projection string for the raster.
    datatype : str, optional
        Data type of the raster. Defaults to 'byte'.
    cog_flag : bool, optional
        If True, converts the raster to COG format. Defaults to False.
    scratch_dir : str, optional
        Directory for intermediate processing. Defaults to '.'.
    """
    gdal_type = np2gdal_conversion[datatype]

    data = np.array(data, dtype=datatype)
    ndim = data.ndim
    number_band = 1 if ndim < 3 else data.shape[0]

    data_start_without_pad = block_param.write_start_line - \
        block_param.read_start_line
    data_end_without_pad = data_start_without_pad + \
        block_param.block_length

    if block_param.write_start_line == 0:
        driver = gdal.GetDriverByName('GTiff')
        ds_data = driver.Create(out_raster,
                                block_param.data_width,
                                block_param.data_length,
                                number_band, gdal_type)
        if not ds_data:
            raise IOError(f"Failed to create raster: {out_raster}")

        ds_data.SetGeoTransform(geotransform)
        ds_data.SetProjection(projection)
    else:
        ds_data = gdal.Open(out_raster, gdal.GA_Update)
        if not ds_data:
            raise IOError(f"Failed to open raster for update: {out_raster}")

    if ndim == 2:
        ds_data.GetRasterBand(1).WriteArray(
                data[data_start_without_pad:data_end_without_pad,
                    :],
                xoff=0,
                yoff=block_param.write_start_line)
    else:
        for im_ind in range(0, number_band):
            ds_data.GetRasterBand(im_ind+1).WriteArray(
                np.squeeze(data[im_ind,
                                data_start_without_pad:data_end_without_pad,
                                :]),
                xoff=0,
                yoff=block_param.write_start_line)

    del ds_data

    # Write COG is cog_flag is True and last block.
    if (block_param.write_start_line + block_param.block_length == \
        block_param.data_length) and cog_flag:
        _save_as_cog(out_raster, scratch_dir)


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
    half_pad_length = pad_length // 2
    half_pad_width = pad_width // 2

    # Calculate number of blocks to break raster into
    num_blocks = int(np.ceil(data_length / lines_per_block))

    for block in range(num_blocks):
        start_line = block * lines_per_block

        # Discriminate between first, last, and middle blocks
        first_block = block == 0
        last_block = block == num_blocks - 1 or num_blocks == 1
        middle_block = not first_block and not last_block

        # Determine block size; Last block uses leftover lines
        block_length = data_length - start_line \
            if last_block else lines_per_block
        # Determine padding along length. Full padding for middle blocks
        # Half padding for start and end blocks
        read_length_pad = pad_length if middle_block else half_pad_length

        # Determine 1st line of output
        write_start_line = block * lines_per_block

        # Determine 1st dataset line to read. Subtract half padding length
        # to account for additional lines to be read.
        read_start_line = block * lines_per_block - half_pad_length

        # If applicable, save negative start line as deficit
        # to account for later
        read_start_line, start_line_deficit = (
            0, read_start_line) if read_start_line < 0 else (
            read_start_line, 0)

        # Initial guess at number lines to read; accounting
        # for negative start at the end
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
            top_pad = half_pad_length
            bottom_pad = abs(end_line_deficit)
        elif last_block:
            # Only the bottom part of the block should be padded
            top_pad = abs(
                start_line_deficit) if start_line_deficit < 0 else 0
            bottom_pad = half_pad_length
        else:
            # Top and bottom should be added taking into account line deficit
            top_pad = abs(
                start_line_deficit) if start_line_deficit < 0 else 0
            bottom_pad = abs(end_line_deficit)

        block_pad = ((top_pad, bottom_pad),
                     (half_pad_width, half_pad_width))

        yield BlockParam(block_length,
                         write_start_line,
                         read_start_line,
                         read_length,
                         block_pad,
                         data_width,
                         data_length)

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
    # applied to left/right (along width). Values in second tuple do not change
    # included in class so one less value is passed between functions.
    block_pad: tuple

    # Width of current block. Value does not change per block; included to
    # in class so one less value is to be passed between functions.
    data_width: int

    data_length: int


def merge_binary_layers(layer_list, value_list, merged_layer_path, 
                        lines_per_block, mode='or', cog_flag=True, 
                        scratch_dir='.'):
    """
    Merges multiple raster layers into a single binary layer based on specified
    values and a logical operation ('and' or 'or').

    Parameters
    ----------
    layer_list : list of str
        List of paths to the raster files (layers) to be merged.
    value_list : list
        List of values corresponding to each raster file. A pixel in the
        output binary layer is set if it matches the value in the respective
        input layer.
    merged_layer_path : str
        Path to save the merged binary layer.
    lines_per_block : int
        Number of lines per block for processing the data in chunks.
    mode : str, optional
        Logical operation to apply for merging ('and' or 'or'). The default is 'or'.
    cog_flag : bool, optional
        Write to COG if True. Defaults to True.
    scratch_dir : str, optional
        Path to scrath dir. Defaults to '.'.

    Returns
    -------
    None
        The function saves the merged binary layer at `merged_layer_path`.
    """

    if len(layer_list) != len(value_list):
        raise ValueError('Number of layers does not match with number of values')

    # Getting metadata from the reference layer
    meta_info = get_meta_from_tif(layer_list[0])
    data_shape = [meta_info['length'], meta_info['width']]

    # Setting padding for block processing
    pad_shape = (0, 0)
    block_params = block_param_generator(
        lines_per_block, data_shape, pad_shape)

    # Determine the logical operation function
    logical_function = np.logical_or if mode == 'or' else np.logical_and

    # Iterating through blocks
    for block_param in block_params:
        combined_binary_image = None

        for layer, value in zip(layer_list, value_list):
            layer_block = get_raster_block(layer, block_param)
            binary_image = (layer_block == value).astype(np.uint8)

            if combined_binary_image is None:
                combined_binary_image = binary_image
            else:
                combined_binary_image = logical_function(
                    combined_binary_image, binary_image).astype(np.uint8)

        # Writing the merged block to the output raster
        write_raster_block(
            out_raster=merged_layer_path,
            data=combined_binary_image,
            block_param=block_param,
            geotransform=meta_info['geotransform'],
            projection=meta_info['projection'],
            datatype='byte',
            cog_flag=cog_flag,
            scratch_dir=scratch_dir)


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
    plt.figure(figsize=(20, 20))
    _, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.imshow(10 * np.log10(intensity),
              cmap=plt.get_cmap('gray'),
              vmin=immin,
              vmax=immax)
    plt.title('RTC')
    plt.savefig(os.path.join(outputdir, f'RTC_intensity_{pol}'))


def block_threshold_visualization(intensity, block_row, block_col,
                                  threshold_tile, output_dir, fig_name):
    """
    Visualize an intensity image overlaid with threshold values from
    specified blocks/subtiles.

    Parameters:
    -----------
    intensity : numpy.ndarray
        A 2D or 3D array representing the intensity of the image.
        If 3D, only the second and third dimensions (rows and columns) are used for visualization.
    block_row : int
        Number of rows in each block/subtile.
    block_col : int
        Number of columns in each block/subtile.
    threshold_tile : numpy.ndarray
        2D array containing the threshold values for each block/subtile.
        Its dimensions should match the number of blocks in the intensity image.
    outputdir : str
        Path to the directory where visualizations will be saved.
    figname : str
        Name for the saved visualization figure.
    """

    if intensity.ndim == 2:
        rows, cols = intensity.shape
    elif intensity.ndim == 3:
        _, rows, cols = intensity.shape

    nrow_tile = int(np.ceil(rows / block_row))
    ncol_tile = int(np.ceil(cols / block_col))
    assert nrow_tile == threshold_tile.shape[0], 'Row tile size error'
    assert ncol_tile == threshold_tile.shape[1], 'Column tile size error'

    intensity_db = 10 * np.log10(intensity)

    plt.figure(figsize=(20, 20))
    vmin, vmax = np.nanpercentile(intensity_db, [5, 95])
    plt.imshow(intensity_db, cmap='gray', vmin=vmin, vmax=vmax)

    threshold_oversample = np.zeros_like(intensity_db)
    for i in range(nrow_tile):
        for j in range(ncol_tile):
            i_end = min((i + 1) * block_row, rows)
            j_end = min((j + 1) * block_col, cols)

            threshold_oversample[i*block_row:i_end, j*block_col:j_end] = \
                threshold_tile[i, j]

            # Draw the tile rectangle
            plt.plot([j*block_col, j_end, j_end, j*block_col, j*block_col],
                     [i*block_row, i*block_row, i_end, i_end, i*block_row],
                     'black')

    threshold_oversample[threshold_oversample == -50] = np.nan
    plt.imshow(threshold_oversample, alpha=0.3, cmap='jet',
               vmin=-20, vmax=-14)

    plt.savefig(os.path.join(output_dir, fig_name))

    plt.close()


def block_threshold_visulaization_rg(intensity, threshold_dict, outputdir, figname):
    """
    Visualize an intensity image overlaid with threshold values from provided blocks/subtiles.

    Parameters
    -----------
    intensity : numpy.ndarray
        2D or 3D array representing the intensity of the image.
        If 3D, the first dimension is considered as the band index.
    threshold_dict : dict
        A dictionary containing:
        * 'array': A nested list of threshold values for each band and block.
        * 'subtile_coord': A nested list of block coordinates for each band and block,
          in the format [[start_row, end_row, start_col, end_col], ...].
    outputdir : str
        Path to the directory where visualizations will be saved.
    figname : str
        Base name for the saved visualization figures.

    Returns
    --------
    None. The visualized figures are saved to the specified directory.
    """
    # Determine the dimensions and the number of bands based on the shape of the intensity array
    if len(intensity.shape) == 2:
        rows, cols = intensity.shape
        bands = [intensity]
    else:
        bands = [intensity[i] for i in range(intensity.shape[0])]
        _, rows, cols = np.shape(intensity)

    for band_index, band in enumerate(bands):
        intensity_db = 10 * np.log10(band)

        plt.figure(figsize=(20, 20))
        vmin = np.nanpercentile(intensity_db, 5)
        vmax = np.nanpercentile(intensity_db, 95)

        # Display the main intensity image
        plt.imshow(intensity_db, cmap='gray', vmin=vmin, vmax=vmax)

        # Prepare a matrix for the overlaid threshold values
        threshold_overlay = np.full((rows, cols), np.nan)

        for threshold, coords in zip(threshold_dict['array'][band_index], threshold_dict['subtile_coord'][band_index]):
            start_row, end_row, start_col, end_col = coords
            threshold_overlay[start_row:end_row, start_col:end_col] = threshold

            # Draw a block boundary for visualization
            plt.plot([start_col, end_col, end_col, start_col, start_col],
                     [start_row, start_row, end_row, end_row, start_row], 'black')

        # Overlay the threshold values on top of the intensity image
        plt.imshow(threshold_overlay, alpha=0.3, cmap='jet', vmin=-20, vmax=-14)

        # Save the visualization to file
        plt.savefig(os.path.join(outputdir, f'{figname}_{band_index}'))
        plt.close()


def _compute_browse_array(
        masked_interpreted_water_layer,
        flag_collapse_wtr_classes=True,
        exclude_inundated_vegetation=False,
        set_not_water_to_nodata=False,
        set_hand_mask_to_nodata=False,
        set_layover_shadow_to_nodata=False,
        set_ocean_masked_to_nodata=True):
    """
    Generate a version of the WTR layer where the
    pixels marked with dark land and bright water in the CLOUD layer
    are designated with unique values per dswx_hls.py constants (see notes).

    Parameters
    ----------
    masked_interpreted_water_layer : numpy.ndarray
        interpreted water layer
        (i.e. the DSWx-S1 WTR layer)
    flag_collapse_wtr_classes : bool
        Collapse interpreted layer water classes following standard
        DSWx-S1 product water classes
    exclude_inundated_vegetation : bool
        True to exclude Inundated vegetation
        in output layer and instead display them as Not Water. 
        False to display these pixels as PSW. Default is False.
    set_not_water_to_nodata : bool
        How to code the Not Water pixels. Defaults to False. Options are:
            True : Not Water pixels will be marked with UINT8_FILL_VALUE
            False : Not Water will remain WATER_NOT_WATER_CLEAR
    set_hand_mask_to_nodata : bool
        How to code the hand mask pixels. Defaults to False. Options are:
            True : cloud pixels will be marked with UINT8_FILL_VALUE
            False : cloud will remain WTR_CLOUD_MASKED
    set_layover_shadow_to_nodata : bool
        How to code the snow pixels. Defaults to False. Options are:
            True : snow pixels will be marked with UINT8_FILL_VALUE
            False : snow will remain WTR_SNOW_MASKED
   set_ocean_masked_to_nodata : bool
        How to code the ocean-masked pixels. Defaults to True. Options are:
            True : ocean-masked pixels will be marked with UINT8_FILL_VALUE
            False : ocean-masked will remain WTR_OCEAN_MASKED
    
    Returns
    -------
    browse_arr : numpy.ndarray
        Interpreted water layer adjusted for the input parameters provided.
    """

    # Create a copy of the masked_interpreted_water_layer.
    browse_arr = masked_interpreted_water_layer.copy()

    if flag_collapse_wtr_classes:
        browse_arr = _collapse_wtr_classes(browse_arr)

    # Discard the Partial Surface Water Aggressive class
    if exclude_inundated_vegetation:
        browse_arr[
            browse_arr == band_assign_value_dict['inundated_vegetation']] = \
                band_assign_value_dict['water']

    if set_not_water_to_nodata:
        browse_arr[
            browse_arr == band_assign_value_dict['no_water']] = \
            band_assign_value_dict['no_data']

    if set_hand_mask_to_nodata:
        browse_arr[
            browse_arr == band_assign_value_dict['hand_mask']] = \
            band_assign_value_dict['no_data']

    if set_layover_shadow_to_nodata:
        browse_arr[
            browse_arr == band_assign_value_dict['layover_shadow_mask']] = \
            band_assign_value_dict['no_data']

    if set_ocean_masked_to_nodata:
        browse_arr[
            browse_arr == band_assign_value_dict['ocean_mask']] = \
            band_assign_value_dict['no_data']

    return browse_arr


def _collapse_wtr_classes(interpreted_layer):
    """
       Collapse interpreted layer classes onto final DSWx-HLS
        product WTR classes

       Parameters
       ----------
       interpreted_layer: np.ndarray
              Interpreted layer

       Returns
       -------
       collapsed_interpreted_layer: np.ndarray
              Interpreted layer with collapsed classes
    """
    collapsed_interpreted_layer = np.full_like(
        interpreted_layer,
        band_assign_value_dict['no_data'])
    for original_value, new_value in collapse_wtr_classes_dict.items():
        collapsed_interpreted_layer[interpreted_layer == original_value] = \
            new_value
    return collapsed_interpreted_layer


def _save_array(input_array, output_file, dswx_metadata_dict, geotransform,
                projection, description = None, scratch_dir = '.',
                output_files_list = None, output_dtype = gdal.GDT_Byte,
                ctable = None, no_data_value = None):
    """Save a generic DSWx-HLS layer (e.g., diagnostic layer, shadow layer, etc.)

       Parameters
       ----------
       input_array: numpy.ndarray
              DSWx-HLS layer to be saved
       output_file: str
              Output filename
       dswx_metadata_dict: dict
              Metadata dictionary to be written into the output file
       geotransform: numpy.ndarray
              Geotransform describing the output file geolocation
       projection: str
              Output file's projection
       description: str (optional)
              Band description
       scratch_dir: str (optional)
              Directory for temporary files
       output_files_list: list (optional)
              Mutable list of output files
       output_dtype: gdal.DataType
              GDAL data type
       ctable: GDAL ColorTable object
              GDAL ColorTable object
       no_data_value: numeric
              No data value
    """
    os.makedirs(scratch_dir, exist_ok=True)

    shape = input_array.shape
    driver = gdal.GetDriverByName("GTiff")
    gdal_ds = driver.Create(output_file, shape[1], shape[0], 1, output_dtype)
    if dswx_metadata_dict is not None:
        gdal_ds.SetMetadata(dswx_metadata_dict)
    gdal_ds.SetGeoTransform(geotransform)
    gdal_ds.SetProjection(projection)
    raster_band = gdal_ds.GetRasterBand(1)
    raster_band.WriteArray(input_array)
    if no_data_value is not None:
        raster_band.SetNoDataValue(no_data_value)

    if description is not None:
        raster_band.SetDescription(description)

    if ctable is not None:
        raster_band.SetRasterColorTable(ctable)
        raster_band.SetRasterColorInterpretation(
                gdal.GCI_PaletteIndex)

    gdal_ds.FlushCache()
    gdal_ds = None

    _save_as_cog(output_file, scratch_dir, logger)

    if output_files_list is not None:
        output_files_list.append(output_file)
    logger.info(f'file saved: {output_file}')


def geotiff2png(src_geotiff_filename,
                dest_png_filename,
                output_height=None,
                output_width=None,
                logger=None,
                ):
    """
    Convert a GeoTIFF file to a png file.

    Parameters
    ----------
    src_geotiff_filename : str
        Name (with path) of the source geotiff file to be 
        converted. This file must already exist.
    dest_png_filename : str
        Name (with path) for the output .png file
    output_height : int, optional.
        Height in Pixels for the output png. If not provided, 
        will default to the height of the source geotiff.
    output_width : int, optional.
        Width in Pixels for the output png. If not provided, 
        will default to the width of the source geotiff.
    logger : Logger, optional
        Logger for the project

    """
    # Load the source dataset
    gdal_ds = gdal.Open(src_geotiff_filename, gdal.GA_ReadOnly)

    # Set output height
    if output_height is None:
        output_height = gdal_ds.GetRasterBand(1).YSize

    # Set output height
    if output_width is None:
        output_width = gdal_ds.GetRasterBand(1).XSize
    # select the resampling algorithm to use based on dtype
    
    gdal_dtype = gdal_ds.GetRasterBand(1).DataType
    dtype_name = gdal.GetDataTypeName(gdal_dtype).lower()
    is_integer = 'byte' in dtype_name  or 'int' in dtype_name

    if is_integer:
        resamp_algorithm = 'NEAREST'
    else:
        resamp_algorithm = 'CUBICSPLINE'

    del gdal_ds  # close the dataset (Python object and pointers)

    # Do not output the .aux.xml file alongside the PNG
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')

    # Translate the existing geotiff to the .png format
    gdal.Translate(dest_png_filename, 
                   src_geotiff_filename, 
                   format='PNG',
                   height=output_height,
                   width=output_width,
                   resampleAlg=resamp_algorithm,
                   nogcp=True,  # do not print GCPs
    )

    if logger is None:
        logger = logging.getLogger('dswx_s1')
    logger.info(f'Browse Image PNG created: {dest_png_filename}')


def create_browse_image(water_geotiff_filename, 
                        output_dir_path,
                        browser_filename,
                        browse_image_height,
                        browse_image_width,
                        scratch_dir,
                        flag_collapse_wtr_classes=True,
                        exclude_inundated_vegetation=False,
                        set_not_water_to_nodata=False,
                        set_hand_mask_to_nodata=False,
                        set_layover_shadow_to_nodata=False,
                        set_ocean_masked_to_nodata=False):
    """
    Process a water-related GeoTIFF file to create a browse image.

    The function performs the following steps:
    - Opens the specified GeoTIFF file and reads the water layer.
    - Extracts relevant metadata for geospatial referencing.
    - Computes a browse array based on various masking and data classification criteria.
    - Forms a color table for data visualization.
    - Saves the processed data as a new GeoTIFF file in a scratch directory.
    - Converts the GeoTIFF to a PNG file, resized to the specified dimensions.

    Parameters
    -----------
    water_geotiff_filename : str
        The path to the input water GeoTIFF file.
    output_dir_path : str
        The directory path where the output PNG will be saved.
    browser_filename : str
        The filename for the output browse image PNG.
    browse_image_height : int
        The height of the output browse image.
    browse_image_width : int 
        The width of the output browse image.
    scratch_dir : str 
        The directory path used for temporary storage during processing.
   flag_collapse_wtr_classes : bool, optional
        Flag to collapse water classes if set to True. Default is True.
    exclude_inundated_vegetation : bool, optional
        Flag to exclude inundated vegetation from processing if set to True. 
        Default is False.
    set_not_water_to_nodata : bool, optional
        Flag to set non-water pixels to NoData value if set to True. 
        Default is False.
    set_hand_mask_to_nodata : bool, optional
        Flag to set HAND mask pixels to NoData value if set to True. 
        Default is False.
    set_layover_shadow_to_nodata : bool, optional
        Flag to set layover and shadow pixels to NoData value if set to True. 
        Default is False.
    set_ocean_masked_to_nodata : bool, optional
        Flag to set ocean-masked pixels to NoData value if set to True. 
        Default is False.

    Returns
    --------
    None
    """
    # # Build the browse image
    # # Create the source image as a geotiff
    # # Reason: gdal.Create() cannot currently create .png files, so we
    # # must start from a GeoTiff, etc.
    # # Source: https://gis.stackexchange.com/questions/132298/gdal-c-api-how-to-create-png-or-jpeg-from-scratch
    with rasterio.open(water_geotiff_filename) as src:
        wtr_layer = src.read(1)
    meta_info = get_meta_from_tif(water_geotiff_filename)

    browse_arr = _compute_browse_array(
        masked_interpreted_water_layer=wtr_layer,  # WTR layer
        flag_collapse_wtr_classes=flag_collapse_wtr_classes,
        exclude_inundated_vegetation=exclude_inundated_vegetation,
        set_not_water_to_nodata=set_not_water_to_nodata,
        set_hand_mask_to_nodata=set_hand_mask_to_nodata,
        set_layover_shadow_to_nodata=set_layover_shadow_to_nodata,
        set_ocean_masked_to_nodata=set_ocean_masked_to_nodata,)

    # Form color table
    browse_ctable = get_interpreted_dswx_s1_ctable()
    water_geotiff_basename = os.path.basename(water_geotiff_filename)
    browse_image_geotiff_filename = os.path.join(scratch_dir, 
                                                 water_geotiff_basename)

    _save_array(
        input_array=browse_arr,
        output_file=browse_image_geotiff_filename,
        dswx_metadata_dict=None,
        geotransform=meta_info['geotransform'],
        projection=meta_info['projection'],
        scratch_dir=scratch_dir,
        output_dtype=gdal.GDT_Byte,  # unsigned int 8
        ctable=browse_ctable,
        no_data_value=band_assign_value_dict['no_data'])

    # Convert the geotiff to a resized PNG to create the browse image PNG
    geotiff2png(
        src_geotiff_filename=browse_image_geotiff_filename,
        dest_png_filename=os.path.join(output_dir_path,
                                       browser_filename),
        output_height=browse_image_height,
        output_width=browse_image_width,
        logger=logger
        )


def check_gdal_raster_s3(path_raster_s3: str, raise_error=True):
    '''
    Check if the GDAL raster in S3 bucket is available

    Parameter
    ---------
    path_raster_s3: str
        Path to the GDAL raster in S3 bucket starts with `/vsis3`
    raise_error: bool
        Raise an error when the file is not accessible, rather than
        returning a boolean flag

    Returns
    -------
    _: Bool
        True when the file is accessible; False otherwise.
        Optional when the parameter `raise_error` is `False`.

    Raises
    ------
    RuntimeError
        When the GDAL raster in AWS S3 is not available.
        Optional when the parameter `raise_error` is `True`.
    '''
    if not path_raster_s3.startswith('/vsis3/'):
        raise RuntimeError(f'The raster path {path_raster_s3} is not a '
                           'valid format for GDAL raster in S3 bucket')

    # Currently `gdal.DontUseExceptions()` is called in this code.
    # In that case, failed `gdal.Open()` will return None
    gdal_in = gdal.Open(path_raster_s3)

    is_gdal_file_exist = gdal_in is not None

    if not is_gdal_file_exist and raise_error:
        raise RuntimeError(f'GDAL raster "{path_raster_s3}" is not available.')

    return is_gdal_file_exist
