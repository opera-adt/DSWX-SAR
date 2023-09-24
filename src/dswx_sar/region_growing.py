import mimetypes
import logging
import os
import time

import numpy as np
from scipy import ndimage
from joblib import Parallel, delayed

from dswx_sar import (dswx_sar_util,
                      generate_log)
from dswx_sar.dswx_runconfig import RunConfig, _get_parser


logger = logging.getLogger('dswx_s1')


def region_growing(likelihood_image,
                   initial_threshold=0.6,
                   relaxed_threshold=0.45,
                   maxiter=200,
                   mode='descending'):
    """The regions are then grown from the seed points to adjacent
    points since it covers the relaxed_threshold values.

    Parameters
    ----------
    likelihood_image : numpy.ndarray
        fuzzy image with values [0, 1] representing
        likelihood of features (e.g. water) where 0 is 0% and 1 is 100%
    initial_threshold : float
        Initial threshold [0 - 1] used to classify
        `fuzz_image` into feature and non-feature pixels.
        If a pixel values exceeds `initial_threshold`
        then it is classified as feature, otherwise it is
        classified as non-feature.
    relaxed_threshold : float
        relaxed threshold to be used for transient area
        between feature and non-feature. Any pixel value
        greater than threshold classified as feature,
        otherwise it's classified as non-feature.
    maxiter : integer
        maximum iteration for region growing.
        Defaults to 0 which translates to infinite iterations.

    Returns
    ----------
    binary_image : numpy.ndarray
        result of region growing algorithm
        1: the pixels involved in region growing (i.e., water)
        0: the pixels not involved in region growing (i.e., non-water)
    """
    if mode == 'descending':
        if initial_threshold <= relaxed_threshold:
            err_str = f"Initial threshold {initial_threshold} " \
                      f" should be larger than relaxed threshold" \
                      f"{relaxed_threshold}."
            raise ValueError(err_str)
    else:
        if initial_threshold >= relaxed_threshold:
            err_str = f"Initial threshold {initial_threshold} " \
                      f" should be smaller than relaxed threshold" \
                      f"{relaxed_threshold}."
            raise ValueError(err_str)

    # Create initial binary image using seed value
    if mode == 'descending':
        binary_image = likelihood_image > initial_threshold
    else:
        binary_image = likelihood_image < initial_threshold

    newpixelmin = 0
    itercount = 0
    number_added = 20

    # Maximum iteration 0 is assumed as that region growing
    # should be carried out until all pixels are included.
    if maxiter == 0:
        maxiter = np.inf

    # Run region growing until maximum iteration reaches
    # and no more pixels are found
    while (itercount < maxiter) and (number_added > newpixelmin):

        # exclude the original binary pixels from buffer binary
        buffer_binary = np.logical_xor(
            ndimage.binary_dilation(binary_image), binary_image)

        # define new_binary for the pixels higher than relaxed_threshold
        if mode == 'descending':
            new_binary = likelihood_image[buffer_binary] > relaxed_threshold
        else:
            new_binary = likelihood_image[buffer_binary] < relaxed_threshold 

        # add new pixels to binary_image
        binary_image[buffer_binary] = new_binary
        number_added = np.sum(new_binary)
        itercount += 1
        logger.info(f"iteration {itercount}: {number_added:.3f} pixels added")

    return binary_image


def process_region_growing_block(block_param,
                                 loopind,
                                 base_dir,
                                 fuzzy_base_name,
                                 input_tif_path,
                                 initial_threshold,
                                 relaxed_threshold,
                                 maxiter):
    """Process region growing for blocks

    Parameters
    ----------
    block_param: BlockParam
        Object specifying where and how much to read and write to out_raster
    loopind: integer
        index value for iteration
    base_dir: str
        path for fuzzy geotiff and input geotiff
    fuzzy_base_name: str
        prefix for temporary file name
    input_tif_path: str
        path of initial fuzzy later
    initial_threshold : float
        Initial threshold [0 - 1] used to classify
        `likelihood_image` into feature and non-feature pixels.
        If a pixel values exceeds `initial_threshold`
        then it is classified as feature, otherwise it is
        classified as non-feature.
    relaxed_threshold : float
        relaxed threshold to be used for transient area
        between feature and non-feature. Any pixel value
        greater than threshold classified as feature,
        otherwise it's classified as non-feature.
    maxiter : integer
        maximum iteration for region growing

    Returns
    ----------
    block_param: BlockParam
        Object specifying where and how much to read and write to out_raster
    data_block: numpy.ndarraya
        fuzzy values after region growing
    """
    # At first loop, read block from intial fuzzy value geotiff
    # Otherwise, read block from previous loop
    if loopind == 0:
        fuzzy_map_temp = input_tif_path
    else:
        fuzzy_map_temp = \
            f'{base_dir}/{fuzzy_base_name}_temp_loop_{loopind}.tif'
    data_block = dswx_sar_util.get_raster_block(
        fuzzy_map_temp, block_param)

    # Run region growing for fuzzy values
    region_grow_sub = region_growing(data_block,
                                     initial_threshold=initial_threshold,
                                     relaxed_threshold=relaxed_threshold,
                                     maxiter=maxiter)
    # replace fuzzy values with 1 for the pixels included by region growing
    data_block[region_grow_sub == 1] = 1

    return block_param, data_block


def run_parallel_region_growing(input_tif_path,
                                output_tif_path,
                                lines_per_block=200,
                                initial_threshold=0.6,
                                relaxed_threshold=0.45,
                                maxiter=200):
    """Perform region growing in parallel

    Parameters
    ----------
    input_tif_path: str
        path of fuzzy-logic value Geotiff
    output_tif_path: str
        path of region-growing
    lines_per_block: int
        lines per block
    initial_threshold: float
        Initial seed values where region-growing starts.
        Initial threshold [0 - 1] used to classify
        `likelihood_image` into feature and non-feature pixels.
        If a pixel values exceeds `initial_threshold`
        then it is classified as feature, otherwise it is
        classified as non-feature.
    relaxed_threshold: float
        value where region-growing stops.
        relaxed threshold to be used for transient area
        between feature and non-feature. Any pixel value
        greater than threshold classified as feature,
        otherwise it's classified as non-feature.
    maxiter: integer
        maximum number of dilation
    """
    meta_dict = dswx_sar_util.get_meta_from_tif(input_tif_path)
    data_length = meta_dict['length']
    data_width = meta_dict['width']
    data_shape = [data_length, data_width]

    # Process fast region-growing with blocks
    # In each iteration, the block size will increase to cover
    # more areas to accelerate processing in challenging
    # areas after the initial iteration
    lines_per_block_list = [lines_per_block,
                            2*lines_per_block,
                            3*lines_per_block]
    num_loop = len(lines_per_block_list)

    for loopind, lines_per_block_loop in enumerate(lines_per_block_list):
        base_dir = os.path.dirname(output_tif_path)
        fuzzy_base_name = os.path.splitext(
            os.path.basename(input_tif_path))[0]
        lines_per_block_loop = min(data_length,
                                   lines_per_block_loop)
        pad_shape = (0, 0)
        block_params = dswx_sar_util.block_param_generator(
            lines_per_block_loop,
            data_shape,
            pad_shape)
        # run region-growing for blocks in parallel
        result = Parallel(n_jobs=-1)(delayed(process_region_growing_block)(
            block_param,
            loopind,
            base_dir,
            fuzzy_base_name,
            input_tif_path,
            initial_threshold,
            relaxed_threshold,
            maxiter)
            for block_param in block_params)

        for block_param, region_grow_block in result:
            fuzzy_map_temp = \
                f'{base_dir}/{fuzzy_base_name}_temp_loop_{loopind + 1}.tif'

            dswx_sar_util.write_raster_block(
                fuzzy_map_temp,
                region_grow_block,
                block_param,
                geotransform=meta_dict['geotransform'],
                projection=meta_dict['projection'],
                DataType='float32')

            # In final loop, write the result to output_tif_path
            if loopind == num_loop - 1:
                dswx_sar_util.write_raster_block(
                    output_tif_path,
                    region_grow_block,
                    block_param,
                    geotransform=meta_dict['geotransform'],
                    projection=meta_dict['projection'])


def run(cfg):
    '''
    Run region growing with parameters in cfg dictionary
    '''
    logger.info('Starting DSWx-S1 Region-Growing')

    t_all = time.time()

    processing_cfg = cfg.groups.processing
    outputdir = cfg.groups.product_path_group.scratch_path
    pol_str = '_'.join(processing_cfg.polarizations)

    # Region growing cfg
    region_growing_cfg = processing_cfg.region_growing
    region_growing_seed = region_growing_cfg.initial_threshold
    region_growing_relaxed_threshold = region_growing_cfg.relaxed_threshold
    region_growing_line_per_block = region_growing_cfg.line_per_block

    print(f'Region Growing Seed: {region_growing_seed}')
    print(f'Region Growing relaxed threshold: {region_growing_relaxed_threshold}')

    fuzzy_tif_path = os.path.join(outputdir,
                                  f'fuzzy_image_{pol_str}.tif')
    feature_meta = dswx_sar_util.get_meta_from_tif(fuzzy_tif_path)
    feature_tif_path = os.path.join(outputdir,
                                  f"region_growing_output_binary_{pol_str}.tif")
    temp_rg_tif_path = os.path.join(outputdir,
                                    f'temp_region_growing_{pol_str}.tif')

    # First, run region-growing algorithm for blocks
    # to avoid to repeatly run with large image.
    run_parallel_region_growing(
        fuzzy_tif_path,
        temp_rg_tif_path,
        lines_per_block=region_growing_line_per_block,
        initial_threshold=region_growing_seed,
        relaxed_threshold=region_growing_relaxed_threshold,
        maxiter=0)

    fuzzy_map = dswx_sar_util.read_geotiff(fuzzy_tif_path)
    temp_rg = dswx_sar_util.read_geotiff(temp_rg_tif_path)

    # replace the fuzzy values with 1 for the pixels
    # where the region-growing already applied
    fuzzy_map[temp_rg == 1] = 1

    # Run region-growing again for entire image
    region_grow_map = region_growing(
        fuzzy_map,
        initial_threshold=region_growing_seed,
        relaxed_threshold=region_growing_relaxed_threshold,
        maxiter=0)

    dswx_sar_util.save_dswx_product(
        region_grow_map,
        feature_tif_path,
        geotransform=feature_meta['geotransform'],
        projection=feature_meta['projection'],
        scratch_dir=outputdir)

    t_all_elapsed = time.time() - t_all
    logger.info(
        f"successfully ran region growing in {t_all_elapsed:.3f} seconds")


def main():
    'Main Function to run region growing'
    parser = _get_parser()

    args = parser.parse_args()
    generate_log.configure_log_file(args.log_file)

    mimetypes.add_type("text/yaml", ".yaml", strict=True)
    flag_first_file_is_text = 'text' in mimetypes.guess_type(
        args.input_yaml[0])[0]

    if not flag_first_file_is_text:
        raise ValueError('input yaml file is not text')

    if len(args.input_yaml) > 1 and flag_first_file_is_text:
        logger.info('ERROR only one runconfig file is allowed')
        return

    cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_s1', args)

    run(cfg)

if __name__ == '__main__':
    main()
