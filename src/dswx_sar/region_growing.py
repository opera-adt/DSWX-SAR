import time
import os
import numpy as np
from scipy import ndimage
import cv2

import time
import logging
import mimetypes
from joblib import Parallel, delayed

from dswx_sar import dswx_sar_util
from dswx_sar.dswx_runconfig import RunConfig, _get_parser


logger = logging.getLogger('dswx_s1')

def region_growing(defuzz, initial_seed=0.6, tolerance=0.45, maxiter=200):
    """The regions are then grown from the seed points to adjacent points
    since it covers the tolerance values.

    Parameters
    ----------
    defuzz : numpy.ndarray
        fuzzy image
    initial_seed : float
        Threshold used for the seed of water body [0 ~ 1]
    tolerance : float
        relaxed threshold to be used for transient area between water and non-water
    maxiter : integer
        maximum iteration for region growing

    Returns
    ----------
    water_binary : numpy.ndarray
        result of region growing algorithm
        1: the pixels involved in region growing (water)
        0: the pixels not involved in region growing (non-water)
    """

    # Create initial binary image using seed value
    water_binary = defuzz > initial_seed

    # Create another layer with uint8
    thresh = np.zeros(defuzz.shape, dtype=np.uint8)
    thresh[water_binary] = 1

    nb_components, _, _, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    nb_components = nb_components - 1

    newpixelmin = 0
    itercount = 0
    number_added = 20

    # Maximum iteration 0 is assumed as that region growing should be carried out
    # until all pixels are included.
    if maxiter == 0:
        maxiter = np.inf

    # Run region growing until maximum iteration reaches and no more pixels are found
    while (itercount < maxiter) and (number_added > newpixelmin):

        # apply binary dilation
        buffer_binary = ndimage.binary_dilation(water_binary)

        # exclude the original binary pixels from buffer binary
        buffer_binary = np.logical_xor(ndimage.binary_dilation(water_binary), water_binary)

        # define new_water for the pixels higher than tolerance
        # new_water = np.logical_and(buffer_binary, defuzz > tolerance)
        new_water = defuzz[buffer_binary] > tolerance

        # add new pixels to water_binary
        water_binary[buffer_binary] = new_water
        number_added = np.sum(new_water)
        itercount = itercount + 1

    return water_binary

def process_region_growing_block(block_param,
                                 loopind,
                                 base_dir,
                                 fuzzy_base_name,
                                 input_tif_path,
                                 initial_seed,
                                 tolerance,
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
    initial_seed : float
        Threshold used for the seed of water body [0 ~ 1]
    tolerance : float
        relaxed threshold to be used for transient area between water and non-water
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
        data_block = dswx_sar_util.get_raster_block(input_tif_path, block_param)
    else:
        fuzzy_map_temp = f'{base_dir}/{fuzzy_base_name}_temp_loop_{loopind}.tif'
        data_block = dswx_sar_util.get_raster_block(fuzzy_map_temp, block_param)

    # Run region growing for fuzzy values
    region_grow_sub = region_growing(data_block,
                                     initial_seed=initial_seed,
                                     tolerance=tolerance,
                                     maxiter=maxiter)
    # replace fuzzy values with 1 for the pixels included by region growing
    data_block[region_grow_sub == 1] = 1

    return block_param, data_block

def run_parallel_region_growing(input_tif_path,
                                output_tif_path,
                                lines_per_block=200,
                                initial_seed=0.6,
                                tolerance=0.45,
                                maxiter=200):
    """Process region growing using parallel

    Parameters
    ----------
    input_tif_path: str
        path of fuzzy-logic value Geotiff
    output_tif_path: str
        path of region-growing
    lines_per_block: int
        lines per block
    initial_seed: float
        initial seed values where region-growing starts
    tolerance: float
        value where region-growing stops
    maxiter: integer
        maximum number of dilation
    """

    meta_dict = dswx_sar_util.get_meta_from_tif(input_tif_path)
    data_length = meta_dict['length']
    data_width = meta_dict['width']
    data_shape = [data_length, data_width]

    # Process fast region-growing with blocks
    lines_per_block_list = [lines_per_block, 2*lines_per_block, 3*lines_per_block]
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
        result = Parallel(n_jobs=10)(delayed(process_region_growing_block)(
                                            block_param,
                                            loopind,
                                            base_dir,
                                            fuzzy_base_name,
                                            input_tif_path,
                                            initial_seed,
                                            tolerance,
                                            maxiter)
                                            for block_param in block_params)

        for block_param, region_grow_block in result:
            fuzzy_map_temp = f'{base_dir}/{fuzzy_base_name}_temp_loop_{loopind + 1}.tif'

            dswx_sar_util.write_raster_block(fuzzy_map_temp,
                                             region_grow_block,
                                             block_param,
                                             geotransform=meta_dict['geotransform'],
                                             projection=meta_dict['projection'],
                                             DataType='float32')

            # In final loop, write the result to output_tif_path
            if loopind == num_loop - 1:
                dswx_sar_util.write_raster_block(output_tif_path,
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
    pol_list = processing_cfg.polarizations
    pol_str = '_'.join(pol_list)

    ## Region growing cfg
    region_growing_cfg = processing_cfg.region_growing
    region_growing_seed = region_growing_cfg.seed
    region_growing_tolerance = region_growing_cfg.tolerance
    region_growing_line_per_block = region_growing_cfg.line_per_block


    fuzzy_tif_str = os.path.join(outputdir, 'fuzzy_image_{}.tif'.format(pol_str))
    water_meta = dswx_sar_util.get_meta_from_tif(fuzzy_tif_str)
    water_tif_str = os.path.join(outputdir, f"region_growing_output_binary_{pol_str}.tif")
    temp_rg_tif_str = os.path.join(outputdir, 'temp_region_growing_{}.tif'.format(pol_str))

    # First, run region-growing algorithm for blocks to avoid to repeatly run with large image.
    run_parallel_region_growing(fuzzy_tif_str,
                                temp_rg_tif_str,
                                lines_per_block=region_growing_line_per_block,
                                initial_seed=region_growing_seed,
                                tolerance=region_growing_tolerance,
                                maxiter=0)

    fuzzy_map = dswx_sar_util.read_geotiff(fuzzy_tif_str)
    temp_rg = dswx_sar_util.read_geotiff(temp_rg_tif_str)

    # replace the fuzzy values with 1 for the pixels
    # where the region-growing already applied
    fuzzy_map[temp_rg==1] = 1

    # Run region-growing again for entire image
    region_grow_map = region_growing(fuzzy_map,
                            initial_seed=region_growing_seed,
                            tolerance=region_growing_tolerance,
                            maxiter=0)

    dswx_sar_util.save_dswx_product(region_grow_map,
                water_tif_str,
                geotransform=water_meta['geotransform'],
                projection=water_meta['projection'],
                description='Water classification (WTR)',
                scratch_dir=outputdir)

    t_all_elapsed = time.time() - t_all
    logger.info(f"successfully ran region growing in {t_all_elapsed:.3f} seconds")

def main():
    parser = _get_parser()

    args = parser.parse_args()

    mimetypes.add_type("text/yaml", ".yaml", strict=True)
    flag_first_file_is_text = 'text' in mimetypes.guess_type(
        args.input_yaml[0])[0]

    if flag_first_file_is_text:
        cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_s1', args)
    run(cfg)

if __name__ == '__main__':
    main()