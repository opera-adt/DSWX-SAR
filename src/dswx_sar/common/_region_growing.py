import copy
import gc
import logging
import os
import time

import numpy as np
from scipy import ndimage
from joblib import Parallel, delayed

from dswx_sar.common import _dswx_sar_util


logger = logging.getLogger('dswx_sar')


def region_growing(likelihood_image,
                   initial_threshold=0.6,
                   relaxed_threshold=0.45,
                   maxiter=200,
                   exclude_area=None,
                   mode='descending',
                   verbose=True):
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
    mode : str
        'ascending' or 'descending'
        If region growing mode is 'ascending',
        then algorithm starts from low value to high value
        to find the pixel lower than relaxed threshold.
        The 'descending' starts from high value and find
        the pixel higher than relaxed threshold.

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
                      " should be larger than relaxed threshold" \
                      f"{relaxed_threshold}."
            raise ValueError(err_str)
    elif mode == 'ascending':
        if initial_threshold >= relaxed_threshold:
            err_str = f"Initial threshold {initial_threshold} " \
                      " should be smaller than relaxed threshold" \
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

    if exclude_area is not None:
        target_area = np.invert(exclude_area)

    # Run region growing until maximum iteration reaches
    # and no more pixels are found
    while (itercount < maxiter) and (number_added > newpixelmin):

        # exclude the original binary pixels from buffer binary
        if exclude_area is not None:
            buffer_binary = np.logical_xor(
                ndimage.binary_dilation(binary_image,
                                        mask=target_area),
                binary_image)
        else:
            buffer_binary = np.logical_xor(
                ndimage.binary_dilation(binary_image),
                binary_image)

        # define new_binary for the pixels higher than relaxed_threshold
        if mode == 'descending':
            new_binary = likelihood_image[buffer_binary] > relaxed_threshold
        else:
            new_binary = likelihood_image[buffer_binary] < relaxed_threshold

        # add new pixels to binary_image
        binary_image[buffer_binary] = new_binary
        number_added = np.sum(new_binary)
        itercount += 1
        if verbose:
            logger.info(f"full region growing iteration {itercount}: "
                        f"{number_added:.3f} pixels added")

    return binary_image


def region_growing_fast(
    likelihood_image: np.ndarray,
    initial_threshold: float = 0.6,
    relaxed_threshold: float = 0.45,
    maxiter: int = 200,  # kept for API compatibility; unused in fast path
    exclude_area: np.ndarray | None = None,
    mode: str = 'descending',
    verbose: bool = True,
    structure: np.ndarray | None = None,  # connectivity kernel (e.g., 3x3 ones)
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """
    Grow regions from seeds into neighbors that satisfy a relaxed threshold.

    Parameters
    ----------
    likelihood_image : np.ndarray
        Float array in [0, 1] expressing likelihood (e.g., of water).
    initial_threshold : float
        Seed threshold. For 'descending', seeds are > initial_threshold.
        For 'ascending', seeds are < initial_threshold.
    relaxed_threshold : float
        Growth eligibility threshold. For 'descending', eligible are > relaxed_threshold.
        For 'ascending', eligible are < relaxed_threshold.
    maxiter : int
        Ignored in the fast path (propagation converges in one call).
        Kept only for backward compatibility.
    exclude_area : np.ndarray or None
        Boolean mask; True where growth is forbidden.
    mode : {'ascending','descending'}
        Controls the inequality direction.
    verbose : bool
        If True, emits a short log message.
    structure : np.ndarray or None
        Binary structuring element for connectivity (None → default 8-connected in 2D).
    logger : logging.Logger or None
        Optional logger for messages.

    Returns
    -------
    np.ndarray of dtype bool
        True where grown; False elsewhere.
    """
    if mode not in ('ascending', 'descending'):
        raise ValueError(f"mode must be 'ascending' or 'descending', got {mode!r}")

    if mode == 'descending' and not (initial_threshold > relaxed_threshold):
        raise ValueError(
            f"Initial threshold {initial_threshold} should be larger than relaxed threshold {relaxed_threshold} "
            "for 'descending' mode."
        )
    if mode == 'ascending' and not (initial_threshold < relaxed_threshold):
        raise ValueError(
            f"Initial threshold {initial_threshold} should be smaller than relaxed threshold {relaxed_threshold} "
            "for 'ascending' mode."
        )

    if likelihood_image.dtype.kind == 'f':
        if np.isnan(likelihood_image).any():
            # Treat NaNs as ineligible for growth
            likelihood = np.nan_to_num(likelihood_image, nan=-np.inf if mode=='descending' else np.inf)
        else:
            likelihood = likelihood_image
    else:
        likelihood = likelihood_image.astype(float, copy=False)

    # Seeds S
    if mode == 'descending':
        seeds = likelihood > initial_threshold
        eligible = likelihood > relaxed_threshold
    else:
        seeds = likelihood < initial_threshold
        eligible = likelihood < relaxed_threshold

    if exclude_area is not None:
        if exclude_area.shape != likelihood.shape:
            raise ValueError("exclude_area shape must match likelihood_image")
        ex = exclude_area.astype(bool, copy=False)
        seeds = seeds & (~ex)
        eligible = eligible & (~ex)

    if not seeds.any():
        if verbose and logger:
            logger.info("region_growing: no initial seeds; returning empty mask.")
        return np.zeros_like(seeds, dtype=bool)

    # Default to 4-connected neighbors in 2D if not provided
    if structure is None:
        structure = ndimage.generate_binary_structure(seeds.ndim, 1)

    # One-shot growth to convergence within eligible mask
    grown = ndimage.binary_propagation(seeds, structure=structure, mask=eligible)

    if verbose and logger:
        logger.info(f"region_growing: grown pixels: {int(grown.sum()):,d}")

    return grown


def process_region_growing_block(block_param,
                                 loopind,
                                 base_dir,
                                 fuzzy_base_name,
                                 input_tif_path,
                                 exclude_area_path,
                                 initial_threshold,
                                 relaxed_threshold,
                                 maxiter,
                                 rg_method='normal'):
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
    data_block: numpy.ndarray
        fuzzy values after region growing
    """
    # At first loop, read block from initial fuzzy value geotiff
    # Otherwise, read block from previous loop

    if loopind == 0:
        fuzzy_map_temp = input_tif_path
    else:
        fuzzy_map_temp = \
            f'{base_dir}/{fuzzy_base_name}_temp_loop_{loopind}.tif'
    data_block = _dswx_sar_util.get_raster_block(
        fuzzy_map_temp, block_param)

    if exclude_area_path is not None:
        exclude_block = _dswx_sar_util.get_raster_block(
            exclude_area_path, block_param)
    else:
        exclude_block = None

    # Run region growing for fuzzy values
    if rg_method == 'normal':
        rg_algorithm = region_growing
    elif rg_method == 'fast':
        rg_algorithm = region_growing_fast
    region_grow_sub = rg_algorithm(data_block,
                                   initial_threshold=initial_threshold,
                                   relaxed_threshold=relaxed_threshold,
                                   maxiter=maxiter,
                                   exclude_area=exclude_block,
                                   verbose=False)

    # replace fuzzy values with 1 for the pixels included by region growing
    data_block[region_grow_sub == 1] = 1
    del region_grow_sub
    if exclude_block is not None:
        del exclude_block
    gc.collect()
    return block_param, data_block


def run_parallel_region_growing(input_tif_path,
                                output_tif_path,
                                exclude_area_path=None,
                                lines_per_block=200,
                                initial_threshold=0.6,
                                relaxed_threshold=0.45,
                                maxiter=200,
                                rg_method='normal'):
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
    meta_dict = _dswx_sar_util.get_meta_from_tif(input_tif_path)
    data_length = meta_dict['length']
    data_width = meta_dict['width']
    data_shape = [data_length, data_width]

    num_available_cpu = os.cpu_count()
    # Process fast region-growing with blocks
    # In each iteration, the block size will increase to cover
    # more areas to accelerate processing in challenging
    # areas after the initial iteration
    # Dynamically compute lines_per_block_list
    lines_per_block_list = [lines_per_block]
    next_lines_per_block = lines_per_block
    multiplier = 2
    while next_lines_per_block < data_length and multiplier < 6:
        next_lines_per_block = lines_per_block * multiplier
        lines_per_block_list.append(next_lines_per_block)
        multiplier += 1

    num_loop = len(lines_per_block_list)

    for loopind, lines_per_block_loop in enumerate(lines_per_block_list):
        base_dir = os.path.dirname(output_tif_path)
        fuzzy_base_name = os.path.splitext(
            os.path.basename(input_tif_path))[0]
        lines_per_block_loop = min(data_length,
                                   lines_per_block_loop)
        pad_shape = (0, 0)
        block_params = _dswx_sar_util.block_param_generator(
            lines_per_block_loop,
            data_shape,
            pad_shape)

        num_block = int(np.ceil(data_length / lines_per_block_loop))
        use_cpu = min(num_available_cpu, num_block)

        # run region-growing for blocks in parallel
        result = Parallel(n_jobs=use_cpu)(
            delayed(process_region_growing_block)(
                block_param,
                loopind,
                base_dir,
                fuzzy_base_name,
                input_tif_path,
                exclude_area_path,
                initial_threshold,
                relaxed_threshold,
                maxiter,
                rg_method=rg_method)
            for block_param in block_params)

        for block_param, region_grow_block in result:
            fuzzy_map_temp = \
                f'{base_dir}/{fuzzy_base_name}_temp_loop_{loopind + 1}.tif'

            _dswx_sar_util.write_raster_block(
                fuzzy_map_temp,
                region_grow_block,
                block_param,
                geotransform=meta_dict['geotransform'],
                projection=meta_dict['projection'],
                datatype='float32',
                cog_flag=True,
                scratch_dir=base_dir)

            # In final loop, write the result to output_tif_path
            if loopind == num_loop - 1:
                _dswx_sar_util.write_raster_block(
                    output_tif_path,
                    region_grow_block,
                    block_param,
                    geotransform=meta_dict['geotransform'],
                    projection=meta_dict['projection'],
                    datatype='float32',
                    cog_flag=True,
                    scratch_dir=base_dir)
        del result, region_grow_block
        gc.collect()  # Invoke garbage collector


def run(cfg):
    '''
    Run region growing with parameters in cfg dictionary
    '''
    logger.info('Starting DSWx-S1 Region-Growing')

    t_all = time.time()

    processing_cfg = cfg.groups.processing
    outputdir = cfg.groups.product_path_group.scratch_path
    pol_list = copy.deepcopy(processing_cfg.polarizations)
    pol_options = processing_cfg.polarimetric_option

    if pol_options is not None:
        pol_list += pol_options

    pol_str = '_'.join(pol_list)

    # Region growing cfg
    region_growing_cfg = processing_cfg.region_growing
    region_growing_seed = region_growing_cfg.initial_threshold
    region_growing_relaxed_threshold = region_growing_cfg.relaxed_threshold
    region_growing_line_per_block = region_growing_cfg.line_per_block

    logger.info(f'Region Growing Seed: {region_growing_seed}')
    logger.info('Region Growing relaxed threshold: '
                f'{region_growing_relaxed_threshold}')

    fuzzy_tif_path = os.path.join(
        outputdir, f'fuzzy_image_{pol_str}.tif')
    feature_meta = _dswx_sar_util.get_meta_from_tif(fuzzy_tif_path)
    feature_tif_path = os.path.join(
        outputdir, f"region_growing_output_binary_{pol_str}.tif")
    temp_rg_tif_path = os.path.join(
        outputdir, f'temp_region_growing_{pol_str}.tif')

    # First, run region-growing algorithm for blocks
    # to avoid repeated run with large image.
    run_parallel_region_growing(
        fuzzy_tif_path,
        temp_rg_tif_path,
        lines_per_block=region_growing_line_per_block,
        initial_threshold=region_growing_seed,
        relaxed_threshold=region_growing_relaxed_threshold,
        maxiter=0)

    fuzzy_map = _dswx_sar_util.read_geotiff(fuzzy_tif_path)
    temp_rg = _dswx_sar_util.read_geotiff(temp_rg_tif_path)

    # replace the fuzzy values with 1 for the pixels
    # where the region-growing already applied
    fuzzy_map[temp_rg == 1] = 1
    del temp_rg

    # Run region-growing again for entire image
    region_grow_map = region_growing(
        fuzzy_map,
        initial_threshold=region_growing_seed,
        relaxed_threshold=region_growing_relaxed_threshold,
        maxiter=0)

    _dswx_sar_util.save_dswx_product(
        region_grow_map,
        feature_tif_path,
        geotransform=feature_meta['geotransform'],
        projection=feature_meta['projection'],
        scratch_dir=outputdir)

    t_all_elapsed = time.time() - t_all
    logger.info(
        f"successfully ran region growing in {t_all_elapsed:.3f} seconds")
