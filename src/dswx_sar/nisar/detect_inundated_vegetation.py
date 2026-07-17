import copy
import logging
import mimetypes
import os
import time

import numpy as np
from collections import deque
from scipy import ndimage

from dswx_sar.common import _filter_SAR, _generate_log
from dswx_sar.common import _dswx_sar_util
from dswx_sar.nisar.dswx_ni_runconfig import DSWX_NI_POL_DICT, _get_parser, RunConfig
from dswx_sar.common import _detect_inundated_vegetation
from dswx_sar.common._pre_processing import pol_ratio
from dswx_sar.common._masking_with_ancillary import FillMaskLandCover
from dswx_sar.common._fuzzy_value_computation import (
    create_slope_angle_geotiff,
    smf,
    zmf,
)

from dswx_sar.common._region_growing import (
    run_parallel_region_growing,
    region_growing_fast,
)


logger = logging.getLogger('dswx_sar')


def _compute_iv_fuzzy_value(
        ratio_db,
        hand,
        slope,
        target_area,
        no_data,
        ratio_min,
        ratio_max,
        hand_min,
        hand_max,
        slope_min,
        slope_max,
        hand_threshold,
        tree_area=None,
        short_vegetation_area=None,
        water_connectivity=None,
        cross_db=None,
        tree_cross_pol_min=None,
        short_vegetation_cross_pol_min=None,
        cross_pol_fuzzy_mode='none',
        cross_pol_soft_margin=2.0,
        cross_pol_weight=0.3,
        short_vegetation_water_connectivity_weight=0.5,
        fuzzy_mode='weighted_sum'):
    """Compute fuzzy score for inundated vegetation refinement.

    Tree pixels use SAR/topographic fuzzy logic.
    Short vegetation pixels can additionally use DEM/open-water connectivity.

    The optional cross-pol term is intended to reduce false positives during
    fuzzy refinement. It should usually be applied as a soft penalty rather
    than as a hard mask for omission-sensitive cases.

    Parameters
    ----------
    cross_pol_fuzzy_mode : {'none', 'soft', 'hard'}
        'none': do not use cross-pol in fuzzy refinement.
        'soft': multiply fuzzy score by ((1 - weight) + weight * cross_s).
        'hard': set fuzzy score to 0 where cross-pol is below threshold.
    cross_pol_soft_margin : float
        Width in dB used by smf() above the cross-pol threshold.
        For example, threshold=-26 and margin=2 gives membership transition
        from -26 to -24 dB.
    cross_pol_weight : float
        Strength of the soft cross-pol penalty, 0-1.
    short_vegetation_water_connectivity_weight : float
        Strength of DEM/open-water connectivity penalty for short vegetation,
        0-1. With 0.5, short_fuzzy = sar_fuzzy * (0.5 + 0.5 * water_conn_s).
    """

    ratio_db = np.asarray(ratio_db, dtype=np.float32)
    hand = np.asarray(hand, dtype=np.float32)
    slope = np.asarray(slope, dtype=np.float32)

    ratio_s = smf(ratio_db, ratio_min, ratio_max)
    hand_z = zmf(hand, hand_min, hand_max)
    slope_z = zmf(slope, slope_min, slope_max)

    if fuzzy_mode == 'product':
        sar_fuzzy = ratio_s * hand_z * slope_z
    else:
        sar_fuzzy = (
            ratio_s * 0.5 +
            hand_z * 0.25 +
            slope_z * 0.25
        )

    sar_fuzzy = np.asarray(sar_fuzzy, dtype=np.float32)

    if tree_area is None:
        tree_area = np.zeros_like(sar_fuzzy, dtype=bool)
    else:
        tree_area = np.asarray(tree_area).astype(bool)

    if short_vegetation_area is None:
        short_vegetation_area = np.zeros_like(sar_fuzzy, dtype=bool)
    else:
        short_vegetation_area = np.asarray(short_vegetation_area).astype(bool)

    # Optional cross-pol false-positive suppression.
    cross_pol_fuzzy_mode = str(cross_pol_fuzzy_mode).lower()
    if cross_pol_fuzzy_mode not in ['none', 'soft', 'hard']:
        raise ValueError(
            f"Invalid cross_pol_fuzzy_mode: {cross_pol_fuzzy_mode}. "
            "Expected 'none', 'soft', or 'hard'."
        )

    if cross_db is not None and cross_pol_fuzzy_mode != 'none':
        cross_db = np.asarray(cross_db, dtype=np.float32)
        cross_s = np.ones_like(sar_fuzzy, dtype=np.float32)

        if tree_cross_pol_min is not None:
            tree_cross_s = smf(
                cross_db,
                tree_cross_pol_min,
                tree_cross_pol_min + cross_pol_soft_margin
            )
            cross_s[tree_area] = tree_cross_s[tree_area]

        if short_vegetation_cross_pol_min is not None:
            short_cross_s = smf(
                cross_db,
                short_vegetation_cross_pol_min,
                short_vegetation_cross_pol_min + cross_pol_soft_margin
            )
            cross_s[short_vegetation_area] = short_cross_s[
                short_vegetation_area]

        cross_s[np.isnan(cross_s)] = 0
        cross_s = np.clip(cross_s, 0, 1)

        if cross_pol_fuzzy_mode == 'hard':
            sar_fuzzy[cross_s <= 0] = 0
        elif cross_pol_fuzzy_mode == 'soft':
            cross_pol_weight = np.clip(cross_pol_weight, 0, 1)
            sar_fuzzy *= ((1.0 - cross_pol_weight) +
                          cross_pol_weight * cross_s)

    if water_connectivity is None:
        water_conn_s = np.zeros_like(sar_fuzzy, dtype=np.float32)
    else:
        water_conn_s = np.asarray(water_connectivity, dtype=np.float32)
        water_conn_s[np.isnan(water_conn_s)] = 0
        water_conn_s = np.clip(water_conn_s, 0, 1)

    # Default: SAR/topography behavior.
    iv_fuzzy = sar_fuzzy.copy()

    # For short vegetation only, DEM/open-water connectivity boosts/suppresses
    # the SAR fuzzy score. This avoids using DEM under trees.
    short_weight = np.clip(short_vegetation_water_connectivity_weight, 0, 1)
    short_fuzzy = sar_fuzzy * (
        (1.0 - short_weight) + short_weight * water_conn_s
    )

    iv_fuzzy[short_vegetation_area] = short_fuzzy[short_vegetation_area]

    # Tree pixels are intentionally left as SAR/topography fuzzy score.
    iv_fuzzy[tree_area] = sar_fuzzy[tree_area]

    valid_area = (
        (target_area > 0) &
        (hand <= hand_threshold) &
        (~no_data)
    )

    iv_fuzzy[~valid_area] = 0
    iv_fuzzy[np.isnan(iv_fuzzy)] = 0

    return iv_fuzzy.astype(np.float32)

def _get_cfg_value(cfg_obj, name, default):
    """Safely read config attribute."""
    if cfg_obj is None:
        return default
    return getattr(cfg_obj, name, default)


def _as_bool_mask(arr):
    """Convert raster values to boolean mask."""
    return np.asarray(arr) > 0



def _compute_water_connectivity_from_open_water(
        dem,
        open_water,
        no_data,
        *,
        dz_max=1.0,
        max_distance_pixels=100,
        barrier_mask=None,
        connectivity=8,
        min_component_pixels=5):
    """DEM-constrained open-water connectivity.

    Water level is estimated as the median DEM elevation of each
    connected open-water component.
    """

    dem = np.asarray(dem, dtype=np.float32)
    open_water = np.asarray(open_water).astype(bool)
    no_data = np.asarray(no_data).astype(bool)

    if barrier_mask is None:
        barrier_mask = np.zeros_like(open_water, dtype=bool)
    else:
        barrier_mask = np.asarray(barrier_mask).astype(bool)

    valid_seed = (
        open_water &
        np.isfinite(dem) &
        (~no_data) &
        (~barrier_mask)
    )

    if connectivity == 4:
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=np.uint8)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        structure = np.ones((3, 3), dtype=np.uint8)
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

    labels, nlabels = ndimage.label(valid_seed, structure=structure)

    conn = np.zeros(open_water.shape, dtype=np.uint8)
    dist = np.full(open_water.shape, np.inf, dtype=np.float32)
    water_level = np.full(open_water.shape, np.nan, dtype=np.float32)

    q = deque()

    for label_id in range(1, nlabels + 1):
        component = labels == label_id
        n_component = np.sum(component)

        if n_component < min_component_pixels:
            continue

        component_dem = dem[component]
        component_dem = component_dem[np.isfinite(component_dem)]

        if component_dem.size == 0:
            continue

        # Estimated local water level for this open-water object.
        wl = np.nanmedian(component_dem).astype(np.float32)

        ys, xs = np.where(component)
        for y, x in zip(ys, xs):
            conn[y, x] = 1
            dist[y, x] = 0.0
            water_level[y, x] = wl
            q.append((y, x))

    height, width = dem.shape

    while q:
        y, x = q.popleft()
        wl = water_level[y, x]

        for dy, dx in neighbors:
            yy = y + dy
            xx = x + dx

            if yy < 0 or yy >= height or xx < 0 or xx >= width:
                continue

            if conn[yy, xx] == 1:
                continue

            if no_data[yy, xx] or barrier_mask[yy, xx]:
                continue

            if not np.isfinite(dem[yy, xx]):
                continue

            new_dist = dist[y, x] + np.hypot(dy, dx)
            if new_dist > max_distance_pixels:
                continue

            # Stop growth if terrain is too high above the propagated
            # component-level water surface.
            if dem[yy, xx] > wl + dz_max:
                continue

            conn[yy, xx] = 1
            dist[yy, xx] = new_dist
            water_level[yy, xx] = wl
            q.append((yy, xx))

    water_conn_s = np.exp(-dist / max_distance_pixels).astype(np.float32)
    water_conn_s[conn == 0] = 0.0
    water_conn_s[open_water] = 1.0
    water_conn_s[np.isnan(water_conn_s)] = 0.0

    return water_conn_s, conn, dist, water_level


def _build_short_vegetation_water_connectivity(
        cfg,
        pol_all_str,
        im_meta,
        short_vegetation_path,
        no_data_raster_path):
    """Build DEM/open-water connectivity fuzzy layer for short vegetation only."""

    processing_cfg = cfg.groups.processing
    scratch_dir = cfg.groups.product_path_group.scratch_path
    inundated_vege_cfg = processing_cfg.inundated_vegetation
    iv_fuzzy_cfg = getattr(inundated_vege_cfg, 'fuzzy_refinement', None)

    dem_path = os.path.join(scratch_dir, 'interpolated_DEM.tif')

    water_conn_cfg = getattr(iv_fuzzy_cfg, 'water_connectivity', None)

    enabled = _get_cfg_value(water_conn_cfg, 'enabled', False)
    if not enabled:
        return None

    open_water_path = os.path.join(
        scratch_dir,
        f'bimodality_output_binary_{pol_all_str}.tif'
    )

    if not os.path.exists(open_water_path):
        logger.warning(
            f'Open water map is not found: {open_water_path}. '
            'DEM/open-water connectivity will not be used.'
        )
        return None

    if not os.path.exists(dem_path):
        logger.warning(
            f'DEM is not found: {dem_path}. '
            'DEM/open-water connectivity will not be used.'
        )
        return None

    logger.info('Build DEM/open-water connectivity for short vegetation.')

    dem = _dswx_sar_util.read_geotiff(dem_path)
    open_water = _dswx_sar_util.read_geotiff(open_water_path)
    short_veg = _dswx_sar_util.read_geotiff(short_vegetation_path)
    no_data = _dswx_sar_util.read_geotiff(no_data_raster_path) == 1

    open_water = _as_bool_mask(open_water)
    short_veg = _as_bool_mask(short_veg)

    # Keep open-water seeds, but final support is only for short vegetation.
    water_conn_s, topo_valid, dist, water_level = (
        _compute_fast_water_connectivity_from_open_water(
            dem=dem,
            open_water=open_water,
            no_data=no_data,
            max_distance_pixels=_get_cfg_value(
                water_conn_cfg,
                'max_distance_pixels',
                100
            ),
            z_max=_get_cfg_value(
                water_conn_cfg,
                'z_max',
                _get_cfg_value(water_conn_cfg, 'dz_max', 1.0)
            ),
            min_water_elevation=_get_cfg_value(
                water_conn_cfg,
                'min_water_elevation',
                None
            ),
            max_water_elevation=_get_cfg_value(
                water_conn_cfg,
                'max_water_elevation',
                None
            )
        )
    )
    water_conn_path = os.path.join(
        scratch_dir,
        f'temp_water_connectivity_{pol_all_str}.tif'
    )

    _dswx_sar_util.save_dswx_product(
        water_conn_s,
        water_conn_path,
        geotransform=im_meta['geotransform'],
        projection=im_meta['projection'],
        scratch_dir=scratch_dir
    )
    water_level_path = os.path.join(
        scratch_dir,
        f'temp_water_level_{pol_all_str}.tif'
    )

    _dswx_sar_util.save_dswx_product(
        water_level,
        water_level_path,
        geotransform=im_meta['geotransform'],
        projection=im_meta['projection'],
        scratch_dir=scratch_dir
    )
    # IMPORTANT:
    # DEM-based support is used only for short vegetation.
    water_conn_s[~short_veg] = 0.0
    topo_valid[~short_veg] = False
    water_conn_path = os.path.join(
        scratch_dir,
        f'temp_short_vegetation_water_connectivity_{pol_all_str}.tif'
    )

    _dswx_sar_util.save_dswx_product(
        water_conn_s,
        water_conn_path,
        geotransform=im_meta['geotransform'],
        projection=im_meta['projection'],
        scratch_dir=scratch_dir
    )
    _dswx_sar_util.save_raster_gdal(
        data=water_conn_s,
        output_file=water_conn_path,
        geotransform=im_meta['geotransform'],
        projection=im_meta['projection'],
        scratch_dir=scratch_dir,
        datatype='float32')

    topo_valid_path = os.path.join(
        scratch_dir,
        f'temp_topo_valid_{pol_all_str}.tif'
    )

    _dswx_sar_util.save_dswx_product(
        topo_valid,
        topo_valid_path,
        geotransform=im_meta['geotransform'],
        projection=im_meta['projection'],
        scratch_dir=scratch_dir
    )
    if processing_cfg.debug_mode:
        # _dswx_sar_util.save_dswx_product(
        #     conn,
        #     os.path.join(
        #         scratch_dir,
        #         f'temp_short_vegetation_water_connected_{pol_all_str}.tif'
        #     ),
        #     geotransform=im_meta['geotransform'],
        #     projection=im_meta['projection'],
        #     scratch_dir=scratch_dir
        # )

        _dswx_sar_util.save_dswx_product(
            dist,
            os.path.join(
                scratch_dir,
                f'temp_short_vegetation_distance_to_water_{pol_all_str}.tif'
            ),
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            scratch_dir=scratch_dir
        )

        _dswx_sar_util.save_dswx_product(
            water_level,
            os.path.join(
                scratch_dir,
                f'temp_short_vegetation_propagated_water_level_{pol_all_str}.tif'
            ),
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            scratch_dir=scratch_dir
        )

    logger.info('Finished DEM/open-water connectivity for short vegetation.')

    return water_conn_path


from scipy import ndimage
import numpy as np


def _compute_fast_water_connectivity_from_open_water(
        dem,
        open_water,
        no_data,
        *,
        max_distance_pixels=100,
        z_max=1.0,
        min_water_elevation=None,
        max_water_elevation=None):
    """Fast DEM/open-water topographic plausibility layer.

    This creates a water-connectivity/topographic-support fuzzy layer by:
    1. finding pixels within max_distance_pixels from open water,
    2. assigning each nearby pixel the DEM elevation of its nearest
       open-water pixel,
    3. testing whether DEM <= nearest_water_elevation + z_max.

    This is much faster than queue-based region growing.

    Parameters
    ----------
    dem : np.ndarray
        DEM in meters.
    open_water : np.ndarray
        Boolean open-water mask.
    no_data : np.ndarray
        Boolean no-data mask.
    max_distance_pixels : int
        Maximum distance from open water in pixels.
    z_max : float
        Allowed elevation above nearby water level, in meters.
    min_water_elevation, max_water_elevation : float or None
        Optional filters to remove unrealistic water DEM values.

    Returns
    -------
    water_conn_s : np.ndarray
        Float32 fuzzy support layer, 0-1.
    topo_valid : np.ndarray
        Boolean topographically plausible area.
    distance : np.ndarray
        Distance to nearest open water, in pixels.
    nearest_water_level : np.ndarray
        DEM elevation of nearest open-water pixel.
    """

    dem = np.asarray(dem, dtype=np.float32)
    open_water = np.asarray(open_water).astype(bool)
    no_data = np.asarray(no_data).astype(bool)

    valid_water = (
        open_water &
        np.isfinite(dem) &
        (~no_data)
    )

    if min_water_elevation is not None:
        valid_water &= dem >= min_water_elevation

    if max_water_elevation is not None:
        valid_water &= dem <= max_water_elevation

    if np.sum(valid_water) == 0:
        shape = dem.shape
        return (
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=bool),
            np.full(shape, np.inf, dtype=np.float32),
            np.full(shape, np.nan, dtype=np.float32)
        )

    # distance_transform_edt computes distance to zeros.
    # Therefore use ~valid_water so valid water pixels are zeros.
    distance, indices = ndimage.distance_transform_edt(
        ~valid_water,
        return_indices=True
    )

    distance = distance.astype(np.float32)

    nearest_water_level = dem[tuple(indices)].astype(np.float32)

    near_water = distance <= max_distance_pixels

    topo_valid = (
        near_water &
        np.isfinite(dem) &
        np.isfinite(nearest_water_level) &
        (~no_data) &
        (dem <= nearest_water_level + z_max)
    )

    # Fuzzy distance decay.
    # Near open water gets high support; farther pixels get lower support.
    water_conn_s = np.exp(-distance / max_distance_pixels).astype(np.float32)

    # Keep only topographically valid nearby pixels.
    water_conn_s[~topo_valid] = 0.0
    water_conn_s[valid_water] = 1.0
    water_conn_s[np.isnan(water_conn_s)] = 0.0

    return water_conn_s, topo_valid, distance, nearest_water_level


def _refine_inundated_vegetation_with_fuzzy_region_growing(
        cfg,
        pol_all_str,
        ratio_db_path,
        cross_db_path,
        inundated_vege_path,
        target_area_path,
        tree_area_path,
        short_vegetation_area_path,
        im_meta):
    """Refine inundated vegetation using IV fuzzy logic and region growing."""

    processing_cfg = cfg.groups.processing
    scratch_dir = cfg.groups.product_path_group.scratch_path

    inundated_vege_cfg = processing_cfg.inundated_vegetation
    fuzzy_cfg = processing_cfg.fuzzy_value
    region_growing_cfg = processing_cfg.region_growing

    # Optional IV-specific config.
    # If these fields do not exist yet, fall back to existing fuzzy/RG params.
    iv_fuzzy_cfg = getattr(inundated_vege_cfg, 'fuzzy_refinement', None)

    if iv_fuzzy_cfg is None:
        logger.info('IV fuzzy refinement is not configured.')
        return

    if not iv_fuzzy_cfg.enabled:
        logger.info('IV fuzzy refinement is disabled.')
        return

    logger.info('Start IV fuzzy refinement with region growing.')

    iv_fuzzy_path = os.path.join(
        scratch_dir,
        f'fuzzy_inundated_vegetation_{pol_all_str}.tif'
    )
    temp_rg_path = os.path.join(
        scratch_dir,
        f'temp_region_growing_inundated_vegetation_{pol_all_str}.tif'
    )
    rg_path = os.path.join(
        scratch_dir,
        f'region_growing_inundated_vegetation_{pol_all_str}.tif'
    )

    lines_per_block = getattr(
        iv_fuzzy_cfg,
        'line_per_block',
        inundated_vege_cfg.line_per_block
    )
    dem_path = os.path.join(scratch_dir, 'interpolated_DEM.tif')
    hand_path = os.path.join(scratch_dir, 'interpolated_hand.tif')
    slope_path = os.path.join(scratch_dir, 'slope.tif')
    no_data_raster_path = os.path.join(
        scratch_dir,
        f'no_data_area_{pol_all_str}.tif'
    )

    if not os.path.exists(slope_path):
        create_slope_angle_geotiff(
            dem_path,
            slope_path,
            lines_per_block=lines_per_block
        )
    ratio_min = getattr(
        iv_fuzzy_cfg,
        'ratio_member_min',
        inundated_vege_cfg.dual_pol_ratio_min
    )
    ratio_max = getattr(
        iv_fuzzy_cfg,
        'ratio_member_max',
        inundated_vege_cfg.dual_pol_ratio_max
    )

    hand_min = getattr(
        iv_fuzzy_cfg,
        'hand_member_min',
        fuzzy_cfg.hand.member_min
    )
    hand_max = getattr(
        iv_fuzzy_cfg,
        'hand_member_max',
        fuzzy_cfg.hand.member_max
    )
    slope_min = getattr(
        iv_fuzzy_cfg,
        'slope_member_min',
        fuzzy_cfg.slope.member_min
    )
    slope_max = getattr(
        iv_fuzzy_cfg,
        'slope_member_max',
        fuzzy_cfg.slope.member_max
    )

    hand_threshold = getattr(
        iv_fuzzy_cfg,
        'hand_threshold',
        processing_cfg.hand.mask_value
    )

    fuzzy_mode = getattr(
        iv_fuzzy_cfg,
        'mode',
        'weighted_sum'
    )

    # Optional cross-pol term inside fuzzy refinement.
    # This is useful for testing whether cross-pol suppresses false positives.
    cross_pol_fuzzy_mode = getattr(
        iv_fuzzy_cfg,
        'cross_pol_fuzzy_mode',
        'none'
    )
    cross_pol_soft_margin = getattr(
        iv_fuzzy_cfg,
        'cross_pol_soft_margin',
        2.0
    )
    cross_pol_weight = getattr(
        iv_fuzzy_cfg,
        'cross_pol_weight',
        0.3
    )

    tree_cross_pol_min = getattr(
        inundated_vege_cfg,
        'tree_cross_pol_min',
        inundated_vege_cfg.cross_pol_min
    )
    short_vegetation_cross_pol_min = getattr(
        inundated_vege_cfg,
        'short_vegetation_cross_pol_min',
        inundated_vege_cfg.cross_pol_min
    )

    water_conn_cfg = getattr(iv_fuzzy_cfg, 'water_connectivity', None)
    short_vegetation_water_connectivity_weight = _get_cfg_value(
        water_conn_cfg,
        'weight',
        0.5
    )

    water_connectivity_path = _build_short_vegetation_water_connectivity(
        cfg=cfg,
        pol_all_str=pol_all_str,
        im_meta=im_meta,
        short_vegetation_path=short_vegetation_area_path,
        no_data_raster_path=no_data_raster_path
    )

    block_params = _dswx_sar_util.block_param_generator(
        lines_per_block=lines_per_block,
        data_shape=(im_meta['length'], im_meta['width']),
        pad_shape=(0, 0)
    )

    for block_ind, block_param in enumerate(block_params):
        logger.info(f'IV fuzzy refinement block {block_ind}')

        ratio_db = _dswx_sar_util.get_raster_block(
            ratio_db_path,
            block_param
        )

        cross_db = _dswx_sar_util.get_raster_block(
            cross_db_path,
            block_param
        )

        hand = _dswx_sar_util.get_raster_block(
            hand_path,
            block_param
        )

        slope = _dswx_sar_util.get_raster_block(
            slope_path,
            block_param
        )

        target_area = _dswx_sar_util.get_raster_block(
            target_area_path,
            block_param
        )
        tree_area = _dswx_sar_util.get_raster_block(
            tree_area_path,
            block_param
        ) > 0

        short_vegetation_area = _dswx_sar_util.get_raster_block(
            short_vegetation_area_path,
            block_param
        ) > 0

        if water_connectivity_path is not None:
            water_connectivity = _dswx_sar_util.get_raster_block(
                water_connectivity_path,
                block_param
            )
        else:
            water_connectivity = None
        no_data = _dswx_sar_util.get_raster_block(
            no_data_raster_path,
            block_param
        ) == 1

        iv_fuzzy = _compute_iv_fuzzy_value(
            ratio_db=ratio_db,
            hand=hand,
            slope=slope,
            target_area=target_area,
            no_data=no_data,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            hand_min=hand_min,
            hand_max=hand_max,
            slope_min=slope_min,
            slope_max=slope_max,
            hand_threshold=hand_threshold,
            tree_area=tree_area,
            short_vegetation_area=short_vegetation_area,
            water_connectivity=water_connectivity,
            cross_db=cross_db,
            tree_cross_pol_min=tree_cross_pol_min,
            short_vegetation_cross_pol_min=short_vegetation_cross_pol_min,
            cross_pol_fuzzy_mode=cross_pol_fuzzy_mode,
            cross_pol_soft_margin=cross_pol_soft_margin,
            cross_pol_weight=cross_pol_weight,
            short_vegetation_water_connectivity_weight=(
                short_vegetation_water_connectivity_weight),
            fuzzy_mode=fuzzy_mode
        )

        _dswx_sar_util.write_raster_block(
            out_raster=iv_fuzzy_path,
            data=iv_fuzzy,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='float32',
            cog_flag=True,
            scratch_dir=scratch_dir
        )

    # First block-wise RG for memory safety.
    run_parallel_region_growing(
        iv_fuzzy_path,
        temp_rg_path,
        lines_per_block=region_growing_cfg.line_per_block,
        initial_threshold=region_growing_cfg.initial_threshold,
        relaxed_threshold=region_growing_cfg.relaxed_threshold,
        maxiter=0,
        rg_method='fast'
    )

    # Then whole-image RG to connect across block boundaries.
    iv_fuzzy = _dswx_sar_util.read_geotiff(iv_fuzzy_path)
    temp_rg = _dswx_sar_util.read_geotiff(temp_rg_path)

    iv_fuzzy[temp_rg == 1] = 1
    del temp_rg

    rg_map = region_growing_fast(
        iv_fuzzy,
        initial_threshold=region_growing_cfg.initial_threshold,
        relaxed_threshold=region_growing_cfg.relaxed_threshold,
        maxiter=0
    )

    del iv_fuzzy

    _dswx_sar_util.save_dswx_product(
        rg_map,
        rg_path,
        geotransform=im_meta['geotransform'],
        projection=im_meta['projection'],
        scratch_dir=scratch_dir
    )

    # Combine with original IV map.
    original_iv = _dswx_sar_util.read_geotiff(inundated_vege_path)
    target_area = _dswx_sar_util.read_geotiff(target_area_path)

    refined_iv = np.zeros_like(original_iv, dtype=np.uint8)
    combine_mode = getattr(iv_fuzzy_cfg, 'combine_mode', 'intersection')

    if combine_mode == 'union':
        refined_mask = (
            ((original_iv == 2) | (rg_map == 1)) &
            (target_area > 0)
        )
    elif combine_mode == 'intersection':
        refined_mask = (
            (original_iv == 2) &
            (rg_map == 1) &
            (target_area > 0)
        )
    else:
        raise ValueError(
            f"Invalid combine_mode for IV fuzzy refinement: {combine_mode}. "
            "Expected 'union' or 'intersection'."
        )
    refined_iv[refined_mask] = 2

    _dswx_sar_util.save_dswx_product(
        refined_iv,
        inundated_vege_path,
        geotransform=im_meta['geotransform'],
        projection=im_meta['projection'],
        scratch_dir=scratch_dir
    )

    if processing_cfg.debug_mode:
        debug_refined_path = os.path.join(
            scratch_dir,
            f'temp_inundated_vegetation_refined_{pol_all_str}.tif'
        )
        _dswx_sar_util.save_dswx_product(
            refined_iv,
            debug_refined_path,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            scratch_dir=scratch_dir
        )

    logger.info('Finished IV fuzzy refinement with region growing.')


def run(cfg):

    logger.info('Start inundated vegetation mapping')

    t_all = time.time()

    processing_cfg = cfg.groups.processing
    scratch_dir = cfg.groups.product_path_group.scratch_path
    pol_list = copy.deepcopy(processing_cfg.polarizations)
    pol_options = processing_cfg.polarimetric_option

    if pol_options is not None:
        pol_list += pol_options

    pol_all_str = '_'.join(pol_list)

    inundated_vege_cfg = processing_cfg.inundated_vegetation
    inundated_vege_ratio_max = inundated_vege_cfg.dual_pol_ratio_max
    inundated_vege_ratio_min = inundated_vege_cfg.dual_pol_ratio_min
    inundated_vege_ratio_threshold = \
        inundated_vege_cfg.dual_pol_ratio_threshold
    inundated_vege_cross_pol_min = inundated_vege_cfg.cross_pol_min
    inundated_vege_copol_threshold = inundated_vege_cfg.copol_threshold
    target_file_type = inundated_vege_cfg.target_area_file_type
    target_worldcover_class = inundated_vege_cfg.target_worldcover_class
    target_glad_class = inundated_vege_cfg.target_glad_class


    target_glad_tree_class = getattr(
        inundated_vege_cfg,
        'target_glad_tree_class',
        []
    )
    target_glad_short_vegetation_class = getattr(
        inundated_vege_cfg,
        'target_glad_short_vegetation_class',
        []
    )

    # Class-specific IV thresholds.
    # If not configured, fall back to the original values.
    tree_ratio_threshold = getattr(
        inundated_vege_cfg,
        'tree_dual_pol_ratio_threshold',
        inundated_vege_ratio_threshold
    )
    short_veg_ratio_threshold = getattr(
        inundated_vege_cfg,
        'short_vegetation_dual_pol_ratio_threshold',
        inundated_vege_ratio_threshold
    )

    tree_cross_pol_min = getattr(
        inundated_vege_cfg,
        'tree_cross_pol_min',
        inundated_vege_cross_pol_min
    )
    short_veg_cross_pol_min = getattr(
        inundated_vege_cfg,
        'short_vegetation_cross_pol_min',
        inundated_vege_cross_pol_min
    )

    tree_copol_threshold = getattr(
        inundated_vege_cfg,
        'tree_copol_threshold',
        inundated_vege_copol_threshold
    )
    short_veg_copol_threshold = getattr(
        inundated_vege_cfg,
        'short_vegetation_copol_threshold',
        inundated_vege_copol_threshold
    )

    line_per_block = inundated_vege_cfg.line_per_block
    filter_options = inundated_vege_cfg.filter
    filter_method = inundated_vege_cfg.filter.method

    iv_fuzzy_cfg = getattr(inundated_vege_cfg, 'fuzzy_refinement', None)
    fuzzy_logic_enabled = (
        iv_fuzzy_cfg is not None and
        getattr(iv_fuzzy_cfg, 'enabled', False)
    )
    interp_glad_path_str = os.path.join(scratch_dir, 'interpolated_glad.tif')
    interp_worldcover_path_str = os.path.join(scratch_dir,
                                              'interpolated_landcover.tif')
    logger.info(f"dual_pol_ratio_threshold: {inundated_vege_ratio_threshold}")
    logger.info(f"dual_pol_ratio_min: {inundated_vege_ratio_min}")
    logger.info(f"dual_pol_ratio_max: {inundated_vege_ratio_max}")
    logger.info(f"cross_pol_min: {inundated_vege_cross_pol_min}")
    logger.info(f"copol_threshold: {inundated_vege_copol_threshold}")
    logger.info(f"filter_method: {filter_method}")
    logger.info(f"filter_options: {filter_options}")
    logger.info(f"tree_ratio_threshold: {tree_ratio_threshold}")
    logger.info(f"short_veg_ratio_threshold: {short_veg_ratio_threshold}")
    logger.info(f"tree_cross_pol_min: {tree_cross_pol_min}")
    logger.info(f"short_veg_cross_pol_min: {short_veg_cross_pol_min}")
    logger.info(f"tree_copol_threshold: {tree_copol_threshold}")
    logger.info(f"short_veg_copol_threshold: {short_veg_copol_threshold}")
    if target_file_type == 'auto':
        if os.path.exists(interp_glad_path_str):
            target_file_type = 'GLAD'
        else:
            target_file_type = 'WorldCover'
    logger.info(f'Vegetation area is extracted from {target_file_type}.')

    # Currently, inundated vegetation for C-band is available for
    # Potential wetland area from Land cover maps
    if target_file_type == 'WorldCover':
        landcover_path_str = interp_worldcover_path_str
    else:
        landcover_path_str = interp_glad_path_str
        sup_mask_obj = FillMaskLandCover(interp_worldcover_path_str,
                                         'WorldCover')
    mask_obj = FillMaskLandCover(landcover_path_str, target_file_type)
    inundated_vege_path = \
        f"{scratch_dir}/temp_inundated_vegetation_{pol_all_str}.tif"
    target_area_path = \
        f"{scratch_dir}/temp_target_area_{pol_all_str}.tif"
    high_ratio_path = \
        f"{scratch_dir}/temp_high_dualpol_ratio_{pol_all_str}.tif"
    ratio_db_path = os.path.join(
        scratch_dir,
        f'temp_intensity_db_ratio_{pol_all_str}.tif'
    )
    cross_db_path = os.path.join(
        scratch_dir,
        f'temp_crosspol_db_{pol_all_str}.tif'
    )

    tree_area_path = \
        f"{scratch_dir}/temp_tree_area_{pol_all_str}.tif"
    short_vegetation_area_path = \
        f"{scratch_dir}/temp_short_vegetation_area_{pol_all_str}.tif"

    dual_pol_flag = False
    if (('HH' in pol_list) and ('HV' in pol_list)) or \
       (('VV' in pol_list) and ('VH' in pol_list)):
        dual_pol_flag = True

    if inundated_vege_cfg.enabled == 'auto':
        if dual_pol_flag:
            inundated_vege_cfg_flag = True
        else:
            inundated_vege_cfg_flag = False
    else:
        inundated_vege_cfg_flag = inundated_vege_cfg.enabled

    if inundated_vege_cfg_flag and not dual_pol_flag:
        err_str = 'Daul polarizations are required for inundated vegetation'
        raise ValueError(err_str)

    for polind, pol in enumerate(pol_list):
        if pol in ['HH', 'VV']:
            copol_ind = polind
        elif pol in ['HV', 'VH']:
            crosspol_ind = polind

    rtc_dual_path = f"{scratch_dir}/filtered_image_{pol_all_str}.tif"
    if not os.path.isfile(rtc_dual_path):
        err_str = f'{rtc_dual_path} is not found.'
        raise FileExistsError(err_str)

    for thr_name, thr_value in [
            ('dual_pol_ratio_threshold', inundated_vege_ratio_threshold),
            ('tree_dual_pol_ratio_threshold', tree_ratio_threshold),
            ('short_vegetation_dual_pol_ratio_threshold',
             short_veg_ratio_threshold)]:
        if (inundated_vege_ratio_min > thr_value) or \
           (inundated_vege_ratio_max < thr_value):
            err_str = f'{thr_name}={thr_value} is not valid.'
            raise ValueError(err_str)

    im_meta = _dswx_sar_util.get_meta_from_tif(rtc_dual_path)

    pad_shape = (filter_options.block_pad, 0)
    block_params = _dswx_sar_util.block_param_generator(
        lines_per_block=line_per_block,
        data_shape=(im_meta['length'],
                    im_meta['width']),
        pad_shape=pad_shape)

    for block_param in block_params:

        rtc_dual = _dswx_sar_util.get_raster_block(
            rtc_dual_path,
            block_param)

        rtc_ratio = pol_ratio(
            np.squeeze(rtc_dual[copol_ind, :, :]),
            np.squeeze(rtc_dual[crosspol_ind, :, :]))

        if filter_method == 'lee':
            filtering_method = _filter_SAR.lee_enhanced_filter
            filter_option = vars(filter_options.lee_filter)

        elif filter_method == 'anisotropic_diffusion':
            filtering_method = _filter_SAR.anisotropic_diffusion
            filter_option = vars(filter_options.anisotropic_diffusion)

        elif filter_method == 'guided_filter':
            filtering_method = _filter_SAR.guided_filter
            filter_option = vars(filter_options.guided_filter)

        elif filter_method == 'bregman':
            filtering_method = _filter_SAR.tv_bregman
            filter_option = vars(filter_options.bregman)

        filt_ratio = filtering_method(
                        rtc_ratio, **filter_option)
        filt_ratio_db = 10 * np.log10(filt_ratio +
                                      _dswx_sar_util.Constants.negligible_value)
        cross_db = 10 * np.log10(
            np.squeeze(rtc_dual[crosspol_ind, :, :]) +
            _dswx_sar_util.Constants.negligible_value)
        co_db = 10 * np.log10(
            np.squeeze(rtc_dual[copol_ind, :, :]) +
            _dswx_sar_util.Constants.negligible_value)


        _dswx_sar_util.write_raster_block(
            out_raster=ratio_db_path,
            data=filt_ratio_db,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='float32',
            cog_flag=True,
            scratch_dir=scratch_dir
        )

        _dswx_sar_util.write_raster_block(
            out_raster=cross_db_path,
            data=cross_db,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='float32',
            cog_flag=True,
            scratch_dir=scratch_dir
        )

        output_data = np.zeros(filt_ratio.shape, dtype='uint8')

        target_cross_pol = cross_db > inundated_vege_cross_pol_min
        target_co_pol = co_db > inundated_vege_copol_threshold
        if target_file_type == 'WorldCover':
            target_inundated_vege_class = mask_obj.get_mask(
                mask_label=target_worldcover_class,
                block_param=block_param)

            # WorldCover fallback:
            # if tree/short classes are not available, treat all target
            # vegetation as tree-like existing behavior.
            target_tree_class = np.array(
                target_inundated_vege_class,
                dtype=bool
            )
            target_short_vegetation_class = np.zeros_like(
                target_tree_class,
                dtype=bool
            )
        elif target_file_type == 'GLAD':
            inundated_vege_target = (
                _detect_inundated_vegetation.parse_ranges(target_glad_class)
            )

            target_inundated_vege_class = mask_obj.get_mask(
                mask_label=inundated_vege_target,
                block_param=block_param)

            tree_target = (
                _detect_inundated_vegetation.parse_ranges(
                    target_glad_tree_class
                )
                if len(target_glad_tree_class) > 0 else []
            )

            short_veg_target = (
                _detect_inundated_vegetation.parse_ranges(
                    target_glad_short_vegetation_class
                )
                if len(target_glad_short_vegetation_class) > 0 else []
            )

            if len(tree_target) > 0:
                target_tree_class = mask_obj.get_mask(
                    mask_label=tree_target,
                    block_param=block_param)
            else:
                target_tree_class = np.zeros_like(
                    target_inundated_vege_class,
                    dtype=bool
                )

            if len(short_veg_target) > 0:
                target_short_vegetation_class = mask_obj.get_mask(
                    mask_label=short_veg_target,
                    block_param=block_param)
            else:
                target_short_vegetation_class = np.zeros_like(
                    target_inundated_vege_class,
                    dtype=bool
                )

            # If no tree/short split is configured, preserve old behavior.
            if (len(tree_target) == 0) and (len(short_veg_target) == 0):
                target_tree_class = np.array(
                    target_inundated_vege_class,
                    dtype=bool
                )
                target_short_vegetation_class = np.zeros_like(
                    target_tree_class,
                    dtype=bool
                )

            # GLAD has no-data values for small island and polar regions.
            glad_no_data = mask_obj.get_mask(
                mask_label=[255],
                block_param=block_param)
            logger.info(f'GLAD has {np.sum(glad_no_data)} no data')

            target_replace_class = sup_mask_obj.get_mask(
                mask_label=target_worldcover_class,
                block_param=block_param)

            target_inundated_vege_class = np.array(
                target_inundated_vege_class,
                dtype='int8'
            )

            target_inundated_vege_class[
                glad_no_data & target_replace_class] = 2

            # For GLAD no-data fallback, assign fallback vegetation to tree
            # to preserve the original behavior and avoid applying DEM method
            # where GLAD short vegetation is unknown.
            target_tree_class = np.asarray(target_tree_class).astype(bool)
            target_short_vegetation_class = np.asarray(
                target_short_vegetation_class
            ).astype(bool)

            target_tree_class[glad_no_data & target_replace_class] = True
            target_short_vegetation_class[
                glad_no_data & target_replace_class
            ] = False

        no_data = np.isnan(filt_ratio)
        target_inundated_vege_class[no_data] = 0
        target_tree_class[no_data] = False
        target_short_vegetation_class[no_data] = False

        tree_inundated_cand = (
            (filt_ratio_db > tree_ratio_threshold) &
            (cross_db > tree_cross_pol_min) &
            (co_db > tree_copol_threshold) &
            target_tree_class
        )

        short_vegetation_inundated_cand = (
            (filt_ratio_db > short_veg_ratio_threshold) &
            (cross_db > short_veg_cross_pol_min) &
            (co_db > short_veg_copol_threshold) &
            target_short_vegetation_class
        )


        all_inundated_cand = (
            tree_inundated_cand |
            short_vegetation_inundated_cand
        )

        inundated_vegetation = (
            all_inundated_cand &
            (target_inundated_vege_class > 0)
        )
        output_data[inundated_vegetation] = 2

        _dswx_sar_util.write_raster_block(
            out_raster=inundated_vege_path,
            data=output_data,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='byte',
            cog_flag=True,
            scratch_dir=scratch_dir)

        _dswx_sar_util.write_raster_block(
            out_raster=target_area_path,
            data=target_inundated_vege_class,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='byte',
            cog_flag=True,
            scratch_dir=scratch_dir)

        _dswx_sar_util.write_raster_block(
            out_raster=high_ratio_path,
            data=all_inundated_cand,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='byte',
            cog_flag=True,
            scratch_dir=scratch_dir)
        _dswx_sar_util.write_raster_block(
            out_raster=tree_area_path,
            data=target_tree_class.astype(np.uint8),
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='byte',
            cog_flag=True,
            scratch_dir=scratch_dir)

        _dswx_sar_util.write_raster_block(
            out_raster=short_vegetation_area_path,
            data=target_short_vegetation_class.astype(np.uint8),
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='byte',
            cog_flag=True,
            scratch_dir=scratch_dir)
        if processing_cfg.debug_mode:

            _dswx_sar_util.write_raster_block(
                out_raster=os.path.join(
                    scratch_dir, f'intensity_db_ratio_{pol_all_str}.tif'),
                data=filt_ratio_db,
                block_param=block_param,
                geotransform=im_meta['geotransform'],
                projection=im_meta['projection'],
                datatype='float32',
                cog_flag=True,
                scratch_dir=scratch_dir)
    if fuzzy_logic_enabled:

        _refine_inundated_vegetation_with_fuzzy_region_growing(
            cfg=cfg,
            pol_all_str=pol_all_str,
            ratio_db_path=ratio_db_path,
            cross_db_path=cross_db_path,
            inundated_vege_path=inundated_vege_path,
            target_area_path=target_area_path,
            tree_area_path=tree_area_path,
            short_vegetation_area_path=short_vegetation_area_path,
            im_meta=im_meta
        )
    t_time_end = time.time()

    logger.info(
        f'total inundated vegetation mapping time: {t_time_end - t_all} sec')


def main():

    parser = _get_parser()

    args = parser.parse_args()

    _generate_log.configure_log_file(args.log_file)

    mimetypes.add_type("text/yaml", ".yaml", strict=True)
    flag_first_file_is_text = 'text' in mimetypes.guess_type(
        args.input_yaml[0])[0]

    if len(args.input_yaml) > 1 and flag_first_file_is_text:
        logger.info('ERROR only one runconfig file is allowed')
        return

    if flag_first_file_is_text:
        cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_ni', args)

    processing_cfg = cfg.groups.processing
    pol_mode = processing_cfg.polarization_mode
    pol_list = processing_cfg.polarizations
    if pol_mode == 'MIX_DUAL_POL':
        proc_pol_set = [DSWX_NI_POL_DICT['DV_POL'],
                        DSWX_NI_POL_DICT['DH_POL']]
    elif pol_mode == 'MIX_SINGLE_POL':
        proc_pol_set = [DSWX_NI_POL_DICT['SV_POL'],
                        DSWX_NI_POL_DICT['SH_POL']]
    else:
        proc_pol_set = [pol_list]

    for pol_set in proc_pol_set:
        processing_cfg.polarizations = pol_set
        run(cfg)


if __name__ == '__main__':
    main()
