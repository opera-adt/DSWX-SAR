import os
import logging
import time

from collections.abc import Iterator
from dataclasses import dataclass
import geopandas as gpd
import mimetypes
from typing import Optional, Tuple

from dswx_sar.common.gcov_reader import RTCReader
from dswx_sar.nisar.dswx_ni_runconfig import (
    _get_parser,
    RunConfig)

logger = logging.getLogger('dswx_sar')

@dataclass(frozen=True)
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def expand(self, margin: float) -> "BBox":
        if margin is None or margin == 0:
            return self
        return BBox(
            xmin=self.xmin - margin,
            ymin=self.ymin - margin,
            xmax=self.xmax + margin,
            ymax=self.ymax + margin,
        )


def _read_mgrs_set_bbox_and_epsg(
    mgrs_collection_db: str,
    mgrs_collection_id: str,
    layer: str = "mgrs_track_frame_db",
    id_col: str = "mgrs_set_id",
    xmin_col: str = "xmin",
    ymin_col: str = "ymin",
    xmax_col: str = "xmax",
    ymax_col: str = "ymax",
    epsg_col: str = "epsg",
) -> Tuple[BBox, Optional[int]]:
    gdf = gpd.read_file(mgrs_collection_db, layer=layer)

    if id_col not in gdf.columns:
        raise KeyError(f"Column '{id_col}' not found in layer '{layer}'")

    filtered = gdf[gdf[id_col] == mgrs_collection_id]
    if filtered.empty:
        raise ValueError(
            f"No rows found for {id_col}='{mgrs_collection_id}' in layer '{layer}'"
        )
    if len(filtered) > 1:
        # If duplicates exist, you can decide how to handle them.
        # Here we take the first row deterministically.
        filtered = filtered.iloc[[0]]

    row = filtered.iloc[0]

    for c in (xmin_col, ymin_col, xmax_col, ymax_col):
        if c not in gdf.columns:
            raise KeyError(f"Column '{c}' not found in layer '{layer}'")

    bbox = BBox(
        xmin=float(row[xmin_col]),
        ymin=float(row[ymin_col]),
        xmax=float(row[xmax_col]),
        ymax=float(row[ymax_col]),
    )

    epsg_val = None
    if epsg_col in gdf.columns:
        v = row[epsg_col]
        epsg_val = None if v is None else int(v)

    return bbox, epsg_val


def run(cfg):
    """Generate mosaic workflow with user-defined args stored
    in dictionary runconfig 'cfg'

    Parameters:
    -----------
    cfg: RunConfig
        RunConfig object with user runconfig options
    """
    logger.info('Starting DSWx-NI mosaic GCOV frames')
    t_all = time.time()
    # Mosaicking parameters
    processing_cfg = cfg.groups.processing

    input_list = cfg.groups.input_file_group.input_file_path

    mosaic_cfg = processing_cfg.mosaic
    mosaic_mode = mosaic_cfg.mosaic_mode
    mosaic_prefix = mosaic_cfg.mosaic_prefix
    mosaic_posting_thresh = mosaic_cfg.mosaic_posting_thresh
    gdal_cache_max_mb = mosaic_cfg.gdal_cache_max_mb
    nisar_uni_mode = processing_cfg.nisar_uni_mode
    # input margin is km.
    mosaic_margin = mosaic_cfg.mosaic_margin * 1000

    nisar_uni_mode = processing_cfg.nisar_uni_mode
    resamp_required = mosaic_cfg.resamp_required
    # Determine if resampling is required
    if not nisar_uni_mode:
        resamp_required = True

    resamp_method = mosaic_cfg.resamp_method
    resamp_out_res = mosaic_cfg.resamp_out_res

    mgrs_collection_db = \
        cfg.groups.static_ancillary_file_group.mgrs_collection_database_file
    mgrs_collection_id = cfg.groups.input_file_group.input_mgrs_collection_id
    mgrs_bbox = None
    mgrs_epsg = None

    if mgrs_collection_db is not None and mgrs_collection_id is not None:
        bbox, epsg_val = _read_mgrs_set_bbox_and_epsg(
            mgrs_collection_db=mgrs_collection_db,
            mgrs_collection_id=mgrs_collection_id,
            layer="mgrs_track_frame_db",
            id_col="mgrs_set_id",
            xmin_col="xmin",
            ymin_col="ymin",
            xmax_col="xmax",
            ymax_col="ymax",
            epsg_col="EPSG",  # change if your column name differs
        )
        bbox = bbox.expand(mosaic_margin)
        mgrs_bbox = (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
        mgrs_epsg = epsg_val

        logger.info(f"MGRS set id: {mgrs_collection_id}")
        logger.info(f"MGRS bbox (expanded): {mgrs_bbox}")
        logger.info(f"MGRS epsg: {mgrs_epsg}")

    logger.info(f"Resampling : {resamp_required}")
    logger.info(f"Resampling : {resamp_method}")
    logger.info(f"Resampling : {resamp_out_res}")
    scratch_dir = cfg.groups.product_path_group.scratch_path
    os.makedirs(scratch_dir, exist_ok=True)

    row_blk_size = mosaic_cfg.read_row_blk_size
    col_blk_size = mosaic_cfg.read_col_blk_size

    # Create reader object
    reader = RTCReader(
        row_blk_size=row_blk_size,
        col_blk_size=col_blk_size,
    )

    # Mosaic input RTC into output Geotiff
    reader.process_rtc_hdf5(
        input_list,
        scratch_dir,
        mosaic_mode,
        mosaic_prefix,
        mosaic_posting_thresh,
        gdal_cache_max_mb,
        resamp_method,
        resamp_out_res,
        resamp_required,
        bbox=mgrs_bbox,      # (xmin, ymin, xmax, ymax) or None
        bbox_epsg=mgrs_epsg, # int or None
    )
    t_all_elapsed = time.time() - t_all
    logger.info("successfully ran mosaic GCOV in "
                f"{t_all_elapsed:.3f} seconds")

if __name__ == "__main__":
    '''Run mosaic rtc products from command line'''
    # load arguments from command line
    parser = _get_parser()

    # parse arguments
    args = parser.parse_args()

    mimetypes.add_type("text/yaml", ".yaml", strict=True)

    cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_ni', args)

    # Run Mosaic RTC workflow
    run(cfg)
