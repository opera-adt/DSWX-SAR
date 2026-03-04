import os
import logging
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any
import math
import mimetypes

import h5py
from h5py import Dataset
import numpy as np
from osgeo import gdal, osr
from pyproj import Transformer
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio import Env

from dswx_sar.common._mosaic import mosaic_single_output_file
from dswx_sar.common._dswx_sar_util import (
    _aggregate_10m_to_30m_conv,
    _aggregate_10m_to_30m_fast,
    _calculate_output_bounds,
    change_epsg_tif,
    majority_element)
from dswx_sar.common._dswx_geogrid import DSWXGeogrid
from dswx_sar.nisar.dswx_ni_runconfig import (
    _get_parser,
    RunConfig,
    get_pol_rtc_hdf5,
    check_rtc_frequency, get_rtc_spacing_list)

from dswx_sar.common.read_h5_s3 import (
    H5Reader, slice_gen, is_s3_path, file_exists
)
import psutil
logger = logging.getLogger('dswx_sar')


proc = psutil.Process(os.getpid())
def rss_mib():
    return proc.memory_info().rss / (1024**2)

def hwm_mib():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB->MiB on Linux
    except Exception:
        return float("nan")


def multi_look_average_gdal(
    input_geotiff: str,
    scratch_dir: str,
    output_res: float,
    geogrid_in
):
    """
    Fast multi-look using gdal.Warp average resampling.
    Supports inputs at 10m or 20m and produces output at output_res (e.g. 30).
    Replaces the previous memory-heavy upsample + numpy aggregation.
    """

    # --- global GDAL tuning (set once per process ideally) ---
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    # cache in MB (tune to available RAM)
    gdal.SetConfigOption("GDAL_CACHEMAX", "2048")

    ds_input = gdal.Open(input_geotiff, gdal.GA_ReadOnly)
    if ds_input is None:
        raise FileNotFoundError(f"Cannot open {input_geotiff}")

    gt = ds_input.GetGeoTransform()
    input_res_x = abs(gt[1])
    input_res_y = abs(gt[5])  # typically negative; abs for magnitude

    if input_res_x != input_res_y:
        ds_input = None
        raise ValueError("Input x/y resolutions mismatch: must be same")

    if int(input_res_x) not in (10, 20):
        ds_input = None
        raise ValueError(f"Input resolution {input_res_x} not supported (expect 10 or 20)")

    # Output filename
    full_path = Path(input_geotiff)
    output_geotiff = f'{full_path.parent}/{full_path.stem}_multi_look.tif'

    # Determine srcNodata if present (prefer dataset value)
    band = ds_input.GetRasterBand(1)
    src_nodata = band.GetNoDataValue()
    # If the dataset uses some designated_value sentinel (e.g. 500), you can set it explicitly:
    # src_nodata = src_nodata if src_nodata is not None else 500

    # WarpOptions: average resampling to output_res
    warp_opts = gdal.WarpOptions(
        format='GTiff',
        xRes=float(output_res),
        yRes=float(output_res),
        resampleAlg='average',
        multithread=True,
        warpMemoryLimit=2048,            # MB, tune as necessary
        srcNodata=src_nodata,
        dstNodata=src_nodata,
        targetAlignedPixels=True,
        creationOptions=[
            "TILED=YES",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
            "COMPRESS=DEFLATE",     # you may skip compression for scratch for speed
            "PREDICTOR=2",
            "BIGTIFF=IF_SAFER",
        ],
    )

    # Run warp (this writes directly to disk, streaming & multi-threaded)
    ds_out = gdal.Warp(output_geotiff, ds_input, options=warp_opts)
    if ds_out is None:
        ds_input = None
        raise RuntimeError("gdal.Warp failed")

    # Update the geogrid and replace original file
    geogrid_in.update_geogrid(output_geotiff)
    ds_out = None
    ds_input = None
    os.replace(output_geotiff, input_geotiff)


def reproject_bbox(bbox, src_epsg: int, dst_epsg: int):
    xmin, ymin, xmax, ymax = bbox
    tf = Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)
    x1, y1 = tf.transform(xmin, ymin)
    x2, y2 = tf.transform(xmax, ymax)
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


class DataReader(ABC):
    def __init__(self, row_blk_size: int, col_blk_size: int,
                 s3_profile: str | None = None, aws_region: str | None = None):
        self.row_blk_size = row_blk_size
        self.col_blk_size = col_blk_size
        self.s3_profile = s3_profile
        self.aws_region = aws_region

    @abstractmethod
    def process_rtc_hdf5(self, input_list: list) -> Any:
        pass

class RTCReader(DataReader):
    def __init__(self, row_blk_size: int, col_blk_size: int,
                 s3_profile: str | None = None, aws_region: str | None = None):

        super().__init__(row_blk_size, col_blk_size, s3_profile, aws_region)

    def process_rtc_hdf5(
            self,
            input_list: list,
            scratch_dir: str,
            mosaic_mode: str,
            mosaic_prefix: str,
            mosaic_posting_thresh: float,
            gdal_cache_max_mb: int,
            resamp_method: str,
            resamp_out_res: float,
            resamp_required: bool,
            bbox=None,
            bbox_epsg=None
    ):
        """Read data from input HDF5s in blocks and generate mosaicked output
           Geotiff

        Parameters
        ----------
        input_list: list
            The HDF5 file paths of input RTCs to be mosaicked.
        scratch_dir: str
            Directory which stores the temporary files
        mosaic_mode: str
            Mosaic algorithm mode choice in 'average', 'first',
            or 'burst_center'
        mosaic_prefix: str
            Mosaicked output file name prefix
        mosaic_posting_thresh: float
            Minimu RTC posting needed for processing
        resamp_required: bool
            Indicates whether resampling (downsampling) needs to be performed
            on input RTC product in Geotiff.
        resamp_method: str
            Set GDAL.Warp() resampling algorithms based on its built-in options
            Default = 'nearest'
        resamp_out_res: float
            User define output resolution for resampled Geotiff
        """
        # Extract Relevant information of input files

        # Read frequency group(s) from each input file
        flag_freq_equal, freq_list = check_rtc_frequency(input_list)

        num_input_files = len(input_list)

        # Extract resolution information
        res_list, res_highest = get_rtc_spacing_list(input_list, freq_list)

        # Generate valid data paths
        data_path = self.generate_nisar_dataset_name(input_list, freq_list)
        new_input_list = []

        # Remove dataset from processing based on minimum image resolution/posting
        for input_idx, input_rtc in enumerate(input_list):
            valid_freqs = []

            for freq_idx, freq_group in enumerate(freq_list[input_idx]):
                if res_list[input_idx][freq_idx] is not None and \
                    res_list[input_idx][freq_idx] <= mosaic_posting_thresh:
                    valid_freqs.append(freq_group)
            if valid_freqs:
                data_path[input_rtc] = {f: data_path[input_rtc][f] for f in valid_freqs}
                new_input_list.append(input_rtc)
            else:
                data_path.pop(input_rtc, None)
                # if freq_group == 'B':
                #     del data_path[input_rtc][freq_group]

        input_list = new_input_list
        logger.info('data path for mosaic')
        logger.info(data_path)
        # Generate layover mask path
        layover_mask_name = 'mask'
        layover_path = str(self.generate_nisar_layover_name(layover_mask_name))

        mask_path = self.generate_nisar_mask_name()
        epsg_array, epsg_same_flag = self.get_nisar_epsg(input_list)

        # Write all RTC HDF5 inputs to intermeidate Geotiff first and re-use
        # existing functions to reproject data and create mosaicked output
        # from intermediate Geotiffs
        (
            geogrid_in,
            input_gtiff_list,
            mask_gtiff_list,
            pol_gtiff_list) = self.write_rtc_geotiff(
                input_list,
                scratch_dir,
                epsg_array,
                data_path,
                mask_path,
                gdal_cache_max_mb,
                bbox=bbox,
                bbox_epsg=bbox_epsg,
            )

        mask_exist = (len(mask_gtiff_list) > 0)
        # To Do: Use flag_mosaic_freq_a and flag_mosaic_freq_b flags to
        # filter frequency A or B from processing

        # Choose Resampling methods
        if resamp_required:
            resampled_geogrid_in = DSWXGeogrid()
            # Apply multi-look technique
            if resamp_method == 'multilook':
                if len(input_gtiff_list) > 0:
                    for idx, input_geotiff in enumerate(input_gtiff_list):
                        multi_look_average_gdal(
                            input_geotiff,
                            scratch_dir,
                            resamp_out_res,
                            resampled_geogrid_in,
                        )

            else:
                # Apply resampling using GDAL.Warp() based on
                # resampling methods
                if len(input_gtiff_list) > 0:
                    for idx, input_geotiff in enumerate(input_gtiff_list):
                        self.resample_rtc(
                            input_geotiff,
                            scratch_dir,
                            resamp_out_res,
                            resampled_geogrid_in,
                            resamp_method,
                            )
            if mask_exist:
                for idx, layover_geotiff in enumerate(mask_gtiff_list):
                    self.resample_rtc(
                        layover_geotiff,
                        scratch_dir,
                        resamp_out_res,
                        resampled_geogrid_in,
                        'nearest',
                        )

            if bbox is not None:
                # bbox is (xmin, ymin, xmax, ymax) in bbox_epsg
                if bbox_epsg is None:
                    raise ValueError("bbox was provided but bbox_epsg is None")

                target_epsg = int(getattr(geogrid_in, "epsg", None))

                # Reproject bbox to target_epsg if needed
                if int(bbox_epsg) != target_epsg:
                    bbox_use = reproject_bbox(bbox, int(bbox_epsg), target_epsg)
                else:
                    bbox_use = bbox
                # Ensure geogrid EPSG matches target_epsg
                ge_epsg = getattr(resampled_geogrid_in, "epsg", None)
                if ge_epsg is None or int(ge_epsg) != target_epsg:
                    resampled_geogrid_in.epsg = target_epsg
                # Now bbox_use is in target_epsg, so pass bbox_epsg=target_epsg
                resampled_geogrid_in.clip_to_bbox(bbox_use, bbox_epsg=target_epsg, snap=True)

                bx = resampled_geogrid_in.bounds()
                print(f"[BBox clip] final geogrid bounds: {bx}, epsg={geogrid_in.epsg}")
            geogrid_in = resampled_geogrid_in

        # Mosaic intermediate geotiffs
        nlooks_list = []
        self.mosaic_rtc_geotiff(
            input_list,
            pol_gtiff_list,
            scratch_dir,
            geogrid_in,
            nlooks_list,
            mosaic_mode,
            mosaic_prefix,
            mask_exist,
        )


    def write_rtc_geotiff(
        self,
        input_list: list,
        scratch_dir: str,
        epsg_array: np.ndarray,
        data_path: dict,
        mask_path: str,
        gdal_cache_max_mb: int,
        bbox=None,
        bbox_epsg=None,
    ):
        """
        Create intermediate GeoTIFFs from input RTCs.

        NEW: If bbox is provided, only reads/writes the intersection window for each RTC.
            RTCs/datasets with no overlap are skipped.
        """
        # Majority EPSG for harmonization
        most_freq_epsg = majority_element(epsg_array)
        designated_value = np.float32(500)

        output_gtiff_list: list[str] = []
        mask_gtiff_list: list[str] = []
        pol_gtiff_list: dict[str, list[str]] = {}

        # Small helper to compute crop window in source pixel coords
        def bbox_to_window_for_ds(
            bbox_in,
            bbox_epsg_in: int,
            ds_epsg_in: int,
            transform_in: Affine,
            width_in: int,
            height_in: int,
        ):
            if bbox_in is None:
                return (0, height_in, 0, width_in)

            if bbox_epsg_in is None:
                raise ValueError("bbox was provided but bbox_epsg is None")

            # Reproject bbox to dataset EPSG if needed
            if int(bbox_epsg_in) != int(ds_epsg_in):
                bbox_ds = reproject_bbox(bbox_in, int(bbox_epsg_in), int(ds_epsg_in))
            else:
                bbox_ds = bbox_in

            xmin, ymin, xmax, ymax = bbox_ds
            xmin, xmax = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
            ymin, ymax = (ymin, ymax) if ymin <= ymax else (ymax, ymin)

            # Dataset bounds in map coords
            xs = [transform_in * (0, 0), transform_in * (width_in, 0),
                transform_in * (0, height_in), transform_in * (width_in, height_in)]
            xvals = [p[0] for p in xs]
            yvals = [p[1] for p in xs]
            ds_xmin, ds_xmax = min(xvals), max(xvals)
            ds_ymin, ds_ymax = min(yvals), max(yvals)

            # Intersection
            ixmin = max(xmin, ds_xmin)
            ixmax = min(xmax, ds_xmax)
            iymin = max(ymin, ds_ymin)
            iymax = min(ymax, ds_ymax)
            if ixmin >= ixmax or iymin >= iymax:
                return None

            inv = ~transform_in
            c0, r0 = inv * (ixmin, iymin)
            c1, r1 = inv * (ixmax, iymax)

            col_min = int(np.floor(min(c0, c1)))
            col_max = int(np.ceil (max(c0, c1)))
            row_min = int(np.floor(min(r0, r1)))
            row_max = int(np.ceil (max(r0, r1)))

            # Clamp
            col_min = max(0, min(width_in,  col_min))
            col_max = max(0, min(width_in,  col_max))
            row_min = max(0, min(height_in, row_min))
            row_max = max(0, min(height_in, row_max))

            if col_min >= col_max or row_min >= row_max:
                return None

            return (row_min, row_max, col_min, col_max)

        # ---- Write science layers (cropped if bbox provided) ----
        for input_idx, input_rtc in enumerate(input_list):
            output_prefix = self.extract_file_name(input_rtc)

            # Geo info + metadata
            geotransform, crs = self.read_geodata_hdf5(input_rtc)
            dswx_metadata_dict = self.read_metadata_hdf5(input_rtc)

            ds_epsg = int(epsg_array[input_idx])

            data_path_input = data_path[input_rtc]
            freq_groups = list(data_path_input.keys())

            for freq_group in freq_groups:
                for dataset_path in data_path_input[freq_group]:
                    data_name = Path(dataset_path).name[:2]
                    pol = dataset_path.split("/")[-1][:2]

                    output_gtiff = f"{scratch_dir}/{output_prefix}_{data_name}_{freq_group}.tif"

                    with H5Reader(input_rtc) as f:
                        if dataset_path not in f:
                            raise RuntimeError(f"Dataset not found in HDF5: {dataset_path}")
                        dset = f[dataset_path]
                        if dset.ndim != 2:
                            raise RuntimeError(f"Expected 2D dataset, got shape {dset.shape} at {dataset_path}")
                        num_rows, num_cols = dset.shape

                        win = bbox_to_window_for_ds(
                            bbox, bbox_epsg, ds_epsg, geotransform, num_cols, num_rows
                        )
                        if win is None:
                            logger.info(f"[BBox] skip {output_prefix} {data_name} {freq_group}: no overlap")
                            continue

                        r0, r1, c0, c1 = win
                        out_rows = r1 - r0
                        out_cols = c1 - c0

                        # Write cropped GeoTIFF
                        self.read_write_rtc_h5py(
                            dset,
                            output_gtiff,
                            num_rows,
                            num_cols,
                            self.row_blk_size,
                            self.col_blk_size,
                            gdal_cache_max_mb,
                            designated_value,
                            geotransform,
                            crs,
                            dswx_metadata_dict,
                            is_mask=False,
                            out_dtype="float32",
                            row0=r0,
                            col0=c0,
                            out_rows=out_rows,
                            out_cols=out_cols,
                            align_to_chunks=True,
                            K_chunks=8,
                        )

                    output_gtiff_list.append(output_gtiff)
                    pol_gtiff_list.setdefault(pol, []).append(output_gtiff)

        # ---- EPSG harmonization + geogrid union ----
        geogrid_in = DSWXGeogrid()

        # Harmonize EPSG if needed, update geogrid
        for input_idx, input_rtc in enumerate(input_list):
            input_prefix = self.extract_file_name(input_rtc)
            ds_epsg = int(epsg_array[input_idx])

            data_path_input = data_path[input_rtc]
            freq_groups = list(data_path_input.keys())

            for freq_group in freq_groups:
                for dataset_path in data_path_input[freq_group]:
                    data_name = Path(dataset_path).name[:2]
                    input_gtiff = f"{scratch_dir}/{input_prefix}_{data_name}_{freq_group}.tif"
                    if not os.path.exists(input_gtiff):
                        continue  # could have been skipped due to bbox no-overlap

                    if ds_epsg != most_freq_epsg:
                        temp_gtiff = f"{scratch_dir}/{input_prefix}_temp_{data_name}_{freq_group}.tif"
                        change_epsg_tif(
                            input_tif=input_gtiff,
                            output_tif=temp_gtiff,
                            epsg_output=most_freq_epsg,
                            output_nodata=255,
                        )
                        geogrid_in.update_geogrid(temp_gtiff)
                        os.replace(temp_gtiff, input_gtiff)
                    else:
                        geogrid_in.update_geogrid(input_gtiff)

        # ---- Layover/Shadow mask (cropped if bbox provided) ----
        for input_idx, input_rtc in enumerate(input_list):
            output_prefix = self.extract_file_name(input_rtc)
            ds_epsg = int(epsg_array[input_idx])

            mask_ds_path = f"/{mask_path}" if not str(mask_path).startswith("/") else str(mask_path)

            # Open once to get shape and window
            with H5Reader(input_rtc) as f:
                if mask_ds_path not in f:
                    warnings.warn(f"\nDataset at {mask_ds_path} does not exist or cannot be opened.", RuntimeWarning)
                    continue
                dset = f[mask_ds_path]
                if dset.ndim != 2:
                    warnings.warn(f"\nMask dataset not 2D: {mask_ds_path}", RuntimeWarning)
                    continue
                num_rows, num_cols = dset.shape

                geotransform, crs = self.read_geodata_hdf5(input_rtc)
                dswx_metadata_dict = self.read_metadata_hdf5(input_rtc)

                win = bbox_to_window_for_ds(
                    bbox, bbox_epsg, ds_epsg, geotransform, num_cols, num_rows
                )
                if win is None:
                    logger.info(f"[BBox] skip {output_prefix} mask: no overlap")
                    continue

                r0, r1, c0, c1 = win
                out_rows = r1 - r0
                out_cols = c1 - c0

                output_mask_gtiff = f"{scratch_dir}/{output_prefix}_mask.tif"

                self.read_write_rtc_h5py(
                    dset,
                    output_mask_gtiff,
                    num_rows,
                    num_cols,
                    self.row_blk_size,
                    self.col_blk_size,
                    gdal_cache_max_mb,
                    np.float32(500),
                    geotransform,
                    crs,
                    dswx_metadata_dict,
                    is_mask=True,
                    out_dtype="uint8",
                    row0=r0,
                    col0=c0,
                    out_rows=out_rows,
                    out_cols=out_cols,
                    align_to_chunks=True,
                    K_chunks=8,
                )

            # EPSG harmonization for mask
            if ds_epsg != most_freq_epsg:
                tmp = f"{scratch_dir}/{output_prefix}_mask_tmp.tif"
                change_epsg_tif(
                    input_tif=output_mask_gtiff,
                    output_tif=tmp,
                    epsg_output=most_freq_epsg,
                    output_nodata=255,
                )
                os.replace(tmp, output_mask_gtiff)

            mask_gtiff_list.append(output_mask_gtiff)
            geogrid_in.update_geogrid(output_mask_gtiff)

        # ---- Optional: final clip of union geogrid to bbox (kept from your original) ----
        if bbox is not None:
            if bbox_epsg is None:
                raise ValueError("bbox was provided but bbox_epsg is None")

            target_epsg = int(getattr(geogrid_in, "epsg", most_freq_epsg))
            if int(bbox_epsg) != target_epsg:
                bbox_use = reproject_bbox(bbox, int(bbox_epsg), target_epsg)
            else:
                bbox_use = bbox

            ge_epsg = getattr(geogrid_in, "epsg", None)
            if ge_epsg is None or int(ge_epsg) != target_epsg:
                geogrid_in.epsg = target_epsg

            geogrid_in.clip_to_bbox(bbox_use, bbox_epsg=target_epsg, snap=True)
            logger.info(f"[BBox clip] final geogrid bounds: {geogrid_in.bounds()}, epsg={geogrid_in.epsg}")

        return geogrid_in, output_gtiff_list, mask_gtiff_list, pol_gtiff_list


    def read_write_rtc_h5py(
            self,
            h5_dset: h5py.Dataset,
            output_gtiff: str,
            num_rows: int,
            num_cols: int,
            row_blk_size: int,
            col_blk_size: int,
            gdal_cache_max_mb,
            designated_value: np.float32,
            geotransform: Affine,
            crs: str,
            dswx_metadata_dict: dict,
            *,
            is_mask: bool = False,
            out_dtype: str | None = None,
            # NEW: crop window in source pixel coordinates
            row0: int = 0,
            col0: int = 0,
            out_rows: int | None = None,
            out_cols: int | None = None,
            # NEW: optionally align row reads to HDF5 chunk rows for speed
            align_to_chunks: bool = True,
            K_chunks: int = 8,
        ):
            """
            Stream a 2D h5py dataset in blocks and write GeoTIFF via rasterio.

            Supports optional cropping: reads source window [row0:row0+out_rows, col0:col0+out_cols]
            and writes to an output GeoTIFF whose transform is shifted accordingly.

            h5_dset: h5py.Dataset with shape (rows, cols)
            """
            if out_dtype is None:
                out_dtype = "uint8" if is_mask else "float32"
            dst_dtype = np.uint8 if out_dtype == "uint8" else np.float32

            # Output shape defaults to full raster
            if out_rows is None:
                out_rows = num_rows - row0
            if out_cols is None:
                out_cols = num_cols - col0

            # Sanity + clamp
            if row0 < 0 or col0 < 0:
                raise ValueError(f"row0/col0 must be >=0, got row0={row0}, col0={col0}")
            if row0 + out_rows > num_rows or col0 + out_cols > num_cols:
                raise ValueError(
                    f"Crop window out of bounds: "
                    f"src ({num_rows},{num_cols}), "
                    f"window row[{row0}:{row0+out_rows}] col[{col0}:{col0+out_cols}]"
                )
            if out_rows <= 0 or out_cols <= 0:
                raise ValueError(f"Invalid output size out_rows={out_rows}, out_cols={out_cols}")

            # Shift transform so pixel (0,0) corresponds to original (col0,row0)
            out_transform = geotransform * Affine.translation(col0, row0)

            profile = {
                "driver": "GTiff",
                "height": out_rows,
                "width": out_cols,
                "count": 1,
                "dtype": out_dtype,
                "crs": crs,
                "transform": out_transform,
                "compress": "DEFLATE",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                'interleave': 'band',
            }
            if is_mask:
                profile["nodata"] = 255

            # Helpers for aligned row reading
            def aligned_ranges(start_row: int, stop_row: int, blk: int, align: int):
                r = start_row
                # align the first read to chunk boundary if requested
                if align > 1 and (r % align) != 0:
                    a = r + (align - (r % align))
                    yield slice(r, min(a, stop_row))
                    r = a
                while r < stop_row:
                    yield slice(r, min(r + blk, stop_row))
                    r += blk

            # Decide row stripe height
            chunk = h5_dset.chunks or (512, 512)
            chunk_rows = int(chunk[0]) if chunk and len(chunk) >= 2 else 512

            if align_to_chunks and chunk_rows > 0:
                row_blk_size_eff = int(K_chunks) * chunk_rows
                align = chunk_rows
            else:
                row_blk_size_eff = int(row_blk_size)
                align = 1

            # We write full-width stripes of the cropped window
            col_blk_size_eff = out_cols

            src_r0 = row0
            src_r1 = row0 + out_rows
            with rasterio.Env(GDAL_CACHEMAX=gdal_cache_max_mb):

                with rasterio.open(output_gtiff, "w", **profile) as dst:
                    for s in aligned_ranges(src_r0, src_r1, row_blk_size_eff, align):
                        rs, re = int(s.start), int(s.stop)

                        # Read from HDF5 window (source coords)
                        ds_blk = h5_dset[rs:re, col0 : col0 + out_cols]

                        # Convert / clean
                        if is_mask:
                            blk = ds_blk.astype(np.int32, copy=False)
                            blk = np.where(np.isfinite(blk), blk, 255).astype(dst_dtype, copy=False)
                        else:
                            blk = ds_blk.astype(np.float32, copy=False)
                            blk[np.isinf(blk)] = designated_value
                            blk[blk > designated_value] = designated_value
                            blk[blk == 0] = np.nan

                        # Write into output at row offset (destination coords start at 0)
                        dst_rs = rs - src_r0
                        dst.write(blk, 1, window=Window(0, dst_rs, out_cols, re - rs))

                    dst.update_tags(**dswx_metadata_dict)


    def mosaic_rtc_geotiff(
        self,
        input_list: list,
        pol_list: dict,
        scratch_dir: str,
        geogrid_in: DSWXGeogrid,
        nlooks_list: list,
        mosaic_mode: str,
        mosaic_prefix: str,
        mask_exist: bool,
    ):
        """ Create mosaicked output Geotiff from a list of input RTCs

        Parameters
        ----------
        input_list: list
            The HDF5 file paths of input RTCs to be mosaicked.
        pol_list: dictionary
            polarization dictionary with Geotiff paths
        scratch_dir: str
            Directory which stores the temporary files
        geogrid_in: DSWXGeogrid object
            A dataclass object representing the geographical grid
            configuration for an RTC (Radar Terrain Correction) run.
        nlooks_list: list
            List of the nlooks raster that corresponds to list_rtc
        mosaic_mode: str
            Mosaic algorithm mode choice in 'average', 'first',
            or 'burst_center'
        mosaic_prefix: str
            Mosaicked output file name prefix
        mask_exist: bool
            Boolean which indicates if a mask layer
            exists in input RTC
        """
        for pol in pol_list:
            input_gtiff_list = pol_list[pol]

            # Mosaic dataset of same polarization into a single Geotiff
            output_mosaic_gtiff = \
                f'{scratch_dir}/{mosaic_prefix}_{pol}.tif'
            mosaic_single_output_file(
                input_gtiff_list,
                nlooks_list,
                output_mosaic_gtiff,
                mosaic_mode,
                self.row_blk_size,
                self.col_blk_size,
                scratch_dir=scratch_dir,
                geogrid_in=geogrid_in,
                temp_files_list=None,
                no_data_value=255,
            )

        # Mosaic layover shadow mask
        if mask_exist:
            mask_inputs: list[str] = []
            for input_rtc in input_list:
                input_prefix = self.extract_file_name(input_rtc)
                mask_inputs.append(f'{scratch_dir}/{input_prefix}_mask.tif')

            mask_mosaic_gtiff = f'{scratch_dir}/{mosaic_prefix}_mask.tif'

            mosaic_single_output_file(
                mask_inputs,
                nlooks_list,
                mask_mosaic_gtiff,
                mosaic_mode,
                self.row_blk_size,
                self.col_blk_size,
                scratch_dir=scratch_dir,
                geogrid_in=geogrid_in,
                temp_files_list=None,
                no_data_value=255,
            )

    def resample_rtc(
        self,
        input_geotiff: str,
        scratch_dir: str,
        output_res: float,
        geogrid_in: DSWXGeogrid,
        resamp_method: str = 'nearest',
    ):
        """Resample input geotfif from their native resolution to desired
        output resolution

        Parameters
        ----------
        input_geotiff: str
            Input geotiff path to be resampled.
        scratch_dir: str
            Directory which stores the temporary files
        output_res: float
            User define output resolution for resampled Geotiff
        geogrid_in: DSWXGeogrid object
            A dataclass object  representing the geographical grid
            configuration for an RTC (Radar Terrain Correction) run.
        resamp_method: str
            Set GDAL.Warp() resampling algorithms based on its built-in options
            Default = 'nearest
        """

        # Check if the file exists
        if not os.path.exists(input_geotiff):
            raise FileNotFoundError(f"The file '{input_geotiff}' does not exist.")

        full_path = Path(input_geotiff)
        output_geotiff = f'{full_path.parent}/{full_path.stem}_resamp.tif'

        ds_input = gdal.Open(input_geotiff)
        geotransform = ds_input.GetGeoTransform()

        # Set GDAL Warp options
        # Resampling method
        #gdal.GRA_Bilinear, gdal.GRA_NearestNeighbour, gdal.GRA_Cubic, gdal.GRA_Average, etc

        options = gdal.WarpOptions(
            xRes=output_res,
            yRes=output_res,
            resampleAlg=resamp_method)

        ds_output = gdal.Warp(output_geotiff, ds_input, options=options)

        # Update Geogrid in output Geotiff and replace the input Geotiff with it
        geogrid_in.update_geogrid(output_geotiff)
        os.replace(output_geotiff, input_geotiff)

        ds_input = None
        ds_output = None


    def multi_look_average(
        self,
        input_geotiff: str,
        scratch_dir: str,
        output_res: float,
        geogrid_in: DSWXGeogrid,
    ):
        """Apply upsampling and multi-look pixel averaging on input geotfif
        to obtain Geotiff with desired output resolution

        Parameters
        ----------
        input_geotiff: str
            Input geotiff path to be resampled.
        scratch_dir: str
            Directory which stores the temporary files
        output_res: float
            User define output resolution for multi-looked Geotiff
        geogrid_in: DSWXGeogrid object
            A dataclass object  representing the geographical grid
            configuration for an RTC (Radar Terrain Correction) run.
        """

        ds_input = gdal.Open(input_geotiff)
        geotransform_input = ds_input.GetGeoTransform()

        input_width = ds_input.RasterXSize
        input_length = ds_input.RasterYSize

        input_res_x = np.abs(geotransform_input[1])
        input_res_y = np.abs(geotransform_input[5])
        xsign = 1.0 if geotransform_input[1] >= 0 else -1.0
        ysign = 1.0 if geotransform_input[5] >= 0 else -1.0
        if input_res_x != input_res_y:
            raise ValueError(
                "x and y resolutions of the input must be the same."
            )

        full_path = Path(input_geotiff)
        output_geotiff = f'{full_path.parent}/{full_path.stem}_multi_look.tif'

        # Multi-look parameters
        input_res_x_i = int(round(input_res_x))
        output_res_i = int(round(output_res))
        if input_res_x_i == output_res_i:
            logger.info(f'multi-look is not necessary because output_spacing {output_res_i}'
                        f'is same as input_spacing {input_res_x_i}')
            return None

        interm_upsamp_res = math.gcd(input_res_x_i, output_res_i)
        output_res_i = int(round(output_res))

        downsamp_ratio = output_res_i // interm_upsamp_res  # ratio = 3

        normalized_flag = True
        if output_res_i % interm_upsamp_res != 0:
            raise ValueError(
                f"output_res ({output_res_i}) not divisible by "
                f"interm_upsamp_res ({interm_upsamp_res})"
            )
        if input_res_x == 20:
            # Perform upsampling to 10 meter resolution
            # Compute upsampled data output bounds
            upsamp_bounds = _calculate_output_bounds(
                geotransform_input,
                input_width,
                input_length,
                interm_upsamp_res,
            )

            # Perform GDAL.warp() in memory for upsampled data
            warp_options = gdal.WarpOptions(
                xRes=xsign * interm_upsamp_res,
                yRes=ysign * interm_upsamp_res,
                outputBounds=upsamp_bounds,
                resampleAlg='nearest',
                format='MEM'  # Use memory as the output format
            )

            ds_upsamp = gdal.Warp('', ds_input, options=warp_options)
            data_upsamp = ds_upsamp.GetRasterBand(1).ReadAsArray()
            geotransform_upsamp = ds_upsamp.GetGeoTransform()
            projection_upsamp = ds_upsamp.GetProjection()

            # Aggregate pixel values in a image to lower resolution to achieve
            # multi-looking effect
            multi_look_output = _aggregate_10m_to_30m_fast(
                data_upsamp,
                downsamp_ratio,
                normalized_flag,
            )

            # Write multi-look averaged data to output geotiff
            self.write_array_to_geotiff(
                ds_upsamp,
                multi_look_output,
                upsamp_bounds,
                output_res,
                output_geotiff,
            )
        elif input_res_x == 10:
            # Directly average 10m resolution input to 30m resolution output
            ds_array = ds_input.GetRasterBand(1).ReadAsArray()

            multi_look_output = _aggregate_10m_to_30m_fast(
                ds_array,
                downsamp_ratio,
                normalized_flag,
            )
            upsamp_bounds = None

            # Write to output geotiff
            self.write_array_to_geotiff(
                ds_input,
                multi_look_output,
                upsamp_bounds,
                output_res,
                output_geotiff,
            )
        else:
            raise ValueError("Input RTC are expected to have only 10m or 20m resolutions.")


        # Update Geogrid in output Geotiff and replace the input Geotiff with it
        geogrid_in.update_geogrid(output_geotiff)
        os.replace(output_geotiff, input_geotiff)


    def write_array_to_geotiff(
        self,
        ds_input,
        output_data: np.ndarray,
        output_bounds,   # [xmin, ymin, xmax, ymax] or None
        output_res: float,
        output_geotiff: str,
        *,
        nodata=None,
        compress="DEFLATE",
        predictor=2,
    ):
        """
        Write output_data to GeoTIFF with correct geotransform.
        Keeps the sign of x/y pixel sizes from ds_input.
        Assumes north-up (no rotation). If rotation exists, use gdal.Warp instead.
        """
        gt_in = ds_input.GetGeoTransform()
        rot_x, rot_y = gt_in[2], gt_in[4]
        if rot_x != 0 or rot_y != 0:
            raise ValueError("Input geotransform has rotation; use GDAL.Warp for correct georeferencing.")

        # Keep sign convention from input
        xsign = 1.0 if gt_in[1] >= 0 else -1.0
        ysign = 1.0 if gt_in[5] >= 0 else -1.0
        px = xsign * float(output_res)
        py = ysign * float(output_res)

        # Normalize output_data shape to (bands, rows, cols)
        arr = np.asarray(output_data)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim == 3:
            pass
        else:
            raise ValueError(f"output_data must be 2D or 3D, got shape {arr.shape}")

        bands, rows, cols = arr.shape

        # Decide origin: from bounds if provided, else from input origin
        if output_bounds is not None:
            xmin, ymin, xmax, ymax = output_bounds
            # For north-up (py < 0): origin Y should be ymax
            # For south-up (py > 0): origin Y should be ymin
            origin_x = xmin
            origin_y = ymax if py < 0 else ymin
        else:
            origin_x = gt_in[0]
            origin_y = gt_in[3]

        gt_out = (origin_x, px, 0.0, origin_y, 0.0, py)

        creation_opts = [f"COMPRESS={compress}"]
        if compress.upper() in ("DEFLATE", "LZW"):
            creation_opts.append(f"PREDICTOR={predictor}")

        driver = gdal.GetDriverByName("GTiff")
        ds_out = driver.Create(
            output_geotiff,
            cols,
            rows,
            bands,
            gdal.GDT_Float32,
            options=creation_opts,
        )
        ds_out.SetGeoTransform(gt_out)
        ds_out.SetProjection(ds_input.GetProjection())

        if nodata is not None:
            for b in range(1, bands + 1):
                ds_out.GetRasterBand(b).SetNoDataValue(float(nodata))

        for b in range(bands):
            ds_out.GetRasterBand(b + 1).WriteArray(arr[b])

        ds_out.FlushCache()
        ds_out = None

    def extract_file_name(self, input_rtc):
        """Extract file name identifier from input file name

        Parameters
        ----------
        input_rtc: str
            The HDF5 RTC input file path

        Returns
        -------
        file_name: str
            file name identifier
        """

        # Check if the file exists
        if not file_exists(input_rtc):
            raise FileNotFoundError(f"The file '{input_rtc}' does not exist.")

        file_name = Path(input_rtc).stem.split('-')[0]

        return file_name

    def extract_nisar_polarization(self, input_list):
        """Extract input RTC dataset polarizations

        Parameters
        ----------
        input_list: list
            The HDF5 file paths of input RTCs to be mosaicked.

        Returns
        -------
        polarizations: list of str
            All dataset polarizations listed in the input HDF5 file
        """

        pol_list_path = \
            '/science/LSAR/GCOV/grids/frequencyA/listOfPolarizations'
        polarizations = []
        pols_rtc = []
        for input_idx, input_rtc in enumerate(input_list):
            # Check if the file exists
            if not os.path.exists(input_rtc):
                raise FileNotFoundError(
                    f"The file '{input_rtc}' does not exist.")
            with H5Reader(input_rtc) as src_h5:
                pols = np.sort(src_h5[pol_list_path][()])
                filtered_pols = [pol.decode('utf-8') for pol in pols]
                polarizations.extend(filtered_pols)

        pols_rtc = set(list(polarizations))

        return pols_rtc

    def generate_nisar_mask_name(self, frequency='A'):
        """GCOV frequency-A mask dataset path"""
        return f'/science/LSAR/GCOV/grids/frequency{frequency}/mask'

    def generate_nisar_dataset_name(self, input_list: str | list[str], freq_list: np.ndarray):
        """Generate dataset paths

        Parameters
        ----------
        input_list: str or list of str
            All dataset polarizations listed in the input HDF5 file
        freq_list: array
            list of available frequency groups for each of the input files
            e.g., [[A, B], [A, B]]

        Returns
        -------
        h5_path_dict: dict
            dictionary containing the RTC file and image paths for
            each polarization
        """
        if isinstance(input_list, str):
            input_list = [input_list]
        h5_path_dict = defaultdict(dict)

        for input_idx, input_rtc in enumerate(input_list):
            if freq_list[input_idx]:
                for freq_idx, freq_group in enumerate(freq_list[input_idx]):
                    pols_rtc = get_pol_rtc_hdf5(input_rtc, freq_group)

                    group = f'/science/LSAR/GCOV/grids/frequency{freq_group}/'

                    data_path = []
                    if isinstance(pols_rtc, str):
                        pols_rtc = [pols_rtc]
                    for dname in pols_rtc:
                        hdf5_path = f'{group}{dname * 2}'
                        data_path.append(hdf5_path)

                    h5_path_dict[input_rtc][freq_group] = data_path
            else:
                warnings.warn(f'\nFrequency group {freq_group} does not exist.'
                              , RuntimeWarning)
                continue


        return h5_path_dict

    def generate_nisar_layover_name(self, layover_name: str):
        """Generate layOverShadowMask dataset path

        Parameters
        ----------
        layover_name: str
            Name of layover and shadow Mask layer in the input HDF5 file

        Returns
        -------
        data_path: str
            RTC dataset path within the HDF5 input file
        """
        group = '/science/LSAR/GCOV/grids/frequencyA/'

        data_path = f'{group}{layover_name}'

        return data_path

    def get_nisar_epsg(self, input_list):
        """extract data from RTC Geo information and store it as a dictionary

        parameters
        ----------
        input_list: list
            The HDF5 file paths of input RTCs to be mosaicked.

        Returns
        -------
        epsg_array: array of int
            EPSG of each of the RTC input HDF5
        epsg_same_flag: bool
            A flag which indicates whether all input EPSG are the same
            if True, all input EPSG are the same and vice versa.
        """
        projA = '/science/LSAR/GCOV/grids/frequencyA/projection'
        projB = '/science/LSAR/GCOV/grids/frequencyB/projection'

        epsg_array = np.zeros(len(input_list), dtype=int)
        for input_idx, input_rtc in enumerate(input_list):

            with H5Reader(input_rtc) as src_h5:

                if projA in src_h5:
                    epsg_array[input_idx] = src_h5[projA][()]

                elif projB in src_h5:
                    epsg_array[input_idx] = src_h5[projB][()]

                else:
                    raise KeyError(
                        f"No projection dataset found in {input_rtc}. "
                        f"Neither {projA} nor {projB} exists."
                    )

        epsg_same_flag = np.all(epsg_array == epsg_array[0])

        return epsg_array, epsg_same_flag

    def read_write_rtc(
        self,
        h5_ds: Dataset,
        output_gtiff,
        num_rows: int,
        num_cols: int,
        row_blk_size: int,
        col_blk_size: int,
        designated_value: np.float32,
        geotransform: Affine,
        crs: str,
        dswx_metadata_dict: dict,
        *,
        is_mask: bool = False,
        out_dtype: str | None = None):
        """Read an level-2 RTC prodcut in HDF5 format and writ it out in
        GeoTiff format in data blocks defined by row_blk_size and col_blk_size.

        Parameters
        ----------
        h5_ds: GDAL Dataset
            GDAL dataset object to be processed
        output_gtiff: str
        Output Geotiff file path and name
            num_rows: int
        The number of rows (height) of the output Geotiff.
            num_cols: int
        The number of columns (width) of the output Geotiff.
            row_blk_size: int
        The number of rows to read each time from the dataset.
            col_blk_size: int
        The number of columns to read each time from the dataset
        designated_value: np.float32
            Identify Inf in the dataset and replace them with
            a designated value
        geotransform: Affine Transformation object
            Transformation matrix which maps pixel locations in (row, col)
            coordinates to (x, y) spatial positions.
        crs: str
            Coordinate Reference System object in EPSG representation
        dswx_metadata_dict: dictionary
            This dictionary metadata extracted from input RTC
        """
        row_blk_size = self.row_blk_size
        col_blk_size = self.col_blk_size

        if out_dtype is None:
            out_dtype = 'uint8' if is_mask else 'float32'
        dst_dtype = np.uint8 if out_dtype == 'uint8' else np.float32

        profile = {
            'driver': 'GTiff',
            'height': num_rows,
            'width': num_cols,
            'count': 1,
            'dtype': out_dtype,
            'crs': crs,
            'transform': geotransform,
            'compress': 'DEFLATE',
        }
        # For masks, define nodata=255 so resampling/warps preserve validity
        if is_mask:
            profile['nodata'] = 255

        with rasterio.open(output_gtiff, 'w', **profile) as dst:
            for slice_row in slice_gen(num_rows, row_blk_size):
                row_slice_size = slice_row.stop - slice_row.start
                for slice_col in slice_gen(num_cols, col_blk_size):
                    col_slice_size = slice_col.stop - slice_col.start

                    ds_blk = h5_ds.ReadAsArray(
                        slice_col.start,
                        slice_row.start,
                        col_slice_size,
                        row_slice_size,
                    )

                    if is_mask:
                        # Keep 0/1 (or general categorical) as-is,
                        # map invalids to 255 if any
                        # If the dataset has >1 values for "valid",
                        # we leave them; caller can binarize upstream if needed.
                        ds_blk = ds_blk.astype(np.int32)  # safe cast before setting nodata
                        ds_blk = np.where(np.isfinite(ds_blk), ds_blk, 255)
                        ds_blk = ds_blk.astype(dst_dtype)
                    else:
                        # Float surface: clamp inf/huge and map zeros to NaN as in your original
                        ds_blk = ds_blk.astype(np.float32, copy=False)
                        ds_blk[np.isinf(ds_blk)] = designated_value
                        ds_blk[ds_blk > designated_value] = designated_value
                        ds_blk[ds_blk == 0] = np.nan
                        # rasterio will write NaN for float32 as is


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

    def read_geodata_hdf5(self, input_rtc):
        """extract data from RTC Geo information and store it as a dictionary

        parameters
        ----------
        input_rtc: str
            The HDF5 RTC input file path

        Returns
        -------
        geotransform: Affine Transformation object
            Transformation matrix which maps pixel locations in (row, col)
            coordinates to (x, y) spatial positions.
        crs: str
            Coordinate Reference System object in EPSG representation
        """
        freq_paths = {
            "frequencyA": "/science/LSAR/GCOV/grids/frequencyA",
            "frequencyB": "/science/LSAR/GCOV/grids/frequencyB",
        }

        # Dataset names we require under each frequency group
        required = {
            "xcoord": "xCoordinates",
            "ycoord": "yCoordinates",
            "xposting": "xCoordinateSpacing",
            "yposting": "yCoordinateSpacing",
            "proj": "projection",
        }

        def _mapping(base: str) -> dict[str, str]:
            return {k: f"{base}/{v}" for k, v in required.items()}

        with H5Reader(input_rtc) as src_h5:

            used_freq = None
            geo = None

            # Try A then B
            for freq_name, base in freq_paths.items():
                m = _mapping(base)
                missing = [path for path in m.values() if path not in src_h5]
                if not missing:
                    used_freq = freq_name
                    geo = m
                    break

            if geo is None:
                # Build a helpful error message listing what's missing for each frequency
                details = []
                for freq_name, base in freq_paths.items():
                    m = _mapping(base)
                    missing = [path for path in m.values() if path not in src_h5]
                    details.append(f"{freq_name} missing: {missing}")
                raise KeyError(
                    f"Geodata datasets not found in {input_rtc}. "
                    f"Neither frequencyA nor frequencyB contains a complete geo set. "
                    + " | ".join(details)
                )

            # Read values
            x = src_h5[geo["xcoord"]][:]
            y = src_h5[geo["ycoord"]][:]
            xmin = float(x[0])
            ymin = float(y[0])

            xres = float(src_h5[geo["xposting"]][()])
            yres = float(src_h5[geo["yposting"]][()])
            epsg = int(src_h5[geo["proj"]][()])

        # Geo transformation
        geotransform = Affine.translation(xmin - xres / 2.0, ymin - yres / 2.0) * Affine.scale(xres, yres)

        # CRS
        crs = f"EPSG:{epsg}"

        # debug log hook if you have logger:
        logger.info(f"read_geodata_hdf5: used {used_freq} for {input_rtc}")

        return geotransform, crs

    def read_metadata_hdf5(self, input_rtc):
        """Read NISAR Level-2 GCOV metadata

        Parameters
        ----------
        input_rtc: str
            The HDF5 RTC input file path

        Returns
        -------
        dswx_metadata_dict: dictionary
            RTC metadata dictionary. Will be written into output GeoTIFF.

        """
        id_path = '/science/LSAR/identification'
        meta_path = '/science/LSAR/GCOV/metadata'
        # Metadata Name Dictionary
        dswx_meta_mapping = {
            'RTC_ORBIT_PASS_DIRECTION': f'{id_path}/orbitPassDirection',
            'RTC_LOOK_DIRECTION': f'{id_path}/lookDirection',
            'RTC_PRODUCT_VERSION': f'{id_path}/productVersion',
            'RTC_SENSING_START_TIME': f'{id_path}/zeroDopplerStartTime',
            'RTC_SENSING_END_TIME': f'{id_path}/zeroDopplerEndTime',
            'RTC_FRAME_NUMBER': f'{id_path}/frameNumber',
            'RTC_TRACK_NUMBER': f'{id_path}/trackNumber',
            'RTC_ABSOLUTE_ORBIT_NUMBER': f'{id_path}/absoluteOrbitNumber',
            'RTC_INPUT_L1_SLC_GRANULES':
                f'{meta_path}/processingInformation/inputs/l1SlcGranules',
            }

        with H5Reader(input_rtc) as src_h5:
            orbit_pass_dir = src_h5[
                dswx_meta_mapping['RTC_ORBIT_PASS_DIRECTION']][()].decode()
            look_dir = src_h5[
                dswx_meta_mapping['RTC_LOOK_DIRECTION']][()].decode()
            prod_ver = src_h5[
                dswx_meta_mapping['RTC_PRODUCT_VERSION']][()].decode()
            zero_dopp_start = src_h5[
                dswx_meta_mapping['RTC_SENSING_START_TIME']][()].decode()
            zero_dopp_end = src_h5[
                dswx_meta_mapping['RTC_SENSING_END_TIME']][()].decode()
            frame_number = src_h5[
                dswx_meta_mapping['RTC_FRAME_NUMBER']][()]
            track_number = src_h5[
                dswx_meta_mapping['RTC_TRACK_NUMBER']][()]
            if isinstance(track_number, bytes):
                track_number = track_number.decode("utf-8")
            abs_orbit_number = src_h5[
                dswx_meta_mapping['RTC_ABSOLUTE_ORBIT_NUMBER']][()]
            track_number = int(track_number)
            frame_number = int(frame_number)
            abs_orbit_number = int(abs_orbit_number)
            try:
                input_slc_granules = os.path.basename(src_h5[
                    dswx_meta_mapping['RTC_INPUT_L1_SLC_GRANULES']][(0)].decode())
                dswx_metadata_dict = {
                    'ORBIT_PASS_DIRECTION': orbit_pass_dir,
                    'LOOK_DIRECTION': look_dir,
                    'PRODUCT_VERSION': prod_ver,
                    'ZERO_DOPPLER_START_TIME': zero_dopp_start,
                    'ZERO_DOPPLER_END_TIME': zero_dopp_end,
                    'FRAME_NUMBER': frame_number,
                    'TRACK_NUMBER': track_number,
                    'ABSOLUTE_ORBIT_NUMBER': abs_orbit_number,
                    'INPUT_L1_SLC_GRANULES': input_slc_granules,
                }
            except:
                print('RTC_INPUT_L1_SLC_GRANULES is not available')
                dswx_metadata_dict = {
                    'ORBIT_PASS_DIRECTION': orbit_pass_dir,
                    'LOOK_DIRECTION': look_dir,
                    'PRODUCT_VERSION': prod_ver,
                    'ZERO_DOPPLER_START_TIME': zero_dopp_start,
                    'ZERO_DOPPLER_END_TIME': zero_dopp_end,
                    'FRAME_NUMBER': frame_number,
                    'TRACK_NUMBER': track_number,
                    'ABSOLUTE_ORBIT_NUMBER': abs_orbit_number,
                }

        return dswx_metadata_dict


def slice_gen(total_size: int,
              batch_size: int,
              combine_rem: bool = True) -> Iterator[slice]:
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
        Iterable slices of data with specified input batch size,
        bounded by start_idx and stop_idx.
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


def run(cfg):
    """Generate mosaic workflow with user-defined args stored
    in dictionary runconfig 'cfg'

    Parameters:
    -----------
    cfg: RunConfig
        RunConfig object with user runconfig options
    """

    # Mosaicking parameters
    processing_cfg = cfg.groups.processing

    input_list = cfg.groups.input_file_group.input_file_path

    mosaic_cfg = processing_cfg.mosaic
    mosaic_mode = mosaic_cfg.mosaic_mode
    mosaic_prefix = mosaic_cfg.mosaic_prefix
    mosaic_posting_thresh = mosaic_cfg.mosaic_posting_thresh
    gdal_cache_max_mb = mosaic_cfg.gdal_cache_max_mb
    nisar_uni_mode = processing_cfg.nisar_uni_mode

    # Determine if resampling is required
    if nisar_uni_mode:
        resamp_required = False
    else:
        resamp_required = True

    resamp_method = mosaic_cfg.resamp_method
    resamp_out_res = mosaic_cfg.resamp_out_res

    scratch_dir = cfg.groups.product_path_group.scratch_path
    os.makedirs(scratch_dir, exist_ok=True)

    row_blk_size = mosaic_cfg.read_row_blk_size
    col_blk_size = mosaic_cfg.read_col_blk_size

    # Create reader object
    reader = RTCReader(
        row_blk_size=row_blk_size,
        col_blk_size=col_blk_size,
    )
    s3_profile = os.environ.get("AWS_PROFILE")
    aws_region = os.environ.get("AWS_REGION")

    reader = RTCReader(row_blk_size=row_blk_size, col_blk_size=col_blk_size,
                       s3_profile=s3_profile, aws_region=aws_region)
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
    )

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
