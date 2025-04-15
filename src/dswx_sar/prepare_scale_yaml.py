#!/usr/bin/env python3
"""
generate_runconfig_from_urls.py (updated)
----------------------------------------
Read a text file containing GCOV *.h5* URLs (one per line), download each file
(if it is not already present), and create a per‑granule YAML *runconfig* file
that points to the downloaded path.

New in this version
-------------------
* **scratch_path** and **sas_output_path** are now filled automatically.
* `--scratch-dir` sets the base directory for *scratch_path* (default: `./scratch`).
* `--product-dir` sets the base directory for *sas_output_path* (default: `./products`).
  A **unique sub‑folder** named after each granule is created under this base
  directory so every runconfig writes to its own location.

Example
-------
```bash
./generate_runconfig_from_urls.py \
    --url-file gcov_urls.txt \
    --download-dir ./data \
    --yaml-dir ./runconfigs \
    --scratch-dir ./scratch \
    --product-dir ./products \
    --token $EARTHDATA_TOKEN
```
"""
from __future__ import annotations

import argparse
import netrc
import os
import sys
from pathlib import Path
from typing import List, Tuple

import requests
import yaml  # PyYAML

URS_HOST = "urs.earthdata.nasa.gov"

# -----------------------------------------------------------------------------
# Authentication helpers (same logic as download_gcov_hls.py)
# -----------------------------------------------------------------------------

def _get_netrc_auth(host: str, netrc_path: Path | None = None) -> Tuple[str, str] | None:
    """Return (user, password) for *host* from .netrc if present."""
    try:
        nrc = netrc.netrc(str(netrc_path) if netrc_path else None)
        auth = nrc.authenticators(host)
        return auth[:2] if auth else None
    except (FileNotFoundError, netrc.NetrcParseError):
        return None


def _init_session(token: str | None, netrc_path: Path | None) -> requests.Session:
    sess = requests.Session()
    if token:
        sess.headers["Authorization"] = f"Bearer {token}"
    else:
        auth = _get_netrc_auth(URS_HOST, netrc_path)
        if auth:
            sess.auth = auth
        else:
            print(
                "Warning: no token and no .netrc credentials found; downloads may fail.",
                file=sys.stderr,
            )
    return sess


# -----------------------------------------------------------------------------
# Download helper
# -----------------------------------------------------------------------------

def _download(url: str, target: Path, session: requests.Session) -> None:
    """Download *url* to *target* using *session*."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(target, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


# -----------------------------------------------------------------------------
# YAML template helper
# -----------------------------------------------------------------------------

def _build_runconfig(h5_path: Path, scratch_base: Path, product_base: Path) -> dict:
    """Return the runconfig dict with paths filled in."""
    granule_name = h5_path.stem  # filename without extension

    scratch_path = scratch_base
    sas_output_path = product_base / granule_name
    product_path = sas_output_path  # store products alongside outputs

    return {
        "runconfig": {
            "name": "dswx_ni_workflow_default",
            "groups": {
                "pge_name_group": {"pge_name": "DSWX_NI_PGE"},
                "input_file_group": {
                    "input_file_path": [str(h5_path)],
                    "input_file_historical_path": None,
                    "input_mgrs_collection_id": None,
                },
                "dynamic_ancillary_file_group": {
                    "dem_file": "/home/shiroma/dat/nisar-dem-copernicus/EPSG4326/EPSG4326.vrt",
                    "dem_file_description": None,
                    "worldcover_file": "/mnt/aurora-r0/jungkyo/data/landcover.vrt",
                    "worldcover_file_description": None,
                    "glad_classification_file": "/mnt/aurora-r0/jungkyo/OPERA/DSWx-NI/landcover_test/glad_landcover_2020/glad_map.vrt",
                    "glad_classification_file_description": None,
                    "reference_water_file": "/mnt/aurora-r0/jungkyo/data/pekel.vrt",
                    "reference_water_file_description": None,
                    "hand_file": "/mnt/aurora-r0/jungkyo/data/hand/data/EPSG4326.vrt",
                    "hand_file_description": None,
                    "shoreline_shapefile": None,
                    "shoreline_shapefile_description": None,
                    "algorithm_parameters": "/mnt/aurora-r0/jungkyo/OPERA/DSWx-NI/develop/new_GCOV_4/algorithm_parameter_ni2.yaml",
                    "mean_backscattering": None,
                    "standard_deviation_backscattering": None,
                },
                "static_ancillary_file_group": {
                    "static_ancillary_inputs_flag": True,
                    "mgrs_database_file": "/mnt/aurora-r0/jungkyo/OPERA/DSWx-NI/R1_interface/sample_data/input_dir/ancillary_data/MGRS_tile.sqlite",
                    "mgrs_collection_database_file": "/mnt/aurora-r0/jungkyo/OPERA/DSWx-NI/R1_interface/sample_data/input_dir/ancillary_data/MGRS_collection_db_DSWx-NI_v0.1.sqlite",
                },
                "primary_executable": {"product_type": "dswx_ni"},
                "product_path_group": {
                    "product_path": str(product_path),
                    "scratch_path": str(scratch_path),
                    "sas_output_path": str(f"{sas_output_path}_output1"),
                    "product_version": 0.3,
                    "output_imagery_format": "COG",
                    "output_imagery_compression": "DEFLATE",
                    "output_imagery_nbits": 32,
                    "output_spacing": 30,
                },
                "browse_image_group": {
                    "save_browse": True,
                    "browse_image_height": 1024,
                    "browse_image_width": 1024,
                    "flag_collapse_wtr_classes": True,
                    "exclude_inundated_vegetation": False,
                    "set_not_water_to_nodata": False,
                    "set_hand_mask_to_nodata": True,
                    "set_layover_shadow_to_nodata": True,
                    "set_ocean_masked_to_nodata": False,
                    "save_tif_to_output": False,
                },
                "log_file": None,
            },
        }
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Download GCOV .h5 files listed in a text file and create YAML runconfigs."
    )
    ap.add_argument("--url-file", required=True, type=Path, help="Path to gcov_urls.txt")
    ap.add_argument(
        "--download-dir",
        default=Path("data"),
        type=Path,
        help="Directory to save downloaded .h5 files (default: ./data)",
    )
    ap.add_argument(
        "--yaml-dir",
        default=Path("runconfigs"),
        type=Path,
        help="Directory to write YAML runconfig files (default: ./runconfigs)",
    )
    ap.add_argument(
        "--scratch-dir",
        default=Path("scratch"),
        type=Path,
        help="Base directory for scratch_path (default: ./scratch)",
    )
    ap.add_argument(
        "--product-dir",
        default=Path("products"),
        type=Path,
        help="Base directory for sas_output_path (default: ./products)",
    )
    ap.add_argument(
        "--token",
        default=os.environ.get("EARTHDATA_TOKEN"),
        help="Bearer token for Earthdata (optional). If omitted, falls back to .netrc.",
    )
    ap.add_argument("--netrc", type=Path, help="Path to a .netrc file (optional)")

    args = ap.parse_args(argv)

    session = _init_session(args.token, args.netrc)

    # read URL list
    try:
        urls = [u.strip() for u in args.url_file.read_text().splitlines() if u.strip()]
    except FileNotFoundError:
        sys.exit(f"URL file not found: {args.url_file}")

    if not urls:
        sys.exit("URL list is empty – nothing to do.")

    # ensure directories exist
    for d in (args.download_dir, args.yaml_dir, args.scratch_dir, args.product_dir):
        d.mkdir(parents=True, exist_ok=True)

    for url in urls:
        fname = Path(url).name
        local_path = args.download_dir / fname

        if not local_path.exists():
            print(f"Downloading {fname} ...")
            try:
                _download(url, local_path, session)
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        else:
            print(f"File {fname} already exists; skipping download.")

        # build and write YAML
        runconfig = _build_runconfig(local_path, args.scratch_dir, args.product_dir)
        yaml_path = args.yaml_dir / f"{fname}.yaml"
        with open(yaml_path, "w") as y:
            yaml.safe_dump(runconfig, y, sort_keys=False)
        print(f"  Wrote runconfig -> {yaml_path}")


if __name__ == "__main__":
    main()
