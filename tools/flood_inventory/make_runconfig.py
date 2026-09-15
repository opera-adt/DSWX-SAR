#!/usr/bin/env python3
"""Generate DSWX-NI runconfigs and download manifests from the NISAR flood inventory.

Given the flood inventory CSV (produced by D828) and an event_id or granule
list, this tool emits:
  (a) A DSWX-NI runconfig YAML validated against src/dswx_sar/schemas/dswx_ni.yaml
  (b) A JSON download manifest for the referenced GCOV granules (asf_search URLs)

Usage
-----
    python -m tools.flood_inventory.make_runconfig \\
        --inventory tools/flood_inventory/output/nisar_flood_inventory.csv \\
        --event-id GDACS-FL-1103107 \\
        --output-dir ./runconfigs

    python -m tools.flood_inventory.make_runconfig \\
        --inventory tools/flood_inventory/output/nisar_flood_inventory.csv \\
        --granule NISAR_L2_PR_GCOV_003_075_A_169_2005_DHDH_A_20251022T104928_20251022T104935_X05010_N_P_J_001 \\
        --output-dir ./runconfigs
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from ruamel.yaml import YAML


# ---------------------------------------------------------------------------
# Inventory loading
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    'event_id', 'granule_id', 'product_type', 'dswx_ni_input',
    'acquisition_date', 'track_number', 'frame_number', 'orbit_pass',
    'polarization', 'collection_short_name',
}


def load_inventory(csv_path: str) -> list[dict]:
    """Load the flood inventory CSV and return rows as dicts."""
    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Inventory CSV missing required columns: {missing}")
        return list(reader)


def filter_rows(
    rows: list[dict],
    event_id: Optional[str] = None,
    granule_ids: Optional[list[str]] = None,
) -> list[dict]:
    """Filter inventory rows to DSWX-NI-eligible GCOV granules.

    Selection criteria:
    - dswx_ni_input == 'True'
    - product_type == 'GCOV'
    - Matches event_id or is in granule_ids list
    """
    filtered = []
    for row in rows:
        if row['dswx_ni_input'] != 'True':
            continue
        if row['product_type'] != 'GCOV':
            continue
        if event_id and row['event_id'] != event_id:
            continue
        if granule_ids and row['granule_id'] not in granule_ids:
            continue
        filtered.append(row)
    return filtered


def group_by_acquisition(rows: list[dict]) -> dict[str, list[dict]]:
    """Group rows by (track, acquisition_date, orbit_pass).

    Each group represents a set of frames from the same pass that can be
    mosaicked together into a single DSWX-NI run.
    """
    groups = defaultdict(list)
    for row in rows:
        key = (
            row['track_number'],
            row['acquisition_date'],
            row['orbit_pass'],
        )
        group_label = f"T{row['track_number']}_{row['acquisition_date']}_{row['orbit_pass'][0]}"
        groups[group_label].append(row)
    # Sort frames within each group by frame number
    for label in groups:
        groups[label].sort(key=lambda r: int(r['frame_number']))
    return dict(groups)


# ---------------------------------------------------------------------------
# Runconfig generation
# ---------------------------------------------------------------------------

ASF_DOWNLOAD_BASE = "https://datapool.asf.alaska.edu"
CMR_GRANULE_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"


def _gcov_download_path(granule_id: str) -> str:
    """Construct placeholder local path for a downloaded GCOV granule."""
    return f"input_dir/{granule_id}/{granule_id}.h5"


def build_runconfig(
    group_label: str,
    rows: list[dict],
    output_dir: str = "output",
    scratch_dir: str = "scratch",
    ancillary_dir: str = "ancillary",
    input_dir: str = "input_dir",
    algorithm_parameters: Optional[str] = None,
) -> dict:
    """Build a DSWX-NI runconfig dict from a group of inventory rows.

    Parameters
    ----------
    group_label : str
        Label for this acquisition group (e.g. T75_2025-10-22_A).
    rows : list[dict]
        Inventory rows for this group (same track/date/pass).
    output_dir : str
        Base directory for product output.
    scratch_dir : str
        Directory for scratch/temp files.
    ancillary_dir : str
        Directory where ancillary files (DEM, HAND, etc.) reside.
    input_dir : str
        Directory where downloaded GCOV files will be placed.
    algorithm_parameters : str, optional
        Path to algorithm parameters YAML. If None, a placeholder is used.

    Returns
    -------
    dict
        Runconfig dictionary ready for YAML serialization.
    """
    input_paths = [
        f"{input_dir}/{row['granule_id']}/{row['granule_id']}.h5"
        for row in rows
    ]

    # Extract event metadata from first row
    event_id = rows[0]['event_id']

    config = {
        'runconfig': {
            'name': f"dswx_ni_{event_id}_{group_label}",
            'groups': {
                'pge_name_group': {
                    'pge_name': 'DSWX_NI_PGE',
                },
                'input_file_group': {
                    'input_file_path': input_paths,
                },
                'dynamic_ancillary_file_group': {
                    'dem_file': f"{ancillary_dir}/dem.tif",
                    'reference_water_file': f"{ancillary_dir}/reference_water.tif",
                    'hand_file': f"{ancillary_dir}/hand.tif",
                    'algorithm_parameters': (
                        algorithm_parameters
                        or f"{ancillary_dir}/algorithm_parameter_ni.yaml"
                    ),
                },
                'static_ancillary_file_group': {
                    'static_ancillary_inputs_flag': True,
                },
                'primary_executable': {
                    'product_type': 'dswx_ni',
                },
                'product_path_group': {
                    'product_path': f"{output_dir}/{event_id}/{group_label}",
                    'scratch_path': f"{scratch_dir}/{event_id}/{group_label}",
                    'sas_output_path': f"{output_dir}/{event_id}/{group_label}",
                },
                'browse_image_group': {
                    'save_browse': True,
                },
            },
        }
    }
    return config


def write_runconfig(config: dict, path: str) -> None:
    """Serialize a runconfig dict to YAML."""
    yaml = YAML()
    yaml.default_flow_style = False
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as fh:
        yaml.dump(config, fh)


# ---------------------------------------------------------------------------
# Download manifest
# ---------------------------------------------------------------------------

def build_manifest(
    rows: list[dict],
    event_id: str,
) -> dict:
    """Build a download manifest for GCOV granules.

    The manifest contains granule metadata and URLs for asf_search /
    earthaccess download.

    Returns
    -------
    dict
        Manifest with event metadata and per-granule download info.
    """
    granules = []
    seen = set()
    for row in rows:
        gid = row['granule_id']
        if gid in seen:
            continue
        seen.add(gid)
        granules.append({
            'granule_id': gid,
            'collection': row['collection_short_name'],
            'acquisition_date': row['acquisition_date'],
            'track_number': int(row['track_number']),
            'frame_number': int(row['frame_number']),
            'orbit_pass': row['orbit_pass'],
            'polarization': row['polarization'],
            'cmr_query': {
                'url': CMR_GRANULE_URL,
                'params': {
                    'short_name': row['collection_short_name'],
                    'readable_granule_name': gid,
                    'provider': 'JPL',
                },
            },
            'asf_search': {
                'platform': 'NISAR',
                'granule_list': [gid],
            },
            'earthaccess': {
                'short_name': row['collection_short_name'],
                'granule_name': gid,
            },
        })

    # Event-level metadata from first row
    first = rows[0]
    manifest = {
        'event_id': event_id,
        'event_name': first.get('event_name', ''),
        'country': first.get('country', ''),
        'total_granules': len(granules),
        'granules': granules,
    }
    return manifest


def write_manifest(manifest: dict, path: str) -> None:
    """Write download manifest to JSON."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(manifest, fh, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DSWX-NI runconfigs and download manifests "
            "from the NISAR flood inventory."
        ),
    )
    parser.add_argument(
        '--inventory', '-i',
        default='tools/flood_inventory/output/nisar_flood_inventory.csv',
        help='Path to the NISAR flood inventory CSV.',
    )
    parser.add_argument(
        '--event-id', '-e',
        help='Filter inventory to this event ID (e.g. GDACS-FL-1103107).',
    )
    parser.add_argument(
        '--granule', '-g',
        action='append',
        dest='granule_ids',
        help='Filter to specific granule ID(s). May be repeated.',
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./runconfigs',
        help='Directory for generated runconfig YAML and manifest JSON.',
    )
    parser.add_argument(
        '--input-dir',
        default='input_dir',
        help='Base directory where downloaded GCOV files will reside.',
    )
    parser.add_argument(
        '--ancillary-dir',
        default='ancillary',
        help='Directory containing ancillary files (DEM, HAND, etc.).',
    )
    parser.add_argument(
        '--algorithm-parameters',
        help='Path to algorithm parameters YAML (optional override).',
    )
    parser.add_argument(
        '--scratch-dir',
        default='scratch',
        help='Directory for scratch/temp files.',
    )
    parser.add_argument(
        '--list-events',
        action='store_true',
        help='List all available event IDs and exit.',
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate generated runconfigs against the DSWX-NI schema.',
    )
    return parser


def validate_runconfig(yaml_path: str) -> bool:
    """Validate a runconfig YAML against the DSWX-NI schema.

    Returns True if valid, raises on failure.
    """
    import yamale

    schema_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'src', 'dswx_sar', 'schemas',
    )
    schema_path = os.path.join(schema_dir, 'dswx_ni.yaml')
    schema = yamale.make_schema(schema_path, parser='ruamel')
    data = yamale.make_data(yaml_path, parser='ruamel')
    yamale.validate(schema, data)
    return True


def main(argv: Optional[list[str]] = None) -> int:
    parser = get_parser()
    args = parser.parse_args(argv)

    if not args.event_id and not args.granule_ids and not args.list_events:
        parser.error("Specify --event-id, --granule, or --list-events.")

    rows = load_inventory(args.inventory)

    if args.list_events:
        events = sorted({
            r['event_id'] for r in rows
            if r['dswx_ni_input'] == 'True' and r['product_type'] == 'GCOV'
        })
        for eid in events:
            # Count granules per event
            n = sum(1 for r in rows
                    if r['event_id'] == eid
                    and r['dswx_ni_input'] == 'True'
                    and r['product_type'] == 'GCOV')
            name = next(
                (r['event_name'] for r in rows if r['event_id'] == eid), '')
            print(f"{eid}\t{n} granules\t{name}")
        return 0

    # Filter to matching rows
    filtered = filter_rows(
        rows,
        event_id=args.event_id,
        granule_ids=args.granule_ids,
    )
    if not filtered:
        print(
            f"No DSWX-NI-eligible GCOV granules found for "
            f"event_id={args.event_id!r}, granules={args.granule_ids!r}",
            file=sys.stderr,
        )
        return 1

    event_id = args.event_id or filtered[0]['event_id']
    print(f"Found {len(filtered)} GCOV granules for {event_id}")

    # Group by track/date/pass
    groups = group_by_acquisition(filtered)
    print(f"Grouped into {len(groups)} acquisition(s):")
    for label, group_rows in groups.items():
        frames = [r['frame_number'] for r in group_rows]
        print(f"  {label}: {len(group_rows)} frames ({', '.join(frames)})")

    # Generate runconfigs
    out_dir = os.path.join(args.output_dir, event_id)
    os.makedirs(out_dir, exist_ok=True)

    runconfig_paths = []
    for label, group_rows in groups.items():
        config = build_runconfig(
            group_label=label,
            rows=group_rows,
            output_dir=args.output_dir,
            scratch_dir=args.scratch_dir,
            ancillary_dir=args.ancillary_dir,
            input_dir=args.input_dir,
            algorithm_parameters=args.algorithm_parameters,
        )
        rc_path = os.path.join(out_dir, f"runconfig_{label}.yaml")
        write_runconfig(config, rc_path)
        runconfig_paths.append(rc_path)
        print(f"  Wrote runconfig: {rc_path}")

    # Generate download manifest
    manifest = build_manifest(filtered, event_id)
    manifest_path = os.path.join(out_dir, "download_manifest.json")
    write_manifest(manifest, manifest_path)
    print(f"  Wrote manifest:  {manifest_path} ({manifest['total_granules']} granules)")

    # Validate if requested
    if args.validate:
        print("\nValidating runconfigs against DSWX-NI schema...")
        all_valid = True
        for rc_path in runconfig_paths:
            try:
                validate_runconfig(rc_path)
                print(f"  PASS: {rc_path}")
            except Exception as exc:
                print(f"  FAIL: {rc_path}: {exc}", file=sys.stderr)
                all_valid = False
        if not all_valid:
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
