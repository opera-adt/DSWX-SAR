"""Tests for tools.flood_inventory.make_runconfig.

Covers:
- CSV loading and filtering
- Acquisition grouping
- Runconfig YAML generation and yamale schema validation
- Download manifest generation
- CLI entry point

All tests run without network access, using the committed inventory CSV
and the project's yamale schemas.
"""
import csv
import json
import os
import tempfile
from pathlib import Path

import pytest
import yamale

from tools.flood_inventory.make_runconfig import (
    build_manifest,
    build_runconfig,
    filter_rows,
    group_by_acquisition,
    load_inventory,
    main,
    write_manifest,
    write_runconfig,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
INVENTORY_CSV = REPO_ROOT / "tools" / "flood_inventory" / "output" / "nisar_flood_inventory.csv"
SCHEMA_PATH = REPO_ROOT / "src" / "dswx_sar" / "schemas" / "dswx_ni.yaml"

# A real event from the committed inventory
TEST_EVENT_ID = "GDACS-FL-1103107"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def inventory_rows():
    """Load the full inventory once for all tests in this module."""
    return load_inventory(str(INVENTORY_CSV))


@pytest.fixture(scope="module")
def gcov_rows(inventory_rows):
    """Filter to DSWX-NI-eligible GCOV rows for the test event."""
    return filter_rows(inventory_rows, event_id=TEST_EVENT_ID)


@pytest.fixture(scope="module")
def yamale_schema():
    """Load the DSWX-NI yamale schema."""
    return yamale.make_schema(str(SCHEMA_PATH), parser='ruamel')


# ---------------------------------------------------------------------------
# Tests: inventory loading
# ---------------------------------------------------------------------------


class TestLoadInventory:

    def test_loads_all_rows(self, inventory_rows):
        assert len(inventory_rows) > 1000, "Inventory should have thousands of rows"

    def test_has_required_columns(self, inventory_rows):
        row = inventory_rows[0]
        for col in ('event_id', 'granule_id', 'product_type',
                     'dswx_ni_input', 'track_number', 'frame_number'):
            assert col in row, f"Missing column: {col}"

    def test_raises_on_bad_csv(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("col_a,col_b\n1,2\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_inventory(str(bad_csv))


# ---------------------------------------------------------------------------
# Tests: filtering
# ---------------------------------------------------------------------------


class TestFilterRows:

    def test_filters_to_event(self, gcov_rows):
        assert len(gcov_rows) > 0
        assert all(r['event_id'] == TEST_EVENT_ID for r in gcov_rows)

    def test_only_gcov(self, gcov_rows):
        assert all(r['product_type'] == 'GCOV' for r in gcov_rows)

    def test_only_dswx_ni_eligible(self, gcov_rows):
        assert all(r['dswx_ni_input'] == 'True' for r in gcov_rows)

    def test_granule_filter(self, inventory_rows):
        # Pick a specific granule
        gid = "NISAR_L2_PR_GCOV_003_075_A_169_2005_DHDH_A_20251022T104928_20251022T104935_X05010_N_P_J_001"
        rows = filter_rows(inventory_rows, granule_ids=[gid])
        assert len(rows) == 1
        assert rows[0]['granule_id'] == gid

    def test_no_match_returns_empty(self, inventory_rows):
        rows = filter_rows(inventory_rows, event_id="NONEXISTENT-EVENT")
        assert rows == []


# ---------------------------------------------------------------------------
# Tests: grouping
# ---------------------------------------------------------------------------


class TestGroupByAcquisition:

    def test_groups_by_track_date_pass(self, gcov_rows):
        groups = group_by_acquisition(gcov_rows)
        assert len(groups) > 0
        # Check label format
        for label in groups:
            assert label.startswith("T"), f"Bad label: {label}"
            parts = label.split("_")
            assert len(parts) == 3, f"Expected T<track>_<date>_<pass>: {label}"

    def test_frames_sorted(self, gcov_rows):
        groups = group_by_acquisition(gcov_rows)
        for label, rows in groups.items():
            frame_nums = [int(r['frame_number']) for r in rows]
            assert frame_nums == sorted(frame_nums), (
                f"Frames not sorted in {label}: {frame_nums}"
            )

    def test_first_group_has_expected_structure(self, gcov_rows):
        groups = group_by_acquisition(gcov_rows)
        label = "T75_2025-10-22_A"
        assert label in groups
        rows = groups[label]
        assert len(rows) == 3  # 3 frames for this track/date


# ---------------------------------------------------------------------------
# Tests: runconfig generation + schema validation
# ---------------------------------------------------------------------------


class TestBuildRunconfig:

    def test_builds_valid_dict(self, gcov_rows):
        groups = group_by_acquisition(gcov_rows)
        label, rows = next(iter(groups.items()))
        config = build_runconfig(label, rows)
        assert 'runconfig' in config
        assert 'groups' in config['runconfig']
        groups_dict = config['runconfig']['groups']
        assert 'input_file_group' in groups_dict
        assert len(groups_dict['input_file_group']['input_file_path']) == len(rows)

    def test_schema_validation_passes(self, gcov_rows, yamale_schema, tmp_path):
        """Core acceptance test: generated runconfig validates against schema."""
        groups = group_by_acquisition(gcov_rows)
        label, rows = next(iter(groups.items()))
        config = build_runconfig(label, rows)

        yaml_path = tmp_path / "test_runconfig.yaml"
        write_runconfig(config, str(yaml_path))

        data = yamale.make_data(str(yaml_path), parser='ruamel')
        yamale.validate(yamale_schema, data)  # raises on failure

    def test_all_groups_validate(self, gcov_rows, yamale_schema, tmp_path):
        """Validate runconfigs for ALL acquisition groups of the test event."""
        groups = group_by_acquisition(gcov_rows)
        for label, rows in groups.items():
            config = build_runconfig(label, rows)
            yaml_path = tmp_path / f"rc_{label}.yaml"
            write_runconfig(config, str(yaml_path))
            data = yamale.make_data(str(yaml_path), parser='ruamel')
            yamale.validate(yamale_schema, data)

    def test_input_paths_use_granule_ids(self, gcov_rows):
        groups = group_by_acquisition(gcov_rows)
        label, rows = next(iter(groups.items()))
        config = build_runconfig(label, rows, input_dir="/data/gcov")
        paths = config['runconfig']['groups']['input_file_group']['input_file_path']
        for path, row in zip(paths, rows):
            assert row['granule_id'] in path
            assert path.startswith("/data/gcov/")
            assert path.endswith(".h5")

    def test_custom_ancillary_dir(self, gcov_rows):
        groups = group_by_acquisition(gcov_rows)
        label, rows = next(iter(groups.items()))
        config = build_runconfig(label, rows, ancillary_dir="/opt/anc")
        anc = config['runconfig']['groups']['dynamic_ancillary_file_group']
        assert anc['dem_file'] == "/opt/anc/dem.tif"
        assert anc['hand_file'] == "/opt/anc/hand.tif"
        assert anc['reference_water_file'] == "/opt/anc/reference_water.tif"


# ---------------------------------------------------------------------------
# Tests: download manifest
# ---------------------------------------------------------------------------


class TestBuildManifest:

    def test_manifest_structure(self, gcov_rows):
        manifest = build_manifest(gcov_rows, TEST_EVENT_ID)
        assert manifest['event_id'] == TEST_EVENT_ID
        assert manifest['total_granules'] > 0
        assert manifest['total_granules'] == len(manifest['granules'])

    def test_granule_dedup(self, gcov_rows):
        """Manifest should contain unique granule IDs."""
        manifest = build_manifest(gcov_rows, TEST_EVENT_ID)
        ids = [g['granule_id'] for g in manifest['granules']]
        assert len(ids) == len(set(ids))

    def test_granule_has_download_info(self, gcov_rows):
        manifest = build_manifest(gcov_rows, TEST_EVENT_ID)
        g = manifest['granules'][0]
        assert 'cmr_query' in g
        assert 'asf_search' in g
        assert 'earthaccess' in g
        assert g['asf_search']['platform'] == 'NISAR'

    def test_write_manifest_json(self, gcov_rows, tmp_path):
        manifest = build_manifest(gcov_rows, TEST_EVENT_ID)
        out = tmp_path / "manifest.json"
        write_manifest(manifest, str(out))
        loaded = json.loads(out.read_text())
        assert loaded['event_id'] == TEST_EVENT_ID
        assert loaded['total_granules'] == manifest['total_granules']


# ---------------------------------------------------------------------------
# Tests: CLI integration
# ---------------------------------------------------------------------------


class TestCLI:

    def test_list_events(self, capsys):
        rc = main(['--inventory', str(INVENTORY_CSV), '--list-events'])
        assert rc == 0
        out = capsys.readouterr().out
        assert TEST_EVENT_ID in out
        assert "granules" in out

    def test_generate_for_event(self, tmp_path):
        out_dir = str(tmp_path / "out")
        rc = main([
            '--inventory', str(INVENTORY_CSV),
            '--event-id', TEST_EVENT_ID,
            '--output-dir', out_dir,
        ])
        assert rc == 0
        event_dir = tmp_path / "out" / TEST_EVENT_ID
        assert event_dir.is_dir()
        yamls = list(event_dir.glob("runconfig_*.yaml"))
        assert len(yamls) > 0
        assert (event_dir / "download_manifest.json").exists()

    def test_generate_with_validation(self, tmp_path):
        out_dir = str(tmp_path / "out")
        rc = main([
            '--inventory', str(INVENTORY_CSV),
            '--event-id', TEST_EVENT_ID,
            '--output-dir', out_dir,
            '--validate',
        ])
        assert rc == 0

    def test_nonexistent_event(self, tmp_path):
        rc = main([
            '--inventory', str(INVENTORY_CSV),
            '--event-id', 'FAKE-EVENT-999',
            '--output-dir', str(tmp_path),
        ])
        assert rc == 1

    def test_no_args_is_error(self):
        with pytest.raises(SystemExit):
            main(['--inventory', str(INVENTORY_CSV)])
