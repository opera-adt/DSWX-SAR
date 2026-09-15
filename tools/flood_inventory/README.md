# NISAR Flood Inventory

A catalogue of **NISAR scenes that observed real flood events**, built for
DSWX-SAR / DSWX-NI development: algorithm tuning, validation case selection,
and sizing the chunking / parallelisation trade-offs in D827.

Everything here is derived from two public services and is fully
regenerable — no manual curation, no hand-edited rows.

## Contents

| File | What it is |
| --- | --- |
| `output/nisar_flood_inventory.csv` | One row per (flood event × NISAR granule). The primary artifact. |
| `output/nisar_flood_inventory.geojson` | Same rows as features; geometry is the **granule** footprint bbox. |
| `output/flood_events.geojson` | One feature per flood event that NISAR actually observed; geometry is the **AOI** bbox. |
| `output/manifest.json` | Provenance: generation time, sources, every CLI parameter, and summary counts. |
| `output/verification.json` | Result of the last `verify_inventory.py` run. |

### CSV columns

**Event**: `event_id` (`GDACS-FL-<id>`), `event_name`, `country`, `iso3`,
`basin_aoi_name`, `alert_level`, `event_start`, `event_end`, `report_url`.

**AOI**: `aoi_source` (`affected_area` — a real GDACS affected-area polygon; or
`centroid_buffer` — a square box around the event centroid when no polygon was
published), `aoi_bbox_{west,south,east,north}`.

**Granule**: `granule_id` (the CMR granule UR), `collection_short_name`,
`product_type`, `maturity` (`provisional` / `beta` / `urgent_response`),
`dswx_ni_input`, `acquisition_date`, `acquisition_start`, `acquisition_end`, `pair_product`,
`days_into_event`, `track_number`, `frame_number`, `orbit_pass`,
`polarization`, `full_frame`, `granule_bbox_{west,south,east,north}`.

### Single scenes vs. pair products

`acquisition_start` / `acquisition_end` are the granule's CMR temporal extent.
For a single scene (GCOV, GSLC) that is a ~25 s acquisition and
`acquisition_date` is simply its date.

For an **interferometric pair** (GOFF, GUNW — flagged by `pair_product=True`)
the extent spans the *reference* acquisition through the *secondary* one, often
12 days apart. `acquisition_date` is therefore taken from the **secondary**,
the observation that actually saw the flood. A consequence worth knowing: a
pair whose reference precedes the flood and whose secondary follows it will
have `acquisition_date > event_end`. That is not an error — it is precisely
the bracketing geometry you want for change detection.

`dswx_ni_input` is `True` only for **GCOV** products (including `UR_GCOV`),
because GCOV is what the DSWX-NI workflow ingests. The urgent-response
collection also carries GSLC / GOFF / GUNW; those are catalogued for context
but are not DSWX inputs — filter on `dswx_ni_input` before handing rows to a
DSWX run.

## How the list was built

1. **Flood events** come from the [GDACS](https://www.gdacs.org) flood archive
   (`eventtype=FL`), queried in ≤31-day slices over the NISAR archive window.
   GDACS is run by the EC Joint Research Centre / UN OCHA, is openly queryable
   and is stable enough to regenerate from. Each event carries an alert level,
   an activity start/end date, affected countries, and — per episode — an
   affected-area polygon. Where an event has multiple episodes we keep the
   highest-numbered one, which carries the final extent.
2. **AOI** is the bounding box of that affected-area polygon. Events whose only
   published geometry is a centroid fall back to a ±0.75° box, flagged via
   `aoi_source=centroid_buffer`. Events whose AOI exceeds
   `--max-aoi-area-deg2` are skipped rather than triggering a continent-wide
   granule search.
3. **NISAR granules** come from [NASA CMR](https://cmr.earthdata.nasa.gov)
   granule search, restricted to the event AOI bbox **and** the event's active
   date range. So a row means: *this NISAR scene was acquired over this flood's
   affected area while the flood was active.*
4. Rows are deduplicated per (event, granule) and sorted by event start date.

### Why these bounds

NISAR L2 products only appear in CMR from **2025-10-12** (beta) / **2025-10-29**
(provisional), so the searchable window opens there, not at launch. Searching
earlier only burns requests.

## Regenerating

Standard library only — no extra dependencies, no Earthdata login (CMR granule
*search* is open; downloading the data is not).

```bash
cd tools/flood_inventory

# Full rebuild, exactly as committed:
python build_inventory.py \
    --start 2025-10-01 --end $(date -u +%F) \
    --alert-levels Green Orange Red \
    --workers 8 --output-dir output

# Major floods only (a much smaller, higher-confidence list):
python build_inventory.py --alert-levels Orange Red --output-dir output_major

# Include a pre-event reference window for change detection:
python build_inventory.py --pre-days 24 --post-days 12 --output-dir output_ref
```

Useful knobs (`--help` for all):

| Flag | Default | Effect |
| --- | --- | --- |
| `--alert-levels` | `Orange Red` | GDACS severity filter. `Green` adds many smaller events. |
| `--collections` | GCOV provisional + beta + `NISAR_UR_L2` | Which CMR collections to search. |
| `--pre-days` / `--post-days` | `0` | Widen the search window around the event. |
| `--centroid-buffer-deg` | `0.75` | AOI half-width when GDACS gives only a centroid. |
| `--max-aoi-area-deg2` | `2000` | Skip AOIs bigger than this. |
| `--max-granules-per-event` | `500` | Cap per event per collection. |
| `--workers` | `8` | Concurrent per-event GDACS/CMR fetches. |

A full rebuild is dominated by per-event HTTP round trips — one GDACS polygon
fetch plus one CMR search per collection — so it is run on a thread pool
(`--workers`, default 8). Serially the same build takes hours. Drop to
`--workers 1` if you are being rate limited. `build_inventory.py` streams
progress to stderr and the summary JSON to stdout.

> The output is **not** frozen in time: GDACS revises event extents and end
> dates, and NISAR products get reprocessed between maturity levels. A rebuild
> on a later date will legitimately differ. `manifest.json` records the
> parameters and generation time of the committed copy.

## Verifying

Every granule in the CSV must resolve in CMR:

```bash
python verify_inventory.py --inventory output/nisar_flood_inventory.csv
```

Exits non-zero and prints `MISSING <granule_id>` for any UR CMR no longer
returns — which is how you notice a reprocessing campaign superseded part of
the inventory. The result is written to `output/verification.json`.

## Tests

```bash
python -m pytest tests/test_flood_inventory.py -q
```

Offline unit tests cover UMM-G parsing, product-type/DSWX-input classification,
GDACS geometry handling and centroid fallback, search-window clipping, and the
output writers. A final class asserts structural invariants of the *committed*
inventory (non-empty, well-formed bboxes, acquisitions inside the event window,
unique event/granule pairs); it skips if `output/` has not been built.

## What is in the committed inventory

Generated 2026-09-15 over `2025-10-01 .. 2026-09-15` at all three GDACS alert
levels. Exact parameters are in `manifest.json`.

| | |
| --- | --- |
| Flood events searched | 622 |
| Events NISAR actually observed | 230 (222 Green, 6 Orange, 2 Red) |
| Rows (event × granule) | 9,542 |
| Unique granules | 8,468 |
| DSWX-NI inputs (GCOV + UR_GCOV) | 8,741 rows |
| By product type | GCOV 8,560 · UR_GCOV 181 · UR_GOFF 315 · UR_GUNW 315 · UR_GSLC 171 |
| AOI source | 622/622 real affected-area polygons; no centroid fallbacks |

All 8,468 granules resolved in CMR at generation time — see
`output/verification.json`.

Coverage is strongly skewed late: NISAR L2 availability ramps through the
archive, so the August–September 2026 events carry hundreds of granules each
while most late-2025 events carry none. For D827 chunking work, the large
recent events (India, China, Nepal, Poland, Germany) are the realistic
stress cases.

## Known limitations

- **AOI is a bounding box, not the flood polygon.** A granule intersecting the
  bbox may not intersect the flooded area itself. Treat rows as candidates.
- **Antimeridian-crossing AOIs** are passed through as-is and will search the
  wrong side of the globe. None are present in the committed inventory.
- **GDACS alert level is a modelled hazard score**, not a confirmed observation
  of flooding. Green events in particular include many minor or forecast-only
  cases.
- **No inundation labels.** This inventory answers *which scenes to look at*,
  not *what is water in them*. Validation labels are a separate problem.
