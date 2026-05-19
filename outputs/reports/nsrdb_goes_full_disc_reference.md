# NSRDB GOES Full Disc Ingest Reference

## Scope

This note captures the current implementation plan for Puerto Rico geophysical
ingest from the National Solar Radiation Database (NSRDB), using the GOES Full
Disc PSM v4 endpoint that already appears in `nsrdb_data_download_PR.py`.

The target use case is downstream statistical analysis, not imagery display.
That means the primary outputs should be reproducible time-series extracts and
geometry-aware aggregates, not ad hoc CSV downloads.

## Primary Endpoint

- Dataset: `nsrdb-GOES-full-disc-v4-0-0-download`
- Base URL: `https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-full-disc-v4-0-0-download.json`
- Dataset coverage: 2018 onward
- Spatial resolution: 2 km
- Supported intervals: 10, 30, 60 minutes
- Required request fields: `api_key`, `wkt`, `names`, `interval`, `email`

The earlier high-level NSRDB catalog page describes GOES Full Disc as a 10-minute
product, but the endpoint-specific documentation explicitly allows `10`, `30`,
and `60` minute intervals. That means the sample script's `interval='30'` is
valid for this endpoint.

## Current Project Attribute Set

The downloaded viewer export in `nsrdb_data_download_PR.py` is already focused on
the variables that matter for the project's analysis layer:

- `air_temperature`
- `clearsky_dhi`
- `clearsky_dni`
- `clearsky_ghi`
- `dhi`
- `dni`
- `ghi`
- `solar_zenith_angle`
- `surface_albedo`

This is a reasonable first pass because it covers ambient temperature, observed
irradiance, clear-sky irradiance, and one surface property. Additional weather
fields such as `relative_humidity`, `wind_speed`, or `surface_pressure` should be
added only if they are needed for a specific regression or feature-engineering step.

## Documented Versus Viewer-Export Inputs

The public endpoint documentation emphasizes `wkt` geometry input, while the
saved viewer export uses `location_ids` batches. The implementation plan should
treat these differently:

- `wkt` is the documented, portable path and should be used for preflight,
  reproduction, and future reruns.
- `location_ids` from the viewer export are a useful bootstrap artifact for the
  current Puerto Rico coverage, but they should not be the only reproducibility
  path.

Recommended approach:

1. Use the documented `nsrdb_data_query` and `site-count` endpoints to confirm
   coverage and expected site counts for Puerto Rico.
2. Preserve the current viewer-export point groups as a draft batching seed.
3. Add a regeneration step that derives the same or equivalent coverage from
   documented geometry-based queries so the workflow does not depend on a manual
   browser export forever.

## Rate Limits And Queue Rules

Use the endpoint-specific GOES Full Disc limits, not the general developer
network defaults.

- `.csv` direct-download requests: up to `10,000` per day, no more than `1`
  request per second
- Non-CSV archive requests such as `.json`: up to `2,000` per day, no more than
  `1` request every `2` seconds
- In-flight request cap: `20`
- Queue fail-safe: the service may reject new requests when the backend queue is full

Operational rule for this project:

- Use `.json` archive requests for island-wide or multi-point jobs
- Use `.csv` only for single-point smoke tests
- Enforce a client-side throttle of at least `2` seconds between archive
  submissions and keep concurrency below `20`

## Output And Delivery Model

The endpoint is asynchronous for large jobs.

- A JSON request returns an acknowledgement plus an `outputs.downloadUrl`
- The service also emails the requester when file generation is ready
- Direct streaming CSV is only supported for a single point and a single year

That means the production workflow should assume a two-step model:

1. Submit request and record the acknowledgement
2. Poll or consume the provided download URL and persist the returned archive

## Recommended Storage Layout

Because the NSRDB endpoint returns time-series data rather than raster tiles in
this workflow, the raw artifacts should live under `data/tabular/`.

Recommended layout:

- `data/tabular/nsrdb/raw/` for downloaded ZIP or CSV payloads
- `data/tabular/nsrdb/normalized/` for cleaned parquet extracts
- `data/tabular/nsrdb/_ledger.parquet` for request and download tracking

Recommended normalized tables:

- `pr_nsrdb_request_ledger` in DuckDB for request metadata
- `pr_nsrdb_sites` for site identifiers, geometry, and batch membership
- `pr_nsrdb_timeseries_2024` for normalized long-form observations
- optional derived aggregate tables keyed by census geography if the analysis
  wants BG-, tract-, or municipio-level summaries directly in DuckDB

## Ledger Requirements

Mirror the design already used in `utils/solar_api.py`, but keep a separate
ledger because the request model, artifact types, and quota rules differ.

Minimum ledger columns:

- `requested_at_utc`
- `endpoint`
- `request_format`
- `year`
- `interval_minutes`
- `attributes`
- `batch_id`
- `site_count`
- `status`
- `http_status`
- `download_url`
- `local_archive_path`
- `notes`

## Implementation Sequence

### Phase 1 — Preflight

1. Confirm Puerto Rico site coverage with `nsrdb_data_query` or `site-count`
2. Decide whether the canonical coverage unit is a Puerto Rico polygon query,
   a regenerated site lattice, or the existing viewer-export `location_ids`
3. Lock the year and interval policy for the first ingest run

### Phase 2 — Client Wrapper

1. Create `utils/nsrdb_api.py`
2. Implement JSON archive submission with `api_key` in the query string
3. Add ledger writes for submission, success, failure, and download completion
4. Add request pacing that respects the stricter GOES Full Disc limits

### Phase 3 — Normalization

1. Unpack returned archives
2. Parse Standard Time Series Data File Format headers separately from the data body
3. Write normalized parquet tables
4. Join site identifiers back to Puerto Rico census geometries in DuckDB

### Phase 4 — Analysis-Ready Aggregates

1. Compute municipal, tract, and block-group summaries from the normalized site table
2. Persist those summaries into DuckDB for direct regression and mapping workflows

## Notes On The Existing Example Script

`nsrdb_data_download_PR.py` should currently be treated as a saved viewer export,
not as the final project client. It already gives us useful starting decisions:

- GOES Full Disc is the chosen dataset
- the project is targeting 2024 first
- the initial attribute bundle is reasonable
- Puerto Rico-wide coverage was split into batches of point IDs

However, it still has several implementation gaps:

- no durable ledger
- no normalized storage target
- no request retry or queue handling
- no documented regeneration path for the `location_ids`
- placeholder credentials instead of project configuration

## Decision For The Sprint

Do not run a bulk island-wide NSRDB ingest in this sprint yet.

The correct sprint deliverable is a validated ingestion design and a thin client
wrapper plan grounded in the current docs, so the actual pull can happen without
guesswork once the statistical-analysis branch is ready for it.