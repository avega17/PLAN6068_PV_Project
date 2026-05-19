# PLAN 6068 Final Project: Rooftop PV Panel-Row Extraction for Puerto Rico

This repository contains the implementation scaffold for a geospatial AI workflow to extract rooftop solar panel-row features from remote sensing imagery and building footprints in Puerto Rico.

## Project Goal

Develop a reproducible pipeline that:
- transforms coarse rooftop and building footprint geometries into granular panel-row features,
- supports exploratory correlation analysis with sociodemographic indicators (US Census 2020), and
- compares those relationships against geophysical variables such as solar irradiance, ground temperature, and land cover.

## Research Focus

This project is motivated by the rapid growth of decentralized rooftop PV in Puerto Rico and the need for a granular spatial inventory to support:
- distribution-level grid planning,
- short-term forecasting context, and
- equity-oriented urban planning analysis.

## Method Summary

The planned workflow combines:
- vector processing for building footprints and PV installation geometries,
- raster ingestion/preprocessing for high-resolution imagery,
- superpixel-driven computer vision segmentation to estimate panel-row rectangles,
- spatial joins and feature engineering with Python, and DuckDB (spatial SQL),
- exploratory analysis of socioeconomic and geophysical associations.

## Data Inputs (Planned)

- Overture Maps building footprints (validated against CRIM cadastres)
- LUMA / NEPR interconnection data
- OpenStreetMap rooftop PV installation vectors (reference/validation)
- US Census 2020 tract-level sociodemographic variables
- Geophysical layers (NSRDB irradiance/temperature, land cover)
- High-resolution imagery from Maxar Open Data, Satellogic EarthView, NAIP STAC, and optional Google Solar API

## Setup & Installation

This project follows the repository rule to use `uv` while strictly managing dependencies through `venv` + `pip` workflows (via `uv pip`).

### Prerequisites

- Python 3.12
- Git
- uv (https://docs.astral.sh/uv/)

### Install uv (if needed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew (macOS)
brew install uv

# Or via Winget (Windows)
winget install -e --id astral-sh.uv
```

### Create environment and install dependencies

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
```

### Activate environment

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### Jupyter kernel setup (VS Code / notebooks)

```bash
uv pip install ipykernel
python -m ipykernel install --user --name pv-pr-312 --display-name "Python 3.12 (pv-pr-312)"
```

Then select the `.venv` interpreter/kernel in VS Code notebooks.

### Default DuckDB Artifact

The project now uses a single repo-local DuckDB artifact by default:

```text
data/PR_PV_plan_data.duckdb
```

The `.env` file sets `VECTOR_DB=data/PR_PV_plan_data.duckdb`. Override it only
when you intentionally want to write project tables somewhere else.

## Development Workflow Rules (Implemented)

- Notebooks are authored first as numbered `.py` scripts using Jupytext light format.
- `.ipynb` counterparts are produced for execution and presentation.
- Complex wrangling/segmentation logic is progressively refactored into `utils/`.
- Critical visual outputs are exported to `outputs/figures/` and `outputs/maps/`.
- A root, narrative-oriented E2E notebook is maintained for policy/planning communication.

## Recommended Run Order

Use the following order for a clean rerun from scratch when you want to refresh
all local vector, tabular, STAC, Contextily-training, and inference artifacts
without relying on Google Solar.

1. `notebooks/vectors/01_census_geometries_ingest.py` — loads municipalities, tracts, block groups, and blocks into DuckDB.
2. `notebooks/vectors/02_osm_pv_ingestion_and_viz.py` — ingests rooftop PV labels and updates `has_PV` flags.
3. `notebooks/vectors/03_overture_buildings_ingest.py` — ingests whole-island Overture buildings and materializes H3 support tables.
4. `notebooks/tabular/04_acs_5year_ingest.py` — persists the local ACS 2020 and 2024 5-year county, tract, and block-group slices.
5. `notebooks/tabular/05_urban_blocks_2020_ingest.py` — builds the curated 2020 urban-block reference and the urban summary views used downstream.
6. `notebooks/vectors/06_bg_tile_manifest.py` — builds the H3-based solar tile manifest from Overture + OSM labels.
7. `notebooks/rasters/08_pr_raster_catalog_indexes.py` — builds the local Puerto Rico NAIP STAC GeoParquet, consolidates the raster catalog, and runs vector-guided preview checks.
8. `notebooks/rasters/09_pr_stac_municipality_fetch.py` — fetches and clips the local STAC rasters for San Juan and Isabela.
9. `notebooks/rasters/10_geoai_training_data.py` — builds the Contextily + OSM training dataset directly from the H3 manifest.
10. `notebooks/rasters/11_geoai_solar_finetune.py` — fine-tunes Mask R-CNN on the paired Contextily chips.
11. `notebooks/rasters/12_geoai_solar_inference.py` — runs inference over the local raster catalog with the fine-tuned model.
12. `notebooks/tabular/13_pv_building_join.py` — joins building footprints to OSM and model-detected PV signals.
13. `notebooks/tabular/14_pv_bg_aggregation.py` — aggregates building/PV/ACS/urban-flag metrics to the block-group level.
14. `notebooks/extra/06_pr_poster_figures.py` — optional publication-style figures after the core tables exist.
15. `00_e2e_master_narrative.py` — optional narrative pass once the upstream tables and artifacts are ready.

### Optional Google Solar Branch

If you want Google Solar assets in the mixed local raster catalog, insert this
branch after step 6:

1. `notebooks/rasters/05_solar_quality_probe.py` — optional quota/quality check before a larger API run.
2. `notebooks/rasters/06_google_solar_api_ingest.py` — downloads Google Solar Data Layers GeoTIFFs.
3. `notebooks/rasters/08_solar_raster_catalog.py` — rebuilds the source-aware local raster catalog to include Google Solar outputs.

After that branch, continue with steps 10–16. The Contextily training dataset
does not require Google Solar, but inference can incorporate Google Solar tiles
once step 8 has refreshed the mixed local catalog.

## Repository Skeleton

```text
Proyecto Final/
├── ASVN_AI_Planning_Proposal_Final_Draft.md
├── project_rules.md
├── README.md
├── requirements.txt
├── 00_e2e_master_narrative.py
├── data/
│   ├── vectors/
│   ├── rasters/
│   └── tabular/
├── notebooks/
│   ├── vectors/
│   ├── rasters/
│   └── tabular/
├── outputs/
│   ├── figures/
│   └── maps/
└── utils/
```

## Status

Repository workflow is active. Use the run order above as the current reference
for full refreshes and end-to-end reruns.
