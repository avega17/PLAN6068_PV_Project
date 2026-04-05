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

## Development Workflow Rules (Implemented)

- Notebooks are authored first as numbered `.py` scripts using Jupytext light format.
- `.ipynb` counterparts are produced for execution and presentation.
- Complex wrangling/segmentation logic is progressively refactored into `utils/`.
- Critical visual outputs are exported to `outputs/figures/` and `outputs/maps/`.
- A root, narrative-oriented E2E notebook is maintained for policy/planning communication.

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

Repository scaffold initialized. Next step is to create the first numbered Jupytext scripts under `notebooks/vectors/`, `notebooks/rasters/`, and `notebooks/tabular/`.
