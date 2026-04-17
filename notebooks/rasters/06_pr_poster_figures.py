# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3.12 (pv-pr-312)
#     language: python
#     name: python3
# ---

# + [markdown] magic_args="[markdown]"
# # Puerto Rico Poster Figures
#
# This notebook/script assembles the three poster-ready figures that were still
# missing from the project workflow:
#
# 1. A static quarterly distributed-generation capacity chart for `2017-Q1` to
#    `2025-Q4` using the March 2026 NEPR Exhibit 2 workbook.
# 2. A top-5 municipality summary table joining OSM rooftop PV labels, Overture
#    buildings, census geography counts, and raster availability metrics.
# 3. A hardcoded three-panel example that combines one building footprint, one
#    readable raster chip, a binary mask derived from OSM panel rows, and a
#    smoothed array vector overlay.
# -

# %%
"""06_pr_poster_figures.py

Jupytext-friendly notebook script for generating the remaining research-poster
figures from the Puerto Rico rooftop PV workflow.
"""

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nest_asyncio
import pandas as pd
from dotenv import load_dotenv
from IPython.display import display


def resolve_project_root(start: Path | None = None) -> Path:
    """Find repository root regardless of active notebook directory."""

    current = (start or Path.cwd()).resolve()
    markers = ("project_rules.md", ".git")
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return current


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
nest_asyncio.apply()

from utils.poster_figure_helpers import PREFERRED_POSTER_SOURCES
from utils.poster_figure_helpers import load_or_materialize_consolidated_catalog
from utils.poster_figure_helpers import plot_example_candidate_contact_sheet
from utils.poster_figure_helpers import plot_three_panel_poster_example
from utils.poster_figure_helpers import plot_top_municipality_table
from utils.poster_figure_helpers import prepare_poster_chip_examples
from utils.poster_figure_helpers import save_figure_variants
from utils.poster_figure_helpers import select_review_candidate_ilocs
from utils.poster_figure_helpers import summarize_example_metadata
from utils.poster_figure_helpers import summarize_top_municipalities_for_poster
from utils.raster_stac_index import create_duckdb_connection
from utils.raster_stac_index import resolve_vector_db_path
from utils.ref_pr_pv_capacity_plot import DEFAULT_EXHIBIT_2_URL
from utils.ref_pr_pv_capacity_plot import build_capacity_clients_bar_figure
from utils.ref_pr_pv_capacity_plot import build_capacity_growth_combo_figure
from utils.ref_pr_pv_capacity_plot import build_poster_capacity_figure
from utils.ref_pr_pv_capacity_plot import load_quarterly_capacity_records
from utils.ref_pr_pv_capacity_plot import save_capacity_figure_variants


FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_DIR = FIGURES_DIR / "review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
TOP24_REVIEW_DIR = REVIEW_DIR / "top24_candidates"
TOP24_REVIEW_DIR.mkdir(parents=True, exist_ok=True)

WORKBOOK_URL = DEFAULT_EXHIBIT_2_URL
QUARTER_START = "2017-01-01"
QUARTER_END = "2025-12-31"

CAPACITY_OUTPUT_STEM = FIGURES_DIR / "pr_pv_capacity_quarterly_2017_2025"
CAPACITY_CLIENTS_OUTPUT_STEM = FIGURES_DIR / "pr_pv_capacity_vs_clients"
CAPACITY_GROWTH_OUTPUT_STEM = FIGURES_DIR / "pr_pv_capacity_vs_growth"
TABLE_OUTPUT_STEM = FIGURES_DIR / "pr_top5_municipality_summary"
EXAMPLE_OUTPUT_STEM = FIGURES_DIR / "pr_pv_three_panel_example"
REVIEW_CONTACT_SHEET_STEM = REVIEW_DIR / "pr_pv_candidate_contact_sheet"
REVIEW_MANIFEST_PATH = REVIEW_DIR / "pr_pv_candidate_manifest.csv"
TOP24_CONTACT_SHEET_STEM = REVIEW_DIR / "pr_pv_candidate_contact_sheet_top24"
TOP24_MANIFEST_PATH = TOP24_REVIEW_DIR / "pr_pv_candidate_manifest_top24.csv"

REVIEW_EXPLICIT_ILOCS = [0, 1, 2]
REVIEW_RANDOM_SEED = 42
REVIEW_RANDOM_COUNT = 3
SELECTED_EXAMPLE_ILOC = 0
EXCLUDED_BLURRY_INITIAL_RANKS = [1, 2, 3, 4, 5]
TOP24_CANDIDATE_COUNT = 24
TOP24_CONTACT_SHEET_COLUMNS = 8
MAXAR_TOP5_CANDIDATE_COUNT = 5
MAXAR_TOP5_OFFSET = 5
MAXAR_TOP5_CONTACT_SHEET_COLUMNS = 5
MAXAR_ONLY_SOURCES = ["maxar_open_data"]
MAXAR_CANDIDATE_SCAN_LIMIT = 5000
MAXAR_QUERY_BATCH_SIZE = 500

MAXAR_TOP5_RANGE_START = MAXAR_TOP5_OFFSET
MAXAR_TOP5_RANGE_END = MAXAR_TOP5_OFFSET + MAXAR_TOP5_CANDIDATE_COUNT - 1
MAXAR_TOP5_RANGE_LABEL = f"maxar_candidates_{MAXAR_TOP5_RANGE_START:02d}_{MAXAR_TOP5_RANGE_END:02d}"
MAXAR_TOP5_REVIEW_DIR = REVIEW_DIR / MAXAR_TOP5_RANGE_LABEL
MAXAR_TOP5_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
MAXAR_TOP5_CONTACT_SHEET_STEM = REVIEW_DIR / f"pr_pv_candidate_contact_sheet_{MAXAR_TOP5_RANGE_LABEL}"
MAXAR_TOP5_MANIFEST_PATH = MAXAR_TOP5_REVIEW_DIR / f"pr_pv_candidate_manifest_{MAXAR_TOP5_RANGE_LABEL}.csv"

# + [markdown] magic_args="[markdown]"
# ## Figure 1: Quarterly PV Capacity
#
# We use the remote NEPR workbook directly, cache it locally via the shared
# helper, and keep one quarter-end observation per quarter from `2017-Q1`
# through `2025-Q4`.
# -

# %%
quarterly_capacity = load_quarterly_capacity_records(
    WORKBOOK_URL,
    start=QUARTER_START,
    end=QUARTER_END,
)
display(quarterly_capacity.tail(12))
print(quarterly_capacity[["quarter_key", "capacity_mw", "client_count"]].tail(8).to_string(index=False))

capacity_figure = build_poster_capacity_figure(quarterly_capacity)
capacity_clients_figure = build_capacity_clients_bar_figure(quarterly_capacity)
capacity_growth_figure = build_capacity_growth_combo_figure(quarterly_capacity)

capacity_output_paths = save_capacity_figure_variants(capacity_figure, CAPACITY_OUTPUT_STEM)
capacity_clients_output_paths = save_capacity_figure_variants(capacity_clients_figure, CAPACITY_CLIENTS_OUTPUT_STEM)
capacity_growth_output_paths = save_capacity_figure_variants(capacity_growth_figure, CAPACITY_GROWTH_OUTPUT_STEM)
plt.close(capacity_figure)
plt.close(capacity_clients_figure)
plt.close(capacity_growth_figure)
quarterly_capacity_csv_path = CAPACITY_OUTPUT_STEM.with_suffix(".csv")
quarterly_capacity.to_csv(quarterly_capacity_csv_path, index=False)

print(f"Capacity chart PNG: {capacity_output_paths['png']}")
print(f"Capacity chart SVG: {capacity_output_paths['svg']}")
print(f"Capacity vs clients PNG: {capacity_clients_output_paths['png']}")
print(f"Capacity vs clients SVG: {capacity_clients_output_paths['svg']}")
print(f"Capacity vs growth PNG: {capacity_growth_output_paths['png']}")
print(f"Capacity vs growth SVG: {capacity_growth_output_paths['svg']}")
print(f"Quarterly capacity data CSV: {quarterly_capacity_csv_path}")

# + [markdown] magic_args="[markdown]"
# ## Load the Consolidated Raster Catalog
#
# This reuses the source-aware consolidated catalog from the raster indexing
# utilities. If the GeoParquet is missing locally, the notebook materializes it
# on demand from NAIP, Maxar Open Data, and EarthView.
# -

# %%
catalog_gdf, catalog_summary, catalog_path = load_or_materialize_consolidated_catalog()
if catalog_summary is not None:
    display(catalog_summary)

print(f"Raster catalog path: {catalog_path}")
print(f"Raster catalog rows: {len(catalog_gdf):,}")
display(catalog_gdf.groupby("source").size().rename("item_rows").reset_index())

# + [markdown] magic_args="[markdown]"
# ## Figure 2: Top-5 Municipality Summary Table
#
# The table keeps the user-requested ranking basis, which is raw OSM rooftop PV
# polygon count, and augments each municipality with Overture building counts,
# estimated share of buildings with PV labels, census block-group / tract counts,
# and raster coverage metrics.
# -

# %%
vector_db_path = resolve_vector_db_path()
vector_con = create_duckdb_connection(db_path=vector_db_path, read_only=True)
municipality_summary = summarize_top_municipalities_for_poster(vector_con, catalog_gdf, top_n=5)
display(municipality_summary)
print(
    municipality_summary[
        [
            "municipality_name",
            "municipality_area_km2",
            "pv_feature_count",
            "building_count",
            "pv_building_pct",
            "block_group_count",
            "tract_count",
            "imagery_resolution_label",
            "rooftop_area_km2",
        ]
    ].to_string(index=False)
)

municipality_table_figure = plot_top_municipality_table(municipality_summary)
municipality_table_paths = save_figure_variants(municipality_table_figure, TABLE_OUTPUT_STEM)
plt.close(municipality_table_figure)
municipality_summary_csv_path = TABLE_OUTPUT_STEM.with_suffix(".csv")
municipality_summary.to_csv(municipality_summary_csv_path, index=False)

print(f"Top municipality table PNG: {municipality_table_paths['png']}")
print(f"Top municipality table SVG: {municipality_table_paths['svg']}")
print(f"Top municipality summary CSV: {municipality_summary_csv_path}")

# + [markdown] magic_args="[markdown]"
# ## Figure 3: Hardcoded Three-Panel Example
#
# This step first creates a scored review set so we can cycle through explicit
# `iloc` picks and seeded random samples, then saves the final selected rooftop
# example as the poster-ready image, mask, and vector overlay sequence.
# -

# %%
candidate_manifest, candidate_examples = prepare_poster_chip_examples(
    vector_con,
    catalog_gdf,
    preferred_sources=PREFERRED_POSTER_SOURCES,
    excluded_initial_ranks=EXCLUDED_BLURRY_INITIAL_RANKS,
)
display(candidate_manifest.head(12))
candidate_manifest.to_csv(REVIEW_MANIFEST_PATH, index=False)

top24_candidate_ilocs = candidate_manifest["candidate_iloc"].astype(int).head(min(TOP24_CANDIDATE_COUNT, len(candidate_manifest))).tolist()
top24_candidate_manifest = candidate_manifest[candidate_manifest["candidate_iloc"].isin(top24_candidate_ilocs)].copy()
top24_candidate_manifest.to_csv(TOP24_MANIFEST_PATH, index=False)

review_candidate_ilocs = select_review_candidate_ilocs(
    candidate_manifest,
    explicit_ilocs=REVIEW_EXPLICIT_ILOCS,
    random_seed=REVIEW_RANDOM_SEED,
    random_count=REVIEW_RANDOM_COUNT,
)
print(f"Review candidate ilocs: {review_candidate_ilocs}")

candidate_contact_sheet = plot_example_candidate_contact_sheet(
    candidate_examples,
    candidate_manifest,
    review_candidate_ilocs,
)
candidate_contact_sheet_paths = save_figure_variants(candidate_contact_sheet, REVIEW_CONTACT_SHEET_STEM)
plt.close(candidate_contact_sheet)

top24_contact_sheet = plot_example_candidate_contact_sheet(
    candidate_examples,
    candidate_manifest,
    top24_candidate_ilocs,
    title=f"Top {len(top24_candidate_ilocs)} poster candidates",
    columns=TOP24_CONTACT_SHEET_COLUMNS,
)
top24_contact_sheet_paths = save_figure_variants(top24_contact_sheet, TOP24_CONTACT_SHEET_STEM)
plt.close(top24_contact_sheet)

for candidate_iloc in review_candidate_ilocs:
    review_figure = plot_three_panel_poster_example(candidate_examples[candidate_iloc])
    review_output_stem = REVIEW_DIR / f"pr_pv_three_panel_candidate_{candidate_iloc:02d}"
    save_figure_variants(review_figure, review_output_stem)
    plt.close(review_figure)

for candidate_iloc in top24_candidate_ilocs:
    top24_figure = plot_three_panel_poster_example(candidate_examples[candidate_iloc])
    top24_output_stem = TOP24_REVIEW_DIR / f"pr_pv_three_panel_candidate_{candidate_iloc:02d}"
    save_figure_variants(top24_figure, top24_output_stem)
    plt.close(top24_figure)

maxar_manifest, maxar_examples = prepare_poster_chip_examples(
    vector_con,
    catalog_gdf,
    preferred_sources=MAXAR_ONLY_SOURCES,
    top_municipality_limit=None,
    candidate_limit=MAXAR_TOP5_OFFSET + MAXAR_TOP5_CANDIDATE_COUNT,
    candidate_search_limit=MAXAR_CANDIDATE_SCAN_LIMIT,
    query_batch_size=MAXAR_QUERY_BATCH_SIZE,
    require_preferred_source=True,
)
maxar_top5_ilocs = maxar_manifest["candidate_iloc"].astype(int).iloc[
    MAXAR_TOP5_OFFSET : MAXAR_TOP5_OFFSET + MAXAR_TOP5_CANDIDATE_COUNT
].tolist()
if not maxar_top5_ilocs:
    raise RuntimeError(
        f"No Maxar candidates were available for iloc range {MAXAR_TOP5_RANGE_START}-{MAXAR_TOP5_RANGE_END}."
    )
maxar_top5_manifest = maxar_manifest[maxar_manifest["candidate_iloc"].isin(maxar_top5_ilocs)].copy()
display(maxar_top5_manifest)
maxar_top5_manifest.to_csv(MAXAR_TOP5_MANIFEST_PATH, index=False)

maxar_top5_contact_sheet = plot_example_candidate_contact_sheet(
    maxar_examples,
    maxar_manifest,
    maxar_top5_ilocs,
    title=(
        f"Maxar-only poster candidates #{MAXAR_TOP5_RANGE_START}"
        f"-{MAXAR_TOP5_RANGE_START + len(maxar_top5_ilocs) - 1}"
    ),
    columns=MAXAR_TOP5_CONTACT_SHEET_COLUMNS,
)
maxar_top5_contact_sheet_paths = save_figure_variants(maxar_top5_contact_sheet, MAXAR_TOP5_CONTACT_SHEET_STEM)
plt.close(maxar_top5_contact_sheet)

for candidate_iloc in maxar_top5_ilocs:
    maxar_top5_figure = plot_three_panel_poster_example(maxar_examples[candidate_iloc])
    maxar_top5_output_stem = MAXAR_TOP5_REVIEW_DIR / f"pr_pv_three_panel_candidate_{candidate_iloc:02d}"
    save_figure_variants(maxar_top5_figure, maxar_top5_output_stem)
    plt.close(maxar_top5_figure)

poster_example = candidate_examples[SELECTED_EXAMPLE_ILOC]
example_metadata = summarize_example_metadata(poster_example)
display(example_metadata)
print(example_metadata.to_string(index=False))

poster_example_figure = plot_three_panel_poster_example(poster_example)
poster_example_paths = save_figure_variants(poster_example_figure, EXAMPLE_OUTPUT_STEM)
plt.close(poster_example_figure)
example_metadata_csv_path = EXAMPLE_OUTPUT_STEM.with_suffix(".csv")
example_metadata.to_csv(example_metadata_csv_path, index=False)

print(f"Candidate manifest CSV: {REVIEW_MANIFEST_PATH}")
print(f"Candidate contact sheet PNG: {candidate_contact_sheet_paths['png']}")
print(f"Candidate contact sheet SVG: {candidate_contact_sheet_paths['svg']}")
print(f"Top-24 candidate manifest CSV: {TOP24_MANIFEST_PATH}")
print(f"Top-24 contact sheet PNG: {top24_contact_sheet_paths['png']}")
print(f"Top-24 contact sheet SVG: {top24_contact_sheet_paths['svg']}")
print(f"Top-24 candidate triptych folder: {TOP24_REVIEW_DIR}")
print(f"Maxar top-5 manifest CSV: {MAXAR_TOP5_MANIFEST_PATH}")
print(f"Maxar top-5 contact sheet PNG: {maxar_top5_contact_sheet_paths['png']}")
print(f"Maxar top-5 contact sheet SVG: {maxar_top5_contact_sheet_paths['svg']}")
print(f"Maxar top-5 candidate triptych folder: {MAXAR_TOP5_REVIEW_DIR}")
print(f"Three-panel example PNG: {poster_example_paths['png']}")
print(f"Three-panel example SVG: {poster_example_paths['svg']}")
print(f"Three-panel example metadata CSV: {example_metadata_csv_path}")

# %%
vector_con.close()
