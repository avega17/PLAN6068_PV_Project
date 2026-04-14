# %% [markdown]
# # Puerto Rico Consolidated Raster Catalog and Vector-Guided Preview
# 
# This notebook does two jobs:
# 1. Materialize one final Puerto Rico AOI-filtered raster catalog GeoParquet.
# 2. Walk through concrete vector-guided previews so we can confirm the catalog is
#    useful for downstream PV and building chip extraction.
# 
# The raster sources stay the same as the current consolidated workflow:
# - NAIP 2021 Puerto Rico via static STAC JSON crawl.
# - Maxar Open Data via remote STAC GeoParquet queried with DuckDB.
# - Satellogic Earthview via remote STAC GeoParquet queried with DuckDB.
# 
# Important note on Maxar filtering:
# - The current pipeline does **not** keep every item from a Maxar event once an
#   event intersects Puerto Rico.
# - The old event-manifest approach is no longer used here.
# - The new consolidated pipeline queries the public Maxar GeoParquet directly and
#   only keeps individual STAC items whose footprints intersect the Puerto Rico AOI.

# %%
"""05_pr_raster_catalog_indexes.py

Jupytext-friendly notebook script for consolidated Puerto Rico raster catalog
materialization plus vector-guided raster preview steps.
"""

# %%
from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import nest_asyncio
import numpy as np
import pandas as pd
import rasterio
from dotenv import load_dotenv
from IPython.display import display
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform, transform_bounds
from shapely.geometry import Point


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

from utils.raster_stac_index import EARTHVIEW_PUBLIC_PARQUET_URI
from utils.raster_stac_index import MAXAR_PUBLIC_PARQUET_URI
from utils.raster_stac_index import create_duckdb_connection
from utils.raster_stac_index import load_puerto_rico_boundary
from utils.raster_stac_index import materialize_consolidated_pr_raster_catalog
from utils.raster_stac_index import resolve_vector_db_path


OUTPUT_CRS = "EPSG:4326"
USER_SAMPLE_SEED: int | None = None
BLOCK_PREVIEW_SIZE = 512
CHIP_SIZE = 512
MUNICIPALITY_PV_PREVIEW_LIMIT = 600
BUILDING_PREVIEW_LIMIT = 600
PREFERRED_PREVIEW_SOURCES = ["naip_2021_pr", "maxar_open_data"]

ACTIVE_SAMPLE_SEED = (
    USER_SAMPLE_SEED
    if USER_SAMPLE_SEED is not None
    else int(pd.Timestamp.utcnow().value % (2**32 - 1))
)

RASTER_STAC_DIR = PROJECT_ROOT / "data" / "rasters" / "stac"
RASTER_STAC_DIR.mkdir(parents=True, exist_ok=True)
CONSOLIDATED_OUTPUT_PATH = RASTER_STAC_DIR / "pr_raster_catalog_items.parquet"

NAIP_2021_CATALOG_URL = (
    "https://coastalimagery.blob.core.windows.net/digitalcoast/"
    "PR_NAIP_2021_9825/stac/catalog.json"
)
MAXAR_REMOTE_PARQUET_URL = MAXAR_PUBLIC_PARQUET_URI
EARTHVIEW_REMOTE_PARQUET_URL = EARTHVIEW_PUBLIC_PARQUET_URI

boundary = load_puerto_rico_boundary()
print(f"Puerto Rico boundary source: {boundary.source}")
print(f"Puerto Rico bounds: {boundary.bounds}")
print(f"Vector DuckDB path: {resolve_vector_db_path()}")
print(f"Active sample seed: {ACTIVE_SAMPLE_SEED}")


def table_exists(con, table_name: str) -> bool:
    """Return True when a table exists in the main schema."""

    row = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
        """,
        [table_name],
    ).fetchone()
    return bool(row and row[0])


def _to_wkb_bytes(value: object) -> bytes:
    """Normalize WKB values fetched from DuckDB."""

    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def fetch_geodataframe(con, query: str, params: list[object] | None = None) -> gpd.GeoDataFrame:
    """Run a SQL query that returns `geometry_wkb` and convert it to GeoPandas."""

    frame = con.execute(query, params or []).fetchdf()
    if frame.empty:
        return gpd.GeoDataFrame(
            frame.drop(columns=["geometry_wkb"], errors="ignore"),
            geometry=gpd.GeoSeries([], crs=OUTPUT_CRS),
            crs=OUTPUT_CRS,
        )

    geometry = gpd.GeoSeries.from_wkb(frame["geometry_wkb"].map(_to_wkb_bytes), crs=OUTPUT_CRS)
    return gpd.GeoDataFrame(frame.drop(columns=["geometry_wkb"]), geometry=geometry, crs=OUTPUT_CRS)


def choose_raster_asset(item_row: pd.Series) -> str | None:
    """Pick the best raster asset href available for preview/chip extraction."""

    for column_name in ["visual_asset_href", "analytic_asset_href"]:
        href = item_row.get(column_name)
        if isinstance(href, str) and href.strip():
            return href
    return None


def derive_random_state(sample_label: str) -> int:
    """Create a stable random state derived from the active notebook seed."""

    digest = hashlib.sha256(f"{ACTIVE_SAMPLE_SEED}:{sample_label}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def sample_rows(frame: pd.DataFrame | gpd.GeoDataFrame, sample_label: str, n: int = 1):
    """Sample rows reproducibly from the active notebook seed."""

    if frame.empty:
        return frame
    return frame.sample(n=min(n, len(frame)), random_state=derive_random_state(sample_label))


def normalize_image(array: np.ndarray) -> np.ndarray:
    """Contrast-stretch arrays for quick notebook visualization."""

    if array.ndim == 2:
        valid = array[np.isfinite(array)]
        if valid.size == 0:
            return np.zeros_like(array, dtype=float)
        lower, upper = np.percentile(valid, [2, 98])
        if upper <= lower:
            upper = lower + 1.0
        return np.clip((array - lower) / (upper - lower), 0.0, 1.0)

    stretched_bands = [normalize_image(array[band_index]) for band_index in range(array.shape[0])]
    return np.dstack(stretched_bands)


def read_raster_preview_from_geometry(
    item_row: pd.Series,
    geometry,
    out_size: int = BLOCK_PREVIEW_SIZE,
) -> dict[str, object] | None:
    """Read a quicklook raster preview for a geometry extent."""

    asset_href = choose_raster_asset(item_row)
    if asset_href is None:
        return None

    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(asset_href) as src:
            minx, miny, maxx, maxy = geometry.bounds
            if src.crs and src.crs.to_string() != OUTPUT_CRS:
                minx, miny, maxx, maxy = transform_bounds(OUTPUT_CRS, src.crs, minx, miny, maxx, maxy, densify_pts=21)

            window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            full_window = Window(0, 0, src.width, src.height)
            window = window.intersection(full_window)
            if window.width <= 0 or window.height <= 0:
                return None

            bands = [1, 2, 3] if src.count >= 3 else [1]
            data = src.read(
                bands,
                window=window,
                out_shape=(len(bands), out_size, out_size),
                resampling=Resampling.bilinear,
            )

    return {
        "asset_href": asset_href,
        "bands": bands,
        "image": normalize_image(data[0] if len(bands) == 1 else data),
    }


def read_raster_chip_at_point(
    item_row: pd.Series,
    point: Point,
    chip_size: int = CHIP_SIZE,
) -> dict[str, object] | None:
    """Read an NxN chip centered on a point."""

    asset_href = choose_raster_asset(item_row)
    if asset_href is None:
        return None

    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(asset_href) as src:
            x_coord, y_coord = point.x, point.y
            if src.crs and src.crs.to_string() != OUTPUT_CRS:
                transformed_x, transformed_y = transform(OUTPUT_CRS, src.crs, [x_coord], [y_coord])
                x_coord, y_coord = transformed_x[0], transformed_y[0]

            row_index, col_index = src.index(x_coord, y_coord)
            half_size = chip_size // 2
            window = Window(
                col_off=col_index - half_size,
                row_off=row_index - half_size,
                width=chip_size,
                height=chip_size,
            )

            bands = [1, 2, 3] if src.count >= 3 else [1]
            data = src.read(
                bands,
                window=window,
                out_shape=(len(bands), chip_size, chip_size),
                boundless=True,
                fill_value=0,
                resampling=Resampling.bilinear,
            )

    return {
        "asset_href": asset_href,
        "bands": bands,
        "image": normalize_image(data[0] if len(bands) == 1 else data),
    }


def sort_catalog_candidates(candidates: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Sort candidate raster items by resolution first, then recency."""

    if candidates.empty:
        return candidates

    ranked = candidates.copy()
    ranked["acquired_at"] = pd.to_datetime(ranked["acquired_at"], utc=True, errors="coerce")
    ranked["gsd_sort"] = pd.to_numeric(ranked["gsd"], errors="coerce").fillna(999999.0)
    ranked = ranked.sort_values(
        by=["gsd_sort", "acquired_at", "source", "item_id"],
        ascending=[True, False, True, True],
        na_position="last",
    ).drop(columns=["gsd_sort"])
    return ranked.reset_index(drop=True)


def select_source_candidates(
    candidates: gpd.GeoDataFrame,
    preferred_sources: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Keep the best intersecting raster item per requested source."""

    if candidates.empty:
        return candidates

    sources = preferred_sources or PREFERRED_PREVIEW_SOURCES
    subset = candidates[candidates["source"].isin(sources)].copy()
    if subset.empty:
        return subset
    return subset.groupby("source", sort=False).head(1).reset_index(drop=True)


def row_to_gdf(row: pd.Series) -> gpd.GeoDataFrame:
    """Wrap a single row with geometry into a one-row GeoDataFrame."""

    return gpd.GeoDataFrame([row], geometry="geometry", crs=OUTPUT_CRS)


def plot_vector_context(
    municipality_gdf: gpd.GeoDataFrame,
    title: str,
    block_gdf: gpd.GeoDataFrame | None = None,
    pv_gdf: gpd.GeoDataFrame | None = None,
    building_gdf: gpd.GeoDataFrame | None = None,
    raster_item_gdf: gpd.GeoDataFrame | None = None,
) -> None:
    """Plot a compact vector context panel for notebook inspection."""

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    municipality_gdf.boundary.plot(ax=ax, color="#0f172a", linewidth=1.25, label="Municipality")

    if raster_item_gdf is not None and not raster_item_gdf.empty:
        raster_item_gdf.boundary.plot(ax=ax, color="#2563eb", linewidth=1.1, label="Raster footprint")

    if block_gdf is not None and not block_gdf.empty:
        block_gdf.boundary.plot(ax=ax, color="#dc2626", linewidth=1.4, label="Sampled census unit")

    if pv_gdf is not None and not pv_gdf.empty:
        pv_gdf.plot(ax=ax, color="#f97316", alpha=0.55, linewidth=0.2, label="Solar panel vectors")

    if building_gdf is not None and not building_gdf.empty:
        building_gdf.plot(ax=ax, color="#10b981", alpha=0.45, linewidth=0.1, label="Building footprints")

    ax.set_title(title)
    ax.set_axis_off()
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        seen = set()
        unique_handles = []
        unique_labels = []
        for handle, label in zip(handles, labels):
            if label not in seen:
                seen.add(label)
                unique_handles.append(handle)
                unique_labels.append(label)
        ax.legend(unique_handles, unique_labels, loc="upper right")
    plt.show()


def plot_raster_preview(preview: dict[str, object], title: str) -> None:
    """Display a raster preview array."""

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    cmap = None if np.asarray(preview["image"]).ndim == 3 else "gray"
    ax.imshow(preview["image"], cmap=cmap)
    ax.set_title(title)
    ax.set_axis_off()
    plt.show()


def plot_multiple_raster_previews(
    previews: list[tuple[pd.Series, dict[str, object]]],
    title_prefix: str,
) -> None:
    """Display one preview panel per sampled raster source."""

    if not previews:
        print("No readable raster previews were generated.")
        return

    figure, axes = plt.subplots(1, len(previews), figsize=(6 * len(previews), 6), constrained_layout=True)
    if len(previews) == 1:
        axes = [axes]

    for axis, (candidate_row, preview) in zip(axes, previews):
        cmap = None if np.asarray(preview["image"]).ndim == 3 else "gray"
        axis.imshow(preview["image"], cmap=cmap)
        axis.set_title(f"{title_prefix}\n{candidate_row['source']}\n{candidate_row['item_id']}")
        axis.set_axis_off()

    plt.show()


def build_geometry_previews(
    candidate_rows: gpd.GeoDataFrame,
    geometry,
) -> list[tuple[pd.Series, dict[str, object]]]:
    """Read one geometry-based preview for each candidate raster row."""

    previews: list[tuple[pd.Series, dict[str, object]]] = []
    for _, candidate_row in candidate_rows.iterrows():
        preview = read_raster_preview_from_geometry(candidate_row, geometry)
        if preview is not None:
            previews.append((candidate_row, preview))
    return previews


def build_point_chip_previews(
    candidate_rows: gpd.GeoDataFrame,
    point: Point,
    chip_size: int = CHIP_SIZE,
) -> list[tuple[pd.Series, dict[str, object]]]:
    """Read one point-centered chip preview for each candidate raster row."""

    previews: list[tuple[pd.Series, dict[str, object]]] = []
    for _, candidate_row in candidate_rows.iterrows():
        preview = read_raster_chip_at_point(candidate_row, point, chip_size=chip_size)
        if preview is not None:
            previews.append((candidate_row, preview))
    return previews


def choose_lonboard_item(*candidate_frames: gpd.GeoDataFrame) -> pd.Series | None:
    """Pick a NAIP item for full-COG lonboard exploration from prior samples."""

    for candidate_frame in candidate_frames:
        if candidate_frame is None or candidate_frame.empty:
            continue
        naip_rows = candidate_frame[candidate_frame["source"] == "naip_2021_pr"]
        if not naip_rows.empty:
            return naip_rows.iloc[0]
    return None

# %% [markdown]
# ## Materialize the Consolidated Puerto Rico Raster Catalog
# 
# Output written by this cell:
# - `data/rasters/stac/pr_raster_catalog_items.parquet`
# 
# The summary table confirms how many **individual items** survived the Puerto Rico
# AOI filter per source. For Maxar, that means item footprints intersect Puerto Rico;
# it does **not** mean we retained every item from a qualifying event collection.

# %%
SUMMARY_FRAME = asyncio.run(
    materialize_consolidated_pr_raster_catalog(
        output_path=CONSOLIDATED_OUTPUT_PATH,
        boundary=boundary,
        naip_catalog_url=NAIP_2021_CATALOG_URL,
        maxar_remote_parquet_url=MAXAR_REMOTE_PARQUET_URL,
        earthview_remote_parquet_url=EARTHVIEW_REMOTE_PARQUET_URL,
    )
)
display(SUMMARY_FRAME)
print(f"Consolidated catalog output: {CONSOLIDATED_OUTPUT_PATH}")

# %% [markdown]
# ## Inspect the Consolidated Raster Catalog
# 
# Before sampling vectors, we load the catalog itself, confirm source coverage, and
# preview the item footprints we now have for Puerto Rico.

# %%
catalog_gdf = gpd.read_parquet(CONSOLIDATED_OUTPUT_PATH)
if catalog_gdf.crs is None:
    catalog_gdf = catalog_gdf.set_crs(OUTPUT_CRS)
else:
    catalog_gdf = catalog_gdf.to_crs(OUTPUT_CRS)

catalog_gdf["acquired_at"] = pd.to_datetime(catalog_gdf["acquired_at"], utc=True, errors="coerce")
print(f"Catalog rows: {len(catalog_gdf):,}")
display(catalog_gdf.groupby("source").size().rename("item_rows").reset_index())
display(
    catalog_gdf[
        [
            "source",
            "item_id",
            "collection_id",
            "acquired_at",
            "gsd",
            "visual_asset_href",
            "analytic_asset_href",
        ]
    ].head(12)
)

fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
gpd.GeoSeries([boundary.geometry], crs=OUTPUT_CRS).boundary.plot(ax=ax, color="#0f172a", linewidth=1.5)
catalog_gdf.groupby("source").head(120).plot(ax=ax, alpha=0.25, column="source", legend=True)
ax.set_title("Sample of consolidated Puerto Rico raster footprints by source")
ax.set_axis_off()
plt.show()

# %% [markdown]
# ## Sample a Municipality from the Top 10 PV Municipalities
# 
# We start from the vector side instead of the raster side. The sample below chooses
# one municipality from the top 10 by rooftop PV polygon count, then previews that
# municipality together with a random subset of its PV vectors.
#
# Sampling note:
# - Set `USER_SAMPLE_SEED` near the top of the notebook to reproduce a specific run.
# - Leave it as `None` to get a fresh seed on each execution.

# %%
vector_db_path = resolve_vector_db_path()
vector_con = create_duckdb_connection(db_path=vector_db_path, read_only=True)

top_municipalities = vector_con.execute(
    """
    SELECT
        municipality_name,
        COUNT(*) AS pv_feature_count
    FROM pr_osm_rooftop_pv_polygons
    GROUP BY 1
    ORDER BY pv_feature_count DESC, municipality_name
    LIMIT 10
    """
).fetchdf()

if top_municipalities.empty:
    raise RuntimeError("No rooftop PV municipalities were found in pr_osm_rooftop_pv_polygons.")

selected_municipality_name = sample_rows(top_municipalities, "top-municipality", n=1).iloc[0]["municipality_name"]
print(f"Sampled municipality from top 10 PV municipalities: {selected_municipality_name}")
display(top_municipalities)

municipality_gdf = fetch_geodataframe(
    vector_con,
    """
    SELECT
        GEOID AS municipality_geoid,
        NAME AS municipality_name,
        ST_AsWKB(geometry) AS geometry_wkb
    FROM pr_municipalities
    WHERE NAME = ?
    """,
    [selected_municipality_name],
)

municipality_pv_gdf = fetch_geodataframe(
    vector_con,
    """
    SELECT
        feature_id,
        municipality_name,
        municipality_geoid,
        ST_AsWKB(geometry) AS geometry_wkb
    FROM pr_osm_rooftop_pv_polygons
    WHERE municipality_name = ?
    """,
    [selected_municipality_name],
)
municipality_pv_gdf = sample_rows(
    municipality_pv_gdf,
    f"municipality-pv-{selected_municipality_name}",
    n=MUNICIPALITY_PV_PREVIEW_LIMIT,
)

plot_vector_context(
    municipality_gdf=municipality_gdf,
    title=f"{selected_municipality_name}: municipality boundary with sampled rooftop PV polygons",
    pv_gdf=municipality_pv_gdf,
)

# %% [markdown]
# ## Sample a Census Unit Inside that Municipality
# 
# The current workspace uses census tracts and block groups. There is no census
# block table in this project, so the finer sampled census unit is always drawn
# from `pr_block_groups`.

# %%
census_unit_table = "pr_block_groups"
census_unit_label = "block group"

sampled_census_unit_gdf = fetch_geodataframe(
    vector_con,
    f"""
    SELECT
        g.GEOID AS census_unit_geoid,
        coalesce(g.NAME, g.GEOID) AS census_unit_name,
        ST_AsWKB(g.geometry) AS geometry_wkb
    FROM {census_unit_table} AS g
    JOIN pr_municipalities AS m
      ON ST_Intersects(m.geometry, g.geometry)
    WHERE m.NAME = ?
    """,
    [selected_municipality_name],
)
sampled_census_unit_gdf = sample_rows(
    sampled_census_unit_gdf,
    f"census-unit-{selected_municipality_name}",
    n=1,
)

if sampled_census_unit_gdf.empty:
    raise RuntimeError(f"No sampled {census_unit_label} was found inside {selected_municipality_name}.")

print(f"Using census geography table: {census_unit_table} ({census_unit_label})")
display(sampled_census_unit_gdf.drop(columns="geometry"))

block_pv_gdf = fetch_geodataframe(
    vector_con,
    """
    SELECT
        feature_id,
        municipality_name,
        municipality_geoid,
        ST_AsWKB(pv.geometry) AS geometry_wkb
    FROM pr_osm_rooftop_pv_polygons AS pv
    WHERE ST_Intersects(pv.geometry, ST_GeomFromText(?))
    """,
    [sampled_census_unit_gdf.geometry.iloc[0].wkt],
)
block_pv_gdf = sample_rows(block_pv_gdf, "block-group-pv-preview", n=300)

plot_vector_context(
    municipality_gdf=municipality_gdf,
    block_gdf=sampled_census_unit_gdf,
    pv_gdf=block_pv_gdf,
    title=f"Sampled {census_unit_label} inside {selected_municipality_name}",
)

# %% [markdown]
# ## Sample a Raster Item that Intersects the Census Unit
# 
# This is the first raster preview that is directly anchored to already-fetched
# vector data. We search the local consolidated catalog for items whose footprint
# intersects the sampled census unit, rank candidates by smaller `gsd` and newer
# acquisition time, then preview the chosen raster against the same vector context.

# %%
sampled_census_unit_geometry = sampled_census_unit_gdf.geometry.iloc[0]
raster_candidates_for_unit = sort_catalog_candidates(
    catalog_gdf[catalog_gdf.geometry.intersects(sampled_census_unit_geometry)].copy()
)
raster_candidates_for_unit = select_source_candidates(raster_candidates_for_unit)

display(
    raster_candidates_for_unit[
        ["source", "item_id", "collection_id", "acquired_at", "gsd", "visual_asset_href", "analytic_asset_href"]
    ].head(10)
)

if raster_candidates_for_unit.empty:
    raise RuntimeError(f"No raster items intersected the sampled {census_unit_label} geometry.")

print(f"Sampled raster sources for the {census_unit_label}: {', '.join(raster_candidates_for_unit['source'].tolist())}")

plot_vector_context(
    municipality_gdf=municipality_gdf,
    block_gdf=sampled_census_unit_gdf,
    pv_gdf=block_pv_gdf,
    raster_item_gdf=raster_candidates_for_unit,
    title=f"Raster footprint intersecting sampled {census_unit_label}",
)

unit_raster_previews = build_geometry_previews(raster_candidates_for_unit, sampled_census_unit_geometry)
if not unit_raster_previews:
    print("No readable raster preview asset was available for the sampled census unit.")
else:
    plot_multiple_raster_previews(
        unit_raster_previews,
        title_prefix=f"Raster preview over sampled {census_unit_label}",
    )
    print("Readable raster preview assets for the sampled census unit:")
    for candidate_row, preview in unit_raster_previews:
        print(f"- {candidate_row['source']}: {preview['asset_href']}")

# %% [markdown]
# ## Fetch an NxN Raster Chip from a Sampled Solar Panel Centroid
# 
# We now move from polygon-overlap preview to point-centered chip extraction. The
# sampled solar panel vector is drawn from the same census unit when possible, and
# the STAC search is performed against the local consolidated catalog using the
# centroid point.

# %%
sampled_pv_gdf = fetch_geodataframe(
    vector_con,
    """
    SELECT
        feature_id,
        municipality_name,
        municipality_geoid,
        ST_AsWKB(geometry) AS geometry_wkb
    FROM pr_osm_rooftop_pv_polygons
    WHERE ST_Intersects(geometry, ST_GeomFromText(?))
    """,
    [sampled_census_unit_geometry.wkt],
)
sampled_pv_gdf = sample_rows(sampled_pv_gdf, "block-group-pv-chip", n=1)

if sampled_pv_gdf.empty:
    sampled_pv_gdf = fetch_geodataframe(
        vector_con,
        """
        SELECT
            feature_id,
            municipality_name,
            municipality_geoid,
            ST_AsWKB(geometry) AS geometry_wkb
        FROM pr_osm_rooftop_pv_polygons
        WHERE municipality_name = ?
        """,
        [selected_municipality_name],
    )
    sampled_pv_gdf = sample_rows(sampled_pv_gdf, f"municipality-pv-chip-{selected_municipality_name}", n=1)

if sampled_pv_gdf.empty:
    raise RuntimeError("No sampled rooftop PV vector was found for centroid chip extraction.")

pv_centroid = sampled_pv_gdf.geometry.iloc[0].centroid
pv_point_candidates = sort_catalog_candidates(
    catalog_gdf[catalog_gdf.geometry.intersects(pv_centroid)].copy()
)
pv_point_candidates = select_source_candidates(pv_point_candidates)
display(
    pv_point_candidates[
        ["source", "item_id", "collection_id", "acquired_at", "gsd", "visual_asset_href", "analytic_asset_href"]
    ].head(8)
)

if pv_point_candidates.empty:
    raise RuntimeError("No raster items intersected the sampled rooftop PV centroid.")

plot_vector_context(
    municipality_gdf=municipality_gdf,
    block_gdf=sampled_census_unit_gdf,
    pv_gdf=sampled_pv_gdf,
    raster_item_gdf=pv_point_candidates,
    title="Sampled PV polygon centroid with matching raster footprint",
)

pv_chip_previews = build_point_chip_previews(pv_point_candidates, pv_centroid, chip_size=CHIP_SIZE)
if not pv_chip_previews:
    print("No readable raster chip asset was available for the sampled PV centroid.")
else:
    plot_multiple_raster_previews(
        pv_chip_previews,
        title_prefix=f"{CHIP_SIZE}x{CHIP_SIZE} chip centered on sampled PV centroid",
    )
    print("Readable raster chip assets for the sampled PV centroid:")
    for candidate_row, preview in pv_chip_previews:
        print(f"- {candidate_row['source']}: {preview['asset_href']}")

# %% [markdown]
# ## Sample NxN Raster Chips from a Building Footprint Centroid
# 
# The final step repeats the centroid-driven search with Overture building
# footprints. Instead of taking only one best item overall, we keep one best chip
# per source so we can compare how the same building context looks across the
# intersecting raster datasets we have available.

# %%
sampled_building_gdf = fetch_geodataframe(
    vector_con,
    f"""
    SELECT
        id AS building_id,
        municipality_name,
        municipality_geoid,
        ST_AsWKB(geometry) AS geometry_wkb
    FROM pr_overture_buildings
    WHERE ST_Intersects(geometry, ST_GeomFromText(?))
    """,
    [sampled_census_unit_geometry.wkt],
)
sampled_building_gdf = sample_rows(sampled_building_gdf, "block-group-building-chip", n=1)

if sampled_building_gdf.empty:
    sampled_building_gdf = fetch_geodataframe(
        vector_con,
        f"""
        SELECT
            id AS building_id,
            municipality_name,
            municipality_geoid,
            ST_AsWKB(geometry) AS geometry_wkb
        FROM pr_overture_buildings
        WHERE municipality_name = ?
        """,
        [selected_municipality_name],
    )
    sampled_building_gdf = sample_rows(
        sampled_building_gdf,
        f"municipality-building-chip-{selected_municipality_name}",
        n=1,
    )

if sampled_building_gdf.empty:
    raise RuntimeError("No sampled Overture building footprint was found for centroid chip extraction.")

sampled_building_centroid = sampled_building_gdf.geometry.iloc[0].centroid
building_point_candidates = sort_catalog_candidates(
    catalog_gdf[catalog_gdf.geometry.intersects(sampled_building_centroid)].copy()
)
building_point_candidates = select_source_candidates(building_point_candidates)

display(
    building_point_candidates[
        ["source", "item_id", "collection_id", "acquired_at", "gsd", "visual_asset_href", "analytic_asset_href"]
    ]
)

plot_vector_context(
    municipality_gdf=municipality_gdf,
    block_gdf=sampled_census_unit_gdf,
    building_gdf=sampled_building_gdf,
    raster_item_gdf=building_point_candidates,
    title="Sampled building centroid with best intersecting raster item per source",
)

if building_point_candidates.empty:
    print("No intersecting raster items were found for the sampled building centroid.")
else:
    previews = build_point_chip_previews(building_point_candidates, sampled_building_centroid, chip_size=CHIP_SIZE)

    if not previews:
        print("Intersecting raster items were found, but none exposed a readable preview/chip asset.")
    else:
        plot_multiple_raster_previews(
            previews,
            title_prefix=f"{CHIP_SIZE}x{CHIP_SIZE} building-centroid chip",
        )
        print("Readable chip assets used for the building centroid preview:")
        for candidate_row, preview in previews:
            print(f"- {candidate_row['source']}: {preview['asset_href']}")


# %% [markdown]
# ## Explore a Full NAIP COG in Lonboard
#
# This final cell uses Lonboard's COG support so the full raster can be explored
# interactively with pan and zoom. We use a sampled NAIP item because its public
# HTTPS COG is the simplest match for the Lonboard + Async-GeoTIFF workflow.

# %%
lonboard_item = choose_lonboard_item(raster_candidates_for_unit, pv_point_candidates, building_point_candidates)
if lonboard_item is None:
    print("No sampled NAIP item was available for the full COG lonboard preview.")
else:
    try:
        import io

        from async_geotiff import GeoTIFF
        from async_geotiff.utils import reshape_as_image
        from lonboard import Map, RasterLayer
        from lonboard.raster import EncodedImage
        from obstore.store import HTTPStore
        from PIL import Image

        naip_asset_href = choose_raster_asset(lonboard_item)
        print(f"Lonboard COG asset: {naip_asset_href}")

        async def open_lonboard_geotiff(cog_href: str):
            store = HTTPStore()
            return await GeoTIFF.open(cog_href, store=store)

        geotiff = asyncio.run(open_lonboard_geotiff(naip_asset_href))

        def render_tile(tile) -> EncodedImage:
            image_array = reshape_as_image(tile.array.data)
            image_array = np.asarray(image_array)
            if image_array.ndim == 2:
                stretched = (normalize_image(image_array) * 255).astype(np.uint8)
                image = Image.fromarray(stretched, mode="L")
            else:
                image_array = image_array[:, :, : min(3, image_array.shape[2])]
                stretched = (normalize_image(np.moveaxis(image_array, -1, 0)) * 255).astype(np.uint8)
                image = Image.fromarray(stretched, mode="RGB")

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return EncodedImage(data=buffer.getvalue(), media_type="image/png")

        lonboard_layer = RasterLayer.from_geotiff(geotiff, render_tile=render_tile)
        display(Map(layers=[lonboard_layer]))
    except ImportError as exc:
        print(f"Lonboard full-COG preview requires extra packages that are not installed: {exc}")

# %% [markdown]
# ## Reference Notes for Follow-on Raster Preview Work
# 
# These references are the ones requested for retrieval, and they are the main
# patterns worth carrying into later notebook expansion:
# 
# 1. Planetary Computer STAC quickstart
#    - `bbox` and `intersects` are the key search patterns.
#    - Treat STAC items as GeoJSON and convert them into GeoDataFrames for quick
#      metadata analysis.
#    - Item assets can be inspected directly and opened with libraries such as
#      `rioxarray` or `rasterio`.
# 
# 2. EODC thumbnail creation with TiTiler
#    - A preview image can be built from a COG URL using a `/cog/preview` endpoint
#      with parameters such as `rescale`, `nodata`, and `dst_crs`.
#    - That pattern is useful if we want richer remote thumbnails without reading
#      raster windows directly in Python.
# 
# 3. Lonboard COG rendering
#    - `RasterLayer.from_geotiff(...)` now supports streaming COG tiles on demand.
#    - This is a strong next step if we want interactive browser-side inspection of
#      candidate raster items without standing up a tile server.
# 
# 4. STAC + xarray + dask
#    - `odc.stac.load(...)` is the main pattern for multi-item raster cubes when we
#      need more than single-scene preview or chip extraction.
#    - That is more appropriate for temporal stacks or model-ready raster tensors
#      than for lightweight notebook previews.

# %%
vector_con.close()


