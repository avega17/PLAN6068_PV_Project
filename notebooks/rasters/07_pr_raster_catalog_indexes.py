# %% [markdown]
# # Puerto Rico Consolidated Raster Catalog and Vector-Guided Preview
#
# This notebook does two jobs:
# 1. Materialize one final Puerto Rico AOI-filtered raster catalog GeoParquet.
# 2. Walk through concrete vector-guided previews so we can confirm the catalog is
#    useful for downstream PV and building chip extraction.
#
# The raster sources stay the same as the current consolidated workflow:
# - Puerto Rico NAIP via a local STAC GeoParquet built from coastal imagery plus
#   Microsoft Planetary Computer NAIP items for 2021 through 2024.
# - Maxar Open Data via remote STAC GeoParquet queried with DuckDB.
# - Satellogic Earthview via remote STAC GeoParquet queried with DuckDB.

# %%
"""07_pr_raster_catalog_indexes.py

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
from rasterio.windows import Window, bounds as window_bounds_fn, from_bounds
from rasterio.warp import transform, transform_bounds
from shapely.geometry import Point, box as shapely_box


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
METRIC_CRS = "EPSG:32620"
USER_SAMPLE_SEED: int | None = None
BLOCK_PREVIEW_SIZE = 512
CHIP_SIZE = 512
CENSUS_UNIT_PREVIEW_SIZE = 1024
MUNICIPALITY_PV_PREVIEW_LIMIT = 600
BUILDING_PREVIEW_LIMIT = 600
PREFERRED_PREVIEW_SOURCES = ["pr_naip", "naip_2021_pr", "maxar_open_data"]

ACTIVE_SAMPLE_SEED = (
    USER_SAMPLE_SEED
    if USER_SAMPLE_SEED is not None
    else int(pd.Timestamp.utcnow().value % (2**32 - 1))
)

RASTER_STAC_DIR = PROJECT_ROOT / "data" / "rasters" / "stac"
RASTER_STAC_DIR.mkdir(parents=True, exist_ok=True)
CONSOLIDATED_OUTPUT_PATH = RASTER_STAC_DIR / "pr_raster_catalog_items.parquet"

NAIP_COASTAL_CATALOG_URL = (
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


def empty_geodataframe(columns: list[str] | None = None) -> gpd.GeoDataFrame:
    """Return an empty GeoDataFrame in the notebook output CRS."""

    return gpd.GeoDataFrame(
        columns=columns or ["geometry"],
        geometry=gpd.GeoSeries([], crs=OUTPUT_CRS),
        crs=OUTPUT_CRS,
    )


def _to_wkb_bytes(value: object) -> bytes:
    """Normalize WKB values fetched from DuckDB."""

    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def fetch_geodataframe(con, query: str, params: list[object] | None = None) -> gpd.GeoDataFrame:
    """Run a SQL query that returns geometry_wkb and convert it to GeoPandas."""

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
    """Pick the best raster asset href available for preview or chip extraction."""

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

    geom_minx, geom_miny, geom_maxx, geom_maxy = geometry.bounds
    bounds_4326 = (geom_minx, geom_miny, geom_maxx, geom_maxy)

    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(asset_href) as src:
            minx, miny, maxx, maxy = bounds_4326
            if src.crs and src.crs.to_string() != OUTPUT_CRS:
                minx, miny, maxx, maxy = transform_bounds(OUTPUT_CRS, src.crs, minx, miny, maxx, maxy, densify_pts=21)

            window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            window = window.intersection(Window(0, 0, src.width, src.height))
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
        "bounds_4326": bounds_4326,
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

            native_bounds = window_bounds_fn(window, src.transform)
            if src.crs and src.crs.to_string() != OUTPUT_CRS:
                chip_bounds_4326 = transform_bounds(src.crs, OUTPUT_CRS, *native_bounds)
            else:
                chip_bounds_4326 = native_bounds

    return {
        "asset_href": asset_href,
        "bands": bands,
        "image": normalize_image(data[0] if len(bands) == 1 else data),
        "bounds_4326": chip_bounds_4326,
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


def fetch_h3_pv_cells_for_municipality(vector_con, municipality_name: str) -> gpd.GeoDataFrame:
    """Fetch H3 cells with OSM PV labels for a municipality."""

    if not table_exists(vector_con, "pr_solar_tile_manifest"):
        return empty_geodataframe(["h3_cell_id", "osm_pv_count", "geometry"])

    return fetch_geodataframe(
        vector_con,
        """
        SELECT
            h3_cell_id,
            osm_pv_count,
            ST_AsWKB(geometry) AS geometry_wkb
        FROM pr_solar_tile_manifest
        WHERE municipio = ?
          AND osm_pv_count > 0
        """,
        [municipality_name],
    )


def fetch_h3_pv_cells_for_census_unit(vector_con, census_unit_geometry) -> gpd.GeoDataFrame:
    """Fetch H3 cells with OSM PV labels intersecting a census unit."""

    if not table_exists(vector_con, "pr_solar_tile_manifest"):
        return empty_geodataframe(["h3_cell_id", "osm_pv_count", "geometry"])

    return fetch_geodataframe(
        vector_con,
        """
        SELECT
            h3_cell_id,
            osm_pv_count,
            ST_AsWKB(geometry) AS geometry_wkb
        FROM pr_solar_tile_manifest
        WHERE ST_Intersects(geometry, ST_GeomFromText(?))
          AND osm_pv_count > 0
        """,
        [census_unit_geometry.wkt],
    )


def find_h3_cell_for_point(vector_con, point: Point) -> str | None:
    """Return the H3 cell whose geometry contains a point."""

    if not table_exists(vector_con, "pr_solar_tile_manifest"):
        return None

    row = vector_con.execute(
        """
        SELECT h3_cell_id
        FROM pr_solar_tile_manifest
        WHERE ST_Contains(geometry, ST_GeomFromText(?))
        LIMIT 1
        """,
        [point.wkt],
    ).fetchone()
    return row[0] if row else None


def compute_area_km2(gdf: gpd.GeoDataFrame) -> float:
    """Compute area in square kilometers using UTM zone 20N for Puerto Rico."""

    if gdf.empty:
        return float("nan")
    return float(gdf.to_crs(METRIC_CRS).geometry.area.iloc[0] / 1_000_000.0)


def bounds_to_wkt(bounds_4326: tuple[float, float, float, float]) -> str:
    """Convert a geographic bounds tuple into WKT polygon text."""

    return shapely_box(*bounds_4326).wkt


def overlay_polygons_on_image(
    ax,
    image_shape: tuple[int, ...],
    polygons_gdf: gpd.GeoDataFrame,
    bounds_4326: tuple[float, float, float, float],
    *,
    edgecolor: str,
    facecolor: str = "none",
    linewidth: float = 1.5,
    alpha: float = 0.8,
) -> None:
    """Overlay polygon outlines or fills on a raster image in pixel space."""

    if polygons_gdf is None or polygons_gdf.empty:
        return

    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon

    minx, miny, maxx, maxy = bounds_4326
    if maxx <= minx or maxy <= miny:
        return

    image_height, image_width = image_shape[:2]
    patches = []

    def coords_to_pixels(coords) -> np.ndarray:
        coords_array = np.asarray(coords)
        x_pixels = ((coords_array[:, 0] - minx) / (maxx - minx)) * image_width
        y_pixels = (1.0 - ((coords_array[:, 1] - miny) / (maxy - miny))) * image_height
        return np.column_stack([x_pixels, y_pixels])

    for geom in polygons_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for part in parts:
            exterior = getattr(part, "exterior", None)
            if exterior is None:
                continue
            patches.append(MplPolygon(coords_to_pixels(exterior.coords), closed=True))

    if not patches:
        return

    ax.add_collection(
        PatchCollection(
            patches,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )
    )


def plot_vector_context(
    municipality_gdf: gpd.GeoDataFrame,
    title: str,
    *,
    block_gdf: gpd.GeoDataFrame | None = None,
    pv_gdf: gpd.GeoDataFrame | None = None,
    building_gdf: gpd.GeoDataFrame | None = None,
    raster_item_gdf: gpd.GeoDataFrame | None = None,
    h3_pv_cells_gdf: gpd.GeoDataFrame | None = None,
) -> None:
    """Plot a compact vector context panel for notebook inspection."""

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    municipality_gdf.boundary.plot(ax=ax, color="#0f172a", linewidth=1.25, label="Municipality")

    if raster_item_gdf is not None and not raster_item_gdf.empty:
        raster_item_gdf.boundary.plot(ax=ax, color="#2563eb", linewidth=1.1, label="Raster footprint")

    if h3_pv_cells_gdf is not None and not h3_pv_cells_gdf.empty:
        h3_pv_cells_gdf.boundary.plot(
            ax=ax,
            color="#7c3aed",
            linewidth=0.7,
            alpha=0.7,
            label="H3 cells w/ PV labels",
        )

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
            if label in seen:
                continue
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
        ax.legend(unique_handles, unique_labels, loc="upper right")
    plt.show()


def plot_raster_with_overlay(
    preview: dict[str, object],
    title: str,
    *,
    overlay_layers: list[dict[str, object]] | None = None,
    figsize: tuple[int, int] = (8, 8),
) -> None:
    """Display a raster preview with optional vector overlays."""

    image = np.asarray(preview["image"])
    bounds_4326 = preview.get("bounds_4326")

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    cmap = None if image.ndim == 3 else "gray"
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.set_axis_off()

    if overlay_layers and bounds_4326 is not None:
        for layer in overlay_layers:
            layer_gdf = layer.get("gdf")
            if layer_gdf is None or layer_gdf.empty:
                continue
            overlay_polygons_on_image(
                ax,
                image.shape,
                layer_gdf,
                bounds_4326,
                edgecolor=str(layer.get("edgecolor", "#ff0000")),
                facecolor=str(layer.get("facecolor", "none")),
                linewidth=float(layer.get("linewidth", 1.5)),
                alpha=float(layer.get("alpha", 0.8)),
            )
    plt.show()


def build_geometry_previews(
    candidate_rows: gpd.GeoDataFrame,
    geometry,
    *,
    out_size: int = BLOCK_PREVIEW_SIZE,
) -> list[tuple[pd.Series, dict[str, object]]]:
    """Read one geometry-based preview for each candidate raster row."""

    previews: list[tuple[pd.Series, dict[str, object]]] = []
    for _, candidate_row in candidate_rows.iterrows():
        preview = read_raster_preview_from_geometry(candidate_row, geometry, out_size=out_size)
        if preview is not None:
            previews.append((candidate_row, preview))
    return previews


def build_point_chip_previews(
    candidate_rows: gpd.GeoDataFrame,
    point: Point,
    *,
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
        naip_rows = candidate_frame[candidate_frame["source"].isin(["pr_naip", "naip_2021_pr"])]
        if not naip_rows.empty:
            return naip_rows.iloc[0]
    return None


# %% [markdown]
# ## Materialize the Consolidated Puerto Rico Raster Catalog

# %%
SUMMARY_FRAME = asyncio.run(
    materialize_consolidated_pr_raster_catalog(
        output_path=CONSOLIDATED_OUTPUT_PATH,
        boundary=boundary,
        naip_coastal_catalog_url=NAIP_COASTAL_CATALOG_URL,
        maxar_remote_parquet_url=MAXAR_REMOTE_PARQUET_URL,
        earthview_remote_parquet_url=EARTHVIEW_REMOTE_PARQUET_URL,
    )
)
display(SUMMARY_FRAME)
print(f"Consolidated catalog output: {CONSOLIDATED_OUTPUT_PATH}")


# %% [markdown]
# ## Inspect the Consolidated Raster Catalog

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

top_names = top_municipalities["municipality_name"].astype(str).tolist()
placeholders = ", ".join(["?"] * len(top_names))
top_municipality_boundaries = fetch_geodataframe(
    vector_con,
    f"""
    SELECT
        GEOID AS municipality_geoid,
        NAME AS municipality_name,
        ST_AsWKB(geometry) AS geometry_wkb
    FROM pr_census_counties
    WHERE NAME IN ({placeholders})
    """,
    top_names,
)

municipalities_with_stac: set[str] = set()
for _, municipality_row in top_municipality_boundaries.iterrows():
    if catalog_gdf.geometry.intersects(municipality_row.geometry).any():
        municipalities_with_stac.add(str(municipality_row["municipality_name"]))

top_municipalities_with_stac = top_municipalities[
    top_municipalities["municipality_name"].isin(municipalities_with_stac)
].reset_index(drop=True)
if top_municipalities_with_stac.empty:
    print("Warning: no top-PV municipality intersects the current STAC catalog; falling back to the full top-10 list.")
    top_municipalities_with_stac = top_municipalities.copy()

selected_municipality_name = sample_rows(
    top_municipalities_with_stac,
    "top-municipality",
    n=1,
).iloc[0]["municipality_name"]
print(f"Sampled municipality (top-10 PV, STAC-intersected when available): {selected_municipality_name}")
display(top_municipalities)

municipality_gdf = fetch_geodataframe(
    vector_con,
    """
    SELECT
        GEOID AS municipality_geoid,
        NAME AS municipality_name,
        ST_AsWKB(geometry) AS geometry_wkb
    FROM pr_census_counties
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

municipality_geometry = municipality_gdf.geometry.iloc[0]
municipality_stac_gdf = catalog_gdf[catalog_gdf.geometry.intersects(municipality_geometry)].copy()
municipality_h3_pv_gdf = fetch_h3_pv_cells_for_municipality(vector_con, selected_municipality_name)
print(
    f"STAC footprints intersecting municipality: {len(municipality_stac_gdf)}, "
    f"H3 PV-labelled cells: {len(municipality_h3_pv_gdf)}"
)

plot_vector_context(
    municipality_gdf,
    f"{selected_municipality_name}: boundary, STAC footprints, H3 PV cells, sampled PV polygons",
    pv_gdf=municipality_pv_gdf,
    raster_item_gdf=municipality_stac_gdf if not municipality_stac_gdf.empty else None,
    h3_pv_cells_gdf=municipality_h3_pv_gdf if not municipality_h3_pv_gdf.empty else None,
)


# %% [markdown]
# ## Sample a Census Unit Inside that Municipality

# %%
census_unit_table = "pr_census_block_groups"
census_unit_label = "block group"

sampled_census_unit_gdf = fetch_geodataframe(
    vector_con,
    f"""
    SELECT
        g.GEOID AS census_unit_geoid,
        COALESCE(g.NAME, g.GEOID) AS census_unit_name,
        ST_AsWKB(g.geometry) AS geometry_wkb
    FROM {census_unit_table} AS g
    JOIN pr_census_counties AS m
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

census_display_name = str(sampled_census_unit_gdf.iloc[0].get("census_unit_name") or "").strip()
if not census_display_name:
    census_display_name = str(sampled_census_unit_gdf.iloc[0].get("census_unit_geoid", "Unknown"))

if census_display_name == str(sampled_census_unit_gdf.iloc[0].get("census_unit_geoid", "")):
    try:
        import osmnx as ox

        osmnx_match = ox.geocode_to_gdf(f"{selected_municipality_name}, Puerto Rico")
        if not osmnx_match.empty:
            fallback_name = str(osmnx_match.iloc[0].get("display_name", "")).split(",")[0].strip()
            if fallback_name:
                census_display_name = fallback_name
    except Exception:
        pass

census_area_km2 = compute_area_km2(sampled_census_unit_gdf)

print(f"Using census geography table: {census_unit_table} ({census_unit_label})")
print(f"Census unit common name : {census_display_name}")
print(f"Census unit area        : {census_area_km2:.3f} km^2")
display(sampled_census_unit_gdf.drop(columns="geometry"))

sampled_census_unit_geometry = sampled_census_unit_gdf.geometry.iloc[0]
block_pv_gdf = fetch_geodataframe(
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
block_pv_gdf = sample_rows(block_pv_gdf, "block-group-pv-preview", n=300)

plot_vector_context(
    municipality_gdf,
    f"Sampled {census_unit_label} inside {selected_municipality_name}",
    block_gdf=sampled_census_unit_gdf,
    pv_gdf=block_pv_gdf,
)


# %% [markdown]
# ## Sample a Raster Item that Intersects the Census Unit

# %%
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
    municipality_gdf,
    f"Raster footprint intersecting sampled {census_unit_label}",
    block_gdf=sampled_census_unit_gdf,
    pv_gdf=block_pv_gdf,
    raster_item_gdf=raster_candidates_for_unit,
)

unit_raster_previews = build_geometry_previews(
    raster_candidates_for_unit,
    sampled_census_unit_geometry,
    out_size=CENSUS_UNIT_PREVIEW_SIZE,
)
if not unit_raster_previews:
    print("No readable raster preview asset was available for the sampled census unit.")
else:
    print("Readable raster preview assets for the sampled census unit:")
    for candidate_row, preview in unit_raster_previews:
        print(f"- {candidate_row['source']}: {preview['asset_href']}")
        plot_raster_with_overlay(
            preview,
            (
                f"Raster preview ({CENSUS_UNIT_PREVIEW_SIZE}px) over sampled {census_unit_label}\n"
                f"{candidate_row['source']} - {candidate_row['item_id']}"
            ),
            overlay_layers=[
                {
                    "gdf": sampled_census_unit_gdf,
                    "edgecolor": "#dc2626",
                    "facecolor": "none",
                    "linewidth": 2.0,
                    "alpha": 0.9,
                }
            ],
            figsize=(9, 9),
        )


# %% [markdown]
# ## Fetch an NxN Raster Chip from a Sampled Solar Panel Centroid

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
pv_point_candidates = sort_catalog_candidates(catalog_gdf[catalog_gdf.geometry.intersects(pv_centroid)].copy())
pv_point_candidates = select_source_candidates(pv_point_candidates)

display(
    pv_point_candidates[
        ["source", "item_id", "collection_id", "acquired_at", "gsd", "visual_asset_href", "analytic_asset_href"]
    ].head(8)
)

if pv_point_candidates.empty:
    raise RuntimeError("No raster items intersected the sampled rooftop PV centroid.")

plot_vector_context(
    municipality_gdf,
    "Sampled PV polygon centroid with matching raster footprint",
    block_gdf=sampled_census_unit_gdf,
    pv_gdf=sampled_pv_gdf,
    raster_item_gdf=pv_point_candidates,
)

census_unit_h3_pv_gdf = fetch_h3_pv_cells_for_census_unit(vector_con, sampled_census_unit_geometry)
selected_h3_cell_id = find_h3_cell_for_point(vector_con, pv_centroid)

if not census_unit_h3_pv_gdf.empty:
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    sampled_census_unit_gdf.boundary.plot(ax=ax, color="#0f172a", linewidth=1.2, label="Census unit")
    census_unit_h3_pv_gdf.boundary.plot(
        ax=ax,
        color="#7c3aed",
        linewidth=0.8,
        alpha=0.7,
        label="H3 cells w/ PV labels",
    )
    if selected_h3_cell_id:
        selected_h3_gdf = census_unit_h3_pv_gdf[census_unit_h3_pv_gdf["h3_cell_id"] == selected_h3_cell_id]
        if not selected_h3_gdf.empty:
            selected_h3_gdf.plot(
                ax=ax,
                color="#facc15",
                edgecolor="#d97706",
                linewidth=1.5,
                alpha=0.6,
                label="Selected H3 cell",
            )
    sampled_pv_gdf.plot(ax=ax, color="#f97316", alpha=0.7, linewidth=0.5, label="Sampled PV polygon")
    ax.set_title(f"H3 PV-labelled cells in sampled {census_unit_label}")
    ax.set_axis_off()
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles, labels, loc="upper right")
    plt.show()
    print(f"H3 cells with PV labels in census unit: {len(census_unit_h3_pv_gdf)}")
    print(f"H3 cell containing PV centroid: {selected_h3_cell_id or 'none found'}")
else:
    print("No H3 PV-labelled cells found in this census unit.")

pv_chip_previews = build_point_chip_previews(pv_point_candidates, pv_centroid, chip_size=CHIP_SIZE)
if not pv_chip_previews:
    print("No readable raster chip asset was available for the sampled PV centroid.")
else:
    print("Readable raster chip assets for the sampled PV centroid:")
    for candidate_row, preview in pv_chip_previews:
        print(f"- {candidate_row['source']}: {preview['asset_href']}")
        local_pv_gdf = fetch_geodataframe(
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
            [bounds_to_wkt(preview["bounds_4326"])],
        )
        plot_raster_with_overlay(
            preview,
            f"{CHIP_SIZE}x{CHIP_SIZE} PV chip - {candidate_row['source']}\nPV labels overlaid in orange",
            overlay_layers=[
                {
                    "gdf": local_pv_gdf,
                    "edgecolor": "#ea580c",
                    "facecolor": "#f97316",
                    "linewidth": 1.1,
                    "alpha": 0.55,
                }
            ]
            if not local_pv_gdf.empty
            else None,
        )


# %% [markdown]
# ## Sample NxN Raster Chips from a Building Footprint Centroid

# %%
sampled_building_gdf = fetch_geodataframe(
    vector_con,
    """
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
        """
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
    municipality_gdf,
    "Sampled building centroid with best intersecting raster item per source",
    block_gdf=sampled_census_unit_gdf,
    building_gdf=sampled_building_gdf,
    raster_item_gdf=building_point_candidates,
)

if building_point_candidates.empty:
    print("No intersecting raster items were found for the sampled building centroid.")
else:
    building_chip_previews = build_point_chip_previews(
        building_point_candidates,
        sampled_building_centroid,
        chip_size=CHIP_SIZE,
    )
    if not building_chip_previews:
        print("Intersecting raster items were found, but none exposed a readable preview or chip asset.")
    else:
        print("Readable chip assets used for the building centroid preview:")
        for candidate_row, preview in building_chip_previews:
            print(f"- {candidate_row['source']}: {preview['asset_href']}")
            local_buildings_gdf = fetch_geodataframe(
                vector_con,
                """
                SELECT
                    id AS building_id,
                    ST_AsWKB(geometry) AS geometry_wkb
                FROM pr_overture_buildings
                WHERE ST_Intersects(geometry, ST_GeomFromText(?))
                """,
                [bounds_to_wkt(preview["bounds_4326"])],
            )
            if not local_buildings_gdf.empty:
                local_buildings_gdf = sample_rows(
                    local_buildings_gdf,
                    f"building-overlay-{candidate_row['source']}-{candidate_row['item_id']}",
                    n=BUILDING_PREVIEW_LIMIT,
                )
            plot_raster_with_overlay(
                preview,
                f"{CHIP_SIZE}x{CHIP_SIZE} building chip - {candidate_row['source']}\nBuilding footprints overlaid in green",
                overlay_layers=[
                    {
                        "gdf": local_buildings_gdf if not local_buildings_gdf.empty else sampled_building_gdf,
                        "edgecolor": "#10b981",
                        "facecolor": "none",
                        "linewidth": 1.5,
                        "alpha": 0.9,
                    }
                ],
            )


# %% [markdown]
# ## Explore a Full NAIP COG in Lonboard

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
            return await GeoTIFF.open("", store=HTTPStore(cog_href))

        geotiff = asyncio.run(open_lonboard_geotiff(naip_asset_href))

        def render_tile(tile) -> EncodedImage:
            masked_array = reshape_as_image(tile.array.as_masked())
            image_array = np.asarray(masked_array.data)
            if image_array.ndim == 2:
                stretched = (normalize_image(image_array) * 255).astype(np.uint8)
                image = Image.fromarray(stretched, mode="L")
            else:
                rgb = image_array[:, :, : min(3, image_array.shape[2])]
                stretched = (normalize_image(np.moveaxis(rgb, -1, 0)) * 255).astype(np.uint8)
                mask = np.asarray(masked_array.mask)
                if mask.ndim == 0:
                    alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
                elif mask.ndim == 2:
                    alpha = (~mask).astype(np.uint8) * 255
                else:
                    alpha = (~mask.any(axis=-1)).astype(np.uint8) * 255
                image = Image.fromarray(np.concatenate([stretched, alpha[..., None]], axis=-1), mode="RGBA")

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return EncodedImage(data=buffer.getvalue(), media_type="image/png")

        lonboard_layer = RasterLayer.from_geotiff(geotiff, render_tile=render_tile)
        layers = [lonboard_layer]

        try:
            from lonboard import HeatmapLayer

            heatmap_layer = None
            if table_exists(vector_con, "pr_solar_tile_manifest"):
                h3_cog_gdf = fetch_geodataframe(
                    vector_con,
                    """
                    SELECT
                        h3_cell_id,
                        osm_pv_count,
                        ST_AsWKB(ST_Centroid(geometry)) AS geometry_wkb
                    FROM pr_solar_tile_manifest
                    WHERE ST_Intersects(geometry, ST_GeomFromText(?))
                      AND osm_pv_count > 0
                    """,
                    [lonboard_item.geometry.wkt],
                )
                if h3_cog_gdf.empty:
                    print("No H3 PV-labelled cells found intersecting this COG footprint; HeatmapLayer skipped.")
                else:
                    print(
                        f"Building HeatmapLayer from {len(h3_cog_gdf)} H3 cells "
                        f"(osm_pv_count total: {int(h3_cog_gdf['osm_pv_count'].sum())})."
                    )
                    heatmap_layer = HeatmapLayer.from_geopandas(
                        h3_cog_gdf[["geometry", "osm_pv_count"]],
                        get_weight="osm_pv_count",
                        radius_pixels=40,
                        intensity=1.5,
                        opacity=0.65,
                        aggregation="SUM",
                    )
            else:
                print("pr_solar_tile_manifest table not found; HeatmapLayer skipped.")

            if heatmap_layer is not None:
                layers.append(heatmap_layer)
        except ImportError:
            print("HeatmapLayer is not available in this lonboard version; displaying raster only.")
        except Exception as exc:
            print(f"HeatmapLayer construction failed: {type(exc).__name__}: {exc}")

        display(Map(layers=layers, height=800))
    except ImportError as exc:
        print(f"Lonboard full-COG preview requires extra packages that are not installed: {exc}")
    except Exception as exc:
        print(f"Lonboard full-COG preview failed: {type(exc).__name__}: {exc}")


# %% [markdown]
# ## Reference Notes for Follow-on Raster Preview Work
#
# 1. Planetary Computer STAC quickstart
#    - bbox and intersects are the key search patterns.
#    - Treat STAC items as GeoJSON and convert them into GeoDataFrames for quick metadata analysis.
#    - Item assets can be inspected directly and opened with libraries such as rioxarray or rasterio.
#
# 2. EODC thumbnail creation with TiTiler
#    - A preview image can be built from a COG URL using a /cog/preview endpoint with parameters such as rescale, nodata, and dst_crs.
#    - That pattern is useful if we want richer remote thumbnails without reading raster windows directly in Python.
#
# 3. Lonboard COG rendering
#    - RasterLayer.from_geotiff(...) supports streaming COG tiles on demand.
#    - This is a strong next step if we want interactive browser-side inspection of candidate raster items without standing up a tile server.
#
# 4. STAC + xarray + dask
#    - odc.stac.load(...) is the main pattern for multi-item raster cubes when we need more than single-scene preview or chip extraction.
#    - That is more appropriate for temporal stacks or model-ready raster tensors than for lightweight notebook previews.

# %%
vector_con.close()