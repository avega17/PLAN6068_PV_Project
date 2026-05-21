# %% [markdown]
# # STAC Raster Fetch and Occupied-H3 Clip
# 
# Uses the consolidated Puerto Rico STAC catalog from `05_pr_raster_catalog_indexes.py`
# to fetch every intersecting raster item for occupied H3 cells in San Juan and
# Isabela, clips each asset to the occupied cell geometry, reprojects to
# EPSG:3857, and writes model-ready local derivatives under
# `data/rasters/stac/local/`.

# %%
"""07_pr_stac_municipality_fetch.py

Fetch and clip intersecting STAC raster assets for occupied H3 cells.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from affine import Affine
import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import rasterio
from rasterio.enums import Resampling
from dotenv import load_dotenv
from rasterio.mask import mask
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_geom
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, mapping
from shapely.ops import unary_union


def resolve_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in ("project_rules.md", ".git")):
            return candidate
    return current


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
from utils.census import CANONICAL_COUNTY_TABLE
from utils.overture import occupied_h3_cells_sql

TARGET_MUNICIPALITIES = ("San Juan", "Isabela")
OVERTURE_BUILDINGS_TABLE = "pr_overture_buildings"
STAC_CATALOG_PATH = PROJECT_ROOT / "data" / "rasters" / "stac" / "pr_raster_catalog_items.parquet"
LOCAL_STAC_ROOT = PROJECT_ROOT / "data" / "rasters" / "stac" / "local"
LOCAL_FETCH_MANIFEST_PATH = PROJECT_ROOT / "data" / "rasters" / "stac" / "pr_local_stac_fetch_manifest.parquet"
OUTPUT_STAC_TILE_ROOT = PROJECT_ROOT / "outputs" / "stac_tiles"
OUTPUT_STAC_TILE_MANIFEST_PATH = OUTPUT_STAC_TILE_ROOT / "pr_stac_tile_manifest.parquet"
MODEL_READY_CRS = "EPSG:3857"
PREFERRED_ASSET_COLUMNS = (
    ("visual_asset_href", "visual"),
    ("analytic_asset_href", "analytic"),
)
MAX_ITEMS_THIS_RUN = int(os.getenv("MAX_STAC_ITEMS_THIS_RUN", "10")) or None
MAX_ASSETS_THIS_RUN = int(os.getenv("MAX_STAC_ASSETS_THIS_RUN", "10")) or None
OVERWRITE_EXISTING = os.getenv("OVERWRITE_STAC_LOCAL_CACHE", "0") == "1"
EXPORT_STAC_SQUARE_TILES = os.getenv("EXPORT_STAC_SQUARE_TILES", "1") == "1"
STAC_TILE_PIXELS = int(os.getenv("STAC_TILE_PIXELS", "512") or "512")
STAC_TILE_PADDING_FACTOR = float(os.getenv("STAC_TILE_PADDING_FACTOR", "1.05") or "1.05")
SHOW_COVERAGE_PREVIEW = os.getenv("SHOW_STAC_COVERAGE_PREVIEW", "1") == "1"

SOURCE_COLORS = {
    "pr_naip": "#1f78b4",
    "naip_2021_pr": "#6baed6",
    "maxar_open_data": "#33a02c",
    "satellogic_earthview": "#ff7f00",
}


def resolve_db_path() -> Path:
    value = os.getenv("VECTOR_DB")
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path if len(path.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / path
        return path
    return PROJECT_ROOT / "data" / "PR_PV_plan_data.duckdb"


def _to_bytes(value: object) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    return bytes(value)


def slugify(value: str) -> str:
    return value.strip().replace(" ", "_").replace("/", "_")


def _project_relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    row = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?;",
        [table_name],
    ).fetchone()
    return bool(row and row[0])


def load_target_h3_cells(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    if not table_exists(con, OVERTURE_BUILDINGS_TABLE):
        raise RuntimeError(
            f"{OVERTURE_BUILDINGS_TABLE} not found; run notebooks/vectors/03_overture_buildings_ingest.py first."
        )

    municipio_sql = ", ".join(f"'{municipio}'" for municipio in TARGET_MUNICIPALITIES)
    occupied_h3_sql = occupied_h3_cells_sql(OVERTURE_BUILDINGS_TABLE)
    frame = con.execute(
        f"""
        WITH occupied_h3 AS ({occupied_h3_sql})
        SELECT
               h3_cell_id,
               h3_resolution,
               municipality_name AS municipio,
               municipality_geoid AS municipio_geoid,
               building_count,
               municipality_building_count,
               crosses_municipality_boundary,
               ST_AsWKB(geometry) AS geometry_wkb
         FROM occupied_h3
        WHERE municipality_name IN ({municipio_sql})
        ORDER BY municipio, h3_cell_id;
        """
    ).fetchdf()
    if frame.empty:
        return gpd.GeoDataFrame(
            columns=[
                "h3_cell_id",
                "h3_resolution",
                "municipio",
                "municipio_geoid",
                "building_count",
                "municipality_building_count",
                "crosses_municipality_boundary",
                "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

    geometry = gpd.GeoSeries.from_wkb(frame["geometry_wkb"].map(_to_bytes), crs="EPSG:4326")
    return gpd.GeoDataFrame(frame.drop(columns=["geometry_wkb"]), geometry=geometry, crs="EPSG:4326")


def load_catalog() -> gpd.GeoDataFrame:
    if not STAC_CATALOG_PATH.exists():
        raise RuntimeError(
            f"STAC catalog not found at {STAC_CATALOG_PATH}; run notebooks/rasters/05_pr_raster_catalog_indexes.py first."
        )
    catalog = gpd.read_parquet(STAC_CATALOG_PATH)
    if catalog.crs is None:
        return catalog.set_crs("EPSG:4326")
    return catalog.to_crs("EPSG:4326")


def load_target_municipalities(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    if not table_exists(con, CANONICAL_COUNTY_TABLE):
        raise RuntimeError(
            f"{CANONICAL_COUNTY_TABLE} not found; run notebooks/vectors/01_census_geometries_ingest.py first."
        )

    names_sql = ", ".join("?" * len(TARGET_MUNICIPALITIES))
    frame = con.execute(
        f"""
        SELECT
            CAST(NAME AS VARCHAR) AS municipio,
            CAST(GEOID AS VARCHAR) AS municipio_geoid,
            ST_AsWKB(geometry) AS geometry_wkb
        FROM {CANONICAL_COUNTY_TABLE}
        WHERE NAME IN ({names_sql})
        ORDER BY NAME;
        """,
        list(TARGET_MUNICIPALITIES),
    ).fetchdf()
    if frame.empty:
        return gpd.GeoDataFrame(columns=["municipio", "municipio_geoid", "geometry"], geometry="geometry", crs="EPSG:4326")

    geometry = gpd.GeoSeries.from_wkb(frame["geometry_wkb"].map(_to_bytes), crs="EPSG:4326")
    return gpd.GeoDataFrame(frame.drop(columns=["geometry_wkb"]), geometry=geometry, crs="EPSG:4326")


def _empty_coverage_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns=["municipio", "municipio_geoid", "source", "geometry"], geometry="geometry", crs="EPSG:4326")


def _geometry_area_km2(geometry) -> float:
    if geometry is None or geometry.is_empty:
        return 0.0
    return float(gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs("EPSG:6933").area.iloc[0] / 1_000_000.0)


def build_catalog_extent_overlay(
    catalog: gpd.GeoDataFrame,
    municipalities: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if catalog.empty or municipalities.empty:
        return _empty_coverage_gdf()

    joined = gpd.sjoin(
        catalog[["source", "item_id", "acquired_at", "gsd", "geometry"]].copy(),
        municipalities[["municipio", "municipio_geoid", "geometry"]],
        how="inner",
        predicate="intersects",
        lsuffix="item",
        rsuffix="municipio",
    )
    if joined.empty:
        return _empty_coverage_gdf()

    overlay = joined.drop_duplicates(subset=["municipio", "source", "item_id"]).reset_index(drop=True)
    return gpd.GeoDataFrame(overlay, geometry="geometry", crs="EPSG:4326")


def build_cached_asset_extent_overlay(
    asset_queue: pd.DataFrame,
    manifest: pd.DataFrame,
    municipalities: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if asset_queue.empty or manifest.empty or municipalities.empty:
        return _empty_coverage_gdf()

    manifest_keys = manifest[manifest["status"].isin({"fetched", "reused"})][_asset_group_columns()].drop_duplicates()
    if manifest_keys.empty:
        return _empty_coverage_gdf()

    overlay = asset_queue.merge(manifest_keys, on=_asset_group_columns(), how="inner")
    if overlay.empty:
        return _empty_coverage_gdf()

    overlay_gdf = gpd.GeoDataFrame(overlay.copy(), geometry="geometry", crs="EPSG:4326")
    joined = gpd.sjoin(
        overlay_gdf,
        municipalities[["municipio", "municipio_geoid", "geometry"]],
        how="inner",
        predicate="intersects",
        lsuffix="asset",
        rsuffix="municipio",
    )
    if joined.empty:
        return _empty_coverage_gdf()

    return gpd.GeoDataFrame(joined, geometry="geometry", crs="EPSG:4326")


def summarize_land_coverage(
    municipalities: gpd.GeoDataFrame,
    coverage_overlay: gpd.GeoDataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for municipality in municipalities.itertuples(index=False):
        subset = coverage_overlay[coverage_overlay["municipio"] == municipality.municipio].copy()
        municipality_area_km2 = _geometry_area_km2(municipality.geometry)
        coverage_geometry = None
        if not subset.empty:
            coverage_geometry = unary_union(subset.geometry.tolist()).intersection(municipality.geometry)
        coverage_area_km2 = _geometry_area_km2(coverage_geometry)
        rows.append(
            {
                "municipio": municipality.municipio,
                "municipio_geoid": municipality.municipio_geoid,
                "municipality_area_km2": municipality_area_km2,
                "coverage_area_km2": coverage_area_km2,
                "coverage_area_pct": 0.0 if municipality_area_km2 <= 0 else (coverage_area_km2 / municipality_area_km2) * 100.0,
                "raster_count": int(len(subset)),
                "sources": ", ".join(sorted(subset["source"].dropna().astype(str).unique().tolist())),
            }
        )
    return pd.DataFrame(rows)


def filter_targets_by_manifest(targets: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    if targets.empty or manifest.empty:
        return targets.iloc[0:0].copy()

    success_keys = manifest[manifest["status"].isin({"fetched", "reused"})][_asset_group_columns()].drop_duplicates()
    if success_keys.empty:
        return targets.iloc[0:0].copy()
    return targets.merge(success_keys, on=_asset_group_columns(), how="inner")


def summarize_building_coverage(
    occupied_h3_cells: gpd.GeoDataFrame,
    covered_targets: pd.DataFrame,
) -> pd.DataFrame:
    total = (
        occupied_h3_cells[["municipio", "h3_cell_id", "building_count"]]
        .drop_duplicates(subset=["municipio", "h3_cell_id"])
        .groupby("municipio", dropna=False)["building_count"]
        .sum()
        .rename("total_buildings")
    )
    covered = (
        covered_targets[["municipio", "h3_cell_id", "building_count"]]
        .drop_duplicates(subset=["municipio", "h3_cell_id"])
        .groupby("municipio", dropna=False)["building_count"]
        .sum()
        .rename("covered_buildings")
    )
    summary = pd.concat([total, covered], axis=1).fillna(0).reset_index()
    summary["total_buildings"] = summary["total_buildings"].astype(int)
    summary["covered_buildings"] = summary["covered_buildings"].astype(int)
    summary["covered_building_pct"] = np.where(
        summary["total_buildings"] > 0,
        (summary["covered_buildings"] / summary["total_buildings"]) * 100.0,
        0.0,
    )
    return summary


def plot_municipality_coverage(
    municipalities: gpd.GeoDataFrame,
    overlay: gpd.GeoDataFrame,
    summary: pd.DataFrame,
    *,
    suptitle: str,
    include_buildings: bool,
) -> None:
    if municipalities.empty:
        print("coverage preview skipped: municipality geometries are unavailable.")
        return

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, len(municipalities), figsize=(8 * len(municipalities), 8), squeeze=False)
    axes_list = list(axes.flat)
    legend_handles: list[Patch] = []

    for ax, municipality in zip(axes_list, municipalities.itertuples(index=False)):
        municipality_gdf = municipalities[municipalities["municipio"] == municipality.municipio]
        municipality_gdf.plot(ax=ax, color="#f2f2f2", edgecolor="#222222", linewidth=1.2)

        subset = overlay[overlay["municipio"] == municipality.municipio].copy() if not overlay.empty else overlay
        if not subset.empty:
            for source, source_group in subset.groupby("source", sort=False):
                color = SOURCE_COLORS.get(str(source), "#9467bd")
                source_group.plot(ax=ax, facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.22)
                if source not in {handle.get_label() for handle in legend_handles}:
                    legend_handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.35, label=str(source)))

        municipality_gdf.boundary.plot(ax=ax, color="#111111", linewidth=1.4)
        summary_row = summary[summary["municipio"] == municipality.municipio].iloc[0]
        text_lines = [
            f"Land area covered: {summary_row['coverage_area_pct']:.1f}%",
            f"Coverage area: {summary_row['coverage_area_km2']:.1f} / {summary_row['municipality_area_km2']:.1f} km²",
            f"Raster footprints: {int(summary_row['raster_count']):,}",
        ]
        if include_buildings:
            text_lines.append(
                f"Buildings covered: {summary_row['covered_building_pct']:.1f}% ({int(summary_row['covered_buildings']):,} / {int(summary_row['total_buildings']):,})"
            )
        if isinstance(summary_row.get("sources"), str) and summary_row["sources"]:
            text_lines.append(f"Sources: {summary_row['sources']}")
        ax.text(
            0.02,
            0.02,
            "\n".join(text_lines),
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.9, "boxstyle": "round,pad=0.4"},
        )
        ax.set_title(municipality.municipio)
        ax.set_axis_off()

    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center", ncol=min(4, len(legend_handles)), frameon=False)
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))


def choose_asset(row: pd.Series) -> tuple[str | None, str | None]:
    for column_name, asset_role in PREFERRED_ASSET_COLUMNS:
        href = row.get(column_name)
        if isinstance(href, str) and href.strip():
            return href, asset_role
    return None, None


def _asset_source_rank(source_name: str) -> int:
    order = {
        "pr_naip": 0,
        "naip_2021_pr": 1,
        "maxar_open_data": 2,
        "satellogic_earthview": 3,
    }
    return order.get(str(source_name), 99)


def _asset_group_columns() -> list[str]:
    return ["source", "item_id", "asset_role", "asset_href"]


def _asset_batch_limit() -> int | None:
    return MAX_ASSETS_THIS_RUN if MAX_ASSETS_THIS_RUN is not None else MAX_ITEMS_THIS_RUN


def _dedupe_exact_asset_candidates(asset_candidates: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop exact footprint/date duplicates while preserving the preferred source row."""

    if asset_candidates.empty:
        return asset_candidates

    deduped = asset_candidates.copy()
    bounds = deduped.geometry.bounds.round(6)
    deduped["_bounds_sig"] = bounds.astype(str).agg("|".join, axis=1)
    deduped["_acquired_sig"] = pd.to_datetime(deduped["acquired_at"], utc=True, errors="coerce").astype(str)
    deduped["_gsd_sig"] = pd.to_numeric(deduped["gsd"], errors="coerce").round(3)
    deduped["_source_rank"] = deduped["source"].map(_asset_source_rank)
    deduped = deduped.sort_values(
        by=["_source_rank", "_acquired_sig", "_gsd_sig", "item_id"],
        ascending=[True, False, True, True],
        na_position="last",
    )
    deduped = deduped.drop_duplicates(
        subset=["asset_role", "_acquired_sig", "_gsd_sig", "_bounds_sig"],
        keep="first",
    )
    return deduped.drop(columns=["_bounds_sig", "_acquired_sig", "_gsd_sig", "_source_rank"]).reset_index(
        drop=True
    )


def resolve_fetch_asset_href(asset_href: str) -> str:
    """Refresh Planetary Computer Azure blob URLs so persisted SAS tokens do not expire the fetch path."""

    parsed = urlsplit(str(asset_href))
    if not parsed.netloc.endswith("blob.core.windows.net"):
        return str(asset_href)

    base_href = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    try:
        refreshed = planetary_computer.sign_url(base_href)
    except Exception:
        return str(asset_href)
    return str(refreshed or asset_href)


def enumerate_target_items(
    catalog: gpd.GeoDataFrame,
    occupied_h3_cells: gpd.GeoDataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if catalog.empty or occupied_h3_cells.empty:
        return pd.DataFrame()

    ranked = catalog.copy()
    ranked["acquired_at"] = pd.to_datetime(ranked["acquired_at"], utc=True, errors="coerce")
    ranked = ranked.sort_values(
        by=["source", "acquired_at", "gsd", "item_id"],
        ascending=[True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)

    candidate_rows: list[dict[str, object]] = []
    for _, item in ranked.iterrows():
        asset_href, asset_role = choose_asset(item)
        if asset_href is None:
            continue
        candidate_rows.append(
            {
                "source": item["source"],
                "item_id": item["item_id"],
                "asset_role": asset_role,
                "asset_href": asset_href,
                "acquired_at": item.get("acquired_at"),
                "gsd": item.get("gsd"),
                "platform": item.get("platform"),
                "catalog_self_href": item.get("self_href"),
                "geometry": item.geometry,
            }
        )

    if not candidate_rows:
        return pd.DataFrame()

    asset_candidates = gpd.GeoDataFrame(candidate_rows, geometry="geometry", crs="EPSG:4326")
    asset_candidates = _dedupe_exact_asset_candidates(asset_candidates)
    cells_with_id = occupied_h3_cells.reset_index(drop=True).copy()
    cells_with_id["_cell_row_id"] = cells_with_id.index
    joined = gpd.sjoin(
        asset_candidates,
        cells_with_id,
        how="inner",
        predicate="intersects",
        lsuffix="asset",
        rsuffix="cell",
    )
    if joined.empty:
        return pd.DataFrame()

    clip_geometries = cells_with_id.set_index("_cell_row_id").geometry
    cell_index_column = "_cell_row_id"
    if cell_index_column not in joined.columns:
        for candidate in ("index_cell", "index_right"):
            if candidate in joined.columns:
                cell_index_column = candidate
                break
    for _, row in joined.iterrows():
        rows.append(
            {
                "source": row["source"],
                "item_id": row["item_id"],
                "municipio": row["municipio"],
                "municipio_geoid": row["municipio_geoid"],
                "h3_cell_id": row["h3_cell_id"],
                "h3_resolution": row["h3_resolution"],
                "building_count": row["building_count"],
                "municipality_building_count": row["municipality_building_count"],
                "crosses_municipality_boundary": row["crosses_municipality_boundary"],
                "asset_role": row["asset_role"],
                "asset_href": row["asset_href"],
                "acquired_at": row.get("acquired_at"),
                "gsd": row.get("gsd"),
                "platform": row.get("platform"),
                "catalog_self_href": row.get("catalog_self_href"),
                "geometry": clip_geometries.loc[int(row[cell_index_column])],
            }
        )

    targets = pd.DataFrame(rows)
    if targets.empty:
        return targets
    targets = targets.drop_duplicates(subset=_asset_group_columns() + ["municipio", "h3_cell_id"])
    targets = targets.sort_values(
        by=["municipio", "h3_cell_id", "source", "acquired_at", "item_id", "asset_role"],
        ascending=[True, True, True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return targets


def _square_bounds_from_geometry(geometry_3857, *, padding_factor: float = STAC_TILE_PADDING_FACTOR) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = geometry_3857.bounds
    width = maxx - minx
    height = maxy - miny
    half_span = max(width, height, 1.0) * max(padding_factor, 1.0) / 2.0
    centroid = geometry_3857.centroid
    return (
        float(centroid.x - half_span),
        float(centroid.y - half_span),
        float(centroid.x + half_span),
        float(centroid.y + half_span),
    )


def _square_polygon_for_geometry(geometry_3857):
    return box(*_square_bounds_from_geometry(geometry_3857))


def build_asset_fetch_queue(targets: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-cell intersections into one cache-fetch row per unique STAC asset."""

    if targets.empty:
        return pd.DataFrame()

    asset_rows: list[dict[str, object]] = []
    for asset_key, group in targets.groupby(_asset_group_columns(), sort=False):
        source, item_id, asset_role, asset_href = asset_key
        municipalities = sorted({str(value) for value in group["municipio"].dropna().astype(str)})
        municipio_label = municipalities[0] if len(municipalities) == 1 else "multiple_municipios"
        cell_geometries_3857 = gpd.GeoSeries(list(group["geometry"]), crs="EPSG:4326").to_crs(MODEL_READY_CRS)
        coverage_polygons_3857 = cell_geometries_3857.map(_square_polygon_for_geometry)
        coverage_geometry_3857 = gpd.GeoSeries(list(coverage_polygons_3857), crs=MODEL_READY_CRS).union_all()
        coverage_geometry = gpd.GeoSeries([coverage_geometry_3857], crs=MODEL_READY_CRS).to_crs("EPSG:4326").iloc[0]
        local_dir = LOCAL_STAC_ROOT / slugify(str(source)) / slugify(municipio_label) / slugify(str(item_id))
        tile_dir = OUTPUT_STAC_TILE_ROOT / slugify(municipio_label) / slugify(str(source)) / slugify(str(item_id))
        first_row = group.iloc[0]
        asset_rows.append(
            {
                "source": source,
                "item_id": item_id,
                "asset_role": asset_role,
                "asset_href": asset_href,
                "primary_municipio": municipio_label,
                "municipios": json.dumps(municipalities, ensure_ascii=True),
                "municipio_count": len(municipalities),
                "municipio_geoid_count": int(group["municipio_geoid"].nunique(dropna=True)),
                "h3_cell_count": int(group["h3_cell_id"].nunique()),
                "building_count": int(group["building_count"].sum()),
                "municipality_building_count": int(group["municipality_building_count"].sum()),
                "crosses_municipality_boundary": bool(group["crosses_municipality_boundary"].any()),
                "acquired_at": first_row.get("acquired_at"),
                "gsd": first_row.get("gsd"),
                "platform": first_row.get("platform"),
                "catalog_self_href": first_row.get("catalog_self_href"),
                "local_asset_path": str(local_dir / f"{item_id}_{asset_role}_epsg3857.tif"),
                "tile_output_dir": str(tile_dir),
                "geometry": coverage_geometry,
            }
        )

    asset_queue = pd.DataFrame(asset_rows)
    if asset_queue.empty:
        return asset_queue

    asset_queue = asset_queue.sort_values(
        by=["primary_municipio", "source", "acquired_at", "item_id", "asset_role"],
        ascending=[True, True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    asset_queue["_batch_round"] = asset_queue.groupby("primary_municipio", sort=False).cumcount()
    asset_queue = asset_queue.sort_values(
        by=["_batch_round", "primary_municipio", "source", "acquired_at", "item_id", "asset_role"],
        ascending=[True, True, True, False, True, True],
        na_position="last",
    ).drop(columns=["_batch_round"]).reset_index(drop=True)
    return asset_queue


def filter_pending_asset_queue(asset_queue: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if asset_queue.empty:
        return asset_queue, asset_queue

    if OVERWRITE_EXISTING:
        return asset_queue.reset_index(drop=True), asset_queue.iloc[0:0].copy()

    cache_exists = asset_queue["local_asset_path"].map(lambda value: Path(str(value)).exists())
    pending = asset_queue.loc[~cache_exists].copy().reset_index(drop=True)
    existing = asset_queue.loc[cache_exists].copy().reset_index(drop=True)
    return pending, existing


def merge_manifest_rows(
    manifest_path: Path,
    rows: pd.DataFrame,
    *,
    key_columns: list[str],
) -> pd.DataFrame:
    if rows.empty:
        return rows

    merged = rows.copy()
    if manifest_path.exists():
        try:
            existing = pd.read_parquet(manifest_path)
        except Exception:
            existing = pd.DataFrame()
        if not existing.empty and set(key_columns).issubset(existing.columns):
            merged = pd.concat([existing, merged], ignore_index=True)
            merged = merged.drop_duplicates(subset=key_columns, keep="last").reset_index(drop=True)
    return merged


def reproject_to_model_ready_crs(
    clipped_image: np.ndarray,
    clipped_transform: rasterio.Affine,
    *,
    src_crs: rasterio.crs.CRS,
    dst_crs: str = MODEL_READY_CRS,
    nodata: float | int | None,
) -> tuple[np.ndarray, rasterio.Affine, str, float | int | None]:
    if str(src_crs).upper() == dst_crs.upper():
        return clipped_image, clipped_transform, dst_crs, nodata

    if nodata is None:
        nodata = 0 if np.issubdtype(clipped_image.dtype, np.integer) else np.nan

    bounds = array_bounds(clipped_image.shape[1], clipped_image.shape[2], clipped_transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        dst_crs,
        clipped_image.shape[2],
        clipped_image.shape[1],
        *bounds,
    )
    if dst_width <= 0 or dst_height <= 0:
        raise RuntimeError("Reprojected raster window is empty.")

    destination = np.full((clipped_image.shape[0], dst_height, dst_width), nodata, dtype=clipped_image.dtype)
    for band_idx in range(clipped_image.shape[0]):
        reproject(
            source=clipped_image[band_idx],
            destination=destination[band_idx],
            src_transform=clipped_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=nodata,
            dst_nodata=nodata,
            resampling=Resampling.bilinear,
        )
    return destination, dst_transform, dst_crs, nodata


def _write_raster(
    path: Path,
    data: np.ndarray,
    *,
    transform,
    crs: str,
    nodata: float | int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "driver": "GTiff",
        "height": int(data.shape[1]),
        "width": int(data.shape[2]),
        "count": int(data.shape[0]),
        "dtype": data.dtype,
        "transform": transform,
        "crs": crs,
        "compress": "deflate",
        "tiled": True,
    }
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        meta["nodata"] = nodata
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data)


def _build_asset_manifest_row(
    asset_row: pd.Series,
    *,
    local_path: Path,
    status: str,
    error: str | None,
    native_proj_epsg: int | None,
    model_ready_proj_epsg: int | None,
    tile_rows_written: int,
    tile_rows_reused: int,
    tile_rows_error: int,
) -> dict[str, object]:
    return {
        "source": asset_row["source"],
        "item_id": asset_row["item_id"],
        "asset_role": asset_row["asset_role"],
        "asset_href": asset_row["asset_href"],
        "primary_municipio": asset_row["primary_municipio"],
        "municipios": asset_row["municipios"],
        "municipio_count": asset_row["municipio_count"],
        "municipio_geoid_count": asset_row["municipio_geoid_count"],
        "h3_cell_count": asset_row["h3_cell_count"],
        "building_count": asset_row["building_count"],
        "municipality_building_count": asset_row["municipality_building_count"],
        "crosses_municipality_boundary": asset_row["crosses_municipality_boundary"],
        "local_asset_path": _project_relative_or_absolute(local_path),
        "tile_output_dir": _project_relative_or_absolute(Path(str(asset_row["tile_output_dir"]))),
        "status": status,
        "error": error,
        "acquired_at": asset_row.get("acquired_at"),
        "gsd": asset_row.get("gsd"),
        "platform": asset_row.get("platform"),
        "catalog_self_href": asset_row.get("catalog_self_href"),
        "native_proj_epsg": native_proj_epsg,
        "model_ready_proj_epsg": model_ready_proj_epsg or 3857,
        "tile_rows_written": tile_rows_written,
        "tile_rows_reused": tile_rows_reused,
        "tile_rows_error": tile_rows_error,
    }


def _write_asset_sidecar(
    asset_row: pd.Series,
    *,
    target_rows: pd.DataFrame,
    sidecar_path: Path,
    local_path: Path,
    native_proj_epsg: int | None,
    model_ready_proj_epsg: int | None,
) -> None:
    acquired_at = asset_row.get("acquired_at")
    if pd.notna(acquired_at):
        acquired_at = pd.Timestamp(acquired_at).isoformat()
    else:
        acquired_at = None
    sidecar_path.write_text(
        json.dumps(
            {
                "source": asset_row["source"],
                "item_id": asset_row["item_id"],
                "asset_role": asset_row["asset_role"],
                "municipios": json.loads(asset_row["municipios"]),
                "h3_cell_ids": sorted(target_rows["h3_cell_id"].dropna().astype(str).unique().tolist()),
                "h3_cell_count": int(asset_row["h3_cell_count"]),
                "building_count": int(asset_row["building_count"]),
                "municipality_building_count": int(asset_row["municipality_building_count"]),
                "crosses_municipality_boundary": bool(asset_row["crosses_municipality_boundary"]),
                "acquired_at": acquired_at,
                "gsd": asset_row.get("gsd"),
                "platform": asset_row.get("platform"),
                "original_asset_href": asset_row["asset_href"],
                "catalog_self_href": asset_row.get("catalog_self_href"),
                "clip_strategy": "square_h3_tile_union",
                "native_proj_epsg": native_proj_epsg,
                "model_ready_proj_epsg": model_ready_proj_epsg or 3857,
                "local_asset_path": _project_relative_or_absolute(local_path),
                "tile_output_dir": _project_relative_or_absolute(Path(str(asset_row["tile_output_dir"]))),
                "tile_pixels": STAC_TILE_PIXELS,
                "tile_padding_factor": STAC_TILE_PADDING_FACTOR,
            },
            indent=2,
        )
    )


def fetch_asset_cache(asset_row: pd.Series, target_rows: pd.DataFrame) -> tuple[dict[str, object], bool]:
    """Fetch one cache raster covering the union of square tile footprints for a STAC asset."""

    local_path = Path(str(asset_row["local_asset_path"]))
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = local_path.with_name(f"{local_path.stem}_meta.json")
    model_ready_proj_epsg = rasterio.crs.CRS.from_string(MODEL_READY_CRS).to_epsg()

    if not OVERWRITE_EXISTING and local_path.exists():
        if not sidecar_path.exists():
            _write_asset_sidecar(
                asset_row,
                target_rows=target_rows,
                sidecar_path=sidecar_path,
                local_path=local_path,
                native_proj_epsg=None,
                model_ready_proj_epsg=model_ready_proj_epsg,
            )
        return (
            _build_asset_manifest_row(
                asset_row,
                local_path=local_path,
                status="reused",
                error=None,
                native_proj_epsg=None,
                model_ready_proj_epsg=model_ready_proj_epsg,
                tile_rows_written=0,
                tile_rows_reused=0,
                tile_rows_error=0,
            ),
            False,
        )

    fetch_asset_href = resolve_fetch_asset_href(str(asset_row["asset_href"]))
    try:
        with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
            with rasterio.open(fetch_asset_href) as src:
                if src.crs is None:
                    raise RuntimeError("Source raster is missing CRS metadata.")
                geometry = mapping(asset_row["geometry"])
                if str(src.crs).upper() != "EPSG:4326":
                    geometry = transform_geom("EPSG:4326", src.crs, geometry, precision=6)
                clipped_image, clipped_transform = mask(src, [geometry], crop=True)
                if clipped_image.shape[1] == 0 or clipped_image.shape[2] == 0:
                    raise RuntimeError("Clip produced an empty raster window.")

                native_proj_epsg = src.crs.to_epsg() if src.crs else None
                clipped_image, clipped_transform, _, nodata = reproject_to_model_ready_crs(
                    clipped_image,
                    clipped_transform,
                    src_crs=src.crs,
                    dst_crs=MODEL_READY_CRS,
                    nodata=src.nodata,
                )
                _write_raster(
                    local_path,
                    clipped_image,
                    transform=clipped_transform,
                    crs=MODEL_READY_CRS,
                    nodata=nodata,
                )
                _write_asset_sidecar(
                    asset_row,
                    target_rows=target_rows,
                    sidecar_path=sidecar_path,
                    local_path=local_path,
                    native_proj_epsg=native_proj_epsg,
                    model_ready_proj_epsg=model_ready_proj_epsg,
                )
        return (
            _build_asset_manifest_row(
                asset_row,
                local_path=local_path,
                status="fetched",
                error=None,
                native_proj_epsg=native_proj_epsg,
                model_ready_proj_epsg=model_ready_proj_epsg,
                tile_rows_written=0,
                tile_rows_reused=0,
                tile_rows_error=0,
            ),
            True,
        )
    except Exception as exc:
        return (
            _build_asset_manifest_row(
                asset_row,
                local_path=local_path,
                status="error",
                error=str(exc),
                native_proj_epsg=None,
                model_ready_proj_epsg=model_ready_proj_epsg,
                tile_rows_written=0,
                tile_rows_reused=0,
                tile_rows_error=0,
            ),
            False,
        )


def _build_tile_manifest_row(
    tile_row: pd.Series,
    asset_row: pd.Series,
    *,
    tile_path: Path,
    status: str,
    error: str | None,
    bounds_3857: tuple[float, float, float, float] | None,
) -> dict[str, object]:
    return {
        "source": asset_row["source"],
        "item_id": asset_row["item_id"],
        "asset_role": asset_row["asset_role"],
        "municipio": tile_row["municipio"],
        "municipio_geoid": tile_row["municipio_geoid"],
        "h3_cell_id": tile_row["h3_cell_id"],
        "h3_resolution": tile_row["h3_resolution"],
        "building_count": tile_row["building_count"],
        "municipality_building_count": tile_row["municipality_building_count"],
        "crosses_municipality_boundary": tile_row["crosses_municipality_boundary"],
        "local_asset_path": _project_relative_or_absolute(Path(str(asset_row["local_asset_path"]))),
        "tile_path": _project_relative_or_absolute(tile_path),
        "status": status,
        "error": error,
        "tile_pixels": STAC_TILE_PIXELS,
        "west_3857": None if bounds_3857 is None else bounds_3857[0],
        "south_3857": None if bounds_3857 is None else bounds_3857[1],
        "east_3857": None if bounds_3857 is None else bounds_3857[2],
        "north_3857": None if bounds_3857 is None else bounds_3857[3],
    }


def export_square_tiles_for_asset(asset_row: pd.Series, target_rows: pd.DataFrame) -> list[dict[str, object]]:
    """Materialize square training/inference tiles from one cached asset raster."""

    if target_rows.empty:
        return []

    local_asset_path = Path(str(asset_row["local_asset_path"]))
    if not local_asset_path.exists():
        return [
            _build_tile_manifest_row(
                target_rows.iloc[idx],
                asset_row,
                tile_path=Path(str(asset_row["tile_output_dir"])) / f"{target_rows.iloc[idx]['h3_cell_id']}_{asset_row['asset_role']}_epsg3857.tif",
                status="error",
                error="Cached asset raster does not exist.",
                bounds_3857=None,
            )
            for idx in range(len(target_rows))
        ]

    tile_rows = target_rows.reset_index(drop=True).copy()
    tile_geometries_3857 = gpd.GeoSeries(list(tile_rows["geometry"]), crs="EPSG:4326").to_crs(MODEL_READY_CRS)
    results: list[dict[str, object]] = []

    with rasterio.open(local_asset_path) as src:
        for idx in range(len(tile_rows)):
            row = tile_rows.iloc[idx]
            bounds_3857 = _square_bounds_from_geometry(tile_geometries_3857.iloc[idx])
            tile_path = Path(str(asset_row["tile_output_dir"])) / f"{row['h3_cell_id']}_{asset_row['asset_role']}_epsg3857.tif"

            if not OVERWRITE_EXISTING and tile_path.exists():
                results.append(
                    _build_tile_manifest_row(
                        row,
                        asset_row,
                        tile_path=tile_path,
                        status="reused",
                        error=None,
                        bounds_3857=bounds_3857,
                    )
                )
                continue

            try:
                window = from_bounds(*bounds_3857, transform=src.transform)
                window = window.intersection(Window(0, 0, src.width, src.height))
                if window.width <= 0 or window.height <= 0:
                    raise RuntimeError("Square tile window is outside the cached raster extent.")
                image = src.read(
                    window=window,
                    out_shape=(src.count, STAC_TILE_PIXELS, STAC_TILE_PIXELS),
                    resampling=Resampling.bilinear,
                )
                base_transform = src.window_transform(window)
                scaled_transform = base_transform * Affine.scale(
                    window.width / STAC_TILE_PIXELS,
                    window.height / STAC_TILE_PIXELS,
                )
                _write_raster(
                    tile_path,
                    image,
                    transform=scaled_transform,
                    crs=str(src.crs),
                    nodata=src.nodata,
                )
                results.append(
                    _build_tile_manifest_row(
                        row,
                        asset_row,
                        tile_path=tile_path,
                        status="fetched",
                        error=None,
                        bounds_3857=bounds_3857,
                    )
                )
            except Exception as exc:
                results.append(
                    _build_tile_manifest_row(
                        row,
                        asset_row,
                        tile_path=tile_path,
                        status="error",
                        error=str(exc),
                        bounds_3857=bounds_3857,
                    )
                )

    return results

# %%
if __name__ == "__main__":
    con = duckdb.connect(str(resolve_db_path()))
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("LOAD h3;")

    municipalities = load_target_municipalities(con)
    occupied_h3_cells = load_target_h3_cells(con)
    print(f"loaded {len(occupied_h3_cells):,} occupied H3 cells for target municipalities")
    if occupied_h3_cells.empty:
        print("no occupied H3 cells found in DuckDB.")
        con.close()
        sys.exit(0)

    catalog = load_catalog()
    print(f"loaded {len(catalog):,} STAC catalog items from {STAC_CATALOG_PATH}")

    targets = enumerate_target_items(catalog, occupied_h3_cells)
    print(f"intersecting occupied-H3 tile footprints queued: {len(targets):,}")
    if targets.empty:
        print("no intersecting STAC assets were found for the occupied H3 cells.")
        con.close()
        sys.exit(0)

    target_groups = {
        asset_key: group.reset_index(drop=True)
        for asset_key, group in targets.groupby(_asset_group_columns(), sort=False)
    }
    asset_queue = build_asset_fetch_queue(targets)
    pending_assets, existing_assets = filter_pending_asset_queue(asset_queue)
    batch_limit = _asset_batch_limit()
    if batch_limit is not None:
        pending_assets = pending_assets.head(batch_limit).reset_index(drop=True)

    asset_summary = (
        asset_queue.groupby(["primary_municipio", "source"])
        .agg(total_assets=("item_id", "size"), total_h3_cells=("h3_cell_count", "sum"))
        .reset_index()
    )
    pending_summary = (
        pending_assets.groupby(["primary_municipio", "source"])
        .agg(pending_assets=("item_id", "size"))
        .reset_index()
        if not pending_assets.empty
        else pd.DataFrame(columns=["primary_municipio", "source", "pending_assets"])
    )
    asset_summary = asset_summary.merge(pending_summary, on=["primary_municipio", "source"], how="left")
    asset_summary["pending_assets"] = asset_summary["pending_assets"].fillna(0).astype(int)
    print("asset queue by municipality/source:")
    print(asset_summary.to_string(index=False))
    print(
        f"cached assets already present: {len(existing_assets):,} | "
        f"assets selected this run: {len(pending_assets):,}"
    )

    asset_manifest_rows: list[dict[str, object]] = []
    tile_manifest_rows: list[dict[str, object]] = []
    for _, asset_row in pending_assets.iterrows():
        asset_key = (
            asset_row["source"],
            asset_row["item_id"],
            asset_row["asset_role"],
            asset_row["asset_href"],
        )
        target_rows = target_groups[asset_key]
        asset_manifest_row, _ = fetch_asset_cache(asset_row, target_rows)
        if EXPORT_STAC_SQUARE_TILES and asset_manifest_row["status"] in {"fetched", "reused"}:
            tile_rows = export_square_tiles_for_asset(asset_row, target_rows)
            asset_manifest_row["tile_rows_written"] = int(sum(1 for row in tile_rows if row["status"] == "fetched"))
            asset_manifest_row["tile_rows_reused"] = int(sum(1 for row in tile_rows if row["status"] == "reused"))
            asset_manifest_row["tile_rows_error"] = int(sum(1 for row in tile_rows if row["status"] == "error"))
            tile_manifest_rows.extend(tile_rows)
        asset_manifest_rows.append(asset_manifest_row)

    manifest = pd.DataFrame(asset_manifest_rows)
    manifest = merge_manifest_rows(
        LOCAL_FETCH_MANIFEST_PATH,
        manifest,
        key_columns=_asset_group_columns(),
    )
    LOCAL_FETCH_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(LOCAL_FETCH_MANIFEST_PATH, index=False)
    print(f"wrote {len(manifest):,} asset-cache rows to {LOCAL_FETCH_MANIFEST_PATH}")
    if not manifest.empty:
        print(manifest.groupby(["primary_municipio", "source", "status"]).size().to_string())

    tile_manifest = pd.DataFrame(tile_manifest_rows)
    if not tile_manifest.empty:
        tile_manifest = merge_manifest_rows(
            OUTPUT_STAC_TILE_MANIFEST_PATH,
            tile_manifest,
            key_columns=["source", "item_id", "asset_role", "municipio", "h3_cell_id"],
        )
        OUTPUT_STAC_TILE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        tile_manifest.to_parquet(OUTPUT_STAC_TILE_MANIFEST_PATH, index=False)
        print(f"wrote {len(tile_manifest):,} square-tile rows to {OUTPUT_STAC_TILE_MANIFEST_PATH}")
        print(tile_manifest.groupby(["municipio", "source", "status"]).size().to_string())
    con.close()

# %% [markdown]
# ## Municipality coverage QA
# 
# The first preview overlays every candidate STAC item footprint intersecting the
# target municipalities, then summarizes the percentage of municipal land area
# touched by those candidate extents. The second preview uses only locally
# cached asset clips (`fetched` or `reused`) and adds the share of
# H3-attributed buildings covered by those clipped rasters.

# %%
if __name__ == "__main__" and SHOW_COVERAGE_PREVIEW:
    catalog_overlay = build_catalog_extent_overlay(catalog, municipalities)
    catalog_coverage_summary = summarize_land_coverage(municipalities, catalog_overlay)
    plot_municipality_coverage(
        municipalities,
        catalog_overlay,
        catalog_coverage_summary,
        suptitle="Candidate STAC item extent coverage by municipality",
        include_buildings=False,
    )

    if manifest.empty:
        print("cached-raster coverage preview skipped: no asset manifest rows are available yet.")
    else:
        cached_overlay = build_cached_asset_extent_overlay(asset_queue, manifest, municipalities)
        covered_targets = filter_targets_by_manifest(targets, manifest)
        cached_summary = summarize_land_coverage(municipalities, cached_overlay)
        building_summary = summarize_building_coverage(occupied_h3_cells, covered_targets)
        cached_summary = cached_summary.merge(building_summary, on="municipio", how="left")
        for column_name in ("total_buildings", "covered_buildings", "covered_building_pct"):
            if column_name not in cached_summary.columns:
                cached_summary[column_name] = 0
        cached_summary[["total_buildings", "covered_buildings"]] = cached_summary[["total_buildings", "covered_buildings"]].fillna(0).astype(int)
        cached_summary["covered_building_pct"] = cached_summary["covered_building_pct"].fillna(0.0)
        plot_municipality_coverage(
            municipalities,
            cached_overlay,
            cached_summary,
            suptitle="Locally cached clipped-raster coverage and building coverage",
            include_buildings=True,
        )


