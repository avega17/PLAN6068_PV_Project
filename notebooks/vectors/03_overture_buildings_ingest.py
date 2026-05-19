# %% [markdown]
# # Puerto Rico Overture Buildings Ingestion (Island-Wide)
# 
# This notebook/script ingests Overture building footprints for Puerto Rico as
# an island-wide cacheable artifact, then persists the cleaned result into the
# project DuckDB. It follows a progressive, step-wise workflow:
# 
# 1. Connect to local DuckDB and extract municipality boundaries.
# 2. Rank municipalities using available PV label evidence.
# 3. Build or reuse a whole-island Overture GeoParquet cache.
# 4. Keep only analysis-critical columns and load to DuckDB.
# 5. Build a spatially optimized table with Hilbert ordering + RTree index.
# 6. Preview a municipality subset in lonboard with optional neighborhood clip.

# %%
"""03_overture_buildings_ingest.py

Jupytext-friendly workflow for ingesting Overture Maps building footprints into
the local Puerto Rico DuckDB database.

Default scope:
- whole-island building fetch via the official overturemaps client
- persisted GeoParquet cache reused across reruns
- column-pruned schema for performant downstream joins
- native DuckDB GEOMETRY storage and spatial indexing
"""

# %%
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import geopandas as gpd
import ipywidgets as widgets
import numpy as np
import overturemaps
import pandas as pd
import osmnx as ox
import pyarrow.parquet as pq
from dotenv import load_dotenv
from IPython.display import display
from lonboard import Map, PolygonLayer
from matplotlib import cm
from matplotlib.colors import LogNorm
from overturemaestro import functions as omt
from shapely import from_wkb
from shapely.geometry.base import BaseGeometry


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

from utils.overture import occupied_h3_cells_sql

load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_CRS = "EPSG:4326"
TARGET_MUNICIPALITY = "San Juan"
DEFAULT_VECTOR_DB_PATH = PROJECT_ROOT / "data" / "PR_PV_plan_data.duckdb"
# Municipalities that must be ingested regardless of top-K PV-label ranking.
# San Juan is already the top-ranked by OSM PV labels; Isabela is a second
# project target (Barrio Mora) that sits outside the default top-10 and is
# force-included here so the Solar API tile pipeline has full coverage.
FORCE_INCLUDE_MUNICIPALITIES: list[str] = ["San Juan", "Isabela"]
TARGET_NEIGHBORHOOD_QUERY: str | None = None
OVERTURE_THEME = "buildings"
OVERTURE_TYPE = "building"
OVERTURE_TABLE = "pr_overture_buildings"
OVERTURE_WORKING_DIR = PROJECT_ROOT / "cache" / "overture_tmp"
OVERTURE_FETCH_BACKEND = os.getenv("OVERTURE_FETCH_BACKEND", "official_island_bbox").strip().lower()
OVERTURE_REFRESH_GEOPARQUET_CACHE = os.getenv("OVERTURE_REFRESH_GEOPARQUET_CACHE", "0") == "1"
CLEAN_LEGACY_OVERTURE_ARTIFACTS = os.getenv("CLEAN_LEGACY_OVERTURE_ARTIFACTS", "1") == "1"
TOP_K_MUNICIPALITIES = 10
OVERTURE_FETCH_STRATEGY = "single_union"
OVERTURE_DOWNLOAD_MAX_WORKERS: int | None = 8
MAP_PREVIEW_LIMIT = 4000
ENABLE_3D_PREVIEW = True
HEIGHT_FROM_FLOOR_METERS = 3.2
MIN_PREVIEW_ELEVATION_METERS = 1.0
OVERTURE_RELEASE = os.getenv("OVERTURE_RELEASE", "2026-03-18.0")
H3_RESOLUTION = int(os.getenv("OVERTURE_BUILDING_H3_RESOLUTION", "10"))
RUN_ISLAND_BBOX_OFFICIAL_EXPORT = True
ISLAND_BBOX_EXPORT_FILENAME = f"pr_overture_buildings_island_bbox_{OVERTURE_RELEASE}.parquet"
ISLAND_BBOX_EXPORT_DIR = PROJECT_ROOT / "data" / "vectors" / "cache"
ISLAND_BBOX_EXPORT_COMPRESSION = "zstd"

CRITICAL_BUILDING_COLUMNS = [
    "id",
    "geometry",
    "subtype",
    "class",
    "names",
    "height",
    "num_floors",
    "roof_material",
    "roof_direction",
    "roof_orientation",
    "roof_height",
]

OVERTURE_COLUMNS_TO_DOWNLOAD = CRITICAL_BUILDING_COLUMNS.copy()


def format_size_bytes(size_bytes: int) -> str:
    """Format bytes into a human-readable size string."""

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(max(size_bytes, 0))
    unit_idx = 0
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"


def resolve_db_path() -> Path:
    """Resolve the project DuckDB path with env override and stable default."""

    db_path_value = os.getenv("VECTOR_DB")
    if db_path_value:
        db_path = Path(db_path_value)
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path if len(db_path.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / db_path
        return db_path

    return DEFAULT_VECTOR_DB_PATH


def create_spatial_connection(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open DuckDB and load extensions needed for geometry + parquet access."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    try:
        con.execute("INSTALL parquet;")
        con.execute("LOAD parquet;")
    except duckdb.Error:
        # Parquet is bundled in many DuckDB builds.
        pass
    load_h3_extension(con)
    return con


def load_h3_extension(con: duckdb.DuckDBPyConnection) -> bool:
    """Load DuckDB's community H3 extension when available."""

    try:
        con.execute("INSTALL h3 FROM community;")
        con.execute("LOAD h3;")
        return True
    except duckdb.Error as exc:
        print(f"[warn] DuckDB h3 extension unavailable: {exc}")
        return False


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """Check if a main-schema table exists in DuckDB."""

    result = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?;
        """,
        [table_name],
    ).fetchone()
    return bool(result and result[0])


def _to_bytes(value: object) -> bytes:
    """Normalize memoryview/bytearray values returned by DuckDB to bytes."""

    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    return bytes(value)


def is_native_duckdb_geometry_type(type_name: object) -> bool:
    """Return True when DuckDB already exposes a native GEOMETRY column."""

    return isinstance(type_name, str) and type_name.strip().upper().startswith("GEOMETRY")


def resolve_geometry_sql_expression(
    con: duckdb.DuckDBPyConnection,
    relation_sql: str,
    geometry_column: str = "geometry",
) -> str:
    """Choose passthrough vs WKB deserialization for a relation's geometry column."""

    schema_df = con.execute(
        f"""
        DESCRIBE SELECT {geometry_column}
        FROM {relation_sql}
        """
    ).fetchdf()
    if schema_df.empty or "column_type" not in schema_df.columns:
        raise RuntimeError(f"Unable to infer geometry type for {relation_sql}.{geometry_column}.")

    geometry_type = schema_df.iloc[0]["column_type"]
    return geometry_column if is_native_duckdb_geometry_type(geometry_type) else f"ST_GeomFromWKB({geometry_column})"


def fetch_municipality_boundaries(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    """Load Puerto Rico municipality polygons from local DuckDB."""

    df = con.execute(
        """
        SELECT
            GEOID AS municipality_geoid,
            NAME AS municipality_name,
            ST_AsWKB(geometry) AS geometry_wkb
        FROM pr_census_counties
        WHERE ST_GeometryType(geometry) IN ('POLYGON', 'MULTIPOLYGON')
        ORDER BY municipality_name;
        """
    ).fetchdf()

    geometry = gpd.GeoSeries(df["geometry_wkb"].map(lambda value: from_wkb(_to_bytes(value))), crs=OUTPUT_CRS)
    gdf = gpd.GeoDataFrame(df.drop(columns=["geometry_wkb"]), geometry=geometry, crs=OUTPUT_CRS)
    return gdf


def fetch_municipality_pv_counts(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, str]:
    """Compute municipality PV label counts from available project tables.

    Prefers cleaned rooftop PV polygons when present, otherwise falls back to a
    spatial join against QuackOSM stage records.
    """

    if table_exists(con, "pr_osm_rooftop_pv_polygons"):
        query = """
        SELECT
            municipality_geoid,
            municipality_name,
            COUNT(*) AS pv_labels
        FROM pr_osm_rooftop_pv_polygons
        GROUP BY ALL
        ORDER BY pv_labels DESC, municipality_name;
        """
        return con.execute(query).fetchdf(), "pr_osm_rooftop_pv_polygons"

    if table_exists(con, "pr_osm_quackosm_stage"):
        query = """
        SELECT
            m.GEOID AS municipality_geoid,
            m.NAME AS municipality_name,
            COUNT(s.feature_id) AS pv_labels
        FROM pr_census_counties AS m
        LEFT JOIN pr_osm_quackosm_stage AS s
          ON ST_Intersects(m.geometry, s.geometry)
         AND (
                lower(coalesce(CAST(s."generator:method" AS VARCHAR), '')) = 'photovoltaic'
             OR lower(coalesce(CAST(s."generator:source" AS VARCHAR), '')) = 'solar'
             OR lower(coalesce(CAST(s."generator:type" AS VARCHAR), '')) = 'solar_photovoltaic_panel'
         )
         AND lower(coalesce(CAST(s.content AS VARCHAR), '')) NOT IN ('hot_water', 'hot', 'water')
         AND lower(coalesce(CAST(s."generator:output:hot_water" AS VARCHAR), '')) <> 'yes'
        GROUP BY ALL
        ORDER BY pv_labels DESC, municipality_name;
        """
        return con.execute(query).fetchdf(), "pr_osm_quackosm_stage"

    query = """
    SELECT
        GEOID AS municipality_geoid,
        NAME AS municipality_name,
        CAST(coalesce(has_PV, FALSE) AS INTEGER) AS pv_labels
    FROM pr_census_counties
    ORDER BY pv_labels DESC, municipality_name;
    """
    return con.execute(query).fetchdf(), "pr_census_counties.has_PV"


def choose_target_municipality(
    municipalities_gdf: gpd.GeoDataFrame,
    pv_counts_df: pd.DataFrame,
    target_name: str,
) -> pd.Series:
    """Choose municipality by name, with top-ranked fallback."""

    ranked = municipalities_gdf.merge(
        pv_counts_df,
        on=["municipality_geoid", "municipality_name"],
        how="left",
    )
    ranked["pv_labels"] = ranked["pv_labels"].fillna(0).astype(int)
    ranked = ranked.sort_values(["pv_labels", "municipality_name"], ascending=[False, True]).reset_index(drop=True)

    explicit = ranked[ranked["municipality_name"].str.casefold() == target_name.casefold()]
    if not explicit.empty:
        return explicit.iloc[0]

    print(f"Requested municipality '{target_name}' was not found. Falling back to the highest PV-label municipality.")
    return ranked.iloc[0]


def rank_municipalities_by_pv_labels(
    municipalities_gdf: gpd.GeoDataFrame,
    pv_counts_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Return municipalities sorted by available PV-label evidence."""

    ranked = municipalities_gdf.merge(
        pv_counts_df,
        on=["municipality_geoid", "municipality_name"],
        how="left",
    )
    ranked["pv_labels"] = ranked["pv_labels"].fillna(0).astype(int)
    ranked = ranked.sort_values(["pv_labels", "municipality_name"], ascending=[False, True]).reset_index(drop=True)
    return ranked


def select_top_k_municipalities(
    ranked_municipalities: gpd.GeoDataFrame,
    top_k: int = TOP_K_MUNICIPALITIES,
    force_include: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Pick top-K municipalities with PV evidence; fallback to ranked order when needed.

    Any municipality names passed via ``force_include`` are always appended to
    the selection even when they fall outside the natural top-K. This is used
    to guarantee Solar-API-relevant municipalities (e.g. Isabela) are ingested
    regardless of their OSM-PV-label rank.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    with_labels = ranked_municipalities[ranked_municipalities["pv_labels"] > 0]
    candidates = with_labels if not with_labels.empty else ranked_municipalities
    selected = candidates.head(top_k).copy().reset_index(drop=True)

    if force_include:
        forced_names = {str(name).casefold() for name in force_include}
        already_selected = {str(name).casefold() for name in selected["municipality_name"].astype(str)}
        missing = forced_names - already_selected
        if missing:
            extras = ranked_municipalities[
                ranked_municipalities["municipality_name"].astype(str).str.casefold().isin(missing)
            ].copy()
            if not extras.empty:
                selected = pd.concat([selected, extras], ignore_index=True)
                selected = selected.drop_duplicates(subset=["municipality_geoid"]).reset_index(drop=True)

    if selected.empty:
        raise ValueError("No municipalities are available for top-K selection.")
    return selected


def build_union_geometry(municipalities_gdf: gpd.GeoDataFrame) -> BaseGeometry:
    """Union multiple municipality geometries into a single fetch boundary."""

    union_geometry = municipalities_gdf.geometry.union_all()
    if union_geometry is None or union_geometry.is_empty:
        raise ValueError("Top-K municipality union geometry is empty.")
    return union_geometry


def cleanup_legacy_overture_artifacts() -> list[Path]:
    """Remove legacy Overture sidecar/cache artifacts outside the parquet cache."""

    deleted_paths: list[Path] = []
    if not CLEAN_LEGACY_OVERTURE_ARTIFACTS:
        return deleted_paths

    legacy_roots = [
        PROJECT_ROOT / "data" / "vectors" / "cache",
        OVERTURE_WORKING_DIR,
    ]
    seen: set[Path] = set()
    for root in legacy_roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)
        for path in root.iterdir():
            if "overture" not in path.name.lower():
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            deleted_paths.append(path)
    return deleted_paths


def fetch_overture_buildings(
    geometry_filter: BaseGeometry | list[BaseGeometry],
    working_directory: Path,
    strategy: str = OVERTURE_FETCH_STRATEGY,
    columns_to_download: list[str] | None = None,
    max_workers: int | None = OVERTURE_DOWNLOAD_MAX_WORKERS,
) -> tuple[gpd.GeoDataFrame, list[Path]]:
    """Fetch Overture building footprints into parquet files, then read as a GeoDataFrame.

    Strategy modes:
    - single_union: one unioned geometry fetch (default)
    - per_geometry: one parquet per geometry in the provided list
    """

    working_directory.mkdir(parents=True, exist_ok=True)

    if strategy not in {"single_union", "per_geometry"}:
        raise ValueError("Unsupported strategy. Use 'single_union' or 'per_geometry'.")

    if strategy == "single_union":
        if isinstance(geometry_filter, list):
            raise TypeError("single_union strategy expects a single unioned geometry.")
        geometry_filters = [geometry_filter]
    else:
        if not isinstance(geometry_filter, list):
            raise TypeError("per_geometry strategy expects a list of geometries.")
        if not geometry_filter:
            raise ValueError("per_geometry strategy requires at least one geometry.")
        geometry_filters = geometry_filter

    parquet_paths: list[Path] = []
    gdf_parts: list[gpd.GeoDataFrame] = []
    effective_columns = columns_to_download or OVERTURE_COLUMNS_TO_DOWNLOAD

    for idx, current_geometry in enumerate(geometry_filters, start=1):
        out_path = working_directory / f"overture_{OVERTURE_THEME}_{OVERTURE_TYPE}_{strategy}_{idx:03d}.parquet"
        parquet_path = omt.convert_geometry_to_parquet(
            theme=OVERTURE_THEME,
            type=OVERTURE_TYPE,
            geometry_filter=current_geometry,
            release=None,
            columns_to_download=effective_columns,
            ignore_cache=False,
            result_file_path=str(out_path),
            working_directory=str(working_directory),
            verbosity_mode="verbose",
            max_workers=max_workers,
            sort_result=True,
        )
        parquet_paths.append(Path(parquet_path))

        current_gdf = gpd.read_parquet(parquet_path)
        if current_gdf.crs is None:
            current_gdf = current_gdf.set_crs(OUTPUT_CRS)
        else:
            current_gdf = current_gdf.to_crs(OUTPUT_CRS)
        gdf_parts.append(current_gdf)

    result = gpd.GeoDataFrame(pd.concat(gdf_parts, ignore_index=True), geometry="geometry", crs=OUTPUT_CRS)
    result = _ensure_overture_id_column(result)

    if strategy == "per_geometry" and "id" in result.columns:
        result = result.drop_duplicates(subset=["id"]).copy()

    return result, parquet_paths


def _ensure_overture_id_column(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure Overture ID is materialized as a regular column.

    OvertureMaestro geodataframe utilities often store id in the GeoDataFrame
    index, so this recovers it into the explicit id column for downstream SQL
    and schema-pruning steps.
    """

    fixed = gdf.copy()
    if "id" not in fixed.columns and str(getattr(fixed.index, "name", "")).casefold() == "id":
        fixed["id"] = fixed.index.astype("string")
    if "id" in fixed.columns:
        fixed["id"] = fixed["id"].astype("string")
    return fixed


def assign_buildings_to_municipalities(
    buildings_gdf: gpd.GeoDataFrame,
    municipalities_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Assign municipality attributes to building footprints using representative points."""

    enriched = buildings_gdf.copy()
    if enriched.empty:
        enriched["municipality_geoid"] = None
        enriched["municipality_name"] = None
        return enriched

    if enriched.crs is None:
        enriched = enriched.set_crs(OUTPUT_CRS)
    else:
        enriched = enriched.to_crs(OUTPUT_CRS)

    municipality_lookup = municipalities_gdf[["municipality_geoid", "municipality_name", "geometry"]].copy()
    if municipality_lookup.crs is None:
        municipality_lookup = municipality_lookup.set_crs(OUTPUT_CRS)
    else:
        municipality_lookup = municipality_lookup.to_crs(OUTPUT_CRS)

    building_points = gpd.GeoDataFrame(
        geometry=enriched.geometry.representative_point(),
        index=enriched.index,
        crs=OUTPUT_CRS,
    )
    point_matches = gpd.sjoin(building_points, municipality_lookup, how="left", predicate="within")
    first_point_matches = point_matches[["municipality_geoid", "municipality_name"]].groupby(level=0).first()

    enriched["municipality_geoid"] = first_point_matches["municipality_geoid"]
    enriched["municipality_name"] = first_point_matches["municipality_name"]

    unresolved = enriched["municipality_name"].isna()
    if unresolved.any():
        unresolved_polygons = gpd.GeoDataFrame(
            enriched.loc[unresolved, ["geometry"]].copy(),
            geometry="geometry",
            crs=OUTPUT_CRS,
        )
        polygon_matches = gpd.sjoin(unresolved_polygons, municipality_lookup, how="left", predicate="intersects")
        first_polygon_matches = polygon_matches[["municipality_geoid", "municipality_name"]].groupby(level=0).first()
        if not first_polygon_matches.empty:
            enriched.loc[first_polygon_matches.index, "municipality_geoid"] = first_polygon_matches["municipality_geoid"]
            enriched.loc[first_polygon_matches.index, "municipality_name"] = first_polygon_matches["municipality_name"]

    return enriched


def _jsonify_value(value: object) -> str | None:
    """Convert nested Overture objects to stable JSON strings."""

    def _to_json_compatible(item: object) -> object:
        if item is None:
            return None

        if isinstance(item, pd.Timestamp):
            return item.isoformat()

        if isinstance(item, dict):
            return {str(key): _to_json_compatible(val) for key, val in item.items()}

        if isinstance(item, (list, tuple, set)):
            return [_to_json_compatible(val) for val in item]

        if hasattr(item, "tolist") and not isinstance(item, (str, bytes, bytearray, memoryview)):
            try:
                return _to_json_compatible(item.tolist())
            except Exception:
                pass

        if hasattr(item, "item") and callable(getattr(item, "item")):
            try:
                return _to_json_compatible(item.item())
            except Exception:
                pass

        try:
            if pd.api.types.is_scalar(item) and pd.isna(item):
                return None
        except (TypeError, ValueError):
            pass

        if isinstance(item, (str, int, float, bool)):
            return item

        return str(item)

    normalized = _to_json_compatible(value)
    if normalized is None:
        return None

    if isinstance(normalized, (dict, list)):
        return json.dumps(normalized, ensure_ascii=True, sort_keys=True)

    return str(normalized)


def clean_buildings_schema(
    raw_gdf: gpd.GeoDataFrame,
    municipalities_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Keep critical columns and normalize nested/numeric values."""

    cleaned = _ensure_overture_id_column(raw_gdf)
    cleaned = assign_buildings_to_municipalities(cleaned, municipalities_gdf)

    selected_columns = CRITICAL_BUILDING_COLUMNS + ["municipality_name", "municipality_geoid"]

    for column_name in selected_columns:
        if column_name not in cleaned.columns:
            cleaned[column_name] = None

    cleaned = cleaned[selected_columns].copy()
    cleaned = cleaned[cleaned.geometry.notnull()].copy()

    if cleaned.crs is None:
        cleaned = cleaned.set_crs(OUTPUT_CRS)
    else:
        cleaned = cleaned.to_crs(OUTPUT_CRS)

    cleaned["names"] = cleaned["names"].map(_jsonify_value)
    for roof_col in ["roof_material", "roof_direction", "roof_orientation"]:
        cleaned[roof_col] = cleaned[roof_col].map(_jsonify_value)

    for numeric_col in ["height", "num_floors", "roof_height"]:
        cleaned[numeric_col] = pd.to_numeric(cleaned[numeric_col], errors="coerce")

    if "municipality_name" not in cleaned.columns:
        cleaned["municipality_name"] = None
    if "municipality_geoid" not in cleaned.columns:
        cleaned["municipality_geoid"] = None

    cleaned["municipality_name"] = cleaned["municipality_name"].astype("string")
    cleaned["municipality_geoid"] = cleaned["municipality_geoid"].astype("string")
    cleaned["loaded_at_utc"] = pd.Timestamp.utcnow().tz_localize(None)

    ordered_columns = [
        "id",
        "municipality_name",
        "municipality_geoid",
        "subtype",
        "class",
        "names",
        "height",
        "num_floors",
        "roof_material",
        "roof_direction",
        "roof_orientation",
        "roof_height",
        "loaded_at_utc",
        "geometry",
    ]
    cleaned = cleaned[ordered_columns]

    return cleaned


def write_buildings_to_duckdb(con: duckdb.DuckDBPyConnection, gdf: gpd.GeoDataFrame) -> None:
    """Persist cleaned Overture footprints with layout optimizations."""

    if not load_h3_extension(con):
        raise RuntimeError(
            "DuckDB h3 extension is required to persist building-level H3 columns. "
            "Install or enable community extensions, then rerun this notebook."
        )

    stage_df = pd.DataFrame(gdf.drop(columns=["geometry"]).copy())
    stage_df["geometry_wkb"] = gdf.geometry.to_wkb()

    con.register("staged_overture_buildings", stage_df)

    # Optimization rationale:
    # 1) We keep only analysis-critical columns to avoid nested-struct bloat.
    # 2) We materialize DuckDB native GEOMETRY from WKB for spatial SQL support.
    # 3) We sort by ST_Hilbert(geometry) to improve spatial locality and selective
    #    predicate performance for downstream bounding-box and intersection filters.
    # 4) We keep both an RTREE on geometry and a composite municipality/H3 index
    #    because downstream notebooks repeatedly derive occupied H3 cells and
    #    municipality-scoped previews from this base table.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {OVERTURE_TABLE} AS
        WITH typed AS (
            SELECT
                CAST(id AS VARCHAR) AS id,
                CAST(municipality_name AS VARCHAR) AS municipality_name,
                CAST(municipality_geoid AS VARCHAR) AS municipality_geoid,
                CAST(subtype AS VARCHAR) AS subtype,
                CAST(class AS VARCHAR) AS class,
                CAST(names AS VARCHAR) AS names,
                TRY_CAST(height AS DOUBLE) AS height,
                TRY_CAST(num_floors AS DOUBLE) AS num_floors,
                CAST(roof_material AS VARCHAR) AS roof_material,
                CAST(roof_direction AS VARCHAR) AS roof_direction,
                CAST(roof_orientation AS VARCHAR) AS roof_orientation,
                TRY_CAST(roof_height AS DOUBLE) AS roof_height,
                CAST(loaded_at_utc AS TIMESTAMP) AS loaded_at_utc,
                ST_GeomFromWKB(geometry_wkb) AS geometry
            FROM staged_overture_buildings
            WHERE geometry_wkb IS NOT NULL
        ),
        pointified AS (
            SELECT
                *,
                ST_PointOnSurface(geometry) AS centroid_geometry
            FROM typed
            WHERE ST_IsValid(geometry)
        )
        SELECT
            id,
            municipality_name,
            municipality_geoid,
            subtype,
            class,
            names,
            height,
            num_floors,
            roof_material,
            roof_direction,
            roof_orientation,
            roof_height,
            loaded_at_utc,
            CAST(ST_X(centroid_geometry) AS DOUBLE) AS building_centroid_lon,
            CAST(ST_Y(centroid_geometry) AS DOUBLE) AS building_centroid_lat,
            CAST({H3_RESOLUTION} AS INTEGER) AS h3_resolution,
            CAST(
                h3_latlng_to_cell(
                    ST_Y(centroid_geometry),
                    ST_X(centroid_geometry),
                    {H3_RESOLUTION}
                ) AS UBIGINT
            ) AS h3_cell_uint,
            CAST(
                h3_latlng_to_cell_string(
                    ST_Y(centroid_geometry),
                    ST_X(centroid_geometry),
                    {H3_RESOLUTION}
                ) AS VARCHAR
            ) AS h3_cell_id,
            geometry
        FROM pointified
        ORDER BY ST_Hilbert(geometry);
        """
    )
    con.execute(f"DROP INDEX IF EXISTS idx_{OVERTURE_TABLE}_geometry;")
    con.execute(f"DROP INDEX IF EXISTS idx_{OVERTURE_TABLE}_h3_cell_id;")
    con.execute(f"DROP INDEX IF EXISTS idx_{OVERTURE_TABLE}_municipality_name_h3_cell_id;")
    con.execute(f"CREATE INDEX idx_{OVERTURE_TABLE}_geometry ON {OVERTURE_TABLE} USING RTREE (geometry);")
    con.execute(f"CREATE INDEX idx_{OVERTURE_TABLE}_h3_cell_id ON {OVERTURE_TABLE} (h3_cell_id);")
    con.execute(
        f"CREATE INDEX idx_{OVERTURE_TABLE}_municipality_name_h3_cell_id "
        f"ON {OVERTURE_TABLE} (municipality_name, h3_cell_id);"
    )
    con.execute(f"ANALYZE {OVERTURE_TABLE};")
    con.unregister("staged_overture_buildings")


def compute_island_bbox_from_municipalities(municipalities_gdf: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    """Compute Puerto Rico island bounding box from municipality polygons."""

    xmin, ymin, xmax, ymax = municipalities_gdf.total_bounds
    return float(xmin), float(ymin), float(xmax), float(ymax)


def fetch_official_overture_buildings_table_for_bbox(
    bbox: tuple[float, float, float, float],
    release: str = OVERTURE_RELEASE,
) -> object:
    """Fetch Overture buildings with the official overturemaps Python client."""

    reader = overturemaps.record_batch_reader(
        OVERTURE_TYPE,
        bbox=bbox,
        release=release,
    )
    if reader is None:
        raise RuntimeError("overturemaps.record_batch_reader returned None for the requested bbox.")
    return reader.read_all()


def export_official_island_bbox_geoparquet_with_sql_assignment(
    con: duckdb.DuckDBPyConnection,
    municipalities_gdf: gpd.GeoDataFrame,
    output_path: Path,
    release: str = OVERTURE_RELEASE,
    compression: str = ISLAND_BBOX_EXPORT_COMPRESSION,
) -> dict[str, object]:
    """Fetch island-wide Overture buildings via official client and export assigned GeoParquet.

    This does not ingest into the project's main DuckDB tables by default; it only
    writes an external GeoParquet artifact.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    island_bbox = compute_island_bbox_from_municipalities(municipalities_gdf)

    total_started = time.perf_counter()
    fetch_started = time.perf_counter()
    official_table = fetch_official_overture_buildings_table_for_bbox(
        island_bbox,
        release=release,
    )
    fetch_elapsed_seconds = time.perf_counter() - fetch_started

    # Persist raw pull to a temporary parquet snapshot for reproducibility/debugging.
    OVERTURE_WORKING_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="official_overture_bbox_", dir=str(OVERTURE_WORKING_DIR)))
    raw_snapshot_path = temp_dir / "official_overture_buildings_raw.parquet"
    pq.write_table(official_table, raw_snapshot_path, compression=compression)
    raw_snapshot_size_bytes = raw_snapshot_path.stat().st_size

    assignment_started = time.perf_counter()
    con.register("official_overture_bbox_arrow", official_table)
    official_geometry_expr = resolve_geometry_sql_expression(con, relation_sql="official_overture_bbox_arrow")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE official_overture_bbox_buildings AS
        SELECT
            ROW_NUMBER() OVER () AS rid,
            CAST(id AS VARCHAR) AS id,
            CAST(subtype AS VARCHAR) AS subtype,
            CAST(class AS VARCHAR) AS class,
            CAST(names AS VARCHAR) AS names,
            TRY_CAST(height AS DOUBLE) AS height,
            TRY_CAST(num_floors AS DOUBLE) AS num_floors,
            CAST(roof_material AS VARCHAR) AS roof_material,
            TRY_CAST(roof_direction AS DOUBLE) AS roof_direction,
            CAST(roof_orientation AS VARCHAR) AS roof_orientation,
            TRY_CAST(roof_height AS DOUBLE) AS roof_height,
            {official_geometry_expr} AS geometry
        FROM official_overture_bbox_arrow
        WHERE geometry IS NOT NULL;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE official_overture_bbox_assigned AS
        WITH buildings AS (
            SELECT *
            FROM official_overture_bbox_buildings
            WHERE ST_IsValid(geometry)
        ),
        point_ranked AS (
            SELECT
                b.rid,
                m.GEOID AS municipality_geoid,
                m.NAME AS municipality_name,
                ROW_NUMBER() OVER (PARTITION BY b.rid ORDER BY m.NAME) AS rank_in_muni
            FROM buildings AS b
            LEFT JOIN pr_census_counties AS m
              ON ST_Within(ST_PointOnSurface(b.geometry), m.geometry)
        ),
        intersect_ranked AS (
            SELECT
                b.rid,
                m.GEOID AS municipality_geoid,
                m.NAME AS municipality_name,
                ROW_NUMBER() OVER (
                    PARTITION BY b.rid
                    ORDER BY ST_Area(ST_Intersection(m.geometry, b.geometry)) DESC NULLS LAST, m.NAME
                ) AS rank_in_muni
            FROM buildings AS b
            JOIN pr_census_counties AS m
              ON ST_Intersects(m.geometry, b.geometry)
        )
        SELECT
            b.id,
            COALESCE(p.municipality_name, i.municipality_name) AS municipality_name,
            COALESCE(p.municipality_geoid, i.municipality_geoid) AS municipality_geoid,
            b.subtype,
            b.class,
            b.names,
            b.height,
            b.num_floors,
            b.roof_material,
            b.roof_direction,
            b.roof_orientation,
            b.roof_height,
            b.geometry
        FROM buildings AS b
        LEFT JOIN point_ranked AS p
          ON b.rid = p.rid
         AND p.rank_in_muni = 1
        LEFT JOIN intersect_ranked AS i
          ON b.rid = i.rid
         AND i.rank_in_muni = 1;
        """
    )

    con.execute(
        f"""
        COPY (
            SELECT *
            FROM official_overture_bbox_assigned
            ORDER BY ST_Hilbert(geometry)
        )
        TO '{output_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION {compression.upper()});
        """
    )

    assignment_elapsed_seconds = time.perf_counter() - assignment_started
    total_elapsed_seconds = time.perf_counter() - total_started

    output_size_bytes = output_path.stat().st_size
    output_row_count = con.execute("SELECT COUNT(*) FROM official_overture_bbox_assigned;").fetchone()[0]
    assigned_row_count = con.execute(
        """
        SELECT COUNT(*)
        FROM official_overture_bbox_assigned
        WHERE municipality_name IS NOT NULL
        """
    ).fetchone()[0]
    unassigned_row_count = int(output_row_count) - int(assigned_row_count)

    municipality_preview = con.execute(
        """
        SELECT municipality_name, COUNT(*) AS building_rows
        FROM official_overture_bbox_assigned
        GROUP BY municipality_name
        ORDER BY building_rows DESC, municipality_name
        LIMIT 20;
        """
    ).fetchdf()

    con.unregister("official_overture_bbox_arrow")

    shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "release": release,
        "bbox": island_bbox,
        "output_path": str(output_path),
        "rows_total": int(output_row_count),
        "rows_assigned": int(assigned_row_count),
        "rows_unassigned": int(unassigned_row_count),
        "fetch_elapsed_seconds": round(fetch_elapsed_seconds, 2),
        "assignment_elapsed_seconds": round(assignment_elapsed_seconds, 2),
        "total_elapsed_seconds": round(total_elapsed_seconds, 2),
        "raw_snapshot_size_bytes": int(raw_snapshot_size_bytes),
        "output_size_bytes": int(output_size_bytes),
        "municipality_preview": municipality_preview,
    }


def load_overture_cache_geodataframe(cache_path: Path) -> gpd.GeoDataFrame:
    """Load a persisted island Overture GeoParquet cache as a GeoDataFrame."""

    gdf = gpd.read_parquet(cache_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(OUTPUT_CRS)
    else:
        gdf = gdf.to_crs(OUTPUT_CRS)
    return _ensure_overture_id_column(gdf)


def summarize_overture_cache(cache_path: Path, gdf: gpd.GeoDataFrame, *, release: str) -> dict[str, object]:
    """Build summary metadata for a persisted island Overture cache."""

    municipio_counts = (
        gdf["municipality_name"]
        .fillna("UNASSIGNED")
        .value_counts()
        .rename_axis("municipality_name")
        .reset_index(name="building_rows")
    )
    assigned = int(gdf["municipality_name"].notna().sum()) if "municipality_name" in gdf.columns else 0
    output_size_bytes = cache_path.stat().st_size if cache_path.exists() else 0
    return {
        "release": release,
        "output_path": str(cache_path),
        "rows_total": int(len(gdf)),
        "rows_assigned": assigned,
        "rows_unassigned": int(len(gdf)) - assigned,
        "fetch_elapsed_seconds": 0.0,
        "assignment_elapsed_seconds": 0.0,
        "total_elapsed_seconds": 0.0,
        "raw_snapshot_size_bytes": 0,
        "output_size_bytes": int(output_size_bytes),
        "municipality_preview": municipio_counts.head(20),
        "used_cache": True,
    }


def build_or_load_official_island_bbox_geoparquet(
    con: duckdb.DuckDBPyConnection,
    municipalities_gdf: gpd.GeoDataFrame,
    output_path: Path,
    *,
    refresh: bool = OVERTURE_REFRESH_GEOPARQUET_CACHE,
    release: str = OVERTURE_RELEASE,
    compression: str = ISLAND_BBOX_EXPORT_COMPRESSION,
) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    """Reuse the island Overture cache when available, otherwise rebuild it."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not refresh:
        cached_gdf = load_overture_cache_geodataframe(output_path)
        return cached_gdf, summarize_overture_cache(output_path, cached_gdf, release=release)

    stats = export_official_island_bbox_geoparquet_with_sql_assignment(
        con,
        municipalities_gdf=municipalities_gdf,
        output_path=output_path,
        release=release,
        compression=compression,
    )
    stats["used_cache"] = False
    refreshed_gdf = load_overture_cache_geodataframe(output_path)
    return refreshed_gdf, stats


def load_duckdb_wkb_query_to_gdf(
    con: duckdb.DuckDBPyConnection,
    query: str,
    params: list[object] | None = None,
) -> gpd.GeoDataFrame:
    """Run SQL returning geometry_wkb and rebuild a GeoDataFrame."""

    frame = con.execute(query, params or []).fetchdf()
    if frame.empty:
        return gpd.GeoDataFrame(frame, geometry=[], crs=OUTPUT_CRS)

    geometry = gpd.GeoSeries(frame["geometry_wkb"].map(lambda value: from_wkb(_to_bytes(value))), crs=OUTPUT_CRS)
    return gpd.GeoDataFrame(frame.drop(columns=["geometry_wkb"]), geometry=geometry, crs=OUTPUT_CRS)


def fetch_preview_subset(
    con: duckdb.DuckDBPyConnection,
    municipality_name: str,
    neighborhood_query: str | None = None,
    limit: int = MAP_PREVIEW_LIMIT,
) -> gpd.GeoDataFrame:
    """Fetch map preview subset with optional neighborhood clipping."""

    neighborhood_geom_wkb: bytes | None = None
    if neighborhood_query:
        try:
            neighborhood_gdf = ox.geocode_to_gdf(neighborhood_query)
            if not neighborhood_gdf.empty:
                if neighborhood_gdf.crs is None:
                    neighborhood_gdf = neighborhood_gdf.set_crs(OUTPUT_CRS)
                else:
                    neighborhood_gdf = neighborhood_gdf.to_crs(OUTPUT_CRS)
                neighborhood_geom = neighborhood_gdf.geometry.union_all()
                neighborhood_geom_wkb = neighborhood_geom.wkb
                print(f"Resolved neighborhood boundary for preview: {neighborhood_query}")
        except Exception as exc:
            print(f"Neighborhood geocoding fallback triggered ({exc}). Using municipality-wide sample.")

    if neighborhood_geom_wkb is not None:
        clipped_query = f"""
        SELECT
            id,
            municipality_name,
            subtype,
            class,
            height,
            num_floors,
            roof_material,
            ST_AsWKB(geometry) AS geometry_wkb
        FROM {OVERTURE_TABLE}
        WHERE municipality_name = ?
          AND ST_Intersects(geometry, ST_GeomFromWKB(?))
        LIMIT ?;
        """
        clipped_gdf = load_duckdb_wkb_query_to_gdf(
            con,
            clipped_query,
            [municipality_name, neighborhood_geom_wkb, limit],
        )
        if not clipped_gdf.empty:
            print(f"Preview subset rows after neighborhood clip: {len(clipped_gdf):,}")
            return clipped_gdf

    municipal_query = f"""
    SELECT
        id,
        municipality_name,
        subtype,
        class,
        height,
        num_floors,
        roof_material,
        ST_AsWKB(geometry) AS geometry_wkb
    FROM {OVERTURE_TABLE}
    WHERE municipality_name = ?
    ORDER BY coalesce(height, 0) DESC NULLS LAST
    LIMIT ?;
    """
    municipal_gdf = load_duckdb_wkb_query_to_gdf(con, municipal_query, [municipality_name, limit])
    print(f"Preview subset rows from municipality-wide fallback: {len(municipal_gdf):,}")
    return municipal_gdf


def fetch_loaded_municipality_names(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return distinct municipality names currently loaded in the Overture table."""

    names_df = con.execute(
        f"""
        SELECT DISTINCT municipality_name
        FROM {OVERTURE_TABLE}
        WHERE municipality_name IS NOT NULL
        ORDER BY municipality_name;
        """
    ).fetchdf()
    if names_df.empty:
        return []
    return names_df["municipality_name"].astype(str).tolist()


def prepare_preview_heights(preview_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Create a robust elevation field for 3D lonboard rendering and diagnostics."""

    prepared = preview_gdf.copy()
    prepared["height"] = pd.to_numeric(prepared["height"], errors="coerce")
    prepared["num_floors"] = pd.to_numeric(prepared["num_floors"], errors="coerce")

    inferred_height = prepared["num_floors"] * HEIGHT_FROM_FLOOR_METERS
    prepared["preview_height_m"] = prepared["height"].where(prepared["height"].notna(), inferred_height)
    prepared["preview_height_m"] = prepared["preview_height_m"].fillna(MIN_PREVIEW_ELEVATION_METERS)
    prepared["preview_height_m"] = prepared["preview_height_m"].clip(lower=MIN_PREVIEW_ELEVATION_METERS)

    explicit_height_rows = int(prepared["height"].notna().sum())
    inferred_from_floors_rows = int(prepared["height"].isna().mul(prepared["num_floors"].notna()).sum())
    fallback_rows = int(prepared["preview_height_m"].eq(MIN_PREVIEW_ELEVATION_METERS).sum())

    stats = {
        "rows_total": int(len(prepared)),
        "rows_with_explicit_height": explicit_height_rows,
        "rows_inferred_from_num_floors": inferred_from_floors_rows,
        "rows_min_fallback": fallback_rows,
    }
    return prepared, stats


def build_log_scaled_height_colors(heights_meters: np.ndarray) -> np.ndarray:
    """Create RGBA colors from heights using log-scaled normalization."""

    if heights_meters.size == 0:
        return np.empty((0, 4), dtype=np.uint8)

    safe_heights = np.nan_to_num(heights_meters.astype(np.float64, copy=False), nan=1.0)
    safe_heights = np.where(safe_heights > 0, safe_heights, 1.0)

    max_height = float(np.nanmax(safe_heights)) if safe_heights.size else 1.0
    if not np.isfinite(max_height) or max_height <= 1.0:
        normalized = np.ones_like(safe_heights, dtype=np.float64)
    else:
        normalized = LogNorm(vmin=1.0, vmax=max_height, clip=True)(safe_heights)

    # Match lonboard tutorial intent: emphasize low-height variation with a log scale.
    rgba_float = cm.get_cmap("Oranges")(normalized)
    rgba_uint8 = np.clip(np.round(rgba_float * 255), 0, 255).astype(np.uint8)
    return rgba_uint8


def _render_lonboard_preview_for_municipality(
    con: duckdb.DuckDBPyConnection,
    municipality_name: str,
    neighborhood_query: str | None = None,
    limit: int = MAP_PREVIEW_LIMIT,
) -> None:
    """Render one municipality preview map for use in static and interactive modes."""

    preview_gdf = fetch_preview_subset(
        con,
        municipality_name=municipality_name,
        neighborhood_query=neighborhood_query,
        limit=limit,
    )

    if preview_gdf.empty:
        print(f"No building features available for municipality preview: {municipality_name}")
        return

    preview_gdf, height_stats = prepare_preview_heights(preview_gdf)
    heights_m = preview_gdf["preview_height_m"].to_numpy(dtype=np.float32, na_value=MIN_PREVIEW_ELEVATION_METERS)
    heights_m = np.nan_to_num(heights_m, nan=MIN_PREVIEW_ELEVATION_METERS)
    heights_m = np.where(heights_m > 0, heights_m, MIN_PREVIEW_ELEVATION_METERS)
    fill_colors = build_log_scaled_height_colors(heights_m)

    layer = PolygonLayer.from_geopandas(
        preview_gdf,
        get_fill_color=fill_colors,
        get_line_color=[164, 45, 24, 220],
        get_elevation=heights_m,
        extruded=ENABLE_3D_PREVIEW,
        line_width_min_pixels=1,
        pickable=True,
    )
    display(Map(layers=[layer]))
    print(f"lonboard preview rendered for {municipality_name} with {len(preview_gdf):,} features.")
    print(
        "Height coverage in preview subset: "
        f"explicit={height_stats['rows_with_explicit_height']:,}, "
        f"inferred_from_floors={height_stats['rows_inferred_from_num_floors']:,}, "
        f"fallback_min_height={height_stats['rows_min_fallback']:,}, "
        f"max_height_m={float(np.max(heights_m)) if heights_m.size else 0.0:.1f}."
    )


def render_interactive_municipality_preview(
    con: duckdb.DuckDBPyConnection,
    default_municipality: str | None = None,
    neighborhood_query: str | None = TARGET_NEIGHBORHOOD_QUERY,
    neighborhood_municipality: str | None = TARGET_MUNICIPALITY,
    limit: int = MAP_PREVIEW_LIMIT,
) -> None:
    """Render a Dropdown widget to switch municipality previews in lonboard."""

    municipality_names = fetch_loaded_municipality_names(con)
    if not municipality_names:
        print("No municipalities were found in the loaded Overture table.")
        return

    chosen_default = default_municipality if default_municipality in municipality_names else municipality_names[0]

    dropdown = widgets.Dropdown(
        options=municipality_names,
        value=chosen_default,
        description="Municipality:",
    )
    output = widgets.Output()

    def _refresh(municipality_name: str) -> None:
        active_neighborhood_query: str | None = None
        if neighborhood_query and (
            neighborhood_municipality is None or municipality_name.casefold() == neighborhood_municipality.casefold()
        ):
            active_neighborhood_query = neighborhood_query

        with output:
            output.clear_output(wait=True)
            _render_lonboard_preview_for_municipality(
                con,
                municipality_name=municipality_name,
                neighborhood_query=active_neighborhood_query,
                limit=limit,
            )

    def _on_dropdown_change(change: dict[str, object]) -> None:
        if change.get("name") == "value":
            _refresh(str(change.get("new")))

    dropdown.observe(_on_dropdown_change, names="value")
    _refresh(chosen_default)
    display(widgets.VBox([dropdown, output]))

# %% [markdown]
# ## Step 1 - Connect to DuckDB and rank municipality candidates by PV labels

# %%
if __name__ == "__main__":
    db_path = resolve_db_path()
    cleaned_paths = cleanup_legacy_overture_artifacts()

    print(f"Resolved project root: {PROJECT_ROOT}")
    print(f"Using DuckDB path: {db_path}")
    print(f"Using Overture backend: {OVERTURE_FETCH_BACKEND}")
    print(f"Using transient Overture temp directory: {OVERTURE_WORKING_DIR}")
    if cleaned_paths:
        print(f"Removed {len(cleaned_paths):,} legacy Overture cache artifacts before starting this run.")

    con = create_spatial_connection(db_path)
    print("Connected to DuckDB with spatial extension loaded.")

    municipalities_gdf = fetch_municipality_boundaries(con)
    pv_counts_df, pv_source = fetch_municipality_pv_counts(con)

    ranked_municipalities = rank_municipalities_by_pv_labels(municipalities_gdf, pv_counts_df)
    top_k_municipalities = select_top_k_municipalities(
        ranked_municipalities,
        top_k=TOP_K_MUNICIPALITIES,
        force_include=FORCE_INCLUDE_MUNICIPALITIES,
    )

    print(f"PV label source used for ranking: {pv_source}")
    print("Top municipality candidates by PV labels:")
    print(ranked_municipalities[["municipality_geoid", "municipality_name", "pv_labels"]].head(12).to_string(index=False))
    # print(f"\nTop-{len(top_k_municipalities)} municipalities by PV evidence (context for QA + previews):")
    # print(top_k_municipalities[["municipality_geoid", "municipality_name", "pv_labels"]].to_string(index=False))
    # print(
    #     "\nScope note: the default ingest path is now whole-island. The top-K ranking remains useful "
    #     "for validation focus, preview selection, and later label-driven QA."
    # )

    target_row = choose_target_municipality(municipalities_gdf, pv_counts_df, TARGET_MUNICIPALITY)
    target_geometry = target_row.geometry

    print("\nSelected default municipality for preview focus:")
    print(
        pd.DataFrame(
            [
                {
                    "municipality_geoid": target_row["municipality_geoid"],
                    "municipality_name": target_row["municipality_name"],
                    "pv_labels": int(target_row.get("pv_labels", 0) or 0),
                    "geometry_type": target_geometry.geom_type,
                }
            ]
        ).to_string(index=False)
    )

# %% [markdown]
# ## Step 2 - Build or reuse the whole-island Overture GeoParquet cache

# %%
if __name__ == "__main__":
    island_export_path = ISLAND_BBOX_EXPORT_DIR / ISLAND_BBOX_EXPORT_FILENAME
    island_cache_stats: dict[str, object] | None = None

    if OVERTURE_FETCH_BACKEND == "official_island_bbox":
        print(f"Building or reusing the whole-island Overture GeoParquet cache at {island_export_path}.")
        overture_raw_gdf, island_cache_stats = build_or_load_official_island_bbox_geoparquet(
            con,
            municipalities_gdf=municipalities_gdf,
            output_path=island_export_path,
            refresh=OVERTURE_REFRESH_GEOPARQUET_CACHE,
            release=OVERTURE_RELEASE,
            compression=ISLAND_BBOX_EXPORT_COMPRESSION,
        )
        overture_parquet_paths = [island_export_path]
        print(
            f"Whole-island cache {'reused' if island_cache_stats['used_cache'] else 'rebuilt'}; "
            f"rows={int(island_cache_stats['rows_total']):,}."
        )
    elif OVERTURE_FETCH_BACKEND == "maestro_geometry":
        print(
            f"Fetching Overture buildings with legacy strategy='{OVERTURE_FETCH_STRATEGY}' "
            f"over top-{len(top_k_municipalities)} municipality union geometry."
        )
        OVERTURE_WORKING_DIR.mkdir(parents=True, exist_ok=True)
        ingest_temp_dir = Path(tempfile.mkdtemp(prefix="overture_ingest_", dir=str(OVERTURE_WORKING_DIR)))
        top_k_union_geometry = build_union_geometry(top_k_municipalities)
        overture_raw_gdf, overture_parquet_paths = fetch_overture_buildings(
            top_k_union_geometry,
            ingest_temp_dir,
            strategy=OVERTURE_FETCH_STRATEGY,
            columns_to_download=OVERTURE_COLUMNS_TO_DOWNLOAD,
            max_workers=OVERTURE_DOWNLOAD_MAX_WORKERS,
        )
    else:
        raise ValueError(
            "Unsupported OVERTURE_FETCH_BACKEND. Expected 'official_island_bbox' or 'maestro_geometry'."
        )

    print(f"Fetched Overture rows: {len(overture_raw_gdf):,}")
    print("GeoParquet artifacts:")
    for parquet_path in overture_parquet_paths:
        print(f" - {parquet_path}")

    id_in_index = str(getattr(overture_raw_gdf.index, "name", "")).casefold() == "id"
    id_null_count = int(overture_raw_gdf["id"].isna().sum()) if "id" in overture_raw_gdf.columns else -1
    print(
        "Raw ID diagnostics: "
        f"id_column_present={'id' in overture_raw_gdf.columns}, "
        f"id_in_index={id_in_index}, "
        f"id_null_count={id_null_count if id_null_count >= 0 else 'N/A'}"
    )
    print(f"Raw Overture columns ({len(overture_raw_gdf.columns)}): {list(overture_raw_gdf.columns)}")
    print("Raw Overture preview:")
    print(overture_raw_gdf.head(5).drop(columns=["geometry"], errors="ignore").to_string(index=False))

# %% [markdown]
# ## Step 3 - Prune/flatten schema for performant local storage

# %%
if __name__ == "__main__":
    overture_clean_gdf = clean_buildings_schema(
        overture_raw_gdf,
        municipalities_gdf=municipalities_gdf,
    )

    print(f"Cleaned Overture rows kept: {len(overture_clean_gdf):,}")
    print(f"Cleaned ID null count: {int(overture_clean_gdf['id'].isna().sum())}")
    print(f"Cleaned schema columns ({len(overture_clean_gdf.columns)}): {list(overture_clean_gdf.columns)}")
    print("Rows by municipality after spatial assignment:")
    print(overture_clean_gdf["municipality_name"].fillna("UNASSIGNED").value_counts().head(12).to_string())
    print("Cleaned Overture preview:")
    print(overture_clean_gdf.head(5).drop(columns=["geometry"], errors="ignore").to_string(index=False))

# %% [markdown]
# ## Step 4 - Load cleaned buildings into DuckDB with spatial optimizations

# %%
if __name__ == "__main__":
    write_buildings_to_duckdb(con, overture_clean_gdf)
    con.execute("DROP TABLE IF EXISTS pr_overture_building_h3;")
    con.execute("DROP TABLE IF EXISTS pr_overture_h3_cells;")

    row_count = con.execute(f"SELECT COUNT(*) AS n_rows FROM {OVERTURE_TABLE};").fetchone()[0]
    print(f"Loaded rows in {OVERTURE_TABLE}: {row_count:,}")

    occupied_h3_sql = occupied_h3_cells_sql(OVERTURE_TABLE)
    h3_building_count = con.execute(
        f"SELECT COUNT(*) AS n_rows FROM {OVERTURE_TABLE} WHERE h3_cell_id IS NOT NULL;"
    ).fetchone()[0]
    h3_cell_count = con.execute(
        f"WITH occupied_h3 AS ({occupied_h3_sql}) SELECT COUNT(*) AS n_rows FROM occupied_h3;"
    ).fetchone()[0]
    print(
        f"Persisted {h3_building_count:,} building rows with embedded H3 assignments and {h3_cell_count:,} "
        f"derived occupied H3 cells at resolution {H3_RESOLUTION}."
    )

    print("Table schema preview:")
    print(con.execute(f"PRAGMA table_info('{OVERTURE_TABLE}')").fetchdf().to_string(index=False))

    print("Sample rows from persisted DuckDB table:")
    print(
        con.execute(
            f"""
            SELECT
                id,
                municipality_name,
                subtype,
                class,
                height,
                num_floors,
                roof_material,
                h3_resolution,
                h3_cell_id
            FROM {OVERTURE_TABLE}
            LIMIT 8;
            """
        ).fetchdf().to_string(index=False)
    )

    print("Rows by municipality in persisted DuckDB table:")
    print(
        con.execute(
            f"""
            SELECT municipality_name, COUNT(*) AS building_rows
            FROM {OVERTURE_TABLE}
            GROUP BY municipality_name
            ORDER BY building_rows DESC, municipality_name
            LIMIT 20;
            """
        ).fetchdf().to_string(index=False)
    )

    print("Occupied H3 cells by municipality:")
    print(
        con.execute(
            f"""
            WITH occupied_h3 AS ({occupied_h3_sql})
            SELECT
                municipality_name,
                COUNT(*) AS occupied_cells,
                SUM(building_count) AS total_buildings,
                SUM(CASE WHEN crosses_municipality_boundary THEN 1 ELSE 0 END) AS boundary_crossing_cells
            FROM occupied_h3
            GROUP BY municipality_name
            ORDER BY occupied_cells DESC, municipality_name
            LIMIT 20;
            """
        ).fetchdf().to_string(index=False)
    )

# %% [markdown]
# ## Step 5 - Whole-island Overture GeoParquet cache summary

# %%
if __name__ == "__main__":
    if OVERTURE_FETCH_BACKEND == "official_island_bbox" and island_cache_stats is not None:
        print("Whole-island Overture GeoParquet cache ready for reruns.")
        print(f"Release: {island_cache_stats['release']}")
        print(f"Output GeoParquet: {island_cache_stats['output_path']}")
        print(
            f"Rows: total={int(island_cache_stats['rows_total']):,}, assigned={int(island_cache_stats['rows_assigned']):,}, "
            f"unassigned={int(island_cache_stats['rows_unassigned']):,}"
        )
        print(
            f"Timing (s): fetch={island_cache_stats['fetch_elapsed_seconds']}, "
            f"assignment+export={island_cache_stats['assignment_elapsed_seconds']}, "
            f"total={island_cache_stats['total_elapsed_seconds']}"
        )
        print(
            "Disk usage: "
            f"raw_snapshot={format_size_bytes(int(island_cache_stats['raw_snapshot_size_bytes']))}, "
            f"compressed_output={format_size_bytes(int(island_cache_stats['output_size_bytes']))}"
        )
        print("Top municipality counts in island export artifact:")
        print(island_cache_stats["municipality_preview"].to_string(index=False))
    else:
        print(
            "Whole-island cache summary is only available when OVERTURE_FETCH_BACKEND='official_island_bbox'."
        )

# %% [markdown]
# ## Step 6 - Interactive lonboard preview by municipality (Dropdown + optional neighborhood clip)

# %%
if __name__ == "__main__":
    render_interactive_municipality_preview(
        con,
        default_municipality=str(target_row["municipality_name"]),
        neighborhood_query=TARGET_NEIGHBORHOOD_QUERY,
        neighborhood_municipality=TARGET_MUNICIPALITY,
        limit=150000,
    )

# %% [markdown]
# ## Step 7 - Close connection

# %%
if __name__ == "__main__":
    if OVERTURE_FETCH_BACKEND == "maestro_geometry" and ingest_temp_dir.exists():
        shutil.rmtree(ingest_temp_dir, ignore_errors=True)
        print(f"Temporary Overture ingest directory removed: {ingest_temp_dir}")
    con.close()
    print("DuckDB connection closed.")


