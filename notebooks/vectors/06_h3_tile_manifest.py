# %% [markdown]
# # Occupied-H3 Tile Manifest for the Google Solar API Pipeline
#
# Builds `pr_solar_tile_manifest` — the prioritized list of Google Solar Data
# Layers fetch targets for the two case-study municipalities.
#
# Important scope note:
# - Despite the historical file name, this notebook no longer lays a new grid
#   over census block groups.
# - Each output row corresponds to one occupied H3 cell derived on demand from
#   `pr_overture_buildings`, with the cell center used as the Solar API query
#   point and the H3 polygon retained as the manifest geometry.
# - The manifest is therefore a fetch-planning table, not the final ML tiling
#   dataset.
#
# Workflow:
# 1. Load occupied H3 cells from the Overture-building ingest for San Juan and
#    Isabela.
# 2. Carry forward municipality attribution, dominant-municipality counts, and
#    the `crosses_municipality_boundary` diagnostic from the H3 source table.
# 3. Count OSM rooftop PV labels per occupied H3 cell and boost priorities for
#    seeded neighborhoods (Puerto Nuevo and Barrio Mora).
# 4. Seed expected imagery quality for downstream Google Solar runs.
# 5. Persist the fetch manifest to DuckDB as `pr_solar_tile_manifest`.
#
# Interpretation note:
# - `crosses_municipality_boundary=True` means an occupied H3 cell contains
#   buildings attributed to more than one municipality in the island-wide
#   Overture H3 summary.
# - It is not a generic "near the edge of San Juan or Isabela" flag, so sparse
#   `True` values are expected in the case-study subset.

# %%
"""04_bg_tile_manifest.py

Jupytext-friendly builder for the Solar-API tile manifest covering San Juan
and Isabela Block Groups.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from shapely import from_wkb
from shapely.geometry import shape


def resolve_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if any((candidate / m).exists() for m in ("project_rules.md", ".git")):
            return candidate
    return current


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from utils.solar_tiling import (  # noqa: E402 - project imports after path setup
    attach_h3_priority,
    build_h3_tile_manifest,
)
from utils.overture import occupied_h3_cells_sql

OUTPUT_CRS = "EPSG:4326"
TARGET_MUNICIPALITIES = ["San Juan", "Isabela"]
MANIFEST_TABLE = "pr_solar_tile_manifest"
OVERTURE_BUILDINGS_TABLE = "pr_overture_buildings"
# Seed neighborhoods to score as priority-3. Pulled from OSM via osmnx on demand.
SEED_NEIGHBORHOODS = [
    {"name": "Puerto Nuevo", "municipio": "San Juan", "osm_query": "Puerto Nuevo, San Juan, Puerto Rico"},
    {"name": "Barrio Mora", "municipio": "Isabela", "osm_query": "Mora, Isabela, Puerto Rico"},
]

# Default imagery-quality guesses before the BuildingInsights probe runs.
# Values here are overwritten by 06a_solar_quality_probe; this only seeds
# sensible defaults for dry runs and sanity checks.
DEFAULT_EXPECTED_QUALITY = {
    "San Juan": "HIGH",
    "Isabela": "MEDIUM",
}


def resolve_db_path() -> Path:
    db_path_value = os.getenv("VECTOR_DB")
    if db_path_value:
        p = Path(db_path_value)
        if not p.is_absolute():
            p = PROJECT_ROOT / p if len(p.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / p
        return p
    return PROJECT_ROOT / "data" / "PR_PV_plan_data.duckdb"


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("LOAD h3;")
    return con


def _to_bytes(value: object) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    return bytes(value) if not isinstance(value, bytes) else value


# %%
def fetch_occupied_h3_cells(con: duckdb.DuckDBPyConnection, municipalities: list[str]) -> gpd.GeoDataFrame:
    """Derive occupied H3 cells for ``municipalities`` from the base Overture table."""

    names_sql = ", ".join("?" * len(municipalities))
    table_count = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?;",
        [OVERTURE_BUILDINGS_TABLE],
    ).fetchone()[0]
    if not table_count:
        raise RuntimeError(
            f"{OVERTURE_BUILDINGS_TABLE} not found; run notebooks/vectors/03_overture_buildings_ingest.py first."
        )

    occupied_h3_sql = occupied_h3_cells_sql(OVERTURE_BUILDINGS_TABLE)
    df = con.execute(
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
            cell_center_lon,
            cell_center_lat,
            ST_AsWKB(geometry) AS geometry_wkb
        FROM occupied_h3
        WHERE municipality_name IN ({names_sql})
        ORDER BY municipio, building_count DESC, h3_cell_id;
        """,
        list(municipalities),
    ).fetchdf()

    geometry = gpd.GeoSeries(df["geometry_wkb"].map(lambda v: from_wkb(_to_bytes(v))), crs=OUTPUT_CRS)
    return gpd.GeoDataFrame(df.drop(columns=["geometry_wkb"]), geometry=geometry, crs=OUTPUT_CRS)


def fetch_osm_pv(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    if not con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'pr_osm_rooftop_pv_polygons'"
    ).fetchone()[0]:
        return gpd.GeoDataFrame(geometry=[], crs=OUTPUT_CRS)
    df = con.execute(
        "SELECT ST_AsWKB(geometry) AS geometry_wkb FROM pr_osm_rooftop_pv_polygons WHERE geometry IS NOT NULL;"
    ).fetchdf()
    geometry = gpd.GeoSeries(df["geometry_wkb"].map(lambda v: from_wkb(_to_bytes(v))), crs=OUTPUT_CRS)
    return gpd.GeoDataFrame(geometry=geometry, crs=OUTPUT_CRS)


def fetch_seed_neighborhoods() -> gpd.GeoDataFrame:
    """Geocode seed neighborhood polygons via osmnx (falls back to empty on failure)."""

    try:
        import osmnx as ox  # imported lazily to keep the module importable without OSMnx
    except ImportError:
        return gpd.GeoDataFrame(geometry=[], crs=OUTPUT_CRS)

    frames: list[gpd.GeoDataFrame] = []
    for seed in SEED_NEIGHBORHOODS:
        try:
            gdf = ox.geocode_to_gdf(seed["osm_query"])
            gdf["seed_name"] = seed["name"]
            gdf["municipio"] = seed["municipio"]
            frames.append(gdf[["seed_name", "municipio", "geometry"]])
        except Exception as exc:
            print(f"[warn] failed to geocode {seed['osm_query']}: {exc}")
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=OUTPUT_CRS)
    result = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=OUTPUT_CRS)
    return result


# %%
def write_manifest_to_duckdb(con: duckdb.DuckDBPyConnection, manifest: gpd.GeoDataFrame, table: str) -> None:
    staged = pd.DataFrame(manifest.drop(columns=["geometry"]).copy())
    staged["geometry_wkb"] = manifest.geometry.to_wkb()
    con.register("staged_manifest", staged)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT
            CAST(tile_id AS VARCHAR) AS tile_id,
            CAST(bg_geoid AS VARCHAR) AS bg_geoid,
            CAST(h3_cell_id AS VARCHAR) AS h3_cell_id,
            CAST(h3_resolution AS INTEGER) AS h3_resolution,
            CAST(municipio AS VARCHAR) AS municipio,
            CAST(municipio_geoid AS VARCHAR) AS municipio_geoid,
            CAST(lon AS DOUBLE) AS lon,
            CAST(lat AS DOUBLE) AS lat,
            CAST(radius_m AS INTEGER) AS radius_m,
            CAST(building_count AS INTEGER) AS building_count,
            CAST(municipality_building_count AS INTEGER) AS municipality_building_count,
            CAST(osm_pv_count AS INTEGER) AS osm_pv_count,
            CAST(crosses_municipality_boundary AS BOOLEAN) AS crosses_municipality_boundary,
            CAST(priority_score AS INTEGER) AS priority_score,
            CAST(expected_quality AS VARCHAR) AS expected_quality,
            CAST(required_quality AS VARCHAR) AS required_quality,
            CAST(status AS VARCHAR) AS status,
            ST_GeomFromWKB(geometry_wkb) AS geometry
        FROM staged_manifest;
        """
    )
    con.execute(f"DROP INDEX IF EXISTS idx_{table}_geometry;")
    con.execute(f"CREATE INDEX idx_{table}_geometry ON {table} USING RTREE (geometry);")
    con.unregister("staged_manifest")


# %% [markdown]
# ## Step 1 — Load occupied H3 cells and label inputs

# %%
if __name__ == "__main__":
    db_path = resolve_db_path()
    print(f"DuckDB: {db_path}")
    con = connect(db_path)

    h3_cells = fetch_occupied_h3_cells(con, TARGET_MUNICIPALITIES)
    osm_pv = fetch_osm_pv(con)
    print(f"occupied H3 cells: {len(h3_cells):,} | osm PV: {len(osm_pv):,}")

    seeds = fetch_seed_neighborhoods()
    print(f"seed neighborhoods: {len(seeds):,}")

# %% [markdown]
# ## Step 2 — Build one manifest row per occupied H3 cell

# %%
if __name__ == "__main__":
    manifest = build_h3_tile_manifest(
        h3_cells,
        municipio_col="municipio",
        municipio_geoid_col="municipio_geoid",
    )
    print(f"occupied H3 manifest rows: {len(manifest):,}")
    print(
        "municipality attribution preview:\n",
        manifest[["municipio", "municipio_geoid"]].value_counts(dropna=False).head(8).to_string(),
    )

# %% [markdown]
# ## Step 3 — Attach priority and explain diagnostic fields

# %%
if __name__ == "__main__":
    manifest = attach_h3_priority(
        manifest,
        osm_pv_polygons=osm_pv,
        seed_neighborhoods=seeds,
    )

    # Occupied H3 cells should already be non-empty, but keep the guard in case
    # future upstream filters materialize placeholder rows.
    manifest = manifest[manifest["building_count"] > 0].copy()

    manifest["expected_quality"] = manifest["municipio"].map(DEFAULT_EXPECTED_QUALITY).fillna("MEDIUM")
    manifest["required_quality"] = manifest["expected_quality"]
    manifest["status"] = "pending"

    manifest = manifest.sort_values(
        ["priority_score", "osm_pv_count", "building_count", "tile_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    print("priority distribution:\n", manifest["priority_score"].value_counts().sort_index(ascending=False).to_string())
    print("tiles per municipality:\n", manifest["municipio"].value_counts().to_string())
    print(
        "boundary diagnostic (dominant municipality differs from full H3 occupancy):\n",
        manifest["crosses_municipality_boundary"].value_counts(dropna=False).to_string(),
    )
    print(f"final tile count (post-drop of empty tiles): {len(manifest):,}")

# %% [markdown]
# ## Step 4 — Persist the fetch manifest to DuckDB

# %%
if __name__ == "__main__":
    write_manifest_to_duckdb(con, manifest, MANIFEST_TABLE)
    preview = con.execute(
        f"SELECT priority_score, municipio, expected_quality, COUNT(*) AS tiles, SUM(building_count) AS buildings "
        f"FROM {MANIFEST_TABLE} GROUP BY 1,2,3 ORDER BY 1 DESC, 2, 3"
    ).fetchdf()
    print(preview.to_string(index=False))
    con.close()
