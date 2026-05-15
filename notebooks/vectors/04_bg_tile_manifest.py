# %% [markdown]
# # Block-Group Tile Manifest for the Google Solar API Pipeline
#
# Builds `pr_solar_tile_manifest` — the prioritized list of ≤175 m-radius
# query points that the Data Layers ingest notebook will consume.
#
# Workflow:
# 1. Load Census Block Groups (from `01_census_geometries_ingest`) and clip to
#    San Juan + Isabela.
# 2. Load Overture building centroids and count buildings per BG.
# 3. Load OSM rooftop PV polygons for BG-level priority boosts.
# 4. Lay a ~247 m grid inside each BG in UTM 19N, keep every point whose
#    175 m disk intersects the BG polygon.
# 5. Attach priority (3 = Puerto Nuevo / Barrio Mora seed; 2 = BG has OSM PV;
#    1 = coverage sweep) and an expected imagery-quality default.
# 6. Persist to DuckDB as `pr_solar_tile_manifest`.
#
# Budget-cap note: the manifest may contain more tiles than the 1,999 Data
# Layers cap. The ingest notebook applies the cap in priority order.

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
    DEFAULT_RADIUS_M,
    DEFAULT_SPACING_M,
    attach_priority,
    build_tile_manifest,
)

OUTPUT_CRS = "EPSG:4326"
TARGET_MUNICIPALITIES = ["San Juan", "Isabela"]
MANIFEST_TABLE = "pr_solar_tile_manifest"
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
    return PROJECT_ROOT / "data" / "vectors" / "PR_vector_data.duckdb"


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")
    return con


def _to_bytes(value: object) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    return bytes(value) if not isinstance(value, bytes) else value


# %%
def fetch_block_groups(con: duckdb.DuckDBPyConnection, municipalities: list[str]) -> gpd.GeoDataFrame:
    """Load BGs for ``municipalities`` by spatial-joining against pr_municipalities."""

    names_sql = ", ".join("?" * len(municipalities))
    df = con.execute(
        f"""
        WITH target_munis AS (
            SELECT GEOID AS municipio_geoid, NAME AS municipio, geometry
            FROM pr_municipalities
            WHERE NAME IN ({names_sql})
        )
        SELECT
            bg.GEOID AS bg_geoid,
            m.municipio,
            m.municipio_geoid,
            ST_AsWKB(bg.geometry) AS geometry_wkb
        FROM pr_block_groups AS bg
        JOIN target_munis AS m
          ON ST_Intersects(m.geometry, bg.geometry)
        WHERE ST_Area(ST_Intersection(m.geometry, bg.geometry))
            / NULLIF(ST_Area(bg.geometry), 0) > 0.5
        ORDER BY m.municipio, bg.GEOID;
        """,
        list(municipalities),
    ).fetchdf()

    geometry = gpd.GeoSeries(df["geometry_wkb"].map(lambda v: from_wkb(_to_bytes(v))), crs=OUTPUT_CRS)
    return gpd.GeoDataFrame(df.drop(columns=["geometry_wkb"]), geometry=geometry, crs=OUTPUT_CRS)


def fetch_building_centroids(con: duckdb.DuckDBPyConnection, municipalities: list[str]) -> gpd.GeoDataFrame:
    """Return Overture building centroids for ``municipalities``."""

    names_sql = ", ".join("?" * len(municipalities))
    df = con.execute(
        f"""
        SELECT
            id AS building_id,
            municipality_name AS municipio,
            ST_X(ST_Centroid(geometry)) AS building_centroid_lon,
            ST_Y(ST_Centroid(geometry)) AS building_centroid_lat
        FROM pr_overture_buildings
        WHERE municipality_name IN ({names_sql});
        """,
        list(municipalities),
    ).fetchdf()
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["building_centroid_lon"], df["building_centroid_lat"]),
        crs=OUTPUT_CRS,
    )


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
            CAST(municipio AS VARCHAR) AS municipio,
            CAST(lon AS DOUBLE) AS lon,
            CAST(lat AS DOUBLE) AS lat,
            CAST(radius_m AS INTEGER) AS radius_m,
            CAST(building_count AS INTEGER) AS building_count,
            CAST(osm_pv_count AS INTEGER) AS osm_pv_count,
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
# ## Step 1 — Load vector inputs

# %%
if __name__ == "__main__":
    db_path = resolve_db_path()
    print(f"DuckDB: {db_path}")
    con = connect(db_path)

    block_groups = fetch_block_groups(con, TARGET_MUNICIPALITIES)
    buildings = fetch_building_centroids(con, TARGET_MUNICIPALITIES)
    osm_pv = fetch_osm_pv(con)
    print(f"block groups: {len(block_groups):,} | buildings: {len(buildings):,} | osm PV: {len(osm_pv):,}")

    seeds = fetch_seed_neighborhoods()
    print(f"seed neighborhoods: {len(seeds):,}")

# %% [markdown]
# ## Step 2 — Generate tile centers per Block Group (UTM 19N, 175 m disks)

# %%
if __name__ == "__main__":
    manifest = build_tile_manifest(
        block_groups.rename(columns={"bg_geoid": "GEOID"}),
        radius_m=DEFAULT_RADIUS_M,
        spacing_m=DEFAULT_SPACING_M,
        geoid_col="GEOID",
        municipio_col="municipio",
    )
    print(f"initial tile candidates: {len(manifest):,}")

# %% [markdown]
# ## Step 3 — Attach priority, building count, and default expected quality

# %%
if __name__ == "__main__":
    manifest = attach_priority(
        manifest,
        osm_pv_polygons=osm_pv,
        seed_neighborhoods=seeds,
        buildings_gdf=buildings,
    )

    # Drop tiles with zero buildings (coverage with no targets is pure waste).
    manifest = manifest[manifest["building_count"] > 0].copy()

    manifest["expected_quality"] = manifest["municipio"].map(DEFAULT_EXPECTED_QUALITY).fillna("MEDIUM")
    manifest["required_quality"] = manifest["expected_quality"]
    manifest["status"] = "pending"

    # Re-index tile_id for stability.
    manifest = manifest.sort_values(["priority_score", "osm_pv_count", "building_count"], ascending=[False, False, False]).reset_index(drop=True)
    print("priority distribution:\n", manifest["priority_score"].value_counts().sort_index(ascending=False).to_string())
    print("tiles per municipality:\n", manifest["municipio"].value_counts().to_string())
    print(f"final tile count (post-drop of empty tiles): {len(manifest):,}")

# %% [markdown]
# ## Step 4 — Persist to DuckDB

# %%
if __name__ == "__main__":
    write_manifest_to_duckdb(con, manifest, MANIFEST_TABLE)
    preview = con.execute(
        f"SELECT priority_score, municipio, expected_quality, COUNT(*) AS tiles, SUM(building_count) AS buildings "
        f"FROM {MANIFEST_TABLE} GROUP BY 1,2,3 ORDER BY 1 DESC, 2, 3"
    ).fetchdf()
    print(preview.to_string(index=False))
    con.close()
