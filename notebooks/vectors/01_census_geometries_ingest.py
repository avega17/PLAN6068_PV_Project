"""01_census_geometries_ingest.py

Jupytext-friendly script for ingesting Puerto Rico census geometries into a
local DuckDB spatial database.

This script fetches three foundational geographies for Puerto Rico:
- Municipalities (Census counties equivalent; state FIPS 72)
- Census tracts
- Block groups

The output is stored in a local DuckDB database so later notebooks can use the
tables as reusable spatial bounding boxes and join targets.
"""

# %%
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
from censusdis.maps import ShapeReader
from dotenv import load_dotenv


def resolve_project_root(start: Path | None = None) -> Path:
    """Find the repository root regardless of notebook working directory."""

    current = (start or Path.cwd()).resolve()
    markers = ("project_rules.md", ".git")
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return current


PROJECT_ROOT = resolve_project_root()
TARGET_STATE_FIPS = "72"
CENSUS_YEAR = 2020
OUTPUT_CRS = "EPSG:4326"

load_dotenv(PROJECT_ROOT / ".env")


def resolve_db_path() -> Path:
    """Resolve the DuckDB file path from the workspace or .env settings."""

    db_path_value = os.getenv("VECTOR_DB")
    if db_path_value:
        db_path = Path(db_path_value)
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path if len(db_path.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / db_path
        return db_path

    return PROJECT_ROOT / "data" / "vectors" / "PR_vector_data.duckdb"


# %%
def fetch_census_geography(reader: ShapeReader, geography: str) -> gpd.GeoDataFrame:
    """Fetch a Census geography layer and filter it to Puerto Rico only."""

    gdf = reader.read_cb_shapefile(
        shapefile_scope="us",
        geography=geography,
        crs=OUTPUT_CRS,
    )

    if gdf.crs is None:
        gdf = gdf.set_crs(OUTPUT_CRS)
    else:
        gdf = gdf.to_crs(OUTPUT_CRS)

    if "STATEFP" in gdf.columns:
        gdf = gdf[gdf["STATEFP"].astype(str) == TARGET_STATE_FIPS].copy()

    return gdf


def prepare_municipalities(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Standardize Puerto Rico county-equivalent geometries for storage."""

    columns = [col for col in ["STATEFP", "COUNTYFP", "GEOID", "NAME", "geometry"] if col in gdf.columns]
    municipalities = gdf.loc[:, columns].copy()
    municipalities.insert(0, "geography_level", "municipality")
    return municipalities


def prepare_tracts(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Standardize Census tract geometries for storage."""

    columns = [
        col
        for col in ["STATEFP", "COUNTYFP", "TRACTCE", "GEOID", "NAME", "ALAND", "AWATER", "geometry"]
        if col in gdf.columns
    ]
    tracts = gdf.loc[:, columns].copy()
    tracts.insert(0, "geography_level", "tract")
    return tracts


def prepare_block_groups(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Standardize Census block group geometries for storage."""

    columns = [
        col
        for col in [
            "STATEFP",
            "COUNTYFP",
            "TRACTCE",
            "BLKGRPCE",
            "GEOID",
            "NAME",
            "ALAND",
            "AWATER",
            "geometry",
        ]
        if col in gdf.columns
    ]
    block_groups = gdf.loc[:, columns].copy()
    block_groups.insert(0, "geography_level", "block_group")
    return block_groups


def create_spatial_connection(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection and load the spatial extension once."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(db_path))
    connection.execute("INSTALL spatial;")
    connection.execute("LOAD spatial;")
    return connection


def upsert_geodataframe(con: duckdb.DuckDBPyConnection, table_name: str, gdf: gpd.GeoDataFrame) -> None:
    """Persist a GeoDataFrame as a native DuckDB GEOMETRY table.

    The staged frame keeps a single column named `geometry`, but it temporarily
    stores WKB bytes so DuckDB can materialize a native GEOMETRY column without
    introducing a second geometry field. This keeps the table schema simple
    while preserving clean Shapely <-> WKB round-tripping between GeoPandas and
    DuckDB.
    """

    staged = pd.DataFrame(gdf.copy())
    staged["geometry"] = gdf.geometry.to_wkb()

    con.register("staged_geography", staged)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT
            * EXCLUDE (geometry),
            ST_GeomFromWKB(geometry) AS geometry,
            CAST(NULL AS BOOLEAN) AS has_PV
        FROM staged_geography;
        """
    )

    # Create a spatial index so later notebooks can clip and filter efficiently
    # when the table is used as a bounding-box source.
    con.execute(f"CREATE INDEX idx_{table_name}_geometry ON {table_name} USING RTREE (geometry);")
    con.unregister("staged_geography")


def list_db_tables(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """List user tables in the active DuckDB database."""

    return con.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
        """
    ).fetchdf()


def preview_table_schema(con: duckdb.DuckDBPyConnection, table_name: str) -> pd.DataFrame:
    """Return DESCRIBE output for a table."""

    return con.execute(f"DESCRIBE {table_name}").fetchdf()


def preview_table_rows(con: duckdb.DuckDBPyConnection, table_name: str, limit: int = 5) -> pd.DataFrame:
    """Return a small sample of table rows."""

    return con.execute(f"FROM {table_name} LIMIT {limit}").fetchdf()


def preview_geometry_samples(gdf: gpd.GeoDataFrame, n: int = 5) -> pd.DataFrame:
    """Return a small sample of WKT geometry strings for quick visual inspection."""

    sample = gdf.head(n).copy()
    sample["geometry_wkt"] = sample.geometry.to_wkt()
    keep = [col for col in ["GEOID", "NAME", "geometry_wkt"] if col in sample.columns]
    return sample[keep]


def main() -> None:
    reader = ShapeReader(year=CENSUS_YEAR)
    db_path = resolve_db_path()
    con = create_spatial_connection(db_path)

    try:
        tables_before = list_db_tables(con)
        print("Tables available before census ingest:")
        print(tables_before.to_string(index=False))

        # The censusdis boundary reader already includes the descriptive fields
        # we need; we filter to Puerto Rico and persist each level separately.
        municipalities = prepare_municipalities(fetch_census_geography(reader, "county"))
        tracts = prepare_tracts(fetch_census_geography(reader, "tract"))
        # Census cartographic boundary files use the abbreviated geography code
        # `bg` for block groups.
        block_groups = prepare_block_groups(fetch_census_geography(reader, "bg"))

        print("\nSample Shapely geometries from fetched municipalities:")
        print(preview_geometry_samples(municipalities, n=5).to_string(index=False))
        print("\nSample Shapely geometries from fetched census tracts:")
        print(preview_geometry_samples(tracts, n=5).to_string(index=False))
        print("\nSample Shapely geometries from fetched block groups:")
        print(preview_geometry_samples(block_groups, n=5).to_string(index=False))

        upsert_geodataframe(con, "pr_municipalities", municipalities)
        upsert_geodataframe(con, "pr_census_tracts", tracts)
        upsert_geodataframe(con, "pr_block_groups", block_groups)

        tables_after = list_db_tables(con)
        print("\nTables available after census ingest:")
        print(tables_after.to_string(index=False))

        print("\nTable schema preview: pr_municipalities")
        print(preview_table_schema(con, "pr_municipalities").to_string(index=False))
        print("\nSample rows: pr_municipalities")
        print(preview_table_rows(con, "pr_municipalities", limit=5).to_string(index=False))

        print("\nTable schema preview: pr_census_tracts")
        print(preview_table_schema(con, "pr_census_tracts").to_string(index=False))
        print("\nSample rows: pr_census_tracts")
        print(preview_table_rows(con, "pr_census_tracts", limit=5).to_string(index=False))

        print("\nTable schema preview: pr_block_groups")
        print(preview_table_schema(con, "pr_block_groups").to_string(index=False))
        print("\nSample rows: pr_block_groups")
        print(preview_table_rows(con, "pr_block_groups", limit=5).to_string(index=False))

        print(f"Saved DuckDB spatial database to: {db_path}")
        print("Created tables: pr_municipalities, pr_census_tracts, pr_block_groups")
    finally:
        # Always close the connection so later notebook runs do not inherit a
        # locked database handle.
        con.close()


# %%
# Notebook/script entrypoint.
main()