"""Shared Census geometry helpers for PLAN6068 notebooks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
from censusdis.maps import MapException, ShapeReader
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
OUTPUT_CRS = "EPSG:4326"
TARGET_STATE_FIPS = "72"
CANONICAL_COUNTY_TABLE = "pr_census_counties"
CANONICAL_BLOCK_GROUP_TABLE = "pr_census_block_groups"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class CensusLayerSpec:
    """Configuration for a project Census geography layer."""

    geography: str
    reader_geographies: tuple[str, ...]
    fetch_method: str
    shapefile_scope: str
    geography_level: str
    table_name: str
    columns: tuple[str, ...]


CENSUS_LAYER_SPECS: dict[str, CensusLayerSpec] = {
    "municipality": CensusLayerSpec(
        geography="municipality",
        reader_geographies=("county",),
        fetch_method="cb",
        shapefile_scope="us",
        geography_level="municipality",
        table_name=CANONICAL_COUNTY_TABLE,
        columns=("STATEFP", "COUNTYFP", "GEOID", "NAME", "geometry"),
    ),
    "tract": CensusLayerSpec(
        geography="tract",
        reader_geographies=("tract",),
        fetch_method="cb",
        shapefile_scope="us",
        geography_level="tract",
        table_name="pr_census_tracts",
        columns=("STATEFP", "COUNTYFP", "TRACTCE", "GEOID", "NAME", "ALAND", "AWATER", "geometry"),
    ),
    "block_group": CensusLayerSpec(
        geography="block_group",
        reader_geographies=("bg",),
        fetch_method="cb",
        shapefile_scope="us",
        geography_level="block_group",
        table_name=CANONICAL_BLOCK_GROUP_TABLE,
        columns=(
            "STATEFP",
            "COUNTYFP",
            "TRACTCE",
            "BLKGRPCE",
            "GEOID",
            "NAME",
            "ALAND",
            "AWATER",
            "geometry",
        ),
    ),
    "block": CensusLayerSpec(
        geography="block",
        reader_geographies=("tabblock20", "tabblock"),
        fetch_method="tiger",
        shapefile_scope="state",
        geography_level="block",
        table_name="pr_census_blocks",
        columns=(
            "STATEFP",
            "COUNTYFP",
            "TRACTCE",
            "BLOCKCE",
            "GEOID",
            "NAME",
            "ALAND",
            "AWATER",
            "geometry",
        ),
    ),
}

CENSUS_LAYER_ORDER = tuple(CENSUS_LAYER_SPECS.keys())

_GEOGRAPHY_ALIASES = {
    "municipality": "municipality",
    "municipalities": "municipality",
    "county": "municipality",
    "counties": "municipality",
    "tract": "tract",
    "tracts": "tract",
    "block_group": "block_group",
    "block_groups": "block_group",
    "bg": "block_group",
    "block": "block",
    "blocks": "block",
    "census_block": "block",
    "census_blocks": "block",
    "tabblock": "block",
    "tabblock20": "block",
}


def get_census_layer_spec(geography: str) -> CensusLayerSpec:
    """Return the project spec for a supported Census geography."""

    canonical = _GEOGRAPHY_ALIASES.get(geography.strip().lower())
    if canonical is None:
        supported = ", ".join(sorted(CENSUS_LAYER_SPECS))
        raise ValueError(f"Unsupported Census geography '{geography}'. Expected one of: {supported}")
    return CENSUS_LAYER_SPECS[canonical]


def resolve_vector_db_path(project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve the DuckDB file path from the workspace or .env settings."""

    db_path_value = os.getenv("VECTOR_DB")
    if db_path_value:
        db_path = Path(db_path_value)
        if not db_path.is_absolute():
            db_path = project_root / db_path if len(db_path.parts) > 1 else project_root / "data" / "vectors" / db_path
        return db_path

    return project_root / "data" / "PR_PV_plan_data.duckdb"


def create_spatial_connection(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection and load the spatial extension once."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(db_path))
    connection.execute("INSTALL spatial;")
    connection.execute("LOAD spatial;")
    return connection


def fetch_census_geography(
    reader: ShapeReader,
    geography: str,
    *,
    state_fips: str = TARGET_STATE_FIPS,
    output_crs: str = OUTPUT_CRS,
    timeout: int = 120,
) -> gpd.GeoDataFrame:
    """Fetch a Census geography layer and filter it to Puerto Rico only."""

    spec = get_census_layer_spec(geography)
    shapefile_scope = state_fips if spec.shapefile_scope == "state" else spec.shapefile_scope
    last_error: Exception | None = None

    for reader_geography in spec.reader_geographies:
        try:
            if spec.fetch_method == "tiger":
                gdf = reader.read_shapefile(
                    shapefile_scope=shapefile_scope,
                    geography=reader_geography,
                    crs=output_crs,
                    timeout=timeout,
                )
            else:
                gdf = reader.read_cb_shapefile(
                    shapefile_scope=shapefile_scope,
                    geography=reader_geography,
                    crs=output_crs,
                    timeout=timeout,
                )
            break
        except MapException as exc:
            last_error = exc
    else:
        detail = str(last_error) if last_error is not None else "unknown error"
        raise RuntimeError(f"Failed to fetch Census geography '{spec.geography}': {detail}") from last_error

    if gdf.crs is None:
        gdf = gdf.set_crs(output_crs)
    else:
        gdf = gdf.to_crs(output_crs)

    if "STATEFP" in gdf.columns:
        gdf = gdf[gdf["STATEFP"].astype(str).str.zfill(2) == state_fips].copy()

    return gdf.reset_index(drop=True)


def prepare_census_geography(gdf: gpd.GeoDataFrame, geography: str) -> gpd.GeoDataFrame:
    """Standardize a Census geography for project storage."""

    spec = get_census_layer_spec(geography)
    columns = [column for column in spec.columns if column in gdf.columns]
    prepared = gdf.loc[:, columns].copy()
    prepared.insert(0, "geography_level", spec.geography_level)
    return prepared


def fetch_prepared_census_geography(
    reader: ShapeReader,
    geography: str,
    *,
    state_fips: str = TARGET_STATE_FIPS,
    output_crs: str = OUTPUT_CRS,
    timeout: int = 120,
) -> gpd.GeoDataFrame:
    """Fetch a project Census layer and immediately standardize its schema."""

    return prepare_census_geography(
        fetch_census_geography(
            reader,
            geography,
            state_fips=state_fips,
            output_crs=output_crs,
            timeout=timeout,
        ),
        geography,
    )


def upsert_geodataframe(con: duckdb.DuckDBPyConnection, table_name: str, gdf: gpd.GeoDataFrame) -> None:
    """Persist a GeoDataFrame as a native DuckDB GEOMETRY table."""

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
    keep = [column for column in ["GEOID", "NAME", "geometry_wkt"] if column in sample.columns]
    return sample[keep]