# %% [markdown]
# # 2020 Urban Areas reference and urban summary views for Puerto Rico
# 
# This notebook now has a narrower durable footprint:
# 
# 1. Materialize a curated **Puerto Rico-only** 2020 urban-block reference table.
# 2. Build urban summary **views** for block groups, tracts, and counties using
#    those 2020 numerators plus the 2020 ACS tables created by the ACS ingest
#    notebook.
# 3. Keep the 2020-versus-2024 census-block comparison as a notebook-local
#    diagnostic rather than persisting extra comparison tables.
# 
# The Urban Areas product itself remains pinned to 2020 because that is still
# the latest Census urban-rural release.

# %%
"""05_urban_blocks_2020_ingest.py"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
from censusdis.maps import ShapeReader
from IPython.display import IFrame, display


def _bootstrap_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    markers = ("project_rules.md", ".git")
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return current


PROJECT_ROOT = _bootstrap_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.acs import table_name_for_acs
from utils.census import create_spatial_connection
from utils.census import fetch_prepared_census_geography
from utils.census import get_census_layer_spec
from utils.census import resolve_vector_db_path
from utils.census import upsert_geodataframe

UA_BLOCKS_URL = "https://www2.census.gov/geo/docs/reference/ua/2020_UA_BLOCKS.txt"
CACHE_DIR = PROJECT_ROOT / "data" / "tabular"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PR_CACHE_FILE = CACHE_DIR / "2020_UA_BLOCKS_PR.parquet"
PR_STATE_FIPS = "72"
UA_READ_CHUNKSIZE = 250_000
NEAR_MATCH_THRESHOLD = 0.999
TARGET_MUNICIPALITIES = ("San Juan", "Isabela")

URBAN_BLOCK_TABLE = "pr_urban_blocks_2020"
BG_URBAN_FLAG_VIEW = "pr_bg_urban_flags"
BG_URBAN_STATS_VIEW = "vw_pr_census_block_group_urban_stats"
TRACT_URBAN_STATS_VIEW = "vw_pr_census_tract_urban_stats"
COUNTY_URBAN_STATS_VIEW = "vw_pr_census_county_urban_stats"
CENSUS_2020_BLOCK_TABLE = get_census_layer_spec("block").table_name
TEMP_CENSUS_2024_BLOCK_TABLE = "tmp_pr_census_blocks_2024"
BLOCK_GROUP_GEOMETRY_TABLE = get_census_layer_spec("block_group").table_name
TRACT_GEOMETRY_TABLE = get_census_layer_spec("tract").table_name
COUNTY_GEOMETRY_TABLE = get_census_layer_spec("municipality").table_name
ACS_YEAR_FOR_URBAN_DENOMINATORS = 2020
URBAN_PREVIEW_MAP_PATH = PROJECT_ROOT / "outputs" / "maps" / "pr_urban_all_geographies_preview.html"
URBAN_PREVIEW_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)


def normalize_urban_blocks_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize UA-block columns and derive reusable GEOIDs."""

    normalized = frame.copy()
    normalized.columns = [column.strip().upper() for column in normalized.columns]

    expected = {"STATE", "COUNTY", "TRACT", "BLOCK"}
    missing = expected - set(normalized.columns)
    if missing:
        raise RuntimeError(f"2020_UA_BLOCKS source is missing columns: {sorted(missing)}")

    pr_blocks = normalized[normalized["STATE"].fillna("").astype(str).str.zfill(2) == PR_STATE_FIPS].copy()
    if pr_blocks.empty:
        return pr_blocks

    pr_blocks["STATE"] = pr_blocks["STATE"].fillna("").astype(str).str.zfill(2)
    pr_blocks["COUNTY"] = pr_blocks["COUNTY"].fillna("").astype(str).str.zfill(3)
    pr_blocks["TRACT"] = pr_blocks["TRACT"].fillna("").astype(str).str.zfill(6)
    pr_blocks["BLOCK"] = pr_blocks["BLOCK"].fillna("").astype(str).str.zfill(4)
    pr_blocks["block_geoid"] = pr_blocks["STATE"] + pr_blocks["COUNTY"] + pr_blocks["TRACT"] + pr_blocks["BLOCK"]
    pr_blocks["bg_geoid"] = pr_blocks["STATE"] + pr_blocks["COUNTY"] + pr_blocks["TRACT"] + pr_blocks["BLOCK"].str[0]
    pr_blocks["tract_geoid"] = pr_blocks["STATE"] + pr_blocks["COUNTY"] + pr_blocks["TRACT"]
    pr_blocks["county_geoid"] = pr_blocks["STATE"] + pr_blocks["COUNTY"]
    return pr_blocks.reset_index(drop=True)


def materialize_pr_urban_blocks_cache(force_refresh: bool = False) -> Path:
    """Build a durable Puerto Rico-only urban-block cache as parquet."""

    if PR_CACHE_FILE.exists() and not force_refresh:
        return PR_CACHE_FILE

    print(f"[cache] materializing Puerto Rico-only UA blocks cache → {PR_CACHE_FILE}")
    pr_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        UA_BLOCKS_URL,
        sep="|",
        dtype=str,
        low_memory=False,
        chunksize=UA_READ_CHUNKSIZE,
        encoding="latin-1",
    ):
        pr_chunk = normalize_urban_blocks_frame(chunk)
        if not pr_chunk.empty:
            pr_chunks.append(pr_chunk)

    if not pr_chunks:
        raise RuntimeError("Failed to extract any Puerto Rico rows from the 2020 Urban Areas block file.")

    pr_blocks = pd.concat(pr_chunks, ignore_index=True)
    pr_blocks.to_parquet(PR_CACHE_FILE, index=False)
    return PR_CACHE_FILE


def load_pr_urban_blocks_cache(path: Path) -> pd.DataFrame:
    """Load the durable Puerto Rico-only urban-block cache."""

    frame = pd.read_parquet(path)
    required_columns = {"block_geoid", "bg_geoid", "tract_geoid", "county_geoid"}
    if required_columns.issubset(frame.columns):
        return frame
    return normalize_urban_blocks_frame(frame)


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """Return whether a table exists in DuckDB."""

    return bool(
        con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
            """,
            [table_name],
        ).fetchone()[0]
    )


def drop_relation_if_exists(con: duckdb.DuckDBPyConnection, relation_name: str) -> None:
    """Drop a table or view by name, whichever currently exists."""

    relation = con.execute(
        """
        SELECT table_type
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
        """,
        [relation_name],
    ).fetchone()
    if relation is None:
        return
    if relation[0] == "VIEW":
        con.execute(f"DROP VIEW IF EXISTS {relation_name};")
        return
    con.execute(f"DROP TABLE IF EXISTS {relation_name};")


def ensure_required_acs_tables(con: duckdb.DuckDBPyConnection, *, year: int = ACS_YEAR_FOR_URBAN_DENOMINATORS) -> None:
    """Fail early unless the ACS notebook has already produced 2020 denominators."""

    required = [table_name_for_acs(year, geography) for geography in ("county", "tract", "block_group")]
    missing = [table_name for table_name in required if not table_exists(con, table_name)]
    if missing:
        details = ", ".join(missing)
        raise RuntimeError(
            f"Missing ACS denominator tables: {details}. Run notebooks/tabular/04_acs_5year_ingest.py first."
        )


def persist_urban_blocks(con: duckdb.DuckDBPyConnection, pr_blocks: pd.DataFrame) -> None:
    """Persist a curated 2020 urban-block numerator table."""

    con.register("pr_urban_blocks_df", pr_blocks)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {URBAN_BLOCK_TABLE} AS
        SELECT
            block_geoid,
            bg_geoid,
            tract_geoid,
            county_geoid,
            STATE AS state_fips,
            COUNTY AS county_fips,
            TRACT AS tract_code,
            BLOCK AS block_code,
            TRY_CAST(AREALAND AS BIGINT) AS urban_land_area_m2,
            TRY_CAST("2020_POP" AS BIGINT) AS urban_pop_2020,
            TRY_CAST("2020_HOU" AS BIGINT) AS urban_housing_units_2020,
            CAST("2020_UACE" AS VARCHAR) AS urban_area_code_2020,
            CAST("2020_UA_NAME" AS VARCHAR) AS urban_area_name_2020
        FROM pr_urban_blocks_df;
        """
    )
    con.unregister("pr_urban_blocks_df")


def create_urban_summary_views(con: duckdb.DuckDBPyConnection) -> None:
    """Create urban summary views across block groups, tracts, and counties."""

    acs_bg_table = table_name_for_acs(ACS_YEAR_FOR_URBAN_DENOMINATORS, "block_group")
    acs_tract_table = table_name_for_acs(ACS_YEAR_FOR_URBAN_DENOMINATORS, "tract")
    acs_county_table = table_name_for_acs(ACS_YEAR_FOR_URBAN_DENOMINATORS, "county")

    con.execute(
        f"""
        CREATE OR REPLACE VIEW {BG_URBAN_STATS_VIEW} AS
        WITH urban_agg AS (
            SELECT bg_geoid,
                   COUNT(*)::BIGINT AS urban_block_count,
                   SUM(urban_pop_2020) AS total_urban_pop,
                   SUM(urban_housing_units_2020) AS total_urban_housing_units,
                   SUM(urban_land_area_m2) AS urban_land_area_m2
            FROM {URBAN_BLOCK_TABLE}
            GROUP BY bg_geoid
        )
        SELECT bg.GEOID AS bg_geoid,
               muni.GEOID AS municipio_geoid,
               muni.NAME AS municipio,
               bg.STATEFP,
               bg.COUNTYFP,
               bg.TRACTCE,
               bg.BLKGRPCE,
               bg.NAME,
               bg.ALAND AS total_land_area_m2,
               COALESCE(urban_agg.urban_block_count, 0) AS urban_block_count,
               COALESCE(urban_agg.total_urban_pop, 0) AS total_urban_pop,
               COALESCE(urban_agg.total_urban_housing_units, 0) AS total_urban_housing_units,
               COALESCE(urban_agg.urban_land_area_m2, 0) AS urban_land_area_m2,
               acs.total_population AS total_population_acs,
               acs.total_housing_units AS total_housing_units_acs,
               acs.median_household_income_usd,
               acs.poverty_rate,
               acs.pct_owner_occupied,
               acs.pct_bachelor_plus,
               CASE
                   WHEN acs.total_population > 0
                       THEN COALESCE(urban_agg.total_urban_pop, 0)::DOUBLE / acs.total_population
                   ELSE NULL
               END AS pct_urban_pop,
               CASE
                   WHEN acs.total_population > 0
                       THEN COALESCE(urban_agg.total_urban_pop, 0)::DOUBLE / acs.total_population
                   ELSE NULL
               END AS pct_urban_population,
               CASE
                   WHEN acs.total_housing_units > 0
                       THEN COALESCE(urban_agg.total_urban_housing_units, 0)::DOUBLE / acs.total_housing_units
                   ELSE NULL
               END AS pct_urban_housing_units,
               CASE
                   WHEN bg.ALAND > 0
                       THEN COALESCE(urban_agg.urban_land_area_m2, 0)::DOUBLE / bg.ALAND
                   ELSE NULL
               END AS pct_urban_land_area,
               COALESCE(urban_agg.urban_block_count, 0) > 0 AS is_urban
        FROM {BLOCK_GROUP_GEOMETRY_TABLE} AS bg
        LEFT JOIN urban_agg
          ON urban_agg.bg_geoid = bg.GEOID
        LEFT JOIN {acs_bg_table} AS acs
          ON acs.bg_geoid = bg.GEOID
        LEFT JOIN {COUNTY_GEOMETRY_TABLE} AS muni
          ON muni.STATEFP = bg.STATEFP AND muni.COUNTYFP = bg.COUNTYFP;
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE VIEW {TRACT_URBAN_STATS_VIEW} AS
        WITH urban_agg AS (
            SELECT tract_geoid,
                   county_geoid,
                   COUNT(*)::BIGINT AS urban_block_count,
                   SUM(urban_pop_2020) AS total_urban_pop,
                   SUM(urban_housing_units_2020) AS total_urban_housing_units,
                   SUM(urban_land_area_m2) AS urban_land_area_m2
            FROM {URBAN_BLOCK_TABLE}
            GROUP BY tract_geoid, county_geoid
        )
        SELECT tract.GEOID AS tract_geoid,
               muni.GEOID AS municipio_geoid,
               muni.NAME AS municipio,
               tract.STATEFP,
               tract.COUNTYFP,
               tract.TRACTCE,
               tract.NAME,
               tract.ALAND AS total_land_area_m2,
               COALESCE(urban_agg.urban_block_count, 0) AS urban_block_count,
               COALESCE(urban_agg.total_urban_pop, 0) AS total_urban_pop,
               COALESCE(urban_agg.total_urban_housing_units, 0) AS total_urban_housing_units,
               COALESCE(urban_agg.urban_land_area_m2, 0) AS urban_land_area_m2,
               acs.total_population AS total_population_acs,
               acs.total_housing_units AS total_housing_units_acs,
               acs.median_household_income_usd,
               acs.poverty_rate,
               acs.pct_owner_occupied,
               acs.pct_bachelor_plus,
               CASE
                   WHEN acs.total_population > 0
                       THEN COALESCE(urban_agg.total_urban_pop, 0)::DOUBLE / acs.total_population
                   ELSE NULL
               END AS pct_urban_pop,
               CASE
                   WHEN acs.total_population > 0
                       THEN COALESCE(urban_agg.total_urban_pop, 0)::DOUBLE / acs.total_population
                   ELSE NULL
               END AS pct_urban_population,
               CASE
                   WHEN acs.total_housing_units > 0
                       THEN COALESCE(urban_agg.total_urban_housing_units, 0)::DOUBLE / acs.total_housing_units
                   ELSE NULL
               END AS pct_urban_housing_units,
               CASE
                   WHEN tract.ALAND > 0
                       THEN COALESCE(urban_agg.urban_land_area_m2, 0)::DOUBLE / tract.ALAND
                   ELSE NULL
               END AS pct_urban_land_area,
               COALESCE(urban_agg.urban_block_count, 0) > 0 AS is_urban
        FROM {TRACT_GEOMETRY_TABLE} AS tract
        LEFT JOIN urban_agg
          ON urban_agg.tract_geoid = tract.GEOID
        LEFT JOIN {acs_tract_table} AS acs
          ON acs.tract_geoid = tract.GEOID
        LEFT JOIN {COUNTY_GEOMETRY_TABLE} AS muni
          ON muni.STATEFP = tract.STATEFP AND muni.COUNTYFP = tract.COUNTYFP;
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE VIEW {COUNTY_URBAN_STATS_VIEW} AS
        WITH urban_agg AS (
            SELECT county_geoid,
                   COUNT(*)::BIGINT AS urban_block_count,
                   SUM(urban_pop_2020) AS total_urban_pop,
                   SUM(urban_housing_units_2020) AS total_urban_housing_units,
                   SUM(urban_land_area_m2) AS urban_land_area_m2
            FROM {URBAN_BLOCK_TABLE}
            GROUP BY county_geoid
        ),
        county_land_area AS (
            SELECT SUBSTR(GEOID, 1, 5) AS county_geoid,
                   SUM(ALAND) AS total_land_area_m2
            FROM {CENSUS_2020_BLOCK_TABLE}
            GROUP BY county_geoid
        )
        SELECT muni.GEOID AS county_geoid,
               muni.GEOID AS municipio_geoid,
               muni.NAME AS municipio,
               muni.STATEFP,
               muni.COUNTYFP,
               muni.NAME,
               county_land_area.total_land_area_m2,
               COALESCE(urban_agg.urban_block_count, 0) AS urban_block_count,
               COALESCE(urban_agg.total_urban_pop, 0) AS total_urban_pop,
               COALESCE(urban_agg.total_urban_housing_units, 0) AS total_urban_housing_units,
               COALESCE(urban_agg.urban_land_area_m2, 0) AS urban_land_area_m2,
               acs.total_population AS total_population_acs,
               acs.total_housing_units AS total_housing_units_acs,
               acs.median_household_income_usd,
               acs.poverty_rate,
               acs.pct_owner_occupied,
               acs.pct_bachelor_plus,
               CASE
                   WHEN acs.total_population > 0
                       THEN COALESCE(urban_agg.total_urban_pop, 0)::DOUBLE / acs.total_population
                   ELSE NULL
               END AS pct_urban_pop,
               CASE
                   WHEN acs.total_population > 0
                       THEN COALESCE(urban_agg.total_urban_pop, 0)::DOUBLE / acs.total_population
                   ELSE NULL
               END AS pct_urban_population,
               CASE
                   WHEN acs.total_housing_units > 0
                       THEN COALESCE(urban_agg.total_urban_housing_units, 0)::DOUBLE / acs.total_housing_units
                   ELSE NULL
               END AS pct_urban_housing_units,
               CASE
                   WHEN county_land_area.total_land_area_m2 > 0
                       THEN COALESCE(urban_agg.urban_land_area_m2, 0)::DOUBLE / county_land_area.total_land_area_m2
                   ELSE NULL
               END AS pct_urban_land_area,
               COALESCE(urban_agg.urban_block_count, 0) > 0 AS is_urban
        FROM {COUNTY_GEOMETRY_TABLE} AS muni
        LEFT JOIN urban_agg
          ON urban_agg.county_geoid = muni.GEOID
        LEFT JOIN county_land_area
          ON county_land_area.county_geoid = muni.GEOID
        LEFT JOIN {acs_county_table} AS acs
          ON acs.county_geoid = muni.GEOID;
        """
    )

    drop_relation_if_exists(con, BG_URBAN_FLAG_VIEW)
    con.execute(
        f"""
        CREATE VIEW {BG_URBAN_FLAG_VIEW} AS
        SELECT bg_geoid,
               municipio,
               municipio_geoid,
               urban_block_count,
               total_urban_pop,
               total_urban_housing_units,
               urban_land_area_m2,
               pct_urban_pop,
               pct_urban_population,
               pct_urban_housing_units,
               pct_urban_land_area,
               is_urban
        FROM {BG_URBAN_STATS_VIEW};
        """
    )


def summarize_urban_blocks(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Return a compact summary of the curated urban-block numerator table."""

    return con.execute(
        f"""
        SELECT COUNT(*) AS urban_block_rows,
               COUNT(DISTINCT bg_geoid) AS urban_bg_count,
               SUM(urban_pop_2020) AS total_urban_pop,
               SUM(urban_housing_units_2020) AS total_urban_housing_units,
               SUM(urban_land_area_m2) AS total_urban_land_area_m2
        FROM {URBAN_BLOCK_TABLE};
        """
    ).fetchdf()


def summarize_urban_views(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Return a compact summary across the urban summary views."""

    summaries = []
    for view_name in (BG_URBAN_STATS_VIEW, TRACT_URBAN_STATS_VIEW, COUNTY_URBAN_STATS_VIEW):
        summaries.append(
            con.execute(
                f"""
                SELECT '{view_name}' AS view_name,
                       COUNT(*) AS geography_rows,
                       SUM(CASE WHEN is_urban THEN 1 ELSE 0 END) AS urban_geography_rows,
                       SUM(CASE WHEN pct_urban_pop > 1 THEN 1 ELSE 0 END) AS over_100_pct_urban_pop_rows,
                       MEDIAN(pct_urban_pop) AS median_pct_urban_pop,
                       MAX(pct_urban_pop) AS max_pct_urban_pop,
                       MEDIAN(pct_urban_land_area) AS median_pct_urban_land_area
                FROM {view_name};
                """
            ).fetchdf()
        )
    return pd.concat(summaries, ignore_index=True)


def summarize_target_block_groups(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Return block-group urban summaries for San Juan and Isabela."""

    municipality_sql = ", ".join(f"'{municipio}'" for municipio in TARGET_MUNICIPALITIES)
    return con.execute(
        f"""
        SELECT municipio,
               COUNT(*) AS bg_rows,
               SUM(CASE WHEN is_urban THEN 1 ELSE 0 END) AS urban_bg_rows,
               SUM(total_urban_pop) AS total_urban_pop,
               SUM(CASE WHEN pct_urban_pop > 1 THEN 1 ELSE 0 END) AS over_100_pct_urban_pop_rows,
               MEDIAN(pct_urban_pop) AS median_pct_urban_pop,
               MAX(pct_urban_pop) AS max_pct_urban_pop,
               MEDIAN(pct_urban_land_area) AS median_pct_urban_land_area
        FROM {BG_URBAN_STATS_VIEW}
        WHERE municipio IN ({municipality_sql})
        GROUP BY municipio
        ORDER BY municipio;
        """
    ).fetchdf()


def sample_population_ratio_anomalies(con: duckdb.DuckDBPyConnection, limit: int = 15) -> pd.DataFrame:
    """Return the highest block-group population ratios that exceed 100%."""

    return con.execute(
        f"""
        SELECT bg_geoid,
               municipio,
               total_urban_pop,
               total_population_acs,
               pct_urban_pop,
               pct_urban_land_area,
               median_household_income_usd,
               poverty_rate
        FROM {BG_URBAN_STATS_VIEW}
        WHERE pct_urban_pop > 1
        ORDER BY pct_urban_pop DESC, bg_geoid
        LIMIT {limit};
        """
    ).fetchdf()


def _to_bytes(value: object) -> object:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def geodataframe_from_wkb(frame: pd.DataFrame, *, geometry_column: str = "geometry_wkb") -> gpd.GeoDataFrame:
    """Construct a GeoDataFrame from a WKB-bearing DataFrame."""

    geometry = gpd.GeoSeries.from_wkb(frame[geometry_column].map(_to_bytes), crs="EPSG:4326")
    data = frame.drop(columns=[geometry_column]).copy()
    return gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")


def load_urban_blocks_preview_gdf(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    """Load the raw 2020 urban blocks with geometry for interactive previewing."""

    frame = con.execute(
        f"""
        SELECT urb.block_geoid,
               urb.bg_geoid,
               urb.tract_geoid,
               urb.county_geoid,
               muni.NAME AS municipio,
               urb.urban_area_code_2020,
               urb.urban_area_name_2020,
               urb.urban_pop_2020,
               urb.urban_housing_units_2020,
               urb.urban_land_area_m2,
               ST_AsWKB(block.geometry) AS geometry_wkb
        FROM {URBAN_BLOCK_TABLE} AS urb
        JOIN {CENSUS_2020_BLOCK_TABLE} AS block
          ON block.GEOID = urb.block_geoid
        LEFT JOIN {COUNTY_GEOMETRY_TABLE} AS muni
          ON muni.GEOID = urb.county_geoid
        ORDER BY municipio, urb.block_geoid;
        """
    ).fetchdf()
    return geodataframe_from_wkb(frame)


def load_urban_view_gdf(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str,
    geoid_column: str,
    geometry_table: str,
) -> gpd.GeoDataFrame:
    """Load one aggregated urban summary view with its geometry attached."""

    frame = con.execute(
        f"""
        SELECT stats.*,
               ST_AsWKB(geom.geometry) AS geometry_wkb
        FROM {view_name} AS stats
        JOIN {geometry_table} AS geom
          ON geom.GEOID = stats.{geoid_column}
        ORDER BY stats.municipio, stats.{geoid_column};
        """
    ).fetchdf()
    return geodataframe_from_wkb(frame)


def build_interactive_urban_preview(
    con: duckdb.DuckDBPyConnection,
    *,
    output_path: Path = URBAN_PREVIEW_MAP_PATH,
) -> Path:
    """Save an interactive all-municipality map for raw and aggregated urban layers."""

    urban_blocks_gdf = load_urban_blocks_preview_gdf(con)
    bg_gdf = load_urban_view_gdf(
        con,
        view_name=BG_URBAN_STATS_VIEW,
        geoid_column="bg_geoid",
        geometry_table=BLOCK_GROUP_GEOMETRY_TABLE,
    )
    tract_gdf = load_urban_view_gdf(
        con,
        view_name=TRACT_URBAN_STATS_VIEW,
        geoid_column="tract_geoid",
        geometry_table=TRACT_GEOMETRY_TABLE,
    )
    county_gdf = load_urban_view_gdf(
        con,
        view_name=COUNTY_URBAN_STATS_VIEW,
        geoid_column="county_geoid",
        geometry_table=COUNTY_GEOMETRY_TABLE,
    )

    map_obj = urban_blocks_gdf.explore(
        name="2020 urban blocks",
        tiles="CartoDB positron",
        tooltip=[
            "municipio",
            "block_geoid",
            "urban_area_name_2020",
            "urban_pop_2020",
            "urban_housing_units_2020",
        ],
        style_kwds={"fillColor": "#ef6c00", "color": "#ef6c00", "weight": 0.2, "fillOpacity": 0.25},
    )

    bg_gdf.explore(
        m=map_obj,
        name="Block groups: pct urban land area",
        column="pct_urban_land_area",
        cmap="YlOrRd",
        tooltip=[
            "municipio",
            "bg_geoid",
            "urban_block_count",
            "pct_urban_pop",
            "pct_urban_land_area",
            "median_household_income_usd",
            "poverty_rate",
            "pct_owner_occupied",
            "pct_bachelor_plus",
        ],
        legend=True,
        show=False,
        style_kwds={"weight": 0.5, "color": "#6d4c41", "fillOpacity": 0.45},
    )
    tract_gdf.explore(
        m=map_obj,
        name="Tracts: pct urban land area",
        column="pct_urban_land_area",
        cmap="YlGnBu",
        tooltip=[
            "municipio",
            "tract_geoid",
            "urban_block_count",
            "pct_urban_pop",
            "pct_urban_land_area",
            "median_household_income_usd",
            "poverty_rate",
            "pct_owner_occupied",
            "pct_bachelor_plus",
        ],
        legend=True,
        show=False,
        style_kwds={"weight": 0.7, "color": "#1b5e20", "fillOpacity": 0.4},
    )
    county_gdf.explore(
        m=map_obj,
        name="Counties: pct urban land area",
        column="pct_urban_land_area",
        cmap="OrRd",
        tooltip=[
            "municipio",
            "county_geoid",
            "urban_block_count",
            "pct_urban_pop",
            "pct_urban_land_area",
            "median_household_income_usd",
            "poverty_rate",
            "pct_owner_occupied",
            "pct_bachelor_plus",
        ],
        legend=True,
        show=False,
        style_kwds={"weight": 1.0, "color": "#8e24aa", "fillOpacity": 0.3},
    )

    map_obj.save(output_path)
    return output_path


def ensure_census_block_table(
    con: duckdb.DuckDBPyConnection,
    *,
    year: int,
    table_name: str,
    force_refresh: bool = False,
) -> None:
    """Persist a Census block geometry table for a given year when needed."""

    if table_exists(con, table_name) and not force_refresh:
        print(f"[reuse] {table_name}")
        return

    print(f"[fetch] census blocks ({year}) …")
    blocks = fetch_prepared_census_geography(ShapeReader(year=year), "block")
    print(f"        {len(blocks):,} rows")
    print(f"[persist] {table_name} …")
    upsert_geodataframe(con, table_name, blocks)


def register_temp_census_block_table(con: duckdb.DuckDBPyConnection, *, year: int) -> None:
    """Register a temporary Census block table for notebook-only diagnostics."""

    print(f"[fetch] temporary census blocks ({year}) …")
    blocks = fetch_prepared_census_geography(ShapeReader(year=year), "block")
    print(f"        {len(blocks):,} rows")

    staged = pd.DataFrame(blocks.copy())
    staged["geometry"] = blocks.geometry.to_wkb()
    con.register("staged_census_blocks_temp", staged)
    con.execute(f"DROP TABLE IF EXISTS {TEMP_CENSUS_2024_BLOCK_TABLE};")
    con.execute(
        f"""
        CREATE TEMP TABLE {TEMP_CENSUS_2024_BLOCK_TABLE} AS
        SELECT * EXCLUDE (geometry),
               ST_GeomFromWKB(geometry) AS geometry
        FROM staged_census_blocks_temp;
        """
    )
    con.unregister("staged_census_blocks_temp")


def block_comparison_sql(census_2024_table: str = TEMP_CENSUS_2024_BLOCK_TABLE) -> str:
    """Return the SQL relation that compares 2020 and 2024 block geometries."""

    return f"""
        WITH matched AS (
            SELECT
                COALESCE(b20.GEOID, b24.GEOID) AS block_geoid,
                COALESCE(b20.COUNTYFP, b24.COUNTYFP) AS countyfp,
                COALESCE(b20.TRACTCE, b24.TRACTCE) AS tractce,
                COALESCE(b20.BLOCKCE, b24.BLOCKCE) AS blockce,
                b20.GEOID IS NOT NULL AS present_2020,
                b24.GEOID IS NOT NULL AS present_2024,
                CASE
                    WHEN b20.GEOID IS NOT NULL AND b24.GEOID IS NOT NULL
                    THEN ST_Equals(b20.geometry, b24.geometry)
                    ELSE NULL
                END AS geometry_exact_match,
                CASE
                    WHEN b20.GEOID IS NOT NULL AND b24.GEOID IS NOT NULL AND ST_Area(b20.geometry) > 0
                    THEN ST_Area(ST_Intersection(b20.geometry, b24.geometry)) / ST_Area(b20.geometry)
                    ELSE NULL
                END AS overlap_share_2020,
                CASE
                    WHEN b20.GEOID IS NOT NULL AND b24.GEOID IS NOT NULL AND ST_Area(b24.geometry) > 0
                    THEN ST_Area(ST_Intersection(b20.geometry, b24.geometry)) / ST_Area(b24.geometry)
                    ELSE NULL
                END AS overlap_share_2024,
                CASE
                    WHEN b20.GEOID IS NOT NULL AND b24.GEOID IS NOT NULL
                    THEN ST_Distance_Sphere(ST_Centroid(b20.geometry), ST_Centroid(b24.geometry))
                    ELSE NULL
                END AS centroid_distance_m
            FROM {CENSUS_2020_BLOCK_TABLE} AS b20
            FULL OUTER JOIN {census_2024_table} AS b24
              ON b20.GEOID = b24.GEOID
        )
        SELECT *,
               CASE
                   WHEN NOT present_2020 THEN 'new_in_2024'
                   WHEN NOT present_2024 THEN 'missing_in_2024'
                   WHEN geometry_exact_match THEN 'exact_match'
                   WHEN overlap_share_2020 >= {NEAR_MATCH_THRESHOLD} AND overlap_share_2024 >= {NEAR_MATCH_THRESHOLD}
                        THEN 'near_match'
                   ELSE 'changed_geometry'
               END AS comparison_status
        FROM matched
    """


def summarize_block_comparison(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return status, overlap, and changed-example summaries for block comparison."""

    comparison_sql = block_comparison_sql()
    status_summary = con.execute(
        f"""
        WITH comparison AS ({comparison_sql})
        SELECT comparison_status,
               COUNT(*) AS block_count,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 3) AS pct_blocks
        FROM comparison
        GROUP BY comparison_status
        ORDER BY block_count DESC, comparison_status;
        """
    ).fetchdf()

    overlap_summary = con.execute(
        f"""
        WITH comparison AS ({comparison_sql})
        SELECT AVG(overlap_share_2020) AS mean_overlap_share_2020,
               MIN(overlap_share_2020) AS min_overlap_share_2020,
               AVG(overlap_share_2024) AS mean_overlap_share_2024,
               MIN(overlap_share_2024) AS min_overlap_share_2024,
               AVG(centroid_distance_m) AS mean_centroid_distance_m,
               MAX(centroid_distance_m) AS max_centroid_distance_m
        FROM comparison
        WHERE present_2020 AND present_2024;
        """
    ).fetchdf()

    changed_examples = con.execute(
        f"""
        WITH comparison AS ({comparison_sql})
        SELECT block_geoid,
               comparison_status,
               overlap_share_2020,
               overlap_share_2024,
               centroid_distance_m
        FROM comparison
        WHERE comparison_status <> 'exact_match'
        ORDER BY COALESCE(LEAST(overlap_share_2020, overlap_share_2024), -1.0) ASC,
                 centroid_distance_m DESC NULLS LAST,
                 block_geoid
        LIMIT 10;
        """
    ).fetchdf()
    return status_summary, overlap_summary, changed_examples


def summarize_target_block_comparison(
    con: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return municipality-scoped diagnostic summaries for project target areas."""

    comparison_sql = block_comparison_sql()
    muni_sql = ", ".join(f"'{municipio}'" for municipio in TARGET_MUNICIPALITIES)
    target_comparison_sql = f"""
        WITH comparison AS ({comparison_sql}),
        block_geometries AS (
            SELECT cmp.*,
                   CASE
                       WHEN b24.geometry IS NOT NULL THEN b24.geometry
                       ELSE b20.geometry
                   END AS geometry
            FROM comparison AS cmp
            LEFT JOIN {CENSUS_2020_BLOCK_TABLE} AS b20
              ON b20.GEOID = cmp.block_geoid
            LEFT JOIN {TEMP_CENSUS_2024_BLOCK_TABLE} AS b24
              ON b24.GEOID = cmp.block_geoid
        ),
        target_municipios AS (
            SELECT NAME AS municipio, geometry
            FROM pr_census_counties
            WHERE NAME IN ({muni_sql})
        )
        SELECT bg.*, muni.municipio
        FROM block_geometries AS bg
        JOIN target_municipios AS muni
          ON ST_Within(ST_Centroid(bg.geometry), muni.geometry)
    """

    status_summary = con.execute(
        f"""
        WITH target_comparison AS ({target_comparison_sql})
        SELECT municipio,
               comparison_status,
               COUNT(*) AS block_count,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY municipio), 3) AS pct_blocks
        FROM target_comparison
        GROUP BY municipio, comparison_status
        ORDER BY municipio, block_count DESC, comparison_status;
        """
    ).fetchdf()

    overlap_summary = con.execute(
        f"""
        WITH target_comparison AS ({target_comparison_sql})
        SELECT municipio,
               AVG(overlap_share_2020) AS mean_overlap_share_2020,
               MIN(overlap_share_2020) AS min_overlap_share_2020,
               AVG(overlap_share_2024) AS mean_overlap_share_2024,
               MIN(overlap_share_2024) AS min_overlap_share_2024,
               AVG(centroid_distance_m) AS mean_centroid_distance_m,
               MAX(centroid_distance_m) AS max_centroid_distance_m
        FROM target_comparison
        WHERE present_2020 AND present_2024
        GROUP BY municipio
        ORDER BY municipio;
        """
    ).fetchdf()

    changed_examples = con.execute(
        f"""
        WITH target_comparison AS ({target_comparison_sql})
        SELECT municipio,
               block_geoid,
               comparison_status,
               overlap_share_2020,
               overlap_share_2024,
               centroid_distance_m
        FROM target_comparison
        WHERE comparison_status <> 'exact_match'
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY municipio
            ORDER BY COALESCE(LEAST(overlap_share_2020, overlap_share_2024), -1.0) ASC,
                     centroid_distance_m DESC NULLS LAST,
                     block_geoid
        ) <= 10;
        """
    ).fetchdf()
    return status_summary, overlap_summary, changed_examples


def print_vintage_guidance(status_summary: pd.DataFrame) -> None:
    """Print a simple recommendation based on the comparison summary."""

    counts = dict(zip(status_summary["comparison_status"], status_summary["block_count"], strict=False))
    exact_or_near = counts.get("exact_match", 0) + counts.get("near_match", 0)
    changed = counts.get("changed_geometry", 0)
    missing = counts.get("missing_in_2024", 0) + counts.get("new_in_2024", 0)

    if changed == 0 and missing == 0:
        print(
            "Guidance: 2020 and 2024 Puerto Rico block geometries match by GEOID; a 2024 ACS block-group vintage is geometry-safe."
        )
        return
    if changed == 0 and missing > 0:
        print(
            "Guidance: matched blocks align geometrically, but the block inventory changes between 2020 and 2024; review the missing/new GEOIDs before updating the ACS vintage."
        )
        return
    if exact_or_near == 0:
        print(
            "Guidance: block geometries diverge substantially; keep the downstream ACS work pinned until the mismatch is resolved."
        )
        return
    print(
        "Guidance: most blocks may still align, but some geometries change; review the changed examples before switching ACS vintage."
    )

# %% [markdown]
# ## Step 1 — Build the Puerto Rico-only Urban Areas cache
# 
# The durable cache is a filtered parquet artifact instead of the full national
# raw text file. The upstream Census text file is read in chunks and only Puerto
# Rico rows are retained locally.

# %%
urban_block_cache_path = materialize_pr_urban_blocks_cache()
pr_urban_blocks = load_pr_urban_blocks_cache(urban_block_cache_path)
print(f"[cache] loaded {len(pr_urban_blocks):,} Puerto Rico urban-block rows from {urban_block_cache_path}")
print(pr_urban_blocks[["block_geoid", "bg_geoid", "tract_geoid", "county_geoid"]].head(5).to_string(index=False))

# %% [markdown]
# ## Step 2 — Persist the curated 2020 urban-block base table and summary views
# 
# `pr_urban_blocks_2020` is the durable block-level numerator table. The urban
# roll-ups are now views so they stay synchronized with the latest 2020 ACS
# denominators without adding more durable project tables.
# 
# Note: `pct_urban_pop` uses 2020 urban-block population numerators over ACS
# 2020 total-population denominators. Those denominators are estimates, so some
# row-level ratios can exceed 1.0. The summaries below therefore report medians
# plus anomaly counts instead of means.

# %%
db_path = resolve_vector_db_path()
con = create_spatial_connection(db_path)
ensure_required_acs_tables(con)
persist_urban_blocks(con, pr_urban_blocks)
ensure_census_block_table(con, year=2020, table_name=CENSUS_2020_BLOCK_TABLE)
create_urban_summary_views(con)
print(summarize_urban_blocks(con).to_string(index=False))
print("\nUrban summary views:")
print(summarize_urban_views(con).to_string(index=False))
print("\nBlock-group population-ratio anomalies (top examples):")
print(sample_population_ratio_anomalies(con).to_string(index=False))
print(f"\nTarget municipality block-group urban summary ({', '.join(TARGET_MUNICIPALITIES)}):")
print(summarize_target_block_groups(con).to_string(index=False))

# %%
urban_preview_map = build_interactive_urban_preview(con)
print(f"\nSaved interactive all-municipality urban preview map to: {urban_preview_map}")
display(IFrame(src=str(urban_preview_map), width="70%", height=700))

# %% [markdown]
# ## Step 3 — Compare 2020 versus 2024 block geometries without persisting diagnostics
# 
# The 2024 block layer is fetched into a temporary table only for this notebook.
# That keeps the mismatch diagnostics visible here without adding extra durable
# comparison tables to the project database.

# %%
register_temp_census_block_table(con, year=2024)
status_summary, overlap_summary, changed_examples = summarize_block_comparison(con)
target_status_summary, target_overlap_summary, target_changed_examples = summarize_target_block_comparison(con)
print(status_summary.to_string(index=False))
print("\nOverlap summary:")
print(overlap_summary.to_string(index=False))
print("\nChanged examples:")
print(changed_examples.to_string(index=False))
print(f"\nTarget municipality summary ({', '.join(TARGET_MUNICIPALITIES)}):")
print(target_status_summary.to_string(index=False))
print("\nTarget municipality overlap summary:")
print(target_overlap_summary.to_string(index=False))
print("\nTarget municipality changed examples:")
print(target_changed_examples.to_string(index=False))
print_vintage_guidance(status_summary)
print(f"\nSaved DuckDB spatial database to: {db_path}")
print("Created/updated durable table:")
print(f"- {URBAN_BLOCK_TABLE}")
print("Created/updated views:")
for view_name in (BG_URBAN_STATS_VIEW, TRACT_URBAN_STATS_VIEW, COUNTY_URBAN_STATS_VIEW, BG_URBAN_FLAG_VIEW):
    print(f"- {view_name}")
print("Temporary diagnostic relation:")
print(f"- {TEMP_CENSUS_2024_BLOCK_TABLE}")
con.close()


