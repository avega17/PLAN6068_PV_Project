# %% [markdown]
# # Tract-Level PV + NSRDB + ACS Aggregation
#
# Produces tract-level summaries over San Juan + Isabela:
#
# 1. **PV signal:** building counts, OSM-labeled count, model-detected count,
#    and public-facing prevalence metrics per 1,000 buildings.
# 2. **H3 context:** distinct occupied H3 cells represented by the building
#    inventory in each tract.
# 3. **NSRDB context:** tract-level means from the building-attached NSRDB
#    irradiance summaries generated in `13_pv_building_join.py`.
# 4. **ACS sociodemographics:** the newest locally available tract slice,
#    preferring 2024 but falling back to an earlier local vintage when needed,
#    with bilingual working-age language indicators surfaced explicitly.
# 5. **Urban context:** tract-level Census 2020 urban summary metrics from
#    `vw_pr_census_tract_urban_stats`.
# 6. **CDC SVI context:** tract-level Social Vulnerability Index values from
#    the CDC/ATSDR 2020 Puerto Rico SVI release.
#
# Choropleth maps for the two project municipalities are emitted to
# `outputs/maps/`, CSVs to `outputs/figures/`, and an indicator narrative guide
# to `outputs/reports/`.
#
# Refs:
# - https://api.census.gov/data/2020/acs/acs5/variables.html
# - https://svi.cdc.gov/Documents/Data/2020/csv/states/PuertoRico.csv
# - https://www.census.gov/programs-surveys/geography/guidance/geo-areas/urban-rural.html
# - https://www2.census.gov/geo/docs/reference/ua/2020_UA_BLOCKS.txt

# %%
"""02_pv_bg_aggregation.py"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv


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

from utils.acs import artifact_path_for_acs
from utils.acs import table_name_for_acs
from utils.census import create_spatial_connection

FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
MAP_DIR = PROJECT_ROOT / "outputs" / "maps"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
AGG_TABLE = "pr_pv_tract_aggregates"
AGG_GEOID_COLUMN = "tract_geoid"
GEOGRAPHY_TABLE = "pr_census_tracts"
URBAN_STATS_VIEW = "vw_pr_census_tract_urban_stats"
SVI_2020_TRACT_URL = "https://svi.cdc.gov/Documents/Data/2020/csv/states/PuertoRico.csv"
SVI_2020_TRACT_CACHE_PATH = PROJECT_ROOT / "data" / "tabular" / "svi" / "cdc_svi_2020_puerto_rico_tracts.csv"
INDICATOR_GUIDE_OUTPUT_PATH = REPORT_DIR / "pv_tract_indicator_guide.md"

TARGET_MUNICIPALITIES = ("San Juan", "Isabela")
ACS_PREFERRED_VINTAGES = (2024, 2020)
REQUIRED_BUILDING_COLUMNS = (
    "building_id",
    "h3_cell_id",
    "has_pv_osm",
    "has_pv_detected",
    "has_pv_any",
    "pv_detected_count",
    "pv_detected_area_deg2",
    "nsrdb_site_id",
    "nsrdb_multiyear_ghi",
    "nsrdb_multiyear_dni",
    "nsrdb_multiyear_dhi",
    "nsrdb_multiyear_air_temperature",
    "geometry",
)

ACS_VARIABLES: dict[str, str] = {
    "B19013_001E": "median_household_income_usd",
    "B25001_001E": "total_housing_units",
    "B25003_001E": "tenure_total",
    "B25003_002E": "tenure_owner_occupied",
    "B25003_003E": "tenure_renter_occupied",
    "B15003_001E": "edu_total_25plus",
    "B15003_022E": "edu_bachelors",
    "B15003_023E": "edu_masters",
    "B15003_024E": "edu_professional",
    "B15003_025E": "edu_doctorate",
    "B16004_001E": "language_population_5plus",
    "B16004_025E": "english_only_18_64",
    "B16004_026E": "spanish_speaking_18_64",
    "B16004_028E": "spanish_english_well_18_64",
    "B16004_029E": "spanish_english_not_well_18_64",
    "B16004_030E": "spanish_english_not_at_all_18_64",
}

PUBLIC_METRIC_LABELS = {
    "osm_labeled_buildings_per_1000": "OSM-labeled PV buildings per 1,000 buildings",
    "detected_pv_buildings_per_1000": "Model-detected PV buildings per 1,000 buildings",
    "any_pv_evidence_buildings_per_1000": "Buildings with any PV evidence per 1,000 buildings",
    "nsrdb_multiyear_ghi_mean": "NSRDB multiyear GHI mean (W/m2)",
    "occupied_h3_cell_count": "Distinct occupied H3 cells",
    "median_household_income_usd": "ACS median household income (USD)",
    "pct_bachelor_plus": "Share age 25+ with Bachelor's or higher",
    "pct_spanish_english_well_18_64": "Share of Spanish-speaking adults 18-64 who speak English well",
    "pct_spanish_english_not_at_all_18_64": "Share of Spanish-speaking adults 18-64 who speak no English",
    "cdc_svi_2020_overall_percentile": "CDC SVI 2020 overall percentile (0-100)",
    "pct_urban_land_area": "Urban land-area share",
}

MAP_FIGURE_SPECS: tuple[dict[str, str], ...] = (
    {
        "column": "any_pv_evidence_buildings_per_1000",
        "title": "Buildings with any PV evidence per 1,000 buildings",
        "filename": "pv_any_evidence_per_1000_buildings_choropleth.png",
        "cmap": "YlOrBr",
        "what": "Counts buildings tagged either by OSM rooftop PV labels or by the GeoAI detection workflow, standardized by 1,000 buildings in each tract.",
        "why": "This is the broadest rooftop-PV prevalence indicator in the project and is the best first map for non-technical readers.",
    },
    {
        "column": "detected_pv_buildings_per_1000",
        "title": "Model-detected PV buildings per 1,000 buildings",
        "filename": "pv_model_detected_per_1000_buildings_choropleth.png",
        "cmap": "YlOrBr",
        "what": "Counts only buildings flagged by the inference model, standardized by 1,000 buildings in each tract.",
        "why": "Compare this map against the broader any-evidence surface to see where model detections diverge from label-assisted evidence.",
    },
    {
        "column": "nsrdb_multiyear_ghi_mean",
        "title": "NSRDB multiyear GHI mean (W/m2)",
        "filename": "nsrdb_ghi_choropleth.png",
        "cmap": "cividis",
        "what": "A tract-average rooftop irradiance context layer derived from the nearest NSRDB 30-minute site summaries attached to buildings.",
        "why": "Higher values indicate stronger long-run solar resource potential, but not necessarily more rooftop PV adoption.",
    },
    {
        "column": "occupied_h3_cell_count",
        "title": "Distinct occupied H3 cells",
        "filename": "occupied_h3_cell_count_choropleth.png",
        "cmap": "YlGnBu",
        "what": "Counts distinct H3 cells containing buildings within each tract.",
        "why": "This is a simple spatial-coverage diagnostic: larger counts often indicate tracts with more distributed building footprints and inference opportunities.",
    },
    {
        "column": "median_household_income_usd",
        "title": "ACS median household income (USD)",
        "filename": "acs_income_choropleth.png",
        "cmap": "viridis",
        "what": "Median household income from the locally available ACS tract slice used in the project pipeline.",
        "why": "Income remains a core planning covariate for testing whether rooftop-PV evidence clusters with higher-earning neighborhoods.",
    },
    {
        "column": "pct_bachelor_plus",
        "title": "Share age 25+ with Bachelor's or higher",
        "filename": "acs_education_choropleth.png",
        "cmap": "BuGn",
        "what": "Share of adults age 25+ with a bachelor's degree or higher.",
        "why": "Educational attainment is still a useful socioeconomic context layer for rooftop-PV adoption analysis.",
    },
    {
        "column": "pct_spanish_english_well_18_64",
        "title": "Share of Spanish-speaking adults 18-64 who speak English well",
        "filename": "acs_spanish_english_well_18_64_choropleth.png",
        "cmap": "BuPu",
        "what": "Among Spanish-speaking adults age 18-64, the share reporting that they speak English well.",
        "why": "This is the positive bilingual-capacity measure requested for the planning narrative because it can proxy access to higher-paying jobs and service networks.",
    },
    {
        "column": "pct_spanish_english_not_at_all_18_64",
        "title": "Share of Spanish-speaking adults 18-64 who speak no English",
        "filename": "acs_spanish_english_not_at_all_18_64_choropleth.png",
        "cmap": "OrRd",
        "what": "Among Spanish-speaking adults age 18-64, the share reporting that they do not speak English at all.",
        "why": "This is the opposite bilingual-capacity extreme and helps interpret where language access barriers may align with lower rooftop-PV uptake.",
    },
    {
        "column": "cdc_svi_2020_overall_percentile",
        "title": "CDC SVI 2020 overall percentile (0-100)",
        "filename": "cdc_svi_2020_overall_percentile_choropleth.png",
        "cmap": "magma",
        "what": "The CDC/ATSDR Social Vulnerability Index overall percentile for 2020 Puerto Rico tracts, scaled from 0 to 100.",
        "why": "This replaces the earlier diversity-index map with an established public-health vulnerability measure that is much more interpretable in the Puerto Rico planning context.",
    },
    {
        "column": "pct_urban_land_area",
        "title": "Urban land-area share",
        "filename": "urban_land_share_choropleth.png",
        "cmap": "PuBuGn",
        "what": "Share of tract land area classified as urban by the Census 2020 urban-area overlay workflow.",
        "why": "This is more informative than a binary urban/rural map because all target tracts are urban to some extent; the gradient still reveals how built-up each tract is.",
    },
)


def resolve_db_path() -> Path:
    v = os.getenv("VECTOR_DB")
    if v:
        p = Path(v)
        if not p.is_absolute():
            p = PROJECT_ROOT / p if len(p.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / p
        return p
    return PROJECT_ROOT / "data" / "PR_PV_plan_data.duckdb"


def _to_bytes(value: object) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    return bytes(value)


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, pd.NA)


def assert_building_table_contract(con: duckdb.DuckDBPyConnection) -> None:
    available_columns = {
        row[0]
        for row in con.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'pr_buildings_with_pv'
            ORDER BY ordinal_position;
            """
        ).fetchall()
    }
    missing_columns = [column_name for column_name in REQUIRED_BUILDING_COLUMNS if column_name not in available_columns]
    if missing_columns:
        raise RuntimeError(
            "pr_buildings_with_pv is missing the updated notebook 13 columns: "
            + ", ".join(missing_columns)
            + ". Re-run notebooks/tabular/13_pv_building_join.py before notebook 14."
        )


# %%
def build_tract_pv_nsrdb_aggregates(con: duckdb.DuckDBPyConnection) -> None:
    assert_building_table_contract(con)
    munis_sql = ", ".join(f"'{m}'" for m in TARGET_MUNICIPALITIES)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {AGG_TABLE} AS
        WITH muni AS (
            SELECT GEOID AS municipio_geoid, NAME AS municipio, geometry AS muni_geom
            FROM pr_census_counties
            WHERE NAME IN ({munis_sql})
        ),
        tract_in_muni AS (
            SELECT tract.GEOID AS tract_geoid,
                   tract.geometry AS tract_geom,
                   m.municipio,
                   ROW_NUMBER() OVER (
                     PARTITION BY tract.GEOID
                     ORDER BY ST_Area(ST_Intersection(tract.geometry, m.muni_geom)) DESC
                   ) AS rn
            FROM {GEOGRAPHY_TABLE} AS tract
            JOIN muni AS m
              ON ST_Intersects(tract.geometry, m.muni_geom)
            WHERE ST_Area(ST_Intersection(tract.geometry, m.muni_geom))
                    / NULLIF(ST_Area(tract.geometry), 0) > 0.5
        ),
        bld_tract AS (
            SELECT
                tim.tract_geoid,
                tim.municipio,
                b.building_id,
                b.h3_cell_id,
                b.has_pv_osm,
                b.has_pv_detected,
                b.has_pv_any,
                b.pv_detected_count,
                b.pv_detected_area_deg2,
                b.nsrdb_site_id,
                b.nsrdb_multiyear_ghi,
                b.nsrdb_multiyear_dni,
                b.nsrdb_multiyear_dhi,
                b.nsrdb_multiyear_air_temperature
            FROM pr_buildings_with_pv AS b
            JOIN tract_in_muni AS tim
              ON tim.rn = 1 AND ST_Within(ST_Centroid(b.geometry), tim.tract_geom)
        ),
        pv_tract AS (
            SELECT
                tract_geoid,
                ANY_VALUE(municipio) AS municipio,
                COUNT(*) AS building_count,
                COUNT(DISTINCT h3_cell_id) FILTER (WHERE h3_cell_id IS NOT NULL) AS occupied_h3_cell_count,
                COUNT(DISTINCT nsrdb_site_id) FILTER (WHERE nsrdb_site_id IS NOT NULL) AS nsrdb_site_count,
                SUM(CAST(has_pv_osm AS INT)) AS osm_pv_count,
                SUM(CAST(has_pv_detected AS INT)) AS detected_pv_count,
                SUM(CAST(has_pv_any AS INT)) AS any_pv_signal_count,
                SUM(CAST(has_pv_osm AND has_pv_detected AS INT)) AS overlap_count,
                SUM(pv_detected_area_deg2) AS total_detected_area_deg2,
                CASE WHEN COUNT(*) = 0 THEN 0.0
                     ELSE SUM(CAST(has_pv_osm AS INT))::DOUBLE / COUNT(*)
                END AS osm_pv_rate,
                 CASE WHEN COUNT(*) = 0 THEN 0.0
                     ELSE 1000.0 * SUM(CAST(has_pv_osm AS INT))::DOUBLE / COUNT(*)
                 END AS osm_labeled_buildings_per_1000,
                CASE WHEN COUNT(*) = 0 THEN 0.0
                     ELSE SUM(CAST(has_pv_detected AS INT))::DOUBLE / COUNT(*)
                END AS detected_pv_rate,
                 CASE WHEN COUNT(*) = 0 THEN 0.0
                     ELSE 1000.0 * SUM(CAST(has_pv_detected AS INT))::DOUBLE / COUNT(*)
                 END AS detected_pv_buildings_per_1000,
                CASE WHEN COUNT(*) = 0 THEN 0.0
                     ELSE SUM(CAST(has_pv_any AS INT))::DOUBLE / COUNT(*)
                END AS any_pv_signal_rate,
                 CASE WHEN COUNT(*) = 0 THEN 0.0
                     ELSE 1000.0 * SUM(CAST(has_pv_any AS INT))::DOUBLE / COUNT(*)
                 END AS any_pv_evidence_buildings_per_1000,
                AVG(nsrdb_multiyear_ghi) AS nsrdb_multiyear_ghi_mean,
                AVG(nsrdb_multiyear_dni) AS nsrdb_multiyear_dni_mean,
                AVG(nsrdb_multiyear_dhi) AS nsrdb_multiyear_dhi_mean,
                AVG(nsrdb_multiyear_air_temperature) AS nsrdb_multiyear_air_temperature_mean
            FROM bld_tract
            GROUP BY tract_geoid
        )
        SELECT *
        FROM pv_tract;
        """
    )


# %%
def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    return bool(
        con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?;
            """,
            [table_name],
        ).fetchone()[0]
    )


def finalize_local_acs_metrics(frame: pd.DataFrame, geoid_column: str) -> pd.DataFrame:
    out = frame.copy()
    required_language_columns = {
        "language_population_5plus",
        "spanish_speaking_18_64",
        "spanish_english_well_18_64",
        "spanish_english_not_at_all_18_64",
    }
    missing_language_columns = sorted(required_language_columns - set(out.columns))
    if missing_language_columns:
        raise RuntimeError(
            "ACS tract slice is missing the bilingual columns required for notebook 14: "
            + ", ".join(missing_language_columns)
            + ". Re-run notebooks/tabular/04_acs_5year_ingest.py with the B16004 variables enabled."
        )

    if "pct_owner_occupied" not in out.columns:
        out["pct_owner_occupied"] = safe_ratio(out["tenure_owner_occupied"], out["tenure_total"])
    if "pct_bachelor_plus" not in out.columns:
        out["pct_bachelor_plus"] = safe_ratio(
            out[["edu_bachelors", "edu_masters", "edu_professional", "edu_doctorate"]].sum(axis=1),
            out["edu_total_25plus"],
        )

    out["pct_spanish_speaking_18_64"] = safe_ratio(out["spanish_speaking_18_64"], out["language_population_5plus"])
    out["pct_spanish_english_well_18_64"] = safe_ratio(out["spanish_english_well_18_64"], out["spanish_speaking_18_64"])
    if "spanish_english_not_well_18_64" in out.columns:
        out["pct_spanish_english_not_well_18_64"] = safe_ratio(out["spanish_english_not_well_18_64"], out["spanish_speaking_18_64"])
    out["pct_spanish_english_not_at_all_18_64"] = safe_ratio(out["spanish_english_not_at_all_18_64"], out["spanish_speaking_18_64"])

    keep = [
        geoid_column,
        *ACS_VARIABLES.values(),
        "pct_owner_occupied",
        "pct_bachelor_plus",
        "pct_spanish_speaking_18_64",
        "pct_spanish_english_well_18_64",
        "pct_spanish_english_not_well_18_64",
        "pct_spanish_english_not_at_all_18_64",
    ]
    return out[[column for column in keep if column in out.columns]].copy()


def load_cdc_svi_2020_tracts() -> pd.DataFrame:
    cache_path = SVI_2020_TRACT_CACHE_PATH
    if cache_path.exists():
        frame = pd.read_csv(cache_path, dtype={"FIPS": str})
    else:
        frame = pd.read_csv(SVI_2020_TRACT_URL, dtype={"FIPS": str})
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index=False)

    if frame.empty:
        raise RuntimeError("CDC SVI 2020 tract file loaded but returned no rows.")

    frame[AGG_GEOID_COLUMN] = frame["FIPS"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(11)
    output = frame[[AGG_GEOID_COLUMN, "LOCATION", "RPL_THEMES"]].copy()
    output = output.rename(columns={"LOCATION": "cdc_svi_2020_location"})
    svi_percentile = pd.to_numeric(output["RPL_THEMES"], errors="coerce")
    svi_percentile = svi_percentile.mask(svi_percentile < 0)
    output["cdc_svi_2020_overall_percentile"] = svi_percentile * 100.0
    return output[[AGG_GEOID_COLUMN, "cdc_svi_2020_location", "cdc_svi_2020_overall_percentile"]].copy()


def load_local_acs_tracts(con: duckdb.DuckDBPyConnection) -> tuple[int, pd.DataFrame]:
    candidate_years: list[int] = []
    for year in ACS_PREFERRED_VINTAGES:
        if year not in candidate_years:
            candidate_years.append(year)

    for row in con.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name LIKE 'pr_acs_%_tracts'
        ORDER BY table_name DESC;
        """
    ).fetchall():
        parts = str(row[0]).split("_")
        if len(parts) >= 4 and parts[2].isdigit():
            year = int(parts[2])
            if year not in candidate_years:
                candidate_years.append(year)

    for year in candidate_years:
        table_name = table_name_for_acs(year, "tract")
        if table_exists(con, table_name):
            df = con.execute(f"SELECT * FROM {table_name};").fetchdf()
            return year, finalize_local_acs_metrics(df, geoid_column=AGG_GEOID_COLUMN)

        artifact_path = artifact_path_for_acs(year, "tract")
        if artifact_path.exists():
            return year, finalize_local_acs_metrics(pd.read_parquet(artifact_path), geoid_column=AGG_GEOID_COLUMN)

    raise RuntimeError(
        "Local ACS tract data not found. Run notebooks/tabular/04_acs_5year_ingest.py for tracts."
    )


def assert_target_acs_coverage(con: duckdb.DuckDBPyConnection, acs_relation_name: str) -> pd.DataFrame:
    coverage = con.execute(
        f"""
        SELECT agg.municipio,
               COUNT(*) AS tract_rows,
               SUM(CASE WHEN acs.{AGG_GEOID_COLUMN} IS NOT NULL THEN 1 ELSE 0 END) AS matched_acs_rows,
               SUM(CASE WHEN acs.{AGG_GEOID_COLUMN} IS NULL THEN 1 ELSE 0 END) AS missing_acs_rows
        FROM {AGG_TABLE} AS agg
        LEFT JOIN {acs_relation_name} AS acs
          ON acs.{AGG_GEOID_COLUMN} = agg.{AGG_GEOID_COLUMN}
        GROUP BY agg.municipio
        ORDER BY agg.municipio;
        """
    ).fetchdf()
    print(coverage.to_string(index=False))

    found = set(coverage["municipio"].tolist())
    missing_municipios = [municipio for municipio in TARGET_MUNICIPALITIES if municipio not in found]
    if missing_municipios:
        raise RuntimeError(f"Missing tract aggregate rows for target municipalities: {missing_municipios}")

    uncovered = coverage[coverage["missing_acs_rows"] > 0]
    if not uncovered.empty:
        raise RuntimeError(
            "Local ACS join left uncovered target tracts:\n"
            f"{uncovered.to_string(index=False)}"
        )
    return coverage


def assert_target_svi_coverage(con: duckdb.DuckDBPyConnection, svi_relation_name: str) -> pd.DataFrame:
    coverage = con.execute(
        f"""
        SELECT agg.municipio,
               COUNT(*) AS tract_rows,
               SUM(CASE WHEN svi.{AGG_GEOID_COLUMN} IS NOT NULL THEN 1 ELSE 0 END) AS matched_svi_rows,
               SUM(CASE WHEN svi.{AGG_GEOID_COLUMN} IS NULL THEN 1 ELSE 0 END) AS missing_svi_rows
        FROM {AGG_TABLE} AS agg
        LEFT JOIN {svi_relation_name} AS svi
          ON svi.{AGG_GEOID_COLUMN} = agg.{AGG_GEOID_COLUMN}
        GROUP BY agg.municipio
        ORDER BY agg.municipio;
        """
    ).fetchdf()
    print(coverage.to_string(index=False))
    uncovered = coverage[coverage["missing_svi_rows"] > 0]
    if not uncovered.empty:
        raise RuntimeError(
            "CDC SVI 2020 join left uncovered target tracts:\n"
            f"{uncovered.to_string(index=False)}"
        )
    return coverage


# %%
def load_urban_flags(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Return tract-level urban context; empty if the tract summary view is absent."""
    has_table = con.execute(
        f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{URBAN_STATS_VIEW}';"
    ).fetchone()[0]
    if not has_table:
        print("note: tract urban summary view not found — run notebooks/tabular/05_urban_blocks_2020_ingest.py")
        return pd.DataFrame(
            columns=[
                AGG_GEOID_COLUMN,
                "urban_block_count",
                "total_urban_pop",
                "total_urban_housing_units",
                "urban_land_area_m2",
                "pct_urban_pop",
                "pct_urban_population",
                "pct_urban_housing_units",
                "pct_urban_land_area",
                "is_urban",
            ]
        )
    return con.execute(
        f"""
        SELECT {AGG_GEOID_COLUMN},
               urban_block_count,
               total_urban_pop,
               total_urban_housing_units,
               urban_land_area_m2,
               pct_urban_pop,
               pct_urban_population,
               pct_urban_housing_units,
               pct_urban_land_area,
               is_urban
        FROM {URBAN_STATS_VIEW};
        """
    ).fetchdf()


# %%
def load_tract_geometries(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    df = con.execute(
        f"""
        SELECT a.*, ST_AsWKB(tract.geometry) AS wkb
        FROM {AGG_TABLE} AS a
        JOIN {GEOGRAPHY_TABLE} AS tract ON tract.GEOID = a.{AGG_GEOID_COLUMN};
        """
    ).fetchdf()
    if df.empty:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")
    from shapely import from_wkb

    geoms = gpd.GeoSeries(
        df["wkb"].map(lambda v: from_wkb(_to_bytes(v))),
        crs="EPSG:4326",
    )
    return gpd.GeoDataFrame(df.drop(columns=["wkb"]), geometry=geoms, crs="EPSG:4326")


def plot_choropleth(gdf: gpd.GeoDataFrame, column: str, title: str, out_path: Path, cmap: str = "viridis") -> None:
    if gdf.empty or gdf[column].dropna().empty:
        print(f"skipping {title}: no data for {column}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, muni in zip(axes, TARGET_MUNICIPALITIES):
        subset = gdf[gdf["municipio"] == muni]
        if subset.empty:
            ax.set_title(f"{muni}: no data")
            ax.set_axis_off()
            continue
        value_count = int(subset[column].dropna().nunique())
        plot_kwargs = {
            "column": column,
            "ax": ax,
            "cmap": cmap,
            "legend": True,
            "edgecolor": "#555",
            "linewidth": 0.2,
            "missing_kwds": {"color": "lightgrey", "label": "no data"},
        }
        if value_count >= 2:
            plot_kwargs["scheme"] = "quantiles"
            plot_kwargs["k"] = min(5, value_count)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Not enough unique values in array to form",
                category=UserWarning,
            )
            subset.plot(**plot_kwargs)
        ax.set_title(f"{muni} — {title}")
        ax.set_axis_off()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_morans_i_results(frame: pd.DataFrame, out_path: Path) -> None:
    if frame.empty:
        print("skipping Moran's I figure: no results")
        return

    plot_df = frame.copy()
    plot_df["metric_label"] = plot_df["metric"].map(lambda value: PUBLIC_METRIC_LABELS.get(value, str(value).replace("_", " ")))
    plot_df["label"] = plot_df["municipio"] + " — " + plot_df["metric_label"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(plot_df["label"], plot_df["morans_I"], color=["#9c6644", "#dda15e", "#6b8f71", "#3d405b"][: len(plot_df)])
    ax.axhline(0.0, color="#444", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Moran's I")
    ax.set_title("Initial Moran's I diagnostics by municipality and PV metric")
    ax.tick_params(axis="x", rotation=20)
    for bar, row in zip(bars, plot_df.itertuples(index=False), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"p={row.p_sim:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _format_metric_range(gdf: gpd.GeoDataFrame, column: str) -> str:
    parts: list[str] = []
    for municipio in TARGET_MUNICIPALITIES:
        series = pd.to_numeric(gdf.loc[gdf["municipio"] == municipio, column], errors="coerce").dropna()
        if series.empty:
            continue
        parts.append(f"{municipio}: {series.min():.2f} to {series.max():.2f}")
    return "; ".join(parts) if parts else "No values available."


def write_indicator_guide(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    lines = [
        "# Tract Indicator Guide",
        "",
        "This note explains the tract maps and Moran diagnostics exported by `14_pv_bg_aggregation.py`.",
        "Public-facing PV indicators are scaled per 1,000 buildings so readers do not have to interpret very small decimals.",
        "",
    ]

    for spec in MAP_FIGURE_SPECS:
        column = spec["column"]
        if column not in gdf.columns or gdf[column].dropna().empty:
            continue
        lines.extend(
            [
                f"## {spec['title']}",
                "",
                f"- Output file: `outputs/maps/{spec['filename']}`",
                f"- What it measures: {spec['what']}",
                f"- How to read it: {spec['why']}",
                f"- Observed range across case-study tracts: {_format_metric_range(gdf, column)}",
                "",
            ]
        )

    lines.extend(
        [
            "## Moran's I diagnostics",
            "",
            "The Moran's I figure summarizes whether neighboring tracts have similar PV prevalence values.",
            "Positive Moran's I means similar values cluster together in space; values near zero suggest little spatial structure; negative values indicate local contrast.",
            "The `p` labels on each bar are permutation-test probabilities, so smaller values indicate stronger evidence that the spatial pattern is not random.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output_path}")


# %%
# Notebook driver: connect, aggregate, merge ACS, render choropleths.
if __name__ == "__main__":
    db_path = resolve_db_path()
    con = create_spatial_connection(db_path)

    print("[1/7] building tract PV + NSRDB aggregates …")
    build_tract_pv_nsrdb_aggregates(con)

    print("[2/7] loading local ACS tract data …")
    acs_year, acs_df = load_local_acs_tracts(con)
    print(f"      loaded {len(acs_df):,} tract rows from local storage (ACS {acs_year})")
    con.register("acs_tract", acs_df)

    print("[3/7] loading CDC SVI 2020 tract context …")
    svi_df = load_cdc_svi_2020_tracts()
    print(f"      loaded {len(svi_df):,} SVI tract rows from CDC 2020 source/cache")
    con.register("svi_tract", svi_df)

    print("[4/7] validating ACS/SVI coverage and joining tract urban summary context …")
    assert_target_acs_coverage(con, "acs_tract")
    assert_target_svi_coverage(con, "svi_tract")
    urban_df = load_urban_flags(con)
    if urban_df.empty:
        urban_df = pd.DataFrame(
            {
                AGG_GEOID_COLUMN: [],
                "urban_block_count": [],
                "total_urban_pop": [],
                "total_urban_housing_units": [],
                "urban_land_area_m2": [],
                "pct_urban_pop": [],
                "pct_urban_population": [],
                "pct_urban_housing_units": [],
                "pct_urban_land_area": [],
                "is_urban": [],
            }
        )
    con.register("urban_bg", urban_df)
    con.execute(
        f"""
         CREATE OR REPLACE TABLE {AGG_TABLE} AS
        SELECT a.*,
                             {acs_year} AS acs_vintage,
                             acs.* EXCLUDE ({AGG_GEOID_COLUMN}),
                             svi.cdc_svi_2020_location,
                             svi.cdc_svi_2020_overall_percentile,
               COALESCE(u.urban_block_count, 0) AS urban_block_count,
               COALESCE(u.total_urban_pop, 0) AS total_urban_pop,
               COALESCE(u.total_urban_housing_units, 0) AS total_urban_housing_units,
               COALESCE(u.urban_land_area_m2, 0) AS urban_land_area_m2,
               u.pct_urban_pop,
               u.pct_urban_population,
               u.pct_urban_housing_units,
               u.pct_urban_land_area,
               COALESCE(u.is_urban, FALSE) AS is_urban
        FROM {AGG_TABLE} AS a
        LEFT JOIN acs_tract AS acs ON acs.{AGG_GEOID_COLUMN} = a.{AGG_GEOID_COLUMN}
        LEFT JOIN svi_tract AS svi ON svi.{AGG_GEOID_COLUMN} = a.{AGG_GEOID_COLUMN}
        LEFT JOIN urban_bg AS u ON u.{AGG_GEOID_COLUMN} = a.{AGG_GEOID_COLUMN};
        """
    )
    con.unregister("acs_tract")
    con.unregister("svi_tract")
    con.unregister("urban_bg")

    # %%
    print("[5/7] exporting aggregate CSV …")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    csv_out = FIG_DIR / "pv_tract_aggregates_sj_isabela.csv"
    con.execute(f"COPY {AGG_TABLE} TO '{csv_out}' (HEADER, DELIMITER ',');")
    print(f"      wrote {csv_out}")

    # %%
    print("[6/7] building choropleths for San Juan + Isabela …")
    tract_gdf = load_tract_geometries(con)
    for spec in MAP_FIGURE_SPECS:
        column = spec["column"]
        if column not in tract_gdf.columns or tract_gdf[column].dropna().empty:
            continue
        plot_choropleth(
            tract_gdf,
            column,
            spec["title"],
            MAP_DIR / spec["filename"],
            cmap=spec["cmap"],
        )
    write_indicator_guide(tract_gdf, INDICATOR_GUIDE_OUTPUT_PATH)

    # %%
    print("[7/7] Moran's I on tract PV prevalence metrics …")
    try:
        from libpysal.weights import Queen
        from esda.moran import Moran

        mo_rows = []
        for metric_name in ("any_pv_evidence_buildings_per_1000", "detected_pv_buildings_per_1000"):
            for muni in TARGET_MUNICIPALITIES:
                subset = tract_gdf[(tract_gdf["municipio"] == muni) & tract_gdf[metric_name].notna()].copy()
                if len(subset) < 5:
                    continue
                w = Queen.from_dataframe(subset, use_index=False)
                w.transform = "r"
                moran = Moran(subset[metric_name].values, w, permutations=999)
                mo_rows.append({
                    "metric": metric_name,
                    "municipio": muni,
                    "n_tracts": len(subset),
                    "morans_I": moran.I,
                    "p_sim": moran.p_sim,
                    "z_sim": moran.z_sim,
                })
        if mo_rows:
            mo_df = pd.DataFrame(mo_rows)
            mo_out = FIG_DIR / "pv_tract_morans_i.csv"
            mo_df.to_csv(mo_out, index=False)
            print(mo_df.to_string(index=False))
            print(f"      wrote {mo_out}")
            plot_morans_i_results(mo_df, FIG_DIR / "pv_tract_morans_i.png")
    except Exception as exc:
        print(f"Moran's I step skipped: {exc}")

    con.close()
