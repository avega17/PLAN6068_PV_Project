# %% [markdown]
# # Block-Group-Level PV + Solar Flux + ACS Aggregation
#
# Produces BG-level summaries over San Juan + Isabela:
#
# 1. **PV signal:** building counts, OSM-labeled count, model-detected count,
#    detection rate per housing unit.
# 2. **Solar flux:** mean / p10 / p90 annual flux across all building rooftops
#    in each BG.
# 3. **ACS 2016-2020 5-yr sociodemographics** via `censusdis` (state FIPS `72`):
#    median household income, education, race×Hispanic (table B03002),
#    tenure, housing units, and a **derived Diversity Index**.
# 4. **Urban/rural flag** per BG derived from the Census 2020 Urban Areas
#    block list (`pr_bg_urban_flags`).
#
# Vintage is pinned to **2020** so the ACS 5-yr estimates are co-temporal with
# the 2020 Urban Areas product and the 2020 decennial geographies.
#
# The Diversity Index follows the Census 2020 methodology: the probability
# that two people drawn at random are of different race/ethnicity groups,
#
#   DI = 1 - sum_i p_i^2
#
# over the seven mutually-exclusive non-Hispanic race groups (B03002_003..009)
# plus Hispanic/Latino of any race (B03002_012).
#
# Choropleth maps for the two project municipalities are emitted to
# `outputs/maps/`, CSVs to `outputs/figures/`.
#
# Refs:
# - https://api.census.gov/data/2020/acs/acs5/variables.html
# - https://www.census.gov/library/visualizations/interactive/racial-and-ethnic-diversity-in-the-united-states-2010-and-2020-census.html
# - https://www.census.gov/programs-surveys/geography/guidance/geo-areas/urban-rural.html
# - https://www2.census.gov/geo/docs/reference/ua/2020_UA_BLOCKS.txt
# - https://censusdis.readthedocs.io/en/1.1.3/nb/nationwide%20diversity%20and%20integration.html

# %%
"""02_pv_bg_aggregation.py"""

from __future__ import annotations

import os
import sys
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

FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
MAP_DIR = PROJECT_ROOT / "outputs" / "maps"
BG_AGG_TABLE = "pr_pv_bg_aggregates"

TARGET_MUNICIPALITIES = ("San Juan", "Isabela")
PR_STATE_FIPS = "72"
ACS_VINTAGE = 2020  # ACS 2016-2020 5-yr — pinned to the 2020 decennial reference geography
ACS_DATASET = "acs/acs5"

# ACS 2016-2020 5-yr variable set.  Race/Hispanic uses B03002 so we can
# compute a Census-2020-style Diversity Index from mutually-exclusive groups.
ACS_VARIABLES: dict[str, str] = {
    "B19013_001E": "median_household_income_usd",
    "B25001_001E": "total_housing_units",
    "B25003_001E": "tenure_total",
    "B25003_002E": "tenure_owner_occupied",
    "B25003_003E": "tenure_renter_occupied",
    # B03002 = Hispanic or Latino origin by race (mutually exclusive groups).
    "B03002_001E": "race_total",
    "B03002_003E": "nh_white_alone",
    "B03002_004E": "nh_black_alone",
    "B03002_005E": "nh_aian_alone",
    "B03002_006E": "nh_asian_alone",
    "B03002_007E": "nh_nhpi_alone",
    "B03002_008E": "nh_other_alone",
    "B03002_009E": "nh_two_or_more",
    "B03002_012E": "hispanic_or_latino",
    "B15003_001E": "edu_total_25plus",
    "B15003_022E": "edu_bachelors",
    "B15003_023E": "edu_masters",
    "B15003_024E": "edu_professional",
    "B15003_025E": "edu_doctorate",
}

# Groups used for the Diversity Index (sum of squared shares over these 8
# mutually-exclusive race/ethnicity groups).
DI_GROUP_COLS = (
    "nh_white_alone",
    "nh_black_alone",
    "nh_aian_alone",
    "nh_asian_alone",
    "nh_nhpi_alone",
    "nh_other_alone",
    "nh_two_or_more",
    "hispanic_or_latino",
)


def resolve_db_path() -> Path:
    v = os.getenv("VECTOR_DB")
    if v:
        p = Path(v)
        if not p.is_absolute():
            p = PROJECT_ROOT / p if len(p.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / p
        return p
    return PROJECT_ROOT / "data" / "vectors" / "PR_vector_data.duckdb"


# %%
def build_bg_pv_flux_aggregates(con: duckdb.DuckDBPyConnection) -> None:
    munis_sql = ", ".join(f"'{m}'" for m in TARGET_MUNICIPALITIES)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {BG_AGG_TABLE} AS
        WITH muni AS (
            SELECT GEOID AS municipio_geoid, NAME AS municipio, geometry AS muni_geom
            FROM pr_municipalities
            WHERE NAME IN ({munis_sql})
        ),
        bg_in_muni AS (
            SELECT bg.GEOID AS bg_geoid,
                   bg.geometry AS bg_geom,
                   m.municipio,
                   ROW_NUMBER() OVER (
                     PARTITION BY bg.GEOID
                     ORDER BY ST_Area(ST_Intersection(bg.geometry, m.muni_geom)) DESC
                   ) AS rn
            FROM pr_block_groups AS bg
            JOIN muni AS m
              ON ST_Intersects(bg.geometry, m.muni_geom)
            WHERE ST_Area(ST_Intersection(bg.geometry, m.muni_geom))
                    / NULLIF(ST_Area(bg.geometry), 0) > 0.5
        ),
        bld_bg AS (
            SELECT
                bim.bg_geoid,
                bim.municipio,
                b.building_id,
                b.has_pv_osm,
                b.has_pv_detected,
                b.pv_detected_count,
                b.pv_detected_area_deg2,
                b.annual_flux_mean_kwh_per_kw_yr,
                b.annual_flux_pixel_count
            FROM pr_buildings_with_pv AS b
            JOIN bg_in_muni AS bim
              ON bim.rn = 1 AND ST_Within(ST_Centroid(b.geometry), bim.bg_geom)
        )
        SELECT
            bg_geoid,
            ANY_VALUE(municipio) AS municipio,
            COUNT(*) AS building_count,
            SUM(CAST(has_pv_osm AS INT)) AS osm_pv_count,
            SUM(CAST(has_pv_detected AS INT)) AS detected_pv_count,
            SUM(CAST(has_pv_osm AND has_pv_detected AS INT)) AS overlap_count,
            SUM(pv_detected_area_deg2) AS total_detected_area_deg2,
            CASE WHEN COUNT(*) = 0 THEN 0.0
                 ELSE SUM(CAST(has_pv_detected AS INT))::DOUBLE / COUNT(*)
            END AS detected_pv_rate,
            -- Pixel-weighted BG flux mean.
            CASE WHEN SUM(COALESCE(annual_flux_pixel_count, 0)) = 0 THEN NULL
                 ELSE SUM(annual_flux_mean_kwh_per_kw_yr * annual_flux_pixel_count)
                      / SUM(annual_flux_pixel_count)
            END AS annual_flux_mean_kwh_per_kw_yr,
            SUM(COALESCE(annual_flux_pixel_count, 0)) AS flux_pixel_count
        FROM bld_bg
        GROUP BY bg_geoid;
        """
    )


# %%
def fetch_acs_block_groups() -> pd.DataFrame:
    """Pull ACS BG-level variables for Puerto Rico via censusdis."""

    import censusdis.data as ced
    from censusdis.datasets import ACS5  # noqa: F401  (dataset registry import)

    variables = list(ACS_VARIABLES.keys())
    df = ced.download(
        dataset=ACS_DATASET,
        vintage=ACS_VINTAGE,
        download_variables=variables,
        state=PR_STATE_FIPS,
        county="*",
        tract="*",
        block_group="*",
        api_key=os.getenv("CENSUS_API_KEY"),
    )
    df = df.rename(columns=ACS_VARIABLES)
    df["bg_geoid"] = (
        df["STATE"].astype(str).str.zfill(2)
        + df["COUNTY"].astype(str).str.zfill(3)
        + df["TRACT"].astype(str).str.zfill(6)
        + df["BLOCK_GROUP"].astype(str)
    )
    # Derived shares.
    df["pct_owner_occupied"] = df["tenure_owner_occupied"] / df["tenure_total"].replace(0, pd.NA)
    df["pct_bachelor_plus"] = (
        df[["edu_bachelors", "edu_masters", "edu_professional", "edu_doctorate"]].sum(axis=1)
        / df["edu_total_25plus"].replace(0, pd.NA)
    )
    # Census 2020 Diversity Index from B03002 shares (mutually exclusive groups).
    total = df["race_total"].replace(0, pd.NA)
    sum_sq = sum((df[col] / total) ** 2 for col in DI_GROUP_COLS)
    df["diversity_index"] = 1 - sum_sq
    keep = [
        "bg_geoid",
        *ACS_VARIABLES.values(),
        "pct_owner_occupied",
        "pct_bachelor_plus",
        "diversity_index",
    ]
    return df[keep].copy()


# %%
def load_urban_flags(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Return BG-level urban flag from `pr_bg_urban_flags`; empty if absent."""
    has_table = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'pr_bg_urban_flags';"
    ).fetchone()[0]
    if not has_table:
        print("note: pr_bg_urban_flags not found — run notebooks/vectors/05_urban_blocks_2020_ingest.py")
        return pd.DataFrame(columns=["bg_geoid", "urban_block_count", "is_urban"])
    return con.execute(
        "SELECT bg_geoid, urban_block_count, is_urban FROM pr_bg_urban_flags;"
    ).fetchdf()


# %%
def load_bg_geometries(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    df = con.execute(
        f"""
        SELECT a.*, ST_AsWKB(bg.geometry) AS wkb
        FROM {BG_AGG_TABLE} AS a
        JOIN pr_block_groups AS bg ON bg.GEOID = a.bg_geoid;
        """
    ).fetchdf()
    if df.empty:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")
    from shapely import from_wkb

    geoms = gpd.GeoSeries(
        df["wkb"].map(lambda v: from_wkb(bytes(v) if not isinstance(v, (bytes, bytearray)) else v)),
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
        subset.plot(
            column=column,
            ax=ax,
            cmap=cmap,
            legend=True,
            edgecolor="#555",
            linewidth=0.2,
            missing_kwds={"color": "lightgrey", "label": "no data"},
            scheme="quantiles",
            k=5,
        )
        ax.set_title(f"{muni} — {title}")
        ax.set_axis_off()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# %%
# Notebook driver: connect, aggregate, merge ACS, render choropleths.
db_path = resolve_db_path()
con = duckdb.connect(str(db_path))
con.execute("INSTALL spatial; LOAD spatial;")

print("[1/6] building BG PV + flux aggregates …")
build_bg_pv_flux_aggregates(con)

print(f"[2/6] fetching ACS {ACS_VINTAGE} 5-yr variables via censusdis …")
acs_df = fetch_acs_block_groups()
print(f"      retrieved {len(acs_df):,} BG rows")
con.register("acs_bg", acs_df)

print("[3/6] joining ACS + urban-blocks 2020 flag …")
urban_df = load_urban_flags(con)
if urban_df.empty:
    urban_df = pd.DataFrame({"bg_geoid": [], "urban_block_count": [], "is_urban": []})
con.register("urban_bg", urban_df)
con.execute(
    f"""
    CREATE OR REPLACE TABLE {BG_AGG_TABLE} AS
    SELECT a.*,
           acs.* EXCLUDE (bg_geoid),
           COALESCE(u.urban_block_count, 0) AS urban_block_count,
           COALESCE(u.is_urban, FALSE) AS is_urban
    FROM {BG_AGG_TABLE} AS a
    LEFT JOIN acs_bg AS acs ON acs.bg_geoid = a.bg_geoid
    LEFT JOIN urban_bg AS u ON u.bg_geoid = a.bg_geoid;
    """
)
con.unregister("acs_bg")
con.unregister("urban_bg")

# %%
print("[4/6] exporting aggregate CSV …")
FIG_DIR.mkdir(parents=True, exist_ok=True)
csv_out = FIG_DIR / "pv_bg_aggregates_sj_isabela.csv"
con.execute(f"COPY {BG_AGG_TABLE} TO '{csv_out}' (HEADER, DELIMITER ',');")
print(f"      wrote {csv_out}")

# %%
print("[5/6] building choropleths for San Juan + Isabela …")
bg_gdf = load_bg_geometries(con)
plot_choropleth(
    bg_gdf, "detected_pv_rate",
    "Detected PV rate (per building)",
    MAP_DIR / "pv_detection_rate_choropleth.png",
    cmap="YlOrBr",
)
plot_choropleth(
    bg_gdf, "annual_flux_mean_kwh_per_kw_yr",
    "Mean annual solar flux (kWh/kW/yr)",
    MAP_DIR / "annual_flux_choropleth.png",
    cmap="plasma",
)
plot_choropleth(
    bg_gdf, "median_household_income_usd",
    "ACS median household income (USD)",
    MAP_DIR / "acs_income_choropleth.png",
    cmap="viridis",
)
plot_choropleth(
    bg_gdf, "pct_bachelor_plus",
    "Share age 25+ with Bachelor's or higher",
    MAP_DIR / "acs_education_choropleth.png",
    cmap="BuGn",
)
plot_choropleth(
    bg_gdf, "diversity_index",
    "Census 2020 Diversity Index (from ACS B03002)",
    MAP_DIR / "acs_diversity_index_choropleth.png",
    cmap="magma",
)
# Urban/rural flag: plot as 0/1 category (no quantile binning needed).
if "is_urban" in bg_gdf.columns and bg_gdf["is_urban"].notna().any():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, muni in zip(axes, TARGET_MUNICIPALITIES):
        subset = bg_gdf[bg_gdf["municipio"] == muni]
        if subset.empty:
            ax.set_title(f"{muni}: no data")
            ax.set_axis_off()
            continue
        subset.assign(is_urban_int=subset["is_urban"].astype(int)).plot(
            column="is_urban_int", ax=ax, cmap="coolwarm",
            categorical=True, legend=True, edgecolor="#555", linewidth=0.2,
        )
        ax.set_title(f"{muni} — urban (2020) vs. rural BGs")
        ax.set_axis_off()
    fig.tight_layout()
    out = MAP_DIR / "urban_rural_2020_choropleth.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

# %%
print("[6/6] Moran's I on detected_pv_rate …")
try:
    from libpysal.weights import Queen
    from esda.moran import Moran

    mo_rows = []
    for muni in TARGET_MUNICIPALITIES:
        subset = bg_gdf[(bg_gdf["municipio"] == muni) & bg_gdf["detected_pv_rate"].notna()].copy()
        if len(subset) < 5:
            continue
        w = Queen.from_dataframe(subset, use_index=False)
        w.transform = "r"
        moran = Moran(subset["detected_pv_rate"].values, w, permutations=999)
        mo_rows.append({
            "municipio": muni,
            "n_bgs": len(subset),
            "morans_I": moran.I,
            "p_sim": moran.p_sim,
            "z_sim": moran.z_sim,
        })
    if mo_rows:
        mo_df = pd.DataFrame(mo_rows)
        mo_out = FIG_DIR / "pv_bg_morans_i.csv"
        mo_df.to_csv(mo_out, index=False)
        print(mo_df.to_string(index=False))
        print(f"      wrote {mo_out}")
except Exception as exc:
    print(f"Moran's I step skipped: {exc}")

con.close()
