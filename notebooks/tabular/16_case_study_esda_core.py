# %% [markdown]
# # Case-Study ESDA Core Draft
#
# Implements the tract-based ESDA path for San Juan and Isabela once the tract
# aggregate surface exists. The notebook focuses on the core 3-day sprint:
#
# - audited tract analysis surface loading,
# - outcome-versus-covariate Pearson and Spearman correlations,
# - per-municipality global Moran's I,
# - one Queen-versus-Rook sensitivity check, and
# - one Local Moran / LISA export and map for the primary PV outcome.
#
# Outputs:
# - `outputs/reports/case_study_esda_correlations.csv`
# - `outputs/reports/case_study_esda_morans_i.csv`
# - `outputs/reports/case_study_esda_weight_sensitivity.csv`
# - `outputs/reports/case_study_esda_lisa_clusters.parquet`
# - `outputs/maps/case_study_esda_lisa_primary_outcome.png`

# %%
"""16_case_study_esda_core.py"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr


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

from utils.census import resolve_vector_db_path

TARGET_MUNICIPALITIES = ("San Juan", "Isabela")
AGG_TABLE = "pr_pv_tract_aggregates"
AGG_CSV_PATH = PROJECT_ROOT / "outputs" / "figures" / "pv_tract_aggregates_sj_isabela.csv"
GEOGRAPHY_TABLE = "pr_census_tracts"
GEOGRAPHY_ID_COLUMN = "tract_geoid"
PRIMARY_OUTCOME_CANDIDATES = (
    "any_pv_evidence_buildings_per_1000",
    "detected_pv_buildings_per_1000",
    "osm_labeled_buildings_per_1000",
)
SENSITIVITY_OUTCOME_CANDIDATES = (
    "detected_pv_buildings_per_1000",
    "osm_labeled_buildings_per_1000",
)
CONTEXTUAL_COVARIATES = (
    "nsrdb_multiyear_ghi_mean",
    "median_household_income_usd",
    "pct_bachelor_plus",
    "pct_owner_occupied",
    "pct_spanish_english_well_18_64",
    "pct_spanish_english_not_at_all_18_64",
    "cdc_svi_2020_overall_percentile",
    "pct_urban_land_area",
)
LISA_SIGNIFICANCE = float(os.getenv("CASE_STUDY_LISA_SIGNIFICANCE", "0.05") or "0.05")
MORAN_PERMUTATIONS = int(os.getenv("CASE_STUDY_MORAN_PERMUTATIONS", "999") or "999")

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
MAP_DIR = PROJECT_ROOT / "outputs" / "maps"
CORRELATION_OUTPUT_PATH = REPORT_DIR / "case_study_esda_correlations.csv"
MORAN_OUTPUT_PATH = REPORT_DIR / "case_study_esda_morans_i.csv"
WEIGHT_SENSITIVITY_OUTPUT_PATH = REPORT_DIR / "case_study_esda_weight_sensitivity.csv"
LISA_OUTPUT_PATH = REPORT_DIR / "case_study_esda_lisa_clusters.parquet"
LISA_MAP_OUTPUT_PATH = MAP_DIR / "case_study_esda_lisa_primary_outcome.png"


def resolve_db_path() -> Path:
    return resolve_vector_db_path(PROJECT_ROOT)


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")
    return con


def _to_bytes(value: object) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    return bytes(value)


def load_analysis_surface(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    has_table = bool(
        con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?;",
            [AGG_TABLE],
        ).fetchone()[0]
    )
    names_sql = ", ".join("?" * len(TARGET_MUNICIPALITIES))

    if has_table:
        frame = con.execute(
            f"""
            SELECT
                agg.*,
                ST_AsWKB(geom.geometry) AS geometry_wkb
            FROM {AGG_TABLE} AS agg
            JOIN {GEOGRAPHY_TABLE} AS geom
              ON geom.GEOID = agg.{GEOGRAPHY_ID_COLUMN}
            WHERE agg.municipio IN ({names_sql})
            ORDER BY agg.municipio, agg.{GEOGRAPHY_ID_COLUMN};
            """,
            list(TARGET_MUNICIPALITIES),
        ).fetchdf()
    elif AGG_CSV_PATH.exists():
        agg_frame = pd.read_csv(AGG_CSV_PATH)
        if agg_frame.empty:
            raise RuntimeError(f"{AGG_CSV_PATH} exists but is empty.")
        con.register("staged_analysis_surface", agg_frame)
        frame = con.execute(
            f"""
            SELECT
                staged_analysis_surface.*,
                ST_AsWKB(geom.geometry) AS geometry_wkb
            FROM staged_analysis_surface
            JOIN {GEOGRAPHY_TABLE} AS geom
              ON geom.GEOID = staged_analysis_surface.{GEOGRAPHY_ID_COLUMN}
            WHERE staged_analysis_surface.municipio IN ({names_sql})
            ORDER BY staged_analysis_surface.municipio, staged_analysis_surface.{GEOGRAPHY_ID_COLUMN};
            """,
            list(TARGET_MUNICIPALITIES),
        ).fetchdf()
        con.unregister("staged_analysis_surface")
    else:
        raise RuntimeError(
            f"Neither {AGG_TABLE} nor {AGG_CSV_PATH} is available. Rebuild the tract aggregate first."
        )

    if frame.empty:
        raise RuntimeError("No tract analysis rows were available for San Juan and Isabela.")

    geometry = gpd.GeoSeries.from_wkb(frame["geometry_wkb"].map(_to_bytes), crs="EPSG:4326")
    return gpd.GeoDataFrame(frame.drop(columns=["geometry_wkb"]), geometry=geometry, crs="EPSG:4326")


def resolve_primary_outcome(frame: pd.DataFrame) -> str:
    for column_name in PRIMARY_OUTCOME_CANDIDATES:
        if column_name in frame.columns:
            return column_name
    raise RuntimeError(
        "No primary PV outcome column was found. Expected one of: " + ", ".join(PRIMARY_OUTCOME_CANDIDATES)
    )


def resolve_analysis_covariates(frame: pd.DataFrame) -> list[str]:
    return [column_name for column_name in CONTEXTUAL_COVARIATES if column_name in frame.columns]


def _valid_numeric_pair(frame: pd.DataFrame, x_column: str, y_column: str) -> pd.DataFrame:
    subset = frame[[x_column, y_column]].apply(pd.to_numeric, errors="coerce").dropna()
    if subset.empty:
        return subset
    if subset[x_column].nunique(dropna=True) < 2 or subset[y_column].nunique(dropna=True) < 2:
        return pd.DataFrame(columns=[x_column, y_column])
    return subset


def build_correlation_table(
    analysis_gdf: gpd.GeoDataFrame,
    *,
    outcome_column: str,
    covariates: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = [("pooled", analysis_gdf)] + [(municipio, subset.copy()) for municipio, subset in analysis_gdf.groupby("municipio", sort=False)]

    for scope_name, scope_frame in scopes:
        for covariate in covariates:
            valid = _valid_numeric_pair(scope_frame, outcome_column, covariate)
            if len(valid) < 3:
                continue

            pearson_r, pearson_p = pearsonr(valid[outcome_column], valid[covariate])
            spearman_rho, spearman_p = spearmanr(valid[outcome_column], valid[covariate])
            rows.append(
                {
                    "scope": scope_name,
                    "outcome": outcome_column,
                    "covariate": covariate,
                    "n": int(len(valid)),
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_rho": float(spearman_rho),
                    "spearman_p": float(spearman_p),
                }
            )

    return pd.DataFrame(rows)


def classify_lisa_cluster(quadrant: int, p_value: float, significance: float = LISA_SIGNIFICANCE) -> str:
    if not math.isfinite(p_value) or p_value >= significance:
        return "not_significant"
    quadrant_map = {
        1: "high_high",
        2: "low_high",
        3: "low_low",
        4: "high_low",
    }
    return quadrant_map.get(int(quadrant), "not_significant")


def run_global_moran(
    analysis_gdf: gpd.GeoDataFrame,
    *,
    value_column: str,
    weight_kind: str,
) -> pd.DataFrame:
    from esda.moran import Moran
    from libpysal.weights import Queen, Rook

    rows: list[dict[str, object]] = []
    weight_factory = Queen if weight_kind == "queen" else Rook

    for municipio, subset in analysis_gdf.groupby("municipio", sort=False):
        subset = subset[subset[value_column].notna()].copy()
        if len(subset) < 5:
            continue
        weights = weight_factory.from_dataframe(subset, use_index=False)
        weights.transform = "r"
        moran = Moran(subset[value_column].to_numpy(), weights, permutations=MORAN_PERMUTATIONS)
        rows.append(
            {
                "metric": value_column,
                "municipio": municipio,
                "weights": weight_kind,
                "n_tracts": int(len(subset)),
                "morans_I": float(moran.I),
                "p_sim": float(moran.p_sim),
                "z_sim": float(moran.z_sim),
            }
        )

    return pd.DataFrame(rows)


def run_local_moran(analysis_gdf: gpd.GeoDataFrame, *, value_column: str) -> gpd.GeoDataFrame:
    from esda.moran import Moran_Local
    from libpysal.weights import Queen

    frames: list[gpd.GeoDataFrame] = []
    for municipio, subset in analysis_gdf.groupby("municipio", sort=False):
        subset = subset[subset[value_column].notna()].copy()
        if len(subset) < 5:
            continue
        weights = Queen.from_dataframe(subset, use_index=False)
        weights.transform = "r"
        local = Moran_Local(subset[value_column].to_numpy(), weights, permutations=MORAN_PERMUTATIONS)
        subset["lisa_metric"] = value_column
        subset["local_moran_i"] = local.Is
        subset["local_moran_p_sim"] = local.p_sim
        subset["lisa_quadrant"] = local.q
        subset["lisa_cluster"] = [
            classify_lisa_cluster(quadrant, p_value)
            for quadrant, p_value in zip(local.q, local.p_sim)
        ]
        frames.append(subset)

    if not frames:
        return gpd.GeoDataFrame(columns=[GEOGRAPHY_ID_COLUMN, "municipio", "lisa_cluster", "geometry"], geometry="geometry", crs=analysis_gdf.crs)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=analysis_gdf.crs)


def plot_lisa_map(lisa_gdf: gpd.GeoDataFrame, *, value_column: str, output_path: Path = LISA_MAP_OUTPUT_PATH) -> None:
    if lisa_gdf.empty:
        print("LISA map skipped: no significant Local Moran surface was produced.")
        return

    cluster_order = ["high_high", "low_low", "low_high", "high_low", "not_significant"]
    color_map = {
        "high_high": "#b2182b",
        "low_low": "#2166ac",
        "low_high": "#67a9cf",
        "high_low": "#ef8a62",
        "not_significant": "#d9d9d9",
    }
    fig, axes = plt.subplots(1, len(TARGET_MUNICIPALITIES), figsize=(14, 7), constrained_layout=True)
    if len(TARGET_MUNICIPALITIES) == 1:
        axes = [axes]

    for ax, municipio in zip(axes, TARGET_MUNICIPALITIES):
        subset = lisa_gdf[lisa_gdf["municipio"] == municipio].copy()
        if subset.empty:
            ax.set_title(f"{municipio}: no LISA surface")
            ax.set_axis_off()
            continue

        for cluster_name in cluster_order:
            cluster_subset = subset[subset["lisa_cluster"] == cluster_name]
            if cluster_subset.empty:
                continue
            cluster_subset.plot(
                ax=ax,
                color=color_map[cluster_name],
                edgecolor="#3a3a3a",
                linewidth=0.2,
            )

        subset.boundary.plot(ax=ax, color="#111111", linewidth=0.3)
        ax.set_title(f"{municipio} — LISA ({value_column})")
        ax.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    db_path = resolve_db_path()
    con = connect(db_path)
    analysis_surface = load_analysis_surface(con)
    con.close()

    outcome_column = resolve_primary_outcome(analysis_surface)
    covariates = resolve_analysis_covariates(analysis_surface)
    sensitivity_outcomes = [column_name for column_name in SENSITIVITY_OUTCOME_CANDIDATES if column_name in analysis_surface.columns]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)

    correlations = build_correlation_table(analysis_surface, outcome_column=outcome_column, covariates=covariates)
    correlations.to_csv(CORRELATION_OUTPUT_PATH, index=False)

    moran_frames = [run_global_moran(analysis_surface, value_column=outcome_column, weight_kind="queen")]
    weight_sensitivity = pd.concat(
        [
            run_global_moran(analysis_surface, value_column=outcome_column, weight_kind="queen"),
            run_global_moran(analysis_surface, value_column=outcome_column, weight_kind="rook"),
        ],
        ignore_index=True,
    )
    for value_column in sensitivity_outcomes:
        moran_frames.append(run_global_moran(analysis_surface, value_column=value_column, weight_kind="queen"))
    morans_i = pd.concat([frame for frame in moran_frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in moran_frames) else pd.DataFrame()

    lisa_gdf = run_local_moran(analysis_surface, value_column=outcome_column)
    if not morans_i.empty:
        morans_i.to_csv(MORAN_OUTPUT_PATH, index=False)
    weight_sensitivity.to_csv(WEIGHT_SENSITIVITY_OUTPUT_PATH, index=False)
    if not lisa_gdf.empty:
        lisa_gdf.to_parquet(LISA_OUTPUT_PATH, index=False)
        plot_lisa_map(lisa_gdf, value_column=outcome_column, output_path=LISA_MAP_OUTPUT_PATH)

    print(f"primary outcome: {outcome_column}")
    print(f"covariates: {', '.join(covariates) if covariates else 'none'}")
    print(f"correlations csv: {CORRELATION_OUTPUT_PATH}")
    if not morans_i.empty:
        print(f"Moran's I csv: {MORAN_OUTPUT_PATH}")
    print(f"weight sensitivity csv: {WEIGHT_SENSITIVITY_OUTPUT_PATH}")
    if not lisa_gdf.empty:
        print(f"LISA parquet: {LISA_OUTPUT_PATH}")
