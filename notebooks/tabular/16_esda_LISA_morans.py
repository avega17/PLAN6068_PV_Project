# %% [markdown]
# # Slide 10 ESDA: LISA Clusters and Limited Bivariate Moran
#
# Replaces the earlier single-outcome ESDA draft with two deliberately scoped
# preliminary methods for the final presentation:
#
# - univariate Local Moran / LISA cluster maps for the slide-10 variables, and
# - compact PV-centered bivariate Moran diagnostics.
#
# Outputs:
# - `outputs/reports/case_study_esda_global_morans_by_variable.csv`
# - `outputs/reports/case_study_esda_lisa_clusters_by_variable.parquet`
# - `outputs/reports/case_study_esda_lisa_cluster_counts.csv`
# - `outputs/reports/case_study_esda_bivariate_morans.csv`
# - `outputs/reports/case_study_esda_local_bivariate_pv_covariates.parquet`
# - `outputs/maps/case_study_esda_lisa_variable_grid.png`
# - `outputs/maps/case_study_esda_lisa_primary_outcome.png`
# - `outputs/maps/case_study_esda_bivariate_moran_pv_covariates.png`
# - `outputs/maps/case_study_esda_bivariate_moran_pv_remaining_covariates.png`
# - `outputs/maps/case_study_esda_local_bivariate_pv_{metric_slug}.png`

# %%
"""Slide-10 LISA and bivariate Moran analysis for San Juan and Isabela."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.patches import Patch


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


@dataclass(frozen=True)
class LisaVariable:
    column: str
    label: str
    short_label: str
    slug: str
    narrative: str


TARGET_MUNICIPALITIES = ("San Juan", "Isabela")
AGG_TABLE = "pr_pv_tract_aggregates"
AGG_CSV_PATH = PROJECT_ROOT / "outputs" / "figures" / "pv_tract_aggregates_sj_isabela.csv"
GEOGRAPHY_TABLE = "pr_census_tracts"
GEOGRAPHY_ID_COLUMN = "tract_geoid"
PRIMARY_PV_COLUMN = "any_pv_evidence_buildings_per_1000"
MIN_OBSERVATIONS = 5

LISA_SIGNIFICANCE = float(os.getenv("CASE_STUDY_LISA_SIGNIFICANCE", "0.05") or "0.05")
MORAN_PERMUTATIONS = int(os.getenv("CASE_STUDY_MORAN_PERMUTATIONS", "999") or "999")
MORAN_RANDOM_SEED = int(os.getenv("CASE_STUDY_MORAN_SEED", "6068") or "6068")

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
MAP_DIR = PROJECT_ROOT / "outputs" / "maps"

GLOBAL_MORAN_OUTPUT_PATH = REPORT_DIR / "case_study_esda_global_morans_by_variable.csv"
LEGACY_MORAN_OUTPUT_PATH = REPORT_DIR / "case_study_esda_morans_i.csv"
LISA_OUTPUT_PATH = REPORT_DIR / "case_study_esda_lisa_clusters_by_variable.parquet"
LEGACY_LISA_OUTPUT_PATH = REPORT_DIR / "case_study_esda_lisa_clusters.parquet"
LISA_CLUSTER_COUNT_OUTPUT_PATH = REPORT_DIR / "case_study_esda_lisa_cluster_counts.csv"
BIVARIATE_MORAN_OUTPUT_PATH = REPORT_DIR / "case_study_esda_bivariate_morans.csv"
LOCAL_BIVARIATE_OUTPUT_PATH = REPORT_DIR / "case_study_esda_local_bivariate_pv_covariates.parquet"
LEGACY_LOCAL_BIVARIATE_OUTPUT_PATH = REPORT_DIR / "case_study_esda_local_bivariate_pv_svi.parquet"
SKIPPED_METRIC_OUTPUT_PATH = REPORT_DIR / "case_study_esda_skipped_metrics.csv"
METHOD_NOTE_OUTPUT_PATH = REPORT_DIR / "case_study_esda_lisa_interpretation.md"

LISA_GRID_OUTPUT_PATH = MAP_DIR / "case_study_esda_lisa_variable_grid.png"
LISA_PRIMARY_OUTPUT_PATH = MAP_DIR / "case_study_esda_lisa_primary_outcome.png"
BIVARIATE_SCATTER_OUTPUT_PATH = MAP_DIR / "case_study_esda_bivariate_moran_pv_covariates.png"
BIVARIATE_SCATTER_REMAINING_OUTPUT_PATH = MAP_DIR / "case_study_esda_bivariate_moran_pv_remaining_covariates.png"
LOCAL_BIVARIATE_MAP_OUTPUT_TEMPLATE = "case_study_esda_local_bivariate_pv_{slug}.png"
LOCAL_BIVARIATE_MAP_OUTPUT_PATH = MAP_DIR / LOCAL_BIVARIATE_MAP_OUTPUT_TEMPLATE.format(slug="svi")

LISA_VARIABLES: tuple[LisaVariable, ...] = (
    LisaVariable(
        column=PRIMARY_PV_COLUMN,
        label="Buildings with any PV evidence per 1,000 buildings",
        short_label="PV density",
        slug="pv_density",
        narrative="Broadest rooftop-PV prevalence signal, combining OSM labels and model detections per 1,000 buildings.",
    ),
    LisaVariable(
        column="median_household_income_usd",
        label="ACS median household income (USD)",
        short_label="Median income",
        slug="income",
        narrative="Socioeconomic purchasing-power context from the tract-level ACS slice.",
    ),
    LisaVariable(
        column="pct_bachelor_plus",
        label="Share age 25+ with Bachelor's or higher",
        short_label="Education",
        slug="education",
        narrative="Educational-attainment context for rooftop-PV adoption patterns.",
    ),
    LisaVariable(
        column="pct_spanish_english_well_18_64",
        label="Share of Spanish-speaking adults 18-64 who speak English well",
        short_label="English proficiency",
        slug="english_well",
        narrative="Bilingual-capacity proxy for access to labor markets and service networks.",
    ),
    LisaVariable(
        column="cdc_svi_2020_overall_percentile",
        label="CDC SVI 2020 overall percentile rank (0-100)",
        short_label="Social vulnerability",
        slug="svi",
        narrative="CDC/ATSDR overall vulnerability percentile, rescaled to 0-100.",
    ),
    LisaVariable(
        column="nsrdb_multiyear_ghi_mean",
        label="NSRDB multiyear GHI mean (W/m2)",
        short_label="Irradiance",
        slug="ghi",
        narrative="Nearest-site NSRDB long-run global horizontal irradiance context attached through buildings.",
    ),
)

BIVARIATE_ONLY_VARIABLES: tuple[LisaVariable, ...] = (
    LisaVariable(
        column="pct_spanish_english_not_at_all_18_64",
        label="Share of Spanish-speaking adults 18-64 who speak no English",
        short_label="No English",
        slug="english_none",
        narrative="Language-access contrast variable already surfaced in the tract choropleth set.",
    ),
)
BIVARIATE_VARIABLES: tuple[LisaVariable, ...] = (*LISA_VARIABLES, *BIVARIATE_ONLY_VARIABLES)

BIVARIATE_COVARIATE_SLUGS = ("income", "education", "english_well", "english_none", "svi", "ghi")
BIVARIATE_FIGURE_SLUGS = ("income", "svi", "ghi")
BIVARIATE_REMAINING_FIGURE_SLUGS = ("education", "english_well", "english_none")

LISA_CLUSTER_ORDER = ("high_high", "low_low", "high_low", "low_high", "not_significant")
LISA_PLOT_ORDER = ("not_significant", "low_low", "low_high", "high_low", "high_high")
LISA_CLUSTER_COLORS = {
    "high_high": "#b2182b",
    "low_low": "#2166ac",
    "high_low": "#ef8a62",
    "low_high": "#67a9cf",
    "not_significant": "#d9d9d9",
}
LISA_CLUSTER_LABELS = {
    "high_high": "High-High (hot spot)",
    "low_low": "Low-Low (cold spot)",
    "high_low": "High-Low (high outlier)",
    "low_high": "Low-High (low outlier)",
    "not_significant": "Not significant",
}
PYSAL_QUADRANT_LABELS = {
    1: "high_high",
    2: "low_high",
    3: "low_low",
    4: "high_low",
}


def resolve_db_path() -> Path:
    return resolve_vector_db_path(PROJECT_ROOT)


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")
    return con


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


def _to_bytes(value: object) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    return bytes(value)


def load_analysis_surface(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    names_sql = ", ".join("?" * len(TARGET_MUNICIPALITIES))

    if table_exists(con, AGG_TABLE):
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
    gdf = gpd.GeoDataFrame(frame.drop(columns=["geometry_wkb"]), geometry=geometry, crs="EPSG:4326")
    return gdf[gdf["municipio"].isin(TARGET_MUNICIPALITIES)].copy()


def variable_by_slug(
    slug: str,
    variables: tuple[LisaVariable, ...] | list[LisaVariable] | None = None,
) -> LisaVariable | None:
    search_variables = BIVARIATE_VARIABLES if variables is None else variables
    return next((variable for variable in search_variables if variable.slug == slug), None)


def available_variables(frame: pd.DataFrame) -> tuple[list[LisaVariable], list[dict[str, object]]]:
    selected: list[LisaVariable] = []
    skipped: list[dict[str, object]] = []

    for variable in LISA_VARIABLES:
        if variable.column not in frame.columns:
            skipped.append(
                {
                    "metric": variable.column,
                    "metric_slug": variable.slug,
                    "reason": "missing column",
                }
            )
            continue
        values = pd.to_numeric(frame[variable.column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.dropna().empty:
            skipped.append(
                {
                    "metric": variable.column,
                    "metric_slug": variable.slug,
                    "reason": "all values are null or non-numeric",
                }
            )
            continue
        selected.append(variable)

    return selected, skipped


def valid_metric_subset(frame: gpd.GeoDataFrame, variable: LisaVariable) -> gpd.GeoDataFrame:
    if variable.column not in frame.columns:
        return gpd.GeoDataFrame(columns=frame.columns, geometry="geometry", crs=frame.crs)

    subset = frame.copy()
    subset[variable.column] = pd.to_numeric(subset[variable.column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    subset = subset[subset[variable.column].notna()].copy()
    if subset.empty:
        return subset
    return gpd.GeoDataFrame(subset, geometry="geometry", crs=frame.crs)


def has_enough_variation(frame: pd.DataFrame, column: str) -> bool:
    return len(frame) >= MIN_OBSERVATIONS and frame[column].nunique(dropna=True) >= 2


def build_queen_weights(frame: gpd.GeoDataFrame):
    from libpysal.weights import Queen

    try:
        weights = Queen.from_dataframe(frame, use_index=False, silence_warnings=True)
    except TypeError:
        weights = Queen.from_dataframe(frame, use_index=False)
    weights.transform = "r"
    return weights


def cluster_from_quadrant(quadrant: int, p_value: float, significance: float = LISA_SIGNIFICANCE) -> str:
    if not math.isfinite(float(p_value)) or float(p_value) >= significance:
        return "not_significant"
    return PYSAL_QUADRANT_LABELS.get(int(quadrant), "not_significant")


def cluster_label(cluster_name: str) -> str:
    return LISA_CLUSTER_LABELS.get(cluster_name, "Not significant")


def _float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def run_global_morans(
    analysis_gdf: gpd.GeoDataFrame,
    variables: list[LisaVariable],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from esda.moran import Moran

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for variable in variables:
        for municipio, muni_frame in analysis_gdf.groupby("municipio", sort=False):
            subset = valid_metric_subset(muni_frame, variable)
            if not has_enough_variation(subset, variable.column):
                skipped.append(
                    {
                        "method": "global_moran",
                        "metric": variable.column,
                        "metric_slug": variable.slug,
                        "municipio": municipio,
                        "reason": "fewer than 5 valid tracts or no variation",
                        "n_valid": int(len(subset)),
                    }
                )
                continue

            weights = build_queen_weights(subset)
            np.random.seed(MORAN_RANDOM_SEED)
            moran = Moran(subset[variable.column].to_numpy(), weights, permutations=MORAN_PERMUTATIONS)
            rows.append(
                {
                    "metric": variable.column,
                    "metric_slug": variable.slug,
                    "metric_label": variable.label,
                    "municipio": municipio,
                    "weights": "queen",
                    "n_tracts": int(len(subset)),
                    "island_count": int(len(weights.islands)),
                    "min_neighbors": int(weights.min_neighbors),
                    "max_neighbors": int(weights.max_neighbors),
                    "mean_neighbors": float(weights.mean_neighbors),
                    "morans_I": float(moran.I),
                    "expected_I": float(moran.EI),
                    "p_sim": _float_or_nan(getattr(moran, "p_sim", np.nan)),
                    "z_sim": _float_or_nan(getattr(moran, "z_sim", np.nan)),
                    "permutations": MORAN_PERMUTATIONS,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(skipped)


def run_local_morans(
    analysis_gdf: gpd.GeoDataFrame,
    variables: list[LisaVariable],
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    from esda.moran import Moran_Local

    frames: list[gpd.GeoDataFrame] = []
    skipped: list[dict[str, object]] = []

    for variable in variables:
        for municipio, muni_frame in analysis_gdf.groupby("municipio", sort=False):
            subset = valid_metric_subset(muni_frame, variable)
            if not has_enough_variation(subset, variable.column):
                skipped.append(
                    {
                        "method": "local_moran",
                        "metric": variable.column,
                        "metric_slug": variable.slug,
                        "municipio": municipio,
                        "reason": "fewer than 5 valid tracts or no variation",
                        "n_valid": int(len(subset)),
                    }
                )
                continue

            weights = build_queen_weights(subset)
            try:
                local = Moran_Local(
                    subset[variable.column].to_numpy(),
                    weights,
                    permutations=MORAN_PERMUTATIONS,
                    seed=MORAN_RANDOM_SEED,
                )
            except TypeError:
                np.random.seed(MORAN_RANDOM_SEED)
                local = Moran_Local(subset[variable.column].to_numpy(), weights, permutations=MORAN_PERMUTATIONS)

            out = subset[[GEOGRAPHY_ID_COLUMN, "municipio", variable.column, "geometry"]].copy()
            out = out.rename(columns={variable.column: "metric_value"})
            out["metric"] = variable.column
            out["metric_slug"] = variable.slug
            out["metric_label"] = variable.label
            out["metric_short_label"] = variable.short_label
            out["local_moran_i"] = local.Is
            out["local_moran_p_sim"] = local.p_sim
            out["lisa_quadrant"] = local.q
            out["lisa_cluster"] = [cluster_from_quadrant(q, p) for q, p in zip(local.q, local.p_sim)]
            out["lisa_cluster_label"] = out["lisa_cluster"].map(cluster_label)
            out["lisa_significant"] = out["lisa_cluster"] != "not_significant"
            out["neighbor_count"] = [int(weights.cardinalities.get(i, 0)) for i in range(len(out))]
            out["is_island"] = [i in weights.islands for i in range(len(out))]
            out["significance"] = LISA_SIGNIFICANCE
            out["permutations"] = MORAN_PERMUTATIONS
            frames.append(gpd.GeoDataFrame(out, geometry="geometry", crs=analysis_gdf.crs))

    if not frames:
        empty = gpd.GeoDataFrame(
            columns=[GEOGRAPHY_ID_COLUMN, "municipio", "metric", "metric_slug", "lisa_cluster", "geometry"],
            geometry="geometry",
            crs=analysis_gdf.crs,
        )
        return empty, pd.DataFrame(skipped)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=analysis_gdf.crs), pd.DataFrame(skipped)


def build_cluster_counts(lisa_gdf: gpd.GeoDataFrame, variables: list[LisaVariable]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variable in variables:
        metric_frame = lisa_gdf[lisa_gdf["metric_slug"] == variable.slug]
        for municipio in TARGET_MUNICIPALITIES:
            subset = metric_frame[metric_frame["municipio"] == municipio]
            if subset.empty:
                continue
            total = len(subset)
            for cluster_name in LISA_CLUSTER_ORDER:
                tract_count = int((subset["lisa_cluster"] == cluster_name).sum())
                rows.append(
                    {
                        "metric": variable.column,
                        "metric_slug": variable.slug,
                        "metric_label": variable.label,
                        "municipio": municipio,
                        "lisa_cluster": cluster_name,
                        "lisa_cluster_label": cluster_label(cluster_name),
                        "tract_count": tract_count,
                        "valid_tract_count": int(total),
                        "pct_of_valid_tracts": 100.0 * tract_count / total if total else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def metric_counts_text(metric_gdf: gpd.GeoDataFrame) -> str:
    if metric_gdf.empty:
        return "no valid tracts"
    abbreviations = {
        "high_high": "HH",
        "low_low": "LL",
        "high_low": "HL",
        "low_high": "LH",
    }
    parts = []
    for cluster_name in ("high_high", "low_low", "high_low", "low_high"):
        parts.append(f"{abbreviations[cluster_name]} {int((metric_gdf['lisa_cluster'] == cluster_name).sum())}")
    return "  ".join(parts)


def bounds_with_padding(gdf: gpd.GeoDataFrame, pad_fraction: float = 0.03) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    width = maxx - minx
    height = maxy - miny
    pad_x = width * pad_fraction if width else 0.01
    pad_y = height * pad_fraction if height else 0.01
    return minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y


def plot_lisa_clusters(
    ax: plt.Axes,
    base_gdf: gpd.GeoDataFrame,
    lisa_metric_gdf: gpd.GeoDataFrame,
) -> None:
    base_gdf.plot(ax=ax, color="#f5f5f5", edgecolor="#777777", linewidth=0.25)
    for cluster_name in LISA_PLOT_ORDER:
        cluster_subset = lisa_metric_gdf[lisa_metric_gdf["lisa_cluster"] == cluster_name]
        if cluster_subset.empty:
            continue
        cluster_subset.plot(
            ax=ax,
            color=LISA_CLUSTER_COLORS[cluster_name],
            edgecolor="#2f2f2f",
            linewidth=0.25,
        )
    base_gdf.boundary.plot(ax=ax, color="#222222", linewidth=0.2)
    ax.set_axis_off()


def lisa_legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=LISA_CLUSTER_COLORS[cluster_name], edgecolor="#444444", label=cluster_label(cluster_name))
        for cluster_name in LISA_CLUSTER_ORDER
    ]


def plot_lisa_metric_panel(
    ax: plt.Axes,
    analysis_gdf: gpd.GeoDataFrame,
    metric_gdf: gpd.GeoDataFrame,
    variable: LisaVariable,
) -> None:
    ax.set_axis_off()
    ax.set_title(variable.short_label, fontsize=13, weight="bold", pad=2)

    inset_specs = (
        ("Isabela", (0.03, 0.22, 0.42, 0.62)),
        ("San Juan", (0.55, 0.22, 0.42, 0.62)),
    )
    for municipio, inset_bounds in inset_specs:
        inset_ax = ax.inset_axes(inset_bounds)
        base_subset = analysis_gdf[analysis_gdf["municipio"] == municipio]
        lisa_subset = metric_gdf[metric_gdf["municipio"] == municipio]
        if base_subset.empty:
            inset_ax.set_axis_off()
            continue
        plot_lisa_clusters(inset_ax, base_subset, lisa_subset)
        xmin, xmax, ymin, ymax = bounds_with_padding(base_subset, pad_fraction=0.06)
        inset_ax.set_xlim(xmin, xmax)
        inset_ax.set_ylim(ymin, ymax)
        inset_ax.set_aspect("equal")
        inset_ax.text(
            0.5,
            -0.08,
            municipio,
            transform=inset_ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#333333",
        )

    ax.text(
        0.5,
        0.06,
        metric_counts_text(metric_gdf),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#1f1f1f",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.0},
    )


def plot_lisa_variable_grid(
    lisa_gdf: gpd.GeoDataFrame,
    analysis_gdf: gpd.GeoDataFrame,
    variables: list[LisaVariable],
    output_path: Path = LISA_GRID_OUTPUT_PATH,
) -> None:
    if lisa_gdf.empty:
        print("LISA grid skipped: no Local Moran surfaces were produced.")
        return

    ncols = 3
    nrows = math.ceil(len(variables) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8.6), constrained_layout=False)
    axes_array = np.atleast_1d(axes).ravel()

    for ax, variable in zip(axes_array, variables):
        metric_gdf = lisa_gdf[lisa_gdf["metric_slug"] == variable.slug]
        plot_lisa_metric_panel(ax, analysis_gdf, metric_gdf, variable)

    for ax in axes_array[len(variables) :]:
        ax.set_axis_off()

    fig.suptitle(
        "Local Moran / LISA Cluster Maps: San Juan and Isabela Census Tracts",
        fontsize=16,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.928,
        f"Queen contiguity, row-standardized weights, {MORAN_PERMUTATIONS} permutations, p < {LISA_SIGNIFICANCE:g}. "
        "HH/LL are spatial clusters; HL/LH are spatial outliers.",
        ha="center",
        va="top",
        fontsize=10,
        color="#333333",
    )
    fig.legend(
        handles=lisa_legend_handles(),
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.89))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def plot_lisa_metric_by_municipio(
    lisa_gdf: gpd.GeoDataFrame,
    analysis_gdf: gpd.GeoDataFrame,
    variable: LisaVariable,
    output_path: Path,
) -> None:
    metric_gdf = lisa_gdf[lisa_gdf["metric_slug"] == variable.slug]
    if metric_gdf.empty:
        print(f"LISA detail map skipped for {variable.column}: no surface was produced.")
        return

    fig, axes = plt.subplots(1, len(TARGET_MUNICIPALITIES), figsize=(14, 7), constrained_layout=True)
    axes_array = np.atleast_1d(axes)

    for ax, municipio in zip(axes_array, TARGET_MUNICIPALITIES):
        base_subset = analysis_gdf[analysis_gdf["municipio"] == municipio]
        lisa_subset = metric_gdf[metric_gdf["municipio"] == municipio]
        plot_lisa_clusters(ax, base_subset, lisa_subset)
        xmin, xmax, ymin, ymax = bounds_with_padding(base_subset)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title(f"{municipio}: {variable.short_label}\n{metric_counts_text(lisa_subset)}", fontsize=11)

    fig.suptitle(variable.label, fontsize=14, weight="bold")
    fig.legend(
        handles=lisa_legend_handles(),
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def plot_lisa_detail_maps(lisa_gdf: gpd.GeoDataFrame, analysis_gdf: gpd.GeoDataFrame, variables: list[LisaVariable]) -> None:
    for variable in variables:
        output_path = MAP_DIR / f"case_study_esda_lisa_{variable.slug}.png"
        plot_lisa_metric_by_municipio(lisa_gdf, analysis_gdf, variable, output_path)


def bivariate_covariates(analysis_gdf: pd.DataFrame, variables: list[LisaVariable]) -> list[LisaVariable]:
    available_lisa_slugs = {variable.slug for variable in variables}
    covariates: list[LisaVariable] = []
    for slug in BIVARIATE_COVARIATE_SLUGS:
        variable = variable_by_slug(slug)
        if variable is None or variable.column not in analysis_gdf.columns:
            continue
        if slug not in available_lisa_slugs and variable not in BIVARIATE_ONLY_VARIABLES:
            continue
        values = pd.to_numeric(analysis_gdf[variable.column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.dropna().empty:
            continue
        covariates.append(variable)
    return covariates


def run_bivariate_morans(
    analysis_gdf: gpd.GeoDataFrame,
    variables: list[LisaVariable],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from esda.moran import Moran_BV

    primary = variable_by_slug("pv_density")
    if primary is None or primary.column not in analysis_gdf.columns:
        return pd.DataFrame(), pd.DataFrame(
            [{"method": "bivariate_moran", "reason": "primary PV density column unavailable"}]
        )

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for covariate in bivariate_covariates(analysis_gdf, variables):
        for municipio, muni_frame in analysis_gdf.groupby("municipio", sort=False):
            subset = muni_frame.copy()
            subset[primary.column] = pd.to_numeric(subset[primary.column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            subset[covariate.column] = pd.to_numeric(subset[covariate.column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            subset = subset[subset[[primary.column, covariate.column]].notna().all(axis=1)].copy()

            if (
                len(subset) < MIN_OBSERVATIONS
                or subset[primary.column].nunique(dropna=True) < 2
                or subset[covariate.column].nunique(dropna=True) < 2
            ):
                skipped.append(
                    {
                        "method": "bivariate_moran",
                        "x_metric": primary.column,
                        "y_metric": covariate.column,
                        "y_metric_slug": covariate.slug,
                        "municipio": municipio,
                        "reason": "fewer than 5 paired tracts or no variation",
                        "n_valid": int(len(subset)),
                    }
                )
                continue

            weights = build_queen_weights(gpd.GeoDataFrame(subset, geometry="geometry", crs=analysis_gdf.crs))
            np.random.seed(MORAN_RANDOM_SEED)
            moran_bv = Moran_BV(
                subset[primary.column].to_numpy(),
                subset[covariate.column].to_numpy(),
                weights,
                permutations=MORAN_PERMUTATIONS,
            )
            rows.append(
                {
                    "x_metric": primary.column,
                    "x_metric_slug": primary.slug,
                    "x_metric_label": primary.label,
                    "y_metric": covariate.column,
                    "y_metric_slug": covariate.slug,
                    "y_metric_label": covariate.label,
                    "relationship": f"{primary.short_label} vs spatial lag of {covariate.short_label}",
                    "municipio": municipio,
                    "weights": "queen",
                    "n_tracts": int(len(subset)),
                    "moran_bv_I": float(moran_bv.I),
                    "p_sim": _float_or_nan(getattr(moran_bv, "p_sim", np.nan)),
                    "p_z_sim": _float_or_nan(getattr(moran_bv, "p_z_sim", np.nan)),
                    "z_sim": _float_or_nan(getattr(moran_bv, "z_sim", np.nan)),
                    "permutations": MORAN_PERMUTATIONS,
                    "interpretation_note": "Bivariate Moran compares the x metric in each tract with the spatial lag of the y metric among neighboring tracts.",
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(skipped)


def zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.nanstd(values))
    if std == 0 or not math.isfinite(std):
        return np.full_like(values, np.nan, dtype=float)
    return (values - float(np.nanmean(values))) / std


def plot_quadrant_labels(ax: plt.Axes) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_pad = (xlim[1] - xlim[0]) * 0.05
    y_pad = (ylim[1] - ylim[0]) * 0.06
    ax.text(xlim[1] - x_pad, ylim[1] - y_pad, "HH", ha="right", va="top", fontsize=9, color="#777777")
    ax.text(xlim[0] + x_pad, ylim[1] - y_pad, "LH", ha="left", va="top", fontsize=9, color="#777777")
    ax.text(xlim[0] + x_pad, ylim[0] + y_pad, "LL", ha="left", va="bottom", fontsize=9, color="#777777")
    ax.text(xlim[1] - x_pad, ylim[0] + y_pad, "HL", ha="right", va="bottom", fontsize=9, color="#777777")


def plot_bivariate_moran_grid(
    analysis_gdf: gpd.GeoDataFrame,
    variables: list[LisaVariable],
    bivariate_df: pd.DataFrame,
    output_path: Path = BIVARIATE_SCATTER_OUTPUT_PATH,
    covariate_slugs: tuple[str, ...] = BIVARIATE_FIGURE_SLUGS,
    title: str = "Limited Bivariate Moran Diagnostics: PV Density vs Neighboring Context",
    subtitle: str = "Each panel compares tract PV density with the row-standardized spatial lag of a contextual variable in neighboring tracts.",
) -> None:
    from libpysal.weights import lag_spatial

    primary = variable_by_slug("pv_density")
    if primary is None or bivariate_df.empty:
        print("Bivariate Moran scatter grid skipped: no bivariate rows were produced.")
        return

    available_covariate_slugs = {variable.slug for variable in bivariate_covariates(analysis_gdf, variables)}
    covariates = [variable_by_slug(slug) for slug in covariate_slugs]
    covariates = [variable for variable in covariates if variable is not None and variable.slug in available_covariate_slugs]
    if not covariates:
        print("Bivariate Moran scatter grid skipped: no figure covariates were available.")
        return

    fig_width = max(5 * len(covariates), 8)
    fig, axes = plt.subplots(len(TARGET_MUNICIPALITIES), len(covariates), figsize=(fig_width, 8), squeeze=False)
    municipio_colors = {"San Juan": "#2a6f97", "Isabela": "#b36b00"}

    for row_idx, municipio in enumerate(TARGET_MUNICIPALITIES):
        muni_frame = analysis_gdf[analysis_gdf["municipio"] == municipio].copy()
        for col_idx, covariate in enumerate(covariates):
            ax = axes[row_idx][col_idx]
            subset = muni_frame.copy()
            subset[primary.column] = pd.to_numeric(subset[primary.column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            subset[covariate.column] = pd.to_numeric(subset[covariate.column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            subset = subset[subset[[primary.column, covariate.column]].notna().all(axis=1)].copy()

            bv_row = bivariate_df[
                (bivariate_df["municipio"] == municipio) & (bivariate_df["y_metric_slug"] == covariate.slug)
            ]
            if (
                subset.empty
                or len(subset) < MIN_OBSERVATIONS
                or subset[primary.column].nunique(dropna=True) < 2
                or subset[covariate.column].nunique(dropna=True) < 2
                or bv_row.empty
            ):
                ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            weights = build_queen_weights(gpd.GeoDataFrame(subset, geometry="geometry", crs=analysis_gdf.crs))
            zx = zscore(subset[primary.column].to_numpy(dtype=float))
            zy = zscore(subset[covariate.column].to_numpy(dtype=float))
            lag_zy = lag_spatial(weights, zy)
            moran_i = float(bv_row.iloc[0]["moran_bv_I"])
            p_value = _float_or_nan(bv_row.iloc[0]["p_sim"])

            ax.scatter(
                zx,
                lag_zy,
                s=34,
                color=municipio_colors.get(municipio, "#333333"),
                edgecolor="white",
                linewidth=0.4,
                alpha=0.82,
            )
            xlim = np.nanmax(np.abs(zx))
            ylim = np.nanmax(np.abs(lag_zy))
            lim = max(float(xlim), float(ylim), 1.0) * 1.12
            line_x = np.linspace(-lim, lim, 100)
            ax.plot(line_x, moran_i * line_x, color="#111111", linewidth=1.0, alpha=0.75)
            ax.axhline(0, color="#9a9a9a", linewidth=0.8, linestyle="--")
            ax.axvline(0, color="#9a9a9a", linewidth=0.8, linestyle="--")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_title(f"{municipio} - neighbor {covariate.short_label}\nI={moran_i:.3f}, p={p_value:.3f}", fontsize=10)
            ax.set_xlabel("PV density (standardized)")
            ax.set_ylabel(f"Spatial lag: {covariate.short_label}")
            plot_quadrant_labels(ax)

    fig.suptitle(title, fontsize=15, weight="bold", y=0.99)
    fig.text(
        0.5,
        0.935,
        subtitle,
        ha="center",
        fontsize=10,
        color="#333333",
    )
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def run_local_bivariate_morans(
    analysis_gdf: gpd.GeoDataFrame,
    variables: list[LisaVariable],
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, list[LisaVariable]]:
    from esda.moran import Moran_Local_BV

    primary = variable_by_slug("pv_density")
    covariates = bivariate_covariates(analysis_gdf, variables)
    if primary is None or primary.column not in analysis_gdf.columns or not covariates:
        return (
            gpd.GeoDataFrame(columns=[GEOGRAPHY_ID_COLUMN, "municipio", "geometry"], geometry="geometry", crs=analysis_gdf.crs),
            pd.DataFrame([{"method": "local_bivariate_moran", "reason": "primary or covariates unavailable"}]),
            covariates,
        )

    frames: list[gpd.GeoDataFrame] = []
    skipped: list[dict[str, object]] = []
    for covariate in covariates:
        for municipio, muni_frame in analysis_gdf.groupby("municipio", sort=False):
            subset = muni_frame.copy()
            subset[primary.column] = pd.to_numeric(subset[primary.column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            subset[covariate.column] = pd.to_numeric(subset[covariate.column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            subset = subset[subset[[primary.column, covariate.column]].notna().all(axis=1)].copy()

            if (
                len(subset) < MIN_OBSERVATIONS
                or subset[primary.column].nunique(dropna=True) < 2
                or subset[covariate.column].nunique(dropna=True) < 2
            ):
                skipped.append(
                    {
                        "method": "local_bivariate_moran",
                        "x_metric": primary.column,
                        "x_metric_slug": primary.slug,
                        "y_metric": covariate.column,
                        "y_metric_slug": covariate.slug,
                        "municipio": municipio,
                        "reason": "fewer than 5 paired tracts or no variation",
                        "n_valid": int(len(subset)),
                    }
                )
                continue

            weights = build_queen_weights(gpd.GeoDataFrame(subset, geometry="geometry", crs=analysis_gdf.crs))
            try:
                local_bv = Moran_Local_BV(
                    subset[primary.column].to_numpy(),
                    subset[covariate.column].to_numpy(),
                    weights,
                    permutations=MORAN_PERMUTATIONS,
                    seed=MORAN_RANDOM_SEED,
                )
            except TypeError:
                np.random.seed(MORAN_RANDOM_SEED)
                local_bv = Moran_Local_BV(
                    subset[primary.column].to_numpy(),
                    subset[covariate.column].to_numpy(),
                    weights,
                    permutations=MORAN_PERMUTATIONS,
                )

            out = subset[[GEOGRAPHY_ID_COLUMN, "municipio", primary.column, covariate.column, "geometry"]].copy()
            out = out.rename(columns={primary.column: "x_metric_value", covariate.column: "y_metric_value"})
            out["x_metric"] = primary.column
            out["x_metric_slug"] = primary.slug
            out["x_metric_label"] = primary.label
            out["x_metric_short_label"] = primary.short_label
            out["y_metric"] = covariate.column
            out["y_metric_slug"] = covariate.slug
            out["y_metric_label"] = covariate.label
            out["y_metric_short_label"] = covariate.short_label
            out["local_bv_moran_i"] = local_bv.Is
            out["local_bv_p_sim"] = local_bv.p_sim
            out["local_bv_quadrant"] = local_bv.q
            out["lisa_cluster"] = [cluster_from_quadrant(q, p) for q, p in zip(local_bv.q, local_bv.p_sim)]
            out["lisa_cluster_label"] = out["lisa_cluster"].map(cluster_label)
            out["lisa_significant"] = out["lisa_cluster"] != "not_significant"
            out["significance"] = LISA_SIGNIFICANCE
            out["permutations"] = MORAN_PERMUTATIONS
            frames.append(gpd.GeoDataFrame(out, geometry="geometry", crs=analysis_gdf.crs))

    if not frames:
        empty = gpd.GeoDataFrame(columns=[GEOGRAPHY_ID_COLUMN, "municipio", "geometry"], geometry="geometry", crs=analysis_gdf.crs)
        return empty, pd.DataFrame(skipped), covariates
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=analysis_gdf.crs), pd.DataFrame(skipped), covariates


def local_bivariate_label(cluster_name: str, covariate: LisaVariable) -> str:
    labels = {
        "high_high": f"High PV / high neighbor {covariate.short_label}",
        "low_low": f"Low PV / low neighbor {covariate.short_label}",
        "high_low": f"High PV / low neighbor {covariate.short_label}",
        "low_high": f"Low PV / high neighbor {covariate.short_label}",
        "not_significant": "Not significant",
    }
    return labels.get(cluster_name, "Not significant")


def plot_local_bivariate_map(
    local_bv_gdf: gpd.GeoDataFrame,
    analysis_gdf: gpd.GeoDataFrame,
    covariate: LisaVariable | None,
    output_path: Path = LOCAL_BIVARIATE_MAP_OUTPUT_PATH,
) -> None:
    if local_bv_gdf.empty or covariate is None:
        print("Local bivariate map skipped: no local bivariate surface was produced.")
        return

    if "y_metric_slug" in local_bv_gdf.columns:
        metric_gdf = local_bv_gdf[local_bv_gdf["y_metric_slug"] == covariate.slug]
    else:
        metric_gdf = local_bv_gdf
    if metric_gdf.empty:
        print(f"Local bivariate map skipped for {covariate.slug}: no surface was produced.")
        return

    fig, axes = plt.subplots(1, len(TARGET_MUNICIPALITIES), figsize=(14, 7), constrained_layout=True)
    axes_array = np.atleast_1d(axes)

    for ax, municipio in zip(axes_array, TARGET_MUNICIPALITIES):
        base_subset = analysis_gdf[analysis_gdf["municipio"] == municipio]
        lisa_subset = metric_gdf[metric_gdf["municipio"] == municipio]
        plot_lisa_clusters(ax, base_subset, lisa_subset)
        xmin, xmax, ymin, ymax = bounds_with_padding(base_subset)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title(f"{municipio}\n{metric_counts_text(lisa_subset)}", fontsize=11)

    handles = [
        Patch(facecolor=LISA_CLUSTER_COLORS[cluster_name], edgecolor="#444444", label=local_bivariate_label(cluster_name, covariate))
        for cluster_name in LISA_CLUSTER_ORDER
    ]
    fig.suptitle(
        f"Local Bivariate Moran: PV density vs Neighboring {covariate.short_label}",
        fontsize=14,
        weight="bold",
    )
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def local_bivariate_map_output_path(covariate: LisaVariable) -> Path:
    return MAP_DIR / LOCAL_BIVARIATE_MAP_OUTPUT_TEMPLATE.format(slug=covariate.slug)


def plot_local_bivariate_maps(
    local_bv_gdf: gpd.GeoDataFrame,
    analysis_gdf: gpd.GeoDataFrame,
    covariates: list[LisaVariable],
) -> None:
    if not covariates:
        print("Local bivariate maps skipped: no covariates were available.")
        return
    for covariate in covariates:
        plot_local_bivariate_map(
            local_bv_gdf,
            analysis_gdf,
            covariate,
            output_path=local_bivariate_map_output_path(covariate),
        )


def write_interpretation_note(variables: list[LisaVariable], local_bv_covariates: list[LisaVariable]) -> None:
    variable_lines = [f"- `{variable.column}`: {variable.label}. {variable.narrative}" for variable in variables]
    covariate_note = ", ".join(covariate.short_label for covariate in local_bv_covariates) if local_bv_covariates else "not produced"
    lines = [
        "# Case-Study ESDA Interpretation Notes",
        "",
        "## LISA Cluster Labels",
        "",
        "- High-High (HH): a high-value tract surrounded by high-value neighbors; interpreted as a hot spot.",
        "- Low-Low (LL): a low-value tract surrounded by low-value neighbors; interpreted as a cold spot.",
        "- High-Low (HL): a high-value tract surrounded by low-value neighbors; interpreted as a high spatial outlier.",
        "- Low-High (LH): a low-value tract surrounded by high-value neighbors; interpreted as a low spatial outlier.",
        "- Not significant: the Local Moran pseudo p-value is not below the configured threshold.",
        "",
        f"Main threshold: p < {LISA_SIGNIFICANCE:g}; permutations: {MORAN_PERMUTATIONS}; weights: row-standardized Queen contiguity, run separately by municipality.",
        "",
        "## Slide Variables",
        "",
        *variable_lines,
        "",
        "## Bivariate Moran Caution",
        "",
        "Bivariate Moran compares one variable in each tract with the spatial lag of another variable in neighboring tracts. It is not the same as same-tract Pearson or Spearman correlation, and it should be interpreted as exploratory neighborhood alignment rather than causality.",
        "",
        f"Local bivariate backup maps: PV density vs neighboring {covariate_note}.",
    ]
    METHOD_NOTE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METHOD_NOTE_OUTPUT_PATH.write_text("\n".join(lines).rstrip() + "\n")
    print(f"wrote {METHOD_NOTE_OUTPUT_PATH}")


def write_outputs(
    global_morans: pd.DataFrame,
    lisa_gdf: gpd.GeoDataFrame,
    cluster_counts: pd.DataFrame,
    bivariate_morans: pd.DataFrame,
    local_bv_gdf: gpd.GeoDataFrame,
    skipped_frames: list[pd.DataFrame],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)

    if not global_morans.empty:
        global_morans.to_csv(GLOBAL_MORAN_OUTPUT_PATH, index=False)
        global_morans.to_csv(LEGACY_MORAN_OUTPUT_PATH, index=False)
        print(f"wrote {GLOBAL_MORAN_OUTPUT_PATH}")
        print(f"wrote {LEGACY_MORAN_OUTPUT_PATH}")
    if not lisa_gdf.empty:
        lisa_gdf.to_parquet(LISA_OUTPUT_PATH, index=False)
        lisa_gdf.to_parquet(LEGACY_LISA_OUTPUT_PATH, index=False)
        print(f"wrote {LISA_OUTPUT_PATH}")
        print(f"wrote {LEGACY_LISA_OUTPUT_PATH}")
    if not cluster_counts.empty:
        cluster_counts.to_csv(LISA_CLUSTER_COUNT_OUTPUT_PATH, index=False)
        print(f"wrote {LISA_CLUSTER_COUNT_OUTPUT_PATH}")
    if not bivariate_morans.empty:
        bivariate_morans.to_csv(BIVARIATE_MORAN_OUTPUT_PATH, index=False)
        print(f"wrote {BIVARIATE_MORAN_OUTPUT_PATH}")
    if not local_bv_gdf.empty:
        local_bv_gdf.to_parquet(LOCAL_BIVARIATE_OUTPUT_PATH, index=False)
        print(f"wrote {LOCAL_BIVARIATE_OUTPUT_PATH}")
        if "y_metric_slug" in local_bv_gdf.columns:
            legacy_local_bv = local_bv_gdf[local_bv_gdf["y_metric_slug"] == "svi"]
            if not legacy_local_bv.empty:
                legacy_local_bv.to_parquet(LEGACY_LOCAL_BIVARIATE_OUTPUT_PATH, index=False)
                print(f"wrote {LEGACY_LOCAL_BIVARIATE_OUTPUT_PATH}")

    skipped = pd.concat([frame for frame in skipped_frames if frame is not None and not frame.empty], ignore_index=True) if any(frame is not None and not frame.empty for frame in skipped_frames) else pd.DataFrame()
    if not skipped.empty:
        skipped.to_csv(SKIPPED_METRIC_OUTPUT_PATH, index=False)
        print(f"wrote {SKIPPED_METRIC_OUTPUT_PATH}")


def main() -> None:
    db_path = resolve_db_path()
    con = connect(db_path)
    analysis_surface = load_analysis_surface(con)
    con.close()

    variables, availability_skips = available_variables(analysis_surface)
    if not variables:
        raise RuntimeError("None of the slide-10 LISA variables are available in the analysis surface.")

    global_morans, global_skips = run_global_morans(analysis_surface, variables)
    lisa_gdf, lisa_skips = run_local_morans(analysis_surface, variables)
    cluster_counts = build_cluster_counts(lisa_gdf, variables)
    bivariate_morans, bivariate_skips = run_bivariate_morans(analysis_surface, variables)
    local_bv_gdf, local_bv_skips, local_bv_covariates = run_local_bivariate_morans(analysis_surface, variables)

    write_outputs(
        global_morans,
        lisa_gdf,
        cluster_counts,
        bivariate_morans,
        local_bv_gdf,
        [pd.DataFrame(availability_skips), global_skips, lisa_skips, bivariate_skips, local_bv_skips],
    )

    plot_lisa_variable_grid(lisa_gdf, analysis_surface, variables, output_path=LISA_GRID_OUTPUT_PATH)
    plot_lisa_detail_maps(lisa_gdf, analysis_surface, variables)
    primary_variable = variable_by_slug("pv_density")
    if primary_variable is not None:
        plot_lisa_metric_by_municipio(lisa_gdf, analysis_surface, primary_variable, LISA_PRIMARY_OUTPUT_PATH)
    plot_bivariate_moran_grid(analysis_surface, variables, bivariate_morans, output_path=BIVARIATE_SCATTER_OUTPUT_PATH)
    plot_bivariate_moran_grid(
        analysis_surface,
        variables,
        bivariate_morans,
        output_path=BIVARIATE_SCATTER_REMAINING_OUTPUT_PATH,
        covariate_slugs=BIVARIATE_REMAINING_FIGURE_SLUGS,
        title="Additional Bivariate Moran Diagnostics: PV Density vs Neighboring Context",
        subtitle="This companion grid extends the PV-centered screen to education and language-access variables.",
    )
    plot_local_bivariate_maps(local_bv_gdf, analysis_surface, local_bv_covariates)
    write_interpretation_note(variables, local_bv_covariates)

    print("slide-10 ESDA variables: " + ", ".join(variable.short_label for variable in variables))
    print(f"LISA significance threshold: p < {LISA_SIGNIFICANCE:g}")
    print(f"Moran permutations: {MORAN_PERMUTATIONS}")


if __name__ == "__main__":
    main()