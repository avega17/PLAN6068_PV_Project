# %% [markdown]
# # Case-Study Block-Group DBSCAN Appendix
#
# Clusters San Juan + Isabela census block groups from the rebuilt BG aggregate
# surface instead of the earlier NSRDB point-site surface. This keeps clustering
# in the appendix while aligning the unit of analysis with the core ESDA path.
#
# Outputs:
# - `outputs/reports/san_juan_isabela_bg_dbscan_clusters.parquet`
# - `outputs/reports/san_juan_isabela_bg_dbscan_summary.csv`
# - `outputs/maps/san_juan_isabela_bg_dbscan_clusters.png`

# %%
"""15_nsrdb_case_study_dbscan.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


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
TARGET_SCOPE_STEM = "_".join(municipio.lower().replace(" ", "_") for municipio in TARGET_MUNICIPALITIES)
ANALYSIS_CRS = "EPSG:32619"
BG_AGG_TABLE = "pr_pv_bg_aggregates"
PREFERRED_FEATURE_COLUMNS = (
    "any_pv_signal_rate",
    "osm_pv_rate",
    "detected_pv_rate",
    "annual_flux_mean_kwh_per_kw_yr",
    "nsrdb_ghi_mean",
    "median_household_income_usd",
    "pct_bachelor_plus",
    "pct_owner_occupied",
    "diversity_index",
    "pct_urban_population",
)
DEFAULT_MIN_SAMPLES = int(os.getenv("CASE_STUDY_BG_DBSCAN_MIN_SAMPLES", "5") or "5")
DEFAULT_EPS_QUANTILE = float(os.getenv("CASE_STUDY_BG_DBSCAN_EPS_QUANTILE", "0.9") or "0.9")
COORDINATE_WEIGHT = float(os.getenv("CASE_STUDY_BG_DBSCAN_COORDINATE_WEIGHT", "0.25") or "0.25")

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
MAP_DIR = PROJECT_ROOT / "outputs" / "maps"
CLUSTER_OUTPUT_PATH = REPORT_DIR / f"{TARGET_SCOPE_STEM}_bg_dbscan_clusters.parquet"
SUMMARY_OUTPUT_PATH = REPORT_DIR / f"{TARGET_SCOPE_STEM}_bg_dbscan_summary.csv"
MAP_OUTPUT_PATH = MAP_DIR / f"{TARGET_SCOPE_STEM}_bg_dbscan_clusters.png"


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


def load_bg_cluster_surface(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    has_table = bool(
        con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?;",
            [BG_AGG_TABLE],
        ).fetchone()[0]
    )
    if not has_table:
        raise RuntimeError(
            f"{BG_AGG_TABLE} is missing. Rebuild the block-group aggregate with notebooks/tabular/14_pv_bg_aggregation.py first."
        )

    names_sql = ", ".join("?" * len(TARGET_MUNICIPALITIES))
    frame = con.execute(
        f"""
        SELECT
            agg.*,
            ST_AsWKB(bg.geometry) AS geometry_wkb
        FROM {BG_AGG_TABLE} AS agg
        JOIN pr_census_block_groups AS bg
          ON bg.GEOID = agg.bg_geoid
        WHERE agg.municipio IN ({names_sql})
        ORDER BY agg.municipio, agg.bg_geoid;
        """,
        list(TARGET_MUNICIPALITIES),
    ).fetchdf()
    if frame.empty:
        raise RuntimeError("The BG aggregate exists but returned no rows for San Juan and Isabela.")

    geometry = gpd.GeoSeries.from_wkb(frame["geometry_wkb"].map(_to_bytes), crs="EPSG:4326")
    return gpd.GeoDataFrame(frame.drop(columns=["geometry_wkb"]), geometry=geometry, crs="EPSG:4326")


def resolve_feature_columns(frame: pd.DataFrame) -> list[str]:
    feature_override = os.getenv("CASE_STUDY_BG_DBSCAN_FEATURES", "").strip()
    if feature_override:
        requested = [column_name.strip() for column_name in feature_override.split(",") if column_name.strip()]
        feature_columns = [column_name for column_name in requested if column_name in frame.columns]
    else:
        feature_columns = [column_name for column_name in PREFERRED_FEATURE_COLUMNS if column_name in frame.columns]

    if not feature_columns:
        raise RuntimeError(
            "No BG clustering feature columns were found. Expected one or more of: "
            + ", ".join(PREFERRED_FEATURE_COLUMNS)
        )
    return feature_columns


def build_feature_matrix(
    frame: gpd.GeoDataFrame,
    feature_columns: list[str],
    *,
    coordinate_weight: float = COORDINATE_WEIGHT,
) -> tuple[np.ndarray, list[str]]:
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)

    centroids = frame.to_crs(ANALYSIS_CRS).geometry.centroid
    coordinate_frame = pd.DataFrame(
        {
            "x_coord_km": centroids.x / 1_000.0,
            "y_coord_km": centroids.y / 1_000.0,
        },
        index=frame.index,
    )
    combined = pd.concat([numeric, coordinate_frame], axis=1)
    scaled = pd.DataFrame(
        StandardScaler().fit_transform(combined),
        index=combined.index,
        columns=combined.columns,
    )
    scaled["x_coord_km"] = scaled["x_coord_km"] * coordinate_weight
    scaled["y_coord_km"] = scaled["y_coord_km"] * coordinate_weight
    return scaled.to_numpy(), combined.columns.tolist()


def estimate_dbscan_eps(
    feature_matrix: np.ndarray,
    *,
    min_samples: int,
    quantile: float,
) -> float:
    if len(feature_matrix) <= 1:
        return 0.5

    neighbor_count = max(2, min(int(min_samples), len(feature_matrix)))
    distances, _ = NearestNeighbors(n_neighbors=neighbor_count).fit(feature_matrix).kneighbors(feature_matrix)
    kth_distances = np.sort(distances[:, -1])
    eps = float(np.quantile(kth_distances, quantile))
    if not np.isfinite(eps) or eps <= 0:
        positive_distances = kth_distances[kth_distances > 0]
        eps = float(np.median(positive_distances)) if len(positive_distances) else 0.5
    return max(eps, 0.05)


def run_dbscan_by_municipality(
    bg_gdf: gpd.GeoDataFrame,
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    eps_quantile: float = DEFAULT_EPS_QUANTILE,
    coordinate_weight: float = COORDINATE_WEIGHT,
) -> gpd.GeoDataFrame:
    if bg_gdf.empty:
        return bg_gdf.copy()

    feature_columns = resolve_feature_columns(bg_gdf)
    clustered_frames: list[gpd.GeoDataFrame] = []

    for municipio, subset in bg_gdf.groupby("municipio", sort=False):
        subset = subset.copy()
        if len(subset) < max(3, min_samples):
            subset["cluster_id"] = -1
            subset["cluster_label"] = "noise"
            subset["is_noise"] = True
            subset["cluster_size"] = 0
            subset["dbscan_eps"] = np.nan
            subset["dbscan_min_samples"] = min_samples
            subset["dbscan_feature_columns"] = ",".join(feature_columns)
            clustered_frames.append(subset)
            continue

        feature_matrix, used_columns = build_feature_matrix(
            subset,
            feature_columns,
            coordinate_weight=coordinate_weight,
        )
        eps = estimate_dbscan_eps(feature_matrix, min_samples=min_samples, quantile=eps_quantile)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(feature_matrix)

        subset["cluster_id"] = labels.astype(int)
        subset["cluster_label"] = [
            "noise" if label == -1 else f"{municipio.lower().replace(' ', '_')}_cluster_{label}"
            for label in labels
        ]
        subset["is_noise"] = subset["cluster_id"] == -1
        subset["dbscan_eps"] = eps
        subset["dbscan_min_samples"] = min_samples
        subset["dbscan_feature_columns"] = ",".join(used_columns)
        clustered_frames.append(subset)

    clustered = pd.concat(clustered_frames, ignore_index=True)
    cluster_sizes = (
        clustered.loc[clustered["cluster_id"] >= 0]
        .groupby(["municipio", "cluster_id"], dropna=False)
        .size()
        .rename("cluster_size")
        .reset_index()
    )
    clustered = clustered.drop(columns=["cluster_size"], errors="ignore").merge(
        cluster_sizes,
        on=["municipio", "cluster_id"],
        how="left",
    )
    clustered["cluster_size"] = clustered["cluster_size"].fillna(0).astype(int)
    return gpd.GeoDataFrame(clustered, geometry="geometry", crs=bg_gdf.crs)


def summarize_clusters(clustered: gpd.GeoDataFrame) -> pd.DataFrame:
    if clustered.empty:
        return pd.DataFrame()

    feature_columns = resolve_feature_columns(clustered)
    aggregations: dict[str, tuple[str, str]] = {
        "bg_count": ("bg_geoid", "size"),
        "building_count": ("building_count", "sum"),
        "dbscan_eps": ("dbscan_eps", "first"),
        "dbscan_min_samples": ("dbscan_min_samples", "first"),
    }
    for column_name in feature_columns:
        aggregations[f"mean_{column_name}"] = (column_name, "mean")

    summary = (
        clustered.groupby(["municipio", "cluster_id", "cluster_label", "is_noise"], dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(["municipio", "cluster_id"])
        .reset_index(drop=True)
    )
    return summary


def plot_cluster_map(
    clustered: gpd.GeoDataFrame,
    output_path: Path = MAP_OUTPUT_PATH,
) -> None:
    if clustered.empty:
        print("cluster map skipped: clustered BG surface is empty.")
        return

    fig, axes = plt.subplots(1, len(TARGET_MUNICIPALITIES), figsize=(14, 7), constrained_layout=True)
    if len(TARGET_MUNICIPALITIES) == 1:
        axes = [axes]

    for ax, municipio in zip(axes, TARGET_MUNICIPALITIES):
        subset = clustered[clustered["municipio"] == municipio].copy()
        if subset.empty:
            ax.set_title(f"{municipio}: no data")
            ax.set_axis_off()
            continue

        clustered_polygons = subset[subset["cluster_id"] >= 0]
        if not clustered_polygons.empty:
            clustered_polygons.plot(
                ax=ax,
                column="cluster_id",
                categorical=True,
                cmap="tab20",
                legend=False,
                edgecolor="#2f2f2f",
                linewidth=0.2,
            )

        noise_polygons = subset[subset["cluster_id"] == -1]
        if not noise_polygons.empty:
            noise_polygons.plot(
                ax=ax,
                color="#d9d9d9",
                edgecolor="#666666",
                linewidth=0.2,
            )

        subset.boundary.plot(ax=ax, color="#111111", linewidth=0.25)
        ax.set_title(f"{municipio} BG DBSCAN appendix")
        ax.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    db_path = resolve_db_path()
    con = connect(db_path)
    bg_surface = load_bg_cluster_surface(con)
    con.close()

    feature_columns = resolve_feature_columns(bg_surface)
    print(f"loaded {len(bg_surface):,} BG rows for {', '.join(TARGET_MUNICIPALITIES)}")
    print(f"feature columns: {', '.join(feature_columns)}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)

    clustered_bg = run_dbscan_by_municipality(bg_surface)
    cluster_summary = summarize_clusters(clustered_bg)

    clustered_bg.to_parquet(CLUSTER_OUTPUT_PATH, index=False)
    cluster_summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    plot_cluster_map(clustered_bg, MAP_OUTPUT_PATH)

    print(f"clustered BG parquet: {CLUSTER_OUTPUT_PATH}")
    print(f"cluster summary csv: {SUMMARY_OUTPUT_PATH}")
    if not cluster_summary.empty:
        print(cluster_summary.to_string(index=False))
