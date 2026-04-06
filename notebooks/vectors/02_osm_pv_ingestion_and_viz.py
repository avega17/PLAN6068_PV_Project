"""02_osm_pv_ingestion_and_viz.py

Jupytext-friendly workflow for chunked OpenStreetMap rooftop solar PV polygon
ingestion and story maps for Puerto Rico.

The notebook/script:
- reads Puerto Rico municipality boundaries from the local DuckDB spatial file,
- queries OSM rooftop solar features municipality-by-municipality with
  concurrent futures,
- filters the resulting features down to polygonal rooftop PV geometries,
- stores the final GeoDataFrame back into DuckDB as a native GEOMETRY table,
- exports a macro map and a micro map for the narrative layer.
"""

# %%
from __future__ import annotations

import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import os

import contextily as ctx
import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from shapely.geometry.base import BaseGeometry
from shapely import wkt
from tqdm.auto import tqdm


def resolve_project_root(start: Path | None = None) -> Path:
    """Find the repository root regardless of notebook working directory."""

    current = (start or Path.cwd()).resolve()
    markers = ("project_rules.md", ".git")
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return current


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osm_pv_workers import fetch_osm_features_for_municipality_worker

OUTPUT_CRS = "EPSG:4326"
PLOT_CRS = "EPSG:3857"
DEFAULT_WORKERS = max(1, int(os.getenv("OSM_MAX_WORKERS", str(min(8, os.cpu_count() or 1)))))
EXECUTOR_MODE = os.getenv("OSM_EXECUTOR_MODE", "process").strip().lower()
TARGET_MUNICIPALITY = "San Juan"

load_dotenv(PROJECT_ROOT / ".env")

OSM_TAG_QUERIES: list[dict[str, object]] = [
    {
        "label": "generator_source_solar",
        "tags": {"generator:source": "solar"},
    },
    {
        "label": "generator_method_photovoltaic",
        "tags": {"generator:method": "photovoltaic"},
    },
    {
        "label": "generator_type_solar_panel",
        "tags": {"generator:type": "solar_photovoltaic_panel"},
    },
    {
        "label": "power_generator_with_solar_source",
        "tags": {"power": "generator", "generator:source": "solar"},
    },
    {
        "label": "power_generator_with_pv_method",
        "tags": {"power": "generator", "generator:method": "photovoltaic"},
    },
    {
        "label": "power_plant_solar_source",
        "tags": {"power": "plant", "plant:source": "solar"},
    },
    {
        "label": "power_plant_pv_method",
        "tags": {"power": "plant", "plant:method": "photovoltaic"},
    },
    {
        "label": "rooftop_solar_source",
        "tags": {"location": "roof", "generator:source": "solar"},
    },
]

# %%
def resolve_db_path() -> Path:
    """Resolve the project DuckDB path from VECTOR_DB or the default filename."""

    db_path_value = os.getenv("VECTOR_DB")
    if db_path_value:
        db_path = Path(db_path_value)
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path if len(db_path.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / db_path
        return db_path

    return PROJECT_ROOT / "data" / "vectors" / "PR_vector_data.duckdb"


def resolve_output_dir() -> Path:
    """Resolve the output directory used for saved maps."""

    output_dir = PROJECT_ROOT / "outputs" / "maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_spatial_connection(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection and load the spatial extension."""

    connection = duckdb.connect(str(db_path))
    connection.execute("INSTALL spatial;")
    connection.execute("LOAD spatial;")
    return connection


def load_municipalities(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    """Load Puerto Rico municipalities from DuckDB as a GeoDataFrame."""

    frame = con.execute(
        """
        SELECT
            NAME AS municipality_name,
            GEOID AS municipality_geoid,
            ST_AsText(geometry) AS geometry_wkt
        FROM pr_municipalities
        ORDER BY NAME
        """
    ).fetchdf()

    if frame.empty:
        return gpd.GeoDataFrame(columns=["municipality_name", "municipality_geoid", "geometry"], geometry="geometry", crs=OUTPUT_CRS)

    geometry = frame["geometry_wkt"].map(wkt.loads)
    municipalities = gpd.GeoDataFrame(
        frame.drop(columns=["geometry_wkt"]),
        geometry=geometry,
        crs=OUTPUT_CRS,
    )
    municipalities["geometry_wkt"] = frame["geometry_wkt"]
    return municipalities


def fetch_osm_features_parallel(
    municipalities: gpd.GeoDataFrame,
    max_workers: int = DEFAULT_WORKERS,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Fetch municipality OSM features concurrently with tqdm progress and stats."""

    municipality_items = list(
        municipalities[["municipality_name", "municipality_geoid", "geometry_wkt"]].itertuples(index=False, name=None)
    )
    collected: list[gpd.GeoDataFrame] = []
    stats_records: list[dict[str, Any]] = []

    if not municipality_items:
        empty_gdf = gpd.GeoDataFrame(columns=["municipality_name", "query_label", "geometry"], geometry="geometry", crs=OUTPUT_CRS)
        return empty_gdf, pd.DataFrame()

    worker_count = max(1, min(max_workers, len(municipality_items)))
    executor_description = "process pool" if EXECUTOR_MODE == "process" else "thread pool"
    executor_factory: type[ThreadPoolExecutor | ProcessPoolExecutor]
    executor_factory = ProcessPoolExecutor if EXECUTOR_MODE == "process" else ThreadPoolExecutor

    with executor_factory(max_workers=worker_count, **({"mp_context": mp.get_context("spawn")} if EXECUTOR_MODE == "process" else {})) as executor:
        future_to_context = {
            executor.submit(
                fetch_osm_features_for_municipality_worker,
                municipality_name,
                municipality_geometry_wkt,
                OSM_TAG_QUERIES,
            ): (municipality_name, municipality_geoid)
            for municipality_name, municipality_geoid, municipality_geometry_wkt in municipality_items
        }

        with tqdm(total=len(future_to_context), desc=f"Municipalities queried ({executor_description})", unit="municipality") as progress:
            for future in as_completed(future_to_context):
                municipality_name, municipality_geoid = future_to_context[future]
                try:
                    frame, stats = future.result()
                except Exception as exc:  # pragma: no cover - execution depends on Overpass responses.
                    tqdm.write(f"[ERROR] {municipality_name}: {exc}")
                    stats_records.append(
                        {
                            "municipality_name": municipality_name,
                            "municipality_geoid": municipality_geoid,
                            "query_count": len(OSM_TAG_QUERIES),
                            "matched_query_count": 0,
                            "feature_count": 0,
                            "error_count": 1,
                            "errors": str(exc),
                        }
                    )
                    progress.update(1)
                    continue

                stats["municipality_geoid"] = municipality_geoid
                stats_records.append(stats)
                if not frame.empty:
                    collected.append(frame)
                    tqdm.write(
                        f"[MATCH] {municipality_name}: {stats['feature_count']} polygons "
                        f"({stats['matched_query_count']}/{stats['query_count']} tag queries)"
                    )
                else:
                    tqdm.write(f"[NO MATCH] {municipality_name}")

                if stats["error_count"]:
                    tqdm.write(f"[WARN] {municipality_name}: {stats['errors']}")
                progress.update(1)

    if not collected:
        empty_gdf = gpd.GeoDataFrame(columns=["municipality_name", "query_label", "geometry"], geometry="geometry", crs=OUTPUT_CRS)
        return empty_gdf, pd.DataFrame(stats_records)

    combined = pd.concat(collected, ignore_index=True)
    dedupe_columns = [col for col in ["osmid", "municipality_name", "query_label"] if col in combined.columns]
    if dedupe_columns:
        combined = combined.drop_duplicates(subset=dedupe_columns)
    stats_frame = pd.DataFrame(stats_records).sort_values(by=["feature_count", "municipality_name"], ascending=[False, True])
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=OUTPUT_CRS), stats_frame


def upsert_geodataframe(con: duckdb.DuckDBPyConnection, table_name: str, gdf: gpd.GeoDataFrame) -> None:
    """Store a GeoDataFrame as a native DuckDB GEOMETRY table."""

    if gdf.empty:
        raise ValueError(f"Refusing to write empty GeoDataFrame to {table_name}")

    staged = pd.DataFrame(gdf.copy())
    staged["geometry"] = gdf.geometry.to_wkb()

    con.register("staged_osm", staged)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT
            * EXCLUDE (geometry),
            ST_GeomFromWKB(geometry) AS geometry
        FROM staged_osm;
        """
    )
    con.execute(f"CREATE INDEX idx_{table_name}_geometry ON {table_name} USING RTREE (geometry);")
    con.unregister("staged_osm")


def update_has_pv_flags(con: duckdb.DuckDBPyConnection) -> None:
    """Populate has_PV booleans for municipality/tract/block group geographies."""

    geography_tables = ["pr_municipalities", "pr_census_tracts", "pr_block_groups"]
    for table_name in geography_tables:
        con.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS has_PV BOOLEAN;")
        con.execute(
            f"""
            UPDATE {table_name} AS g
            SET has_PV = EXISTS (
                SELECT 1
                FROM pr_osm_rooftop_pv_polygons AS pv
                WHERE ST_Intersects(g.geometry, pv.geometry)
            );
            """
        )


def plot_macro_map(municipalities: gpd.GeoDataFrame, osm_features: gpd.GeoDataFrame, output_dir: Path) -> Path:
    """Save an island-wide count map of rooftop PV features by municipality."""

    feature_counts = osm_features.groupby("municipality_name").size().rename("feature_count")
    macro = municipalities.merge(feature_counts, on="municipality_name", how="left")
    macro["feature_count"] = macro["feature_count"].fillna(0)

    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    macro.to_crs(OUTPUT_CRS).plot(
        ax=ax,
        column="feature_count",
        cmap="YlOrRd",
        linewidth=0.8,
        edgecolor="#1f2937",
        legend=True,
        legend_kwds={"label": "Detected rooftop PV polygons"},
    )
    ax.set_title("Puerto Rico rooftop solar PV polygons by municipality", pad=16, fontsize=16)
    ax.set_axis_off()

    output_path = output_dir / "02_osm_pv_macro_map.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_micro_map(osm_features: gpd.GeoDataFrame, municipalities: gpd.GeoDataFrame, output_dir: Path) -> Path:
    """Save a zoomed map centered on the San Juan urban area."""

    if TARGET_MUNICIPALITY in municipalities["municipality_name"].values:
        target_name = TARGET_MUNICIPALITY
    else:
        target_name = (
            osm_features.groupby("municipality_name").size().sort_values(ascending=False).index[0]
            if not osm_features.empty
            else municipalities.iloc[0]["municipality_name"]
        )

    target_boundary = municipalities[municipalities["municipality_name"] == target_name].to_crs(PLOT_CRS)
    target_features = osm_features[osm_features["municipality_name"] == target_name].to_crs(PLOT_CRS)

    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    target_boundary.plot(ax=ax, facecolor="none", edgecolor="#f8fafc", linewidth=1.5, zorder=2)
    if not target_features.empty:
        target_features.plot(ax=ax, facecolor="#34d399", edgecolor="#064e3b", alpha=0.7, linewidth=0.4, zorder=3)

    bounds = target_boundary.total_bounds
    padding_x = (bounds[2] - bounds[0]) * 0.18 or 250
    padding_y = (bounds[3] - bounds[1]) * 0.18 or 250
    ax.set_xlim(bounds[0] - padding_x, bounds[2] + padding_x)
    ax.set_ylim(bounds[1] - padding_y, bounds[3] + padding_y)

    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs=PLOT_CRS, alpha=1.0)
    ax.set_title(f"{target_name} rooftop solar PV polygons on satellite imagery", pad=16, fontsize=16)
    ax.set_axis_off()

    output_path = output_dir / "02_osm_pv_micro_map_san_juan.png"
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


# %%
# Step 1: initialize paths + connection.
DB_PATH = resolve_db_path()
OUTPUT_DIR = resolve_output_dir()
con = create_spatial_connection(DB_PATH)
print(f"Connected to vector database: {DB_PATH}")


# %%
# Step 2: load municipality boundaries.
municipalities = load_municipalities(con)
print(f"Loaded municipalities: {len(municipalities)}")


# %%
# Step 3: fetch OSM rooftop PV polygons with parallel municipality queries.
osm_features, municipality_fetch_stats = fetch_osm_features_parallel(
    municipalities,
    max_workers=DEFAULT_WORKERS,
)
print(f"Detected rooftop PV polygons: {len(osm_features)}")
if not municipality_fetch_stats.empty:
    print(municipality_fetch_stats.head(15).to_string(index=False))


# %%
# Step 4: persist OSM geometries and update has_PV indicators.
if osm_features.empty:
    raise RuntimeError("No rooftop PV polygons were returned from OSM for the tested municipalities")

upsert_geodataframe(con, "pr_osm_rooftop_pv_polygons", osm_features)
update_has_pv_flags(con)
print("Created table: pr_osm_rooftop_pv_polygons")
print("Updated has_PV flags in: pr_municipalities, pr_census_tracts, pr_block_groups")


# %%
# Step 5: generate narrative outputs.
macro_path = plot_macro_map(municipalities, osm_features, OUTPUT_DIR)
micro_path = plot_micro_map(osm_features, municipalities, OUTPUT_DIR)
print(f"Saved macro map to: {macro_path}")
print(f"Saved micro map to: {micro_path}")


# %%
# Optional: close DB handle at the end of the notebook run.
con.close()
print("Closed DuckDB connection")
