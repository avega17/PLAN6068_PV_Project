# %% [markdown]
# # Puerto Rico Rooftop Solar PV Ingestion Pipeline
# 
# This workflow orchestrates the extraction and processing of OpenStreetMap (OSM) data to identify rooftop solar photovoltaics (PV) across Puerto Rico.
# 
# **Our Technical Approach**:
# 1. **Geometry Filtering**: We query `osmnx` for the high-fidelity boundary of 'Puerto Rico' to constrain our search space.
# 2. **Direct QuackOSM -> DuckDB Loading**: We use `quackosm.convert_geometry_to_duckdb()` to download matching OSM extracts and materialize rows directly into our DuckDB file and staging table.
# 3. **DuckDB Spatial Cleaning**: We apply SQL filters for rooftop-PV semantics (including content exclusion for hot-water systems) and keep polygonal geometries only.
# 4. **EDA + Mapping**: We summarize feature counts/areas and block coverage, then render a macro static map and a micro interactive Folium map using a contextily provider URL.

# %%
"""02_osm_pv_ingestion_and_viz.py

Jupytext-friendly workflow for island-wide OpenStreetMap rooftop solar PV
ingestion and narrative maps for Puerto Rico using QuackOSM + DuckDB.

The notebook/script:
- reads Puerto Rico municipality boundaries from the local DuckDB spatial file,
- geocodes Puerto Rico with osmnx and uses that geometry as QuackOSM filter,
- ingests filtered OSM data directly into a DuckDB staging table,
- uses vectorized DuckDB SQL to filter non-PV records and normalize GEOMETRY,
- updates has_PV flags and exports macro/micro narrative map artifacts.
"""

# %%
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
import os

import contextily as ctx
import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import quackosm as qosm
import folium
from folium.plugins import MarkerCluster
from dotenv import load_dotenv
from IPython.display import IFrame, Image, display
from shapely.geometry.base import BaseGeometry
from shapely import wkt


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

load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_CRS = "EPSG:4326"
PLOT_CRS = "EPSG:3857"
PV_TABLE_NAME = "pr_osm_rooftop_pv_polygons"
PV_STAGE_TABLE_NAME = "pr_osm_quackosm_stage"
TARGET_MUNICIPALITY = "San Juan"

QUACKOSM_TAGS_FILTER: dict[str, object] = {
    "power": "generator",
    "generator:source": "solar",
    "generator:type": "solar_photovoltaic_panel",
}
OSM_EXTRACT_SOURCE = os.getenv("OSM_EXTRACT_SOURCE", "any").strip() or "any"


def slugify(value: str) -> str:
    """Create a stable filename-safe slug from an arbitrary label."""

    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "unknown"

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


def resolve_cache_dir() -> Path:
    """Resolve QuackOSM working directory for temporary extraction artifacts."""

    cache_dir = PROJECT_ROOT / "data" / "vectors" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def create_spatial_connection(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection and load the spatial extension."""

    connection = duckdb.connect(str(db_path))
    connection.execute("INSTALL spatial;")
    connection.execute("LOAD spatial;")
    try:
        connection.execute("INSTALL parquet;")
        connection.execute("LOAD parquet;")
    except duckdb.Error:
        # Parquet support is bundled in many DuckDB builds.
        pass
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


def resolve_puerto_rico_geometry() -> BaseGeometry:
    """Fetch Puerto Rico boundary geometry using osmnx geocoding."""
    print("Initiating osmnx geocoding for Puerto Rico geometry...")

    puerto_rico_gdf = ox.geocode_to_gdf('Puerto Rico')
    if puerto_rico_gdf.empty:
        raise RuntimeError("osmnx.geocode_to_gdf('Puerto Rico') returned no boundary geometry")

    print(f"Successfully fetched Puerto Rico boundary. Original CRS: {puerto_rico_gdf.crs}")
    
    if puerto_rico_gdf.crs is None:
        puerto_rico_gdf = puerto_rico_gdf.set_crs(OUTPUT_CRS)
    else:
        puerto_rico_gdf = puerto_rico_gdf.to_crs(OUTPUT_CRS)

    print(f"Standardized Puerto Rico boundary CRS: {puerto_rico_gdf.crs}")
    
    pr_geom = puerto_rico_gdf.geometry.union_all()
    # Ensuring valid shapely geometry with no latent CRS binding properties attached to object
    print(f"Geometry filter type resolved: {pr_geom.geom_type}")
    
    return pr_geom


def count_features_in_table(con: duckdb.DuckDBPyConnection, table_name: str) -> int:
    """Count features from a DuckDB table."""

    return con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]


def list_db_tables(con: duckdb.DuckDBPyConnection):
    """List user tables in the active DuckDB database."""

    return con.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
        """
    ).fetchdf()


def preview_table_schema(con: duckdb.DuckDBPyConnection, table_name: str):
    """Return DESCRIBE output for a table."""

    return con.execute(f"DESCRIBE {table_name}").fetchdf()


def preview_table_rows(con: duckdb.DuckDBPyConnection, table_name: str, limit: int = 5):
    """Return a small sample of table rows."""

    return con.execute(f"FROM {table_name} LIMIT {limit}").fetchdf()


def preview_geometry_samples(gdf: gpd.GeoDataFrame, n: int = 5):
    """Return a small sample of WKT geometry strings for quick visual inspection."""

    sample = gdf.head(n).copy()
    sample["geometry_wkt"] = sample.geometry.to_wkt()
    keep = [col for col in ["feature_id", "municipality_name", "geometry_wkt"] if col in sample.columns]
    return sample[keep]


def run_extraction(
    geometry_filter: BaseGeometry,
    db_path: Path,
    cache_dir: Path,
    stage_table_name: str = PV_STAGE_TABLE_NAME,
) -> dict[str, object]:
    """Run QuackOSM extraction directly into DuckDB and return timing metadata."""
    print("Running QuackOSM extraction via convert_geometry_to_duckdb...")
    print(f"Target DuckDB file: {db_path}")
    print(f"Target staging table: {stage_table_name}")
    print(f"Working/cache directory: {cache_dir}")
    print(f"OSM extract source selection: {OSM_EXTRACT_SOURCE}")
    print(f"QuackOSM tags filter: {QUACKOSM_TAGS_FILTER}")

    # Ensure shape is a strictly plain BaseGeometry without unexpected bindings:
    active_filter = geometry_filter.buffer(0)
    if not active_filter.is_valid:
        raise RuntimeError("Resolved Puerto Rico geometry is invalid after normalization")

    started = time.perf_counter()
    duckdb_result_path = qosm.convert_geometry_to_duckdb(
        geometry_filter=active_filter,
        tags_filter=QUACKOSM_TAGS_FILTER,
        osm_extract_source=OSM_EXTRACT_SOURCE,
        result_file_path=str(db_path),
        duckdb_table_name=stage_table_name,
        working_directory=str(cache_dir),
        ignore_cache=False,
        verbosity_mode="verbose",
    )
    elapsed = time.perf_counter() - started
    
    print(f"QuackOSM extraction complete in {elapsed:.2f} seconds.")
    print(f"Updated DuckDB artifact: {duckdb_result_path}")
    
    return {
        "scope": "island_wide",
        "strategy": "quackosm_geometry_to_duckdb",
        "elapsed_seconds": round(elapsed, 2),
        "duckdb_path": str(duckdb_result_path),
        "stage_table": stage_table_name,
    }


def create_clean_pv_table(
    con: duckdb.DuckDBPyConnection,
    stage_table_name: str = PV_STAGE_TABLE_NAME,
    table_name: str = PV_TABLE_NAME,
) -> None:
    """Create cleaned rooftop PV geometries table from QuackOSM stage table."""

    print(f"Building cleaned PV table from stage table: {stage_table_name}")

    stage_schema = con.execute(f"PRAGMA table_info('{stage_table_name}')").fetchdf()
    stage_columns = set(stage_schema["name"].tolist())

    def stage_text_expr(column_name: str) -> str:
        if column_name in stage_columns:
            return f"lower(coalesce(CAST(\"{column_name}\" AS VARCHAR), ''))"
        return "''"

    if "tags" in stage_columns:
        power_expr = f"lower(coalesce(CAST(tags['power'] AS VARCHAR), {stage_text_expr('power')}))"
        source_expr = f"lower(coalesce(CAST(tags['generator:source'] AS VARCHAR), {stage_text_expr('generator:source')}))"
        type_expr = f"lower(coalesce(CAST(tags['generator:type'] AS VARCHAR), {stage_text_expr('generator:type')}))"
        content_expr = f"lower(coalesce(CAST(tags['content'] AS VARCHAR), {stage_text_expr('content')}))"
        tags_select_expr = "tags"
    else:
        power_expr = stage_text_expr("power")
        source_expr = stage_text_expr("generator:source")
        type_expr = stage_text_expr("generator:type")
        content_expr = stage_text_expr("content")
        tags_select_expr = "map_from_entries([]::STRUCT(k VARCHAR, v VARCHAR)[]) AS tags"

    print(f"Detected QuackOSM stage columns: {', '.join(sorted(stage_columns))}")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table_name} AS
        WITH staged AS (
            SELECT
                feature_id,
                {tags_select_expr},
                CAST(geometry AS GEOMETRY) AS geometry,
                {content_expr} AS content_value,
                {power_expr} AS power_value,
                {source_expr} AS generator_source_value,
                {type_expr} AS generator_type_value
            FROM {stage_table_name}
        ),
        filtered AS (
            SELECT *
            FROM staged
            WHERE content_value <> 'hot_water'
              AND (
                    power_value = 'generator'
                 OR generator_source_value = 'solar'
                 OR generator_type_value = 'solar_photovoltaic_panel'
              )
              AND ST_GeometryType(geometry) IN ('POLYGON', 'MULTIPOLYGON')
        ),
        municipality_ranked AS (
            SELECT
                f.feature_id,
                f.tags,
                f.geometry,
                m.NAME AS municipality_name,
                m.GEOID AS municipality_geoid,
                ROW_NUMBER() OVER (
                    PARTITION BY f.feature_id
                    ORDER BY ST_Area(ST_Intersection(m.geometry, f.geometry)) DESC NULLS LAST
                ) AS municipality_rank
            FROM filtered AS f
            LEFT JOIN pr_municipalities AS m
              ON ST_Intersects(m.geometry, f.geometry)
        )
        SELECT
            feature_id,
            municipality_name,
            municipality_geoid,
            tags,
            geometry
        FROM municipality_ranked
        WHERE municipality_rank = 1;
        """
    )
    con.execute(f"DROP INDEX IF EXISTS idx_{table_name}_geometry;")
    con.execute(f"CREATE INDEX idx_{table_name}_geometry ON {table_name} USING RTREE (geometry);")
    print(f"Created cleaned table and RTree index: {table_name}")


def summarize_municipality_pv(con: duckdb.DuckDBPyConnection, table_name: str = PV_TABLE_NAME):
    """Return municipality-level PV feature counts and area totals in m2."""

    return con.execute(
        f"""
        SELECT
            coalesce(pv.municipality_name, m.NAME) AS municipality_name,
            COUNT(*) AS pv_feature_count,
            SUM(ST_Area(ST_Transform(pv.geometry, 'EPSG:4326', 'EPSG:3857'))) AS pv_area_m2
        FROM {table_name} AS pv
        LEFT JOIN pr_municipalities AS m
          ON pv.municipality_geoid = m.GEOID
        GROUP BY 1
        ORDER BY pv_feature_count DESC, municipality_name
        """
    ).fetchdf()


def summarize_pv_area_stats(con: duckdb.DuckDBPyConnection, table_name: str = PV_TABLE_NAME):
    """Return descriptive statistics of rooftop PV polygon area (m2)."""

    return con.execute(
        f"""
        WITH areas AS (
            SELECT ST_Area(ST_Transform(geometry, 'EPSG:4326', 'EPSG:3857')) AS area_m2
            FROM {table_name}
        )
        SELECT
            COUNT(*) AS polygon_count,
            MIN(area_m2) AS min_area_m2,
            AVG(area_m2) AS mean_area_m2,
            quantile_cont(area_m2, 0.5) AS median_area_m2,
            quantile_cont(area_m2, 0.9) AS p90_area_m2,
            MAX(area_m2) AS max_area_m2
        FROM areas
        """
    ).fetchdf()


def summarize_block_coverage(con: duckdb.DuckDBPyConnection):
    """Return census block group and tract PV-coverage percentages."""

    return con.execute(
        """
        SELECT 'block_groups' AS geography, COUNT(*) AS total_units, SUM(CASE WHEN has_PV THEN 1 ELSE 0 END) AS pv_units,
               100.0 * SUM(CASE WHEN has_PV THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS pct_with_pv
        FROM pr_block_groups
        UNION ALL
        SELECT 'census_tracts' AS geography, COUNT(*) AS total_units, SUM(CASE WHEN has_PV THEN 1 ELSE 0 END) AS pv_units,
               100.0 * SUM(CASE WHEN has_PV THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS pct_with_pv
        FROM pr_census_tracts
        """
    ).fetchdf()


def plot_eda_summary(municipality_summary, output_dir: Path) -> Path:
    """Save an EDA bar chart of top municipalities by PV feature count."""

    top_muni = municipality_summary.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.barh(top_muni["municipality_name"], top_muni["pv_feature_count"], color="#16a34a")
    ax.set_title("Top 15 municipalities by rooftop PV OSM feature count", pad=12, fontsize=13)
    ax.set_xlabel("PV feature count")
    output_path = output_dir / "pv_eda_top15_municipalities.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


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
                FROM {PV_TABLE_NAME} AS pv
                WHERE ST_Intersects(g.geometry, pv.geometry)
            );
            """
        )


def load_pv_features(con: duckdb.DuckDBPyConnection, table_name: str = PV_TABLE_NAME) -> gpd.GeoDataFrame:
    """Load cleaned rooftop PV polygons from DuckDB as a GeoDataFrame."""

    frame = con.execute(
        f"""
        SELECT
            feature_id,
            municipality_name,
            municipality_geoid,
            ST_AsText(geometry) AS geometry_wkt
        FROM {table_name}
        """
    ).fetchdf()

    if frame.empty:
        return gpd.GeoDataFrame(
            columns=["feature_id", "municipality_name", "municipality_geoid", "geometry"],
            geometry="geometry",
            crs=OUTPUT_CRS,
        )

    geometry = frame["geometry_wkt"].map(wkt.loads)
    return gpd.GeoDataFrame(
        frame.drop(columns=["geometry_wkt"]),
        geometry=geometry,
        crs=OUTPUT_CRS,
    )


def plot_macro_map(municipalities: gpd.GeoDataFrame, osm_features: gpd.GeoDataFrame, output_dir: Path) -> Path:
    """Save an island-wide count map using log-scaled municipality counts to reduce skew."""

    feature_counts = osm_features.groupby("municipality_name").size().rename("feature_count")
    macro = municipalities.merge(feature_counts, on="municipality_name", how="left")
    macro["feature_count"] = macro["feature_count"].fillna(0)
    macro["feature_count_log1p"] = np.log1p(macro["feature_count"])

    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    macro.to_crs(OUTPUT_CRS).plot(
        ax=ax,
        column="feature_count_log1p",
        cmap="YlOrRd",
        linewidth=0.8,
        edgecolor="#1f2937",
        legend=True,
        legend_kwds={"label": "log(1 + detected rooftop PV polygons)"},
    )
    ax.set_title("Puerto Rico rooftop solar PV polygons by municipality (log scale)", pad=16, fontsize=16)
    ax.set_axis_off()

    output_path = output_dir / "pr_macro_pv_map.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_micro_map(osm_features: gpd.GeoDataFrame, municipalities: gpd.GeoDataFrame, output_dir: Path) -> Path:
    """Build and save an interactive Folium micro-map using a contextily provider tile URL."""

    if TARGET_MUNICIPALITY in municipalities["municipality_name"].values:
        target_name = TARGET_MUNICIPALITY
    else:
        target_name = (
            osm_features.groupby("municipality_name").size().sort_values(ascending=False).index[0]
            if not osm_features.empty
            else municipalities.iloc[0]["municipality_name"]
        )

    target_boundary = municipalities[municipalities["municipality_name"] == target_name]
    target_features = osm_features[osm_features["municipality_name"] == target_name]

    provider = ctx.providers.Esri.WorldImagery
    tile_url = provider.build_url()
    tile_attr = provider.get("html_attribution") or provider.get("attribution") or ""

    centroid = target_boundary.geometry.union_all().centroid
    micro_map = folium.Map(location=[centroid.y, centroid.x], zoom_start=13, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles=tile_url,
        attr=tile_attr,
        name="Esri WorldImagery",
        overlay=False,
        control=True,
        max_zoom=provider.get("max_zoom", 20),
    ).add_to(micro_map)

    folium.GeoJson(
        target_boundary.__geo_interface__,
        name="Municipality boundary",
        style_function=lambda _: {"fillColor": "#00000000", "color": "#f8fafc", "weight": 2},
    ).add_to(micro_map)

    feature_layer = folium.FeatureGroup(name="PV polygons", show=True)
    folium.GeoJson(
        target_features.__geo_interface__,
        style_function=lambda _: {"fillColor": "#34d399", "color": "#065f46", "weight": 1, "fillOpacity": 0.65},
    ).add_to(feature_layer)
    feature_layer.add_to(micro_map)

    marker_cluster = MarkerCluster(name="PV centroids", overlay=True, control=True).add_to(micro_map)
    target_features_3857 = target_features.to_crs(PLOT_CRS)
    for idx, row in target_features.iterrows():
        centroid_point = row.geometry.centroid
        area_m2 = float(target_features_3857.loc[idx].geometry.area)
        popup_text = f"feature_id: {row.feature_id}<br>area_m2: {area_m2:,.1f}"
        folium.Marker(
            location=[centroid_point.y, centroid_point.x],
            popup=folium.Popup(popup_text, max_width=280),
            icon=folium.Icon(color="green", icon="bolt", prefix="fa"),
        ).add_to(marker_cluster)

    folium.LayerControl(collapsed=False).add_to(micro_map)

    output_path = output_dir / "puerto_nuevo_pv.html"
    micro_map.save(str(output_path))
    return micro_map

# %%
# %% [markdown]
# ## Execution Steps
# 1. Connect to the project DuckDB and load municipality boundaries.
# 2. Resolve Puerto Rico geometry with OSMnx and ingest matching OSM features directly into DuckDB via QuackOSM.
# 3. Build cleaned PV polygons from the QuackOSM stage table using SQL tag logic.
# 4. Update `has_PV` flags, run EDA summaries, and export macro + interactive micro maps.

# %%
# Step 1: initialize paths + connection.
DB_PATH = resolve_db_path()
OUTPUT_DIR = resolve_output_dir()
CACHE_DIR = resolve_cache_dir()
con = create_spatial_connection(DB_PATH)
print(f"Connected to vector database: {DB_PATH}")
tables_before_quackosm = list_db_tables(con)
print("Tables available before QuackOSM ingestion:")
print(tables_before_quackosm.to_string(index=False))

# %%
# Step 2: load municipality boundaries.
municipalities = load_municipalities(con)
print(f"Loaded municipalities: {len(municipalities)}")

# %%
# Step 3: fetch Puerto Rico geometry and run island-wide QuackOSM extraction.
puerto_rico_geometry = resolve_puerto_rico_geometry()
island_run = run_extraction(
    puerto_rico_geometry,
    DB_PATH,
    CACHE_DIR,
)
island_feature_count = count_features_in_table(con, island_run["stage_table"])
island_run["feature_count"] = island_feature_count
print(f"Island-wide staging table: {island_run['stage_table']}")
print(f"Island-wide feature count before SQL cleaning (QuackOSM stage rows already prefiltered by tags_filter): {island_feature_count}")
print(f"Island-wide extraction elapsed: {island_run['elapsed_seconds']}s")
tables_after_quackosm = list_db_tables(con)
print("\nTables available after QuackOSM ingestion:")
print(tables_after_quackosm.to_string(index=False))
print("\nStage table schema preview:")
print(preview_table_schema(con, island_run["stage_table"]).to_string(index=False))
print("\nStage table sample rows:")
print(preview_table_rows(con, island_run["stage_table"], limit=5).to_string(index=False))


# %%
# Step 4: clean + persist PV geometries in DuckDB and update has_PV indicators.
create_clean_pv_table(
    con,
    stage_table_name=island_run["stage_table"],
    table_name=PV_TABLE_NAME,
)
osm_features = load_pv_features(con, table_name=PV_TABLE_NAME)
if osm_features.empty:
    raise RuntimeError("No rooftop PV polygons were produced after QuackOSM ingest and SQL cleaning")

update_has_pv_flags(con)
print(f"Created table: {PV_TABLE_NAME}")
print("Updated has_PV flags in: pr_municipalities, pr_census_tracts, pr_block_groups")
print(f"Detected rooftop PV polygons after SQL cleaning: {len(osm_features)}")
print("\nCleaned PV table schema preview:")
print(preview_table_schema(con, PV_TABLE_NAME).to_string(index=False))
print("\nCleaned PV table sample rows:")
print(preview_table_rows(con, PV_TABLE_NAME, limit=5).to_string(index=False))
print("\nSample Shapely geometries from cleaned PV features:")
print(preview_geometry_samples(osm_features, n=5).to_string(index=False))

# %%
# Step 4b: EDA summaries of cleaned rooftop PV features.
municipality_summary = summarize_municipality_pv(con, table_name=PV_TABLE_NAME)
area_stats = summarize_pv_area_stats(con, table_name=PV_TABLE_NAME)
coverage_stats = summarize_block_coverage(con)
eda_plot_path = plot_eda_summary(municipality_summary, OUTPUT_DIR)

print("Municipality PV summary (top 10):")
print(municipality_summary.head(10).to_string(index=False))
print("\nArea statistics (m^2):")
print(area_stats.to_string(index=False))
print("\nCoverage statistics (% with at least one PV feature):")
print(coverage_stats.to_string(index=False))
print(f"Saved EDA bar plot to: {eda_plot_path}")
display(Image(filename=str(eda_plot_path)))

# %%
# Step 5: generate narrative outputs.
macro_path = plot_macro_map(municipalities, osm_features, OUTPUT_DIR)
micro_map = plot_micro_map(osm_features, municipalities, OUTPUT_DIR)
micro_path = OUTPUT_DIR / "puerto_nuevo_pv.html"
print(f"Saved macro map to: {macro_path}")
print(f"Saved micro map to: {micro_path}")
display(Image(filename=str(macro_path)))

# %%
# Step 5b: display interactive Folium map inline in notebook.
micro_map

# %%
# Optional: close DB handle at the end of the notebook run.
con.close()
print("Closed DuckDB connection")


