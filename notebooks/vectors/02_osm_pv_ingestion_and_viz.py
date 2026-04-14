# %% [markdown]
# # Puerto Rico Rooftop Solar PV Ingestion Pipeline
# 
# This workflow orchestrates the extraction and processing of OpenStreetMap (OSM) data to identify rooftop solar photovoltaics (PV) across Puerto Rico.
# 
# **Our Technical Approach**:
# 1. **Geometry Filtering**: We query `osmnx` for the high-fidelity boundary of 'Puerto Rico' to constrain our search space.
# 2. **Direct QuackOSM -> DuckDB Loading**: We use `quackosm.convert_geometry_to_duckdb()` to download matching OSM extracts and materialize rows directly into DuckDB staging tables.
# 3. **Expanded Stage Schema**: We set `keep_all_tags=True`, `explode_tags=True`, and `ignore_metadata_tags=False` to preserve all available tags plus metadata fields such as user and timestamp.
# 4. **Dual Staging Targets**: We ingest one stage for rooftop-PV features and a second stage for larger solar-plant features (`plant:method=photovoltaic` OR `plant:source=solar`).
# 5. **Local Metadata Enrichment**: We enrich metadata from a cached local `.osm.pbf` file for faster throughput than remote API calls.
# 6. **DuckDB Spatial Cleaning + EDA + Mapping**: We clean rooftop PV polygons with SQL, summarize counts/areas/coverage, and render macro + interactive micro maps.

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
import pandas as pd
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
SOLAR_PLANT_STAGE_TABLE_NAME = "pr_osm_solar_plant_stage"
TARGET_MUNICIPALITY = "San Juan"

QUACKOSM_TAGS_FILTER: dict[str, object] = {
    "generator:method": "photovoltaic",
    "generator:source": "solar",
    "generator:type": "solar_photovoltaic_panel",
}
SOLAR_PLANT_TAGS_FILTER: dict[str, object] = {
    "plant:method": "photovoltaic",
    "plant:source": "solar",
}
QUACKOSM_KEEP_ALL_TAGS = True
QUACKOSM_EXPLODE_TAGS = True
QUACKOSM_IGNORE_METADATA_TAGS = False
OSM_EXTRACT_SOURCE = os.getenv("OSM_EXTRACT_SOURCE", "any").strip() or "any"

PV_FILTER_TAG_COLUMNS = ["generator:method", "generator:source", "generator:type"]
PV_METADATA_COLUMNS = ["user", "timestamp", "uid", "version", "changeset"]
LOCAL_METADATA_UPDATE_CHUNK_SIZE = 5000


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


def resolve_cached_osm_pbf(cache_dir: Path) -> Path | None:
    """Resolve a local cached OSM .osm.pbf path for fast metadata enrichment."""

    explicit = os.getenv("OSM_PBF_PATH", "").strip()
    if explicit != '':
        explicit_path = Path(explicit)
        if explicit_path.exists():
            return explicit_path

    candidates = sorted(cache_dir.glob("*.osm.pbf"))
    if not candidates:
        return None

    preferred = [
        path
        for path in candidates
        if "puerto_rico" in path.name.lower() or "puerto-rico" in path.name.lower()
    ]
    return preferred[0] if preferred else candidates[0]


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
    tags_filter: dict[str, object],
    stage_table_name: str = PV_STAGE_TABLE_NAME,
    keep_all_tags: bool = QUACKOSM_KEEP_ALL_TAGS,
    explode_tags: bool = QUACKOSM_EXPLODE_TAGS,
    ignore_metadata_tags: bool = QUACKOSM_IGNORE_METADATA_TAGS,
) -> dict[str, object]:
    """Run QuackOSM extraction directly into DuckDB and return timing metadata."""
    print("Running QuackOSM extraction via convert_geometry_to_duckdb...")
    print(f"Target DuckDB file: {db_path}")
    print(f"Target staging table: {stage_table_name}")
    print(f"Working/cache directory: {cache_dir}")
    print(f"OSM extract source selection: {OSM_EXTRACT_SOURCE}")
    print(f"QuackOSM tags filter: {tags_filter}")
    print(f"keep_all_tags={keep_all_tags}, explode_tags={explode_tags}, ignore_metadata_tags={ignore_metadata_tags}")

    # Ensure shape is a strictly plain BaseGeometry without unexpected bindings:
    active_filter = geometry_filter.buffer(0)
    if not active_filter.is_valid:
        raise RuntimeError("Resolved Puerto Rico geometry is invalid after normalization")

    started = time.perf_counter()
    duckdb_result_path = qosm.convert_geometry_to_duckdb(
        geometry_filter=active_filter,
        tags_filter=tags_filter,
        osm_extract_source=OSM_EXTRACT_SOURCE,
        result_file_path=str(db_path),
        keep_all_tags=keep_all_tags,
        explode_tags=explode_tags,
        ignore_metadata_tags=ignore_metadata_tags,
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
        "tags_filter": tags_filter,
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

    def quoted_identifier(column_name: str) -> str:
        return f'"{column_name.replace("\"", "\"\"")}"'

    def stage_text_expr(column_name: str) -> str:
        if column_name in stage_columns:
            return f"lower(coalesce(CAST({quoted_identifier(column_name)} AS VARCHAR), ''))"
        return "''"

    method_expr = stage_text_expr("generator:method")
    source_expr = stage_text_expr("generator:source")
    type_expr = stage_text_expr("generator:type")
    content_expr = stage_text_expr("content")
    hot_water_output_expr = stage_text_expr("generator:output:hot_water")

    keep_stage_columns = [
        column_name
        for column_name in [*PV_FILTER_TAG_COLUMNS, *PV_METADATA_COLUMNS]
        if column_name in stage_columns
    ]

    staged_select_items = [
        "feature_id",
        "CAST(geometry AS GEOMETRY) AS geometry",
        f"{content_expr} AS content_value",
        f"{method_expr} AS generator_method_value",
        f"{source_expr} AS generator_source_value",
        f"{type_expr} AS generator_type_value",
        f"{hot_water_output_expr} AS generator_output_hot_water_value",
        *[f"{quoted_identifier(column_name)} AS {quoted_identifier(column_name)}" for column_name in keep_stage_columns],
    ]
    municipality_ranked_items = [
        "f.feature_id",
        "f.geometry",
        *[f"f.{quoted_identifier(column_name)} AS {quoted_identifier(column_name)}" for column_name in keep_stage_columns],
        "m.NAME AS municipality_name",
        "m.GEOID AS municipality_geoid",
        "ROW_NUMBER() OVER (PARTITION BY f.feature_id ORDER BY ST_Area(ST_Intersection(m.geometry, f.geometry)) DESC NULLS LAST) AS municipality_rank",
    ]
    final_select_items = [
        "feature_id",
        "municipality_name",
        "municipality_geoid",
        *[quoted_identifier(column_name) for column_name in keep_stage_columns],
        "geometry",
    ]

    staged_select_sql = ",\n                ".join(staged_select_items)
    municipality_ranked_sql = ",\n                ".join(municipality_ranked_items)
    final_select_sql = ",\n            ".join(final_select_items)

    print(f"Detected QuackOSM stage columns: {', '.join(sorted(stage_columns))}")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table_name} AS
        WITH staged AS (
            SELECT
                {staged_select_sql}
            FROM {stage_table_name}
        ),
        filtered AS (
            SELECT *
            FROM staged
            WHERE content_value NOT IN ('hot_water', 'hot', 'water')
              AND generator_output_hot_water_value <> 'yes'
              AND (
                    generator_method_value = 'photovoltaic'
                 OR generator_source_value = 'solar'
                 OR generator_type_value = 'solar_photovoltaic_panel'
              )
              AND ST_GeometryType(geometry) IN ('POLYGON', 'MULTIPOLYGON')
        ),
        municipality_ranked AS (
            SELECT
                {municipality_ranked_sql}
            FROM filtered AS f
            LEFT JOIN pr_municipalities AS m
              ON ST_Intersects(m.geometry, f.geometry)
        )
        SELECT
            {final_select_sql}
        FROM municipality_ranked
        WHERE municipality_rank = 1;
        """
    )
    con.execute(f"DROP INDEX IF EXISTS idx_{table_name}_geometry;")
    con.execute(f"CREATE INDEX idx_{table_name}_geometry ON {table_name} USING RTREE (geometry);")
    print(f"Created cleaned table and RTree index: {table_name}")


def enrich_stage_metadata_from_local_pbf(
    con: duckdb.DuckDBPyConnection,
    stage_table_name: str,
    pbf_path: Path | None,
    update_chunk_size: int = LOCAL_METADATA_UPDATE_CHUNK_SIZE,
) -> None:
    """Fetch OSM metadata from a local cached PBF file and update stage table metadata fields."""

    if pbf_path is None or not pbf_path.exists():
        print(f"No local .osm.pbf available for metadata enrichment in: {stage_table_name}")
        return

    feature_df = con.execute(
        f"""
        SELECT feature_id
        FROM {stage_table_name}
        WHERE feature_id LIKE 'way/%'
        """
    ).fetchdf()
    if feature_df.empty:
        print(f"No way features found to enrich in: {stage_table_name}")
        return

    way_ids = sorted(
        {
            int(feature_id.split("/", 1)[1])
            for feature_id in feature_df["feature_id"].dropna().tolist()
            if isinstance(feature_id, str) and feature_id.startswith("way/") and feature_id.split("/", 1)[1].isdigit()
        }
    )
    if not way_ids:
        print(f"No valid way IDs found for metadata enrichment in: {stage_table_name}")
        return

    try:
        import osmium  # type: ignore
    except ImportError:
        print(
            "pyosmium is not installed; skipping local metadata enrichment. "
            "Install with: uv pip install pyosmium"
        )
        return

    target_ids = set(way_ids)
    updates: list[dict[str, str | int | None]] = []

    class WayMetadataHandler(osmium.SimpleHandler):
        def __init__(self, selected_ids: set[int]):
            super().__init__()
            self.selected_ids = selected_ids

        def way(self, way):
            way_id = int(way.id)
            if way_id not in self.selected_ids:
                return

            updates.append(
                {
                    "way_id": way_id,
                    "user": getattr(way, "user", None),
                    "timestamp": str(getattr(way, "timestamp", "")) if getattr(way, "timestamp", None) else None,
                    "uid": str(getattr(way, "uid", "")) if getattr(way, "uid", None) is not None else None,
                    "version": str(getattr(way, "version", "")) if getattr(way, "version", None) is not None else None,
                    "changeset": str(getattr(way, "changeset", "")) if getattr(way, "changeset", None) is not None else None,
                }
            )

    print(
        f"Reading local OSM metadata for {len(way_ids)} way features from {pbf_path.name} "
        f"into {stage_table_name}"
    )
    handler = WayMetadataHandler(target_ids)
    handler.apply_file(str(pbf_path), locations=False)

    if not updates:
        print(f"Local PBF parsing returned no metadata updates for: {stage_table_name}")
        return

    metadata_columns = ["user", "timestamp", "uid", "version", "changeset"]
    for column_name in metadata_columns:
        con.execute(f'ALTER TABLE {stage_table_name} ADD COLUMN IF NOT EXISTS "{column_name}" VARCHAR;')

    metadata_updates = pd.DataFrame(updates).drop_duplicates(subset=["way_id"])
    total_rows = len(metadata_updates)

    for start in range(0, total_rows, update_chunk_size):
        chunk = metadata_updates.iloc[start : start + update_chunk_size].copy()
        con.register("metadata_updates", chunk)
        con.execute(
            f"""
            UPDATE {stage_table_name} AS s
            SET
                "user" = coalesce(m.user, s."user"),
                "timestamp" = coalesce(m.timestamp, s."timestamp"),
                "uid" = coalesce(m.uid, s."uid"),
                "version" = coalesce(m.version, s."version"),
                "changeset" = coalesce(m.changeset, s."changeset")
            FROM metadata_updates AS m
            WHERE s.feature_id = 'way/' || CAST(m.way_id AS VARCHAR);
            """
        )
        con.unregister("metadata_updates")

    print(f"Updated local PBF metadata rows for {stage_table_name}: {total_rows}")


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

    top_muni = municipality_summary.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.barh(top_muni["municipality_name"], top_muni["pv_feature_count"], color="#16a34a")
    # add bar value labels
    for i, count in enumerate(top_muni["pv_feature_count"]):
        ax.text(count + 0.5, i, str(count), va="center", fontsize=9)
    ax.set_title("Top 10 municipalities by rooftop PV OSM feature count", pad=12, fontsize=13)
    ax.set_xlabel("PV feature count")
    output_path = output_dir / "pv_eda_top10_municipalities.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
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
# 2. Resolve Puerto Rico geometry with OSMnx and ingest matching rooftop-PV + solar-plant OSM features directly into DuckDB via QuackOSM.
# 3. Enrich stage metadata from local cached OSM PBF and validate stage schemas/samples.
# 4. Build cleaned rooftop PV polygons from the QuackOSM stage table using SQL tag logic.
# 5. Update `has_PV` flags, run EDA summaries, and export macro + interactive micro maps.

# %%
# Step 1: initialize paths + connection.
DB_PATH = resolve_db_path()
OUTPUT_DIR = resolve_output_dir()
CACHE_DIR = resolve_cache_dir()
OSM_PBF_PATH = resolve_cached_osm_pbf(CACHE_DIR)
con = create_spatial_connection(DB_PATH)
print(f"Connected to duckdb database: {DB_PATH}")
print(f"Local OSM PBF for metadata enrichment: {OSM_PBF_PATH if OSM_PBF_PATH else 'not found'}")
tables_before_quackosm = list_db_tables(con)
print("Tables available before QuackOSM ingestion:")
print(tables_before_quackosm.to_string(index=False))

# %%
# Step 2: load municipality boundaries.
municipalities = load_municipalities(con)
print(f"Loaded municipalities: {len(municipalities)}")

# %%
# Step 3: fetch Puerto Rico geometry and run island-wide QuackOSM extractions.
puerto_rico_geometry = resolve_puerto_rico_geometry()
island_run = run_extraction(
    puerto_rico_geometry,
    DB_PATH,
    CACHE_DIR,
    tags_filter=QUACKOSM_TAGS_FILTER,
)
island_feature_count = count_features_in_table(con, island_run["stage_table"])
island_run["feature_count"] = island_feature_count

solar_plant_run = run_extraction(
    puerto_rico_geometry,
    DB_PATH,
    CACHE_DIR,
    tags_filter=SOLAR_PLANT_TAGS_FILTER,
    stage_table_name=SOLAR_PLANT_STAGE_TABLE_NAME,
)
solar_plant_feature_count = count_features_in_table(con, solar_plant_run["stage_table"])
solar_plant_run["feature_count"] = solar_plant_feature_count

enrich_stage_metadata_from_local_pbf(con, island_run["stage_table"], OSM_PBF_PATH)
enrich_stage_metadata_from_local_pbf(con, solar_plant_run["stage_table"], OSM_PBF_PATH)

print(f"Island-wide staging table: {island_run['stage_table']}")
print(f"Island-wide feature count before SQL cleaning (QuackOSM stage rows already prefiltered by tags_filter): {island_feature_count}")
print(f"Island-wide extraction elapsed: {island_run['elapsed_seconds']}s")
print(f"Solar-plant staging table: {solar_plant_run['stage_table']}")
print(f"Solar-plant feature count (plant:method=photovoltaic OR plant:source=solar): {solar_plant_feature_count}")
print(f"Solar-plant extraction elapsed: {solar_plant_run['elapsed_seconds']}s")
tables_after_quackosm = list_db_tables(con)
print("\nTables available after QuackOSM ingestion:")
print(tables_after_quackosm.to_string(index=False))
print("\nStage table schema preview:")
print(preview_table_schema(con, island_run["stage_table"]).to_string(index=False))
print("\nStage table sample rows:")
print(preview_table_rows(con, island_run["stage_table"], limit=5).to_string(index=False))
print("\nSolar-plant stage schema preview:")
print(preview_table_schema(con, solar_plant_run["stage_table"]).to_string(index=False))
print("\nSolar-plant stage sample rows:")
print(preview_table_rows(con, solar_plant_run["stage_table"], limit=5).to_string(index=False))

stage_metadata_cols = {"user", "timestamp"}
available_stage_cols = set(preview_table_schema(con, island_run["stage_table"])["column_name"].tolist())
print(f"\nMetadata column check in rooftop stage table -> user/timestamp present: {stage_metadata_cols.issubset(available_stage_cols)}")
metadata_counts = con.execute(
    f"""
    SELECT
        SUM(CASE WHEN "user" IS NOT NULL THEN 1 ELSE 0 END) AS user_non_null,
        SUM(CASE WHEN "timestamp" IS NOT NULL THEN 1 ELSE 0 END) AS timestamp_non_null
    FROM {island_run['stage_table']}
    """
).fetchdf()
print("Rooftop stage metadata non-null counts:")
print(metadata_counts.to_string(index=False))

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


