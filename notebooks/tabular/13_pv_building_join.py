# %% [markdown]
# # Building-Level PV Join with Annual-Flux Zonal Stats
#
# For each Overture building in `San Juan` + `Isabela`:
# - flag `has_pv_osm` if the building intersects `pr_osm_rooftop_pv_polygons`,
# - flag `has_pv_detected` + area from `pr_solar_pv_detections` and any
#   available local inference GeoJSON outputs,
# - attach annual-flux zonal stats from `data/rasters/solar/**/*_annualFlux_*.tif`.
#
# Output: DuckDB table `pr_buildings_with_pv` with geometry in EPSG:4326.

# %%
"""01_pv_building_join.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from shapely import from_wkb


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

from utils.census import resolve_vector_db_path

_env_solar_root = os.getenv("SOLAR_RASTER_ROOT")
SOLAR_ROOT = (
    (PROJECT_ROOT / _env_solar_root)
    if _env_solar_root and not Path(_env_solar_root).is_absolute()
    else Path(_env_solar_root or PROJECT_ROOT / "data" / "rasters" / "solar")
)

OUTPUT_TABLE = "pr_buildings_with_pv"
TARGET_MUNICIPALITIES = ("San Juan", "Isabela")
LOCAL_INFERENCE_ROOTS = (
    PROJECT_ROOT / "outputs" / "geoai_inference",
)


def resolve_db_path() -> Path:
    return resolve_vector_db_path(PROJECT_ROOT)


def _to_bytes(v: object) -> bytes:
    if isinstance(v, memoryview):
        return v.tobytes()
    if isinstance(v, bytearray):
        return bytes(v)
    return bytes(v) if not isinstance(v, bytes) else v


def load_target_geometry(con: duckdb.DuckDBPyConnection):
    munis_sql = ", ".join(f"'{m}'" for m in TARGET_MUNICIPALITIES)
    row = con.execute(
        f"""
        SELECT ST_AsWKB(ST_Union_Agg(geometry)) AS wkb
        FROM pr_census_counties
        WHERE NAME IN ({munis_sql});
        """
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return from_wkb(_to_bytes(row[0]))


def list_local_inference_vector_paths() -> list[Path]:
    preferred_paths: list[Path] = []
    preferred_stems: set[str] = set()
    fallback_paths: list[Path] = []

    for root in LOCAL_INFERENCE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*_pred_props.geojson")):
            preferred_paths.append(path)
            preferred_stems.add(path.name.removesuffix("_pred_props.geojson"))

    for root in LOCAL_INFERENCE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*_pred.geojson")):
            stem = path.name.removesuffix("_pred.geojson")
            if stem not in preferred_stems:
                fallback_paths.append(path)

    return preferred_paths + fallback_paths


def infer_detection_source(path: Path) -> str:
    relative_path = path.relative_to(PROJECT_ROOT)
    path_text = relative_path.as_posix()
    if path_text.startswith("outputs/geoai_inference/"):
        return "geoai_inference"
    if "smp_grounded" in path.parts:
        return "smp_grounded"
    return path.parent.name


def load_local_inference_vectors(target_geometry=None) -> gpd.GeoDataFrame:
    rows: list[gpd.GeoDataFrame] = []
    for index, path in enumerate(list_local_inference_vector_paths(), start=1):
        try:
            gdf = gpd.read_file(path)
        except Exception as exc:
            print(f"warning: failed to read local inference vector {path}: {exc}")
            continue
        if gdf.empty or "geometry" not in gdf.columns:
            continue

        gdf = gdf.loc[gdf.geometry.notna()].copy()
        if gdf.empty:
            continue
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        else:
            gdf = gdf.to_crs("EPSG:4326")
        gdf = gdf.loc[~gdf.geometry.is_empty].copy()
        if target_geometry is not None:
            gdf = gdf[gdf.intersects(target_geometry)].copy()
        if gdf.empty:
            continue

        gdf["detection_key"] = [f"local_{index}_{row_index}" for row_index in range(len(gdf))]
        gdf["inference_source"] = infer_detection_source(path)
        gdf["source_path"] = str(path.relative_to(PROJECT_ROOT))
        rows.append(gdf[["detection_key", "inference_source", "source_path", "geometry"]].copy())

    if not rows:
        return gpd.GeoDataFrame(columns=["detection_key", "inference_source", "source_path", "geometry"], geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame(pd.concat(rows, ignore_index=True), geometry="geometry", crs="EPSG:4326")


def stage_inference_vectors(con: duckdb.DuckDBPyConnection) -> None:
    target_geometry = load_target_geometry(con)
    local_vectors = load_local_inference_vectors(target_geometry=target_geometry)
    has_detection_table = bool(
        con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='pr_solar_pv_detections';"
        ).fetchone()[0]
    )

    if not local_vectors.empty:
        staged = pd.DataFrame(local_vectors.drop(columns=["geometry"]))
        staged["geometry_wkb"] = local_vectors.geometry.to_wkb()
        con.register("staged_local_inference_vectors", staged)
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE local_inference_vectors AS
            SELECT * EXCLUDE (geometry_wkb),
                   ST_GeomFromWKB(geometry_wkb) AS geometry
            FROM staged_local_inference_vectors;
            """
        )
    else:
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE local_inference_vectors AS
            SELECT CAST(NULL AS VARCHAR) AS detection_key,
                   CAST(NULL AS VARCHAR) AS inference_source,
                   CAST(NULL AS VARCHAR) AS source_path,
                   CAST(NULL AS GEOMETRY) AS geometry
            WHERE FALSE;
            """
        )

    if has_detection_table:
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE db_inference_vectors AS
            SELECT
                CONCAT('db_', CAST(ROW_NUMBER() OVER () AS VARCHAR)) AS detection_key,
                'pr_solar_pv_detections' AS inference_source,
                'duckdb:pr_solar_pv_detections' AS source_path,
                geometry
            FROM pr_solar_pv_detections
            WHERE geometry IS NOT NULL;
            """
        )
    else:
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE db_inference_vectors AS
            SELECT CAST(NULL AS VARCHAR) AS detection_key,
                   CAST(NULL AS VARCHAR) AS inference_source,
                   CAST(NULL AS VARCHAR) AS source_path,
                   CAST(NULL AS GEOMETRY) AS geometry
            WHERE FALSE;
            """
        )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE pv_inference_vectors AS
        SELECT detection_key, inference_source, source_path, geometry
        FROM db_inference_vectors
        UNION ALL
        SELECT detection_key, inference_source, source_path, geometry
        FROM local_inference_vectors;
        """
    )


# %%
def load_buildings(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    munis_sql = ", ".join(f"'{m}'" for m in TARGET_MUNICIPALITIES)
    df = con.execute(
        f"""
        SELECT
            b.id AS building_id,
            b.municipality_name AS municipio,
            b.municipality_geoid,
            ST_AsWKB(b.geometry) AS wkb
        FROM pr_overture_buildings AS b
        WHERE b.municipality_name IN ({munis_sql})
          AND b.geometry IS NOT NULL;
        """
    ).fetchdf()
    if df.empty:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")
    geoms = gpd.GeoSeries(df["wkb"].map(lambda v: from_wkb(_to_bytes(v))), crs="EPSG:4326")
    return gpd.GeoDataFrame(df.drop(columns=["wkb"]), geometry=geoms, crs="EPSG:4326")


def build_pv_flags_table(con: duckdb.DuckDBPyConnection) -> None:
    munis_sql = ", ".join(f"'{m}'" for m in TARGET_MUNICIPALITIES)
    stage_inference_vectors(con)

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE pr_buildings_pv_flags AS
        WITH buildings AS (
            SELECT id AS building_id, municipality_name AS municipio,
                   municipality_geoid, geometry
            FROM pr_overture_buildings
            WHERE municipality_name IN ({munis_sql}) AND geometry IS NOT NULL
        ),
        det_join AS (
            SELECT
                b.building_id,
                COUNT(DISTINCT d.detection_key) AS pv_detected_count,
                COUNT(DISTINCT d.inference_source) AS pv_detection_source_count,
                CASE
                    WHEN COUNT(DISTINCT d.detection_key) = 0 THEN 0.0
                    ELSE ST_Area(ST_Intersection(ANY_VALUE(b.geometry), ST_Union_Agg(d.geometry)))
                END AS pv_detected_area_deg2
            FROM buildings AS b
            LEFT JOIN pv_inference_vectors AS d
              ON ST_Intersects(b.geometry, d.geometry)
            GROUP BY b.building_id
        ),
        osm_join AS (
            SELECT b.building_id, TRUE AS has_pv_osm
            FROM buildings AS b
            JOIN pr_osm_rooftop_pv_polygons AS o
              ON ST_Intersects(b.geometry, o.geometry)
            GROUP BY b.building_id
        )
        SELECT
            b.building_id,
            b.municipio,
            b.municipality_geoid,
            COALESCE(o.has_pv_osm, FALSE) AS has_pv_osm,
            COALESCE(d.pv_detected_count, 0) > 0 AS has_pv_detected,
            COALESCE(d.pv_detected_count, 0) AS pv_detected_count,
            COALESCE(d.pv_detection_source_count, 0) AS pv_detection_source_count,
            COALESCE(d.pv_detected_area_deg2, 0.0) AS pv_detected_area_deg2,
            COALESCE(o.has_pv_osm, FALSE) OR COALESCE(d.pv_detected_count, 0) > 0 AS has_pv_any,
            b.geometry
        FROM buildings AS b
        LEFT JOIN det_join AS d USING (building_id)
        LEFT JOIN osm_join AS o USING (building_id);
        """
    )


# %%
def compute_annual_flux_zonal_stats(buildings: gpd.GeoDataFrame, solar_root: Path) -> pd.DataFrame:
    """Weighted (by pixel count) zonal mean/p10/p90 per building across all
    overlapping annualFlux tiles."""

    import numpy as np
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import transform_bounds
    from shapely.geometry import box

    empty_cols = [
        "building_id",
        "annual_flux_mean_kwh_per_kw_yr",
        "annual_flux_p10_kwh_per_kw_yr",
        "annual_flux_p90_kwh_per_kw_yr",
        "annual_flux_pixel_count",
    ]
    flux_tifs = sorted(solar_root.rglob("*_annualFlux_*.tif"))
    if not flux_tifs or buildings.empty:
        print("no annualFlux GeoTIFFs (or no buildings) — returning empty stats.")
        return pd.DataFrame(columns=empty_cols)

    per_building: dict[str, list[tuple[float, float, float, int]]] = {}
    for tif in flux_tifs:
        try:
            with rasterio.open(tif) as ds:
                if ds.crs is None:
                    continue
                left, bottom, right, top = ds.bounds
                lonlat = transform_bounds(ds.crs, "EPSG:4326", left, bottom, right, top, densify_pts=21)
                tile_poly = box(*lonlat)
                candidates = buildings[buildings.intersects(tile_poly)]
                if candidates.empty:
                    continue
                cr = candidates.to_crs(ds.crs)
                nodata = ds.nodata
                for bid, geom in zip(cr["building_id"], cr.geometry):
                    if geom.is_empty:
                        continue
                    try:
                        arr, _ = rio_mask(ds, [geom], crop=True, filled=False)
                    except ValueError:
                        continue
                    band = arr[0]
                    if hasattr(band, "compressed"):
                        valid = band.compressed()
                    else:
                        valid = band.ravel()
                        if nodata is not None:
                            valid = valid[valid != nodata]
                    valid = valid[np.isfinite(valid)]
                    if valid.size == 0:
                        continue
                    per_building.setdefault(bid, []).append(
                        (float(valid.mean()), float(np.percentile(valid, 10)),
                         float(np.percentile(valid, 90)), int(valid.size))
                    )
        except rasterio.errors.RasterioIOError:
            continue

    rows = []
    for bid, entries in per_building.items():
        total = sum(e[3] for e in entries)
        if total == 0:
            continue
        rows.append({
            "building_id": bid,
            "annual_flux_mean_kwh_per_kw_yr": sum(e[0] * e[3] for e in entries) / total,
            "annual_flux_p10_kwh_per_kw_yr": sum(e[1] * e[3] for e in entries) / total,
            "annual_flux_p90_kwh_per_kw_yr": sum(e[2] * e[3] for e in entries) / total,
            "annual_flux_pixel_count": total,
        })
    return pd.DataFrame(rows, columns=empty_cols) if not rows else pd.DataFrame(rows)


# %%
# Notebook driver: connect, build flags, compute zonal stats, persist.
if __name__ == "__main__":
    db_path = resolve_db_path()
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")

    print("[1/3] flagging OSM + detections …")
    build_pv_flags_table(con)

    local_vectors = load_local_inference_vectors(target_geometry=load_target_geometry(con))
    print(f"      local inference vectors discovered: {len(local_vectors):,}")

    print("[2/3] loading buildings for zonal stats …")
    buildings = load_buildings(con)
    print(f"      {len(buildings):,} buildings in {TARGET_MUNICIPALITIES}")

    print(f"[3/3] computing annual-flux zonal stats under {SOLAR_ROOT} …")
    flux_df = compute_annual_flux_zonal_stats(buildings, SOLAR_ROOT)
    print(f"      stats for {len(flux_df):,} buildings")

    con.register("flux_stats", flux_df)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {OUTPUT_TABLE} AS
        SELECT f.*,
               fs.annual_flux_mean_kwh_per_kw_yr,
               fs.annual_flux_p10_kwh_per_kw_yr,
               fs.annual_flux_p90_kwh_per_kw_yr,
               fs.annual_flux_pixel_count
        FROM pr_buildings_pv_flags AS f
        LEFT JOIN flux_stats AS fs USING (building_id);
        """
    )
    con.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_geom ON {OUTPUT_TABLE} USING RTREE (geometry);"
    )
    con.unregister("flux_stats")

    # %%
    summary = con.execute(
        f"""
        SELECT municipio,
               COUNT(*) AS buildings,
               SUM(CAST(has_pv_osm AS INT)) AS osm_labeled,
               SUM(CAST(has_pv_detected AS INT)) AS model_detected,
               SUM(CAST(has_pv_any AS INT)) AS any_pv_signal,
               SUM(CAST(has_pv_osm AND has_pv_detected AS INT)) AS overlap,
               AVG(annual_flux_mean_kwh_per_kw_yr) AS avg_flux_kwh_per_kw_yr
        FROM {OUTPUT_TABLE}
        GROUP BY 1
        ORDER BY 1;
        """
    ).fetchdf()
    print("\npr_buildings_with_pv summary:")
    print(summary.to_string(index=False))
    con.close()
