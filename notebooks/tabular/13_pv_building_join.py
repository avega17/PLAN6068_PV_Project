# %% [markdown]
# # Building-Level PV Join with Annual-Flux Zonal Stats
#
# For each Overture building in `San Juan` + `Isabela`:
# - flag `has_pv_osm` if the building intersects `pr_osm_rooftop_pv_polygons`,
# - flag `has_pv_detected` + area from `pr_solar_pv_detections`,
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

_env_solar_root = os.getenv("SOLAR_RASTER_ROOT")
SOLAR_ROOT = (
    (PROJECT_ROOT / _env_solar_root)
    if _env_solar_root and not Path(_env_solar_root).is_absolute()
    else Path(_env_solar_root or PROJECT_ROOT / "data" / "rasters" / "solar")
)

OUTPUT_TABLE = "pr_buildings_with_pv"
TARGET_MUNICIPALITIES = ("San Juan", "Isabela")


def resolve_db_path() -> Path:
    v = os.getenv("VECTOR_DB")
    if v:
        p = Path(v)
        if not p.is_absolute():
            p = PROJECT_ROOT / p if len(p.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / p
        return p
    return PROJECT_ROOT / "data" / "PR_PV_plan_data.duckdb"


def _to_bytes(v: object) -> bytes:
    if isinstance(v, memoryview):
        return v.tobytes()
    if isinstance(v, bytearray):
        return bytes(v)
    return bytes(v) if not isinstance(v, bytes) else v


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
    has_dets = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='pr_solar_pv_detections';"
    ).fetchone()[0]

    det_cte = (
        """
        det_join AS (
            SELECT b.building_id,
                   COUNT(d.geometry) AS pv_detected_count,
                   SUM(ST_Area(ST_Intersection(b.geometry, d.geometry))) AS pv_detected_area_deg2
            FROM buildings AS b
            LEFT JOIN pr_solar_pv_detections AS d
              ON ST_Intersects(b.geometry, d.geometry)
            GROUP BY b.building_id
        )
        """
        if has_dets
        else """
        det_join AS (
            SELECT building_id, 0 AS pv_detected_count, 0.0 AS pv_detected_area_deg2
            FROM buildings
        )
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE pr_buildings_pv_flags AS
        WITH buildings AS (
            SELECT id AS building_id, municipality_name AS municipio,
                   municipality_geoid, geometry
            FROM pr_overture_buildings
            WHERE municipality_name IN ({munis_sql}) AND geometry IS NOT NULL
        ),
        {det_cte},
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
            COALESCE(d.pv_detected_area_deg2, 0.0) AS pv_detected_area_deg2,
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
db_path = resolve_db_path()
con = duckdb.connect(str(db_path))
con.execute("INSTALL spatial; LOAD spatial;")

print("[1/3] flagging OSM + detections …")
build_pv_flags_table(con)

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
