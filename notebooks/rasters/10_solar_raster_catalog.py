# %% [markdown]
# # Solar API STAC Geoparquet Catalog
#
# Scans fetched Solar-API GeoTIFFs under `data/rasters/solar/` and produces a
# STAC-lite geoparquet catalog at
# `data/rasters/stac/pr_solar_catalog_items.parquet`.
#
# It also builds a mixed local raster catalog at
# `data/rasters/stac/pr_local_raster_catalog_items.parquet` so downstream
# inference notebooks can work from one local source-aware index that includes
# both Google Solar rasters and occupied-H3-cell STAC derivatives.

# %%
"""08_solar_raster_catalog.py

Build a STAC-geoparquet index over fetched Google Solar API GeoTIFFs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from dotenv import load_dotenv
from shapely.geometry import box


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
SOLAR_ROOT = (PROJECT_ROOT / _env_solar_root) if _env_solar_root and not Path(_env_solar_root).is_absolute() else Path(_env_solar_root or PROJECT_ROOT / "data" / "rasters" / "solar")
CATALOG_DIR = PROJECT_ROOT / "data" / "rasters" / "stac"
CATALOG_PATH = CATALOG_DIR / "pr_solar_catalog_items.parquet"
LOCAL_STAC_ROOT = CATALOG_DIR / "local"
LOCAL_CATALOG_PATH = CATALOG_DIR / "pr_local_raster_catalog_items.parquet"
POST_MARIA_CUTOFF = pd.Timestamp("2017-09-20")
POST_MARIA_ONLY = True
TRAINING_ELIGIBLE_QUALITIES = {"BASE", "MEDIUM"}
STAC_TRAINING_MAX_GSD_M = 1.5


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _is_post_maria(value: object) -> bool:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return False
    cutoff = POST_MARIA_CUTOFF
    if getattr(timestamp, "tzinfo", None) is not None:
        timestamp = timestamp.tz_convert("UTC")
        cutoff = cutoff.tz_localize("UTC")
    return bool(timestamp >= cutoff)


# %%
def scan_google_solar_geotiffs(root: Path) -> list[dict]:
    records: list[dict] = []
    for tif in sorted(root.rglob("*.tif")):
        stem_parts = tif.stem.split("_")
        if len(stem_parts) < 3:
            continue
        quality = stem_parts[-1]
        layer = stem_parts[-2]
        tile_id = "_".join(stem_parts[:-2])
        bg_geoid = tif.parent.name
        municipio = tif.parent.parent.name.replace("_", " ")
        sidecar = tif.parent / f"{tile_id}_meta.json"
        meta = _read_json(sidecar)
        imagery_date = pd.to_datetime(meta.get("imagery_date"), errors="coerce")
        post_maria = _is_post_maria(imagery_date)

        try:
            with rasterio.open(tif) as ds:
                src_crs = ds.crs
                bounds = ds.bounds
                gsd_x = abs(ds.transform.a)
                gsd_y = abs(ds.transform.e)
                width, height = ds.width, ds.height
                bands = ds.count
        except rasterio.errors.RasterioIOError:
            continue

        if src_crs is not None and str(src_crs).upper() != "EPSG:4326":
            lonlat = transform_bounds(src_crs, "EPSG:4326", *bounds, densify_pts=21)
        else:
            lonlat = tuple(bounds)
        geometry = box(*lonlat)

        proj_epsg = src_crs.to_epsg() if src_crs else None
        records.append({
            "source_family": "google_solar",
            "source": "google_solar",
            "item_id": f"{tile_id}_{layer}_{quality}",
            "tile_id": tile_id,
            "bg_geoid": bg_geoid,
            "h3_cell_id": meta.get("h3_cell_id"),
            "h3_resolution": meta.get("h3_resolution"),
            "municipio": municipio,
            "layer": layer,
            "asset_role": layer,
            "imagery_quality": quality,
            "asset_href": str(tif.relative_to(PROJECT_ROOT)),
            "local_asset_path": str(tif.relative_to(PROJECT_ROOT)),
            "original_asset_href": None,
            "proj_epsg": proj_epsg,
            "native_proj_epsg": proj_epsg,
            "model_ready_proj_epsg": proj_epsg,
            "gsd_m": float((gsd_x + gsd_y) / 2),
            "width": int(width),
            "height": int(height),
            "bands": int(bands),
            "bbox_minx": float(lonlat[0]),
            "bbox_miny": float(lonlat[1]),
            "bbox_maxx": float(lonlat[2]),
            "bbox_maxy": float(lonlat[3]),
            "imagery_date": meta.get("imagery_date"),
            "imagery_processed_date": meta.get("imagery_processed_date"),
            "acquired_at": meta.get("imagery_date"),
            "post_maria": post_maria,
            "training_eligible": bool(post_maria and quality in TRAINING_ELIGIBLE_QUALITIES and layer == "rgb"),
            "inference_eligible": bool(post_maria and layer == "rgb"),
            "radius_m": meta.get("radius_m"),
            "view": meta.get("view"),
            "platform": "google_solar",
            "catalog_self_href": None,
            "clip_strategy": meta.get("clip_strategy") or "google_solar_poi",
            "geometry": geometry,
        })
    return records


def scan_local_stac_geotiffs(root: Path) -> list[dict]:
    records: list[dict] = []
    if not root.exists():
        return records

    for tif in sorted(root.rglob("*.tif")):
        sidecar = tif.with_name(f"{tif.stem}_meta.json")
        meta = _read_json(sidecar)
        source = meta.get("source") or tif.parent.parent.name
        municipio = meta.get("municipio") or tif.parent.name.replace("_", " ")
        asset_role = meta.get("asset_role") or tif.stem.rsplit("_", 1)[-1]
        item_id = meta.get("item_id") or tif.stem.removesuffix(f"_{asset_role}")
        acquired_at = pd.to_datetime(meta.get("acquired_at"), utc=True, errors="coerce")
        post_maria = _is_post_maria(acquired_at)

        try:
            with rasterio.open(tif) as ds:
                src_crs = ds.crs
                bounds = ds.bounds
                gsd_x = abs(ds.transform.a)
                gsd_y = abs(ds.transform.e)
                width, height = ds.width, ds.height
                bands = ds.count
        except rasterio.errors.RasterioIOError:
            continue

        if src_crs is not None and str(src_crs).upper() != "EPSG:4326":
            lonlat = transform_bounds(src_crs, "EPSG:4326", *bounds, densify_pts=21)
        else:
            lonlat = tuple(bounds)
        geometry = box(*lonlat)
        gsd_m = float((gsd_x + gsd_y) / 2)

        native_proj_epsg = meta.get("native_proj_epsg")
        model_ready_proj_epsg = meta.get("model_ready_proj_epsg") or (src_crs.to_epsg() if src_crs else None)
        inference_eligible = bool(
            post_maria
            and asset_role in {"visual", "analytic"}
            and bands >= 3
            and gsd_m <= STAC_TRAINING_MAX_GSD_M
            and model_ready_proj_epsg == 3857
        )

        records.append({
            "source_family": "stac",
            "source": source,
            "item_id": item_id,
            "tile_id": None,
            "bg_geoid": None,
            "h3_cell_id": meta.get("h3_cell_id"),
            "h3_resolution": meta.get("h3_resolution"),
            "municipio": municipio,
            "layer": asset_role,
            "asset_role": asset_role,
            "imagery_quality": None,
            "asset_href": str(tif.relative_to(PROJECT_ROOT)),
            "local_asset_path": str(tif.relative_to(PROJECT_ROOT)),
            "original_asset_href": meta.get("original_asset_href"),
            "proj_epsg": src_crs.to_epsg() if src_crs else None,
            "native_proj_epsg": native_proj_epsg,
            "model_ready_proj_epsg": model_ready_proj_epsg,
            "gsd_m": gsd_m,
            "width": int(width),
            "height": int(height),
            "bands": int(bands),
            "bbox_minx": float(lonlat[0]),
            "bbox_miny": float(lonlat[1]),
            "bbox_maxx": float(lonlat[2]),
            "bbox_maxy": float(lonlat[3]),
            "imagery_date": None,
            "imagery_processed_date": None,
            "acquired_at": meta.get("acquired_at"),
            "post_maria": post_maria,
            "training_eligible": False,
            "inference_eligible": inference_eligible,
            "radius_m": None,
            "view": None,
            "platform": meta.get("platform"),
            "catalog_self_href": meta.get("catalog_self_href"),
            "clip_strategy": meta.get("clip_strategy") or "occupied_h3_cell",
            "geometry": geometry,
        })
    return records


# %%
if __name__ == "__main__":
    print(f"scanning {SOLAR_ROOT}")
    solar_rows = scan_google_solar_geotiffs(SOLAR_ROOT)
    if not solar_rows:
        print("no GeoTIFFs found — run 07_google_solar_api_ingest first.")
    else:
        gdf = gpd.GeoDataFrame(solar_rows, geometry="geometry", crs="EPSG:4326")
        if POST_MARIA_ONLY:
            gdf = gdf[gdf["post_maria"]].reset_index(drop=True)
        CATALOG_DIR.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(CATALOG_PATH, index=False)
        print(f"wrote {len(gdf):,} items to {CATALOG_PATH}")

        local_rows = solar_rows + scan_local_stac_geotiffs(LOCAL_STAC_ROOT)
        local_gdf = gpd.GeoDataFrame(local_rows, geometry="geometry", crs="EPSG:4326")
        local_gdf.to_parquet(LOCAL_CATALOG_PATH, index=False)
        print(f"wrote {len(local_gdf):,} items to {LOCAL_CATALOG_PATH}")
        print("solar-only catalog summary:")
        print(gdf.groupby(["municipio", "layer", "imagery_quality", "post_maria"]).size().head(20).to_string())
        print("\nlocal mixed catalog summary:")
        print(local_gdf.groupby(["source_family", "source", "layer", "inference_eligible"]).size().head(30).to_string())
