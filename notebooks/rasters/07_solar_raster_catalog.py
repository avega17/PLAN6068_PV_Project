# %% [markdown]
# # Solar API STAC Geoparquet Catalog
#
# Scans fetched Solar-API GeoTIFFs under `data/rasters/solar/` and produces a
# STAC-lite geoparquet catalog at
# `data/rasters/stac/pr_solar_catalog_items.parquet`. Keeps the existing
# windowed-reader pattern used by the poster-figure notebooks.

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
POST_MARIA_CUTOFF = pd.Timestamp("2017-09-20")
POST_MARIA_ONLY = True
TRAINING_ELIGIBLE_QUALITIES = {"BASE", "MEDIUM"}


# %%
def scan_geotiffs(root: Path) -> list[dict]:
    records: list[dict] = []
    for tif in sorted(root.rglob("*.tif")):
        stem_parts = tif.stem.split("_")
        if len(stem_parts) < 3:
            continue
        quality = stem_parts[-1]
        layer = stem_parts[-2]
        tile_id = "_".join(stem_parts[:-2])
        bg_geoid = tif.parent.name
        municipio = tif.parent.parent.name
        sidecar = tif.parent / f"{tile_id}_meta.json"
        meta: dict = {}
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text())
            except json.JSONDecodeError:
                pass
        imagery_date = pd.to_datetime(meta.get("imagery_date"), errors="coerce")
        post_maria = bool(pd.notna(imagery_date) and imagery_date >= POST_MARIA_CUTOFF)

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

        records.append({
            "item_id": f"{tile_id}_{layer}_{quality}",
            "tile_id": tile_id,
            "bg_geoid": bg_geoid,
            "municipio": municipio,
            "layer": layer,
            "imagery_quality": quality,
            "asset_href": str(tif.relative_to(PROJECT_ROOT)),
            "proj_epsg": src_crs.to_epsg() if src_crs else None,
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
            "post_maria": post_maria,
            "training_eligible": bool(post_maria and quality in TRAINING_ELIGIBLE_QUALITIES),
            "radius_m": meta.get("radius_m"),
            "view": meta.get("view"),
            "geometry": geometry,
        })
    return records


# %%
if __name__ == "__main__":
    print(f"scanning {SOLAR_ROOT}")
    rows = scan_geotiffs(SOLAR_ROOT)
    if not rows:
        print("no GeoTIFFs found — run 07_google_solar_api_ingest first.")
    else:
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        if POST_MARIA_ONLY:
            gdf = gdf[gdf["post_maria"]].reset_index(drop=True)
        CATALOG_DIR.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(CATALOG_PATH, index=False)
        print(f"wrote {len(gdf):,} items to {CATALOG_PATH}")
        print(gdf.groupby(["municipio", "layer", "imagery_quality", "post_maria"]).size().head(20).to_string())
