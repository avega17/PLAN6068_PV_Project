# %% [markdown]
# # GeoAI Training Data Preparation
#
# Uses fetched Solar RGB GeoTIFFs (priority-3 tiles: Puerto Nuevo + Barrio Mora)
# and `pr_osm_rooftop_pv_polygons` as seed masks to generate chips for
# fine-tuning `geoai.SolarPanelDetector`.
#
# Exports COCO annotations so the workflow does not depend on GeoAI's optional
# `lxml`-backed PASCAL VOC writer.

# %%
"""09_geoai_training_data.py

Export matched image/mask chips for Mask R-CNN training.
"""

from __future__ import annotations

import json
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
SOLAR_ROOT = (PROJECT_ROOT / _env_solar_root) if _env_solar_root and not Path(_env_solar_root).is_absolute() else Path(_env_solar_root or PROJECT_ROOT / "data" / "rasters" / "solar")
TRAIN_IMAGE_DIR = PROJECT_ROOT / "output" / "geoai_train" / "images"
TRAIN_MASK_FILE = PROJECT_ROOT / "output" / "geoai_train" / "osm_pv_seed_masks.geojson"
TILE_SIZE = 512
STRIDE = 256
BUFFER_RADIUS_M = 0.5
CLASS_VALUE_FIELD = "class"
POST_MARIA_CUTOFF = pd.Timestamp("2017-09-20")
TRAINING_ELIGIBLE_QUALITIES = {"BASE", "MEDIUM"}
TRAIN_METADATA_FORMAT = "COCO"


def resolve_db_path() -> Path:
    value = os.getenv("VECTOR_DB")
    if value:
        p = Path(value)
        if not p.is_absolute():
            p = PROJECT_ROOT / p if len(p.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / p
        return p
    return PROJECT_ROOT / "data" / "vectors" / "PR_vector_data.duckdb"


def _to_bytes(v: object) -> bytes:
    if isinstance(v, memoryview):
        return v.tobytes()
    if isinstance(v, bytearray):
        return bytes(v)
    return bytes(v) if not isinstance(v, bytes) else v


def _sidecar_meta(tile_dir: Path, tile_id: str) -> dict:
    sidecar = tile_dir / f"{tile_id}_meta.json"
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


# %%
def gather_priority3_rgb_tiles(con: duckdb.DuckDBPyConnection) -> list[Path]:
    """Locate post-Maria RGB GeoTIFFs for priority-3 tiles that were fetched."""

    rows = con.execute(
        """
        SELECT tile_id, bg_geoid, municipio
        FROM pr_solar_tile_manifest
        WHERE priority_score = 3 AND status = 'fetched'
        ORDER BY tile_id;
        """
    ).fetchdf()
    tifs: list[Path] = []
    for _, row in rows.iterrows():
        tile_dir = SOLAR_ROOT / str(row["municipio"]).replace(" ", "_") / str(row["bg_geoid"])
        meta = _sidecar_meta(tile_dir, str(row["tile_id"]))
        imagery_date = pd.to_datetime(meta.get("imagery_date"), errors="coerce")
        imagery_quality = meta.get("returned_quality")
        if pd.isna(imagery_date) or imagery_date < POST_MARIA_CUTOFF:
            continue
        if imagery_quality not in TRAINING_ELIGIBLE_QUALITIES:
            continue
        for candidate in tile_dir.glob(f"{row['tile_id']}_rgb_*.tif"):
            tifs.append(candidate)
    return tifs


def export_osm_pv_seed_masks(con: duckdb.DuckDBPyConnection, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = con.execute(
        "SELECT ST_AsWKB(geometry) AS wkb FROM pr_osm_rooftop_pv_polygons WHERE geometry IS NOT NULL;"
    ).fetchdf()
    if df.empty:
        return 0
    gdf = gpd.GeoDataFrame(
        {"class": 1},
        geometry=df["wkb"].map(lambda v: from_wkb(_to_bytes(v))),
        crs="EPSG:4326",
        index=df.index,
    )
    gdf.to_file(out_path, driver="GeoJSON")
    return len(gdf)


# %%
if __name__ == "__main__":
    db_path = resolve_db_path()
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")

    count = export_osm_pv_seed_masks(con, TRAIN_MASK_FILE)
    print(f"wrote {count:,} seed PV polygons to {TRAIN_MASK_FILE}")

    rgb_tifs = gather_priority3_rgb_tiles(con)
    print(f"priority-3 post-Maria RGB tiles: {len(rgb_tifs):,}")
    if not rgb_tifs:
        print("no post-Maria RGB tiles available yet — run the ingest notebook first.")
        con.close()
        sys.exit(0)

    # Stage tiles into a single flat folder for geoai.export_geotiff_tiles_batch.
    TRAIN_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for tif in rgb_tifs:
        dest = TRAIN_IMAGE_DIR / tif.name
        if not dest.exists():
            dest.symlink_to(tif.resolve())

    import geoai  # imported lazily to keep top-level import cheap

    geoai.export_geotiff_tiles_batch(
        images_folder=str(TRAIN_IMAGE_DIR),
        masks_file=str(TRAIN_MASK_FILE),
        output_folder=str(PROJECT_ROOT / "output" / "geoai_train"),
        tile_size=TILE_SIZE,
        stride=STRIDE,
        buffer_radius=BUFFER_RADIUS_M,
        class_value_field=CLASS_VALUE_FIELD,
        skip_empty_tiles=True,
        metadata_format=TRAIN_METADATA_FORMAT,
    )
    con.close()
