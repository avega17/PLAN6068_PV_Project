# %% [markdown]
# # GeoAI Solar Detector Inference
#
# Runs the fine-tuned Mask R-CNN over every Solar RGB tile, orthogonalizes
# predictions to polygons, attaches geometric properties, filters, and writes
# `pr_solar_pv_detections` to DuckDB.

# %%
"""11_geoai_solar_inference.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv


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
MODEL_PATH = PROJECT_ROOT / "output" / "models" / "best_model.pth"
INFER_ROOT = PROJECT_ROOT / "output" / "geoai_inference"
DETECTION_TABLE = "pr_solar_pv_detections"

WINDOW_SIZE = 400
OVERLAP = 100
CONFIDENCE_THRESHOLD = 0.4
MIN_AREA_M2 = 3.0
MAX_AREA_M2 = 500.0
MAX_ELONGATION = 10.0


def resolve_db_path() -> Path:
    value = os.getenv("VECTOR_DB")
    if value:
        p = Path(value)
        if not p.is_absolute():
            p = PROJECT_ROOT / p if len(p.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / p
        return p
    return PROJECT_ROOT / "data" / "vectors" / "PR_vector_data.duckdb"


# %%
if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print(f"model not found at {MODEL_PATH} — run 10_geoai_solar_finetune first.")
        sys.exit(0)

    import geoai

    INFER_ROOT.mkdir(parents=True, exist_ok=True)
    all_polygons: list[gpd.GeoDataFrame] = []

    for rgb in sorted(SOLAR_ROOT.rglob("*_rgb_*.tif")):
        mask_out = INFER_ROOT / f"{rgb.stem}_mask.tif"
        vector_out = INFER_ROOT / f"{rgb.stem}_pred.geojson"
        if not mask_out.exists():
            geoai.object_detection(
                raster_path=str(rgb),
                output_path=str(mask_out),
                model_path=str(MODEL_PATH),
                window_size=WINDOW_SIZE,
                overlap=OVERLAP,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                batch_size=4,
                num_channels=3,
            )
        if mask_out.exists() and not vector_out.exists():
            geoai.orthogonalize(str(mask_out), str(vector_out), epsilon=0.2)

        if vector_out.exists():
            try:
                gdf = gpd.read_file(vector_out)
            except Exception:
                continue
            if gdf.empty:
                continue
            enriched = INFER_ROOT / f"{rgb.stem}_pred_props.geojson"
            geoai.add_geometric_properties(str(vector_out), str(enriched))
            gdf = gpd.read_file(enriched)
            gdf["source_raster"] = str(rgb.relative_to(PROJECT_ROOT))
            all_polygons.append(gdf)

    if not all_polygons:
        print("no detections produced.")
        sys.exit(0)

    merged = gpd.GeoDataFrame(pd.concat(all_polygons, ignore_index=True), crs=all_polygons[0].crs)
    merged = merged.to_crs("EPSG:4326")

    if "area_m2" in merged.columns and "elongation" in merged.columns:
        merged = merged[
            (merged["area_m2"].between(MIN_AREA_M2, MAX_AREA_M2))
            & (merged["elongation"] < MAX_ELONGATION)
        ].copy()

    db_path = resolve_db_path()
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")

    staged = pd.DataFrame(merged.drop(columns=["geometry"]))
    staged["geometry_wkb"] = merged.geometry.to_wkb()
    con.register("staged_dets", staged)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {DETECTION_TABLE} AS
        SELECT * EXCLUDE (geometry_wkb),
               ST_GeomFromWKB(geometry_wkb) AS geometry
        FROM staged_dets;
        """
    )
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{DETECTION_TABLE}_geom ON {DETECTION_TABLE} USING RTREE (geometry);")
    con.unregister("staged_dets")
    print(f"wrote {len(merged):,} detections to {DETECTION_TABLE}")
    con.close()
