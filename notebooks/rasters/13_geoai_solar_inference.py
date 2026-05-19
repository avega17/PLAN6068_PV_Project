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
import random
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


def _resolve_configured_path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

_env_solar_root = os.getenv("SOLAR_RASTER_ROOT")
SOLAR_ROOT = (PROJECT_ROOT / _env_solar_root) if _env_solar_root and not Path(_env_solar_root).is_absolute() else Path(_env_solar_root or PROJECT_ROOT / "data" / "rasters" / "solar")
MODEL_PATH = _resolve_configured_path("GEOAI_MODEL_PATH", PROJECT_ROOT / "outputs" / "models" / "best_model.pth")
INFER_ROOT = PROJECT_ROOT / "outputs" / "geoai_inference"
DETECTION_TABLE = "pr_solar_pv_detections"
LOCAL_RASTER_CATALOG = PROJECT_ROOT / "data" / "rasters" / "stac" / "pr_local_raster_catalog_items.parquet"
SOLAR_CATALOG = PROJECT_ROOT / "data" / "rasters" / "stac" / "pr_solar_catalog_items.parquet"
TARGET_MUNICIPALITIES = ("San Juan", "Isabela")
INFERENCE_MODE = os.getenv("GEOAI_INFERENCE_MODE", "sample").strip().lower()
SAMPLE_COUNT = max(1, int(os.getenv("GEOAI_SAMPLE_COUNT", "1")))
SAMPLE_SEED = int(os.getenv("GEOAI_SAMPLE_SEED", "42"))

WINDOW_SIZE = 400
OVERLAP = 100
CONFIDENCE_THRESHOLD = 0.4
MIN_AREA_M2 = 3.0
MAX_AREA_M2 = 500.0
MAX_ELONGATION = 10.0


def resolve_local_asset_path(row: pd.Series) -> Path | None:
    for column_name in ("local_asset_path", "asset_href", "visual_asset_href", "analytic_asset_href"):
        value = row.get(column_name)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists() and candidate.suffix.lower() in {".tif", ".tiff"}:
            return candidate
    return None


def load_catalog_rasters() -> pd.DataFrame:
    catalog_path = LOCAL_RASTER_CATALOG if LOCAL_RASTER_CATALOG.exists() else SOLAR_CATALOG
    if not catalog_path.exists():
        return pd.DataFrame(columns=["raster_path", "source", "item_id", "municipio", "catalog_path"])

    frame = pd.read_parquet(catalog_path)
    if frame.empty:
        return pd.DataFrame(columns=["raster_path", "source", "item_id", "municipio", "catalog_path"])

    if "municipio" in frame.columns:
        frame = frame[frame["municipio"].isin(TARGET_MUNICIPALITIES)].copy()
    if "inference_eligible" in frame.columns:
        frame = frame[frame["inference_eligible"].fillna(False)].copy()
    elif "training_eligible" in frame.columns:
        frame = frame[frame["training_eligible"].fillna(False)].copy()
    if "layer" in frame.columns:
        frame = frame[frame["layer"].fillna("").str.lower().isin({"rgb", "visual"})].copy()

    if frame.empty:
        return pd.DataFrame(columns=["raster_path", "source", "item_id", "municipio", "catalog_path"])

    frame["raster_path"] = frame.apply(resolve_local_asset_path, axis=1)
    frame = frame[frame["raster_path"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=["raster_path", "source", "item_id", "municipio", "catalog_path"])

    if "source" not in frame.columns:
        frame["source"] = "google_solar"
    if "item_id" not in frame.columns:
        frame["item_id"] = frame["raster_path"].map(lambda path: Path(path).stem)
    if "municipio" not in frame.columns:
        frame["municipio"] = None

    frame["catalog_path"] = str(catalog_path.relative_to(PROJECT_ROOT))
    frame["raster_path"] = frame["raster_path"].map(lambda path: str(Path(path).resolve()))
    frame = frame.drop_duplicates(subset=["raster_path"]).reset_index(drop=True)
    return frame[["raster_path", "source", "item_id", "municipio", "catalog_path"]].copy()


def scan_solar_rasters() -> pd.DataFrame:
    rows = []
    for rgb in sorted(SOLAR_ROOT.rglob("*_rgb_*.tif")):
        municipio = rgb.parent.parent.name if len(rgb.parents) >= 2 else None
        if municipio not in TARGET_MUNICIPALITIES:
            continue
        rows.append({
            "raster_path": str(rgb.resolve()),
            "source": "google_solar",
            "item_id": rgb.stem,
            "municipio": municipio,
            "catalog_path": None,
        })
    return pd.DataFrame(rows)


def list_candidate_rasters() -> pd.DataFrame:
    catalog_rows = load_catalog_rasters()
    if not catalog_rows.empty:
        return catalog_rows
    return scan_solar_rasters()


def select_inference_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    if INFERENCE_MODE == "all":
        return candidates.reset_index(drop=True)
    rng = random.Random(SAMPLE_SEED)
    indices = list(candidates.index)
    rng.shuffle(indices)
    chosen = indices[: min(SAMPLE_COUNT, len(indices))]
    return candidates.loc[chosen].reset_index(drop=True)


def build_output_stub(source: str, raster_path: Path) -> str:
    safe_source = source.replace(":", "_").replace("/", "_")
    return f"{safe_source}_{raster_path.stem}"


def render_review_visuals(geoai_module, *, raster_path: Path, vector_path: Path) -> None:
    try:
        review_map = geoai_module.view_vector_interactive(str(vector_path), tiles=str(raster_path))
        if hasattr(review_map, "save"):
            review_map.save(str(vector_path.with_name(f"{vector_path.stem}_review_map.html")))
    except Exception as exc:
        print(f"review visualization skipped for {raster_path.name}: {exc}")
        return

    try:
        geoai_module.create_split_map(
            left_layer=str(vector_path),
            right_layer=str(raster_path),
            left_args={"style": {"color": "red", "fillOpacity": 0.2}},
            basemap=str(raster_path),
            m=review_map,
        )
        if hasattr(review_map, "save"):
            review_map.save(str(vector_path.with_name(f"{vector_path.stem}_split_map.html")))
    except Exception as exc:
        print(f"split-map visualization skipped for {raster_path.name}: {exc}")


def resolve_db_path() -> Path:
    value = os.getenv("VECTOR_DB")
    if value:
        p = Path(value)
        if not p.is_absolute():
            p = PROJECT_ROOT / p if len(p.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / p
        return p
    return PROJECT_ROOT / "data" / "PR_PV_plan_data.duckdb"


# %%
if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print(f"model not found at {MODEL_PATH} — run 10_geoai_solar_finetune first.")
        sys.exit(0)

    import geoai

    INFER_ROOT.mkdir(parents=True, exist_ok=True)
    all_polygons: list[gpd.GeoDataFrame] = []

    candidates = list_candidate_rasters()
    selected = select_inference_rows(candidates)
    print(f"inference mode={INFERENCE_MODE} selected {len(selected):,} raster(s)")
    if selected.empty:
        print("no candidate rasters available for inference.")
        sys.exit(0)

    for row in selected.itertuples(index=False):
        rgb = Path(row.raster_path)
        output_stub = build_output_stub(str(row.source), rgb)
        mask_out = INFER_ROOT / f"{output_stub}_mask.tif"
        vector_out = INFER_ROOT / f"{output_stub}_pred.geojson"
        if not mask_out.exists():
            geoai.object_detection(
                input_path=str(rgb),
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
            enriched = INFER_ROOT / f"{output_stub}_pred_props.geojson"
            gdf = geoai.add_geometric_properties(gdf)
            gdf.to_file(enriched, driver="GeoJSON")
            gdf["source_raster"] = str(rgb.relative_to(PROJECT_ROOT))
            gdf["raster_source"] = str(row.source)
            gdf["catalog_path"] = row.catalog_path
            gdf["sample_mode"] = INFERENCE_MODE
            all_polygons.append(gdf)

            if INFERENCE_MODE != "all":
                render_review_visuals(geoai, raster_path=rgb, vector_path=enriched)

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
