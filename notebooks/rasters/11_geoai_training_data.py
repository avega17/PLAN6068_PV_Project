# %% [markdown]
# # GeoAI Training Data Preparation
#
# Builds a clean Mask R-CNN fine-tuning dataset from Contextily/Esri
# WorldImagery chips aligned to occupied H3 cells that already contain OSM
# rooftop PV labels. The output is a paired `images/` + `masks/` directory
# in EPSG:3857 with matching filenames and binary `uint8` masks.

# %%
"""09_geoai_training_data.py

Export matched Contextily image/mask chips for Mask R-CNN training.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path

import contextily as ctx
import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from dotenv import load_dotenv
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from shapely import from_wkb
from shapely.geometry import box


def resolve_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in ("project_rules.md", ".git")):
            return candidate
    return current


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_CRS = "EPSG:4326"
MODEL_CRS = "EPSG:3857"
MANIFEST_TABLE = "pr_solar_tile_manifest"
PV_TABLE = "pr_osm_rooftop_pv_polygons"
DEFAULT_TRAIN_ROOT = PROJECT_ROOT / "outputs" / "geoai_train_contextily"


def _resolve_configured_path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


TRAIN_ROOT = _resolve_configured_path("GEOAI_TRAIN_ROOT", DEFAULT_TRAIN_ROOT)
TRAIN_IMAGE_DIR = TRAIN_ROOT / "images"
TRAIN_MASK_DIR = TRAIN_ROOT / "masks"
TRAIN_MANIFEST = TRAIN_ROOT / "training_chip_manifest.csv"
TRAIN_SUMMARY = TRAIN_ROOT / "training_chip_summary.json"
TRAIN_SOURCE = ctx.providers.Esri.WorldImagery

CHIP_PIXELS = int(os.getenv("GEOAI_CHIP_PIXELS", "512"))
CHIP_PADDING_FACTOR = float(os.getenv("GEOAI_CHIP_PADDING_FACTOR", "1.15"))
CHIP_SPAN_OVERRIDE_M = float(os.getenv("GEOAI_CHIP_SPAN_M", "0") or "0")
CONTEXTILY_ZOOM_RAW = (os.getenv("GEOAI_CONTEXTILY_ZOOM", "19") or "19").strip()
CONTEXTILY_USE_CACHE = os.getenv("GEOAI_CONTEXTILY_USE_CACHE", "1") == "1"
RESET_TRAIN_ROOT = os.getenv("GEOAI_RESET_TRAIN_ROOT", "1") == "1"
MAX_TILES = int(os.getenv("GEOAI_MAX_TILES", "0") or "0")
MAX_TILES_PER_MUNICIPALITY = int(os.getenv("GEOAI_MAX_TILES_PER_MUNICIPALITY", "0") or "0")
MIN_PRIORITY_SCORE = int(os.getenv("GEOAI_MIN_PRIORITY_SCORE", "3") or "3")


def _parse_csv_env(env_name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(env_name)
    if not value:
        return default
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    return parts or default


TARGET_MUNICIPALITIES = _parse_csv_env("GEOAI_TARGET_MUNICIPALITIES", ("San Juan", "Isabela"))


def resolve_db_path() -> Path:
    value = os.getenv("VECTOR_DB")
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path if len(path.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / path
        return path
    return PROJECT_ROOT / "data" / "PR_PV_plan_data.duckdb"


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")
    return con


def _to_bytes(value: object) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    return bytes(value) if not isinstance(value, bytes) else value


def _table_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    rows = con.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = ?;
        """,
        [table_name],
    ).fetchall()
    return {str(row[0]) for row in rows}


def _optional_cast_expr(
    available_columns: set[str],
    column_name: str,
    sql_type: str,
    *,
    alias: str | None = None,
    fallback: str,
) -> str:
    target = alias or column_name
    if column_name in available_columns:
        return f"CAST({column_name} AS {sql_type}) AS {target}"
    return f"{fallback} AS {target}"


def _resolve_zoom(value: str) -> int | str:
    return value if value.lower() == "auto" else int(value)


def _ensure_safe_output_root(train_root: Path) -> None:
    resolved = train_root.resolve()
    forbidden = {PROJECT_ROOT.resolve(), (PROJECT_ROOT / "output").resolve()}
    if resolved in forbidden:
        raise ValueError(f"Refusing to use overly broad training root: {train_root}")


def _reset_training_root(train_root: Path) -> None:
    _ensure_safe_output_root(train_root)
    for path in (train_root / "images", train_root / "masks", train_root / "annotations"):
        if path.exists():
            shutil.rmtree(path)
    for path in (train_root / "training_chip_manifest.csv", train_root / "training_chip_summary.json"):
        if path.exists():
            path.unlink()


def _prepare_training_root(train_root: Path) -> None:
    if RESET_TRAIN_ROOT:
        _reset_training_root(train_root)
    (train_root / "images").mkdir(parents=True, exist_ok=True)
    (train_root / "masks").mkdir(parents=True, exist_ok=True)


def load_training_cells(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    table_exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?;",
        [MANIFEST_TABLE],
    ).fetchone()[0]
    if not table_exists:
        raise RuntimeError(
            f"{MANIFEST_TABLE} not found; run notebooks/vectors/04_bg_tile_manifest.py first."
        )

    available_columns = _table_columns(con, MANIFEST_TABLE)
    names_sql = ", ".join("?" * len(TARGET_MUNICIPALITIES))
    municipio_geoid_expr = _optional_cast_expr(
        available_columns,
        "municipio_geoid",
        "VARCHAR",
        fallback="CAST(NULL AS VARCHAR)",
    )
    h3_cell_id_expr = _optional_cast_expr(
        available_columns,
        "h3_cell_id",
        "VARCHAR",
        fallback="CAST(tile_id AS VARCHAR)",
    )
    h3_resolution_expr = _optional_cast_expr(
        available_columns,
        "h3_resolution",
        "INTEGER",
        fallback="CAST(NULL AS INTEGER)",
    )
    df = con.execute(
        f"""
        SELECT
            CAST(tile_id AS VARCHAR) AS tile_id,
            CAST(bg_geoid AS VARCHAR) AS bg_geoid,
            {h3_cell_id_expr},
            {h3_resolution_expr},
            CAST(municipio AS VARCHAR) AS municipio,
            {municipio_geoid_expr},
            CAST(radius_m AS INTEGER) AS radius_m,
            CAST(building_count AS INTEGER) AS building_count,
            CAST(osm_pv_count AS INTEGER) AS osm_pv_count,
            CAST(priority_score AS INTEGER) AS priority_score,
            ST_AsWKB(geometry) AS geometry_wkb
        FROM {MANIFEST_TABLE}
        WHERE geometry IS NOT NULL
          AND municipio IN ({names_sql})
          AND COALESCE(osm_pv_count, 0) > 0
          AND COALESCE(priority_score, 0) >= ?
        ORDER BY priority_score DESC, osm_pv_count DESC, building_count DESC, tile_id;
        """,
        [*TARGET_MUNICIPALITIES, MIN_PRIORITY_SCORE],
    ).fetchdf()
    if df.empty:
        return gpd.GeoDataFrame(columns=["tile_id", "h3_cell_id", "municipio", "geometry"], geometry="geometry", crs=OUTPUT_CRS)

    if MAX_TILES_PER_MUNICIPALITY > 0:
        df = (
            df.groupby("municipio", sort=False, group_keys=False)
            .head(MAX_TILES_PER_MUNICIPALITY)
            .reset_index(drop=True)
        )

    if MAX_TILES > 0:
        df = df.head(MAX_TILES).copy()

    geometry = gpd.GeoSeries(df["geometry_wkb"].map(lambda value: from_wkb(_to_bytes(value))), crs=OUTPUT_CRS)
    return gpd.GeoDataFrame(df.drop(columns=["geometry_wkb"]), geometry=geometry, crs=OUTPUT_CRS)


def load_osm_pv_polygons(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    table_exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?;",
        [PV_TABLE],
    ).fetchone()[0]
    if not table_exists:
        raise RuntimeError(
            f"{PV_TABLE} not found; run notebooks/vectors/02_osm_pv_ingestion_and_viz.py first."
        )

    df = con.execute(
        f"""
        SELECT ST_AsWKB(geometry) AS geometry_wkb
        FROM {PV_TABLE}
        WHERE geometry IS NOT NULL;
        """
    ).fetchdf()
    if df.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=OUTPUT_CRS)

    geometry = gpd.GeoSeries(df["geometry_wkb"].map(lambda value: from_wkb(_to_bytes(value))), crs=OUTPUT_CRS)
    return gpd.GeoDataFrame(geometry=geometry, crs=OUTPUT_CRS)


def infer_chip_span_m(cells_3857: gpd.GeoDataFrame) -> int:
    if CHIP_SPAN_OVERRIDE_M > 0:
        return int(CHIP_SPAN_OVERRIDE_M)
    bounds = cells_3857.geometry.bounds
    max_span = float(
        max(
            (bounds["maxx"] - bounds["minx"]).max(),
            (bounds["maxy"] - bounds["miny"]).max(),
        )
    )
    padded = max_span * max(CHIP_PADDING_FACTOR, 1.0)
    return max(64, int(math.ceil(padded / 16.0) * 16))


def chip_bounds_from_geometry(geometry, chip_span_m: int) -> tuple[float, float, float, float]:
    centroid = geometry.centroid
    half_span = chip_span_m / 2.0
    return (
        float(centroid.x - half_span),
        float(centroid.y - half_span),
        float(centroid.x + half_span),
        float(centroid.y + half_span),
    )


def select_polygons_for_bounds(polygons_3857: gpd.GeoDataFrame, bounds: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    if polygons_3857.empty:
        return polygons_3857.iloc[0:0].copy()

    window = box(*bounds)
    try:
        candidate_idx = polygons_3857.sindex.query(window, predicate="intersects")
        subset = polygons_3857.iloc[list(candidate_idx)].copy()
    except Exception:
        subset = polygons_3857[polygons_3857.geometry.intersects(window)].copy()

    if subset.empty:
        return subset
    return subset[subset.geometry.intersects(window)].copy()


def fetch_contextily_chip(bounds: tuple[float, float, float, float], target_transform) -> np.ndarray:
    west, south, east, north = bounds
    image, extent = ctx.bounds2img(
        west,
        south,
        east,
        north,
        zoom=_resolve_zoom(CONTEXTILY_ZOOM_RAW),
        source=TRAIN_SOURCE,
        ll=False,
        use_cache=CONTEXTILY_USE_CACHE,
    )

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] > 3:
        image = image[:, :, :3]
    image = np.clip(np.asarray(image), 0, 255).astype(np.uint8)

    src_west, src_east, src_south, src_north = extent
    src_transform = from_bounds(src_west, src_south, src_east, src_north, image.shape[1], image.shape[0])

    out = np.zeros((3, CHIP_PIXELS, CHIP_PIXELS), dtype=np.uint8)
    for band_index in range(3):
        reproject(
            source=image[:, :, band_index],
            destination=out[band_index],
            src_transform=src_transform,
            src_crs=MODEL_CRS,
            dst_transform=target_transform,
            dst_crs=MODEL_CRS,
            resampling=Resampling.bilinear,
        )
    return out


def rasterize_labels(polygons_3857: gpd.GeoDataFrame, target_transform) -> np.ndarray:
    shapes = [
        (geometry, 1)
        for geometry in polygons_3857.geometry
        if geometry is not None and not geometry.is_empty
    ]
    return rasterize(
        shapes,
        out_shape=(CHIP_PIXELS, CHIP_PIXELS),
        transform=target_transform,
        fill=0,
        dtype="uint8",
    )


def write_chip(path: Path, data: np.ndarray, *, transform, crs: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = int(data.shape[0]) if data.ndim == 3 else 1
    height = int(data.shape[-2])
    width = int(data.shape[-1])
    write_data = data if data.ndim == 3 else data[np.newaxis, :, :]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=write_data.dtype,
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(write_data)


def export_training_dataset(cells: gpd.GeoDataFrame, osm_pv: gpd.GeoDataFrame) -> pd.DataFrame:
    if cells.empty:
        return pd.DataFrame()

    cells_3857 = cells.to_crs(MODEL_CRS)
    osm_pv_3857 = osm_pv.to_crs(MODEL_CRS)
    chip_span_m = infer_chip_span_m(cells_3857)
    records: list[dict[str, object]] = []

    for row in cells_3857.itertuples(index=False):
        bounds = chip_bounds_from_geometry(row.geometry, chip_span_m)
        label_subset = select_polygons_for_bounds(osm_pv_3857, bounds)
        if label_subset.empty:
            continue

        transform = from_bounds(*bounds, CHIP_PIXELS, CHIP_PIXELS)
        mask = rasterize_labels(label_subset, transform)
        if not mask.any():
            continue

        try:
            image = fetch_contextily_chip(bounds, transform)
        except Exception as exc:
            print(f"[warn] failed to fetch imagery for {row.tile_id}: {exc}")
            continue

        stem = str(row.tile_id)
        image_path = TRAIN_IMAGE_DIR / f"{stem}.tif"
        mask_path = TRAIN_MASK_DIR / f"{stem}.tif"
        write_chip(image_path, image, transform=transform, crs=MODEL_CRS)
        write_chip(mask_path, mask, transform=transform, crs=MODEL_CRS)

        records.append(
            {
                "tile_id": row.tile_id,
                "bg_geoid": row.bg_geoid,
                "h3_cell_id": row.h3_cell_id,
                "h3_resolution": row.h3_resolution,
                "municipio": row.municipio,
                "municipio_geoid": row.municipio_geoid,
                "priority_score": row.priority_score,
                "building_count": row.building_count,
                "osm_pv_count": row.osm_pv_count,
                "chip_pixels": CHIP_PIXELS,
                "chip_span_m": chip_span_m,
                "contextily_zoom": CONTEXTILY_ZOOM_RAW,
                "provider": "Esri.WorldImagery",
                "crs": MODEL_CRS,
                "west": bounds[0],
                "south": bounds[1],
                "east": bounds[2],
                "north": bounds[3],
                "label_polygon_count": int(len(label_subset)),
                "positive_pixels": int(mask.sum()),
                "image_path": str(image_path.relative_to(PROJECT_ROOT)),
                "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
            }
        )

    return pd.DataFrame.from_records(records)


def write_training_summary(records: pd.DataFrame) -> None:
    TRAIN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(TRAIN_MANIFEST, index=False)
    summary = {
        "train_root": str(TRAIN_ROOT.relative_to(PROJECT_ROOT)),
        "image_count": int(len(records)),
        "mask_count": int(len(records)),
        "chip_pixels": CHIP_PIXELS,
        "chip_span_m": int(records["chip_span_m"].iloc[0]) if not records.empty else None,
        "contextily_zoom": CONTEXTILY_ZOOM_RAW,
        "provider": "Esri.WorldImagery",
        "min_priority_score": MIN_PRIORITY_SCORE,
        "max_tiles_per_municipality": MAX_TILES_PER_MUNICIPALITY or None,
        "municipalities": sorted(records["municipio"].dropna().astype(str).unique().tolist()) if not records.empty else [],
    }
    TRAIN_SUMMARY.write_text(json.dumps(summary, indent=2))


# %%
if __name__ == "__main__":
    db_path = resolve_db_path()
    print(f"DuckDB: {db_path}")
    print(f"Training root: {TRAIN_ROOT}")

    con = connect(db_path)
    cells = load_training_cells(con)
    osm_pv = load_osm_pv_polygons(con)
    con.close()

    print(f"target municipalities: {', '.join(TARGET_MUNICIPALITIES)}")
    print(f"minimum priority score: {MIN_PRIORITY_SCORE}")
    if MAX_TILES_PER_MUNICIPALITY > 0:
        print(f"per-municipality tile cap: {MAX_TILES_PER_MUNICIPALITY}")
    if MAX_TILES > 0:
        print(f"global tile cap: {MAX_TILES}")
    print(f"selected H3 tiles with OSM PV labels: {len(cells):,}")
    print(f"OSM rooftop PV polygons: {len(osm_pv):,}")
    if cells.empty:
        print("no manifest tiles with OSM PV labels were found; run the manifest builder first.")
        sys.exit(0)
    if osm_pv.empty:
        print("no OSM rooftop PV polygons were found; run the OSM ingestion notebook first.")
        sys.exit(0)

    _prepare_training_root(TRAIN_ROOT)
    manifest = export_training_dataset(cells, osm_pv)
    if manifest.empty:
        print("no training chips were written; inspect Contextily connectivity and manifest coverage.")
        sys.exit(1)

    write_training_summary(manifest)
    print(f"wrote {len(manifest):,} Contextily image chips to {TRAIN_IMAGE_DIR}")
    print(f"wrote {len(manifest):,} binary mask chips to {TRAIN_MASK_DIR}")
    print(f"training manifest: {TRAIN_MANIFEST}")
