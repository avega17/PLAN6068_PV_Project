# %% [markdown]
# # GeoAI Training Data Preparation
# 
# Builds a clean training dataset from Contextily/Esri WorldImagery chips
# aligned to occupied H3 cells that already contain OSM rooftop PV labels.
# The exporter now writes both the original raw OSM masks and a second
# footprint-grounded mask variant clipped to matched Overture buildings, plus
# per-tile prompt JSON artifacts for downstream SAM benchmarking.

# %%
"""09_geoai_training_data.py

Export matched Contextily image/mask chips for GeoAI model training.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
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
from shapely.ops import unary_union


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

from utils.overture import DEFAULT_OVERTURE_BUILDINGS_TABLE

OUTPUT_CRS = "EPSG:4326"
MODEL_CRS = "EPSG:3857"
MANIFEST_TABLE = "pr_solar_tile_manifest"
PV_TABLE = "pr_osm_rooftop_pv_polygons"
OVERTURE_BUILDINGS_TABLE = DEFAULT_OVERTURE_BUILDINGS_TABLE
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
TRAIN_GROUNDED_MASK_DIR = TRAIN_ROOT / "grounded_masks"
TRAIN_PROMPT_DIR = TRAIN_ROOT / "prompt_artifacts"
TRAIN_REVIEW_DIR = TRAIN_ROOT / "review_artifacts"
TRAIN_MANIFEST = TRAIN_ROOT / "training_chip_manifest.csv"
TRAIN_SUMMARY = TRAIN_ROOT / "training_chip_summary.json"
PRIORITY2_TRAIN_ROOT = _resolve_configured_path("GEOAI_PRIORITY2_TRAIN_ROOT", PROJECT_ROOT / "outputs" / "geoai_train_contextily_priority")
PRIORITY1_TRAIN_ROOT = _resolve_configured_path("GEOAI_PRIORITY1_TRAIN_ROOT", PROJECT_ROOT / "outputs" / "geoai_train_contextily_priority")
TRAIN_SOURCE = ctx.providers.Esri.WorldImagery

CHIP_PIXELS = int(os.getenv("GEOAI_CHIP_PIXELS", "512"))
CHIP_PADDING_FACTOR = float(os.getenv("GEOAI_CHIP_PADDING_FACTOR", "1.15"))
CHIP_SPAN_OVERRIDE_M = float(os.getenv("GEOAI_CHIP_SPAN_M", "0") or "0")
CONTEXTILY_ZOOM_RAW = (os.getenv("GEOAI_CONTEXTILY_ZOOM", "19") or "19").strip()
CONTEXTILY_USE_CACHE = os.getenv("GEOAI_CONTEXTILY_USE_CACHE", "1") == "1"
RESET_TRAIN_ROOT = os.getenv("GEOAI_RESET_TRAIN_ROOT", "0") == "1"
RESET_PRIORITY_BAND_ROOTS = os.getenv("GEOAI_RESET_PRIORITY_BAND_ROOTS", "0") == "1"
OVERWRITE_EXISTING_CHIPS = os.getenv("GEOAI_OVERWRITE_EXISTING_CHIPS", "0") == "1"
MAX_TILES = int(os.getenv("GEOAI_MAX_TILES", "0") or "0")
MAX_TILES_PER_MUNICIPALITY = int(os.getenv("GEOAI_MAX_TILES_PER_MUNICIPALITY", "0") or "0")
MIN_PRIORITY_SCORE = int(os.getenv("GEOAI_MIN_PRIORITY_SCORE", "3") or "3")
MAX_PRIORITY_SCORE = int(os.getenv("GEOAI_MAX_PRIORITY_SCORE", "0") or "0") or None
PREVIEW_SAMPLE_COUNT = int(os.getenv("GEOAI_PREVIEW_SAMPLE_COUNT", "6") or "6")
PREVIEW_OVERLAY_ALPHA = float(os.getenv("GEOAI_PREVIEW_OVERLAY_ALPHA", "0.30") or "0.30")
SHOW_PREVIEW = os.getenv("GEOAI_SHOW_PREVIEW", "1") == "1"
WRITE_REVIEW_ARTIFACTS = os.getenv("GEOAI_WRITE_REVIEW_ARTIFACTS", "1") == "1"
EXPORT_PRIORITY2_VALIDATION = os.getenv("GEOAI_EXPORT_PRIORITY2_VALIDATION", "0") == "1"
EXPORT_PRIORITY1_VALIDATION = os.getenv("GEOAI_EXPORT_PRIORITY1_VALIDATION", "0") == "1"


def _parse_split_weights(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(
            "GEOAI_DATASET_SPLITS must contain three comma-separated weights for train,val,test."
        )
    weights = tuple(float(part) for part in parts)
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("GEOAI_DATASET_SPLITS must be non-negative and sum to a positive value.")
    total = sum(weights)
    return tuple(weight / total for weight in weights)


TRAIN_SPLIT_WEIGHTS = _parse_split_weights(os.getenv("GEOAI_DATASET_SPLITS", "0.70,0.15,0.15"))


def _parse_csv_env(env_name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(env_name)
    if not value:
        return default
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    return parts or default


TARGET_MUNICIPALITIES = _parse_csv_env("GEOAI_TARGET_MUNICIPALITIES", ("San Juan", "Isabela"))


@dataclass(frozen=True)
class TrainingLayout:
    train_root: Path
    image_dir: Path
    mask_dir: Path
    grounded_mask_dir: Path
    prompt_dir: Path
    review_dir: Path
    manifest_path: Path
    summary_path: Path


def resolve_training_layout(train_root: Path) -> TrainingLayout:
    return TrainingLayout(
        train_root=train_root,
        image_dir=train_root / "images",
        mask_dir=train_root / "masks",
        grounded_mask_dir=train_root / "grounded_masks",
        prompt_dir=train_root / "prompt_artifacts",
        review_dir=train_root / "review_artifacts",
        manifest_path=train_root / "training_chip_manifest.csv",
        summary_path=train_root / "training_chip_summary.json",
    )


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
    for path in (
        train_root / "images",
        train_root / "masks",
        train_root / "grounded_masks",
        train_root / "prompt_artifacts",
        train_root / "review_artifacts",
        train_root / "annotations",
    ):
        if path.exists():
            shutil.rmtree(path)
    for path in (train_root / "training_chip_manifest.csv", train_root / "training_chip_summary.json"):
        if path.exists():
            path.unlink()


def _prepare_training_root(train_root: Path, *, reset_root: bool) -> None:
    if reset_root:
        _reset_training_root(train_root)
    (train_root / "images").mkdir(parents=True, exist_ok=True)
    (train_root / "masks").mkdir(parents=True, exist_ok=True)
    (train_root / "grounded_masks").mkdir(parents=True, exist_ok=True)
    (train_root / "prompt_artifacts").mkdir(parents=True, exist_ok=True)
    (train_root / "review_artifacts").mkdir(parents=True, exist_ok=True)


def load_training_cells(
    con: duckdb.DuckDBPyConnection,
    *,
    target_municipalities: tuple[str, ...] = TARGET_MUNICIPALITIES,
    min_priority_score: int = MIN_PRIORITY_SCORE,
    max_priority_score: int | None = MAX_PRIORITY_SCORE,
    max_tiles: int = MAX_TILES,
    max_tiles_per_municipality: int = MAX_TILES_PER_MUNICIPALITY,
) -> gpd.GeoDataFrame:
    table_exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?;",
        [MANIFEST_TABLE],
    ).fetchone()[0]
    if not table_exists:
        raise RuntimeError(
            f"{MANIFEST_TABLE} not found; run notebooks/vectors/04_bg_tile_manifest.py first."
        )

    available_columns = _table_columns(con, MANIFEST_TABLE)
    names_sql = ", ".join("?" * len(target_municipalities))
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
    priority_max_sql = ""
    params: list[object] = [*target_municipalities, min_priority_score]
    if max_priority_score is not None:
        priority_max_sql = "\n          AND COALESCE(priority_score, 0) <= ?"
        params.append(max_priority_score)
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
          {priority_max_sql}
        ORDER BY priority_score DESC, osm_pv_count DESC, building_count DESC, tile_id;
        """,
        params,
    ).fetchdf()
    if df.empty:
        return gpd.GeoDataFrame(columns=["tile_id", "h3_cell_id", "municipio", "geometry"], geometry="geometry", crs=OUTPUT_CRS)

    if max_tiles_per_municipality > 0:
        df = (
            df.groupby("municipio", sort=False, group_keys=False)
            .head(max_tiles_per_municipality)
            .reset_index(drop=True)
        )

    if max_tiles > 0:
        df = df.head(max_tiles).copy()

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
        SELECT
            CAST(ROW_NUMBER() OVER () AS BIGINT) AS osm_label_id,
            ST_AsWKB(geometry) AS geometry_wkb
        FROM {PV_TABLE}
        WHERE geometry IS NOT NULL;
        """
    ).fetchdf()
    if df.empty:
        return gpd.GeoDataFrame(columns=["osm_label_id", "geometry"], geometry="geometry", crs=OUTPUT_CRS)

    geometry = gpd.GeoSeries(df["geometry_wkb"].map(lambda value: from_wkb(_to_bytes(value))), crs=OUTPUT_CRS)
    return gpd.GeoDataFrame(df.drop(columns=["geometry_wkb"]), geometry=geometry, crs=OUTPUT_CRS)


def load_overture_buildings(
    con: duckdb.DuckDBPyConnection,
    *,
    target_municipalities: tuple[str, ...] = TARGET_MUNICIPALITIES,
    h3_cell_ids: tuple[str, ...] | None = None,
) -> gpd.GeoDataFrame:
    table_exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?;",
        [OVERTURE_BUILDINGS_TABLE],
    ).fetchone()[0]
    if not table_exists:
        raise RuntimeError(
            f"{OVERTURE_BUILDINGS_TABLE} not found; run notebooks/vectors/03_overture_buildings_ingest.py first."
        )

    municipio_sql = ", ".join("?" * len(target_municipalities))
    params: list[object] = [*target_municipalities]
    h3_filter_sql = ""
    normalized_h3_ids = tuple(sorted({str(value) for value in (h3_cell_ids or ()) if value}))
    if normalized_h3_ids:
        h3_filter_sql = f"\n          AND h3_cell_id IN ({', '.join('?' * len(normalized_h3_ids))})"
        params.extend(normalized_h3_ids)

    df = con.execute(
        f"""
        SELECT
            CAST(id AS VARCHAR) AS building_id,
            CAST(municipality_name AS VARCHAR) AS municipio,
            CAST(municipality_geoid AS VARCHAR) AS municipio_geoid,
            CAST(h3_cell_id AS VARCHAR) AS h3_cell_id,
            ST_AsWKB(geometry) AS geometry_wkb
        FROM {OVERTURE_BUILDINGS_TABLE}
        WHERE geometry IS NOT NULL
                    AND municipality_name IN ({municipio_sql})
          {h3_filter_sql}
        ORDER BY municipio, building_id;
        """,
        params,
    ).fetchdf()
    if df.empty:
        return gpd.GeoDataFrame(
            columns=["building_id", "municipio", "municipio_geoid", "h3_cell_id", "geometry"],
            geometry="geometry",
            crs=OUTPUT_CRS,
        )

    geometry = gpd.GeoSeries(df["geometry_wkb"].map(lambda value: from_wkb(_to_bytes(value))), crs=OUTPUT_CRS)
    return gpd.GeoDataFrame(df.drop(columns=["geometry_wkb"]), geometry=geometry, crs=OUTPUT_CRS)


def assign_dataset_split(key: object) -> str:
    digest = hashlib.sha1(str(key).encode("utf-8")).hexdigest()
    fraction = int(digest[:12], 16) / float((16**12) - 1)
    train_cutoff = TRAIN_SPLIT_WEIGHTS[0]
    val_cutoff = TRAIN_SPLIT_WEIGHTS[0] + TRAIN_SPLIT_WEIGHTS[1]
    if fraction < train_cutoff:
        return "train"
    if fraction < val_cutoff:
        return "val"
    return "test"


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
    with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / "cache")) as tmpdir:
        raster_path = Path(tmpdir) / "contextily_chip.tif"
        ctx.bounds2raster(
            west,
            south,
            east,
            north,
            str(raster_path),
            zoom=_resolve_zoom(CONTEXTILY_ZOOM_RAW),
            source=TRAIN_SOURCE,
            ll=False,
            use_cache=CONTEXTILY_USE_CACHE,
        )
        with rasterio.open(raster_path) as src:
            image = src.read()
            src_transform = src.transform
            src_crs = src.crs or MODEL_CRS

    if image.ndim == 2:
        image = np.repeat(image[np.newaxis, :, :], 3, axis=0)
    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)
    if image.shape[0] > 3:
        image = image[:3]
    image = np.clip(np.asarray(image), 0, 255).astype(np.uint8)

    out = np.zeros((3, CHIP_PIXELS, CHIP_PIXELS), dtype=np.uint8)
    for band_index in range(3):
        reproject(
            source=image[band_index],
            destination=out[band_index],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=target_transform,
            dst_crs=MODEL_CRS,
            resampling=Resampling.bilinear,
        )
    return out


def preview_training_samples(
    records: pd.DataFrame,
    osm_pv: gpd.GeoDataFrame,
    *,
    sample_count: int = PREVIEW_SAMPLE_COUNT,
    overlay_alpha: float = PREVIEW_OVERLAY_ALPHA,
) -> None:
    if records.empty or sample_count <= 0:
        return

    import matplotlib.pyplot as plt

    sample = records.sample(n=min(sample_count, len(records)), random_state=0).reset_index(drop=True)
    polygons_3857 = osm_pv.to_crs(MODEL_CRS)
    ncols = min(3, len(sample))
    nrows = int(math.ceil(len(sample) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)

    for ax, row in zip(axes.flat, sample.itertuples(index=False)):
        image_path = PROJECT_ROOT / str(row.image_path)
        with rasterio.open(image_path) as src:
            image = src.read()
            bounds = src.bounds
            chip_crs = str(src.crs or MODEL_CRS)

        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
        if image.shape[0] > 3:
            image = image[:3]
        image = np.moveaxis(image, 0, -1).astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0

        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
        label_bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
        ax.imshow(image, extent=extent)

        label_subset = select_polygons_for_bounds(polygons_3857, label_bounds)
        if not label_subset.empty:
            label_subset.to_crs(chip_crs).plot(
                ax=ax,
                facecolor=(1.0, 0.3, 0.2, overlay_alpha),
                edgecolor=(1.0, 1.0, 1.0, 0.9),
                linewidth=0.8,
            )

        ax.set_title(f"{row.tile_id} | {row.municipio} | priority {row.priority_score}")
        ax.set_axis_off()

    for ax in axes.flat[len(sample):]:
        ax.set_axis_off()
    fig.suptitle("Training chip preview with OSM PV overlays", fontsize=14)
    fig.tight_layout()


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


def ground_labels_to_overture_buildings(
    label_subset_3857: gpd.GeoDataFrame,
    building_subset_3857: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, int | float]]:
    empty_grounded = gpd.GeoDataFrame(
        columns=[
            "osm_label_id",
            "building_id",
            "municipio",
            "municipio_geoid",
            "h3_cell_id",
            "overlap_area_m2",
            "raw_area_m2",
            "grounding_ratio",
            "geometry",
        ],
        geometry="geometry",
        crs=MODEL_CRS,
    )
    empty_prompts = pd.DataFrame(
        columns=[
            "building_id",
            "municipio",
            "municipio_geoid",
            "h3_cell_id",
            "matched_label_count",
            "matched_osm_label_ids",
            "match_overlap_area_m2",
            "building_geometry",
        ]
    )
    raw_label_count = int(len(label_subset_3857))
    diagnostics: dict[str, int | float] = {
        "raw_label_count": raw_label_count,
        "grounded_label_count": 0,
        "matched_building_count": 0,
        "unmatched_label_count": raw_label_count,
        "ambiguous_label_count": 0,
        "mean_grounding_ratio": 0.0,
    }
    if label_subset_3857.empty or building_subset_3857.empty:
        return empty_grounded, empty_prompts, diagnostics

    intersections = gpd.overlay(
        label_subset_3857[["osm_label_id", "geometry"]],
        building_subset_3857[["building_id", "municipio", "municipio_geoid", "h3_cell_id", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    if intersections.empty:
        return empty_grounded, empty_prompts, diagnostics

    intersections["overlap_area_m2"] = intersections.geometry.area.astype(float)
    match_counts = intersections.groupby("osm_label_id")["building_id"].nunique()
    grounded = (
        intersections.sort_values(["osm_label_id", "overlap_area_m2", "building_id"], ascending=[True, False, True])
        .drop_duplicates(subset=["osm_label_id"], keep="first")
        .reset_index(drop=True)
    )
    raw_area_lookup = label_subset_3857.set_index("osm_label_id").geometry.area.to_dict()
    grounded["raw_area_m2"] = grounded["osm_label_id"].map(raw_area_lookup).astype(float)
    grounded["grounding_ratio"] = np.where(
        grounded["raw_area_m2"] > 0,
        grounded["overlap_area_m2"] / grounded["raw_area_m2"],
        np.nan,
    )
    building_geometry_lookup = building_subset_3857.set_index("building_id").geometry.to_dict()
    grounded["building_geometry"] = grounded["building_id"].map(building_geometry_lookup)

    matched_label_counts = grounded.groupby("building_id")["osm_label_id"].nunique()
    overlap_by_building = grounded.groupby("building_id")["overlap_area_m2"].sum()
    label_ids_by_building = grounded.groupby("building_id")["osm_label_id"].apply(
        lambda values: [int(value) for value in sorted(values.tolist())]
    )
    prompt_geometry_by_building = grounded.groupby("building_id")["geometry"].apply(
        lambda values: unary_union(
            [geometry for geometry in values.tolist() if geometry is not None and not geometry.is_empty]
        )
    )
    prompt_features = grounded.drop_duplicates(subset=["building_id"], keep="first").copy()
    prompt_features["matched_label_count"] = prompt_features["building_id"].map(matched_label_counts).astype(int)
    prompt_features["matched_osm_label_ids"] = prompt_features["building_id"].map(label_ids_by_building)
    prompt_features["match_overlap_area_m2"] = prompt_features["building_id"].map(overlap_by_building).astype(float)
    prompt_features["prompt_geometry"] = prompt_features["building_id"].map(prompt_geometry_by_building)
    prompt_features = prompt_features[
        [
            "building_id",
            "municipio",
            "municipio_geoid",
            "h3_cell_id",
            "matched_label_count",
            "matched_osm_label_ids",
            "match_overlap_area_m2",
            "prompt_geometry",
            "building_geometry",
        ]
    ].reset_index(drop=True)

    diagnostics = {
        "raw_label_count": raw_label_count,
        "grounded_label_count": int(len(grounded)),
        "matched_building_count": int(prompt_features["building_id"].nunique()),
        "unmatched_label_count": int(max(raw_label_count - len(grounded), 0)),
        "ambiguous_label_count": int((match_counts > 1).sum()),
        "mean_grounding_ratio": float(grounded["grounding_ratio"].fillna(0.0).mean()) if not grounded.empty else 0.0,
    }
    grounded = gpd.GeoDataFrame(grounded.drop(columns=["building_geometry"]), geometry="geometry", crs=MODEL_CRS)
    return grounded, prompt_features, diagnostics


def _clamp_pixel_coordinate(value: float) -> float:
    return float(min(max(value, 0.0), float(CHIP_PIXELS - 1)))


def _world_to_pixel(transform, x: float, y: float) -> tuple[float, float]:
    col, row = (~transform) * (x, y)
    return _clamp_pixel_coordinate(float(col)), _clamp_pixel_coordinate(float(row))


def _bounds_to_pixel_bbox(bounds: tuple[float, float, float, float], transform) -> list[float]:
    minx, miny, maxx, maxy = bounds
    left_px, top_px = _world_to_pixel(transform, minx, maxy)
    right_px, bottom_px = _world_to_pixel(transform, maxx, miny)
    left_px, right_px = sorted((left_px, right_px))
    top_px, bottom_px = sorted((top_px, bottom_px))
    return [left_px, top_px, right_px, bottom_px]


def _centroid_to_pixel(transform, geometry) -> list[float]:
    centroid = geometry.centroid
    x_px, y_px = _world_to_pixel(transform, float(centroid.x), float(centroid.y))
    return [x_px, y_px]


def write_prompt_artifact(
    path: Path,
    row,
    *,
    bounds: tuple[float, float, float, float],
    transform,
    prompt_features: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tile_id": str(row.tile_id),
        "h3_cell_id": str(row.h3_cell_id) if pd.notna(row.h3_cell_id) else None,
        "municipio": str(row.municipio),
        "municipio_geoid": str(row.municipio_geoid) if pd.notna(row.municipio_geoid) else None,
        "chip_pixels": CHIP_PIXELS,
        "crs": MODEL_CRS,
        "chip_bounds_m": {
            "west": float(bounds[0]),
            "south": float(bounds[1]),
            "east": float(bounds[2]),
            "north": float(bounds[3]),
        },
        "prompt_source": "pv_label_union_by_building",
        "prompt_source_version": 2,
        "prompt_count": 0,
        "prompts": [],
    }
    for prompt in prompt_features.itertuples(index=False):
        geometry = prompt.prompt_geometry
        if geometry is None or geometry.is_empty:
            continue
        building_geometry = prompt.building_geometry
        building_bbox_model_crs = (
            [float(value) for value in building_geometry.bounds]
            if building_geometry is not None and not building_geometry.is_empty
            else None
        )
        building_bbox_pixels = (
            _bounds_to_pixel_bbox(building_geometry.bounds, transform)
            if building_geometry is not None and not building_geometry.is_empty
            else None
        )
        building_centroid_model_crs = (
            [float(building_geometry.centroid.x), float(building_geometry.centroid.y)]
            if building_geometry is not None and not building_geometry.is_empty
            else None
        )
        building_centroid_pixels = (
            _centroid_to_pixel(transform, building_geometry)
            if building_geometry is not None and not building_geometry.is_empty
            else None
        )
        payload["prompts"].append(
            {
                "building_id": str(prompt.building_id),
                "municipio": str(prompt.municipio) if pd.notna(prompt.municipio) else None,
                "municipio_geoid": str(prompt.municipio_geoid) if pd.notna(prompt.municipio_geoid) else None,
                "h3_cell_id": str(prompt.h3_cell_id) if pd.notna(prompt.h3_cell_id) else None,
                "prompt_geometry_kind": "pv_label_union_within_building",
                "matched_label_count": int(prompt.matched_label_count),
                "matched_osm_label_ids": [int(value) for value in prompt.matched_osm_label_ids],
                "match_overlap_area_m2": float(prompt.match_overlap_area_m2),
                "bbox_model_crs": [float(value) for value in geometry.bounds],
                "bbox_pixels": _bounds_to_pixel_bbox(geometry.bounds, transform),
                "centroid_model_crs": [float(geometry.centroid.x), float(geometry.centroid.y)],
                "centroid_pixels": _centroid_to_pixel(transform, geometry),
                "building_bbox_model_crs": building_bbox_model_crs,
                "building_bbox_pixels": building_bbox_pixels,
                "building_centroid_model_crs": building_centroid_model_crs,
                "building_centroid_pixels": building_centroid_pixels,
            }
        )
    payload["prompt_count"] = int(len(payload["prompts"]))
    path.write_text(json.dumps(payload, indent=2))


def _prepare_image_for_plot(image: np.ndarray) -> np.ndarray:
    display = np.asarray(image)
    if display.ndim == 2:
        display = np.repeat(display[np.newaxis, :, :], 3, axis=0)
    if display.shape[0] == 1:
        display = np.repeat(display, 3, axis=0)
    if display.shape[0] > 3:
        display = display[:3]
    display = np.moveaxis(display, 0, -1).astype(np.float32)
    if display.max() > 1.0:
        display /= 255.0
    return display


def write_review_artifact(
    path: Path,
    row,
    *,
    image: np.ndarray,
    raw_mask: np.ndarray,
    grounded_mask: np.ndarray,
    bounds: tuple[float, float, float, float],
    split_name: str,
    matched_building_count: int,
    overlay_alpha: float = PREVIEW_OVERLAY_ALPHA,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    display_image = _prepare_image_for_plot(image)
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    raw_overlay = np.ma.masked_where(raw_mask == 0, raw_mask)
    grounded_overlay = np.ma.masked_where(grounded_mask == 0, grounded_mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
    axes = axes[0]
    for ax in axes:
        ax.imshow(display_image, extent=extent)
        ax.set_axis_off()

    axes[0].set_title("Source raster")

    axes[1].imshow(raw_overlay, extent=extent, cmap="Reds", alpha=overlay_alpha, vmin=0, vmax=1)
    axes[1].set_title(f"Raw OSM mask | pixels={int(raw_mask.sum())}")

    axes[2].imshow(grounded_overlay, extent=extent, cmap="Greens", alpha=overlay_alpha, vmin=0, vmax=1)
    axes[2].set_title(
        f"Grounded mask | pixels={int(grounded_mask.sum())} | buildings={matched_building_count}"
    )

    fig.suptitle(f"{row.tile_id} | {row.municipio} | split={split_name}", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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


def _build_manifest_row(
    row,
    *,
    bounds: tuple[float, float, float, float],
    chip_span_m: int,
    raw_label_polygon_count: int,
    grounded_label_polygon_count: int,
    matched_building_count: int,
    unmatched_label_count: int,
    ambiguous_label_count: int,
    mean_grounding_ratio: float,
    raw_positive_pixels: int,
    grounded_positive_pixels: int,
    image_path: Path,
    mask_path: Path,
    grounded_mask_path: Path,
    prompt_path: Path,
    review_path: Path | None,
    split_name: str,
    split_key: str,
    status: str,
) -> dict[str, object]:
    return {
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
        "dataset_split": split_name,
        "split_key": split_key,
        "west": bounds[0],
        "south": bounds[1],
        "east": bounds[2],
        "north": bounds[3],
        "label_polygon_count": int(raw_label_polygon_count),
        "raw_label_polygon_count": int(raw_label_polygon_count),
        "grounded_label_polygon_count": int(grounded_label_polygon_count),
        "matched_building_count": int(matched_building_count),
        "unmatched_label_count": int(unmatched_label_count),
        "ambiguous_label_count": int(ambiguous_label_count),
        "mean_grounding_ratio": float(mean_grounding_ratio),
        "positive_pixels": int(raw_positive_pixels),
        "raw_positive_pixels": int(raw_positive_pixels),
        "grounded_positive_pixels": int(grounded_positive_pixels),
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
        "raw_mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
        "grounded_mask_path": str(grounded_mask_path.relative_to(PROJECT_ROOT)),
        "prompt_artifact_path": str(prompt_path.relative_to(PROJECT_ROOT)),
        "review_artifact_path": str(review_path.relative_to(PROJECT_ROOT)) if review_path is not None else None,
        "status": status,
    }


def _read_positive_pixels(mask_path: Path) -> int:
    with rasterio.open(mask_path) as src:
        return int(src.read(1).sum())


def merge_training_manifest(manifest_path: Path, records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records
    merged = records.copy()
    if manifest_path.exists():
        try:
            existing = pd.read_csv(manifest_path)
        except Exception:
            existing = pd.DataFrame()
        if not existing.empty and "tile_id" in existing.columns:
            merged = pd.concat([existing, merged], ignore_index=True)
            merged = merged.drop_duplicates(subset=["tile_id"], keep="last").reset_index(drop=True)
    return merged


def export_training_dataset(
    cells: gpd.GeoDataFrame,
    osm_pv: gpd.GeoDataFrame,
    overture_buildings: gpd.GeoDataFrame,
    *,
    layout: TrainingLayout,
    overwrite_existing: bool = OVERWRITE_EXISTING_CHIPS,
) -> pd.DataFrame:
    if cells.empty:
        return pd.DataFrame()

    cells_3857 = cells.to_crs(MODEL_CRS)
    osm_pv_3857 = osm_pv.to_crs(MODEL_CRS)
    overture_buildings_3857 = overture_buildings.to_crs(MODEL_CRS)
    chip_span_m = infer_chip_span_m(cells_3857)
    records: list[dict[str, object]] = []

    for row in cells_3857.itertuples(index=False):
        bounds = chip_bounds_from_geometry(row.geometry, chip_span_m)
        label_subset = select_polygons_for_bounds(osm_pv_3857, bounds)
        if label_subset.empty:
            continue

        transform = from_bounds(*bounds, CHIP_PIXELS, CHIP_PIXELS)
        raw_mask = rasterize_labels(label_subset, transform)
        if not raw_mask.any():
            continue

        building_subset = select_polygons_for_bounds(overture_buildings_3857, bounds)
        grounded_labels, prompt_features, grounding_metrics = ground_labels_to_overture_buildings(
            label_subset,
            building_subset,
        )
        grounded_mask = rasterize_labels(grounded_labels, transform)

        stem = str(row.tile_id)
        image_path = layout.image_dir / f"{stem}.tif"
        mask_path = layout.mask_dir / f"{stem}.tif"
        grounded_mask_path = layout.grounded_mask_dir / f"{stem}.tif"
        prompt_path = layout.prompt_dir / f"{stem}.json"
        review_path = layout.review_dir / f"{stem}_review.png"

        image: np.ndarray | None = None
        fetched_image = False
        updated_assets = False
        if overwrite_existing or not image_path.exists():
            try:
                image = fetch_contextily_chip(bounds, transform)
            except Exception as exc:
                print(f"[warn] failed to fetch imagery for {row.tile_id}: {exc}")
                continue
            write_chip(image_path, image, transform=transform, crs=MODEL_CRS)
            fetched_image = True

        if overwrite_existing or not mask_path.exists():
            write_chip(mask_path, raw_mask, transform=transform, crs=MODEL_CRS)
            raw_positive_pixels = int(raw_mask.sum())
            updated_assets = True
        else:
            raw_positive_pixels = _read_positive_pixels(mask_path)

        if overwrite_existing or not grounded_mask_path.exists():
            write_chip(grounded_mask_path, grounded_mask, transform=transform, crs=MODEL_CRS)
            grounded_positive_pixels = int(grounded_mask.sum())
            updated_assets = True
        else:
            grounded_positive_pixels = _read_positive_pixels(grounded_mask_path)

        if overwrite_existing or not prompt_path.exists():
            write_prompt_artifact(
                prompt_path,
                row,
                bounds=bounds,
                transform=transform,
                prompt_features=prompt_features,
            )
            updated_assets = True

        split_key = str(row.h3_cell_id) if pd.notna(row.h3_cell_id) else str(row.tile_id)
        split_name = assign_dataset_split(split_key)
        if WRITE_REVIEW_ARTIFACTS and (overwrite_existing or not review_path.exists()):
            if image is None:
                with rasterio.open(image_path) as src:
                    image = src.read()
            write_review_artifact(
                review_path,
                row,
                image=image,
                raw_mask=raw_mask,
                grounded_mask=grounded_mask,
                bounds=bounds,
                split_name=split_name,
                matched_building_count=int(grounding_metrics["matched_building_count"]),
            )
            updated_assets = True

        status = "fetched" if fetched_image else "updated" if updated_assets else "reused"

        records.append(
            _build_manifest_row(
                row,
                bounds=bounds,
                chip_span_m=chip_span_m,
                raw_label_polygon_count=int(grounding_metrics["raw_label_count"]),
                grounded_label_polygon_count=int(grounding_metrics["grounded_label_count"]),
                matched_building_count=int(grounding_metrics["matched_building_count"]),
                unmatched_label_count=int(grounding_metrics["unmatched_label_count"]),
                ambiguous_label_count=int(grounding_metrics["ambiguous_label_count"]),
                mean_grounding_ratio=float(grounding_metrics["mean_grounding_ratio"]),
                raw_positive_pixels=raw_positive_pixels,
                grounded_positive_pixels=grounded_positive_pixels,
                image_path=image_path,
                mask_path=mask_path,
                grounded_mask_path=grounded_mask_path,
                prompt_path=prompt_path,
                review_path=review_path if WRITE_REVIEW_ARTIFACTS else None,
                split_name=split_name,
                split_key=split_key,
                status=status,
            )
        )

    return pd.DataFrame.from_records(records)


def write_training_summary(
    records: pd.DataFrame,
    *,
    layout: TrainingLayout,
    min_priority_score: int,
    max_priority_score: int | None,
) -> pd.DataFrame:
    layout.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    merged_records = merge_training_manifest(layout.manifest_path, records)
    merged_records.to_csv(layout.manifest_path, index=False)
    status_counts = merged_records["status"].fillna("unknown").value_counts().to_dict() if not merged_records.empty else {}
    summary = {
        "train_root": str(layout.train_root.relative_to(PROJECT_ROOT)),
        "image_count": int(len(merged_records)),
        "mask_count": int(len(merged_records)),
        "grounded_mask_count": int(len(merged_records)),
        "prompt_artifact_count": int(len(merged_records)),
        "review_artifact_count": int(merged_records["review_artifact_path"].notna().sum()) if not merged_records.empty and "review_artifact_path" in merged_records.columns else 0,
        "chip_pixels": CHIP_PIXELS,
        "chip_span_m": int(merged_records["chip_span_m"].iloc[0]) if not merged_records.empty else None,
        "contextily_zoom": CONTEXTILY_ZOOM_RAW,
        "provider": "Esri.WorldImagery",
        "dataset_splits": merged_records["dataset_split"].value_counts().to_dict() if not merged_records.empty else {},
        "raw_positive_pixels_total": int(merged_records["raw_positive_pixels"].sum()) if not merged_records.empty else 0,
        "grounded_positive_pixels_total": int(merged_records["grounded_positive_pixels"].sum()) if not merged_records.empty else 0,
        "matched_buildings_total": int(merged_records["matched_building_count"].sum()) if not merged_records.empty else 0,
        "mean_grounding_ratio": float(merged_records["mean_grounding_ratio"].mean()) if not merged_records.empty else 0.0,
        "min_priority_score": min_priority_score,
        "max_priority_score": max_priority_score,
        "max_tiles_per_municipality": MAX_TILES_PER_MUNICIPALITY or None,
        "overwrite_existing_chips": OVERWRITE_EXISTING_CHIPS,
        "status_counts": status_counts,
        "municipalities": sorted(merged_records["municipio"].dropna().astype(str).unique().tolist()) if not merged_records.empty else [],
    }
    layout.summary_path.write_text(json.dumps(summary, indent=2))
    return merged_records


def export_priority_band_dataset(
    *,
    train_root: Path,
    min_priority_score: int,
    max_priority_score: int,
    reset_root: bool,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame] | None:
    layout = resolve_training_layout(train_root)
    con = connect(resolve_db_path())
    cells = load_training_cells(
        con,
        min_priority_score=min_priority_score,
        max_priority_score=max_priority_score,
    )
    osm_pv = load_osm_pv_polygons(con)
    cell_h3_ids = tuple(sorted({str(value) for value in cells["h3_cell_id"].dropna().astype(str).tolist()})) if not cells.empty else ()
    overture_buildings = load_overture_buildings(con, h3_cell_ids=cell_h3_ids)
    con.close()

    if cells.empty:
        print(f"no tiles found for priority band {min_priority_score}..{max_priority_score} under {train_root}.")
        return None
    if osm_pv.empty:
        print("no OSM rooftop PV polygons were found; run the OSM ingestion notebook first.")
        return None

    _prepare_training_root(train_root, reset_root=reset_root)
    manifest = export_training_dataset(cells, osm_pv, overture_buildings, layout=layout)
    if manifest.empty:
        print(f"no training chips were written for priority band {min_priority_score}..{max_priority_score}.")
        return None

    merged_manifest = write_training_summary(
        manifest,
        layout=layout,
        min_priority_score=min_priority_score,
        max_priority_score=max_priority_score,
    )
    print(
        f"priority {min_priority_score}..{max_priority_score} export wrote {len(merged_manifest):,} chips to {layout.image_dir}"
    )
    print(f"manifest: {layout.manifest_path}")
    return merged_manifest, osm_pv

# %%
if __name__ == "__main__":
    db_path = resolve_db_path()
    print(f"DuckDB: {db_path}")
    print(f"Training root: {TRAIN_ROOT}")
    print(f"Contextily HTTP cache enabled: {CONTEXTILY_USE_CACHE}")
    print(f"Reuse existing chips: {not OVERWRITE_EXISTING_CHIPS}")
    if not RESET_TRAIN_ROOT:
        print("training root reset disabled: existing image/mask chips will be reused when possible.")

    con = connect(db_path)
    cells = load_training_cells(con)
    osm_pv = load_osm_pv_polygons(con)
    cell_h3_ids = tuple(sorted({str(value) for value in cells["h3_cell_id"].dropna().astype(str).tolist()})) if not cells.empty else ()
    overture_buildings = load_overture_buildings(con, h3_cell_ids=cell_h3_ids)
    con.close()
    train_layout = resolve_training_layout(TRAIN_ROOT)

    print(f"target municipalities: {', '.join(TARGET_MUNICIPALITIES)}")
    print(f"minimum priority score: {MIN_PRIORITY_SCORE}")
    if MAX_PRIORITY_SCORE is not None:
        print(f"maximum priority score: {MAX_PRIORITY_SCORE}")
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
    if overture_buildings.empty:
        print("no Overture buildings were found for the selected training cells; run the Overture ingestion notebook first.")
        sys.exit(1)

    _prepare_training_root(TRAIN_ROOT, reset_root=RESET_TRAIN_ROOT)
    manifest = export_training_dataset(cells, osm_pv, overture_buildings, layout=train_layout)
    if manifest.empty:
        print("no training chips were written; inspect Contextily connectivity and manifest coverage.")
        sys.exit(1)

    manifest = write_training_summary(
        manifest,
        layout=train_layout,
        min_priority_score=MIN_PRIORITY_SCORE,
        max_priority_score=MAX_PRIORITY_SCORE,
    )
    print(f"wrote {len(manifest):,} Contextily image chips to {train_layout.image_dir}")
    print(f"wrote {len(manifest):,} raw binary mask chips to {train_layout.mask_dir}")
    print(f"wrote {len(manifest):,} grounded mask chips to {train_layout.grounded_mask_dir}")
    print(f"wrote {len(manifest):,} prompt artifacts to {train_layout.prompt_dir}")
    if WRITE_REVIEW_ARTIFACTS:
        print(f"wrote {len(manifest):,} review PNG artifacts to {train_layout.review_dir}")
    print(f"training manifest: {train_layout.manifest_path}")
    print("To rebuild from scratch, set GEOAI_RESET_TRAIN_ROOT=1. To widen beyond the seed-neighborhood subset, rerun with GEOAI_MIN_PRIORITY_SCORE=1 and optional municipality/cap overrides.")

# %% [markdown]
# ## Preview exported chips with OSM PV overlays
# 
# Uses the saved GeoTIFF chips plus the source OSM rooftop PV polygons for a
# quick visual QA pass. Set `GEOAI_SHOW_PREVIEW=0` or `GEOAI_PREVIEW_SAMPLE_COUNT=0`
# to skip this step.

# %%
if __name__ == "__main__":
    if SHOW_PREVIEW and 'manifest' in locals() and not manifest.empty:
        preview_training_samples(manifest, osm_pv)

# %% [markdown]
# ## Optional priority-2 validation export
# 
# Set `GEOAI_EXPORT_PRIORITY2_VALIDATION=1` to write the next validation band
# (priority score exactly 2) into a separate root without disturbing the seed
# training dataset.

# %%
if __name__ == "__main__" and EXPORT_PRIORITY2_VALIDATION:
    priority2_result = export_priority_band_dataset(
        train_root=PRIORITY2_TRAIN_ROOT,
        min_priority_score=2,
        max_priority_score=2,
        reset_root=RESET_PRIORITY_BAND_ROOTS,
    )
    if SHOW_PREVIEW and priority2_result is not None:
        priority2_manifest, priority2_osm_pv = priority2_result
        preview_training_samples(priority2_manifest, priority2_osm_pv)

# %% [markdown]
# ## Optional priority-1 validation export
# 
# Set `GEOAI_EXPORT_PRIORITY1_VALIDATION=1` to write the long-tail priority-1
# tiles into a separate root. This keeps the lower-priority OSM validation band
# isolated from the default seed-neighborhood training subset.

# %%
if __name__ == "__main__" and EXPORT_PRIORITY1_VALIDATION:
    priority1_result = export_priority_band_dataset(
        train_root=PRIORITY1_TRAIN_ROOT,
        min_priority_score=1,
        max_priority_score=1,
        reset_root=RESET_PRIORITY_BAND_ROOTS,
    )
    if SHOW_PREVIEW and priority1_result is not None:
        priority1_manifest, priority1_osm_pv = priority1_result
        preview_training_samples(priority1_manifest, priority1_osm_pv)


