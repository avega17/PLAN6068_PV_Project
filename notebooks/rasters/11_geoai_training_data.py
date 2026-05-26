# %% [markdown]
# # GeoAI Training Data Preparation
# 
# Builds a clean training dataset from either Contextily/Esri WorldImagery or
# local NAIP STAC chips aligned to occupied H3 cells that already contain OSM
# rooftop PV labels. The exporter writes both the original raw OSM masks and a
# second footprint-grounded mask variant clipped to matched Overture buildings,
# plus per-tile prompt JSON artifacts for downstream benchmarking.

# %%
"""09_geoai_training_data.py

Export matched ESRI or NAIP image/mask chips for GeoAI model training.
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
from shapely import from_wkb, union_all
from shapely.geometry import box, shape

try:
    import ipywidgets as widgets
    from IPython.display import display
except Exception:
    widgets = None
    display = None


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

from utils.geoai_preview_sources import collect_naip_stac_preview_rasters, infer_stac_source_name
from utils.overture import DEFAULT_OVERTURE_BUILDINGS_TABLE
from utils.census import CANONICAL_COUNTY_TABLE
from utils.raster_stac_index import resolve_vector_db_path

OUTPUT_CRS = "EPSG:4326"
MODEL_CRS = "EPSG:3857"
MANIFEST_TABLE = "pr_solar_tile_manifest"
PV_TABLE = "pr_osm_rooftop_pv_polygons"
OVERTURE_BUILDINGS_TABLE = DEFAULT_OVERTURE_BUILDINGS_TABLE
NAIP_SOURCE_NAMES = ("pr_naip", "naip_2021_pr")
ESRI_CONTEXTILY_SOURCE = ctx.providers.Esri.WorldImagery
STAC_TILE_MANIFEST = PROJECT_ROOT / "outputs" / "stac_tiles" / "pr_stac_tile_manifest.parquet"
MAP_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "maps"
VALID_IMAGERY_SOURCES = {"esri", "naip"}
VALID_DATASET_COHORTS = {"train_pool", "holdout_priority3", "priority_band"}
NOTEBOOK_DATASET_COHORTS = ("train_pool", "holdout_priority3")
HOLDOUT_GROUP_LABEL = "priority3_seed_neighborhoods"
SEED_NEIGHBORHOODS = (
    {"seed_name": "Puerto Nuevo", "municipio": "San Juan", "cache_name": "Puerto Nuevo", "osm_query": "Puerto Nuevo, San Juan, Puerto Rico"},
    {"seed_name": "Mora", "municipio": "Isabela", "cache_name": "Mora", "osm_query": "Mora, Isabela, Puerto Rico"},
)
IMAGERY_SOURCE_LABELS = {
    "esri": "Esri WorldImagery",
    "naip": "Local NAIP STAC",
}
DATASET_COHORT_LABELS = {
    "train_pool": "Train/val/test pool with seed-neighborhood test tiles",
    "holdout_priority3": "Seed-neighborhood holdout only",
    "priority_band": "Priority band export",
}


def _resolve_configured_path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _normalize_choice(value: str, *, name: str, valid_values: set[str]) -> str:
    normalized = (value or "").strip().lower()
    if normalized not in valid_values:
        allowed = ", ".join(sorted(valid_values))
        raise ValueError(f"{name} must be one of: {allowed}. Received: {value!r}")
    return normalized


def _default_train_root(imagery_source: str, dataset_cohort: str) -> Path:
    imagery_source = _normalize_choice(imagery_source, name="imagery_source", valid_values=VALID_IMAGERY_SOURCES)
    dataset_cohort = _normalize_choice(dataset_cohort, name="dataset_cohort", valid_values=VALID_DATASET_COHORTS)
    imagery_slug = "contextily" if imagery_source == "esri" else imagery_source
    if dataset_cohort == "train_pool":
        return PROJECT_ROOT / f"outputs/geoai_train_{imagery_slug}"
    if dataset_cohort == "holdout_priority3":
        return PROJECT_ROOT / f"outputs/geoai_holdout_{imagery_slug}_priority3"
    return PROJECT_ROOT / f"outputs/geoai_{imagery_slug}_priority_band"


IMAGERY_SOURCE = _normalize_choice(
    os.getenv("GEOAI_IMAGERY_SOURCE", "esri"),
    name="GEOAI_IMAGERY_SOURCE",
    valid_values=VALID_IMAGERY_SOURCES,
)
DATASET_COHORT = _normalize_choice(
    os.getenv("GEOAI_DATASET_COHORT", "train_pool"),
    name="GEOAI_DATASET_COHORT",
    valid_values=set(NOTEBOOK_DATASET_COHORTS),
)
SPLIT_POLICY = (os.getenv("GEOAI_SPLIT_POLICY", "priority3_holdout") or "priority3_holdout").strip().lower()
HOLDOUT_PRIORITY_SCORE = int(os.getenv("GEOAI_HOLDOUT_PRIORITY_SCORE", "3") or "3")
ENABLE_WIDGETS = os.getenv("GEOAI_ENABLE_WIDGETS", "1") == "1"


DEFAULT_TRAIN_ROOT = _default_train_root(IMAGERY_SOURCE, DATASET_COHORT)


TRAIN_ROOT = _resolve_configured_path("GEOAI_TRAIN_ROOT", DEFAULT_TRAIN_ROOT)
PRIORITY2_TRAIN_ROOT = _resolve_configured_path("GEOAI_PRIORITY2_TRAIN_ROOT", PROJECT_ROOT / "outputs" / "geoai_train_contextily_priority")
PRIORITY1_TRAIN_ROOT = _resolve_configured_path("GEOAI_PRIORITY1_TRAIN_ROOT", PROJECT_ROOT / "outputs" / "geoai_train_contextily_priority")

CHIP_PIXELS = int(os.getenv("GEOAI_CHIP_PIXELS", "512"))
CHIP_PADDING_FACTOR = float(os.getenv("GEOAI_CHIP_PADDING_FACTOR", "1.15"))
CHIP_SPAN_OVERRIDE_M = float(os.getenv("GEOAI_CHIP_SPAN_M", "0") or "0")
CONTEXTILY_ZOOM_RAW = (os.getenv("GEOAI_CONTEXTILY_ZOOM", "19") or "19").strip()
CONTEXTILY_USE_CACHE = os.getenv("GEOAI_CONTEXTILY_USE_CACHE", "1") == "1"
RESET_TRAIN_ROOT = os.getenv("GEOAI_RESET_TRAIN_ROOT", "0") == "1"
RESET_PRIORITY_BAND_ROOTS = os.getenv("GEOAI_RESET_PRIORITY_BAND_ROOTS", "0") == "1"
OVERWRITE_EXISTING_CHIPS = os.getenv("GEOAI_OVERWRITE_EXISTING_CHIPS", "0") == "1"
REPAIR_EXISTING_SPLITS_ONLY = os.getenv("GEOAI_REPAIR_EXISTING_SPLITS_ONLY", "0") == "1"
REFETCH_PRIORITY_SEED_MISMATCHES_ONLY = os.getenv("GEOAI_REFETCH_PRIORITY_SEED_MISMATCHES_ONLY", "0") == "1"
MAX_TILES = int(os.getenv("GEOAI_MAX_TILES", "0") or "0")
MAX_TILES_PER_MUNICIPALITY = int(os.getenv("GEOAI_MAX_TILES_PER_MUNICIPALITY", "0") or "0")
MIN_PRIORITY_SCORE = int(os.getenv("GEOAI_MIN_PRIORITY_SCORE", "1") or "1")
MAX_PRIORITY_SCORE = int(os.getenv("GEOAI_MAX_PRIORITY_SCORE", "0") or "0") or None
PREVIEW_SAMPLE_COUNT = int(os.getenv("GEOAI_PREVIEW_SAMPLE_COUNT", "6") or "6")
PREVIEW_OVERLAY_ALPHA = float(os.getenv("GEOAI_PREVIEW_OVERLAY_ALPHA", "0.30") or "0.30")
MIN_VALIDATION_SHARE = float(os.getenv("GEOAI_MIN_VALIDATION_SHARE", "0.10") or "0.10")
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
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] in "([{" and cleaned[-1] in ")]}":
        cleaned = cleaned[1:-1]

    parts: list[str] = []
    for raw_part in cleaned.split(","):
        part = raw_part.strip().strip("\"'").strip()
        part = part.strip("()[]{}")
        if part:
            parts.append(part)

    parsed = tuple(parts)
    if parsed and parsed != tuple(default):
        if any(token != token.strip("\"'").strip() for token in parsed):
            print(f"note: normalized {env_name} tokens to {parsed}")
    parts = parsed
    return parts or default


CASE_STUDY_MUNICIPALITIES = _parse_csv_env(
    "GEOAI_CASE_STUDY_MUNICIPALITIES",
    ("San Juan", "Isabela"),
)


# Empty means "all municipalities". Set GEOAI_TARGET_MUNICIPALITIES to a
# comma-separated subset (or tuple-style string) when you want a scoped export.
_TARGET_MUNICIPALITIES_RAW = (os.getenv("GEOAI_TARGET_MUNICIPALITIES", "") or "").strip()
if _TARGET_MUNICIPALITIES_RAW.lower() in {"", "all", "*"}:
    TARGET_MUNICIPALITIES: tuple[str, ...] = tuple()
else:
    TARGET_MUNICIPALITIES = _parse_csv_env("GEOAI_TARGET_MUNICIPALITIES", tuple())

_TRAINING_WIDGETS: dict[str, object] | None = None


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


def _widgets_enabled() -> bool:
    return ENABLE_WIDGETS and widgets is not None and display is not None and "ipykernel" in sys.modules


def _maybe_initialize_training_widgets() -> None:
    global _TRAINING_WIDGETS
    if _TRAINING_WIDGETS is not None or not _widgets_enabled():
        return

    imagery_widget = widgets.Dropdown(
        options=[(IMAGERY_SOURCE_LABELS[key], key) for key in sorted(VALID_IMAGERY_SOURCES)],
        value=IMAGERY_SOURCE,
        description="Imagery",
        layout=widgets.Layout(width="320px"),
    )
    cohort_widget = widgets.Dropdown(
        options=[(DATASET_COHORT_LABELS[key], key) for key in NOTEBOOK_DATASET_COHORTS],
        value=DATASET_COHORT,
        description="Cohort",
        layout=widgets.Layout(width="420px"),
    )
    help_text = widgets.HTML(
        value=(
            "<b>Export controls</b><br>"
            "Use ESRI or NAIP imagery for the same chip contract. "
            "The train pool keeps seed-neighborhood tiles as the test split, while the holdout cohort exports only those seed tiles."
        )
    )
    display(widgets.VBox([help_text, widgets.HBox([imagery_widget, cohort_widget])]))
    _TRAINING_WIDGETS = {
        "imagery_source": imagery_widget,
        "dataset_cohort": cohort_widget,
    }


def apply_training_widget_overrides() -> None:
    global IMAGERY_SOURCE, DATASET_COHORT, TRAIN_ROOT
    if not _TRAINING_WIDGETS:
        return

    IMAGERY_SOURCE = _normalize_choice(
        str(_TRAINING_WIDGETS["imagery_source"].value),
        name="imagery_source",
        valid_values=VALID_IMAGERY_SOURCES,
    )
    DATASET_COHORT = _normalize_choice(
        str(_TRAINING_WIDGETS["dataset_cohort"].value),
        name="dataset_cohort",
        valid_values=set(NOTEBOOK_DATASET_COHORTS),
    )
    if "GEOAI_TRAIN_ROOT" not in os.environ:
        TRAIN_ROOT = _default_train_root(IMAGERY_SOURCE, DATASET_COHORT)


def resolve_db_path() -> Path:
    return resolve_vector_db_path()


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
    dataset_cohort: str = DATASET_COHORT,
    holdout_priority_score: int = HOLDOUT_PRIORITY_SCORE,
) -> gpd.GeoDataFrame:
    dataset_cohort = _normalize_choice(
        dataset_cohort,
        name="dataset_cohort",
        valid_values=VALID_DATASET_COHORTS,
    )
    table_exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?;",
        [MANIFEST_TABLE],
    ).fetchone()[0]
    if not table_exists:
        raise RuntimeError(
            f"{MANIFEST_TABLE} not found; run notebooks/vectors/04_bg_tile_manifest.py first."
        )

    available_columns = _table_columns(con, MANIFEST_TABLE)
    municipio_filter_sql = ""
    params: list[object] = []
    if target_municipalities:
        names_sql = ", ".join("?" * len(target_municipalities))
        municipio_filter_sql = f"\n          AND municipio IN ({names_sql})"
        params.extend(target_municipalities)
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
    priority_filters: list[str] = ["COALESCE(priority_score, 0) >= ?"]
    params.append(min_priority_score)
    if max_priority_score is not None:
        priority_filters.append("COALESCE(priority_score, 0) <= ?")
        params.append(max_priority_score)
    priority_sql = "" if not priority_filters else "\n          AND " + "\n          AND ".join(priority_filters)
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
                    {municipio_filter_sql}
          AND COALESCE(osm_pv_count, 0) > 0
                    {priority_sql}
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

    params: list[object] = []
    municipio_filter_sql = ""
    if target_municipalities:
        municipio_sql = ", ".join("?" * len(target_municipalities))
        municipio_filter_sql = f"\n          AND municipality_name IN ({municipio_sql})"
        params.extend(target_municipalities)
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
                    {municipio_filter_sql}
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


def load_municipality_boundaries(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    table_exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?;",
        [CANONICAL_COUNTY_TABLE],
    ).fetchone()[0]
    if not table_exists:
        return gpd.GeoDataFrame(columns=["GEOID", "NAME", "geometry"], geometry="geometry", crs=OUTPUT_CRS)

    df = con.execute(
        f"""
        SELECT
            CAST(GEOID AS VARCHAR) AS geoid,
            CAST(NAME AS VARCHAR) AS name,
            ST_AsWKB(geometry) AS geometry_wkb
        FROM {CANONICAL_COUNTY_TABLE}
        WHERE geometry IS NOT NULL
        ORDER BY name;
        """
    ).fetchdf()
    if df.empty:
        return gpd.GeoDataFrame(columns=["geoid", "name", "geometry"], geometry="geometry", crs=OUTPUT_CRS)

    geometry = gpd.GeoSeries(df["geometry_wkb"].map(lambda value: from_wkb(_to_bytes(value))), crs=OUTPUT_CRS)
    return gpd.GeoDataFrame(df.drop(columns=["geometry_wkb"]), geometry=geometry, crs=OUTPUT_CRS)


def _seed_cache_directories() -> tuple[Path, ...]:
    return (
        PROJECT_ROOT / "cache",
        PROJECT_ROOT / "notebooks" / "vectors" / "cache",
    )


def _load_cached_seed_geometry(seed_name: str, municipio: str):
    for cache_dir in _seed_cache_directories():
        if not cache_dir.exists():
            continue
        for path in sorted(cache_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            if not isinstance(payload, list):
                continue
            for record in payload:
                if str(record.get("name") or "").strip() != seed_name:
                    continue
                display_name = str(record.get("display_name") or "")
                if municipio not in display_name:
                    continue
                geojson = record.get("geojson")
                if not isinstance(geojson, dict):
                    continue
                if str(geojson.get("type") or "") not in {"Polygon", "MultiPolygon"}:
                    continue
                return shape(geojson)
    return None


def load_seed_neighborhoods() -> gpd.GeoDataFrame:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for seed in SEED_NEIGHBORHOODS:
        geometry = _load_cached_seed_geometry(str(seed["cache_name"]), str(seed["municipio"]))
        if geometry is None:
            missing.append(str(seed["seed_name"]))
            continue
        rows.append(
            {
                "seed_name": str(seed["seed_name"]),
                "municipio": str(seed["municipio"]),
                "geometry": geometry,
            }
        )
    if missing:
        print(f"[warn] missing cached seed neighborhood polygons: {', '.join(missing)}")
    if not rows:
        return gpd.GeoDataFrame(columns=["seed_name", "municipio", "geometry"], geometry="geometry", crs=OUTPUT_CRS)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=OUTPUT_CRS)


def _dataset_split_fraction(key: object) -> float:
    digest = hashlib.sha1(str(key).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float((16**12) - 1)


def assign_dataset_split(key: object) -> str:
    fraction = _dataset_split_fraction(key)
    train_cutoff = TRAIN_SPLIT_WEIGHTS[0]
    val_cutoff = TRAIN_SPLIT_WEIGHTS[0] + TRAIN_SPLIT_WEIGHTS[1]
    if fraction < train_cutoff:
        return "train"
    if fraction < val_cutoff:
        return "val"
    return "test"


def assign_case_study_split(key: object) -> str:
    active_weight = TRAIN_SPLIT_WEIGHTS[0] + TRAIN_SPLIT_WEIGHTS[1]
    if active_weight <= 0 or TRAIN_SPLIT_WEIGHTS[1] <= 0:
        return "train"
    train_cutoff = TRAIN_SPLIT_WEIGHTS[0] / active_weight
    return "train" if _dataset_split_fraction(key) < train_cutoff else "val"


def annotate_split_geography(
    cells: gpd.GeoDataFrame,
    *,
    seed_neighborhoods: gpd.GeoDataFrame,
    case_study_municipalities: tuple[str, ...] = CASE_STUDY_MUNICIPALITIES,
) -> gpd.GeoDataFrame:
    if cells.empty:
        annotated = cells.copy()
        annotated["seed_name"] = pd.Series(dtype="object")
        annotated["is_seed_neighborhood"] = pd.Series(dtype="bool")
        annotated["is_case_study_municipality"] = pd.Series(dtype="bool")
        annotated["split_key"] = pd.Series(dtype="object")
        annotated["dataset_split"] = pd.Series(dtype="object")
        annotated["split_bucket"] = pd.Series(dtype="object")
        return annotated

    annotated = cells.copy()
    annotated["seed_name"] = pd.Series([None] * len(annotated), dtype="object")
    if not seed_neighborhoods.empty:
        cells_3857 = annotated.to_crs(MODEL_CRS)
        chip_span_m = infer_chip_span_m(cells_3857)
        chip_footprints = gpd.GeoDataFrame(
            annotated[["tile_id"]].copy(),
            geometry=[box(*chip_bounds_from_geometry(geometry, chip_span_m)) for geometry in cells_3857.geometry],
            crs=MODEL_CRS,
        ).to_crs(OUTPUT_CRS)
        seed_join = gpd.sjoin(
            chip_footprints,
            seed_neighborhoods[["seed_name", "geometry"]],
            predicate="intersects",
            how="left",
        )
        if not seed_join.empty:
            seed_lookup = (
                seed_join.dropna(subset=["seed_name"])
                .sort_values(["tile_id", "seed_name"], kind="stable")
                .drop_duplicates(subset=["tile_id"], keep="first")
                [["tile_id", "seed_name"]]
            )
            annotated = annotated.merge(seed_lookup, on="tile_id", how="left", suffixes=("", "_matched"))
            if "seed_name_matched" in annotated.columns:
                annotated["seed_name"] = annotated["seed_name_matched"].combine_first(annotated["seed_name"])
                annotated = annotated.drop(columns=["seed_name_matched"])

    annotated["is_case_study_municipality"] = annotated["municipio"].fillna("").isin(case_study_municipalities)
    annotated["is_seed_neighborhood"] = annotated["seed_name"].notna() & annotated["is_case_study_municipality"]
    annotated["split_key"] = annotated["h3_cell_id"].where(annotated["h3_cell_id"].notna(), annotated["tile_id"]).astype(str)
    annotated["dataset_split"] = "train"
    case_study_mask = (~annotated["is_seed_neighborhood"]) & annotated["is_case_study_municipality"]
    if case_study_mask.any():
        annotated.loc[case_study_mask, "dataset_split"] = annotated.loc[case_study_mask, "split_key"].map(assign_case_study_split)
    annotated.loc[annotated["is_seed_neighborhood"], "dataset_split"] = "test"
    annotated["split_bucket"] = np.select(
        [
            annotated["is_seed_neighborhood"],
            case_study_mask & (annotated["dataset_split"] == "val"),
            case_study_mask & (annotated["dataset_split"] == "train"),
        ],
        [
            "seed_neighborhood_test",
            "case_study_val",
            "case_study_train",
        ],
        default="island_train",
    )
    return annotated


def select_cells_for_dataset_cohort(cells: gpd.GeoDataFrame, *, dataset_cohort: str) -> gpd.GeoDataFrame:
    dataset_cohort = _normalize_choice(
        dataset_cohort,
        name="dataset_cohort",
        valid_values=VALID_DATASET_COHORTS,
    )
    if dataset_cohort == "holdout_priority3":
        return cells[cells["dataset_split"] == "test"].copy()
    return cells.copy()


def rebalance_validation_split(
    cells: gpd.GeoDataFrame,
    *,
    min_validation_share: float = MIN_VALIDATION_SHARE,
) -> gpd.GeoDataFrame:
    if cells.empty or min_validation_share <= 0:
        return cells

    adjusted = cells.copy()
    test_count = int((adjusted["dataset_split"] == "test").sum())
    if test_count <= 0:
        return adjusted

    total_count = int(len(adjusted))
    target_share = min(float(min_validation_share), float(test_count) / float(max(total_count, 1)))
    target_val_count = int(math.ceil(total_count * target_share))
    current_val_count = int((adjusted["dataset_split"] == "val").sum())
    promote_count = max(target_val_count - current_val_count, 0)
    if promote_count <= 0:
        return adjusted

    promotion_pool = adjusted[adjusted["split_bucket"] == "case_study_train"].copy()
    if promotion_pool.empty:
        return adjusted

    promotion_pool["split_fraction"] = promotion_pool["split_key"].map(_dataset_split_fraction)
    promotions = promotion_pool.sort_values(
        ["split_fraction", "osm_pv_count", "building_count", "tile_id"],
        ascending=[False, False, False, True],
        kind="stable",
    ).head(promote_count)
    if promotions.empty:
        return adjusted

    promotion_ids = set(promotions["tile_id"].astype(str))
    adjusted.loc[adjusted["tile_id"].astype(str).isin(promotion_ids), "dataset_split"] = "val"
    adjusted.loc[adjusted["tile_id"].astype(str).isin(promotion_ids), "split_bucket"] = "case_study_val"
    return adjusted


def repair_existing_manifest_splits(
    records: pd.DataFrame,
    *,
    cells: gpd.GeoDataFrame,
    dataset_cohort: str,
) -> pd.DataFrame:
    if records.empty:
        return records.copy()

    join_key = "tile_id" if "tile_id" in records.columns and "tile_id" in cells.columns else "h3_cell_id"
    lookup_columns = [
        join_key,
        "dataset_split",
        "split_key",
        "split_bucket",
        "seed_name",
        "is_seed_neighborhood",
        "is_case_study_municipality",
    ]
    split_lookup = cells[[column_name for column_name in lookup_columns if column_name in cells.columns]].drop_duplicates(
        subset=[join_key],
        keep="first",
    )
    repaired = records.drop(
        columns=[
            column_name
            for column_name in (
                "dataset_split",
                "split_key",
                "split_bucket",
                "seed_name",
                "is_seed_neighborhood",
                "is_case_study_municipality",
                "holdout_group",
                "split_policy",
                "split_seed",
            )
            if column_name in records.columns
        ]
    ).copy()
    repaired = repaired.merge(split_lookup, on=join_key, how="left")
    missing_lookup = repaired["dataset_split"].isna()
    missing_count = int(missing_lookup.sum())
    if missing_count:
        print(
            f"warning: {missing_count:,} existing manifest rows were missing from the current split lookup; "
            "falling back to split_key hashing for those rows."
        )
    key_series = repaired.get("h3_cell_id", repaired.get("tile_id", pd.Series(index=repaired.index, dtype="object")))
    repaired["split_key"] = repaired["split_key"].fillna(key_series.astype(str))
    repaired["dataset_split"] = repaired["dataset_split"].fillna(repaired["split_key"].map(assign_dataset_split))
    if "split_bucket" in repaired.columns:
        inferred_split_bucket = pd.Series(
            np.select(
                [
                    repaired["dataset_split"] == "test",
                    repaired["dataset_split"] == "val",
                    repaired.get("is_case_study_municipality", False).fillna(False),
                ],
                [
                    "seed_neighborhood_test",
                    "case_study_val",
                    "case_study_train",
                ],
                default="island_train",
            ),
            index=repaired.index,
            dtype="object",
        )
        repaired["split_bucket"] = repaired["split_bucket"].fillna(
            inferred_split_bucket
        )
    repaired["holdout_group"] = np.where(repaired["dataset_split"] == "test", HOLDOUT_GROUP_LABEL, None)
    repaired["split_policy"] = SPLIT_POLICY
    repaired["split_seed"] = "sha1(split_key)"
    if "dataset_cohort" in repaired.columns:
        repaired["dataset_cohort"] = dataset_cohort
    if dataset_cohort == "holdout_priority3":
        repaired = repaired[repaired["dataset_split"] == "test"].copy()
    return repaired.reset_index(drop=True)


def run_training_export_workflow() -> tuple[pd.DataFrame | None, gpd.GeoDataFrame | None]:
    con = connect(resolve_db_path())
    cells = load_training_cells(
        con,
        dataset_cohort=DATASET_COHORT,
        target_municipalities=TARGET_MUNICIPALITIES,
        min_priority_score=MIN_PRIORITY_SCORE,
        max_priority_score=MAX_PRIORITY_SCORE,
        max_tiles_per_municipality=MAX_TILES_PER_MUNICIPALITY,
        max_tiles=MAX_TILES,
    )
    osm_pv = load_osm_pv_polygons(con)
    municipality_boundaries = load_municipality_boundaries(con)
    cell_h3_ids = tuple(sorted({str(value) for value in cells["h3_cell_id"].dropna().astype(str).tolist()})) if not cells.empty else ()
    overture_buildings = load_overture_buildings(con, h3_cell_ids=cell_h3_ids)
    con.close()

    seed_neighborhoods = load_seed_neighborhoods()
    cells = annotate_split_geography(cells, seed_neighborhoods=seed_neighborhoods)
    cells = rebalance_validation_split(cells)
    export_cells = select_cells_for_dataset_cohort(cells, dataset_cohort=DATASET_COHORT)
    train_layout = resolve_training_layout(TRAIN_ROOT)
    highlighted_boundaries = municipality_boundaries[
        municipality_boundaries["name"].fillna("").isin(CASE_STUDY_MUNICIPALITIES)
    ].copy() if not municipality_boundaries.empty else municipality_boundaries

    if TARGET_MUNICIPALITIES:
        print(f"target municipalities: {', '.join(TARGET_MUNICIPALITIES)}")
        if len(TARGET_MUNICIPALITIES) <= 2:
            print(
                "note: municipality scope is constrained; for island-wide export set "
                "GEOAI_TARGET_MUNICIPALITIES=all (or unset it)."
            )
    else:
        print("target municipalities: ALL (island-wide)")
    print(f"minimum priority score: {MIN_PRIORITY_SCORE}")
    if MAX_PRIORITY_SCORE is not None:
        print(f"maximum priority score: {MAX_PRIORITY_SCORE}")
    if MAX_TILES_PER_MUNICIPALITY > 0:
        print(f"per-municipality tile cap: {MAX_TILES_PER_MUNICIPALITY}")
    if MAX_TILES > 0:
        print(f"global tile cap: {MAX_TILES}")
    if CASE_STUDY_MUNICIPALITIES:
        print(f"case-study municipalities: {', '.join(CASE_STUDY_MUNICIPALITIES)}")
    print(f"selected H3 tiles with OSM PV labels: {len(cells):,}")
    print(f"planned geographic split counts: {cells['dataset_split'].value_counts().to_dict() if not cells.empty else {}}")
    print(f"OSM rooftop PV polygons: {len(osm_pv):,}")
    if cells.empty:
        print("no manifest tiles with OSM PV labels were found; run the manifest builder first.")
        return None, osm_pv
    if osm_pv.empty:
        print("no OSM rooftop PV polygons were found; run the OSM ingestion notebook first.")
        return None, osm_pv
    if overture_buildings.empty:
        print("no Overture buildings were found for the selected training cells; run the Overture ingestion notebook first.")
        return None, osm_pv

    split_map_path = write_split_spatial_overview(
        cells,
        layout=train_layout,
        title="Candidate training tiles by split (pre-fetch preview)",
        file_name="dataset_split_plan_overview.png",
        boundaries=municipality_boundaries,
        highlight_boundaries=highlighted_boundaries,
    )
    if split_map_path is not None:
        print(f"pre-fetch split overview: {split_map_path}")

    if REFETCH_PRIORITY_SEED_MISMATCHES_ONLY:
        mismatch_cells = identify_priority_seed_mismatch_cells(cells)
        if mismatch_cells.empty:
            print("priority/seed mismatch refetch skipped: no mismatch cells found.")
            return None, osm_pv
        if not train_layout.manifest_path.exists():
            print("priority/seed mismatch refetch skipped: no existing training manifest was found.")
            return None, osm_pv

        mismatch_tile_ids = set(mismatch_cells["tile_id"].dropna().astype(str))
        print(f"priority/seed mismatch cells selected for refetch: {len(mismatch_tile_ids):,}")
        remove_manifest_rows_and_artifacts(
            train_layout,
            tile_ids=mismatch_tile_ids,
            imagery_source=IMAGERY_SOURCE,
            dataset_cohort=DATASET_COHORT,
        )
        refreshed_records = export_training_dataset(
            mismatch_cells,
            osm_pv,
            overture_buildings,
            layout=train_layout,
            overwrite_existing=True,
            imagery_source=IMAGERY_SOURCE,
            dataset_cohort=DATASET_COHORT,
        )
        if refreshed_records.empty:
            print("priority/seed mismatch refetch wrote no replacement chips; inspect imagery and label coverage.")
            return refreshed_records, osm_pv
        refreshed_manifest = write_training_summary(
            refreshed_records,
            layout=train_layout,
            min_priority_score=MIN_PRIORITY_SCORE,
            max_priority_score=MAX_PRIORITY_SCORE,
            imagery_source=IMAGERY_SOURCE,
            dataset_cohort=DATASET_COHORT,
        )
        summarize_dataset_split_diagnostics(cells, refreshed_manifest)
        refreshed_map_path = write_split_spatial_overview(
            refreshed_manifest,
            layout=train_layout,
            title="Training manifest by split after priority/seed mismatch refetch",
            boundaries=municipality_boundaries,
            highlight_boundaries=highlighted_boundaries,
        )
        if refreshed_map_path is not None:
            print(f"refreshed split spatial overview: {refreshed_map_path}")
        print(f"refetched {len(refreshed_records):,} priority/seed mismatch chips into {train_layout.train_root}")
        return refreshed_manifest, osm_pv

    if REPAIR_EXISTING_SPLITS_ONLY:
        if not train_layout.manifest_path.exists():
            print("split repair skipped: no existing training manifest was found.")
            return None, osm_pv
        existing_manifest = pd.read_csv(train_layout.manifest_path)
        repaired_manifest = repair_existing_manifest_splits(
            existing_manifest,
            cells=cells,
            dataset_cohort=DATASET_COHORT,
        )
        repaired_manifest = write_training_summary(
            repaired_manifest,
            layout=train_layout,
            min_priority_score=MIN_PRIORITY_SCORE,
            max_priority_score=MAX_PRIORITY_SCORE,
            imagery_source=IMAGERY_SOURCE,
            dataset_cohort=DATASET_COHORT,
        )
        summarize_dataset_split_diagnostics(cells, repaired_manifest)
        repaired_map_path = write_split_spatial_overview(
            repaired_manifest,
            layout=train_layout,
            title="Existing training manifest by split (repaired metadata)",
            boundaries=municipality_boundaries,
            highlight_boundaries=highlighted_boundaries,
        )
        if repaired_map_path is not None:
            print(f"repaired split spatial overview: {repaired_map_path}")
        print(f"repaired split metadata for {len(repaired_manifest):,} existing chips in {train_layout.manifest_path}")
        return repaired_manifest, osm_pv

    _prepare_training_root(TRAIN_ROOT, reset_root=RESET_TRAIN_ROOT)
    manifest = export_training_dataset(
        export_cells,
        osm_pv,
        overture_buildings,
        layout=train_layout,
        imagery_source=IMAGERY_SOURCE,
        dataset_cohort=DATASET_COHORT,
    )
    if manifest.empty:
        print("no training chips were written; inspect imagery availability, NAIP coverage, and manifest filtering.")
        return manifest, osm_pv

    manifest = write_training_summary(
        manifest,
        layout=train_layout,
        min_priority_score=MIN_PRIORITY_SCORE,
        max_priority_score=MAX_PRIORITY_SCORE,
        imagery_source=IMAGERY_SOURCE,
        dataset_cohort=DATASET_COHORT,
    )
    summarize_dataset_split_diagnostics(export_cells, manifest)
    split_map_path = write_split_spatial_overview(
        manifest,
        layout=train_layout,
        boundaries=municipality_boundaries,
        highlight_boundaries=highlighted_boundaries,
    )
    if split_map_path is not None:
        print(f"split spatial overview: {split_map_path}")
    print(f"wrote {len(manifest):,} {IMAGERY_SOURCE_LABELS[IMAGERY_SOURCE]} image chips to {train_layout.image_dir}")
    print(f"wrote {len(manifest):,} raw binary mask chips to {train_layout.mask_dir}")
    print(f"wrote {len(manifest):,} grounded mask chips to {train_layout.grounded_mask_dir}")
    return manifest, osm_pv


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
            source=ESRI_CONTEXTILY_SOURCE,
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


def load_naip_tile_index(
    *,
    target_municipalities: tuple[str, ...] = TARGET_MUNICIPALITIES,
) -> dict[str, dict[str, object]]:
    source_priority = {name: index for index, name in enumerate(NAIP_SOURCE_NAMES)}

    if STAC_TILE_MANIFEST.exists():
        frame = pd.read_parquet(STAC_TILE_MANIFEST)
        if frame.empty:
            return {}
        if "source" in frame.columns:
            frame = frame[frame["source"].isin(NAIP_SOURCE_NAMES)].copy()
        if "asset_role" in frame.columns:
            frame = frame[frame["asset_role"].fillna("").str.lower() == "visual"].copy()
        if "status" in frame.columns:
            frame = frame[frame["status"].fillna("").str.lower() == "fetched"].copy()
        if target_municipalities and "municipio" in frame.columns:
            frame = frame[frame["municipio"].isin(target_municipalities)].copy()
        if frame.empty or "h3_cell_id" not in frame.columns:
            return {}

        frame["tile_abs_path"] = frame["tile_path"].map(lambda value: (PROJECT_ROOT / str(value)).resolve())
        frame["local_asset_abs_path"] = frame["local_asset_path"].map(
            lambda value: (PROJECT_ROOT / str(value)).resolve() if isinstance(value, str) and value else None
        )
        recency_candidates = (
            "acquired",
            "datetime",
            "item_datetime",
            "start_datetime",
            "end_datetime",
            "updated",
            "fetched_at",
            "ingested_at",
            "created",
        )
        recency_series = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
        for column_name in recency_candidates:
            if column_name not in frame.columns:
                continue
            parsed = pd.to_datetime(frame[column_name], errors="coerce", utc=True)
            recency_series = recency_series.where(parsed.isna(), parsed)
        frame["item_recency"] = recency_series
        frame["source_rank"] = frame["source"].map(lambda value: source_priority.get(str(value), len(source_priority)))
        frame = frame.sort_values(
            ["item_recency", "source_rank", "building_count", "h3_cell_id"],
            ascending=[False, True, False, True],
            kind="stable",
        )

        index: dict[str, dict[str, object]] = {}
        for item in frame.itertuples(index=False):
            h3_cell_id = str(getattr(item, "h3_cell_id", "") or "")
            if not h3_cell_id or h3_cell_id in index:
                continue

            source_path = None
            local_asset_path = getattr(item, "local_asset_abs_path", None)
            tile_path = getattr(item, "tile_abs_path", None)
            if isinstance(local_asset_path, Path) and local_asset_path.exists():
                source_path = local_asset_path
            elif isinstance(tile_path, Path) and tile_path.exists():
                source_path = tile_path
            if source_path is None:
                continue

            index[h3_cell_id] = {
                "source": str(getattr(item, "source", "naip")),
                "source_path": source_path,
                "tile_path": tile_path if isinstance(tile_path, Path) else source_path,
                # Bounds written by notebook 08 (EPSG:3857). Present when the
                # tile was exported via export_square_tiles_for_asset; fall
                # back to None so the caller can read them from the GeoTIFF.
                "west_3857": getattr(item, "west_3857", None) or None,
                "south_3857": getattr(item, "south_3857", None) or None,
                "east_3857": getattr(item, "east_3857", None) or None,
                "north_3857": getattr(item, "north_3857", None) or None,
            }
        return index

    candidates = collect_naip_stac_preview_rasters(PROJECT_ROOT, exclude_stems=set())
    index: dict[str, dict[str, object]] = {}
    for path in candidates:
        h3_cell_id = path.stem.split("_visual")[0]
        if not h3_cell_id or h3_cell_id in index:
            continue
        index[h3_cell_id] = {
            "source": infer_stac_source_name(path),
            "source_path": path,
            "tile_path": path,
        }
    return index


def fetch_local_raster_chip(source_path: Path, target_transform) -> np.ndarray:
    out = np.zeros((3, CHIP_PIXELS, CHIP_PIXELS), dtype=np.float32)
    with rasterio.open(source_path) as src:
        source_band_count = max(1, min(src.count, 3))
        if source_band_count == 1:
            band_indices = (1, 1, 1)
        elif source_band_count == 2:
            band_indices = (1, 2, 2)
        else:
            band_indices = (1, 2, 3)

        for destination_index, source_band_index in enumerate(band_indices):
            reproject(
                source=rasterio.band(src, source_band_index),
                destination=out[destination_index],
                src_transform=src.transform,
                src_crs=src.crs or MODEL_CRS,
                dst_transform=target_transform,
                dst_crs=MODEL_CRS,
                resampling=Resampling.bilinear,
            )

    if out.max() <= 1.0:
        out *= 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def fetch_naip_chip(row, target_transform, naip_tile_index: dict[str, dict[str, object]]) -> tuple[np.ndarray, str]:
    if pd.isna(row.h3_cell_id):
        raise RuntimeError(f"{row.tile_id} is missing h3_cell_id; cannot resolve NAIP tile")

    h3_cell_id = str(row.h3_cell_id)
    tile_record = naip_tile_index.get(h3_cell_id)
    if tile_record is None:
        raise RuntimeError(f"no local NAIP tile found for h3_cell_id={h3_cell_id}")

    source_path = Path(tile_record["source_path"])
    return fetch_local_raster_chip(source_path, target_transform), str(tile_record["source"])


def fetch_imagery_chip(
    row,
    bounds: tuple[float, float, float, float],
    target_transform,
    *,
    imagery_source: str,
    naip_tile_index: dict[str, dict[str, object]] | None,
) -> tuple[np.ndarray, str]:
    imagery_source = _normalize_choice(
        imagery_source,
        name="imagery_source",
        valid_values=VALID_IMAGERY_SOURCES,
    )
    if imagery_source == "esri":
        return fetch_contextily_chip(bounds, target_transform), "Esri.WorldImagery"
    if not naip_tile_index:
        raise RuntimeError("NAIP imagery requested, but no local NAIP STAC tiles were found.")
    return fetch_naip_chip(row, target_transform, naip_tile_index)


def summarize_dataset_split_diagnostics(cells: gpd.GeoDataFrame, records: pd.DataFrame) -> None:
    if cells.empty:
        print("split diagnostics skipped: no candidate cells available.")
        return

    eligible = cells.copy()
    total_eligible = int(len(eligible))
    holdout_eligible = int((eligible["priority_score"].fillna(0).astype(int) == HOLDOUT_PRIORITY_SCORE).sum())
    print(
        "eligible manifest cells: "
        f"{total_eligible:,} total | priority-{HOLDOUT_PRIORITY_SCORE} holdout candidates={holdout_eligible:,} "
        f"({(100.0 * holdout_eligible / max(total_eligible, 1)):.1f}%)"
    )

    if records.empty:
        print("split diagnostics skipped: no exported records.")
        return

    split_counts = records["dataset_split"].value_counts().to_dict()
    print(f"exported split counts: {split_counts}")

    if "split_bucket" in records.columns:
        print(f"split bucket counts: {records['split_bucket'].value_counts().to_dict()}")

    if "priority_score" in records.columns:
        priority_summary = (
            records.groupby(["dataset_split", "priority_score"]).size().rename("tiles").reset_index()
            .sort_values(["priority_score", "dataset_split"], ascending=[False, True])
        )
        print("split x priority summary:")
        print(priority_summary.to_string(index=False))

        if "split_bucket" in records.columns:
            mismatch_mask = (
                ((records["priority_score"].fillna(0).astype(int) == HOLDOUT_PRIORITY_SCORE) & (records["split_bucket"] != "seed_neighborhood_test"))
                | ((records["priority_score"].fillna(0).astype(int) != HOLDOUT_PRIORITY_SCORE) & (records["split_bucket"] == "seed_neighborhood_test"))
            )
            mismatch_count = int(mismatch_mask.sum())
            if mismatch_count > 0:
                print(
                    "priority/seed mismatch count: "
                    f"{mismatch_count:,} rows differ because manifest priority uses broader H3 overlap, "
                    "while the dataset split uses the square training chip footprint."
                )


def write_split_spatial_overview(
    records: pd.DataFrame | gpd.GeoDataFrame,
    *,
    layout: TrainingLayout,
    title: str = "Exported training tiles by split (centroid overlay)",
    file_name: str = "dataset_split_spatial_overview.png",
    boundaries: gpd.GeoDataFrame | None = None,
    highlight_boundaries: gpd.GeoDataFrame | None = None,
) -> Path | None:
    if records.empty:
        return None

    import matplotlib.pyplot as plt

    split_palette = {
        "train": "#2ca02c",
        "val": "#1f77b4",
        "test": "#d62728",
    }
    if isinstance(records, gpd.GeoDataFrame) and "geometry" in records.columns:
        points_gdf = records.to_crs(OUTPUT_CRS).copy()
        points_gdf = gpd.GeoDataFrame(
            points_gdf.drop(columns=["geometry"]),
            geometry=points_gdf.geometry.representative_point(),
            crs=OUTPUT_CRS,
        )
    else:
        points = records.copy()
        required_cols = {"west", "south", "east", "north", "dataset_split"}
        if not required_cols.issubset(points.columns):
            return None
        points["x"] = (points["west"].astype(float) + points["east"].astype(float)) / 2.0
        points["y"] = (points["south"].astype(float) + points["north"].astype(float)) / 2.0
        points_gdf = gpd.GeoDataFrame(
            points,
            geometry=gpd.points_from_xy(points["x"], points["y"], crs=MODEL_CRS),
            crs=MODEL_CRS,
        ).to_crs(OUTPUT_CRS)

    extent_geom = points_gdf.geometry.union_all().convex_hull
    fig, ax = plt.subplots(figsize=(12, 6))
    if boundaries is not None and not boundaries.empty:
        boundaries.to_crs(OUTPUT_CRS).boundary.plot(ax=ax, color="#9ca3af", linewidth=0.4, alpha=0.55)
    if highlight_boundaries is not None and not highlight_boundaries.empty:
        highlight_boundaries.to_crs(OUTPUT_CRS).boundary.plot(ax=ax, color="#111827", linewidth=1.0, alpha=0.85)
    gpd.GeoSeries([extent_geom], crs=OUTPUT_CRS).boundary.plot(ax=ax, color="#1f2937", linewidth=1.0)

    for split_name in ("train", "val", "test"):
        subset = points_gdf[points_gdf["dataset_split"] == split_name]
        if subset.empty:
            continue
        subset.plot(
            ax=ax,
            markersize=8,
            color=split_palette[split_name],
            alpha=0.5,
            label=f"{split_name} ({len(subset):,})",
        )

    ax.set_title(title)
    ax.set_axis_off()
    ax.legend(loc="best")
    figure_path = MAP_OUTPUT_DIR / file_name
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path = layout.review_dir / file_name
    if legacy_path.exists():
        legacy_path.unlink()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_path


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
        lambda values: union_all(
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


def link_stac_tile_to_layout(stac_tile_path: Path, layout_path: Path) -> bool:
    """Point layout_path at the existing STAC tile using the cheapest mechanism.

    Preference order:
    1. Hard link (os.link) – same inode, zero storage cost, completely
       transparent to rasterio / PyTorch DataLoader.  Fails across devices.
    2. Absolute symlink – works across devices; follows correctly as long as
       the stac_tiles directory is not moved.
    Returns True if a link was created, False if the caller should fall back
    to a full write (e.g. both strategies failed).
    """
    layout_path.parent.mkdir(parents=True, exist_ok=True)

    if layout_path.exists() or layout_path.is_symlink():
        return True  # already present from a prior run

    try:
        os.link(stac_tile_path, layout_path)
        return True
    except OSError:
        pass  # cross-device or unsupported filesystem

    try:
        os.symlink(stac_tile_path.resolve(), layout_path)
        return True
    except OSError:
        return False


def resolve_dataset_split(
    key: object,
    *,
    priority_score: object,
    dataset_cohort: str,
    is_seed_neighborhood: object = False,
    is_case_study_municipality: object = False,
) -> str:
    if bool(is_seed_neighborhood):
        return "test"
    if bool(is_case_study_municipality):
        return assign_case_study_split(key)
    if dataset_cohort == "holdout_priority3":
        return "test"
    if SPLIT_POLICY == "priority3_holdout" and pd.notna(priority_score) and int(priority_score) == HOLDOUT_PRIORITY_SCORE:
        return "test"
    return assign_dataset_split(key)


def _manifest_file_list_hash(records: pd.DataFrame) -> str | None:
    if records.empty:
        return None
    columns = [
        column_name
        for column_name in ("tile_id", "h3_cell_id", "imagery_source", "dataset_cohort", "dataset_split", "split_key")
        if column_name in records.columns
    ]
    if not columns:
        return None
    ordered = (
        records[columns]
        .fillna("")
        .astype(str)
        .sort_values(columns, kind="stable")
        .to_dict(orient="records")
    )
    return hashlib.sha1(json.dumps(ordered, sort_keys=True).encode("utf-8")).hexdigest()


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
    imagery_source: str,
    provider_name: str,
    dataset_cohort: str,
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
        "meters_per_pixel": float(chip_span_m / CHIP_PIXELS),
        "imagery_source": imagery_source,
        "contextily_zoom": CONTEXTILY_ZOOM_RAW,
        "provider": provider_name,
        "crs": MODEL_CRS,
        "dataset_cohort": dataset_cohort,
        "holdout_group": HOLDOUT_GROUP_LABEL if split_name == "test" else None,
        "dataset_split": split_name,
        "split_policy": SPLIT_POLICY,
        "split_seed": "sha1(split_key)",
        "split_key": split_key,
        "split_bucket": getattr(row, "split_bucket", None),
        "seed_name": getattr(row, "seed_name", None),
        "is_seed_neighborhood": bool(getattr(row, "is_seed_neighborhood", False)),
        "is_case_study_municipality": bool(getattr(row, "is_case_study_municipality", False)),
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
            dedupe_columns = [
                column_name
                for column_name in ("tile_id", "imagery_source", "dataset_cohort")
                if column_name in merged.columns
            ]
            merged = merged.drop_duplicates(subset=dedupe_columns or ["tile_id"], keep="last").reset_index(drop=True)
    return merged


def identify_priority_seed_mismatch_cells(cells: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if cells.empty or "priority_score" not in cells.columns or "split_bucket" not in cells.columns:
        return cells.iloc[0:0].copy()

    priority_holdout = cells["priority_score"].fillna(0).astype(int) == HOLDOUT_PRIORITY_SCORE
    seed_test = cells["split_bucket"] == "seed_neighborhood_test"
    return cells[priority_holdout.ne(seed_test)].copy()


def remove_manifest_rows_and_artifacts(
    layout: TrainingLayout,
    *,
    tile_ids: set[str],
    imagery_source: str,
    dataset_cohort: str,
) -> pd.DataFrame:
    if not layout.manifest_path.exists() or not tile_ids:
        return pd.DataFrame()

    manifest = pd.read_csv(layout.manifest_path)
    if manifest.empty or "tile_id" not in manifest.columns:
        return manifest

    row_mask = manifest["tile_id"].astype(str).isin(tile_ids)
    if "imagery_source" in manifest.columns:
        row_mask &= manifest["imagery_source"].fillna(imagery_source).astype(str).eq(imagery_source)
    if "dataset_cohort" in manifest.columns:
        row_mask &= manifest["dataset_cohort"].fillna(dataset_cohort).astype(str).eq(dataset_cohort)

    rows_to_remove = manifest[row_mask].copy()
    artifact_columns = [
        "image_path",
        "mask_path",
        "raw_mask_path",
        "grounded_mask_path",
        "prompt_artifact_path",
        "review_artifact_path",
    ]
    protected_root = layout.train_root.resolve()
    removed_paths: set[Path] = set()
    for column_name in artifact_columns:
        if column_name not in rows_to_remove.columns:
            continue
        for raw_path in rows_to_remove[column_name].dropna().astype(str):
            artifact_path = (PROJECT_ROOT / raw_path).resolve()
            try:
                artifact_path.relative_to(protected_root)
            except ValueError:
                continue
            if artifact_path in removed_paths or not artifact_path.exists():
                continue
            artifact_path.unlink()
            removed_paths.add(artifact_path)

    remaining = manifest[~row_mask].reset_index(drop=True)
    remaining.to_csv(layout.manifest_path, index=False)
    print(
        f"removed {len(rows_to_remove):,} priority/seed mismatch rows and {len(removed_paths):,} local artifacts "
        f"from {layout.manifest_path}"
    )
    return rows_to_remove.reset_index(drop=True)


def export_training_dataset(
    cells: gpd.GeoDataFrame,
    osm_pv: gpd.GeoDataFrame,
    overture_buildings: gpd.GeoDataFrame,
    *,
    layout: TrainingLayout,
    overwrite_existing: bool = OVERWRITE_EXISTING_CHIPS,
    imagery_source: str = IMAGERY_SOURCE,
    dataset_cohort: str = DATASET_COHORT,
) -> pd.DataFrame:
    if cells.empty:
        return pd.DataFrame()

    cells_3857 = cells.to_crs(MODEL_CRS)
    osm_pv_3857 = osm_pv.to_crs(MODEL_CRS)
    overture_buildings_3857 = overture_buildings.to_crs(MODEL_CRS)
    chip_span_m = infer_chip_span_m(cells_3857)
    naip_tile_index = load_naip_tile_index(target_municipalities=TARGET_MUNICIPALITIES) if imagery_source == "naip" else None
    records: list[dict[str, object]] = []

    for row in cells_3857.itertuples(index=False):
        # ── Resolve chip bounds ───────────────────────────────────────────────
        # For NAIP: use the per-cell bounds already written by notebook 08 into
        # the STAC tile manifest (west_3857 … north_3857).  This lets the image
        # chip be a direct hard link / symlink to the existing stac_tiles
        # GeoTIFF — no re-projection needed, no new derived copy on disk.
        # For ESRI: keep the batch-uniform span so all contextily chips are
        # square and identically sized.
        h3_key = str(row.h3_cell_id) if pd.notna(row.h3_cell_id) else str(row.tile_id)
        naip_tile_record: dict[str, object] | None = (
            naip_tile_index.get(h3_key) if naip_tile_index is not None else None
        )

        if naip_tile_record is not None:
            w = naip_tile_record.get("west_3857")
            s = naip_tile_record.get("south_3857")
            e = naip_tile_record.get("east_3857")
            n = naip_tile_record.get("north_3857")
            if None in (w, s, e, n):
                # Bounds not in manifest – read from the GeoTIFF header.
                with rasterio.open(naip_tile_record["tile_path"]) as _src:
                    _b = _src.bounds
                    w, s, e, n = _b.left, _b.bottom, _b.right, _b.top
            bounds = (float(w), float(s), float(e), float(n))
        else:
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
        provider_name = "Esri.WorldImagery" if imagery_source == "esri" else "naip"
        fetched_image = False
        updated_assets = False
        if overwrite_existing or not image_path.exists():
            if naip_tile_record is not None:
                # ── NAIP: link directly to the existing stac_tiles GeoTIFF ───
                # outputs/stac_tiles is the canonical source; image_path in the
                # training layout is a hard link (same inode, zero extra storage)
                # or an absolute symlink if the layout is on a different device.
                stac_tile_path = Path(naip_tile_record["tile_path"])
                if not link_stac_tile_to_layout(stac_tile_path, image_path):
                    # Both link strategies failed; fall back to a file copy.
                    try:
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(stac_tile_path, image_path)
                    except Exception as exc:
                        print(f"[warn] failed to link/copy stac tile for {row.tile_id}: {exc}")
                        continue
                provider_name = str(naip_tile_record.get("source", "naip"))
                fetched_image = True
            else:
                # ── ESRI (or NAIP tile missing from index) ────────────────────
                try:
                    image, provider_name = fetch_imagery_chip(
                        row,
                        bounds,
                        transform,
                        imagery_source=imagery_source,
                        naip_tile_index=naip_tile_index,
                    )
                except Exception as exc:
                    print(f"[warn] failed to fetch {imagery_source} imagery for {row.tile_id}: {exc}")
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
        if hasattr(row, "split_key") and pd.notna(row.split_key):
            split_key = str(row.split_key)
        if hasattr(row, "dataset_split") and pd.notna(row.dataset_split):
            split_name = str(row.dataset_split)
        else:
            split_name = resolve_dataset_split(
                split_key,
                priority_score=row.priority_score,
                dataset_cohort=dataset_cohort,
                is_seed_neighborhood=getattr(row, "is_seed_neighborhood", False),
                is_case_study_municipality=getattr(row, "is_case_study_municipality", False),
            )
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
        if not fetched_image and naip_tile_record is not None and (image_path.is_symlink() or (image_path.exists() and image_path.stat().st_nlink > 1)):
            status = "linked"

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
                imagery_source=imagery_source,
                provider_name=provider_name,
                dataset_cohort=dataset_cohort,
            )
        )

    return pd.DataFrame.from_records(records)


def write_training_summary(
    records: pd.DataFrame,
    *,
    layout: TrainingLayout,
    min_priority_score: int,
    max_priority_score: int | None,
    imagery_source: str = IMAGERY_SOURCE,
    dataset_cohort: str = DATASET_COHORT,
) -> pd.DataFrame:
    layout.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    merged_records = merge_training_manifest(layout.manifest_path, records)
    merged_records.to_csv(layout.manifest_path, index=False)
    status_counts = merged_records["status"].fillna("unknown").value_counts().to_dict() if not merged_records.empty else {}
    providers = sorted(merged_records["provider"].dropna().astype(str).unique().tolist()) if not merged_records.empty else []
    summary = {
        "train_root": str(layout.train_root.relative_to(PROJECT_ROOT)),
        "image_count": int(len(merged_records)),
        "mask_count": int(len(merged_records)),
        "grounded_mask_count": int(len(merged_records)),
        "prompt_artifact_count": int(len(merged_records)),
        "review_artifact_count": int(merged_records["review_artifact_path"].notna().sum()) if not merged_records.empty and "review_artifact_path" in merged_records.columns else 0,
        "chip_pixels": CHIP_PIXELS,
        "chip_span_m": int(merged_records["chip_span_m"].iloc[0]) if not merged_records.empty else None,
        "meters_per_pixel": float(merged_records["meters_per_pixel"].iloc[0]) if not merged_records.empty else None,
        "imagery_source": imagery_source,
        "dataset_cohort": dataset_cohort,
        "contextily_zoom": CONTEXTILY_ZOOM_RAW,
        "providers": providers,
        "dataset_splits": merged_records["dataset_split"].value_counts().to_dict() if not merged_records.empty else {},
        "split_policy": SPLIT_POLICY,
        "holdout_priority_score": HOLDOUT_PRIORITY_SCORE,
        "holdout_group": HOLDOUT_GROUP_LABEL,
        "split_weights": {
            "train": TRAIN_SPLIT_WEIGHTS[0],
            "val": TRAIN_SPLIT_WEIGHTS[1],
            "test": TRAIN_SPLIT_WEIGHTS[2],
        },
        "file_list_hash": _manifest_file_list_hash(merged_records),
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
    imagery_source: str = IMAGERY_SOURCE,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame] | None:
    layout = resolve_training_layout(train_root)
    con = connect(resolve_db_path())
    cells = load_training_cells(
        con,
        min_priority_score=min_priority_score,
        max_priority_score=max_priority_score,
        dataset_cohort="priority_band",
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
    manifest = export_training_dataset(
        cells,
        osm_pv,
        overture_buildings,
        layout=layout,
        imagery_source=imagery_source,
        dataset_cohort="priority_band",
    )
    if manifest.empty:
        print(f"no training chips were written for priority band {min_priority_score}..{max_priority_score}.")
        return None

    merged_manifest = write_training_summary(
        manifest,
        layout=layout,
        min_priority_score=min_priority_score,
        max_priority_score=max_priority_score,
        imagery_source=imagery_source,
        dataset_cohort="priority_band",
    )
    print(
        f"priority {min_priority_score}..{max_priority_score} export wrote {len(merged_manifest):,} chips to {layout.image_dir}"
    )
    print(f"manifest: {layout.manifest_path}")
    return merged_manifest, osm_pv


# %% [markdown]
# ## Optional export controls
#
# In notebook mode, use the dropdowns below to switch between ESRI and NAIP
# imagery and between the train pool and the dedicated priority-3 holdout.

# %%
if _widgets_enabled():
    _maybe_initialize_training_widgets()

# %%
if __name__ == "__main__":
    apply_training_widget_overrides()
    db_path = resolve_db_path()
    print(f"DuckDB: {db_path}")
    print(f"Training root: {TRAIN_ROOT}")
    print(f"Imagery source: {IMAGERY_SOURCE_LABELS[IMAGERY_SOURCE]}")
    print(f"Dataset cohort: {DATASET_COHORT_LABELS[DATASET_COHORT]}")
    print(f"Split policy: {SPLIT_POLICY}")
    print(f"Contextily HTTP cache enabled: {CONTEXTILY_USE_CACHE}")
    print(f"Reuse existing chips: {not OVERWRITE_EXISTING_CHIPS}")
    if not RESET_TRAIN_ROOT:
        print("training root reset disabled: existing image/mask chips will be reused when possible.")
    manifest, osm_pv = run_training_export_workflow()
    if manifest is not None and not manifest.empty:
        train_layout = resolve_training_layout(TRAIN_ROOT)
        print(f"wrote {len(manifest):,} prompt artifacts to {train_layout.prompt_dir}")
        if WRITE_REVIEW_ARTIFACTS:
            print(f"wrote {len(manifest):,} review PNG artifacts to {train_layout.review_dir}")
        print(f"training manifest: {train_layout.manifest_path}")
        print(
            "To rebuild from scratch, set GEOAI_RESET_TRAIN_ROOT=1. "
            "Use GEOAI_IMAGERY_SOURCE=naip for NAIP exports and GEOAI_DATASET_COHORT=holdout_priority3 to export the dedicated seed-neighborhood holdout."
        )

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
        imagery_source=IMAGERY_SOURCE,
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
        imagery_source=IMAGERY_SOURCE,
    )
    if SHOW_PREVIEW and priority1_result is not None:
        priority1_manifest, priority1_osm_pv = priority1_result
        preview_training_samples(priority1_manifest, priority1_osm_pv)


