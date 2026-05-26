# %% [markdown]
# # GeoAI Solar Segmentation Inference on NAIP STAC Tiles
# 
# Runs either the fine-tuned DINOv3 or SMP grounded-segmentation model over the
# fetched NAIP STAC tiles for the case-study municipalities, vectorizes the
# predicted masks, writes `pr_solar_pv_detections`, and materializes tile- and
# building-level summary tables for downstream tabular notebooks.

# %%
"""13_geoai_solar_inference.py"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from dotenv import load_dotenv
from rasterio.features import geometry_mask
from shapely import from_wkb
from shapely.geometry import mapping
from tqdm.auto import tqdm


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
from utils.geoai_review import render_prediction_review_bundle
from utils.geoai_segmentation import mask_to_geodataframe


def _resolve_optional_path(raw_value: str | None) -> Path | None:
    if not raw_value:
        return None
    path = Path(raw_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


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
    return tuple(parts) or default


def _slugify(value: object) -> str:
    text = str(value).strip().lower()
    slug_chars = [char if char.isalnum() else "-" for char in text]
    slug = "".join(slug_chars)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "default"


NAIP_SOURCE_NAMES = ("pr_naip", "naip_2021_pr")
TARGET_MUNICIPALITIES = _parse_csv_env("GEOAI_TARGET_MUNICIPALITIES", ("San Juan", "Isabela"))
INFERENCE_MODEL_FAMILY = (os.getenv("GEOAI_INFERENCE_MODEL_FAMILY", "dinov3") or "dinov3").strip().lower()
VALID_MODEL_FAMILIES = {"dinov3", "smp"}
EXPLICIT_MODEL_DIR = _resolve_optional_path(os.getenv("GEOAI_INFERENCE_MODEL_DIR"))
EXPLICIT_MODEL_PATH = _resolve_optional_path(os.getenv("GEOAI_INFERENCE_MODEL_PATH"))
EXPLICIT_INFERENCE_ROOT = _resolve_optional_path(os.getenv("GEOAI_INFERENCE_ROOT"))

STAC_TILE_MANIFEST = PROJECT_ROOT / "outputs" / "stac_tiles" / "pr_stac_tile_manifest.parquet"
INFERENCE_MODE = (os.getenv("GEOAI_INFERENCE_MODE", "sample") or "sample").strip().lower()
INFERENCE_PHASE = (os.getenv("GEOAI_INFERENCE_PHASE", "both") or "both").strip().lower()
VALID_INFERENCE_PHASES = {"both", "inference", "reporting"}
if INFERENCE_PHASE not in VALID_INFERENCE_PHASES:
    raise RuntimeError(
        f"unsupported GEOAI_INFERENCE_PHASE={INFERENCE_PHASE!r}; expected one of {sorted(VALID_INFERENCE_PHASES)}"
    )
SAMPLE_COUNT = max(1, int(os.getenv("GEOAI_SAMPLE_COUNT", "1") or "1"))
SAMPLE_SEED = int(os.getenv("GEOAI_SAMPLE_SEED", "323") or "323")
MIN_TILE_BUILDING_COUNT = int(os.getenv("GEOAI_MIN_TILE_BUILDING_COUNT", "1") or "1")
INCLUDE_CROSS_BOUNDARY_TILES = os.getenv("GEOAI_INCLUDE_CROSS_BOUNDARY_TILES", "1") == "1"
REVIEW_COUNT = int(os.getenv("GEOAI_REVIEW_COUNT", "12") or "12")
OVERWRITE_INFERENCE_ARTIFACTS = os.getenv("GEOAI_OVERWRITE_INFERENCE_ARTIFACTS", "0") == "1"
WRITE_STATIC_REVIEW_ARTIFACTS = os.getenv("GEOAI_WRITE_STATIC_REVIEW_ARTIFACTS", "1") == "1"
WRITE_INTERACTIVE_REVIEW_ARTIFACTS = os.getenv("GEOAI_WRITE_INTERACTIVE_REVIEW_ARTIFACTS", "0") == "1"
SHOW_WINDOW_PROGRESS = os.getenv("GEOAI_SHOW_WINDOW_PROGRESS", "0") == "1"
DB_COMMIT = os.getenv("GEOAI_DB_COMMIT", "0") == "1"
INFERENCE_ARTIFACT_MODE = (os.getenv("GEOAI_INFERENCE_ARTIFACT_MODE", "compact") or "compact").strip().lower()
VALID_INFERENCE_ARTIFACT_MODES = {"compact", "debug"}
if INFERENCE_ARTIFACT_MODE not in VALID_INFERENCE_ARTIFACT_MODES:
    raise RuntimeError(
        "unsupported GEOAI_INFERENCE_ARTIFACT_MODE="
        f"{INFERENCE_ARTIFACT_MODE!r}; expected one of {sorted(VALID_INFERENCE_ARTIFACT_MODES)}"
    )
CLIP_INPUT_RASTERS_TO_BUILDINGS = os.getenv("GEOAI_CLIP_INPUT_RASTERS_TO_BUILDINGS", "1") == "1"
CLIP_PREDICTION_MASK_TO_BUILDINGS = os.getenv("GEOAI_CLIP_PREDICTION_MASK_TO_BUILDINGS", "0") == "1"
WRITE_INPUT_CLIP_PREVIEW_ARTIFACTS = os.getenv("GEOAI_WRITE_INPUT_CLIP_PREVIEW_ARTIFACTS", "1") == "1"
WRITE_PROBABILITY_RASTERS = os.getenv("GEOAI_WRITE_PROBABILITY_RASTERS", "0") == "1"
PRESERVE_TILE_MASK_ARTIFACTS = os.getenv(
    "GEOAI_PRESERVE_TILE_MASK_ARTIFACTS",
    "1" if INFERENCE_ARTIFACT_MODE == "debug" else "0",
) == "1"
PRESERVE_CLIPPED_INPUT_ARTIFACTS = os.getenv(
    "GEOAI_PRESERVE_CLIPPED_INPUT_ARTIFACTS",
    "1" if INFERENCE_ARTIFACT_MODE == "debug" else "0",
) == "1"
INFERENCE_MIXED_PRECISION = (os.getenv("GEOAI_INFERENCE_MIXED_PRECISION", "bf16") or "bf16").strip().lower()
INFERENCE_FLOAT32_MATMUL_PRECISION = (
    os.getenv("GEOAI_INFERENCE_FLOAT32_MATMUL_PRECISION", "high") or "high"
).strip().lower()
ENABLE_INFERENCE_TF32 = os.getenv("GEOAI_INFERENCE_ENABLE_TF32", "1") == "1"
ENABLE_INFERENCE_CUDNN_BENCHMARK = os.getenv("GEOAI_INFERENCE_ENABLE_CUDNN_BENCHMARK", "1") == "1"
CUDA_EMPTY_CACHE_BEFORE_INFERENCE = os.getenv("GEOAI_INFERENCE_EMPTY_CACHE_BEFORE_RUN", "0") == "1"

WINDOW_SIZE = int(os.getenv("GEOAI_INFERENCE_WINDOW_SIZE", "512") or "512")
OVERLAP = int(os.getenv("GEOAI_INFERENCE_OVERLAP", "256") or "256")
INFERENCE_BATCH_SIZE = int(os.getenv("GEOAI_INFERENCE_BATCH_SIZE", "16") or "16")
INFERENCE_TILE_BATCH_SIZE = max(
    1,
    int(os.getenv("GEOAI_INFERENCE_TILE_BATCH_SIZE", str(INFERENCE_BATCH_SIZE)) or str(INFERENCE_BATCH_SIZE)),
)
ENABLE_SOURCE_GROUP_TILE_BATCHING = os.getenv("GEOAI_ENABLE_SOURCE_GROUP_TILE_BATCHING", "1") == "1"
PROBABILITY_THRESHOLD = float(os.getenv("GEOAI_INFERENCE_PROBABILITY_THRESHOLD", "0.5") or "0.5")
MIN_AREA_M2 = float(os.getenv("GEOAI_INFERENCE_MIN_AREA_M2", "2.0") or "2.0")
MAX_AREA_M2 = float(os.getenv("GEOAI_INFERENCE_MAX_AREA_M2", "1800.0") or "1800.0")
MAX_ELONGATION = float(os.getenv("GEOAI_INFERENCE_MAX_ELONGATION", "30.0") or "30.0")
CLIP_TO_BUILDINGS = os.getenv("GEOAI_CLIP_TO_BUILDINGS", "0") == "1"
BUILDING_CLIP_BUFFER_M = float(os.getenv("GEOAI_BUILDING_CLIP_BUFFER_M", "2.0") or "2.0")
WRITE_PRESENTATION_OUTPUTS = os.getenv("GEOAI_WRITE_PRESENTATION_OUTPUTS", "1") == "1"
PRESENTATION_AGGREGATIONS = _parse_csv_env("GEOAI_PRESENTATION_AGGREGATIONS", ("h3", "tract", "block_group"))
MAPS_ROOT = PROJECT_ROOT / "outputs" / "maps" / "geoai_inference"
REPORTS_ROOT = PROJECT_ROOT / "outputs" / "reports"

DETECTION_TABLE = "pr_solar_pv_detections"
ALL_MODEL_DETECTION_TABLE = "pr_solar_pv_detections_all_models"
TILE_SUMMARY_TABLE = "pr_solar_pv_detection_tiles"
BUILDING_HITS_TABLE = "pr_solar_pv_building_hits"
ALL_MODEL_BUILDING_TABLE = "pr_solar_pv_building_detections_all_models"
MUNICIPIO_SUMMARY_TABLE = "pr_solar_pv_detection_municipio_summary"
MULTIMODEL_BUILDING_SUMMARY_OUTPUT_PATH = REPORTS_ROOT / "pv_building_detections_by_model_arch_summary.csv"
MULTIMODEL_BUILDING_MAP_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "maps" / "pv_building_detections_by_model_arch.png"
LEGACY_MODEL_FAMILY_BACKFILL = "dinov3"
LEGACY_MODEL_RUN_NAME_BACKFILL = "vitl16-sat493m__hub-dinov3-vitl16__df512__ps16__e25__b16__lr5em04__lora-r8__data-esri-train-pool-n5409-e0263da2"
MODEL_ARCH_ORDER = ("DINOv3", "PAN", "FPN", "Segformer", "DeepLabV3+", "UPerNet", "Shared")
MODEL_ARCH_COLORS = {
    "DINOv3": "#1b9e77",
    "PAN": "#d95f02",
    "FPN": "#7570b3",
    "Segformer": "#e7298a",
    "DeepLabV3+": "#66a61e",
    "UPerNet": "#e6ab02",
    "Shared": "#4d4d4d",
}
INFERENCE_NOTEBOOK_WIDGETS: dict[str, object] | None = None


def resolve_db_path() -> Path:
    value = os.getenv("VECTOR_DB")
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path if len(path.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / path
        return path
    return PROJECT_ROOT / "data" / "PR_PV_plan_data.duckdb"


def resolve_model_arch_info(
    model_family: str | None,
    model_run_name: str | None,
    metadata: dict[str, object] | None = None,
) -> tuple[str, str]:
    family = str(model_family or "").strip().lower()
    run_name = str(model_run_name or "").strip()
    raw_arch = ""
    if metadata is not None:
        raw_arch = str(metadata.get("architecture") or metadata.get("model_name") or "").strip()
    if not raw_arch and family == "dinov3":
        raw_arch = "dinov3"
    if not raw_arch and run_name:
        raw_arch = run_name.split("__", 1)[0]

    normalized = raw_arch.lower().replace("-", "").replace("_", "")
    if family == "dinov3" or "dinov3" in normalized:
        return "dinov3", "DINOv3"

    key_map = {
        "pan": "pan",
        "fpn": "fpn",
        "segformer": "segformer",
        "deeplabv3plus": "deeplabv3plus",
        "upernet": "upernet",
    }
    label_map = {
        "pan": "PAN",
        "fpn": "FPN",
        "segformer": "Segformer",
        "deeplabv3plus": "DeepLabV3+",
        "upernet": "UPerNet",
    }
    key = key_map.get(normalized, normalized or family or "unknown")
    label = label_map.get(key, (raw_arch or family or "unknown").replace("_", " ").replace("-", " ").title())
    return key, label


def running_in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def discovered_model_options(model_family: str) -> list[tuple[str, str]]:
    metadata_name = "smp_metadata.json" if model_family == "smp" else "dinov3_metadata.json"
    base_root = PROJECT_ROOT / "outputs" / "models" / ("smp_grounded" if model_family == "smp" else "dinov3_grounded")
    options = [(path.name, str(path)) for path in list_model_run_dirs(base_root, metadata_name)]
    return options or [(f"No {model_family} runs found", "")]


def _maybe_initialize_inference_widgets() -> dict[str, object] | None:
    global INFERENCE_NOTEBOOK_WIDGETS
    if INFERENCE_NOTEBOOK_WIDGETS is not None:
        return INFERENCE_NOTEBOOK_WIDGETS
    if not running_in_notebook():
        return None

    try:
        import ipywidgets as widgets
        from IPython.display import Markdown, display
    except ImportError:
        return None

    family_widget = widgets.Dropdown(
        options=[("DINOv3", "dinov3"), ("SMP", "smp")],
        value=INFERENCE_MODEL_FAMILY,
        description="Family",
        layout=widgets.Layout(width="260px"),
    )
    _initial_run_options = discovered_model_options(INFERENCE_MODEL_FAMILY)
    _initial_run_values = {opt_value for _, opt_value in _initial_run_options}
    _preferred_run_value = str(EXPLICIT_MODEL_DIR) if EXPLICIT_MODEL_DIR is not None else ""
    _initial_run_value = _preferred_run_value if _preferred_run_value in _initial_run_values else (
        _initial_run_options[0][1] if _initial_run_options else ""
    )
    run_widget = widgets.Dropdown(
        options=_initial_run_options,
        value=_initial_run_value or None,
        description="Run",
        layout=widgets.Layout(width="720px"),
    )
    mode_widget = widgets.Dropdown(
        options=[("Sample", "sample"), ("All tiles", "all")],
        value=INFERENCE_MODE,
        description="Mode",
        layout=widgets.Layout(width="220px"),
    )
    phase_widget = widgets.Dropdown(
        options=[("Inference + reporting", "both"), ("Inference only", "inference"), ("Reporting only", "reporting")],
        value=INFERENCE_PHASE,
        description="Phase",
        layout=widgets.Layout(width="260px"),
    )
    artifact_mode_widget = widgets.Dropdown(
        options=[("Compact", "compact"), ("Debug", "debug")],
        value=INFERENCE_ARTIFACT_MODE,
        description="Artifacts",
        layout=widgets.Layout(width="220px"),
    )
    sample_count_widget = widgets.IntSlider(
        value=SAMPLE_COUNT,
        min=1,
        max=5000,
        step=1,
        description="Samples",
        layout=widgets.Layout(width="220px"),
    )
    batch_range_widget = widgets.IntRangeSlider(
        value=(1, max(1, INFERENCE_BATCH_SIZE)),
        min=1,
        max=64,
        step=1,
        description="Batch range",
        continuous_update=False,
        layout=widgets.Layout(width="360px"),
    )
    min_building_widget = widgets.BoundedIntText(
        value=MIN_TILE_BUILDING_COUNT,
        min=0,
        max=10000,
        description="Min blds",
        layout=widgets.Layout(width="220px"),
    )
    review_count_widget = widgets.BoundedIntText(
        value=REVIEW_COUNT,
        min=0,
        max=500,
        description="Reviews",
        layout=widgets.Layout(width="220px"),
    )
    cross_boundary_widget = widgets.Checkbox(
        value=INCLUDE_CROSS_BOUNDARY_TILES,
        description="Include boundary-crossing tiles",
        indent=False,
    )
    clip_widget = widgets.Checkbox(
        value=CLIP_TO_BUILDINGS,
        description="Enable building clipping",
        indent=False,
    )
    aggregation_widget = widgets.SelectMultiple(
        options=[("H3 cells", "h3"), ("Census tracts", "tract"), ("Census block groups", "block_group")],
        value=tuple(value for value in PRESENTATION_AGGREGATIONS if value in {"h3", "tract", "block_group"}) or ("h3",),
        description="Aggregate",
        layout=widgets.Layout(width="360px", height="90px"),
    )
    commit_widget = widgets.Checkbox(
        value=DB_COMMIT,
        description="Commit DuckDB writes",
        indent=False,
    )
    window_progress_widget = widgets.Checkbox(
        value=SHOW_WINDOW_PROGRESS,
        description="Show per-window progress",
        indent=False,
    )
    input_preview_widget = widgets.Checkbox(
        value=WRITE_INPUT_CLIP_PREVIEW_ARTIFACTS,
        description="Preview clipped inputs",
        indent=False,
    )

    def _refresh_run_options(_change=None) -> None:
        del _change
        options = discovered_model_options(str(family_widget.value))
        allowed_values = {value for _, value in options}
        current_value = run_widget.value
        # Reset value to a safe sentinel before swapping options to avoid TraitError.
        first_value = options[0][1] if options else None
        run_widget.value = None
        run_widget.options = options
        run_widget.value = current_value if current_value in allowed_values else first_value

    family_widget.observe(_refresh_run_options, names="value")

    display(
        Markdown(
            "**Inference controls**: sample mode always stays local-only; DuckDB writes are only allowed for full runs when the commit box is checked."
        )
    )
    display(
        Markdown(
            "Sample count controls how many candidate tiles run in sample mode; full inference ignores it and processes all selected case-study tiles. The phase selector can skip reporting or rerun reporting from existing local outputs. Building clipping is off by default so inference keeps full tile context unless you explicitly enable footprint masking/clipping. Compact artifact mode keeps the run-level outputs and sample reviews while dropping the per-tile probability and vector files. The batch range upper value is used as the inference batch size."
        )
    )
    display(
        widgets.VBox(
            [
                widgets.HBox([family_widget, run_widget]),
                widgets.HBox([mode_widget, phase_widget, artifact_mode_widget, sample_count_widget, min_building_widget, review_count_widget]),
                widgets.HBox([batch_range_widget, aggregation_widget]),
                widgets.HBox([cross_boundary_widget, clip_widget, commit_widget, window_progress_widget, input_preview_widget]),
            ]
        )
    )

    INFERENCE_NOTEBOOK_WIDGETS = {
        "family": family_widget,
        "run_dir": run_widget,
        "mode": mode_widget,
        "phase": phase_widget,
        "artifact_mode": artifact_mode_widget,
        "sample_count": sample_count_widget,
        "batch_range": batch_range_widget,
        "min_building_count": min_building_widget,
        "review_count": review_count_widget,
        "include_cross_boundary": cross_boundary_widget,
        "clip_to_buildings": clip_widget,
        "aggregations": aggregation_widget,
        "db_commit": commit_widget,
        "show_window_progress": window_progress_widget,
        "write_input_clip_preview_artifacts": input_preview_widget,
    }
    return INFERENCE_NOTEBOOK_WIDGETS


def apply_inference_widget_overrides() -> None:
    global INFERENCE_MODEL_FAMILY, EXPLICIT_MODEL_DIR, INFERENCE_MODE, INFERENCE_PHASE, INFERENCE_ARTIFACT_MODE, SAMPLE_COUNT
    global MIN_TILE_BUILDING_COUNT, REVIEW_COUNT, INCLUDE_CROSS_BOUNDARY_TILES, DB_COMMIT
    global INFERENCE_BATCH_SIZE, CLIP_TO_BUILDINGS, PRESENTATION_AGGREGATIONS, SHOW_WINDOW_PROGRESS
    global WRITE_INPUT_CLIP_PREVIEW_ARTIFACTS
    widgets = _maybe_initialize_inference_widgets()
    if widgets is None:
        return

    INFERENCE_MODEL_FAMILY = str(widgets["family"].value)
    EXPLICIT_MODEL_DIR = _resolve_optional_path(str(widgets["run_dir"].value) or None)
    INFERENCE_MODE = str(widgets["mode"].value)
    INFERENCE_PHASE = str(widgets["phase"].value)
    INFERENCE_ARTIFACT_MODE = str(widgets["artifact_mode"].value)
    SAMPLE_COUNT = max(1, int(widgets["sample_count"].value))
    INFERENCE_BATCH_SIZE = max(1, int(widgets["batch_range"].value[1]))
    MIN_TILE_BUILDING_COUNT = int(widgets["min_building_count"].value)
    REVIEW_COUNT = int(widgets["review_count"].value)
    INCLUDE_CROSS_BOUNDARY_TILES = bool(widgets["include_cross_boundary"].value)
    CLIP_TO_BUILDINGS = bool(widgets["clip_to_buildings"].value)
    PRESENTATION_AGGREGATIONS = tuple(str(value) for value in widgets["aggregations"].value)
    DB_COMMIT = bool(widgets["db_commit"].value)
    SHOW_WINDOW_PROGRESS = bool(widgets["show_window_progress"].value)
    WRITE_INPUT_CLIP_PREVIEW_ARTIFACTS = bool(widgets["write_input_clip_preview_artifacts"].value)


def resolve_runtime_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_runtime_device(device: torch.device) -> str:
    if device.index is None:
        return device.type
    return f"{device.type}:{device.index}"


def resolve_inference_amp_dtype(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or INFERENCE_MIXED_PRECISION in {"", "0", "off", "false", "none", "no"}:
        return None
    if INFERENCE_MIXED_PRECISION in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if INFERENCE_MIXED_PRECISION in {"fp16", "float16", "16"}:
        return torch.float16
    raise RuntimeError("GEOAI_INFERENCE_MIXED_PRECISION must be one of: off, bf16, fp16.")


def format_amp_dtype(amp_dtype: torch.dtype | None) -> str:
    return str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "off"


def configure_torch_inference_performance(device: torch.device) -> torch.dtype | None:
    valid_matmul_precision = {"highest", "high", "medium"}
    if INFERENCE_FLOAT32_MATMUL_PRECISION not in valid_matmul_precision:
        raise RuntimeError(
            "GEOAI_INFERENCE_FLOAT32_MATMUL_PRECISION must be one of "
            f"{sorted(valid_matmul_precision)}; got {INFERENCE_FLOAT32_MATMUL_PRECISION!r}."
        )
    torch.set_float32_matmul_precision(INFERENCE_FLOAT32_MATMUL_PRECISION)

    amp_dtype = resolve_inference_amp_dtype(device)
    if device.type != "cuda":
        print(
            "Inference GPU tuning: CUDA unavailable; using CPU-safe settings "
            f"mixed_precision={format_amp_dtype(amp_dtype)} | "
            f"matmul_precision={INFERENCE_FLOAT32_MATMUL_PRECISION}"
        )
        return amp_dtype

    torch.backends.cuda.matmul.allow_tf32 = ENABLE_INFERENCE_TF32
    torch.backends.cudnn.allow_tf32 = ENABLE_INFERENCE_TF32
    torch.backends.cudnn.benchmark = ENABLE_INFERENCE_CUDNN_BENCHMARK
    if CUDA_EMPTY_CACHE_BEFORE_INFERENCE:
        torch.cuda.empty_cache()

    index = 0 if device.index is None else device.index
    gpu_name = torch.cuda.get_device_name(index)
    total_vram_gb = torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
    print(
        "Inference GPU tuning: "
        f"{gpu_name} ({total_vram_gb:.1f} GB) | mixed_precision={format_amp_dtype(amp_dtype)} | "
        f"matmul_precision={INFERENCE_FLOAT32_MATMUL_PRECISION} | tf32={ENABLE_INFERENCE_TF32} | "
        f"cudnn_benchmark={ENABLE_INFERENCE_CUDNN_BENCHMARK}"
    )
    return amp_dtype


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _to_bytes(value: object) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    return bytes(value) if not isinstance(value, bytes) else value


def list_model_run_dirs(base_root: Path, metadata_name: str) -> list[Path]:
    if not base_root.exists():
        return []
    return sorted(
        [path for path in base_root.iterdir() if path.is_dir() and (path / metadata_name).exists()],
        key=lambda path: -path.stat().st_mtime,
    )


def find_latest_dinov3_checkpoint(model_dir: Path) -> Path:
    last_candidates = sorted(
        [path for path in model_dir.rglob("last.ckpt") if path.is_file()],
        key=lambda path: -path.stat().st_mtime,
    )
    if last_candidates:
        return last_candidates[0]

    candidates = sorted(
        [
            path
            for path in model_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".ckpt", ".pth", ".pt"}
        ],
        key=lambda path: -path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no DINOv3 checkpoints found under {model_dir}")
    return candidates[0]


def resolve_model_configuration() -> dict[str, object]:
    if INFERENCE_MODEL_FAMILY not in VALID_MODEL_FAMILIES:
        raise RuntimeError(
            f"unsupported GEOAI_INFERENCE_MODEL_FAMILY={INFERENCE_MODEL_FAMILY!r}; expected one of {sorted(VALID_MODEL_FAMILIES)}"
        )

    if EXPLICIT_MODEL_DIR is not None:
        model_dir = EXPLICIT_MODEL_DIR
    elif INFERENCE_MODEL_FAMILY == "smp":
        candidates = list_model_run_dirs(PROJECT_ROOT / "outputs" / "models" / "smp_grounded", "smp_metadata.json")
        if not candidates:
            raise FileNotFoundError("no SMP model directories with smp_metadata.json found under outputs/models/smp_grounded")
        model_dir = candidates[0]
    else:
        candidates = list_model_run_dirs(PROJECT_ROOT / "outputs" / "models" / "dinov3_grounded", "dinov3_metadata.json")
        if not candidates:
            raise FileNotFoundError("no DINOv3 model directories with dinov3_metadata.json found under outputs/models/dinov3_grounded")
        model_dir = candidates[0]

    if not model_dir.exists():
        raise FileNotFoundError(f"configured model dir not found: {model_dir}")

    if INFERENCE_MODEL_FAMILY == "smp":
        metadata_path = model_dir / "smp_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"missing SMP metadata file: {metadata_path}")
        metadata = _load_json(metadata_path)
        model_path = EXPLICIT_MODEL_PATH or (model_dir / "best_model.pth")
        if not model_path.exists():
            raise FileNotFoundError(f"missing SMP model checkpoint: {model_path}")
    else:
        metadata_path = model_dir / "dinov3_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"missing DINOv3 metadata file: {metadata_path}")
        metadata = _load_json(metadata_path)
        model_path = EXPLICIT_MODEL_PATH or find_latest_dinov3_checkpoint(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"missing DINOv3 checkpoint: {model_path}")

    run_name = model_dir.name
    inference_root = EXPLICIT_INFERENCE_ROOT or (PROJECT_ROOT / "outputs" / "geoai_inference" / f"{INFERENCE_MODEL_FAMILY}_{run_name}")
    return {
        "model_family": INFERENCE_MODEL_FAMILY,
        "model_dir": model_dir,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "model_path": model_path,
        "run_name": run_name,
        "inference_root": inference_root,
    }


_maybe_initialize_inference_widgets()


def normalize_dinov3_image_tensor(image: torch.Tensor, metadata: dict[str, object]) -> torch.Tensor:
    mean = torch.tensor(metadata.get("input_mean", (0.430, 0.411, 0.296)), dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.tensor(metadata.get("input_std", (0.213, 0.156, 0.143)), dtype=image.dtype, device=image.device).view(-1, 1, 1)
    channels = min(image.shape[0], mean.shape[0])
    normalized = image.clone()
    normalized[:channels] = (normalized[:channels] - mean[:channels]) / std[:channels]
    return normalized


def load_dinov3_segmenter(checkpoint_path: Path, metadata: dict[str, object], device: torch.device):
    from geoai.dinov3_finetune import DINOv3Segmenter

    weights_path_value = metadata.get("weights_path")
    weights_path = None
    if isinstance(weights_path_value, str) and weights_path_value:
        candidate = Path(weights_path_value)
        weights_path = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()

    if checkpoint_path.suffix.lower() == ".ckpt":
        checkpoint_kwargs = {}
        if weights_path is not None:
            checkpoint_kwargs["weights_path"] = str(weights_path)
        model_module = DINOv3Segmenter.load_from_checkpoint(
            str(checkpoint_path),
            map_location=device,
            **checkpoint_kwargs,
        )
    else:
        model_module = DINOv3Segmenter(
            model_name=str(metadata["model_name"]),
            weights_path=str(weights_path) if weights_path is not None else None,
            num_classes=int(metadata["num_classes"]),
            decoder_features=int(metadata["decoder_features"]),
        )
        state_dict = torch.load(checkpoint_path, map_location=device)
        model_module.load_state_dict(state_dict, strict=False)

    model_module = model_module.to(device)
    model_module.eval()
    return model_module


def load_smp_segmenter(model_path: Path, metadata: dict[str, object], device: torch.device):
    from geoai.train import get_smp_model

    model_module = get_smp_model(
        architecture=str(metadata["architecture"]),
        encoder_name=str(metadata["encoder_name"]),
        encoder_weights=None,
        in_channels=int(metadata.get("num_channels", 3) or 3),
        classes=int(metadata.get("num_classes", 2) or 2),
        activation=None,
    )
    state_dict = torch.load(model_path, map_location=device)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model_module.load_state_dict(state_dict)
    model_module = model_module.to(device)
    model_module.eval()
    return model_module


def build_smp_inference_runtime(
    model_path: Path,
    metadata: dict[str, object],
    *,
    device: torch.device | None = None,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, object]:
    runtime_device = device or resolve_runtime_device()
    model_module = load_smp_segmenter(model_path, metadata, runtime_device)
    return {
        "device": runtime_device,
        "model_module": model_module,
        "amp_dtype": amp_dtype if runtime_device.type == "cuda" else None,
    }


def build_dinov3_inference_runtime(
    checkpoint_path: Path,
    metadata: dict[str, object],
    *,
    device: torch.device | None = None,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, object]:
    runtime_device = device or resolve_runtime_device()
    model_module = load_dinov3_segmenter(checkpoint_path, metadata, runtime_device)
    return {
        "device": runtime_device,
        "model_module": model_module,
        "amp_dtype": amp_dtype if runtime_device.type == "cuda" else None,
    }


def dinov3_segment_geotiff_with_runtime_settings(
    *,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    metadata: dict[str, object],
    window_size: int,
    overlap: int,
    batch_size: int,
    runtime: dict[str, object] | None = None,
) -> None:
    from rasterio.windows import Window

    if overlap >= window_size:
        raise ValueError(f"overlap ({overlap}) must be less than window_size ({window_size})")

    if runtime is None:
        device = resolve_runtime_device()
        model_module = load_dinov3_segmenter(checkpoint_path, metadata, device)
        amp_dtype = resolve_inference_amp_dtype(device)
    else:
        device = runtime["device"]
        model_module = runtime["model_module"]
        amp_dtype = runtime.get("amp_dtype")
    patch_size = int(getattr(model_module, "patch_size", metadata.get("patch_size", 16)))

    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        height, width = src.shape
        num_channels = min(src.count, int(metadata.get("num_channels", 3) or 3))

        stride = window_size - overlap
        n_rows = max(1, int(np.ceil((height - overlap) / stride)))
        n_cols = max(1, int(np.ceil((width - overlap) / stride)))
        votes = np.zeros((int(metadata["num_classes"]), height, width), dtype=np.float32)
        count = np.zeros((height, width), dtype=np.float32)
        padded_h = window_size + (patch_size - window_size % patch_size) % patch_size
        padded_w = padded_h

        def _prepare_window(img: np.ndarray) -> tuple[np.ndarray, int, int]:
            if img.shape[0] > num_channels:
                img = img[:num_channels]
            elif img.shape[0] < num_channels:
                padded = np.zeros((num_channels, img.shape[1], img.shape[2]), dtype=np.float32)
                padded[: img.shape[0]] = img
                img = padded

            if img.max() > 1.0:
                img = img / 255.0

            h, w = img.shape[1], img.shape[2]
            if h < padded_h or w < padded_w:
                padded = np.zeros((num_channels, padded_h, padded_w), dtype=np.float32)
                padded[:, :h, :w] = img
                img = padded

            img_tensor = normalize_dinov3_image_tensor(torch.from_numpy(img), metadata).cpu()
            return img_tensor.numpy(), h, w

        def _flush_batch(batch_imgs: list[np.ndarray], batch_meta: list[tuple[int, int, int, int, int, int]]) -> None:
            tensor = torch.from_numpy(np.stack(batch_imgs)).to(device)
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None and device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                logits = model_module(tensor)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

            for index, (row_start, row_end, col_start, col_end, h, w) in enumerate(batch_meta):
                votes[:, row_start:row_end, col_start:col_end] += probs[index, :, :h, :w]
                count[row_start:row_end, col_start:col_end] += 1.0

        with torch.no_grad():
            batch_imgs: list[np.ndarray] = []
            batch_meta: list[tuple[int, int, int, int, int, int]] = []
            progress = tqdm(
                total=n_rows * n_cols,
                disable=not SHOW_WINDOW_PROGRESS,
                desc=f"DINOv3 windows | {input_path.stem}",
                leave=False,
                position=1,
                unit="window",
            )

            for row_index in range(n_rows):
                for col_index in range(n_cols):
                    row_start = row_index * stride
                    col_start = col_index * stride
                    row_end = min(row_start + window_size, height)
                    col_end = min(col_start + window_size, width)

                    window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                    raw = src.read(window=window).astype(np.float32)
                    img, h, w = _prepare_window(raw)

                    batch_imgs.append(img)
                    batch_meta.append((row_start, row_end, col_start, col_end, h, w))

                    if len(batch_imgs) == batch_size:
                        _flush_batch(batch_imgs, batch_meta)
                        progress.update(len(batch_imgs))
                        batch_imgs.clear()
                        batch_meta.clear()

            if batch_imgs:
                _flush_batch(batch_imgs, batch_meta)
                progress.update(len(batch_imgs))

            progress.close()

        count = np.maximum(count, 1.0)
        votes /= count[np.newaxis, :, :]
        output = np.argmax(votes, axis=0).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta.update({"count": 1, "dtype": "uint8", "compress": "lzw"})
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(output, 1)


def smp_segment_geotiff_with_runtime_settings(
    *,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    metadata: dict[str, object],
    window_size: int,
    overlap: int,
    batch_size: int,
    probability_path: Path | None = None,
    probability_threshold: float | None = None,
    runtime: dict[str, object] | None = None,
) -> None:
    from rasterio.windows import Window

    if overlap >= window_size:
        raise ValueError(f"overlap ({overlap}) must be less than window_size ({window_size})")

    if probability_threshold is not None and int(metadata.get("num_classes", 2) or 2) != 2:
        raise ValueError("probability_threshold is only supported for binary classification (num_classes=2)")

    if runtime is None:
        device = resolve_runtime_device()
        model_module = load_smp_segmenter(checkpoint_path, metadata, device)
        amp_dtype = resolve_inference_amp_dtype(device)
    else:
        device = runtime["device"]
        model_module = runtime["model_module"]
        amp_dtype = runtime.get("amp_dtype")

    num_channels = int(metadata.get("num_channels", 3) or 3)
    num_classes = int(metadata.get("num_classes", 2) or 2)

    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        height, width = src.shape

        stride = window_size - overlap
        n_rows = max(1, int(np.ceil((height - overlap) / stride)))
        n_cols = max(1, int(np.ceil((width - overlap) / stride)))
        votes = np.zeros((num_classes, height, width), dtype=np.float32)
        count = np.zeros((height, width), dtype=np.float32)

        def _prepare_window(img: np.ndarray) -> tuple[np.ndarray, int, int]:
            img = img.astype(np.float32)
            if img.shape[0] > num_channels:
                img = img[:num_channels]
            elif img.shape[0] < num_channels:
                padded = np.zeros((num_channels, img.shape[1], img.shape[2]), dtype=np.float32)
                padded[: img.shape[0]] = img
                img = padded

            if img.max() > 1.0:
                img = img / 255.0

            h, w = img.shape[1], img.shape[2]
            if h < window_size or w < window_size:
                padded = np.zeros((num_channels, window_size, window_size), dtype=np.float32)
                padded[:, :h, :w] = img
                img = padded

            return img, h, w

        def _window_weight(h: int, w: int) -> np.ndarray:
            if overlap <= 0:
                return np.ones((h, w), dtype=np.float32)

            y_grid, x_grid = np.mgrid[0:h, 0:w]
            edge_distance = np.minimum.reduce(
                [
                    x_grid,
                    w - x_grid - 1,
                    y_grid,
                    h - y_grid - 1,
                ]
            ).astype(np.float32)
            edge_distance = np.minimum(edge_distance, overlap / 2)
            return np.maximum(edge_distance / (overlap / 2), 0.1).astype(np.float32)

        def _flush_batch(batch_imgs: list[np.ndarray], batch_meta: list[tuple[int, int, int, int, int, int]]) -> None:
            tensor = torch.from_numpy(np.stack(batch_imgs)).to(device)
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None and device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                logits = model_module(tensor)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

            for index, (row_start, row_end, col_start, col_end, h, w) in enumerate(batch_meta):
                weight = _window_weight(h, w)
                window_probs = probs[index, :, :h, :w]
                votes[:, row_start:row_end, col_start:col_end] += window_probs * weight[np.newaxis, :, :]
                count[row_start:row_end, col_start:col_end] += weight

        with torch.no_grad():
            batch_imgs: list[np.ndarray] = []
            batch_meta: list[tuple[int, int, int, int, int, int]] = []
            progress = tqdm(
                total=n_rows * n_cols,
                disable=not SHOW_WINDOW_PROGRESS,
                desc=f"SMP windows | {input_path.stem}",
                leave=False,
                position=1,
                unit="window",
            )

            for row_index in range(n_rows):
                for col_index in range(n_cols):
                    row_start = row_index * stride
                    col_start = col_index * stride
                    row_end = min(row_start + window_size, height)
                    col_end = min(col_start + window_size, width)

                    window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                    raw = src.read(window=window)
                    img, h, w = _prepare_window(raw)

                    batch_imgs.append(img)
                    batch_meta.append((row_start, row_end, col_start, col_end, h, w))

                    if len(batch_imgs) == batch_size:
                        _flush_batch(batch_imgs, batch_meta)
                        progress.update(len(batch_imgs))
                        batch_imgs.clear()
                        batch_meta.clear()

            if batch_imgs:
                _flush_batch(batch_imgs, batch_meta)
                progress.update(len(batch_imgs))

            progress.close()

        valid_pixels = count > 0
        normalized_probs = np.zeros_like(votes)
        normalized_probs[:, valid_pixels] = votes[:, valid_pixels] / count[valid_pixels]

        if probability_threshold is not None and num_classes == 2:
            output = (normalized_probs[1] >= probability_threshold).astype(np.uint8)
        else:
            output = np.argmax(normalized_probs, axis=0).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta.update({"count": 1, "dtype": "uint8", "compress": "lzw"})
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(output, 1)

    if probability_path is not None:
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        probability_meta = meta.copy()
        probability_meta.update({"count": num_classes, "dtype": "float32"})
        with rasterio.open(probability_path, "w", **probability_meta) as dst:
            dst.write(normalized_probs.astype(np.float32))


def resolve_inference_group_key(row: pd.Series | object) -> str:
    if isinstance(row, pd.Series):
        local_asset_path = row.get("local_asset_path")
        item_id = row.get("item_id")
        source = row.get("source")
        tile_path = row.get("tile_path")
    else:
        local_asset_path = getattr(row, "local_asset_path", None)
        item_id = getattr(row, "item_id", None)
        source = getattr(row, "source", None)
        tile_path = getattr(row, "tile_path", None)

    if local_asset_path:
        return f"asset::{Path(str(local_asset_path)).as_posix()}"
    if item_id:
        source_text = str(source) if source is not None else "unknown"
        return f"item::{source_text}::{item_id}"
    if tile_path:
        return f"tile::{Path(str(tile_path)).as_posix()}"
    return "tile::unknown"


def prepare_selected_for_inference_batches(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()

    ordered = selected.copy()
    ordered["inference_group_key"] = ordered.apply(resolve_inference_group_key, axis=1)
    if ENABLE_SOURCE_GROUP_TILE_BATCHING:
        ordered = ordered.sort_values(
            by=["inference_group_key", "municipio", "item_id", "h3_cell_id", "tile_path"],
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)
    else:
        ordered = ordered.reset_index(drop=True)
    ordered["processing_order"] = np.arange(len(ordered), dtype=np.int64)
    return ordered


def segment_exact_tile_batch_with_runtime_settings(
    *,
    batch_contexts: list[dict[str, object]],
    config: dict[str, object],
    smp_runtime: dict[str, object] | None = None,
    dinov3_runtime: dict[str, object] | None = None,
) -> bool:
    if not batch_contexts:
        return True

    metadata = config["metadata"]
    model_family = str(config["model_family"])
    prepared_imgs: list[np.ndarray] = []
    metas: list[dict[str, object]] = []
    original_sizes: list[tuple[int, int]] = []

    if model_family == "smp":
        if smp_runtime is None:
            raise RuntimeError("SMP runtime is required for exact tile batching.")
        device = smp_runtime["device"]
        model_module = smp_runtime["model_module"]
        amp_dtype = smp_runtime.get("amp_dtype")
        num_channels = int(metadata.get("num_channels", 3) or 3)
        num_classes = int(metadata.get("num_classes", 2) or 2)
        if PROBABILITY_THRESHOLD is not None and num_classes != 2:
            raise ValueError("probability_threshold is only supported for binary classification (num_classes=2)")

        for context in batch_contexts:
            input_path = Path(context["inference_input_path"])
            with rasterio.open(input_path) as src:
                if src.width != WINDOW_SIZE or src.height != WINDOW_SIZE:
                    return False
                metas.append(src.meta.copy())
                raw = src.read().astype(np.float32)

            if raw.shape[0] > num_channels:
                raw = raw[:num_channels]
            elif raw.shape[0] < num_channels:
                padded = np.zeros((num_channels, raw.shape[1], raw.shape[2]), dtype=np.float32)
                padded[: raw.shape[0]] = raw
                raw = padded
            if raw.max() > 1.0:
                raw = raw / 255.0
            prepared_imgs.append(raw)
            original_sizes.append((raw.shape[1], raw.shape[2]))

        tensor = torch.from_numpy(np.stack(prepared_imgs)).to(device)
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None and device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad():
            with autocast_context:
                logits = model_module(tensor)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

        for index, context in enumerate(batch_contexts):
            meta = metas[index].copy()
            h, w = original_sizes[index]
            normalized_probs = probs[index, :, :h, :w].astype(np.float32)
            if PROBABILITY_THRESHOLD is not None and num_classes == 2:
                output = (normalized_probs[1] >= PROBABILITY_THRESHOLD).astype(np.uint8)
            else:
                output = np.argmax(normalized_probs, axis=0).astype(np.uint8)

            mask_path = Path(context["mask_path"])
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            meta.update({"count": 1, "dtype": "uint8", "compress": "lzw"})
            with rasterio.open(mask_path, "w", **meta) as dst:
                dst.write(output, 1)

            probability_path = context.get("probability_path")
            if probability_path is not None:
                probability_meta = meta.copy()
                probability_meta.update({"count": num_classes, "dtype": "float32"})
                probability_path = Path(probability_path)
                probability_path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(probability_path, "w", **probability_meta) as dst:
                    dst.write(normalized_probs)
        return True

    if model_family == "dinov3":
        if dinov3_runtime is None:
            raise RuntimeError("DINOv3 runtime is required for exact tile batching.")
        device = dinov3_runtime["device"]
        model_module = dinov3_runtime["model_module"]
        amp_dtype = dinov3_runtime.get("amp_dtype")
        patch_size = int(getattr(model_module, "patch_size", metadata.get("patch_size", 16)))
        num_channels = min(3, int(metadata.get("num_channels", 3) or 3))
        padded_h = WINDOW_SIZE + (patch_size - WINDOW_SIZE % patch_size) % patch_size
        padded_w = padded_h

        for context in batch_contexts:
            input_path = Path(context["inference_input_path"])
            with rasterio.open(input_path) as src:
                if src.width != WINDOW_SIZE or src.height != WINDOW_SIZE:
                    return False
                metas.append(src.meta.copy())
                raw = src.read().astype(np.float32)

            if raw.shape[0] > num_channels:
                raw = raw[:num_channels]
            elif raw.shape[0] < num_channels:
                padded = np.zeros((num_channels, raw.shape[1], raw.shape[2]), dtype=np.float32)
                padded[: raw.shape[0]] = raw
                raw = padded
            if raw.max() > 1.0:
                raw = raw / 255.0

            h, w = raw.shape[1], raw.shape[2]
            if h < padded_h or w < padded_w:
                padded = np.zeros((num_channels, padded_h, padded_w), dtype=np.float32)
                padded[:, :h, :w] = raw
                raw = padded

            img_tensor = normalize_dinov3_image_tensor(torch.from_numpy(raw), metadata).cpu().numpy()
            prepared_imgs.append(img_tensor)
            original_sizes.append((h, w))

        tensor = torch.from_numpy(np.stack(prepared_imgs)).to(device)
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None and device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad():
            with autocast_context:
                logits = model_module(tensor)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

        for index, context in enumerate(batch_contexts):
            meta = metas[index].copy()
            h, w = original_sizes[index]
            output = np.argmax(probs[index, :, :h, :w], axis=0).astype(np.uint8)
            mask_path = Path(context["mask_path"])
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            meta.update({"count": 1, "dtype": "uint8", "compress": "lzw"})
            with rasterio.open(mask_path, "w", **meta) as dst:
                dst.write(output, 1)
        return True

    return False


def load_naip_tile_candidates() -> pd.DataFrame:
    if STAC_TILE_MANIFEST.exists():
        frame = pd.read_parquet(STAC_TILE_MANIFEST)
        if frame.empty:
            return frame
        if "source" in frame.columns:
            frame = frame[frame["source"].isin(NAIP_SOURCE_NAMES)].copy()
        if "asset_role" in frame.columns:
            frame = frame[frame["asset_role"].fillna("").str.lower() == "visual"].copy()
        if "status" in frame.columns:
            frame = frame[frame["status"].fillna("").str.lower() == "fetched"].copy()
        if "municipio" in frame.columns:
            frame = frame[frame["municipio"].isin(TARGET_MUNICIPALITIES)].copy()
        if "building_count" in frame.columns:
            frame = frame[frame["building_count"].fillna(0).astype(int) >= MIN_TILE_BUILDING_COUNT].copy()
        if not INCLUDE_CROSS_BOUNDARY_TILES and "crosses_municipality_boundary" in frame.columns:
            frame = frame[~frame["crosses_municipality_boundary"].fillna(False)].copy()

        if frame.empty:
            return frame

        frame["tile_abs_path"] = frame["tile_path"].map(lambda value: (PROJECT_ROOT / str(value)).resolve())
        frame = frame[frame["tile_abs_path"].map(Path.exists)].copy()
        frame["local_asset_abs_path"] = frame["local_asset_path"].map(
            lambda value: (PROJECT_ROOT / str(value)).resolve() if isinstance(value, str) and value else None
        )
        return frame.reset_index(drop=True)

    candidates = collect_naip_stac_preview_rasters(PROJECT_ROOT, exclude_stems=set())
    rows: list[dict[str, object]] = []
    for path in candidates:
        parts = list(path.parts)
        municipio_slug = next((part for part in parts if part in {"San_Juan", "Isabela"}), None)
        municipio = municipio_slug.replace("_", " ") if municipio_slug else None
        if municipio and municipio not in TARGET_MUNICIPALITIES:
            continue
        rows.append(
            {
                "source": infer_stac_source_name(path),
                "item_id": path.parent.name,
                "asset_role": "visual",
                "municipio": municipio,
                "municipio_geoid": None,
                "h3_cell_id": path.stem.split("_visual")[0],
                "building_count": None,
                "municipality_building_count": None,
                "crosses_municipality_boundary": None,
                "local_asset_path": None,
                "tile_path": str(path.relative_to(PROJECT_ROOT)),
                "tile_abs_path": path.resolve(),
                "local_asset_abs_path": None,
            }
        )
    return pd.DataFrame.from_records(rows)


def load_overture_building_footprints(selected: pd.DataFrame) -> gpd.GeoDataFrame:
    if selected.empty:
        return gpd.GeoDataFrame(columns=["building_id", "municipio", "h3_cell_id", "geometry"], geometry="geometry", crs="EPSG:4326")

    db_path = resolve_db_path()
    try:
        con = duckdb.connect(str(db_path), read_only=True)
    except duckdb.ConnectionException:
        con = duckdb.connect(str(db_path))
    except duckdb.IOException as exc:
        print(f"building footprint lookup skipped: could not open DuckDB read-only ({exc})")
        return gpd.GeoDataFrame(columns=["building_id", "municipio", "h3_cell_id", "geometry"], geometry="geometry", crs="EPSG:4326")
    try:
        table_is_available = bool(
            con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'pr_overture_buildings';"
            ).fetchone()[0]
        )
        if not table_is_available:
            return gpd.GeoDataFrame(columns=["building_id", "municipio", "h3_cell_id", "geometry"], geometry="geometry", crs="EPSG:4326")

        municipios = sorted({str(value) for value in selected.get("municipio", pd.Series(dtype=str)).dropna().astype(str).tolist()})
        h3_ids = sorted({str(value) for value in selected.get("h3_cell_id", pd.Series(dtype=str)).dropna().astype(str).tolist()})

        where_parts = ["geometry IS NOT NULL"]
        params: list[object] = []
        if municipios:
            where_parts.append(f"municipality_name IN ({', '.join(['?'] * len(municipios))})")
            params.extend(municipios)
        if h3_ids:
            where_parts.append(f"h3_cell_id IN ({', '.join(['?'] * len(h3_ids))})")
            params.extend(h3_ids)

        query = f"""
            SELECT
                CAST(id AS VARCHAR) AS building_id,
                CAST(municipality_name AS VARCHAR) AS municipio,
                CAST(h3_cell_id AS VARCHAR) AS h3_cell_id,
                ST_AsWKB(geometry) AS geometry_wkb
            FROM pr_overture_buildings
            WHERE {' AND '.join(where_parts)};
        """
        frame = con.execute(query, params).fetchdf()
    finally:
        con.close()

    if frame.empty:
        return gpd.GeoDataFrame(columns=["building_id", "municipio", "h3_cell_id", "geometry"], geometry="geometry", crs="EPSG:4326")

    geometry = gpd.GeoSeries(frame["geometry_wkb"].map(lambda value: from_wkb(_to_bytes(value))), crs="EPSG:4326")
    return gpd.GeoDataFrame(frame.drop(columns=["geometry_wkb"]), geometry=geometry, crs="EPSG:4326")


def clip_detections_to_buildings(
    detections: gpd.GeoDataFrame,
    *,
    row: pd.Series | object,
    building_footprints: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if detections.empty:
        return detections.iloc[0:0].copy()
    if building_footprints.empty:
        return detections

    row_h3 = str(getattr(row, "h3_cell_id", "") or "")
    row_municipio = str(getattr(row, "municipio", "") or "")
    candidate = building_footprints
    if row_h3:
        candidate = candidate[candidate["h3_cell_id"].fillna("") == row_h3].copy()
    if candidate.empty and row_municipio:
        candidate = building_footprints[building_footprints["municipio"].fillna("") == row_municipio].copy()
    if candidate.empty:
        return detections.iloc[0:0].copy()

    candidate = candidate.to_crs(detections.crs)
    clipped = gpd.overlay(
        detections,
        candidate[["building_id", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    if clipped.empty:
        return detections.iloc[0:0].copy()
    clipped = clipped.drop(columns=["building_id"], errors="ignore")
    return gpd.GeoDataFrame(clipped, geometry="geometry", crs=detections.crs)


def select_building_footprints_for_row(row: pd.Series | object, building_footprints: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if building_footprints.empty:
        return building_footprints

    row_h3 = str(getattr(row, "h3_cell_id", "") or "")
    row_municipio = str(getattr(row, "municipio", "") or "")
    candidates = building_footprints
    if row_h3 and "h3_cell_id" in candidates.columns:
        candidates = candidates[candidates["h3_cell_id"].fillna("").astype(str) == row_h3].copy()
    if candidates.empty and row_municipio and "municipio" in building_footprints.columns:
        candidates = building_footprints[building_footprints["municipio"].fillna("").astype(str) == row_municipio].copy()
    return candidates


def buffered_building_geometries_for_raster(
    buildings: gpd.GeoDataFrame,
    raster_crs,
    *,
    buffer_m: float,
) -> list[object]:
    if buildings.empty or raster_crs is None:
        return []
    raster_buildings = buildings.to_crs(raster_crs)
    if buffer_m <= 0:
        return [geom for geom in raster_buildings.geometry if geom is not None and not geom.is_empty]

    buffered = buildings.to_crs("EPSG:3857").geometry.buffer(buffer_m)
    buffered = gpd.GeoSeries(buffered, crs="EPSG:3857").to_crs(raster_crs)
    return [geom for geom in buffered if geom is not None and not geom.is_empty]


def mask_raster_to_buildings(
    raster_path: Path | None,
    *,
    row: pd.Series | object,
    building_footprints: gpd.GeoDataFrame,
    buffer_m: float,
) -> bool:
    if raster_path is None or not raster_path.exists() or building_footprints.empty:
        return False

    buildings = select_building_footprints_for_row(row, building_footprints)
    if buildings.empty:
        return False

    with rasterio.open(raster_path, "r+") as dataset:
        shapes = buffered_building_geometries_for_raster(buildings, dataset.crs, buffer_m=buffer_m)
        if not shapes:
            return False
        inside_buildings = geometry_mask(
            [mapping(shape) for shape in shapes],
            out_shape=(dataset.height, dataset.width),
            transform=dataset.transform,
            invert=True,
        )
        data = dataset.read()
        data[:, ~inside_buildings] = 0
        dataset.write(data)
    return True


def write_building_masked_raster(
    source_raster_path: Path,
    output_path: Path,
    *,
    row: pd.Series | object,
    building_footprints: gpd.GeoDataFrame,
    buffer_m: float,
    mask_output_path: Path | None = None,
) -> bool:
    if not source_raster_path.exists() or building_footprints.empty:
        return False

    buildings = select_building_footprints_for_row(row, building_footprints)
    if buildings.empty:
        return False

    with rasterio.open(source_raster_path) as dataset:
        shapes = buffered_building_geometries_for_raster(buildings, dataset.crs, buffer_m=buffer_m)
        if not shapes:
            return False
        inside_buildings = geometry_mask(
            [mapping(shape) for shape in shapes],
            out_shape=(dataset.height, dataset.width),
            transform=dataset.transform,
            invert=True,
        )
        data = dataset.read()
        data[:, ~inside_buildings] = 0
        meta = dataset.meta.copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta.update(compress="lzw")
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(data)

    if mask_output_path is not None:
        mask_meta = meta.copy()
        mask_meta.update(count=1, dtype="uint8")
        with rasterio.open(mask_output_path, "w", **mask_meta) as dst:
            dst.write(inside_buildings.astype(np.uint8), 1)
    return True


def cleanup_artifact_path(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    try:
        path.unlink()
    except OSError:
        pass


def cleanup_output_path(path: Path, counters: dict[str, int]) -> None:
    if not path.exists():
        return
    try:
        if path.is_file():
            counters["deleted_bytes"] += int(path.stat().st_size)
            path.unlink()
            counters["deleted_files"] += 1
        elif path.is_dir():
            path.rmdir()
            counters["deleted_directories"] += 1
    except OSError:
        pass


def cleanup_overwrite_outputs(
    inference_root: Path,
    config: dict[str, object],
    *,
    include_tile_artifacts: bool,
    preserve_local_reporting_inputs: bool,
) -> dict[str, int]:
    counters = {"deleted_files": 0, "deleted_directories": 0, "deleted_bytes": 0}
    seen: set[Path] = set()
    root_filenames = [
        "building_level_inference_metrics.geojson",
        "building_level_inference_metrics.csv",
        "building_level_inference_summary.csv",
        "inference_db_schema_reference.csv",
        "reporting_summary.json",
        "inference_summary.json",
        "dry_run_summary.json",
        "db_commit_summary.json",
        "inference_commit_summary.json",
    ]
    if not preserve_local_reporting_inputs:
        root_filenames.extend(["tile_summary.csv", "merged_detections.parquet"])

    for filename in root_filenames:
        path = inference_root / filename
        if path in seen:
            continue
        seen.add(path)
        cleanup_output_path(path, counters)

    if include_tile_artifacts:
        recursive_patterns = (
            "*_pred.tif",
            "*_prob.tif",
            "*_pred.geojson",
            "*_pred_props.geojson",
            "*_input_clip.tif",
            "*_input_clip_mask.tif",
            "*_interactive_vectors.geojson",
            "*_review.png",
            "*_vector_review.png",
            "*_review_map.html",
        )
        for pattern in recursive_patterns:
            for path in inference_root.rglob(pattern):
                if not path.is_file() or path in seen:
                    continue
                seen.add(path)
                cleanup_output_path(path, counters)

        for directory in sorted(
            (path for path in inference_root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            cleanup_output_path(directory, counters)

    map_prefix = f"{config['model_family']}_{config['run_name']}_"
    if MAPS_ROOT.exists():
        for path in MAPS_ROOT.glob(f"{map_prefix}*"):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            cleanup_output_path(path, counters)

    return counters


def should_persist_tile_mask_artifacts() -> bool:
    return PRESERVE_TILE_MASK_ARTIFACTS


def should_persist_clipped_input_artifacts() -> bool:
    return PRESERVE_CLIPPED_INPUT_ARTIFACTS


def merged_detections_artifact_path(inference_root: Path) -> Path:
    return inference_root / "merged_detections.parquet"


def write_merged_detections_artifact(inference_root: Path, detections: gpd.GeoDataFrame) -> Path | None:
    artifact_path = merged_detections_artifact_path(inference_root)
    if detections.empty:
        cleanup_artifact_path(artifact_path)
        return None
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    detections.to_crs("EPSG:4326").to_parquet(artifact_path, index=False)
    return artifact_path


def load_local_detections_from_artifact(inference_root: Path) -> gpd.GeoDataFrame | None:
    artifact_path = merged_detections_artifact_path(inference_root)
    if not artifact_path.exists():
        return None
    try:
        detections = gpd.read_parquet(artifact_path)
    except Exception as exc:
        print(f"local reporting reload skipped for {artifact_path.name}: {exc}")
        return None
    if detections.empty:
        return empty_detection_gdf()
    return detections.to_crs("EPSG:4326")


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


def build_output_stem(inference_root: Path, row: pd.Series | object) -> Path:
    municipio = getattr(row, "municipio", None) if not isinstance(row, pd.Series) else row.get("municipio")
    municipio_slug = _slugify(municipio or "unknown")
    tile_key = getattr(row, "h3_cell_id", None) if not isinstance(row, pd.Series) else row.get("h3_cell_id")
    if not tile_key:
        tile_path_value = getattr(row, "tile_path", None) if not isinstance(row, pd.Series) else row.get("tile_path")
        tile_key = Path(str(tile_path_value)).stem if tile_path_value else "tile"
    return inference_root / municipio_slug / str(tile_key)


def run_segmentation_inference(
    geoai_module,
    *,
    raster_path: Path,
    mask_path: Path,
    probability_path: Path | None,
    config: dict[str, object],
    smp_runtime: dict[str, object] | None = None,
    dinov3_runtime: dict[str, object] | None = None,
) -> None:
    metadata = config["metadata"]
    if config["model_family"] == "smp":
        smp_segment_geotiff_with_runtime_settings(
            input_path=raster_path,
            output_path=mask_path,
            checkpoint_path=Path(config["model_path"]),
            metadata=metadata,
            window_size=WINDOW_SIZE,
            overlap=OVERLAP,
            batch_size=INFERENCE_BATCH_SIZE,
            probability_path=probability_path,
            probability_threshold=PROBABILITY_THRESHOLD,
            runtime=smp_runtime,
        )
        return

    dinov3_segment_geotiff_with_runtime_settings(
        input_path=raster_path,
        output_path=mask_path,
        checkpoint_path=Path(config["model_path"]),
        metadata=metadata,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        batch_size=INFERENCE_BATCH_SIZE,
        runtime=dinov3_runtime,
    )


def enrich_prediction_vectors(geoai_module, detections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if detections.empty:
        return detections
    gdf = detections.copy()
    if hasattr(geoai_module, "add_geometric_properties"):
        gdf = geoai_module.add_geometric_properties(gdf)
    if gdf.columns.duplicated().any():
        gdf = gdf.loc[:, ~gdf.columns.duplicated(keep="last")]
    if "area_m2" in gdf.columns:
        gdf = gdf[gdf["area_m2"].between(MIN_AREA_M2, MAX_AREA_M2)].copy()
    if "elongation" in gdf.columns:
        gdf = gdf[gdf["elongation"] < MAX_ELONGATION].copy()
    return gdf


def render_review_visuals(
    geoai_module,
    *,
    raster_path: Path,
    inference_input_path: Path | None,
    input_clip_mask_path: Path | None,
    mask_path: Path | None,
    detections: gpd.GeoDataFrame,
    output_stem: Path,
) -> str | None:
    review_png_path = None
    if WRITE_STATIC_REVIEW_ARTIFACTS:
        review_outputs = render_prediction_review_bundle(
            image_path=raster_path,
            output_stem=output_stem,
            transformed_image_path=inference_input_path if inference_input_path is not None and inference_input_path != raster_path else None,
            transformed_mask_path=input_clip_mask_path,
            predicted_mask_path=mask_path,
            vector_gdf=detections if not detections.empty else None,
            suptitle=output_stem.name,
        )
        review_png = review_outputs.get("review_png_path")
        if review_png is not None:
            review_png_path = str(review_png.relative_to(PROJECT_ROOT))
        if running_in_notebook():
            try:
                from IPython.display import Image, display
            except ImportError:
                pass
            else:
                display(Image(filename=str(review_outputs["review_png_path"])))
                vector_review_png = review_outputs.get("vector_review_png_path")
                if vector_review_png is not None:
                    display(Image(filename=str(vector_review_png)))

    if not WRITE_INTERACTIVE_REVIEW_ARTIFACTS:
        return review_png_path

    temp_vector_path = None
    if not detections.empty:
        temp_vector_path = output_stem.with_name(f"{output_stem.name}_interactive_vectors.geojson")
        detections.to_file(temp_vector_path, driver="GeoJSON")

    try:
        if temp_vector_path is not None and temp_vector_path.exists():
            review_map = geoai_module.view_vector_interactive(
                str(temp_vector_path),
                tiles=str(inference_input_path or raster_path),
            )
            if hasattr(review_map, "save"):
                review_map.save(str(output_stem.with_name(f"{output_stem.name}_review_map.html")))
    except Exception as exc:
        print(f"interactive review map skipped for {raster_path.name}: {exc}")

    try:
        if mask_path is not None and mask_path.exists():
            split_map = geoai_module.create_split_map(
                left_layer=str(mask_path),
                right_layer=str(inference_input_path or raster_path),
                left_label="Predicted mask",
                right_label="Inference input",
            )
            if hasattr(split_map, "save"):
                split_map.save(str(output_stem.with_name(f"{output_stem.name}_split_map.html")))
    except Exception as exc:
        print(f"interactive split map skipped for {raster_path.name}: {exc}")
    finally:
        cleanup_artifact_path(temp_vector_path)
    return review_png_path


def build_detection_frame(
    gdf: gpd.GeoDataFrame,
    row: pd.Series | object,
    config: dict[str, object],
    *,
    mask_path: Path | None,
    vector_path: Path | None,
    probability_path: Path | None,
) -> gpd.GeoDataFrame:
    gdf = gdf.reset_index(drop=True).copy()
    model_arch_key, model_arch = resolve_model_arch_info(
        config.get("model_family"),
        config.get("run_name"),
        metadata=config.get("metadata") if isinstance(config, dict) else None,
    )
    gdf["detection_key"] = [f"{getattr(row, 'h3_cell_id', None) or Path(str(getattr(row, 'tile_path', 'tile'))).stem}_{index}" for index in range(len(gdf))]
    gdf["model_family"] = config["model_family"]
    gdf["model_run_name"] = config["run_name"]
    gdf["model_arch_key"] = model_arch_key
    gdf["model_arch"] = model_arch
    gdf["source"] = getattr(row, "source", None)
    gdf["item_id"] = getattr(row, "item_id", None)
    gdf["municipio"] = getattr(row, "municipio", None)
    gdf["municipio_geoid"] = getattr(row, "municipio_geoid", None)
    gdf["h3_cell_id"] = getattr(row, "h3_cell_id", None)
    gdf["building_count"] = getattr(row, "building_count", None)
    gdf["municipality_building_count"] = getattr(row, "municipality_building_count", None)
    gdf["crosses_municipality_boundary"] = getattr(row, "crosses_municipality_boundary", None)
    gdf["tile_path"] = str(Path(str(getattr(row, "tile_path", mask_path or "tile"))).as_posix())
    gdf["local_asset_path"] = str(Path(str(getattr(row, "local_asset_path", ""))).as_posix()) if getattr(row, "local_asset_path", None) else None
    gdf["predicted_mask_path"] = str(mask_path.relative_to(PROJECT_ROOT)) if mask_path is not None and mask_path.exists() else None
    gdf["probability_path"] = str(probability_path.relative_to(PROJECT_ROOT)) if probability_path is not None and probability_path.exists() else None
    gdf["vector_path"] = str(vector_path.relative_to(PROJECT_ROOT)) if vector_path is not None and vector_path.exists() else None
    gdf["inference_mode"] = INFERENCE_MODE
    return gdf.to_crs("EPSG:4326")


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    return bool(
        con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?;",
            [table_name],
        ).fetchone()[0]
    )


def write_detection_table(con: duckdb.DuckDBPyConnection, detections: gpd.GeoDataFrame) -> None:
    if detections.empty:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {DETECTION_TABLE} AS
            SELECT CAST(NULL AS VARCHAR) AS detection_key,
                   CAST(NULL AS VARCHAR) AS model_family,
                   CAST(NULL AS VARCHAR) AS model_run_name,
                     CAST(NULL AS VARCHAR) AS model_arch_key,
                     CAST(NULL AS VARCHAR) AS model_arch,
                   CAST(NULL AS VARCHAR) AS source,
                   CAST(NULL AS VARCHAR) AS item_id,
                   CAST(NULL AS VARCHAR) AS municipio,
                   CAST(NULL AS VARCHAR) AS municipio_geoid,
                   CAST(NULL AS VARCHAR) AS h3_cell_id,
                   CAST(NULL AS INTEGER) AS building_count,
                   CAST(NULL AS INTEGER) AS municipality_building_count,
                   CAST(NULL AS BOOLEAN) AS crosses_municipality_boundary,
                   CAST(NULL AS VARCHAR) AS tile_path,
                   CAST(NULL AS VARCHAR) AS local_asset_path,
                   CAST(NULL AS VARCHAR) AS predicted_mask_path,
                   CAST(NULL AS VARCHAR) AS probability_path,
                   CAST(NULL AS VARCHAR) AS vector_path,
                   CAST(NULL AS VARCHAR) AS inference_mode,
                   CAST(NULL AS GEOMETRY) AS geometry
            WHERE FALSE;
            """
        )
        return

    staged = pd.DataFrame(detections.drop(columns=["geometry"]))
    staged["geometry_wkb"] = detections.geometry.to_wkb()
    con.register("staged_solar_detections", staged)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {DETECTION_TABLE} AS
        SELECT * EXCLUDE (geometry_wkb),
               ST_GeomFromWKB(geometry_wkb) AS geometry
        FROM staged_solar_detections;
        """
    )
    con.unregister("staged_solar_detections")
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{DETECTION_TABLE}_geom ON {DETECTION_TABLE} USING RTREE (geometry);")


def backfill_detection_table_model_arch(con: duckdb.DuckDBPyConnection) -> None:
    if not table_exists(con, DETECTION_TABLE):
        return

    con.execute(f"ALTER TABLE {DETECTION_TABLE} ADD COLUMN IF NOT EXISTS model_arch_key VARCHAR;")
    con.execute(f"ALTER TABLE {DETECTION_TABLE} ADD COLUMN IF NOT EXISTS model_arch VARCHAR;")
    con.execute(
        f"""
        UPDATE {DETECTION_TABLE}
        SET model_family = COALESCE(NULLIF(TRIM(CAST(model_family AS VARCHAR)), ''), '{LEGACY_MODEL_FAMILY_BACKFILL}'),
            model_run_name = COALESCE(NULLIF(TRIM(CAST(model_run_name AS VARCHAR)), ''), '{LEGACY_MODEL_RUN_NAME_BACKFILL}')
        WHERE TRUE;
        """
    )
    con.execute(
        f"""
        UPDATE {DETECTION_TABLE}
        SET model_arch_key = CASE
                WHEN LOWER(COALESCE(model_family, '')) = 'dinov3' OR LOWER(COALESCE(model_run_name, '')) LIKE '%dinov3%' THEN 'dinov3'
                WHEN LOWER(COALESCE(model_run_name, '')) LIKE 'pan__%' THEN 'pan'
                WHEN LOWER(COALESCE(model_run_name, '')) LIKE 'fpn__%' THEN 'fpn'
                WHEN LOWER(COALESCE(model_run_name, '')) LIKE 'segformer__%' THEN 'segformer'
                WHEN LOWER(COALESCE(model_run_name, '')) LIKE 'deeplabv3plus__%' THEN 'deeplabv3plus'
                WHEN LOWER(COALESCE(model_run_name, '')) LIKE 'upernet__%' THEN 'upernet'
                ELSE COALESCE(NULLIF(TRIM(CAST(model_arch_key AS VARCHAR)), ''), LOWER(COALESCE(model_family, 'unknown')))
            END,
            model_arch = CASE
                WHEN LOWER(COALESCE(model_family, '')) = 'dinov3' OR LOWER(COALESCE(model_run_name, '')) LIKE '%dinov3%' THEN 'DINOv3'
                WHEN LOWER(COALESCE(model_run_name, '')) LIKE 'pan__%' THEN 'PAN'
                WHEN LOWER(COALESCE(model_run_name, '')) LIKE 'fpn__%' THEN 'FPN'
                WHEN LOWER(COALESCE(model_run_name, '')) LIKE 'segformer__%' THEN 'Segformer'
                WHEN LOWER(COALESCE(model_run_name, '')) LIKE 'deeplabv3plus__%' THEN 'DeepLabV3+'
                WHEN LOWER(COALESCE(model_run_name, '')) LIKE 'upernet__%' THEN 'UPerNet'
                ELSE COALESCE(NULLIF(TRIM(CAST(model_arch AS VARCHAR)), ''), 'Unknown')
            END
        WHERE model_arch_key IS NULL
           OR TRIM(CAST(model_arch_key AS VARCHAR)) = ''
           OR model_arch IS NULL
           OR TRIM(CAST(model_arch AS VARCHAR)) = '';
        """
    )


def write_tile_summary_table(con: duckdb.DuckDBPyConnection, tile_df: pd.DataFrame) -> None:
    con.register("staged_tile_summary", tile_df)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {TILE_SUMMARY_TABLE} AS
        SELECT *
        FROM staged_tile_summary;
        """
    )
    con.unregister("staged_tile_summary")


def build_building_hits_table(con: duckdb.DuckDBPyConnection, config: dict[str, object]) -> bool:
    if not table_exists(con, "pr_overture_buildings"):
        print("building-level detection summary skipped: pr_overture_buildings is unavailable in the vector DB.")
        return False

    has_osm_labels = table_exists(con, "pr_osm_rooftop_pv_polygons")
    munis_sql = ", ".join(f"'{municipio}'" for municipio in TARGET_MUNICIPALITIES)
    model_family = str(config["model_family"])
    run_name = str(config["run_name"])
    _, model_arch = resolve_model_arch_info(
        model_family,
        run_name,
        metadata=config.get("metadata") if isinstance(config, dict) else None,
    )
    osm_join_cte = """
        , osm_join AS (
            SELECT
                b.building_id,
                COUNT(*) FILTER (WHERE o.geometry IS NOT NULL) AS osm_pv_label_count
            FROM buildings AS b
            LEFT JOIN pr_osm_rooftop_pv_polygons AS o
              ON ST_Intersects(b.geometry, o.geometry)
            GROUP BY b.building_id
        )
    """ if has_osm_labels else ""
    osm_select_sql = """
            COALESCE(o.osm_pv_label_count, 0) > 0 AS has_osm_pv_label,
            COALESCE(o.osm_pv_label_count, 0) AS osm_pv_label_count,
            (COALESCE(d.pv_detected_count, 0) > 0 AND COALESCE(o.osm_pv_label_count, 0) > 0) AS is_true_positive,
            (COALESCE(d.pv_detected_count, 0) > 0 AND COALESCE(o.osm_pv_label_count, 0) = 0) AS is_false_positive,
            (COALESCE(d.pv_detected_count, 0) = 0 AND COALESCE(o.osm_pv_label_count, 0) > 0) AS is_false_negative,
    """ if has_osm_labels else """
            FALSE AS has_osm_pv_label,
            0 AS osm_pv_label_count,
            FALSE AS is_true_positive,
            COALESCE(d.pv_detected_count, 0) > 0 AS is_false_positive,
            FALSE AS is_false_negative,
    """
    osm_join_sql = "LEFT JOIN osm_join AS o USING (building_id)" if has_osm_labels else ""
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {BUILDING_HITS_TABLE} AS
        WITH buildings AS (
            SELECT id AS building_id,
                   municipality_name AS municipio,
                   municipality_geoid,
                   geometry
            FROM pr_overture_buildings
            WHERE municipality_name IN ({munis_sql})
              AND geometry IS NOT NULL
        ),
        det_join AS (
            SELECT
                b.building_id,
                COUNT(DISTINCT d.detection_key) AS pv_detected_count,
                COUNT(DISTINCT d.source) AS pv_detection_source_count,
                COUNT(DISTINCT d.h3_cell_id) AS contributing_tile_count,
                CASE
                    WHEN COUNT(DISTINCT d.detection_key) = 0 THEN 0.0
                    ELSE ST_Area(ST_Intersection(ANY_VALUE(b.geometry), ST_Union_Agg(d.geometry)))
                END AS pv_detected_area_deg2
            FROM buildings AS b
            LEFT JOIN {DETECTION_TABLE} AS d
              ON ST_Intersects(b.geometry, d.geometry)
            GROUP BY b.building_id
        )
                {osm_join_cte}
        SELECT
            b.building_id,
            b.municipio,
            b.municipality_geoid,
            COALESCE(d.pv_detected_count, 0) > 0 AS has_pv_detected,
            COALESCE(d.pv_detected_count, 0) AS pv_detected_count,
            COALESCE(d.pv_detection_source_count, 0) AS pv_detection_source_count,
            COALESCE(d.contributing_tile_count, 0) AS contributing_tile_count,
            COALESCE(d.pv_detected_area_deg2, 0.0) AS pv_detected_area_deg2,
{osm_select_sql}
            '{model_family}' AS model_family,
            '{run_name}' AS model_run_name,
            '{model_arch}' AS model_arch,
            b.geometry
        FROM buildings AS b
        LEFT JOIN det_join AS d USING (building_id)
        {osm_join_sql};
        """
    )
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{BUILDING_HITS_TABLE}_geom ON {BUILDING_HITS_TABLE} USING RTREE (geometry);")
    return True


def build_municipio_summary_table(con: duckdb.DuckDBPyConnection, has_building_hits: bool) -> None:
    if has_building_hits:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {MUNICIPIO_SUMMARY_TABLE} AS
            WITH tile_rollup AS (
                SELECT
                    municipio,
                    COUNT(*) AS processed_tiles,
                    SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS tiles_with_detections,
                    SUM(detection_count) AS detection_count,
                    SUM(detected_area_m2) AS detected_area_m2,
                    SUM(COALESCE(building_count, 0)) AS tile_building_count
                FROM {TILE_SUMMARY_TABLE}
                GROUP BY municipio
            ),
            building_rollup AS (
                SELECT
                    municipio,
                    SUM(CAST(has_pv_detected AS INT)) AS detected_building_count,
                    SUM(pv_detected_count) AS detected_polygon_count_in_buildings,
                    SUM(pv_detected_area_deg2) AS detected_area_deg2_in_buildings,
                    SUM(CAST(has_osm_pv_label AS INT)) AS known_osm_pv_building_count,
                    SUM(CAST(is_true_positive AS INT)) AS true_positive_buildings,
                    SUM(CAST(is_false_positive AS INT)) AS false_positive_buildings,
                    SUM(CAST(is_false_negative AS INT)) AS false_negative_buildings
                FROM {BUILDING_HITS_TABLE}
                GROUP BY municipio
            )
            SELECT
                t.municipio,
                t.processed_tiles,
                t.tiles_with_detections,
                t.detection_count,
                t.detected_area_m2,
                t.tile_building_count,
                COALESCE(b.detected_building_count, 0) AS detected_building_count,
                COALESCE(b.detected_polygon_count_in_buildings, 0) AS detected_polygon_count_in_buildings,
                COALESCE(b.detected_area_deg2_in_buildings, 0.0) AS detected_area_deg2_in_buildings,
                COALESCE(b.known_osm_pv_building_count, 0) AS known_osm_pv_building_count,
                COALESCE(b.true_positive_buildings, 0) AS true_positive_buildings,
                COALESCE(b.false_positive_buildings, 0) AS false_positive_buildings,
                COALESCE(b.false_negative_buildings, 0) AS false_negative_buildings,
                CASE
                    WHEN COALESCE(b.known_osm_pv_building_count, 0) = 0 THEN 0.0
                    ELSE COALESCE(b.true_positive_buildings, 0)::DOUBLE / b.known_osm_pv_building_count
                END AS recall_vs_osm,
                CASE
                    WHEN COALESCE(b.detected_building_count, 0) = 0 THEN 0.0
                    ELSE COALESCE(b.true_positive_buildings, 0)::DOUBLE / b.detected_building_count
                END AS precision_lower_bound_vs_osm
            FROM tile_rollup AS t
            LEFT JOIN building_rollup AS b USING (municipio)
            ORDER BY t.municipio;
            """
        )
        return

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {MUNICIPIO_SUMMARY_TABLE} AS
        SELECT
            municipio,
            COUNT(*) AS processed_tiles,
            SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS tiles_with_detections,
            SUM(detection_count) AS detection_count,
            SUM(detected_area_m2) AS detected_area_m2,
            SUM(COALESCE(building_count, 0)) AS tile_building_count
        FROM {TILE_SUMMARY_TABLE}
        GROUP BY municipio
        ORDER BY municipio;
        """
    )


def _display_if_notebook(obj: object) -> None:
    if not running_in_notebook():
        return
    try:
        from IPython.display import display
    except ImportError:
        return
    display(obj)


def should_run_inference_phase() -> bool:
    return INFERENCE_PHASE in {"both", "inference"}


def should_run_reporting_phase() -> bool:
    return INFERENCE_PHASE in {"both", "reporting"}


def empty_detection_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")


def record_stage_duration(stage_timings: dict[str, float], stage_name: str, started_at: float) -> None:
    stage_timings[stage_name] = float(stage_timings.get(stage_name, 0.0) + (time.perf_counter() - started_at))


def summarize_directory_tree(root: Path) -> dict[str, int]:
    if not root.exists():
        return {"file_count": 0, "total_bytes": 0}
    file_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        file_count += 1
        try:
            total_bytes += path.stat().st_size
        except OSError:
            continue
    return {"file_count": file_count, "total_bytes": int(total_bytes)}


def reconstruct_selected_from_tile_summary(tile_summary_df: pd.DataFrame) -> pd.DataFrame:
    selected = tile_summary_df.copy()
    for column in (
        "source",
        "item_id",
        "municipio",
        "municipio_geoid",
        "h3_cell_id",
        "building_count",
        "municipality_building_count",
        "crosses_municipality_boundary",
        "tile_path",
        "local_asset_path",
    ):
        if column not in selected.columns:
            selected[column] = None
    selected["tile_abs_path"] = selected["tile_path"].map(
        lambda value: (PROJECT_ROOT / str(value)).resolve() if isinstance(value, str) and value else None
    )
    selected["local_asset_abs_path"] = selected["local_asset_path"].map(
        lambda value: (PROJECT_ROOT / str(value)).resolve() if isinstance(value, str) and value else None
    )
    return selected[
        [
            "source",
            "item_id",
            "municipio",
            "municipio_geoid",
            "h3_cell_id",
            "building_count",
            "municipality_building_count",
            "crosses_municipality_boundary",
            "tile_path",
            "tile_abs_path",
            "local_asset_path",
            "local_asset_abs_path",
        ]
    ].drop_duplicates().reset_index(drop=True)


def load_local_detections_from_tile_summary(tile_summary_df: pd.DataFrame, config: dict[str, object]) -> gpd.GeoDataFrame:
    if tile_summary_df.empty or "vector_path" not in tile_summary_df.columns:
        return empty_detection_gdf()

    frames: list[gpd.GeoDataFrame] = []
    available_rows = tile_summary_df[tile_summary_df["vector_path"].fillna("").astype(str).str.len() > 0].copy()
    for row in available_rows.itertuples(index=False):
        vector_path = PROJECT_ROOT / str(getattr(row, "vector_path", ""))
        if not vector_path.exists():
            continue
        try:
            gdf = gpd.read_file(vector_path)
        except Exception as exc:
            print(f"local reporting reload skipped for {vector_path.name}: {exc}")
            continue
        if gdf.empty:
            continue

        if "detection_key" not in gdf.columns:
            tile_key = getattr(row, "h3_cell_id", None) or Path(str(getattr(row, "tile_path", vector_path.stem))).stem
            gdf["detection_key"] = [f"{tile_key}_{index}" for index in range(len(gdf))]
        gdf["model_family"] = config["model_family"]
        gdf["model_run_name"] = config["run_name"]
        gdf["source"] = getattr(row, "source", None)
        gdf["item_id"] = getattr(row, "item_id", None)
        gdf["municipio"] = getattr(row, "municipio", None)
        gdf["municipio_geoid"] = getattr(row, "municipio_geoid", None)
        gdf["h3_cell_id"] = getattr(row, "h3_cell_id", None)
        gdf["building_count"] = getattr(row, "building_count", None)
        gdf["municipality_building_count"] = getattr(row, "municipality_building_count", None)
        gdf["crosses_municipality_boundary"] = getattr(row, "crosses_municipality_boundary", None)
        gdf["tile_path"] = getattr(row, "tile_path", None)
        gdf["local_asset_path"] = getattr(row, "local_asset_path", None)
        gdf["predicted_mask_path"] = getattr(row, "predicted_mask_path", None)
        gdf["probability_path"] = getattr(row, "probability_path", None)
        gdf["vector_path"] = getattr(row, "vector_path", None)
        gdf["inference_mode"] = getattr(row, "inference_mode", None)
        frames.append(gdf.to_crs("EPSG:4326"))

    if not frames:
        return empty_detection_gdf()
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs).to_crs("EPSG:4326")


def load_reporting_inputs_from_local_outputs(
    inference_root: Path,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame, str] | None:
    tile_summary_path = inference_root / "tile_summary.csv"
    if not tile_summary_path.exists():
        print(f"reporting phase skipped: no local tile summary found at {tile_summary_path}")
        return None
    tile_summary_df = pd.read_csv(tile_summary_path)
    if tile_summary_df.empty:
        print(f"reporting phase skipped: tile summary is empty at {tile_summary_path}")
        return None
    selected = reconstruct_selected_from_tile_summary(tile_summary_df)
    detections = load_local_detections_from_artifact(inference_root)
    reporting_source = "merged_detection_artifact"
    if detections is None:
        detections = load_local_detections_from_tile_summary(tile_summary_df, config)
        reporting_source = "legacy_vector_artifacts"
    print(
        f"reporting phase reloaded {len(tile_summary_df):,} tile rows and {len(detections):,} detections from local outputs"
    )
    return selected, tile_summary_df, detections, reporting_source


def load_osm_pv_labels() -> gpd.GeoDataFrame:
    db_path = resolve_db_path()
    try:
        con = duckdb.connect(str(db_path), read_only=True)
    except duckdb.IOException as exc:
        print(f"OSM PV label lookup skipped: could not open DuckDB read-only ({exc})")
        return gpd.GeoDataFrame(columns=["osm_label_id", "geometry"], geometry="geometry", crs="EPSG:4326")
    try:
        if not table_exists(con, "pr_osm_rooftop_pv_polygons"):
            print("OSM PV label metrics skipped: pr_osm_rooftop_pv_polygons is unavailable in the vector DB.")
            return gpd.GeoDataFrame(columns=["osm_label_id", "geometry"], geometry="geometry", crs="EPSG:4326")
        frame = con.execute(
            """
            SELECT
                CAST(ROW_NUMBER() OVER () AS VARCHAR) AS osm_label_id,
                ST_AsWKB(geometry) AS geometry_wkb
            FROM pr_osm_rooftop_pv_polygons
            WHERE geometry IS NOT NULL;
            """
        ).fetchdf()
    finally:
        con.close()

    if frame.empty:
        return gpd.GeoDataFrame(columns=["osm_label_id", "geometry"], geometry="geometry", crs="EPSG:4326")
    geometry = gpd.GeoSeries(frame["geometry_wkb"].map(lambda value: from_wkb(_to_bytes(value))), crs="EPSG:4326")
    return gpd.GeoDataFrame(frame.drop(columns=["geometry_wkb"]), geometry=geometry, crs="EPSG:4326")


def mark_building_intersections(buildings: gpd.GeoDataFrame, targets: gpd.GeoDataFrame, id_column: str) -> pd.Series:
    if buildings.empty or targets.empty:
        return pd.Series(0, index=buildings.index, dtype="int64")
    target_geoms = targets[["geometry"]].dropna().to_crs(buildings.crs)
    if target_geoms.empty:
        return pd.Series(0, index=buildings.index, dtype="int64")
    joined = gpd.sjoin(
        buildings[[id_column, "geometry"]],
        target_geoms,
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        return pd.Series(0, index=buildings.index, dtype="int64")
    counts = joined.groupby(id_column).size()
    return buildings[id_column].map(counts).fillna(0).astype("int64")


def build_building_level_metrics(
    *,
    building_footprints: gpd.GeoDataFrame,
    detections: gpd.GeoDataFrame,
    config: dict[str, object],
) -> gpd.GeoDataFrame:
    if building_footprints.empty:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")

    buildings = building_footprints.to_crs("EPSG:4326").copy()
    buildings["building_id"] = buildings["building_id"].astype(str)
    model_arch_key, model_arch = resolve_model_arch_info(
        config.get("model_family"),
        config.get("run_name"),
        metadata=config.get("metadata") if isinstance(config, dict) else None,
    )
    osm_labels = load_osm_pv_labels()
    if not osm_labels.empty:
        osm_labels = gpd.clip(osm_labels.to_crs(buildings.crs), buildings.total_bounds)

    buildings["model_detection_count"] = mark_building_intersections(buildings, detections, "building_id")
    buildings["osm_pv_label_count"] = mark_building_intersections(buildings, osm_labels, "building_id")
    buildings["has_model_detection"] = buildings["model_detection_count"] > 0
    buildings["has_osm_pv_label"] = buildings["osm_pv_label_count"] > 0
    buildings["is_true_positive"] = buildings["has_model_detection"] & buildings["has_osm_pv_label"]
    buildings["is_false_positive"] = buildings["has_model_detection"] & ~buildings["has_osm_pv_label"]
    buildings["is_false_negative"] = ~buildings["has_model_detection"] & buildings["has_osm_pv_label"]
    buildings["model_family"] = str(config["model_family"])
    buildings["model_run_name"] = str(config["run_name"])
    buildings["model_arch_key"] = model_arch_key
    buildings["model_arch"] = model_arch
    return buildings


def discover_persisted_detection_runs() -> list[dict[str, object]]:
    inference_runs_root = PROJECT_ROOT / "outputs" / "geoai_inference"
    if not inference_runs_root.exists():
        return []

    run_records: list[dict[str, object]] = []
    for inference_root in sorted(path for path in inference_runs_root.iterdir() if path.is_dir()):
        artifact_path = inference_root / "merged_detections.parquet"
        if not artifact_path.exists():
            continue

        summary_path = inference_root / "inference_commit_summary.json"
        summary: dict[str, object] = {}
        if summary_path.exists():
            try:
                summary = _load_json(summary_path)
            except Exception as exc:
                print(f"warning: could not read {summary_path.name}: {exc}")

        root_parts = inference_root.name.split("_", 1)
        root_family = root_parts[0].strip().lower()
        root_run_name = root_parts[1] if len(root_parts) == 2 else inference_root.name
        model_family = str(summary.get("model_family") or root_family or LEGACY_MODEL_FAMILY_BACKFILL).strip().lower()
        model_run_name = str(summary.get("model_run_name") or root_run_name or LEGACY_MODEL_RUN_NAME_BACKFILL).strip()

        metadata_name = "smp_metadata.json" if model_family == "smp" else "dinov3_metadata.json"
        model_root = PROJECT_ROOT / "outputs" / "models" / ("smp_grounded" if model_family == "smp" else "dinov3_grounded")
        metadata_path = model_root / model_run_name / metadata_name
        metadata: dict[str, object] | None = None
        if metadata_path.exists():
            try:
                metadata = _load_json(metadata_path)
            except Exception as exc:
                print(f"warning: could not read {metadata_path.name}: {exc}")

        model_arch_key, model_arch = resolve_model_arch_info(model_family, model_run_name, metadata=metadata)
        run_records.append(
            {
                "inference_root": inference_root,
                "artifact_path": artifact_path,
                "artifact_relpath": str(artifact_path.relative_to(PROJECT_ROOT)),
                "model_family": model_family,
                "model_run_name": model_run_name,
                "model_arch_key": model_arch_key,
                "model_arch": model_arch,
            }
        )
    return run_records


def load_persisted_detection_artifact(run_record: dict[str, object]) -> gpd.GeoDataFrame:
    artifact_path = Path(run_record["artifact_path"])
    try:
        gdf = gpd.read_parquet(artifact_path)
    except Exception as exc:
        print(f"warning: could not read detection artifact {artifact_path.name}: {exc}")
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")

    if gdf.empty:
        return gpd.GeoDataFrame(columns=list(gdf.columns), geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326").copy()
    fill_values = {
        "model_family": str(run_record["model_family"]),
        "model_run_name": str(run_record["model_run_name"]),
        "model_arch_key": str(run_record["model_arch_key"]),
        "model_arch": str(run_record["model_arch"]),
    }
    for column_name, fallback_value in fill_values.items():
        if column_name not in gdf.columns:
            gdf[column_name] = fallback_value
            continue
        missing_mask = gdf[column_name].isna() | gdf[column_name].fillna("").astype(str).str.strip().eq("")
        gdf.loc[missing_mask, column_name] = fallback_value

    if "detection_key" not in gdf.columns:
        gdf["detection_key"] = [f"artifact_{index}" for index in range(len(gdf))]
    gdf["catalog_detection_key"] = [
        f"{run_record['model_run_name']}::{key}"
        for key in gdf["detection_key"].fillna("").astype(str)
    ]
    gdf["source_detection_artifact"] = str(run_record["artifact_relpath"])
    return gdf


def build_multimodel_detection_catalog() -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    for run_record in discover_persisted_detection_runs():
        detections = load_persisted_detection_artifact(run_record)
        if detections.empty:
            continue
        frames.append(detections)

    if not frames:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:4326")


def build_multimodel_building_detection_catalog(multimodel_detections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if multimodel_detections.empty:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")

    base_buildings = load_overture_building_footprints(pd.DataFrame({"municipio": list(TARGET_MUNICIPALITIES)}))
    if base_buildings.empty:
        print("multi-model building catalog skipped: no Overture building footprints are available.")
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")

    base_buildings = base_buildings.to_crs("EPSG:4326").copy()
    base_buildings["building_id"] = base_buildings["building_id"].astype(str)
    detected_frames: list[gpd.GeoDataFrame] = []

    group_columns = ["model_family", "model_run_name", "model_arch_key", "model_arch", "source_detection_artifact"]
    for group_values, detection_subset in multimodel_detections.groupby(group_columns, sort=False):
        model_family, model_run_name, model_arch_key, model_arch, source_detection_artifact = group_values
        detection_subset = detection_subset.to_crs(base_buildings.crs)
        candidate_buildings = base_buildings[base_buildings["municipio"].isin(detection_subset["municipio"].dropna().astype(str).unique())].copy()
        h3_ids = {
            str(value)
            for value in detection_subset.get("h3_cell_id", pd.Series(dtype=str)).dropna().astype(str).tolist()
            if str(value).strip()
        }
        if h3_ids:
            candidate_buildings = candidate_buildings[candidate_buildings["h3_cell_id"].fillna("").astype(str).isin(h3_ids)].copy()
        if candidate_buildings.empty:
            continue

        joined = gpd.sjoin(
            candidate_buildings[["building_id", "municipio", "h3_cell_id", "geometry"]],
            detection_subset[["geometry"]],
            how="inner",
            predicate="intersects",
        )
        if joined.empty:
            continue

        counts = joined.groupby("building_id").size()
        detected_buildings = candidate_buildings[candidate_buildings["building_id"].isin(counts.index.astype(str))].copy()
        detected_buildings["model_detection_count"] = detected_buildings["building_id"].map(counts).fillna(0).astype("int64")
        detected_buildings["model_family"] = str(model_family)
        detected_buildings["model_run_name"] = str(model_run_name)
        detected_buildings["model_arch_key"] = str(model_arch_key)
        detected_buildings["model_arch"] = str(model_arch)
        detected_buildings["source_detection_artifact"] = str(source_detection_artifact)
        detected_frames.append(detected_buildings)

    if not detected_frames:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=base_buildings.crs), crs=base_buildings.crs)

    stacked = gpd.GeoDataFrame(pd.concat(detected_frames, ignore_index=True), geometry="geometry", crs=base_buildings.crs)
    geometry_lookup = stacked[["building_id", "geometry"]].drop_duplicates(subset=["building_id"]).copy()
    rolled = (
        stacked.drop(columns=["geometry"])
        .groupby(["building_id", "municipio"], as_index=False)
        .agg(
            h3_cell_id=("h3_cell_id", "first"),
            model_arch_count=("model_arch", lambda values: int(pd.Series(values).dropna().nunique())),
            model_arch_list=("model_arch", lambda values: ", ".join(sorted(pd.Series(values).dropna().unique().tolist()))),
            model_run_count=("model_run_name", lambda values: int(pd.Series(values).dropna().nunique())),
            model_run_list=("model_run_name", lambda values: ", ".join(sorted(pd.Series(values).dropna().unique().tolist()))),
            model_family_list=("model_family", lambda values: ", ".join(sorted(pd.Series(values).dropna().unique().tolist()))),
            total_detection_polygons=("model_detection_count", "sum"),
        )
    )
    rolled["shared_across_models"] = rolled["model_arch_count"] > 1
    rolled["model_arch_display"] = rolled.apply(
        lambda row: "Shared" if bool(row["shared_across_models"]) else str(row["model_arch_list"]),
        axis=1,
    )
    rolled = rolled.merge(geometry_lookup, on="building_id", how="left")
    return gpd.GeoDataFrame(rolled, geometry="geometry", crs=base_buildings.crs)


def write_multimodel_detection_table(con: duckdb.DuckDBPyConnection, multimodel_detections: gpd.GeoDataFrame) -> None:
    if multimodel_detections.empty:
        return
    staged = pd.DataFrame(multimodel_detections.drop(columns=["geometry"]))
    staged["geometry_wkb"] = multimodel_detections.geometry.to_wkb()
    con.register("staged_multimodel_detections", staged)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {ALL_MODEL_DETECTION_TABLE} AS
        SELECT * EXCLUDE (geometry_wkb),
               ST_GeomFromWKB(geometry_wkb) AS geometry
        FROM staged_multimodel_detections;
        """
    )
    con.unregister("staged_multimodel_detections")
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{ALL_MODEL_DETECTION_TABLE}_geom ON {ALL_MODEL_DETECTION_TABLE} USING RTREE (geometry);")


def write_multimodel_building_catalog_table(con: duckdb.DuckDBPyConnection, building_catalog: gpd.GeoDataFrame) -> None:
    if building_catalog.empty:
        return
    staged = pd.DataFrame(building_catalog.drop(columns=["geometry"]))
    staged["geometry_wkb"] = building_catalog.geometry.to_wkb()
    con.register("staged_multimodel_buildings", staged)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {ALL_MODEL_BUILDING_TABLE} AS
        SELECT * EXCLUDE (geometry_wkb),
               ST_GeomFromWKB(geometry_wkb) AS geometry
        FROM staged_multimodel_buildings;
        """
    )
    con.unregister("staged_multimodel_buildings")
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{ALL_MODEL_BUILDING_TABLE}_geom ON {ALL_MODEL_BUILDING_TABLE} USING RTREE (geometry);")


def load_target_municipality_boundaries() -> gpd.GeoDataFrame:
    db_path = resolve_db_path()
    try:
        con = duckdb.connect(str(db_path), read_only=True)
    except duckdb.ConnectionException:
        con = duckdb.connect(str(db_path))
    except duckdb.IOException as exc:
        print(f"municipality boundary lookup skipped: could not open DuckDB read-only ({exc})")
        return gpd.GeoDataFrame(columns=["municipio", "geometry"], geometry="geometry", crs="EPSG:4326")
    try:
        if not table_exists(con, "pr_census_counties"):
            return gpd.GeoDataFrame(columns=["municipio", "geometry"], geometry="geometry", crs="EPSG:4326")
        placeholders = ", ".join("?" for _ in TARGET_MUNICIPALITIES)
        frame = con.execute(
            f"""
            SELECT CAST(NAME AS VARCHAR) AS municipio,
                   ST_AsWKB(geometry) AS geometry_wkb
            FROM pr_census_counties
            WHERE NAME IN ({placeholders});
            """,
            list(TARGET_MUNICIPALITIES),
        ).fetchdf()
    finally:
        con.close()

    if frame.empty:
        return gpd.GeoDataFrame(columns=["municipio", "geometry"], geometry="geometry", crs="EPSG:4326")
    geometry = gpd.GeoSeries(frame["geometry_wkb"].map(lambda value: from_wkb(_to_bytes(value))), crs="EPSG:4326")
    return gpd.GeoDataFrame(frame.drop(columns=["geometry_wkb"]), geometry=geometry, crs="EPSG:4326")


def write_multimodel_building_summary(building_catalog: gpd.GeoDataFrame, output_path: Path = MULTIMODEL_BUILDING_SUMMARY_OUTPUT_PATH) -> pd.DataFrame:
    if building_catalog.empty:
        return pd.DataFrame(columns=["municipio", "model_arch_display", "detected_buildings"])
    summary = (
        building_catalog.groupby(["municipio", "model_arch_display"], as_index=False)
        .agg(
            detected_buildings=("building_id", "nunique"),
            total_detection_polygons=("total_detection_polygons", "sum"),
        )
        .sort_values(["municipio", "detected_buildings"], ascending=[True, False])
        .reset_index(drop=True)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"wrote {output_path}")
    return summary


def plot_multimodel_building_catalog_map(
    building_catalog: gpd.GeoDataFrame,
    *,
    output_path: Path = MULTIMODEL_BUILDING_MAP_OUTPUT_PATH,
) -> None:
    if building_catalog.empty:
        print("multi-model building map skipped: no detected buildings were cataloged.")
        return

    boundaries = load_target_municipality_boundaries()
    plot_gdf = building_catalog.to_crs("EPSG:32619").copy()
    plot_gdf["geometry"] = plot_gdf.geometry.centroid
    plot_gdf = plot_gdf.to_crs("EPSG:4326")

    fig, axes = plt.subplots(1, len(TARGET_MUNICIPALITIES), figsize=(14, 7), constrained_layout=True)
    if len(TARGET_MUNICIPALITIES) == 1:
        axes = [axes]

    for ax, municipio in zip(axes, TARGET_MUNICIPALITIES):
        subset = plot_gdf[plot_gdf["municipio"] == municipio].copy()
        if not boundaries.empty:
            boundary = boundaries[boundaries["municipio"] == municipio]
            if not boundary.empty:
                boundary.boundary.plot(ax=ax, color="#111111", linewidth=0.8)
        if subset.empty:
            ax.set_title(f"{municipio}: no detected buildings")
            ax.set_axis_off()
            continue

        for arch_name in MODEL_ARCH_ORDER:
            arch_subset = subset[subset["model_arch_display"] == arch_name]
            if arch_subset.empty:
                continue
            arch_subset.plot(
                ax=ax,
                color=MODEL_ARCH_COLORS.get(arch_name, "#666666"),
                markersize=6,
                alpha=0.75,
                label=arch_name,
            )
        ax.set_title(f"{municipio} — PV buildings by model architecture")
        ax.set_axis_off()

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def summarize_building_metrics(building_metrics: gpd.GeoDataFrame) -> pd.DataFrame:
    if building_metrics.empty:
        return pd.DataFrame(
            [
                {
                    "building_count": 0,
                    "known_osm_pv_buildings": 0,
                    "detected_pv_buildings": 0,
                    "true_positive_buildings": 0,
                    "false_positive_buildings": 0,
                    "false_negative_buildings": 0,
                    "recall_vs_osm": 0.0,
                    "precision_lower_bound_vs_osm": 0.0,
                    "detected_building_rate": 0.0,
                }
            ]
        )
    known = int(building_metrics["has_osm_pv_label"].sum())
    detected = int(building_metrics["has_model_detection"].sum())
    true_positive = int(building_metrics["is_true_positive"].sum())
    return pd.DataFrame(
        [
            {
                "building_count": int(len(building_metrics)),
                "known_osm_pv_buildings": known,
                "detected_pv_buildings": detected,
                "true_positive_buildings": true_positive,
                "false_positive_buildings": int(building_metrics["is_false_positive"].sum()),
                "false_negative_buildings": int(building_metrics["is_false_negative"].sum()),
                "recall_vs_osm": true_positive / known if known else 0.0,
                "precision_lower_bound_vs_osm": true_positive / detected if detected else 0.0,
                "detected_building_rate": detected / len(building_metrics) if len(building_metrics) else 0.0,
            }
        ]
    )


def load_aggregation_units(level: str, selected: pd.DataFrame) -> gpd.GeoDataFrame:
    table_by_level = {
        "h3": "pr_solar_tile_manifest",
        "tract": "pr_census_tracts",
        "block_group": "pr_census_block_groups",
    }
    table_name = table_by_level.get(level)
    if table_name is None:
        raise ValueError(f"unsupported aggregation level: {level}")

    db_path = resolve_db_path()
    try:
        con = duckdb.connect(str(db_path), read_only=True)
    except duckdb.IOException as exc:
        print(f"aggregation skipped for {level}: could not open DuckDB read-only ({exc})")
        return gpd.GeoDataFrame(columns=["unit_id", "unit_name", "geometry"], geometry="geometry", crs="EPSG:4326")
    try:
        if not table_exists(con, table_name):
            print(f"aggregation skipped for {level}: {table_name} is unavailable in the vector DB.")
            return gpd.GeoDataFrame(columns=["unit_id", "unit_name", "geometry"], geometry="geometry", crs="EPSG:4326")
        if level == "h3":
            h3_ids = sorted({str(value) for value in selected.get("h3_cell_id", pd.Series(dtype=str)).dropna().astype(str).tolist()})
            if not h3_ids:
                return gpd.GeoDataFrame(columns=["unit_id", "unit_name", "geometry"], geometry="geometry", crs="EPSG:4326")
            placeholders = ", ".join("?" for _ in h3_ids)
            frame = con.execute(
                f"""
                SELECT
                    CAST(h3_cell_id AS VARCHAR) AS unit_id,
                    CAST(h3_cell_id AS VARCHAR) AS unit_name,
                    ST_AsWKB(geometry) AS geometry_wkb
                FROM {table_name}
                WHERE h3_cell_id IN ({placeholders})
                  AND geometry IS NOT NULL;
                """,
                h3_ids,
            ).fetchdf()
        else:
            frame = con.execute(
                f"""
                SELECT
                    CAST(GEOID AS VARCHAR) AS unit_id,
                    CAST(NAME AS VARCHAR) AS unit_name,
                    ST_AsWKB(geometry) AS geometry_wkb
                FROM {table_name}
                WHERE geometry IS NOT NULL;
                """
            ).fetchdf()
    finally:
        con.close()

    if frame.empty:
        return gpd.GeoDataFrame(columns=["unit_id", "unit_name", "geometry"], geometry="geometry", crs="EPSG:4326")
    geometry = gpd.GeoSeries(frame["geometry_wkb"].map(lambda value: from_wkb(_to_bytes(value))), crs="EPSG:4326")
    return gpd.GeoDataFrame(frame.drop(columns=["geometry_wkb"]), geometry=geometry, crs="EPSG:4326")


def aggregate_building_metrics_by_unit(
    building_metrics: gpd.GeoDataFrame,
    units: gpd.GeoDataFrame,
    *,
    level: str,
) -> gpd.GeoDataFrame:
    if building_metrics.empty or units.empty:
        return gpd.GeoDataFrame(columns=["unit_id", "unit_name", "geometry"], geometry="geometry", crs="EPSG:4326")

    metric_points = building_metrics.to_crs("EPSG:4326").copy()
    metric_points["geometry"] = metric_points.geometry.representative_point()
    joined = gpd.sjoin(
        metric_points[
            [
                "building_id",
                "has_model_detection",
                "has_osm_pv_label",
                "is_true_positive",
                "is_false_positive",
                "is_false_negative",
                "geometry",
            ]
        ],
        units.to_crs(metric_points.crs)[["unit_id", "unit_name", "geometry"]],
        how="inner",
        predicate="within",
    )
    if joined.empty:
        return units.iloc[0:0].copy()

    grouped = joined.groupby(["unit_id", "unit_name"], dropna=False).agg(
        building_count=("building_id", "nunique"),
        detected_pv_buildings=("has_model_detection", "sum"),
        known_osm_pv_buildings=("has_osm_pv_label", "sum"),
        true_positive_buildings=("is_true_positive", "sum"),
        false_positive_buildings=("is_false_positive", "sum"),
        false_negative_buildings=("is_false_negative", "sum"),
    ).reset_index()
    grouped["detected_building_rate"] = grouped["detected_pv_buildings"] / grouped["building_count"].replace(0, np.nan)
    grouped["recall_vs_osm"] = grouped["true_positive_buildings"] / grouped["known_osm_pv_buildings"].replace(0, np.nan)
    grouped["precision_lower_bound_vs_osm"] = grouped["true_positive_buildings"] / grouped["detected_pv_buildings"].replace(0, np.nan)
    grouped = grouped.fillna(0)
    grouped["aggregation_level"] = level
    return units.merge(grouped, on=["unit_id", "unit_name"], how="inner")


def save_aggregate_map(gdf: gpd.GeoDataFrame, *, column: str, title: str, output_path: Path) -> None:
    if gdf.empty or column not in gdf.columns:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(
        ax=ax,
        column=column,
        cmap="viridis",
        legend=True,
        edgecolor="#1f2937",
        linewidth=0.2,
        missing_kwds={"color": "#f3f4f6"},
    )
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def display_h3_lonboard_heatmap(h3_metrics: gpd.GeoDataFrame) -> None:
    if not running_in_notebook() or h3_metrics.empty:
        return
    try:
        from lonboard import HeatmapLayer, Map
    except ImportError:
        print("lonboard HeatmapLayer is unavailable; static maps were still written.")
        return

    heatmap_points = h3_metrics.to_crs("EPSG:3857").copy()
    heatmap_points["geometry"] = heatmap_points.geometry.centroid
    heatmap_points = heatmap_points.to_crs("EPSG:4326")
    heatmap_points["detected_per_100_buildings"] = heatmap_points["detected_building_rate"] * 100.0
    heatmap_weights = (
        heatmap_points["detected_per_100_buildings"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    layer = HeatmapLayer.from_geopandas(
        heatmap_points[["geometry"]],
        get_weight=heatmap_weights,
        radius_pixels=45,
        intensity=1.3,
        opacity=0.65,
        aggregation="SUM",
    )
    _display_if_notebook(Map(layers=[layer], height=650))


def write_inference_schema_reference(inference_root: Path) -> Path:
    rows = [
        {"table": DETECTION_TABLE, "field": "detection_key", "description": "Unique detection polygon key within the model run."},
        {"table": DETECTION_TABLE, "field": "model_family", "description": "Inference model family: smp or dinov3."},
        {"table": DETECTION_TABLE, "field": "model_run_name", "description": "Model run directory used for inference."},
        {"table": DETECTION_TABLE, "field": "model_arch", "description": "Short public-facing architecture label used in legends, such as DINOv3, PAN, or Segformer."},
        {"table": DETECTION_TABLE, "field": "h3_cell_id", "description": "Source tile H3 cell identifier."},
        {"table": DETECTION_TABLE, "field": "predicted_mask_path", "description": "Relative path to the per-tile prediction raster when mask artifacts are preserved."},
        {"table": DETECTION_TABLE, "field": "probability_path", "description": "Relative path to the probability raster when explicitly emitted; compact runs leave this null."},
        {"table": DETECTION_TABLE, "field": "geometry", "description": "Detected PV polygon geometry in EPSG:4326."},
        {"table": ALL_MODEL_DETECTION_TABLE, "field": "catalog_detection_key", "description": "Cross-run unique detection key prefixed by the model run name."},
        {"table": ALL_MODEL_DETECTION_TABLE, "field": "model_arch", "description": "Short architecture label attached to each persisted detection artifact."},
        {"table": TILE_SUMMARY_TABLE, "field": "status", "description": "Per-tile inference status: ok, no_positive_mask, outside_buildings, filtered_empty, or mask_missing."},
        {"table": TILE_SUMMARY_TABLE, "field": "building_clip_applied", "description": "Whether the inference tile was masked to building footprints plus buffer before or after prediction."},
        {"table": TILE_SUMMARY_TABLE, "field": "building_clip_buffer_m", "description": "Metric buffer applied around Overture building footprints before raster clipping."},
        {"table": TILE_SUMMARY_TABLE, "field": "input_clip_applied", "description": "Whether the raster fed into inference was masked to the building footprint buffer."},
        {"table": BUILDING_HITS_TABLE, "field": "has_pv_detected", "description": "Whether any model detection intersects the Overture building."},
        {"table": BUILDING_HITS_TABLE, "field": "pv_detected_count", "description": "Number of model detection polygons intersecting the building."},
        {"table": BUILDING_HITS_TABLE, "field": "has_osm_pv_label", "description": "Whether any known OSM rooftop PV label intersects the Overture building."},
        {"table": BUILDING_HITS_TABLE, "field": "osm_pv_label_count", "description": "Number of known OSM rooftop PV labels intersecting the building."},
        {"table": BUILDING_HITS_TABLE, "field": "is_true_positive", "description": "Building has both a model detection and an OSM PV label."},
        {"table": BUILDING_HITS_TABLE, "field": "is_false_positive", "description": "Building has a model detection but no known OSM PV label; this is a conservative false-positive proxy."},
        {"table": BUILDING_HITS_TABLE, "field": "is_false_negative", "description": "Building has a known OSM PV label but no model detection."},
        {"table": BUILDING_HITS_TABLE, "field": "model_arch", "description": "Short architecture label for the current-run building-hit table."},
        {"table": ALL_MODEL_BUILDING_TABLE, "field": "model_arch_display", "description": "Legend-ready architecture label for each detected building; buildings seen by multiple models are labeled Shared."},
        {"table": ALL_MODEL_BUILDING_TABLE, "field": "shared_across_models", "description": "Whether a detected building was identified by more than one model architecture."},
        {"table": MUNICIPIO_SUMMARY_TABLE, "field": "detected_building_count", "description": "Detected buildings summarized by municipality when building-hit tables are available."},
        {"table": MUNICIPIO_SUMMARY_TABLE, "field": "recall_vs_osm", "description": "True-positive detected buildings divided by buildings with known OSM PV labels."},
        {"table": MUNICIPIO_SUMMARY_TABLE, "field": "precision_lower_bound_vs_osm", "description": "True-positive detected buildings divided by detected buildings; conservative because OSM labels are incomplete."},
    ]
    schema_df = pd.DataFrame(rows)
    inference_root.mkdir(parents=True, exist_ok=True)
    MAPS_ROOT.mkdir(parents=True, exist_ok=True)
    schema_path = inference_root / "inference_db_schema_reference.csv"
    schema_df.to_csv(schema_path, index=False)
    schema_df.to_csv(MAPS_ROOT / "inference_db_schema_reference.csv", index=False)
    _display_if_notebook(schema_df)
    return schema_path


def write_presentation_outputs(
    *,
    inference_root: Path,
    selected: pd.DataFrame,
    building_footprints: gpd.GeoDataFrame,
    detections: gpd.GeoDataFrame,
    config: dict[str, object],
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    MAPS_ROOT.mkdir(parents=True, exist_ok=True)
    building_metrics = build_building_level_metrics(
        building_footprints=building_footprints,
        detections=detections,
        config=config,
    )
    if building_metrics.empty:
        print("presentation metrics skipped: no building footprints are available for the selected inference tiles.")
        return outputs

    metrics_path = inference_root / "building_level_inference_metrics.geojson"
    metrics_csv_path = inference_root / "building_level_inference_metrics.csv"
    summary_path = inference_root / "building_level_inference_summary.csv"
    building_metrics.to_file(metrics_path, driver="GeoJSON")
    pd.DataFrame(building_metrics.drop(columns=["geometry"])).to_csv(metrics_csv_path, index=False)
    summary_df = summarize_building_metrics(building_metrics)
    summary_df.to_csv(summary_path, index=False)
    _display_if_notebook(summary_df)
    outputs.update(
        {
            "building_metrics_geojson": str(metrics_path.relative_to(PROJECT_ROOT)),
            "building_metrics_csv": str(metrics_csv_path.relative_to(PROJECT_ROOT)),
            "building_summary_csv": str(summary_path.relative_to(PROJECT_ROOT)),
        }
    )

    for level in PRESENTATION_AGGREGATIONS:
        units = load_aggregation_units(level, selected)
        aggregate = aggregate_building_metrics_by_unit(building_metrics, units, level=level)
        if aggregate.empty:
            continue
        aggregate_base = f"{config['model_family']}_{config['run_name']}_{level}_inference_metrics"
        aggregate_geojson = MAPS_ROOT / f"{aggregate_base}.geojson"
        aggregate_csv = MAPS_ROOT / f"{aggregate_base}.csv"
        aggregate.to_file(aggregate_geojson, driver="GeoJSON")
        pd.DataFrame(aggregate.drop(columns=["geometry"])).to_csv(aggregate_csv, index=False)
        save_aggregate_map(
            aggregate,
            column="detected_pv_buildings",
            title=f"Detected PV buildings by {level.replace('_', ' ')}",
            output_path=MAPS_ROOT / f"{aggregate_base}_detected_buildings.png",
        )
        save_aggregate_map(
            aggregate,
            column="detected_building_rate",
            title=f"Detected PV building rate by {level.replace('_', ' ')}",
            output_path=MAPS_ROOT / f"{aggregate_base}_detected_rate.png",
        )
        save_aggregate_map(
            aggregate,
            column="recall_vs_osm",
            title=f"Model recall vs OSM PV labels by {level.replace('_', ' ')}",
            output_path=MAPS_ROOT / f"{aggregate_base}_recall_vs_osm.png",
        )
        outputs[f"{level}_metrics_geojson"] = str(aggregate_geojson.relative_to(PROJECT_ROOT))
        outputs[f"{level}_metrics_csv"] = str(aggregate_csv.relative_to(PROJECT_ROOT))
        if level == "h3":
            display_h3_lonboard_heatmap(aggregate)

    return outputs


def should_commit_to_db() -> bool:
    if INFERENCE_MODE == "sample":
        return False
    return DB_COMMIT


def write_local_run_summary(
    inference_root: Path,
    *,
    config: dict[str, object],
    selected: pd.DataFrame,
    tile_summary_df: pd.DataFrame,
    detections: gpd.GeoDataFrame,
    db_committed: bool,
    presentation_outputs: dict[str, str] | None = None,
    stage_timings: dict[str, float] | None = None,
    output_metrics: dict[str, dict[str, int]] | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> Path:
    inference_root.mkdir(parents=True, exist_ok=True)
    tile_summary_path = inference_root / "tile_summary.csv"
    tile_summary_df.to_csv(tile_summary_path, index=False)

    cleaned_stage_timings = {
        key: round(float(value), 6)
        for key, value in (stage_timings or {}).items()
    }
    processed_tile_count = int(len(tile_summary_df))
    stage_seconds_per_processed_tile = {
        key: round(value / processed_tile_count, 6)
        for key, value in cleaned_stage_timings.items()
        if processed_tile_count > 0
    }

    summary = {
        "model_family": str(config["model_family"]),
        "model_run_name": str(config["run_name"]),
        "inference_mode": INFERENCE_MODE,
        "inference_phase": INFERENCE_PHASE,
        "inference_artifact_mode": INFERENCE_ARTIFACT_MODE,
        "show_window_progress": SHOW_WINDOW_PROGRESS,
        "clip_to_buildings": CLIP_TO_BUILDINGS,
        "clip_input_rasters_to_buildings": CLIP_INPUT_RASTERS_TO_BUILDINGS,
        "clip_prediction_mask_to_buildings": CLIP_PREDICTION_MASK_TO_BUILDINGS,
        "write_input_clip_preview_artifacts": WRITE_INPUT_CLIP_PREVIEW_ARTIFACTS,
        "preserve_tile_mask_artifacts": PRESERVE_TILE_MASK_ARTIFACTS,
        "preserve_clipped_input_artifacts": PRESERVE_CLIPPED_INPUT_ARTIFACTS,
        "write_probability_rasters": WRITE_PROBABILITY_RASTERS,
        "db_commit_requested": DB_COMMIT,
        "db_committed": db_committed,
        "selected_tile_count": int(len(selected)),
        "processed_tile_count": processed_tile_count,
        "detection_count": int(len(detections)),
        "tile_status_counts": tile_summary_df["status"].value_counts().to_dict() if not tile_summary_df.empty and "status" in tile_summary_df.columns else {},
        "tile_summary_csv": str(tile_summary_path.relative_to(PROJECT_ROOT)),
        "duckdb_tables": [
            DETECTION_TABLE,
            ALL_MODEL_DETECTION_TABLE,
            TILE_SUMMARY_TABLE,
            BUILDING_HITS_TABLE,
            ALL_MODEL_BUILDING_TABLE,
            MUNICIPIO_SUMMARY_TABLE,
        ],
        "presentation_outputs": presentation_outputs or {},
        "stage_timings_seconds": cleaned_stage_timings,
        "stage_seconds_per_processed_tile": stage_seconds_per_processed_tile,
        "output_metrics": output_metrics or {},
    }
    if extra_metadata:
        summary.update(extra_metadata)

    if INFERENCE_PHASE == "reporting":
        summary_name = "reporting_summary.json"
    elif INFERENCE_PHASE == "inference":
        summary_name = "inference_commit_summary.json" if db_committed else "inference_summary.json"
    else:
        summary_name = "db_commit_summary.json" if db_committed else "dry_run_summary.json"
    summary_path = inference_root / summary_name
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path

# %%
def run_inference_workflow() -> Path | None:
    workflow_started = time.perf_counter()
    stage_timings: dict[str, float] = defaultdict(float)
    apply_inference_widget_overrides()
    config = resolve_model_configuration()
    inference_root = Path(config["inference_root"])
    inference_root.mkdir(parents=True, exist_ok=True)
    config["model_family"] = INFERENCE_MODEL_FAMILY

    print(
        f"model_family={INFERENCE_MODEL_FAMILY} | run={config['run_name']} | mode={INFERENCE_MODE} | "
        f"phase={INFERENCE_PHASE} | artifacts={INFERENCE_ARTIFACT_MODE}"
    )
    runtime_device = resolve_runtime_device()
    amp_dtype = configure_torch_inference_performance(runtime_device)
    selected: pd.DataFrame
    merged_detections: gpd.GeoDataFrame
    tile_summary_df: pd.DataFrame
    reporting_source = "inference_loop"
    extra_metadata: dict[str, object] = {
        "runtime_device": describe_runtime_device(runtime_device),
        "runtime_mixed_precision": format_amp_dtype(amp_dtype),
        "inference_gpu_tuning": {
            "float32_matmul_precision": INFERENCE_FLOAT32_MATMUL_PRECISION,
            "tf32": ENABLE_INFERENCE_TF32,
            "cudnn_benchmark": ENABLE_INFERENCE_CUDNN_BENCHMARK,
            "empty_cache_before_run": CUDA_EMPTY_CACHE_BEFORE_INFERENCE,
        },
    }

    if OVERWRITE_INFERENCE_ARTIFACTS:
        stage_started = time.perf_counter()
        overwrite_cleanup = cleanup_overwrite_outputs(
            inference_root,
            config,
            include_tile_artifacts=should_run_inference_phase(),
            preserve_local_reporting_inputs=not should_run_inference_phase(),
        )
        record_stage_duration(stage_timings, "overwrite_cleanup_seconds", stage_started)
        extra_metadata["overwrite_cleanup"] = overwrite_cleanup
        if overwrite_cleanup["deleted_files"] or overwrite_cleanup["deleted_directories"]:
            print(
                "overwrite cleanup: "
                f"files={overwrite_cleanup['deleted_files']:,} | "
                f"directories={overwrite_cleanup['deleted_directories']:,} | "
                f"bytes={overwrite_cleanup['deleted_bytes']:,}"
            )
        elif should_run_inference_phase():
            print("overwrite cleanup: no existing run artifacts matched the cleanup rules.")
        else:
            print("overwrite cleanup: no stale reporting artifacts matched; preserved local inference inputs.")

    if should_run_inference_phase():
        import geoai

        stage_started = time.perf_counter()
        candidates = load_naip_tile_candidates()
        record_stage_duration(stage_timings, "candidate_load_seconds", stage_started)
        if candidates.empty:
            print("no NAIP STAC tiles available for inference.")
            return None

        stage_started = time.perf_counter()
        selected = select_inference_rows(candidates)
        record_stage_duration(stage_timings, "tile_selection_seconds", stage_started)
        print(f"selected_tiles={len(selected):,}")
        if INFERENCE_MODE == "sample":
            print("sample mode forces local-only execution; DuckDB writes are disabled.")
        elif not DB_COMMIT:
            print("DuckDB commit disabled; this full inference run will stay local-only unless GEOAI_DB_COMMIT=1 is set.")
        if selected.empty:
            print("tile selection returned zero rows.")
            return None

        selected = prepare_selected_for_inference_batches(selected)
        grouped_asset_count = int(selected["inference_group_key"].nunique()) if not selected.empty else 0
        batching_mode = "enabled" if ENABLE_SOURCE_GROUP_TILE_BATCHING else "disabled"
        print(
            "source-group tile batching: "
            f"{batching_mode} | groups={grouped_asset_count:,} | "
            f"max_tiles_per_batch={INFERENCE_TILE_BATCH_SIZE}"
        )
        extra_metadata["inference_batching"] = {
            "source_group_tile_batching": bool(ENABLE_SOURCE_GROUP_TILE_BATCHING),
            "max_tiles_per_batch": int(INFERENCE_TILE_BATCH_SIZE),
            "group_count": grouped_asset_count,
        }

        stage_started = time.perf_counter()
        needs_building_footprints = CLIP_TO_BUILDINGS or WRITE_PRESENTATION_OUTPUTS
        building_footprints = load_overture_building_footprints(selected) if needs_building_footprints else gpd.GeoDataFrame()
        record_stage_duration(stage_timings, "building_footprint_load_seconds", stage_started)
        if CLIP_TO_BUILDINGS:
            print(f"building clipping enabled | footprints loaded={len(building_footprints):,}")
            if building_footprints.empty:
                print("building clipping requested but no building footprints were found in the vector DB; leaving inference tiles unclipped.")
        elif WRITE_PRESENTATION_OUTPUTS and not building_footprints.empty:
            print(f"building footprints loaded for reporting | footprints loaded={len(building_footprints):,}")
        elif WRITE_PRESENTATION_OUTPUTS:
            print("building footprints unavailable; presentation metrics may be skipped.")

        sample_tile_path = Path(selected.iloc[0].tile_abs_path)
        with rasterio.open(sample_tile_path) as sample_tile:
            sample_tile_width = sample_tile.width
            sample_tile_height = sample_tile.height
        sample_stride = WINDOW_SIZE - OVERLAP
        sample_window_rows = max(1, int(np.ceil((sample_tile_height - OVERLAP) / sample_stride)))
        sample_window_cols = max(1, int(np.ceil((sample_tile_width - OVERLAP) / sample_stride)))
        print(
            "sample inference tiling: "
            f"{sample_tile_width}x{sample_tile_height}px | "
            f"windows_per_tile={sample_window_rows * sample_window_cols} | "
            f"per-tile window_batch_size={INFERENCE_BATCH_SIZE} | "
            f"max_tiles_per_batch={INFERENCE_TILE_BATCH_SIZE}"
        )

        smp_runtime = None
        dinov3_runtime = None
        if INFERENCE_MODEL_FAMILY == "smp":
            stage_started = time.perf_counter()
            smp_runtime = build_smp_inference_runtime(
                Path(config["model_path"]),
                config["metadata"],
                device=runtime_device,
                amp_dtype=amp_dtype,
            )
            record_stage_duration(stage_timings, "runtime_build_seconds", stage_started)
            print("SMP inference runtime loaded once for the full tile loop; model weights will not be reloaded per tile.")
        elif INFERENCE_MODEL_FAMILY == "dinov3":
            stage_started = time.perf_counter()
            dinov3_runtime = build_dinov3_inference_runtime(
                Path(config["model_path"]),
                config["metadata"],
                device=runtime_device,
                amp_dtype=amp_dtype,
            )
            record_stage_duration(stage_timings, "runtime_build_seconds", stage_started)

        all_detections: list[gpd.GeoDataFrame] = []
        tile_records: list[dict[str, object]] = []
        status_counts: Counter[str] = Counter()
        progress_counts = {
            "mask_cache_hits": 0,
            "tiles_inferred": 0,
            "total_detections": 0,
            "input_clipped_tiles": 0,
        }
        tile_loop_started = time.perf_counter()
        tile_progress = tqdm(
            total=len(selected),
            desc=f"Tiles | {INFERENCE_MODEL_FAMILY}:{config['run_name']}",
            unit="tile",
            leave=True,
            position=0,
        )

        def finalize_tile_progress(status: str, detection_count: int) -> None:
            status_counts[status] += 1
            progress_counts["total_detections"] += int(detection_count)
            tile_progress.update(1)
            tile_progress.set_postfix(
                {
                    "ok": status_counts.get("ok", 0),
                    "det": progress_counts["total_detections"],
                    "cache": progress_counts["mask_cache_hits"],
                },
                refresh=False,
            )

        def cleanup_tile_artifacts(context: dict[str, object]) -> None:
            cleanup_artifact_path(Path(context["legacy_raw_vector_path"]))
            cleanup_artifact_path(Path(context["legacy_enriched_vector_path"]))
            if not WRITE_PROBABILITY_RASTERS:
                cleanup_artifact_path(Path(context["legacy_probability_path"]))
            if not should_persist_tile_mask_artifacts():
                cleanup_artifact_path(Path(context["mask_path"]))
            if not should_persist_clipped_input_artifacts():
                cleanup_artifact_path(Path(context["clipped_input_path"]))
                cleanup_artifact_path(context.get("input_clip_mask_path"))

        def maybe_render_review(context: dict[str, object], current_detections: gpd.GeoDataFrame) -> str | None:
            if int(context["row_index"]) >= REVIEW_COUNT:
                return None
            review_started = time.perf_counter()
            raster_path = Path(context["raster_path"])
            clipped_input_path = Path(context["clipped_input_path"])
            input_clip_mask_path = context.get("input_clip_mask_path")
            mask_path = Path(context["mask_path"])
            review_png = render_review_visuals(
                geoai,
                raster_path=raster_path,
                inference_input_path=clipped_input_path if clipped_input_path.exists() else raster_path,
                input_clip_mask_path=input_clip_mask_path if input_clip_mask_path is not None and Path(input_clip_mask_path).exists() else None,
                mask_path=mask_path if mask_path.exists() else None,
                detections=current_detections,
                output_stem=Path(context["output_stem"]),
            )
            record_stage_duration(stage_timings, "review_render_seconds", review_started)
            return review_png

        def build_tile_context(row: object) -> dict[str, object]:
            row_index = int(getattr(row, "processing_order"))
            raster_path = Path(row.tile_abs_path)
            output_stem = build_output_stem(inference_root, row)
            output_stem.parent.mkdir(parents=True, exist_ok=True)

            mask_path = output_stem.with_name(f"{output_stem.name}_pred.tif")
            legacy_probability_path = output_stem.with_name(f"{output_stem.name}_prob.tif")
            probability_path = legacy_probability_path if INFERENCE_MODEL_FAMILY == "smp" and WRITE_PROBABILITY_RASTERS else None
            legacy_raw_vector_path = output_stem.with_name(f"{output_stem.name}_pred.geojson")
            legacy_enriched_vector_path = output_stem.with_name(f"{output_stem.name}_pred_props.geojson")
            clipped_input_path = output_stem.with_name(f"{output_stem.name}_input_clip.tif")
            input_clip_mask_path = (
                output_stem.with_name(f"{output_stem.name}_input_clip_mask.tif")
                if row_index < REVIEW_COUNT and WRITE_INPUT_CLIP_PREVIEW_ARTIFACTS
                else None
            )
            return {
                "row": row,
                "row_index": row_index,
                "raster_path": raster_path,
                "output_stem": output_stem,
                "mask_path": mask_path,
                "legacy_probability_path": legacy_probability_path,
                "probability_path": probability_path,
                "legacy_raw_vector_path": legacy_raw_vector_path,
                "legacy_enriched_vector_path": legacy_enriched_vector_path,
                "clipped_input_path": clipped_input_path,
                "input_clip_mask_path": input_clip_mask_path,
                "inference_input_path": raster_path,
                "input_clip_applied": False,
            }

        try:
            grouped_selected = (
                selected.groupby("inference_group_key", sort=False)
                if ENABLE_SOURCE_GROUP_TILE_BATCHING
                else [("all", selected)]
            )
            for _, group_df in grouped_selected:
                group_contexts: list[dict[str, object]] = []
                for row in group_df.itertuples(index=False):
                    context = build_tile_context(row)
                    row_value = context["row"]
                    mask_path = Path(context["mask_path"])
                    if OVERWRITE_INFERENCE_ARTIFACTS:
                        for artifact_path in (
                            context["mask_path"],
                            context["legacy_probability_path"],
                            context["legacy_raw_vector_path"],
                            context["legacy_enriched_vector_path"],
                            context["clipped_input_path"],
                            context["input_clip_mask_path"],
                        ):
                            cleanup_artifact_path(artifact_path)

                    if INFERENCE_ARTIFACT_MODE == "compact":
                        cleanup_artifact_path(Path(context["legacy_raw_vector_path"]))
                        cleanup_artifact_path(Path(context["legacy_enriched_vector_path"]))
                        if not WRITE_PROBABILITY_RASTERS:
                            cleanup_artifact_path(Path(context["legacy_probability_path"]))

                    if mask_path.exists():
                        progress_counts["mask_cache_hits"] += 1
                    else:
                        if CLIP_TO_BUILDINGS and CLIP_INPUT_RASTERS_TO_BUILDINGS:
                            stage_started = time.perf_counter()
                            input_clip_applied = write_building_masked_raster(
                                Path(context["raster_path"]),
                                Path(context["clipped_input_path"]),
                                row=row_value,
                                building_footprints=building_footprints,
                                buffer_m=BUILDING_CLIP_BUFFER_M,
                                mask_output_path=context["input_clip_mask_path"],
                            )
                            record_stage_duration(stage_timings, "input_raster_clip_seconds", stage_started)
                            if input_clip_applied:
                                context["inference_input_path"] = Path(context["clipped_input_path"])
                                context["input_clip_applied"] = True
                                progress_counts["input_clipped_tiles"] += 1
                    group_contexts.append(context)

                pending_contexts = [context for context in group_contexts if not Path(context["mask_path"]).exists()]
                for batch_start in range(0, len(pending_contexts), INFERENCE_TILE_BATCH_SIZE):
                    batch_contexts = pending_contexts[batch_start : batch_start + INFERENCE_TILE_BATCH_SIZE]
                    stage_started = time.perf_counter()
                    batch_written = False
                    if ENABLE_SOURCE_GROUP_TILE_BATCHING and len(batch_contexts) > 1:
                        batch_written = segment_exact_tile_batch_with_runtime_settings(
                            batch_contexts=batch_contexts,
                            config=config,
                            smp_runtime=smp_runtime,
                            dinov3_runtime=dinov3_runtime,
                        )
                    if not batch_written:
                        for context in batch_contexts:
                            run_segmentation_inference(
                                geoai,
                                raster_path=Path(context["inference_input_path"]),
                                mask_path=Path(context["mask_path"]),
                                probability_path=context["probability_path"],
                                config=config,
                                smp_runtime=smp_runtime,
                                dinov3_runtime=dinov3_runtime,
                            )
                    record_stage_duration(stage_timings, "model_inference_seconds", stage_started)
                    progress_counts["tiles_inferred"] += len(batch_contexts)

                for context in group_contexts:
                    row = context["row"]
                    mask_path = Path(context["mask_path"])
                    probability_path = context["probability_path"]
                    building_clip_applied = bool(context["input_clip_applied"])
                    if CLIP_TO_BUILDINGS and CLIP_PREDICTION_MASK_TO_BUILDINGS:
                        stage_started = time.perf_counter()
                        building_clip_applied = mask_raster_to_buildings(
                            mask_path,
                            row=row,
                            building_footprints=building_footprints,
                            buffer_m=BUILDING_CLIP_BUFFER_M,
                        ) or building_clip_applied
                        if probability_path is not None and Path(probability_path).exists():
                            mask_raster_to_buildings(
                                probability_path,
                                row=row,
                                building_footprints=building_footprints,
                                buffer_m=BUILDING_CLIP_BUFFER_M,
                            )
                        record_stage_duration(stage_timings, "mask_building_clip_seconds", stage_started)

                    if not mask_path.exists():
                        tile_records.append(
                            {
                                "model_family": INFERENCE_MODEL_FAMILY,
                                "model_run_name": config["run_name"],
                                "source": row.source,
                                "item_id": row.item_id,
                                "municipio": row.municipio,
                                "municipio_geoid": row.municipio_geoid,
                                "h3_cell_id": row.h3_cell_id,
                                "building_count": row.building_count,
                                "municipality_building_count": row.municipality_building_count,
                                "crosses_municipality_boundary": row.crosses_municipality_boundary,
                                "tile_path": row.tile_path,
                                "local_asset_path": row.local_asset_path,
                                "predicted_mask_path": None,
                                "probability_path": None,
                                "vector_path": None,
                                "review_png_path": None,
                                "inference_mode": INFERENCE_MODE,
                                "building_clip_applied": building_clip_applied,
                                "input_clip_applied": bool(context["input_clip_applied"]),
                                "building_clip_buffer_m": BUILDING_CLIP_BUFFER_M,
                                "status": "mask_missing",
                                "detection_count": 0,
                                "detected_area_m2": 0.0,
                            }
                        )
                        cleanup_tile_artifacts(context)
                        finalize_tile_progress("mask_missing", 0)
                        continue

                    stage_started = time.perf_counter()
                    detections = mask_to_geodataframe(mask_path)
                    record_stage_duration(stage_timings, "vectorization_seconds", stage_started)

                    if detections.empty:
                        review_png_path = maybe_render_review(context, empty_detection_gdf())
                        tile_records.append(
                            {
                                "model_family": INFERENCE_MODEL_FAMILY,
                                "model_run_name": config["run_name"],
                                "source": row.source,
                                "item_id": row.item_id,
                                "municipio": row.municipio,
                                "municipio_geoid": row.municipio_geoid,
                                "h3_cell_id": row.h3_cell_id,
                                "building_count": row.building_count,
                                "municipality_building_count": row.municipality_building_count,
                                "crosses_municipality_boundary": row.crosses_municipality_boundary,
                                "tile_path": row.tile_path,
                                "local_asset_path": row.local_asset_path,
                                "predicted_mask_path": str(mask_path.relative_to(PROJECT_ROOT)) if should_persist_tile_mask_artifacts() else None,
                                "probability_path": str(Path(probability_path).relative_to(PROJECT_ROOT)) if probability_path is not None and Path(probability_path).exists() else None,
                                "vector_path": None,
                                "review_png_path": review_png_path,
                                "inference_mode": INFERENCE_MODE,
                                "building_clip_applied": building_clip_applied,
                                "input_clip_applied": bool(context["input_clip_applied"]),
                                "building_clip_buffer_m": BUILDING_CLIP_BUFFER_M,
                                "status": "no_positive_mask",
                                "detection_count": 0,
                                "detected_area_m2": 0.0,
                            }
                        )
                        cleanup_tile_artifacts(context)
                        finalize_tile_progress("no_positive_mask", 0)
                        continue

                    stage_started = time.perf_counter()
                    detections = enrich_prediction_vectors(geoai, detections)
                    empty_status = "filtered_empty"
                    if CLIP_TO_BUILDINGS:
                        detections = clip_detections_to_buildings(
                            detections,
                            row=row,
                            building_footprints=building_footprints,
                        )
                        if detections.empty:
                            empty_status = "outside_buildings"
                        else:
                            detections = enrich_prediction_vectors(geoai, detections)
                    if detections.empty:
                        record_stage_duration(stage_timings, "detection_postprocess_seconds", stage_started)
                        review_png_path = maybe_render_review(context, empty_detection_gdf())
                        tile_records.append(
                            {
                                "model_family": INFERENCE_MODEL_FAMILY,
                                "model_run_name": config["run_name"],
                                "source": row.source,
                                "item_id": row.item_id,
                                "municipio": row.municipio,
                                "municipio_geoid": row.municipio_geoid,
                                "h3_cell_id": row.h3_cell_id,
                                "building_count": row.building_count,
                                "municipality_building_count": row.municipality_building_count,
                                "crosses_municipality_boundary": row.crosses_municipality_boundary,
                                "tile_path": row.tile_path,
                                "local_asset_path": row.local_asset_path,
                                "predicted_mask_path": str(mask_path.relative_to(PROJECT_ROOT)) if should_persist_tile_mask_artifacts() else None,
                                "probability_path": str(Path(probability_path).relative_to(PROJECT_ROOT)) if probability_path is not None and Path(probability_path).exists() else None,
                                "vector_path": None,
                                "review_png_path": review_png_path,
                                "inference_mode": INFERENCE_MODE,
                                "building_clip_applied": building_clip_applied,
                                "input_clip_applied": bool(context["input_clip_applied"]),
                                "building_clip_buffer_m": BUILDING_CLIP_BUFFER_M,
                                "status": empty_status,
                                "detection_count": 0,
                                "detected_area_m2": 0.0,
                            }
                        )
                        cleanup_tile_artifacts(context)
                        finalize_tile_progress(empty_status, 0)
                        continue

                    persisted_mask_path = mask_path if should_persist_tile_mask_artifacts() else None
                    detections = build_detection_frame(
                        detections,
                        row,
                        config,
                        mask_path=persisted_mask_path,
                        vector_path=None,
                        probability_path=probability_path,
                    )
                    all_detections.append(detections)
                    record_stage_duration(stage_timings, "detection_postprocess_seconds", stage_started)

                    review_png_path = maybe_render_review(context, detections)

                    detected_area_m2 = float(detections["area_m2"].sum()) if "area_m2" in detections.columns else 0.0
                    tile_records.append(
                        {
                            "model_family": INFERENCE_MODEL_FAMILY,
                            "model_run_name": config["run_name"],
                            "source": row.source,
                            "item_id": row.item_id,
                            "municipio": row.municipio,
                            "municipio_geoid": row.municipio_geoid,
                            "h3_cell_id": row.h3_cell_id,
                            "building_count": row.building_count,
                            "municipality_building_count": row.municipality_building_count,
                            "crosses_municipality_boundary": row.crosses_municipality_boundary,
                            "tile_path": row.tile_path,
                            "local_asset_path": row.local_asset_path,
                            "predicted_mask_path": str(mask_path.relative_to(PROJECT_ROOT)) if persisted_mask_path is not None and persisted_mask_path.exists() else None,
                            "probability_path": str(Path(probability_path).relative_to(PROJECT_ROOT)) if probability_path is not None and Path(probability_path).exists() else None,
                            "vector_path": None,
                            "review_png_path": review_png_path,
                            "inference_mode": INFERENCE_MODE,
                            "building_clip_applied": building_clip_applied,
                            "input_clip_applied": bool(context["input_clip_applied"]),
                            "building_clip_buffer_m": BUILDING_CLIP_BUFFER_M,
                            "status": "ok",
                            "detection_count": int(len(detections)),
                            "detected_area_m2": detected_area_m2,
                        }
                    )
                    cleanup_tile_artifacts(context)
                    finalize_tile_progress("ok", int(len(detections)))
        finally:
            tile_progress.close()
        record_stage_duration(stage_timings, "tile_loop_seconds", tile_loop_started)
        selected = selected.drop(columns=["inference_group_key", "processing_order"], errors="ignore")

        tile_summary_df = pd.DataFrame.from_records(tile_records)
        merged_detections = (
            gpd.GeoDataFrame(pd.concat(all_detections, ignore_index=True), crs=all_detections[0].crs)
            if all_detections
            else empty_detection_gdf()
        )
        extra_metadata.update(
            {
                "progress_counts": {
                    "mask_cache_hits": int(progress_counts["mask_cache_hits"]),
                    "tiles_inferred": int(progress_counts["tiles_inferred"]),
                    "total_detections": int(progress_counts["total_detections"]),
                    "input_clipped_tiles": int(progress_counts["input_clipped_tiles"]),
                },
            }
        )
        stage_started = time.perf_counter()
        merged_detection_artifact = write_merged_detections_artifact(inference_root, merged_detections)
        record_stage_duration(stage_timings, "merged_detection_artifact_write_seconds", stage_started)
        extra_metadata["merged_detections_artifact"] = (
            str(merged_detection_artifact.relative_to(PROJECT_ROOT)) if merged_detection_artifact is not None else None
        )
    else:
        stage_started = time.perf_counter()
        local_inputs = load_reporting_inputs_from_local_outputs(inference_root, config)
        record_stage_duration(stage_timings, "local_reporting_input_reload_seconds", stage_started)
        if local_inputs is None:
            return None
        selected, tile_summary_df, merged_detections, reporting_source = local_inputs
        building_footprints = load_overture_building_footprints(selected) if WRITE_PRESENTATION_OUTPUTS else gpd.GeoDataFrame()
        extra_metadata["reporting_source"] = reporting_source

    presentation_outputs: dict[str, str] = {}
    stage_started = time.perf_counter()
    schema_path = write_inference_schema_reference(inference_root)
    record_stage_duration(stage_timings, "schema_write_seconds", stage_started)
    presentation_outputs["schema_reference_csv"] = str(schema_path.relative_to(PROJECT_ROOT))

    if should_run_reporting_phase() and WRITE_PRESENTATION_OUTPUTS:
        if "building_footprints" not in locals():
            stage_started = time.perf_counter()
            building_footprints = load_overture_building_footprints(selected)
            record_stage_duration(stage_timings, "building_footprint_load_seconds", stage_started)
        stage_started = time.perf_counter()
        presentation_outputs.update(
            write_presentation_outputs(
                inference_root=inference_root,
                selected=selected,
                building_footprints=building_footprints,
                detections=merged_detections,
                config=config,
            )
        )
        record_stage_duration(stage_timings, "presentation_reporting_seconds", stage_started)
    elif not should_run_reporting_phase():
        print("reporting phase skipped by GEOAI_INFERENCE_PHASE.")

    output_metrics = {
        "inference_root": summarize_directory_tree(inference_root),
        "maps_root": summarize_directory_tree(MAPS_ROOT),
    }

    if not should_run_inference_phase() or not should_commit_to_db():
        record_stage_duration(stage_timings, "workflow_total_seconds", workflow_started)
        summary_path = write_local_run_summary(
            inference_root,
            config=config,
            selected=selected,
            tile_summary_df=tile_summary_df,
            detections=merged_detections,
            db_committed=False,
            presentation_outputs=presentation_outputs,
            stage_timings=stage_timings,
            output_metrics=output_metrics,
            extra_metadata=extra_metadata,
        )
        print(f"local run summary: {summary_path}")
        return summary_path

    stage_started = time.perf_counter()
    db_path = resolve_db_path()
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")

    write_detection_table(con, merged_detections)
    backfill_detection_table_model_arch(con)
    write_tile_summary_table(con, tile_summary_df)
    has_building_hits = build_building_hits_table(con, config)
    build_municipio_summary_table(con, has_building_hits)
    multimodel_detections = build_multimodel_detection_catalog()
    write_multimodel_detection_table(con, multimodel_detections)
    multimodel_building_catalog = build_multimodel_building_detection_catalog(multimodel_detections)
    write_multimodel_building_catalog_table(con, multimodel_building_catalog)
    record_stage_duration(stage_timings, "duckdb_write_seconds", stage_started)

    print(f"wrote {len(merged_detections):,} detections to {DETECTION_TABLE}")
    print(f"wrote {len(tile_summary_df):,} tile rows to {TILE_SUMMARY_TABLE}")
    if not multimodel_detections.empty:
        print(f"wrote {len(multimodel_detections):,} cross-run detections to {ALL_MODEL_DETECTION_TABLE}")
    if not multimodel_building_catalog.empty:
        print(f"wrote {len(multimodel_building_catalog):,} detected buildings to {ALL_MODEL_BUILDING_TABLE}")

    municipio_summary = con.execute(
        f"SELECT * FROM {MUNICIPIO_SUMMARY_TABLE} ORDER BY municipio;"
    ).fetchdf()
    if not municipio_summary.empty:
        print("municipio inference summary:")
        print(municipio_summary.to_string(index=False))

    if has_building_hits:
        building_summary = con.execute(
            f"""
            SELECT municipio,
                   COUNT(*) AS buildings,
                   SUM(CAST(has_pv_detected AS INT)) AS detected_buildings,
                   SUM(pv_detected_count) AS detected_polygons
            FROM {BUILDING_HITS_TABLE}
            GROUP BY municipio
            ORDER BY municipio;
            """
        ).fetchdf()
        print("building hit summary:")
        print(building_summary.to_string(index=False))

    if not multimodel_building_catalog.empty:
        multimodel_summary = write_multimodel_building_summary(multimodel_building_catalog)
        if not multimodel_summary.empty:
            print("multi-model building detection summary:")
            print(multimodel_summary.to_string(index=False))
        plot_multimodel_building_catalog_map(multimodel_building_catalog)

    con.close()

    record_stage_duration(stage_timings, "workflow_total_seconds", workflow_started)
    summary_path = write_local_run_summary(
        inference_root,
        config=config,
        selected=selected,
        tile_summary_df=tile_summary_df,
        detections=merged_detections,
        db_committed=True,
        presentation_outputs=presentation_outputs,
        stage_timings=stage_timings,
        output_metrics=output_metrics,
        extra_metadata=extra_metadata,
    )
    print(f"commit summary: {summary_path}")
    return summary_path


if __name__ == "__main__":
    run_inference_workflow()

# %%




