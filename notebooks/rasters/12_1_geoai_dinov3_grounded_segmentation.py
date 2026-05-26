# %% [markdown]
# # DINOv3 Grounded Segmentation
# 
# Fine-tunes GeoAI's DINOv3 semantic-segmentation stack on the footprint-
# grounded masks exported by `11_geoai_training_data.py`. The notebook is safe
# by default: a normal script run performs manifest and dependency preflight,
# while actual training only starts when `GEOAI_RUN_DINOV3_TRAINING=1`.

# %%
"""12_1_geoai_dinov3_grounded_segmentation.py"""

from __future__ import annotations

import json
import os
import random
import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from dotenv import load_dotenv


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

from utils.geoai_review import render_prediction_review_bundle
from utils.geoai_preview_sources import collect_naip_stac_preview_rasters, infer_stac_source_name
from utils.geoai_segmentation import (
    compute_binary_mask_metrics,
    evaluate_holdout_building_metrics,
    summarize_holdout_building_metrics,
)
from utils.geoai_training_contract import (
    build_training_contract,
    compare_training_contracts,
    project_relative_path,
    training_contract_run_fragment,
)


def _resolve_configured_path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_optional_path(raw_value: str | None) -> Path | None:
    if not raw_value:
        return None
    path = Path(raw_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_num_workers() -> int:
    cpu_count = os.cpu_count() or 0
    if cpu_count <= 1:
        return 0
    return min(8, cpu_count - 1)

DINOV3_NORMALIZATION_PROFILES = {
    "sat493m": {
        "label": "SAT-493M satellite imagery",
        "mean": (0.430, 0.411, 0.296),
        "std": (0.213, 0.156, 0.143),
    },
    "lvd1689m": {
        "label": "LVD-1689M web imagery",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}
DINOV3_BACKBONE_PRESETS = {
    "vitl16_sat493m": {
        "label": "ViT-L/16 SAT-493M (GeoAI-ready)",
        "model_name": "dinov3_vitl16",
        "normalization_profile": "sat493m",
        "hub_weights": "SAT493M",
        "requires_explicit_weights": False,
        "note": "Correct GeoAI hub model is dinov3_vitl16; this preset keeps GeoAI's working SAT-493M .pth fallback plus the SAT normalization profile.",
    },
    "vit7b16_sat493m": {
        "label": "ViT-7B/16 SAT-493M (official hub / optional .pth)",
        "model_name": "dinov3_vit7b16",
        "normalization_profile": "sat493m",
        "hub_weights": "SAT493M",
        "requires_explicit_weights": False,
        "note": "When no local .pth is supplied, the notebook now requests the official DINOv3 SAT-493M checkpoint URL for dinov3_vit7b16 instead of GeoAI's incorrect ViT-L fallback.",
    },
    "vitl16_lvd1689m": {
        "label": "ViT-L/16 LVD-1689M (official hub / optional .pth)",
        "model_name": "dinov3_vitl16",
        "normalization_profile": "lvd1689m",
        "hub_weights": "LVD1689M",
        "requires_explicit_weights": False,
        "note": "When no local .pth is supplied, the notebook now requests the official DINOv3 LVD-1689M checkpoint URL instead of GeoAI's incorrect SAT fallback.",
    },
    "vitb16_lvd1689m": {
        "label": "ViT-B/16 LVD-1689M (official hub / optional .pth)",
        "model_name": "dinov3_vitb16",
        "normalization_profile": "lvd1689m",
        "hub_weights": "LVD1689M",
        "requires_explicit_weights": False,
        "note": "When no local .pth is supplied, the notebook now requests the official DINOv3 LVD-1689M checkpoint URL instead of GeoAI's incorrect SAT fallback.",
    },
}
DINOV3_ALLOWED_HUB_MODELS = {
    "dinov3_vits16",
    "dinov3_vits16plus",
    "dinov3_vitb16",
    "dinov3_vitl16",
    "dinov3_vith16plus",
    "dinov3_vit7b16",
    "dinov3_convnext_tiny",
    "dinov3_convnext_small",
    "dinov3_convnext_base",
    "dinov3_convnext_large",
}

BACKBONE_PRESET = (os.getenv("GEOAI_DINOV3_BACKBONE_PRESET", "vitl16_sat493m") or "vitl16_sat493m").strip().lower()
BACKBONE_WEIGHTS_PATH = _resolve_optional_path(os.getenv("GEOAI_DINOV3_WEIGHTS_PATH"))
MODEL_NAME = "dinov3_vitl16"
BACKBONE_HUB_WEIGHTS = "SAT493M"
INPUT_NORMALIZATION_PROFILE = "sat493m"
INPUT_NORMALIZATION_MEAN = DINOV3_NORMALIZATION_PROFILES[INPUT_NORMALIZATION_PROFILE]["mean"]
INPUT_NORMALIZATION_STD = DINOV3_NORMALIZATION_PROFILES[INPUT_NORMALIZATION_PROFILE]["std"]
BACKBONE_NOTE = ""
NUM_CLASSES = int(os.getenv("GEOAI_DINOV3_NUM_CLASSES", "2"))
DECODER_FEATURES = int(os.getenv("GEOAI_DINOV3_DECODER_FEATURES", "256"))
NUM_EPOCHS = int(os.getenv("GEOAI_DINOV3_NUM_EPOCHS", "20"))
BATCH_SIZE = int(os.getenv("GEOAI_DINOV3_BATCH_SIZE", "16"))
LEARNING_RATE = float(os.getenv("GEOAI_DINOV3_LEARNING_RATE", "1e-4"))
WEIGHT_DECAY = float(os.getenv("GEOAI_DINOV3_WEIGHT_DECAY", "1e-4"))
NUM_WORKERS = int(os.getenv("GEOAI_DINOV3_NUM_WORKERS", str(default_num_workers())))
LIGHTNING_PRECISION = (
    os.getenv("GEOAI_DINOV3_LIGHTNING_PRECISION", "bf16-mixed" if torch.cuda.is_available() else "32-true")
    or "32-true"
).strip()
FLOAT32_MATMUL_PRECISION = (os.getenv("GEOAI_DINOV3_FLOAT32_MATMUL_PRECISION", "high") or "high").strip().lower()
ENABLE_TF32 = os.getenv("GEOAI_DINOV3_ENABLE_TF32", "1") == "1"
ENABLE_CUDNN_BENCHMARK = os.getenv("GEOAI_DINOV3_CUDNN_BENCHMARK", "1") == "1"
CUDA_EMPTY_CACHE_BEFORE_TRAINING = os.getenv("GEOAI_DINOV3_EMPTY_CACHE_BEFORE_TRAINING", "1") == "1"
FREEZE_BACKBONE = os.getenv("GEOAI_DINOV3_FREEZE_BACKBONE", "1") == "1"
USE_LORA = os.getenv("GEOAI_DINOV3_USE_LORA", "0") == "1"
LORA_RANK = int(os.getenv("GEOAI_DINOV3_LORA_RANK", "4"))
TARGET_SIZE_RAW = int(os.getenv("GEOAI_DINOV3_TARGET_SIZE", "512") or "0")
TARGET_SIZE = TARGET_SIZE_RAW if TARGET_SIZE_RAW > 0 else None
PATCH_SIZE = int(os.getenv("GEOAI_DINOV3_PATCH_SIZE", "16") or "16")

DEFAULT_TRAIN_ROOT = PROJECT_ROOT / "outputs" / "geoai_train_contextily"
TRAIN_ROOT = _resolve_configured_path("GEOAI_TRAIN_ROOT", DEFAULT_TRAIN_ROOT)
TRAIN_MANIFEST = TRAIN_ROOT / "training_chip_manifest.csv"
IMAGES_DIR = TRAIN_ROOT / "images"
RAW_MASKS_DIR = TRAIN_ROOT / "masks"
GROUNDED_MASKS_DIR = TRAIN_ROOT / "grounded_masks"
PROMPT_ARTIFACTS_DIR = TRAIN_ROOT / "prompt_artifacts"
BASE_MODEL_ROOT = PROJECT_ROOT / "outputs" / "models" / "dinov3_grounded"
EXPLICIT_MODEL_OUT = _resolve_optional_path(os.getenv("GEOAI_DINOV3_MODEL_OUT"))
EXPLICIT_PREVIEW_ROOT = _resolve_optional_path(os.getenv("GEOAI_DINOV3_PREVIEW_ROOT"))
EXPLICIT_EVAL_ROOT = _resolve_optional_path(os.getenv("GEOAI_DINOV3_EVAL_ROOT"))
MODEL_OUT = EXPLICIT_MODEL_OUT or BASE_MODEL_ROOT / "dinov3-vitl16__df256__ps16__e10__b16__lr1em4__frozen"
PREVIEW_ROOT = EXPLICIT_PREVIEW_ROOT or MODEL_OUT / "preview"
EVAL_ROOT = EXPLICIT_EVAL_ROOT or MODEL_OUT / "evaluation"
METADATA_PATH = MODEL_OUT / "dinov3_metadata.json"
EVAL_METRICS_CSV = EVAL_ROOT / "split_metrics.csv"
EVAL_BUILDING_METRICS_CSV = EVAL_ROOT / "holdout_building_metrics.csv"
EVAL_SUMMARY_JSON = EVAL_ROOT / "split_summary.json"

ACCELERATOR = (os.getenv("GEOAI_DINOV3_ACCELERATOR", "auto") or "auto").strip()
DEVICES = (os.getenv("GEOAI_DINOV3_DEVICES", "auto") or "auto").strip()
EARLY_STOPPING_PATIENCE = int(os.getenv("GEOAI_DINOV3_PATIENCE", "15") or "15")
SEED = int(os.getenv("GEOAI_DINOV3_SEED", "323") or "323")
RUN_DINOV3_TRAINING = os.getenv("GEOAI_RUN_DINOV3_TRAINING", "1") == "1"
RUN_DINOV3_PREVIEW = os.getenv("GEOAI_RUN_DINOV3_PREVIEW", "1") == "1"
RUN_DINOV3_EVAL = os.getenv("GEOAI_RUN_DINOV3_EVAL", "0") == "1"
RUN_DINOV3_TRANSFORM_PREVIEW = os.getenv("GEOAI_RUN_DINOV3_TRANSFORM_PREVIEW", "1") == "1"
DINOV3_TRANSFORM_PREVIEW_COUNT = int(os.getenv("GEOAI_DINOV3_TRANSFORM_PREVIEW_COUNT", "6") or "6")
PREVIEW_SPLIT = (os.getenv("GEOAI_DINOV3_PREVIEW_SPLIT", "val") or "val").strip().lower()
DINOV3_SIMILARITY_SPLIT_OVERRIDE = os.getenv("GEOAI_DINOV3_SIMILARITY_SPLIT")
DINOV3_SIMILARITY_SPLIT = (DINOV3_SIMILARITY_SPLIT_OVERRIDE or PREVIEW_SPLIT).strip().lower()
RUN_DINOV3_SIMILARITY_PREVIEW = os.getenv("GEOAI_RUN_DINOV3_SIMILARITY_PREVIEW", "0") == "1"
DINOV3_SIMILARITY_PREVIEW_COUNT = int(os.getenv("GEOAI_DINOV3_SIMILARITY_PREVIEW_COUNT", "3") or "3")
DINOV3_SIMILARITY_TARGET_SIZE = int(os.getenv("GEOAI_DINOV3_SIMILARITY_TARGET_SIZE", "1024") or "1024")
DINOV3_SIMILARITY_COLORMAP = (os.getenv("GEOAI_DINOV3_SIMILARITY_COLORMAP", "turbo") or "turbo").strip()
DINOV3_SIMILARITY_ALPHA = float(os.getenv("GEOAI_DINOV3_SIMILARITY_ALPHA", "0.75") or "0.75")
PREVIEW_COUNT = int(os.getenv("GEOAI_DINOV3_PREVIEW_COUNT", "5") or "5")
EXTERNAL_PREVIEW_COUNT = int(os.getenv("GEOAI_DINOV3_EXTERNAL_PREVIEW_COUNT", "1") or "1")
PREVIEW_WINDOW_SIZE = int(os.getenv("GEOAI_DINOV3_PREVIEW_WINDOW_SIZE", "512") or "512")
PREVIEW_OVERLAP = int(os.getenv("GEOAI_DINOV3_PREVIEW_OVERLAP", "256") or "256")
PREVIEW_BATCH_SIZE = int(os.getenv("GEOAI_DINOV3_PREVIEW_BATCH_SIZE", "8") or "8")
EVAL_BATCH_SIZE = int(
    os.getenv("GEOAI_DINOV3_EVAL_BATCH_SIZE", str(max(PREVIEW_BATCH_SIZE, BATCH_SIZE)))
    or str(max(PREVIEW_BATCH_SIZE, BATCH_SIZE))
)
EVAL_SPLITS = tuple(
    split_name.strip().lower()
    for split_name in (os.getenv("GEOAI_DINOV3_EVAL_SPLITS", "val,test") or "val,test").split(",")
    if split_name.strip()
)
EVAL_MAX_ROWS = int(os.getenv("GEOAI_DINOV3_EVAL_MAX_ROWS", "0") or "0")
EVAL_REVIEW_COUNT = int(os.getenv("GEOAI_DINOV3_EVAL_REVIEW_COUNT", "12") or "12")
RUN_DINOV3_TEST_DETECTION_PREVIEW = os.getenv("GEOAI_DINOV3_RUN_TEST_DETECTION_PREVIEW", "1") == "1"
TEST_DETECTION_SAMPLE_COUNT = int(os.getenv("GEOAI_DINOV3_TEST_DETECTION_SAMPLE_COUNT", "24") or "24")
STAC_TILE_ROOT = PROJECT_ROOT / "outputs" / "stac_tiles"
LOCAL_STAC_CACHE_ROOT = PROJECT_ROOT / "data" / "rasters" / "stac" / "local"
SOLAR_RASTER_ROOT = PROJECT_ROOT / "data" / "rasters" / "solar"
SUPPORTED_IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
RUN_NAMING_VERSION = 2
RESUME_TRAINING = os.getenv("GEOAI_DINOV3_RESUME_TRAINING", "1") == "1"
RESUME_RUN_DIR = _resolve_optional_path(os.getenv("GEOAI_DINOV3_RESUME_RUN_DIR"))
RESUME_CHECKPOINT = _resolve_optional_path(os.getenv("GEOAI_DINOV3_RESUME_CHECKPOINT"))
TRAINING_CONTRACT: dict[str, object] | None = None
DINOV3_OFFICIAL_BASE_URL = "https://dl.fbaipublicfiles.com/dinov3"
_ORIGINAL_DINOV3_PROCESSOR_LOAD_MODEL = None
_ORIGINAL_DINOV3_SEGMENTER_LOAD_BACKBONE = None


def _slugify(value: object) -> str:
    text = str(value).strip().lower()
    slug_chars = [char if char.isalnum() else "-" for char in text]
    slug = "".join(slug_chars)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "default"


def _float_slug(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 0.01:
        return f"{value:.0e}".replace("e-", "em").replace("e+", "ep")
    return f"{value:g}".replace(".", "p")


def running_in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def get_backbone_preset(preset_key: str | None = None) -> dict[str, object]:
    key = (preset_key or BACKBONE_PRESET).strip().lower()
    if key not in DINOV3_BACKBONE_PRESETS:
        raise RuntimeError(
            f"unsupported GEOAI_DINOV3_BACKBONE_PRESET={key!r}; expected one of {sorted(DINOV3_BACKBONE_PRESETS)}"
        )
    return DINOV3_BACKBONE_PRESETS[key]


def apply_backbone_settings() -> None:
    global MODEL_NAME, BACKBONE_HUB_WEIGHTS, INPUT_NORMALIZATION_PROFILE, INPUT_NORMALIZATION_MEAN, INPUT_NORMALIZATION_STD, BACKBONE_NOTE
    preset = get_backbone_preset()
    MODEL_NAME = str(preset["model_name"])
    BACKBONE_HUB_WEIGHTS = str(preset.get("hub_weights") or "").strip().upper()
    INPUT_NORMALIZATION_PROFILE = str(preset["normalization_profile"])
    profile = DINOV3_NORMALIZATION_PROFILES[INPUT_NORMALIZATION_PROFILE]
    INPUT_NORMALIZATION_MEAN = tuple(float(value) for value in profile["mean"])
    INPUT_NORMALIZATION_STD = tuple(float(value) for value in profile["std"])
    BACKBONE_NOTE = str(preset["note"])


def resolve_backbone_weights_path() -> Path | None:
    preset = get_backbone_preset()
    if BACKBONE_WEIGHTS_PATH is not None:
        if not BACKBONE_WEIGHTS_PATH.exists():
            raise RuntimeError(f"GEOAI_DINOV3_WEIGHTS_PATH not found: {BACKBONE_WEIGHTS_PATH}")
        if BACKBONE_WEIGHTS_PATH.suffix.lower() == ".safetensors":
            raise RuntimeError(
                "GeoAI DINOv3 fine-tuning expects a compatible .pth backbone state dict, not a Transformers .safetensors checkpoint."
            )
        return BACKBONE_WEIGHTS_PATH

    if bool(preset["requires_explicit_weights"]) and not str(preset.get("hub_weights") or "").strip():
        raise RuntimeError(
            f"Backbone preset {BACKBONE_PRESET!r} requires GEOAI_DINOV3_WEIGHTS_PATH to point to a compatible local .pth file."
        )
    return None


def resolve_backbone_hub_weights_name(metadata: dict[str, object] | None = None) -> str:
    if metadata is not None:
        raw_hub_weights = metadata.get("hub_weights")
        if isinstance(raw_hub_weights, str) and raw_hub_weights.strip():
            return raw_hub_weights.strip().upper()
        raw_preset = metadata.get("backbone_preset")
        if isinstance(raw_preset, str) and raw_preset.strip():
            preset = get_backbone_preset(raw_preset)
            hub_weights = str(preset.get("hub_weights") or "").strip().upper()
            if hub_weights:
                return hub_weights

    preset = get_backbone_preset()
    hub_weights = str(preset.get("hub_weights") or "").strip().upper()
    if not hub_weights:
        raise RuntimeError(f"Backbone preset {BACKBONE_PRESET!r} does not define a DINOv3 weight family.")
    return hub_weights


def build_official_dinov3_weights_url(model_name: str, hub_weights_name: str) -> str:
    if model_name not in DINOV3_ALLOWED_HUB_MODELS:
        raise RuntimeError(
            f"Unsupported DINOv3 hub model {model_name!r}; expected one of {sorted(DINOV3_ALLOWED_HUB_MODELS)}"
        )
    if hub_weights_name not in {"SAT493M", "LVD1689M"}:
        raise RuntimeError(
            f"Unsupported DINOv3 weight family {hub_weights_name!r}; expected SAT493M or LVD1689M."
        )
    weights_slug = hub_weights_name.lower()
    return f"{DINOV3_OFFICIAL_BASE_URL}/{model_name}/{model_name}_pretrain_{weights_slug}.pth"


def resolve_dinov3_hub_location() -> tuple[str, str]:
    dinov3_location = os.getenv("DINOV3_LOCATION", "facebookresearch/dinov3")
    dinov3_source = "local" if dinov3_location != "facebookresearch/dinov3" else "github"
    if dinov3_location != "facebookresearch/dinov3" and dinov3_location not in sys.path:
        sys.path.append(dinov3_location)
    return dinov3_location, dinov3_source


def _should_use_geoai_sat_fallback(model_name: str, hub_weights_name: str) -> bool:
    return model_name == "dinov3_vitl16" and hub_weights_name == "SAT493M"


def _load_backbone_from_official_dinov3_hub(
    *,
    model_name: str,
    hub_weights_name: str,
    device: torch.device | None = None,
) -> torch.nn.Module:
    dinov3_location, dinov3_source = resolve_dinov3_hub_location()
    weights_url = build_official_dinov3_weights_url(model_name, hub_weights_name)
    try:
        model = torch.hub.load(
            repo_or_dir=dinov3_location,
            model=model_name,
            source=dinov3_source,
            pretrained=True,
            weights=weights_url,
            trust_repo=True,
            skip_validation=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to auto-resolve {model_name} with official DINOv3 {hub_weights_name} weights ({weights_url}). "
            "Set GEOAI_DINOV3_WEIGHTS_PATH to a compatible local .pth if this environment cannot reach the upstream checkpoint."
        ) from exc

    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def configure_geoai_dinov3_weight_loaders(metadata: dict[str, object] | None = None) -> None:
    import importlib
    import geoai.dinov3 as dinov3_module
    import geoai.dinov3_finetune as finetune_module

    global _ORIGINAL_DINOV3_PROCESSOR_LOAD_MODEL, _ORIGINAL_DINOV3_SEGMENTER_LOAD_BACKBONE

    def _is_plan6068_patch(func: object) -> bool:
        return bool(getattr(func, "_plan6068_patched", False))

    def _resolve_original_loader(
        owner: object,
        *,
        attr_name: str,
        storage_attr: str,
        cached: object | None,
    ) -> object | None:
        stored = getattr(owner, storage_attr, None)
        if callable(stored):
            return stored

        current = getattr(owner, attr_name)
        if callable(cached) and not _is_plan6068_patch(cached):
            original = cached
        elif callable(current) and not _is_plan6068_patch(current):
            original = current
        else:
            return None

        setattr(owner, storage_attr, original)
        return original

    hub_weights_name = resolve_backbone_hub_weights_name(metadata)
    processor_original = _resolve_original_loader(
        dinov3_module.DINOv3GeoProcessor,
        attr_name="_load_model",
        storage_attr="_plan6068_original_load_model",
        cached=_ORIGINAL_DINOV3_PROCESSOR_LOAD_MODEL,
    )
    segmenter_original = _resolve_original_loader(
        finetune_module.DINOv3Segmenter,
        attr_name="_load_backbone",
        storage_attr="_plan6068_original_load_backbone",
        cached=_ORIGINAL_DINOV3_SEGMENTER_LOAD_BACKBONE,
    )

    if processor_original is None or segmenter_original is None:
        dinov3_module = importlib.reload(dinov3_module)
        finetune_module = importlib.reload(finetune_module)
        processor_original = dinov3_module.DINOv3GeoProcessor._load_model
        segmenter_original = finetune_module.DINOv3Segmenter._load_backbone
        dinov3_module.DINOv3GeoProcessor._plan6068_original_load_model = processor_original
        finetune_module.DINOv3Segmenter._plan6068_original_load_backbone = segmenter_original

    _ORIGINAL_DINOV3_PROCESSOR_LOAD_MODEL = processor_original
    _ORIGINAL_DINOV3_SEGMENTER_LOAD_BACKBONE = segmenter_original

    def _patched_load_model(self, weights_path: str | None = None) -> torch.nn.Module:
        if weights_path:
            candidate = Path(weights_path)
            if not candidate.exists():
                raise RuntimeError(f"Configured DINOv3 weights_path not found: {candidate}")
            return _ORIGINAL_DINOV3_PROCESSOR_LOAD_MODEL(self, str(candidate))

        if _should_use_geoai_sat_fallback(self.model_name, hub_weights_name):
            return _ORIGINAL_DINOV3_PROCESSOR_LOAD_MODEL(self, None)

        return _load_backbone_from_official_dinov3_hub(
            model_name=self.model_name,
            hub_weights_name=hub_weights_name,
            device=self.device,
        )

    def _patched_load_backbone(model_name: str, weights_path: str | None) -> torch.nn.Module:
        if weights_path:
            candidate = Path(weights_path)
            if not candidate.exists():
                raise RuntimeError(f"Configured DINOv3 weights_path not found: {candidate}")
            return _ORIGINAL_DINOV3_SEGMENTER_LOAD_BACKBONE(model_name, str(candidate))

        if _should_use_geoai_sat_fallback(model_name, hub_weights_name):
            return _ORIGINAL_DINOV3_SEGMENTER_LOAD_BACKBONE(model_name, None)

        return _load_backbone_from_official_dinov3_hub(
            model_name=model_name,
            hub_weights_name=hub_weights_name,
        )

    _patched_load_model._plan6068_patched = True
    _patched_load_backbone._plan6068_patched = True

    dinov3_module.DINOv3GeoProcessor._load_model = _patched_load_model
    dinov3_module.DINOv3GeoProcessor._plan6068_hub_weights = hub_weights_name
    finetune_module.DINOv3Segmenter._load_backbone = staticmethod(_patched_load_backbone)
    finetune_module.DINOv3Segmenter._plan6068_hub_weights = hub_weights_name


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def normalize_dinov3_image_tensor(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(INPUT_NORMALIZATION_MEAN, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.tensor(INPUT_NORMALIZATION_STD, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    channels = min(image.shape[0], mean.shape[0])
    normalized = image.clone()
    normalized[:channels] = (normalized[:channels] - mean[:channels]) / std[:channels]
    return normalized


def build_dinov3_segmentation_transform():
    def _transform(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return normalize_dinov3_image_tensor(image.float()), mask

    return _transform


def resolve_runtime_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_torch_gpu_performance() -> None:
    valid_matmul_precision = {"highest", "high", "medium"}
    if FLOAT32_MATMUL_PRECISION not in valid_matmul_precision:
        raise RuntimeError(
            "GEOAI_DINOV3_FLOAT32_MATMUL_PRECISION must be one of "
            f"{sorted(valid_matmul_precision)}; got {FLOAT32_MATMUL_PRECISION!r}."
        )
    torch.set_float32_matmul_precision(FLOAT32_MATMUL_PRECISION)

    if not torch.cuda.is_available():
        print(
            "GPU tuning: CUDA unavailable; using CPU-safe settings "
            f"precision={LIGHTNING_PRECISION} | matmul_precision={FLOAT32_MATMUL_PRECISION}"
        )
        return

    torch.backends.cuda.matmul.allow_tf32 = ENABLE_TF32
    torch.backends.cudnn.allow_tf32 = ENABLE_TF32
    torch.backends.cudnn.benchmark = ENABLE_CUDNN_BENCHMARK
    if CUDA_EMPTY_CACHE_BEFORE_TRAINING:
        torch.cuda.empty_cache()

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(
        "GPU tuning: "
        f"{gpu_name} ({total_vram_gb:.1f} GB) | precision={LIGHTNING_PRECISION} | "
        f"matmul_precision={FLOAT32_MATMUL_PRECISION} | tf32={ENABLE_TF32} | "
        f"cudnn_benchmark={ENABLE_CUDNN_BENCHMARK}"
    )


def load_dinov3_segmenter(checkpoint_path: Path, metadata: dict[str, object], device: torch.device):
    from geoai.dinov3_finetune import DINOv3Segmenter

    configure_geoai_dinov3_weight_loaders(metadata=metadata)

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


def build_dinov3_inference_runtime(checkpoint_path: Path, metadata: dict[str, object]) -> dict[str, object]:
    device = resolve_runtime_device()
    model_module = load_dinov3_segmenter(checkpoint_path, metadata, device)
    return {
        "device": device,
        "model_module": model_module,
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
    progress_desc: str = "DINOv3 windows",
) -> None:
    from rasterio.windows import Window
    from tqdm import tqdm

    if overlap >= window_size:
        raise ValueError(f"overlap ({overlap}) must be less than window_size ({window_size})")

    if runtime is None:
        device = resolve_runtime_device()
        model_module = load_dinov3_segmenter(checkpoint_path, metadata, device)
    else:
        device = runtime["device"]
        model_module = runtime["model_module"]
    patch_size = int(model_module.patch_size)

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
                pad_arr = np.zeros((num_channels, img.shape[1], img.shape[2]), dtype=np.float32)
                pad_arr[: img.shape[0]] = img
                img = pad_arr

            if img.max() > 1.0:
                img = img / 255.0

            h, w = img.shape[1], img.shape[2]
            if h < padded_h or w < padded_w:
                padded = np.zeros((num_channels, padded_h, padded_w), dtype=np.float32)
                padded[:, :h, :w] = img
                img = padded

            img_tensor = normalize_dinov3_image_tensor(torch.from_numpy(img)).cpu()
            return img_tensor.numpy(), h, w

        def _flush_batch(batch_imgs: list[np.ndarray], batch_meta: list[tuple[int, int, int, int, int, int]]) -> None:
            tensor = torch.from_numpy(np.stack(batch_imgs)).to(device)
            logits = model_module(tensor)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

            for index, (row_start, row_end, col_start, col_end, h, w) in enumerate(batch_meta):
                votes[:, row_start:row_end, col_start:col_end] += probs[index, :, :h, :w]
                count[row_start:row_end, col_start:col_end] += 1.0

        with torch.no_grad():
            batch_imgs: list[np.ndarray] = []
            batch_meta: list[tuple[int, int, int, int, int, int]] = []
            progress = tqdm(total=n_rows * n_cols, disable=False, desc=progress_desc)

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


def refresh_runtime_paths() -> None:
    global MODEL_OUT, PREVIEW_ROOT, EVAL_ROOT, METADATA_PATH, EVAL_METRICS_CSV, EVAL_BUILDING_METRICS_CSV, EVAL_SUMMARY_JSON
    resume_model_dir = resolve_resume_model_dir() if RESUME_TRAINING else None
    if EXPLICIT_MODEL_OUT is not None:
        model_out = EXPLICIT_MODEL_OUT
    elif resume_model_dir is not None:
        model_out = resume_model_dir
    else:
        mode_slug = f"lora-r{LORA_RANK}" if USE_LORA else "frozen"
        data_slug = training_contract_run_fragment(TRAINING_CONTRACT)
        model_out = BASE_MODEL_ROOT / (
            f"{_slugify(BACKBONE_PRESET)}__hub-{_slugify(MODEL_NAME)}__df{DECODER_FEATURES}__ps{PATCH_SIZE}"
            f"__e{NUM_EPOCHS}__b{BATCH_SIZE}__lr{_float_slug(LEARNING_RATE)}__{mode_slug}__{data_slug}"
        )
    MODEL_OUT = model_out
    PREVIEW_ROOT = EXPLICIT_PREVIEW_ROOT or MODEL_OUT / "preview"
    EVAL_ROOT = EXPLICIT_EVAL_ROOT or MODEL_OUT / "evaluation"
    METADATA_PATH = MODEL_OUT / "dinov3_metadata.json"
    EVAL_METRICS_CSV = EVAL_ROOT / "split_metrics.csv"
    EVAL_BUILDING_METRICS_CSV = EVAL_ROOT / "holdout_building_metrics.csv"
    EVAL_SUMMARY_JSON = EVAL_ROOT / "split_summary.json"


def resolve_resume_model_dir() -> Path | None:
    if RESUME_RUN_DIR is not None:
        return RESUME_RUN_DIR
    if RESUME_CHECKPOINT is not None:
        return RESUME_CHECKPOINT.parent if RESUME_CHECKPOINT.is_file() else RESUME_CHECKPOINT
    return None


def resolve_resume_checkpoint() -> Path | None:
    if not RESUME_TRAINING:
        return None
    if RESUME_CHECKPOINT is not None:
        return RESUME_CHECKPOINT

    model_dir = resolve_resume_model_dir() or MODEL_OUT
    candidates = [
        model_dir / "models" / "last.ckpt",
        model_dir / "last.ckpt",
        model_dir / "models" / "final.ckpt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    ckpt_candidates = sorted(model_dir.rglob("*.ckpt"), key=lambda path: (-path.stat().st_mtime, path.name))
    if ckpt_candidates:
        return ckpt_candidates[0]

    weight_candidates = sorted(
        [path for path in model_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".pth", ".pt"}],
        key=lambda path: (-path.stat().st_mtime, path.name),
    )
    return weight_candidates[0] if weight_candidates else None


def apply_manifest_training_contract(manifest: pd.DataFrame) -> None:
    global TRAINING_CONTRACT
    TRAINING_CONTRACT = build_training_contract(
        manifest,
        train_root=TRAIN_ROOT,
        manifest_path=TRAIN_MANIFEST,
        project_root=PROJECT_ROOT,
    )
    refresh_runtime_paths()


def normalize_model_identity_value(value: object) -> object:
    if isinstance(value, tuple):
        return list(value)
    return value


def current_model_identity() -> dict[str, object]:
    return {
        "backbone_preset": BACKBONE_PRESET,
        "model_name": MODEL_NAME,
        "hub_weights": BACKBONE_HUB_WEIGHTS,
        "weights_path": _display_path(resolve_backbone_weights_path()),
        "input_normalization_profile": INPUT_NORMALIZATION_PROFILE,
        "num_classes": NUM_CLASSES,
        "decoder_features": DECODER_FEATURES,
        "patch_size": PATCH_SIZE,
        "target_size": TARGET_SIZE,
        "freeze_backbone": FREEZE_BACKBONE,
        "use_lora": USE_LORA,
        "lora_rank": LORA_RANK,
    }


def load_saved_dinov3_metadata(model_dir: Path) -> dict[str, object]:
    metadata_path = model_dir / "dinov3_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text())


def _format_resume_mismatches(mismatches: dict[str, dict[str, object]]) -> str:
    return "; ".join(
        f"{key}: current={values['current']!r} saved={values['saved']!r}"
        for key, values in sorted(mismatches.items())
    )


def ensure_resume_compatible() -> Path | None:
    if not RESUME_TRAINING:
        return None

    resume_checkpoint = resolve_resume_checkpoint()
    if resume_checkpoint is None or not resume_checkpoint.exists():
        raise FileNotFoundError(
            "resume training requested, but no DINOv3 checkpoint was found. "
            "Set GEOAI_DINOV3_RESUME_CHECKPOINT or GEOAI_DINOV3_RESUME_RUN_DIR."
        )

    resume_model_dir = resolve_resume_model_dir() or resume_checkpoint.parent.parent if resume_checkpoint.parent.name == "models" else resume_checkpoint.parent
    metadata = load_saved_dinov3_metadata(resume_model_dir)
    if metadata:
        saved_model_identity = {
            key: normalize_model_identity_value(metadata.get(key))
            for key in (
                "backbone_preset",
                "model_name",
                "hub_weights",
                "weights_path",
                "input_normalization_profile",
                "num_classes",
                "decoder_features",
                "patch_size",
                "target_size",
                "freeze_backbone",
                "use_lora",
                "lora_rank",
            )
            if key in metadata
        }
        model_mismatches = {
            key: {
                "current": normalize_model_identity_value(current_model_identity()[key]),
                "saved": saved_model_identity[key],
            }
            for key in saved_model_identity
            if normalize_model_identity_value(current_model_identity()[key]) != saved_model_identity[key]
        }
        if model_mismatches:
            raise RuntimeError(
                "DINOv3 resume configuration does not match the saved model identity: "
                + _format_resume_mismatches(model_mismatches)
            )

        saved_contract = metadata.get("training_contract")
        if isinstance(saved_contract, dict):
            contract_mismatches = compare_training_contracts(TRAINING_CONTRACT, saved_contract)
            if contract_mismatches:
                raise RuntimeError(
                    "DINOv3 resume configuration does not match the saved training contract: "
                    + _format_resume_mismatches(contract_mismatches)
                )
        else:
            metadata["training_contract"] = TRAINING_CONTRACT
            metadata["legacy_training_contract_inferred"] = True
            (resume_model_dir / "dinov3_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    return resume_checkpoint

# %% [markdown]
# ## Notebook Controls
# 
# These controls expose the most impactful DINOv3 sweep parameters without
# changing the safe script defaults. GeoAI's built-in DINOv3 helpers still
# hardcode a ViT-L SAT fallback when `weights_path` is omitted, so this notebook
# patches the non-default presets to request the matching official DINOv3 weight
# family instead.

# %%
DINOV3_MODEL_CHOICES = [
    (preset["label"], preset_key)
    for preset_key, preset in DINOV3_BACKBONE_PRESETS.items()
]
DINOV3_LEARNING_RATE_CHOICES = [5e-5, 1e-4, 2e-4, 5e-4]
DINOV3_DECODER_FEATURE_CHOICES = [128, 256, 384, 512]
DINOV3_PATCH_SIZE_CHOICES = [16, 32]
DINOV3_LORA_RANK_CHOICES = [4, 8, 16]
DINOV3_NOTEBOOK_WIDGETS: dict[str, object] | None = None


def _maybe_initialize_dinov3_widgets() -> dict[str, object] | None:
    global DINOV3_NOTEBOOK_WIDGETS
    if DINOV3_NOTEBOOK_WIDGETS is not None:
        return DINOV3_NOTEBOOK_WIDGETS
    if not running_in_notebook():
        return None

    try:
        import ipywidgets as widgets
        from IPython.display import Markdown, display
    except ImportError:
        return None

    backbone_preset_widget = widgets.Dropdown(
        options=DINOV3_MODEL_CHOICES,
        value=BACKBONE_PRESET,
        description="Backbone",
        layout=widgets.Layout(width="420px"),
    )
    weights_path_widget = widgets.Text(
        value=str(BACKBONE_WEIGHTS_PATH) if BACKBONE_WEIGHTS_PATH is not None else "",
        description="Weights",
        placeholder="Optional local .pth path",
        layout=widgets.Layout(width="620px"),
    )
    learning_rate_widget = widgets.Dropdown(
        options=DINOV3_LEARNING_RATE_CHOICES,
        value=LEARNING_RATE,
        description="LR",
        layout=widgets.Layout(width="220px"),
    )
    num_epochs_widget = widgets.BoundedIntText(
        value=NUM_EPOCHS,
        min=1,
        max=500,
        description="Epochs",
        layout=widgets.Layout(width="220px"),
    )
    batch_size_widget = widgets.BoundedIntText(
        value=BATCH_SIZE,
        min=1,
        max=256,
        description="Batch",
        layout=widgets.Layout(width="220px"),
    )
    decoder_features_widget = widgets.Dropdown(
        options=DINOV3_DECODER_FEATURE_CHOICES,
        value=DECODER_FEATURES,
        description="Decoder",
        layout=widgets.Layout(width="220px"),
    )
    patch_size_widget = widgets.Dropdown(
        options=DINOV3_PATCH_SIZE_CHOICES,
        value=PATCH_SIZE,
        description="Patch pad",
        layout=widgets.Layout(width="220px"),
    )
    use_lora_widget = widgets.Checkbox(
        value=USE_LORA,
        description="Enable LoRA",
        indent=False,
    )
    lora_rank_widget = widgets.Dropdown(
        options=DINOV3_LORA_RANK_CHOICES,
        value=LORA_RANK,
        description="LoRA rank",
        layout=widgets.Layout(width="220px"),
    )
    run_training_widget = widgets.Checkbox(
        value=RUN_DINOV3_TRAINING,
        description="Run training",
        indent=False,
    )
    run_preview_widget = widgets.Checkbox(
        value=RUN_DINOV3_PREVIEW,
        description="Run preview",
        indent=False,
    )
    run_eval_widget = widgets.Checkbox(
        value=RUN_DINOV3_EVAL,
        description="Run eval",
        indent=False,
    )
    run_similarity_widget = widgets.Checkbox(
        value=RUN_DINOV3_SIMILARITY_PREVIEW,
        description="Run similarity preview",
        indent=False,
    )
    preview_split_widget = widgets.Dropdown(
        options=["train", "val", "test"],
        value=PREVIEW_SPLIT,
        description="Preview split",
        layout=widgets.Layout(width="220px"),
    )

    def _toggle_lora_rank(change) -> None:
        lora_rank_widget.disabled = not bool(change["new"])

    use_lora_widget.observe(_toggle_lora_rank, names="value")
    _toggle_lora_rank({"new": use_lora_widget.value})

    display(
        Markdown(
            "**Backbone note**: GeoAI's current DINOv3 loaders always fall back to the ViT-L SAT-493M checkpoint when `weights_path` is omitted. "
            "This notebook keeps that working path for `vitl16_sat493m`, but for `vit7b16_sat493m` and the LVD presets it now requests the matching official DINOv3 checkpoint URL before asking for a local `.pth`."
        )
    )
    display(
        Markdown(
            "**LoRA guidance**: keep `use_lora=False` for the first baseline. "
            "Enable LoRA when larger DINOv3 variants are memory-constrained or when the frozen-backbone run stops improving."
        )
    )
    display(
        widgets.VBox(
            [
                widgets.HBox([backbone_preset_widget, learning_rate_widget]),
                weights_path_widget,
                widgets.HBox([num_epochs_widget, batch_size_widget, preview_split_widget]),
                widgets.HBox([decoder_features_widget, patch_size_widget]),
                widgets.HBox([use_lora_widget, lora_rank_widget]),
                widgets.HBox([run_training_widget, run_preview_widget, run_eval_widget, run_similarity_widget]),
            ]
        )
    )

    DINOV3_NOTEBOOK_WIDGETS = {
        "backbone_preset": backbone_preset_widget,
        "weights_path": weights_path_widget,
        "learning_rate": learning_rate_widget,
        "num_epochs": num_epochs_widget,
        "batch_size": batch_size_widget,
        "decoder_features": decoder_features_widget,
        "patch_size": patch_size_widget,
        "use_lora": use_lora_widget,
        "lora_rank": lora_rank_widget,
        "run_training": run_training_widget,
        "run_preview": run_preview_widget,
        "run_eval": run_eval_widget,
        "run_similarity": run_similarity_widget,
        "preview_split": preview_split_widget,
    }
    return DINOV3_NOTEBOOK_WIDGETS


def apply_dinov3_widget_overrides() -> None:
    global BACKBONE_PRESET, BACKBONE_WEIGHTS_PATH, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE
    global DECODER_FEATURES, PATCH_SIZE, USE_LORA, LORA_RANK
    global RUN_DINOV3_TRAINING, RUN_DINOV3_PREVIEW, RUN_DINOV3_EVAL, RUN_DINOV3_SIMILARITY_PREVIEW, PREVIEW_SPLIT, DINOV3_SIMILARITY_SPLIT
    widgets = _maybe_initialize_dinov3_widgets()
    if widgets is not None:
        BACKBONE_PRESET = str(widgets["backbone_preset"].value)
        BACKBONE_WEIGHTS_PATH = _resolve_optional_path(str(widgets["weights_path"].value).strip() or None)
        LEARNING_RATE = float(widgets["learning_rate"].value)
        NUM_EPOCHS = int(widgets["num_epochs"].value)
        BATCH_SIZE = int(widgets["batch_size"].value)
        DECODER_FEATURES = int(widgets["decoder_features"].value)
        PATCH_SIZE = int(widgets["patch_size"].value)
        USE_LORA = bool(widgets["use_lora"].value)
        LORA_RANK = int(widgets["lora_rank"].value)
        RUN_DINOV3_TRAINING = bool(widgets["run_training"].value)
        RUN_DINOV3_PREVIEW = bool(widgets["run_preview"].value)
        RUN_DINOV3_EVAL = bool(widgets["run_eval"].value)
        RUN_DINOV3_SIMILARITY_PREVIEW = bool(widgets["run_similarity"].value)
        PREVIEW_SPLIT = str(widgets["preview_split"].value)
        if DINOV3_SIMILARITY_SPLIT_OVERRIDE is None:
            DINOV3_SIMILARITY_SPLIT = PREVIEW_SPLIT
    apply_backbone_settings()
    refresh_runtime_paths()


def validate_dinov3_runtime_configuration() -> None:
    get_backbone_preset()
    if MODEL_NAME not in DINOV3_ALLOWED_HUB_MODELS:
        raise RuntimeError(
            f"Resolved DINOv3 hub model {MODEL_NAME!r} is unsupported; expected one of {sorted(DINOV3_ALLOWED_HUB_MODELS)}"
        )
    if PATCH_SIZE < 16 or PATCH_SIZE % 16 != 0:
        raise RuntimeError(
            f"PATCH_SIZE must be a multiple of 16 for the supported DINOv3 backbones; got {PATCH_SIZE}."
        )
    if USE_LORA and LORA_RANK < 1:
        raise RuntimeError(f"LORA_RANK must be >= 1 when USE_LORA is enabled; got {LORA_RANK}.")
    if BATCH_SIZE < 1:
        raise RuntimeError(f"BATCH_SIZE must be >= 1; got {BATCH_SIZE}.")
    if NUM_EPOCHS < 1:
        raise RuntimeError(f"NUM_EPOCHS must be >= 1; got {NUM_EPOCHS}.")
    resolve_backbone_weights_path()


apply_backbone_settings()
refresh_runtime_paths()
_maybe_initialize_dinov3_widgets()


def assign_dataset_split(key: object) -> str:
    digest = hashlib.sha1(str(key).encode("utf-8")).hexdigest()
    fraction = int(digest[:12], 16) / float((16**12) - 1)
    if fraction < 0.70:
        return "train"
    if fraction < 0.85:
        return "val"
    return "test"


def build_manifest_from_training_root() -> pd.DataFrame:
    if not IMAGES_DIR.exists() or not GROUNDED_MASKS_DIR.exists():
        raise FileNotFoundError(
            f"training directories not found under {TRAIN_ROOT}; run 11_geoai_training_data.py first"
        )

    image_files = {path.stem: path for path in sorted(IMAGES_DIR.iterdir()) if path.is_file()}
    grounded_mask_files = {path.stem: path for path in sorted(GROUNDED_MASKS_DIR.iterdir()) if path.is_file()}
    raw_mask_files = {path.stem: path for path in sorted(RAW_MASKS_DIR.iterdir()) if path.is_file()} if RAW_MASKS_DIR.exists() else {}
    prompt_artifact_files = {path.stem: path for path in sorted(PROMPT_ARTIFACTS_DIR.iterdir()) if path.is_file()} if PROMPT_ARTIFACTS_DIR.exists() else {}
    common_stems = sorted(set(image_files) & set(grounded_mask_files))
    if not common_stems:
        raise RuntimeError(f"no grounded image/mask pairs found under {TRAIN_ROOT}")

    rows = []
    for stem in common_stems:
        image_path = image_files[stem]
        grounded_mask_path = grounded_mask_files[stem]
        raw_mask_path = raw_mask_files.get(stem)
        prompt_artifact_path = prompt_artifact_files.get(stem)
        rows.append(
            {
                "tile_id": stem,
                "image_path": str(image_path.relative_to(PROJECT_ROOT)),
                "grounded_mask_path": str(grounded_mask_path.relative_to(PROJECT_ROOT)),
                "raw_mask_path": str(raw_mask_path.relative_to(PROJECT_ROOT)) if raw_mask_path is not None else None,
                "prompt_artifact_path": str(prompt_artifact_path.relative_to(PROJECT_ROOT)) if prompt_artifact_path is not None else None,
                "dataset_split": assign_dataset_split(stem),
                "matched_building_count": None,
                "grounded_positive_pixels": None,
            }
        )
    return pd.DataFrame.from_records(rows)


def load_grounded_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else build_manifest_from_training_root()
    required = {"image_path", "grounded_mask_path", "dataset_split"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise RuntimeError(
            "grounded manifest is missing required columns: " + ", ".join(missing)
        )

    manifest = manifest.copy()
    manifest["image_abs_path"] = manifest["image_path"].map(lambda value: (PROJECT_ROOT / str(value)).resolve())
    manifest["grounded_mask_abs_path"] = manifest["grounded_mask_path"].map(
        lambda value: (PROJECT_ROOT / str(value)).resolve()
    )
    raw_mask_column = "raw_mask_path" if "raw_mask_path" in manifest.columns else "mask_path"
    manifest["raw_mask_abs_path"] = manifest[raw_mask_column].map(
        lambda value: (PROJECT_ROOT / str(value)).resolve() if isinstance(value, str) and value else None
    )
    prompt_artifact_column = "prompt_artifact_path" if "prompt_artifact_path" in manifest.columns else None
    manifest["prompt_artifact_abs_path"] = manifest[prompt_artifact_column].map(
        lambda value: (PROJECT_ROOT / str(value)).resolve() if isinstance(value, str) and value else None
    ) if prompt_artifact_column else None
    manifest["image_stem"] = manifest["image_abs_path"].map(lambda path: Path(path).stem)
    manifest["dataset_split"] = manifest["dataset_split"].fillna("train").astype(str).str.lower()

    exists_mask = manifest["image_abs_path"].map(Path.exists) & manifest["grounded_mask_abs_path"].map(Path.exists)
    manifest = manifest[exists_mask].reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError("grounded manifest contains no usable image/grounded-mask pairs.")
    return manifest


def summarize_manifest(manifest: pd.DataFrame) -> None:
    summary = (
        manifest.groupby("dataset_split")
        .agg(
            chip_count=("tile_id", "count"),
            matched_buildings=("matched_building_count", "sum"),
            grounded_pixels=("grounded_positive_pixels", "sum"),
        )
        .reset_index()
    )
    print("grounded manifest summary:")
    print(summary.to_string(index=False))


def infer_num_channels(image_path: Path) -> int:
    with rasterio.open(image_path) as src:
        return src.count


def build_split_dataset(geoai_module, manifest: pd.DataFrame, split_name: str):
    subset = manifest[manifest["dataset_split"] == split_name].reset_index(drop=True)
    if subset.empty:
        return None
    return geoai_module.DINOv3SegmentationDataset(
        image_paths=[str(path) for path in subset["image_abs_path"]],
        mask_paths=[str(path) for path in subset["grounded_mask_abs_path"]],
        patch_size=PATCH_SIZE,
        target_size=TARGET_SIZE,
        num_channels=min(3, infer_num_channels(Path(subset.iloc[0]["image_abs_path"]))),
        transform=build_dinov3_segmentation_transform(),
    )


def _unpack_dataset_sample(sample: object) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if isinstance(sample, dict):
        image = sample.get("image")
        mask = sample.get("mask")
        if mask is None:
            mask = sample.get("label")
        if mask is None:
            mask = sample.get("target")
        return image, mask
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        image = sample[0]
        mask = sample[1]
        return image, mask
    return None, None


def preview_transformed_samples(geoai_module, manifest: pd.DataFrame) -> None:
    if not RUN_DINOV3_TRANSFORM_PREVIEW or DINOV3_TRANSFORM_PREVIEW_COUNT <= 0:
        return

    import matplotlib.pyplot as plt

    preview_root = PREVIEW_ROOT / "transform_preview"
    preview_root.mkdir(parents=True, exist_ok=True)
    per_split = max(1, DINOV3_TRANSFORM_PREVIEW_COUNT // 3)

    for split_name in ("train", "val", "test"):
        subset = manifest[manifest["dataset_split"] == split_name].reset_index(drop=True)
        if subset.empty:
            continue
        sample_rows = subset.sample(n=min(per_split, len(subset)), random_state=SEED).reset_index(drop=True)
        dataset = geoai_module.DINOv3SegmentationDataset(
            image_paths=[str(path) for path in sample_rows["image_abs_path"]],
            mask_paths=[str(path) for path in sample_rows["grounded_mask_abs_path"]],
            patch_size=PATCH_SIZE,
            target_size=TARGET_SIZE,
            num_channels=min(3, infer_num_channels(Path(sample_rows.iloc[0]["image_abs_path"]))),
            transform=build_dinov3_segmentation_transform(),
        )

        for sample_index in range(len(dataset)):
            image_tensor, mask_tensor = _unpack_dataset_sample(dataset[sample_index])
            if image_tensor is None or mask_tensor is None:
                continue

            image = image_tensor.detach().cpu().numpy()
            if image.ndim == 3:
                image = np.moveaxis(image, 0, -1)
            if image.shape[-1] > 3:
                image = image[..., :3]
            image = image.astype(np.float32)
            image_min = float(np.nanmin(image))
            image_max = float(np.nanmax(image))
            if image_max > image_min:
                image = (image - image_min) / (image_max - image_min)

            mask = mask_tensor.detach().cpu().numpy()
            if mask.ndim == 3:
                mask = np.argmax(mask, axis=0)
            mask = mask.astype(np.uint8)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
            axes = axes[0]
            axes[0].imshow(image)
            axes[0].set_title(f"{split_name} transformed image")
            axes[0].set_axis_off()
            axes[1].imshow(mask, cmap="gray", vmin=0, vmax=max(1, NUM_CLASSES - 1))
            axes[1].set_title(f"{split_name} transformed mask")
            axes[1].set_axis_off()
            fig.tight_layout()
            tile_id = str(sample_rows.iloc[sample_index]["tile_id"])
            fig_path = preview_root / f"{split_name}_{tile_id}_transform_preview.png"
            fig.savefig(fig_path, dpi=180, bbox_inches="tight")
            plt.close(fig)

    print(f"wrote transform previews under {preview_root.relative_to(PROJECT_ROOT)}")


def _normalize_preview_display_image(data: np.ndarray) -> np.ndarray:
    if data.ndim == 3:
        if data.shape[0] <= 3:
            display_img = np.transpose(data, (1, 2, 0))
        else:
            display_img = np.transpose(data[:3], (1, 2, 0))
    else:
        display_img = data

    if display_img.dtype == np.uint8:
        return display_img.astype(np.float32) / 255.0

    display_img = display_img.astype(np.float32)
    if display_img.ndim == 2:
        finite = np.isfinite(display_img)
        if not finite.any():
            return np.zeros_like(display_img, dtype=np.float32)
        p2, p98 = np.percentile(display_img[finite], [2, 98])
        if p98 > p2:
            return np.clip((display_img - p2) / (p98 - p2), 0, 1)
        scale = 255.0 if float(np.nanmax(display_img[finite])) > 1.0 else 1.0
        return np.clip(display_img / scale, 0, 1)

    normalized = np.zeros_like(display_img, dtype=np.float32)
    for band_index in range(display_img.shape[2]):
        band = display_img[:, :, band_index]
        finite = np.isfinite(band)
        if not finite.any():
            continue
        p2, p98 = np.percentile(band[finite], [2, 98])
        if p98 > p2:
            normalized[:, :, band_index] = np.clip((band - p2) / (p98 - p2), 0, 1)
        else:
            scale = 255.0 if float(np.nanmax(band[finite])) > 1.0 else 1.0
            normalized[:, :, band_index] = np.clip(band / scale, 0, 1)
    return normalized


def _load_positive_prompt_examples(prompt_artifact_path: Path) -> list[dict[str, object]]:
    payload = json.loads(prompt_artifact_path.read_text())
    prompts = payload.get("prompts") or []
    positive_prompts: list[dict[str, object]] = []
    for prompt in prompts:
        matched_label_count = int(prompt.get("matched_label_count") or 0)
        centroid = prompt.get("building_centroid_pixels") or prompt.get("centroid_pixels")
        if matched_label_count < 1:
            continue
        if not isinstance(centroid, (list, tuple)) or len(centroid) != 2:
            continue
        positive_prompts.append(prompt)
    return positive_prompts


def select_similarity_rows(
    manifest: pd.DataFrame,
    split_name: str,
    count: int,
) -> list[tuple[pd.Series, dict[str, object]]]:
    if count <= 0:
        return []
    if split_name == "all":
        subset = manifest.copy()
    else:
        subset = manifest[manifest["dataset_split"] == split_name].copy()
    if subset.empty:
        return []

    rng = random.Random(SEED)
    indices = list(subset.index)
    rng.shuffle(indices)
    selected: list[tuple[pd.Series, dict[str, object]]] = []
    for index in indices:
        row = subset.loc[index]
        prompt_path = row.get("prompt_artifact_abs_path")
        if prompt_path is None:
            continue
        prompt_artifact_path = Path(prompt_path)
        if not prompt_artifact_path.exists():
            continue
        try:
            positive_prompts = _load_positive_prompt_examples(prompt_artifact_path)
        except Exception as exc:
            print(f"similarity preview skipped prompt artifact {prompt_artifact_path.name}: {exc}")
            continue
        if not positive_prompts:
            continue
        selected.append((row, positive_prompts[0]))
        if len(selected) >= count:
            break
    return selected


def preview_similarity_rows(manifest: pd.DataFrame) -> None:
    if not RUN_DINOV3_SIMILARITY_PREVIEW or DINOV3_SIMILARITY_PREVIEW_COUNT <= 0:
        return

    preview_examples = select_similarity_rows(
        manifest,
        DINOV3_SIMILARITY_SPLIT,
        DINOV3_SIMILARITY_PREVIEW_COUNT,
    )
    if not preview_examples:
        print(
            "DINOv3 similarity preview skipped: no OSM-positive prompt artifacts found for "
            f"split={DINOV3_SIMILARITY_SPLIT!r}."
        )
        return

    metadata = load_metadata()
    configure_geoai_dinov3_weight_loaders(metadata=metadata)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from geoai.dinov3 import DINOv3GeoProcessor, visualize_similarity_results
    except Exception as exc:
        print(f"DINOv3 similarity preview skipped: {exc}")
        return

    weights_path_value = metadata.get("weights_path")
    weights_path = None
    if isinstance(weights_path_value, str) and weights_path_value:
        candidate = Path(weights_path_value)
        weights_path = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()

    try:
        processor = DINOv3GeoProcessor(
            model_name=str(metadata["model_name"]),
            weights_path=str(weights_path) if weights_path is not None else None,
        )
    except Exception as exc:
        print(f"DINOv3 similarity preview skipped: {exc}")
        return

    similarity_root = PREVIEW_ROOT / "similarity_preview"
    similarity_root.mkdir(parents=True, exist_ok=True)
    rendered_count = 0

    for row, prompt in preview_examples:
        image_path = Path(row["image_abs_path"])
        tile_id = str(row["tile_id"])
        query_coords = prompt.get("building_centroid_pixels") or prompt.get("centroid_pixels")
        if not isinstance(query_coords, (list, tuple)) or len(query_coords) != 2:
            continue

        tile_root = similarity_root / tile_id
        tile_root.mkdir(parents=True, exist_ok=True)
        quick_save_path = tile_root / f"{tile_id}_geoai_similarity.png"
        detail_save_path = tile_root / f"{tile_id}_osm_pv_similarity_detail.png"
        similarity_array_path = tile_root / f"{tile_id}_similarity.npy"
        summary_path = tile_root / f"{tile_id}_similarity_summary.json"

        try:
            results = visualize_similarity_results(
                input_image=str(image_path),
                query_coords=(float(query_coords[0]), float(query_coords[1])),
                output_dir=str(tile_root),
                model_name=str(metadata["model_name"]),
                weights_path=str(weights_path) if weights_path is not None else None,
                colormap=DINOV3_SIMILARITY_COLORMAP,
                alpha=DINOV3_SIMILARITY_ALPHA,
                save_path=str(quick_save_path),
                overlay=False,
                target_size=DINOV3_SIMILARITY_TARGET_SIZE,
            )
        except Exception as exc:
            print(f"DINOv3 similarity preview failed for {tile_id}: {exc}")
            continue

        similarity_data = np.asarray(results["image_dict"]["image"][0], dtype=np.float32)
        np.save(similarity_array_path, similarity_data)
        overlay_img = processor.create_similarity_overlay(
            source=str(image_path),
            similarity_data=similarity_data,
            colormap=DINOV3_SIMILARITY_COLORMAP,
            alpha=DINOV3_SIMILARITY_ALPHA,
        )
        image_data, _ = processor.load_image(str(image_path))
        display_img = _normalize_preview_display_image(image_data)

        matched_osm_label_ids = prompt.get("matched_osm_label_ids") or []
        matched_label_count = int(prompt.get("matched_label_count") or len(matched_osm_label_ids))
        bbox = prompt.get("building_bbox_pixels") or prompt.get("bbox_pixels")
        patch_coords = tuple(int(value) for value in results["patch_coords"])
        patch_grid_size = tuple(int(value) for value in results["patch_grid_size"])
        sim_min = float(np.nanmin(similarity_data))
        sim_max = float(np.nanmax(similarity_data))
        sim_max = sim_max if sim_max > sim_min else sim_min + 1e-6

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False)
        axes = axes[0]
        if display_img.ndim == 2:
            axes[0].imshow(display_img, cmap="gray")
        else:
            axes[0].imshow(display_img)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x_min, y_min, x_max, y_max = (float(value) for value in bbox)
            axes[0].add_patch(
                Rectangle(
                    (x_min, y_min),
                    max(1.0, x_max - x_min),
                    max(1.0, y_max - y_min),
                    fill=False,
                    edgecolor="cyan",
                    linewidth=2.0,
                )
            )
        axes[0].scatter([float(query_coords[0])], [float(query_coords[1])], c="yellow", s=42, marker="x")
        axes[0].set_title(
            "OSM-positive building\n"
            f"labels={matched_label_count} ids={matched_osm_label_ids[:4]}"
        )
        axes[0].set_axis_off()

        similarity_im = axes[1].imshow(
            similarity_data,
            cmap=DINOV3_SIMILARITY_COLORMAP,
            vmin=sim_min,
            vmax=sim_max,
        )
        axes[1].set_title(
            "Patch similarity\n"
            f"query_patch={patch_coords} grid={patch_grid_size[1]}x{patch_grid_size[0]}"
        )
        axes[1].set_axis_off()
        fig.colorbar(similarity_im, ax=axes[1], fraction=0.046, pad=0.04)

        if np.asarray(overlay_img).ndim == 2:
            axes[2].imshow(overlay_img, cmap="gray")
        else:
            axes[2].imshow(overlay_img)
        axes[2].set_title(f"{metadata['model_name']} overlay")
        axes[2].set_axis_off()

        fig.tight_layout()
        fig.savefig(detail_save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        quick_figure = results.get("visualization")
        if quick_figure is not None:
            plt.close(quick_figure)

        summary_payload = {
            "tile_id": tile_id,
            "dataset_split": str(row["dataset_split"]),
            "image_path": row["image_path"],
            "prompt_artifact_path": row.get("prompt_artifact_path"),
            "model_name": metadata["model_name"],
            "hub_weights": resolve_backbone_hub_weights_name(metadata),
            "weights_path": metadata.get("weights_path"),
            "query_coords": [float(query_coords[0]), float(query_coords[1])],
            "patch_coords": list(patch_coords),
            "patch_grid_size": list(patch_grid_size),
            "matched_label_count": matched_label_count,
            "matched_osm_label_ids": matched_osm_label_ids,
            "similarity_min": sim_min,
            "similarity_max": float(np.nanmax(similarity_data)),
            "quick_visualization_path": _display_path(quick_save_path),
            "detail_visualization_path": _display_path(detail_save_path),
            "similarity_array_path": _display_path(similarity_array_path),
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
        rendered_count += 1

    if rendered_count:
        print(
            f"wrote {rendered_count} DINOv3 similarity previews under "
            f"{_display_path(similarity_root)}"
        )
    else:
        print("DINOv3 similarity preview finished without any rendered examples.")


def find_latest_checkpoint(model_dir: Path) -> Path:
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


def collect_external_preview_rasters(exclude_stems: set[str]) -> list[Path]:
    return collect_naip_stac_preview_rasters(
        PROJECT_ROOT,
        exclude_stems=exclude_stems,
        supported_image_extensions=SUPPORTED_IMAGE_EXTENSIONS,
    )


def write_metadata(num_channels: int) -> None:
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    weights_path = resolve_backbone_weights_path()
    payload = {
        "run_name": MODEL_OUT.name,
        "run_naming_version": RUN_NAMING_VERSION,
        "backbone_preset": BACKBONE_PRESET,
        "model_name": MODEL_NAME,
        "hub_weights": BACKBONE_HUB_WEIGHTS,
        "weights_path": _display_path(weights_path),
        "backbone_note": BACKBONE_NOTE,
        "input_normalization_profile": INPUT_NORMALIZATION_PROFILE,
        "input_mean": list(INPUT_NORMALIZATION_MEAN),
        "input_std": list(INPUT_NORMALIZATION_STD),
        "num_classes": NUM_CLASSES,
        "decoder_features": DECODER_FEATURES,
        "num_channels": num_channels,
        "patch_size": PATCH_SIZE,
        "target_size": TARGET_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "lightning_precision": LIGHTNING_PRECISION,
        "float32_matmul_precision": FLOAT32_MATMUL_PRECISION,
        "enable_tf32": ENABLE_TF32,
        "cudnn_benchmark": ENABLE_CUDNN_BENCHMARK,
        "freeze_backbone": FREEZE_BACKBONE,
        "use_lora": USE_LORA,
        "lora_rank": LORA_RANK,
        "training_contract": TRAINING_CONTRACT,
        "resume_training": RESUME_TRAINING,
        "resume_checkpoint": _display_path(resolve_resume_checkpoint()),
    }
    METADATA_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def load_metadata() -> dict[str, object]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    weights_path = resolve_backbone_weights_path()
    return {
        "run_name": MODEL_OUT.name,
        "run_naming_version": RUN_NAMING_VERSION,
        "backbone_preset": BACKBONE_PRESET,
        "model_name": MODEL_NAME,
        "hub_weights": BACKBONE_HUB_WEIGHTS,
        "weights_path": _display_path(weights_path),
        "backbone_note": BACKBONE_NOTE,
        "input_normalization_profile": INPUT_NORMALIZATION_PROFILE,
        "input_mean": list(INPUT_NORMALIZATION_MEAN),
        "input_std": list(INPUT_NORMALIZATION_STD),
        "num_classes": NUM_CLASSES,
        "decoder_features": DECODER_FEATURES,
        "num_channels": 3,
        "patch_size": PATCH_SIZE,
        "target_size": TARGET_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "lightning_precision": LIGHTNING_PRECISION,
        "float32_matmul_precision": FLOAT32_MATMUL_PRECISION,
        "enable_tf32": ENABLE_TF32,
        "cudnn_benchmark": ENABLE_CUDNN_BENCHMARK,
        "freeze_backbone": FREEZE_BACKBONE,
        "use_lora": USE_LORA,
        "lora_rank": LORA_RANK,
        "training_contract": TRAINING_CONTRACT,
        "resume_training": RESUME_TRAINING,
        "resume_checkpoint": _display_path(resolve_resume_checkpoint()),
    }


def select_preview_rows(manifest: pd.DataFrame, split_name: str, count: int) -> pd.DataFrame:
    subset = manifest[manifest["dataset_split"] == split_name].copy()
    if subset.empty:
        return subset
    rng = random.Random(SEED)
    indices = list(subset.index)
    rng.shuffle(indices)
    chosen = indices[: min(count, len(indices))]
    return subset.loc[chosen].reset_index(drop=True)


def select_evaluation_rows(manifest: pd.DataFrame, split_names: tuple[str, ...], max_rows: int) -> pd.DataFrame:
    subset = manifest[manifest["dataset_split"].isin(split_names)].copy()
    if subset.empty:
        return subset
    if max_rows <= 0 or len(subset) <= max_rows:
        return subset.reset_index(drop=True)

    rng = random.Random(SEED)
    chosen_indices: list[int] = []
    for split_name in split_names:
        split_subset = subset[subset["dataset_split"] == split_name]
        split_indices = list(split_subset.index)
        rng.shuffle(split_indices)
        chosen_indices.extend(split_indices)
    return subset.loc[chosen_indices[:max_rows]].reset_index(drop=True)


def preview_training_rows(geoai_module, manifest: pd.DataFrame, checkpoint_path: Path) -> None:
    metadata = load_metadata()
    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    preview_rows = select_preview_rows(manifest, PREVIEW_SPLIT, PREVIEW_COUNT)
    if preview_rows.empty:
        print(f"no rows found for preview split={PREVIEW_SPLIT!r}")
        return

    runtime = build_dinov3_inference_runtime(checkpoint_path, metadata)

    for row in preview_rows.itertuples(index=False):
        output_stem = PREVIEW_ROOT / f"{PREVIEW_SPLIT}_{Path(row.image_path).stem}"
        predicted_mask_path = output_stem.with_name(f"{output_stem.name}_pred.tif")
        dinov3_segment_geotiff_with_runtime_settings(
            input_path=Path(row.image_abs_path),
            output_path=predicted_mask_path,
            checkpoint_path=checkpoint_path,
            metadata=metadata,
            window_size=PREVIEW_WINDOW_SIZE,
            overlap=PREVIEW_OVERLAP,
            batch_size=PREVIEW_BATCH_SIZE,
            runtime=runtime,
            progress_desc=f"DINOv3 windows | {Path(row.image_path).stem}",
        )
        render_prediction_review_bundle(
            image_path=Path(row.image_abs_path),
            output_stem=output_stem,
            predicted_mask_path=predicted_mask_path,
            raw_mask_path=Path(row.raw_mask_abs_path) if row.raw_mask_abs_path else None,
            grounded_mask_path=Path(row.grounded_mask_abs_path),
            suptitle=f"DINOv3 | {row.dataset_split} | {Path(row.image_path).stem}",
        )


def preview_external_rows(geoai_module, manifest: pd.DataFrame, checkpoint_path: Path) -> None:
    metadata = load_metadata()
    external_candidates = collect_external_preview_rasters(set(manifest["image_stem"].tolist()))
    if not external_candidates:
        print("no external preview rasters found.")
        return

    runtime = build_dinov3_inference_runtime(checkpoint_path, metadata)

    for image_path in external_candidates[:EXTERNAL_PREVIEW_COUNT]:
        output_stem = PREVIEW_ROOT / f"external_{image_path.stem}"
        predicted_mask_path = output_stem.with_name(f"{output_stem.name}_pred.tif")
        source_name = infer_stac_source_name(image_path)
        dinov3_segment_geotiff_with_runtime_settings(
            input_path=image_path,
            output_path=predicted_mask_path,
            checkpoint_path=checkpoint_path,
            metadata=metadata,
            window_size=PREVIEW_WINDOW_SIZE,
            overlap=PREVIEW_OVERLAP,
            batch_size=PREVIEW_BATCH_SIZE,
            runtime=runtime,
            progress_desc=f"DINOv3 windows | {image_path.stem}",
        )
        render_prediction_review_bundle(
            image_path=image_path,
            output_stem=output_stem,
            predicted_mask_path=predicted_mask_path,
            suptitle=f"DINOv3 | {source_name} external | {image_path.stem}",
        )


def write_evaluation_summary(metrics_df: pd.DataFrame, building_metrics_df: pd.DataFrame | None = None) -> None:
    pixel_summary: dict[str, dict[str, float | int]] = {}
    for split_name, frame in metrics_df.groupby("dataset_split"):
        pixel_summary[split_name] = {
            "row_count": int(len(frame)),
            "mean_iou": float(frame["iou"].mean()),
            "median_iou": float(frame["iou"].median()),
            "mean_f1": float(frame["f1"].mean()),
            "mean_precision": float(frame["precision"].mean()),
            "mean_recall": float(frame["recall"].mean()),
        }

    summary: dict[str, object] = {"pixel_metrics": pixel_summary}
    if building_metrics_df is not None and not building_metrics_df.empty:
        summary["building_metrics"] = summarize_holdout_building_metrics(building_metrics_df)
        summary["building_metrics_note"] = (
            "reference_precision_lower_bound treats the OSM-grounded reference buildings as incomplete labels; "
            "unmatched predictions should be interpreted as lower-bound extras, not confirmed false positives."
        )

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    EVAL_SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")


def evaluate_split_rows(geoai_module, manifest: pd.DataFrame, checkpoint_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = load_metadata()
    eval_rows = select_evaluation_rows(manifest, EVAL_SPLITS, EVAL_MAX_ROWS)
    if eval_rows.empty:
        print(f"no evaluation rows found for splits={EVAL_SPLITS!r}")
        return pd.DataFrame(), pd.DataFrame()

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    runtime = build_dinov3_inference_runtime(checkpoint_path, metadata)
    records: list[dict[str, object]] = []
    for row_index, row in enumerate(eval_rows.itertuples(index=False)):
        split_root = EVAL_ROOT / row.dataset_split
        split_root.mkdir(parents=True, exist_ok=True)
        output_stem = split_root / f"{Path(row.image_path).stem}"
        predicted_mask_path = output_stem.with_name(f"{output_stem.name}_pred.tif")

        dinov3_segment_geotiff_with_runtime_settings(
            input_path=Path(row.image_abs_path),
            output_path=predicted_mask_path,
            checkpoint_path=checkpoint_path,
            metadata=metadata,
            window_size=PREVIEW_WINDOW_SIZE,
            overlap=PREVIEW_OVERLAP,
            batch_size=EVAL_BATCH_SIZE,
            runtime=runtime,
            progress_desc=f"DINOv3 eval windows | {Path(row.image_path).stem}",
        )
        metrics = compute_binary_mask_metrics(predicted_mask_path, Path(row.grounded_mask_abs_path))

        review_png_path = None
        if row_index < EVAL_REVIEW_COUNT:
            review_outputs = render_prediction_review_bundle(
                image_path=Path(row.image_abs_path),
                output_stem=output_stem,
                predicted_mask_path=predicted_mask_path,
                raw_mask_path=Path(row.raw_mask_abs_path) if row.raw_mask_abs_path else None,
                grounded_mask_path=Path(row.grounded_mask_abs_path),
                suptitle=f"DINOv3 eval | {row.dataset_split} | {Path(row.image_path).stem}",
            )
            review_png_path = str(review_outputs["review_png_path"].relative_to(PROJECT_ROOT))

        records.append(
            {
                "tile_id": row.tile_id,
                "dataset_split": row.dataset_split,
                "prompt_artifact_path": getattr(row, "prompt_artifact_path", None),
                "prompt_artifact_abs_path": str(getattr(row, "prompt_artifact_abs_path", "")) if getattr(row, "prompt_artifact_abs_path", None) else None,
                "predicted_mask_path": str(predicted_mask_path.relative_to(PROJECT_ROOT)),
                "review_png_path": review_png_path,
                **metrics,
            }
        )

    metrics_df = pd.DataFrame.from_records(records)
    metrics_df.to_csv(EVAL_METRICS_CSV, index=False)
    metrics_df["predicted_mask_abs_path"] = metrics_df["predicted_mask_path"].map(lambda value: (PROJECT_ROOT / str(value)).resolve())
    building_metrics_df = evaluate_holdout_building_metrics(
        metrics_df,
        predicted_mask_column="predicted_mask_abs_path",
        prompt_artifact_column="prompt_artifact_abs_path",
    )
    if not building_metrics_df.empty:
        building_metrics_df.to_csv(EVAL_BUILDING_METRICS_CSV, index=False)
    elif metrics_df["prompt_artifact_path"].notna().sum() == 0:
        print("building-level holdout metrics skipped: prompt_artifact_path unavailable in the grounded manifest.")
    write_evaluation_summary(metrics_df, building_metrics_df)
    return metrics_df, building_metrics_df


def render_test_detection_gallery(manifest: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    if not RUN_DINOV3_TEST_DETECTION_PREVIEW or TEST_DETECTION_SAMPLE_COUNT <= 0 or metrics_df.empty:
        return

    test_rows = metrics_df[metrics_df["dataset_split"] == "test"].copy()
    if test_rows.empty:
        print("test detection gallery skipped: no test rows in evaluation output.")
        return

    detection_rows = test_rows[
        (test_rows["pred_positive_pixels"] > 0) | (test_rows["target_positive_pixels"] > 0)
    ].copy()
    if detection_rows.empty:
        print("test detection gallery skipped: no PV detections found in test split.")
        return

    detection_rows = detection_rows.sort_values(
        ["pred_positive_pixels", "target_positive_pixels", "iou"],
        ascending=[False, False, False],
        kind="stable",
    ).head(TEST_DETECTION_SAMPLE_COUNT)

    gallery_root = EVAL_ROOT / "test_detection_gallery"
    gallery_root.mkdir(parents=True, exist_ok=True)
    manifest_by_tile = manifest.set_index("tile_id", drop=False)

    for row in detection_rows.itertuples(index=False):
        if row.tile_id not in manifest_by_tile.index:
            continue
        manifest_row = manifest_by_tile.loc[row.tile_id]
        if isinstance(manifest_row, pd.DataFrame):
            manifest_row = manifest_row.iloc[0]

        predicted_mask_path = (PROJECT_ROOT / str(row.predicted_mask_path)).resolve()
        if not predicted_mask_path.exists():
            continue

        output_stem = gallery_root / f"test_{row.tile_id}"
        render_prediction_review_bundle(
            image_path=Path(manifest_row["image_abs_path"]),
            output_stem=output_stem,
            predicted_mask_path=predicted_mask_path,
            raw_mask_path=Path(manifest_row["raw_mask_abs_path"]) if manifest_row.get("raw_mask_abs_path") else None,
            grounded_mask_path=Path(manifest_row["grounded_mask_abs_path"]),
            suptitle=f"DINOv3 test detection | {row.tile_id}",
        )
    print(
        f"wrote test detection gallery for {len(detection_rows):,} tiles under "
        f"{gallery_root.relative_to(PROJECT_ROOT)}"
    )

# %%
if __name__ == "__main__":
    apply_dinov3_widget_overrides()
    validate_dinov3_runtime_configuration()
    configure_torch_gpu_performance()

    manifest = load_grounded_manifest(TRAIN_MANIFEST)
    apply_manifest_training_contract(manifest)
    summarize_manifest(manifest)

    split_counts = manifest["dataset_split"].value_counts().to_dict()
    print(f"usable grounded tiles: {len(manifest):,}")
    print(f"split counts: {split_counts}")

    sample_image = Path(manifest.iloc[0]["image_abs_path"])
    num_channels = min(3, infer_num_channels(sample_image))
    print(f"sample image: {sample_image.name} | channels used: {num_channels}")
    print(
        f"backbone preset: {BACKBONE_PRESET} | hub model: {MODEL_NAME} | "
        f"hub weights: {BACKBONE_HUB_WEIGHTS} | normalization: {INPUT_NORMALIZATION_PROFILE}"
    )
    if BACKBONE_WEIGHTS_PATH is None and BACKBONE_PRESET == "vitl16_sat493m":
        print("backbone weights: GeoAI SAT-493M default (.pth auto-resolved by geoai)")
    elif BACKBONE_WEIGHTS_PATH is not None:
        print(f"backbone weights: {BACKBONE_WEIGHTS_PATH}")
    else:
        print(
            "backbone weights: official DINOv3 auto-resolution for "
            f"{BACKBONE_HUB_WEIGHTS} (set GEOAI_DINOV3_WEIGHTS_PATH to override with a local .pth)"
        )
    print(f"model output dir: {MODEL_OUT}")
    resume_checkpoint = ensure_resume_compatible()
    if resume_checkpoint is not None:
        print(f"resume checkpoint: {resume_checkpoint}")
    write_metadata(num_channels)

    import geoai

    configure_geoai_dinov3_weight_loaders(metadata=load_metadata())

    train_dataset = build_split_dataset(geoai, manifest, "train")
    val_dataset = build_split_dataset(geoai, manifest, "val")
    preview_transformed_samples(geoai, manifest)
    preview_similarity_rows(manifest)
    if train_dataset is None:
        raise RuntimeError("no train split found in grounded manifest")
    if val_dataset is None:
        print("warning: no val split found; DINOv3 training will run without validation.")

    print(
        f"DINOv3 preflight complete | model={MODEL_NAME} | epochs={NUM_EPOCHS} | "
        f"freeze_backbone={FREEZE_BACKBONE} | use_lora={USE_LORA} | precision={LIGHTNING_PRECISION}"
    )

    if RUN_DINOV3_TRAINING:
        MODEL_OUT.mkdir(parents=True, exist_ok=True)
        geoai.train_dinov3_segmentation(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_name=MODEL_NAME,
            weights_path=str(BACKBONE_WEIGHTS_PATH) if BACKBONE_WEIGHTS_PATH is not None else None,
            num_classes=NUM_CLASSES,
            decoder_features=DECODER_FEATURES,
            output_dir=str(MODEL_OUT),
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            num_workers=NUM_WORKERS,
            freeze_backbone=FREEZE_BACKBONE,
            use_lora=USE_LORA,
            lora_rank=LORA_RANK,
            accelerator=ACCELERATOR,
            devices=DEVICES,
            patience=EARLY_STOPPING_PATIENCE,
            checkpoint_path=str(resume_checkpoint) if resume_checkpoint is not None else None,
            precision=LIGHTNING_PRECISION,
        )
        write_metadata(num_channels)
        print(f"DINOv3 training complete under {MODEL_OUT}")
    else:
        print("training disabled; set GEOAI_RUN_DINOV3_TRAINING=1 to launch fine-tuning.")

    if RUN_DINOV3_PREVIEW or RUN_DINOV3_EVAL:
        try:
            checkpoint_path = find_latest_checkpoint(MODEL_OUT)
        except FileNotFoundError as exc:
            print(f"checkpoint-dependent steps skipped: {exc}")
        else:
            print(f"checkpoint: {checkpoint_path}")
            if RUN_DINOV3_PREVIEW:
                preview_training_rows(geoai, manifest, checkpoint_path)
                preview_external_rows(geoai, manifest, checkpoint_path)
            if RUN_DINOV3_EVAL:
                evaluation_metrics, building_metrics = evaluate_split_rows(geoai, manifest, checkpoint_path)
                if not evaluation_metrics.empty:
                    print("evaluation metrics by split:")
                    print(
                        evaluation_metrics.groupby("dataset_split")[["iou", "f1", "precision", "recall"]]
                        .mean()
                        .reset_index()
                        .to_string(index=False)
                    )
                    render_test_detection_gallery(manifest, evaluation_metrics)
                if not building_metrics.empty:
                    print("holdout building metrics by split:")
                    print(
                        building_metrics.groupby("dataset_split")[[
                            "building_recall",
                            "reference_precision_lower_bound",
                            "reference_f1_lower_bound",
                            "predicted_to_reference_ratio",
                        ]]
                        .mean()
                        .reset_index()
                        .to_string(index=False)
                    )

# %%





