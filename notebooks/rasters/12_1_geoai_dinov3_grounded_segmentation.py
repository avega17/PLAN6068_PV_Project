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
        "requires_explicit_weights": False,
        "note": "Correct GeoAI hub model is dinov3_vitl16; the satellite specialization comes from SAT-493M weights and normalization.",
    },
    "vit7b16_sat493m": {
        "label": "ViT-7B/16 SAT-493M (compatible .pth required)",
        "model_name": "dinov3_vit7b16",
        "normalization_profile": "sat493m",
        "requires_explicit_weights": True,
        "note": "The Facebook Hugging Face repo exposes Transformers safetensors, not a GeoAI-compatible torch.hub .pth backbone file.",
    },
    "vitl16_lvd1689m": {
        "label": "ViT-L/16 LVD-1689M (compatible .pth required)",
        "model_name": "dinov3_vitl16",
        "normalization_profile": "lvd1689m",
        "requires_explicit_weights": True,
        "note": "GeoAI's finetune helper only auto-resolves the SAT-493M ViT-L weights, so web-pretrained runs need an explicit local .pth.",
    },
    "vitb16_lvd1689m": {
        "label": "ViT-B/16 LVD-1689M (compatible .pth required)",
        "model_name": "dinov3_vitb16",
        "normalization_profile": "lvd1689m",
        "requires_explicit_weights": True,
        "note": "GeoAI's finetune helper only auto-resolves the SAT-493M ViT-L weights, so web-pretrained runs need an explicit local .pth.",
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
EARLY_STOPPING_PATIENCE = int(os.getenv("GEOAI_DINOV3_PATIENCE", "10") or "10")
SEED = int(os.getenv("GEOAI_DINOV3_SEED", "323") or "323")
RUN_DINOV3_TRAINING = os.getenv("GEOAI_RUN_DINOV3_TRAINING", "1") == "1"
RUN_DINOV3_PREVIEW = os.getenv("GEOAI_RUN_DINOV3_PREVIEW", "1") == "1"
RUN_DINOV3_EVAL = os.getenv("GEOAI_RUN_DINOV3_EVAL", "0") == "1"
PREVIEW_SPLIT = (os.getenv("GEOAI_DINOV3_PREVIEW_SPLIT", "val") or "val").strip().lower()
PREVIEW_COUNT = int(os.getenv("GEOAI_DINOV3_PREVIEW_COUNT", "5") or "5")
EXTERNAL_PREVIEW_COUNT = int(os.getenv("GEOAI_DINOV3_EXTERNAL_PREVIEW_COUNT", "1") or "1")
PREVIEW_WINDOW_SIZE = int(os.getenv("GEOAI_DINOV3_PREVIEW_WINDOW_SIZE", "512") or "512")
PREVIEW_OVERLAP = int(os.getenv("GEOAI_DINOV3_PREVIEW_OVERLAP", "256") or "256")
PREVIEW_BATCH_SIZE = int(os.getenv("GEOAI_DINOV3_PREVIEW_BATCH_SIZE", "8") or "8")
EVAL_SPLITS = tuple(
    split_name.strip().lower()
    for split_name in (os.getenv("GEOAI_DINOV3_EVAL_SPLITS", "val,test") or "val,test").split(",")
    if split_name.strip()
)
EVAL_MAX_ROWS = int(os.getenv("GEOAI_DINOV3_EVAL_MAX_ROWS", "0") or "0")
EVAL_REVIEW_COUNT = int(os.getenv("GEOAI_DINOV3_EVAL_REVIEW_COUNT", "6") or "6")
STAC_TILE_ROOT = PROJECT_ROOT / "outputs" / "stac_tiles"
LOCAL_STAC_CACHE_ROOT = PROJECT_ROOT / "data" / "rasters" / "stac" / "local"
SOLAR_RASTER_ROOT = PROJECT_ROOT / "data" / "rasters" / "solar"
SUPPORTED_IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


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
    global MODEL_NAME, INPUT_NORMALIZATION_PROFILE, INPUT_NORMALIZATION_MEAN, INPUT_NORMALIZATION_STD, BACKBONE_NOTE
    preset = get_backbone_preset()
    MODEL_NAME = str(preset["model_name"])
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

    if bool(preset["requires_explicit_weights"]):
        raise RuntimeError(
            f"Backbone preset {BACKBONE_PRESET!r} requires GEOAI_DINOV3_WEIGHTS_PATH to point to a compatible local .pth file."
        )
    return None


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


def dinov3_segment_geotiff_with_runtime_settings(
    *,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    metadata: dict[str, object],
    window_size: int,
    overlap: int,
    batch_size: int,
) -> None:
    from rasterio.windows import Window
    from tqdm import tqdm

    if overlap >= window_size:
        raise ValueError(f"overlap ({overlap}) must be less than window_size ({window_size})")

    device = resolve_runtime_device()
    model_module = load_dinov3_segmenter(checkpoint_path, metadata, device)
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
            progress = tqdm(total=n_rows * n_cols, disable=False, desc="DINOv3 windows")

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
    if EXPLICIT_MODEL_OUT is not None:
        model_out = EXPLICIT_MODEL_OUT
    else:
        mode_slug = f"lora-r{LORA_RANK}" if USE_LORA else "frozen"
        model_out = BASE_MODEL_ROOT / (
            f"{_slugify(BACKBONE_PRESET)}__hub-{_slugify(MODEL_NAME)}__df{DECODER_FEATURES}__ps{PATCH_SIZE}"
            f"__e{NUM_EPOCHS}__b{BATCH_SIZE}__lr{_float_slug(LEARNING_RATE)}__{mode_slug}"
        )
    MODEL_OUT = model_out
    PREVIEW_ROOT = EXPLICIT_PREVIEW_ROOT or MODEL_OUT / "preview"
    EVAL_ROOT = EXPLICIT_EVAL_ROOT or MODEL_OUT / "evaluation"
    METADATA_PATH = MODEL_OUT / "dinov3_metadata.json"
    EVAL_METRICS_CSV = EVAL_ROOT / "split_metrics.csv"
    EVAL_BUILDING_METRICS_CSV = EVAL_ROOT / "holdout_building_metrics.csv"
    EVAL_SUMMARY_JSON = EVAL_ROOT / "split_summary.json"

# %% [markdown]
# ## Notebook Controls
# 
# These controls expose the most impactful DINOv3 sweep parameters without
# changing the safe script defaults. The SAT-493M preset still uses the hub
# model name `dinov3_vitl16`; the satellite specialization comes from the
# backbone weights and input normalization, not from a separate GeoAI model id.

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
            "**Backbone note**: `facebook/dinov3-vitl16-pretrain-sat493m` is a Hugging Face repository id, not a GeoAI `model_name`. "
            "For the SAT-493M backbone in this notebook, use the preset that resolves to hub model `dinov3_vitl16` and SAT-specific normalization."
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
                widgets.HBox([run_training_widget, run_preview_widget, run_eval_widget]),
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
        "preview_split": preview_split_widget,
    }
    return DINOV3_NOTEBOOK_WIDGETS


def apply_dinov3_widget_overrides() -> None:
    global BACKBONE_PRESET, BACKBONE_WEIGHTS_PATH, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE
    global DECODER_FEATURES, PATCH_SIZE, USE_LORA, LORA_RANK
    global RUN_DINOV3_TRAINING, RUN_DINOV3_PREVIEW, RUN_DINOV3_EVAL, PREVIEW_SPLIT
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
        PREVIEW_SPLIT = str(widgets["preview_split"].value)
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


def find_latest_checkpoint(model_dir: Path) -> Path:
    candidates = sorted(
        [
            path
            for path in model_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".ckpt", ".pth", ".pt"}
        ],
        key=lambda path: ("last" in path.name.lower(), -path.stat().st_mtime),
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
        "backbone_preset": BACKBONE_PRESET,
        "model_name": MODEL_NAME,
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
        "freeze_backbone": FREEZE_BACKBONE,
        "use_lora": USE_LORA,
        "lora_rank": LORA_RANK,
    }
    METADATA_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def load_metadata() -> dict[str, object]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    weights_path = resolve_backbone_weights_path()
    return {
        "backbone_preset": BACKBONE_PRESET,
        "model_name": MODEL_NAME,
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
        "freeze_backbone": FREEZE_BACKBONE,
        "use_lora": USE_LORA,
        "lora_rank": LORA_RANK,
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
            batch_size=PREVIEW_BATCH_SIZE,
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

# %%
if __name__ == "__main__":
    apply_dinov3_widget_overrides()
    validate_dinov3_runtime_configuration()

    manifest = load_grounded_manifest(TRAIN_MANIFEST)
    summarize_manifest(manifest)

    split_counts = manifest["dataset_split"].value_counts().to_dict()
    print(f"usable grounded tiles: {len(manifest):,}")
    print(f"split counts: {split_counts}")

    sample_image = Path(manifest.iloc[0]["image_abs_path"])
    num_channels = min(3, infer_num_channels(sample_image))
    print(f"sample image: {sample_image.name} | channels used: {num_channels}")
    print(f"backbone preset: {BACKBONE_PRESET} | hub model: {MODEL_NAME} | normalization: {INPUT_NORMALIZATION_PROFILE}")
    if BACKBONE_WEIGHTS_PATH is None and BACKBONE_PRESET == "vitl16_sat493m":
        print("backbone weights: GeoAI SAT-493M default (.pth auto-resolved by geoai)")
    elif BACKBONE_WEIGHTS_PATH is not None:
        print(f"backbone weights: {BACKBONE_WEIGHTS_PATH}")
    print(f"model output dir: {MODEL_OUT}")
    write_metadata(num_channels)

    import geoai

    train_dataset = build_split_dataset(geoai, manifest, "train")
    val_dataset = build_split_dataset(geoai, manifest, "val")
    if train_dataset is None:
        raise RuntimeError("no train split found in grounded manifest")
    if val_dataset is None:
        print("warning: no val split found; DINOv3 training will run without validation.")

    print(
        f"DINOv3 preflight complete | model={MODEL_NAME} | epochs={NUM_EPOCHS} | "
        f"freeze_backbone={FREEZE_BACKBONE} | use_lora={USE_LORA}"
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




