# %% [markdown]
# # SMP Grounded Segmentation
# 
# Trains a segmentation-models-pytorch baseline on the grounded image/mask
# export from `11_geoai_training_data.py` while preserving the manifest-defined
# train/val/test split. The notebook is safe by default: a normal script run
# performs preflight and dataset validation, while training starts only when
# `GEOAI_RUN_SMP_TRAINING=1`.

# %%
"""12_3_geoai_smp_grounded_segmentation.py"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader


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
    mask_has_positive_pixels,
    summarize_holdout_building_metrics,
    vectorize_binary_mask,
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


DEFAULT_TRAIN_ROOT = PROJECT_ROOT / "outputs" / "geoai_train_contextily"
TRAIN_ROOT = _resolve_configured_path("GEOAI_TRAIN_ROOT", DEFAULT_TRAIN_ROOT)
TRAIN_MANIFEST = TRAIN_ROOT / "training_chip_manifest.csv"
IMAGES_DIR = TRAIN_ROOT / "images"
RAW_MASKS_DIR = TRAIN_ROOT / "masks"
GROUNDED_MASKS_DIR = TRAIN_ROOT / "grounded_masks"
PROMPT_ARTIFACTS_DIR = TRAIN_ROOT / "prompt_artifacts"

BASE_MODEL_ROOT = PROJECT_ROOT / "outputs" / "models" / "smp_grounded"
EXPLICIT_MODEL_OUT = _resolve_optional_path(os.getenv("GEOAI_SMP_MODEL_OUT"))
EXPLICIT_PREVIEW_ROOT = _resolve_optional_path(os.getenv("GEOAI_SMP_PREVIEW_ROOT"))
EXPLICIT_EVAL_ROOT = _resolve_optional_path(os.getenv("GEOAI_SMP_EVAL_ROOT"))
MODEL_OUT = EXPLICIT_MODEL_OUT or BASE_MODEL_ROOT / "unet__resnet34__imagenet__e10__lr1em03__full-ft"
PREVIEW_ROOT = EXPLICIT_PREVIEW_ROOT or MODEL_OUT / "preview"
EVAL_ROOT = EXPLICIT_EVAL_ROOT or MODEL_OUT / "evaluation"
METADATA_PATH = MODEL_OUT / "smp_metadata.json"
PREVIEW_METRICS_CSV = PREVIEW_ROOT / "preview_metrics.csv"
EVAL_METRICS_CSV = EVAL_ROOT / "split_metrics.csv"
EVAL_BUILDING_METRICS_CSV = EVAL_ROOT / "holdout_building_metrics.csv"
EVAL_SUMMARY_JSON = EVAL_ROOT / "split_summary.json"

ARCHITECTURE = (os.getenv("GEOAI_SMP_ARCHITECTURE", "unet") or "unet").strip().lower()
ENCODER_NAME = (os.getenv("GEOAI_SMP_ENCODER_NAME", "resnet34") or "resnet34").strip()
ENCODER_WEIGHTS_RAW = (os.getenv("GEOAI_SMP_ENCODER_WEIGHTS", "imagenet") or "imagenet").strip().lower()
ENCODER_WEIGHTS = None if ENCODER_WEIGHTS_RAW in {"", "none", "null"} else ENCODER_WEIGHTS_RAW
NUM_CLASSES = int(os.getenv("GEOAI_SMP_NUM_CLASSES", "2") or "2")
NUM_EPOCHS = int(os.getenv("GEOAI_SMP_NUM_EPOCHS", "20") or "10")
BATCH_SIZE = int(os.getenv("GEOAI_SMP_BATCH_SIZE", "16") or "4")
LEARNING_RATE = float(os.getenv("GEOAI_SMP_LEARNING_RATE", "1e-3") or "1e-3")
WEIGHT_DECAY = float(os.getenv("GEOAI_SMP_WEIGHT_DECAY", "1e-4") or "1e-4")
NUM_WORKERS = int(os.getenv("GEOAI_SMP_NUM_WORKERS", str(default_num_workers())))
TARGET_SIZE_RAW = int(os.getenv("GEOAI_SMP_TARGET_SIZE", "512") or "0")
TARGET_SIZE = (TARGET_SIZE_RAW, TARGET_SIZE_RAW) if TARGET_SIZE_RAW > 0 else None
RESIZE_MODE = (os.getenv("GEOAI_SMP_RESIZE_MODE", "resize") or "resize").strip().lower()
FREEZE_ENCODER = os.getenv("GEOAI_SMP_FREEZE_ENCODER", "0") == "1"
EARLY_STOPPING_PATIENCE = int(os.getenv("GEOAI_SMP_PATIENCE", "10") or "10")
SEED = int(os.getenv("GEOAI_SMP_SEED", "323") or "323")
RUN_SMP_TRAINING = os.getenv("GEOAI_RUN_SMP_TRAINING", "1") == "1"
RUN_SMP_PREVIEW = os.getenv("GEOAI_RUN_SMP_PREVIEW", "1") == "1"
PREVIEW_SPLIT = (os.getenv("GEOAI_SMP_PREVIEW_SPLIT", "val") or "val").strip().lower()
PREVIEW_COUNT = int(os.getenv("GEOAI_SMP_PREVIEW_COUNT", "3") or "3")
EXTERNAL_PREVIEW_COUNT = int(os.getenv("GEOAI_SMP_EXTERNAL_PREVIEW_COUNT", "1") or "1")
PREVIEW_WINDOW_SIZE = int(os.getenv("GEOAI_SMP_PREVIEW_WINDOW_SIZE", "512") or "512")
PREVIEW_OVERLAP = int(os.getenv("GEOAI_SMP_PREVIEW_OVERLAP", "256") or "256")
PREVIEW_BATCH_SIZE = int(os.getenv("GEOAI_SMP_PREVIEW_BATCH_SIZE", "8") or "8")
PREVIEW_PROBABILITY_THRESHOLD = float(os.getenv("GEOAI_SMP_PREVIEW_PROBABILITY_THRESHOLD", "0.5") or "0.5")
DEVICE_REQUEST = (os.getenv("GEOAI_SMP_DEVICE", "auto") or "auto").strip().lower()
STAC_TILE_ROOT = PROJECT_ROOT / "outputs" / "stac_tiles"
LOCAL_STAC_CACHE_ROOT = PROJECT_ROOT / "data" / "rasters" / "stac" / "local"
SOLAR_RASTER_ROOT = PROJECT_ROOT / "data" / "rasters" / "solar"
SUPPORTED_IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
RUN_SMP_EVAL = os.getenv("GEOAI_RUN_SMP_EVAL", "0") == "1"
EVAL_SPLITS = tuple(
    split_name.strip().lower()
    for split_name in (os.getenv("GEOAI_SMP_EVAL_SPLITS", "val,test") or "val,test").split(",")
    if split_name.strip()
)
EVAL_MAX_ROWS = int(os.getenv("GEOAI_SMP_EVAL_MAX_ROWS", "0") or "0")
EVAL_REVIEW_COUNT = int(os.getenv("GEOAI_SMP_EVAL_REVIEW_COUNT", "6") or "6")


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


def refresh_runtime_paths() -> None:
    global MODEL_OUT, PREVIEW_ROOT, EVAL_ROOT, METADATA_PATH, PREVIEW_METRICS_CSV
    global EVAL_METRICS_CSV, EVAL_BUILDING_METRICS_CSV, EVAL_SUMMARY_JSON
    if EXPLICIT_MODEL_OUT is not None:
        model_out = EXPLICIT_MODEL_OUT
    else:
        freeze_slug = "frozen-enc" if FREEZE_ENCODER else "full-ft"
        weights_slug = _slugify(ENCODER_WEIGHTS or "random")
        model_out = BASE_MODEL_ROOT / (
            f"{_slugify(ARCHITECTURE)}__{_slugify(ENCODER_NAME)}__{weights_slug}"
            f"__e{NUM_EPOCHS}__b{BATCH_SIZE}__lr{_float_slug(LEARNING_RATE)}__{freeze_slug}"
        )
    MODEL_OUT = model_out
    PREVIEW_ROOT = EXPLICIT_PREVIEW_ROOT or MODEL_OUT / "preview"
    EVAL_ROOT = EXPLICIT_EVAL_ROOT or MODEL_OUT / "evaluation"
    METADATA_PATH = MODEL_OUT / "smp_metadata.json"
    PREVIEW_METRICS_CSV = PREVIEW_ROOT / "preview_metrics.csv"
    EVAL_METRICS_CSV = EVAL_ROOT / "split_metrics.csv"
    EVAL_BUILDING_METRICS_CSV = EVAL_ROOT / "holdout_building_metrics.csv"
    EVAL_SUMMARY_JSON = EVAL_ROOT / "split_summary.json"

# %% [markdown]
# ## Notebook Controls
# 
# These widgets make it easier to switch across the shortlisted SMP
# architecture and encoder variants. The current shortlist spans a simple
# baseline, denser skip connections, a context-heavy decoder, an efficient
# multiscale option, and a transformer-style model.

# %%
SMP_ARCHITECTURE_CHOICES = [
    ("UNet", "unet"),
    ("UNet++", "unetplusplus"),
    ("DeepLabV3+", "deeplabv3plus"),
    ("FPN", "fpn"),
    ("UPerNet", "upernet"),
    ("SegFormer", "segformer"),
]
SMP_ENCODER_CHOICES = [
    ("ResNet34", "resnet34"),
    ("ResNet50", "resnet50"),
    ("ResNet101", "resnet101"),
    ("EfficientNet-B3", "efficientnet-b3"),
    ("EfficientNet-B4", "efficientnet-b4"),
    ("MiT-B2", "mit_b2"),
    ("MiT-B3", "mit_b3"),
]
SMP_ENCODER_WEIGHT_CHOICES = [
    ("ImageNet", "imagenet"),
    ("Random init", "none"),
]
SMP_LEARNING_RATE_CHOICES = [1e-4, 3e-4, 1e-3, 3e-3]
SMP_NOTEBOOK_WIDGETS: dict[str, object] | None = None


def _maybe_initialize_smp_widgets() -> dict[str, object] | None:
    global SMP_NOTEBOOK_WIDGETS
    if SMP_NOTEBOOK_WIDGETS is not None:
        return SMP_NOTEBOOK_WIDGETS
    if not running_in_notebook():
        return None

    try:
        import ipywidgets as widgets
        from IPython.display import Markdown, display
    except ImportError:
        return None

    architecture_widget = widgets.Dropdown(
        options=SMP_ARCHITECTURE_CHOICES,
        value=ARCHITECTURE,
        description="Arch",
        layout=widgets.Layout(width="260px"),
    )
    encoder_widget = widgets.Dropdown(
        options=SMP_ENCODER_CHOICES,
        value=ENCODER_NAME,
        description="Encoder",
        layout=widgets.Layout(width="260px"),
    )
    encoder_weights_widget = widgets.Dropdown(
        options=SMP_ENCODER_WEIGHT_CHOICES,
        value=ENCODER_WEIGHTS or "none",
        description="Weights",
        layout=widgets.Layout(width="260px"),
    )
    learning_rate_widget = widgets.Dropdown(
        options=SMP_LEARNING_RATE_CHOICES,
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
    preview_split_widget = widgets.Dropdown(
        options=["train", "val", "test"],
        value=PREVIEW_SPLIT,
        description="Preview split",
        layout=widgets.Layout(width="220px"),
    )
    freeze_encoder_widget = widgets.Checkbox(
        value=FREEZE_ENCODER,
        description="Freeze encoder",
        indent=False,
    )
    run_training_widget = widgets.Checkbox(
        value=RUN_SMP_TRAINING,
        description="Run training",
        indent=False,
    )
    run_preview_widget = widgets.Checkbox(
        value=RUN_SMP_PREVIEW,
        description="Run preview",
        indent=False,
    )
    run_eval_widget = widgets.Checkbox(
        value=RUN_SMP_EVAL,
        description="Run eval",
        indent=False,
    )

    display(
        Markdown(
            "**Validated shortlist**: UNet/ResNet34, UNet++/ResNet50, DeepLabV3+/ResNet50, "
            "FPN/EfficientNet-B3, UPerNet/ResNet101, SegFormer/MiT-B2, plus the expanded "
            "ResNet101, EfficientNet-B4, and MiT-B3 encoder options all initialize successfully here."
        )
    )
    display(
        widgets.VBox(
            [
                widgets.HBox([architecture_widget, encoder_widget, encoder_weights_widget]),
                widgets.HBox([learning_rate_widget, num_epochs_widget, batch_size_widget]),
                widgets.HBox([preview_split_widget, freeze_encoder_widget]),
                widgets.HBox([run_training_widget, run_preview_widget, run_eval_widget]),
            ]
        )
    )

    SMP_NOTEBOOK_WIDGETS = {
        "architecture": architecture_widget,
        "encoder_name": encoder_widget,
        "encoder_weights": encoder_weights_widget,
        "learning_rate": learning_rate_widget,
        "num_epochs": num_epochs_widget,
        "batch_size": batch_size_widget,
        "preview_split": preview_split_widget,
        "freeze_encoder": freeze_encoder_widget,
        "run_training": run_training_widget,
        "run_preview": run_preview_widget,
        "run_eval": run_eval_widget,
    }
    return SMP_NOTEBOOK_WIDGETS


def apply_smp_widget_overrides() -> None:
    global ARCHITECTURE, ENCODER_NAME, ENCODER_WEIGHTS, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE
    global PREVIEW_SPLIT, FREEZE_ENCODER, RUN_SMP_TRAINING, RUN_SMP_PREVIEW, RUN_SMP_EVAL
    widgets = _maybe_initialize_smp_widgets()
    if widgets is not None:
        ARCHITECTURE = str(widgets["architecture"].value)
        ENCODER_NAME = str(widgets["encoder_name"].value)
        encoder_weights_value = str(widgets["encoder_weights"].value)
        ENCODER_WEIGHTS = None if encoder_weights_value in {"none", "null", ""} else encoder_weights_value
        LEARNING_RATE = float(widgets["learning_rate"].value)
        NUM_EPOCHS = int(widgets["num_epochs"].value)
        BATCH_SIZE = int(widgets["batch_size"].value)
        PREVIEW_SPLIT = str(widgets["preview_split"].value)
        FREEZE_ENCODER = bool(widgets["freeze_encoder"].value)
        RUN_SMP_TRAINING = bool(widgets["run_training"].value)
        RUN_SMP_PREVIEW = bool(widgets["run_preview"].value)
        RUN_SMP_EVAL = bool(widgets["run_eval"].value)
    refresh_runtime_paths()


refresh_runtime_paths()
_maybe_initialize_smp_widgets()


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
        raw_mask_path = raw_mask_files.get(stem)
        prompt_artifact_path = prompt_artifact_files.get(stem)
        rows.append(
            {
                "tile_id": stem,
                "image_path": str(image_files[stem].relative_to(PROJECT_ROOT)),
                "grounded_mask_path": str(grounded_mask_files[stem].relative_to(PROJECT_ROOT)),
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


def resolve_device(request: str) -> torch.device:
    if request in {"", "auto"}:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GEOAI_SMP_DEVICE requested CUDA but torch.cuda.is_available() is False.")
    return device


def describe_device(device: torch.device) -> None:
    print(f"training device: {device}")
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        print(f"cuda device name: {torch.cuda.get_device_name(index)}")


def write_metadata(num_channels: int) -> None:
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": ARCHITECTURE,
        "encoder_name": ENCODER_NAME,
        "encoder_weights": ENCODER_WEIGHTS,
        "num_channels": num_channels,
        "num_classes": NUM_CLASSES,
        "target_size": TARGET_SIZE,
        "resize_mode": RESIZE_MODE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "freeze_encoder": FREEZE_ENCODER,
    }
    METADATA_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def load_metadata() -> dict[str, object]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return {
        "architecture": ARCHITECTURE,
        "encoder_name": ENCODER_NAME,
        "encoder_weights": ENCODER_WEIGHTS,
        "num_channels": 3,
        "num_classes": NUM_CLASSES,
        "target_size": TARGET_SIZE,
        "resize_mode": RESIZE_MODE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "freeze_encoder": FREEZE_ENCODER,
    }


def find_best_model(model_dir: Path) -> Path:
    candidate = model_dir / "best_model.pth"
    if not candidate.exists():
        raise FileNotFoundError(f"best_model.pth not found under {model_dir}")
    return candidate


def find_training_history_path(model_dir: Path) -> Path | None:
    candidate = model_dir / "training_history.pth"
    return candidate if candidate.exists() else None


def load_training_history_summary(history_path: Path) -> pd.DataFrame:
    history = torch.load(history_path, map_location="cpu")
    max_len = max((len(values) for values in history.values() if isinstance(values, list)), default=0)
    frame = pd.DataFrame({"epoch": list(range(1, max_len + 1))})
    column_map = {
        "train_losses": "train_loss",
        "val_losses": "val_loss",
        "val_ious": "val_iou",
        "val_f1s": "val_f1",
        "val_precisions": "val_precision",
        "val_recalls": "val_recall",
    }
    for source_key, target_key in column_map.items():
        values = history.get(source_key)
        if isinstance(values, list) and values:
            frame[target_key] = values
    return frame


def collect_external_preview_rasters(exclude_stems: set[str]) -> list[Path]:
    return collect_naip_stac_preview_rasters(
        PROJECT_ROOT,
        exclude_stems=exclude_stems,
        supported_image_extensions=SUPPORTED_IMAGE_EXTENSIONS,
    )


def select_preview_rows(manifest: pd.DataFrame, split_name: str, count: int) -> pd.DataFrame:
    subset = manifest[manifest["dataset_split"] == split_name].copy()
    if subset.empty:
        return subset
    rng = random.Random(SEED)
    indices = list(subset.index)
    rng.shuffle(indices)
    chosen = indices[: min(count, len(indices))]
    return subset.loc[chosen].reset_index(drop=True)


def build_fixed_split_datasets(manifest: pd.DataFrame, num_channels: int):
    from geoai.train import SemanticSegmentationDataset, get_semantic_transform

    train_rows = manifest[manifest["dataset_split"] == "train"].reset_index(drop=True)
    val_rows = manifest[manifest["dataset_split"] == "val"].reset_index(drop=True)
    if train_rows.empty:
        raise RuntimeError("no train split rows found in grounded manifest")
    if val_rows.empty:
        raise RuntimeError("no val split rows found in grounded manifest")

    train_dataset = SemanticSegmentationDataset(
        image_paths=[str(path) for path in train_rows["image_abs_path"]],
        label_paths=[str(path) for path in train_rows["grounded_mask_abs_path"]],
        transforms=get_semantic_transform(train=True),
        num_channels=num_channels,
        target_size=TARGET_SIZE,
        resize_mode=RESIZE_MODE,
        num_classes=NUM_CLASSES,
    )
    val_dataset = SemanticSegmentationDataset(
        image_paths=[str(path) for path in val_rows["image_abs_path"]],
        label_paths=[str(path) for path in val_rows["grounded_mask_abs_path"]],
        transforms=get_semantic_transform(train=False),
        num_channels=num_channels,
        target_size=TARGET_SIZE,
        resize_mode=RESIZE_MODE,
        num_classes=NUM_CLASSES,
    )
    return train_dataset, val_dataset


def validate_smp_model_configuration(num_channels: int) -> None:
    from geoai.train import get_smp_model

    probe_model = get_smp_model(
        architecture=ARCHITECTURE,
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=num_channels,
        classes=NUM_CLASSES,
    )
    del probe_model


def train_smp_model(manifest: pd.DataFrame, *, num_channels: int, device: torch.device) -> None:
    from geoai.train import (
        evaluate_semantic,
        get_smp_model,
        train_semantic_one_epoch,
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset, val_dataset = build_fixed_split_datasets(manifest, num_channels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    model = get_smp_model(
        architecture=ARCHITECTURE,
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=num_channels,
        classes=NUM_CLASSES,
    )
    if FREEZE_ENCODER and hasattr(model, "encoder"):
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, EARLY_STOPPING_PATIENCE // 2),
    )

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    history = {
        "train_losses": [],
        "val_losses": [],
        "val_ious": [],
        "val_f1s": [],
        "val_precisions": [],
        "val_recalls": [],
    }
    best_iou = -1.0
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        train_loss = train_semantic_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            criterion,
            print_freq=max(1, len(train_loader) // 5),
            verbose=True,
        )
        val_metrics = evaluate_semantic(model, val_loader, device, criterion, num_classes=NUM_CLASSES)
        scheduler.step(val_metrics["loss"])

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_metrics["loss"])
        history["val_ious"].append(val_metrics["IoU"])
        history["val_f1s"].append(val_metrics["F1"])
        history["val_precisions"].append(val_metrics["Precision"])
        history["val_recalls"].append(val_metrics["Recall"])

        print(
            f"epoch {epoch + 1}/{NUM_EPOCHS} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_iou={val_metrics['IoU']:.4f} | "
            f"val_f1={val_metrics['F1']:.4f}"
        )

        if val_metrics["IoU"] > best_iou:
            best_iou = val_metrics["IoU"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_OUT / "best_model.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"early stopping after {epoch + 1} epochs without IoU improvement")
                break

    torch.save(model.state_dict(), MODEL_OUT / "final_model.pth")
    torch.save(history, MODEL_OUT / "training_history.pth")

    summary_lines = [
        f"architecture: {ARCHITECTURE}",
        f"encoder_name: {ENCODER_NAME}",
        f"num_epochs: {NUM_EPOCHS}",
        f"best_val_iou: {best_iou:.4f}",
        f"freeze_encoder: {FREEZE_ENCODER}",
    ]
    (MODEL_OUT / "training_summary.txt").write_text("\n".join(summary_lines) + "\n")


def preview_rows(geoai_module, manifest: pd.DataFrame, *, model_path: Path, num_channels: int) -> pd.DataFrame:
    metadata = load_metadata()
    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    preview_rows = select_preview_rows(manifest, PREVIEW_SPLIT, PREVIEW_COUNT)
    if preview_rows.empty:
        print(f"no rows found for preview split={PREVIEW_SPLIT!r}")
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for row in preview_rows.itertuples(index=False):
        output_stem = PREVIEW_ROOT / f"{PREVIEW_SPLIT}_{Path(row.image_path).stem}_{ARCHITECTURE}_{ENCODER_NAME}"
        predicted_mask_path = output_stem.with_name(f"{output_stem.name}_pred.tif")
        probability_path = output_stem.with_name(f"{output_stem.name}_prob.tif")
        vector_path = output_stem.with_name(f"{output_stem.name}_pred.geojson")

        geoai_module.semantic_segmentation(
            input_path=str(row.image_abs_path),
            output_path=str(predicted_mask_path),
            model_path=str(model_path),
            architecture=str(metadata["architecture"]),
            encoder_name=str(metadata["encoder_name"]),
            num_channels=num_channels,
            num_classes=int(metadata["num_classes"]),
            window_size=PREVIEW_WINDOW_SIZE,
            overlap=PREVIEW_OVERLAP,
            batch_size=PREVIEW_BATCH_SIZE,
            probability_path=str(probability_path),
            probability_threshold=PREVIEW_PROBABILITY_THRESHOLD,
            quiet=False,
        )
        vectorized_path = vectorize_binary_mask(predicted_mask_path, vector_path) if mask_has_positive_pixels(predicted_mask_path) else None
        metrics = compute_binary_mask_metrics(predicted_mask_path, Path(row.grounded_mask_abs_path))
        review_outputs = render_prediction_review_bundle(
            image_path=Path(row.image_abs_path),
            output_stem=output_stem,
            predicted_mask_path=predicted_mask_path,
            raw_mask_path=Path(row.raw_mask_abs_path) if row.raw_mask_abs_path else None,
            grounded_mask_path=Path(row.grounded_mask_abs_path),
            vector_path=vectorized_path,
            suptitle=f"SMP | {ARCHITECTURE}/{ENCODER_NAME} | {row.dataset_split} | {Path(row.image_path).stem}",
        )
        records.append(
            {
                "tile_id": row.tile_id,
                "dataset_split": row.dataset_split,
                "predicted_mask_path": str(predicted_mask_path.relative_to(PROJECT_ROOT)),
                "probability_path": str(probability_path.relative_to(PROJECT_ROOT)),
                "vector_path": str(vectorized_path.relative_to(PROJECT_ROOT)) if vectorized_path is not None else None,
                "review_png_path": str(review_outputs["review_png_path"].relative_to(PROJECT_ROOT)),
                "vector_review_png_path": str(review_outputs.get("vector_review_png_path").relative_to(PROJECT_ROOT)) if review_outputs.get("vector_review_png_path") is not None else None,
                **metrics,
            }
        )
    metrics_df = pd.DataFrame.from_records(records)
    metrics_df.to_csv(PREVIEW_METRICS_CSV, index=False)
    return metrics_df


def preview_external_rows(geoai_module, manifest: pd.DataFrame, *, model_path: Path, num_channels: int) -> None:
    metadata = load_metadata()
    external_candidates = collect_external_preview_rasters(set(manifest["image_stem"].tolist()))
    if not external_candidates:
        print("no external preview rasters found.")
        return

    for image_path in external_candidates[:EXTERNAL_PREVIEW_COUNT]:
        output_stem = PREVIEW_ROOT / f"external_{image_path.stem}_{ARCHITECTURE}_{ENCODER_NAME}"
        predicted_mask_path = output_stem.with_name(f"{output_stem.name}_pred.tif")
        probability_path = output_stem.with_name(f"{output_stem.name}_prob.tif")
        vector_path = output_stem.with_name(f"{output_stem.name}_pred.geojson")
        source_name = infer_stac_source_name(image_path)

        geoai_module.semantic_segmentation(
            input_path=str(image_path),
            output_path=str(predicted_mask_path),
            model_path=str(model_path),
            architecture=str(metadata["architecture"]),
            encoder_name=str(metadata["encoder_name"]),
            num_channels=num_channels,
            num_classes=int(metadata["num_classes"]),
            window_size=PREVIEW_WINDOW_SIZE,
            overlap=PREVIEW_OVERLAP,
            batch_size=PREVIEW_BATCH_SIZE,
            probability_path=str(probability_path),
            probability_threshold=PREVIEW_PROBABILITY_THRESHOLD,
            quiet=False,
        )
        vectorized_path = vectorize_binary_mask(predicted_mask_path, vector_path) if mask_has_positive_pixels(predicted_mask_path) else None
        render_prediction_review_bundle(
            image_path=image_path,
            output_stem=output_stem,
            predicted_mask_path=predicted_mask_path,
            vector_path=vectorized_path,
            suptitle=f"SMP | {ARCHITECTURE}/{ENCODER_NAME} | {source_name} external | {image_path.stem}",
        )


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


def evaluate_split_rows(geoai_module, manifest: pd.DataFrame, *, model_path: Path, num_channels: int) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        output_stem = split_root / f"{Path(row.image_path).stem}_{ARCHITECTURE}_{ENCODER_NAME}"
        predicted_mask_path = output_stem.with_name(f"{output_stem.name}_pred.tif")
        probability_path = output_stem.with_name(f"{output_stem.name}_prob.tif")
        vector_path = output_stem.with_name(f"{output_stem.name}_pred.geojson")

        geoai_module.semantic_segmentation(
            input_path=str(row.image_abs_path),
            output_path=str(predicted_mask_path),
            model_path=str(model_path),
            architecture=str(metadata["architecture"]),
            encoder_name=str(metadata["encoder_name"]),
            num_channels=num_channels,
            num_classes=int(metadata["num_classes"]),
            window_size=PREVIEW_WINDOW_SIZE,
            overlap=PREVIEW_OVERLAP,
            batch_size=PREVIEW_BATCH_SIZE,
            probability_path=str(probability_path),
            probability_threshold=PREVIEW_PROBABILITY_THRESHOLD,
            quiet=False,
        )
        vectorized_path = vectorize_binary_mask(predicted_mask_path, vector_path) if mask_has_positive_pixels(predicted_mask_path) else None
        metrics = compute_binary_mask_metrics(predicted_mask_path, Path(row.grounded_mask_abs_path))

        review_png_path = None
        vector_review_png_path = None
        if row_index < EVAL_REVIEW_COUNT:
            review_outputs = render_prediction_review_bundle(
                image_path=Path(row.image_abs_path),
                output_stem=output_stem,
                predicted_mask_path=predicted_mask_path,
                raw_mask_path=Path(row.raw_mask_abs_path) if row.raw_mask_abs_path else None,
                grounded_mask_path=Path(row.grounded_mask_abs_path),
                vector_path=vectorized_path,
                suptitle=f"SMP eval | {ARCHITECTURE}/{ENCODER_NAME} | {row.dataset_split} | {Path(row.image_path).stem}",
            )
            review_png_path = str(review_outputs["review_png_path"].relative_to(PROJECT_ROOT))
            vector_review_png_path = (
                str(review_outputs.get("vector_review_png_path").relative_to(PROJECT_ROOT))
                if review_outputs.get("vector_review_png_path") is not None
                else None
            )

        records.append(
            {
                "tile_id": row.tile_id,
                "dataset_split": row.dataset_split,
                "prompt_artifact_path": getattr(row, "prompt_artifact_path", None),
                "prompt_artifact_abs_path": str(getattr(row, "prompt_artifact_abs_path", "")) if getattr(row, "prompt_artifact_abs_path", None) else None,
                "predicted_mask_path": str(predicted_mask_path.relative_to(PROJECT_ROOT)),
                "probability_path": str(probability_path.relative_to(PROJECT_ROOT)),
                "vector_path": str(vectorized_path.relative_to(PROJECT_ROOT)) if vectorized_path is not None else None,
                "review_png_path": review_png_path,
                "vector_review_png_path": vector_review_png_path,
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
    apply_smp_widget_overrides()

    manifest = load_grounded_manifest(TRAIN_MANIFEST)
    summarize_manifest(manifest)

    split_counts = manifest["dataset_split"].value_counts().to_dict()
    print(f"usable grounded tiles: {len(manifest):,}")
    print(f"split counts: {split_counts}")

    sample_image = Path(manifest.iloc[0]["image_abs_path"])
    num_channels = min(3, infer_num_channels(sample_image))
    print(f"sample image: {sample_image.name} | channels used: {num_channels}")
    validate_smp_model_configuration(num_channels)

    device = resolve_device(DEVICE_REQUEST)
    describe_device(device)
    print(f"model output dir: {MODEL_OUT}")
    write_metadata(num_channels)

    train_count = int((manifest["dataset_split"] == "train").sum())
    val_count = int((manifest["dataset_split"] == "val").sum())
    if train_count == 0 or val_count == 0:
        raise RuntimeError("SMP training requires non-empty train and val splits in the grounded manifest.")

    print(
        f"SMP preflight complete | architecture={ARCHITECTURE} | encoder={ENCODER_NAME} | "
        f"epochs={NUM_EPOCHS} | freeze_encoder={FREEZE_ENCODER}"
    )

    if RUN_SMP_TRAINING:
        train_smp_model(manifest, num_channels=num_channels, device=device)
        print(f"SMP training complete under {MODEL_OUT}")
    else:
        print("training disabled; set GEOAI_RUN_SMP_TRAINING=1 to launch fixed-split SMP fine-tuning.")

    history_path = find_training_history_path(MODEL_OUT)
    if history_path is not None:
        history_summary = load_training_history_summary(history_path)
        if not history_summary.empty:
            print("latest SMP training metrics:")
            print(history_summary.tail().to_string(index=False))

    if RUN_SMP_PREVIEW:
        try:
            model_path = find_best_model(MODEL_OUT)
        except FileNotFoundError as exc:
            print(f"preview skipped: {exc}")
        else:
            import geoai

            preview_metrics = preview_rows(geoai, manifest, model_path=model_path, num_channels=num_channels)
            if not preview_metrics.empty:
                print("preview metrics:")
                print(
                    preview_metrics[
                        [
                            "tile_id",
                            "dataset_split",
                            "iou",
                            "f1",
                            "precision",
                            "recall",
                            "pred_positive_pixels",
                            "target_positive_pixels",
                        ]
                    ].to_string(index=False)
                )
            preview_external_rows(geoai, manifest, model_path=model_path, num_channels=num_channels)
            if RUN_SMP_EVAL:
                evaluation_metrics, building_metrics = evaluate_split_rows(
                    geoai,
                    manifest,
                    model_path=model_path,
                    num_channels=num_channels,
                )
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



