# %% [markdown]
# # GeoAI Solar Detector Fine-Tuning
#
# Fine-tunes Mask R-CNN from the pretrained `geoai.SolarPanelDetector` weights
# on the paired Contextily image/mask chips exported by
# `09_geoai_training_data.py`.

# %%
"""10_geoai_solar_finetune.py"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
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


def _resolve_configured_path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_num_workers() -> int:
    cpu_count = os.cpu_count() or 0
    if cpu_count <= 1:
        return 0
    return min(4, cpu_count - 1)


DEFAULT_TRAIN_ROOT = PROJECT_ROOT / "outputs" / "geoai_train_contextily"
TRAIN_ROOT = _resolve_configured_path("GEOAI_TRAIN_ROOT", DEFAULT_TRAIN_ROOT)
IMAGES = TRAIN_ROOT / "images"
MASKS = TRAIN_ROOT / "masks"
MODEL_OUT = _resolve_configured_path("GEOAI_MODEL_OUT", PROJECT_ROOT / "outputs" / "models")

NUM_CLASSES = 2        # background + PV
NUM_EPOCHS = int(os.getenv("GEOAI_NUM_EPOCHS", "30"))
BATCH_SIZE = int(os.getenv("GEOAI_BATCH_SIZE", "4"))
LEARNING_RATE = float(os.getenv("GEOAI_LEARNING_RATE", "1e-4"))
VAL_SPLIT = float(os.getenv("GEOAI_VAL_SPLIT", "0.2"))
SEED = int(os.getenv("GEOAI_SEED", "42"))
NUM_WORKERS = int(os.getenv("GEOAI_NUM_WORKERS", str(default_num_workers())))
VISUALIZE = os.getenv("GEOAI_VISUALIZE", "0") == "1"
MODEL_OPTIONS = [
    "maskrcnn_resnet50_fpn",
    "maskrcnn_resnet50_fpn_v2",
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "retinanet_resnet50_fpn",
    "retinanet_resnet50_fpn_v2",
]
MODEL_NAME = os.getenv("GEOAI_MODEL_NAME", MODEL_OPTIONS[0])
INSTANCE_LABELS = os.getenv("GEOAI_INSTANCE_LABELS", "0") == "1"
MULTICLASS = os.getenv("GEOAI_MULTICLASS", "0") == "1"
DEVICE_REQUEST = (os.getenv("GEOAI_DEVICE", "auto") or "auto").strip()
SUPPORTED_IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def selected_model_supports_masks(model_name: str) -> bool:
    return model_name.startswith("maskrcnn_")


def collect_raster_files(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    }


def inspect_dataset(images_dir: Path, masks_dir: Path) -> tuple[list[str], int]:
    image_files = collect_raster_files(images_dir)
    mask_files = collect_raster_files(masks_dir)
    image_stems = set(image_files)
    mask_stems = set(mask_files)
    common = sorted(image_stems & mask_stems)
    missing_masks = sorted(image_stems - mask_stems)
    missing_images = sorted(mask_stems - image_stems)

    print(f"images: {len(image_files):,} | masks: {len(mask_files):,} | matched pairs: {len(common):,}")
    if missing_masks:
        print(f"missing masks for {len(missing_masks):,} image stems")
    if missing_images:
        print(f"missing images for {len(missing_images):,} mask stems")

    if not common:
        raise RuntimeError("No matching image/mask stems were found under the training root.")
    if missing_masks or missing_images:
        raise RuntimeError(
            "Training dataset is inconsistent; rerun 09_geoai_training_data.py to rebuild a clean paired dataset."
        )

    sample_image = image_files[common[0]]
    sample_mask = mask_files[common[0]]
    with rasterio.open(sample_image) as src:
        num_channels = src.count
        image_shape = (src.height, src.width)
        image_crs = src.crs
    with rasterio.open(sample_mask) as src:
        if src.count != 1:
            raise RuntimeError(f"Expected single-band masks, found {src.count} bands in {sample_mask}.")
        mask_values = np.unique(src.read(1))
        mask_dtype = src.dtypes[0]
        mask_crs = src.crs

    print(f"sample image shape: {image_shape} | channels: {num_channels} | crs: {image_crs}")
    print(f"sample mask dtype: {mask_dtype} | unique values: {mask_values.tolist()} | crs: {mask_crs}")

    if num_channels not in {1, 3, 4}:
        raise RuntimeError(f"Unsupported image channel count: {num_channels}")
    return common, num_channels


def validate_split_feasibility(pair_count: int, val_split: float) -> None:
    if pair_count < 2:
        raise RuntimeError(
            "Need at least 2 matched image/mask pairs for training; rerun "
            "09_geoai_training_data.py with a larger tile export."
        )

    val_count = int(math.ceil(pair_count * val_split))
    train_count = pair_count - val_count
    if val_count < 1 or train_count < 1:
        raise RuntimeError(
            f"val_split={val_split} is incompatible with {pair_count} training pairs; "
            "adjust GEOAI_VAL_SPLIT or export more tiles."
        )


def resolve_device(request: str) -> torch.device:
    normalized = request.strip().lower()
    if normalized in {"", "auto"}:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GEOAI_DEVICE requested CUDA but torch.cuda.is_available() is False.")
    return device


def describe_device(device: torch.device) -> None:
    print(f"training device: {device}")
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        print(f"cuda device name: {torch.cuda.get_device_name(index)}")
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        print(f"cuda memory free/total: {free_bytes / 1e9:.2f} GB / {total_bytes / 1e9:.2f} GB")


# %%
if __name__ == "__main__":
    if not IMAGES.exists() or not MASKS.exists():
        print(f"expected training data at {TRAIN_ROOT}; run 09_geoai_training_data first.")
        sys.exit(0)

    if MODEL_NAME not in MODEL_OPTIONS:
        print(f"unsupported GEOAI_MODEL_NAME={MODEL_NAME!r}")
        print(f"choose one of: {', '.join(MODEL_OPTIONS)}")
        sys.exit(1)

    if not selected_model_supports_masks(MODEL_NAME):
        print(
            "this notebook stays Mask R-CNN-only for now because the downstream "
            "inference workflow depends on mask outputs."
        )
        sys.exit(1)

    if not (0.0 < VAL_SPLIT < 1.0):
        raise RuntimeError(f"GEOAI_VAL_SPLIT must be between 0 and 1, got {VAL_SPLIT}.")

    matched_stems, num_channels = inspect_dataset(IMAGES, MASKS)
    validate_split_feasibility(len(matched_stems), VAL_SPLIT)
    print(f"validated {len(matched_stems):,} clean image/mask pairs under {TRAIN_ROOT}")

    device = resolve_device(DEVICE_REQUEST)
    describe_device(device)

    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    import geoai

    geoai.train_MaskRCNN_model(
        images_dir=str(IMAGES),
        labels_dir=str(MASKS),
        output_dir=str(MODEL_OUT),
        input_format="directory",
        num_channels=num_channels,
        num_classes=NUM_CLASSES,
        pretrained=True,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        val_split=VAL_SPLIT,
        seed=SEED,
        visualize=VISUALIZE,
        device=device,
        num_workers=NUM_WORKERS,
        model_name=MODEL_NAME,
        instance_labels=INSTANCE_LABELS,
        multiclass=MULTICLASS,
    )
    print(f"training complete — checkpoints under {MODEL_OUT}")
