# %% [markdown]
# # GeoAI Solar Detector Fine-Tuning
# 
# Fine-tunes Mask R-CNN from the pretrained `geoai.SolarPanelDetector` weights
# on the paired Contextily image/mask chips exported by
# `09_geoai_training_data.py`.

# %%
"""10_geoai_solar_finetune.py"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
import rasterio
import numpy as np


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
from utils.geoai_preview_sources import collect_naip_stac_preview_rasters


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
    return min(8, cpu_count - 1)


DEFAULT_TRAIN_ROOT = PROJECT_ROOT / "outputs" / "geoai_train_contextily"
TRAIN_ROOT = _resolve_configured_path("GEOAI_TRAIN_ROOT", DEFAULT_TRAIN_ROOT)
IMAGES = TRAIN_ROOT / "images"
MASKS = TRAIN_ROOT / "masks"
MODEL_OUT = _resolve_configured_path("GEOAI_MODEL_OUT", PROJECT_ROOT / "outputs" / "models")

NUM_CLASSES = 2        # background + PV
NUM_EPOCHS = int(os.getenv("GEOAI_NUM_EPOCHS", "1"))
BATCH_SIZE = int(os.getenv("GEOAI_BATCH_SIZE", "8"))
LEARNING_RATE = float(os.getenv("GEOAI_LEARNING_RATE", "1e-4"))
VAL_SPLIT = float(os.getenv("GEOAI_VAL_SPLIT", "0.2"))
SEED = int(os.getenv("GEOAI_SEED", "42"))
NUM_WORKERS = int(os.getenv("GEOAI_NUM_WORKERS", str(default_num_workers())))
LIBRARY_VISUALIZE_REQUESTED = os.getenv("GEOAI_VISUALIZE", "0") == "1"
MODEL_OPTIONS = [
    'fasterrcnn_mobilenet_v3_large_fpn',
    'fasterrcnn_resnet50_fpn_v2',
    'fcos_resnet50_fpn',
    'maskrcnn_resnet50_fpn',
    'retinanet_resnet50_fpn_v2'
]
MODEL_NAME = os.getenv("GEOAI_MODEL_NAME", MODEL_OPTIONS[3])
DEFAULT_CLASS_NAMES = ["background", "pv"]
INSTANCE_LABELS = os.getenv("GEOAI_INSTANCE_LABELS", "0") == "1"
MULTICLASS = os.getenv("GEOAI_MULTICLASS", "0") == "1"
DEVICE_REQUEST = (os.getenv("GEOAI_DEVICE", "auto") or "auto").strip()
SUPPORTED_IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
PRINT_FREQ = int(os.getenv("GEOAI_PRINT_FREQ", min(1, NUM_EPOCHS // 10)) or "1")
VERBOSE = os.getenv("GEOAI_VERBOSE", "1") == "1"
TRAIN_PREVIEW_COUNT = int(os.getenv("GEOAI_TRAIN_PREVIEW_COUNT", "3") or "3")
EXTERNAL_PREVIEW_COUNT = int(os.getenv("GEOAI_EXTERNAL_PREVIEW_COUNT", "1") or "1")
PREVIEW_SCORE_THRESHOLD = float(os.getenv("GEOAI_PREVIEW_SCORE_THRESHOLD", "0.5") or "0.5")
STAC_TILE_ROOT = PROJECT_ROOT / "outputs" / "stac_tiles"
LOCAL_STAC_CACHE_ROOT = PROJECT_ROOT / "data" / "rasters" / "stac" / "local"
SOLAR_RASTER_ROOT = PROJECT_ROOT / "data" / "rasters" / "solar"
PREVIEW_ROOT = _resolve_configured_path("GEOAI_PREVIEW_ROOT", MODEL_OUT / "preview")
TRAIN_MANIFEST = TRAIN_ROOT / "training_chip_manifest.csv"
PREVIEW_WINDOW_SIZE = int(os.getenv("GEOAI_PREVIEW_WINDOW_SIZE", "512") or "512")
PREVIEW_OVERLAP = int(os.getenv("GEOAI_PREVIEW_OVERLAP", "256") or "256")
PREVIEW_BATCH_SIZE = int(os.getenv("GEOAI_PREVIEW_BATCH_SIZE", "4") or "4")
PREVIEW_EPSILON = float(os.getenv("GEOAI_PREVIEW_EPSILON", "0.2") or "0.2")
OVERWRITE_PREVIEW_ARTIFACTS = os.getenv("GEOAI_OVERWRITE_PREVIEW_ARTIFACTS", "0") == "1"
WRITE_STATIC_REVIEW_ARTIFACTS = os.getenv("GEOAI_WRITE_STATIC_REVIEW_ARTIFACTS", "1") == "1"
WRITE_INTERACTIVE_REVIEW_ARTIFACTS = os.getenv("GEOAI_WRITE_INTERACTIVE_REVIEW_ARTIFACTS", "0") == "1"


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


def find_latest_checkpoint(model_dir: Path) -> Path:
    checkpoint_candidates = sorted(
        [
            p
            for p in model_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".pt", ".pth", ".ckpt"}
            and "history" not in p.name.lower()
        ],
        key=lambda p: ("best" not in p.name.lower(), -p.stat().st_mtime),
    )
    if not checkpoint_candidates:
        raise FileNotFoundError(f"No checkpoint found under {model_dir}")
    return checkpoint_candidates[0]


def find_training_history_path(model_dir: Path) -> Path | None:
    candidates = sorted(model_dir.rglob("training_history.pth"), key=lambda p: -p.stat().st_mtime)
    return candidates[0] if candidates else None


def load_training_history_summary(history_path: Path) -> pd.DataFrame:
    history = torch.load(history_path, weights_only=True)
    epoch_count = max((len(values) for values in history.values() if isinstance(values, list)), default=0)
    epochs = history.get("epochs") or list(range(1, epoch_count + 1))
    frame = pd.DataFrame({"epoch": epochs})
    for key in ("train_loss", "val_loss", "val_iou", "lr"):
        values = history.get(key)
        if isinstance(values, list) and values:
            frame[key] = values
    return frame


def collect_external_preview_rasters(exclude_stems: set[str]) -> list[Path]:
    return collect_naip_stac_preview_rasters(
        PROJECT_ROOT,
        exclude_stems=exclude_stems,
        supported_image_extensions=SUPPORTED_IMAGE_EXTENSIONS,
    )


def normalize_checkpoint_state_dict(checkpoint_payload: object) -> dict[str, object]:
    if isinstance(checkpoint_payload, dict):
        for nested_key in ("state_dict", "model_state_dict"):
            nested = checkpoint_payload.get(nested_key)
            if isinstance(nested, dict) and nested:
                checkpoint_payload = nested
                break

    if not isinstance(checkpoint_payload, dict) or not checkpoint_payload:
        raise RuntimeError("Checkpoint does not contain a usable state_dict.")

    state_dict = checkpoint_payload
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value for key, value in state_dict.items()
        }
    return state_dict


def infer_model_name_from_state_dict(state_dict: dict[str, object]) -> str:
    if "roi_heads.mask_predictor.conv5_mask.weight" in state_dict:
        return "maskrcnn_resnet50_fpn"
    if "roi_heads.box_predictor.cls_score.weight" in state_dict:
        if any(key.startswith("roi_heads.mask_predictor.") for key in state_dict):
            return "maskrcnn_resnet50_fpn"
        return "fasterrcnn_resnet50_fpn_v2"
    if "head.classification_head.cls_logits.weight" in state_dict:
        if any(key.startswith("anchor_generator.") for key in state_dict):
            return "retinanet_resnet50_fpn_v2"
        return "fcos_resnet50_fpn"
    return MODEL_NAME


def infer_num_classes_from_state_dict(state_dict: dict[str, object], model_name: str) -> int:
    rcnn_key = "roi_heads.box_predictor.cls_score.weight"
    retina_key = "head.classification_head.cls_logits.weight"
    if rcnn_key in state_dict:
        return int(state_dict[rcnn_key].shape[0])
    if retina_key in state_dict:
        out_channels = int(state_dict[retina_key].shape[0])
        num_anchors = 9 if model_name == "retinanet_resnet50_fpn_v2" else 1
        return max(1, out_channels // num_anchors)
    return NUM_CLASSES


def default_class_names(num_classes: int) -> list[str]:
    if num_classes == len(DEFAULT_CLASS_NAMES):
        return DEFAULT_CLASS_NAMES.copy()
    return ["background", *[f"class_{index}" for index in range(1, num_classes)]]


def resolve_checkpoint_metadata(model_path: Path) -> dict[str, object]:
    class_info_path = model_path.parent / "class_info.json"
    class_info: dict[str, object] = {}
    if class_info_path.exists():
        class_info = json.loads(class_info_path.read_text())

    checkpoint_payload = torch.load(model_path, map_location="cpu")
    state_dict = normalize_checkpoint_state_dict(checkpoint_payload)

    model_name = str(class_info.get("model_name") or infer_model_name_from_state_dict(state_dict))
    num_classes = int(class_info.get("num_classes") or infer_num_classes_from_state_dict(state_dict, model_name))
    class_names = class_info.get("class_names")
    if not isinstance(class_names, list) or len(class_names) != num_classes:
        class_names = default_class_names(num_classes)

    resolved = {
        "model_name": model_name,
        "num_classes": num_classes,
        "class_names": class_names,
    }
    if class_info != resolved:
        class_info_path.write_text(json.dumps(resolved, indent=2) + "\n")
    return resolved


def load_training_review_lookup(manifest_path: Path) -> dict[str, dict[str, object]]:
    if not manifest_path.exists():
        return {}
    frame = pd.read_csv(manifest_path)
    if frame.empty or "image_path" not in frame.columns:
        return {}

    lookup: dict[str, dict[str, object]] = {}
    for row in frame.to_dict(orient="records"):
        image_value = row.get("image_path")
        if not isinstance(image_value, str) or not image_value:
            continue
        image_path = (PROJECT_ROOT / image_value).resolve()
        lookup[str(image_path)] = row
        lookup[Path(image_value).stem] = row
    return lookup


def resolve_ground_truth_paths(
    image_path: Path,
    training_lookup: dict[str, dict[str, object]] | None,
) -> tuple[Path | None, Path | None]:
    if not training_lookup:
        return None, None
    row = training_lookup.get(str(image_path.resolve())) or training_lookup.get(image_path.stem)
    if row is None:
        return None, None

    def _resolve(column_name: str) -> Path | None:
        value = row.get(column_name)
        if not isinstance(value, str) or not value:
            return None
        candidate = PROJECT_ROOT / value
        return candidate if candidate.exists() else None

    raw_mask_path = _resolve("raw_mask_path") or _resolve("mask_path")
    grounded_mask_path = _resolve("grounded_mask_path")
    return raw_mask_path, grounded_mask_path


def run_geoai_preview(
    geoai_module,
    *,
    image_path: Path,
    model_path: Path,
    num_channels: int,
    preview_group: str,
    training_lookup: dict[str, dict[str, object]] | None = None,
) -> dict[str, Path] | None:
    preview_dir = PREVIEW_ROOT / preview_group
    preview_dir.mkdir(parents=True, exist_ok=True)

    output_stub = f"{preview_group}_{image_path.stem}"
    mask_path = preview_dir / f"{output_stub}_mask.tif"
    vector_path = preview_dir / f"{output_stub}_pred.geojson"
    enriched_path = preview_dir / f"{output_stub}_pred_props.geojson"
    review_png_path = preview_dir / f"{output_stub}_review.png"
    review_map_path = preview_dir / f"{output_stub}_review_map.html"
    split_map_path = preview_dir / f"{output_stub}_split_map.html"
    checkpoint_metadata = resolve_checkpoint_metadata(model_path)

    if OVERWRITE_PREVIEW_ARTIFACTS or not mask_path.exists():
        geoai_module.object_detection(
            input_path=str(image_path),
            output_path=str(mask_path),
            model_path=str(model_path),
            model_name=str(checkpoint_metadata["model_name"]),
            num_classes=int(checkpoint_metadata["num_classes"]),
            class_names=list(checkpoint_metadata["class_names"]),
            window_size=PREVIEW_WINDOW_SIZE,
            overlap=PREVIEW_OVERLAP,
            confidence_threshold=PREVIEW_SCORE_THRESHOLD,
            batch_size=PREVIEW_BATCH_SIZE,
            num_channels=num_channels,
        )
    if not mask_path.exists():
        print(f"preview skipped for {image_path.name}: GeoAI did not produce a mask raster.")
        return None

    if OVERWRITE_PREVIEW_ARTIFACTS or not vector_path.exists():
        geoai_module.orthogonalize(str(mask_path), str(vector_path), epsilon=PREVIEW_EPSILON)
    if not vector_path.exists():
        print(f"preview skipped for {image_path.name}: GeoAI did not produce a vector output.")
        return None

    import geopandas as gpd

    try:
        gdf = geoai_module.add_geometric_properties(gpd.read_file(vector_path))
    except Exception:
        gdf = gpd.read_file(vector_path)
    if hasattr(gdf, "empty") and gdf.empty:
        print(f"preview for {image_path.name}: no detections above threshold.")
        if WRITE_STATIC_REVIEW_ARTIFACTS:
            raw_mask_path, grounded_mask_path = resolve_ground_truth_paths(image_path, training_lookup)
            rendered = render_prediction_review_bundle(
                image_path=image_path,
                output_stem=review_png_path.with_suffix(""),
                predicted_mask_path=mask_path,
                raw_mask_path=raw_mask_path,
                grounded_mask_path=grounded_mask_path,
                suptitle=f"{preview_group} | {image_path.stem}",
            )
        else:
            rendered = {}
        return {
            "mask_path": mask_path,
            "vector_path": vector_path,
            **rendered,
        }

    gdf.to_file(enriched_path, driver="GeoJSON")
    raw_mask_path, grounded_mask_path = resolve_ground_truth_paths(image_path, training_lookup)
    rendered = {}
    if WRITE_STATIC_REVIEW_ARTIFACTS:
        rendered = render_prediction_review_bundle(
            image_path=image_path,
            output_stem=review_png_path.with_suffix(""),
            predicted_mask_path=mask_path,
            raw_mask_path=raw_mask_path,
            grounded_mask_path=grounded_mask_path,
            vector_path=enriched_path,
            suptitle=f"{preview_group} | {image_path.stem}",
        )

    if WRITE_INTERACTIVE_REVIEW_ARTIFACTS:
        review_map = geoai_module.view_vector_interactive(str(enriched_path), tiles=str(image_path))
        if hasattr(review_map, "save"):
            review_map.save(str(review_map_path))
        try:
            split_map = geoai_module.create_split_map(
                left_layer=str(mask_path),
                right_layer=str(image_path),
                left_label="Predicted mask",
                right_label="Source image",
            )
            if hasattr(split_map, "save"):
                split_map.save(str(split_map_path))
        except Exception as exc:
            print(f"split-map export skipped for {image_path.name}: {exc}")
    print(f"preview artifacts for {image_path.name}: {enriched_path}")
    return {
        "mask_path": mask_path,
        "vector_path": vector_path,
        "enriched_path": enriched_path,
        **rendered,
        **({"review_map_path": review_map_path} if review_map_path.exists() else {}),
        **({"split_map_path": split_map_path} if split_map_path.exists() else {}),
    }

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
    print(f"training progress settings: verbose={VERBOSE} | print_freq={PRINT_FREQ}")
    if LIBRARY_VISUALIZE_REQUESTED:
        print(
            "training-time GeoAI visualizations are disabled in this notebook because "
            "geoai.train.visualize_predictions can fail on outputs without a 'masks' key; "
            "the post-training preview cell below remains the supported visualization path."
        )

    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    import geoai

    trained_model = geoai.train_MaskRCNN_model(
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
        visualize=False,
        device=device,
        num_workers=NUM_WORKERS,
        print_freq=PRINT_FREQ,
        verbose=VERBOSE,
        model_name=MODEL_NAME,
        instance_labels=INSTANCE_LABELS,
        multiclass=MULTICLASS,
    )
    print(f"training complete — checkpoints under {MODEL_OUT}")

    history_path = find_training_history_path(MODEL_OUT)
    if history_path is None:
        print("training history file not found under the model output directory.")
    else:
        print(f"training history: {history_path}")
        history_summary = load_training_history_summary(history_path)
        if not history_summary.empty:
            print("latest training metrics:")
            print(history_summary.tail().to_string(index=False))
        geoai.plot_detection_training_history(str(history_path))

# %% [markdown]
# ## Preview predictions on training and external tiles
# 
# The first group previews held-out/training-set imagery from the exported chip
# directory. The second group looks for non-training rasters under
# `outputs/stac_tiles/`, then the local STAC cache, then `data/rasters/solar/`.

# %%
if __name__ == "__main__":
    if 'matched_stems' not in locals():
        matched_stems, num_channels = inspect_dataset(IMAGES, MASKS)
    if 'geoai' not in locals():
        import geoai
    if 'num_channels' not in locals():
        _, num_channels = inspect_dataset(IMAGES, MASKS)

    train_image_files = collect_raster_files(IMAGES)
    training_lookup = load_training_review_lookup(TRAIN_MANIFEST)
    best_checkpoint = find_latest_checkpoint(MODEL_OUT)
    print(f"using checkpoint: {best_checkpoint}")
    checkpoint_metadata = resolve_checkpoint_metadata(best_checkpoint)
    print(
        "checkpoint metadata: "
        f"model_name={checkpoint_metadata['model_name']} | "
        f"num_classes={checkpoint_metadata['num_classes']}"
    )
    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)

    rng = __import__('random').Random(SEED)
    training_stems = rng.sample(matched_stems, k=min(TRAIN_PREVIEW_COUNT, len(matched_stems)))
    for stem in training_stems:
        image_path = train_image_files[stem]
        run_geoai_preview(
            geoai,
            image_path=image_path,
            model_path=best_checkpoint,
            num_channels=num_channels,
            preview_group="training_sample",
            training_lookup=training_lookup,
        )

    external_candidates = collect_external_preview_rasters(set(matched_stems))
    if not external_candidates:
        print("no out-of-training preview rasters found under outputs/stac_tiles, data/rasters/stac/local, or data/rasters/solar.")
    else:
        for image_path in external_candidates[:EXTERNAL_PREVIEW_COUNT]:
            run_geoai_preview(
                geoai,
                image_path=image_path,
                model_path=best_checkpoint,
                num_channels=num_channels,
                preview_group="external_tile",
                training_lookup=training_lookup,
            )


