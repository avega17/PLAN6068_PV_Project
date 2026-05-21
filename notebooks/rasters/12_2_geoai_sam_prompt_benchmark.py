# %% [markdown]
# # SAM3 Prompt Benchmark
#
# Prepares prompt-benchmark requests from the grounded training manifest and the
# `prompt_artifacts/` JSON files exported by `11_geoai_training_data.py`, then
# runs SAM3 instance-prompt inference when `GEOAI_RUN_SAM_BENCHMARK=1`.

# %%
"""12_2_geoai_sam_prompt_benchmark.py"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import hashlib
from pathlib import Path

import pandas as pd
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
from utils.geoai_segmentation import (
    compute_binary_mask_metrics,
    mask_has_positive_pixels,
    vectorize_binary_mask,
)


def _resolve_configured_path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


DEFAULT_TRAIN_ROOT = PROJECT_ROOT / "outputs" / "geoai_train_contextily"
TRAIN_ROOT = _resolve_configured_path("GEOAI_TRAIN_ROOT", DEFAULT_TRAIN_ROOT)
TRAIN_MANIFEST = TRAIN_ROOT / "training_chip_manifest.csv"
IMAGES_DIR = TRAIN_ROOT / "images"
RAW_MASKS_DIR = TRAIN_ROOT / "masks"
GROUNDED_MASKS_DIR = TRAIN_ROOT / "grounded_masks"
PROMPT_DIR = TRAIN_ROOT / "prompt_artifacts"
BENCHMARK_ROOT = _resolve_configured_path("GEOAI_SAM_BENCHMARK_ROOT", PROJECT_ROOT / "outputs" / "sam_prompt_benchmark")
REQUESTS_CSV = BENCHMARK_ROOT / "sam_prompt_requests.csv"
SUMMARY_JSON = BENCHMARK_ROOT / "sam_prompt_summary.json"
METRICS_CSV = BENCHMARK_ROOT / "sam_prompt_metrics.csv"
PREDICTION_ROOT = BENCHMARK_ROOT / "predictions"
REVIEW_ROOT = BENCHMARK_ROOT / "review"

SAM_VARIANT = (os.getenv("GEOAI_SAM_VARIANT", "sam3") or "sam3").strip().lower()
PROMPT_SPLIT = (os.getenv("GEOAI_SAM_PROMPT_SPLIT", "val") or "val").strip().lower()
PROMPT_SAMPLE_COUNT = int(os.getenv("GEOAI_SAM_PROMPT_SAMPLE_COUNT", "12") or "12")
PROMPT_REVIEW_COUNT = int(os.getenv("GEOAI_SAM_PROMPT_REVIEW_COUNT", "4") or "4")
SEED = int(os.getenv("GEOAI_SAM_SEED", "42") or "42")
RUN_SAM_BENCHMARK = os.getenv("GEOAI_RUN_SAM_BENCHMARK", "0") == "1"
SAM3_BACKEND = (os.getenv("GEOAI_SAM3_BACKEND", "auto") or "auto").strip().lower()
SAM3_MODEL_ID = (os.getenv("GEOAI_SAM3_MODEL_ID", "facebook/sam3") or "facebook/sam3").strip()
SAM_DEVICE = (os.getenv("GEOAI_SAM_DEVICE", "") or "").strip() or None
SAM_CONFIDENCE_THRESHOLD = float(os.getenv("GEOAI_SAM_CONFIDENCE_THRESHOLD", "0.5") or "0.5")
SAM_MASK_THRESHOLD = float(os.getenv("GEOAI_SAM_MASK_THRESHOLD", "0.5") or "0.5")
SAM_MIN_MASK_SIZE = int(os.getenv("GEOAI_SAM_MIN_MASK_SIZE", "0") or "0")
SAM_MAX_MASK_SIZE = int(os.getenv("GEOAI_SAM_MAX_MASK_SIZE", "0") or "0") or None
SAM_MAX_PROMPTS_PER_TILE = int(os.getenv("GEOAI_SAM_MAX_PROMPTS_PER_TILE", "0") or "0")
SAM_BOX_MULTIMASK_OUTPUT = os.getenv("GEOAI_SAM_BOX_MULTIMASK_OUTPUT", "0") == "1"
SAM_POINT_MULTIMASK_OUTPUT = os.getenv("GEOAI_SAM_POINT_MULTIMASK_OUTPUT", "0") == "1"
SAM_USE_POINT_FALLBACK = os.getenv("GEOAI_SAM_USE_POINT_FALLBACK", "1") == "1"
SAM_UNIQUE_MASKS = os.getenv("GEOAI_SAM_UNIQUE_MASKS", "1") == "1"
SAM_ALLOW_BACKEND_FALLBACK = os.getenv("GEOAI_SAM_ALLOW_BACKEND_FALLBACK", "1") == "1"
SAM_PROMPT_MODE = (os.getenv("GEOAI_SAM_PROMPT_MODE", "boxes") or "boxes").strip().lower()
VALID_SAM_PROMPT_MODES = {"boxes", "points", "boxes_then_points"}
HF_TOKEN_ENV_VARS = ("HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def assign_dataset_split(key: object) -> str:
    digest = hashlib.sha1(str(key).encode("utf-8")).hexdigest()
    fraction = int(digest[:12], 16) / float((16**12) - 1)
    if fraction < 0.70:
        return "train"
    if fraction < 0.85:
        return "val"
    return "test"


def build_manifest_from_training_root() -> pd.DataFrame:
    if not IMAGES_DIR.exists() or not GROUNDED_MASKS_DIR.exists() or not PROMPT_DIR.exists():
        raise FileNotFoundError(
            f"training directories not found under {TRAIN_ROOT}; run 11_geoai_training_data.py first"
        )

    image_files = {path.stem: path for path in sorted(IMAGES_DIR.iterdir()) if path.is_file()}
    grounded_mask_files = {path.stem: path for path in sorted(GROUNDED_MASKS_DIR.iterdir()) if path.is_file()}
    raw_mask_files = {path.stem: path for path in sorted(RAW_MASKS_DIR.iterdir()) if path.is_file()} if RAW_MASKS_DIR.exists() else {}
    prompt_files = {path.stem: path for path in sorted(PROMPT_DIR.iterdir()) if path.is_file() and path.suffix.lower() == ".json"}
    common_stems = sorted(set(image_files) & set(grounded_mask_files) & set(prompt_files))
    if not common_stems:
        raise RuntimeError(f"no prompt benchmark pairs found under {TRAIN_ROOT}")

    rows = []
    for stem in common_stems:
        raw_mask_path = raw_mask_files.get(stem)
        rows.append(
            {
                "tile_id": stem,
                "image_path": str(image_files[stem].relative_to(PROJECT_ROOT)),
                "grounded_mask_path": str(grounded_mask_files[stem].relative_to(PROJECT_ROOT)),
                "raw_mask_path": str(raw_mask_path.relative_to(PROJECT_ROOT)) if raw_mask_path is not None else None,
                "prompt_artifact_path": str(prompt_files[stem].relative_to(PROJECT_ROOT)),
                "dataset_split": assign_dataset_split(stem),
            }
        )
    return pd.DataFrame.from_records(rows)


def load_grounded_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else build_manifest_from_training_root()
    required = {"image_path", "prompt_artifact_path", "grounded_mask_path", "dataset_split"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise RuntimeError(
            "prompt benchmark manifest is missing required columns: " + ", ".join(missing)
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
    manifest["prompt_abs_path"] = manifest["prompt_artifact_path"].map(
        lambda value: (PROJECT_ROOT / str(value)).resolve()
    )
    manifest["dataset_split"] = manifest["dataset_split"].fillna("train").astype(str).str.lower()
    exists_mask = (
        manifest["image_abs_path"].map(Path.exists)
        & manifest["grounded_mask_abs_path"].map(Path.exists)
        & manifest["prompt_abs_path"].map(Path.exists)
    )
    manifest = manifest[exists_mask].reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError("no usable prompt benchmark rows found in the grounded manifest.")
    return manifest


def load_prompt_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def build_prompt_requests(manifest: pd.DataFrame, split_name: str, sample_count: int) -> pd.DataFrame:
    subset = manifest[manifest["dataset_split"] == split_name].copy()
    if subset.empty:
        return subset

    rng = random.Random(SEED)
    indices = list(subset.index)
    rng.shuffle(indices)
    subset = subset.loc[indices[: min(sample_count, len(indices))]].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for row in subset.itertuples(index=False):
        payload = load_prompt_payload(Path(row.prompt_abs_path))
        prompts = payload.get("prompts") if isinstance(payload.get("prompts"), list) else []
        first_prompt = prompts[0] if prompts else {}
        rows.append(
            {
                "tile_id": row.tile_id,
                "dataset_split": row.dataset_split,
                "image_path": row.image_path,
                "prompt_artifact_path": row.prompt_artifact_path,
                "grounded_mask_path": row.grounded_mask_path,
                "raw_mask_path": getattr(row, "raw_mask_path", getattr(row, "mask_path", None)),
                "prompt_source": payload.get("prompt_source", "legacy_building_prompt"),
                "prompt_count": int(payload.get("prompt_count", len(prompts))),
                "building_id": first_prompt.get("building_id"),
                "prompt_geometry_kind": first_prompt.get("prompt_geometry_kind", "building_footprint"),
                "bbox_pixels": json.dumps(first_prompt.get("bbox_pixels")),
                "centroid_pixels": json.dumps(first_prompt.get("centroid_pixels")),
                "matched_label_count": int(first_prompt.get("matched_label_count", 0) or 0),
            }
        )
    return pd.DataFrame.from_records(rows)


def write_prompt_summary(requests: pd.DataFrame) -> None:
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    requests.to_csv(REQUESTS_CSV, index=False)
    summary = {
        "benchmark_root": str(BENCHMARK_ROOT.relative_to(PROJECT_ROOT)),
        "request_count": int(len(requests)),
        "split_counts": requests["dataset_split"].value_counts().to_dict() if not requests.empty else {},
        "prompt_count_total": int(requests["prompt_count"].sum()) if not requests.empty else 0,
        "median_prompt_count": float(requests["prompt_count"].median()) if not requests.empty else 0.0,
        "prompt_source_counts": requests["prompt_source"].value_counts().to_dict() if not requests.empty else {},
        "prompt_geometry_kind_counts": requests["prompt_geometry_kind"].value_counts().to_dict() if not requests.empty else {},
        "sam_variant": SAM_VARIANT,
        "sam_prompt_mode": SAM_PROMPT_MODE,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")


def update_prompt_summary(metrics: pd.DataFrame) -> None:
    if SUMMARY_JSON.exists():
        summary = json.loads(SUMMARY_JSON.read_text())
    else:
        summary = {}

    summary["benchmark_run"] = {
        "row_count": int(len(metrics)),
        "successful_prediction_count": int(metrics["pred_positive_pixels"].gt(0).sum()) if not metrics.empty else 0,
        "strategy_counts": metrics["prompt_strategy"].value_counts().to_dict() if not metrics.empty else {},
        "mean_iou": float(metrics["iou"].mean()) if not metrics.empty else 0.0,
        "mean_f1": float(metrics["f1"].mean()) if not metrics.empty else 0.0,
        "mean_precision": float(metrics["precision"].mean()) if not metrics.empty else 0.0,
        "mean_recall": float(metrics["recall"].mean()) if not metrics.empty else 0.0,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")


def render_prompt_review_samples(manifest: pd.DataFrame, requests: pd.DataFrame) -> None:
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    request_lookup = requests.set_index("tile_id").to_dict(orient="index") if not requests.empty else {}
    chosen_tiles = requests["tile_id"].head(PROMPT_REVIEW_COUNT).tolist()
    sample_rows = manifest[manifest["tile_id"].isin(chosen_tiles)].reset_index(drop=True)

    for row in sample_rows.itertuples(index=False):
        request_row = request_lookup.get(row.tile_id, {})
        prompt_count = int(request_row.get("prompt_count", 0) or 0)
        render_prediction_review_bundle(
            image_path=Path(row.image_abs_path),
            output_stem=REVIEW_ROOT / f"{row.dataset_split}_{Path(row.image_path).stem}",
            raw_mask_path=Path(row.raw_mask_abs_path) if row.raw_mask_abs_path else None,
            grounded_mask_path=Path(row.grounded_mask_abs_path),
            suptitle=f"SAM prompt scaffold | {row.dataset_split} | prompts={prompt_count}",
        )


def resolve_sam_status() -> dict[str, object]:
    token, token_source = resolve_hf_token()
    status = {
        "samgeo_installed": importlib.util.find_spec("samgeo") is not None,
        "requested_variant": SAM_VARIANT,
        "sam3_backend": SAM3_BACKEND,
        "sam3_model_id": SAM3_MODEL_ID,
        "hf_token_present": token is not None,
        "hf_token_source": token_source,
        "recommended_install": "pip install segment-geospatial[samgeo3]",
        "sam3_note": "SAM3 also requires Hugging Face access to the documented weights.",
    }
    return status


def resolve_hf_token() -> tuple[str | None, str | None]:
    for env_name in HF_TOKEN_ENV_VARS:
        token = (os.getenv(env_name, "") or "").strip()
        if token:
            return token, env_name
    return None, None


def authenticate_huggingface() -> str:
    token, token_source = resolve_hf_token()
    if token is None:
        raise RuntimeError(
            "No Hugging Face token found. Set one of: " + ", ".join(HF_TOKEN_ENV_VARS)
        )

    from huggingface_hub import login

    login(token=token, add_to_git_credential=False, skip_if_logged_in=True)
    return token_source or "unknown"


def extract_prompt_inputs(payload: dict[str, object]) -> tuple[list[dict[str, object]], list[list[float]], list[list[float]]]:
    prompts = payload.get("prompts") if isinstance(payload.get("prompts"), list) else []
    if SAM_MAX_PROMPTS_PER_TILE > 0:
        prompts = prompts[:SAM_MAX_PROMPTS_PER_TILE]

    boxes: list[list[float]] = []
    points: list[list[float]] = []
    for prompt in prompts:
        bbox = prompt.get("bbox_pixels")
        if isinstance(bbox, list) and len(bbox) == 4:
            boxes.append([float(value) for value in bbox])
        centroid = prompt.get("centroid_pixels")
        if isinstance(centroid, list) and len(centroid) == 2:
            points.append([float(value) for value in centroid])
    return prompts, boxes, points


def initialize_sam_model():
    if SAM_VARIANT != "sam3":
        raise NotImplementedError(
            f"This notebook currently enables SAM3 runtime execution only; got GEOAI_SAM_VARIANT={SAM_VARIANT!r}."
        )

    from samgeo import SamGeo3

    if SAM3_BACKEND == "auto":
        backend_candidates = ["meta", "transformers"]
    else:
        backend_candidates = [SAM3_BACKEND]
        if SAM3_BACKEND == "meta" and SAM_ALLOW_BACKEND_FALLBACK:
            backend_candidates.append("transformers")

    failures: list[str] = []
    for backend in backend_candidates:
        try:
            model = SamGeo3(
                backend=backend,
                model_id=SAM3_MODEL_ID,
                device=SAM_DEVICE,
                load_from_HF=True,
                enable_inst_interactivity=backend == "meta",
                confidence_threshold=SAM_CONFIDENCE_THRESHOLD,
                mask_threshold=SAM_MASK_THRESHOLD,
            )
            print(f"initialized SAM3 with backend={backend}")
            return model
        except Exception as exc:
            failures.append(f"{backend}: {exc}")

    raise RuntimeError("failed to initialize any SAM3 backend: " + " | ".join(failures))


def _mask_dtype() -> str:
    return "uint16" if SAM_UNIQUE_MASKS else "uint8"


def save_sam_masks(sam_model, predicted_mask_path: Path, score_map_path: Path) -> None:
    sam_model.save_masks(
        output=str(predicted_mask_path),
        save_scores=str(score_map_path),
        unique=SAM_UNIQUE_MASKS,
        min_size=SAM_MIN_MASK_SIZE,
        max_size=SAM_MAX_MASK_SIZE,
        dtype=_mask_dtype(),
    )


def _relative_or_none(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.relative_to(PROJECT_ROOT))


def run_sam_inference_for_request(sam_model, manifest_row: pd.Series, request_row) -> dict[str, object]:
    image_path = Path(manifest_row["image_abs_path"])
    grounded_mask_path = Path(manifest_row["grounded_mask_abs_path"])
    raw_mask_path = Path(manifest_row["raw_mask_abs_path"]) if manifest_row.get("raw_mask_abs_path") else None
    prompt_payload = load_prompt_payload(Path(manifest_row["prompt_abs_path"]))
    prompts, boxes, points = extract_prompt_inputs(prompt_payload)
    prompt_geometry_kinds = sorted(
        {
            str(prompt.get("prompt_geometry_kind", "building_footprint"))
            for prompt in prompts
        }
    )

    output_stub = f"{request_row.tile_id}_{SAM_VARIANT}"
    predicted_mask_path = PREDICTION_ROOT / f"{output_stub}_mask.tif"
    score_map_path = PREDICTION_ROOT / f"{output_stub}_scores.tif"
    vector_path = PREDICTION_ROOT / f"{output_stub}_pred.geojson"
    review_stem = REVIEW_ROOT / f"{request_row.dataset_split}_{output_stub}"
    PREDICTION_ROOT.mkdir(parents=True, exist_ok=True)
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    for path in (predicted_mask_path, score_map_path, vector_path):
        if path.exists():
            path.unlink()

    prompt_strategy = "empty"
    errors: list[str] = []
    used_prompt_count = int(len(prompts))
    try_boxes = SAM_PROMPT_MODE in {"boxes", "boxes_then_points"}
    try_points = SAM_PROMPT_MODE in {"points", "boxes_then_points"}

    if try_boxes and boxes:
        try:
            sam_model.set_image(str(image_path))
            sam_model.generate_masks_by_boxes(
                boxes=boxes,
                min_size=SAM_MIN_MASK_SIZE,
                max_size=SAM_MAX_MASK_SIZE,
            )
            save_sam_masks(sam_model, predicted_mask_path, score_map_path)
            if predicted_mask_path.exists() and mask_has_positive_pixels(predicted_mask_path):
                prompt_strategy = "boxes"
        except Exception as exc:
            errors.append(f"boxes: {exc}")

    if (
        prompt_strategy == "empty"
        and try_points
        and SAM_USE_POINT_FALLBACK
        and points
        and getattr(sam_model, "backend", None) == "meta"
    ):
        try:
            if predicted_mask_path.exists():
                predicted_mask_path.unlink()
            if score_map_path.exists():
                score_map_path.unlink()
            sam_model.set_image(str(image_path))
            sam_model.generate_masks_by_points(
                point_coords=points,
                point_labels=[1] * len(points),
                min_size=SAM_MIN_MASK_SIZE,
                max_size=SAM_MAX_MASK_SIZE,
                multimask_output=SAM_POINT_MULTIMASK_OUTPUT,
            )
            save_sam_masks(sam_model, predicted_mask_path, score_map_path)
            if predicted_mask_path.exists() and mask_has_positive_pixels(predicted_mask_path):
                prompt_strategy = "points"
        except Exception as exc:
            errors.append(f"points: {exc}")
    elif prompt_strategy == "empty" and try_points and points and getattr(sam_model, "backend", None) != "meta":
        errors.append("points: point prompts require the meta SAM3 backend in samgeo")

    metrics: dict[str, float | int]
    vectorized_path: Path | None = None
    if predicted_mask_path.exists():
        metrics = compute_binary_mask_metrics(predicted_mask_path, grounded_mask_path)
        if mask_has_positive_pixels(predicted_mask_path):
            vectorized_path = vectorize_binary_mask(predicted_mask_path, vector_path)
    else:
        grounded_metrics = compute_binary_mask_metrics(grounded_mask_path, grounded_mask_path)
        metrics = {
            **grounded_metrics,
            "intersection_pixels": 0,
            "union_pixels": int(grounded_metrics["target_positive_pixels"]),
            "pred_positive_pixels": 0,
            "iou": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    review_outputs = render_prediction_review_bundle(
        image_path=image_path,
        output_stem=review_stem,
        predicted_mask_path=predicted_mask_path if predicted_mask_path.exists() else None,
        raw_mask_path=raw_mask_path if raw_mask_path is not None and raw_mask_path.exists() else None,
        grounded_mask_path=grounded_mask_path,
        vector_path=vectorized_path,
        suptitle=f"SAM3 | {request_row.dataset_split} | {request_row.tile_id} | {prompt_strategy}",
    )

    return {
        "tile_id": request_row.tile_id,
        "dataset_split": request_row.dataset_split,
        "sam_variant": SAM_VARIANT,
        "prompt_source": prompt_payload.get("prompt_source", "legacy_building_prompt"),
        "prompt_geometry_kind": ",".join(prompt_geometry_kinds) if prompt_geometry_kinds else "building_footprint",
        "prompt_strategy": prompt_strategy,
        "prompt_count": int(request_row.prompt_count),
        "used_prompt_count": used_prompt_count,
        "box_prompt_count": int(len(boxes)),
        "point_prompt_count": int(len(points)),
        "predicted_mask_path": _relative_or_none(predicted_mask_path) if predicted_mask_path.exists() else None,
        "score_map_path": _relative_or_none(score_map_path) if score_map_path.exists() else None,
        "vector_path": _relative_or_none(vectorized_path),
        "review_png_path": _relative_or_none(review_outputs.get("review_png_path")),
        "vector_review_png_path": _relative_or_none(review_outputs.get("vector_review_png_path")),
        "error": "; ".join(errors) if errors else None,
        **metrics,
    }


def run_sam_benchmark(manifest: pd.DataFrame, requests: pd.DataFrame) -> pd.DataFrame:
    token_source = authenticate_huggingface()
    print(f"huggingface login succeeded via {token_source}")
    sam_model = initialize_sam_model()

    manifest_by_tile = manifest.set_index("tile_id", drop=False)
    records: list[dict[str, object]] = []
    for request_row in requests.itertuples(index=False):
        if request_row.tile_id not in manifest_by_tile.index:
            records.append(
                {
                    "tile_id": request_row.tile_id,
                    "dataset_split": request_row.dataset_split,
                    "sam_variant": SAM_VARIANT,
                    "prompt_strategy": "missing_manifest_row",
                    "prompt_count": int(request_row.prompt_count),
                    "used_prompt_count": 0,
                    "box_prompt_count": 0,
                    "point_prompt_count": 0,
                    "predicted_mask_path": None,
                    "score_map_path": None,
                    "vector_path": None,
                    "review_png_path": None,
                    "vector_review_png_path": None,
                    "error": "tile_id missing from grounded manifest",
                    "intersection_pixels": 0,
                    "union_pixels": 0,
                    "pred_positive_pixels": 0,
                    "target_positive_pixels": 0,
                    "iou": 0.0,
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                }
            )
            continue

        manifest_row = manifest_by_tile.loc[request_row.tile_id]
        try:
            records.append(run_sam_inference_for_request(sam_model, manifest_row, request_row))
        except Exception as exc:
            records.append(
                {
                    "tile_id": request_row.tile_id,
                    "dataset_split": request_row.dataset_split,
                    "sam_variant": SAM_VARIANT,
                    "prompt_strategy": "error",
                    "prompt_count": int(request_row.prompt_count),
                    "used_prompt_count": 0,
                    "box_prompt_count": 0,
                    "point_prompt_count": 0,
                    "predicted_mask_path": None,
                    "score_map_path": None,
                    "vector_path": None,
                    "review_png_path": None,
                    "vector_review_png_path": None,
                    "error": str(exc),
                    "intersection_pixels": 0,
                    "union_pixels": 0,
                    "pred_positive_pixels": 0,
                    "target_positive_pixels": 0,
                    "iou": 0.0,
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                }
            )

    metrics = pd.DataFrame.from_records(records)
    metrics.to_csv(METRICS_CSV, index=False)
    update_prompt_summary(metrics)
    return metrics


# %%
if __name__ == "__main__":
    if SAM_PROMPT_MODE not in VALID_SAM_PROMPT_MODES:
        raise RuntimeError(
            f"unsupported GEOAI_SAM_PROMPT_MODE={SAM_PROMPT_MODE!r}; expected one of {sorted(VALID_SAM_PROMPT_MODES)}"
        )

    manifest = load_grounded_manifest(TRAIN_MANIFEST)
    requests = build_prompt_requests(manifest, PROMPT_SPLIT, PROMPT_SAMPLE_COUNT)
    if requests.empty:
        print(f"no prompt benchmark rows found for split={PROMPT_SPLIT!r}")
        sys.exit(0)

    legacy_prompt_rows = int(requests["prompt_geometry_kind"].eq("building_footprint").sum())
    if legacy_prompt_rows:
        print(
            "warning: prompt artifacts appear to use legacy building-footprint prompts; "
            "rerun 11_geoai_training_data.py to export pv_label_union prompts before benchmarking."
        )

    write_prompt_summary(requests)
    render_prompt_review_samples(manifest, requests)

    print(f"wrote {len(requests):,} SAM prompt benchmark requests to {REQUESTS_CSV}")
    print(f"wrote prompt-benchmark summary to {SUMMARY_JSON}")
    print(f"wrote review artifacts under {REVIEW_ROOT}")

    sam_status = resolve_sam_status()
    print("SAM dependency status:")
    print(json.dumps(sam_status, indent=2))

    if not sam_status["samgeo_installed"]:
        print(
            "samgeo is not installed in the current environment; this scaffold stops after "
            "request export and review generation. Install the dependency, then implement the "
            "backend-specific benchmark call path in this notebook."
        )
        sys.exit(0)

    if RUN_SAM_BENCHMARK:
        metrics = run_sam_benchmark(manifest, requests)
        print(f"wrote {len(metrics):,} SAM benchmark metric rows to {METRICS_CSV}")
        if not metrics.empty:
            print("benchmark metrics preview:")
            print(
                metrics[
                    [
                        "tile_id",
                        "prompt_strategy",
                        "iou",
                        "f1",
                        "precision",
                        "recall",
                        "pred_positive_pixels",
                        "target_positive_pixels",
                    ]
                ].to_string(index=False)
            )
    else:
        print("SAM benchmark execution disabled; set GEOAI_RUN_SAM_BENCHMARK=1 to run SAM3 inference.")