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
from pathlib import Path
from glob import glob

import duckdb
import geopandas as gpd
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

from utils.geoai_preview_sources import collect_naip_stac_preview_rasters, infer_stac_source_name
from utils.geoai_review import render_prediction_review_bundle
from utils.geoai_segmentation import mask_has_positive_pixels, vectorize_binary_mask


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


def _parse_csv_env(env_name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(env_name)
    if not value:
        return default
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    return parts or default


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
SAMPLE_COUNT = max(1, int(os.getenv("GEOAI_SAMPLE_COUNT", "1") or "1"))
SAMPLE_SEED = int(os.getenv("GEOAI_SAMPLE_SEED", "42") or "42")
MIN_TILE_BUILDING_COUNT = int(os.getenv("GEOAI_MIN_TILE_BUILDING_COUNT", "1") or "1")
INCLUDE_CROSS_BOUNDARY_TILES = os.getenv("GEOAI_INCLUDE_CROSS_BOUNDARY_TILES", "1") == "1"
REVIEW_COUNT = int(os.getenv("GEOAI_REVIEW_COUNT", "6") or "6")
OVERWRITE_INFERENCE_ARTIFACTS = os.getenv("GEOAI_OVERWRITE_INFERENCE_ARTIFACTS", "0") == "1"
WRITE_STATIC_REVIEW_ARTIFACTS = os.getenv("GEOAI_WRITE_STATIC_REVIEW_ARTIFACTS", "1") == "1"
WRITE_INTERACTIVE_REVIEW_ARTIFACTS = os.getenv("GEOAI_WRITE_INTERACTIVE_REVIEW_ARTIFACTS", "0") == "1"

WINDOW_SIZE = int(os.getenv("GEOAI_INFERENCE_WINDOW_SIZE", "512") or "512")
OVERLAP = int(os.getenv("GEOAI_INFERENCE_OVERLAP", "256") or "256")
INFERENCE_BATCH_SIZE = int(os.getenv("GEOAI_INFERENCE_BATCH_SIZE", "4") or "4")
PROBABILITY_THRESHOLD = float(os.getenv("GEOAI_INFERENCE_PROBABILITY_THRESHOLD", "0.5") or "0.5")
MIN_AREA_M2 = float(os.getenv("GEOAI_INFERENCE_MIN_AREA_M2", "3.0") or "3.0")
MAX_AREA_M2 = float(os.getenv("GEOAI_INFERENCE_MAX_AREA_M2", "1500.0") or "1500.0")
MAX_ELONGATION = float(os.getenv("GEOAI_INFERENCE_MAX_ELONGATION", "25.0") or "25.0")

DETECTION_TABLE = "pr_solar_pv_detections"
TILE_SUMMARY_TABLE = "pr_solar_pv_detection_tiles"
BUILDING_HITS_TABLE = "pr_solar_pv_building_hits"
MUNICIPIO_SUMMARY_TABLE = "pr_solar_pv_detection_municipio_summary"


def resolve_db_path() -> Path:
    value = os.getenv("VECTOR_DB")
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path if len(path.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / path
        return path
    return PROJECT_ROOT / "data" / "PR_PV_plan_data.duckdb"


def resolve_runtime_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def list_model_run_dirs(base_root: Path, metadata_name: str) -> list[Path]:
    if not base_root.exists():
        return []
    return sorted(
        [path for path in base_root.iterdir() if path.is_dir() and (path / metadata_name).exists()],
        key=lambda path: -path.stat().st_mtime,
    )


def find_latest_dinov3_checkpoint(model_dir: Path) -> Path:
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
        "model_dir": model_dir,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "model_path": model_path,
        "run_name": run_name,
        "inference_root": inference_root,
    }


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
            logits = model_module(tensor)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

            for index, (row_start, row_end, col_start, col_end, h, w) in enumerate(batch_meta):
                votes[:, row_start:row_end, col_start:col_end] += probs[index, :, :h, :w]
                count[row_start:row_end, col_start:col_end] += 1.0

        with torch.no_grad():
            batch_imgs: list[np.ndarray] = []
            batch_meta: list[tuple[int, int, int, int, int, int]] = []
            progress = tqdm(total=n_rows * n_cols, disable=False, desc=f"DINOv3 windows | {input_path.stem}")

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


def run_segmentation_inference(geoai_module, *, raster_path: Path, mask_path: Path, probability_path: Path | None, config: dict[str, object]) -> None:
    metadata = config["metadata"]
    if config["model_family"] == "smp":
        geoai_module.semantic_segmentation(
            input_path=str(raster_path),
            output_path=str(mask_path),
            model_path=str(config["model_path"]),
            architecture=str(metadata["architecture"]),
            encoder_name=str(metadata["encoder_name"]),
            num_channels=int(metadata.get("num_channels", 3) or 3),
            num_classes=int(metadata.get("num_classes", 2) or 2),
            window_size=WINDOW_SIZE,
            overlap=OVERLAP,
            batch_size=INFERENCE_BATCH_SIZE,
            probability_path=str(probability_path) if probability_path is not None else None,
            probability_threshold=PROBABILITY_THRESHOLD,
            quiet=False,
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
    )


def enrich_prediction_vectors(geoai_module, raw_vector_path: Path, enriched_vector_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(raw_vector_path)
    if gdf.empty:
        return gdf
    if hasattr(geoai_module, "add_geometric_properties"):
        gdf = geoai_module.add_geometric_properties(gdf)
    if "area_m2" in gdf.columns:
        gdf = gdf[gdf["area_m2"].between(MIN_AREA_M2, MAX_AREA_M2)].copy()
    if "elongation" in gdf.columns:
        gdf = gdf[gdf["elongation"] < MAX_ELONGATION].copy()
    if gdf.empty:
        return gdf
    enriched_vector_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(enriched_vector_path, driver="GeoJSON")
    return gdf


def render_review_visuals(geoai_module, *, raster_path: Path, mask_path: Path, vector_path: Path, output_stem: Path) -> str | None:
    review_png_path = None
    if WRITE_STATIC_REVIEW_ARTIFACTS:
        review_outputs = render_prediction_review_bundle(
            image_path=raster_path,
            output_stem=output_stem,
            predicted_mask_path=mask_path,
            vector_path=vector_path,
            suptitle=output_stem.name,
        )
        review_png = review_outputs.get("review_png_path")
        if review_png is not None:
            review_png_path = str(review_png.relative_to(PROJECT_ROOT))

    if not WRITE_INTERACTIVE_REVIEW_ARTIFACTS:
        return review_png_path

    try:
        review_map = geoai_module.view_vector_interactive(str(vector_path), tiles=str(raster_path))
        if hasattr(review_map, "save"):
            review_map.save(str(vector_path.with_name(f"{vector_path.stem}_review_map.html")))
    except Exception as exc:
        print(f"interactive review map skipped for {raster_path.name}: {exc}")

    try:
        split_map = geoai_module.create_split_map(
            left_layer=str(mask_path),
            right_layer=str(raster_path),
            left_label="Predicted mask",
            right_label="Source image",
        )
        if hasattr(split_map, "save"):
            split_map.save(str(vector_path.with_name(f"{vector_path.stem}_split_map.html")))
    except Exception as exc:
        print(f"interactive split map skipped for {raster_path.name}: {exc}")
    return review_png_path


def build_detection_frame(gdf: gpd.GeoDataFrame, row: pd.Series | object, config: dict[str, object], *, mask_path: Path, vector_path: Path, probability_path: Path | None) -> gpd.GeoDataFrame:
    gdf = gdf.reset_index(drop=True).copy()
    gdf["detection_key"] = [f"{getattr(row, 'h3_cell_id', None) or Path(str(getattr(row, 'tile_path', 'tile'))).stem}_{index}" for index in range(len(gdf))]
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
    gdf["tile_path"] = str(Path(str(getattr(row, "tile_path", mask_path))).as_posix())
    gdf["local_asset_path"] = str(Path(str(getattr(row, "local_asset_path", ""))).as_posix()) if getattr(row, "local_asset_path", None) else None
    gdf["predicted_mask_path"] = str(mask_path.relative_to(PROJECT_ROOT))
    gdf["probability_path"] = str(probability_path.relative_to(PROJECT_ROOT)) if probability_path is not None and probability_path.exists() else None
    gdf["vector_path"] = str(vector_path.relative_to(PROJECT_ROOT))
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

    munis_sql = ", ".join(f"'{municipio}'" for municipio in TARGET_MUNICIPALITIES)
    model_family = str(config["model_family"])
    run_name = str(config["run_name"])
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
        SELECT
            b.building_id,
            b.municipio,
            b.municipality_geoid,
            COALESCE(d.pv_detected_count, 0) > 0 AS has_pv_detected,
            COALESCE(d.pv_detected_count, 0) AS pv_detected_count,
            COALESCE(d.pv_detection_source_count, 0) AS pv_detection_source_count,
            COALESCE(d.contributing_tile_count, 0) AS contributing_tile_count,
            COALESCE(d.pv_detected_area_deg2, 0.0) AS pv_detected_area_deg2,
            '{model_family}' AS model_family,
            '{run_name}' AS model_run_name,
            b.geometry
        FROM buildings AS b
        LEFT JOIN det_join AS d USING (building_id);
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
                    SUM(pv_detected_area_deg2) AS detected_area_deg2_in_buildings
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
                COALESCE(b.detected_area_deg2_in_buildings, 0.0) AS detected_area_deg2_in_buildings
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


# %%
if __name__ == "__main__":
    config = resolve_model_configuration()
    inference_root = Path(config["inference_root"])
    inference_root.mkdir(parents=True, exist_ok=True)
    config["model_family"] = INFERENCE_MODEL_FAMILY

    import geoai

    candidates = load_naip_tile_candidates()
    if candidates.empty:
        print("no NAIP STAC tiles available for inference.")
        sys.exit(0)

    selected = select_inference_rows(candidates)
    print(
        f"model_family={INFERENCE_MODEL_FAMILY} | run={config['run_name']} | mode={INFERENCE_MODE} | "
        f"selected_tiles={len(selected):,}"
    )
    if selected.empty:
        print("tile selection returned zero rows.")
        sys.exit(0)

    all_detections: list[gpd.GeoDataFrame] = []
    tile_records: list[dict[str, object]] = []

    for row_index, row in enumerate(selected.itertuples(index=False)):
        raster_path = Path(row.tile_abs_path)
        output_stem = build_output_stem(inference_root, row)
        output_stem.parent.mkdir(parents=True, exist_ok=True)

        mask_path = output_stem.with_name(f"{output_stem.name}_pred.tif")
        probability_path = output_stem.with_name(f"{output_stem.name}_prob.tif") if INFERENCE_MODEL_FAMILY == "smp" else None
        raw_vector_path = output_stem.with_name(f"{output_stem.name}_pred.geojson")
        enriched_vector_path = output_stem.with_name(f"{output_stem.name}_pred_props.geojson")

        if OVERWRITE_INFERENCE_ARTIFACTS:
            for artifact_path in (mask_path, probability_path, raw_vector_path, enriched_vector_path):
                if artifact_path is not None and artifact_path.exists():
                    artifact_path.unlink()

        if not mask_path.exists():
            run_segmentation_inference(
                geoai,
                raster_path=raster_path,
                mask_path=mask_path,
                probability_path=probability_path,
                config=config,
            )

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
                    "status": "mask_missing",
                    "detection_count": 0,
                    "detected_area_m2": 0.0,
                }
            )
            continue

        vectorized_path = None
        if mask_has_positive_pixels(mask_path):
            vectorized_path = vectorize_binary_mask(mask_path, raw_vector_path)

        if vectorized_path is None or not vectorized_path.exists():
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
                    "predicted_mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
                    "probability_path": str(probability_path.relative_to(PROJECT_ROOT)) if probability_path is not None and probability_path.exists() else None,
                    "vector_path": None,
                    "review_png_path": None,
                    "inference_mode": INFERENCE_MODE,
                    "status": "no_positive_mask",
                    "detection_count": 0,
                    "detected_area_m2": 0.0,
                }
            )
            continue

        detections = enrich_prediction_vectors(geoai, vectorized_path, enriched_vector_path)
        if detections.empty:
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
                    "predicted_mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
                    "probability_path": str(probability_path.relative_to(PROJECT_ROOT)) if probability_path is not None and probability_path.exists() else None,
                    "vector_path": str(enriched_vector_path.relative_to(PROJECT_ROOT)),
                    "review_png_path": None,
                    "inference_mode": INFERENCE_MODE,
                    "status": "filtered_empty",
                    "detection_count": 0,
                    "detected_area_m2": 0.0,
                }
            )
            continue

        detections = build_detection_frame(
            detections,
            row,
            config,
            mask_path=mask_path,
            vector_path=enriched_vector_path,
            probability_path=probability_path,
        )
        all_detections.append(detections)

        review_png_path = None
        if row_index < REVIEW_COUNT:
            review_png_path = render_review_visuals(
                geoai,
                raster_path=raster_path,
                mask_path=mask_path,
                vector_path=enriched_vector_path,
                output_stem=output_stem,
            )

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
                "predicted_mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
                "probability_path": str(probability_path.relative_to(PROJECT_ROOT)) if probability_path is not None and probability_path.exists() else None,
                "vector_path": str(enriched_vector_path.relative_to(PROJECT_ROOT)),
                "review_png_path": review_png_path,
                "inference_mode": INFERENCE_MODE,
                "status": "ok",
                "detection_count": int(len(detections)),
                "detected_area_m2": detected_area_m2,
            }
        )

    tile_summary_df = pd.DataFrame.from_records(tile_records)
    merged_detections = (
        gpd.GeoDataFrame(pd.concat(all_detections, ignore_index=True), crs=all_detections[0].crs)
        if all_detections
        else gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")
    )

    db_path = resolve_db_path()
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")

    write_detection_table(con, merged_detections)
    write_tile_summary_table(con, tile_summary_df)
    has_building_hits = build_building_hits_table(con, config)
    build_municipio_summary_table(con, has_building_hits)

    print(f"wrote {len(merged_detections):,} detections to {DETECTION_TABLE}")
    print(f"wrote {len(tile_summary_df):,} tile rows to {TILE_SUMMARY_TABLE}")

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

    con.close()

# %%
