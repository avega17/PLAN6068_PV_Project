"""Shared segmentation metrics, vectorization, and building-level eval helpers."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import box, shape


def load_binary_mask(mask_path: Path, *, threshold: int | float = 0) -> np.ndarray:
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return (np.asarray(mask) > threshold).astype(np.uint8)


def mask_has_positive_pixels(mask_path: Path, *, threshold: int | float = 0) -> bool:
    return bool(load_binary_mask(mask_path, threshold=threshold).any())


def compute_binary_mask_metrics(
    predicted_mask_path: Path,
    target_mask_path: Path,
    *,
    threshold: int | float = 0,
) -> dict[str, float | int]:
    predicted = load_binary_mask(predicted_mask_path, threshold=threshold).astype(bool)
    target = load_binary_mask(target_mask_path, threshold=threshold).astype(bool)
    if predicted.shape != target.shape:
        raise ValueError(
            "predicted and target masks must share the same shape; "
            f"got {predicted.shape} vs {target.shape}"
        )

    intersection = int(np.logical_and(predicted, target).sum())
    union = int(np.logical_or(predicted, target).sum())
    predicted_positive_pixels = int(predicted.sum())
    target_positive_pixels = int(target.sum())
    precision = intersection / predicted_positive_pixels if predicted_positive_pixels else 0.0
    recall = intersection / target_positive_pixels if target_positive_pixels else 0.0
    iou = intersection / union if union else 0.0
    f1 = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "intersection_pixels": intersection,
        "union_pixels": union,
        "pred_positive_pixels": predicted_positive_pixels,
        "target_positive_pixels": target_positive_pixels,
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def load_prompt_artifact(prompt_artifact_path: Path) -> dict[str, object]:
    return json.loads(prompt_artifact_path.read_text())


def prompt_artifact_to_building_geodataframe(
    prompt_artifact_path: Path,
    *,
    geometry_key: str = "building_bbox_model_crs",
) -> gpd.GeoDataFrame:
    payload = load_prompt_artifact(prompt_artifact_path)
    crs = payload.get("crs") or None
    rows: list[dict[str, object]] = []

    for prompt in payload.get("prompts", []):
        if not isinstance(prompt, dict):
            continue
        bounds = prompt.get(geometry_key) or prompt.get("bbox_model_crs")
        if not isinstance(bounds, list) or len(bounds) != 4:
            continue
        geometry = box(*[float(value) for value in bounds])
        if geometry.is_empty:
            continue
        rows.append(
            {
                "building_id": str(prompt.get("building_id")) if prompt.get("building_id") is not None else None,
                "municipio": prompt.get("municipio"),
                "municipio_geoid": prompt.get("municipio_geoid"),
                "h3_cell_id": prompt.get("h3_cell_id"),
                "matched_label_count": int(prompt.get("matched_label_count", 0) or 0),
                "geometry": geometry,
            }
        )

    if not rows:
        return gpd.GeoDataFrame(columns=["building_id", "municipio", "municipio_geoid", "h3_cell_id", "matched_label_count", "geometry"], geometry="geometry", crs=crs)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def mask_to_geodataframe(
    mask_path: Path,
    *,
    threshold: int | float = 0,
) -> gpd.GeoDataFrame:
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

    features: list[dict[str, object]] = []
    for geometry, value in shapes(mask, mask=mask > threshold, transform=transform):
        if float(value) <= threshold:
            continue
        polygon = shape(geometry)
        if polygon.is_empty:
            continue
        features.append({"value": int(value), "geometry": polygon})

    if not features:
        return gpd.GeoDataFrame(columns=["value", "geometry"], geometry="geometry", crs=crs)
    return gpd.GeoDataFrame(features, geometry="geometry", crs=crs)


def load_prediction_geometries(
    *,
    predicted_mask_path: Path | None = None,
    predicted_vector_path: Path | None = None,
    threshold: int | float = 0,
) -> gpd.GeoDataFrame:
    if predicted_vector_path is not None and predicted_vector_path.exists():
        gdf = gpd.read_file(predicted_vector_path)
        if gdf.empty:
            return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=gdf.crs)
        gdf = gdf.loc[gdf.geometry.notna()].copy()
        return gdf.loc[~gdf.geometry.is_empty].copy()

    if predicted_mask_path is None:
        raise ValueError("Either predicted_mask_path or predicted_vector_path must be provided.")
    return mask_to_geodataframe(predicted_mask_path, threshold=threshold)


def compute_building_detection_metrics(
    *,
    prompt_artifact_path: Path,
    predicted_mask_path: Path | None = None,
    predicted_vector_path: Path | None = None,
    threshold: int | float = 0,
    geometry_key: str = "building_bbox_model_crs",
) -> dict[str, float | int]:
    reference_buildings = prompt_artifact_to_building_geodataframe(
        prompt_artifact_path,
        geometry_key=geometry_key,
    )
    predicted_geometries = load_prediction_geometries(
        predicted_mask_path=predicted_mask_path,
        predicted_vector_path=predicted_vector_path,
        threshold=threshold,
    )

    if not predicted_geometries.empty and reference_buildings.crs and predicted_geometries.crs and predicted_geometries.crs != reference_buildings.crs:
        predicted_geometries = predicted_geometries.to_crs(reference_buildings.crs)

    matched_reference_ids: set[str] = set()
    matched_prediction_indices: set[int] = set()
    if not reference_buildings.empty and not predicted_geometries.empty:
        reference_ids = reference_buildings["building_id"].fillna("")
        for prediction_index, prediction_geometry in enumerate(predicted_geometries.geometry):
            hits = reference_buildings.geometry.intersects(prediction_geometry)
            if not bool(hits.any()):
                continue
            matched_prediction_indices.add(prediction_index)
            matched_reference_ids.update(
                building_id
                for building_id in reference_ids.loc[hits].tolist()
                if building_id
            )

    reference_building_count = int(len(reference_buildings))
    predicted_building_count = int(len(predicted_geometries))
    matched_reference_building_count = int(len(matched_reference_ids))
    matched_prediction_count = int(len(matched_prediction_indices))

    building_recall = (
        matched_reference_building_count / reference_building_count
        if reference_building_count
        else 0.0
    )
    reference_precision_lower_bound = (
        matched_prediction_count / predicted_building_count
        if predicted_building_count
        else 0.0
    )
    reference_f1_lower_bound = (
        (2.0 * building_recall * reference_precision_lower_bound) / (building_recall + reference_precision_lower_bound)
        if (building_recall + reference_precision_lower_bound) > 0
        else 0.0
    )

    return {
        "reference_building_count": reference_building_count,
        "predicted_building_count": predicted_building_count,
        "matched_reference_building_count": matched_reference_building_count,
        "matched_prediction_count": matched_prediction_count,
        "unmatched_reference_building_count": int(max(0, reference_building_count - matched_reference_building_count)),
        "unmatched_prediction_count": int(max(0, predicted_building_count - matched_prediction_count)),
        "building_recall": float(building_recall),
        "reference_precision_lower_bound": float(reference_precision_lower_bound),
        "reference_f1_lower_bound": float(reference_f1_lower_bound),
        "predicted_to_reference_ratio": float(predicted_building_count / reference_building_count) if reference_building_count else 0.0,
    }


def evaluate_holdout_building_metrics(
    rows: pd.DataFrame,
    *,
    predicted_mask_column: str,
    prompt_artifact_column: str = "prompt_artifact_abs_path",
    predicted_vector_column: str | None = None,
    dataset_split_column: str = "dataset_split",
    tile_id_column: str = "tile_id",
    threshold: int | float = 0,
    geometry_key: str = "building_bbox_model_crs",
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in rows.to_dict(orient="records"):
        prompt_artifact_value = row.get(prompt_artifact_column)
        predicted_mask_value = row.get(predicted_mask_column)
        predicted_vector_value = row.get(predicted_vector_column) if predicted_vector_column else None
        if not prompt_artifact_value or not predicted_mask_value:
            continue

        prompt_artifact_path = Path(str(prompt_artifact_value))
        predicted_mask_path = Path(str(predicted_mask_value))
        predicted_vector_path = Path(str(predicted_vector_value)) if predicted_vector_value else None
        if not prompt_artifact_path.exists() or not predicted_mask_path.exists():
            continue

        metrics = compute_building_detection_metrics(
            prompt_artifact_path=prompt_artifact_path,
            predicted_mask_path=predicted_mask_path,
            predicted_vector_path=predicted_vector_path if predicted_vector_path and predicted_vector_path.exists() else None,
            threshold=threshold,
            geometry_key=geometry_key,
        )
        records.append(
            {
                tile_id_column: row.get(tile_id_column),
                dataset_split_column: row.get(dataset_split_column),
                "prompt_artifact_path": str(prompt_artifact_path),
                "predicted_mask_path": str(predicted_mask_path),
                "predicted_vector_path": str(predicted_vector_path) if predicted_vector_path else None,
                **metrics,
            }
        )

    return pd.DataFrame.from_records(records)


def summarize_holdout_building_metrics(
    metrics_df: pd.DataFrame,
    *,
    group_column: str = "dataset_split",
) -> dict[str, dict[str, float | int]]:
    if metrics_df.empty:
        return {}

    def _summarize_frame(frame: pd.DataFrame) -> dict[str, float | int]:
        return {
            "row_count": int(len(frame)),
            "reference_building_count": int(frame["reference_building_count"].sum()),
            "predicted_building_count": int(frame["predicted_building_count"].sum()),
            "matched_reference_building_count": int(frame["matched_reference_building_count"].sum()),
            "matched_prediction_count": int(frame["matched_prediction_count"].sum()),
            "mean_building_recall": float(frame["building_recall"].mean()),
            "median_building_recall": float(frame["building_recall"].median()),
            "mean_reference_precision_lower_bound": float(frame["reference_precision_lower_bound"].mean()),
            "mean_reference_f1_lower_bound": float(frame["reference_f1_lower_bound"].mean()),
            "mean_predicted_to_reference_ratio": float(frame["predicted_to_reference_ratio"].mean()),
        }

    summary = {"pooled": _summarize_frame(metrics_df)}
    if group_column in metrics_df.columns:
        for group_value, frame in metrics_df.groupby(group_column):
            summary[str(group_value)] = _summarize_frame(frame)
    return summary


def vectorize_binary_mask(
    mask_path: Path,
    output_path: Path,
    *,
    threshold: int | float = 0,
) -> Path | None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    gdf = mask_to_geodataframe(mask_path, threshold=threshold)
    if gdf.empty:
        return None
    gdf.to_file(output_path, driver="GeoJSON")
    return output_path