"""Helpers for tracking training-data identity across GeoAI notebook runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


HASH_COLUMNS = (
    "tile_id",
    "h3_cell_id",
    "image_path",
    "grounded_mask_path",
    "raw_mask_path",
    "dataset_split",
    "imagery_source",
    "dataset_cohort",
    "split_policy",
    "split_key",
)


def _slugify(value: object) -> str:
    text = str(value).strip().lower()
    slug_chars = [char if char.isalnum() else "-" for char in text]
    slug = "".join(slug_chars)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "default"


def project_relative_path(path: Path | None, project_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path.resolve())


def _single_value(series: pd.Series | None) -> object | None:
    if series is None:
        return None
    cleaned = [value for value in series.dropna().tolist() if str(value).strip()]
    if not cleaned:
        return None
    normalized = {json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value) for value in cleaned}
    if len(normalized) == 1:
        return cleaned[0]
    return None


def _infer_imagery_source(manifest: pd.DataFrame, train_root: Path) -> str:
    if "imagery_source" in manifest.columns:
        value = _single_value(manifest["imagery_source"].astype(str))
        if value:
            return str(value)
    root_slug = train_root.name.lower()
    if "naip" in root_slug:
        return "naip"
    return "esri"


def _infer_dataset_cohort(manifest: pd.DataFrame) -> str:
    if "dataset_cohort" in manifest.columns:
        value = _single_value(manifest["dataset_cohort"].astype(str))
        if value:
            return str(value)
    if "priority_score" in manifest.columns:
        values = sorted({int(value) for value in manifest["priority_score"].dropna().astype(int).tolist()})
        if values == [3]:
            return "holdout_priority3"
    return "train_pool"


def compute_training_file_hash(manifest: pd.DataFrame) -> str:
    columns = [column_name for column_name in HASH_COLUMNS if column_name in manifest.columns]
    if not columns:
        columns = list(manifest.columns)
    ordered_records = (
        manifest[columns]
        .fillna("")
        .astype(str)
        .sort_values(columns, kind="stable")
        .to_dict(orient="records")
    )
    return hashlib.sha1(json.dumps(ordered_records, sort_keys=True).encode("utf-8")).hexdigest()


def build_training_contract(
    manifest: pd.DataFrame,
    *,
    train_root: Path,
    manifest_path: Path,
    project_root: Path,
) -> dict[str, object]:
    dataset_splits = (
        manifest["dataset_split"].fillna("missing").astype(str).str.lower().value_counts().to_dict()
        if "dataset_split" in manifest.columns
        else {}
    )
    providers = (
        sorted(manifest["provider"].dropna().astype(str).unique().tolist())
        if "provider" in manifest.columns
        else []
    )
    split_policy = None
    if "split_policy" in manifest.columns:
        split_policy = _single_value(manifest["split_policy"].astype(str))
    if split_policy is None:
        split_policy = "manifest_defined" if "dataset_split" in manifest.columns else "unknown"

    holdout_group = None
    if "holdout_group" in manifest.columns:
        holdout_group = _single_value(manifest["holdout_group"].astype(str))

    return {
        "contract_version": 2,
        "train_root": project_relative_path(train_root, project_root),
        "manifest_path": project_relative_path(manifest_path, project_root),
        "imagery_source": _infer_imagery_source(manifest, train_root),
        "dataset_cohort": _infer_dataset_cohort(manifest),
        "split_policy": str(split_policy),
        "holdout_group": str(holdout_group) if holdout_group else None,
        "dataset_splits": dataset_splits,
        "tile_count": int(len(manifest)),
        "providers": providers,
        "file_list_hash": compute_training_file_hash(manifest),
    }


def training_contract_run_fragment(contract: dict[str, object] | None) -> str:
    if not contract:
        return "data-unresolved"
    imagery_source = _slugify(contract.get("imagery_source") or "unknown")
    dataset_cohort = _slugify(contract.get("dataset_cohort") or "unknown")
    tile_count = int(contract.get("tile_count") or 0)
    hash_prefix = str(contract.get("file_list_hash") or "unknown")[:8]
    return f"data-{imagery_source}-{dataset_cohort}-n{tile_count}-{hash_prefix}"


def compare_training_contracts(
    current_contract: dict[str, object] | None,
    saved_contract: dict[str, object] | None,
) -> dict[str, dict[str, object]]:
    if not current_contract or not saved_contract:
        return {}

    mismatches: dict[str, dict[str, object]] = {}
    for key in ("train_root", "imagery_source", "dataset_cohort", "split_policy", "holdout_group", "file_list_hash"):
        current_value = current_contract.get(key)
        saved_value = saved_contract.get(key)
        if current_value is None or saved_value is None:
            continue
        if current_value != saved_value:
            mismatches[key] = {"current": current_value, "saved": saved_value}
    return mismatches