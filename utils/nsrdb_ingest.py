"""Shared NSRDB ingest helpers for the 06 notebook workflow.

This module intentionally holds the heavier planning, archive, CSV fallback,
and summary logic that grew out of the notebook. The notebook remains the
source-of-truth analysis surface, while these helpers keep the Jupytext script
readable and easier to iterate on.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from io import StringIO
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from shapely.geometry import Point

from .census import CANONICAL_COUNTY_TABLE, resolve_vector_db_path
from .nsrdb_api import (
    DEFAULT_ATTRIBUTES,
    DEFAULT_LEDGER_PATH,
    GOES_FULL_DISC_DATASET,
    GOES_FULL_DISC_DOWNLOAD_URL,
    MAX_REQUEST_WEIGHT,
    MIN_ARCHIVE_REQUEST_SPACING_SECONDS,
    NORMALIZED_ROOT,
    NSRDB_ROOT,
    RAW_ROOT,
    append_ledger_record,
    compute_request_weight,
    download_archive,
    extract_archive,
    max_sites_per_request,
    plan_download_batches,
    query_site_count,
    simplify_wkt_for_site_count,
    submit_planned_requests,
)


def resolve_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in ("project_rules.md", ".git")):
            return candidate
    return current


PROJECT_ROOT = resolve_project_root()
load_dotenv(PROJECT_ROOT / ".env")


def _parse_csv_env(env_name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = os.getenv(env_name, "")
    if not raw_value.strip():
        return default

    parsed = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    return parsed or default


def _slugify_scope(values: tuple[str, ...]) -> str:
    if not values:
        return "puerto_rico"

    slug_parts: list[str] = []
    for value in values:
        normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
        collapsed = "_".join(part for part in normalized.split("_") if part)
        if collapsed:
            slug_parts.append(collapsed)
    return "_".join(slug_parts) if slug_parts else "puerto_rico"


def _normalize_choice(raw_value: str, *, env_name: str, aliases: dict[str, str]) -> str:
    normalized = aliases.get((raw_value or "").strip().lower())
    if normalized is None:
        options = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"{env_name} must be one of: {options}")
    return normalized


def _resolve_scope_municipalities(
    scope_name: str,
    *,
    case_study_municipalities: tuple[str, ...],
    allow_none: bool = False,
) -> tuple[str, ...] | None:
    if scope_name == "all_pr":
        return ()
    if scope_name == "case_study":
        return case_study_municipalities
    if allow_none and scope_name == "none":
        return None
    raise ValueError(f"Unsupported scope name: {scope_name}")


def _resolve_scope_label(scope_name: str, *, case_study_municipalities: tuple[str, ...]) -> str:
    if scope_name == "all_pr":
        return "Puerto Rico"
    if scope_name == "case_study":
        return " + ".join(case_study_municipalities) if case_study_municipalities else "Case study"
    if scope_name == "none":
        return "No archive fetching"
    raise ValueError(f"Unsupported scope name: {scope_name}")


def _resolve_scope_stem(scope_name: str, *, case_study_municipalities: tuple[str, ...]) -> str:
    if scope_name == "all_pr":
        return "puerto_rico"
    if scope_name == "case_study":
        return _slugify_scope(case_study_municipalities)
    if scope_name == "none":
        return "fetch_disabled"
    raise ValueError(f"Unsupported scope name: {scope_name}")


def _legacy_archive_workflow_mode() -> str:
    if os.getenv("NSRDB_NORMALIZE_ARCHIVE_EXPORTS", "") == "1":
        return "normalize"
    if os.getenv("NSRDB_EXTRACT_ARCHIVES", "") == "1":
        return "extract"
    if os.getenv("NSRDB_DOWNLOAD_ARCHIVES", "") == "1":
        return "download"
    if os.getenv("NSRDB_SUBMIT_ARCHIVE_REQUESTS", os.getenv("NSRDB_SUBMIT_REQUESTS", "")) == "1":
        return "submit"
    return "status_only"


def _legacy_csv_fallback_mode() -> str:
    if os.getenv("NSRDB_FETCH_CSV_REQUESTS", "") == "1":
        return "execute"
    if os.getenv("NSRDB_REBUILD_SITE_DISCOVERY", "") == "1":
        return "plan_only"
    return "disabled"


SCOPE_ALIASES = {
    "all": "all_pr",
    "all_pr": "all_pr",
    "pr": "all_pr",
    "puerto_rico": "all_pr",
    "case_study": "case_study",
    "target": "case_study",
}
FETCH_SCOPE_ALIASES = {
    **SCOPE_ALIASES,
    "disabled": "none",
    "none": "none",
}
ARCHIVE_WORKFLOW_ALIASES = {
    "status_only": "status_only",
    "plan_only": "status_only",
    "submit": "submit",
    "download": "download",
    "extract": "extract",
    "normalize": "normalize",
}
CSV_FALLBACK_ALIASES = {
    "disabled": "disabled",
    "none": "disabled",
    "plan_only": "plan_only",
    "discover": "plan_only",
    "execute": "execute",
    "fetch": "execute",
}


NSRDB_API_KEY = os.getenv("NSRDB_API_KEY") or os.getenv("NREL_API_KEY") or os.getenv("NLR_API_KEY") or ""
NSRDB_EMAIL = os.getenv("NSRDB_EMAIL") or os.getenv("EMAIL") or ""
NSRDB_FULL_NAME = os.getenv("NSRDB_FULL_NAME") or None
NSRDB_AFFILIATION = os.getenv("NSRDB_AFFILIATION") or None
NSRDB_REASON = os.getenv("NSRDB_REASON", "Academic PV planning analysis") or None
NSRDB_MAILING_LIST = os.getenv("NSRDB_MAILING_LIST", "0") == "1"

CASE_STUDY_MUNICIPALITIES = _parse_csv_env(
    "NSRDB_CASE_STUDY_MUNICIPALITIES",
    _parse_csv_env("NSRDB_TARGET_MUNICIPALITIES", ("San Juan", "Isabela")),
)
PLAN_SCOPE = _normalize_choice(
    os.getenv("NSRDB_PLAN_SCOPE", "all_pr"),
    env_name="NSRDB_PLAN_SCOPE",
    aliases=SCOPE_ALIASES,
)
FETCH_SCOPE = _normalize_choice(
    os.getenv("NSRDB_FETCH_SCOPE", "case_study"),
    env_name="NSRDB_FETCH_SCOPE",
    aliases=FETCH_SCOPE_ALIASES,
)
ARCHIVE_WORKFLOW_MODE = _normalize_choice(
    os.getenv("NSRDB_ARCHIVE_WORKFLOW_MODE", _legacy_archive_workflow_mode()),
    env_name="NSRDB_ARCHIVE_WORKFLOW_MODE",
    aliases=ARCHIVE_WORKFLOW_ALIASES,
)
CSV_FALLBACK_MODE = _normalize_choice(
    os.getenv("NSRDB_CSV_FALLBACK_MODE", _legacy_csv_fallback_mode()),
    env_name="NSRDB_CSV_FALLBACK_MODE",
    aliases=CSV_FALLBACK_ALIASES,
)
RUN_BATCH_LIMIT = int(os.getenv("NSRDB_RUN_BATCH_LIMIT", os.getenv("NSRDB_MAX_REQUESTS_THIS_RUN", "20")) or "0") or None

PLAN_SCOPE_MUNICIPALITIES = _resolve_scope_municipalities(
    PLAN_SCOPE,
    case_study_municipalities=CASE_STUDY_MUNICIPALITIES,
)
FETCH_SCOPE_MUNICIPALITIES = _resolve_scope_municipalities(
    FETCH_SCOPE,
    case_study_municipalities=CASE_STUDY_MUNICIPALITIES,
    allow_none=True,
)
PLAN_SCOPE_LABEL = _resolve_scope_label(PLAN_SCOPE, case_study_municipalities=CASE_STUDY_MUNICIPALITIES)
FETCH_SCOPE_LABEL = _resolve_scope_label(FETCH_SCOPE, case_study_municipalities=CASE_STUDY_MUNICIPALITIES)
PLAN_SCOPE_STEM = _resolve_scope_stem(PLAN_SCOPE, case_study_municipalities=CASE_STUDY_MUNICIPALITIES)
FETCH_SCOPE_STEM = _resolve_scope_stem(FETCH_SCOPE, case_study_municipalities=CASE_STUDY_MUNICIPALITIES)

YEARS = tuple(range(2018, 2025))
ATTRIBUTES = DEFAULT_ATTRIBUTES
INTERVAL_MINUTES = int(os.getenv("NSRDB_INTERVAL_MINUTES", "30") or "30")
REQUEST_UTC = os.getenv("NSRDB_REQUEST_UTC", "1") == "1"
INCLUDE_LEAP_DAY = os.getenv("NSRDB_INCLUDE_LEAP_DAY", "1") == "1"
MAX_BATCH_SPLIT_DEPTH = int(os.getenv("NSRDB_MAX_BATCH_SPLIT_DEPTH", "8") or "8")
MIN_BATCH_AREA_KM2 = float(os.getenv("NSRDB_MIN_BATCH_AREA_KM2", "2") or "2")
MAX_REQUESTS_THIS_RUN = RUN_BATCH_LIMIT
SUBMIT_ARCHIVE_REQUESTS = ARCHIVE_WORKFLOW_MODE in {"submit", "download", "extract", "normalize"}
REQUEST_PAUSE_SECONDS = max(
    MIN_ARCHIVE_REQUEST_SPACING_SECONDS,
    float(os.getenv("NSRDB_REQUEST_PAUSE_SECONDS", str(MIN_ARCHIVE_REQUEST_SPACING_SECONDS)) or MIN_ARCHIVE_REQUEST_SPACING_SECONDS),
)
ARCHIVE_DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("NSRDB_ARCHIVE_TIMEOUT_SECONDS", "600") or "600")
ARCHIVE_DOWNLOAD_MAX_ATTEMPTS = int(os.getenv("NSRDB_ARCHIVE_DOWNLOAD_MAX_ATTEMPTS", "15") or "15")
ARCHIVE_DOWNLOAD_POLL_SECONDS = max(5.0, float(os.getenv("NSRDB_ARCHIVE_DOWNLOAD_POLL_SECONDS", "15") or "15"))
DOWNLOAD_ARCHIVES = ARCHIVE_WORKFLOW_MODE in {"download", "extract", "normalize"}
EXTRACT_ARCHIVES = ARCHIVE_WORKFLOW_MODE in {"extract", "normalize"}
NORMALIZE_ARCHIVE_EXPORTS = ARCHIVE_WORKFLOW_MODE == "normalize"
OVERWRITE_EXISTING_ARCHIVES = os.getenv("NSRDB_OVERWRITE_EXISTING_ARCHIVES", "0") == "1"
OVERWRITE_EXTRACTED_ARCHIVES = os.getenv("NSRDB_OVERWRITE_EXTRACTED_ARCHIVES", "0") == "1"

REQUEST_PLAN_PATH = NSRDB_ROOT / f"{PLAN_SCOPE_STEM}_request_plan.parquet"
REQUEST_PLAN_GEOJSON = NSRDB_ROOT / f"{PLAN_SCOPE_STEM}_request_plan.geojson"
ARCHIVE_DOWNLOAD_ROOT = RAW_ROOT / "archive_batches" / FETCH_SCOPE_STEM
ARCHIVE_EXTRACT_ROOT = RAW_ROOT / "archive_batches_extracted" / FETCH_SCOPE_STEM
ARCHIVE_MANIFEST_PATH = NSRDB_ROOT / f"{FETCH_SCOPE_STEM}_nsrdb_archive_manifest.parquet"
CSV_DIRECT_DOWNLOAD_URL = GOES_FULL_DISC_DOWNLOAD_URL.replace(".json", ".csv")
DISCOVERY_GRID_CRS = "EPSG:32619"
DISCOVERY_GRID_SPACING_METERS = float(os.getenv("NSRDB_DISCOVERY_GRID_SPACING_METERS", "2000") or "2000")
DISCOVERY_GRID_OFFSETS = (
    (0.0, 0.0),
    (0.5, 0.5),
    (0.5, 0.0),
    (0.0, 0.5),
)
DISCOVERY_MAX_PASSES = max(1, min(len(DISCOVERY_GRID_OFFSETS), int(os.getenv("NSRDB_DISCOVERY_MAX_PASSES", "1") or "1")))
DISCOVERY_YEAR = int(os.getenv("NSRDB_DISCOVERY_YEAR", str(max(YEARS))) or str(max(YEARS)))
DISCOVERY_ATTRIBUTES = tuple(
    value.strip()
    for value in os.getenv("NSRDB_DISCOVERY_ATTRIBUTES", "ghi").split(",")
    if value.strip()
) or ("ghi",)
DISCOVERY_INTERVAL_MINUTES = int(os.getenv("NSRDB_DISCOVERY_INTERVAL_MINUTES", "60") or "60")
DISCOVERY_TIMEOUT_SECONDS = int(os.getenv("NSRDB_DISCOVERY_TIMEOUT_SECONDS", "240") or "240")
CSV_TIMEOUT_SECONDS = int(os.getenv("NSRDB_CSV_TIMEOUT_SECONDS", "600") or "600")
CSV_REQUEST_PAUSE_SECONDS = max(1.05, float(os.getenv("NSRDB_CSV_REQUEST_PAUSE_SECONDS", "1.05") or "1.05"))
MAX_SITE_PROBES_THIS_RUN = int(os.getenv("NSRDB_MAX_SITE_PROBES_THIS_RUN", "0") or "0") or None
_csv_request_limit_raw = os.getenv(
    "NSRDB_MAX_CSV_DOWNLOADS_THIS_RUN",
    os.getenv("NSRDB_RUN_BATCH_LIMIT", os.getenv("NSRDB_MAX_REQUESTS_THIS_RUN", "25")),
)
MAX_CSV_DOWNLOADS_THIS_RUN = int(_csv_request_limit_raw or "25")
if MAX_CSV_DOWNLOADS_THIS_RUN <= 0:
    MAX_CSV_DOWNLOADS_THIS_RUN = None
FETCH_CSV_REQUESTS = CSV_FALLBACK_MODE == "execute"
REBUILD_SITE_DISCOVERY = CSV_FALLBACK_MODE in {"plan_only", "execute"}
OVERWRITE_EXISTING_CSV = os.getenv("NSRDB_OVERWRITE_EXISTING_CSV", "0") == "1"
CSV_DAILY_REQUEST_LIMIT = 10_000
DISCOVERED_SITES_PATH = NSRDB_ROOT / f"{FETCH_SCOPE_STEM}_nsrdb_sites.parquet"
SITE_DISCOVERY_LOG_PATH = NSRDB_ROOT / f"{FETCH_SCOPE_STEM}_nsrdb_site_discovery_log.parquet"
CSV_REQUEST_PLAN_PATH = NSRDB_ROOT / f"{FETCH_SCOPE_STEM}_nsrdb_csv_request_plan.parquet"
NSRDB_FLUX_SUMMARY_COLUMNS = (
    "ghi",
    "dni",
    "dhi",
    "air_temperature",
    "clearsky_ghi",
    "clearsky_dni",
    "clearsky_dhi",
    "surface_albedo",
)
MONTHLY_FLUX_MEANS_PATH = NORMALIZED_ROOT / f"{FETCH_SCOPE_STEM}_nsrdb_monthly_flux_means.parquet"
ANNUAL_FLUX_MEANS_PATH = NORMALIZED_ROOT / f"{FETCH_SCOPE_STEM}_nsrdb_annual_flux_means.parquet"
MULTIYEAR_FLUX_MEANS_PATH = NORMALIZED_ROOT / f"{FETCH_SCOPE_STEM}_nsrdb_multiyear_flux_means.parquet"


def resolve_db_path() -> Path:
    return resolve_vector_db_path(PROJECT_ROOT)


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")
    return con


def _to_bytes(value: object) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    return bytes(value)


def load_target_municipalities(
    con: duckdb.DuckDBPyConnection,
    target_municipalities: tuple[str, ...] = CASE_STUDY_MUNICIPALITIES,
) -> gpd.GeoDataFrame:
    table_exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?;",
        [CANONICAL_COUNTY_TABLE],
    ).fetchone()[0]
    if not table_exists:
        raise RuntimeError(
            f"{CANONICAL_COUNTY_TABLE} not found; run notebooks/vectors/01_census_geometries_ingest.py first."
        )

    if target_municipalities:
        names_sql = ", ".join("?" * len(target_municipalities))
        frame = con.execute(
            f"""
            SELECT
                CAST(NAME AS VARCHAR) AS municipio,
                CAST(GEOID AS VARCHAR) AS municipio_geoid,
                ST_AsWKB(geometry) AS geometry_wkb
            FROM {CANONICAL_COUNTY_TABLE}
            WHERE NAME IN ({names_sql})
            ORDER BY NAME;
            """,
            list(target_municipalities),
        ).fetchdf()
    else:
        frame = con.execute(
            f"""
            SELECT
                CAST(NAME AS VARCHAR) AS municipio,
                CAST(GEOID AS VARCHAR) AS municipio_geoid,
                ST_AsWKB(geometry) AS geometry_wkb
            FROM {CANONICAL_COUNTY_TABLE}
            ORDER BY NAME;
            """
        ).fetchdf()
    if frame.empty:
        return gpd.GeoDataFrame(columns=["municipio", "municipio_geoid", "geometry"], geometry="geometry", crs="EPSG:4326")

    geometry = gpd.GeoSeries.from_wkb(frame["geometry_wkb"].map(_to_bytes), crs="EPSG:4326")
    return gpd.GeoDataFrame(frame.drop(columns=["geometry_wkb"]), geometry=geometry, crs="EPSG:4326")


def load_puerto_rico_municipalities(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    return load_target_municipalities(con, ())


def write_request_plan(plan: gpd.GeoDataFrame) -> None:
    REQUEST_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.to_parquet(REQUEST_PLAN_PATH, index=False)
    plan.to_file(REQUEST_PLAN_GEOJSON, driver="GeoJSON")


def plot_request_plan(municipalities: gpd.GeoDataFrame, plan: gpd.GeoDataFrame) -> None:
    if municipalities.empty or plan.empty:
        print("request-plan plot skipped: municipality geometry or request batches are empty.")
        return

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    municipalities.plot(ax=ax, color="#f3f3f3", edgecolor="#444444", linewidth=0.6)
    plan.plot(
        ax=ax,
        column="site_count",
        cmap="YlOrRd",
        alpha=0.35,
        edgecolor="#b30000",
        linewidth=1.0,
        legend=True,
        legend_kwds={
            "label": "NSRDB sites per request polygon",
            "orientation": "horizontal",
            "pad": 0.02,
            "shrink": 0.7,
        },
    )
    municipalities.boundary.plot(ax=ax, color="#111111", linewidth=0.8)
    if len(plan) <= 40:
        label_points = plan.geometry.representative_point()
        for row, label_point in zip(plan.itertuples(index=False), label_points, strict=False):
            ax.text(
                label_point.x,
                label_point.y,
                batch_id_numeric_label(row.batch_id),
                fontsize=7,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            )
    ax.set_title(f"{PLAN_SCOPE_LABEL} NSRDB request polygons")
    ax.set_axis_off()
    fig.tight_layout()


def empty_municipality_frame() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns=["municipio", "municipio_geoid", "geometry"], geometry="geometry", crs="EPSG:4326")


def namespace_plan_batch_ids(plan: gpd.GeoDataFrame, *, scope_stem: str) -> gpd.GeoDataFrame:
    if plan.empty:
        return plan

    legacy_case_study_stem = _slugify_scope(CASE_STUDY_MUNICIPALITIES)
    if scope_stem == legacy_case_study_stem:
        return plan

    prefix = f"{scope_stem}_"
    frame = plan.copy()
    frame["batch_id"] = frame["batch_id"].astype(str).map(
        lambda value: value if value.startswith(prefix) else f"{prefix}{value}"
    )
    return frame


def read_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def batch_id_numeric_label(batch_id: object) -> str:
    match = re.search(r"(\d+)$", str(batch_id))
    if match is not None:
        return match.group(1)
    return str(batch_id)


def filter_archive_queue_to_batch_ids(queue: pd.DataFrame, batch_ids: set[str]) -> pd.DataFrame:
    if queue.empty:
        return queue
    if not batch_ids:
        return queue.iloc[0:0].copy()
    return queue.loc[queue["batch_id"].astype(str).isin(batch_ids)].copy().reset_index(drop=True)


def utc_now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def print_request_weight_guidance() -> None:
    print("request_weight_pct = 100 * request_weight / NSRDB's max allowed weight for one polygon request.")
    print("Use it only as an API-limit meter: under 50 is comfortable, 50-80 is moderate, 80-100 is close to the cap, and values above 100 must be split further.")


def load_ledger_entries(ledger_path: Path = DEFAULT_LEDGER_PATH) -> pd.DataFrame:
    return read_parquet_if_exists(ledger_path)


def load_archive_manifest(manifest_path: Path = ARCHIVE_MANIFEST_PATH) -> pd.DataFrame:
    return read_parquet_if_exists(manifest_path)


def summarize_batch_municipality_overlap(
    plan: gpd.GeoDataFrame,
    municipalities: gpd.GeoDataFrame,
    *,
    names_column: str,
    count_column: str,
) -> pd.DataFrame:
    if plan.empty or municipalities.empty:
        return pd.DataFrame(columns=["batch_id", names_column, count_column])

    municipality_rows = list(municipalities[["municipio", "geometry"]].itertuples(index=False, name=None))
    rows: list[dict[str, object]] = []
    for batch_row in plan[["batch_id", "geometry"]].itertuples(index=False):
        matches = sorted(
            {
                str(municipio)
                for municipio, geometry in municipality_rows
                if geometry is not None and not geometry.is_empty and batch_row.geometry.intersects(geometry)
            }
        )
        rows.append(
            {
                "batch_id": str(batch_row.batch_id),
                names_column: ", ".join(matches),
                count_column: len(matches),
            }
        )

    return pd.DataFrame(rows)


ARCHIVE_STATUS_ORDER = [
    "normalized",
    "extracted",
    "downloaded",
    "submitted",
    "planned",
    "dry_run",
    "outside_fetch_scope",
    "fetch_disabled",
    "error",
]
ARCHIVE_STATUS_COLORS = {
    "normalized": "#0b6e4f",
    "extracted": "#3b7a57",
    "downloaded": "#4e79a7",
    "submitted": "#9c755f",
    "planned": "#f2cf5b",
    "dry_run": "#d8b365",
    "outside_fetch_scope": "#c7c7c7",
    "fetch_disabled": "#e0e0e0",
    "error": "#b22222",
}


def derive_archive_batch_status(
    *,
    in_fetch_scope: bool,
    fetch_scope_enabled: bool,
    batch_ledger: pd.DataFrame,
    manifest_summary: pd.Series | None,
) -> tuple[str, str, int]:
    normalized_files = int(manifest_summary.get("normalized_files", 0)) if manifest_summary is not None else 0
    if normalized_files > 0:
        return "normalized", f"{normalized_files:,} normalized parquet file(s).", normalized_files

    note_values = [
        str(value).strip()
        for value in batch_ledger.get("notes", pd.Series(dtype=object)).fillna("").astype(str)
        if str(value).strip()
    ]
    unique_notes = list(dict.fromkeys(note_values))
    note_text = " | ".join(unique_notes[-2:]) if unique_notes else ""
    statuses = {str(value) for value in batch_ledger.get("status", pd.Series(dtype=object)).dropna().astype(str)}
    waiting_for_archive = any(
        "not ready yet" in note.lower() or "generation in progress" in note.lower()
        for note in unique_notes
    )

    if "extracted" in statuses:
        return "extracted", note_text or "Archive zip extracted locally.", normalized_files
    if "downloaded" in statuses or (
        "reused" in statuses
        and batch_ledger.get("local_archive_path", pd.Series(dtype=object)).fillna("").astype(str).str.endswith(".zip").any()
    ):
        return "downloaded", note_text or "Archive zip downloaded locally.", normalized_files
    if "submitted" in statuses or waiting_for_archive:
        detail = "Archive requested; NSRDB is still preparing the zip." if waiting_for_archive else (note_text or "Archive request submitted.")
        return "submitted", detail, normalized_files
    if "dry_run" in statuses:
        return "dry_run", "Batch was only planned; no request was executed.", normalized_files
    if "error" in statuses:
        return "error", note_text or "Archive workflow hit an error.", normalized_files
    if not fetch_scope_enabled:
        return "fetch_disabled", "Archive execution is disabled for this run.", normalized_files
    if not in_fetch_scope:
        return "outside_fetch_scope", f"Planned for {PLAN_SCOPE_LABEL}; current fetch scope is {FETCH_SCOPE_LABEL}.", normalized_files
    return "planned", f"Ready to submit within the current fetch scope ({FETCH_SCOPE_LABEL}).", normalized_files


def build_request_batch_status(
    plan: gpd.GeoDataFrame,
    municipalities: gpd.GeoDataFrame,
    fetch_municipalities: gpd.GeoDataFrame,
    *,
    fetch_scope_enabled: bool,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    manifest_path: Path = ARCHIVE_MANIFEST_PATH,
) -> gpd.GeoDataFrame:
    if plan.empty:
        return plan.copy()

    status_frame = plan.copy()
    status_frame["batch_id"] = status_frame["batch_id"].astype(str)

    overlap = summarize_batch_municipality_overlap(
        status_frame,
        municipalities,
        names_column="covered_municipalities",
        count_column="municipality_count",
    )
    status_frame = status_frame.merge(overlap, on="batch_id", how="left")

    if fetch_scope_enabled and not fetch_municipalities.empty:
        fetch_overlap = summarize_batch_municipality_overlap(
            status_frame,
            fetch_municipalities,
            names_column="fetch_scope_municipalities",
            count_column="fetch_scope_municipality_count",
        )
        status_frame = status_frame.merge(fetch_overlap, on="batch_id", how="left")
        status_frame["in_fetch_scope"] = status_frame["fetch_scope_municipality_count"].fillna(0).astype(int) > 0
    else:
        status_frame["fetch_scope_municipalities"] = ""
        status_frame["fetch_scope_municipality_count"] = 0
        status_frame["in_fetch_scope"] = False if not fetch_scope_enabled else True

    for column_name in (
        "covered_municipalities",
        "fetch_scope_municipalities",
        "municipality_count",
        "fetch_scope_municipality_count",
    ):
        if column_name not in status_frame.columns:
            status_frame[column_name] = "" if column_name.endswith("municipalities") else 0

    ledger = load_ledger_entries(ledger_path)
    if not ledger.empty:
        ledger = ledger.loc[
            (ledger["request_format"].astype(str) == "json")
            & ledger["batch_id"].astype(str).isin(status_frame["batch_id"])
        ].copy()

    manifest = load_archive_manifest(manifest_path)
    manifest_summary = pd.DataFrame(columns=["batch_id", "normalized_files"]).set_index("batch_id")
    if not manifest.empty and "batch_id" in manifest.columns:
        manifest_summary = (
            manifest.assign(batch_id=manifest["batch_id"].astype(str))
            .groupby("batch_id", dropna=False)
            .agg(
                normalized_files=(
                    "status",
                    lambda values: int(pd.Series(values).astype(str).isin(["normalized", "reused"]).sum()),
                )
            )
        )

    display_statuses: list[str] = []
    status_details: list[str] = []
    normalized_counts: list[int] = []
    for batch_row in status_frame[["batch_id", "in_fetch_scope"]].itertuples(index=False):
        batch_ledger = ledger.loc[ledger["batch_id"].astype(str) == batch_row.batch_id].copy() if not ledger.empty else pd.DataFrame()
        batch_manifest = manifest_summary.loc[batch_row.batch_id] if batch_row.batch_id in manifest_summary.index else None
        display_status, status_detail, normalized_count = derive_archive_batch_status(
            in_fetch_scope=bool(batch_row.in_fetch_scope),
            fetch_scope_enabled=fetch_scope_enabled,
            batch_ledger=batch_ledger,
            manifest_summary=batch_manifest,
        )
        display_statuses.append(display_status)
        status_details.append(status_detail)
        normalized_counts.append(normalized_count)

    status_frame["display_status"] = pd.Categorical(display_statuses, categories=ARCHIVE_STATUS_ORDER, ordered=True)
    status_frame["status_detail"] = status_details
    status_frame["normalized_file_count"] = normalized_counts
    return status_frame.sort_values(["display_status", "split_depth", "batch_id"]).reset_index(drop=True)


def build_batch_status_summary(status_frame: gpd.GeoDataFrame) -> pd.DataFrame:
    if status_frame.empty:
        return pd.DataFrame(columns=["display_status", "batch_count", "site_count", "area_km2"])

    summary = (
        status_frame.groupby("display_status", observed=True)
        .agg(
            batch_count=("batch_id", "size"),
            site_count=("site_count", "sum"),
            area_km2=("area_km2", "sum"),
        )
        .reset_index()
    )
    summary["display_status"] = summary["display_status"].astype(str)
    summary["area_km2"] = summary["area_km2"].round(2)
    return summary.loc[summary["batch_count"] > 0].reset_index(drop=True)


def build_ledger_activity_summary(ledger_path: Path = DEFAULT_LEDGER_PATH) -> pd.DataFrame:
    ledger = load_ledger_entries(ledger_path)
    if ledger.empty:
        return pd.DataFrame(columns=["request_format", "status", "request_count"])

    summary = (
        ledger.assign(
            request_format=ledger["request_format"].astype(str),
            status=ledger["status"].astype(str),
        )
        .groupby(["request_format", "status"], dropna=False)
        .size()
        .reset_index(name="request_count")
        .sort_values(["request_format", "request_count", "status"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return summary


def plot_request_status_map(
    municipalities: gpd.GeoDataFrame,
    status_frame: gpd.GeoDataFrame,
    *,
    fetch_municipalities: gpd.GeoDataFrame | None = None,
) -> None:
    if municipalities.empty or status_frame.empty:
        print("request-status plot skipped: municipality geometry or batch status table is empty.")
        return

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    municipalities.plot(ax=ax, color="#fafafa", edgecolor="#9a9a9a", linewidth=0.5)
    for status_name in ARCHIVE_STATUS_ORDER:
        subset = status_frame.loc[status_frame["display_status"].astype(str) == status_name]
        if subset.empty:
            continue
        subset.plot(
            ax=ax,
            color=ARCHIVE_STATUS_COLORS[status_name],
            alpha=0.55,
            edgecolor="#2f2f2f",
            linewidth=0.9,
        )
    municipalities.boundary.plot(ax=ax, color="#222222", linewidth=0.7)
    if fetch_municipalities is not None and not fetch_municipalities.empty:
        fetch_municipalities.boundary.plot(ax=ax, color="#000000", linewidth=1.6)

    if len(status_frame) <= 40:
        label_points = status_frame.geometry.representative_point()
        for row, label_point in zip(status_frame.itertuples(index=False), label_points, strict=False):
            ax.text(
                label_point.x,
                label_point.y,
                batch_id_numeric_label(row.batch_id),
                fontsize=7,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
            )

    legend_handles = [
        Patch(facecolor=ARCHIVE_STATUS_COLORS[status_name], edgecolor="#2f2f2f", label=status_name.replace("_", " "))
        for status_name in ARCHIVE_STATUS_ORDER
        if not status_frame.loc[status_frame["display_status"].astype(str) == status_name].empty
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=min(4, len(legend_handles)), frameon=True, title="Batch status")
    ax.set_title(f"{PLAN_SCOPE_LABEL} request polygons with current archive status")
    ax.set_axis_off()
    fig.tight_layout()


def build_site_flux_geodataframe(multiyear_flux_means: pd.DataFrame) -> gpd.GeoDataFrame:
    if multiyear_flux_means.empty or not {"latitude", "longitude"}.issubset(multiyear_flux_means.columns):
        return gpd.GeoDataFrame(columns=list(multiyear_flux_means.columns) + ["geometry"], geometry="geometry", crs="EPSG:4326")

    frame = multiyear_flux_means.loc[
        multiyear_flux_means["latitude"].notna() & multiyear_flux_means["longitude"].notna()
    ].copy()
    if frame.empty:
        return gpd.GeoDataFrame(columns=list(multiyear_flux_means.columns) + ["geometry"], geometry="geometry", crs="EPSG:4326")

    geometry = gpd.points_from_xy(frame["longitude"], frame["latitude"])
    return gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:4326")


def filter_site_points_to_municipalities(
    site_points: gpd.GeoDataFrame,
    municipalities: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if site_points.empty or municipalities.empty:
        return site_points

    municipality_geometries = list(municipalities.geometry)
    mask = site_points.geometry.apply(
        lambda point: any(geometry is not None and not geometry.is_empty and geometry.intersects(point) for geometry in municipality_geometries)
    )
    return site_points.loc[mask].copy()


def plot_flux_attribute_preview(
    municipalities: gpd.GeoDataFrame,
    multiyear_flux_means: pd.DataFrame,
    *,
    focus_label: str,
    attribute_columns: tuple[str, ...] = ("ghi", "dni", "dhi", "air_temperature"),
) -> None:
    if municipalities.empty:
        print("flux attribute plot skipped: municipality geometry is empty.")
        return

    site_points = filter_site_points_to_municipalities(build_site_flux_geodataframe(multiyear_flux_means), municipalities)
    available_columns = [column_name for column_name in attribute_columns if column_name in site_points.columns]
    if site_points.empty or not available_columns:
        print("flux attribute plot skipped: no mapped NSRDB site means are available for the selected municipalities.")
        return

    import matplotlib.pyplot as plt

    ncols = 2
    nrows = int(np.ceil(len(available_columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows))
    axes_array = np.atleast_1d(axes).ravel()

    for axis, column_name in zip(axes_array, available_columns, strict=False):
        municipalities.plot(ax=axis, color="#fbfbfb", edgecolor="#777777", linewidth=0.6)
        site_points.plot(
            ax=axis,
            column=column_name,
            cmap="YlOrRd",
            legend=True,
            markersize=28,
            legend_kwds={"shrink": 0.7},
        )
        municipalities.boundary.plot(ax=axis, color="#111111", linewidth=0.8)
        axis.set_title(column_name.replace("_", " ").title())
        axis.set_axis_off()

    for axis in axes_array[len(available_columns) :]:
        axis.set_visible(False)

    fig.suptitle(f"{focus_label} multiyear NSRDB attribute preview", fontsize=14)
    fig.tight_layout()


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def point_wkt(longitude: float, latitude: float) -> str:
    return f"POINT({float(longitude):.6f} {float(latitude):.6f})"


def build_candidate_discovery_points(
    geometry,
    *,
    spacing_meters: float = DISCOVERY_GRID_SPACING_METERS,
    max_passes: int = DISCOVERY_MAX_PASSES,
) -> gpd.GeoDataFrame:
    if geometry is None or geometry.is_empty:
        return gpd.GeoDataFrame(columns=["candidate_id", "pass_index", "grid_x", "grid_y", "geometry"], geometry="geometry", crs="EPSG:4326")

    offsets = DISCOVERY_GRID_OFFSETS[: max(1, min(max_passes, len(DISCOVERY_GRID_OFFSETS)))]
    geometry_projected = gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs(DISCOVERY_GRID_CRS).iloc[0]
    minx, miny, maxx, maxy = geometry_projected.bounds
    rows: list[dict[str, object]] = []
    candidate_id = 0

    for pass_index, (offset_x, offset_y) in enumerate(offsets, start=1):
        xs = np.arange(minx + (offset_x * spacing_meters), maxx + spacing_meters, spacing_meters)
        ys = np.arange(miny + (offset_y * spacing_meters), maxy + spacing_meters, spacing_meters)
        for grid_x, x_coord in enumerate(xs):
            for grid_y, y_coord in enumerate(ys):
                point_projected = Point(float(x_coord), float(y_coord))
                if not geometry_projected.covers(point_projected):
                    continue
                candidate_id += 1
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "pass_index": pass_index,
                        "grid_x": grid_x,
                        "grid_y": grid_y,
                        "geometry": point_projected,
                    }
                )

    if not rows:
        return gpd.GeoDataFrame(columns=["candidate_id", "pass_index", "grid_x", "grid_y", "geometry"], geometry="geometry", crs="EPSG:4326")

    points = gpd.GeoDataFrame(rows, geometry="geometry", crs=DISCOVERY_GRID_CRS).to_crs("EPSG:4326")
    points["longitude"] = points.geometry.x.round(6)
    points["latitude"] = points.geometry.y.round(6)
    return points.sort_values(["pass_index", "candidate_id"]).reset_index(drop=True)


def parse_csv_header_lines(
    lines: list[str],
    *,
    requested_attributes: tuple[str, ...],
) -> dict[str, object]:
    if len(lines) < 3:
        raise RuntimeError("NSRDB CSV header probe returned fewer than three lines.")

    metadata_header = next(csv.reader([lines[0]]))
    metadata_values = next(csv.reader([lines[1]]))
    data_header = next(csv.reader([lines[2]]))
    metadata = {header: value for header, value in zip(metadata_header, metadata_values)}

    canonical_header = ["year", "month", "day", "hour", "minute"]
    requested_tail = list(requested_attributes[: max(0, len(data_header) - 5)])
    canonical_header.extend(requested_tail)
    if len(canonical_header) < len(data_header):
        canonical_header.extend(
            column.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
            for column in data_header[len(canonical_header) :]
        )

    def first_metadata_value(*keys: str) -> object:
        for key in keys:
            value = metadata.get(key)
            if value not in (None, ""):
                return value
        return None

    return {
        "source": first_metadata_value("Source", "Dataset", "dataset"),
        "site_id": _safe_int(first_metadata_value("Location ID", "SiteID", "site_id", "siteid")),
        "site_latitude": _safe_float(first_metadata_value("Latitude", "latitude")),
        "site_longitude": _safe_float(first_metadata_value("Longitude", "longitude")),
        "site_timezone": _safe_float(first_metadata_value("Time Zone", "Site Timezone", "site_timezone")),
        "site_local_timezone": _safe_float(first_metadata_value("Local Time Zone", "Data Timezone", "local_time_zone")),
        "site_elevation": _safe_float(first_metadata_value("Elevation", "Site Elevation", "elevation")),
        "country": first_metadata_value("Country", "country"),
        "version": first_metadata_value("Version", "version"),
        "raw_metadata_json": json.dumps(metadata, ensure_ascii=True),
        "raw_data_header_json": json.dumps(data_header, ensure_ascii=True),
        "canonical_data_header_json": json.dumps(canonical_header, ensure_ascii=True),
    }


def probe_csv_site_metadata(
    *,
    longitude: float,
    latitude: float,
    api_key: str,
    email: str,
    year: int,
    attributes: tuple[str, ...],
    interval_minutes: int,
    timeout: int = DISCOVERY_TIMEOUT_SECONDS,
    utc: bool = REQUEST_UTC,
    leap_day: bool = INCLUDE_LEAP_DAY,
    session: requests.Session | None = None,
) -> dict[str, object]:
    client = session or requests.Session()
    params = {
        "api_key": api_key,
        "email": email,
        "wkt": point_wkt(longitude, latitude),
        "attributes": ",".join(attributes),
        "names": str(year),
        "interval": str(interval_minutes),
        "utc": str(bool(utc)).lower(),
        "leap_day": str(bool(leap_day)).lower(),
    }

    response = client.get(CSV_DIRECT_DOWNLOAD_URL, params=params, stream=True, timeout=timeout)
    try:
        if not response.ok:
            raise RuntimeError(f"NSRDB CSV request failed with HTTP {response.status_code}: {response.text[:500]}")
        header_lines: list[str] = []
        for line in response.iter_lines(decode_unicode=True):
            if line is None:
                continue
            header_lines.append(str(line))
            if len(header_lines) >= 3:
                break
    finally:
        response.close()

    metadata = parse_csv_header_lines(header_lines, requested_attributes=attributes)
    metadata["request_wkt"] = params["wkt"]
    return metadata


def discover_sites_from_points(
    candidate_points: gpd.GeoDataFrame,
    *,
    target_geometry,
    api_key: str,
    email: str,
    year: int,
    attributes: tuple[str, ...],
    interval_minutes: int,
    expected_site_count: int | None = None,
    max_requests: int | None = None,
    pause_seconds: float = CSV_REQUEST_PAUSE_SECONDS,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    if candidate_points.empty:
        empty_sites = gpd.GeoDataFrame(columns=["site_id", "latitude", "longitude", "geometry"], geometry="geometry", crs="EPSG:4326")
        return empty_sites, pd.DataFrame()

    selected = candidate_points.head(max_requests).copy() if max_requests is not None else candidate_points.copy()
    session = requests.Session()
    discovered_sites: dict[int, dict[str, object]] = {}
    probe_rows: list[dict[str, object]] = []

    for probe_index, point_row in enumerate(selected.itertuples(index=False), start=1):
        if expected_site_count is not None and len(discovered_sites) >= expected_site_count:
            break

        probe_record = {
            "probe_index": probe_index,
            "candidate_id": point_row.candidate_id,
            "pass_index": point_row.pass_index,
            "grid_x": point_row.grid_x,
            "grid_y": point_row.grid_y,
            "probe_longitude": point_row.longitude,
            "probe_latitude": point_row.latitude,
            "site_id": None,
            "site_longitude": None,
            "site_latitude": None,
            "status": None,
            "error": None,
        }

        try:
            metadata = probe_csv_site_metadata(
                longitude=point_row.longitude,
                latitude=point_row.latitude,
                api_key=api_key,
                email=email,
                year=year,
                attributes=attributes,
                interval_minutes=interval_minutes,
                session=session,
            )
            probe_record.update(metadata)
            site_id = metadata.get("site_id")
            site_point = None
            if metadata.get("site_longitude") is not None and metadata.get("site_latitude") is not None:
                site_point = Point(float(metadata["site_longitude"]), float(metadata["site_latitude"]))

            if site_id is None or site_point is None:
                probe_record["status"] = "invalid_metadata"
            elif not target_geometry.covers(site_point):
                probe_record["status"] = "outside_target"
            elif site_id in discovered_sites:
                probe_record["status"] = "duplicate"
            else:
                probe_record["status"] = "discovered"
                discovered_sites[int(site_id)] = {
                    "site_id": int(site_id),
                    "latitude": float(metadata["site_latitude"]),
                    "longitude": float(metadata["site_longitude"]),
                    "timezone": metadata.get("site_timezone"),
                    "local_timezone": metadata.get("site_local_timezone"),
                    "elevation": metadata.get("site_elevation"),
                    "country": metadata.get("country"),
                    "source": metadata.get("source"),
                    "version": metadata.get("version"),
                    "first_probe_candidate_id": point_row.candidate_id,
                    "geometry": site_point,
                }
        except Exception as exc:
            probe_record["status"] = "error"
            probe_record["error"] = str(exc)

        probe_rows.append(probe_record)
        time.sleep(max(pause_seconds, 1.05))

    if discovered_sites:
        sites = gpd.GeoDataFrame(list(discovered_sites.values()), geometry="geometry", crs="EPSG:4326")
        sites = sites.sort_values("site_id").reset_index(drop=True)
    else:
        sites = gpd.GeoDataFrame(columns=["site_id", "latitude", "longitude", "geometry"], geometry="geometry", crs="EPSG:4326")

    return sites, pd.DataFrame(probe_rows)


def build_csv_request_plan(sites: gpd.GeoDataFrame, *, years: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    request_weight = compute_request_weight(1, len(ATTRIBUTES), 1, INTERVAL_MINUTES)
    for site_row in sites.itertuples(index=False):
        for year in years:
            raw_csv_path = RAW_ROOT / f"site_{site_row.site_id}" / f"nsrdb_site_{site_row.site_id}_{year}_{INTERVAL_MINUTES}min.csv"
            normalized_path = NORMALIZED_ROOT / f"nsrdb_site_{site_row.site_id}_{year}_{INTERVAL_MINUTES}min.parquet"
            rows.append(
                {
                    "batch_id": f"site_{site_row.site_id}_{year}",
                    "site_id": int(site_row.site_id),
                    "year": int(year),
                    "latitude": float(site_row.latitude),
                    "longitude": float(site_row.longitude),
                    "request_wkt": point_wkt(site_row.longitude, site_row.latitude),
                    "interval_minutes": int(INTERVAL_MINUTES),
                    "attributes": ",".join(ATTRIBUTES),
                    "site_count": 1,
                    "request_weight": request_weight,
                    "raw_csv_path": str(raw_csv_path),
                    "normalized_parquet_path": str(normalized_path),
                    "raw_csv_exists": raw_csv_path.exists(),
                    "normalized_parquet_exists": normalized_path.exists(),
                }
            )

    request_plan = pd.DataFrame(rows)
    if request_plan.empty:
        return request_plan

    request_plan["status"] = np.where(
        request_plan["raw_csv_exists"] & request_plan["normalized_parquet_exists"],
        "existing",
        "pending",
    )
    return request_plan.sort_values(["site_id", "year"]).reset_index(drop=True)


def parse_nsrdb_csv_text(csv_text: str) -> tuple[dict[str, str], list[str], pd.DataFrame]:
    lines = csv_text.splitlines()
    if len(lines) < 3:
        raise RuntimeError("NSRDB CSV response did not include the expected metadata and data header rows.")

    metadata_header = next(csv.reader([lines[0]]))
    metadata_values = next(csv.reader([lines[1]]))
    metadata = {header: value for header, value in zip(metadata_header, metadata_values)}
    data_header = next(csv.reader([lines[2]]))
    data_frame = pd.read_csv(StringIO("\n".join(lines[2:])))
    return metadata, data_header, data_frame


def normalize_nsrdb_csv_text(
    csv_text: str,
    *,
    request_row: pd.Series,
    utc: bool,
) -> pd.DataFrame:
    metadata, data_header, data_frame = parse_nsrdb_csv_text(csv_text)
    requested_attributes = tuple(
        value.strip()
        for value in str(request_row.get("attributes", "")).split(",")
        if value.strip()
    )
    canonical_header = ["year", "month", "day", "hour", "minute"] + list(requested_attributes[: max(0, len(data_header) - 5)])
    if len(canonical_header) < len(data_header):
        canonical_header.extend(
            column.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
            for column in data_header[len(canonical_header) :]
        )
    data_frame.columns = canonical_header[: len(data_frame.columns)]

    for column_name in data_frame.columns:
        data_frame[column_name] = pd.to_numeric(data_frame[column_name], errors="coerce")

    timestamp_frame = data_frame[["year", "month", "day", "hour", "minute"]].copy()
    timestamp_frame["second"] = 0
    data_frame["timestamp"] = pd.to_datetime(timestamp_frame, utc=utc, errors="coerce")
    if utc:
        data_frame["timestamp_utc"] = data_frame["timestamp"]

    def metadata_value(*keys: str) -> object:
        for key in keys:
            value = metadata.get(key)
            if value not in (None, ""):
                return value
        return None

    site_id = _safe_int(metadata_value("Location ID", "SiteID", "site_id", "siteid"))
    if site_id is None:
        site_id = _safe_int(request_row.get("site_id"))
    if site_id is not None:
        data_frame["site_id"] = int(site_id)

    latitude = _safe_float(metadata_value("Latitude", "latitude"))
    if latitude is None:
        latitude = _safe_float(request_row.get("latitude"))
    if latitude is not None:
        data_frame["latitude"] = float(latitude)

    longitude = _safe_float(metadata_value("Longitude", "longitude"))
    if longitude is None:
        longitude = _safe_float(request_row.get("longitude"))
    if longitude is not None:
        data_frame["longitude"] = float(longitude)

    data_frame["source"] = metadata_value("Source", "Dataset", "dataset")
    data_frame["country"] = metadata_value("Country", "country")
    data_frame["version"] = metadata_value("Version", "version")
    data_frame["time_zone"] = _safe_float(metadata_value("Time Zone", "Site Timezone", "site_timezone"))
    data_frame["local_time_zone"] = _safe_float(metadata_value("Local Time Zone", "Data Timezone", "local_time_zone"))
    data_frame["elevation"] = _safe_float(metadata_value("Elevation", "Site Elevation", "elevation"))
    request_year = _safe_int(request_row.get("year"))
    if request_year is None and "year" in data_frame.columns and not data_frame["year"].dropna().empty:
        unique_years = sorted({int(value) for value in data_frame["year"].dropna().unique()})
        if len(unique_years) == 1:
            request_year = unique_years[0]
    data_frame["request_year"] = request_year

    interval_minutes = _safe_int(request_row.get("interval_minutes"))
    if interval_minutes is not None:
        data_frame["interval_minutes"] = int(interval_minutes)

    data_frame["requested_attributes"] = str(request_row.get("attributes", ""))
    data_frame["raw_data_header_json"] = json.dumps(data_header, ensure_ascii=True)
    return data_frame


def load_archive_download_queue(ledger_path: Path = DEFAULT_LEDGER_PATH) -> pd.DataFrame:
    if not ledger_path.exists():
        return pd.DataFrame()

    ledger = pd.read_parquet(ledger_path)
    if ledger.empty:
        return ledger

    queue = ledger.loc[
        (ledger["request_format"] == "json")
        & ledger["download_url"].notna()
        & ledger["status"].isin(["submitted", "downloaded", "reused", "extracted"])
    ].copy()
    if queue.empty:
        return queue

    queue["requested_at_utc"] = pd.to_datetime(queue["requested_at_utc"], utc=True, errors="coerce")
    queue = queue.sort_values(["requested_at_utc", "batch_id"]).drop_duplicates("batch_id", keep="last")
    return queue.reset_index(drop=True)


def resolve_archive_zip_path(batch_id: str) -> Path:
    return ARCHIVE_DOWNLOAD_ROOT / f"{batch_id}.zip"


def resolve_archive_extract_dir(batch_id: str) -> Path:
    return ARCHIVE_EXTRACT_ROOT / batch_id


def download_archive_queue(
    queue: pd.DataFrame,
    *,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    max_requests: int | None = None,
    dry_run: bool = True,
    overwrite_existing: bool = OVERWRITE_EXISTING_ARCHIVES,
    timeout: int = ARCHIVE_DOWNLOAD_TIMEOUT_SECONDS,
    max_attempts: int = ARCHIVE_DOWNLOAD_MAX_ATTEMPTS,
    poll_seconds: float = ARCHIVE_DOWNLOAD_POLL_SECONDS,
) -> pd.DataFrame:
    if queue.empty:
        return pd.DataFrame()

    selected = queue.head(max_requests).copy() if max_requests is not None else queue.copy()
    session = requests.Session()
    rows: list[dict[str, object]] = []

    for request_row in selected.itertuples(index=False):
        archive_path = resolve_archive_zip_path(str(request_row.batch_id))
        ledger_record = {
            "requested_at_utc": utc_now_iso(),
            "endpoint": GOES_FULL_DISC_DOWNLOAD_URL,
            "dataset": GOES_FULL_DISC_DATASET,
            "request_format": "json",
            "batch_id": str(request_row.batch_id),
            "years": str(request_row.years),
            "interval_minutes": _safe_int(request_row.interval_minutes),
            "attributes": str(request_row.attributes),
            "site_count": _safe_int(request_row.site_count),
            "request_weight": _safe_int(request_row.request_weight),
            "download_url": str(request_row.download_url),
            "local_archive_path": str(archive_path),
            "notes": None,
        }

        if dry_run:
            ledger_record["status"] = "dry_run"
            ledger_record["http_status"] = None
            ledger_record["notes"] = "Archive download queued but not executed."
            append_ledger_record(ledger_path, ledger_record)
            rows.append(ledger_record)
            continue

        if archive_path.exists() and not overwrite_existing:
            ledger_record["status"] = "reused"
            ledger_record["http_status"] = None
            ledger_record["notes"] = "Existing archive reused."
            append_ledger_record(ledger_path, ledger_record)
            rows.append(ledger_record)
            continue

        try:
            download_archive(
                str(request_row.download_url),
                archive_path,
                session=session,
                timeout=timeout,
                max_attempts=max_attempts,
                poll_seconds=poll_seconds,
            )
            ledger_record["status"] = "downloaded"
            ledger_record["http_status"] = 200
            ledger_record["notes"] = f"Archive downloaded to {archive_path}"
        except Exception as exc:
            ledger_record["status"] = "error"
            ledger_record["http_status"] = None
            ledger_record["notes"] = str(exc)

        append_ledger_record(ledger_path, ledger_record)
        rows.append(ledger_record)

    return pd.DataFrame(rows)


def extract_archive_queue(
    queue: pd.DataFrame,
    *,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    overwrite_existing: bool = OVERWRITE_EXTRACTED_ARCHIVES,
) -> pd.DataFrame:
    if queue.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for request_row in queue.itertuples(index=False):
        archive_path = Path(str(request_row.local_archive_path))
        extract_dir = resolve_archive_extract_dir(str(request_row.batch_id))
        ledger_record = {
            "requested_at_utc": utc_now_iso(),
            "endpoint": GOES_FULL_DISC_DOWNLOAD_URL,
            "dataset": GOES_FULL_DISC_DATASET,
            "request_format": "json",
            "batch_id": str(request_row.batch_id),
            "years": str(request_row.years),
            "interval_minutes": _safe_int(request_row.interval_minutes),
            "attributes": str(request_row.attributes),
            "site_count": _safe_int(request_row.site_count),
            "request_weight": _safe_int(request_row.request_weight),
            "download_url": str(request_row.download_url),
            "local_archive_path": str(archive_path),
            "notes": None,
        }

        if not archive_path.exists():
            ledger_record["status"] = "error"
            ledger_record["http_status"] = None
            ledger_record["notes"] = f"Archive file is missing: {archive_path}"
            append_ledger_record(ledger_path, ledger_record)
            rows.append(ledger_record)
            continue

        try:
            extracted_files = extract_archive(archive_path, extract_dir, overwrite=overwrite_existing)
            ledger_record["status"] = "extracted"
            ledger_record["http_status"] = None
            ledger_record["notes"] = f"Extracted {len(extracted_files):,} files to {extract_dir}"
        except Exception as exc:
            ledger_record["status"] = "error"
            ledger_record["http_status"] = None
            ledger_record["notes"] = str(exc)

        append_ledger_record(ledger_path, ledger_record)
        rows.append(ledger_record)

    return pd.DataFrame(rows)


def _archive_year_label(frame: pd.DataFrame, fallback: str = "unknown") -> str:
    if "year" not in frame.columns or frame["year"].dropna().empty:
        return fallback

    years = sorted({int(value) for value in frame["year"].dropna().unique()})
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}_{years[-1]}"


def normalize_archive_exports(
    queue: pd.DataFrame,
    *,
    overwrite_existing: bool = False,
    utc: bool = REQUEST_UTC,
) -> pd.DataFrame:
    if queue.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    NORMALIZED_ROOT.mkdir(parents=True, exist_ok=True)

    for request_row in queue.itertuples(index=False):
        extract_dir = resolve_archive_extract_dir(str(request_row.batch_id))
        if not extract_dir.exists():
            continue

        csv_paths = sorted(path for path in extract_dir.rglob("*.csv") if path.is_file())
        for csv_path in csv_paths:
            try:
                normalized_frame = normalize_nsrdb_csv_text(
                    csv_path.read_text(encoding="utf-8", errors="replace"),
                    request_row=pd.Series(
                        {
                            "attributes": str(request_row.attributes),
                            "interval_minutes": _safe_int(request_row.interval_minutes),
                        }
                    ),
                    utc=utc,
                )
                site_series = normalized_frame["site_id"].dropna() if "site_id" in normalized_frame.columns else pd.Series(dtype=float)
                if site_series.empty:
                    raise RuntimeError(f"Normalized archive export is missing site_id metadata: {csv_path}")

                site_id = int(site_series.iloc[0])
                year_label = _archive_year_label(normalized_frame, fallback=str(request_row.years).replace(",", "_"))
                normalized_path = NORMALIZED_ROOT / f"nsrdb_site_{site_id}_{year_label}_{INTERVAL_MINUTES}min.parquet"
                status = "normalized"
                if normalized_path.exists() and not overwrite_existing:
                    status = "reused"
                else:
                    normalized_frame.to_parquet(normalized_path, index=False)

                rows.append(
                    {
                        "batch_id": str(request_row.batch_id),
                        "source_csv_path": str(csv_path),
                        "normalized_parquet_path": str(normalized_path),
                        "site_id": site_id,
                        "year_label": year_label,
                        "row_count": int(len(normalized_frame)),
                        "status": status,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "batch_id": str(request_row.batch_id),
                        "source_csv_path": str(csv_path),
                        "normalized_parquet_path": None,
                        "site_id": None,
                        "year_label": None,
                        "row_count": 0,
                        "status": f"error: {exc}",
                    }
                )

    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        ARCHIVE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_parquet(ARCHIVE_MANIFEST_PATH, index=False)
    return manifest


def download_csv_request_plan(
    request_plan: pd.DataFrame,
    *,
    api_key: str,
    email: str,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    max_requests: int | None = MAX_CSV_DOWNLOADS_THIS_RUN,
    dry_run: bool = True,
    pause_seconds: float = CSV_REQUEST_PAUSE_SECONDS,
    overwrite_existing: bool = OVERWRITE_EXISTING_CSV,
    utc: bool = REQUEST_UTC,
    leap_day: bool = INCLUDE_LEAP_DAY,
    timeout: int = CSV_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    if request_plan.empty:
        return pd.DataFrame()

    selected = request_plan.head(max_requests).copy() if max_requests is not None else request_plan.copy()
    session = requests.Session()
    rows: list[dict[str, object]] = []

    for request_row in selected.itertuples(index=False):
        raw_csv_path = Path(str(request_row.raw_csv_path))
        normalized_path = Path(str(request_row.normalized_parquet_path))
        ledger_record = {
            "requested_at_utc": utc_now_iso(),
            "endpoint": CSV_DIRECT_DOWNLOAD_URL,
            "dataset": GOES_FULL_DISC_DATASET,
            "request_format": "csv",
            "batch_id": request_row.batch_id,
            "years": str(request_row.year),
            "interval_minutes": int(request_row.interval_minutes),
            "attributes": str(request_row.attributes),
            "site_count": 1,
            "request_weight": int(request_row.request_weight),
            "download_url": None,
            "local_archive_path": str(raw_csv_path),
            "notes": None,
        }

        if dry_run:
            ledger_record["status"] = "dry_run"
            ledger_record["http_status"] = None
            ledger_record["notes"] = "Direct CSV request planned but not executed."
            append_ledger_record(ledger_path, ledger_record)
            rows.append(ledger_record)
            continue

        if not overwrite_existing and raw_csv_path.exists() and normalized_path.exists():
            ledger_record["status"] = "reused"
            ledger_record["http_status"] = None
            ledger_record["notes"] = "Existing raw CSV and normalized parquet reused."
            append_ledger_record(ledger_path, ledger_record)
            rows.append(ledger_record)
            continue

        params = {
            "api_key": api_key,
            "email": email,
            "wkt": request_row.request_wkt,
            "attributes": request_row.attributes,
            "names": str(request_row.year),
            "interval": str(request_row.interval_minutes),
            "utc": str(bool(utc)).lower(),
            "leap_day": str(bool(leap_day)).lower(),
        }

        try:
            response = session.get(CSV_DIRECT_DOWNLOAD_URL, params=params, timeout=timeout)
            ledger_record["http_status"] = int(response.status_code)
            if not response.ok:
                raise RuntimeError(f"NSRDB CSV request failed with HTTP {response.status_code}: {response.text[:500]}")

            raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
            raw_csv_path.write_text(response.text)
            normalized_frame = normalize_nsrdb_csv_text(response.text, request_row=pd.Series(request_row._asdict()), utc=utc)
            normalized_path.parent.mkdir(parents=True, exist_ok=True)
            normalized_frame.to_parquet(normalized_path, index=False)

            ledger_record["status"] = "downloaded"
            ledger_record["notes"] = f"Normalized parquet written to {normalized_path}"
        except Exception as exc:
            ledger_record["status"] = "error"
            ledger_record["notes"] = str(exc)

        append_ledger_record(ledger_path, ledger_record)
        rows.append(ledger_record)
        time.sleep(max(pause_seconds, 1.05))

    return pd.DataFrame(rows)


def list_normalized_nsrdb_parquets(normalized_root: Path = NORMALIZED_ROOT) -> list[Path]:
    if not normalized_root.exists():
        return []

    excluded_names = {
        MONTHLY_FLUX_MEANS_PATH.name,
        ANNUAL_FLUX_MEANS_PATH.name,
        MULTIYEAR_FLUX_MEANS_PATH.name,
    }
    return sorted(path for path in normalized_root.glob("*.parquet") if path.name not in excluded_names)


def load_normalized_nsrdb_timeseries(normalized_root: Path = NORMALIZED_ROOT) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in list_normalized_nsrdb_parquets(normalized_root):
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["_source_file"] = path.name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def filter_timeseries_to_target_municipalities(
    frame: pd.DataFrame,
    target_municipalities: tuple[str, ...] | None = FETCH_SCOPE_MUNICIPALITIES,
) -> pd.DataFrame:
    if frame.empty or not target_municipalities:
        return frame

    for column_name in ("municipio", "municipality_name"):
        if column_name in frame.columns:
            return frame[frame[column_name].isin(target_municipalities)].copy()
    return frame


def resolve_time_column(frame: pd.DataFrame) -> str:
    for column_name in ("timestamp_utc", "timestamp", "time", "datetime", "period_end"):
        if column_name in frame.columns:
            return column_name
    raise RuntimeError(
        "Normalized NSRDB tables must include one of: timestamp_utc, timestamp, time, datetime, or period_end."
    )


def resolve_site_columns(frame: pd.DataFrame) -> list[str]:
    site_columns = [column_name for column_name in ("site_id", "gid") if column_name in frame.columns]
    coordinate_columns = [column_name for column_name in ("latitude", "longitude") if column_name in frame.columns]
    if site_columns:
        return site_columns + [column_name for column_name in coordinate_columns if column_name not in site_columns]
    if len(coordinate_columns) == 2:
        return coordinate_columns
    return []


def build_flux_mean_tables(
    timeseries: pd.DataFrame,
    *,
    summary_columns: tuple[str, ...] = NSRDB_FLUX_SUMMARY_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if timeseries.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    frame = timeseries.copy()
    time_column = resolve_time_column(frame)
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    frame = frame.loc[frame[time_column].notna()].copy()
    if frame.empty:
        raise RuntimeError("Normalized NSRDB data does not contain any parseable timestamps.")

    available_columns = [column_name for column_name in summary_columns if column_name in frame.columns]
    if not available_columns:
        raise RuntimeError(
            "Normalized NSRDB data does not contain any of the expected summary columns: "
            + ", ".join(summary_columns)
        )

    frame["year"] = frame[time_column].dt.year
    frame["month"] = frame[time_column].dt.month
    site_columns = resolve_site_columns(frame)

    monthly_group_columns = site_columns + ["year", "month"] if site_columns else ["year", "month"]
    annual_group_columns = site_columns + ["year"] if site_columns else ["year"]

    monthly = (
        frame.groupby(monthly_group_columns, dropna=False)[available_columns]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(monthly_group_columns)
        .reset_index(drop=True)
    )
    annual = (
        frame.groupby(annual_group_columns, dropna=False)[available_columns]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(annual_group_columns)
        .reset_index(drop=True)
    )

    if site_columns:
        multiyear = (
            annual.groupby(site_columns, dropna=False)[available_columns]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(site_columns)
            .reset_index(drop=True)
        )
    else:
        multiyear = pd.DataFrame([frame[available_columns].mean(numeric_only=True).to_dict()])

    return monthly, annual, multiyear


def write_flux_mean_tables(
    monthly: pd.DataFrame,
    annual: pd.DataFrame,
    multiyear: pd.DataFrame,
) -> None:
    NORMALIZED_ROOT.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(MONTHLY_FLUX_MEANS_PATH, index=False)
    annual.to_parquet(ANNUAL_FLUX_MEANS_PATH, index=False)
    multiyear.to_parquet(MULTIYEAR_FLUX_MEANS_PATH, index=False)


def preview_table(name: str, frame: pd.DataFrame, *, rows: int = 5) -> None:
    print(f"{name}: {len(frame):,} rows")
    if frame.empty:
        return
    print(frame.head(rows).to_string(index=False))