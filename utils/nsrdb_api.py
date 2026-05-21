"""NSRDB GOES Full Disc request planning and submission helpers.

The helpers in this module are intentionally scoped to the first-draft ingest
workflow for PLAN6068:

- estimate request size with the documented NSRDB weight formula,
- query the NSRDB site-count endpoint for arbitrary WKT geometries,
- recursively split a polygon into request-sized batches,
- submit GOES Full Disc archive requests with the API key in the query string,
- append a parquet ledger row for dry runs, submissions, failures, and
  downloads.

The dynamic download endpoint is asynchronous. A successful JSON submission only
acknowledges that file generation has started and returns a download URL for the
archive when it is ready.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from shutil import rmtree
from zipfile import ZipFile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, Sequence

import geopandas as gpd
import pandas as pd
import requests
from dotenv import load_dotenv
from shapely import wkt as shapely_wkt
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry


def _resolve_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in ("project_rules.md", ".git")):
            return candidate
    return current


PROJECT_ROOT = _resolve_project_root()
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


NSRDB_ROOT = _resolve_path("NSRDB_ROOT", PROJECT_ROOT / "data" / "tabular" / "nsrdb")
RAW_ROOT = NSRDB_ROOT / "raw"
NORMALIZED_ROOT = NSRDB_ROOT / "normalized"
DEFAULT_LEDGER_PATH = NSRDB_ROOT / "_ledger.parquet"
DEFAULT_PLAN_PATH = NSRDB_ROOT / "request_plan.parquet"

GOES_FULL_DISC_DATASET = "nsrdb-GOES-full-disc-v4-0-0"
GOES_FULL_DISC_DOWNLOAD_URL = "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-full-disc-v4-0-0-download.json"
SITE_COUNT_URL = "https://developer.nlr.gov/api/nsrdb/v2/site-count.json"

MAX_REQUEST_WEIGHT = 175_000_100
MIN_ARCHIVE_REQUEST_SPACING_SECONDS = 2.05
MAX_INFLIGHT_REQUESTS = 20
ARCHIVE_RETRYABLE_STATUS_CODES = (202, 403, 404, 409, 425, 429, 500, 502, 503, 504)
SITE_COUNT_GET_MAX_WKT_CHARS = int(os.getenv("NSRDB_SITE_COUNT_GET_MAX_WKT_CHARS", "1500") or "1500")
SITE_COUNT_SIMPLIFY_TOLERANCES_METERS = (50.0, 100.0, 250.0, 500.0, 1_000.0, 2_000.0, 5_000.0)

DEFAULT_ATTRIBUTES = (
    "air_temperature",
    "clearsky_dhi",
    "clearsky_dni",
    "clearsky_ghi",
    "dhi",
    "dni",
    "ghi",
    "solar_zenith_angle",
    "surface_albedo",
)

LEDGER_COLUMNS = [
    "requested_at_utc",
    "endpoint",
    "dataset",
    "request_format",
    "batch_id",
    "years",
    "interval_minutes",
    "attributes",
    "site_count",
    "request_weight",
    "status",
    "http_status",
    "download_url",
    "local_archive_path",
    "notes",
]


@dataclass(frozen=True)
class RequestBatch:
    batch_id: str
    geometry: BaseGeometry
    split_depth: int
    site_count: int
    request_weight: int
    request_weight_pct: float
    area_km2: float
    over_limit: bool


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_geometry(geometry: BaseGeometry) -> BaseGeometry:
    if geometry.is_empty:
        return geometry
    normalized = geometry.buffer(0)
    return normalized if not normalized.is_empty else geometry


def polygonal_parts(geometry: BaseGeometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [part for part in geometry.geoms if not part.is_empty]
    if isinstance(geometry, GeometryCollection):
        parts: list[Polygon] = []
        for part in geometry.geoms:
            parts.extend(polygonal_parts(part))
        return parts
    return []


def geometry_area_km2(geometry: BaseGeometry) -> float:
    if geometry.is_empty:
        return 0.0
    return float(gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs("EPSG:6933").area.iloc[0] / 1_000_000.0)


def simplify_geometry_for_site_count(
    geometry: BaseGeometry,
    *,
    max_wkt_chars: int = SITE_COUNT_GET_MAX_WKT_CHARS,
    tolerances_meters: Sequence[float] = SITE_COUNT_SIMPLIFY_TOLERANCES_METERS,
) -> BaseGeometry:
    normalized = normalize_geometry(geometry)
    if normalized.is_empty or len(normalized.wkt) <= max_wkt_chars:
        return normalized

    projected = gpd.GeoSeries([normalized], crs="EPSG:4326").to_crs("EPSG:6933").iloc[0]
    best_geometry = normalized
    best_wkt_length = len(normalized.wkt)

    for tolerance_meters in tolerances_meters:
        simplified_projected = normalize_geometry(projected.simplify(tolerance_meters, preserve_topology=True))
        if simplified_projected.is_empty:
            continue
        simplified = gpd.GeoSeries([simplified_projected], crs="EPSG:6933").to_crs("EPSG:4326").iloc[0]
        simplified = normalize_geometry(simplified)
        if simplified.is_empty:
            continue
        simplified_wkt_length = len(simplified.wkt)
        if simplified_wkt_length < best_wkt_length:
            best_geometry = simplified
            best_wkt_length = simplified_wkt_length
        if simplified_wkt_length <= max_wkt_chars:
            return simplified

    return best_geometry


def simplify_wkt_for_site_count(
    geometry_wkt: str,
    *,
    max_wkt_chars: int = SITE_COUNT_GET_MAX_WKT_CHARS,
) -> str:
    if len(geometry_wkt) <= max_wkt_chars:
        return geometry_wkt
    geometry = shapely_wkt.loads(geometry_wkt)
    return simplify_geometry_for_site_count(geometry, max_wkt_chars=max_wkt_chars).wkt


def compute_request_weight(
    site_count: int,
    attribute_count: int,
    year_count: int,
    interval_minutes: int,
) -> int:
    intervals_per_year = int((60 / interval_minutes) * 24 * 365)
    return int(site_count) * int(attribute_count) * int(year_count) * intervals_per_year


def max_sites_per_request(
    *,
    attribute_count: int,
    year_count: int,
    interval_minutes: int,
    max_request_weight: int = MAX_REQUEST_WEIGHT,
) -> int:
    if attribute_count <= 0 or year_count <= 0:
        raise ValueError("attribute_count and year_count must both be positive.")
    intervals_per_year = int((60 / interval_minutes) * 24 * 365)
    denominator = attribute_count * year_count * intervals_per_year
    if denominator <= 0:
        raise ValueError("Computed request-weight denominator must be positive.")
    return max(1, int(max_request_weight // denominator))


def subdivide_geometry(geometry: BaseGeometry) -> list[BaseGeometry]:
    normalized = normalize_geometry(geometry)
    if normalized.is_empty:
        return []

    minx, miny, maxx, maxy = normalized.bounds
    midx = (minx + maxx) / 2.0
    midy = (miny + maxy) / 2.0
    quadrants = [
        box(minx, miny, midx, midy),
        box(midx, miny, maxx, midy),
        box(minx, midy, midx, maxy),
        box(midx, midy, maxx, maxy),
    ]

    pieces: list[BaseGeometry] = []
    for quadrant in quadrants:
        clipped = normalize_geometry(normalized.intersection(quadrant))
        for part in polygonal_parts(clipped):
            if not part.is_empty:
                pieces.append(part)
    return pieces


def _load_ledger(ledger_path: Path) -> pd.DataFrame:
    if ledger_path.exists():
        try:
            return pd.read_parquet(ledger_path)
        except Exception:
            pass
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def append_ledger_record(ledger_path: Path, record: dict[str, object]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    frame = _load_ledger(ledger_path)
    row = {column: record.get(column) for column in LEDGER_COLUMNS}
    if frame.empty:
        frame = pd.DataFrame([row], columns=LEDGER_COLUMNS)
    else:
        frame = pd.concat([frame, pd.DataFrame([row], columns=LEDGER_COLUMNS)], ignore_index=True)
    tmp_path = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    frame.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, ledger_path)


def parse_json_response(response: requests.Response) -> dict[str, object]:
    if not response.ok:
        raise RuntimeError(f"NSRDB request failed with HTTP {response.status_code}: {response.text[:500]}")
    payload = response.json()
    errors = payload.get("errors") or []
    if errors:
        raise RuntimeError("NSRDB returned errors: " + " | ".join(str(error) for error in errors))
    return payload


def extract_download_url(outputs: dict[str, object] | None) -> str | None:
    if not outputs:
        return None

    for key in ("downloadUrl", "download_url", "downloadURL"):
        value = outputs.get(key)
        if value:
            return str(value)
    return None


def query_site_count(
    geometry_wkt: str,
    *,
    api_key: str,
    dataset: str = GOES_FULL_DISC_DATASET,
    session: requests.Session | None = None,
    timeout: int = 120,
    method: str | None = None,
) -> int:
    client = session or requests.Session()
    request_method = (method or "get").lower()
    if request_method == "post":
        response = client.post(
            SITE_COUNT_URL,
            params={"api_key": api_key},
            data={"wkt": geometry_wkt},
            headers={"content-type": "application/x-www-form-urlencoded"},
            timeout=timeout,
        )
    elif request_method == "get":
        geometry_wkt = simplify_wkt_for_site_count(geometry_wkt)
        response = client.get(
            SITE_COUNT_URL,
            params={"api_key": api_key, "wkt": geometry_wkt},
            timeout=timeout,
        )
    else:
        raise ValueError("method must be 'get', 'post', or None.")
    payload = parse_json_response(response)
    outputs = payload.get("outputs") or {}
    return int(outputs.get(dataset, 0))


def plan_download_batches(
    geometry: BaseGeometry,
    *,
    attributes: Sequence[str],
    years: Sequence[int | str],
    interval_minutes: int,
    api_key: str | None = None,
    dataset: str = GOES_FULL_DISC_DATASET,
    max_request_weight: int = MAX_REQUEST_WEIGHT,
    max_sites_override: int | None = None,
    max_depth: int = 8,
    min_area_km2: float = 1.0,
    session: requests.Session | None = None,
    site_count_resolver: Callable[[BaseGeometry], int] | None = None,
) -> gpd.GeoDataFrame:
    if not attributes:
        raise ValueError("At least one NSRDB attribute is required.")
    if not years:
        raise ValueError("At least one NSRDB year is required.")

    if site_count_resolver is None:
        if not api_key:
            raise ValueError("api_key is required when no custom site_count_resolver is supplied.")

        def site_count_resolver(candidate_geometry: BaseGeometry) -> int:
            return query_site_count(candidate_geometry.wkt, api_key=api_key, dataset=dataset, session=session)

    max_sites = max_sites_override or max_sites_per_request(
        attribute_count=len(attributes),
        year_count=len(years),
        interval_minutes=interval_minutes,
        max_request_weight=max_request_weight,
    )

    pending: list[tuple[BaseGeometry, int]] = [(normalize_geometry(geometry), 0)]
    accepted: list[RequestBatch] = []
    batch_index = 0

    while pending:
        candidate_geometry, split_depth = pending.pop()
        if candidate_geometry.is_empty:
            continue

        site_count = int(site_count_resolver(candidate_geometry))
        if site_count <= 0:
            continue

        request_weight = compute_request_weight(site_count, len(attributes), len(years), interval_minutes)
        area_km2 = geometry_area_km2(candidate_geometry)
        over_limit = site_count > max_sites
        if over_limit and split_depth < max_depth and area_km2 > min_area_km2:
            children = subdivide_geometry(candidate_geometry)
            if children:
                pending.extend((child, split_depth + 1) for child in reversed(children))
                continue

        batch_index += 1
        accepted.append(
            RequestBatch(
                batch_id=f"pr_nsrdb_{batch_index:03d}",
                geometry=candidate_geometry,
                split_depth=split_depth,
                site_count=site_count,
                request_weight=request_weight,
                request_weight_pct=(request_weight / max_request_weight) * 100.0,
                area_km2=area_km2,
                over_limit=over_limit,
            )
        )

    rows = [
        {
            "batch_id": batch.batch_id,
            "split_depth": batch.split_depth,
            "site_count": batch.site_count,
            "request_weight": batch.request_weight,
            "request_weight_pct": batch.request_weight_pct,
            "area_km2": batch.area_km2,
            "over_limit": batch.over_limit,
            "geometry_wkt": batch.geometry.wkt,
            "geometry": batch.geometry,
        }
        for batch in accepted
    ]
    if not rows:
        return gpd.GeoDataFrame(columns=["batch_id", "site_count", "request_weight", "geometry_wkt", "geometry"], geometry="geometry", crs="EPSG:4326")

    plan = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return plan.sort_values(by=["over_limit", "split_depth", "site_count", "batch_id"], ascending=[True, True, False, True]).reset_index(drop=True)


def submit_download_request(
    *,
    api_key: str,
    email: str,
    geometry_wkt: str,
    years: Sequence[int | str],
    attributes: Sequence[str],
    interval_minutes: int,
    utc: bool = True,
    leap_day: bool = True,
    full_name: str | None = None,
    affiliation: str | None = None,
    reason: str | None = None,
    mailing_list: bool = False,
    session: requests.Session | None = None,
    timeout: int = 180,
) -> dict[str, object]:
    client = session or requests.Session()
    payload = {
        "wkt": geometry_wkt,
        "attributes": ",".join(str(attribute) for attribute in attributes),
        "names": ",".join(str(year) for year in years),
        "interval": str(interval_minutes),
        "utc": str(bool(utc)).lower(),
        "leap_day": str(bool(leap_day)).lower(),
        "email": email,
    }
    if full_name:
        payload["full_name"] = full_name
    if affiliation:
        payload["affiliation"] = affiliation
    if reason:
        payload["reason"] = reason
    if mailing_list:
        payload["mailing_list"] = "true"

    response = client.post(
        GOES_FULL_DISC_DOWNLOAD_URL,
        params={"api_key": api_key},
        data=payload,
        headers={"content-type": "application/x-www-form-urlencoded"},
        timeout=timeout,
    )
    return parse_json_response(response)


def submit_planned_requests(
    plan: gpd.GeoDataFrame,
    *,
    api_key: str,
    email: str,
    years: Sequence[int | str],
    attributes: Sequence[str],
    interval_minutes: int,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    max_requests: int | None = None,
    dry_run: bool = True,
    pause_seconds: float = MIN_ARCHIVE_REQUEST_SPACING_SECONDS,
    utc: bool = True,
    leap_day: bool = True,
    full_name: str | None = None,
    affiliation: str | None = None,
    reason: str | None = None,
    mailing_list: bool = False,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    client = session or requests.Session()
    rows: list[dict[str, object]] = []
    selected = plan.head(max_requests).copy() if max_requests is not None else plan.copy()

    for request_row in selected.itertuples(index=False):
        record = {
            "requested_at_utc": utc_now_iso(),
            "endpoint": GOES_FULL_DISC_DOWNLOAD_URL,
            "dataset": GOES_FULL_DISC_DATASET,
            "request_format": "json",
            "batch_id": request_row.batch_id,
            "years": ",".join(str(year) for year in years),
            "interval_minutes": int(interval_minutes),
            "attributes": ",".join(str(attribute) for attribute in attributes),
            "site_count": int(request_row.site_count),
            "request_weight": int(request_row.request_weight),
            "download_url": None,
            "local_archive_path": None,
            "notes": None,
        }

        if dry_run:
            record["status"] = "dry_run"
            record["http_status"] = None
            record["notes"] = "Request planned but not submitted."
            append_ledger_record(ledger_path, record)
            rows.append(record)
            continue

        try:
            payload = submit_download_request(
                api_key=api_key,
                email=email,
                geometry_wkt=request_row.geometry_wkt,
                years=years,
                attributes=attributes,
                interval_minutes=interval_minutes,
                utc=utc,
                leap_day=leap_day,
                full_name=full_name,
                affiliation=affiliation,
                reason=reason,
                mailing_list=mailing_list,
                session=client,
            )
            outputs = payload.get("outputs") or {}
            record["status"] = "submitted"
            record["http_status"] = int(payload.get("status") or 200)
            record["download_url"] = extract_download_url(outputs)
            record["notes"] = outputs.get("message")
        except Exception as exc:
            record["status"] = "error"
            record["http_status"] = None
            record["notes"] = str(exc)

        append_ledger_record(ledger_path, record)
        rows.append(record)
        time.sleep(max(pause_seconds, MIN_ARCHIVE_REQUEST_SPACING_SECONDS))

    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def download_archive(
    download_url: str,
    destination_path: Path,
    *,
    session: requests.Session | None = None,
    timeout: int = 600,
    max_attempts: int = 1,
    poll_seconds: float = 0.0,
    retryable_status_codes: Sequence[int] = ARCHIVE_RETRYABLE_STATUS_CODES,
) -> Path:
    client = session or requests.Session()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination_path.with_suffix(destination_path.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, max(1, int(max_attempts)) + 1):
        try:
            with client.get(download_url, stream=True, timeout=timeout) as response:
                if response.status_code in retryable_status_codes:
                    message = f"Archive download not ready yet (HTTP {response.status_code})"
                    if attempt >= max_attempts:
                        raise RuntimeError(message)
                    last_error = RuntimeError(message)
                else:
                    response.raise_for_status()
                    with tmp_path.open("wb") as file_handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                file_handle.write(chunk)
                    os.replace(tmp_path, destination_path)
                    return destination_path
        except Exception as exc:
            last_error = exc
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        if attempt < max_attempts:
            time.sleep(max(0.0, float(poll_seconds)))

    if last_error is not None:
        raise last_error
    raise RuntimeError("Archive download failed without a captured exception.")


def extract_archive(
    archive_path: Path,
    destination_dir: Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    if overwrite and destination_dir.exists():
        rmtree(destination_dir)

    destination_dir.mkdir(parents=True, exist_ok=True)
    root_dir = destination_dir.resolve()
    extracted_paths: list[Path] = []

    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            member_path = (destination_dir / member.filename).resolve()
            if member_path != root_dir and root_dir not in member_path.parents:
                raise RuntimeError(f"Refusing to extract archive member outside destination: {member.filename}")

            archive.extract(member, destination_dir)
            extracted_paths.append(member_path)

    return extracted_paths


def summarize_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> dict[str, object]:
    ledger = _load_ledger(ledger_path)
    if ledger.empty:
        return {
            "rows": 0,
            "submitted": 0,
            "dry_run": 0,
            "downloaded": 0,
            "errors": 0,
        }
    status_counts = ledger["status"].fillna("unknown").value_counts().to_dict()
    return {
        "rows": int(len(ledger)),
        "submitted": int(status_counts.get("submitted", 0)),
        "dry_run": int(status_counts.get("dry_run", 0)),
        "downloaded": int(status_counts.get("downloaded", 0)),
        "errors": int(status_counts.get("error", 0)),
    }