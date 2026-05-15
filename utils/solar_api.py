"""Google Solar API client for the PLAN6068 PV project.

SDK-based wrapper around ``buildingInsights:findClosest`` and
``dataLayers:get`` + ``geoTiff:get`` that:

- uses the official ``google-maps-solar`` SDK (``SolarClient`` for probes,
  ``SolarAsyncClient`` for data layers + GeoTIFF downloads in parallel),
- picks a quality-appropriate pixel size and auto-adjusts it so every request
  satisfies the Google Solar API constraint ``radius_meters <= pixel_size_meters
  * 1000`` (otherwise the API returns HTTP 400 InvalidArgument for every call),
- downloads RGB + building-mask + annual-flux + DSM GeoTIFFs to a deterministic
  project path ``data/rasters/solar/{municipio}/{bg_geoid}/``,
- caches the Data Layers JSON envelope on disk so re-runs don't burn paid
  calls,
- maintains a persistent parquet ledger that enforces a hard 1,999-call quota
  cap and tracks cost estimates (ok, billed calls only),
- records structured error bodies from ``GoogleAPICallError`` for any failed
  tile so the root cause is visible in the ledger.

Auth: Application Default Credentials (``gcloud auth application-default
login`` plus ``set-quota-project``).  ADC is used because the SDK's
API-key-via-``ClientOptions`` path does not attach the key to Data Layers
calls in ``google-maps-solar`` 0.x.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal
from urllib.parse import parse_qs, urlparse

import pandas as pd
from dotenv import load_dotenv

from google.api_core.exceptions import (
    GoogleAPICallError,
    NotFound,
    ResourceExhausted,
    ServiceUnavailable,
)
from google.maps import solar_v1
from google.maps.solar_v1.types import (
    DataLayerView as _DataLayerViewEnum,
    Experiment as _ExperimentEnum,
    FindClosestBuildingInsightsRequest,
    GetDataLayersRequest,
    GetGeoTiffRequest,
    ImageryQuality as _ImageryQualityEnum,
)
from google.type.latlng_pb2 import LatLng

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

def _resolve_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in ("project_rules.md", ".git")):
            return candidate
    return current


PROJECT_ROOT = _resolve_project_root()
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(env_value: str | None, fallback: Path) -> Path:
    if not env_value:
        return fallback
    path = Path(env_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


DEFAULT_SOLAR_ROOT = _resolve_path(os.getenv("SOLAR_RASTER_ROOT"), PROJECT_ROOT / "data" / "rasters" / "solar")
DEFAULT_CACHE_DIR = DEFAULT_SOLAR_ROOT / "cache"
DEFAULT_LEDGER_PATH = DEFAULT_SOLAR_ROOT / "_ledger.parquet"

# Hard budget: 1,000 free + user-covered delta at $75/1K. Cap below 2,000 to
# guarantee we never silently overshoot the user's paid quota.
DATA_LAYERS_BUDGET_CAP = 1_999
DATA_LAYERS_FREE_TIER_CALLS = 1_000  # first 1K dataLayers calls per month are free
BUILDING_INSIGHTS_SOFT_CAP = 9_500

DATA_LAYERS_UNIT_COST_USD = 0.075
BUILDING_INSIGHTS_UNIT_COST_USD = 0.0  # within 10K/mo free tier for this project

# Narrow string-literal aliases for external callers; internally we map to enums.
DataLayerView = Literal[
    "DSM_LAYER",
    "IMAGERY_LAYERS",
    "IMAGERY_AND_ANNUAL_FLUX_LAYERS",
    "IMAGERY_AND_ALL_FLUX_LAYERS",
    "FULL_LAYERS",
]
ImageryQuality = Literal["HIGH", "MEDIUM", "BASE"]

DEFAULT_VIEW: DataLayerView = "IMAGERY_AND_ANNUAL_FLUX_LAYERS"

_VIEW_ENUM: dict[str, _DataLayerViewEnum] = {
    "DSM_LAYER": _DataLayerViewEnum.DSM_LAYER,
    "IMAGERY_LAYERS": _DataLayerViewEnum.IMAGERY_LAYERS,
    "IMAGERY_AND_ANNUAL_FLUX_LAYERS": _DataLayerViewEnum.IMAGERY_AND_ANNUAL_FLUX_LAYERS,
    "IMAGERY_AND_ALL_FLUX_LAYERS": _DataLayerViewEnum.IMAGERY_AND_ALL_FLUX_LAYERS,
    "FULL_LAYERS": _DataLayerViewEnum.FULL_LAYERS,
}

_QUALITY_ENUM: dict[str, _ImageryQualityEnum] = {
    "HIGH": _ImageryQualityEnum.HIGH,
    "MEDIUM": _ImageryQualityEnum.MEDIUM,
    "BASE": _ImageryQualityEnum.BASE,
}

# Canonical short names for each layer field on the `DataLayers` response.
# These correspond to attributes on the protobuf message returned by the SDK.
REQUESTED_LAYERS: dict[str, str] = {
    "rgb_url": "rgb",
    "mask_url": "mask",
    "annual_flux_url": "annualFlux",
    "dsm_url": "dsm",
}


def quality_to_pixel_size(quality: ImageryQuality) -> float:
    """Default per-pixel resolution (meters) for a given imagery quality tier.

    NOTE: HIGH-quality imagery is *natively* available at 0.10 m, but the
    Solar API enforces ``radius_meters <= pixel_size_meters * 1000``.  Our
    standard tile radius is 175 m, so 0.10 m would be rejected with HTTP 400
    INVALID_ARGUMENT for every call.  We therefore default HIGH to 0.25 m,
    which is the finest resolution compatible with our 175 m tile footprint.
    Callers who explicitly want native 0.10 m must also pass radius_m <= 100
    and an explicit pixel_size_m.
    """

    return 0.25


def enforce_radius_pixel_constraint(radius_m: float, pixel_size_m: float) -> float:
    """Return a valid ``pixel_size_m`` that satisfies the Google constraint.

    The Solar API requires ``radius_meters <= pixel_size_meters * 1000`` for
    any radius > 100 m.  When the caller asks for a finer pixel grid than
    the radius allows, we bump ``pixel_size_m`` to the smallest allowed
    value among Google's supported resolutions (0.1, 0.25, 0.5, 1.0) and log
    a warning.
    """
    if radius_m <= 100:
        return pixel_size_m
    allowed = [0.1, 0.25, 0.5, 1.0]
    for candidate in allowed:
        if candidate >= pixel_size_m and radius_m <= candidate * 1000:
            if candidate != pixel_size_m:
                logger.warning(
                    "pixel_size=%.2fm incompatible with radius=%dm; bumping to %.2fm.",
                    pixel_size_m, int(radius_m), candidate,
                )
            return candidate
    raise ValueError(f"radius={radius_m}m cannot be satisfied by any supported pixel size.")


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

LEDGER_COLUMNS = [
    "timestamp_utc",
    "endpoint",
    "tile_id",
    "bg_geoid",
    "municipio",
    "lon",
    "lat",
    "radius_m",
    "view",
    "required_quality",
    "pixel_size_m",
    "status",
    "returned_quality",
    "imagery_date",
    "cache_hit",
    "http_status",
    "bytes",
    "cost_estimate_usd",
    "notes",
]


def _load_ledger(ledger_path: Path) -> pd.DataFrame:
    if ledger_path.exists():
        try:
            return pd.read_parquet(ledger_path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to read ledger at %s (%s); starting fresh.", ledger_path, exc)
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def _append_ledger(ledger_path: Path, record: dict) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    df = _load_ledger(ledger_path)
    for col in LEDGER_COLUMNS:
        record.setdefault(col, None)
    df = pd.concat([df, pd.DataFrame([record], columns=LEDGER_COLUMNS)], ignore_index=True)
    # Atomic replace: write to a temp file then rename so a SIGKILL mid-write
    # can never leave a half-written parquet behind.
    tmp_path = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, ledger_path)


def ledger_summary(ledger_path: Path = DEFAULT_LEDGER_PATH) -> dict[str, object]:
    """Return a compact summary of spend to date.

    Google Solar Data Layers pricing: first ``DATA_LAYERS_FREE_TIER_CALLS``
    successful calls per billing period are free; every call beyond that is
    billed at ``DATA_LAYERS_UNIT_COST_USD``.  The cost estimate below is
    computed from the *count* of ok/non-cached Data Layers calls, not from
    the per-row ``cost_estimate_usd`` column (which is a rolling per-call
    number and doesn't know about the free tier).
    """
    df = _load_ledger(ledger_path)
    if df.empty:
        return {
            "total_rows": 0,
            "data_layers_successful_calls": 0,
            "data_layers_free_tier_used": 0,
            "data_layers_free_tier_remaining": DATA_LAYERS_FREE_TIER_CALLS,
            "data_layers_billable_calls": 0,
            "data_layers_remaining_to_cap": DATA_LAYERS_BUDGET_CAP,
            "building_insights_calls": 0,
            "total_cost_estimate_usd": 0.0,
        }
    successful_calls = int(
        ((df["endpoint"] == "dataLayers:get") & (df["cache_hit"] == False) & (df["status"] == "ok")).sum()  # noqa: E712
    )
    bi_calls = int(
        ((df["endpoint"] == "buildingInsights:findClosest") & (df["cache_hit"] == False) & (df["status"] == "ok")).sum()  # noqa: E712
    )
    free_tier_used = min(successful_calls, DATA_LAYERS_FREE_TIER_CALLS)
    billable_calls = max(0, successful_calls - DATA_LAYERS_FREE_TIER_CALLS)
    return {
        "total_rows": int(len(df)),
        "data_layers_successful_calls": successful_calls,
        "data_layers_free_tier_used": free_tier_used,
        "data_layers_free_tier_remaining": DATA_LAYERS_FREE_TIER_CALLS - free_tier_used,
        "data_layers_billable_calls": billable_calls,
        "data_layers_remaining_to_cap": DATA_LAYERS_BUDGET_CAP - successful_calls,
        "building_insights_calls": bi_calls,
        "total_cost_estimate_usd": round(billable_calls * DATA_LAYERS_UNIT_COST_USD, 4),
    }


# ---------------------------------------------------------------------------
# Cache keys (JSON envelope cache for Data Layers responses)
# ---------------------------------------------------------------------------

def _cache_key(
    endpoint: str,
    lon: float,
    lat: float,
    *,
    radius_m: int | None = None,
    view: str | None = None,
    pixel_size_m: float | None = None,
    quality: str | None = None,
) -> str:
    parts = [endpoint, f"{round(float(lon), 6):.6f}", f"{round(float(lat), 6):.6f}"]
    if radius_m is not None:
        parts.append(f"r{int(radius_m)}")
    if view is not None:
        parts.append(view)
    if pixel_size_m is not None:
        parts.append(f"p{float(pixel_size_m):g}")
    if quality is not None:
        parts.append(quality)
    return "__".join(parts).replace(":", "_")


def _json_cache_path(key: str, cache_dir: Path) -> Path:
    return cache_dir / f"{key}.json"


def _load_cached_json(key: str, cache_dir: Path) -> dict | None:
    path = _json_cache_path(key, cache_dir)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read cached JSON %s (%s); ignoring.", path, exc)
        return None


def _write_cached_json(key: str, payload: dict, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _json_cache_path(key, cache_dir)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# SDK client factories
# ---------------------------------------------------------------------------

_sync_client: solar_v1.SolarClient | None = None


def _get_sync_client() -> solar_v1.SolarClient:
    global _sync_client
    if _sync_client is None:
        _sync_client = solar_v1.SolarClient()
    return _sync_client


def _date_to_iso(date_proto) -> str | None:
    if not date_proto:
        return None
    y = getattr(date_proto, "year", 0)
    m = getattr(date_proto, "month", 0)
    d = getattr(date_proto, "day", 0)
    if y and m and d:
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return None


def _quality_name(q) -> str | None:
    try:
        return _ImageryQualityEnum(q).name if q is not None else None
    except Exception:
        return str(q) if q is not None else None


# ---------------------------------------------------------------------------
# Probe: buildingInsights:findClosest (sync; free tier)
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    ok: bool
    imagery_quality: str | None
    imagery_date: str | None
    status: str            # "ok" | "not_found" | "error"
    http_status: int | None
    raw: dict | None


def probe_quality(
    lon: float,
    lat: float,
    *,
    required_quality: ImageryQuality | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    tile_id: str | None = None,
    bg_geoid: str | None = None,
    municipio: str | None = None,
) -> ProbeResult:
    """Probe a point's imagery quality via ``buildingInsights:findClosest``.

    Uses an on-disk JSON cache; logs a ledger row on each call (cached or
    network).  Free-tier (10K/mo) so cost is booked as $0.
    """

    key = _cache_key("buildingInsights", lon, lat, quality=required_quality or "ANY")
    cached = _load_cached_json(key, cache_dir)
    if cached is not None:
        status = cached.get("_status", "ok")
        result = ProbeResult(
            ok=(status == "ok"),
            imagery_quality=cached.get("imageryQuality"),
            imagery_date=cached.get("imageryDate"),
            status=status,
            http_status=cached.get("_http_status"),
            raw=cached,
        )
        _append_ledger(ledger_path, {
            "timestamp_utc": pd.Timestamp.utcnow().tz_localize(None).isoformat(),
            "endpoint": "buildingInsights:findClosest",
            "tile_id": tile_id, "bg_geoid": bg_geoid, "municipio": municipio,
            "lon": lon, "lat": lat,
            "required_quality": required_quality,
            "status": status, "returned_quality": result.imagery_quality,
            "imagery_date": result.imagery_date,
            "cache_hit": True, "http_status": result.http_status,
            "cost_estimate_usd": 0.0, "notes": "cache_hit",
        })
        return result

    req = FindClosestBuildingInsightsRequest(
        location=LatLng(latitude=lat, longitude=lon),
    )
    if required_quality is not None:
        req.required_quality = _QUALITY_ENUM[required_quality]

    client = _get_sync_client()
    record: dict
    try:
        resp = client.find_closest_building_insights(request=req)
        record = {
            "imageryQuality": _quality_name(resp.imagery_quality),
            "imageryDate": _date_to_iso(resp.imagery_date),
            "imageryProcessedDate": _date_to_iso(resp.imagery_processed_date),
            "_status": "ok",
            "_http_status": 200,
        }
        _write_cached_json(key, record, cache_dir)
        result = ProbeResult(True, record["imageryQuality"], record["imageryDate"], "ok", 200, record)
    except NotFound:
        record = {"_status": "not_found", "_http_status": 404}
        _write_cached_json(key, record, cache_dir)
        result = ProbeResult(False, None, None, "not_found", 404, record)
    except GoogleAPICallError as exc:
        record = {"_status": "error", "_http_status": getattr(exc, "code", None), "body": str(exc)[:500]}
        result = ProbeResult(False, None, None, "error", record.get("_http_status"), record)

    _append_ledger(ledger_path, {
        "timestamp_utc": pd.Timestamp.utcnow().tz_localize(None).isoformat(),
        "endpoint": "buildingInsights:findClosest",
        "tile_id": tile_id, "bg_geoid": bg_geoid, "municipio": municipio,
        "lon": lon, "lat": lat,
        "required_quality": required_quality,
        "status": result.status, "returned_quality": result.imagery_quality,
        "imagery_date": result.imagery_date,
        "cache_hit": False, "http_status": result.http_status,
        "cost_estimate_usd": 0.0,
        "notes": None if result.ok else str(record.get("body"))[:180],
    })
    return result


# ---------------------------------------------------------------------------
# Data Layers + GeoTIFF fetch (async SDK, wrapped for sync callers)
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    ok: bool
    status: str                  # "ok" | "not_found" | "budget_exhausted" | "error"
    tile_dir: Path | None
    saved_files: dict[str, Path]  # short-name -> local GeoTIFF path
    imagery_quality: str | None
    imagery_date: str | None
    http_status: int | None
    raw: dict | None


def _extract_geotiff_id(url: str) -> str | None:
    """Extract the ``id`` query parameter from a Solar GeoTIFF URL."""
    if not url:
        return None
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    ids = qs.get("id")
    return ids[0] if ids else None


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(value))


def _run_coro_sync(coro_factory):
    """Run an async coroutine from a sync context, even if the caller is
    inside a running event loop (e.g. a Jupyter kernel).

    ``asyncio.run`` and ``loop.run_until_complete`` both raise when invoked
    from a thread that already has a running loop, which is the default in
    Jupyter / IPython.  We detect that case and run the coroutine on a
    dedicated worker thread with its own fresh event loop.

    ``coro_factory`` is a zero-arg callable that constructs the coroutine.
    A factory (rather than a bare coroutine) is required because a coroutine
    object is bound to the loop in which it was created and cannot be
    awaited from another loop.
    """
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False

    if not running:
        return asyncio.run(coro_factory())

    # We're inside a running loop (Jupyter). Hand off to a worker thread.
    import concurrent.futures

    def _runner():
        return asyncio.run(coro_factory())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_runner).result()


async def _download_geotiff(async_client: solar_v1.SolarAsyncClient, url: str, out_path: Path) -> int:
    """Download one GeoTIFF via ``get_geo_tiff``; returns bytes written."""
    asset_id = _extract_geotiff_id(url)
    if asset_id is None:
        raise ValueError(f"could not extract id from URL: {url}")
    body = await async_client.get_geo_tiff(request=GetGeoTiffRequest(id=asset_id))
    # HttpBody.data is bytes.
    with out_path.open("wb") as fh:
        fh.write(body.data)
    return len(body.data)


async def _fetch_tile_async(
    lon: float,
    lat: float,
    *,
    required_quality: ImageryQuality,
    radius_m: int,
    view: DataLayerView,
    pixel_size_m: float,
    tile_id: str,
    bg_geoid: str,
    municipio: str,
    solar_root: Path,
    cache_dir: Path,
    ledger_path: Path,
    overwrite: bool,
    requested_layers: dict[str, str],
) -> FetchResult:
    # Note: do NOT mkdir the tile_dir here. A failed Data Layers call would
    # otherwise pollute data/rasters/solar/ with empty bg_geoid directories.
    # We create it only after a successful response, right before writing
    # GeoTIFFs.
    tile_dir = solar_root / _safe(municipio) / _safe(bg_geoid)

    # Short-circuit if every expected GeoTIFF already exists for this tile_id.
    if not overwrite and tile_dir.exists():
        expected = {short: tile_dir / f"{tile_id}_{short}_{required_quality}.tif" for short in requested_layers.values()}
        if all(p.exists() for p in expected.values()):
            return FetchResult(True, "ok", tile_dir, expected, required_quality, None, None, None)

    cache_key = _cache_key(
        "dataLayers", lon, lat,
        radius_m=radius_m, view=view, pixel_size_m=pixel_size_m, quality=required_quality,
    )
    cached = _load_cached_json(cache_key, cache_dir)

    if cached is None:
        # Enforce budget cap *before* the paid call.
        summary = ledger_summary(ledger_path)
        if summary["data_layers_successful_calls"] >= DATA_LAYERS_BUDGET_CAP:
            logger.error("Data Layers budget cap (%d) reached; refusing to call API.", DATA_LAYERS_BUDGET_CAP)
            _append_ledger(ledger_path, {
                "timestamp_utc": pd.Timestamp.utcnow().tz_localize(None).isoformat(),
                "endpoint": "dataLayers:get",
                "tile_id": tile_id, "bg_geoid": bg_geoid, "municipio": municipio,
                "lon": lon, "lat": lat, "radius_m": radius_m, "view": view,
                "required_quality": required_quality, "pixel_size_m": pixel_size_m,
                "status": "budget_exhausted", "cache_hit": False,
                "cost_estimate_usd": 0.0, "notes": f"cap={DATA_LAYERS_BUDGET_CAP}",
            })
            return FetchResult(False, "budget_exhausted", tile_dir, {}, None, None, None, None)

    async_client = solar_v1.SolarAsyncClient()

    # 1) Data Layers JSON envelope (cached or fresh).
    if cached is not None:
        payload = cached
        http_status = cached.get("_http_status", 200)
        cache_hit = True
    else:
        req = GetDataLayersRequest(
            location=LatLng(latitude=lat, longitude=lon),
            radius_meters=float(radius_m),
            view=_VIEW_ENUM[view],
            required_quality=_QUALITY_ENUM[required_quality],
            pixel_size_meters=float(pixel_size_m),
        )
        if required_quality == "BASE":
            req.experiments = [_ExperimentEnum.EXPANDED_COVERAGE]

        cache_hit = False
        # Retry transient 429/503 with exponential backoff before falling
        # through to the normal error path.  3 attempts, 1s/2s/4s.
        resp = None
        last_exc: GoogleAPICallError | None = None
        for attempt in range(3):
            try:
                resp = await async_client.get_data_layers(request=req)
                break
            except (ResourceExhausted, ServiceUnavailable) as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "transient %s on tile %s (attempt %d/3); sleeping %ds.",
                    type(exc).__name__, tile_id, attempt + 1, wait,
                )
                await asyncio.sleep(wait)
            except NotFound:
                payload = {"_status": "not_found", "_http_status": 404}
                _write_cached_json(cache_key, payload, cache_dir)
                _append_ledger(ledger_path, {
                    "timestamp_utc": pd.Timestamp.utcnow().tz_localize(None).isoformat(),
                    "endpoint": "dataLayers:get",
                    "tile_id": tile_id, "bg_geoid": bg_geoid, "municipio": municipio,
                    "lon": lon, "lat": lat, "radius_m": radius_m, "view": view,
                    "required_quality": required_quality, "pixel_size_m": pixel_size_m,
                    "status": "not_found", "cache_hit": False, "http_status": 404,
                    "cost_estimate_usd": 0.0, "notes": None,
                })
                return FetchResult(False, "not_found", tile_dir, {}, None, None, 404, None)
            except GoogleAPICallError as exc:
                body_text = str(exc)[:500]
                http_code = getattr(exc, "code", None)
                _append_ledger(ledger_path, {
                    "timestamp_utc": pd.Timestamp.utcnow().tz_localize(None).isoformat(),
                    "endpoint": "dataLayers:get",
                    "tile_id": tile_id, "bg_geoid": bg_geoid, "municipio": municipio,
                    "lon": lon, "lat": lat, "radius_m": radius_m, "view": view,
                    "required_quality": required_quality, "pixel_size_m": pixel_size_m,
                    "status": "error", "cache_hit": False, "http_status": http_code,
                    "cost_estimate_usd": 0.0, "notes": body_text[:180],
                })
                return FetchResult(False, "error", tile_dir, {}, None, None, http_code, {"body": body_text})

        if resp is None:
            # Exhausted retries on transient errors.
            body_text = str(last_exc)[:500] if last_exc else "transient retries exhausted"
            http_code = getattr(last_exc, "code", None) if last_exc else 429
            _append_ledger(ledger_path, {
                "timestamp_utc": pd.Timestamp.utcnow().tz_localize(None).isoformat(),
                "endpoint": "dataLayers:get",
                "tile_id": tile_id, "bg_geoid": bg_geoid, "municipio": municipio,
                "lon": lon, "lat": lat, "radius_m": radius_m, "view": view,
                "required_quality": required_quality, "pixel_size_m": pixel_size_m,
                "status": "error", "cache_hit": False, "http_status": http_code,
                "cost_estimate_usd": 0.0, "notes": body_text[:180],
            })
            return FetchResult(False, "error", tile_dir, {}, None, None, http_code, {"body": body_text})

        payload = {
            "rgb_url": resp.rgb_url,
            "mask_url": resp.mask_url,
            "annual_flux_url": resp.annual_flux_url,
            "dsm_url": resp.dsm_url,
            "monthly_flux_url": getattr(resp, "monthly_flux_url", "") or None,
            "imageryQuality": _quality_name(resp.imagery_quality),
            "imageryDate": _date_to_iso(resp.imagery_date),
            "imageryProcessedDate": _date_to_iso(resp.imagery_processed_date),
            "_status": "ok",
            "_http_status": 200,
        }
        _write_cached_json(cache_key, payload, cache_dir)
        http_status = 200

    returned_quality = payload.get("imageryQuality") or required_quality
    imagery_date = payload.get("imageryDate")

    # 2) Download each requested layer GeoTIFF in parallel via the async client.
    # We now know the response is valid, so materialize the tile directory.
    tile_dir.mkdir(parents=True, exist_ok=True)
    saved_files: dict[str, Path] = {}
    total_bytes = 0
    download_tasks: list[tuple[str, Path, asyncio.Task]] = []
    for response_field, short_name in requested_layers.items():
        url = payload.get(response_field)
        if not url:
            continue
        out_path = tile_dir / f"{tile_id}_{short_name}_{returned_quality}.tif"
        if out_path.exists() and not overwrite:
            saved_files[short_name] = out_path
            total_bytes += out_path.stat().st_size
            continue
        task = asyncio.create_task(_download_geotiff(async_client, url, out_path))
        download_tasks.append((short_name, out_path, task))

    for short_name, out_path, task in download_tasks:
        try:
            n = await task
        except (GoogleAPICallError, Exception) as exc:
            logger.warning("GeoTIFF download %s failed for tile %s: %s", short_name, tile_id, exc)
            # Remove partial file if created.
            if out_path.exists():
                try:
                    out_path.unlink()
                except OSError:
                    pass
            continue
        saved_files[short_name] = out_path
        total_bytes += n

    # 3) Sidecar metadata per tile.
    sidecar = tile_dir / f"{tile_id}_meta.json"
    sidecar_data = {
        "tile_id": tile_id, "bg_geoid": bg_geoid, "municipio": municipio,
        "lon": lon, "lat": lat,
        "radius_m": radius_m, "view": view,
        "required_quality": required_quality,
        "returned_quality": returned_quality,
        "pixel_size_m": pixel_size_m,
        "imagery_date": imagery_date,
        "imagery_processed_date": payload.get("imageryProcessedDate"),
        "cache_hit": cache_hit,
        "layers": {k: str(v) for k, v in saved_files.items()},
    }
    with sidecar.open("w", encoding="utf-8") as fh:
        json.dump(sidecar_data, fh, indent=2, sort_keys=True)

    _append_ledger(ledger_path, {
        "timestamp_utc": pd.Timestamp.utcnow().tz_localize(None).isoformat(),
        "endpoint": "dataLayers:get",
        "tile_id": tile_id, "bg_geoid": bg_geoid, "municipio": municipio,
        "lon": lon, "lat": lat, "radius_m": radius_m, "view": view,
        "required_quality": required_quality, "pixel_size_m": pixel_size_m,
        "status": "ok", "returned_quality": returned_quality,
        "imagery_date": imagery_date, "cache_hit": cache_hit,
        "http_status": http_status, "bytes": total_bytes,
        "cost_estimate_usd": 0.0 if cache_hit else DATA_LAYERS_UNIT_COST_USD,
        "notes": None,
    })

    return FetchResult(
        ok=True, status="ok", tile_dir=tile_dir, saved_files=saved_files,
        imagery_quality=returned_quality, imagery_date=imagery_date,
        http_status=http_status, raw=payload,
    )


def fetch_tile(
    lon: float,
    lat: float,
    *,
    required_quality: ImageryQuality,
    radius_m: int = 175,
    view: DataLayerView = DEFAULT_VIEW,
    pixel_size_m: float | None = None,
    tile_id: str,
    bg_geoid: str,
    municipio: str,
    solar_root: Path = DEFAULT_SOLAR_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    overwrite: bool = False,
    requested_layers: dict[str, str] | None = None,
) -> FetchResult:
    """Fetch Data Layers for a tile + download every layer GeoTIFF (sync wrapper).

    Default ``radius_m=175`` maximizes coverage per request; for HIGH-quality
    inputs the native 0.1 m pixel size is auto-bumped to 0.25 m to satisfy
    the Google constraint ``radius <= pixel_size * 1000`` (see
    ``enforce_radius_pixel_constraint``).
    """
    requested_layers = requested_layers or REQUESTED_LAYERS
    pixel_size_m = pixel_size_m or quality_to_pixel_size(required_quality)
    pixel_size_m = enforce_radius_pixel_constraint(radius_m, pixel_size_m)

    coro_factory = lambda: _fetch_tile_async(  # noqa: E731
        lon, lat,
        required_quality=required_quality, radius_m=radius_m, view=view,
        pixel_size_m=pixel_size_m,
        tile_id=tile_id, bg_geoid=bg_geoid, municipio=municipio,
        solar_root=solar_root, cache_dir=cache_dir, ledger_path=ledger_path,
        overwrite=overwrite, requested_layers=requested_layers,
    )
    return _run_coro_sync(coro_factory)


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def probe_points(
    points: Iterable[tuple[str, str, str, float, float]],
    *,
    required_quality: ImageryQuality | None = None,
    sleep_between_seconds: float = 0.02,
    **kwargs,
) -> pd.DataFrame:
    """Probe a batch of points sequentially via the sync client."""
    rows: list[dict] = []
    for tile_id, bg_geoid, municipio, lon, lat in points:
        result = probe_quality(
            lon, lat,
            required_quality=required_quality,
            tile_id=tile_id, bg_geoid=bg_geoid, municipio=municipio,
            **kwargs,
        )
        rows.append({
            "tile_id": tile_id, "bg_geoid": bg_geoid, "municipio": municipio,
            "lon": lon, "lat": lat,
            "status": result.status,
            "expected_quality": result.imagery_quality,
            "imagery_date": result.imagery_date,
            "http_status": result.http_status,
        })
        if sleep_between_seconds:
            time.sleep(sleep_between_seconds)
    return pd.DataFrame(rows)


def fetch_tiles_budgeted(
    manifest: pd.DataFrame,
    *,
    budget_cap: int = DATA_LAYERS_BUDGET_CAP,
    radius_m: int = 175,
    view: DataLayerView = DEFAULT_VIEW,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    sleep_between_seconds: float = 0.05,
    **kwargs,
) -> pd.DataFrame:
    """Iterate a prioritized tile manifest, fetching until budget is exhausted.

    The manifest must contain:
    ``tile_id, bg_geoid, municipio, lon, lat, required_quality`` and
    optionally ``priority_score`` and ``radius_m``.  If a per-tile
    ``radius_m`` column is present it overrides the ``radius_m`` argument.
    """
    required_cols = {"tile_id", "bg_geoid", "municipio", "lon", "lat", "required_quality"}
    missing = required_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")

    sort_cols = [c for c in ["priority_score", "tile_id"] if c in manifest.columns]
    ascending = [False] + [True] * (len(sort_cols) - 1)
    ordered = manifest.sort_values(sort_cols, ascending=ascending).reset_index(drop=True) if sort_cols else manifest

    results: list[dict] = []
    for _, row in ordered.iterrows():
        summary = ledger_summary(ledger_path)
        if summary["data_layers_successful_calls"] >= budget_cap:
            logger.warning("Budget cap %d reached; stopping.", budget_cap)
            break
        row_radius = int(row["radius_m"]) if ("radius_m" in manifest.columns and pd.notna(row.get("radius_m"))) else radius_m
        fr = fetch_tile(
            lon=float(row["lon"]), lat=float(row["lat"]),
            required_quality=str(row["required_quality"]),
            radius_m=row_radius, view=view,
            tile_id=str(row["tile_id"]),
            bg_geoid=str(row["bg_geoid"]),
            municipio=str(row["municipio"]),
            ledger_path=ledger_path,
            **kwargs,
        )
        results.append({
            "tile_id": row["tile_id"],
            "status": fr.status,
            "returned_quality": fr.imagery_quality,
            "imagery_date": fr.imagery_date,
            "num_layers": len(fr.saved_files),
            "http_status": fr.http_status,
            "note": None if fr.ok else (str(fr.raw)[:140] if fr.raw else None),
        })
        if sleep_between_seconds:
            time.sleep(sleep_between_seconds)
    return pd.DataFrame(results)


__all__ = [
    "DATA_LAYERS_BUDGET_CAP",
    "DATA_LAYERS_FREE_TIER_CALLS",
    "DATA_LAYERS_UNIT_COST_USD",
    "BUILDING_INSIGHTS_SOFT_CAP",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_LEDGER_PATH",
    "DEFAULT_SOLAR_ROOT",
    "DEFAULT_VIEW",
    "FetchResult",
    "ProbeResult",
    "REQUESTED_LAYERS",
    "enforce_radius_pixel_constraint",
    "fetch_tile",
    "fetch_tiles_budgeted",
    "ledger_summary",
    "probe_points",
    "probe_quality",
    "quality_to_pixel_size",
]
