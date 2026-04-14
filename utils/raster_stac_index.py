"""Shared helpers for Puerto Rico raster STAC indexing workflows."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urljoin

import aiofiles
import aiohttp
import duckdb
import geopandas as gpd
import osmnx as ox
import pandas as pd
import rustac
from dotenv import load_dotenv
from shapely import from_wkb
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep
from stac_geoparquet.arrow import parse_stac_ndjson_to_parquet


def resolve_project_root(start: Path | None = None) -> Path:
    """Find the repository root from an arbitrary working directory."""

    current = (start or Path.cwd()).resolve()
    markers = ("project_rules.md", ".git")
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return current


PROJECT_ROOT = resolve_project_root()
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

OUTPUT_CRS = "EPSG:4326"
MAX_CONCURRENT_REQUESTS = 24
MAX_HTTP_RETRIES = 3
HTTP_RETRY_BACKOFF_SECONDS = 1.5
TEMP_NDJSON_FILENAME = "temp_items.ndjson"
RUSTAC_PREVIEW_LIMIT = 3
RUSTAC_PREVIEW_TIMEOUT_SECONDS = 30
USER_AGENT = "PLAN6068-PR-Raster-Indexer/1.0"
EARTHVIEW_PUBLIC_PARQUET_URI = os.getenv(
    "EARTHVIEW_PUBLIC_PARQUET_URI",
    "s3://satellogic-earthview-stac-geoparquet/satellogic-earthview-stac-items.parquet",
)
MAXAR_PUBLIC_PARQUET_URI = os.getenv(
    "MAXAR_PUBLIC_PARQUET_URI",
    "https://data.source.coop/maxar/maxar-opendata/maxar-opendata.parquet",
)
EARTHVIEW_S3_REGION = os.getenv("EARTHVIEW_S3_REGION", "us-west-2")

NORMALIZED_INDEX_COLUMNS = [
    "source",
    "strategy",
    "item_id",
    "collection_id",
    "acquired_at",
    "stac_version",
    "item_type",
    "license",
    "platform",
    "constellation",
    "grid_code",
    "gsd",
    "proj_code",
    "proj_epsg",
    "cloud_cover",
    "view_off_nadir",
    "self_href",
    "primary_asset_name",
    "primary_asset_href",
    "visual_asset_name",
    "visual_asset_href",
    "analytic_asset_name",
    "analytic_asset_href",
    "preview_asset_name",
    "preview_asset_href",
    "thumbnail_asset_name",
    "thumbnail_asset_href",
    "asset_count",
    "available_asset_names",
    "bbox_minx",
    "bbox_miny",
    "bbox_maxx",
    "bbox_maxy",
    "geometry",
]

MANIFEST_COLUMNS = [
    "event_id",
    "event_title",
    "catalog_url",
    "extent_source",
    "intersects_puerto_rico",
    "child_catalog_count",
    "bbox_count",
    "status",
    "notes",
]

CONSOLIDATED_CATALOG_COLUMNS = [
    "source",
    "item_id",
    "collection_id",
    "acquired_at",
    "platform",
    "constellation",
    "gsd",
    "cloud_cover",
    "self_href",
    "visual_asset_href",
    "analytic_asset_href",
    "preview_asset_href",
    "thumbnail_asset_href",
    "asset_count",
    "bbox_minx",
    "bbox_miny",
    "bbox_maxx",
    "bbox_maxy",
    "geometry",
]


@dataclass(frozen=True)
class BoundaryContext:
    """Prepared Puerto Rico boundary and helper metadata."""

    geometry: BaseGeometry
    source: str
    bounds: tuple[float, float, float, float]
    envelope: BaseGeometry
    geojson: dict[str, Any]
    wkb: bytes
    prepared_geometry: Any


@dataclass(frozen=True)
class SourceArtifacts:
    """Output locations for one source."""

    stac_path: Path
    index_path: Path
    manifest_path: Path | None = None
    report_path: Path | None = None


@dataclass(frozen=True)
class SourceConfig:
    """Configuration for one source-specific indexing strategy."""

    source_name: str
    strategy: str
    artifacts: SourceArtifacts
    catalog_url: str | None = None
    remote_parquet_url: str | None = None
    follow_child_links: bool = True
    prune_catalogs_by_extent: bool = True
    filter_items_by_geometry: bool = True


@dataclass
class SourceRunSummary:
    """Compact execution summary for notebook reporting."""

    source_name: str
    strategy: str
    item_count: int
    stac_path: str | None
    index_path: str | None
    manifest_path: str | None = None
    report_path: str | None = None
    file_size_mb: float | None = None
    preview_bbox_count: int | None = None
    preview_intersects_count: int | None = None
    preview_arrow_rows: int | None = None
    sample_item_ids: list[str] | None = None
    notes: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a flat dictionary suitable for DataFrame construction."""

        return asdict(self)


def _to_bytes(value: object) -> bytes:
    """Normalize DuckDB binary values to plain bytes."""

    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    return bytes(value)


def _json_default(value: object) -> Any:
    """Convert common geospatial/Pandas objects to JSON-safe values."""

    if value is None:
        return None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, BaseGeometry):
        return mapping(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return value.hex()
    if isinstance(value, Mapping):
        return {str(key): _json_default(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, str):
        try:
            return _json_default(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _json_default(value.item())
        except Exception:
            pass
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _clean_string(value: object) -> str | None:
    """Return a stripped string or None for empty-like values."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_sequence(value: object) -> list[Any]:
    """Normalize arrays, tuples, and JSON arrays to Python lists."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except Exception:
            return []
        return loaded if isinstance(loaded, list) else []
    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray, memoryview)):
        try:
            converted = value.tolist()
            return converted if isinstance(converted, list) else []
        except Exception:
            return []
    return []


def _coerce_mapping(value: object) -> dict[str, Any]:
    """Normalize dict-like or JSON-string values to a Python dict."""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except Exception:
            return {}
        if isinstance(loaded, Mapping):
            return {str(key): item for key, item in loaded.items()}
        return {}
    return {}


def _coerce_float(value: object) -> float | None:
    """Convert scalar-like values to float when possible."""

    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _coerce_int(value: object) -> int | None:
    """Convert scalar-like values to int when possible."""

    numeric = _coerce_float(value)
    if numeric is None:
        return None
    try:
        return int(numeric)
    except Exception:
        return None


def _coerce_timestamp(value: object) -> pd.Timestamp | None:
    """Convert STAC datetime-like values to UTC-normalized timestamps."""

    if value is None:
        return None
    try:
        timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(timestamp):
        return None
    if isinstance(timestamp, pd.DatetimeIndex):
        if len(timestamp) == 0:
            return None
        timestamp = timestamp[0]
    return timestamp.tz_convert("UTC")


def _coerce_geometry(value: object) -> BaseGeometry | None:
    """Normalize GeoParquet/STAC geometry values to Shapely geometry."""

    if value is None:
        return None
    if isinstance(value, BaseGeometry):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return from_wkb(_to_bytes(value))
        except Exception:
            return None
    if isinstance(value, Mapping):
        try:
            return shape(value)
        except Exception:
            return None
    return None


def _coerce_bbox(value: object) -> tuple[float, float, float, float] | None:
    """Normalize STAC bbox values stored as list or struct-like objects."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        if {"xmin", "ymin", "xmax", "ymax"}.issubset(value.keys()):
            return (
                float(value["xmin"]),
                float(value["ymin"]),
                float(value["xmax"]),
                float(value["ymax"]),
            )
        if {"minx", "miny", "maxx", "maxy"}.issubset(value.keys()):
            return (
                float(value["minx"]),
                float(value["miny"]),
                float(value["maxx"]),
                float(value["maxy"]),
            )
    sequence = _coerce_sequence(value)
    if len(sequence) >= 4:
        try:
            return (
                float(sequence[0]),
                float(sequence[1]),
                float(sequence[2]),
                float(sequence[3]),
            )
        except Exception:
            return None
    return None


def _extract_value(row: Mapping[str, Any], *candidates: str) -> Any:
    """Extract a value from a row or nested properties dict using fallback keys."""

    properties = _coerce_mapping(row.get("properties"))
    for candidate in candidates:
        if candidate in row and row[candidate] is not None:
            return row[candidate]
        if candidate in properties and properties[candidate] is not None:
            return properties[candidate]
    return None


def _extract_self_href(row: Mapping[str, Any]) -> str | None:
    """Extract the STAC self href from a GeoParquet row."""

    direct = _clean_string(_extract_value(row, "self_href", "href"))
    if direct is not None:
        return direct

    for link in _coerce_sequence(row.get("links")):
        link_map = _coerce_mapping(link)
        if _clean_string(link_map.get("rel")) == "self":
            return _clean_string(link_map.get("href"))
    return None


def _pick_asset(
    assets: Mapping[str, Any],
    exact_names: Sequence[str] = (),
    role_hints: Sequence[str] = (),
    keyword_hints: Sequence[str] = (),
    disallow_keywords: Sequence[str] = (),
    fallback_first: bool = False,
) -> tuple[str | None, str | None]:
    """Pick a preferred asset name/href pair from a STAC assets dict."""

    normalized_assets = {str(name): _coerce_mapping(asset) for name, asset in assets.items()}
    if not normalized_assets:
        return None, None

    for preferred_name in exact_names:
        asset = normalized_assets.get(preferred_name)
        href = _clean_string(asset.get("href")) if asset else None
        if href is not None:
            return preferred_name, href

    for name, asset in normalized_assets.items():
        name_lower = name.casefold()
        if any(blocked in name_lower for blocked in disallow_keywords):
            continue
        roles = [str(role).casefold() for role in _coerce_sequence(asset.get("roles"))]
        if role_hints and any(role_hint in roles for role_hint in role_hints):
            href = _clean_string(asset.get("href"))
            if href is not None:
                return name, href

    for name, asset in normalized_assets.items():
        name_lower = name.casefold()
        title_lower = _clean_string(asset.get("title"))
        href_lower = _clean_string(asset.get("href"))
        haystacks = [name_lower]
        if title_lower is not None:
            haystacks.append(title_lower.casefold())
        if href_lower is not None:
            haystacks.append(href_lower.casefold())
        if any(blocked in name_lower for blocked in disallow_keywords):
            continue
        if keyword_hints and any(keyword in haystack for keyword in keyword_hints for haystack in haystacks):
            href = _clean_string(asset.get("href"))
            if href is not None:
                return name, href

    if fallback_first:
        for name, asset in normalized_assets.items():
            name_lower = name.casefold()
            if any(blocked in name_lower for blocked in disallow_keywords):
                continue
            href = _clean_string(asset.get("href"))
            if href is not None:
                return name, href

    return None, None


def _available_asset_names(assets: Mapping[str, Any]) -> str:
    """Serialize available asset names for quick downstream inspection."""

    return json.dumps(sorted(str(name) for name in assets.keys()), ensure_ascii=True)


def _normalize_stac_row(row: Mapping[str, Any], source_name: str, strategy: str) -> dict[str, Any] | None:
    """Flatten one STAC GeoParquet row into a source-agnostic index row."""

    geometry = _coerce_geometry(row.get("geometry"))
    bbox = _coerce_bbox(row.get("bbox"))
    if geometry is None and bbox is not None:
        geometry = box(*bbox)
    if geometry is None or geometry.is_empty:
        return None

    if bbox is None:
        bbox = tuple(float(value) for value in geometry.bounds)

    assets = _coerce_mapping(row.get("assets"))
    primary_name, primary_href = _pick_asset(
        assets,
        role_hints=("data",),
        keyword_hints=("visual", "analytic", "image", "rgb"),
        disallow_keywords=("thumbnail", "preview"),
        fallback_first=True,
    )
    visual_name, visual_href = _pick_asset(
        assets,
        exact_names=("visual", "image"),
        role_hints=("visual",),
        keyword_hints=("visual", "rgb"),
        disallow_keywords=("thumbnail", "preview", "analytic"),
        fallback_first=primary_href is None,
    )
    analytic_name, analytic_href = _pick_asset(
        assets,
        exact_names=("analytic", "data"),
        role_hints=("data",),
        keyword_hints=("analytic", "cog", "ortho", "ms", "nir"),
        disallow_keywords=("thumbnail", "preview", "visual"),
        fallback_first=False,
    )
    preview_name, preview_href = _pick_asset(
        assets,
        exact_names=("preview",),
        keyword_hints=("preview", "overview"),
        fallback_first=False,
    )
    thumbnail_name, thumbnail_href = _pick_asset(
        assets,
        exact_names=("thumbnail",),
        keyword_hints=("thumbnail", "thumb"),
        fallback_first=False,
    )

    proj_code = _clean_string(_extract_value(row, "proj:code"))
    proj_epsg = _coerce_int(_extract_value(row, "proj:epsg", "epsg"))
    if proj_epsg is None and proj_code is not None and proj_code.upper().startswith("EPSG:"):
        proj_epsg = _coerce_int(proj_code.split(":", 1)[1])

    return {
        "source": source_name,
        "strategy": strategy,
        "item_id": _clean_string(_extract_value(row, "id")),
        "collection_id": _clean_string(_extract_value(row, "collection")),
        "acquired_at": _coerce_timestamp(_extract_value(row, "datetime", "start_datetime")),
        "stac_version": _clean_string(_extract_value(row, "stac_version")),
        "item_type": _clean_string(_extract_value(row, "type")) or "Feature",
        "license": _clean_string(_extract_value(row, "license")),
        "platform": _clean_string(_extract_value(row, "platform")),
        "constellation": _clean_string(_extract_value(row, "constellation")),
        "grid_code": _clean_string(_extract_value(row, "grid:code")),
        "gsd": _coerce_float(_extract_value(row, "gsd", "eo:gsd", "proj:gsd")),
        "proj_code": proj_code,
        "proj_epsg": proj_epsg,
        "cloud_cover": _coerce_float(_extract_value(row, "eo:cloud_cover", "cloud_cover")),
        "view_off_nadir": _coerce_float(_extract_value(row, "view:off_nadir", "sat:off_nadir")),
        "self_href": _extract_self_href(row),
        "primary_asset_name": primary_name,
        "primary_asset_href": primary_href,
        "visual_asset_name": visual_name or primary_name,
        "visual_asset_href": visual_href or primary_href,
        "analytic_asset_name": analytic_name,
        "analytic_asset_href": analytic_href,
        "preview_asset_name": preview_name,
        "preview_asset_href": preview_href,
        "thumbnail_asset_name": thumbnail_name,
        "thumbnail_asset_href": thumbnail_href,
        "asset_count": len(assets),
        "available_asset_names": _available_asset_names(assets),
        "bbox_minx": bbox[0],
        "bbox_miny": bbox[1],
        "bbox_maxx": bbox[2],
        "bbox_maxy": bbox[3],
        "geometry": geometry,
    }


def _empty_normalized_index() -> gpd.GeoDataFrame:
    """Return an empty normalized index GeoDataFrame with stable columns."""

    frame = pd.DataFrame(columns=[column for column in NORMALIZED_INDEX_COLUMNS if column != "geometry"])
    geometry = gpd.GeoSeries([], crs=OUTPUT_CRS)
    return gpd.GeoDataFrame(frame, geometry=geometry, crs=OUTPUT_CRS)


def _write_normalized_index_from_stac_parquet(
    stac_path: Path,
    output_path: Path,
    source_name: str,
    strategy: str,
) -> int:
    """Read a STAC GeoParquet file and write the flattened source index."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not stac_path.exists():
        _empty_normalized_index().to_parquet(output_path, index=False, compression="zstd")
        return 0

    stac_gdf = gpd.read_parquet(stac_path)
    if stac_gdf.crs is None:
        stac_gdf = stac_gdf.set_crs(OUTPUT_CRS)
    else:
        stac_gdf = stac_gdf.to_crs(OUTPUT_CRS)

    records = []
    for row in stac_gdf.to_dict(orient="records"):
        normalized = _normalize_stac_row(row, source_name=source_name, strategy=strategy)
        if normalized is not None:
            records.append(normalized)

    if not records:
        normalized_gdf = _empty_normalized_index()
    else:
        normalized_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=OUTPUT_CRS)
        normalized_gdf = normalized_gdf[NORMALIZED_INDEX_COLUMNS]

    normalized_gdf.to_parquet(output_path, index=False, compression="zstd")
    return len(normalized_gdf)


def _write_json_report(report: Mapping[str, Any], output_path: Path) -> None:
    """Persist a JSON report with safe serialization defaults."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _write_manifest(manifest_rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    """Persist the Maxar event manifest as Parquet."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_frame = pd.DataFrame(manifest_rows)
    if manifest_frame.empty:
        manifest_frame = pd.DataFrame(columns=MANIFEST_COLUMNS)
    else:
        manifest_frame = manifest_frame.reindex(columns=MANIFEST_COLUMNS)
    manifest_frame.to_parquet(output_path, index=False)


def resolve_vector_db_path() -> Path:
    """Resolve the project DuckDB file with conservative fallbacks."""

    db_path_value = os.getenv("VECTOR_DB")
    if db_path_value:
        db_path = Path(db_path_value)
        if not db_path.is_absolute():
            if len(db_path.parts) > 1:
                return PROJECT_ROOT / db_path
            return PROJECT_ROOT / "data" / "vectors" / db_path
        return db_path

    preferred = PROJECT_ROOT / "data" / "vectors" / "pv_database.db"
    if preferred.exists():
        return preferred

    return PROJECT_ROOT / "data" / "vectors" / "PR_vector_data.duckdb"


def create_duckdb_connection(
    db_path: Path | None = None,
    load_httpfs: bool = False,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Create DuckDB connection with spatial support and optional remote I/O."""

    connect_target = ":memory:" if db_path is None else str(db_path)
    if db_path is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(connect_target, read_only=read_only and db_path is not None)
    try:
        con.execute("LOAD spatial;")
    except duckdb.Error:
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")

    if load_httpfs:
        try:
            con.execute("LOAD httpfs;")
        except duckdb.Error:
            con.execute("INSTALL httpfs;")
            con.execute("LOAD httpfs;")
        con.execute(f"SET s3_region='{EARTHVIEW_S3_REGION}';")
        con.execute("SET s3_url_style='path';")

    return con


def duckdb_table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """Return True when a main-schema DuckDB table exists."""

    row = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?;
        """,
        [table_name],
    ).fetchone()
    return bool(row and row[0])


def load_puerto_rico_boundary() -> BoundaryContext:
    """Load the Puerto Rico boundary from DuckDB first, then OSMnx fallback."""

    db_path = resolve_vector_db_path()
    if db_path.exists():
        con: duckdb.DuckDBPyConnection | None = None
        try:
            con = create_duckdb_connection(db_path=db_path, read_only=True)
            if duckdb_table_exists(con, "pr_municipalities"):
                row = con.execute(
                    """
                    SELECT ST_AsWKB(ST_Union_Agg(geometry)) AS geometry_wkb
                    FROM pr_municipalities
                    WHERE geometry IS NOT NULL;
                    """
                ).fetchone()
                if row and row[0] is not None:
                    geometry = from_wkb(_to_bytes(row[0]))
                    if geometry is not None and not geometry.is_empty:
                        bounds = tuple(float(value) for value in geometry.bounds)
                        return BoundaryContext(
                            geometry=geometry,
                            source=f"duckdb:{db_path.name}",
                            bounds=bounds,
                            envelope=box(*bounds),
                            geojson=mapping(geometry),
                            wkb=geometry.wkb,
                            prepared_geometry=prep(geometry),
                        )
        except Exception:
            pass
        finally:
            if con is not None:
                con.close()

    geometry = ox.geocode_to_gdf("Puerto Rico").geometry.union_all()
    bounds = tuple(float(value) for value in geometry.bounds)
    return BoundaryContext(
        geometry=geometry,
        source="osmnx",
        bounds=bounds,
        envelope=box(*bounds),
        geojson=mapping(geometry),
        wkb=geometry.wkb,
        prepared_geometry=prep(geometry),
    )


def resolve_href(base_url: str, href: str) -> str:
    """Resolve a potentially relative STAC href against its parent URL."""

    if href.startswith(("http://", "https://", "s3://")):
        return href
    return urljoin(base_url, href)


def catalog_might_intersect_boundary(catalog_dict: Mapping[str, Any], boundary: BoundaryContext) -> bool:
    """Return True when a catalog or collection extent might intersect PR."""

    try:
        bboxes = catalog_dict.get("extent", {}).get("spatial", {}).get("bbox", [])
        if not bboxes:
            return True
        return any(
            len(current_bbox) >= 4 and box(*current_bbox[:4]).intersects(boundary.envelope)
            for current_bbox in bboxes
        )
    except Exception:
        return True


def item_intersects_boundary(item_dict: Mapping[str, Any], boundary: BoundaryContext) -> bool:
    """Return True when a STAC item intersects Puerto Rico."""

    geometry = _coerce_geometry(item_dict.get("geometry"))
    if geometry is None:
        bbox = _coerce_bbox(item_dict.get("bbox"))
        if bbox is not None:
            geometry = box(*bbox)

    if geometry is None or geometry.is_empty:
        return False
    if not geometry.intersects(boundary.envelope):
        return False
    return bool(boundary.prepared_geometry.intersects(geometry))


def make_work_dir(output_path: Path) -> Path:
    """Create a deterministic work directory for one source run."""

    work_dir = output_path.parent / "_tmp_stac" / output_path.stem
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    log: logging.Logger,
) -> dict[str, Any] | None:
    """Fetch JSON with bounded concurrency and basic retry handling."""

    async with semaphore:
        for attempt in range(1, MAX_HTTP_RETRIES + 1):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status != 200:
                        log.warning("HTTP %d for %s", response.status, url)
                        return None
                    return await response.json(content_type=None)
            except Exception as exc:
                if attempt == MAX_HTTP_RETRIES:
                    log.warning("Fetch failed for %s after %d attempts: %s", url, attempt, exc)
                    return None
                await asyncio.sleep(HTTP_RETRY_BACKOFF_SECONDS * attempt)
    return None


async def crawl_catalog_roots_to_ndjson(
    session: aiohttp.ClientSession,
    root_catalog_urls: Sequence[str],
    ndjson_path: Path,
    boundary: BoundaryContext,
    follow_child_links: bool,
    prune_catalogs_by_extent: bool,
    filter_items_by_geometry: bool,
    log: logging.Logger,
) -> int:
    """Breadth-first crawl one or more static STAC roots into NDJSON."""

    item_count = 0
    visited_catalogs: set[str] = set()
    pending_catalog_urls: list[str] = [str(url) for url in root_catalog_urls]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiofiles.open(ndjson_path, mode="w", encoding="utf-8") as ndjson_file:
        while pending_catalog_urls:
            batch_urls = [url for url in pending_catalog_urls if url not in visited_catalogs]
            pending_catalog_urls = []
            if not batch_urls:
                break

            visited_catalogs.update(batch_urls)
            catalog_payloads = await asyncio.gather(
                *[fetch_json(session, url, semaphore, log) for url in batch_urls]
            )

            item_urls: list[str] = []
            for catalog_dict, base_url in zip(catalog_payloads, batch_urls):
                if catalog_dict is None:
                    continue
                if prune_catalogs_by_extent and not catalog_might_intersect_boundary(catalog_dict, boundary):
                    continue

                for link in catalog_dict.get("links", []):
                    rel = link.get("rel")
                    href = link.get("href")
                    if not rel or not href:
                        continue
                    resolved_href = resolve_href(base_url, href)
                    if rel == "item":
                        item_urls.append(resolved_href)
                    elif rel == "child" and follow_child_links:
                        pending_catalog_urls.append(resolved_href)

            if not item_urls:
                continue

            item_payloads = await asyncio.gather(
                *[fetch_json(session, item_url, semaphore, log) for item_url in item_urls]
            )
            for item_dict in item_payloads:
                if item_dict is None:
                    continue
                if filter_items_by_geometry and not item_intersects_boundary(item_dict, boundary):
                    continue

                await ndjson_file.write(json.dumps(item_dict, separators=(",", ":")) + "\n")
                item_count += 1
                if item_count % 250 == 0:
                    log.info("Wrote %d qualifying items to %s", item_count, ndjson_path.name)

    return item_count


async def preview_output_with_rustac(
    output_path: Path,
    boundary: BoundaryContext,
    limit: int = RUSTAC_PREVIEW_LIMIT,
) -> dict[str, Any]:
    """Run lightweight rustac searches against a generated STAC GeoParquet file."""

    log = logging.getLogger("raster_stac.preview")
    try:
        bbox_preview = await asyncio.wait_for(
            rustac.search(output_path.as_posix(), bbox=list(boundary.bounds), limit=limit),
            timeout=RUSTAC_PREVIEW_TIMEOUT_SECONDS,
        )
        intersects_preview = await asyncio.wait_for(
            rustac.search(
                output_path.as_posix(),
                intersects=boundary.geojson,
                limit=limit,
            ),
            timeout=RUSTAC_PREVIEW_TIMEOUT_SECONDS,
        )
        duckdb_client = rustac.DuckdbClient()
        arrow_preview = await asyncio.wait_for(
            asyncio.to_thread(
                duckdb_client.search_to_arrow,
                output_path.as_posix(),
                intersects=boundary.geojson,
                limit=limit,
            ),
            timeout=RUSTAC_PREVIEW_TIMEOUT_SECONDS,
        )
        return {
            "bbox_preview_count": len(bbox_preview),
            "intersects_preview_count": len(intersects_preview),
            "arrow_preview_rows": int(arrow_preview.num_rows),
            "sample_item_ids": [item.get("id") for item in intersects_preview],
        }
    except Exception as exc:
        log.warning(
            "rustac preview failed for %s (%s: %s). Falling back to DuckDB spatial preview.",
            output_path.name,
            type(exc).__name__,
            exc,
        )
        try:
            return _preview_output_with_duckdb(output_path, boundary, limit=limit)
        except Exception as fallback_exc:
            log.warning(
                "DuckDB preview failed for %s (%s: %s). Returning empty preview.",
                output_path.name,
                type(fallback_exc).__name__,
                fallback_exc,
            )
            return {
                "bbox_preview_count": 0,
                "intersects_preview_count": 0,
                "arrow_preview_rows": 0,
                "sample_item_ids": [],
            }


def _preview_output_with_duckdb(
    output_path: Path,
    boundary: BoundaryContext,
    limit: int = RUSTAC_PREVIEW_LIMIT,
) -> dict[str, Any]:
    """Run preview queries directly in DuckDB when rustac cannot read the file."""

    parquet_uri = output_path.as_posix()
    con = create_duckdb_connection(load_httpfs=False)
    try:
        geometry_expr = _parquet_geometry_expression(con, parquet_uri, context=output_path.name)
        bbox_preview = con.execute(
            f"""
            SELECT id
            FROM read_parquet(?)
            WHERE ST_Intersects({geometry_expr}, ST_GeomFromText(?))
            LIMIT ?;
            """,
            [parquet_uri, boundary.envelope.wkt, limit],
        ).fetchdf()
        intersects_preview = con.execute(
            f"""
            SELECT id
            FROM read_parquet(?)
            WHERE ST_Intersects({geometry_expr}, ST_GeomFromText(?))
            LIMIT ?;
            """,
            [parquet_uri, boundary.geometry.wkt, limit],
        ).fetchdf()
        arrow_preview = con.execute(
            f"""
            SELECT *
            FROM read_parquet(?)
            WHERE ST_Intersects({geometry_expr}, ST_GeomFromText(?))
            LIMIT ?;
            """,
            [parquet_uri, boundary.geometry.wkt, limit],
        ).fetchdf()
        return {
            "bbox_preview_count": int(len(bbox_preview)),
            "intersects_preview_count": int(len(intersects_preview)),
            "arrow_preview_rows": int(len(arrow_preview)),
            "sample_item_ids": intersects_preview["id"].dropna().astype(str).tolist(),
        }
    finally:
        con.close()


async def _process_static_catalog_source(
    config: SourceConfig,
    boundary: BoundaryContext,
    root_catalog_urls: Sequence[str],
    log: logging.Logger,
) -> SourceRunSummary:
    """Crawl static STAC roots, write STAC GeoParquet, and flatten an index."""

    stac_path = config.artifacts.stac_path
    index_path = config.artifacts.index_path
    stac_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = make_work_dir(stac_path)
    ndjson_path = work_dir / TEMP_NDJSON_FILENAME
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS, limit_per_host=10)

    try:
        async with aiohttp.ClientSession(
            connector=connector,
            headers={"User-Agent": USER_AGENT},
        ) as session:
            item_count = await crawl_catalog_roots_to_ndjson(
                session=session,
                root_catalog_urls=root_catalog_urls,
                ndjson_path=ndjson_path,
                boundary=boundary,
                follow_child_links=config.follow_child_links,
                prune_catalogs_by_extent=config.prune_catalogs_by_extent,
                filter_items_by_geometry=config.filter_items_by_geometry,
                log=log,
            )

        if stac_path.exists():
            stac_path.unlink()

        if item_count == 0:
            _empty_normalized_index().to_parquet(index_path, index=False, compression="zstd")
            return SourceRunSummary(
                source_name=config.source_name,
                strategy=config.strategy,
                item_count=0,
                stac_path=None,
                index_path=index_path.as_posix(),
                notes="No Puerto Rico-intersecting items were materialized.",
            )

        parse_stac_ndjson_to_parquet(input_path=ndjson_path, output_path=stac_path)
        preview = await preview_output_with_rustac(stac_path, boundary)
        normalized_rows = _write_normalized_index_from_stac_parquet(
            stac_path,
            index_path,
            source_name=config.source_name,
            strategy=config.strategy,
        )
        file_size_mb = stac_path.stat().st_size / (1024 * 1024)
        return SourceRunSummary(
            source_name=config.source_name,
            strategy=config.strategy,
            item_count=item_count,
            stac_path=stac_path.as_posix(),
            index_path=index_path.as_posix(),
            file_size_mb=file_size_mb,
            preview_bbox_count=preview["bbox_preview_count"],
            preview_intersects_count=preview["intersects_preview_count"],
            preview_arrow_rows=preview["arrow_preview_rows"],
            sample_item_ids=preview["sample_item_ids"],
            notes=f"normalized_rows={normalized_rows}",
        )
    finally:
        ndjson_path.unlink(missing_ok=True)
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


async def index_flat_static_source(config: SourceConfig, boundary: BoundaryContext) -> SourceRunSummary:
    """Index a flat or shallow static STAC catalog such as Puerto Rico NAIP."""

    if config.catalog_url is None:
        raise ValueError("catalog_url is required for flat static catalog indexing.")

    log = logging.getLogger(f"raster_stac.{config.source_name}")
    return await _process_static_catalog_source(
        config=config,
        boundary=boundary,
        root_catalog_urls=[config.catalog_url],
        log=log,
    )


async def _inspect_catalog_relevance(
    session: aiohttp.ClientSession,
    catalog_url: str,
    boundary: BoundaryContext,
    semaphore: asyncio.Semaphore,
    log: logging.Logger,
) -> dict[str, Any]:
    """Inspect one event catalog and decide whether it merits a targeted crawl."""

    catalog_dict = await fetch_json(session, catalog_url, semaphore, log)
    if catalog_dict is None:
        return {
            "catalog_url": catalog_url,
            "event_id": None,
            "event_title": None,
            "extent_source": "fetch_failed",
            "intersects_puerto_rico": False,
            "child_catalog_count": 0,
            "bbox_count": 0,
            "status": "fetch_failed",
            "notes": "Unable to fetch event catalog.",
        }

    event_id = _clean_string(catalog_dict.get("id"))
    event_title = _clean_string(catalog_dict.get("title")) or event_id
    bboxes = catalog_dict.get("extent", {}).get("spatial", {}).get("bbox", [])
    child_links = [
        resolve_href(catalog_url, link.get("href", ""))
        for link in catalog_dict.get("links", [])
        if link.get("rel") == "child" and link.get("href")
    ]

    if bboxes:
        return {
            "catalog_url": catalog_url,
            "event_id": event_id,
            "event_title": event_title,
            "extent_source": "event_catalog",
            "intersects_puerto_rico": catalog_might_intersect_boundary(catalog_dict, boundary),
            "child_catalog_count": len(child_links),
            "bbox_count": len(bboxes),
            "status": "evaluated",
            "notes": "Event extent evaluated directly.",
        }

    if not child_links:
        return {
            "catalog_url": catalog_url,
            "event_id": event_id,
            "event_title": event_title,
            "extent_source": "missing",
            "intersects_puerto_rico": False,
            "child_catalog_count": 0,
            "bbox_count": 0,
            "status": "skipped_missing_extent",
            "notes": "No event extent and no child catalogs to inspect.",
        }

    child_payloads = await asyncio.gather(
        *[fetch_json(session, url, semaphore, log) for url in child_links]
    )
    child_hits = [
        payload for payload in child_payloads if payload is not None and catalog_might_intersect_boundary(payload, boundary)
    ]
    child_bbox_count = sum(
        len(payload.get("extent", {}).get("spatial", {}).get("bbox", []))
        for payload in child_payloads
        if payload is not None
    )

    return {
        "catalog_url": catalog_url,
        "event_id": event_id,
        "event_title": event_title,
        "extent_source": "child_catalog",
        "intersects_puerto_rico": bool(child_hits),
        "child_catalog_count": len(child_links),
        "bbox_count": child_bbox_count,
        "status": "evaluated",
        "notes": "Inspected one level deeper because event extent was missing.",
    }


async def index_event_static_source(config: SourceConfig, boundary: BoundaryContext) -> SourceRunSummary:
    """Build a Maxar-style event manifest, then crawl only shortlisted events."""

    if config.catalog_url is None:
        raise ValueError("catalog_url is required for event static catalog indexing.")
    if config.artifacts.manifest_path is None or config.artifacts.report_path is None:
        raise ValueError("manifest_path and report_path are required for event static catalog indexing.")

    log = logging.getLogger(f"raster_stac.{config.source_name}")
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS, limit_per_host=10)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession(
        connector=connector,
        headers={"User-Agent": USER_AGENT},
    ) as session:
        root_catalog = await fetch_json(session, config.catalog_url, semaphore, log)
        if root_catalog is None:
            manifest_rows = [
                {
                    "event_id": None,
                    "event_title": None,
                    "catalog_url": config.catalog_url,
                    "extent_source": "root_catalog",
                    "intersects_puerto_rico": False,
                    "child_catalog_count": 0,
                    "bbox_count": 0,
                    "status": "root_fetch_failed",
                    "notes": "Unable to fetch the Maxar events root catalog.",
                }
            ]
            shortlisted_urls: list[str] = []
        else:
            event_urls = [
                resolve_href(config.catalog_url, link.get("href", ""))
                for link in root_catalog.get("links", [])
                if link.get("rel") == "child" and link.get("href")
            ]
            manifest_rows = await asyncio.gather(
                *[
                    _inspect_catalog_relevance(
                        session=session,
                        catalog_url=event_url,
                        boundary=boundary,
                        semaphore=semaphore,
                        log=log,
                    )
                    for event_url in event_urls
                ]
            )
            shortlisted_urls = [
                row["catalog_url"]
                for row in manifest_rows
                if bool(row.get("intersects_puerto_rico"))
            ]

    _write_manifest(manifest_rows, config.artifacts.manifest_path)
    report = {
        "source": config.source_name,
        "strategy": config.strategy,
        "root_catalog_url": config.catalog_url,
        "manifest_path": config.artifacts.manifest_path,
        "shortlisted_event_count": len(shortlisted_urls),
        "evaluated_event_count": len(manifest_rows),
        "shortlisted_event_urls": shortlisted_urls,
    }
    _write_json_report(report, config.artifacts.report_path)

    if config.artifacts.stac_path.exists():
        config.artifacts.stac_path.unlink()

    if not shortlisted_urls:
        _empty_normalized_index().to_parquet(config.artifacts.index_path, index=False, compression="zstd")
        return SourceRunSummary(
            source_name=config.source_name,
            strategy=config.strategy,
            item_count=0,
            stac_path=None,
            index_path=config.artifacts.index_path.as_posix(),
            manifest_path=config.artifacts.manifest_path.as_posix(),
            report_path=config.artifacts.report_path.as_posix(),
            notes="No Puerto Rico-intersecting event catalogs were shortlisted.",
        )

    summary = await _process_static_catalog_source(
        config=config,
        boundary=boundary,
        root_catalog_urls=shortlisted_urls,
        log=log,
    )
    summary.manifest_path = config.artifacts.manifest_path.as_posix()
    summary.report_path = config.artifacts.report_path.as_posix()
    summary.notes = (
        f"shortlisted_events={len(shortlisted_urls)}; {summary.notes}"
        if summary.notes is not None
        else f"shortlisted_events={len(shortlisted_urls)}"
    )
    return summary


def _serialize_remote_row_to_item(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a remote GeoParquet row back into a STAC item dict."""

    item: dict[str, Any] = {}
    for key, value in row.items():
        if key == "geometry":
            geometry = _coerce_geometry(value)
            item[key] = mapping(geometry) if geometry is not None else None
            continue
        if key == "bbox":
            bbox = _coerce_bbox(value)
            item[key] = list(bbox) if bbox is not None else None
            continue
        if isinstance(value, pd.Timestamp):
            item[key] = value.isoformat().replace("+00:00", "Z")
            continue
        item[key] = _json_default(value)

    if item.get("type") is None:
        item["type"] = "Feature"
    return item


def _sql_quote(value: str) -> str:
    """Quote a Python string for safe inline SQL usage."""

    return "'" + str(value).replace("'", "''") + "'"


def _parquet_geometry_expression(
    con: duckdb.DuckDBPyConnection,
    parquet_uri: str,
    context: str = "parquet",
) -> str:
    """Resolve the geometry expression required to spatially query a parquet file."""

    parquet_uri_sql = _sql_quote(parquet_uri)
    geometry_type = con.execute(
        f"SELECT typeof(geometry) FROM read_parquet({parquet_uri_sql}) LIMIT 1;"
    ).fetchone()
    if not geometry_type or geometry_type[0] is None:
        raise RuntimeError(f"Unable to infer geometry type from {context}.")
    type_name = str(geometry_type[0]).upper()
    if "GEOMETRY" in type_name:
        return "geometry"
    return "ST_GeomFromWKB(geometry)"


def _query_remote_earthview_rows(
    remote_uri: str,
    boundary: BoundaryContext,
) -> list[dict[str, Any]]:
    """Query the public Earthview GeoParquet mirror for Puerto Rico rows."""

    con = create_duckdb_connection(load_httpfs=True)
    try:
        remote_uri_sql = _sql_quote(remote_uri)
        geometry_expr = _parquet_geometry_expression(con, remote_uri, context="remote Earthview parquet")
        boundary_minx, boundary_miny, boundary_maxx, boundary_maxy = boundary.bounds
        query = f"""
        WITH source AS (
            SELECT
                *,
                {geometry_expr} AS geometry_native
            FROM read_parquet({remote_uri_sql})
            WHERE bbox.xmin <= {boundary_maxx}
              AND bbox.xmax >= {boundary_minx}
              AND bbox.ymin <= {boundary_maxy}
              AND bbox.ymax >= {boundary_miny}
        )
        SELECT * EXCLUDE (geometry_native)
        FROM source
        WHERE ST_Intersects(geometry_native, ST_GeomFromText(?))
        ORDER BY datetime DESC NULLS LAST;
        """
        frame = con.execute(query, [boundary.geometry.wkt]).fetchdf()
        if frame.empty:
            return []
        return frame.to_dict(orient="records")
    finally:
        con.close()


def _write_item_records_to_ndjson(item_records: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    """Write STAC item dictionaries to newline-delimited JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item_record in item_records:
            handle.write(json.dumps(item_record, separators=(",", ":"), default=_json_default) + "\n")


async def index_remote_geoparquet_source(config: SourceConfig, boundary: BoundaryContext) -> SourceRunSummary:
    """Subset the public Earthview GeoParquet mirror to a Puerto Rico STAC file."""

    if config.remote_parquet_url is None:
        raise ValueError("remote_parquet_url is required for remote GeoParquet indexing.")

    log = logging.getLogger(f"raster_stac.{config.source_name}")
    stac_path = config.artifacts.stac_path
    index_path = config.artifacts.index_path
    work_dir = make_work_dir(stac_path)
    ndjson_path = work_dir / TEMP_NDJSON_FILENAME

    try:
        started_at = time.perf_counter()
        row_records = _query_remote_earthview_rows(config.remote_parquet_url, boundary)
        item_records = [_serialize_remote_row_to_item(row) for row in row_records]
        item_count = len(item_records)

        if stac_path.exists():
            stac_path.unlink()

        if item_count == 0:
            _empty_normalized_index().to_parquet(index_path, index=False, compression="zstd")
            return SourceRunSummary(
                source_name=config.source_name,
                strategy=config.strategy,
                item_count=0,
                stac_path=None,
                index_path=index_path.as_posix(),
                notes="Remote Earthview parquet returned no Puerto Rico-intersecting items.",
            )

        _write_item_records_to_ndjson(item_records, ndjson_path)
        parse_stac_ndjson_to_parquet(input_path=ndjson_path, output_path=stac_path)
        preview = await preview_output_with_rustac(stac_path, boundary)
        normalized_rows = _write_normalized_index_from_stac_parquet(
            stac_path,
            index_path,
            source_name=config.source_name,
            strategy=config.strategy,
        )
        elapsed_seconds = time.perf_counter() - started_at
        file_size_mb = stac_path.stat().st_size / (1024 * 1024)
        return SourceRunSummary(
            source_name=config.source_name,
            strategy=config.strategy,
            item_count=item_count,
            stac_path=stac_path.as_posix(),
            index_path=index_path.as_posix(),
            file_size_mb=file_size_mb,
            preview_bbox_count=preview["bbox_preview_count"],
            preview_intersects_count=preview["intersects_preview_count"],
            preview_arrow_rows=preview["arrow_preview_rows"],
            sample_item_ids=preview["sample_item_ids"],
            notes=f"normalized_rows={normalized_rows}; elapsed_seconds={elapsed_seconds:.1f}",
        )
    finally:
        ndjson_path.unlink(missing_ok=True)
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def build_combined_index(index_paths: Sequence[Path], output_path: Path) -> Path:
    """Combine per-source normalized indexes into one Puerto Rico raster index."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[gpd.GeoDataFrame] = []
    for index_path in index_paths:
        if not index_path.exists():
            continue
        current = gpd.read_parquet(index_path)
        if current.empty:
            continue
        if current.crs is None:
            current = current.set_crs(OUTPUT_CRS)
        else:
            current = current.to_crs(OUTPUT_CRS)
        frames.append(current)

    if not frames:
        combined = _empty_normalized_index()
    else:
        combined = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True),
            geometry="geometry",
            crs=OUTPUT_CRS,
        )
        combined = combined.sort_values(
            by=["source", "acquired_at", "item_id"],
            ascending=[True, False, True],
            na_position="last",
        ).reset_index(drop=True)

    combined.to_parquet(output_path, index=False, compression="zstd")
    return output_path


def summary_to_dataframe(summaries: Sequence[SourceRunSummary]) -> pd.DataFrame:
    """Convert source run summaries into a tabular notebook-friendly view."""

    if not summaries:
        return pd.DataFrame(
            columns=[
                "source_name",
                "strategy",
                "item_count",
                "stac_path",
                "index_path",
                "file_size_mb",
                "preview_bbox_count",
                "preview_intersects_count",
                "preview_arrow_rows",
                "notes",
            ]
        )
    return pd.DataFrame([summary.as_dict() for summary in summaries])


def _empty_consolidated_catalog() -> gpd.GeoDataFrame:
    """Return an empty GeoDataFrame for the unified raster catalog schema."""

    frame = pd.DataFrame(columns=[column for column in CONSOLIDATED_CATALOG_COLUMNS if column != "geometry"])
    geometry = gpd.GeoSeries([], crs=OUTPUT_CRS)
    return gpd.GeoDataFrame(frame, geometry=geometry, crs=OUTPUT_CRS)


def _to_consolidated_catalog_row(
    row: Mapping[str, Any],
    source_name: str,
    boundary: BoundaryContext,
) -> dict[str, Any] | None:
    """Normalize one STAC-like mapping to the consolidated output schema."""

    geometry = _coerce_geometry(row.get("geometry"))
    bbox = _coerce_bbox(row.get("bbox"))
    if geometry is None and bbox is not None:
        geometry = box(*bbox)
    if geometry is None or geometry.is_empty:
        return None
    if not geometry.intersects(boundary.envelope):
        return None
    if not boundary.prepared_geometry.intersects(geometry):
        return None

    if bbox is None:
        bbox = tuple(float(value) for value in geometry.bounds)

    assets = _coerce_mapping(row.get("assets"))
    _, visual_href = _pick_asset(
        assets,
        exact_names=("visual", "image"),
        role_hints=("visual",),
        keyword_hints=("visual", "rgb"),
        disallow_keywords=("thumbnail", "preview"),
        fallback_first=True,
    )
    _, analytic_href = _pick_asset(
        assets,
        exact_names=("analytic", "data"),
        role_hints=("data",),
        keyword_hints=("analytic", "cog", "ortho", "ms", "nir"),
        disallow_keywords=("thumbnail", "preview", "visual"),
        fallback_first=False,
    )
    _, preview_href = _pick_asset(
        assets,
        exact_names=("preview",),
        keyword_hints=("preview", "overview"),
        fallback_first=False,
    )
    _, thumbnail_href = _pick_asset(
        assets,
        exact_names=("thumbnail",),
        keyword_hints=("thumbnail", "thumb"),
        fallback_first=False,
    )

    return {
        "source": source_name,
        "item_id": _clean_string(_extract_value(row, "id")),
        "collection_id": _clean_string(_extract_value(row, "collection")),
        "acquired_at": _coerce_timestamp(_extract_value(row, "datetime", "start_datetime")),
        "platform": _clean_string(_extract_value(row, "platform")),
        "constellation": _clean_string(_extract_value(row, "constellation")),
        "gsd": _coerce_float(_extract_value(row, "gsd", "eo:gsd", "proj:gsd")),
        "cloud_cover": _coerce_float(_extract_value(row, "eo:cloud_cover", "cloud_cover")),
        "self_href": _extract_self_href(row),
        "visual_asset_href": visual_href,
        "analytic_asset_href": analytic_href,
        "preview_asset_href": preview_href,
        "thumbnail_asset_href": thumbnail_href,
        "asset_count": len(assets),
        "bbox_minx": bbox[0],
        "bbox_miny": bbox[1],
        "bbox_maxx": bbox[2],
        "bbox_maxy": bbox[3],
        "geometry": geometry,
    }


async def _collect_catalog_items_for_boundary(
    catalog_url: str,
    boundary: BoundaryContext,
    follow_child_links: bool,
    prune_catalogs_by_extent: bool,
    filter_items_by_geometry: bool,
    source_tag: str,
    log: logging.Logger,
) -> list[dict[str, Any]]:
    """Crawl a static STAC JSON catalog and return qualifying item dictionaries."""

    work_anchor = PROJECT_ROOT / "cache" / f"{source_tag}.parquet"
    work_dir = make_work_dir(work_anchor)
    ndjson_path = work_dir / TEMP_NDJSON_FILENAME
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS, limit_per_host=10)

    try:
        async with aiohttp.ClientSession(
            connector=connector,
            headers={"User-Agent": USER_AGENT},
        ) as session:
            await crawl_catalog_roots_to_ndjson(
                session=session,
                root_catalog_urls=[catalog_url],
                ndjson_path=ndjson_path,
                boundary=boundary,
                follow_child_links=follow_child_links,
                prune_catalogs_by_extent=prune_catalogs_by_extent,
                filter_items_by_geometry=filter_items_by_geometry,
                log=log,
            )

        if not ndjson_path.exists():
            return []

        records: list[dict[str, Any]] = []
        with ndjson_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                try:
                    records.append(json.loads(payload))
                except json.JSONDecodeError:
                    continue
        return records
    finally:
        ndjson_path.unlink(missing_ok=True)
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def _query_remote_parquet_rows(
    remote_uri: str,
    boundary: BoundaryContext,
    context: str,
) -> list[dict[str, Any]]:
    """Query a remote STAC GeoParquet and return rows intersecting Puerto Rico."""

    con = create_duckdb_connection(load_httpfs=True)
    try:
        remote_uri_sql = _sql_quote(remote_uri)
        geometry_expr = _parquet_geometry_expression(con, remote_uri, context=context)
        schema_frame = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({remote_uri_sql});"
        ).fetchdf()
        has_datetime = bool((schema_frame["column_name"] == "datetime").any())

        query = f"""
        WITH source AS (
            SELECT
                *,
                {geometry_expr} AS geometry_native
            FROM read_parquet({remote_uri_sql})
        )
        SELECT * EXCLUDE (geometry_native)
        FROM source
        WHERE geometry_native IS NOT NULL
          AND ST_Intersects(geometry_native, ST_GeomFromText(?))
        """
        if has_datetime:
            query += " ORDER BY datetime DESC NULLS LAST"
        query += ";"

        frame = con.execute(query, [boundary.geometry.wkt]).fetchdf()
        if frame.empty:
            return []
        return frame.to_dict(orient="records")
    finally:
        con.close()


async def materialize_consolidated_pr_raster_catalog(
    output_path: Path,
    boundary: BoundaryContext,
    naip_catalog_url: str,
    maxar_remote_parquet_url: str = MAXAR_PUBLIC_PARQUET_URI,
    earthview_remote_parquet_url: str = EARTHVIEW_PUBLIC_PARQUET_URI,
) -> pd.DataFrame:
    """Materialize one Puerto Rico AOI-filtered consolidated GeoParquet across all sources."""

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("raster_stac.consolidated")
    consolidated_rows: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []

    naip_items = await _collect_catalog_items_for_boundary(
        catalog_url=naip_catalog_url,
        boundary=boundary,
        follow_child_links=False,
        prune_catalogs_by_extent=False,
        filter_items_by_geometry=True,
        source_tag="naip_2021_pr",
        log=log,
    )
    naip_rows = [
        row
        for row in (
            _to_consolidated_catalog_row(item, source_name="naip_2021_pr", boundary=boundary)
            for item in naip_items
        )
        if row is not None
    ]
    consolidated_rows.extend(naip_rows)
    source_summaries.append(
        {
            "source": "naip_2021_pr",
            "strategy": "static_catalog_json",
            "catalog_reference": naip_catalog_url,
            "item_rows": len(naip_rows),
            "notes": None,
        }
    )

    for source_name, remote_uri, context_label in (
        ("maxar_open_data", maxar_remote_parquet_url, "remote Maxar parquet"),
        ("satellogic_earthview", earthview_remote_parquet_url, "remote Earthview parquet"),
    ):
        try:
            remote_rows = _query_remote_parquet_rows(
                remote_uri=remote_uri,
                boundary=boundary,
                context=context_label,
            )
            normalized_rows = [
                row
                for row in (
                    _to_consolidated_catalog_row(remote_row, source_name=source_name, boundary=boundary)
                    for remote_row in remote_rows
                )
                if row is not None
            ]
            consolidated_rows.extend(normalized_rows)
            source_summaries.append(
                {
                    "source": source_name,
                    "strategy": "remote_geoparquet_duckdb",
                    "catalog_reference": remote_uri,
                    "item_rows": len(normalized_rows),
                    "notes": None,
                }
            )
        except Exception as exc:
            source_summaries.append(
                {
                    "source": source_name,
                    "strategy": "remote_geoparquet_duckdb",
                    "catalog_reference": remote_uri,
                    "item_rows": 0,
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )

    if not consolidated_rows:
        consolidated_gdf = _empty_consolidated_catalog()
    else:
        consolidated_gdf = gpd.GeoDataFrame(consolidated_rows, geometry="geometry", crs=OUTPUT_CRS)
        consolidated_gdf = consolidated_gdf[CONSOLIDATED_CATALOG_COLUMNS]
        consolidated_gdf = consolidated_gdf.drop_duplicates(subset=["source", "item_id"], keep="first")
        consolidated_gdf = consolidated_gdf.sort_values(
            by=["source", "acquired_at", "item_id"],
            ascending=[True, False, True],
            na_position="last",
        ).reset_index(drop=True)

    consolidated_gdf.to_parquet(output_path, index=False, compression="zstd")

    summary_frame = pd.DataFrame(source_summaries)
    summary_frame["output_path"] = output_path.as_posix()
    summary_frame["output_rows_total"] = int(len(consolidated_gdf))
    return summary_frame