# %% [markdown]
# # Puerto Rico Static STAC to GeoParquet
#
# This notebook materializes static STAC catalogs into STAC-GeoParquet for the
# raster ingestion pipeline described in the project planning docs.
#
# Workflow per catalog:
# 1. Crawl the static catalog tree asynchronously.
# 2. Append qualifying STAC Item JSON dicts to `temp_items.ndjson` with
#    `aiofiles`.
# 3. Convert the NDJSON to GeoParquet with
#    `stac_geoparquet.arrow.parse_stac_ndjson_to_parquet`.
# 4. Run post-conversion `rustac` searches against the GeoParquet as a direct
#    QA check that the output is searchable with STAC-style `bbox` and
#    `intersects` filters.

# %%
"""04_PR_stac_geoparquet.py

Jupytext-friendly notebook script for crawling multiple static STAC catalogs,
writing qualifying STAC Items to newline-delimited JSON, and converting the
result to STAC-GeoParquet for Puerto Rico-focused downstream use.

Target catalogs
---------------
- NAIP 2021 Puerto Rico
- Satellogic Earthview
- Maxar Open Data
"""

# %%
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import aiofiles
import aiohttp
import duckdb
import nest_asyncio
import osmnx as ox
import rustac
from dotenv import load_dotenv
from shapely import from_wkb
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from stac_geoparquet.arrow import parse_stac_ndjson_to_parquet

nest_asyncio.apply()


def resolve_project_root(start: Path | None = None) -> Path:
    """Find the repository root from an arbitrary working directory."""

    current = (start or Path.cwd()).resolve()
    markers = ("project_rules.md", ".git")
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return current


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TARGET_CATALOGS: dict[str, str] = {
    "https://coastalimagery.blob.core.windows.net/digitalcoast/PR_NAIP_2021_9825/stac/catalog.json": "data/rasters/pr_naip_2021.parquet",
    "https://satellogic-earthview.s3.us-west-2.amazonaws.com/stac/catalog.json": "data/rasters/satellogic_earthview.parquet",
    "https://maxar-opendata.s3.amazonaws.com/events/catalog.json": "data/rasters/maxar_opendata.parquet",
}

GLOBAL_CATALOGS: set[str] = {
    "https://satellogic-earthview.s3.us-west-2.amazonaws.com/stac/catalog.json",
    "https://maxar-opendata.s3.amazonaws.com/events/catalog.json",
}

MAX_CONCURRENT_REQUESTS = 24
MAX_HTTP_RETRIES = 3
HTTP_RETRY_BACKOFF_SECONDS = 1.5
TEMP_NDJSON_FILENAME = "temp_items.ndjson"
RUSTAC_PREVIEW_LIMIT = 3


def _to_bytes(value: object) -> bytes:
    """Normalize DuckDB binary values to plain bytes."""

    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    return bytes(value)


def resolve_vector_db_path() -> Path:
    """Resolve the local DuckDB path with a conservative fallback order."""

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


def create_spatial_connection(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open DuckDB and load extensions required for geometry access."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("LOAD spatial;")
    except duckdb.Error:
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")
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


def load_puerto_rico_boundary() -> tuple[BaseGeometry, str]:
    """Load the Puerto Rico boundary from DuckDB first, then fall back to osmnx."""

    db_path = resolve_vector_db_path()
    if db_path.exists():
        con: duckdb.DuckDBPyConnection | None = None
        try:
            con = create_spatial_connection(db_path)
            if duckdb_table_exists(con, "pr_municipalities"):
                row = con.execute(
                    """
                    SELECT ST_AsWKB(ST_Union_Agg(geometry)) AS geometry_wkb
                    FROM pr_municipalities
                    WHERE geometry IS NOT NULL;
                    """
                ).fetchone()
                if row and row[0] is not None:
                    boundary = from_wkb(_to_bytes(row[0]))
                    if boundary is not None and not boundary.is_empty:
                        return boundary, f"duckdb:{db_path.name}"
        except Exception as exc:
            log.warning("DuckDB boundary load failed (%s). Falling back to osmnx.", exc)
        finally:
            if con is not None:
                con.close()

    log.info("Loading Puerto Rico boundary from osmnx geocoder.")
    boundary = ox.geocode_to_gdf("Puerto Rico").geometry.union_all()
    return boundary, "osmnx"


PR_BOUNDARY, PR_BOUNDARY_SOURCE = load_puerto_rico_boundary()
PR_BOUNDARY_PREPARED = prep(PR_BOUNDARY)
PR_BOUNDARY_ENVELOPE = box(*PR_BOUNDARY.bounds)
PR_BOUNDARY_BOUNDS = list(PR_BOUNDARY.bounds)
PR_BOUNDARY_GEOJSON = mapping(PR_BOUNDARY)

log.info(
    "Puerto Rico boundary loaded from %s with bounds %s",
    PR_BOUNDARY_SOURCE,
    PR_BOUNDARY_BOUNDS,
)


def resolve_output_path(relative_path: str) -> Path:
    """Resolve a repository-relative output path and ensure its parent exists."""

    output_path = PROJECT_ROOT / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def make_work_dir(output_path: Path) -> Path:
    """Create a deterministic temporary work directory for one catalog run."""

    work_dir = output_path.parent / "_tmp_stac" / output_path.stem
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def item_intersects_puerto_rico(item_dict: dict[str, Any]) -> bool:
    """Return True when the item footprint intersects the Puerto Rico boundary."""

    item_geometry: BaseGeometry | None = None
    geometry_json = item_dict.get("geometry")
    if geometry_json:
        try:
            item_geometry = shape(geometry_json)
        except Exception:
            item_geometry = None

    if item_geometry is None:
        bbox_values = item_dict.get("bbox")
        if bbox_values and len(bbox_values) >= 4:
            item_geometry = box(*bbox_values[:4])
        else:
            return False

    if item_geometry.is_empty:
        return False

    if not item_geometry.intersects(PR_BOUNDARY_ENVELOPE):
        return False

    return PR_BOUNDARY_PREPARED.intersects(item_geometry)


def catalog_might_intersect_puerto_rico(catalog_dict: dict[str, Any]) -> bool:
    """Return True when a catalog or collection extent might intersect Puerto Rico."""

    try:
        bboxes = catalog_dict.get("extent", {}).get("spatial", {}).get("bbox", [])
        if not bboxes:
            return True
        return any(
            len(current_bbox) >= 4
            and box(*current_bbox[:4]).intersects(PR_BOUNDARY_ENVELOPE)
            for current_bbox in bboxes
        )
    except Exception:
        return True


async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """Fetch JSON with bounded concurrency and basic retry handling."""

    async with semaphore:
        for attempt in range(1, MAX_HTTP_RETRIES + 1):
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
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


def resolve_href(base_url: str, href: str) -> str:
    """Resolve a potentially relative STAC href against its parent URL."""

    if href.startswith(("http://", "https://", "s3://")):
        return href
    return urljoin(base_url, href)


async def crawl_catalog_to_ndjson(
    session: aiohttp.ClientSession,
    catalog_url: str,
    ndjson_path: Path,
    needs_pr_filter: bool,
) -> int:
    """Breadth-first crawl of a static STAC catalog into NDJSON."""

    item_count = 0
    visited_catalogs: set[str] = set()
    pending_catalog_urls: list[str] = [catalog_url]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiofiles.open(ndjson_path, mode="w", encoding="utf-8") as ndjson_file:
        while pending_catalog_urls:
            batch_urls = [
                current_url
                for current_url in pending_catalog_urls
                if current_url not in visited_catalogs
            ]
            pending_catalog_urls = []

            if not batch_urls:
                break

            visited_catalogs.update(batch_urls)
            catalog_payloads = await asyncio.gather(
                *[fetch_json(session, current_url, semaphore) for current_url in batch_urls]
            )

            item_urls: list[str] = []
            for catalog_dict, base_url in zip(catalog_payloads, batch_urls):
                if catalog_dict is None:
                    continue

                if needs_pr_filter and not catalog_might_intersect_puerto_rico(catalog_dict):
                    log.debug(
                        "Pruned subtree outside Puerto Rico: %s",
                        catalog_dict.get("id", base_url),
                    )
                    continue

                for link in catalog_dict.get("links", []):
                    rel = link.get("rel")
                    href = link.get("href")
                    if not rel or not href:
                        continue

                    resolved_href = resolve_href(base_url, href)
                    if rel == "child":
                        pending_catalog_urls.append(resolved_href)
                    elif rel == "item":
                        item_urls.append(resolved_href)

            if not item_urls:
                continue

            log.info(
                "Fetching %d item documents after visiting %d catalog nodes.",
                len(item_urls),
                len(visited_catalogs),
            )

            item_payloads = await asyncio.gather(
                *[fetch_json(session, item_url, semaphore) for item_url in item_urls]
            )

            for item_dict in item_payloads:
                if item_dict is None:
                    continue

                # Global catalogs are filtered during crawl so the temporary NDJSON
                # and the final GeoParquet stay bounded to Puerto Rico-relevant rows.
                if needs_pr_filter and not item_intersects_puerto_rico(item_dict):
                    continue

                await ndjson_file.write(json.dumps(item_dict, separators=(",", ":")) + "\n")
                item_count += 1

                if item_count % 250 == 0:
                    log.info("Wrote %d qualifying items to %s", item_count, ndjson_path.name)

    return item_count


async def preview_output_with_rustac(output_path: Path) -> dict[str, Any]:
    """Run lightweight rustac searches against the generated GeoParquet output."""

    # This post-conversion QA step uses rustac's built-in STAC-style search
    # helpers directly on the GeoParquet file. The same pattern can be reused by
    # downstream notebooks to query the materialized metadata without traversing
    # the original static catalog again.
    bbox_preview = await rustac.search(
        output_path.as_posix(),
        bbox=PR_BOUNDARY_BOUNDS,
        limit=RUSTAC_PREVIEW_LIMIT,
    )
    intersects_preview = await rustac.search(
        output_path.as_posix(),
        intersects=PR_BOUNDARY_GEOJSON,
        limit=RUSTAC_PREVIEW_LIMIT,
    )
    duckdb_client = rustac.DuckdbClient()
    arrow_preview = duckdb_client.search_to_arrow(
        output_path.as_posix(),
        intersects=PR_BOUNDARY_GEOJSON,
        limit=RUSTAC_PREVIEW_LIMIT,
    )

    return {
        "bbox_preview_count": len(bbox_preview),
        "intersects_preview_count": len(intersects_preview),
        "arrow_preview_rows": int(arrow_preview.num_rows),
        "sample_item_ids": [item.get("id") for item in intersects_preview],
    }


async def process_catalog(catalog_url: str, relative_output_path: str) -> dict[str, Any]:
    """Crawl one static catalog, convert it to GeoParquet, and validate it."""

    output_path = resolve_output_path(relative_output_path)
    work_dir = make_work_dir(output_path)
    ndjson_path = work_dir / TEMP_NDJSON_FILENAME
    needs_pr_filter = catalog_url in GLOBAL_CATALOGS

    log.info("=" * 72)
    log.info("Catalog: %s", catalog_url)
    log.info("Output:  %s", output_path)
    log.info("PR filter enabled: %s", needs_pr_filter)
    log.info("=" * 72)

    crawl_started_at = time.perf_counter()
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS, limit_per_host=10)
    headers = {"User-Agent": "ProyectoFinal-STACGeoParquet/1.0"}

    try:
        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            item_count = await crawl_catalog_to_ndjson(
                session=session,
                catalog_url=catalog_url,
                ndjson_path=ndjson_path,
                needs_pr_filter=needs_pr_filter,
            )

        crawl_elapsed = time.perf_counter() - crawl_started_at
        log.info("Crawl complete: %d items written in %.1f seconds", item_count, crawl_elapsed)

        if item_count == 0:
            return {
                "catalog_url": catalog_url,
                "output_path": None,
                "item_count": 0,
                "file_size_mb": None,
                "rustac_preview": None,
            }

        if output_path.exists():
            output_path.unlink()

        # parse_stac_ndjson_to_parquet performs a two-pass chunked conversion:
        # first it scans the NDJSON to infer a strict Arrow schema across all
        # observed STAC properties, then it re-reads the file in bounded chunks
        # and streams those batches to a ParquetWriter. This avoids materializing
        # the full catalog in memory while still producing a schema-stable,
        # GeoParquet-compliant STAC output.
        convert_started_at = time.perf_counter()
        parse_stac_ndjson_to_parquet(input_path=ndjson_path, output_path=output_path)
        convert_elapsed = time.perf_counter() - convert_started_at
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        log.info(
            "GeoParquet written: %s (%.2f MB) in %.1f seconds",
            output_path.name,
            file_size_mb,
            convert_elapsed,
        )

        rustac_preview = await preview_output_with_rustac(output_path)
        log.info(
            "rustac QA -> bbox=%d | intersects=%d | arrow=%d | sample_ids=%s",
            rustac_preview["bbox_preview_count"],
            rustac_preview["intersects_preview_count"],
            rustac_preview["arrow_preview_rows"],
            rustac_preview["sample_item_ids"],
        )

        ndjson_path.unlink(missing_ok=True)
        log.info("Deleted intermediate NDJSON: %s", ndjson_path)

        return {
            "catalog_url": catalog_url,
            "output_path": output_path,
            "item_count": item_count,
            "file_size_mb": file_size_mb,
            "rustac_preview": rustac_preview,
        }
    finally:
        ndjson_path.unlink(missing_ok=True)
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


# %% [markdown]
# ## Run the Puerto Rico STAC materialization workflow
#
# The loop below processes each target catalog sequentially. The output files are
# the requested Puerto Rico-focused GeoParquet artifacts in `data/rasters/`.

# %%
async def main() -> list[dict[str, Any]]:
    """Process all configured catalogs and log a concise summary."""

    summaries: list[dict[str, Any]] = []
    for catalog_url, relative_output_path in TARGET_CATALOGS.items():
        summaries.append(await process_catalog(catalog_url, relative_output_path))

    log.info("")
    log.info("=" * 72)
    log.info("SUMMARY")
    log.info("=" * 72)
    for summary in summaries:
        output_path = summary["output_path"]
        if output_path is None:
            log.info("SKIPPED  %s -> no qualifying items", summary["catalog_url"])
            continue

        log.info(
            "OK       %s -> %s | items=%d | size=%.2f MB",
            summary["catalog_url"],
            output_path,
            summary["item_count"],
            summary["file_size_mb"],
        )

    return summaries


# %%
if __name__ == "__main__":
    asyncio.run(main())