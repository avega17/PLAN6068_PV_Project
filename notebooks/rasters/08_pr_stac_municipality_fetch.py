# %% [markdown]
# # STAC Raster Fetch and Occupied-H3 Clip
#
# Uses the consolidated Puerto Rico STAC catalog from `05_pr_raster_catalog_indexes.py`
# to fetch every intersecting raster item for occupied H3 cells in San Juan and
# Isabela, clips each asset to the occupied cell geometry, reprojects to
# EPSG:3857, and writes model-ready local derivatives under
# `data/rasters/stac/local/`.

# %%
"""07_pr_stac_municipality_fetch.py

Fetch and clip intersecting STAC raster assets for occupied H3 cells.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import rasterio
from rasterio.enums import Resampling
from dotenv import load_dotenv
from rasterio.mask import mask
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_geom
from shapely.geometry import mapping


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
from utils.overture import occupied_h3_cells_sql

TARGET_MUNICIPALITIES = ("San Juan", "Isabela")
OVERTURE_BUILDINGS_TABLE = "pr_overture_buildings"
STAC_CATALOG_PATH = PROJECT_ROOT / "data" / "rasters" / "stac" / "pr_raster_catalog_items.parquet"
LOCAL_STAC_ROOT = PROJECT_ROOT / "data" / "rasters" / "stac" / "local"
LOCAL_FETCH_MANIFEST_PATH = PROJECT_ROOT / "data" / "rasters" / "stac" / "pr_local_stac_fetch_manifest.parquet"
MODEL_READY_CRS = "EPSG:3857"
PREFERRED_ASSET_COLUMNS = (
    ("visual_asset_href", "visual"),
    ("analytic_asset_href", "analytic"),
)
MAX_ITEMS_THIS_RUN = int(os.getenv("MAX_STAC_ITEMS_THIS_RUN", "0")) or None
MAX_ASSETS_THIS_RUN = int(os.getenv("MAX_STAC_ASSETS_THIS_RUN", "0")) or None
OVERWRITE_EXISTING = os.getenv("OVERWRITE_STAC_LOCAL_CACHE", "0") == "1"


def resolve_db_path() -> Path:
    value = os.getenv("VECTOR_DB")
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path if len(path.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / path
        return path
    return PROJECT_ROOT / "data" / "PR_PV_plan_data.duckdb"


def _to_bytes(value: object) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    return bytes(value)


def slugify(value: str) -> str:
    return value.strip().replace(" ", "_").replace("/", "_")


def _project_relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    row = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?;",
        [table_name],
    ).fetchone()
    return bool(row and row[0])


def load_target_h3_cells(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    if not table_exists(con, OVERTURE_BUILDINGS_TABLE):
        raise RuntimeError(
            f"{OVERTURE_BUILDINGS_TABLE} not found; run notebooks/vectors/03_overture_buildings_ingest.py first."
        )

    municipio_sql = ", ".join(f"'{municipio}'" for municipio in TARGET_MUNICIPALITIES)
    occupied_h3_sql = occupied_h3_cells_sql(OVERTURE_BUILDINGS_TABLE)
    frame = con.execute(
        f"""
        WITH occupied_h3 AS ({occupied_h3_sql})
        SELECT
               h3_cell_id,
               h3_resolution,
               municipality_name AS municipio,
               municipality_geoid AS municipio_geoid,
               building_count,
               municipality_building_count,
               crosses_municipality_boundary,
               ST_AsWKB(geometry) AS geometry_wkb
         FROM occupied_h3
        WHERE municipality_name IN ({municipio_sql})
        ORDER BY municipio, h3_cell_id;
        """
    ).fetchdf()
    if frame.empty:
        return gpd.GeoDataFrame(
            columns=[
                "h3_cell_id",
                "h3_resolution",
                "municipio",
                "municipio_geoid",
                "building_count",
                "municipality_building_count",
                "crosses_municipality_boundary",
                "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

    geometry = gpd.GeoSeries.from_wkb(frame["geometry_wkb"].map(_to_bytes), crs="EPSG:4326")
    return gpd.GeoDataFrame(frame.drop(columns=["geometry_wkb"]), geometry=geometry, crs="EPSG:4326")


def load_catalog() -> gpd.GeoDataFrame:
    if not STAC_CATALOG_PATH.exists():
        raise RuntimeError(
            f"STAC catalog not found at {STAC_CATALOG_PATH}; run notebooks/rasters/05_pr_raster_catalog_indexes.py first."
        )
    catalog = gpd.read_parquet(STAC_CATALOG_PATH)
    if catalog.crs is None:
        return catalog.set_crs("EPSG:4326")
    return catalog.to_crs("EPSG:4326")


def choose_asset(row: pd.Series) -> tuple[str | None, str | None]:
    for column_name, asset_role in PREFERRED_ASSET_COLUMNS:
        href = row.get(column_name)
        if isinstance(href, str) and href.strip():
            return href, asset_role
    return None, None


def _asset_source_rank(source_name: str) -> int:
    order = {
        "pr_naip": 0,
        "naip_2021_pr": 1,
        "maxar_open_data": 2,
        "satellogic_earthview": 3,
    }
    return order.get(str(source_name), 99)


def _asset_group_columns() -> list[str]:
    return ["source", "item_id", "asset_role", "asset_href"]


def _dedupe_exact_asset_candidates(asset_candidates: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop exact footprint/date duplicates while preserving the preferred source row."""

    if asset_candidates.empty:
        return asset_candidates

    deduped = asset_candidates.copy()
    bounds = deduped.geometry.bounds.round(6)
    deduped["_bounds_sig"] = bounds.astype(str).agg("|".join, axis=1)
    deduped["_acquired_sig"] = pd.to_datetime(deduped["acquired_at"], utc=True, errors="coerce").astype(str)
    deduped["_gsd_sig"] = pd.to_numeric(deduped["gsd"], errors="coerce").round(3)
    deduped["_source_rank"] = deduped["source"].map(_asset_source_rank)
    deduped = deduped.sort_values(
        by=["_source_rank", "_acquired_sig", "_gsd_sig", "item_id"],
        ascending=[True, False, True, True],
        na_position="last",
    )
    deduped = deduped.drop_duplicates(
        subset=["asset_role", "_acquired_sig", "_gsd_sig", "_bounds_sig"],
        keep="first",
    )
    return deduped.drop(columns=["_bounds_sig", "_acquired_sig", "_gsd_sig", "_source_rank"]).reset_index(
        drop=True
    )


def resolve_fetch_asset_href(asset_href: str) -> str:
    """Refresh Planetary Computer Azure blob URLs so persisted SAS tokens do not expire the fetch path."""

    parsed = urlsplit(str(asset_href))
    if not parsed.netloc.endswith("blob.core.windows.net"):
        return str(asset_href)

    base_href = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    try:
        refreshed = planetary_computer.sign_url(base_href)
    except Exception:
        return str(asset_href)
    return str(refreshed or asset_href)


def enumerate_target_items(
    catalog: gpd.GeoDataFrame,
    occupied_h3_cells: gpd.GeoDataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if catalog.empty or occupied_h3_cells.empty:
        return pd.DataFrame()

    ranked = catalog.copy()
    ranked["acquired_at"] = pd.to_datetime(ranked["acquired_at"], utc=True, errors="coerce")
    ranked = ranked.sort_values(
        by=["source", "acquired_at", "gsd", "item_id"],
        ascending=[True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)

    candidate_rows: list[dict[str, object]] = []
    for _, item in ranked.iterrows():
        asset_href, asset_role = choose_asset(item)
        if asset_href is None:
            continue
        candidate_rows.append(
            {
                "source": item["source"],
                "item_id": item["item_id"],
                "asset_role": asset_role,
                "asset_href": asset_href,
                "acquired_at": item.get("acquired_at"),
                "gsd": item.get("gsd"),
                "platform": item.get("platform"),
                "catalog_self_href": item.get("self_href"),
                "geometry": item.geometry,
            }
        )

    if not candidate_rows:
        return pd.DataFrame()

    asset_candidates = gpd.GeoDataFrame(candidate_rows, geometry="geometry", crs="EPSG:4326")
    asset_candidates = _dedupe_exact_asset_candidates(asset_candidates)
    cells_with_id = occupied_h3_cells.reset_index(drop=True).copy()
    cells_with_id["_cell_row_id"] = cells_with_id.index
    joined = gpd.sjoin(
        asset_candidates,
        cells_with_id,
        how="inner",
        predicate="intersects",
        lsuffix="asset",
        rsuffix="cell",
    )
    if joined.empty:
        return pd.DataFrame()

    clip_geometries = cells_with_id.set_index("_cell_row_id").geometry
    cell_index_column = "_cell_row_id"
    if cell_index_column not in joined.columns:
        for candidate in ("index_cell", "index_right"):
            if candidate in joined.columns:
                cell_index_column = candidate
                break
    for _, row in joined.iterrows():
        local_dir = (
            LOCAL_STAC_ROOT
            / slugify(str(row["source"]))
            / slugify(str(row["municipio"]))
            / str(row["h3_cell_id"])
        )
        local_path = local_dir / f"{row['item_id']}_{row['asset_role']}_epsg3857.tif"
        rows.append(
            {
                "source": row["source"],
                "item_id": row["item_id"],
                "municipio": row["municipio"],
                "municipio_geoid": row["municipio_geoid"],
                "h3_cell_id": row["h3_cell_id"],
                "h3_resolution": row["h3_resolution"],
                "building_count": row["building_count"],
                "municipality_building_count": row["municipality_building_count"],
                "crosses_municipality_boundary": row["crosses_municipality_boundary"],
                "asset_role": row["asset_role"],
                "asset_href": row["asset_href"],
                "local_asset_path": str(local_path),
                "acquired_at": row.get("acquired_at"),
                "gsd": row.get("gsd"),
                "platform": row.get("platform"),
                "catalog_self_href": row.get("catalog_self_href"),
                "geometry": clip_geometries.loc[int(row[cell_index_column])],
            }
        )

    targets = pd.DataFrame(rows)
    if targets.empty:
        return targets
    targets = targets.drop_duplicates(subset=_asset_group_columns() + ["municipio", "h3_cell_id"])
    targets = targets.sort_values(
        by=["municipio", "h3_cell_id", "source", "acquired_at", "item_id", "asset_role"],
        ascending=[True, True, True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    if MAX_ITEMS_THIS_RUN is not None:
        targets = targets.head(MAX_ITEMS_THIS_RUN).reset_index(drop=True)
    return targets


def reproject_to_model_ready_crs(
    clipped_image: np.ndarray,
    clipped_transform: rasterio.Affine,
    *,
    src_crs: rasterio.crs.CRS,
    dst_crs: str = MODEL_READY_CRS,
    nodata: float | int | None,
) -> tuple[np.ndarray, rasterio.Affine, str, float | int | None]:
    if str(src_crs).upper() == dst_crs.upper():
        return clipped_image, clipped_transform, dst_crs, nodata

    if nodata is None:
        nodata = 0 if np.issubdtype(clipped_image.dtype, np.integer) else np.nan

    bounds = array_bounds(clipped_image.shape[1], clipped_image.shape[2], clipped_transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        dst_crs,
        clipped_image.shape[2],
        clipped_image.shape[1],
        *bounds,
    )
    if dst_width <= 0 or dst_height <= 0:
        raise RuntimeError("Reprojected raster window is empty.")

    destination = np.full((clipped_image.shape[0], dst_height, dst_width), nodata, dtype=clipped_image.dtype)
    for band_idx in range(clipped_image.shape[0]):
        reproject(
            source=clipped_image[band_idx],
            destination=destination[band_idx],
            src_transform=clipped_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=nodata,
            dst_nodata=nodata,
            resampling=Resampling.bilinear,
        )
    return destination, dst_transform, dst_crs, nodata


def _build_manifest_row(
    row: pd.Series,
    *,
    local_path: Path,
    status: str,
    error: str | None,
    model_ready_proj_epsg: int | None,
) -> dict[str, object]:
    return {
        "source": row["source"],
        "item_id": row["item_id"],
        "municipio": row["municipio"],
        "municipio_geoid": row["municipio_geoid"],
        "h3_cell_id": row["h3_cell_id"],
        "h3_resolution": row["h3_resolution"],
        "building_count": row["building_count"],
        "municipality_building_count": row["municipality_building_count"],
        "crosses_municipality_boundary": row["crosses_municipality_boundary"],
        "asset_role": row["asset_role"],
        "asset_href": row["asset_href"],
        "local_asset_path": _project_relative_or_absolute(local_path),
        "status": status,
        "error": error,
        "acquired_at": row.get("acquired_at"),
        "gsd": row.get("gsd"),
        "platform": row.get("platform"),
        "model_ready_proj_epsg": model_ready_proj_epsg or 3857,
    }


def _write_sidecar(
    row: pd.Series,
    *,
    sidecar_path: Path,
    local_path: Path,
    native_proj_epsg: int | None,
    model_ready_proj_epsg: int | None,
) -> None:
    acquired_at = row.get("acquired_at")
    if pd.notna(acquired_at):
        acquired_at = pd.Timestamp(acquired_at).isoformat()
    else:
        acquired_at = None
    sidecar_path.write_text(
        json.dumps(
            {
                "source": row["source"],
                "item_id": row["item_id"],
                "municipio": row["municipio"],
                "municipio_geoid": row["municipio_geoid"],
                "h3_cell_id": row["h3_cell_id"],
                "h3_resolution": int(row["h3_resolution"]),
                "building_count": int(row["building_count"]),
                "municipality_building_count": int(row["municipality_building_count"]),
                "crosses_municipality_boundary": bool(row["crosses_municipality_boundary"]),
                "asset_role": row["asset_role"],
                "acquired_at": acquired_at,
                "gsd": row.get("gsd"),
                "platform": row.get("platform"),
                "original_asset_href": row["asset_href"],
                "catalog_self_href": row.get("catalog_self_href"),
                "clip_strategy": "occupied_h3_cell",
                "native_proj_epsg": native_proj_epsg,
                "model_ready_proj_epsg": model_ready_proj_epsg or 3857,
                "local_asset_path": _project_relative_or_absolute(local_path),
            },
            indent=2,
        )
    )


def clip_asset_to_h3_cells(asset_rows: pd.DataFrame) -> list[dict[str, object]]:
    """Open one remote asset once and write all intersecting H3-cell clips."""

    if asset_rows.empty:
        return []

    results: list[dict[str, object]] = []
    first_row = asset_rows.iloc[0]
    model_ready_proj_epsg = rasterio.crs.CRS.from_string(MODEL_READY_CRS).to_epsg()
    fetch_asset_href = resolve_fetch_asset_href(str(first_row["asset_href"]))
    pending_rows: list[pd.Series] = []

    for row in asset_rows.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        local_path = Path(str(row_series["local_asset_path"]))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not OVERWRITE_EXISTING and local_path.exists():
            results.append(
                _build_manifest_row(
                    row_series,
                    local_path=local_path,
                    status="reused",
                    error=None,
                    model_ready_proj_epsg=model_ready_proj_epsg,
                )
            )
            continue
        pending_rows.append(row_series)

    if not pending_rows:
        return results

    try:
        with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
            with rasterio.open(fetch_asset_href) as src:
                if src.crs is None:
                    raise RuntimeError("Source raster is missing CRS metadata.")

                native_proj_epsg = src.crs.to_epsg() if src.crs else None
                for row in pending_rows:
                    local_path = Path(str(row["local_asset_path"]))
                    sidecar_path = local_path.with_name(f"{local_path.stem}_meta.json")
                    try:
                        geometry = mapping(row["geometry"])
                        if str(src.crs).upper() != "EPSG:4326":
                            geometry = transform_geom("EPSG:4326", src.crs, geometry, precision=6)
                        clipped_image, clipped_transform = mask(src, [geometry], crop=True)
                        if clipped_image.shape[1] == 0 or clipped_image.shape[2] == 0:
                            raise RuntimeError("Clip produced an empty raster window.")

                        clipped_image, clipped_transform, _, nodata = reproject_to_model_ready_crs(
                            clipped_image,
                            clipped_transform,
                            src_crs=src.crs,
                            dst_crs=MODEL_READY_CRS,
                            nodata=src.nodata,
                        )
                        meta = src.meta.copy()
                        meta.update(
                            driver="GTiff",
                            height=clipped_image.shape[1],
                            width=clipped_image.shape[2],
                            transform=clipped_transform,
                            crs=MODEL_READY_CRS,
                            compress="deflate",
                            tiled=True,
                        )
                        if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
                            meta["nodata"] = nodata
                        else:
                            meta.pop("nodata", None)
                        with rasterio.open(local_path, "w", **meta) as dst:
                            dst.write(clipped_image)
                        _write_sidecar(
                            row,
                            sidecar_path=sidecar_path,
                            local_path=local_path,
                            native_proj_epsg=native_proj_epsg,
                            model_ready_proj_epsg=model_ready_proj_epsg,
                        )
                        results.append(
                            _build_manifest_row(
                                row,
                                local_path=local_path,
                                status="fetched",
                                error=None,
                                model_ready_proj_epsg=model_ready_proj_epsg,
                            )
                        )
                    except Exception as exc:
                        results.append(
                            _build_manifest_row(
                                row,
                                local_path=local_path,
                                status="error",
                                error=str(exc),
                                model_ready_proj_epsg=model_ready_proj_epsg,
                            )
                        )
    except Exception as exc:
        error = str(exc)
        for row in pending_rows:
            local_path = Path(str(row["local_asset_path"]))
            results.append(
                _build_manifest_row(
                    row,
                    local_path=local_path,
                    status="error",
                    error=error,
                    model_ready_proj_epsg=model_ready_proj_epsg,
                )
            )

    return results


# %%
if __name__ == "__main__":
    con = duckdb.connect(str(resolve_db_path()))
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("LOAD h3;")

    occupied_h3_cells = load_target_h3_cells(con)
    print(f"loaded {len(occupied_h3_cells):,} occupied H3 cells for target municipalities")
    if occupied_h3_cells.empty:
        print("no occupied H3 cells found in DuckDB.")
        con.close()
        sys.exit(0)

    catalog = load_catalog()
    print(f"loaded {len(catalog):,} STAC catalog items from {STAC_CATALOG_PATH}")

    targets = enumerate_target_items(catalog, occupied_h3_cells)
    print(f"intersecting occupied-H3 clips queued: {len(targets):,}")
    if targets.empty:
        print("no intersecting STAC assets were found for the occupied H3 cells.")
        con.close()
        sys.exit(0)

    print(targets.groupby(["municipio", "source", "asset_role"]).size().to_string())
    asset_groups = list(targets.groupby(_asset_group_columns(), sort=False))
    if MAX_ASSETS_THIS_RUN is not None:
        asset_groups = asset_groups[:MAX_ASSETS_THIS_RUN]
    queued_clip_count = int(sum(len(group) for _, group in asset_groups))
    print(f"unique remote assets queued: {len(asset_groups):,} across {queued_clip_count:,} occupied-H3 clips")

    manifest_rows: list[dict[str, object]] = []
    for _, asset_rows in asset_groups:
        manifest_rows.extend(clip_asset_to_h3_cells(asset_rows.reset_index(drop=True)))
    manifest = pd.DataFrame(manifest_rows)
    LOCAL_FETCH_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(LOCAL_FETCH_MANIFEST_PATH, index=False)
    print(f"wrote {len(manifest):,} fetch rows to {LOCAL_FETCH_MANIFEST_PATH}")
    print(manifest.groupby(["municipio", "source", "status"]).size().to_string())
    con.close()