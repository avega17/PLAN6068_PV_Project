"""Worker utilities for parallel OSM rooftop PV fetching.

This module is kept importable so ProcessPoolExecutor can spawn workers on
Windows without depending on notebook cell state.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely import wkt
from shapely.geometry.base import BaseGeometry


EXCLUDED_CONTENT_VALUE = "hot_water"
THERMAL_VALUE = "thermal"
OUTPUT_CRS = "EPSG:4326"


def configure_osm_settings() -> None:
    """Apply the OSMnx settings needed for parallel Overpass access."""

    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.overpass_rate_limit = False
    ox.settings.requests_timeout = 180
    ox.settings.timeout = 180


def polygonal_only(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only polygonal rooftop solar geometries."""

    if frame.empty:
        return frame

    mask = frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    return frame.loc[mask].copy()


def filter_non_photovoltaic(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Exclude thermal/hot-water solar features from the candidate set."""

    filtered = frame.copy()
    if "content" in filtered.columns:
        filtered = filtered[filtered["content"].astype(str) != EXCLUDED_CONTENT_VALUE].copy()
    if "generator:method" in filtered.columns:
        filtered = filtered[filtered["generator:method"].astype(str) != THERMAL_VALUE].copy()
    if "plant:method" in filtered.columns:
        filtered = filtered[filtered["plant:method"].astype(str) != THERMAL_VALUE].copy()
    return filtered


def normalize_osm_frame(frame: gpd.GeoDataFrame, municipality_name: str, query_label: str) -> gpd.GeoDataFrame:
    """Standardize OSMnx output for downstream storage and plotting."""

    if frame.empty:
        return frame

    normalized = frame.reset_index().copy()
    if "index" in normalized.columns and "osmid" not in normalized.columns:
        normalized = normalized.rename(columns={"index": "osmid"})
    normalized["municipality_name"] = municipality_name
    normalized["query_label"] = query_label
    normalized = polygonal_only(normalized)
    normalized = filter_non_photovoltaic(normalized)
    dedupe_columns = [col for col in ["osmid", "municipality_name", "query_label"] if col in normalized.columns]
    if dedupe_columns:
        normalized = normalized.drop_duplicates(subset=dedupe_columns)
    normalized = gpd.GeoDataFrame(normalized, geometry="geometry", crs=frame.crs)
    if normalized.crs is None:
        normalized = normalized.set_crs(OUTPUT_CRS)
    else:
        normalized = normalized.to_crs(OUTPUT_CRS)
    return normalized


def fetch_osm_features_for_municipality_worker(
    municipality_name: str,
    municipality_geometry_wkt: str,
    tag_queries: list[dict[str, object]],
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Fetch rooftop solar PV features for one municipality.

    The worker keeps Overpass calls serial inside one municipality, but the
    outer executor can now run multiple municipalities in parallel using real
    processes.
    """

    configure_osm_settings()
    municipality_geometry: BaseGeometry = wkt.loads(municipality_geometry_wkt)

    collected_frames: list[gpd.GeoDataFrame] = []
    failed_queries: list[str] = []
    matched_queries = 0

    for query in tag_queries:
        try:
            frame = ox.features_from_polygon(municipality_geometry, tags=query["tags"])
        except Exception as exc:  # pragma: no cover - Overpass/network errors are environment-specific.
            if "No matching features" in str(exc):
                continue
            failed_queries.append(f"{query['label']}: {exc}")
            continue

        if frame.empty:
            continue

        normalized = normalize_osm_frame(frame, municipality_name, str(query["label"]))
        if normalized.empty:
            continue

        matched_queries += 1
        collected_frames.append(normalized)

    if not collected_frames:
        empty_columns = ["municipality_name", "query_label", "geometry"]
        stats = {
            "municipality_name": municipality_name,
            "query_count": len(tag_queries),
            "matched_query_count": 0,
            "feature_count": 0,
            "error_count": len(failed_queries),
            "errors": " | ".join(failed_queries),
        }
        return gpd.GeoDataFrame(columns=empty_columns, geometry="geometry", crs=OUTPUT_CRS), stats

    combined = pd.concat(collected_frames, ignore_index=False)
    dedupe_columns = [col for col in ["osmid", "municipality_name", "query_label"] if col in combined.columns]
    if dedupe_columns:
        combined = combined.drop_duplicates(subset=dedupe_columns)

    combined_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs=OUTPUT_CRS)
    stats = {
        "municipality_name": municipality_name,
        "query_count": len(tag_queries),
        "matched_query_count": matched_queries,
        "feature_count": len(combined_gdf),
        "error_count": len(failed_queries),
        "errors": " | ".join(failed_queries),
    }
    return combined_gdf, stats