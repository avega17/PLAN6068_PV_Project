"""Geometry helpers for tiling Census Block Groups into Solar-API-sized chunks.

The Google Solar API ``dataLayers:get`` endpoint is valid at radius ≤ 175 m
when the requested view excludes monthly flux and hourly shade. To cover each
Block Group we lay down a regular grid of tile centers in a metric CRS and
keep only those whose 175 m disk intersects the BG polygon.

Spacing between tile centers is deliberately ~ ``radius * sqrt(2)`` so
neighboring disks overlap slightly — guaranteeing no sub-tile gap inside the
BG — while keeping the tile count minimal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

DEFAULT_RADIUS_M = 175
DEFAULT_SPACING_M = int(round(DEFAULT_RADIUS_M * math.sqrt(2)))  # ~247 m
PR_METRIC_CRS = "EPSG:32619"   # UTM 19N — covers all of Puerto Rico


@dataclass
class TileCandidate:
    tile_id: str
    bg_geoid: str
    municipio: str
    lon: float
    lat: float
    radius_m: int


def _coerce_metric(gdf: gpd.GeoDataFrame, metric_crs: str = PR_METRIC_CRS) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS.")
    return gdf if str(gdf.crs).upper() == metric_crs.upper() else gdf.to_crs(metric_crs)


def build_h3_tile_manifest(
    occupied_h3_cells_gdf: gpd.GeoDataFrame,
    *,
    metric_crs: str = PR_METRIC_CRS,
    h3_id_col: str = "h3_cell_id",
    resolution_col: str = "h3_resolution",
    municipio_col: str = "municipality_name",
    municipio_geoid_col: str = "municipality_geoid",
    building_count_col: str = "building_count",
    municipality_building_count_col: str = "municipality_building_count",
) -> gpd.GeoDataFrame:
    """Build a manifest from occupied H3 cells while preserving legacy columns.

    ``bg_geoid`` is retained as a compatibility alias so the downstream Solar API
    storage layout can migrate in a later step without breaking callers now.
    """

    columns = [
        "tile_id",
        "bg_geoid",
        "h3_cell_id",
        "h3_resolution",
        "municipio",
        "municipio_geoid",
        "lon",
        "lat",
        "radius_m",
        "building_count",
        "municipality_building_count",
        "crosses_municipality_boundary",
        "geometry",
    ]
    if occupied_h3_cells_gdf.empty:
        return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs="EPSG:4326")

    if h3_id_col not in occupied_h3_cells_gdf.columns:
        raise ValueError(f"Expected occupied H3 cells to include column {h3_id_col!r}.")

    cells = occupied_h3_cells_gdf.copy()
    if cells.crs is None:
        cells = cells.set_crs("EPSG:4326")
    else:
        cells = cells.to_crs("EPSG:4326")

    metric_cells = _coerce_metric(cells[["geometry"]].copy(), metric_crs)
    equivalent_radius = np.ceil(np.sqrt(metric_cells.geometry.area.to_numpy(dtype=float) / math.pi)).astype(int)

    if {"cell_center_lon", "cell_center_lat"}.issubset(cells.columns):
        lon = pd.to_numeric(cells["cell_center_lon"], errors="coerce")
        lat = pd.to_numeric(cells["cell_center_lat"], errors="coerce")
    else:
        centroids = cells.geometry.centroid
        lon = centroids.x
        lat = centroids.y

    result = gpd.GeoDataFrame(
        {
            "tile_id": cells[h3_id_col].astype("string"),
            "bg_geoid": cells[h3_id_col].astype("string"),
            "h3_cell_id": cells[h3_id_col].astype("string"),
            "h3_resolution": pd.to_numeric(cells.get(resolution_col, pd.Series(index=cells.index, dtype="int64")), errors="coerce").astype("Int64"),
            "municipio": cells.get(municipio_col, pd.Series(index=cells.index, dtype="string")).astype("string"),
            "municipio_geoid": cells.get(municipio_geoid_col, pd.Series(index=cells.index, dtype="string")).astype("string"),
            "lon": lon.astype(float),
            "lat": lat.astype(float),
            "radius_m": equivalent_radius.astype(int),
            "building_count": pd.to_numeric(cells.get(building_count_col, 0), errors="coerce").fillna(0).astype(int),
            "municipality_building_count": pd.to_numeric(
                cells.get(municipality_building_count_col, cells.get(building_count_col, 0)),
                errors="coerce",
            ).fillna(0).astype(int),
            "crosses_municipality_boundary": cells.get(
                "crosses_municipality_boundary",
                pd.Series(False, index=cells.index, dtype="bool"),
            ).fillna(False).astype(bool),
        },
        geometry=cells.geometry.copy(),
        crs="EPSG:4326",
    )
    return result[columns].reset_index(drop=True)


def attach_h3_priority(
    tile_manifest: gpd.GeoDataFrame,
    *,
    osm_pv_polygons: gpd.GeoDataFrame | None = None,
    seed_neighborhoods: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Attach priority metadata to an occupied-H3-cell manifest."""

    result = tile_manifest.copy()
    if result.empty:
        result["osm_pv_count"] = pd.Series(dtype="int64")
        result["priority_score"] = pd.Series(dtype="int64")
        return result

    result["building_count"] = pd.to_numeric(result.get("building_count", 0), errors="coerce").fillna(0).astype(int)
    result["osm_pv_count"] = 0
    result["priority_score"] = 1

    if osm_pv_polygons is not None and not osm_pv_polygons.empty:
        osm = osm_pv_polygons.to_crs("EPSG:4326")
        pv_points = gpd.GeoDataFrame(geometry=osm.geometry.representative_point(), crs="EPSG:4326")
        joined = gpd.sjoin(pv_points, result[["h3_cell_id", "geometry"]], predicate="within", how="inner")
        if not joined.empty:
            counts = joined.groupby("h3_cell_id").size().rename("osm_pv_count")
            result = result.merge(counts, on="h3_cell_id", how="left", suffixes=("", "_new"))
            result["osm_pv_count"] = result["osm_pv_count_new"].fillna(result["osm_pv_count"]).astype(int)
            result = result.drop(columns=[c for c in result.columns if c.endswith("_new")])
        result["priority_score"] = result["priority_score"].where(result["osm_pv_count"] == 0, 2)

    if seed_neighborhoods is not None and not seed_neighborhoods.empty:
        seeds = seed_neighborhoods.to_crs("EPSG:4326")
        seed_union = unary_union(seeds.geometry.values)
        result["priority_score"] = result["priority_score"].where(~result.geometry.intersects(seed_union), 3)

    result["osm_pv_count"] = result["osm_pv_count"].fillna(0).astype(int)
    result["priority_score"] = result["priority_score"].fillna(1).astype(int)
    return result


def _grid_points(polygon: Polygon, spacing_m: float) -> list[Point]:
    """Interior grid covering the polygon envelope, spaced ``spacing_m`` apart."""

    minx, miny, maxx, maxy = polygon.bounds
    # Snap to spacing so tiles from adjacent BGs align.
    x0 = math.floor(minx / spacing_m) * spacing_m
    y0 = math.floor(miny / spacing_m) * spacing_m
    xs = np.arange(x0, maxx + spacing_m, spacing_m)
    ys = np.arange(y0, maxy + spacing_m, spacing_m)
    return [Point(float(x), float(y)) for x in xs for y in ys]


def tile_block_group(
    bg_row: pd.Series,
    *,
    radius_m: int = DEFAULT_RADIUS_M,
    spacing_m: float | None = None,
    metric_crs: str = PR_METRIC_CRS,
    geometry_col: str = "geometry",
    geoid_col: str = "GEOID",
    municipio_col: str = "municipio",
) -> list[TileCandidate]:
    """Generate tile candidates covering a single Block Group row.

    The BG geometry is expected in ``metric_crs`` already; if not, pass a
    re-projected row. Returns tile candidates whose 175 m disk intersects the
    BG polygon.
    """

    spacing_m = spacing_m or DEFAULT_SPACING_M
    polygon: Polygon = bg_row[geometry_col]
    if polygon is None or polygon.is_empty:
        return []

    buffered = polygon.buffer(radius_m)
    candidates = _grid_points(polygon, spacing_m)
    bg_geoid = str(bg_row[geoid_col])
    municipio = str(bg_row.get(municipio_col, "")) or "UNKNOWN"

    accepted: list[TileCandidate] = []
    for idx, pt in enumerate(candidates):
        if not buffered.contains(pt):
            continue
        # Keep only if the 175 m disk actually touches the BG polygon.
        if not polygon.intersects(pt.buffer(radius_m)):
            continue
        tile_id = f"{bg_geoid}_t{idx:04d}"
        accepted.append(TileCandidate(
            tile_id=tile_id,
            bg_geoid=bg_geoid,
            municipio=municipio,
            lon=float(pt.x),   # metric — will be reprojected downstream
            lat=float(pt.y),
            radius_m=radius_m,
        ))
    return accepted


def build_tile_manifest(
    block_groups_gdf: gpd.GeoDataFrame,
    *,
    radius_m: int = DEFAULT_RADIUS_M,
    spacing_m: float | None = None,
    metric_crs: str = PR_METRIC_CRS,
    geoid_col: str = "GEOID",
    municipio_col: str = "municipio",
) -> gpd.GeoDataFrame:
    """Build a tile manifest covering all rows of ``block_groups_gdf``.

    Output columns:
    ``tile_id, bg_geoid, municipio, lon, lat, radius_m, geometry``
    where ``lon, lat`` are in EPSG:4326 (ready for the Solar API) and
    ``geometry`` is the 175 m disk in EPSG:4326 for downstream spatial joins.
    """

    if block_groups_gdf.empty:
        return gpd.GeoDataFrame(
            columns=["tile_id", "bg_geoid", "municipio", "lon", "lat", "radius_m", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    metric_gdf = _coerce_metric(block_groups_gdf.copy(), metric_crs)
    tiles: list[TileCandidate] = []
    for _, row in metric_gdf.iterrows():
        tiles.extend(tile_block_group(
            row,
            radius_m=radius_m,
            spacing_m=spacing_m,
            geoid_col=geoid_col,
            municipio_col=municipio_col,
        ))
    if not tiles:
        return gpd.GeoDataFrame(
            columns=["tile_id", "bg_geoid", "municipio", "lon", "lat", "radius_m", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    df = pd.DataFrame([t.__dict__ for t in tiles])
    metric_points = gpd.GeoSeries(
        [Point(xy) for xy in zip(df["lon"], df["lat"])],
        crs=metric_crs,
    )
    metric_disks = metric_points.buffer(radius_m)
    wgs_points = metric_points.to_crs("EPSG:4326")
    wgs_disks = metric_disks.to_crs("EPSG:4326")
    df["lon"] = wgs_points.x.values
    df["lat"] = wgs_points.y.values
    return gpd.GeoDataFrame(df, geometry=wgs_disks, crs="EPSG:4326")


def attach_priority(
    tile_manifest: gpd.GeoDataFrame,
    *,
    osm_pv_polygons: gpd.GeoDataFrame | None = None,
    seed_neighborhoods: gpd.GeoDataFrame | None = None,
    buildings_gdf: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Attach ``priority_score``, ``osm_pv_count``, ``building_count``.

    Scoring:
    * 3 — tile overlaps a seed neighborhood polygon (Puerto Nuevo, Barrio Mora).
    * 2 — tile's BG contains any OSM PV polygon.
    * 1 — everything else (coverage sweep).

    All spatial joins are done in EPSG:4326 against the tile disk. Call sites
    are expected to filter manifest to San Juan + Isabela beforehand.
    """

    result = tile_manifest.copy()
    result["building_count"] = 0
    result["osm_pv_count"] = 0
    result["priority_score"] = 1

    if buildings_gdf is not None and not buildings_gdf.empty:
        bld = buildings_gdf.to_crs("EPSG:4326")
        if "building_centroid_lon" in bld.columns and "building_centroid_lat" in bld.columns:
            centroids = gpd.GeoDataFrame(
                bld[["building_centroid_lon", "building_centroid_lat"]].copy(),
                geometry=gpd.points_from_xy(bld["building_centroid_lon"], bld["building_centroid_lat"]),
                crs="EPSG:4326",
            )
        else:
            centroids = gpd.GeoDataFrame(geometry=bld.geometry.representative_point(), crs="EPSG:4326")
        joined = gpd.sjoin(centroids, result[["tile_id", "geometry"]], predicate="within", how="inner")
        counts = joined.groupby("tile_id").size().rename("building_count")
        result = result.merge(counts, on="tile_id", how="left", suffixes=("", "_new"))
        result["building_count"] = result["building_count_new"].fillna(result["building_count"]).astype(int)
        result = result.drop(columns=[c for c in result.columns if c.endswith("_new")])

    # BG-level OSM PV label counts → applied to every tile in that BG.
    if osm_pv_polygons is not None and not osm_pv_polygons.empty:
        osm = osm_pv_polygons.to_crs("EPSG:4326")
        bg_pv_counts = (
            result[["bg_geoid"]].drop_duplicates()
            .merge(
                _count_by_bg(result, osm),
                on="bg_geoid", how="left",
            )
        )
        bg_pv_counts["osm_pv_count"] = bg_pv_counts["osm_pv_count"].fillna(0).astype(int)
        result = result.drop(columns=["osm_pv_count"]).merge(bg_pv_counts, on="bg_geoid", how="left")
        result["priority_score"] = result["priority_score"].where(result["osm_pv_count"] == 0, 2)

    if seed_neighborhoods is not None and not seed_neighborhoods.empty:
        seeds = seed_neighborhoods.to_crs("EPSG:4326")
        seed_union = unary_union(seeds.geometry.values)
        result["priority_score"] = result["priority_score"].where(
            ~result.geometry.intersects(seed_union), 3
        )

    result["osm_pv_count"] = result["osm_pv_count"].fillna(0).astype(int)
    result["building_count"] = result["building_count"].fillna(0).astype(int)
    return result


def _count_by_bg(tile_manifest: gpd.GeoDataFrame, osm: gpd.GeoDataFrame) -> pd.DataFrame:
    """Count OSM PV polygons per BG by spatial-joining PV centroids to tile disks."""

    pv_points = gpd.GeoDataFrame(geometry=osm.geometry.representative_point(), crs="EPSG:4326")
    joined = gpd.sjoin(pv_points, tile_manifest[["bg_geoid", "geometry"]], predicate="within", how="inner")
    if joined.empty:
        return pd.DataFrame({"bg_geoid": [], "osm_pv_count": []})
    # A PV polygon may fall in multiple overlapping tile disks in the same BG;
    # dedupe on (pv_index, bg_geoid) before counting.
    joined = joined.reset_index().drop_duplicates(subset=["index", "bg_geoid"])
    return joined.groupby("bg_geoid").size().rename("osm_pv_count").reset_index()


__all__ = [
    "DEFAULT_RADIUS_M",
    "DEFAULT_SPACING_M",
    "PR_METRIC_CRS",
    "TileCandidate",
    "attach_h3_priority",
    "attach_priority",
    "build_h3_tile_manifest",
    "build_tile_manifest",
    "tile_block_group",
]
