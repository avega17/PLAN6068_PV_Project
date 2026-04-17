"""Shared helpers for poster-ready Puerto Rico PV figure generation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Rectangle
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds
from shapely import from_wkb
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from utils.raster_stac_index import EARTHVIEW_PUBLIC_PARQUET_URI
from utils.raster_stac_index import MAXAR_PUBLIC_PARQUET_URI
from utils.raster_stac_index import OUTPUT_CRS
from utils.raster_stac_index import PROJECT_ROOT
from utils.raster_stac_index import create_duckdb_connection
from utils.raster_stac_index import load_puerto_rico_boundary
from utils.raster_stac_index import materialize_consolidated_pr_raster_catalog
from utils.raster_stac_index import resolve_vector_db_path

NAIP_2021_CATALOG_URL = (
    "https://coastalimagery.blob.core.windows.net/digitalcoast/"
    "PR_NAIP_2021_9825/stac/catalog.json"
)
CONSOLIDATED_CATALOG_DEFAULT_PATH = PROJECT_ROOT / "data" / "rasters" / "stac" / "pr_raster_catalog_items.parquet"
PREFERRED_POSTER_SOURCES = ["maxar_open_data", "naip_2021_pr", "satellogic_earthview"]
DEFAULT_CHIP_SIZE = 512
METRIC_CRS = "EPSG:32619"
SOURCE_DISPLAY_NAMES = {
    "naip_2021_pr": "NAIP 2021",
    "maxar_open_data": "Maxar Open Data",
    "satellogic_earthview": "EarthView",
}
SOURCE_GSD_CM_FALLBACKS = {
    "naip_2021_pr": 60.0,
}
TABLE_RESOLUTION_BUCKET_THRESHOLD_CM = 45.0


@dataclass
class PosterChipExample:
    """Materialized data needed to render the poster's three-panel example."""

    candidate_index: int | None
    candidate_score: float | None
    municipality_name: str
    building_id: str
    source: str
    gsd_cm: float | None
    item_id: str
    asset_href: str
    building_gdf: gpd.GeoDataFrame
    panel_rows_gdf: gpd.GeoDataFrame
    array_gdf: gpd.GeoDataFrame
    mask: np.ndarray
    chip: dict[str, Any]


def _empty_gdf(crs: str = OUTPUT_CRS) -> gpd.GeoDataFrame:
    """Return an empty GeoDataFrame with a geometry column and CRS."""

    return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=crs), crs=crs)


def _to_wkb_bytes(value: object) -> bytes:
    """Normalize DuckDB binary values to plain bytes."""

    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    return bytes(value)


def fetch_geodataframe(con, query: str, params: list[object] | None = None) -> gpd.GeoDataFrame:
    """Run a WKB-returning query and convert it into a GeoDataFrame."""

    frame = con.execute(query, params or []).fetchdf()
    if frame.empty:
        return _empty_gdf()

    geometry = gpd.GeoSeries.from_wkb(frame["geometry_wkb"].map(_to_wkb_bytes), crs=OUTPUT_CRS)
    return gpd.GeoDataFrame(frame.drop(columns=["geometry_wkb"]), geometry=geometry, crs=OUTPUT_CRS)


def save_figure_variants(figure: plt.Figure, output_stem: Path, dpi: int = 300) -> dict[str, Path]:
    """Save PNG and SVG variants for a Matplotlib figure."""

    base_path = output_stem if output_stem.is_absolute() else PROJECT_ROOT / output_stem
    if base_path.suffix:
        base_path = base_path.with_suffix("")
    base_path.parent.mkdir(parents=True, exist_ok=True)

    png_path = base_path.with_suffix(".png")
    svg_path = base_path.with_suffix(".svg")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    return {"png": png_path, "svg": svg_path}


def load_or_materialize_consolidated_catalog(
    output_path: Path = CONSOLIDATED_CATALOG_DEFAULT_PATH,
    force_refresh: bool = False,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame | None, Path]:
    """Load the local raster catalog, materializing it first when missing."""

    catalog_path = output_path if output_path.is_absolute() else PROJECT_ROOT / output_path
    summary_frame: pd.DataFrame | None = None

    if force_refresh or not catalog_path.exists():
        boundary = load_puerto_rico_boundary()
        summary_frame = asyncio.run(
            materialize_consolidated_pr_raster_catalog(
                output_path=catalog_path,
                boundary=boundary,
                naip_catalog_url=NAIP_2021_CATALOG_URL,
                maxar_remote_parquet_url=MAXAR_PUBLIC_PARQUET_URI,
                earthview_remote_parquet_url=EARTHVIEW_PUBLIC_PARQUET_URI,
            )
        )

    catalog_gdf = gpd.read_parquet(catalog_path)
    if catalog_gdf.crs is None:
        catalog_gdf = catalog_gdf.set_crs(OUTPUT_CRS)
    else:
        catalog_gdf = catalog_gdf.to_crs(OUTPUT_CRS)

    catalog_gdf["acquired_at"] = pd.to_datetime(catalog_gdf["acquired_at"], utc=True, errors="coerce")
    return catalog_gdf, summary_frame, catalog_path


def choose_raster_asset(item_row: pd.Series) -> str | None:
    """Pick a preview-ready raster asset from one catalog row."""

    for column_name in ["visual_asset_href", "analytic_asset_href", "primary_asset_href"]:
        href = item_row.get(column_name)
        if isinstance(href, str) and href.strip():
            return href
    return None


def _source_label(source_name: str) -> str:
    """Return a poster-friendly source label."""

    return SOURCE_DISPLAY_NAMES.get(source_name, source_name.replace("_", " ").title())


def _format_resolution_cm(value: float | int | None) -> str:
    """Format a GSD value in centimeters for table and figure labels."""

    if value is None or pd.isna(value):
        return "n/a"
    numeric = float(value)
    rounded = round(numeric)
    if abs(numeric - rounded) < 0.05:
        return f"{int(rounded)} cm"
    return f"{numeric:.1f} cm"


def _bucket_resolution_cm(value: float | int | None) -> float | None:
    """Collapse raw image resolutions into poster-friendly ~30 / ~60 cm buckets."""

    if value is None or pd.isna(value):
        return None
    return 30.0 if float(value) < TABLE_RESOLUTION_BUCKET_THRESHOLD_CM else 60.0


def _format_bucketed_resolution_cm(value: float | int | None) -> str:
    """Format a bucketed resolution label for the municipality table."""

    if value is None or pd.isna(value):
        return "n/a"
    return f"~{int(round(float(value)))} cm"


def _resolve_gsd_cm(item_row: pd.Series, asset_href: str | None = None) -> float | None:
    """Resolve image resolution in centimeters with source-level fallbacks."""

    gsd_value = pd.to_numeric(pd.Series([item_row.get("gsd")]), errors="coerce").iloc[0]
    if pd.notna(gsd_value):
        return float(gsd_value) * 100.0

    source_name = str(item_row.get("source", ""))
    if source_name in SOURCE_GSD_CM_FALLBACKS:
        return SOURCE_GSD_CM_FALLBACKS[source_name]

    if asset_href:
        with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
            with rasterio.open(asset_href) as src:
                if src.crs and src.crs.is_projected:
                    pixel_width, pixel_height = src.res
                    return float((abs(pixel_width) + abs(pixel_height)) / 2.0 * 100.0)
    return None


def _geometry_area_km2(geometry: BaseGeometry) -> float:
    """Compute one geometry area in km² using a Puerto Rico metric CRS."""

    if geometry is None or geometry.is_empty:
        return 0.0
    return float(gpd.GeoSeries([geometry], crs=OUTPUT_CRS).to_crs(METRIC_CRS).area.iloc[0] / 1_000_000.0)


def sort_catalog_candidates(
    candidates: gpd.GeoDataFrame,
    preferred_sources: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Rank candidate raster items by resolution, recency, then source preference."""

    if candidates.empty:
        return candidates

    ranked = candidates.copy()
    ranked["acquired_at"] = pd.to_datetime(ranked["acquired_at"], utc=True, errors="coerce")
    ranked["gsd_sort"] = pd.to_numeric(ranked["gsd"], errors="coerce").fillna(999999.0)
    source_rank_map = {source_name: index for index, source_name in enumerate(preferred_sources or PREFERRED_POSTER_SOURCES)}
    ranked["source_rank"] = ranked["source"].map(source_rank_map).fillna(len(source_rank_map)).astype(float)
    ranked = ranked.sort_values(
        by=["gsd_sort", "acquired_at", "source_rank", "source", "item_id"],
        ascending=[True, False, True, True, True],
        na_position="last",
    )
    return ranked.drop(columns=["gsd_sort", "source_rank"]).reset_index(drop=True)


def catalog_candidates_for_geometry(
    catalog_gdf: gpd.GeoDataFrame,
    geometry: BaseGeometry,
    preferred_sources: list[str] | None = None,
    one_per_source: bool = False,
) -> gpd.GeoDataFrame:
    """Return sorted catalog rows whose footprints intersect a target geometry."""

    if catalog_gdf.empty or geometry is None or geometry.is_empty:
        return catalog_gdf.iloc[0:0].copy()

    candidates = catalog_gdf[catalog_gdf.geometry.intersects(geometry)].copy()
    if candidates.empty:
        return candidates

    ranked = sort_catalog_candidates(candidates, preferred_sources=preferred_sources)
    if preferred_sources:
        preferred_subset = ranked[ranked["source"].isin(preferred_sources)].copy()
        if not preferred_subset.empty:
            ranked = preferred_subset

    if one_per_source:
        ranked = ranked.groupby("source", sort=False).head(1).reset_index(drop=True)
    return ranked.reset_index(drop=True)


def normalize_image(array: np.ndarray) -> np.ndarray:
    """Contrast-stretch raster data for quick visual inspection."""

    if array.ndim == 2:
        valid = array[np.isfinite(array)]
        if valid.size == 0:
            return np.zeros_like(array, dtype=float)
        lower, upper = np.percentile(valid, [2, 98])
        if upper <= lower:
            upper = lower + 1.0
        return np.clip((array - lower) / (upper - lower), 0.0, 1.0)

    stretched_bands = [normalize_image(array[band_index]) for band_index in range(array.shape[0])]
    return np.dstack(stretched_bands)


def read_raster_chip_for_geometry(
    item_row: pd.Series,
    geometry: BaseGeometry,
    out_size: int = DEFAULT_CHIP_SIZE,
    pad_fraction: float = 0.18,
) -> dict[str, Any] | None:
    """Read a square raster chip around a target geometry."""

    asset_href = choose_raster_asset(item_row)
    if asset_href is None:
        return None

    minx, miny, maxx, maxy = geometry.bounds
    width = max(maxx - minx, 0.0002)
    height = max(maxy - miny, 0.0002)
    minx -= width * pad_fraction
    maxx += width * pad_fraction
    miny -= height * pad_fraction
    maxy += height * pad_fraction

    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(asset_href) as src:
            src_bounds = (minx, miny, maxx, maxy)
            if src.crs and src.crs.to_string() != OUTPUT_CRS:
                src_bounds = transform_bounds(OUTPUT_CRS, src.crs, *src_bounds, densify_pts=21)

            dataset_bounds = src.bounds
            clipped_bounds = (
                max(src_bounds[0], dataset_bounds.left),
                max(src_bounds[1], dataset_bounds.bottom),
                min(src_bounds[2], dataset_bounds.right),
                min(src_bounds[3], dataset_bounds.top),
            )
            if clipped_bounds[0] >= clipped_bounds[2] or clipped_bounds[1] >= clipped_bounds[3]:
                return None

            window = from_bounds(*clipped_bounds, transform=src.transform)
            full_window = Window(0, 0, src.width, src.height)
            window = window.intersection(full_window)
            if window.width <= 0 or window.height <= 0:
                return None

            bands = [1, 2, 3] if src.count >= 3 else [1]
            data = src.read(
                bands,
                window=window,
                out_shape=(len(bands), out_size, out_size),
                boundless=True,
                fill_value=0,
                resampling=Resampling.bilinear,
            )

            base_transform = src.window_transform(window)
            scaled_transform = base_transform * Affine.scale(window.width / out_size, window.height / out_size)
            return {
                "asset_href": asset_href,
                "item_id": item_row.get("item_id"),
                "source": item_row.get("source"),
                "gsd_cm": _resolve_gsd_cm(item_row, asset_href=asset_href),
                "bands": bands,
                "data": data,
                "image": normalize_image(data[0] if len(bands) == 1 else data),
                "transform": scaled_transform,
                "crs": src.crs.to_string() if src.crs else OUTPUT_CRS,
            }


def build_binary_mask(
    chip: dict[str, Any],
    geometries: gpd.GeoSeries | list[BaseGeometry],
    geometry_crs: str = OUTPUT_CRS,
) -> np.ndarray:
    """Rasterize vector geometries into the chip grid."""

    if isinstance(geometries, gpd.GeoSeries):
        geo_series = geometries.copy()
        if geo_series.crs is None:
            geo_series = geo_series.set_crs(geometry_crs)
    else:
        geo_series = gpd.GeoSeries(list(geometries), crs=geometry_crs)

    geo_series = geo_series[geo_series.notna()]
    geo_series = geo_series[~geo_series.geometry.is_empty]
    if geo_series.empty:
        return np.zeros(chip["image"].shape[:2], dtype=np.uint8)

    chip_crs = chip.get("crs") or OUTPUT_CRS
    if chip_crs != str(geo_series.crs):
        geo_series = geo_series.to_crs(chip_crs)

    mask = rasterize(
        [(geometry, 1) for geometry in geo_series.tolist()],
        out_shape=chip["image"].shape[:2],
        transform=chip["transform"],
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )
    return mask


def build_array_geometry_from_panel_rows(
    panel_rows_gdf: gpd.GeoDataFrame,
    bridge_distance_m: float = 0.9,
    shrink_distance_m: float = 0.35,
    final_expand_m: float = 0.10,
    simplify_tolerance_m: float = 0.12,
    min_area_m2: float = 8.0,
) -> gpd.GeoDataFrame:
    """Build more contiguous array polygons by closing gaps between nearby panel rows."""

    if panel_rows_gdf.empty:
        return _empty_gdf()

    metric_rows = panel_rows_gdf.to_crs(METRIC_CRS)
    merged_geometry = unary_union(
        metric_rows.geometry.buffer(bridge_distance_m, cap_style=2, join_style=2).tolist()
    )
    if merged_geometry.is_empty:
        return _empty_gdf()

    smoothed_geometry = merged_geometry
    if shrink_distance_m > 0:
        smoothed_geometry = smoothed_geometry.buffer(-shrink_distance_m, cap_style=2, join_style=2)
        if smoothed_geometry.is_empty:
            smoothed_geometry = merged_geometry
    if final_expand_m > 0:
        smoothed_geometry = smoothed_geometry.buffer(final_expand_m, cap_style=2, join_style=2)
    if simplify_tolerance_m > 0:
        smoothed_geometry = smoothed_geometry.simplify(simplify_tolerance_m, preserve_topology=True)

    smoothed_series = gpd.GeoSeries([smoothed_geometry], crs=METRIC_CRS).explode(index_parts=False)
    smoothed_series = smoothed_series[~smoothed_series.is_empty]
    if smoothed_series.empty:
        return _empty_gdf()

    array_gdf = gpd.GeoDataFrame(geometry=smoothed_series, crs=METRIC_CRS)
    array_gdf = array_gdf[array_gdf.geometry.area >= min_area_m2].copy()
    if array_gdf.empty:
        return _empty_gdf()

    array_gdf = array_gdf.assign(_area_m2=array_gdf.geometry.area)
    array_gdf = array_gdf.sort_values(by="_area_m2", ascending=False).drop(columns="_area_m2").reset_index(drop=True)
    return array_gdf.to_crs(OUTPUT_CRS)


def mask_to_array_geometry(
    mask: np.ndarray,
    chip: dict[str, Any],
    smooth_pixel_factor: float = 0.9,
) -> gpd.GeoDataFrame:
    """Convert a binary mask into a smoothed dissolved array polygon."""

    if mask.size == 0 or int(mask.max()) == 0:
        return _empty_gdf()

    polygon_shapes = [
        shape(geometry_mapping)
        for geometry_mapping, value in shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=chip["transform"])
        if int(value) == 1
    ]
    if not polygon_shapes:
        return _empty_gdf()

    chip_crs = chip.get("crs") or OUTPUT_CRS
    polygon_gdf = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(polygon_shapes, crs=chip_crs),
        crs=chip_crs,
    )
    polygon_gdf = polygon_gdf.explode(index_parts=False).reset_index(drop=True)
    polygon_gdf = polygon_gdf[~polygon_gdf.geometry.is_empty]
    if polygon_gdf.empty:
        return _empty_gdf()

    metric_gdf = polygon_gdf.to_crs("EPSG:3857")
    bounds = metric_gdf.total_bounds
    pixel_width = max((bounds[2] - bounds[0]) / mask.shape[1], 0.5)
    pixel_height = max((bounds[3] - bounds[1]) / mask.shape[0], 0.5)
    smooth_distance = max(pixel_width, pixel_height) * smooth_pixel_factor

    merged_geometry = unary_union(metric_gdf.geometry.tolist())
    smoothed_geometry = merged_geometry.buffer(smooth_distance).buffer(-smooth_distance)
    if smoothed_geometry.is_empty:
        smoothed_geometry = merged_geometry

    smoothed_series = gpd.GeoSeries([smoothed_geometry], crs="EPSG:3857").explode(index_parts=False)
    smoothed_series = smoothed_series[~smoothed_series.is_empty]
    if smoothed_series.empty:
        return _empty_gdf()

    smoothed_gdf = gpd.GeoDataFrame(geometry=smoothed_series, crs="EPSG:3857")
    return smoothed_gdf.to_crs(OUTPUT_CRS).reset_index(drop=True)


def _compute_catalog_coverage(
    catalog_gdf: gpd.GeoDataFrame,
    municipality_geometry: BaseGeometry,
) -> tuple[BaseGeometry | None, int, list[str]]:
    """Collapse intersecting raster footprints into one municipality coverage geometry."""

    candidates = catalog_gdf[catalog_gdf.geometry.intersects(municipality_geometry)].copy()
    if candidates.empty:
        return None, 0, []

    coverage_geometry = unary_union(candidates.geometry.tolist())
    coverage_geometry = coverage_geometry.intersection(municipality_geometry)
    sources = sorted(candidates["source"].dropna().astype(str).unique().tolist())
    return coverage_geometry, int(len(candidates)), sources


def _format_imagery_resolution_summary(candidates: gpd.GeoDataFrame) -> str:
    """Format municipality raster coverage as item-count : resolution pairs."""

    if candidates.empty:
        return "0 : n/a"

    summary = candidates.copy()
    summary["gsd_cm"] = (pd.to_numeric(summary["gsd"], errors="coerce") * 100.0).round()
    missing_resolution = summary["gsd_cm"].isna()
    if missing_resolution.any():
        summary.loc[missing_resolution, "gsd_cm"] = summary.loc[missing_resolution, "source"].map(SOURCE_GSD_CM_FALLBACKS)
    summary["resolution_bucket_cm"] = summary["gsd_cm"].map(_bucket_resolution_cm)
    grouped = (
        summary.groupby("resolution_bucket_cm", dropna=False)
        .size()
        .rename("imagery_count")
        .reset_index()
        .sort_values(by="resolution_bucket_cm", na_position="last")
    )

    parts: list[str] = []
    for row in grouped.itertuples(index=False):
        parts.append(f"{int(row.imagery_count):,} : {_format_bucketed_resolution_cm(row.resolution_bucket_cm)}")
    return "\n".join(parts)


def summarize_top_municipalities_for_poster(
    con,
    catalog_gdf: gpd.GeoDataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Assemble the poster's municipality summary table."""

    municipality_summary = con.execute(
        """
        SELECT
            municipality_name,
            COUNT(*)::BIGINT AS pv_feature_count,
            SUM(ST_Area(ST_Transform(geometry, 'EPSG:4326', 'EPSG:3857'))) AS pv_area_m2
        FROM pr_osm_rooftop_pv_polygons
        GROUP BY 1
        ORDER BY pv_feature_count DESC, municipality_name
        LIMIT ?
        """,
        [top_n],
    ).fetchdf()
    if municipality_summary.empty:
        raise RuntimeError("No municipality-level rooftop PV features were found.")

    building_counts = con.execute(
        """
        SELECT
            municipality_name,
            COUNT(*)::BIGINT AS building_count,
            SUM(ST_Area(ST_Transform(geometry, 'EPSG:4326', 'EPSG:32619'))) AS rooftop_area_m2
        FROM pr_overture_buildings
        GROUP BY 1
        """
    ).fetchdf()

    buildings_with_pv = con.execute(
        """
        SELECT
            b.municipality_name,
            COUNT(DISTINCT b.id)::BIGINT AS buildings_with_pv
        FROM pr_overture_buildings AS b
        JOIN pr_osm_rooftop_pv_polygons AS pv
          ON ST_Intersects(b.geometry, pv.geometry)
        GROUP BY 1
        """
    ).fetchdf()

    tract_counts = con.execute(
        """
        SELECT m.NAME AS municipality_name, COUNT(*)::BIGINT AS tract_count
        FROM pr_municipalities AS m
        JOIN pr_census_tracts AS t
          ON ST_Intersects(m.geometry, t.geometry)
        GROUP BY 1
        """
    ).fetchdf()

    block_group_counts = con.execute(
        """
        SELECT m.NAME AS municipality_name, COUNT(*)::BIGINT AS block_group_count
        FROM pr_municipalities AS m
        JOIN pr_block_groups AS g
          ON ST_Intersects(m.geometry, g.geometry)
        GROUP BY 1
        """
    ).fetchdf()

    municipality_gdf = fetch_geodataframe(
        con,
        """
        SELECT
            GEOID AS municipality_geoid,
            NAME AS municipality_name,
            ST_AsWKB(geometry) AS geometry_wkb
        FROM pr_municipalities
        ORDER BY municipality_name
        """
    )

    summary = municipality_summary.merge(building_counts, on="municipality_name", how="left")
    summary = summary.merge(buildings_with_pv, on="municipality_name", how="left")
    summary = summary.merge(tract_counts, on="municipality_name", how="left")
    summary = summary.merge(block_group_counts, on="municipality_name", how="left")
    summary = summary.merge(
        municipality_gdf[["municipality_name", "geometry"]],
        on="municipality_name",
        how="left",
    )

    summary["building_count"] = summary["building_count"].fillna(0).astype(int)
    summary["buildings_with_pv"] = summary["buildings_with_pv"].fillna(0).astype(int)
    summary["tract_count"] = summary["tract_count"].fillna(0).astype(int)
    summary["block_group_count"] = summary["block_group_count"].fillna(0).astype(int)
    summary["rooftop_area_m2"] = summary["rooftop_area_m2"].fillna(0.0)
    summary["municipality_area_km2"] = summary["geometry"].map(_geometry_area_km2)

    coverage_rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        municipality_candidates = catalog_candidates_for_geometry(
            catalog_gdf,
            row.geometry,
            preferred_sources=PREFERRED_POSTER_SOURCES,
            one_per_source=False,
        )

        coverage_rows.append(
            {
                "municipality_name": row.municipality_name,
                "raster_item_count": int(len(municipality_candidates)),
                "raster_sources": ", ".join(sorted(municipality_candidates["source"].dropna().astype(str).unique().tolist())),
                "imagery_resolution_label": _format_imagery_resolution_summary(municipality_candidates),
            }
        )

    summary = summary.merge(pd.DataFrame(coverage_rows), on="municipality_name", how="left")
    summary["raster_item_count"] = summary["raster_item_count"].fillna(0).astype(int)
    summary["pv_building_pct"] = np.where(
        summary["building_count"] > 0,
        100.0 * summary["buildings_with_pv"] / summary["building_count"],
        np.nan,
    )
    summary["rooftop_area_km2"] = summary["rooftop_area_m2"] / 1_000_000.0
    summary["census_units_label"] = summary.apply(
        lambda row: f"{int(row['block_group_count']):,} / {int(row['tract_count']):,}",
        axis=1,
    )

    return summary.drop(columns=["geometry"]).sort_values(
        ["pv_feature_count", "municipality_name"],
        ascending=[False, True],
    ).reset_index(drop=True)


def plot_top_municipality_table(
    summary: pd.DataFrame,
    title: str = "Top 5 municipalities by rooftop PV label count",
) -> plt.Figure:
    """Render a poster-ready summary table with compact metrics."""

    display_frame = pd.DataFrame(
        {
            "Municipality (Total Km2)": summary.apply(
                lambda row: f"{row['municipality_name']} ({row['municipality_area_km2']:.1f} km²)",
                axis=1,
            ),
            "# OSM PV": summary["pv_feature_count"].map(lambda value: f"{int(value):,}"),
            "# Buildings": summary["building_count"].map(lambda value: f"{int(value):,}"),
            "% w/ PV": summary["pv_building_pct"].map(
                lambda value: "n/a" if pd.isna(value) else f"{value:.1f}%"
            ),
            "Census BG/Tracts": summary["census_units_label"],
            "Imagery : Resolution": summary["imagery_resolution_label"],
            "Approx. Rooftop Area": summary["rooftop_area_km2"].map(
                lambda value: f"{value:.2f} km²"
            ),
        }
    )

    line_counts = display_frame.map(lambda value: str(value).count("\n") + 1).max(axis=1)
    figure_height = 2.7 + 0.72 * len(display_frame) + 0.22 * float(line_counts.sum() - len(display_frame))
    figure, axis = plt.subplots(figsize=(17.2, figure_height), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    figure.suptitle(title, fontsize=20, fontweight="bold", color="#0f172a", x=0.012, ha="left")

    headers = display_frame.columns.tolist()
    column_widths = [0.23, 0.085, 0.095, 0.095, 0.13, 0.17, 0.15]
    x_positions = np.cumsum([0.01, *column_widths[:-1]]).tolist()
    top_y = 0.84
    header_height = 0.11
    row_heights = [0.106 + 0.038 * (int(line_count) - 1) for line_count in line_counts]

    for x_position, width, header in zip(x_positions, column_widths, headers):
        axis.add_patch(
            Rectangle(
                (x_position, top_y),
                width,
                header_height,
                facecolor="#0f172a",
                edgecolor="white",
                linewidth=1.2,
            )
        )
        axis.text(
            x_position + width / 2,
            top_y + header_height / 2,
            header,
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
        )

    running_y = top_y
    for row_index, (_, row) in enumerate(display_frame.iterrows()):
        row_height = row_heights[row_index]
        row_y = running_y - row_height
        fill_color = "#f8fafc" if row_index % 2 == 0 else "#ecfeff"
        for column_index, (x_position, width, header) in enumerate(zip(x_positions, column_widths, headers)):
            axis.add_patch(
                Rectangle(
                    (x_position, row_y),
                    width,
                    row_height,
                    facecolor=fill_color,
                    edgecolor="#cbd5e1",
                    linewidth=0.9,
                )
            )

            text_value = str(row[header])
            if column_index == 0:
                axis.text(
                    x_position + 0.012,
                    row_y + row_height / 2,
                    text_value,
                    ha="left",
                    va="center",
                    fontsize=11,
                    color="#0f172a",
                    fontweight="semibold",
                )
            elif header == "Imagery : Resolution":
                axis.text(
                    x_position + 0.012,
                    row_y + row_height / 2,
                    text_value,
                    ha="left",
                    va="center",
                    fontsize=10.0,
                    color="#0f172a",
                    linespacing=1.25,
                )
            else:
                axis.text(
                    x_position + width / 2,
                    row_y + row_height / 2,
                    text_value,
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color="#0f172a",
                )

        running_y = row_y

    return figure


def _fallback_array_geometry(panel_rows_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve panel rows when mask-based vectorization produces no output."""

    if panel_rows_gdf.empty:
        return _empty_gdf()
    dissolved = unary_union(panel_rows_gdf.geometry.tolist())
    return gpd.GeoDataFrame(geometry=gpd.GeoSeries([dissolved], crs=OUTPUT_CRS), crs=OUTPUT_CRS)


def _filter_candidate_rows_by_catalog_coverage(
    candidate_frame: pd.DataFrame,
    coverage_geometry: BaseGeometry | None,
) -> pd.DataFrame:
    """Keep only candidate rows that intersect a preferred-source footprint union."""

    if candidate_frame.empty:
        return candidate_frame
    if coverage_geometry is None or coverage_geometry.is_empty:
        return candidate_frame.iloc[0:0].copy()

    filtered = candidate_frame.copy()
    filtered["candidate_geometry"] = filtered["geometry_wkb"].map(_to_wkb_bytes).map(from_wkb)
    filtered = filtered[
        filtered["candidate_geometry"].map(
            lambda geometry: geometry is not None and not geometry.is_empty and geometry.intersects(coverage_geometry)
        )
    ].copy()
    return filtered.drop(columns=["candidate_geometry"])


def _query_example_candidate_rows(
    con,
    top_municipality_limit: int | None,
    candidate_limit: int,
    candidate_offset: int = 0,
) -> pd.DataFrame:
    """Query building footprints with intersecting rooftop PV rows for review."""

    if top_municipality_limit is None:
        return con.execute(
            """
            SELECT
                b.id AS building_id,
                b.municipality_name,
                COUNT(DISTINCT pv.feature_id)::BIGINT AS pv_feature_count,
                SUM(
                    ST_Area(
                        ST_Transform(
                            ST_Intersection(b.geometry, pv.geometry),
                            'EPSG:4326',
                            'EPSG:32619'
                        )
                    )
                ) AS pv_overlap_area_m2,
                ST_AsWKB(b.geometry) AS geometry_wkb
            FROM pr_overture_buildings AS b
            JOIN pr_osm_rooftop_pv_polygons AS pv
              ON ST_Intersects(b.geometry, pv.geometry)
            GROUP BY 1, 2, 5
            ORDER BY pv_feature_count DESC, pv_overlap_area_m2 DESC, b.municipality_name, building_id
            LIMIT ?
                        OFFSET ?
            """,
                        [candidate_limit, candidate_offset],
        ).fetchdf()

    return con.execute(
        """
        WITH top_municipalities AS (
            SELECT municipality_name
            FROM pr_osm_rooftop_pv_polygons
            GROUP BY 1
            ORDER BY COUNT(*) DESC, municipality_name
            LIMIT ?
        )
        SELECT
            b.id AS building_id,
            b.municipality_name,
            COUNT(DISTINCT pv.feature_id)::BIGINT AS pv_feature_count,
            SUM(
                ST_Area(
                    ST_Transform(
                        ST_Intersection(b.geometry, pv.geometry),
                        'EPSG:4326',
                        'EPSG:32619'
                    )
                )
            ) AS pv_overlap_area_m2,
            ST_AsWKB(b.geometry) AS geometry_wkb
        FROM pr_overture_buildings AS b
        JOIN top_municipalities AS t
          ON b.municipality_name = t.municipality_name
        JOIN pr_osm_rooftop_pv_polygons AS pv
          ON ST_Intersects(b.geometry, pv.geometry)
        GROUP BY 1, 2, 5
        ORDER BY pv_feature_count DESC, pv_overlap_area_m2 DESC, b.municipality_name, building_id
        LIMIT ?
                OFFSET ?
        """,
                [top_municipality_limit, candidate_limit, candidate_offset],
    ).fetchdf()


def _score_example_candidate(metrics: dict[str, Any]) -> float:
    """Favor detailed imagery, larger panel sets, and fewer array fragments."""

    panel_row_count = float(metrics.get("panel_row_count", 0.0) or 0.0)
    array_polygon_count = int(metrics.get("array_polygon_count", 0) or 0)
    building_coverage_pct = float(metrics.get("building_panel_coverage_pct", 0.0) or 0.0)
    array_to_panel_ratio = metrics.get("array_to_panel_area_ratio")
    gsd_cm = metrics.get("gsd_cm")
    source = str(metrics.get("source", ""))

    row_score = min(panel_row_count / 28.0, 3.2)
    coverage_score = min(building_coverage_pct / 12.0, 2.0)
    contiguity_score = max(0.0, 2.25 - max(array_polygon_count - 1, 0) * 0.55)
    if array_to_panel_ratio is None or pd.isna(array_to_panel_ratio):
        ratio_score = 0.0
    else:
        ratio_score = max(0.0, 1.6 - abs(float(array_to_panel_ratio) - 1.22) * 3.5)
    if gsd_cm is None or pd.isna(gsd_cm):
        resolution_score = 0.0
    else:
        resolution_score = max(0.0, (120.0 - float(gsd_cm)) / 20.0)
    source_bonus = {"maxar_open_data": 0.6, "naip_2021_pr": 0.35, "satellogic_earthview": 0.2}.get(source, 0.0)

    return round(row_score + coverage_score + contiguity_score + ratio_score + resolution_score + source_bonus, 3)


def _materialize_example_from_candidate_row(
    con,
    catalog_gdf: gpd.GeoDataFrame,
    row: pd.Series,
    preferred_sources: list[str] | None = None,
    require_preferred_source: bool = False,
    chip_size: int = DEFAULT_CHIP_SIZE,
) -> PosterChipExample | None:
    """Build one poster example from a candidate building row."""

    building_geometry = from_wkb(_to_wkb_bytes(row["geometry_wkb"]))
    if building_geometry is None or building_geometry.is_empty:
        return None

    raster_candidates = catalog_candidates_for_geometry(
        catalog_gdf,
        building_geometry,
        preferred_sources=preferred_sources or PREFERRED_POSTER_SOURCES,
        one_per_source=False,
    )
    if raster_candidates.empty:
        return None

    if require_preferred_source and preferred_sources:
        raster_candidates = raster_candidates[raster_candidates["source"].isin(preferred_sources)].reset_index(drop=True)
        if raster_candidates.empty:
            return None

    selected_item = raster_candidates.iloc[0]
    panel_rows_gdf = fetch_geodataframe(
        con,
        """
        SELECT
            feature_id,
            municipality_name,
            municipality_geoid,
            ST_AsWKB(ST_Intersection(geometry, ST_GeomFromText(?))) AS geometry_wkb
        FROM pr_osm_rooftop_pv_polygons
        WHERE municipality_name = ?
          AND ST_Intersects(geometry, ST_GeomFromText(?))
        """,
        [building_geometry.wkt, row["municipality_name"], building_geometry.wkt],
    )
    if panel_rows_gdf.empty:
        return None

    panel_rows_gdf = panel_rows_gdf[~panel_rows_gdf.geometry.is_empty].reset_index(drop=True)
    if panel_rows_gdf.empty:
        return None

    chip = read_raster_chip_for_geometry(selected_item, building_geometry, out_size=chip_size)
    if chip is None:
        return None

    mask = build_binary_mask(chip, panel_rows_gdf.geometry, geometry_crs=OUTPUT_CRS)
    if int(mask.max()) == 0:
        return None

    array_gdf = build_array_geometry_from_panel_rows(panel_rows_gdf)
    if array_gdf.empty:
        array_gdf = mask_to_array_geometry(mask, chip, smooth_pixel_factor=1.2)
    if array_gdf.empty:
        array_gdf = _fallback_array_geometry(panel_rows_gdf)

    building_gdf = gpd.GeoDataFrame(
        [{"building_id": row["building_id"], "municipality_name": row["municipality_name"]}],
        geometry=gpd.GeoSeries([building_geometry], crs=OUTPUT_CRS),
        crs=OUTPUT_CRS,
    )

    gsd_cm = chip.get("gsd_cm") or _resolve_gsd_cm(selected_item, asset_href=str(chip["asset_href"]))

    return PosterChipExample(
        candidate_index=None,
        candidate_score=None,
        municipality_name=str(row["municipality_name"]),
        building_id=str(row["building_id"]),
        source=str(selected_item["source"]),
        gsd_cm=gsd_cm,
        item_id=str(selected_item["item_id"]),
        asset_href=str(chip["asset_href"]),
        building_gdf=building_gdf,
        panel_rows_gdf=panel_rows_gdf,
        array_gdf=array_gdf,
        mask=mask,
        chip=chip,
    )


def prepare_poster_chip_examples(
    con,
    catalog_gdf: gpd.GeoDataFrame,
    preferred_sources: list[str] | None = None,
    top_municipality_limit: int | None = 10,
    candidate_limit: int = 60,
    candidate_search_limit: int | None = None,
    query_batch_size: int | None = None,
    chip_size: int = DEFAULT_CHIP_SIZE,
    excluded_initial_ranks: list[int] | None = None,
    require_preferred_source: bool = False,
) -> tuple[pd.DataFrame, list[PosterChipExample]]:
    """Materialize and score multiple poster-example candidates for review."""

    candidate_pairs: list[tuple[dict[str, Any], PosterChipExample]] = []
    preferred_coverage_geometry: BaseGeometry | None = None
    if require_preferred_source and preferred_sources:
        preferred_catalog = catalog_gdf[catalog_gdf["source"].isin(preferred_sources)]
        if preferred_catalog.empty:
            raise RuntimeError("No catalog items were available for the requested preferred imagery sources.")
        preferred_coverage_geometry = unary_union(preferred_catalog.geometry.tolist())

    search_limit = max(int(candidate_limit), int(candidate_search_limit or candidate_limit))
    batch_size = max(1, int(query_batch_size or candidate_limit))
    candidate_offset = 0
    scanned_candidates = 0
    seen_building_ids: set[str] = set()

    while scanned_candidates < search_limit and len(candidate_pairs) < candidate_limit:
        current_batch_size = min(batch_size, search_limit - scanned_candidates)
        candidate_frame = _query_example_candidate_rows(
            con,
            top_municipality_limit,
            current_batch_size,
            candidate_offset=candidate_offset,
        )
        if candidate_frame.empty:
            break

        raw_batch_row_count = len(candidate_frame)
        candidate_offset += raw_batch_row_count
        scanned_candidates += raw_batch_row_count

        if preferred_coverage_geometry is not None:
            candidate_frame = _filter_candidate_rows_by_catalog_coverage(candidate_frame, preferred_coverage_geometry)

        if candidate_frame.empty:
            continue

        for _, row in candidate_frame.iterrows():
            building_id = str(row["building_id"])
            if building_id in seen_building_ids:
                continue
            seen_building_ids.add(building_id)

            example = _materialize_example_from_candidate_row(
                con,
                catalog_gdf,
                row,
                preferred_sources=preferred_sources,
                require_preferred_source=require_preferred_source,
                chip_size=chip_size,
            )
            if example is None:
                continue

            metadata = summarize_example_metadata(example).iloc[0].to_dict()
            metadata["pv_feature_count"] = int(row["pv_feature_count"])
            metadata["pv_overlap_area_m2"] = float(row["pv_overlap_area_m2"] or 0.0)
            metadata["source_label"] = _source_label(example.source)
            metadata["building_panel_coverage_pct"] = (
                100.0 * metadata["panel_row_area_m2"] / metadata["building_area_m2"]
                if metadata["building_area_m2"]
                else np.nan
            )
            metadata["array_to_panel_area_ratio"] = (
                metadata["array_area_m2"] / metadata["panel_row_area_m2"]
                if metadata["panel_row_area_m2"]
                else np.nan
            )
            metadata["candidate_score"] = _score_example_candidate(metadata)
            candidate_pairs.append((metadata, example))

            if len(candidate_pairs) >= candidate_limit:
                break

    if scanned_candidates == 0:
        raise RuntimeError("No building examples intersecting rooftop PV polygons were found.")

    if not candidate_pairs:
        raise RuntimeError("No readable raster-backed PV building example was found for the poster figure.")

    candidate_pairs.sort(key=lambda pair: pair[0]["candidate_score"], reverse=True)
    excluded_rank_set = {int(rank) for rank in (excluded_initial_ranks or [])}
    if excluded_rank_set:
        candidate_pairs = [pair for rank, pair in enumerate(candidate_pairs) if rank not in excluded_rank_set]
    if not candidate_pairs:
        raise RuntimeError("All scored poster candidates were excluded before manifest generation.")

    manifest_rows: list[dict[str, Any]] = []
    examples: list[PosterChipExample] = []
    for candidate_index, (metadata, example) in enumerate(candidate_pairs):
        metadata["candidate_iloc"] = candidate_index
        example.candidate_index = candidate_index
        example.candidate_score = float(metadata["candidate_score"])
        manifest_rows.append(metadata)
        examples.append(example)

    manifest = pd.DataFrame(manifest_rows)
    manifest = manifest[
        [
            "candidate_iloc",
            "candidate_score",
            "municipality_name",
            "building_id",
            "source",
            "source_label",
            "gsd_cm",
            "item_id",
            "asset_href",
            "pv_feature_count",
            "pv_overlap_area_m2",
            "panel_row_count",
            "array_polygon_count",
            "building_area_m2",
            "panel_row_area_m2",
            "array_area_m2",
            "building_panel_coverage_pct",
            "array_to_panel_area_ratio",
            "chip_height_px",
            "chip_width_px",
        ]
    ]
    return manifest, examples


def prepare_poster_chip_example(
    con,
    catalog_gdf: gpd.GeoDataFrame,
    preferred_sources: list[str] | None = None,
    top_municipality_limit: int | None = 10,
    candidate_limit: int = 60,
    candidate_search_limit: int | None = None,
    query_batch_size: int | None = None,
    chip_size: int = DEFAULT_CHIP_SIZE,
    candidate_iloc: int = 0,
    excluded_initial_ranks: list[int] | None = None,
    require_preferred_source: bool = False,
) -> PosterChipExample:
    """Select one scored building/PV/raster example for the poster panel."""

    manifest, examples = prepare_poster_chip_examples(
        con,
        catalog_gdf,
        preferred_sources=preferred_sources,
        top_municipality_limit=top_municipality_limit,
        candidate_limit=candidate_limit,
        candidate_search_limit=candidate_search_limit,
        query_batch_size=query_batch_size,
        chip_size=chip_size,
        excluded_initial_ranks=excluded_initial_ranks,
        require_preferred_source=require_preferred_source,
    )
    if candidate_iloc < 0 or candidate_iloc >= len(examples):
        raise IndexError(f"Requested candidate_iloc={candidate_iloc} is outside 0..{len(examples) - 1}.")
    return examples[candidate_iloc]


def select_review_candidate_ilocs(
    manifest: pd.DataFrame,
    explicit_ilocs: list[int] | None = None,
    random_seed: int | None = None,
    random_count: int = 0,
) -> list[int]:
    """Combine manual ilocs and a seeded random sample for review output."""

    selected_ilocs: list[int] = []
    max_index = len(manifest) - 1
    for candidate_iloc in explicit_ilocs or []:
        if 0 <= candidate_iloc <= max_index and candidate_iloc not in selected_ilocs:
            selected_ilocs.append(int(candidate_iloc))

    if random_seed is not None and random_count > 0 and len(manifest) > 0:
        remaining = [index for index in manifest["candidate_iloc"].astype(int).tolist() if index not in selected_ilocs]
        if remaining:
            rng = np.random.default_rng(random_seed)
            sample_size = min(random_count, len(remaining))
            sampled = sorted(rng.choice(remaining, size=sample_size, replace=False).tolist())
            selected_ilocs.extend(sampled)

    if not selected_ilocs:
        selected_ilocs = manifest["candidate_iloc"].astype(int).head(min(6, len(manifest))).tolist()
    return selected_ilocs


def summarize_example_metadata(example: PosterChipExample) -> pd.DataFrame:
    """Return one-row metadata summary for the selected poster chip example."""

    building_area_m2 = example.building_gdf.to_crs(METRIC_CRS).area.sum()
    panel_row_area_m2 = example.panel_rows_gdf.to_crs(METRIC_CRS).area.sum()
    array_area_m2 = example.array_gdf.to_crs(METRIC_CRS).area.sum() if not example.array_gdf.empty else 0.0
    array_ratio = (float(array_area_m2) / float(panel_row_area_m2)) if panel_row_area_m2 else np.nan
    building_coverage_pct = (100.0 * float(panel_row_area_m2) / float(building_area_m2)) if building_area_m2 else np.nan
    return pd.DataFrame(
        [
            {
                "candidate_iloc": example.candidate_index,
                "candidate_score": example.candidate_score,
                "municipality_name": example.municipality_name,
                "building_id": example.building_id,
                "source": example.source,
                "source_label": _source_label(example.source),
                "gsd_cm": example.gsd_cm,
                "item_id": example.item_id,
                "asset_href": example.asset_href,
                "panel_row_count": int(len(example.panel_rows_gdf)),
                "array_polygon_count": int(len(example.array_gdf)),
                "building_area_m2": float(building_area_m2),
                "panel_row_area_m2": float(panel_row_area_m2),
                "array_area_m2": float(array_area_m2),
                "array_to_panel_area_ratio": array_ratio,
                "building_panel_coverage_pct": building_coverage_pct,
                "chip_height_px": int(example.mask.shape[0]),
                "chip_width_px": int(example.mask.shape[1]),
            }
        ]
    )


def _chip_extent(chip: dict[str, Any]) -> tuple[float, float, float, float]:
    """Compute the plotting extent for an image chip."""

    image = np.asarray(chip["image"])
    height, width = image.shape[:2]
    left, top = chip["transform"] * (0, 0)
    right, bottom = chip["transform"] * (width, height)
    left, right = sorted([left, right])
    bottom, top = sorted([bottom, top])
    return left, right, bottom, top


def plot_three_panel_poster_example(example: PosterChipExample) -> plt.Figure:
    """Render the poster's image, mask, and vector panel sequence."""

    figure, axes = plt.subplots(1, 3, figsize=(18.8, 6.8), constrained_layout=False)
    chip = example.chip
    image = np.asarray(chip["image"])
    chip_crs = chip.get("crs") or OUTPUT_CRS
    extent = _chip_extent(chip)

    building_chip = example.building_gdf.to_crs(chip_crs)
    panel_rows_chip = example.panel_rows_gdf.to_crs(chip_crs)

    image_cmap = None if image.ndim == 3 else "gray"
    axes[0].imshow(image, extent=extent, origin="upper", cmap=image_cmap)
    panel_rows_chip.boundary.plot(ax=axes[0], color="#f97316", linewidth=1.05)
    building_chip.boundary.plot(ax=axes[0], color="#082f49", linewidth=3.8, alpha=0.95)
    building_chip.boundary.plot(ax=axes[0], color="#38bdf8", linewidth=2.2, alpha=0.98)
    axes[0].set_title("1. Image + building + panel rows", fontsize=13.5, fontweight="bold")
    axes[0].set_axis_off()

    axes[1].imshow(example.mask, extent=extent, origin="upper", cmap="gray", vmin=0, vmax=1)
    building_chip.boundary.plot(ax=axes[1], color="#22d3ee", linewidth=1.05)
    axes[1].set_title("2. Binary segmentation mask", fontsize=13.5, fontweight="bold")
    axes[1].set_axis_off()

    axes[2].set_facecolor("#f8fafc")
    example.building_gdf.boundary.plot(ax=axes[2], color="#64748b", linewidth=1.25, linestyle="--")
    if not example.array_gdf.empty:
        example.array_gdf.plot(
            ax=axes[2],
            color="#5eead4",
            alpha=0.62,
            edgecolor="#0f766e",
            linewidth=1.35,
        )
    example.panel_rows_gdf.boundary.plot(ax=axes[2], color="#ea580c", linewidth=0.95)
    axes[2].set_title("3. Array polygon + panel-row vectors", fontsize=13.5, fontweight="bold")
    axes[2].set_axis_off()
    axes[2].set_aspect("equal")

    minx, miny, maxx, maxy = example.building_gdf.total_bounds
    pad_x = max((maxx - minx) * 0.25, 0.00015)
    pad_y = max((maxy - miny) * 0.25, 0.00015)
    axes[2].set_xlim(minx - pad_x, maxx + pad_x)
    axes[2].set_ylim(miny - pad_y, maxy + pad_y)

    title_bits = [example.municipality_name, _source_label(example.source)]
    if example.gsd_cm is not None:
        title_bits.append(_format_resolution_cm(example.gsd_cm))
    figure.suptitle(" | ".join(title_bits), fontsize=17.0, fontweight="bold", color="#0f172a", y=0.985)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return figure


def plot_example_candidate_contact_sheet(
    examples: list[PosterChipExample],
    manifest: pd.DataFrame,
    selected_ilocs: list[int],
    title: str = "Poster example review set",
    columns: int = 3,
) -> plt.Figure:
    """Render an image-first contact sheet for manual example review."""

    review_examples = [examples[candidate_iloc] for candidate_iloc in selected_ilocs if 0 <= candidate_iloc < len(examples)]
    if not review_examples:
        raise RuntimeError("No review examples were available for the contact sheet.")

    row_count = int(np.ceil(len(review_examples) / columns))
    figure, axes = plt.subplots(row_count, columns, figsize=(6.35 * columns, 5.35 * row_count), constrained_layout=False)
    axes_array = np.asarray(axes).reshape(row_count, columns)

    for axis in axes_array.ravel():
        axis.axis("off")

    for axis, example in zip(axes_array.ravel(), review_examples):
        chip = example.chip
        image = np.asarray(chip["image"])
        chip_crs = chip.get("crs") or OUTPUT_CRS
        extent = _chip_extent(chip)
        building_chip = example.building_gdf.to_crs(chip_crs)
        panel_rows_chip = example.panel_rows_gdf.to_crs(chip_crs)

        axis.imshow(image, extent=extent, origin="upper", cmap=None if image.ndim == 3 else "gray")
        panel_rows_chip.boundary.plot(ax=axis, color="#f97316", linewidth=0.9)
        building_chip.boundary.plot(ax=axis, color="#0f172a", linewidth=2.0, alpha=0.92)
        building_chip.boundary.plot(ax=axis, color="#f8fafc", linewidth=1.1)
        axis.set_axis_off()

        manifest_row = manifest.loc[manifest["candidate_iloc"] == example.candidate_index].iloc[0]
        axis.set_title(
            f"#{int(example.candidate_index)} {example.municipality_name}\n"
            f"{_source_label(example.source)} | {_format_resolution_cm(example.gsd_cm)} | rows {int(manifest_row['panel_row_count'])}",
            fontsize=11,
            fontweight="bold",
            color="#0f172a",
        )
        axis.text(
            0.02,
            0.03,
            f"score {manifest_row['candidate_score']:.2f} · arrays {int(manifest_row['array_polygon_count'])} · ratio {manifest_row['array_to_panel_area_ratio']:.2f}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.1,
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=(1, 1, 1, 0.72), edgecolor="#cbd5e1", linewidth=0.8),
        )

    figure.suptitle(title, fontsize=18, fontweight="bold", color="#0f172a", y=0.99)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return figure
