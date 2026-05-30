"""Export presentation datasets and summary artifacts.

This script derives compact, shareable vectors and summary tables from the
local project DuckDB and raster catalog for the final slide deck.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "PR_PV_plan_data.duckdb"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "presentation"
DEFAULT_MAX_BUILDING_BYTES = 9_000_000
SAN_JUAN_EXTRA_SAMPLE_FRACTION = 0.5
OUTPUT_CRS = "EPSG:4326"
PLOT_CRS = "EPSG:3857"
CASE_MUNICIPALITIES = ("San Juan", "Isabela")
HIGH_COVERAGE_AREAS = (
    ("Puerto Nuevo, San Juan", "Puerto Nuevo, San Juan, Puerto Rico", "San Juan"),
    ("Mora, Isabela", "Mora, Isabela, Puerto Rico", "Isabela"),
)
BUILDING_COLUMNS = [
    "id AS building_id",
    "municipality_name",
    "municipality_geoid",
    "subtype",
    "class",
    "height",
    "num_floors",
    "roof_material",
    "roof_direction",
    "roof_orientation",
    "roof_height",
    "ST_X(ST_PointOnSurface(geometry)) AS lon",
    "ST_Y(ST_PointOnSurface(geometry)) AS lat",
    "ST_AsWKB(geometry) AS geometry_wkb",
]
PARQUET_COMPRESSION_OPTIONS = {
    "compression": "zstd",
    "compression_level": 19,
    "use_dictionary": True,
}
DATABASE_SUMMARY_TABLES = [
    (
        "Spatial frame",
        "Puerto Rico municipalities",
        "pr_census_counties",
        "Municipal boundaries used for case-study selection and joining datasets by municipio.",
    ),
    (
        "Spatial frame",
        "Census tracts",
        "pr_census_tracts",
        "Tract units used for LISA, Moran's I, and socioeconomic interpretation.",
    ),
    (
        "Spatial frame",
        "Census block groups",
        "pr_census_block_groups",
        "Finer planning units used for aggregation and future equity analysis.",
    ),
    (
        "PV labels",
        "OSM rooftop PV polygons",
        "pr_osm_rooftop_pv_polygons",
        "Crowdsourced weak labels for training, validation, and exploratory adoption evidence.",
    ),
    (
        "Building denominator",
        "Overture building footprints",
        "pr_overture_buildings",
        "Building units used to normalize PV evidence and translate detections into planning metrics.",
    ),
    (
        "Building denominator",
        "Buildings joined to PV evidence",
        "pr_buildings_with_pv",
        "Candidate building records that connect footprint geometry with PV label or model evidence.",
    ),
    (
        "Imagery workflow",
        "Solar tile manifest",
        "pr_solar_tile_manifest",
        "H3/tile-level imagery plan used to organize training, validation, and inference imagery.",
    ),
    (
        "Model output",
        "Model detection tiles",
        "pr_solar_pv_detection_tiles",
        "Tiles where model inference results are tracked before aggregation to buildings and Census units.",
    ),
    (
        "Model output",
        "All-model PV detections",
        "pr_solar_pv_detections_all_models",
        "Detection records from the model comparison workflow across architectures.",
    ),
    (
        "Planning aggregates",
        "PV tract aggregates",
        "pr_pv_tract_aggregates",
        "Tract-level PV, building, socioeconomic, and context metrics used in spatial analysis.",
    ),
    (
        "Planning aggregates",
        "PV block-group aggregates",
        "pr_pv_bg_aggregates",
        "Block-group metrics prepared for finer follow-up planning interpretation.",
    ),
]


def _to_bytes(value: object) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def fetch_gdf(con: duckdb.DuckDBPyConnection, query: str, params: list[object] | None = None) -> gpd.GeoDataFrame:
    """Run SQL returning geometry_wkb and rebuild a GeoDataFrame."""

    frame = con.execute(query, params or []).fetchdf()
    if frame.empty:
        return gpd.GeoDataFrame(
            frame.drop(columns=["geometry_wkb"], errors="ignore"),
            geometry=gpd.GeoSeries([], crs=OUTPUT_CRS),
            crs=OUTPUT_CRS,
        )

    geometry = gpd.GeoSeries.from_wkb(frame["geometry_wkb"].map(_to_bytes), crs=OUTPUT_CRS)
    return gpd.GeoDataFrame(frame.drop(columns=["geometry_wkb"]), geometry=geometry, crs=OUTPUT_CRS)


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def write_buildings_parquet(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    gdf.to_parquet(output_path, index=False, **PARQUET_COMPRESSION_OPTIONS)


def export_core_vectors(con: duckdb.DuckDBPyConnection, output_dir: Path) -> dict[str, int]:
    """Export municipality, OSM PV, and high-coverage barrio context."""

    output_dir.mkdir(parents=True, exist_ok=True)
    case_boundaries = fetch_gdf(
        con,
        """
        SELECT GEOID AS municipality_geoid, NAME AS municipality_name, ST_AsWKB(geometry) AS geometry_wkb
        FROM pr_census_counties
        WHERE NAME IN (?, ?)
        ORDER BY NAME
        """,
        list(CASE_MUNICIPALITIES),
    )
    case_boundaries.to_file(output_dir / "case_study_municipalities.geojson", driver="GeoJSON")

    case_pv = fetch_gdf(
        con,
        """
        SELECT
            feature_id,
            municipality_name,
            municipality_geoid,
            ST_AsWKB(geometry) AS geometry_wkb
        FROM pr_osm_rooftop_pv_polygons
        WHERE municipality_name IN (?, ?)
        ORDER BY municipality_name, feature_id
        """,
        list(CASE_MUNICIPALITIES),
    )
    if not case_pv.empty:
        case_pv["pv_area_m2"] = case_pv.to_crs(PLOT_CRS).geometry.area.round(2).to_numpy()
    case_pv.to_file(output_dir / "case_study_osm_pv_polygons.geojson", driver="GeoJSON")

    barrio_records = []
    try:
        import osmnx as ox

        for label, query, municipality in HIGH_COVERAGE_AREAS:
            try:
                barrio = ox.geocode_to_gdf(query).to_crs(OUTPUT_CRS)
            except Exception as exc:
                print(f"Warning: could not geocode {label}: {exc}")
                continue
            if barrio.empty:
                continue
            barrio = barrio[["geometry"]].copy()
            barrio["study_area_label"] = label
            barrio["municipality_name"] = municipality
            barrio_records.append(barrio)
    except Exception as exc:
        print(f"Warning: osmnx unavailable for barrio export: {exc}")

    if barrio_records:
        barrios = gpd.GeoDataFrame(pd.concat(barrio_records, ignore_index=True), geometry="geometry", crs=OUTPUT_CRS)
        barrios.to_file(output_dir / "case_study_high_coverage_barrios.geojson", driver="GeoJSON")
    else:
        (output_dir / "case_study_high_coverage_barrios.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": []}, indent=2)
        )

    return {
        "case_municipality_count": int(len(case_boundaries)),
        "case_pv_polygon_count": int(len(case_pv)),
    }


def export_stac_subset(output_dir: Path) -> int:
    """Export a compact case-study STAC footprint subset for map diagnostics."""

    stac_path = PROJECT_ROOT / "data" / "rasters" / "stac" / "pr_raster_catalog_items.parquet"
    case_path = output_dir / "case_study_municipalities.geojson"
    output_path = output_dir / "case_study_stac_footprints.geojson"
    if not stac_path.exists() or not case_path.exists():
        output_path.write_text(json.dumps({"type": "FeatureCollection", "features": []}, indent=2))
        return 0

    stac = gpd.read_parquet(stac_path).to_crs(OUTPUT_CRS)
    case_boundaries = gpd.read_file(case_path).to_crs(OUTPUT_CRS)
    stac_subset = stac[stac.geometry.intersects(case_boundaries.geometry.union_all())].copy()
    if "acquired_at" in stac_subset.columns:
        stac_subset["acquired_at"] = pd.to_datetime(stac_subset["acquired_at"], utc=True, errors="coerce").astype(str)

    keep_cols = [
        column
        for column in [
            "source",
            "item_id",
            "collection_id",
            "acquired_at",
            "gsd",
            "visual_asset_href",
            "analytic_asset_href",
            "geometry",
        ]
        if column in stac_subset.columns
    ]
    stac_subset = stac_subset[keep_cols]
    stac_subset = stac_subset.groupby("source", group_keys=False).head(80).reset_index(drop=True)
    stac_subset.to_file(output_path, driver="GeoJSON")
    return int(len(stac_subset))


def export_database_summary(con: duckdb.DuckDBPyConnection, output_dir: Path) -> int:
    """Export curated row counts for the full research database."""

    records = []
    for summary_group, component, source_table, planning_use in DATABASE_SUMMARY_TABLES:
        record_count = con.execute(f"SELECT COUNT(*) FROM {source_table}").fetchone()[0]
        records.append(
            {
                "summary_group": summary_group,
                "component": component,
                "source_table": source_table,
                "record_count": int(record_count),
                "planning_use": planning_use,
            }
        )
    summary = pd.DataFrame(records)
    summary.to_csv(output_dir / "research_database_summary.csv", index=False)
    return int(len(summary))


def building_scope_sql(scope: str, output_dir: Path) -> tuple[str, list[object], str]:
    """Return WHERE clause, params, and label for a building-export scope."""

    if scope == "case_municipalities":
        return "municipality_name IN (?, ?)", list(CASE_MUNICIPALITIES), "San Juan + Isabela"
    if scope == "san_juan":
        return "municipality_name = ?", ["San Juan"], "San Juan"
    if scope == "isabela":
        return "municipality_name = ?", ["Isabela"], "Isabela"
    if scope == "high_coverage_barrios":
        barrios_path = output_dir / "case_study_high_coverage_barrios.geojson"
        if not barrios_path.exists():
            raise FileNotFoundError("High-coverage barrio GeoJSON must be exported before barrio building scope.")
        barrios = gpd.read_file(barrios_path).to_crs(OUTPUT_CRS)
        if barrios.empty:
            raise RuntimeError("High-coverage barrio GeoJSON is empty; cannot export barrio building scope.")
        wkt_parts = barrios.geometry.to_wkt().tolist()
        intersects_sql = " OR ".join(["ST_Intersects(geometry, ST_GeomFromText(?))"] * len(wkt_parts))
        return f"({intersects_sql})", wkt_parts, "high-coverage barrios"
    raise ValueError(f"Unknown building export scope: {scope}")


def high_coverage_area_wkt(output_dir: Path, label: str) -> str:
    barrios_path = output_dir / "case_study_high_coverage_barrios.geojson"
    if not barrios_path.exists():
        raise FileNotFoundError("High-coverage barrio GeoJSON must be exported before sampled building scopes.")
    barrios = gpd.read_file(barrios_path).to_crs(OUTPUT_CRS)
    match = barrios[barrios["study_area_label"].eq(label)]
    if match.empty:
        raise RuntimeError(f"Missing high-coverage area boundary: {label}")
    return match.geometry.iloc[0].wkt


def fetch_buildings_for_scope(con: duckdb.DuckDBPyConnection, scope: str, output_dir: Path) -> tuple[gpd.GeoDataFrame, str]:
    """Fetch Overture buildings for a named presentation scope."""

    where_sql, params, label = building_scope_sql(scope, output_dir)
    query = f"""
        SELECT
            {", ".join(BUILDING_COLUMNS)}
        FROM pr_overture_buildings
        WHERE {where_sql}
          AND geometry IS NOT NULL
          AND ST_IsValid(geometry)
        ORDER BY municipality_name, id
    """
    return fetch_gdf(con, query, params), label


def fetch_san_juan_puerto_nuevo_buildings(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> tuple[gpd.GeoDataFrame, str]:
    """Fetch San Juan buildings with a Puerto Nuevo coverage flag."""

    puerto_nuevo_wkt = high_coverage_area_wkt(output_dir, "Puerto Nuevo, San Juan")
    query = f"""
        SELECT
            {", ".join(BUILDING_COLUMNS)},
            CAST(ST_Intersects(geometry, ST_GeomFromText(?)) AS BOOLEAN) AS in_puerto_nuevo
        FROM pr_overture_buildings
        WHERE municipality_name = ?
          AND geometry IS NOT NULL
          AND ST_IsValid(geometry)
        ORDER BY
            CASE WHEN ST_Intersects(geometry, ST_GeomFromText(?)) THEN 0 ELSE 1 END,
            md5(id)
    """
    return fetch_gdf(con, query, [puerto_nuevo_wkt, "San Juan", puerto_nuevo_wkt]), "San Juan: Puerto Nuevo + sampled buildings"


def select_sampled_buildings_under_limit(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    max_building_bytes: int,
    extra_row_fraction: float = 1.0,
) -> dict[str, int]:
    """Write all Puerto Nuevo buildings plus a deterministic San Juan sample under a byte limit."""

    puerto_nuevo = gdf[gdf["in_puerto_nuevo"]].copy()
    san_juan_sample_pool = gdf[~gdf["in_puerto_nuevo"]].copy()

    low = 0
    high = len(san_juan_sample_pool)
    best_extra_rows = 0
    best_size = 0
    while low <= high:
        midpoint = (low + high) // 2
        probe = pd.concat([puerto_nuevo, san_juan_sample_pool.head(midpoint)], ignore_index=True)
        write_buildings_parquet(probe, output_path)
        probe_size = file_size(output_path)
        if probe_size <= max_building_bytes:
            best_extra_rows = midpoint
            best_size = probe_size
            low = midpoint + 1
        else:
            high = midpoint - 1

    selected_extra_rows = best_extra_rows
    if best_extra_rows:
        selected_extra_rows = max(1, int(best_extra_rows * extra_row_fraction))

    selected = pd.concat([puerto_nuevo, san_juan_sample_pool.head(selected_extra_rows)], ignore_index=True)
    write_buildings_parquet(selected, output_path)
    return {
        "puerto_nuevo_rows": int(len(puerto_nuevo)),
        "max_extra_san_juan_sample_rows": int(best_extra_rows),
        "selected_extra_san_juan_sample_rows": int(selected_extra_rows),
        "selected_rows": int(len(selected)),
        "selected_bytes": int(file_size(output_path) or best_size),
    }


def write_duckdb_extract(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Write one compact DuckDB extract table for size comparison."""

    output_path.unlink(missing_ok=True)
    frame = pd.DataFrame(gdf.drop(columns="geometry"))
    frame["geometry_wkb"] = gdf.geometry.to_wkb()
    con = duckdb.connect(str(output_path))
    con.register("building_extract", frame)
    con.execute(
        """
        CREATE TABLE case_study_overture_buildings AS
        SELECT
            * EXCLUDE (geometry_wkb),
            ST_GeomFromWKB(geometry_wkb) AS geometry
        FROM building_extract
        """
    )
    con.execute("CHECKPOINT;")
    con.close()


def export_building_candidates(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
    max_building_bytes: int,
    keep_candidate_files: bool = False,
) -> dict[str, object]:
    """Evaluate compressed building-footprint export scopes and keep the best fit."""

    candidate_dir = Path(tempfile.mkdtemp(prefix="presentation_building_exports_", dir=str(output_dir)))
    scopes = ["case_municipalities", "san_juan", "san_juan_puerto_nuevo_sample", "isabela", "high_coverage_barrios"]
    candidate_rows: list[dict[str, object]] = []
    chosen: dict[str, object] | None = None

    try:
        for scope in scopes:
            parquet_path = candidate_dir / f"{scope}.parquet"
            duckdb_path = candidate_dir / f"{scope}.duckdb"
            row_extras: dict[str, object] = {}
            if scope == "san_juan_puerto_nuevo_sample":
                gdf, label = fetch_san_juan_puerto_nuevo_buildings(con, output_dir)
                sample_stats = select_sampled_buildings_under_limit(
                    gdf,
                    parquet_path,
                    max_building_bytes,
                    extra_row_fraction=SAN_JUAN_EXTRA_SAMPLE_FRACTION,
                )
                selected_gdf = pd.read_parquet(parquet_path)
                geometry = gpd.GeoSeries.from_wkb(selected_gdf["geometry"].map(_to_bytes), crs=OUTPUT_CRS)
                gdf = gpd.GeoDataFrame(selected_gdf.drop(columns="geometry"), geometry=geometry, crs=OUTPUT_CRS)
                row_extras.update(sample_stats)
            else:
                gdf, label = fetch_buildings_for_scope(con, scope, output_dir)
                write_buildings_parquet(gdf, parquet_path)
            write_duckdb_extract(gdf, duckdb_path)
            row = {
                "scope": scope,
                "label": label,
                "rows": int(len(gdf)),
                "parquet_bytes": int(file_size(parquet_path)),
                "duckdb_bytes": int(file_size(duckdb_path)),
            }
            row.update(row_extras)
            candidate_rows.append(row)
            if chosen is None and row["parquet_bytes"] <= max_building_bytes:
                chosen = row
                shutil.copy2(parquet_path, output_dir / "case_study_overture_buildings.parquet")
                if keep_candidate_files:
                    shutil.copy2(duckdb_path, output_dir / "case_study_overture_buildings.duckdb")

        if chosen is None:
            raise RuntimeError(f"No building export candidate fit under {max_building_bytes:,} bytes.")
    finally:
        if not keep_candidate_files:
            shutil.rmtree(candidate_dir, ignore_errors=True)

    return {
        "building_export_candidates": candidate_rows,
        "selected_building_export": chosen,
        "building_export_max_bytes": int(max_building_bytes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-building-bytes", type=int, default=DEFAULT_MAX_BUILDING_BYTES)
    parser.add_argument("--keep-duckdb", action="store_true", help="Also keep the selected DuckDB extract for inspection.")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(args.db_path), read_only=True)
    con.execute("LOAD spatial;")
    manifest: dict[str, object] = {}
    manifest["research_database_summary_rows"] = export_database_summary(con, output_dir)
    manifest.update(export_core_vectors(con, output_dir))
    manifest.update(export_building_candidates(con, output_dir, args.max_building_bytes, keep_candidate_files=args.keep_duckdb))
    con.close()

    output_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    manifest["output_files"] = [str(path.relative_to(output_dir)) for path in output_files]
    manifest["output_file_bytes"] = {str(path.relative_to(output_dir)): int(file_size(path)) for path in output_files}
    manifest_path = output_dir / "presentation_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()