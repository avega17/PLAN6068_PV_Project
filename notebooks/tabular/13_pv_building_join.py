# %% [markdown]
# # Building-Level PV Join with NSRDB Irradiance Summaries
#
# For each Overture building in `San Juan` + `Isabela`:
# - flag `has_pv_osm` if the building intersects `pr_osm_rooftop_pv_polygons`,
# - flag `has_pv_detected` + area from `pr_solar_pv_detections` and any
#   available local inference GeoJSON outputs,
# - preserve the source building H3 assignment from `pr_overture_buildings`,
# - attach nearest-site NSRDB irradiance summaries derived from the fetched
#   30-minute normalized dataset,
# - optionally emit monthly municipality animations for a selected irradiance
#   measure.
#
# Output: DuckDB table `pr_buildings_with_pv` with geometry in EPSG:4326.

# %%
"""01_pv_building_join.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from shapely import from_wkb


def resolve_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if any((candidate / m).exists() for m in ("project_rules.md", ".git")):
            return candidate
    return current


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from utils.census import resolve_vector_db_path

OUTPUT_TABLE = "pr_buildings_with_pv"
TARGET_MUNICIPALITIES = ("San Juan", "Isabela")
LOCAL_INFERENCE_ROOTS = (
    PROJECT_ROOT / "outputs" / "geoai_inference",
)
TARGET_SCOPE_STEM = "_".join(municipio.lower().replace(" ", "_") for municipio in TARGET_MUNICIPALITIES)
NSRDB_SUMMARY_COLUMNS = (
    "ghi",
    "dni",
    "dhi",
    "air_temperature",
    "clearsky_ghi",
    "clearsky_dni",
    "clearsky_dhi",
    "surface_albedo",
)
_env_nsrdb_root = os.getenv("NSRDB_ROOT")
NSRDB_ROOT = (
    (PROJECT_ROOT / _env_nsrdb_root)
    if _env_nsrdb_root and not Path(_env_nsrdb_root).is_absolute()
    else Path(_env_nsrdb_root or PROJECT_ROOT / "data" / "tabular" / "nsrdb")
)
NSRDB_NORMALIZED_ROOT = NSRDB_ROOT / "normalized"
NSRDB_MONTHLY_FLUX_PATH = NSRDB_NORMALIZED_ROOT / f"{TARGET_SCOPE_STEM}_nsrdb_monthly_flux_means.parquet"
NSRDB_ANNUAL_FLUX_PATH = NSRDB_NORMALIZED_ROOT / f"{TARGET_SCOPE_STEM}_nsrdb_annual_flux_means.parquet"
NSRDB_MULTIYEAR_FLUX_PATH = NSRDB_NORMALIZED_ROOT / f"{TARGET_SCOPE_STEM}_nsrdb_multiyear_flux_means.parquet"
NSRDB_ANIMATION_MEASURE = os.getenv("NSRDB_ANIMATION_MEASURE", "ghi").strip().lower()
NSRDB_ANIMATION_YEAR = int(os.getenv("NSRDB_ANIMATION_YEAR", "0") or "0")
WRITE_NSRDB_MONTHLY_ANIMATION = os.getenv("NSRDB_WRITE_MONTHLY_ANIMATION", "1") == "1"
MAP_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "maps"


def resolve_db_path() -> Path:
    return resolve_vector_db_path(PROJECT_ROOT)


def _to_bytes(v: object) -> bytes:
    if isinstance(v, memoryview):
        return v.tobytes()
    if isinstance(v, bytearray):
        return bytes(v)
    return bytes(v) if not isinstance(v, bytes) else v


def load_target_geometry(con: duckdb.DuckDBPyConnection):
    munis_sql = ", ".join(f"'{m}'" for m in TARGET_MUNICIPALITIES)
    row = con.execute(
        f"""
        SELECT ST_AsWKB(ST_Union_Agg(geometry)) AS wkb
        FROM pr_census_counties
        WHERE NAME IN ({munis_sql});
        """
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return from_wkb(_to_bytes(row[0]))


def load_target_municipalities(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    munis_sql = ", ".join(f"'{m}'" for m in TARGET_MUNICIPALITIES)
    df = con.execute(
        f"""
        SELECT
            NAME AS municipio,
            GEOID AS municipality_geoid,
            ST_AsWKB(geometry) AS wkb
        FROM pr_census_counties
        WHERE NAME IN ({munis_sql})
        ORDER BY NAME;
        """
    ).fetchdf()
    if df.empty:
        return gpd.GeoDataFrame(columns=["municipio", "municipality_geoid", "geometry"], geometry="geometry", crs="EPSG:4326")
    geoms = gpd.GeoSeries(df["wkb"].map(lambda v: from_wkb(_to_bytes(v))), crs="EPSG:4326")
    return gpd.GeoDataFrame(df.drop(columns=["wkb"]), geometry=geoms, crs="EPSG:4326")


def list_local_inference_vector_paths() -> list[Path]:
    preferred_paths: list[Path] = []
    preferred_stems: set[str] = set()
    fallback_paths: list[Path] = []

    for root in LOCAL_INFERENCE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*_pred_props.geojson")):
            preferred_paths.append(path)
            preferred_stems.add(path.name.removesuffix("_pred_props.geojson"))

    for root in LOCAL_INFERENCE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*_pred.geojson")):
            stem = path.name.removesuffix("_pred.geojson")
            if stem not in preferred_stems:
                fallback_paths.append(path)

    return preferred_paths + fallback_paths


def infer_detection_source(path: Path) -> str:
    relative_path = path.relative_to(PROJECT_ROOT)
    path_text = relative_path.as_posix()
    if path_text.startswith("outputs/geoai_inference/"):
        return "geoai_inference"
    if "smp_grounded" in path.parts:
        return "smp_grounded"
    return path.parent.name


def load_local_inference_vectors(target_geometry=None) -> gpd.GeoDataFrame:
    rows: list[gpd.GeoDataFrame] = []
    for index, path in enumerate(list_local_inference_vector_paths(), start=1):
        try:
            gdf = gpd.read_file(path)
        except Exception as exc:
            print(f"warning: failed to read local inference vector {path}: {exc}")
            continue
        if gdf.empty or "geometry" not in gdf.columns:
            continue

        gdf = gdf.loc[gdf.geometry.notna()].copy()
        if gdf.empty:
            continue
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        else:
            gdf = gdf.to_crs("EPSG:4326")
        gdf = gdf.loc[~gdf.geometry.is_empty].copy()
        if target_geometry is not None:
            gdf = gdf[gdf.intersects(target_geometry)].copy()
        if gdf.empty:
            continue

        gdf["detection_key"] = [f"local_{index}_{row_index}" for row_index in range(len(gdf))]
        gdf["inference_source"] = infer_detection_source(path)
        gdf["source_path"] = str(path.relative_to(PROJECT_ROOT))
        rows.append(gdf[["detection_key", "inference_source", "source_path", "geometry"]].copy())

    if not rows:
        return gpd.GeoDataFrame(columns=["detection_key", "inference_source", "source_path", "geometry"], geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame(pd.concat(rows, ignore_index=True), geometry="geometry", crs="EPSG:4326")


def stage_inference_vectors(con: duckdb.DuckDBPyConnection) -> None:
    target_geometry = load_target_geometry(con)
    local_vectors = load_local_inference_vectors(target_geometry=target_geometry)
    has_detection_table = bool(
        con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='pr_solar_pv_detections';"
        ).fetchone()[0]
    )

    if not local_vectors.empty:
        staged = pd.DataFrame(local_vectors.drop(columns=["geometry"]))
        staged["geometry_wkb"] = local_vectors.geometry.to_wkb()
        con.register("staged_local_inference_vectors", staged)
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE local_inference_vectors AS
            SELECT * EXCLUDE (geometry_wkb),
                   ST_GeomFromWKB(geometry_wkb) AS geometry
            FROM staged_local_inference_vectors;
            """
        )
    else:
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE local_inference_vectors AS
            SELECT CAST(NULL AS VARCHAR) AS detection_key,
                   CAST(NULL AS VARCHAR) AS inference_source,
                   CAST(NULL AS VARCHAR) AS source_path,
                   CAST(NULL AS GEOMETRY) AS geometry
            WHERE FALSE;
            """
        )

    if has_detection_table:
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE db_inference_vectors AS
            SELECT
                CONCAT('db_', CAST(ROW_NUMBER() OVER () AS VARCHAR)) AS detection_key,
                'pr_solar_pv_detections' AS inference_source,
                'duckdb:pr_solar_pv_detections' AS source_path,
                geometry
            FROM pr_solar_pv_detections
            WHERE geometry IS NOT NULL;
            """
        )
    else:
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE db_inference_vectors AS
            SELECT CAST(NULL AS VARCHAR) AS detection_key,
                   CAST(NULL AS VARCHAR) AS inference_source,
                   CAST(NULL AS VARCHAR) AS source_path,
                   CAST(NULL AS GEOMETRY) AS geometry
            WHERE FALSE;
            """
        )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE pv_inference_vectors AS
        SELECT detection_key, inference_source, source_path, geometry
        FROM db_inference_vectors
        UNION ALL
        SELECT detection_key, inference_source, source_path, geometry
        FROM local_inference_vectors;
        """
    )


# %%
def load_buildings(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    munis_sql = ", ".join(f"'{m}'" for m in TARGET_MUNICIPALITIES)
    df = con.execute(
        f"""
        SELECT
            b.id AS building_id,
            b.municipality_name AS municipio,
            b.municipality_geoid,
                        CAST(b.h3_cell_id AS VARCHAR) AS h3_cell_id,
                        CAST(b.h3_resolution AS INTEGER) AS h3_resolution,
            ST_AsWKB(b.geometry) AS wkb
        FROM pr_overture_buildings AS b
        WHERE b.municipality_name IN ({munis_sql})
          AND b.geometry IS NOT NULL;
        """
    ).fetchdf()
    if df.empty:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")
    geoms = gpd.GeoSeries(df["wkb"].map(lambda v: from_wkb(_to_bytes(v))), crs="EPSG:4326")
    return gpd.GeoDataFrame(df.drop(columns=["wkb"]), geometry=geoms, crs="EPSG:4326")


def build_pv_flags_table(con: duckdb.DuckDBPyConnection) -> None:
    munis_sql = ", ".join(f"'{m}'" for m in TARGET_MUNICIPALITIES)
    stage_inference_vectors(con)

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE pr_buildings_pv_flags AS
        WITH buildings AS (
            SELECT id AS building_id,
                   municipality_name AS municipio,
                   municipality_geoid,
                   CAST(h3_cell_id AS VARCHAR) AS h3_cell_id,
                   CAST(h3_resolution AS INTEGER) AS h3_resolution,
                   geometry
            FROM pr_overture_buildings
            WHERE municipality_name IN ({munis_sql}) AND geometry IS NOT NULL
        ),
        det_join AS (
            SELECT
                b.building_id,
                COUNT(DISTINCT d.detection_key) AS pv_detected_count,
                COUNT(DISTINCT d.inference_source) AS pv_detection_source_count,
                CASE
                    WHEN COUNT(DISTINCT d.detection_key) = 0 THEN 0.0
                    ELSE ST_Area(ST_Intersection(ANY_VALUE(b.geometry), ST_Union_Agg(d.geometry)))
                END AS pv_detected_area_deg2
            FROM buildings AS b
            LEFT JOIN pv_inference_vectors AS d
              ON ST_Intersects(b.geometry, d.geometry)
            GROUP BY b.building_id
        ),
        osm_join AS (
            SELECT b.building_id, TRUE AS has_pv_osm
            FROM buildings AS b
            JOIN pr_osm_rooftop_pv_polygons AS o
              ON ST_Intersects(b.geometry, o.geometry)
            GROUP BY b.building_id
        )
        SELECT
            b.building_id,
            b.municipio,
            b.municipality_geoid,
            b.h3_cell_id,
            b.h3_resolution,
            COALESCE(o.has_pv_osm, FALSE) AS has_pv_osm,
            COALESCE(d.pv_detected_count, 0) > 0 AS has_pv_detected,
            COALESCE(d.pv_detected_count, 0) AS pv_detected_count,
            COALESCE(d.pv_detection_source_count, 0) AS pv_detection_source_count,
            COALESCE(d.pv_detected_area_deg2, 0.0) AS pv_detected_area_deg2,
            COALESCE(o.has_pv_osm, FALSE) OR COALESCE(d.pv_detected_count, 0) > 0 AS has_pv_any,
            b.geometry
        FROM buildings AS b
        LEFT JOIN det_join AS d USING (building_id)
        LEFT JOIN osm_join AS o USING (building_id);
        """
    )


# %%
def _load_nsrdb_summary(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(required_columns))

    frame = pd.read_parquet(path)
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"NSRDB summary is missing required columns in {path.name}: {', '.join(missing)}")
    keep = [column_name for column_name in required_columns if column_name in frame.columns]
    return frame[keep].copy()


def load_nsrdb_multiyear_flux_means() -> pd.DataFrame:
    return _load_nsrdb_summary(
        NSRDB_MULTIYEAR_FLUX_PATH,
        ("site_id", "latitude", "longitude", *NSRDB_SUMMARY_COLUMNS),
    )


def load_nsrdb_annual_flux_means() -> pd.DataFrame:
    return _load_nsrdb_summary(
        NSRDB_ANNUAL_FLUX_PATH,
        ("site_id", "latitude", "longitude", "year", *NSRDB_SUMMARY_COLUMNS),
    )


def load_nsrdb_monthly_flux_means() -> pd.DataFrame:
    return _load_nsrdb_summary(
        NSRDB_MONTHLY_FLUX_PATH,
        ("site_id", "latitude", "longitude", "year", "month", *NSRDB_SUMMARY_COLUMNS),
    )


def attach_nsrdb_site_stats(
    con: duckdb.DuckDBPyConnection,
    buildings: gpd.GeoDataFrame,
) -> pd.DataFrame:
    empty_cols = [
        "building_id",
        "nsrdb_site_id",
        "nsrdb_site_latitude",
        "nsrdb_site_longitude",
        "nsrdb_site_distance_m",
        "nsrdb_annual_year",
        *[f"nsrdb_multiyear_{column_name}" for column_name in NSRDB_SUMMARY_COLUMNS],
        *[f"nsrdb_annual_{column_name}" for column_name in NSRDB_SUMMARY_COLUMNS],
    ]
    if buildings.empty:
        return pd.DataFrame(columns=empty_cols)

    multiyear = load_nsrdb_multiyear_flux_means()
    if multiyear.empty:
        print("no NSRDB multiyear summary parquet found — returning empty site stats.")
        return pd.DataFrame(columns=empty_cols)

    annual = load_nsrdb_annual_flux_means()
    latest_annual = pd.DataFrame(
        columns=[
            "site_id",
            "nsrdb_annual_year",
            *[f"nsrdb_annual_{column_name}" for column_name in NSRDB_SUMMARY_COLUMNS],
        ]
    )
    if not annual.empty:
        latest_year = int(pd.to_numeric(annual["year"], errors="coerce").dropna().max())
        latest_annual = annual.loc[annual["year"] == latest_year].copy()
        latest_annual = latest_annual.rename(
            columns={
                "year": "nsrdb_annual_year",
                **{
                    column_name: f"nsrdb_annual_{column_name}"
                    for column_name in NSRDB_SUMMARY_COLUMNS
                    if column_name in latest_annual.columns
                },
            }
        )
        latest_annual = latest_annual[
            [
                "site_id",
                "nsrdb_annual_year",
                *[
                    f"nsrdb_annual_{column_name}"
                    for column_name in NSRDB_SUMMARY_COLUMNS
                    if f"nsrdb_annual_{column_name}" in latest_annual.columns
                ],
            ]
        ].copy()

    site_frame = multiyear.rename(
        columns={
            "site_id": "nsrdb_site_id",
            "latitude": "nsrdb_site_latitude",
            "longitude": "nsrdb_site_longitude",
            **{
                column_name: f"nsrdb_multiyear_{column_name}"
                for column_name in NSRDB_SUMMARY_COLUMNS
                if column_name in multiyear.columns
            },
        }
    )
    site_frame = site_frame.merge(
        latest_annual,
        left_on="nsrdb_site_id",
        right_on="site_id",
        how="left",
    ).drop(columns=["site_id"], errors="ignore")

    sites = gpd.GeoDataFrame(
        site_frame,
        geometry=gpd.points_from_xy(site_frame["nsrdb_site_longitude"], site_frame["nsrdb_site_latitude"]),
        crs="EPSG:4326",
    )
    municipalities = load_target_municipalities(con)
    if municipalities.empty:
        return pd.DataFrame(columns=empty_cols)

    sites = gpd.sjoin(
        sites,
        municipalities[["municipio", "geometry"]],
        how="inner",
        predicate="within",
    ).drop(columns=["index_right"])
    if sites.empty:
        print("NSRDB site summaries did not intersect the target municipalities.")
        return pd.DataFrame(columns=empty_cols)

    building_points = buildings[["building_id", "municipio", "geometry"]].copy().to_crs("EPSG:3857")
    building_points.geometry = building_points.geometry.centroid
    sites_metric = sites.to_crs("EPSG:3857")

    site_value_columns = [
        "nsrdb_site_id",
        "nsrdb_site_latitude",
        "nsrdb_site_longitude",
        "nsrdb_annual_year",
        *[
            f"nsrdb_multiyear_{column_name}"
            for column_name in NSRDB_SUMMARY_COLUMNS
            if f"nsrdb_multiyear_{column_name}" in sites_metric.columns
        ],
        *[
            f"nsrdb_annual_{column_name}"
            for column_name in NSRDB_SUMMARY_COLUMNS
            if f"nsrdb_annual_{column_name}" in sites_metric.columns
        ],
        "geometry",
    ]
    rows: list[pd.DataFrame] = []
    for municipio in TARGET_MUNICIPALITIES:
        building_slice = building_points[building_points["municipio"] == municipio].copy()
        if building_slice.empty:
            continue

        site_slice = sites_metric[sites_metric["municipio"] == municipio].copy()
        if site_slice.empty:
            missing = building_slice[["building_id"]].copy()
            missing["nsrdb_site_id"] = pd.NA
            missing["nsrdb_site_latitude"] = pd.NA
            missing["nsrdb_site_longitude"] = pd.NA
            missing["nsrdb_site_distance_m"] = pd.NA
            missing["nsrdb_annual_year"] = pd.NA
            for column_name in NSRDB_SUMMARY_COLUMNS:
                missing[f"nsrdb_multiyear_{column_name}"] = pd.NA
                missing[f"nsrdb_annual_{column_name}"] = pd.NA
            rows.append(missing[empty_cols])
            continue

        joined = gpd.sjoin_nearest(
            building_slice,
            site_slice[site_value_columns],
            how="left",
            distance_col="nsrdb_site_distance_m",
        )
        rows.append(joined[empty_cols])

    if not rows:
        return pd.DataFrame(columns=empty_cols)
    return pd.concat(rows, ignore_index=True)


def write_monthly_nsrdb_animation(con: duckdb.DuckDBPyConnection) -> list[Path]:
    if not WRITE_NSRDB_MONTHLY_ANIMATION:
        return []

    monthly = load_nsrdb_monthly_flux_means()
    if monthly.empty:
        print("no NSRDB monthly summary parquet found — skipping animation export.")
        return []

    measure = NSRDB_ANIMATION_MEASURE
    if measure not in monthly.columns:
        supported = ", ".join(
            column_name for column_name in NSRDB_SUMMARY_COLUMNS if column_name in monthly.columns
        )
        raise RuntimeError(f"Unsupported NSRDB animation measure '{measure}'. Expected one of: {supported}")

    yearly_values = pd.to_numeric(monthly["year"], errors="coerce").dropna()
    if yearly_values.empty:
        print("NSRDB monthly summary did not include valid years — skipping animation export.")
        return []
    animation_year = NSRDB_ANIMATION_YEAR or int(yearly_values.max())

    frame = monthly.loc[
        monthly["year"] == animation_year,
        ["site_id", "latitude", "longitude", "month", measure],
    ].copy()
    frame = frame.dropna(subset=["latitude", "longitude", "month", measure])
    if frame.empty:
        print(f"no NSRDB monthly rows found for {animation_year} — skipping animation export.")
        return []

    sites = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]),
        crs="EPSG:4326",
    )
    municipalities = load_target_municipalities(con)
    if municipalities.empty:
        return []

    sites = gpd.sjoin(
        sites,
        municipalities[["municipio", "geometry"]],
        how="inner",
        predicate="within",
    ).drop(columns=["index_right"])
    if sites.empty:
        print("NSRDB monthly site summaries did not intersect the target municipalities.")
        return []

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import Normalize

    vmin = float(pd.to_numeric(sites[measure], errors="coerce").dropna().min())
    vmax = float(pd.to_numeric(sites[measure], errors="coerce").dropna().max())
    if vmin == vmax:
        vmax = vmin + 1.0

    written: list[Path] = []
    for municipio in TARGET_MUNICIPALITIES:
        muni_boundary = municipalities[municipalities["municipio"] == municipio]
        muni_sites = sites[sites["municipio"] == municipio].copy()
        if muni_boundary.empty or muni_sites.empty:
            continue

        months = sorted(pd.to_numeric(muni_sites["month"], errors="coerce").dropna().astype(int).unique().tolist())
        if not months:
            continue

        fig, (ax, cax) = plt.subplots(
            1,
            2,
            figsize=(8.5, 7.0),
            gridspec_kw={"width_ratios": [20, 1]},
        )
        norm = Normalize(vmin=vmin, vmax=vmax)
        scalar = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
        fig.colorbar(scalar, cax=cax, label=f"{measure.upper()} monthly mean")

        def _draw(month: int) -> None:
            ax.clear()
            muni_boundary.boundary.plot(ax=ax, color="#1f2937", linewidth=1.2)
            subset = muni_sites[muni_sites["month"] == month]
            if not subset.empty:
                subset.plot(
                    ax=ax,
                    column=measure,
                    cmap="viridis",
                    markersize=44,
                    vmin=vmin,
                    vmax=vmax,
                    legend=False,
                )
            ax.set_title(f"{municipio} NSRDB {measure.upper()} monthly mean ({animation_year}-{month:02d})")
            ax.set_axis_off()

        animation = FuncAnimation(
            fig,
            lambda frame_index: _draw(months[frame_index]),
            frames=len(months),
            interval=900,
            repeat_delay=1600,
        )
        slug = municipio.lower().replace(" ", "_")
        out_path = MAP_OUTPUT_DIR / f"{slug}_nsrdb_{measure}_{animation_year}_monthly.gif"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(out_path, writer=PillowWriter(fps=1))
        plt.close(fig)
        written.append(out_path)
        print(f"      wrote {out_path}")

    return written


# %%
# Notebook driver: connect, build flags, attach NSRDB summaries, persist.
if __name__ == "__main__":
    db_path = resolve_db_path()
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")

    print("[1/4] flagging OSM + detections …")
    build_pv_flags_table(con)

    local_vectors = load_local_inference_vectors(target_geometry=load_target_geometry(con))
    print(f"      local inference vectors discovered: {len(local_vectors):,}")

    print("[2/4] loading buildings for NSRDB site attachment …")
    buildings = load_buildings(con)
    print(f"      {len(buildings):,} buildings in {TARGET_MUNICIPALITIES}")

    print(f"[3/4] attaching NSRDB summaries from {NSRDB_NORMALIZED_ROOT} …")
    nsrdb_df = attach_nsrdb_site_stats(con, buildings)
    print(f"      NSRDB site matches for {len(nsrdb_df):,} buildings")

    con.register("nsrdb_building_stats", nsrdb_df)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {OUTPUT_TABLE} AS
        SELECT f.*,
               ns.nsrdb_site_id,
               ns.nsrdb_site_latitude,
               ns.nsrdb_site_longitude,
               ns.nsrdb_site_distance_m,
               ns.nsrdb_annual_year,
               ns.nsrdb_multiyear_ghi,
               ns.nsrdb_multiyear_dni,
               ns.nsrdb_multiyear_dhi,
               ns.nsrdb_multiyear_air_temperature,
               ns.nsrdb_multiyear_clearsky_ghi,
               ns.nsrdb_multiyear_clearsky_dni,
               ns.nsrdb_multiyear_clearsky_dhi,
               ns.nsrdb_multiyear_surface_albedo,
               ns.nsrdb_annual_ghi,
               ns.nsrdb_annual_dni,
               ns.nsrdb_annual_dhi,
               ns.nsrdb_annual_air_temperature,
               ns.nsrdb_annual_clearsky_ghi,
               ns.nsrdb_annual_clearsky_dni,
               ns.nsrdb_annual_clearsky_dhi,
               ns.nsrdb_annual_surface_albedo
        FROM pr_buildings_pv_flags AS f
        LEFT JOIN nsrdb_building_stats AS ns USING (building_id);
        """
    )
    con.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_geom ON {OUTPUT_TABLE} USING RTREE (geometry);"
    )
    con.unregister("nsrdb_building_stats")

    # %%
    summary = con.execute(
        f"""
        SELECT municipio,
               COUNT(*) AS buildings,
               SUM(CAST(has_pv_osm AS INT)) AS osm_labeled,
               SUM(CAST(has_pv_detected AS INT)) AS model_detected,
               SUM(CAST(has_pv_any AS INT)) AS any_pv_signal,
               SUM(CAST(has_pv_osm AND has_pv_detected AS INT)) AS overlap,
               AVG(nsrdb_multiyear_ghi) AS avg_nsrdb_multiyear_ghi,
               AVG(nsrdb_multiyear_dni) AS avg_nsrdb_multiyear_dni,
               AVG(nsrdb_multiyear_dhi) AS avg_nsrdb_multiyear_dhi
        FROM {OUTPUT_TABLE}
        GROUP BY 1
        ORDER BY 1;
        """
    ).fetchdf()
    print("\npr_buildings_with_pv summary:")
    print(summary.to_string(index=False))
    print("[4/4] exporting municipality NSRDB monthly animations …")
    write_monthly_nsrdb_animation(con)
    con.close()
