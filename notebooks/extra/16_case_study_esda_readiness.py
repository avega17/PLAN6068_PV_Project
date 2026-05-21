# %% [markdown]
# # Case-Study ESDA Readiness Audit
#
# Freezes the core analysis unit at the San Juan + Isabela census block-group
# aggregate and writes a pragmatic readiness report for the 3-day ESDA sprint.
#
# Outputs:
# - `outputs/reports/case_study_esda_readiness.md`
# - `outputs/reports/case_study_esda_data_readiness.csv`
# - `outputs/reports/case_study_esda_variable_audit.csv`
# - `outputs/reports/case_study_esda_method_matrix.csv`

# %%
"""16_case_study_esda_readiness.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv


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

from utils.acs import artifact_path_for_acs, table_name_for_acs

TARGET_MUNICIPALITIES = ("San Juan", "Isabela")
ACS_VINTAGE = 2024
COUNTY_TABLE = "pr_census_counties"
BLOCK_GROUP_TABLE = "pr_census_block_groups"
URBAN_FLAGS_TABLE = "pr_bg_urban_flags"
BUILDING_JOIN_TABLE = "pr_buildings_with_pv"
BG_AGG_TABLE = "pr_pv_bg_aggregates"
OVERTURE_TABLE = "pr_overture_buildings"
OSM_PV_TABLE = "pr_osm_rooftop_pv_polygons"
DETECTIONS_TABLE = "pr_solar_pv_detections"

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
READINESS_CSV_PATH = REPORT_DIR / "case_study_esda_data_readiness.csv"
VARIABLE_AUDIT_CSV_PATH = REPORT_DIR / "case_study_esda_variable_audit.csv"
METHOD_MATRIX_CSV_PATH = REPORT_DIR / "case_study_esda_method_matrix.csv"
REPORT_PATH = REPORT_DIR / "case_study_esda_readiness.md"

NSRDB_NORMALIZED_ROOT = PROJECT_ROOT / "data" / "tabular" / "nsrdb" / "normalized"


def resolve_db_path() -> Path:
    env_value = os.getenv("VECTOR_DB")
    if env_value:
        candidate = Path(env_value)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / env_value if len(candidate.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / candidate
        return candidate
    return PROJECT_ROOT / "data" / "PR_PV_plan_data.duckdb"


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")
    return con


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    return bool(
        con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?;",
            [table_name],
        ).fetchone()[0]
    )


def list_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
    if not table_exists(con, table_name):
        return []
    rows = con.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ?
        ORDER BY ordinal_position;
        """,
        [table_name],
    ).fetchall()
    return [column_name for (column_name,) in rows]


def count_rows(con: duckdb.DuckDBPyConnection, table_name: str) -> int | None:
    if not table_exists(con, table_name):
        return None
    return int(con.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0])


def first_existing_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match is not None:
            return match
    return None


def count_rows_by_target(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    *,
    column_candidates: tuple[str, ...],
) -> dict[str, int]:
    columns = list_columns(con, table_name)
    name_column = first_existing_column(columns, column_candidates)
    if name_column is None:
        return {}

    names_sql = ", ".join("?" * len(TARGET_MUNICIPALITIES))
    frame = con.execute(
        f"""
        SELECT CAST({name_column} AS VARCHAR) AS municipio, COUNT(*) AS row_count
        FROM {table_name}
        WHERE CAST({name_column} AS VARCHAR) IN ({names_sql})
        GROUP BY 1
        ORDER BY 1;
        """,
        list(TARGET_MUNICIPALITIES),
    ).fetchdf()
    return {row.municipio: int(row.row_count) for row in frame.itertuples(index=False)}


def count_target_block_groups(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    if not table_exists(con, COUNTY_TABLE) or not table_exists(con, BLOCK_GROUP_TABLE):
        return {}

    names_sql = ", ".join("?" * len(TARGET_MUNICIPALITIES))
    frame = con.execute(
        f"""
        WITH muni AS (
            SELECT NAME AS municipio, geometry AS municipio_geom
            FROM {COUNTY_TABLE}
            WHERE NAME IN ({names_sql})
        ),
        assigned AS (
            SELECT
                m.municipio,
                bg.GEOID AS bg_geoid,
                ROW_NUMBER() OVER (
                    PARTITION BY bg.GEOID
                    ORDER BY ST_Area(ST_Intersection(bg.geometry, m.municipio_geom)) DESC
                ) AS overlap_rank
            FROM {BLOCK_GROUP_TABLE} AS bg
            JOIN muni AS m
              ON ST_Intersects(bg.geometry, m.municipio_geom)
            WHERE ST_Area(ST_Intersection(bg.geometry, m.municipio_geom)) / NULLIF(ST_Area(bg.geometry), 0) > 0.5
        )
        SELECT municipio, COUNT(*) AS bg_count
        FROM assigned
        WHERE overlap_rank = 1
        GROUP BY municipio
        ORDER BY municipio;
        """,
        list(TARGET_MUNICIPALITIES),
    ).fetchdf()
    return {row.municipio: int(row.bg_count) for row in frame.itertuples(index=False)}


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "n/a"
    return ", ".join(f"{municipio}: {count:,}" for municipio, count in sorted(counts.items()))


def build_readiness_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    acs_table = table_name_for_acs(ACS_VINTAGE, "block_group")
    acs_artifact = artifact_path_for_acs(ACS_VINTAGE, "block_group")
    county_counts = count_rows_by_target(con, COUNTY_TABLE, column_candidates=("name",))
    block_group_counts = count_target_block_groups(con)
    building_counts = count_rows_by_target(con, OVERTURE_TABLE, column_candidates=("municipality_name", "municipio"))
    osm_counts = count_rows_by_target(con, OSM_PV_TABLE, column_candidates=("municipality_name", "municipio"))
    detection_counts = count_rows_by_target(con, DETECTIONS_TABLE, column_candidates=("municipality_name", "municipio"))
    detection_total = count_rows(con, DETECTIONS_TABLE) or 0
    nsrdb_files = sorted(path.name for path in NSRDB_NORMALIZED_ROOT.glob("*.parquet")) if NSRDB_NORMALIZED_ROOT.exists() else []

    rows = [
        {
            "dataset": "Case-study municipality polygons",
            "status": "ready" if len(county_counts) == len(TARGET_MUNICIPALITIES) else "partial",
            "evidence": f"{COUNTY_TABLE}: {format_counts(county_counts)}",
            "notes": "Target municipality boundaries are the geometry anchor for the BG core.",
        },
        {
            "dataset": "Census block-group geometries",
            "status": "ready" if block_group_counts else "partial",
            "evidence": f"{BLOCK_GROUP_TABLE}: {format_counts(block_group_counts)}",
            "notes": "This is the frozen core analysis unit for ESDA and correlations.",
        },
        {
            "dataset": "ACS 2024 5-year block-group attributes",
            "status": "ready" if table_exists(con, acs_table) or acs_artifact.exists() else "partial",
            "evidence": (
                f"table={table_exists(con, acs_table)} ({acs_table}), artifact={acs_artifact.exists()} ({relative_path(acs_artifact)})"
            ),
            "notes": "Ready via local parquet fallback even though the DuckDB table is not currently materialized.",
        },
        {
            "dataset": "Urban context flags",
            "status": "ready" if table_exists(con, URBAN_FLAGS_TABLE) else "partial",
            "evidence": f"{URBAN_FLAGS_TABLE}: {count_rows(con, URBAN_FLAGS_TABLE) or 0:,} rows",
            "notes": "Block-group urban context is materialized and joinable now.",
        },
        {
            "dataset": "Overture buildings",
            "status": "ready" if building_counts else "partial",
            "evidence": f"{OVERTURE_TABLE}: {format_counts(building_counts)}",
            "notes": "Building footprints are present and scoped to the case-study municipalities.",
        },
        {
            "dataset": "OSM rooftop PV polygons",
            "status": "ready" if osm_counts else "partial",
            "evidence": f"{OSM_PV_TABLE}: {format_counts(osm_counts)}",
            "notes": "Usable as the conservative fallback PV signal once joined back to buildings and BGs.",
        },
        {
            "dataset": "GeoAI PV detections",
            "status": "partial" if table_exists(con, DETECTIONS_TABLE) else "exploratory",
            "evidence": f"{DETECTIONS_TABLE}: total rows={detection_total:,}, by target={format_counts(detection_counts)}",
            "notes": "Detections exist but are too sparse for the core narrative; retain as sensitivity only.",
        },
        {
            "dataset": "Building-level PV + annual-flux join",
            "status": "partial",
            "evidence": f"{BUILDING_JOIN_TABLE}: materialized={table_exists(con, BUILDING_JOIN_TABLE)}",
            "notes": "This is the immediate gate. The join table is missing, so BG PV and flux outcomes are not audit-ready yet.",
        },
        {
            "dataset": "Block-group aggregate surface",
            "status": "partial",
            "evidence": f"{BG_AGG_TABLE}: materialized={table_exists(con, BG_AGG_TABLE)}",
            "notes": "Core ESDA should remain BG-based, but the aggregate must be rebuilt before maps, correlations, or Moran diagnostics can run.",
        },
        {
            "dataset": "NSRDB multiyear add-on",
            "status": "exploratory" if not nsrdb_files else "partial",
            "evidence": (
                "normalized parquet files: " + (", ".join(nsrdb_files[:5]) if nsrdb_files else "none present under data/tabular/nsrdb/normalized/")
            ),
            "notes": "Keep NSRDB outside the core ESDA gate until the BG join path is restored; use it later as a comparison branch.",
        },
    ]
    return pd.DataFrame(rows)


def build_variable_audit() -> pd.DataFrame:
    rows = [
        {
            "variable": "detected_pv_count",
            "status": "partial",
            "recommended_role": "Sensitivity only",
            "notes": "Blocked by missing building join and materially weakened by sparse detections.",
        },
        {
            "variable": "detected_pv_rate",
            "status": "partial",
            "recommended_role": "Sensitivity only",
            "notes": "Same dependency as detected_pv_count; do not use as the lead outcome until the join is rebuilt and zero-inflation is checked.",
        },
        {
            "variable": "osm_pv_count / overlap_count fallback",
            "status": "partial",
            "recommended_role": "Preferred conservative PV outcome",
            "notes": "OSM polygons are present now, but the building and BG joins still need to be materialized.",
        },
        {
            "variable": "annual_flux_mean_kwh_per_kw_yr",
            "status": "partial",
            "recommended_role": "Core covariate once join is rebuilt",
            "notes": "Blocked on the missing building-level join; do not claim BG-level flux readiness yet.",
        },
        {
            "variable": "flux_pixel_count",
            "status": "partial",
            "recommended_role": "Coverage QC metric",
            "notes": "Needed to assess flux coverage defensibility after the join is rebuilt.",
        },
        {
            "variable": "median_household_income_usd",
            "status": "ready",
            "recommended_role": "Core contextual covariate",
            "notes": "Available via the ACS 2024 parquet fallback and ready to join into the BG aggregate.",
        },
        {
            "variable": "pct_bachelor_plus",
            "status": "ready",
            "recommended_role": "Core contextual covariate",
            "notes": "Derived directly from the ACS 2024 block-group slice.",
        },
        {
            "variable": "pct_owner_occupied",
            "status": "ready",
            "recommended_role": "Core contextual covariate",
            "notes": "Derived directly from the ACS 2024 block-group slice.",
        },
        {
            "variable": "diversity_index",
            "status": "ready",
            "recommended_role": "Core contextual covariate",
            "notes": "Derivable now from ACS B03002 once the BG aggregate is rebuilt.",
        },
        {
            "variable": "is_urban / pct_urban_population",
            "status": "ready",
            "recommended_role": "Core contextual covariate",
            "notes": "Materialized in pr_bg_urban_flags and ready for the BG join.",
        },
    ]
    return pd.DataFrame(rows)


def build_method_matrix() -> pd.DataFrame:
    rows = [
        {
            "method_family": "Core",
            "method": "Fixed-classification municipality choropleths",
            "status": "recommended",
            "implementation_notes": "Use shared breaks across San Juan and Isabela; keep municipality small multiples instead of pooling the disconnected geography.",
            "suggested_modules": "geopandas.plot, mapclassify",
        },
        {
            "method_family": "Core",
            "method": "Pearson / Spearman matrix and targeted scatterplots",
            "status": "recommended",
            "implementation_notes": "Use BG-level outcomes plus ACS and urban covariates once the aggregate exists.",
            "suggested_modules": "pandas, scipy.stats, seaborn",
        },
        {
            "method_family": "Core",
            "method": "Queen weights, Moran scatter, global Moran's I",
            "status": "recommended",
            "implementation_notes": "Run primarily per municipality; pooled results are descriptive only.",
            "suggested_modules": "libpysal.weights.Queen, esda.moran.Moran",
        },
        {
            "method_family": "Core",
            "method": "Local Moran / LISA",
            "status": "recommended",
            "implementation_notes": "Produce one main PV-outcome LISA map and one optional contextual map once the BG outcome is defensible.",
            "suggested_modules": "esda.moran.Moran_Local",
        },
        {
            "method_family": "Sensitivity",
            "method": "Queen-versus-Rook weights check",
            "status": "recommended",
            "implementation_notes": "Retain only if the main spatial interpretation changes materially.",
            "suggested_modules": "libpysal.weights.Queen, libpysal.weights.Rook",
        },
        {
            "method_family": "Appendix",
            "method": "Correlogram or one focused bivariate/partial Moran question",
            "status": "optional",
            "implementation_notes": "Only after the core ESDA tables and maps land cleanly.",
            "suggested_modules": "libpysal, esda",
        },
        {
            "method_family": "Appendix",
            "method": "DBSCAN on NSRDB or other point surfaces",
            "status": "optional",
            "implementation_notes": "Keep off the BG core. If used, treat it as a separate point-based appendix rather than a BG clustering result.",
            "suggested_modules": "sklearn.cluster.DBSCAN",
        },
        {
            "method_family": "Excluded from 3-day core",
            "method": "Point-pattern methods or DBSCAN on BG centroids",
            "status": "exclude",
            "implementation_notes": "Not aligned with the frozen BG analysis unit for the core deliverable.",
            "suggested_modules": "n/a",
        },
        {
            "method_family": "Excluded from 3-day core",
            "method": "Full spatial econometrics, GWR, predictive ML",
            "status": "exclude",
            "implementation_notes": "Too far beyond the current data readiness and timeline.",
            "suggested_modules": "n/a",
        },
    ]
    return pd.DataFrame(rows)


def to_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""

    normalized = frame.fillna("").astype(str)
    headers = normalized.columns.tolist()
    rows = normalized.values.tolist()

    def escape_cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ").strip()

    header_row = "| " + " | ".join(escape_cell(value) for value in headers) + " |"
    divider_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = [
        "| " + " | ".join(escape_cell(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header_row, divider_row, *body_rows])


def write_report(
    readiness: pd.DataFrame,
    variable_audit: pd.DataFrame,
    method_matrix: pd.DataFrame,
    *,
    report_path: Path = REPORT_PATH,
) -> None:
    ready_items = readiness.loc[readiness["status"] == "ready", "dataset"].tolist()
    partial_items = readiness.loc[readiness["status"] == "partial", "dataset"].tolist()
    exploratory_items = readiness.loc[readiness["status"] == "exploratory", "dataset"].tolist()

    lines = [
        "# Case-Study ESDA Readiness",
        "",
        "## Core Decision",
        "",
        "Freeze the core analysis unit at the census block-group aggregate for San Juan and Isabela. The current repo is ready on geometry, ACS fallback data, and urban context, but it is not yet ready on the materialized building join and BG aggregate tables that the core ESDA workflow depends on.",
        "",
        "Current gate:",
        "- Rebuild `pr_buildings_with_pv` from the building-level join notebook.",
        "- Rebuild `pr_pv_bg_aggregates` after that join exists.",
        "- Only then run the core ESDA maps, correlations, and Moran diagnostics.",
        "",
        "## Dataset Status",
        "",
        to_markdown_table(readiness),
        "",
        f"Ready now: {', '.join(ready_items) if ready_items else 'none'}.",
        f"Partial / blocked surfaces: {', '.join(partial_items) if partial_items else 'none'}.",
        f"Exploratory add-ons: {', '.join(exploratory_items) if exploratory_items else 'none'}.",
        "",
        "## Variable Audit",
        "",
        to_markdown_table(variable_audit),
        "",
        "Primary interpretation for the 3-day core:",
        "- Keep the BG aggregate as the analysis unit.",
        "- Prefer a conservative OSM or overlap-style PV outcome if the rebuilt detection outputs remain sparse or zero-inflated.",
        "- Treat detections as sensitivity only unless the rebuilt join shows materially better coverage than the current 58-row detection table suggests.",
        "",
        "## Method Matrix",
        "",
        to_markdown_table(method_matrix),
        "",
        "## Core ESDA Execution Order",
        "",
        "1. Materialize `pr_buildings_with_pv` and then `pr_pv_bg_aggregates`.",
        "2. Export one audited BG table for San Juan and Isabela with PV, flux, ACS, and urban fields.",
        "3. Build municipality small-multiple choropleths with fixed classification across both municipalities.",
        "4. Compute Pearson and Spearman correlations plus targeted scatterplots for the chosen PV outcome against flux, income, education, tenure, diversity, and urban context.",
        "5. Build Queen weights per municipality, inspect neighbor counts and isolates, then run global Moran's I primarily per municipality.",
        "6. Produce one main LISA map for the primary PV outcome and one optional contextual LISA map if the first result is stable.",
        "7. Run a Queen-versus-Rook sensitivity check only if the interpretation changes materially.",
        "",
        "## NSRDB Add-On Path",
        "",
        "Treat NSRDB as non-blocking for the core BG sprint. Once normalized site or summary parquet files exist, aggregate them to BG or municipio and compare whether they materially change the story relative to the Google-flux path. Until then, keep NSRDB in the report as a future or appendix branch rather than a core finding.",
        "",
        "## Report Structure",
        "",
        "1. Project status by dataset.",
        "2. Analysis-unit choice and caveats.",
        "3. Core ESDA workflow.",
        "4. PySAL implementation notes.",
        "5. Optional clustering appendix.",
        "6. Contingent NSRDB add-on path.",
        "",
        "## 3-Day Schedule",
        "",
        "1. Day 1: rebuild the building join and BG aggregate; audit the candidate PV outcome and flux coverage fields.",
        "2. Day 2: produce choropleths, correlation tables, weights diagnostics, and Moran results per municipality.",
        "3. Day 3: finish LISA outputs, write the guide, and add only the smallest optional appendix that survives the readiness checks.",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).rstrip() + "\n")


# %%
if __name__ == "__main__":
    db_path = resolve_db_path()
    con = connect(db_path)

    readiness = build_readiness_table(con)
    variable_audit = build_variable_audit()
    method_matrix = build_method_matrix()
    con.close()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    readiness.to_csv(READINESS_CSV_PATH, index=False)
    variable_audit.to_csv(VARIABLE_AUDIT_CSV_PATH, index=False)
    method_matrix.to_csv(METHOD_MATRIX_CSV_PATH, index=False)
    write_report(readiness, variable_audit, method_matrix, report_path=REPORT_PATH)

    print(f"wrote {READINESS_CSV_PATH}")
    print(f"wrote {VARIABLE_AUDIT_CSV_PATH}")
    print(f"wrote {METHOD_MATRIX_CSV_PATH}")
    print(f"wrote {REPORT_PATH}")
    print(readiness.to_string(index=False))