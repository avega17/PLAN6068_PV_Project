"""Shared ACS 5-year helpers for PLAN6068 notebooks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv

from utils.census import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

ACS_API_BASE_URL = "https://api.census.gov/data"
ACS5_DATASET = "acs/acs5"
PR_STATE_FIPS = "72"
ACS_SUMMARY_LEVELS_PATH = PROJECT_ROOT / "data" / "tabular" / "2024_DataProductList_5year_summary_levels.csv"
ACS_SENTINEL_FLOOR = -100_000_000
ACS_API_MAX_GET_VARIABLES = 50
ACS_API_FIXED_GET_FIELDS = ("NAME",)
ACS_API_MAX_DATA_VARIABLES_PER_REQUEST = ACS_API_MAX_GET_VARIABLES - len(ACS_API_FIXED_GET_FIELDS)


@dataclass(frozen=True)
class ACSGeographySpec:
    """Configuration for a supported ACS geography query."""

    geography: str
    output_name: str
    summary_level: str
    request_for: str
    request_in: str
    geoid_components: tuple[str, ...]
    geoid_column: str
    geometry_table: str


ACS_GEOGRAPHY_SPECS: dict[str, ACSGeographySpec] = {
    "county": ACSGeographySpec(
        geography="county",
        output_name="counties",
        summary_level="050",
        request_for="county:*",
        request_in="state:{state_fips}",
        geoid_components=("state", "county"),
        geoid_column="county_geoid",
        geometry_table="pr_census_counties",
    ),
    "tract": ACSGeographySpec(
        geography="tract",
        output_name="tracts",
        summary_level="140",
        request_for="tract:*",
        request_in="state:{state_fips} county:*",
        geoid_components=("state", "county", "tract"),
        geoid_column="tract_geoid",
        geometry_table="pr_census_tracts",
    ),
    "block_group": ACSGeographySpec(
        geography="block_group",
        output_name="block_groups",
        summary_level="150",
        request_for="block group:*",
        request_in="state:{state_fips} county:* tract:*",
        geoid_components=("state", "county", "tract", "block group"),
        geoid_column="bg_geoid",
        geometry_table="pr_census_block_groups",
    ),
}

ACS_GEOGRAPHY_ORDER = tuple(ACS_GEOGRAPHY_SPECS.keys())

_GEOGRAPHY_ALIASES = {
    "county": "county",
    "counties": "county",
    "tract": "tract",
    "tracts": "tract",
    "block_group": "block_group",
    "block_groups": "block_group",
    "bg": "block_group",
}

_GEOID_WIDTHS = {
    "state": 2,
    "county": 3,
    "tract": 6,
    "block group": 1,
}


def get_acs_geography_spec(geography: str) -> ACSGeographySpec:
    """Return the ACS geography spec for a supported geography."""

    canonical = _GEOGRAPHY_ALIASES.get(geography.strip().lower())
    if canonical is None:
        supported = ", ".join(sorted(ACS_GEOGRAPHY_SPECS))
        raise ValueError(f"Unsupported ACS geography '{geography}'. Expected one of: {supported}")
    return ACS_GEOGRAPHY_SPECS[canonical]


def table_name_for_acs(year: int, geography: str) -> str:
    """Return the project DuckDB table name for an ACS geography slice."""

    spec = get_acs_geography_spec(geography)
    return f"pr_acs_{year}_{spec.output_name}"


def artifact_path_for_acs(year: int, geography: str, *, project_root: Path = PROJECT_ROOT) -> Path:
    """Return the durable parquet artifact path for an ACS geography slice."""

    spec = get_acs_geography_spec(geography)
    return project_root / "data" / "tabular" / f"acs_{year}_5yr_{spec.output_name}.parquet"


def download_acs_dataset(
    year: int,
    variables: list[str],
    geography: str,
    *,
    dataset: str = ACS5_DATASET,
    state_fips: str = PR_STATE_FIPS,
    api_key: str | None = None,
    timeout: int = 120,
) -> pd.DataFrame:
    """Download ACS data directly from the Census REST API.

    The Census API limits the ``get=`` parameter to 50 fields total. We always
    request ``NAME`` plus the requested estimate columns, so wide pulls are
    split into multiple legal requests and merged back on the geography keys.
    """

    key = api_key or os.getenv("CENSUS_API_KEY")
    if not key:
        raise RuntimeError("CENSUS_API_KEY is required to query the Census ACS API.")

    spec = get_acs_geography_spec(geography)
    variable_chunks = [
        variables[idx : idx + ACS_API_MAX_DATA_VARIABLES_PER_REQUEST]
        for idx in range(0, len(variables), ACS_API_MAX_DATA_VARIABLES_PER_REQUEST)
    ] or [[]]
    merge_keys = [*ACS_API_FIXED_GET_FIELDS, *spec.geoid_components]
    merged_frame: pd.DataFrame | None = None

    for chunk_index, variable_chunk in enumerate(variable_chunks, start=1):
        response = requests.get(
            f"{ACS_API_BASE_URL}/{year}/{dataset}",
            params={
                "get": ",".join((*ACS_API_FIXED_GET_FIELDS, *variable_chunk)),
                "for": spec.request_for,
                "in": spec.request_in.format(state_fips=state_fips),
                "key": key,
            },
            timeout=timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"ACS request failed for {geography} {year} chunk {chunk_index}/{len(variable_chunks)}: "
                f"{response.status_code} {response.text[:500]}"
            )

        payload = response.json()
        if not payload:
            raise RuntimeError(f"ACS request returned no rows for {geography} {year} chunk {chunk_index}.")

        chunk_frame = pd.DataFrame(payload[1:], columns=payload[0])
        if merged_frame is None:
            merged_frame = chunk_frame
            continue

        merged_frame = merged_frame.merge(
            chunk_frame,
            on=merge_keys,
            how="inner",
            validate="one_to_one",
        )

    if merged_frame is None:
        raise RuntimeError(f"ACS request returned no rows for {geography} {year}.")

    ordered_columns = list(dict.fromkeys([*ACS_API_FIXED_GET_FIELDS, *variables, *spec.geoid_components]))
    return merged_frame[ordered_columns].copy()


def coerce_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert ACS estimate columns from strings to numeric values."""

    coerced = frame.copy()
    for column in columns:
        if column in coerced.columns:
            coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return sanitize_acs_numeric_values(coerced, columns)


def sanitize_acs_numeric_values(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    sentinel_floor: int = ACS_SENTINEL_FLOOR,
) -> pd.DataFrame:
    """Replace Census sentinel values with NA after numeric coercion.

    Detailed ACS tables use very large negative sentinels (for example
    ``-666666666``) for suppressed or unavailable values. Those values should
    not survive into ratios, choropleths, or downstream summaries.
    """

    sanitized = frame.copy()
    for column in columns:
        if column not in sanitized.columns:
            continue
        series = sanitized[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        sanitized[column] = series.mask(series <= sentinel_floor)
    return sanitized


def append_geoid(frame: pd.DataFrame, geography: str) -> pd.DataFrame:
    """Append the project GEOID column for a given ACS geography."""

    spec = get_acs_geography_spec(geography)
    geoid_parts: list[pd.Series] = []
    for component in spec.geoid_components:
        width = _GEOID_WIDTHS[component]
        geoid_parts.append(frame[component].astype(str).str.zfill(width))

    geoid_frame = frame.copy()
    geoid_frame[spec.geoid_column] = geoid_parts[0]
    for part in geoid_parts[1:]:
        geoid_frame[spec.geoid_column] = geoid_frame[spec.geoid_column] + part
    return geoid_frame


def persist_dataframe(con: duckdb.DuckDBPyConnection, table_name: str, frame: pd.DataFrame) -> None:
    """Persist a plain tabular DataFrame into DuckDB."""

    con.register("staged_acs_frame", frame)
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM staged_acs_frame;")
    con.unregister("staged_acs_frame")


def _normalize_catalog_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten whitespace-heavy Census catalog column names."""

    normalized = frame.copy()
    normalized.columns = [
        " ".join(
            str(column)
            .replace("\ufeff", "")
            .replace("ï»¿", "")
            .replace("\n", " ")
            .split()
        )
        for column in normalized.columns
    ]
    return normalized


def product_catalog_path_for_year(year: int, *, project_root: Path = PROJECT_ROOT) -> Path:
    """Return the repository catalog path for a given ACS vintage."""

    candidates = [
        project_root / "data" / "tabular" / f"{year}_DataProductList_5Year.csv",
        project_root / "data" / "tabular" / f"{year}_DataProductList.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No ACS product catalog found for {year}. Tried: {tried}")


def load_product_catalog(*, year: int | None = None, path: Path | None = None) -> pd.DataFrame:
    """Load an ACS product catalog supplied with the repository."""

    chosen_path = path or product_catalog_path_for_year(year or 2024)
    return _normalize_catalog_columns(pd.read_csv(chosen_path, dtype=str, encoding="utf-8-sig"))


def load_summary_level_catalog(path: Path = ACS_SUMMARY_LEVELS_PATH) -> pd.DataFrame:
    """Load the ACS 5-year summary-level catalog supplied with the repository."""

    return _normalize_catalog_columns(pd.read_csv(path, dtype=str, encoding="latin-1", header=1))