# %% [markdown]
# # 2020 Urban Areas — Census Blocks ingest (Puerto Rico)
#
# Downloads `2020_UA_BLOCKS.txt` from the Census Bureau, filters to
# Puerto Rico (STATE FIPS `72`), and persists two tables into the project
# DuckDB:
#
# 1. `pr_urban_blocks_2020` — the urban blocks themselves (one row per
#    2020 tabulation block present in an Urban Area), with the derived
#    `bg_geoid` (12 chars) and `block_geoid` (15 chars).
# 2. `pr_bg_urban_flags` — BG-level roll-up with `urban_block_count` and
#    a boolean `is_urban` (any urban block present).  Consumed by
#    `notebooks/tabular/02_pv_bg_aggregation.py`.
#
# According to the Census Bureau, the 2020 Urban Areas dataset is the
# latest release of urban-vs-rural classification and replaces the
# 2010 vintage.  A BG is classified as **rural** if it has no urban
# blocks in the 2020 list.
#
# Refs:
# - https://www.census.gov/programs-surveys/geography/guidance/geo-areas/urban-rural.html
# - https://www2.census.gov/geo/docs/reference/ua/2020_UA_BLOCKS.txt
# - https://www.census.gov/geographies/reference-files/2020/geo/2020-ua.html

# %%
"""05_urban_blocks_2020_ingest.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv


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

UA_BLOCKS_URL = "https://www2.census.gov/geo/docs/reference/ua/2020_UA_BLOCKS.txt"
CACHE_DIR = PROJECT_ROOT / "data" / "tabular"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "2020_UA_BLOCKS.txt"
PR_STATE_FIPS = "72"


def resolve_db_path() -> Path:
    v = os.getenv("VECTOR_DB")
    if v:
        p = Path(v)
        if not p.is_absolute():
            p = PROJECT_ROOT / p if len(p.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / p
        return p
    return PROJECT_ROOT / "data" / "vectors" / "PR_vector_data.duckdb"


# %%
def download_ua_blocks(force: bool = False) -> Path:
    """Download (and cache) the 2020 Urban Areas blocks file."""
    if CACHE_FILE.exists() and not force:
        return CACHE_FILE
    print(f"downloading {UA_BLOCKS_URL} → {CACHE_FILE}")
    r = requests.get(UA_BLOCKS_URL, timeout=120)
    r.raise_for_status()
    CACHE_FILE.write_bytes(r.content)
    return CACHE_FILE


# %%
def load_pr_urban_blocks(path: Path) -> pd.DataFrame:
    """Read the UA blocks file and filter to PR (STATE='72').

    The Census file is pipe-delimited with a header row.  Column names
    vary slightly across vintages so we detect STATE/COUNTY/TRACT/BLOCK
    headers case-insensitively.
    """
    # The file is small (~240 MB raw, ~10 MB for PR).  Stream via pandas.
    df = pd.read_csv(path, sep="|", dtype=str, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]

    # Expected columns include STATE, COUNTY, TRACT, BLOCK, UACE20
    # (2020 Urban Area Code).
    expected = {"STATE", "COUNTY", "TRACT", "BLOCK"}
    missing = expected - set(df.columns)
    if missing:
        raise RuntimeError(f"2020_UA_BLOCKS.txt missing columns: {missing}; got {df.columns.tolist()}")

    pr = df[df["STATE"].str.zfill(2) == PR_STATE_FIPS].copy()
    pr["STATE"] = pr["STATE"].str.zfill(2)
    pr["COUNTY"] = pr["COUNTY"].str.zfill(3)
    pr["TRACT"] = pr["TRACT"].str.zfill(6)
    pr["BLOCK"] = pr["BLOCK"].str.zfill(4)
    pr["block_geoid"] = pr["STATE"] + pr["COUNTY"] + pr["TRACT"] + pr["BLOCK"]
    # BG id = first digit of the block number.
    pr["bg_geoid"] = pr["STATE"] + pr["COUNTY"] + pr["TRACT"] + pr["BLOCK"].str[0]
    return pr.reset_index(drop=True)


# %%
# Notebook driver.
path = download_ua_blocks()
print(f"reading {path} …")
pr_blocks = load_pr_urban_blocks(path)
print(f"      {len(pr_blocks):,} PR urban blocks (2020)")

db_path = resolve_db_path()
con = duckdb.connect(str(db_path))

# Persist urban blocks and a BG-level roll-up.
con.register("pr_urban_blocks_df", pr_blocks)
con.execute(
    """
    CREATE OR REPLACE TABLE pr_urban_blocks_2020 AS
    SELECT * FROM pr_urban_blocks_df;
    """
)
con.execute(
    """
    CREATE OR REPLACE TABLE pr_bg_urban_flags AS
    SELECT bg_geoid,
           COUNT(*)::BIGINT AS urban_block_count,
           TRUE AS is_urban
    FROM pr_urban_blocks_2020
    GROUP BY bg_geoid;
    """
)
con.unregister("pr_urban_blocks_df")

summary = con.execute(
    """
    SELECT COUNT(*) AS urban_bg_count,
           SUM(urban_block_count) AS total_urban_blocks
    FROM pr_bg_urban_flags;
    """
).fetchdf()
print(summary.to_string(index=False))
con.close()
