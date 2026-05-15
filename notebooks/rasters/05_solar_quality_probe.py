# %% [markdown]
# # Solar API Coverage + Quality Probe
#
# Uses the Building Insights endpoint (10K/month free tier) to tag each tile in
# `pr_solar_tile_manifest` with its best-available imagery quality before we
# spend any Data Layers quota.
#
# One BG centroid is typically sufficient; we sample up to
# `MAX_PROBES_PER_BG` tile centers per BG for robustness against edge cases.

# %%
"""05_solar_quality_probe.py

Probe buildingInsights:findClosest across the tile manifest and update
`pr_solar_tile_manifest.expected_quality` + `required_quality` + `status`.
"""

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
        if any((candidate / m).exists() for m in ("project_rules.md", ".git")):
            return candidate
    return current


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from utils.solar_api import probe_points, ledger_summary  # noqa: E402

MANIFEST_TABLE = "pr_solar_tile_manifest"
MAX_PROBES_PER_BG = 1  # bump to 2-3 if we observe within-BG quality heterogeneity


def resolve_db_path() -> Path:
    value = os.getenv("VECTOR_DB")
    if value:
        p = Path(value)
        if not p.is_absolute():
            p = PROJECT_ROOT / p if len(p.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / p
        return p
    return PROJECT_ROOT / "data" / "vectors" / "PR_vector_data.duckdb"


# %%
def load_probe_points(con: duckdb.DuckDBPyConnection, max_per_bg: int = MAX_PROBES_PER_BG) -> pd.DataFrame:
    return con.execute(
        f"""
        WITH ranked AS (
            SELECT
                tile_id, bg_geoid, municipio, lon, lat, building_count,
                ROW_NUMBER() OVER (
                    PARTITION BY bg_geoid
                    ORDER BY building_count DESC, tile_id
                ) AS rn
            FROM {MANIFEST_TABLE}
        )
        SELECT tile_id, bg_geoid, municipio, lon, lat
        FROM ranked
        WHERE rn <= {max_per_bg}
        ORDER BY bg_geoid, rn;
        """
    ).fetchdf()


def collapse_bg_quality(probe_df: pd.DataFrame) -> pd.DataFrame:
    """Per BG, pick the best quality any probed point returned (HIGH>MEDIUM>BASE)."""

    rank = {"HIGH": 3, "MEDIUM": 2, "BASE": 1}
    probe_df = probe_df.copy()
    probe_df["rank"] = probe_df["expected_quality"].map(rank).fillna(0).astype(int)
    best = probe_df.sort_values("rank", ascending=False).groupby("bg_geoid", as_index=False).first()
    return best[["bg_geoid", "expected_quality", "status", "imagery_date"]]


# %% [markdown]
# ## Step 1 — Read manifest, run probe

# %%
if __name__ == "__main__":
    db_path = resolve_db_path()
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")

    points_df = load_probe_points(con)
    print(f"probing {len(points_df):,} points across {points_df['bg_geoid'].nunique():,} BGs")

    tuples = list(points_df[["tile_id", "bg_geoid", "municipio", "lon", "lat"]].itertuples(index=False, name=None))
    probe_df = probe_points(tuples)
    print(probe_df["expected_quality"].value_counts(dropna=False).to_string())
    print(probe_df["status"].value_counts(dropna=False).to_string())

# %% [markdown]
# ## Step 2 — Collapse to BG-level best quality, update manifest

# %%
if __name__ == "__main__":
    best_per_bg = collapse_bg_quality(probe_df)
    con.register("bg_quality", best_per_bg.rename(columns={"status": "probe_status"}))
    con.execute(
        f"""
        UPDATE {MANIFEST_TABLE} AS m
        SET
            expected_quality = q.expected_quality,
            required_quality = q.expected_quality,
            status = CASE
                WHEN q.probe_status = 'not_found' THEN 'no_coverage'
                WHEN q.expected_quality IS NULL THEN 'probe_failed'
                ELSE 'ready'
            END
        FROM bg_quality AS q
        WHERE m.bg_geoid = q.bg_geoid;
        """
    )
    con.unregister("bg_quality")

    summary = con.execute(
        f"""
        SELECT status, expected_quality, COUNT(*) AS tiles
        FROM {MANIFEST_TABLE}
        GROUP BY 1, 2
        ORDER BY 1, 2;
        """
    ).fetchdf()
    print(summary.to_string(index=False))
    print("ledger so far:", ledger_summary())
    con.close()
