# %% [markdown]
# # Solar API Coverage + Quality Probe
#
# Uses the Building Insights endpoint (10K/month free tier) to tag each tile in
# `pr_solar_tile_manifest` with its best-available imagery quality before we
# spend any Data Layers quota.
#
# The manifest is now H3-cell based, so probing one row per legacy `bg_geoid`
# would hit every tile. We instead probe a stratified subset of candidate tiles
# per municipality and classify them as `ready`, `stale_pre_maria`,
# `no_coverage`, or `probe_failed`.

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
PROBE_PENDING_STATUSES = ("pending", "probe_failed")
PROBE_BUCKETS = 8
MAX_PROBES_PER_BUCKET = 6
POST_MARIA_CUTOFF = pd.Timestamp("2017-09-20")


def resolve_db_path() -> Path:
    value = os.getenv("VECTOR_DB")
    if value:
        p = Path(value)
        if not p.is_absolute():
            p = PROJECT_ROOT / p if len(p.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / p
        return p
    return PROJECT_ROOT / "data" / "PR_PV_plan_data.duckdb"


# %%
def load_probe_points(
    con: duckdb.DuckDBPyConnection,
    *,
    statuses: tuple[str, ...] = PROBE_PENDING_STATUSES,
    probe_buckets: int = PROBE_BUCKETS,
    max_per_bucket: int = MAX_PROBES_PER_BUCKET,
) -> pd.DataFrame:
    status_sql = ", ".join(f"'{status}'" for status in statuses)
    return con.execute(
        f"""
        WITH candidates AS (
            SELECT
                tile_id,
                bg_geoid,
                h3_cell_id,
                municipio,
                lon,
                lat,
                priority_score,
                building_count,
                osm_pv_count,
                status,
                required_quality
            FROM {MANIFEST_TABLE}
            WHERE status IN ({status_sql})
        ),
        bucketed AS (
            SELECT
                *,
                NTILE({probe_buckets}) OVER (
                    PARTITION BY municipio
                    ORDER BY priority_score DESC, osm_pv_count DESC, building_count DESC, tile_id
                ) AS priority_bucket
            FROM candidates
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY municipio, priority_bucket
                    ORDER BY building_count DESC, osm_pv_count DESC, tile_id
                ) AS bucket_rank
            FROM bucketed
        )
        SELECT tile_id, bg_geoid, h3_cell_id, municipio, lon, lat, priority_score, priority_bucket, status, required_quality
        FROM ranked
        WHERE bucket_rank <= {max_per_bucket}
        ORDER BY municipio, priority_bucket, bucket_rank, tile_id;
        """
    ).fetchdf()


def ensure_probe_columns(con: duckdb.DuckDBPyConnection) -> None:
    """Add probe-status detail columns to the manifest when missing."""

    existing = set(
        con.execute(
            f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{MANIFEST_TABLE}';
            """
        ).fetchdf()["column_name"]
    )
    required_columns = {
        "probe_imagery_date": "DATE",
        "probe_http_status": "INTEGER",
        "probe_last_checked_utc": "TIMESTAMP",
    }
    for column_name, column_type in required_columns.items():
        if column_name not in existing:
            con.execute(f"ALTER TABLE {MANIFEST_TABLE} ADD COLUMN {column_name} {column_type};")


def collapse_probe_quality(probe_df: pd.DataFrame) -> pd.DataFrame:
    """Per tile, keep the best returned quality plus post-Maria classification."""

    rank = {"HIGH": 3, "MEDIUM": 2, "BASE": 1}
    probe_df = probe_df.copy()
    probe_df["imagery_date"] = pd.to_datetime(probe_df["imagery_date"], errors="coerce")
    probe_df["quality_rank"] = probe_df["expected_quality"].map(rank).fillna(0).astype(int)
    probe_df["is_post_maria"] = probe_df["imagery_date"] >= POST_MARIA_CUTOFF
    best = (
        probe_df.sort_values(["quality_rank", "imagery_date"], ascending=[False, False])
        .groupby("tile_id", as_index=False)
        .first()
    )
    best["manifest_status"] = "probe_failed"
    best.loc[best["status"] == "not_found", "manifest_status"] = "no_coverage"
    best.loc[(best["status"] == "ok") & best["is_post_maria"], "manifest_status"] = "ready"
    best.loc[(best["status"] == "ok") & ~best["is_post_maria"], "manifest_status"] = "stale_pre_maria"
    return best[["tile_id", "expected_quality", "status", "http_status", "imagery_date", "manifest_status"]]


def summarize_probe_results(probe_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return compact quality/date diagnostics for the current probe sample."""

    probe_df = probe_df.copy()
    probe_df["imagery_date"] = pd.to_datetime(probe_df["imagery_date"], errors="coerce")
    probe_df["imagery_period"] = pd.Series(pd.NA, index=probe_df.index, dtype="object")
    probe_df.loc[probe_df["imagery_date"].notna(), "imagery_period"] = "pre_maria"
    probe_df.loc[probe_df["imagery_date"] >= POST_MARIA_CUTOFF, "imagery_period"] = "post_maria"

    by_status = (
        probe_df.groupby(["municipio", "status", "expected_quality", "imagery_period"], dropna=False)
        .size()
        .reset_index(name="tiles")
        .sort_values(["municipio", "tiles", "status"], ascending=[True, False, True])
    )
    failures = (
        probe_df[probe_df["status"] != "ok"]
        .groupby(["status", "http_status"], dropna=False)
        .size()
        .reset_index(name="tiles")
        .sort_values("tiles", ascending=False)
    )
    return by_status, failures


# %% [markdown]
# ## Step 1 — Read manifest, run probe

# %%
if __name__ == "__main__":
    db_path = resolve_db_path()
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")
    ensure_probe_columns(con)

    points_df = load_probe_points(con)
    print(
        f"probing {len(points_df):,} stratified tile points across "
        f"{points_df['municipio'].nunique():,} municipalities and {points_df['priority_bucket'].nunique():,} priority buckets"
    )

    tuples = list(points_df[["tile_id", "bg_geoid", "municipio", "lon", "lat"]].itertuples(index=False, name=None))
    probe_df = probe_points(tuples)
    print(probe_df["expected_quality"].value_counts(dropna=False).to_string())
    print(probe_df["status"].value_counts(dropna=False).to_string())
    sample_summary, failure_summary = summarize_probe_results(probe_df)
    print("\nProbe sample summary:")
    print(sample_summary.to_string(index=False))
    if not failure_summary.empty:
        print("\nProbe failures:")
        print(failure_summary.to_string(index=False))

# %% [markdown]
# ## Step 2 — Collapse to BG-level best quality, update manifest

# %%
if __name__ == "__main__":
    best_per_tile = collapse_probe_quality(probe_df)
    con.register("tile_quality", best_per_tile.rename(columns={"status": "probe_status"}))
    con.execute(
        f"""
        UPDATE {MANIFEST_TABLE} AS m
        SET
            expected_quality = q.expected_quality,
            required_quality = COALESCE(q.expected_quality, m.required_quality),
            status = CASE
                WHEN q.probe_status = 'not_found' THEN 'no_coverage'
                WHEN q.manifest_status = 'ready' THEN 'ready'
                WHEN q.manifest_status = 'stale_pre_maria' THEN 'stale_pre_maria'
                ELSE 'probe_failed'
            END
            ,probe_imagery_date = CAST(q.imagery_date AS DATE)
            ,probe_http_status = q.http_status
            ,probe_last_checked_utc = CURRENT_TIMESTAMP
        FROM tile_quality AS q
        WHERE m.tile_id = q.tile_id;
        """
    )
    con.unregister("tile_quality")

    summary = con.execute(
        f"""
        SELECT status, expected_quality, required_quality, COUNT(*) AS tiles
        FROM {MANIFEST_TABLE}
        GROUP BY 1, 2, 3
        ORDER BY tiles DESC, status, expected_quality, required_quality;
        """
    ).fetchdf()
    print(summary.to_string(index=False))
    post_maria_summary = con.execute(
        f"""
        SELECT
            CASE
                WHEN probe_imagery_date >= DATE '{POST_MARIA_CUTOFF.date().isoformat()}' THEN 'post_maria'
                WHEN probe_imagery_date IS NULL THEN 'unknown'
                ELSE 'pre_maria'
            END AS imagery_period,
            status,
            COUNT(*) AS tiles
        FROM {MANIFEST_TABLE}
        GROUP BY 1, 2
        ORDER BY tiles DESC, imagery_period, status;
        """
    ).fetchdf()
    print("\nManifest imagery-period summary:")
    print(post_maria_summary.to_string(index=False))
    print("ledger so far:", ledger_summary())
    con.close()
