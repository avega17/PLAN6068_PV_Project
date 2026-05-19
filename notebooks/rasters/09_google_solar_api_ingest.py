# %% [markdown]
# # Google Solar API — Data Layers Ingestion
#
# Consumes `pr_solar_tile_manifest` in priority order and downloads RGB,
# building mask, DSM, and annual flux GeoTIFFs for each tile into
# `data/rasters/solar/{municipio}/{bg_geoid}/`. Stops at the 1,999-call cap.
# Supports `--resume` semantics via the local cache + ledger.

# %%
"""06_google_solar_api_ingest.py

Budgeted Data Layers fetcher for the San Juan + Isabela tile manifest.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from dotenv import load_dotenv
from rasterio.merge import merge as merge_rasters


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

# Force-reload utils.solar_api so a stale Jupyter kernel can't silently
# re-run with an outdated version of the module (this has bitten us twice).
import utils.solar_api as _solar_api  # noqa: E402
importlib.reload(_solar_api)
from utils.solar_api import (  # noqa: E402
    DATA_LAYERS_BUDGET_CAP,
    enforce_radius_pixel_constraint,
    fetch_tiles_budgeted,
    ledger_summary,
    quality_to_pixel_size,
)

# Self-test: prove the loaded module enforces the radius/pixel constraint.
# At our standard tile radius (175 m), HIGH-quality requests MUST resolve to
# a 0.25 m pixel size, otherwise the API returns HTTP 400 INVALID_ARGUMENT
# for every tile.
_high_px = quality_to_pixel_size("HIGH")
_enforced_px = enforce_radius_pixel_constraint(175, _high_px)
assert _enforced_px == 0.25, (
    f"stale utils.solar_api detected: HIGH+175m -> pixel={_enforced_px}; expected 0.25. "
    "Restart the Jupyter kernel."
)
print(f"utils.solar_api OK: HIGH default pixel={_high_px}, enforced for r=175m -> {_enforced_px}")

MANIFEST_TABLE = "pr_solar_tile_manifest"
MAX_TILES_THIS_RUN: int | None = None  # set to an int to limit this run below the global cap
MAX_BUILDING_INSIGHTS_THIS_RUN: int | None = None  # set to an int to limit probes; limit is balanced across municipalities
BUILDING_INSIGHTS_MIN_PRIORITY = 2
BUILDING_INSIGHTS_REQUIRED_QUALITIES = ("BASE", "MEDIUM")
DATA_LAYERS_DEFAULT_REQUIRED_QUALITY = "BASE"
POST_MARIA_FETCH_ALLOWED_QUALITIES = ("BASE", "MEDIUM")
FETCH_ONLY_POST_MARIA_CANDIDATES = True
BUILDING_INSIGHTS_SLEEP_SECONDS = 0.02
BUILDING_INSIGHTS_ENDPOINT = "https://solar.googleapis.com/v1/buildingInsights:findClosest"
POST_MARIA_CUTOFF = pd.Timestamp("2017-09-20")
PREVIEW_POST_MARIA_ONLY = True
PREVIEW_ALLOWED_QUALITIES = POST_MARIA_FETCH_ALLOWED_QUALITIES
PREVIEW_MOSAIC_RES_M = 0.5
PREVIEW_MAX_TILES_PER_MUNICIPIO: int | None = None

_env_solar_root = os.getenv("SOLAR_RASTER_ROOT")
SOLAR_ROOT = (
    PROJECT_ROOT / _env_solar_root
    if _env_solar_root and not Path(_env_solar_root).is_absolute()
    else Path(_env_solar_root or PROJECT_ROOT / "data" / "rasters" / "solar")
)


def resolve_db_path() -> Path:
    value = os.getenv("VECTOR_DB")
    if value:
        p = Path(value)
        if not p.is_absolute():
            p = PROJECT_ROOT / p if len(p.parts) > 1 else PROJECT_ROOT / "data" / "vectors" / p
        return p
    return PROJECT_ROOT / "data" / "PR_PV_plan_data.duckdb"


# %%
def load_ready_manifest(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(
        f"""
        SELECT tile_id, bg_geoid, municipio, lon, lat, radius_m,
               required_quality, priority_score, building_count, osm_pv_count
        FROM {MANIFEST_TABLE}
        WHERE status = 'ready'
        ORDER BY priority_score DESC, osm_pv_count DESC, building_count DESC, tile_id;
        """
    ).fetchdf()
    return df


def load_candidate_manifest(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load the full tile universe for post-Maria re-probing / re-fetching.

    This intentionally does NOT filter to `status = 'ready'`. Earlier runs
    marked many tiles as `fetched` or `no_coverage` based on pre-Maria HIGH
    imagery. For the post-Maria BASE/MEDIUM workflow we want to revisit the
    whole manifest and let Building Insights tell us whether newer 0.25 m
    imagery exists.
    """

    df = con.execute(
        f"""
        SELECT tile_id, bg_geoid, municipio, lon, lat, radius_m,
               required_quality, priority_score, building_count, osm_pv_count, status
        FROM {MANIFEST_TABLE}
        ORDER BY priority_score DESC, osm_pv_count DESC, building_count DESC, tile_id;
        """
    ).fetchdf()
    return df


def build_probe_manifest(
    manifest: pd.DataFrame,
    *,
    min_priority: int = BUILDING_INSIGHTS_MIN_PRIORITY,
    max_tiles: int | None = MAX_BUILDING_INSIGHTS_THIS_RUN,
) -> pd.DataFrame:
    probe_manifest = manifest[manifest["priority_score"] >= min_priority].copy()
    if max_tiles is None or len(probe_manifest) <= max_tiles:
        return probe_manifest.reset_index(drop=True)

    municipios = sorted(probe_manifest["municipio"].dropna().unique().tolist())
    per_municipio = max(1, int(np.ceil(max_tiles / max(1, len(municipios)))))
    balanced = pd.concat(
        [probe_manifest[probe_manifest["municipio"] == municipio].head(per_municipio) for municipio in municipios],
        ignore_index=True,
    )
    balanced = (
        balanced.sort_values(
            ["priority_score", "osm_pv_count", "building_count", "tile_id"],
            ascending=[False, False, False, True],
        )
        .head(max_tiles)
        .reset_index(drop=True)
    )
    return balanced


def date_dict_to_timestamp(value: dict | None) -> pd.Timestamp | pd.NaT:
    if not value:
        return pd.NaT
    year = value.get("year")
    month = value.get("month")
    day = value.get("day")
    if not (year and month and day):
        return pd.NaT
    return pd.Timestamp(year=int(year), month=int(month), day=int(day))


def find_closest_building_rest(
    session: requests.Session,
    *,
    lat: float,
    lon: float,
    api_key: str,
    required_quality: str = BUILDING_INSIGHTS_REQUIRED_QUALITIES[0],
    timeout_seconds: int = 60,
) -> dict:
    params = {
        "location.latitude": f"{float(lat):.8f}",
        "location.longitude": f"{float(lon):.8f}",
        "requiredQuality": required_quality,
        "key": api_key,
    }
    response = session.get(BUILDING_INSIGHTS_ENDPOINT, params=params, timeout=timeout_seconds)

    if response.status_code == 404:
        return {
            "status": "not_found",
            "http_status": 404,
            "building_name": None,
            "imagery_quality": None,
            "imagery_date": pd.NaT,
            "imagery_processed_date": pd.NaT,
            "postal_code": None,
            "administrative_area": None,
            "statistical_area": None,
            "max_array_panels_count": None,
            "roof_area_m2": None,
            "notes": "not_found",
        }

    if not response.ok:
        payload = response.json() if "json" in response.headers.get("content-type", "") else {"text": response.text[:500]}
        return {
            "status": "error",
            "http_status": response.status_code,
            "building_name": None,
            "imagery_quality": None,
            "imagery_date": pd.NaT,
            "imagery_processed_date": pd.NaT,
            "postal_code": None,
            "administrative_area": None,
            "statistical_area": None,
            "max_array_panels_count": None,
            "roof_area_m2": None,
            "notes": str(payload)[:200],
        }

    payload = response.json()
    solar_potential = payload.get("solarPotential") or {}
    roof_stats = solar_potential.get("wholeRoofStats") or {}
    return {
        "status": "ok",
        "http_status": response.status_code,
        "building_name": payload.get("name"),
        "imagery_quality": payload.get("imageryQuality"),
        "imagery_date": date_dict_to_timestamp(payload.get("imageryDate")),
        "imagery_processed_date": date_dict_to_timestamp(payload.get("imageryProcessedDate")),
        "postal_code": payload.get("postalCode"),
        "administrative_area": payload.get("administrativeArea"),
        "statistical_area": payload.get("statisticalArea"),
        "max_array_panels_count": solar_potential.get("maxArrayPanelsCount"),
        "roof_area_m2": roof_stats.get("areaMeters2"),
        "notes": None,
    }


def probe_building_insights_rest(
    manifest: pd.DataFrame,
    *,
    api_key: str,
    required_quality: str = BUILDING_INSIGHTS_REQUIRED_QUALITIES[0],
    sleep_between_seconds: float = BUILDING_INSIGHTS_SLEEP_SECONDS,
) -> pd.DataFrame:
    rows: list[dict] = []
    with requests.Session() as session:
        for _, row in manifest.iterrows():
            result = find_closest_building_rest(
                session,
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                api_key=api_key,
                required_quality=required_quality,
            )
            rows.append({
                "tile_id": row["tile_id"],
                "bg_geoid": row["bg_geoid"],
                "municipio": row["municipio"],
                "priority_score": row["priority_score"],
                "building_count": row["building_count"],
                "osm_pv_count": row["osm_pv_count"],
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                **result,
            })
            if sleep_between_seconds:
                time.sleep(sleep_between_seconds)
    return pd.DataFrame(rows)


def probe_building_insights_multi_quality(
    manifest: pd.DataFrame,
    *,
    api_key: str,
    required_qualities: tuple[str, ...] = BUILDING_INSIGHTS_REQUIRED_QUALITIES,
    sleep_between_seconds: float = BUILDING_INSIGHTS_SLEEP_SECONDS,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for required_quality in required_qualities:
        frame = probe_building_insights_rest(
            manifest,
            api_key=api_key,
            required_quality=required_quality,
            sleep_between_seconds=sleep_between_seconds,
        )
        frame["requested_quality"] = required_quality
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def select_post_maria_candidates(
    building_probe: pd.DataFrame,
    *,
    cutoff: pd.Timestamp = POST_MARIA_CUTOFF,
    allowed_qualities: tuple[str, ...] = POST_MARIA_FETCH_ALLOWED_QUALITIES,
) -> pd.DataFrame:
    if building_probe.empty:
        return building_probe.copy()

    candidates = building_probe[
        (building_probe["status"] == "ok")
        & (building_probe["imagery_date"] >= cutoff)
        & (building_probe["imagery_quality"].isin(allowed_qualities))
    ].copy()
    if candidates.empty:
        return candidates

    returned_rank = {"MEDIUM": 2, "BASE": 1}
    requested_rank = {"MEDIUM": 2, "BASE": 1}
    candidates["_returned_rank"] = candidates["imagery_quality"].map(returned_rank).fillna(0)
    candidates["_requested_rank"] = candidates["requested_quality"].map(requested_rank).fillna(0)
    candidates = (
        candidates.sort_values(
            [
                "imagery_date",
                "_returned_rank",
                "_requested_rank",
                "priority_score",
                "osm_pv_count",
                "building_count",
                "tile_id",
            ],
            ascending=[False, False, False, False, False, False, True],
        )
        .drop_duplicates(subset=["tile_id"], keep="first")
        .reset_index(drop=True)
    )
    candidates["fetch_required_quality"] = candidates["imagery_quality"]
    return candidates.drop(columns=["_returned_rank", "_requested_rank"])


def build_fetch_manifest(
    manifest: pd.DataFrame,
    *,
    required_quality: str = DATA_LAYERS_DEFAULT_REQUIRED_QUALITY,
    post_maria_candidates: pd.DataFrame | None = None,
    fetch_only_post_maria_candidates: bool = FETCH_ONLY_POST_MARIA_CANDIDATES,
) -> pd.DataFrame:
    fetch_manifest = manifest.copy()
    fetch_manifest["required_quality"] = required_quality

    if fetch_only_post_maria_candidates:
        if post_maria_candidates is None:
            raise RuntimeError(
                "Run Step 1b first so post_maria_candidates exists before targeted BASE ingestion."
            )
        if post_maria_candidates.empty:
            raise RuntimeError(
                "Step 1b found zero post-Maria candidates. Nothing to fetch with targeted BASE ingestion."
            )

        candidate_cols = [
            "tile_id",
            "requested_quality",
            "fetch_required_quality",
            "imagery_date",
            "imagery_quality",
            "building_name",
            "postal_code",
            "max_array_panels_count",
            "roof_area_m2",
        ]
        candidate_meta = (
            post_maria_candidates[candidate_cols]
            .drop_duplicates(subset=["tile_id"])
            .rename(columns={
                "requested_quality": "bi_requested_quality",
                "fetch_required_quality": "bi_fetch_required_quality",
                "imagery_date": "bi_imagery_date",
                "imagery_quality": "bi_imagery_quality",
            })
        )
        fetch_manifest = fetch_manifest.merge(candidate_meta, on="tile_id", how="inner")
        fetch_manifest["required_quality"] = fetch_manifest["bi_fetch_required_quality"].fillna(required_quality)
        fetch_manifest = fetch_manifest.sort_values(
            [
                "bi_imagery_date",
                "priority_score",
                "osm_pv_count",
                "building_count",
                "tile_id",
            ],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)
    else:
        fetch_manifest["bi_imagery_date"] = pd.NaT
        fetch_manifest["bi_requested_quality"] = None
        fetch_manifest["bi_fetch_required_quality"] = required_quality
        fetch_manifest["bi_imagery_quality"] = None
        fetch_manifest["building_name"] = None
        fetch_manifest["postal_code"] = None
        fetch_manifest["max_array_panels_count"] = None
        fetch_manifest["roof_area_m2"] = None

    return fetch_manifest


def scan_fetched_rgb_assets(root: Path = SOLAR_ROOT) -> pd.DataFrame:
    records: list[dict] = []
    for sidecar in sorted(root.rglob("*_meta.json")):
        try:
            meta = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        layers = meta.get("layers") or {}
        rgb_path = layers.get("rgb")
        if not rgb_path:
            continue
        rgb = Path(rgb_path)
        if not rgb.is_absolute():
            rgb = PROJECT_ROOT / rgb
        imagery_date = pd.to_datetime(meta.get("imagery_date"), errors="coerce")
        records.append({
            "tile_id": meta.get("tile_id"),
            "bg_geoid": meta.get("bg_geoid"),
            "municipio": meta.get("municipio"),
            "imagery_quality": meta.get("returned_quality"),
            "imagery_date": imagery_date,
            "post_maria": imagery_date >= POST_MARIA_CUTOFF if pd.notna(imagery_date) else False,
            "rgb_path": rgb,
        })
    return pd.DataFrame(records)


def stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype("float32")
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    arr = arr[:3]
    out = np.zeros_like(arr, dtype="float32")
    for band_index in range(arr.shape[0]):
        band = arr[band_index]
        lo, hi = np.nanpercentile(band, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(band))
            hi = float(np.nanmax(band))
        if hi > lo:
            out[band_index] = np.clip((band - lo) / (hi - lo), 0, 1)
    return np.moveaxis(out, 0, -1)


def plot_preview_mosaic(
    rgb_assets: pd.DataFrame,
    municipio: str,
    *,
    post_maria_only: bool = PREVIEW_POST_MARIA_ONLY,
    allowed_qualities: tuple[str, ...] = PREVIEW_ALLOWED_QUALITIES,
    mosaic_res_m: float = PREVIEW_MOSAIC_RES_M,
    max_tiles: int | None = PREVIEW_MAX_TILES_PER_MUNICIPIO,
) -> None:
    if rgb_assets.empty:
        print(f"no RGB assets available for preview ({municipio}).")
        return

    subset = rgb_assets[rgb_assets["municipio"] == municipio].copy()
    if post_maria_only:
        subset = subset[subset["post_maria"]]
    if allowed_qualities:
        subset = subset[subset["imagery_quality"].isin(allowed_qualities)]
    subset = subset.sort_values(["imagery_date", "tile_id"], ascending=[False, True])
    if max_tiles is not None:
        subset = subset.head(int(max_tiles))
    if subset.empty:
        print(f"no preview mosaic tiles for {municipio} after post-Maria / quality filters.")
        return

    sources = []
    try:
        for path in subset["rgb_path"]:
            sources.append(rasterio.open(path))
        mosaic, _ = merge_rasters(sources, indexes=[1, 2, 3], res=(mosaic_res_m, mosaic_res_m))
    finally:
        for src in sources:
            src.close()

    plt.figure(figsize=(14, 14))
    plt.imshow(stretch_rgb(mosaic))
    date_min = subset["imagery_date"].min().date()
    date_max = subset["imagery_date"].max().date()
    plt.title(
        f"{municipio} RGB preview mosaic ({len(subset)} tiles, {date_min} -> {date_max}, res={mosaic_res_m}m)"
    )
    plt.axis("off")
    plt.show()


# %% [markdown]
# ## Step 1 — Load manifest, show budget

# %%
if __name__ == "__main__":
    db_path = resolve_db_path()
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")

    manifest = load_ready_manifest(con)
    candidate_manifest = load_candidate_manifest(con)
    print(f"ready tiles: {len(manifest):,}")
    print(f"priority counts:\n{manifest['priority_score'].value_counts().sort_index(ascending=False).to_string()}")
    print(f"candidate tiles for post-Maria re-probing: {len(candidate_manifest):,}")
    print(
        "candidate status counts:\n"
        f"{candidate_manifest['status'].value_counts().to_string()}"
    )

    pre_summary = ledger_summary()
    print(f"ledger before run: {pre_summary}")
    cap = pre_summary["data_layers_remaining_to_cap"]
    if MAX_TILES_THIS_RUN is not None:
        cap = min(cap, MAX_TILES_THIS_RUN)
    print(f"will fetch up to {cap} tiles this run (global cap {DATA_LAYERS_BUDGET_CAP}).")

# %% [markdown]
# ## Step 1b — Probe Building Insights For Post-Maria Candidates
#
# This uses the plain REST endpoint with `SOLAR_API_KEY` from `.env`, not the
# Python SDK. We probe both `BASE` and `MEDIUM`, balance the request queue
# across municipalities when a probe cap is set, and then keep only candidates
# whose returned imagery is post-Maria and has the 0.25 m tiers we want for
# downstream GeoAI work.

# %%
if __name__ == "__main__":
    solar_api_key = os.getenv("SOLAR_API_KEY")
    if not solar_api_key:
        raise RuntimeError("SOLAR_API_KEY not found in .env or environment.")

    probe_manifest = build_probe_manifest(
        candidate_manifest,
        min_priority=BUILDING_INSIGHTS_MIN_PRIORITY,
        max_tiles=MAX_BUILDING_INSIGHTS_THIS_RUN,
    )

    print(
        f"probing {len(probe_manifest):,} tile centroids via buildingInsights REST "
        f"(priority >= {BUILDING_INSIGHTS_MIN_PRIORITY}, requestedQualities={BUILDING_INSIGHTS_REQUIRED_QUALITIES})"
    )
    print("probe manifest by municipio:\n", probe_manifest["municipio"].value_counts().to_string())

    building_probe = probe_building_insights_multi_quality(
        probe_manifest,
        api_key=solar_api_key,
        required_qualities=BUILDING_INSIGHTS_REQUIRED_QUALITIES,
    )
    building_probe["post_maria"] = building_probe["imagery_date"] >= POST_MARIA_CUTOFF

    print(
        "buildingInsights status counts by requested quality:\n",
        building_probe.groupby(["requested_quality", "status"]).size().to_string(),
    )
    ok_probe = building_probe[building_probe["status"] == "ok"].copy()
    if not ok_probe.empty:
        print(
            f"imagery_date range among successful probes: "
            f"{ok_probe['imagery_date'].min().date()} -> {ok_probe['imagery_date'].max().date()}"
        )
        print(
            "imagery quality counts by requested quality:\n",
            ok_probe.groupby(["requested_quality", "imagery_quality"]).size().to_string(),
        )

    post_maria_candidates = select_post_maria_candidates(
        building_probe,
        cutoff=POST_MARIA_CUTOFF,
        allowed_qualities=POST_MARIA_FETCH_ALLOWED_QUALITIES,
    )
    print(f"post-Maria candidates found: {len(post_maria_candidates):,}")
    if not post_maria_candidates.empty:
        print(
            "post-Maria candidates by municipio:\n",
            post_maria_candidates["municipio"].value_counts().to_string(),
        )
        print(
            "post-Maria candidates by fetch quality:\n",
            post_maria_candidates["fetch_required_quality"].value_counts().to_string(),
        )
    post_maria_candidates[
        [
            "tile_id",
            "bg_geoid",
            "municipio",
            "priority_score",
            "osm_pv_count",
            "building_count",
            "requested_quality",
            "fetch_required_quality",
            "imagery_quality",
            "imagery_date",
            "building_name",
            "postal_code",
            "max_array_panels_count",
            "roof_area_m2",
        ]
    ].head(50)


# %% [markdown]
# ## Step 2 — Fetch post-Maria BASE / MEDIUM candidates

# %%
if __name__ == "__main__":
    fetch_manifest = build_fetch_manifest(
        candidate_manifest,
        required_quality=DATA_LAYERS_DEFAULT_REQUIRED_QUALITY,
        post_maria_candidates=globals().get("post_maria_candidates"),
        fetch_only_post_maria_candidates=FETCH_ONLY_POST_MARIA_CANDIDATES,
    )
    print(
        f"fetching {len(fetch_manifest):,} tiles with default required_quality={DATA_LAYERS_DEFAULT_REQUIRED_QUALITY} "
        f"(post-Maria-only={FETCH_ONLY_POST_MARIA_CANDIDATES})"
    )
    if "municipio" in fetch_manifest.columns:
        print("fetch manifest by municipio:\n", fetch_manifest["municipio"].value_counts().to_string())
    if "status" in fetch_manifest.columns:
        print("fetch manifest source status counts:\n", fetch_manifest["status"].value_counts().to_string())
    if "required_quality" in fetch_manifest.columns:
        print(
            "dataLayers required_quality among fetch candidates:\n",
            fetch_manifest["required_quality"].value_counts(dropna=False).to_string(),
        )
    if "bi_imagery_date" in fetch_manifest.columns and fetch_manifest["bi_imagery_date"].notna().any():
        print(
            f"buildingInsights imagery_date range among fetch candidates: "
            f"{fetch_manifest['bi_imagery_date'].min().date()} -> {fetch_manifest['bi_imagery_date'].max().date()}"
        )

    results = fetch_tiles_budgeted(
        fetch_manifest.head(int(cap)) if cap is not None else fetch_manifest,
        budget_cap=DATA_LAYERS_BUDGET_CAP,
    )
    print("fetch result counts:\n", results["status"].value_counts(dropna=False).to_string())
    print("ledger after run:", ledger_summary())

# %% [markdown]
# ## Step 3 — Sync manifest status column from ledger

# %%
if __name__ == "__main__":
    from utils.solar_api import DEFAULT_LEDGER_PATH

    if Path(DEFAULT_LEDGER_PATH).exists():
        ledger_df = pd.read_parquet(DEFAULT_LEDGER_PATH)
        latest = (
            ledger_df[ledger_df["endpoint"] == "dataLayers:get"]
            .sort_values("timestamp_utc")
            .groupby("tile_id", as_index=False)
            .last()[["tile_id", "status"]]
        )
        # Only promote tiles on TERMINAL ledger statuses. Transient
        # 'error' / 'budget_exhausted' rows must leave the manifest at
        # 'ready' so the next run can retry them — otherwise a single
        # storm of 4xx errors permanently strands high-priority tiles
        # (this happened the first 3 runs of 07).
        latest = latest[latest["status"].isin(["ok", "not_found"])]
        con.register("ledger_latest", latest)
        con.execute(
            f"""
            UPDATE {MANIFEST_TABLE} AS m
            SET status = CASE l.status
                WHEN 'ok' THEN 'fetched'
                WHEN 'not_found' THEN 'no_coverage'
                ELSE m.status
            END
            FROM ledger_latest AS l
            WHERE m.tile_id = l.tile_id;
            """
        )
        con.unregister("ledger_latest")

    summary = con.execute(
        f"SELECT status, COUNT(*) AS tiles FROM {MANIFEST_TABLE} GROUP BY 1 ORDER BY 1;"
    ).fetchdf()
    print(summary.to_string(index=False))
    con.close()

# %% [markdown]
# ## Step 4 — Imagery-date distribution of fetched tiles
#
# Google Solar's aerial imagery vintage varies by location. For our
# hurricane-Maria-recovery framing we care about capture dates relative to
# 2017-09-20. The ledger already stores `imagery_date` per fetch, so we can
# summarize without materializing anything to disk.

# %%
if __name__ == "__main__":
    from utils.solar_api import DEFAULT_LEDGER_PATH

    MARIA_LANDFALL = pd.Timestamp("2017-09-20")

    ledger_df = pd.read_parquet(DEFAULT_LEDGER_PATH)
    dl_ok = ledger_df[
        (ledger_df["endpoint"] == "dataLayers:get") & (ledger_df["status"] == "ok")
    ].copy()
    dl_ok["imagery_date"] = pd.to_datetime(dl_ok["imagery_date"], errors="coerce")
    dl_ok["imagery_year"] = dl_ok["imagery_date"].dt.year.astype("Int64")
    dl_ok["post_maria"] = dl_ok["imagery_date"] >= MARIA_LANDFALL

    year_dist = (
        dl_ok.groupby(["imagery_year", "municipio"], dropna=False)
        .size()
        .unstack("municipio", fill_value=0)
    )
    year_dist["total"] = year_dist.sum(axis=1)
    year_dist["pct"] = (year_dist["total"] / year_dist["total"].sum() * 100).round(1)
    year_dist = year_dist.sort_index()

# %%
if __name__ == "__main__":
    era_counts = dl_ok["post_maria"].value_counts().rename({True: "post_Maria", False: "pre_Maria"})
    print(f"successful dataLayers fetches: {len(dl_ok):,}")
    print(f"imagery_date range: {dl_ok['imagery_date'].min().date()} → {dl_ok['imagery_date'].max().date()}")
    print("pre/post Hurricane Maria (2017-09-20):")
    print(era_counts.to_string())
    year_dist  # displays as dataframe in Jupyter


# %% [markdown]
# ## Step 5 — Preview mosaics of fetched post-Maria RGB imagery
#
# Build per-municipio preview mosaics from fetched RGB GeoTIFFs so we can
# visually inspect the post-Maria coverage before moving on to GeoAI chip
# generation and panel detection.

# %%
if __name__ == "__main__":
    fetched_rgb_assets = scan_fetched_rgb_assets(SOLAR_ROOT)
    if fetched_rgb_assets.empty:
        print("no fetched RGB assets found — run Step 2 first.")
    else:
        print(
            "fetched RGB assets by municipio / quality / post_maria:\n",
            fetched_rgb_assets.groupby(["municipio", "imagery_quality", "post_maria"]).size().to_string(),
        )
        preview_source = (
            fetched_rgb_assets[fetched_rgb_assets["post_maria"]].copy()
            if PREVIEW_POST_MARIA_ONLY
            else fetched_rgb_assets.copy()
        )
        preview_municipios = sorted(preview_source["municipio"].dropna().unique().tolist())
        for municipio in preview_municipios:
            plot_preview_mosaic(
                fetched_rgb_assets,
                municipio,
                post_maria_only=PREVIEW_POST_MARIA_ONLY,
                allowed_qualities=PREVIEW_ALLOWED_QUALITIES,
                mosaic_res_m=PREVIEW_MOSAIC_RES_M,
                max_tiles=PREVIEW_MAX_TILES_PER_MUNICIPIO,
            )
