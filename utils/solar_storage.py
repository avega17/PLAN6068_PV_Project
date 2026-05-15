"""Sync Google Solar API GeoTIFFs from repo-local storage to the external drive.

Kept out of the main ingest loop so dev work stays portable. Uses `rsync` when
available (preferred) and falls back to a pure-Python mirror copy.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv


def _resolve_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if any((candidate / m).exists() for m in ("project_rules.md", ".git")):
            return candidate
    return current


PROJECT_ROOT = _resolve_project_root()
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(env_value: str | None, fallback: Path) -> Path:
    if not env_value:
        return fallback
    p = Path(env_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


DEFAULT_LOCAL_ROOT = _resolve_path(os.getenv("SOLAR_RASTER_ROOT"), PROJECT_ROOT / "data" / "rasters" / "solar")
DEFAULT_EXTERNAL_ROOT = _resolve_path(os.getenv("SOLAR_RASTER_EXTERNAL"), Path("/mnt/p/plan6068_dataset/solar"))


def sync_to_external(
    local_root: Path = DEFAULT_LOCAL_ROOT,
    external_root: Path = DEFAULT_EXTERNAL_ROOT,
    *,
    dry_run: bool = False,
    use_rsync: bool = True,
) -> dict[str, int | str]:
    """One-way sync from ``local_root`` to ``external_root``.

    Returns a small summary dict. Does not delete files that no longer exist
    locally — treat the external drive as append-only.
    """

    if not local_root.exists():
        return {"status": "local_missing", "local": str(local_root), "files_copied": 0}

    external_root.mkdir(parents=True, exist_ok=True)

    rsync = shutil.which("rsync") if use_rsync else None
    if rsync:
        cmd = [rsync, "-a", "--info=stats2", "--prune-empty-dirs"]
        if dry_run:
            cmd.append("--dry-run")
        cmd += [f"{local_root}/", f"{external_root}/"]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "status": "ok" if proc.returncode == 0 else "rsync_error",
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }

    # Pure-Python fallback (slow but portable).
    files_copied = 0
    for src_path in local_root.rglob("*"):
        if not src_path.is_file():
            continue
        rel = src_path.relative_to(local_root)
        dest = external_root / rel
        if dest.exists() and dest.stat().st_size == src_path.stat().st_size:
            continue
        if dry_run:
            files_copied += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest)
        files_copied += 1
    return {"status": "ok", "files_copied": files_copied, "mode": "python_fallback"}


__all__ = ["DEFAULT_LOCAL_ROOT", "DEFAULT_EXTERNAL_ROOT", "sync_to_external"]
