"""Helpers for selecting external preview rasters from preferred STAC sources."""

from __future__ import annotations

from pathlib import Path


NAIP_SOURCE_NAMES = ("pr_naip", "naip_2021_pr")
DEFAULT_IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def collect_naip_stac_preview_rasters(
    project_root: Path,
    *,
    exclude_stems: set[str] | None = None,
    source_names: tuple[str, ...] = NAIP_SOURCE_NAMES,
    supported_image_extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
    required_name_fragment: str | None = "visual_epsg3857",
) -> list[Path]:
    exclude_stems = exclude_stems or set()
    stac_tile_root = project_root / "outputs" / "stac_tiles"
    if not stac_tile_root.exists():
        return []

    source_name_lookup = {name.lower(): name for name in source_names}
    required_fragment = required_name_fragment.lower() if required_name_fragment else None
    candidates: list[Path] = []
    seen: set[Path] = set()

    for path in stac_tile_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in supported_image_extensions:
            continue
        if path.stem in exclude_stems:
            continue
        if required_fragment is not None and required_fragment not in path.name.lower():
            continue
        if not any(part.lower() in source_name_lookup for part in path.parts):
            continue

        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(resolved)

    return sorted(candidates)


def infer_stac_source_name(
    path: Path,
    source_names: tuple[str, ...] = NAIP_SOURCE_NAMES,
) -> str:
    source_name_lookup = {name.lower(): name for name in source_names}
    for part in path.parts:
        normalized = part.lower()
        if normalized in source_name_lookup:
            return source_name_lookup[normalized]
    return "external"