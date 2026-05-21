"""Shared static review helpers for GeoAI training and inference notebooks."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio


def _prepare_image_for_plot(image: np.ndarray) -> np.ndarray:
    display = np.asarray(image)
    if display.ndim == 2:
        display = np.repeat(display[np.newaxis, :, :], 3, axis=0)
    if display.shape[0] == 1:
        display = np.repeat(display, 3, axis=0)
    if display.shape[0] > 3:
        display = display[:3]
    display = np.moveaxis(display, 0, -1).astype(np.float32)
    if display.size and display.max() > 1.0:
        display /= 255.0
    return display


def _read_image(image_path: Path) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    with rasterio.open(image_path) as src:
        image = src.read()
        bounds = src.bounds
    extent = (float(bounds.left), float(bounds.right), float(bounds.bottom), float(bounds.top))
    return _prepare_image_for_plot(image), extent


def _read_mask(mask_path: Path) -> np.ndarray:
    with rasterio.open(mask_path) as src:
        return src.read(1)


def render_prediction_review_bundle(
    *,
    image_path: Path,
    output_stem: Path,
    predicted_mask_path: Path | None = None,
    raw_mask_path: Path | None = None,
    grounded_mask_path: Path | None = None,
    vector_path: Path | None = None,
    suptitle: str | None = None,
) -> dict[str, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    image, extent = _read_image(image_path)

    panels: list[tuple[str, np.ndarray | None, str | None]] = [("Source raster", None, None)]
    if raw_mask_path is not None and raw_mask_path.exists():
        panels.append(("Raw label", _read_mask(raw_mask_path), "Reds"))
    if grounded_mask_path is not None and grounded_mask_path.exists():
        panels.append(("Grounded label", _read_mask(grounded_mask_path), "Greens"))
    if predicted_mask_path is not None and predicted_mask_path.exists():
        panels.append(("Predicted mask", _read_mask(predicted_mask_path), "Blues"))

    review_png = output_stem.with_name(f"{output_stem.name}_review.png")
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5), squeeze=False)
    for ax, (title, mask, cmap) in zip(axes[0], panels):
        ax.imshow(image, extent=extent)
        if mask is not None:
            overlay = np.ma.masked_where(mask <= 0, mask)
            ax.imshow(overlay, extent=extent, cmap=cmap, alpha=0.35, vmin=0, vmax=max(1, int(mask.max())))
        ax.set_title(title)
        ax.set_axis_off()
    title = suptitle or image_path.stem
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(review_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    outputs = {"review_png_path": review_png}
    if vector_path is not None and vector_path.exists():
        try:
            vectors = gpd.read_file(vector_path)
        except Exception:
            vectors = None
        if vectors is not None and not vectors.empty:
            vector_png = output_stem.with_name(f"{output_stem.name}_vector_review.png")
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(image, extent=extent)
            vectors.plot(
                ax=ax,
                facecolor=(1.0, 0.2, 0.2, 0.25),
                edgecolor=(1.0, 1.0, 1.0, 0.9),
                linewidth=0.8,
            )
            ax.set_title("Prediction polygons")
            ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(vector_png, dpi=200, bbox_inches="tight")
            plt.close(fig)
            outputs["vector_review_png_path"] = vector_png
    return outputs