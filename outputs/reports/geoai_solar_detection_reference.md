# GeoAI — Solar Panel Detection Reference

Compiled from the `geoai-py` examples (Opengeos) covering
`solar_panel_detection`, `train_solar_panel_detection`, `image_tiling`,
`create_training_data`, and `dinov3`. Scope: fine-tuning Mask R-CNN on
OSM-seeded Puerto Rico chips and running inference on Google Solar API RGB
tiles.

## 1. Pretrained baseline — `geoai.SolarPanelDetector`

```python
import geoai

detector = geoai.SolarPanelDetector()  # downloads Davis, CA NAIP 0.6 m Mask R-CNN weights

detector.generate_masks(
    raster_path="data/rasters/solar/San_Juan/7211013001/tile_00042_rgb_HIGH.tif",
    output_path="output/pv_masks/tile_00042.tif",
    confidence_threshold=0.4,
    mask_threshold=0.5,
    min_object_area=100,
    overlap=0.25,
    chip_size=(400, 400),
    batch_size=4,
)
```

- Model: Mask R-CNN (torchvision) pretrained on NAIP 0.6 m Davis, CA tile from
  the HuggingFace `giswqs/geospatial` repo.
- Input: any 3-band RGB GeoTIFF. The detector internally chips the raster.
- Output: single-channel GeoTIFF with instance IDs + per-instance confidence
  sidecar CSV.

**Resolution caveat**: Solar API RGB is 0.1 m at HIGH and 0.25 m at
MEDIUM/BASE. The baseline model was trained at 0.6 m, so it will under-detect
on native Solar imagery. Fine-tuning is expected; downsampling Solar RGB to
0.6 m is a viable fallback for zero-shot inference.

## 2. Inference pipeline helpers

- `geoai.object_detection(raster_path, output_path, model_path, window_size=400, overlap=100, confidence_threshold=0.4, batch_size=4, num_channels=3)` —
  generic wrapper; takes a fine-tuned Mask R-CNN `.pth` and emits a
  georeferenced instance mask.
- `geoai.orthogonalize(input_mask, output_vector, epsilon=0.2)` — polygonize +
  Ramer–Douglas–Peucker simplification to regular rectangles (epsilon in m).
- `geoai.add_geometric_properties(input_vector, output_vector)` — appends
  `area_m2`, `perimeter_m`, `elongation`, `compactness`, `convexity` columns.

Post-processing filter used in this project:
`elongation < 10 AND 3 <= area_m2 <= 500`.

## 3. Training data creation

### 3.1 `geoai.export_geotiff_tiles`

Single-image tiling with labels:

```python
geoai.export_geotiff_tiles(
    in_raster="solar_rgb.tif",
    out_folder="output/geoai_train",
    in_class_data="pr_osm_rooftop_pv_polygons.geojson",
    tile_size=512,
    stride=256,
    buffer_radius=0.5,            # metres; expand small PV polygons for seg
    class_value_field="class",    # label column
    skip_empty_tiles=True,
)
```

Produces `images/` and `masks/` folders with matched filenames plus a COCO-ish
`annotations.json`. Masks are integer class indices (0=background).

### 3.2 `geoai.export_geotiff_tiles_batch`

Batch over many rasters. Three binding modes:

1. Single vector file applied to all rasters in `images_folder`.
2. Folder of per-raster vectors — `match_by_name=True` (expects the
   image-and-vector filenames to share a stem).
3. Folder of per-raster vectors — `match_by_name=False`, lexicographic sort
   alignment.

Example used here:

```python
geoai.export_geotiff_tiles_batch(
    images_folder="data/rasters/solar/San_Juan/",
    masks_file="data/vectors/osm_pv_for_training.geojson",
    output_folder="output/geoai_train",
    tile_size=512,
    stride=256,
    buffer_radius=0.5,
    class_value_field="class",
    skip_empty_tiles=True,
    recursive=True,
)
```

### 3.3 `geoai.create_training_data`

Higher-level helper that wraps the above and prints a COCO dataset summary.
Use it for quick sanity-checks; use `export_geotiff_tiles_batch` for the real
pipeline.

## 4. Fine-tuning — `geoai.train_MaskRCNN_model`

```python
geoai.train_MaskRCNN_model(
    images_folder="output/geoai_train/images",
    masks_folder="output/geoai_train/masks",
    output_folder="output/models",
    num_classes=2,                # background + PV
    pretrained=True,
    resume_from=None,             # or path to Davis checkpoint to warm-start
    num_epochs=30,
    batch_size=4,
    learning_rate=1e-4,
    val_split=0.2,
    seed=42,
    save_best_only=True,
    device=None,                  # autodetect CUDA
)
```

Saves `best_model.pth` + training curves PNG. Our project writes
`pr_solar_mask_rcnn.pth` as a renamed copy of the best checkpoint.

## 5. Optional validation — DINOv3 similarity

```python
geoai.create_similarity_map(
    input_image="tile_00042_rgb.tif",
    query_coords=[(lon, lat)],             # seed = confirmed OSM PV centroid
    output_dir="output/dinov3",
    model_name="dinov3_vitl16",
)
```

Produces a patch-level similarity heatmap that we use as a sanity check on
low-confidence detector predictions. Not part of the main pipeline.

## 6. Expected per-phase wiring in this project

| Project notebook | GeoAI call |
|---|---|
| `09_geoai_training_data.py` | `export_geotiff_tiles_batch` over priority-3 tiles + OSM PV polygons |
| `10_geoai_solar_finetune.py` | `train_MaskRCNN_model` warm-started from `SolarPanelDetector` weights |
| `11_geoai_solar_inference.py` | `object_detection` → `orthogonalize` → `add_geometric_properties` |

## 7. Key references

- <https://opengeoai.org/examples/solar_panel_detection/>
- <https://opengeoai.org/examples/train_solar_panel_detection/>
- <https://opengeoai.org/examples/image_tiling/>
- <https://opengeoai.org/examples/create_training_data/>
- <https://opengeoai.org/examples/dinov3/>
- PyPI: `pip install geoai-py` — already pinned in `requirements.txt`.
