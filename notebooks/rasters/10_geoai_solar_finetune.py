# %% [markdown]
# # GeoAI Solar Detector Fine-Tuning
#
# Fine-tunes Mask R-CNN from the pretrained `geoai.SolarPanelDetector` weights
# on the chips exported by `09_geoai_training_data.py`.

# %%
"""10_geoai_solar_finetune.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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

TRAIN_ROOT = PROJECT_ROOT / "output" / "geoai_train"
IMAGES = TRAIN_ROOT / "images"
MASKS = TRAIN_ROOT / "masks"
MODEL_OUT = PROJECT_ROOT / "output" / "models"

NUM_CLASSES = 2        # background + PV
NUM_EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
SEED = 42


# %%
if __name__ == "__main__":
    if not IMAGES.exists() or not MASKS.exists():
        print(f"expected training data at {TRAIN_ROOT}; run 09_geoai_training_data first.")
        sys.exit(0)

    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    import geoai

    geoai.train_MaskRCNN_model(
        images_folder=str(IMAGES),
        masks_folder=str(MASKS),
        output_folder=str(MODEL_OUT),
        num_classes=NUM_CLASSES,
        pretrained=True,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        val_split=VAL_SPLIT,
        seed=SEED,
        save_best_only=True,
    )
    print(f"training complete — checkpoints under {MODEL_OUT}")
