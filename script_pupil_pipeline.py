# %%
"""
Script-based workflow for pupil tracking with the pupil_track package.

This script demonstrates both the U-Net detection workflow (using a
pre-trained model) and the U-Net training workflow (label → train → detect).
All functionality lives in the Pupil class — this script just calls methods
in the right order.

Edit the parameters in the section below, then run:
    python script_pupil_track.py
"""

import logging
from pathlib import Path

from pupil_track import Pupil

# Show log messages so you can follow what the pipeline is doing
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Training parameters (only used if you run the training section) ──
TRAIN_IMAGES_DIR = r""  # directory of cropped grayscale PNGs
TRAIN_MASKS_DIR  = r""  # directory of matching binary mask PNGs
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
PATIENCE = 10             # stop early if no improvement for this many epochs
MODLE_SAVE_DIR = r"output"  # where to save model checkpoints during training

# # --- Train the U-Net ---
# # Uses the image+mask pairs from the annotation step (or from your own
# # directories if you set TRAIN_IMAGES_DIR / TRAIN_MASKS_DIR above).
# # Returns the path to the best checkpoint (highest validation Dice score).
# #
# # If you have your own labeled data, pass data_dirs:
# #   best = p.train(data_dirs=[(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)], ...)
# # Otherwise it uses p.img_dir and p.mask_dir from the annotation step.

p = Pupil()
best_model = p.train(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    patience=PATIENCE,
    data_dirs=[(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)],  # uncomment to use custom dirs
    checkpoint_dir= Path(MODLE_SAVE_DIR) / "checkpoints",
)
