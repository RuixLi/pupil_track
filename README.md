# pupil_track

Python package for mouse pupil segmentation in videos. Supports multiple detection methods (U-Net, integrodifferential operator, starburst) with both a script-based workflow and an interactive PyQt5 GUI.

## Features

- **U-Net segmentation** — shallow U-Net trained on labeled eye frames, fast and accurate, robust to noise

- **Manual correction** — manually fix missed detections or false positives after inference, then re-run post-processing without retraining

- **Interactive annotation** — draw pupil masks in GUI on your data, then train your own model

- **Post-processing** — temporal smoothing, outlier removal, gap filling

- **Export** — CSV results, binary masks (.npy), time-series plots, annotated preview videos

- **Reproducible configs** — JSON configuration saved with every run (video info, ROI, model, parameters)

- **Batch processing** — define ROI for multiple videos, then run detection in a second phase using saved configs without intervention

## Installation

Requires Python 3.10+ and PyTorch.

```bash
# Core package
pip install .

# With GUI support
pip install ".[gui]"
```

### GPU acceleration (recommended)

For faster inference and training, install PyTorch with CUDA support **before** installing this package:

```bash
# CUDA 12.1 (recommended for most modern GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install pupil_track
pip install .
```

To verify GPU is available:

```python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Shows GPU name
```

**Note:** CPU-only inference is functional but ~10-50x slower. For batch processing hundreds of videos, GPU is strongly recommended.

### Dependencies

- torch >= 2.0
- numpy >= 1.23
- opencv-python >= 4.8
- albumentations >= 1.3
- scipy >= 1.9
- matplotlib >= 3.6
- tqdm >= 4.64
- PyQt5 >= 5.15 (optional, for GUI)

## Quick start

### GUI

```bash
python -m pupil_track gui
```

### Batch processing (script workflow)

Batch processing runs in two phases:

**Phase 1 — Define ROI** (`batch_roi.py`): For each video in a folder, interactively select the region of interest and save a config JSON with ROI + model path. This is fast.

```bash
python batch_roi.py
```

**Phase 2 — Detect** (`batch_detect.py`): For each video with a saved config, run U-Net detection on all frames, post-process, and export results. This can be slow.

```bash
python batch_detect.py
```

Edit the `VIDEO_DIR`, `OUTPUT_DIR`, and `MODEL_PATH` parameters at the top of each script before running.

### CLI training

```bash
python -m pupil_track train --data-dir ./images ./masks --epochs 50
```

## Project structure

```
src/pupil_track/
    __init__.py          # Package entry, exports Pupil class
    pupil.py             # Main Pupil class — central pipeline object
    config.py            # Config dataclass with JSON serialization
    roi.py               # Auto and manual ROI detection
    video_io.py          # VideoReader wrapper around cv2.VideoCapture
    annotate.py          # Interactive mask annotation GUI
    train.py             # U-Net training loop with early stopping
    infer.py             # Model loading and inference utilities
    postprocess.py       # Feature extraction, smoothing, gap filling
    visualize.py         # Preview playback with mask/ellipse overlays
    gui.py               # PyQt5 GUI application
    model/               # ShallowUNet architecture
    detectors/           # Detection backends (unet, intdiff, starburst)
    utils/               # Dataset, evaluation, loss functions
```

## Usage example

```python
from pupil_track import Pupil

p = Pupil(video_path="eye_video.avi", output_dir="./results", input_size=128)

# Set ROI
p.auto_roi()
p.show_roi()

# Load model and detect
p.model_path = "best_model.pth"
masks = p.detect("unet", frame_indices=p.get_all_indices(), threshold=0.5)

# Post-process and save
raw, smoothed = p.postprocess("unet")
p.save_results("unet")

# Preview
p.preview("unet")
p.close()
```

## License

MIT
