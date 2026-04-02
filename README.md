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

### Prerequisites

- Python 3.10 or higher
- [Git](https://git-scm.com/downloads) (for cloning the repository)
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended for environment management)

### Step-by-Step Installation

#### 1. Clone the repository

Open a terminal (Anaconda Prompt on Windows) and run:

```bash
git clone https://github.com/YOUR_USERNAME/pupil_track.git
cd pupil_track
```

**Note:** Replace `YOUR_USERNAME` with the actual repository owner's GitHub username.

#### 2. Create a conda environment

Create an isolated environment with Python 3.10:

```bash
conda create -n pupil_track python=3.10 -y
conda activate pupil_track
```

**Note:** Always activate this environment before using pupil_track:
```bash
conda activate pupil_track
```

#### 3. Install PyTorch

**Option A: GPU (recommended for faster processing)**

If you have an NVIDIA GPU with CUDA support:

```bash
# For CUDA 12.4 (most modern GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# For CUDA 11.8 (older GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Option B: CPU only**

```bash
pip install torch torchvision
```

Verify your PyTorch installation:

```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

If CUDA is available, you should see `CUDA available: True`.

#### 4. Install pupil_track

**Basic installation (core features only):**

```bash
pip install .
```

**Full installation (includes GUI, recommended):**

```bash
pip install ".[gui]"
```

#### 5. Verify installation

```bash
python -c "import pupil_track; print('pupil_track version:', pupil_track.__version__)"
pupil-track --help
```

### Quick Installation (Advanced Users)

```bash
git clone https://github.com/YOUR_USERNAME/pupil_track.git
cd pupil_track
conda create -n pupil_track python=3.10 -y
conda activate pupil_track
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ".[gui]"
```

### Using pupil_track After Installation

Each time you open a new terminal and want to use pupil_track:

```bash
conda activate pupil_track
cd path/to/pupil_track  # Navigate to your project directory if needed
```

To deactivate the environment when you're done:

```bash
conda deactivate
```

### Performance Notes

- **GPU acceleration:** ~10-50x faster than CPU for inference and training
- **CPU only:** Functional but slower; suitable for small datasets or occasional use
- **VRAM requirements:** ~2GB for inference, ~4GB for training

### Dependencies

All dependencies are automatically installed when you run `pip install .` or `pip install ".[gui]"`:

**Core dependencies:**
- torch >= 2.0
- numpy >= 1.23
- opencv-python >= 4.8
- albumentations >= 1.3
- scipy >= 1.9
- matplotlib >= 3.6
- tqdm >= 4.64
- Pillow >= 9.3

**Optional dependencies:**
- PyQt5 >= 5.15 (for GUI, install with `pip install ".[gui]"`)
- pytest >= 7.0 (for development, install with `pip install ".[dev]"`)

### Troubleshooting

**"Command not found: conda"**
- Install Anaconda or Miniconda from the links in Prerequisites

**"No module named 'pupil_track'"**
- Make sure you're in the pupil_track directory and have run `pip install .`
- Activate your conda environment: `conda activate pupil_track`

**"CUDA out of memory"**
- Reduce batch size or use CPU mode
- Close other GPU-intensive applications

**Import errors after installation**
- Try reinstalling: `pip install --force-reinstall .`

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
