"""Phase 2 — Run U-Net detection on all videos with saved configs.

For each video found in VIDEO_DIR:
  1. Load config JSON from Phase 1 (ROI, model path)
  2. Detect pupils on all frames
  3. Post-process (smoothing, gap filling)
  4. Save results (CSV, masks, plot, preview video, updated config)

Requires: run batch_roi.py first to create config JSONs.

Usage:
    python batch_detect.py
"""

import json
import logging
from pathlib import Path

from pupil_track import Pupil

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  USER PARAMETERS — edit these before running                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Directory containing config JSON files from Phase 1
# (also used to find corresponding video files)
CONFIG_DIR = r"V:\li\RTWF-20251212-ANALYSIS\pupil_new"

# Directory where configs and results are stored.
# If empty, defaults to the same folder as each video.
OUTPUT_DIR = r"V:\li\RTWF-20251212-ANALYSIS\pupil_new"

# Input size — must match Phase 1 and training.
INPUT_SIZE = 128

# Detection threshold (probability above which a pixel is "pupil")
THRESHOLD = 0.5

# Post-processing: smoothing window size (in frames)
SMOOTH_WINDOW = 10

# For testing: limit number of frames (0 = process all frames)
MAX_FRAMES = 0  # Set to 0 to process all frames

# ════════════════════════════════════════════════════════════════════════

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov"}


def load_config_video_path(config_path: Path) -> Path | None:
    """Load video path from config JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        video_path = Path(config['data']['path'])
        return video_path if video_path.exists() else None
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return None


def collect_configs(directory: str | Path) -> list[Path]:
    """Return sorted list of config JSON files in directory."""
    d = Path(directory)
    return [f for f in sorted(d.iterdir()) if f.name.endswith('_config.json')]


def main():
    configs = collect_configs(CONFIG_DIR)
    if not configs:
        print(f"No config JSON files found in {CONFIG_DIR}")
        print("Run batch_roi.py first to create config files.")
        return

    print(f"Found {len(configs)} config file(s) in {CONFIG_DIR}\n")

    skipped = []
    for i, config_path in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {config_path.name}")
        
        # Read video path from config file
        video_path = load_config_video_path(config_path)
        
        if video_path is None:
            print(f"  SKIPPED — video file not found or invalid config\n")
            skipped.append(config_path.name)
            continue
            
        output_dir = OUTPUT_DIR or str(video_path.parent)
        print(f"  Processing: {video_path.name}")

        p = Pupil(
            video_path=str(video_path),
            output_dir=output_dir,
            input_size=INPUT_SIZE,
        )

        # Load config from Phase 1 (we know it exists)
        p.load_config()
        # No need to check if p.roi is None since we found the config file

        p.config.threshold = THRESHOLD
        p.config.smooth_window = SMOOTH_WINDOW

        # Get frame info and detect
        all_indices = p.get_all_indices()
        if MAX_FRAMES > 0 and len(all_indices) > MAX_FRAMES:
            all_indices = all_indices[:MAX_FRAMES]
            print(f"  Limited to first {MAX_FRAMES} frames for testing")
            
        total_frames = len(all_indices)
        print(f"  Detecting pupils in {total_frames} frames...")
        
        # Detect (this may take a while for large videos)
        p.detect("unet", frame_indices=all_indices, threshold=THRESHOLD)
        print(f"  Detection complete!")

        # Post-process and save
        p.postprocess("unet")
        p.save_results("unet")

        print(f"  Done — results saved to {output_dir}\n")
        p.close()

    # Summary
    n_done = len(configs) - len(skipped)
    print(f"Complete: {n_done}/{len(configs)} videos processed.")
    if skipped:
        print(f"Skipped ({len(skipped)}): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
