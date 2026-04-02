"""Phase 1 — Define and confirm ROI for all videos in a directory.

For each video found in VIDEO_DIR:
  1. Auto-detect ROI
  2. Show the ROI overlay — press any key to accept, or 'q' to quit
  3. Save config JSON (with ROI + model path) to OUTPUT_DIR

After this script, every video has a {stem}_config.json ready for Phase 2
(batch_detect.py).

Usage:
    python batch_roi.py
"""

import logging
from pathlib import Path

from pupil_track import Pupil

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  USER PARAMETERS — edit these before running                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Directory containing eye video files (.avi, .mp4)
VIDEO_DIR = r"G:\pupil_demo\movie"

# Directory where config JSONs are saved.
# If empty, defaults to the same folder as each video.
OUTPUT_DIR = r""

# Path to a trained U-Net model (.pth file).
# Saved into each config so Phase 2 knows which model to use.
MODEL_PATH = r"G:\pupil_demo\train\checkpoints\best_model.pth"

# Input size — must match the size used during training.
INPUT_SIZE = 128

# ROI display scale (fraction of original frame size)
ROI_SCALE = 0.5

# ════════════════════════════════════════════════════════════════════════

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov"}


def collect_videos(directory: str | Path) -> list[Path]:
    """Return sorted list of video files in directory."""
    d = Path(directory)
    videos = [f for f in sorted(d.iterdir()) if f.suffix.lower() in VIDEO_EXTENSIONS]
    return videos


def main():
    videos = collect_videos(VIDEO_DIR)
    if not videos:
        print(f"No video files found in {VIDEO_DIR}")
        return

    print(f"Found {len(videos)} video(s) in {VIDEO_DIR}\n")

    for i, video_path in enumerate(videos, 1):
        output_dir = OUTPUT_DIR or str(video_path.parent)
        print(f"[{i}/{len(videos)}] {video_path.name}")

        p = Pupil(
            video_path=str(video_path),
            output_dir=output_dir,
            input_size=INPUT_SIZE,
        )

        p.manual_roi()
        p.show_roi(scale=ROI_SCALE)  # press any key to continue, 'q' to quit

        if MODEL_PATH:
            p.model_path = Path(MODEL_PATH)

        p.save_config()
        print(f"  Config saved: {p.output_dir / (p.video_stem + '_config.json')}\n")
        p.close()

    print("Phase 1 complete. Run batch_detect.py for Phase 2.")


if __name__ == "__main__":
    main()
