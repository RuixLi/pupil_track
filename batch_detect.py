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

import logging
from pathlib import Path

from pupil_track import Pupil

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  USER PARAMETERS — edit these before running                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Directory containing eye video files (.avi, .mp4)
VIDEO_DIR = r"G:\pupil_demo\movie"

# Directory where configs and results are stored.
# If empty, defaults to the same folder as each video.
OUTPUT_DIR = r""

# Input size — must match Phase 1 and training.
INPUT_SIZE = 128

# Detection threshold (probability above which a pixel is "pupil")
THRESHOLD = 0.5

# Post-processing: smoothing window size (in frames)
SMOOTH_WINDOW = 10

# ════════════════════════════════════════════════════════════════════════

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov"}


def collect_videos(directory: str | Path) -> list[Path]:
    """Return sorted list of video files in directory."""
    d = Path(directory)
    return [f for f in sorted(d.iterdir()) if f.suffix.lower() in VIDEO_EXTENSIONS]


def main():
    videos = collect_videos(VIDEO_DIR)
    if not videos:
        print(f"No video files found in {VIDEO_DIR}")
        return

    print(f"Found {len(videos)} video(s) in {VIDEO_DIR}\n")

    skipped = []
    for i, video_path in enumerate(videos, 1):
        output_dir = OUTPUT_DIR or str(video_path.parent)
        print(f"[{i}/{len(videos)}] {video_path.name}")

        p = Pupil(
            video_path=str(video_path),
            output_dir=output_dir,
            input_size=INPUT_SIZE,
        )

        # Load config from Phase 1
        p.load_config()
        if p.roi is None:
            print(f"  SKIPPED — no config found. Run batch_roi.py first.\n")
            skipped.append(video_path.name)
            p.close()
            continue

        p.config.threshold = THRESHOLD
        p.config.smooth_window = SMOOTH_WINDOW

        # Detect
        all_indices = p.get_all_indices()
        p.detect("unet", frame_indices=all_indices, threshold=THRESHOLD)

        # Post-process and save
        p.postprocess("unet")
        p.save_results("unet")

        print(f"  Done — results saved to {output_dir}\n")
        p.close()

    # Summary
    n_done = len(videos) - len(skipped)
    print(f"Complete: {n_done}/{len(videos)} videos processed.")
    if skipped:
        print(f"Skipped ({len(skipped)}): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
