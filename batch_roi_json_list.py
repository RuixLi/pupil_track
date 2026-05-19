"""Phase 1 — Define and confirm ROI for all videos listed in JSON file.

For each entry in the JSON file (can be directory paths or direct video file paths):
  If directory: process all videos found in that directory
  If video file: process that specific video file
    1. Auto-detect ROI
    2. Show the ROI overlay — press any key to accept, or 'q' to quit
    3. Save config JSON (with ROI + model path) to OUTPUT_DIR

After this script, every video has a {stem}_config.json ready for Phase 2
(batch_detect_json_list.py).

JSON file format (list of directory paths or video file paths):
[
  "G:\\pupil_demo\\movie1",
  "G:\\pupil_demo\\movie2\\video1.mp4",
  "G:\\pupil_demo\\movie3"
]

Usage:
    python batch_roi_json_list.py
"""

import json
import logging
from pathlib import Path

from pupil_track import Pupil

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  USER PARAMETERS — edit these before running                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

# JSON file containing list of video directories or direct video file paths
JSON_FILE = r"V:\li\RTWF-20251212-DATASTORE\_pupil_video_list.json"

# Directory where config JSONs are saved.
# If empty, defaults to the same folder as each video.
OUTPUT_DIR = r"V:\li\RTWF-20251212-ANALYSIS\pupil_new"

# Path to a trained U-Net model (.pth file).
# Saved into each config so Phase 2 knows which model to use.
MODEL_PATH = r"C:\Users\bdd2\OneDrive - 同志社大学\Data\pupil_dataset\dataset_260419\checkpoint3\best_model.pth"

# Input size — must match the size used during training.
INPUT_SIZE = 128

# ROI display scale (fraction of original frame size)
ROI_SCALE = 0.5

# ════════════════════════════════════════════════════════════════════════

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov"}


def load_video_paths(json_file: str | Path) -> list[Path]:
    """Load list of video directories or direct video file paths from JSON file."""
    json_path = Path(json_file)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        paths = json.load(f)
    
    if not isinstance(paths, list):
        raise ValueError("JSON file must contain a list of paths")
    
    return [Path(p) for p in paths]


def collect_videos_from_paths(paths: list[Path]) -> list[tuple[Path, Path]]:
    """Return list of (video_path, source_path) tuples from directories or direct video files."""
    all_videos = []
    
    for path in paths:
        if not path.exists():
            print(f"Warning: Path not found: {path}")
            continue
            
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            # Direct video file path
            all_videos.append((path, path.parent))
        elif path.is_dir():
            # Directory path - find all videos in it
            videos = [f for f in sorted(path.iterdir()) if f.suffix.lower() in VIDEO_EXTENSIONS]
            all_videos.extend([(video, path) for video in videos])
        else:
            print(f"Warning: Path is neither a video file nor a directory: {path}")
    
    return all_videos


def main():
    try:
        video_paths = load_video_paths(JSON_FILE)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading JSON file: {e}")
        return

    print(f"Loaded {len(video_paths)} path(s) from {JSON_FILE}")
    
    all_videos = collect_videos_from_paths(video_paths)
    if not all_videos:
        print("No video files found in any of the specified paths")
        return

    print(f"Found {len(all_videos)} video(s) from all specified paths\n")

    for i, (video_path, source_dir) in enumerate(all_videos, 1):
        output_dir = OUTPUT_DIR or str(video_path.parent)
        print(f"[{i}/{len(all_videos)}] {video_path.name} (from {source_dir.name})")

        p = Pupil(
            video_path=str(video_path),
            output_dir=output_dir,
            input_size=INPUT_SIZE,
        )

        p.manual_roi(resize=True)
        p.show_roi(scale=ROI_SCALE)  # press any key to continue, 'q' to quit

        if MODEL_PATH:
            p.model_path = Path(MODEL_PATH)

        p.save_config()
        print(f"  Config saved: {p.output_dir / (p.video_stem + '_config.json')}\n")
        p.close()

    print("Phase 1 complete. Run batch_detect_json_list.py for Phase 2.")


if __name__ == "__main__":
    main()