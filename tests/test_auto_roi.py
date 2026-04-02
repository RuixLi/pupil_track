"""Minimal script to test auto-ROI detection across multiple videos and methods.

Usage:
    cd pupil_track
    python test_auto_roi.py video1.avi video2.mp4 ...
    python test_auto_roi.py /path/to/videos/       # all .avi/.mp4 in a folder
    python test_auto_roi.py video.avi --methods hough center_dark
    python test_auto_roi.py video.avi --frame 50    # test on a specific frame
    python test_auto_roi.py video.avi --resize      # test resize mode

Press any key to advance to the next result, 'q' to quit.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow running from the pupil_track directory
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pupil_track.roi import auto_roi, AUTO_METHODS
from pupil_track.video_io import VideoReader


def draw_roi(frame: np.ndarray, roi: dict, label: str, color=(0, 255, 0)) -> np.ndarray:
    """Draw ROI rectangle + label on a BGR copy of the frame."""
    if frame.ndim == 2:
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        vis = frame.copy()
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
    # label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(vis, (x, y - th - 8), (x + tw + 4, y), color, -1)
    cv2.putText(vis, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return vis


def run_test(
    video_path: str,
    methods: list[str],
    frame_idx: int = 0,
    target_size: int = 128,
    resize: bool = False,
    show: bool = True,
):
    """Test all methods on a single video and optionally display results."""
    print(f"\n{'='*60}")
    print(f"Video: {video_path}")

    try:
        reader = VideoReader(video_path)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return

    frame = reader.read_frame(frame_idx)
    if frame is None:
        print(f"  ERROR: cannot read frame {frame_idx}")
        return

    print(f"  Size: {reader.width}x{reader.height}, {reader.n_frames} frames, {reader.fps:.1f} fps")
    print(f"  Frame: {frame_idx}, target_size: {target_size}, resize: {resize}")
    print(f"  {'Method':<15} {'ROI':<40} {'Time':>8}")
    print(f"  {'-'*15} {'-'*40} {'-'*8}")

    results = {}
    colors = {
        "dark_blob": (0, 255, 0),
        "hough": (255, 100, 0),
        "center_dark": (0, 200, 255),
    }

    for method in methods:
        t0 = time.perf_counter()
        try:
            roi = auto_roi(frame, target_size=target_size, method=method, resize=resize)
            dt = time.perf_counter() - t0
            roi_str = f"x={roi['x']}, y={roi['y']}, w={roi['w']}, h={roi['h']}"
            print(f"  {method:<15} {roi_str:<40} {dt*1000:>6.1f}ms")
            results[method] = roi
        except Exception as e:
            dt = time.perf_counter() - t0
            print(f"  {method:<15} {'FAILED: ' + str(e):<40} {dt*1000:>6.1f}ms")

    if not show or not results:
        return

    # --- Display: all methods overlaid on the same frame ---
    combined = frame.copy()
    if combined.ndim == 2:
        combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

    for method, roi in results.items():
        color = colors.get(method, (255, 255, 255))
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        cv2.rectangle(combined, (x, y), (x + w, y + h), color, 2)
        cv2.putText(combined, method, (x + 4, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Resize for display if too large
    disp_h, disp_w = combined.shape[:2]
    max_disp = 800
    if max(disp_h, disp_w) > max_disp:
        scale = max_disp / max(disp_h, disp_w)
        combined = cv2.resize(combined, None, fx=scale, fy=scale)

    win_name = f"Auto ROI - {Path(video_path).name}"
    cv2.imshow(win_name, combined)

    # Also show individual crops side by side
    crops = []
    for method, roi in results.items():
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        crop = frame[y:y+h, x:x+w]
        crop = cv2.resize(crop, (target_size, target_size))
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        # add method label
        cv2.putText(crop, method, (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(method, (255, 255, 255)), 1)
        crops.append(crop)

    if crops:
        strip = np.hstack(crops)
        cv2.imshow("Crops (side-by-side)", strip)

    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    if key == ord("q"):
        sys.exit(0)


## ── Settings ──────────────────────────────────────────────────────────────────
## Paste your video paths here (one per line). If non-empty, these are used
## instead of command-line arguments.

VIDEOS = [
    "D:\\OneDrive - 同志社大学\\Data\\pupil_video\\251107F03FM-20260224_152959.avi",
    "D:\\OneDrive - 同志社大学\\Data\\pupil_video\\251107F07FM-20260224_175922.avi",
    "D:\\OneDrive - 同志社大学\\Data\\pupil_video\\pupil_video250123_153514_WT5_5rpm_d1t2.mp4",
    "D:\\OneDrive - 同志社大学\\Data\\pupil_video\\526_002-20240618_103044_Camera01.mp4",
    "D:\\OneDrive - 同志社大学\\Data\\pupil_video\\r-2808_002-20250607_082654_Camera01.mp4",
    "D:\\OneDrive - 同志社大学\\Data\\pupil_video\\r-3301_001-20250912_121130_Camera01.mp4"
]

METHODS = list(AUTO_METHODS)       # ["dark_blob", "hough", "center_dark"]
FRAME = 0                          # frame index to test
TARGET_SIZE = 128                  # target ROI size
RESIZE = False                     # test resize mode
NO_SHOW = False                     # True = print only, no cv2 windows

## ──────────────────────────────────────────────────────────────────────────────


def main():
    # Use VIDEOS list if populated, otherwise fall back to CLI args
    if VIDEOS:
        class Args:
            inputs = VIDEOS
            methods = METHODS
            frame = FRAME
            target_size = TARGET_SIZE
            resize = RESIZE
            no_show = NO_SHOW
        args = Args()
    else:
        parser = argparse.ArgumentParser(
            description="Test auto-ROI detection across videos and methods",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__,
        )
        parser.add_argument(
            "inputs", nargs="+",
            help="Video file(s) or directory containing videos",
        )
        parser.add_argument(
            "--methods", nargs="+", default=list(AUTO_METHODS),
            choices=list(AUTO_METHODS),
            help=f"Methods to test (default: all). Options: {', '.join(AUTO_METHODS)}",
        )
        parser.add_argument("--frame", type=int, default=0, help="Frame index to test (default: 0)")
        parser.add_argument("--target-size", type=int, default=128, help="Target ROI size (default: 128)")
        parser.add_argument("--resize", action="store_true", help="Test in resize mode")
        parser.add_argument("--no-show", action="store_true", help="Print results only, no cv2 windows")
        args = parser.parse_args()

    # Collect video paths
    video_exts = {".avi", ".mp4", ".mkv", ".mov", ".wmv"}
    videos = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            for ext in video_exts:
                videos.extend(sorted(p.glob(f"*{ext}")))
        elif p.is_file() and p.suffix.lower() in video_exts:
            videos.append(p)
        else:
            print(f"Skipping: {inp}")

    if not videos:
        print("No video files found.")
        sys.exit(1)

    print(f"Testing {len(videos)} video(s) with methods: {', '.join(args.methods)}")

    for vpath in videos:
        run_test(
            str(vpath),
            methods=args.methods,
            frame_idx=args.frame,
            target_size=args.target_size,
            resize=args.resize,
            show=not args.no_show,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
