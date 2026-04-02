"""Visualization: playback with overlays, time-series plots, video export."""

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .utils.masks import mask_to_features

logger = logging.getLogger(__name__)


def _draw_mask_contour(display: np.ndarray, mask: np.ndarray, color=(255, 255, 255)):
    """Draw raw binary mask contour on display image."""
    if mask is None or mask.sum() == 0:
        return
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(display, contours, -1, color, 1)


def _draw_ellipse_overlay(display: np.ndarray, mask: np.ndarray, color=(255, 255, 255)):
    """Draw fitted ellipse contour on display image (instead of raw mask contour)."""
    if mask is None or mask.sum() == 0:
        return
    features = mask_to_features(mask)
    if features is None or features["ellipse"] is None:
        # Fallback to raw contour if ellipse fit fails
        _draw_mask_contour(display, mask, color)
        return
    e = features["ellipse"]
    center = (int(round(e["center"][0])), int(round(e["center"][1])))
    axes = (int(round(e["axes"][0] / 2)), int(round(e["axes"][1] / 2)))
    angle = e["angle"]
    cv2.ellipse(display, center, axes, angle, 0, 360, color, 1)


def _draw_centroid(display: np.ndarray, results: np.ndarray, frame_idx: int, color=(0, 255, 0)):
    """Draw centroid circle from the N×8 results array."""
    if results is None:
        return
    if frame_idx < len(results) and not np.isnan(results[frame_idx, 2]):
        cx = int(round(results[frame_idx, 2]))
        cy = int(round(results[frame_idx, 3]))
        cv2.circle(display, (cx, cy), 2, color, -1)


def preview(
    video_path: str | Path,
    masks: dict[int, np.ndarray] | None = None,
    results: np.ndarray | None = None,
    roi: dict | None = None,
    input_size: int = 128,
    start_frame: int = 0,
    scale: float = 2.0,
):
    """Interactive playback with overlay.

    Overlay logic:
        - If results (N×8) are provided → fitted ellipse + centroid
        - Else if masks are provided → raw mask contour
        - If both → fitted ellipse + centroid (results take priority)

    Controls:
        Space: pause/resume
        Left/Right arrow: step frame
        Q: quit
    """
    from .video_io import VideoReader

    reader = VideoReader(video_path)
    current = start_frame
    paused = False
    use_fitted = results is not None

    win_w = int(input_size * scale)
    win_h = int(input_size * scale)
    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Preview", win_w, win_h)

    while current < reader.n_frames:
        frame = reader.read_frame(current)
        if frame is None:
            break

        # Apply ROI crop
        if roi is not None:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            frame = frame[y : y + h, x : x + w]

        # Resize to match detection resolution
        frame = cv2.resize(frame, (input_size, input_size))
        display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Overlay
        if use_fitted:
            # Fitted ellipse from mask + centroid from results
            if masks is not None and current in masks:
                _draw_ellipse_overlay(display, masks[current])
            _draw_centroid(display, results, current)
        else:
            # Raw mask contour only
            if masks is not None and current in masks:
                _draw_mask_contour(display, masks[current])

        # Frame number
        cv2.putText(display, f"F{current}", (2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        cv2.imshow("Preview", display)

        wait_time = 1 if paused else 30
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == 81 or key == 2:  # left arrow
            current = max(current - 1, 0)
            continue
        elif key == 83 or key == 3:  # right arrow
            current = min(current + 1, reader.n_frames - 1)
            continue

        # Check if window was closed via X button (after waitKey processes events)
        try:
            if cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        if not paused:
            current += 1

    cv2.destroyAllWindows()
    reader.close()


def plot_timeseries(results: np.ndarray, output_path: str | Path | None = None):
    """Plot area and centroid time-series with frame number on X axis.

    Args:
        results: (N, 8) array with columns [frame, time, cx, cy, area, ...]
        output_path: if provided, save figure to file instead of showing
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    frames = results[:, 0]

    # Area
    axes[0].plot(frames, results[:, 4], "b-", linewidth=0.5)
    axes[0].set_ylabel("Area (px)")
    axes[0].set_title("Pupil Tracking Results")

    # Center X
    axes[1].plot(frames, results[:, 2], "r-", linewidth=0.5)
    axes[1].set_ylabel("Center X (px)")

    # Center Y
    axes[2].plot(frames, results[:, 3], "g-", linewidth=0.5)
    axes[2].set_ylabel("Center Y (px)")
    axes[2].set_xlabel("Frame")

    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), dpi=150)
        logger.info(f"Time-series plot saved to {output_path}")
        plt.close()
    else:
        plt.show()


def export_video(
    video_path: str | Path,
    output_path: str | Path,
    results: np.ndarray,
    masks: dict[int, np.ndarray] | None = None,
    roi: dict | None = None,
    input_size: int = 128,
    fps: int = 30,
):
    """Export annotated video with fitted ellipse and centroid overlays."""
    from .video_io import VideoReader

    reader = VideoReader(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (input_size, input_size), isColor=True)

    for idx in range(reader.n_frames):
        frame = reader.read_frame(idx)
        if frame is None:
            continue

        if roi is not None:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            frame = frame[y : y + h, x : x + w]

        frame = cv2.resize(frame, (input_size, input_size))
        display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Fitted ellipse contour
        if masks is not None and idx in masks:
            _draw_ellipse_overlay(display, masks[idx])

        # Centroid
        _draw_centroid(display, results, idx)

        # Compact frame number
        cv2.putText(display, f"F{idx}", (2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        writer.write(display)

    writer.release()
    reader.close()
    logger.info(f"Annotated video exported to {output_path}")
