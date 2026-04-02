"""Post-processing: feature extraction, smoothing, export.

Translated from MATLAB: sort_data.m, smooth.m, mask_to_contour.m
"""

import csv
import logging
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import medfilt

from .utils.masks import mask_to_features

logger = logging.getLogger(__name__)


def extract_all_features(
    masks: dict[int, np.ndarray], fps: float = 30.0
) -> np.ndarray:
    """Extract pupil features from all detection masks.

    Args:
        masks: dict mapping frame_index -> binary mask
        fps: video frame rate for time computation

    Returns:
        Structured array with columns:
        [frame, time, center_x, center_y, area, major_axis, minor_axis, angle]
        NaN for frames where detection failed.
    """
    if not masks:
        return np.empty((0, 8))

    max_frame = max(masks.keys())
    n_frames = max_frame + 1
    data = np.full((n_frames, 8), np.nan)

    for idx in range(n_frames):
        data[idx, 0] = idx
        data[idx, 1] = idx / fps

        if idx not in masks:
            continue

        features = mask_to_features(masks[idx])
        if features is None:
            continue

        data[idx, 2] = features["centroid_x"]
        data[idx, 3] = features["centroid_y"]
        data[idx, 4] = features["area"]

        if features["ellipse"] is not None:
            axes = features["ellipse"]["axes"]
            data[idx, 5] = max(axes)  # major axis
            data[idx, 6] = min(axes)  # minor axis
            data[idx, 7] = features["ellipse"]["angle"]

    return data


def fill_empty_frames(data: np.ndarray) -> np.ndarray:
    """Fill NaN frames by nearest-neighbor propagation.

    For each NaN row, search backward first, then forward.
    Columns 2+ are filled (skip frame and time).
    """
    result = data.copy()
    n = len(result)
    fill_cols = slice(2, None)

    nan_mask = np.isnan(result[:, 2])
    nan_indices = np.where(nan_mask)[0]

    for j in nan_indices:
        # Search backward
        m = 1
        found = False
        while j - m >= 0:
            if not np.isnan(result[j - m, 2]):
                result[j, fill_cols] = result[j - m, fill_cols]
                found = True
                break
            m += 1
        if not found:
            # Search forward
            m = 1
            while j + m < n:
                if not np.isnan(result[j + m, 2]):
                    result[j, fill_cols] = result[j + m, fill_cols]
                    break
                m += 1

    return result


def temporal_smooth(signal: np.ndarray, window: int = 10) -> np.ndarray:
    """Temporal smoothing with outlier removal.

    Translated from MATLAB smooth.m:
    1. Sliding window median
    2. Replace outliers (> std/3 from median) with median
    3. Final median filter (kernel=3) to remove jitter
    """
    x = signal.copy()
    valid = ~np.isnan(x)

    if valid.sum() < 3:
        return x

    # Sliding window median
    win = int(2 * np.ceil(window / 2))
    med = median_filter(x, size=win, mode="nearest")

    # Interpolate NaN positions
    nan_idx = np.where(np.isnan(x))[0]
    valid_idx = np.where(~np.isnan(x))[0]
    if len(nan_idx) > 0 and len(valid_idx) > 1:
        x[nan_idx] = np.interp(nan_idx, valid_idx, med[valid_idx])
        med[nan_idx] = x[nan_idx]

    # Replace outliers
    threshold = np.nanstd(x) / 3.0
    outliers = np.abs(x - med) > threshold
    x[outliers] = med[outliers]

    # Remove jitter with median filter
    x = medfilt(x, kernel_size=3)

    return x


def smooth_results(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Apply temporal smoothing to all signal columns.

    Args:
        data: (N, 8) array from extract_all_features
        window: smoothing window size

    Returns:
        (N, 8) array with smoothed center_x, center_y, area, axes.
    """
    result = data.copy()
    # Smooth columns: center_x(2), center_y(3), area(4), major(5), minor(6)
    for col in [2, 3, 4, 5, 6]:
        result[:, col] = temporal_smooth(result[:, col], window)
    return result


def export_csv(data: np.ndarray, path: str | Path):
    """Export results to CSV file (no time column — frame-based analysis)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["frame", "center_x", "center_y", "area", "major_axis", "minor_axis", "angle"]
    # Column indices to export (skip index 1 = time)
    cols = [0, 2, 3, 4, 5, 6, 7]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data:
            writer.writerow([f"{v:.4f}" if not np.isnan(v) else "" for v in row[cols]])
    logger.info(f"Results exported to {path}")


def export_npy(data: np.ndarray, path: str | Path):
    """Export results to .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), data)
    logger.info(f"Results exported to {path}")


def postprocess(
    masks: dict[int, np.ndarray],
    fps: float = 30.0,
    smooth_window: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Full post-processing pipeline.

    Returns:
        (raw_data, smoothed_data) — both (N, 8) arrays.
    """
    raw = extract_all_features(masks, fps)
    filled = fill_empty_frames(raw)
    smoothed = smooth_results(filled, smooth_window)
    return raw, smoothed
