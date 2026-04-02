"""Detection dispatcher: routes to U-Net, intdiff, or starburst."""

import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import Config
from .video_io import VideoReader

logger = logging.getLogger(__name__)


def crop_and_resize(
    frame: np.ndarray,
    roi: dict | None,
    target_size: int,
) -> np.ndarray:
    """Apply ROI crop and resize to target_size x target_size."""
    if roi is not None:
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        frame = frame[y : y + h, x : x + w]
    return cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def detect(
    video_path: str | Path,
    config: Config,
    frame_indices: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Run pupil detection on a video.

    Args:
        video_path: path to video file
        config: pipeline configuration
        frame_indices: optional subset of frame indices to process

    Returns:
        dict mapping frame_index -> binary mask (uint8, input_size x input_size).
    """
    reader = VideoReader(video_path)

    if frame_indices is None:
        frame_indices = np.arange(reader.n_frames)

    # Collect frames with ROI crop + resize
    logger.info(f"Loading {len(frame_indices)} frames from {video_path}")
    frames = []
    for idx, frame in tqdm(
        reader.frames(frame_indices), total=len(frame_indices), desc="Loading", unit="frame"
    ):
        cropped = crop_and_resize(frame, config.roi, config.input_size)
        frames.append((idx, cropped))
    reader.close()

    logger.info(f"Detecting with method={config.method}")

    if config.method == "unet":
        from .detectors.unet_detect import detect_unet
        return detect_unet(
            frames,
            model_path=config.model_path,
            input_size=config.input_size,
            threshold=config.threshold,
        )
    elif config.method == "intdiff":
        from .detectors.intdiff import detect_intdiff
        return detect_intdiff(
            frames,
            rmin=config.intdiff_rmin,
            rmax=config.intdiff_rmax,
            downscale=config.intdiff_downscale,
        )
    elif config.method == "starburst":
        from .detectors.starburst import detect_starburst
        return detect_starburst(
            frames,
            edge_thresh=config.starburst_edge_thresh,
            n_rays=config.starburst_rays,
            sigma=config.starburst_sigma,
            min_features=config.starburst_min_features,
            max_ransac=config.starburst_max_ransac,
            remove_cr=config.starburst_remove_cr,
        )
    else:
        raise ValueError(f"Unknown detection method: {config.method}")
