"""ROI selection: manual and automatic modes.

Auto-ROI methods (each embodies a distinct search strategy):
    dark_blob   — The pupil is the darkest, roundest region in the frame.
    hough       — The pupil has a clear circular edge.
    center_dark — The pupil is a dark spot near the center of the FOV.
"""

import logging

import cv2
import numpy as np

from .config import Config

logger = logging.getLogger(__name__)

AUTO_METHODS = ("dark_blob", "hough", "center_dark")


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: build ROI dict from centroid
# ═══════════════════════════════════════════════════════════════════════════════

def _roi_from_centroid(
    cx: int,
    cy: int,
    frame_w: int,
    frame_h: int,
    target_size: int,
    resize: bool,
    resize_side: int | None = None,
) -> dict:
    """Build a square ROI dict centred on (cx, cy).

    resize=False → ROI is exactly target_size.
    resize=True  → ROI is resize_side (or target_size if None), clamped to frame.
    """
    side = resize_side if (resize and resize_side) else target_size
    side = min(side, frame_w, frame_h)
    roi_x = max(cx - side // 2, 0)
    roi_y = max(cy - side // 2, 0)
    roi_x = min(roi_x, frame_w - side)
    roi_y = min(roi_y, frame_h - side)
    return {"x": roi_x, "y": roi_y, "w": side, "h": side}


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 1 — dark_blob
# Logic: the pupil is the DARKEST and ROUNDEST region in the frame.
#
# Steps:
#   1. Blur to suppress noise.
#   2. Adaptive threshold to binarise dark regions (works across lighting).
#   3. Morphological cleanup to remove small noise and merge fragments.
#   4. Score each connected component by  darkness × circularity.
#   5. Pick the highest-scoring component; its centroid = pupil centre.
# ═══════════════════════════════════════════════════════════════════════════════

def _auto_dark_blob(
    frame: np.ndarray,
    target_size: int = 128,
    resize: bool = False,
    resize_margin: float = 1.5,
    **kwargs,
) -> dict:
    """Find the darkest, roundest blob — assumed to be the pupil."""
    h, w = frame.shape[:2]

    # 1. Blur
    blurred = cv2.GaussianBlur(frame, (7, 7), 2.0)

    # 2. Adaptive threshold (catches dark regions regardless of global brightness)
    bw = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=51, C=10,
    )

    # 3. Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)   # remove small noise
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)  # fill small holes

    # 4. Score components by darkness × circularity
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bw, connectivity=8,
    )
    if num_labels <= 1:
        logger.warning("dark_blob: no components found — falling back to frame centre.")
        return _roi_from_centroid(w // 2, h // 2, w, h, target_size, resize)

    best_idx, best_score = -1, -1.0
    total_pixels = h * w

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Reject too-small (noise) and too-large (background) blobs
        if area < 30 or area > total_pixels * 0.25:
            continue

        # Circularity from contour perimeter
        comp_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(
            comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        perim = cv2.arcLength(contours[0], True)
        circularity = (4 * np.pi * area) / (perim * perim) if perim > 0 else 0

        # Mean darkness of the region in the original frame (lower = darker)
        mean_val = cv2.mean(frame, mask=comp_mask)[0]
        # Normalise: 0 (white) → 0 score, 255 (black) → 1 score
        darkness = 1.0 - mean_val / 255.0

        score = darkness * circularity
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx < 0:
        logger.warning("dark_blob: no valid component — falling back to frame centre.")
        return _roi_from_centroid(w // 2, h // 2, w, h, target_size, resize)

    # 5. Centroid → ROI
    cx = int(round(centroids[best_idx][0]))
    cy = int(round(centroids[best_idx][1]))
    logger.info("dark_blob: best component at (%d, %d), score=%.3f", cx, cy, best_score)

    resize_side = None
    if resize:
        lw = stats[best_idx, cv2.CC_STAT_WIDTH]
        lh = stats[best_idx, cv2.CC_STAT_HEIGHT]
        resize_side = max(int(max(lw, lh) * resize_margin), target_size)

    return _roi_from_centroid(cx, cy, w, h, target_size, resize, resize_side)


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 2 — hough
# Logic: the pupil is defined by a CIRCULAR EDGE.
#
# Steps:
#   1. Blur to suppress noise.
#   2. Run Canny edge detection.
#   3. Run Hough Circle Transform on the edge map.
#   4. Among detected circles, pick the one with the darkest interior
#      (disambiguates pupil from iris or reflections).
#   5. Circle centre = pupil centre; diameter informs ROI size.
# ═══════════════════════════════════════════════════════════════════════════════

def _auto_hough(
    frame: np.ndarray,
    target_size: int = 128,
    resize: bool = False,
    resize_margin: float = 1.5,
    min_radius: int = 8,
    max_radius: int = 80,
    **kwargs,
) -> dict:
    """Detect a circular edge — assumed to be the pupil boundary."""
    h, w = frame.shape[:2]

    # 1. Blur
    blurred = cv2.GaussianBlur(frame, (7, 7), 2.0)

    # 2–3. Hough Circle Transform (uses internal Canny)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=max(h, w) // 6,
        param1=80,   # upper Canny threshold
        param2=30,   # accumulator threshold (lower = more sensitive)
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        logger.warning("hough: no circles detected — falling back to frame centre.")
        return _roi_from_centroid(w // 2, h // 2, w, h, target_size, resize)

    circles = np.round(circles[0]).astype(int)  # shape (N, 3): x, y, r
    logger.info("hough: found %d candidate circle(s)", len(circles))

    # 4. Pick the circle with the darkest interior
    best_idx, best_darkness = 0, -1.0
    for i, (cx, cy, r) in enumerate(circles):
        # Create a circular mask for this candidate
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        mean_val = cv2.mean(frame, mask=mask)[0]
        darkness = 1.0 - mean_val / 255.0
        if darkness > best_darkness:
            best_darkness = darkness
            best_idx = i

    cx, cy, r = int(circles[best_idx][0]), int(circles[best_idx][1]), int(circles[best_idx][2])
    logger.info("hough: best circle at (%d, %d) r=%d, darkness=%.3f", cx, cy, r, best_darkness)

    # 5. Circle → ROI
    resize_side = None
    if resize:
        resize_side = max(int(r * 2 * resize_margin), target_size)

    return _roi_from_centroid(cx, cy, w, h, target_size, resize, resize_side)


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 3 — center_dark
# Logic: the pupil is a DARK SPOT NEAR THE CENTRE of the field of view.
#
# Steps:
#   1. Apply a Gaussian spatial prior centred on the frame (down-weight edges).
#   2. Invert the frame so dark pixels become high values.
#   3. Multiply inverted frame × spatial prior → combined map.
#   4. Blur the combined map to get a smooth peak.
#   5. Peak of the combined map = pupil centre.
# ═══════════════════════════════════════════════════════════════════════════════

def _auto_center_dark(
    frame: np.ndarray,
    target_size: int = 128,
    resize: bool = False,
    resize_margin: float = 1.5,
    center_sigma_frac: float = 0.3,
    **kwargs,
) -> dict:
    """Find the darkest spot near the frame centre — assumed to be the pupil."""
    h, w = frame.shape[:2]

    # 1. Gaussian spatial prior (peaks at centre, decays toward edges)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    sigma_y = h * center_sigma_frac
    sigma_x = w * center_sigma_frac
    spatial_prior = np.exp(
        -((xx - w / 2) ** 2) / (2 * sigma_x ** 2)
        - ((yy - h / 2) ** 2) / (2 * sigma_y ** 2)
    )

    # 2. Invert so dark → high
    inverted = 255.0 - frame.astype(np.float64)

    # 3. Combined map
    combined = inverted * spatial_prior

    # 4. Smooth to get a clean peak
    combined = cv2.GaussianBlur(combined, (0, 0), sigmaX=15, sigmaY=15)

    # 5. Peak location
    _, _, _, max_loc = cv2.minMaxLoc(combined)
    cx, cy = int(max_loc[0]), int(max_loc[1])
    logger.info("center_dark: peak at (%d, %d)", cx, cy)

    # Estimate ROI size from the dark region around the peak
    resize_side = None
    if resize:
        # Threshold the region around the peak to estimate pupil extent
        patch_r = min(max_radius := 80, h // 2, w // 2)
        x1 = max(cx - patch_r, 0)
        y1 = max(cy - patch_r, 0)
        x2 = min(cx + patch_r, w)
        y2 = min(cy + patch_r, h)
        patch = frame[y1:y2, x1:x2]
        _, bw = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dark_pixels = cv2.countNonZero(bw)
        equiv_diameter = int(np.sqrt(4 * dark_pixels / np.pi))
        resize_side = max(int(equiv_diameter * resize_margin), target_size)

    return _roi_from_centroid(cx, cy, w, h, target_size, resize, resize_side)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

_AUTO_DISPATCH = {
    "dark_blob": _auto_dark_blob,
    "hough": _auto_hough,
    "center_dark": _auto_center_dark,
}


def auto_roi(
    frame: np.ndarray,
    target_size: int = 128,
    method: str = "dark_blob",
    resize: bool = False,
    **kwargs,
) -> dict:
    """Automatically detect ROI around the pupil.

    Args:
        frame: full-resolution grayscale frame (uint8)
        target_size: side length of the square ROI
        method: 'dark_blob', 'hough', or 'center_dark'
        resize: if True, allow ROI larger than target_size (resized later)
        **kwargs: forwarded to the chosen method

    Returns:
        dict {"x": int, "y": int, "w": int, "h": int}
    """
    if method not in _AUTO_DISPATCH:
        raise ValueError(f"Unknown auto-ROI method: {method!r}. Choose from {AUTO_METHODS}")

    roi = _AUTO_DISPATCH[method](frame, target_size=target_size, resize=resize, **kwargs)
    logger.info("Auto ROI [%s]: %s", method, roi)
    return roi


def manual_roi(frame: np.ndarray, target_size: int = 128, resize: bool = False, scale: float = 0.5) -> dict:
    """Manually select ROI via OpenCV GUI.

    When resize=False (default), the selection is snapped to a target_size square.
    When resize=True, the drawn rectangle is rounded to a square (max side),
    keeping its center — the crop will later be resized to target_size.
    """
    window_name = "Select ROI (draw rectangle, press Enter/Space to confirm)"
    fh, fw = frame.shape[:2]
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(fw * scale), int(fh * scale))
    rect = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    x, y, w, h = rect
    fh, fw = frame.shape[:2]

    if w == 0 or h == 0:
        logger.warning("No ROI selected. Using center crop.")
        x = max(fw // 2 - target_size // 2, 0)
        y = max(fh // 2 - target_size // 2, 0)
        roi = {"x": x, "y": y, "w": target_size, "h": target_size}
    elif resize:
        side = max(w, h)
        cx, cy = x + w // 2, y + h // 2
        rx = max(cx - side // 2, 0)
        ry = max(cy - side // 2, 0)
        side = min(side, fw, fh)
        rx = min(rx, fw - side)
        ry = min(ry, fh - side)
        roi = {"x": rx, "y": ry, "w": side, "h": side}
    else:
        cx, cy = x + w // 2, y + h // 2
        x = max(cx - target_size // 2, 0)
        y = max(cy - target_size // 2, 0)
        x = min(x, fw - target_size)
        y = min(y, fh - target_size)
        roi = {"x": x, "y": y, "w": target_size, "h": target_size}

    logger.info(f"Manual ROI selected: {roi}")
    return roi


def select_roi(
    frame: np.ndarray,
    mode: str = "auto",
    target_size: int = 128,
    method: str = "dark_blob",
    **kwargs,
) -> dict:
    """Select ROI using specified mode.

    Args:
        frame: grayscale frame
        mode: "auto" or "manual"
        target_size: side length of square ROI
        method: auto-ROI method (only used when mode="auto")

    Returns:
        dict {"x": int, "y": int, "w": int, "h": int}
    """
    if mode == "auto":
        return auto_roi(frame, target_size, method=method, **kwargs)
    elif mode == "manual":
        return manual_roi(frame, target_size)
    else:
        raise ValueError(f"Unknown ROI mode: {mode}")
