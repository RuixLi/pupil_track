"""Integrodifferential operator (Daugman's method) for pupil detection.

Translated from MATLAB: intdiff_thresh.m, intdiff_partiald.m,
intdiff_lineint.m, intdiff_search.m, intdiff_segment_pupil.m
"""

import logging

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core integrodifferential functions
# ---------------------------------------------------------------------------

def lineint(image: np.ndarray, center: tuple[int, int], r: int, n: int = 600) -> float:
    """Normalized line integral around a circular contour.

    Args:
        image: (H, W) float64 preprocessed image
        center: (row, col) circle center
        r: radius
        n: number of polygon sides

    Returns:
        Mean intensity along the circle, or 0 if out of bounds.
    """
    rows, cols = image.shape
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # row = center_row - r * sin(theta), col = center_col + r * cos(theta)
    y = center[0] - r * np.sin(theta)
    x = center[1] + r * np.cos(theta)

    # Bounds check
    yr = np.round(y).astype(int)
    xr = np.round(x).astype(int)
    if np.any(yr < 0) or np.any(yr >= rows) or np.any(xr < 0) or np.any(xr >= cols):
        return 0.0

    vals = image[yr, xr]
    return float(np.mean(vals))


def partiald(
    image: np.ndarray,
    center: tuple[int, int],
    rmin: int,
    rmax: int,
    sigma: float = 0.5,
    n: int = 600,
) -> tuple[float, int]:
    """Partial derivative of the normalized line integral.

    Evaluates the integrodifferential operator: smoothed finite differences
    of the line integral over a range of radii.

    Returns:
        (peak_magnitude, best_radius)
    """
    radii = np.arange(rmin, rmax + 1)
    L = np.zeros(len(radii))

    for i, r in enumerate(radii):
        val = lineint(image, center, r, n)
        if val == 0.0:
            L = L[:i]
            radii = radii[:i]
            break
        L[i] = val

    if len(L) < 2:
        return 0.0, rmin

    # Finite differences
    D = np.concatenate(([0.0], np.diff(L)))

    # Smoothing
    if sigma == np.inf:
        kernel = np.ones(7) / 7.0
        blur = np.abs(np.convolve(D, kernel, mode="same"))
    else:
        blur = np.abs(gaussian_filter1d(D, sigma))

    best_idx = int(np.argmax(blur))
    return float(blur[best_idx]), int(radii[best_idx])


def search(
    image: np.ndarray,
    rmin: int,
    rmax: int,
    x: int,
    y: int,
    sigma: float = 0.5,
    n: int = 600,
) -> tuple[int, int, int]:
    """Fine-grain search in 11x11 neighborhood.

    Args:
        image: preprocessed image
        rmin, rmax: radius bounds
        x, y: center point (row, col)

    Returns:
        (best_row, best_col, best_radius)
    """
    rows, cols = image.shape
    xi = np.arange(max(x - 5, 0), min(x + 6, rows))
    yi = np.arange(max(y - 5, 0), min(y + 6, cols))

    best_b = 0.0
    best_r = rmin
    best_xy = (x, y)

    for rx in xi:
        for ry in yi:
            b, r = partiald(image, (rx, ry), rmin, rmax, sigma, n)
            if b > best_b:
                best_b = b
                best_r = r
                best_xy = (rx, ry)

    return best_xy[0], best_xy[1], best_r


def segment_pupil(
    image: np.ndarray,
    cp: tuple[int, int, int],
    height: int,
    width: int,
) -> np.ndarray:
    """Segment pupil using detected circle as seed.

    Uses Otsu thresholding within 1.4x radius margin,
    morphological cleanup, and shape validation.

    Returns:
        (height, width) uint8 binary mask, or zeros if invalid.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    row, col, r = cp
    expected_area = np.pi * r * r

    # 1.4x radius search region
    margin = round(r * 1.4)
    r1 = max(round(row - margin), 0)
    r2 = min(round(row + margin), height)
    c1 = max(round(col - margin), 0)
    c2 = min(round(col + margin), width)
    roi = image[r1:r2, c1:c2]

    if roi.size == 0:
        return mask

    # Otsu threshold (on uint8 version)
    roi_u8 = (roi * 255).clip(0, 255).astype(np.uint8)
    _, bw = cv2.threshold(roi_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = (bw > 0).astype(np.uint8)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    # Largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1
    roi_area = areas[largest_idx - 1]
    bw = (labels == largest_idx).astype(np.uint8)

    # Validate: area ratio
    if roi_area > expected_area * 2.5 or roi_area < expected_area * 0.2:
        return mask

    # Validate: doesn't touch ROI boundary
    if np.any(bw[0, :]) or np.any(bw[-1, :]) or np.any(bw[:, 0]) or np.any(bw[:, -1]):
        return mask

    # Validate: circularity
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        perimeter = cv2.arcLength(contours[0], True)
        if perimeter > 0:
            circularity = 4 * np.pi * roi_area / (perimeter * perimeter)
            if circularity < 0.5:
                return mask

    # Place into full mask
    mask[r1:r2, c1:c2] = bw
    return mask


# ---------------------------------------------------------------------------
# Main detection: two-pass integrodifferential
# ---------------------------------------------------------------------------

def preprocess(image: np.ndarray, rmin: int) -> np.ndarray:
    """Preprocess image for integrodifferential detection.

    1. Fill holes in complement (remove specular reflections)
    2. Background subtraction via morphological opening
    3. Normalize to [0, 1]
    4. Gaussian smooth
    """
    img = image.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0

    # Invert, fill holes, invert back
    inv = (1.0 - img * 255).clip(0, 255).astype(np.uint8)
    # Use morphological closing as approximation of imfill on complement
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inv_filled = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel_fill)
    img = 1.0 - inv_filled.astype(np.float64) / 255.0

    # Background subtraction: morphological opening
    r = max(rmin, 15)
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    bg = cv2.morphologyEx((img * 255).astype(np.uint8), cv2.MORPH_OPEN, kernel_bg)
    img = img - bg.astype(np.float64) / 255.0

    # Normalize to [0, 1]
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()

    # Gaussian smooth
    img = cv2.GaussianBlur(img, (0, 0), 1.5)

    return img


def intdiff_thresh(
    image: np.ndarray,
    rmin: int = 10,
    rmax: int = 20,
    downscale: float = 0.5,
    prior: tuple[int, int, int] | None = None,
) -> tuple[tuple[int, int, int] | None, np.ndarray]:
    """Two-pass integrodifferential pupil detection.

    Args:
        image: grayscale image (uint8 or float64)
        rmin, rmax: expected pupil radius range
        downscale: factor for coarse pass
        prior: (row, col, radius) from previous frame, or None

    Returns:
        (cp, preprocessed_image) where cp = (row, col, radius) or None.
    """
    img = preprocess(image, rmin)
    rows_orig, cols_orig = img.shape

    # Prior-based mode: skip coarse pass
    if prior is not None:
        rmin_fine = max(round(prior[2] * 0.8), rmin)
        rmax_fine = min(round(prior[2] * 1.2), rmax)
        cx = max(min(prior[0], rows_orig - rmax_fine - 1), rmax_fine + 1)
        cy = max(min(prior[1], cols_orig - rmax_fine - 1), rmax_fine + 1)
        row, col, rad = search(img, rmin_fine, rmax_fine, cx, cy)
        return (row, col, rad), img

    # PASS 1: Coarse search on downsampled image
    img_ds = cv2.resize(img, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_LINEAR)
    rmin_ds = max(round(rmin * downscale), 2)
    rmax_ds = max(round(rmax * downscale), rmin_ds + 1)
    rows_ds, cols_ds = img_ds.shape

    # Find darkest region centroid
    thresh = np.quantile(img_ds, 0.2)
    dark_mask = (img_ds <= thresh).astype(np.uint8)

    # Remove border
    margin = rmin_ds
    border = np.zeros_like(dark_mask)
    if 2 * margin < rows_ds and 2 * margin < cols_ds:
        border[margin : rows_ds - margin, margin : cols_ds - margin] = 1
    dark_mask = dark_mask & border

    if not np.any(dark_mask):
        return None, img

    # Largest dark component centroid
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    if num_labels <= 1:
        return None, img

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1
    center_col, center_row = centroids[largest_idx]
    center_row, center_col = int(round(center_row)), int(round(center_col))

    # Evaluate operator in neighborhood
    best_b = 0.0
    best_r = rmin_ds
    best_xy = (center_row, center_col)
    search_range = 3

    for di in range(-search_range, search_range + 1):
        for dj in range(-search_range, search_range + 1):
            ci = center_row + di
            cj = center_col + dj
            if ci <= rmax_ds or cj <= rmax_ds or ci >= rows_ds - rmax_ds or cj >= cols_ds - rmax_ds:
                continue
            b, r = partiald(img_ds, (ci, cj), rmin_ds, rmax_ds, sigma=np.inf, n=600)
            if b > best_b:
                best_b = b
                best_r = r
                best_xy = (ci, cj)

    if best_b == 0:
        return None, img

    # Upscale to original resolution
    coarse_row = round(best_xy[0] / downscale)
    coarse_col = round(best_xy[1] / downscale)
    coarse_r = round(best_r / downscale)

    # PASS 2: Fine search
    rmin_fine = max(round(coarse_r * 0.7), rmin)
    rmax_fine = min(round(coarse_r * 1.3), rmax)
    coarse_row = max(min(coarse_row, rows_orig - rmax_fine - 1), rmax_fine + 1)
    coarse_col = max(min(coarse_col, cols_orig - rmax_fine - 1), rmax_fine + 1)

    row, col, rad = search(img, rmin_fine, rmax_fine, coarse_row, coarse_col)
    return (row, col, rad), img


# ---------------------------------------------------------------------------
# Frame-level detection dispatcher
# ---------------------------------------------------------------------------

def detect_intdiff(
    frames: list[tuple[int, np.ndarray]],
    rmin: int = 10,
    rmax: int = 20,
    downscale: float = 0.5,
    sequential: bool = True,
) -> dict[int, np.ndarray]:
    """Run integrodifferential detection on a list of (index, frame) pairs.

    Args:
        frames: list of (frame_index, grayscale_frame) tuples
        rmin, rmax: radius bounds
        downscale: coarse pass downscale factor
        sequential: if True, use previous frame result as prior

    Returns:
        dict mapping frame_index -> binary mask (uint8).
    """
    results = {}
    prior = None

    for idx, frame in frames:
        cp, img_proc = intdiff_thresh(frame, rmin, rmax, downscale, prior)

        if cp is not None:
            h, w = frame.shape[:2]
            mask = segment_pupil(img_proc, cp, h, w)
            results[idx] = mask
            if sequential:
                prior = cp
        else:
            h, w = frame.shape[:2]
            results[idx] = np.zeros((h, w), dtype=np.uint8)

    return results
