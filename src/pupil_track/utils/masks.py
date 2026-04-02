"""Mask utilities: creation, conversion, feature extraction."""

import cv2
import numpy as np


def circle_to_mask(height: int, width: int, cx: float, cy: float, r: float) -> np.ndarray:
    """Create a filled circle binary mask.

    Args:
        height, width: output mask dimensions
        cx, cy: circle center (x=col, y=row)
        r: radius in pixels

    Returns:
        (height, width) uint8 mask with 1 inside, 0 outside.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r)), 1, -1)
    return mask


def ellipse_to_mask(
    height: int, width: int, a: float, b: float, cx: float, cy: float, theta: float
) -> np.ndarray:
    """Create a filled ellipse binary mask.

    Args:
        height, width: output mask dimensions
        a, b: semi-major and semi-minor axes
        cx, cy: ellipse center (x=col, y=row)
        theta: rotation angle in radians

    Returns:
        (height, width) uint8 mask with 1 inside, 0 outside.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    angle_deg = np.degrees(theta)
    cv2.ellipse(
        mask,
        center=(int(round(cx)), int(round(cy))),
        axes=(int(round(a)), int(round(b))),
        angle=angle_deg,
        startAngle=0,
        endAngle=360,
        color=1,
        thickness=-1,
    )
    return mask


def mask_to_features(mask: np.ndarray):
    """Extract pupil features from a binary mask.

    Args:
        mask: (H, W) binary uint8 mask

    Returns:
        dict with keys: centroid_x, centroid_y, area, ellipse (optional).
        Returns None if mask is empty.
    """
    if mask.sum() == 0:
        return None

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    if num_labels <= 1:
        return None

    # Component 0 is background; find largest foreground
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1
    component_mask = (labels == largest_idx).astype(np.uint8)
    area = int(areas[largest_idx - 1])

    # Centroid via moments
    M = cv2.moments(component_mask)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    result = {"centroid_x": cx, "centroid_y": cy, "area": area, "ellipse": None}

    # Ellipse fit (needs >= 5 contour points)
    contours, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if contours and len(contours[0]) >= 5:
        ellipse = cv2.fitEllipse(contours[0])
        # ellipse = ((cx, cy), (major, minor), angle)
        result["ellipse"] = {
            "center": ellipse[0],
            "axes": ellipse[1],
            "angle": ellipse[2],
        }

    return result
