"""Starburst pupil detection: ray-casting + RANSAC ellipse fitting.

Translated from MATLAB openEyes ToolKit (Li & Parkhurst, 2005):
starburst_process_frame.m, starburst_detect.m,
starburst_fit_ellipse_ransac.m, starburst_fit_ellipse_model.m,
starburst_locate_cr.m, starburst_fit_cr_radius.m, starburst_remove_cr.m,
starburst_convert_conic.m, starburst_normalize_coords.m, starburst_denormalize.m
"""

import logging

import cv2
import numpy as np
from scipy.optimize import minimize

from ..utils.masks import ellipse_to_mask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate normalization (for numerical stability in SVD)
# ---------------------------------------------------------------------------

def normalize_coords(x: np.ndarray, y: np.ndarray):
    """Normalize point coordinates for numerical stability.

    Returns:
        (nx, ny, H) where H is the 3x3 normalization matrix.
    """
    cx, cy = x.mean(), y.mean()
    mean_dist = np.mean(np.sqrt(x**2 + y**2))
    if mean_dist < 1e-10:
        mean_dist = 1.0
    dist_scale = np.sqrt(2) / mean_dist
    H = np.array([
        [dist_scale, 0, -dist_scale * cx],
        [0, dist_scale, -dist_scale * cy],
        [0, 0, 1],
    ])
    nx = dist_scale * x - dist_scale * cx
    ny = dist_scale * y - dist_scale * cy
    return nx, ny, H


def denormalize(ne: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Denormalize ellipse parameters [a, b, cx, cy, theta]."""
    return np.array([
        ne[0] / H[0, 0],
        ne[1] / H[1, 1],
        (ne[2] - H[0, 2]) / H[0, 0],
        (ne[3] - H[1, 2]) / H[1, 1],
        ne[4],
    ])


def convert_conic(c: np.ndarray) -> np.ndarray:
    """Convert 6-vector conic parameters to ellipse [a, b, cx, cy, theta].

    Returns [0,0,0,0,0] if not a valid real ellipse.
    """
    A, B, C, D, E, F = c
    theta = np.arctan2(B, A - C) / 2.0

    ct, st = np.cos(theta), np.sin(theta)
    ap = A * ct**2 + B * ct * st + C * st**2
    cp = A * st**2 - B * ct * st + C * ct**2

    T = np.array([[A, B / 2], [B / 2, C]])
    try:
        t = -np.linalg.solve(2 * T, np.array([D, E]))
    except np.linalg.LinAlgError:
        return np.zeros(5)

    scale_inv = t @ T @ t - F

    if scale_inv <= 0 or ap <= 0 or cp <= 0:
        return np.zeros(5)

    a = np.sqrt(scale_inv / ap)
    b = np.sqrt(scale_inv / cp)

    if not (np.isreal(a) and np.isreal(b)):
        return np.zeros(5)

    return np.array([float(a), float(b), float(t[0]), float(t[1]), float(theta)])


# ---------------------------------------------------------------------------
# Corneal reflection detection and removal
# ---------------------------------------------------------------------------

def locate_cr(
    image: np.ndarray, cx: int, cy: int, window_width: int = 301
) -> tuple[float, float, float]:
    """Locate corneal reflection using adaptive thresholding.

    Returns (x, y, radius) or (0, 0, 0) on failure.
    """
    height, width = image.shape
    r = (window_width - 1) // 2
    sx = max(round(cx - r), 0)
    ex = min(round(cx + r), width)
    sy = max(round(cy - r), 0)
    ey = min(round(cy + r), height)
    Iw = image[sy:ey, sx:ex]

    if Iw.size == 0:
        return 0.0, 0.0, 0.0

    prev_score = 0.0
    result = (0.0, 0.0, 0.0)

    for threshold in range(int(Iw.max()), 0, -1):
        bw = (Iw >= threshold).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)

        if num_labels < 3:  # background + at least 2 components
            continue

        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = np.argmax(areas) + 1
        max_area = areas[max_idx - 1]
        other_area = areas.sum() - max_area

        if other_area <= 0:
            continue

        score = max_area / other_area
        if score < prev_score:
            # Peak found
            cx_local = centroids[max_idx][0] + sx
            cy_local = centroids[max_idx][1] + sy
            equiv_r = np.sqrt(max_area / np.pi)
            result = (cx_local, cy_local, equiv_r)
            break
        prev_score = score

    return result


def fit_cr_radius(
    image: np.ndarray,
    crx: float,
    cry: float,
    crar: float,
    angle_delta: float = np.pi / 180,
) -> float | None:
    """Fit corneal reflection radius via Nelder-Mead optimization."""
    if crx == 0 or cry == 0 or crar == 0:
        return None

    height, width = image.shape

    def objective(r_arr):
        r = float(r_arr[0])
        m = np.arange(0, 2 * np.pi, angle_delta)
        cos_m, sin_m = np.cos(m), np.sin(m)

        x1 = crx + (r + 1) * cos_m
        y1 = cry + (r + 1) * sin_m
        x2 = crx + (r - 1) * cos_m
        y2 = cry + (r - 1) * sin_m

        valid = (
            (x1 > 0) & (y1 > 0) & (x1 < width) & (y1 < height)
            & (x2 > 0) & (y2 > 0) & (x2 < width) & (y2 < height)
        )
        if not np.any(valid):
            return 1.0

        ix1 = np.clip(np.ceil(x1[valid]).astype(int), 0, width - 1)
        iy1 = np.clip(np.ceil(y1[valid]).astype(int), 0, height - 1)
        ix2 = np.clip(np.ceil(x2[valid]).astype(int), 0, width - 1)
        iy2 = np.clip(np.ceil(y2[valid]).astype(int), 0, height - 1)

        Isum_outer = image[iy1, ix1].astype(np.float64).sum()
        Isum_inner = image[iy2, ix2].astype(np.float64).sum() + 1e-6
        return Isum_outer / Isum_inner

    res = minimize(objective, [crar], method="Nelder-Mead",
                   options={"maxiter": 200, "xatol": 0.5})
    r = float(res.x[0])
    return r if r > 0 else None


def remove_cr(
    image: np.ndarray,
    crx: float,
    cry: float,
    crr: int,
    angle_delta: float = np.pi / 180,
) -> np.ndarray:
    """Remove corneal reflection by polar interpolation."""
    if crx == 0 or cry == 0 or crr <= 0:
        return image

    height, width = image.shape
    img = image.copy().astype(np.float64)

    crx_i, cry_i = int(round(crx)), int(round(cry))
    if crx_i - crr < 0 or crx_i + crr >= width or cry_i - crr < 0 or cry_i + crr >= height:
        return image

    theta = np.arange(0, 2 * np.pi, np.pi / 360)
    n_theta = len(theta)

    # Sample perimeter intensities
    px = np.round(crx + crr * np.cos(theta)).astype(int)
    py = np.round(cry + crr * np.sin(theta)).astype(int)
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    perimeter_vals = img[py, px]
    avg = perimeter_vals.mean()

    # Interpolate for each radius
    for r in range(1, crr + 1):
        w = r / crr
        ix = np.round(crx + r * np.cos(theta)).astype(int)
        iy = np.round(cry + r * np.sin(theta)).astype(int)
        ix = np.clip(ix, 0, width - 1)
        iy = np.clip(iy, 0, height - 1)
        interp = avg * (1 - w) + perimeter_vals * w
        img[iy, ix] = interp

    return img.astype(image.dtype)


# ---------------------------------------------------------------------------
# Edge detection: starburst ray-casting
# ---------------------------------------------------------------------------

def _locate_edge_points(
    image: np.ndarray,
    cx: float,
    cy: float,
    dis: int,
    angle_step: float,
    angle_normal: float,
    angle_spread: float,
    edge_thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast rays and find edge points where intensity jumps above threshold.

    Note: MATLAB coordinate convention — cx=x (col), cy=y (row).
    """
    height, width = image.shape
    angles = np.arange(
        angle_normal - angle_spread / 2 + 0.0001,
        angle_normal + angle_spread / 2,
        angle_step,
    )
    max_dist = max(height, width) // dis

    epx, epy, dirs = [], [], []

    for a in angles:
        cos_a, sin_a = np.cos(a), np.sin(a)
        for s in range(2, max_dist):
            x1 = round(cx + s * dis * cos_a)
            y1 = round(cy + s * dis * sin_a)
            if y1 >= height or y1 < 0 or x1 >= width or x1 < 0:
                break
            x0 = round(cx + max(s - 1, 1) * dis * cos_a)
            y0 = round(cy + max(s - 1, 1) * dis * sin_a)
            if y0 >= height or y0 < 0 or x0 >= width or x0 < 0:
                continue
            d = float(image[y1, x1]) - float(image[y0, x0])
            if d >= edge_thresh:
                epx.append(x0)
                epy.append(y0)
                dirs.append(d)
                break

    return np.array(epx), np.array(epy), np.array(dirs)


def starburst_detect(
    image: np.ndarray,
    cx: float,
    cy: float,
    edge_thresh: float = 10,
    n_rays: int = 36,
    min_features: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative starburst edge detection.

    Returns:
        (epx, epy) edge point arrays, or empty arrays on failure.
    """
    dis = 7
    angle_spread = 100 * np.pi / 180
    angle_step = 2 * np.pi / n_rays

    loop_count = 0
    et = edge_thresh

    while et > 5 and loop_count <= 10:
        epx = np.array([])
        epy = np.array([])

        while len(epx) < min_features and et > 5:
            epx, epy, epd = _locate_edge_points(
                image, cx, cy, dis, angle_step, 0, 2 * np.pi, et
            )
            if len(epx) < min_features:
                et -= 1

        if et <= 5:
            break

        # Refine: recast from each edge point
        all_epx, all_epy = [epx], [epy]
        angle_normals = np.arctan2(cy - epy, cx - epx)

        for i in range(len(epx)):
            step = angle_step * (et / (epd[i] + 1e-6))
            tepx, tepy, _ = _locate_edge_points(
                image, epx[i], epy[i], dis, step, angle_normals[i], angle_spread, et
            )
            if len(tepx) > 0:
                all_epx.append(tepx)
                all_epy.append(tepy)

        epx = np.concatenate(all_epx)
        epy = np.concatenate(all_epy)

        loop_count += 1
        new_cx, new_cy = epx.mean(), epy.mean()
        if abs(new_cx - cx) + abs(new_cy - cy) < 10:
            break
        cx, cy = new_cx, new_cy

    if loop_count > 10 or et <= 5:
        return np.array([]), np.array([])

    return epx, epy


# ---------------------------------------------------------------------------
# RANSAC ellipse fitting
# ---------------------------------------------------------------------------

def fit_ellipse_ransac(
    x: np.ndarray, y: np.ndarray, max_iterations: int = 10000
) -> np.ndarray | None:
    """RANSAC-based robust ellipse fitting.

    Returns [a, b, cx, cy, theta] or None on failure.
    """
    n = len(x)
    if n < 5:
        return None

    nx, ny, H = normalize_coords(x, y)
    dist_thresh = np.sqrt(3.84) * H[0, 0]

    ep = np.column_stack([nx, ny, np.ones(n)])

    best_inliers = 0
    best_ellipse = None
    best_inlier_idx = None
    N = float("inf")
    iteration = 0
    adaptive = False

    while N > iteration and iteration < max_iterations:
        if not adaptive:
            idx = np.random.choice(n, 5, replace=False)
        else:
            idx = np.random.choice(best_inlier_idx, min(5, len(best_inlier_idx)), replace=False)

        nxi, nyi = nx[idx], ny[idx]
        A = np.column_stack([
            nxi * nxi, nxi * nyi, nyi * nyi, nxi, nyi, np.ones(len(idx))
        ])
        try:
            _, _, Vt = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            iteration += 1
            continue

        conic = Vt[-1]
        conic_mat = np.array([
            [conic[0], conic[1] / 2, conic[3] / 2],
            [conic[1] / 2, conic[2], conic[4] / 2],
            [conic[3] / 2, conic[4] / 2, conic[5]],
        ])

        diserr = np.sum((ep @ conic_mat) * ep, axis=1)
        inlier_mask = np.abs(diserr) < dist_thresh
        n_inliers = inlier_mask.sum()

        if n_inliers > best_inliers:
            ne = convert_conic(conic)
            if ne[0] > 0 and ne[1] > 0:
                e = denormalize(ne, H)
                er = e[0] / e[1] if e[1] > 0 else 0
                if 0.75 < er < 1.34:
                    best_inliers = n_inliers
                    best_inlier_idx = np.where(inlier_mask)[0]
                    best_ellipse = e
                    ratio = (n_inliers / n) ** 5
                    if ratio < 1.0:
                        N = np.log(1 - 0.99) / np.log(1 - ratio + 1e-15)
                    else:
                        N = 0
                    adaptive = True

        iteration += 1

    return best_ellipse


# ---------------------------------------------------------------------------
# Model-based ellipse refinement (Nelder-Mead)
# ---------------------------------------------------------------------------

def fit_ellipse_model(
    image: np.ndarray,
    ellipse: np.ndarray,
    angle_delta: float = np.pi / 180,
) -> np.ndarray | None:
    """Refine ellipse via Nelder-Mead: maximize outer/inner intensity ratio."""
    height, width = image.shape

    def objective(v):
        a, b, cx, cy, theta = v
        m = np.arange(0, 2 * np.pi, angle_delta)
        cos_m, sin_m = np.cos(m), np.sin(m)
        rc, rs = np.abs(cos_m), np.abs(sin_m)
        ct, st = np.cos(theta), np.sin(theta)

        # Outer perimeter
        x1 = (a + rc) * cos_m
        y1 = (b + rs) * sin_m
        xr1 = x1 * ct - y1 * st + cx
        yr1 = x1 * st + y1 * ct + cy

        # Inner perimeter
        x2 = (a - rc) * cos_m
        y2 = (b - rs) * sin_m
        xr2 = x2 * ct - y2 * st + cx
        yr2 = x2 * st + y2 * ct + cy

        valid = (
            (xr1 > 0) & (yr1 > 0) & (xr1 < width) & (yr1 < height)
            & (xr2 > 0) & (yr2 > 0) & (xr2 < width) & (yr2 < height)
        )
        if not np.any(valid):
            return 0.0

        ix1 = np.clip(np.ceil(xr1[valid]).astype(int), 0, width - 1)
        iy1 = np.clip(np.ceil(yr1[valid]).astype(int), 0, height - 1)
        ix2 = np.clip(np.ceil(xr2[valid]).astype(int), 0, width - 1)
        iy2 = np.clip(np.ceil(yr2[valid]).astype(int), 0, height - 1)

        Isum_outer = image[iy1, ix1].astype(np.float64).sum()
        Isum_inner = image[iy2, ix2].astype(np.float64).sum() + 1e-6
        return -(Isum_outer / Isum_inner)

    res = minimize(objective, ellipse, method="Nelder-Mead",
                   options={"maxiter": 1000, "xatol": 0.1, "fatol": 1e-4})
    result = res.x

    # Validate: center within image
    if result[2] < 0 or result[2] >= width or result[3] < 0 or result[3] >= height:
        return None
    if result[0] <= 0 or result[1] <= 0:
        return None

    return result


# ---------------------------------------------------------------------------
# Single-frame processing
# ---------------------------------------------------------------------------

def process_frame(
    image: np.ndarray,
    sx: float,
    sy: float,
    edge_thresh: float = 10,
    n_rays: int = 36,
    sigma: float = 4.0,
    min_features: int = 10,
    max_ransac: int = 10000,
    remove_cr_flag: bool = False,
    angle_delta: float = np.pi / 180,
) -> np.ndarray | None:
    """Process a single frame with the starburst algorithm.

    Args:
        image: grayscale float64 image
        sx, sy: starting search point (x=col, y=row)
        All other params match MATLAB starburst_process_frame.m

    Returns:
        [a, b, cx, cy, theta] or None on failure.
    """
    img = image.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0

    # Gaussian smoothing
    ksize = int(np.ceil(2.5 * sigma))
    if ksize % 2 == 0:
        ksize += 1
    img = cv2.GaussianBlur(img, (ksize, ksize), sigma)

    # Corneal reflection removal
    if remove_cr_flag:
        cr_window = min(301, min(img.shape) - 1)
        if cr_window % 2 == 0:
            cr_window -= 1
        crx, cry, crar = locate_cr(img, sx, sy, cr_window)
        if crx > 0 and cry > 0 and crar > 0:
            crr = fit_cr_radius(img, crx, cry, crar, angle_delta)
            if crr is not None and crr > 0:
                crr = int(np.ceil(crr * 2.5))
                img = remove_cr(img, crx, cry, crr, angle_delta)

    # Scale to uint8-like range for edge detection
    img_scaled = (img * 255).clip(0, 255)

    # Starburst edge detection
    epx, epy = starburst_detect(img_scaled, sx, sy, edge_thresh, n_rays, min_features)
    if len(epx) < min_features:
        return None

    # RANSAC ellipse fitting
    ellipse = fit_ellipse_ransac(epx, epy, max_ransac)
    if ellipse is None:
        return None

    # Model-based refinement
    result = fit_ellipse_model(img_scaled, ellipse, angle_delta)
    if result is None:
        return None

    # Validate center within image
    if result[2] < 0 or result[2] >= image.shape[1] or result[3] < 0 or result[3] >= image.shape[0]:
        return None

    return result


# ---------------------------------------------------------------------------
# Frame-level detection dispatcher
# ---------------------------------------------------------------------------

def detect_starburst(
    frames: list[tuple[int, np.ndarray]],
    edge_thresh: float = 10,
    n_rays: int = 36,
    sigma: float = 4.0,
    min_features: int = 10,
    max_ransac: int = 10000,
    remove_cr: bool = False,
    sequential: bool = True,
) -> dict[int, np.ndarray]:
    """Run starburst detection on a list of (index, frame) pairs.

    Returns:
        dict mapping frame_index -> binary mask (uint8).
    """
    results = {}

    # Initial start point: center of first frame
    if not frames:
        return results

    h, w = frames[0][1].shape[:2]
    sx, sy = w / 2, h / 2

    for idx, frame in frames:
        pe = process_frame(
            frame, sx, sy,
            edge_thresh=edge_thresh,
            n_rays=n_rays,
            sigma=sigma,
            min_features=min_features,
            max_ransac=max_ransac,
            remove_cr_flag=remove_cr,
        )

        if pe is not None:
            a, b, cx, cy, theta = pe
            mask = ellipse_to_mask(h, w, a, b, cx, cy, theta)
            results[idx] = mask
            if sequential:
                sx, sy = cx, cy
        else:
            results[idx] = np.zeros((h, w), dtype=np.uint8)

    return results
