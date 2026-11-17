"""scanner.py — find a document in a photo and flatten it.

This is the computer-vision heart of the project. The pipeline:

    photo ──► resize ──► grayscale ──► Gaussian blur ──► Canny edges
          ──► contours ──► 4-point polygon ──► order corners ──► perspective warp

Why detect on a *small* copy of the photo?
    A phone photo has ~12 million pixels. Edge detection on that is slow and
    picks up every wood-grain line. At 500 px tall the page outline is still
    obvious, noise is gone and everything runs in milliseconds. We then scale
    the corners back up and warp the **full-resolution** photo, so the final
    scan stays sharp.

Everything here uses classic OpenCV operations. No machine-learning model.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from utils import (
    DocumentNotFoundError,
    ScannerError,
    ensure_valid_image,
    resize_keep_aspect,
)

logger = logging.getLogger(__name__)

MAX_CANDIDATES = 10
MAX_AREA_RATIO = 0.98
CORNER_LABELS = ("TL", "TR", "BR", "BL")


@dataclass(frozen=True)
class DetectionParams:
    """Tunable settings for document detection.

    Attributes:
        blur_kernel:     Gaussian blur window (odd number). Bigger = smoother,
                         fewer false edges, but blurrier page borders.
        canny_low:       Canny lower threshold. Edge pixels weaker than this are dropped.
        canny_high:      Canny upper threshold. Pixels stronger than this are always edges.
        min_area_ratio:  The document must cover at least this fraction of the photo.
        detection_height: Height (px) of the working copy used for detection.
        approx_epsilon:  Polygon simplification strength, as a fraction of the
                         contour perimeter (see ``cv2.approxPolyDP``).
    """

    blur_kernel: int = 5
    canny_low: int = 50
    canny_high: int = 150
    min_area_ratio: float = 0.15
    detection_height: int = 500
    approx_epsilon: float = 0.02

    def __post_init__(self) -> None:
        if self.blur_kernel < 3 or self.blur_kernel % 2 == 0:
            raise ValueError("Blur strength must be an odd number of at least 3.")
        if not 0 <= self.canny_low < self.canny_high:
            raise ValueError("The Canny low threshold must be smaller than the high threshold.")
        if not 0.01 <= self.min_area_ratio < 1.0:
            raise ValueError("Minimum document size must be between 1% and 99% of the photo.")
        if self.detection_height < 100:
            raise ValueError("Detection height must be at least 100 px.")
        if not 0.005 <= self.approx_epsilon <= 0.1:
            raise ValueError("Polygon epsilon must be between 0.005 and 0.1.")


@dataclass(eq=False)
class DetectionResult:
    """What the detector found.

    Attributes:
        corners:  4x2 float32 array in **original-image** pixels, ordered
                  top-left, top-right, bottom-right, bottom-left.
        method:   Which strategy succeeded ("canny", "auto-canny", "otsu", ...).
        edges:    The binary map the contours were taken from (detection resolution).
        contours: The biggest candidate contours (original coordinates), for debugging.
    """

    corners: np.ndarray
    method: str
    edges: np.ndarray | None
    contours: list[np.ndarray]


@dataclass(eq=False)
class ScanResult:
    """A detection plus the flattened ("warped") document image."""

    warped: np.ndarray
    detection: DetectionResult


def order_points(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    ``cv2.getPerspectiveTransform`` pairs source and destination points *by
    position in the list*, so if the corners arrive in a random order the page
    would be mirrored or twisted. We must sort them first.

    The classic textbook trick uses ``x + y`` (smallest = top-left, largest =
    bottom-right) and ``y - x`` for the other two. It breaks when the page is
    rotated close to 45 degrees, because two corners then tie.

    This version is more robust:
        1. Compute the centroid (average of the 4 points).
        2. Sort the points by their angle around the centroid. In image
           coordinates (y points down) increasing angle runs *clockwise on
           screen*, so the order is automatically TL -> TR -> BR -> BL
           up to a rotation of the list.
        3. Rotate the list so it starts with the point whose ``x + y`` is
           smallest (the top-left one).

    Raises:
        ValueError: if ``points`` cannot be reshaped to 4x2.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected exactly 4 points, got {pts.shape[0]}.")

    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    clockwise = pts[np.argsort(angles)]
    start = int(np.argmin(clockwise.sum(axis=1)))
    return np.roll(clockwise, -start, axis=0).astype(np.float32)


def _is_plausible_quad(rect: np.ndarray, min_angle: float = 40.0, max_angle: float = 140.0) -> bool:
    """Reject slivers: every interior angle of a photographed page is roughly 90 degrees."""
    for i in range(4):
        v1 = rect[i - 1] - rect[i]
        v2 = rect[(i + 1) % 4] - rect[i]
        denominator = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denominator < 1e-6:
            return False
        cosine = float(np.clip(np.dot(v1, v2) / denominator, -1.0, 1.0))
        if not min_angle <= np.degrees(np.arccos(cosine)) <= max_angle:
            return False
    return True


def _canny_edges(blurred: np.ndarray, low: int, high: int) -> np.ndarray:
    """Canny edge detector followed by a morphological *closing*.

    Canny keeps a pixel when its gradient is above ``high``, or above ``low``
    **and** connected to a strong pixel (hysteresis). Closing = dilate then
    erode; it bridges 1-2 px gaps so a slightly broken page border becomes one
    closed loop that ``findContours`` can follow.
    """
    edges = cv2.Canny(blurred, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def _auto_canny_edges(blurred: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny with thresholds derived from the image's median brightness.

    Useful when the user's fixed thresholds are wrong for a very dark or very
    bright photo. ``low = (1 - sigma) * median``, ``high = (1 + sigma) * median``.
    """
    median = float(np.median(blurred))
    low = int(max(0.0, (1.0 - sigma) * median))
    high = int(min(255.0, (1.0 + sigma) * median))
    if high <= low:
        high = low + 1
    return _canny_edges(blurred, low, high)


def _otsu_mask(blurred: np.ndarray, invert: bool) -> np.ndarray:
    """Segment the page as a blob instead of tracing edges.

    Otsu's method picks the gray level that best separates two pixel groups
    (it maximises the variance *between* the groups). A bright page on a dark
    desk becomes one white blob. ``invert=True`` handles a dark page on a
    bright surface. Opening removes specks, closing fills holes made by text.
    """
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(blurred, 0, 255, mode + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _quad_ok(rect: np.ndarray, min_area: float, max_area: float) -> bool:
    area = float(cv2.contourArea(rect.reshape(-1, 1, 2)))
    return min_area <= area <= max_area and _is_plausible_quad(rect)


def _find_quad(
    binary: np.ndarray, min_area: float, max_area: float, epsilon: float
) -> tuple[np.ndarray | None, list[np.ndarray]]:
    """Find the largest convex four-sided polygon in a binary map.

    Pass 1 (strict): simplify each big contour with ``approxPolyDP``. A page
    border simplifies to exactly 4 vertices.
    Pass 2 (relaxed): if no contour simplifies cleanly (curled page, a thumb
    over a corner), take its *convex hull* and simplify harder.

    Returns:
        ``(ordered_quad_or_None, biggest_contours)``.
    """
    found = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    contours = sorted(found, key=cv2.contourArea, reverse=True)[:MAX_CANDIDATES]

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            break
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            rect = order_points(approx.reshape(4, 2))
            if _quad_ok(rect, min_area, max_area):
                return rect, contours

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            break
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        for eps in np.linspace(epsilon, epsilon * 3.0, 5):
            approx = cv2.approxPolyDP(hull, float(eps) * perimeter, True)
            if len(approx) < 4:
                break
            if len(approx) == 4:
                rect = order_points(approx.reshape(4, 2))
                if _quad_ok(rect, min_area, max_area):
                    return rect, contours
                break
    return None, contours


def detect_document(image: np.ndarray, params: DetectionParams | None = None) -> DetectionResult:
    """Detect the four corners of the document in ``image``.

    Strategies are tried in order until one yields a plausible quadrilateral:
    ``canny`` (your thresholds), ``auto-canny``, ``otsu``, ``otsu-inverted``.

    Raises:
        InvalidImageError:     if ``image`` is not a valid array.
        DocumentNotFoundError: if no strategy finds a document. The exception
                               carries the Canny edge map for debugging.
    """
    params = params or DetectionParams()
    ensure_valid_image(image)

    if image.shape[0] > params.detection_height:
        small, scale = resize_keep_aspect(image, height=params.detection_height)
    else:
        small, scale = image, 1.0

    gray = small if small.ndim == 2 else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (params.blur_kernel, params.blur_kernel), 0)

    height, width = blurred.shape
    min_area = params.min_area_ratio * height * width
    max_area = MAX_AREA_RATIO * height * width

    strategies: tuple[tuple[str, Callable[[], np.ndarray]], ...] = (
        ("canny", lambda: _canny_edges(blurred, params.canny_low, params.canny_high)),
        ("auto-canny", lambda: _auto_canny_edges(blurred)),
        ("otsu", lambda: _otsu_mask(blurred, invert=False)),
        ("otsu-inverted", lambda: _otsu_mask(blurred, invert=True)),
    )

    first_map: np.ndarray | None = None
    for name, build_map in strategies:
        binary = build_map()
        if first_map is None:
            first_map = binary
        quad, contours = _find_quad(binary, min_area, max_area, params.approx_epsilon)
        if quad is None:
            logger.debug("Strategy %s found no document", name)
            continue

        orig_h, orig_w = image.shape[:2]
        corners = quad / scale
        corners[:, 0] = np.clip(corners[:, 0], 0, orig_w - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, orig_h - 1)
        candidates = [np.round(c.astype(np.float32) / scale).astype(np.int32) for c in contours[:5]]
        logger.debug("Document found with strategy %s", name)
        return DetectionResult(corners.astype(np.float32), name, binary, candidates)

    raise DocumentNotFoundError(
        "No document outline was found. Make sure all four corners are visible, "
        "the page contrasts with the surface under it (a dark table works best) "
        "and the photo is well lit. You can also lower the minimum document size "
        "in the detection settings, or use the whole photo.",
        edges=first_map,
    )


def full_image_detection(image: np.ndarray) -> DetectionResult:
    """Treat the whole photo as the document (manual fallback, no detection)."""
    ensure_valid_image(image)
    height, width = image.shape[:2]
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    return DetectionResult(corners, "full-image", None, [])


def _output_size(rect: np.ndarray) -> tuple[int, int]:
    """Width and height of the flattened page, from the quad's side lengths.

    A tilted page has two different top/bottom lengths (the near edge looks
    longer). We take the **longer** of each pair so no detail is squeezed out.
    """
    top_left, top_right, bottom_right, bottom_left = rect
    width = max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left))
    height = max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left))
    return int(round(float(width))) + 1, int(round(float(height))) + 1


def four_point_transform(
    image: np.ndarray, corners: np.ndarray, max_side: int = 3000
) -> np.ndarray:
    """Warp the quadrilateral given by ``corners`` into a flat rectangle.

    Idea: a flat page seen from a camera is related to the ideal front view by
    a 3x3 *homography* matrix ``H``::

        [x']   [h11 h12 h13] [x]
        [y'] ~ [h21 h22 h23] [y]      (x', y') = (X/Z, Y/Z)
        [ 1 ]  [h31 h32 h33] [1]

    ``H`` has 8 unknowns (it is defined up to scale). Each point pair gives 2
    equations, so **4 corner pairs** determine it exactly.
    ``cv2.getPerspectiveTransform`` solves that linear system, and
    ``cv2.warpPerspective`` then, for every output pixel, looks up where it
    came from in the photo (inverse mapping) and interpolates.

    Args:
        image:    Source photo (full resolution).
        corners:  Four corner points in the photo, any order.
        max_side: Longest allowed side of the output, to bound memory use.

    Raises:
        ScannerError: if the quadrilateral is degenerate (tiny).
    """
    ensure_valid_image(image)
    rect = order_points(corners)
    width, height = _output_size(rect)
    if min(width, height) < 10:
        raise ScannerError("The detected document area is too small to scan.")

    shrink = min(1.0, max_side / max(width, height))
    width = max(2, round(width * shrink))
    height = max(2, round(height * shrink))

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def scan_document(
    image: np.ndarray,
    params: DetectionParams | None = None,
    *,
    use_full_image: bool = False,
) -> ScanResult:
    """Detect (or assume) the document and return the flattened result."""
    detection = full_image_detection(image) if use_full_image else detect_document(image, params)
    warped = four_point_transform(image, detection.corners)
    return ScanResult(warped=warped, detection=detection)


def draw_detection(image: np.ndarray, detection: DetectionResult) -> np.ndarray:
    """Return a copy of ``image`` with candidate contours, the chosen quad and corners drawn."""
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = canvas.shape[:2]
    thickness = max(2, round(max(height, width) / 400))

    for contour in detection.contours:
        cv2.polylines(canvas, [contour.reshape(-1, 1, 2)], True, (0, 140, 255), max(1, thickness // 2))

    points = np.round(detection.corners).astype(np.int32)
    cv2.polylines(canvas, [points.reshape(-1, 1, 2)], True, (0, 200, 0), thickness)
    for (x, y), label in zip(points, CORNER_LABELS):
        cv2.circle(canvas, (int(x), int(y)), thickness * 3, (0, 0, 255), -1)
        cv2.putText(
            canvas, label, (int(x) + thickness * 4, int(y) - thickness * 4),
            cv2.FONT_HERSHEY_SIMPLEX, max(0.6, thickness * 0.35), (0, 0, 255),
            max(1, thickness // 2), cv2.LINE_AA,
        )
    return canvas
