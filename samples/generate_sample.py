"""samples/generate_sample.py — build a synthetic "photo of a document".

Used for two things:
  * creating ``samples/sample_document.jpg`` (run this file directly), and
  * giving the test-suite a photo whose true corners are known exactly.

The scene is made by drawing a fake page, warping it with a known homography
(so we know the ground-truth corners), adding a drop shadow, uneven lighting
and sensor noise.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PAGE_SIZE = (850, 1100)
DEFAULT_CORNERS_REL = np.float32([[0.17, 0.10], [0.84, 0.15], [0.79, 0.90], [0.12, 0.84]])

_WORDS = (
    "scanner document perspective contour threshold adaptive gradient edge corner "
    "matrix kernel blur canny pixel image warp quality report summary results "
    "invoice total amount customer order payment date reference account"
).split()


def make_page(seed: int = 3) -> np.ndarray:
    """Draw a flat page: title, text lines, a table, a coloured stamp, uneven light."""
    rng = np.random.default_rng(seed)
    width, height = PAGE_SIZE
    page = np.full((height, width, 3), (238, 242, 245), np.uint8)
    ink = (45, 45, 45)

    cv2.putText(page, "QUARTERLY REPORT", (60, 110), cv2.FONT_HERSHEY_DUPLEX, 1.5, ink, 2, cv2.LINE_AA)
    cv2.line(page, (60, 140), (width - 60, 140), ink, 2)

    y = 190
    for _ in range(14):
        words = rng.choice(_WORDS, size=9)
        cv2.putText(page, " ".join(words)[:62], (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ink, 1, cv2.LINE_AA)
        y += 34

    top, bottom = 700, 940
    cv2.rectangle(page, (60, top), (width - 60, bottom), ink, 2)
    for row in range(1, 5):
        cv2.line(page, (60, top + row * 48), (width - 60, top + row * 48), ink, 1)
    for col in (330, 560):
        cv2.line(page, (col, top), (col, bottom), ink, 1)

    cv2.circle(page, (700, 1010), 40, (40, 40, 200), 3, cv2.LINE_AA)
    cv2.putText(page, "OK", (676, 1022), cv2.FONT_HERSHEY_DUPLEX, 0.9, (40, 40, 200), 2, cv2.LINE_AA)

    light = np.linspace(0.75, 1.0, width, dtype=np.float32)[None, :, None]
    light = light * np.linspace(1.0, 0.9, height, dtype=np.float32)[:, None, None]
    return np.clip(page * light, 0, 255).astype(np.uint8)


def make_document_photo(
    size: tuple[int, int] = (1600, 1200),
    corners_rel: np.ndarray | None = None,
    background_bgr: tuple[int, int, int] = (52, 60, 72),
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(photo_bgr, true_corners)``; corners are TL, TR, BR, BL in pixels."""
    rng = np.random.default_rng(seed)
    width, height = size
    rel = DEFAULT_CORNERS_REL if corners_rel is None else np.asarray(corners_rel, np.float32)
    corners = (rel * np.float32([width, height])).astype(np.float32)

    desk = np.empty((height, width, 3), np.float32)
    desk[:] = background_bgr
    grain = cv2.GaussianBlur(rng.normal(0, 7, (height, width)).astype(np.float32), (0, 0), 2.5)
    stripes = 4 * np.sin(np.arange(height, dtype=np.float32) / 6.0)[:, None]
    desk += (grain + stripes)[..., None]

    page = make_page()
    page_h, page_w = page.shape[:2]
    source = np.float32([[0, 0], [page_w - 1, 0], [page_w - 1, page_h - 1], [0, page_h - 1]])
    matrix = cv2.getPerspectiveTransform(source, corners)
    warped = cv2.warpPerspective(page, matrix, (width, height), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((page_h, page_w), 255, np.uint8), matrix, (width, height))

    shadow = cv2.GaussianBlur(np.roll(mask, (14, 14), axis=(0, 1)), (0, 0), 14) / 255.0 * 0.55
    desk *= (1.0 - shadow)[..., None]

    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    photo = desk * (1.0 - alpha) + warped.astype(np.float32) * alpha
    photo += rng.normal(0, 3, photo.shape).astype(np.float32)
    return np.clip(photo, 0, 255).astype(np.uint8), corners


if __name__ == "__main__":
    output = Path(__file__).with_name("sample_document.jpg")
    photo, _ = make_document_photo()
    cv2.imwrite(str(output), photo, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Wrote {output}")
