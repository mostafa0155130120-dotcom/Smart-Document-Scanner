"""image_processing.py — scanning filters and image adjustments.

After the perspective warp the page is flat, but it still looks like a photo:
grey-ish paper, a shadow across one side, a colour cast. This module turns it
into something that looks scanned.

The key idea: illumination normalisation
----------------------------------------
Imagine the page *without any ink*: that is just the paper lit by uneven
light. If we can estimate that "paper brightness" ``B(x, y)`` at every pixel,
then dividing the photo by it cancels the lighting::

    scan(x, y) = photo(x, y) / B(x, y) * 255

Paper becomes 255 (white) everywhere, ink stays dark. We estimate ``B`` by
removing the ink from the picture:

    1. *Dilation* replaces every pixel with the maximum of its neighbourhood,
       so thin dark strokes disappear (paper is brighter than ink).
    2. A *median blur* then smooths what is left into a soft lighting map.

Common failure: large dark areas (photos, big logos, solid colour blocks) are
mistaken for "shadow" and get washed out.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from utils import ensure_valid_image

logger = logging.getLogger(__name__)


class ScanMode(str, Enum):
    """Scanning looks. The value is the label shown in the UI."""

    ORIGINAL = "Original Color"
    ENHANCED = "Enhanced Color"
    GRAYSCALE = "Grayscale"
    BLACK_WHITE = "Black & White"
    ADAPTIVE = "Adaptive Threshold"


BINARY_MODES = frozenset({ScanMode.BLACK_WHITE, ScanMode.ADAPTIVE})


@dataclass(frozen=True)
class ScanSettings:
    """Everything the user can change after the document has been flattened.

    Attributes:
        mode:                Which scanning look to apply.
        brightness:          -100..100. In the two binary modes it shifts the
                             threshold instead (positive = lighter result).
        contrast:            -80..100 (%). Ignored by the binary modes.
        sharpness:           0..3 unsharp-mask strength.
        quarter_turns:       Clockwise 90-degree turns (lossless).
        fine_rotation:       Extra clockwise rotation in degrees, to straighten text.
        adaptive_block_size: Neighbourhood size for adaptive threshold; 0 = automatic.
        adaptive_c:          Constant subtracted from the local mean; raise it to
                             remove speckle, lower it to keep faint ink.
    """

    mode: ScanMode = ScanMode.ENHANCED
    brightness: int = 0
    contrast: int = 0
    sharpness: float = 0.0
    quarter_turns: int = 0
    fine_rotation: float = 0.0
    adaptive_block_size: int = 0
    adaptive_c: int = 10

    def __post_init__(self) -> None:
        if self.adaptive_block_size != 0 and (
            self.adaptive_block_size < 3 or self.adaptive_block_size % 2 == 0
        ):
            raise ValueError("The adaptive neighbourhood size must be odd and at least 3 (or 0 for automatic).")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """BGR -> gray using ``cv2.cvtColor`` (gray = 0.299 R + 0.587 G + 0.114 B)."""
    ensure_valid_image(image)
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def estimate_background(channel: np.ndarray) -> np.ndarray:
    """Estimate the lighting map ``B(x, y)`` of a single-channel image.

    The estimate is computed on a copy shrunk to <= 600 px (fast, and a
    lighting map is smooth anyway) and then scaled back up.
    """
    height, width = channel.shape
    scale = min(1.0, 600.0 / max(height, width))
    small = (
        cv2.resize(channel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else channel
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    background = cv2.dilate(small, kernel)
    background = cv2.medianBlur(background, 21)
    if scale < 1.0:
        background = cv2.resize(background, (width, height), interpolation=cv2.INTER_LINEAR)
    return background


def normalize_illumination(channel: np.ndarray) -> np.ndarray:
    """Divide a channel by its estimated background: paper -> white, lighting removed."""
    background = np.maximum(estimate_background(channel), 1)
    return cv2.divide(channel, background, scale=255)


def enhance_color(image: np.ndarray) -> np.ndarray:
    """Colour scan with shadows removed and paper turned white.

    1. Normalise each of B, G and R separately. This also removes a colour cast
       (yellow indoor light makes the paper's blue channel darker, for example).
    2. Convert to LAB, where lightness ``L`` is separate from colour, and apply
       CLAHE (contrast-limited adaptive histogram equalisation) to ``L`` only.
       CLAHE stretches contrast inside small tiles; ``clipLimit`` caps how much,
       so noise is not exaggerated.
    """
    ensure_valid_image(image)
    if image.ndim == 2:
        return enhance_grayscale(image)
    balanced = cv2.merge([normalize_illumination(c) for c in cv2.split(image)])
    lightness, a_channel, b_channel = cv2.split(cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB))
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    merged = cv2.merge([clahe.apply(lightness), a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def enhance_grayscale(image: np.ndarray) -> np.ndarray:
    """Shadow-free grayscale (keeps shading, e.g. in stamps and photographs)."""
    return normalize_illumination(to_grayscale(image))


def apply_black_white(image: np.ndarray, bias: int = 0) -> np.ndarray:
    """Pure black-and-white using one global threshold chosen by Otsu's method.

    Otsu tries every threshold ``t`` from 0 to 255 and keeps the one that
    maximises the *between-class variance* of the two groups it creates
    (dark ink vs bright paper). It works well here because
    ``normalize_illumination`` has already made the histogram two-humped.

    ``bias`` (-100..100) moves the threshold: positive = lighter result, so
    faint ink disappears; negative = darker result, so thin strokes thicken.
    """
    gray = cv2.GaussianBlur(normalize_illumination(to_grayscale(image)), (3, 3), 0)
    otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(np.clip(otsu_threshold - 0.5 * bias, 1, 254))
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def _automatic_block_size(shape: tuple[int, ...]) -> int:
    """About 1/30 of the shorter side, odd, at least 11 px."""
    block = max(11, min(shape[:2]) // 30)
    return block if block % 2 == 1 else block + 1


def apply_adaptive_threshold(
    image: np.ndarray, block_size: int = 0, c: int = 10, bias: int = 0
) -> np.ndarray:
    """Black-and-white with a *different threshold for every pixel*.

    For each pixel, ``cv2.adaptiveThreshold`` computes a Gaussian-weighted mean
    of the ``block_size x block_size`` neighbourhood, subtracts ``c`` and
    compares::

        output = 255 if pixel > local_mean - c else 0

    Because the threshold follows the local brightness, shadows and gradients
    do not matter. Parameters:
        * ``block_size``: must be odd and larger than a letter's stroke width,
          but small enough to stay "local". Too small -> hollow letters;
          too large -> behaves like a global threshold.
        * ``c``: bigger = whiter background and less speckle, but faint ink vanishes.
    """
    gray = cv2.GaussianBlur(to_grayscale(image), (3, 3), 0)
    block = block_size or _automatic_block_size(gray.shape)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, float(c + 0.2 * bias)
    )


def adjust_brightness_contrast(image: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    """Linear adjustment ``out = alpha * (in - 128) + 128 + brightness``.

    Pivoting contrast around mid-gray (128) means raising contrast darkens the
    darks and lightens the lights instead of simply brightening everything.
    ``cv2.convertScaleAbs`` does the arithmetic and clips to 0..255.
    """
    ensure_valid_image(image)
    alpha = 1.0 + float(np.clip(contrast, -80, 100)) / 100.0
    beta = 128.0 * (1.0 - alpha) + float(np.clip(brightness, -100, 100))
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def sharpen(image: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Unsharp mask: ``out = (1 + amount) * image - amount * blurred(image)``.

    Subtracting a blurred copy leaves only fine detail, which is added back
    with weight ``amount``. Too much creates bright halos around letters.
    """
    ensure_valid_image(image)
    if amount <= 0:
        return image.copy()
    blurred = cv2.GaussianBlur(image, (0, 0), 1.5)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate **clockwise** by ``angle`` degrees.

    Multiples of 90 use ``cv2.rotate`` (exact, no resampling). Other angles use
    ``cv2.warpAffine`` on an enlarged canvas so nothing is cut off, with a
    white border.
    """
    ensure_valid_image(image)
    angle = float(angle) % 360.0
    if angle < 1e-6 or angle > 360.0 - 1e-6:
        return image.copy()
    for degrees, code in (
        (90.0, cv2.ROTATE_90_CLOCKWISE),
        (180.0, cv2.ROTATE_180),
        (270.0, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ):
        if abs(angle - degrees) < 1e-6:
            return cv2.rotate(image, code)

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(np.ceil(height * sin + width * cos))
    new_h = int(np.ceil(height * cos + width * sin))
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    return cv2.warpAffine(
        image, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )


def process_scan(image: np.ndarray, settings: ScanSettings | None = None) -> np.ndarray:
    """Apply the chosen scanning look, the adjustments and the rotation.

    Order matters:
        * Colour/gray looks: filter first, then brightness/contrast/sharpness
          (otherwise the normalisation would undo the adjustments).
        * Binary looks: sharpen first (crisper edges to threshold); brightness
          acts on the threshold itself.
        * Rotation is always last.
    """
    settings = settings or ScanSettings()
    ensure_valid_image(image)
    mode = settings.mode

    if mode is ScanMode.BLACK_WHITE:
        result = apply_black_white(sharpen(to_grayscale(image), settings.sharpness), settings.brightness)
    elif mode is ScanMode.ADAPTIVE:
        result = apply_adaptive_threshold(
            sharpen(to_grayscale(image), settings.sharpness),
            settings.adaptive_block_size, settings.adaptive_c, settings.brightness,
        )
    else:
        if mode is ScanMode.ORIGINAL:
            result = image.copy()
        elif mode is ScanMode.ENHANCED:
            result = enhance_color(image)
        elif mode is ScanMode.GRAYSCALE:
            result = enhance_grayscale(image)
        else:  # pragma: no cover - guards against future enum members
            raise ValueError(f"Unknown scan mode: {mode!r}")
        result = adjust_brightness_contrast(result, settings.brightness, settings.contrast)
        result = sharpen(result, settings.sharpness)

    result = rotate_image(result, (settings.quarter_turns % 4) * 90)
    if abs(settings.fine_rotation) > 0.01:
        result = rotate_image(result, settings.fine_rotation)
    return result
