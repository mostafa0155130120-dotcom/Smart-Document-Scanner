"""utils.py — validation, decoding, encoding and resizing helpers.

Everything that several modules need lives here so that ``scanner.py``,
``image_processing.py`` and ``pdf_generator.py`` stay focused on one job each.

Conventions used across the project
-----------------------------------
* Images are NumPy arrays of dtype ``uint8``.
* Colour images use OpenCV's **BGR** channel order (not RGB).
* Grayscale / black-and-white images are 2-D arrays (height x width).
* Corner points are ``(x, y)`` pairs in pixels.
"""
from __future__ import annotations

import io
import logging
import os

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #
ALLOWED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")
ALLOWED_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG"})  # Pillow format names
MAX_FILE_SIZE_MB = 15
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MIN_SIDE_PX = 100  # anything smaller cannot hold a readable document
MAX_PIXELS = 60_000_000  # ~60 megapixels; protects memory on small servers


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class ScannerError(Exception):
    """Base class for every error this project raises on purpose.

    The UI catches ``ScannerError`` and shows its message to the user, so
    messages must be written for end users, not for developers.
    """


class InvalidImageError(ScannerError):
    """The input is not a usable image (wrong format, corrupt, too big, ...)."""


class DocumentNotFoundError(ScannerError):
    """No four-sided document outline could be found in the image.

    ``edges`` optionally carries the edge map the detector saw, so the UI can
    show it and help the user understand why detection failed.
    """

    def __init__(self, message: str, edges: np.ndarray | None = None) -> None:
        super().__init__(message)
        self.edges = edges


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_extension(filename: str | None) -> str:
    """Return the lower-case extension of ``filename`` or raise.

    Raises:
        InvalidImageError: if the extension is not JPG, JPEG or PNG.
    """
    extension = os.path.splitext(filename or "")[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(ALLOWED_EXTENSIONS)
        raise InvalidImageError(
            f"Unsupported file type '{extension or 'unknown'}'. Please use one of: {allowed}."
        )
    return extension


def validate_file_size(num_bytes: int) -> None:
    """Raise if a file is empty or larger than ``MAX_FILE_SIZE_MB``."""
    if num_bytes <= 0:
        raise InvalidImageError("The file is empty.")
    if num_bytes > MAX_FILE_SIZE_BYTES:
        size_mb = num_bytes / (1024 * 1024)
        raise InvalidImageError(
            f"The file is {size_mb:.1f} MB; the limit is {MAX_FILE_SIZE_MB} MB. "
            "Try a smaller photo or lower the camera resolution."
        )


def ensure_valid_image(image: object) -> None:
    """Raise ``InvalidImageError`` unless ``image`` is a usable uint8 array."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise InvalidImageError("No image data was provided.")
    if image.dtype != np.uint8:
        raise InvalidImageError("Images must be 8-bit (uint8) arrays.")
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3):
        return
    raise InvalidImageError("Images must be grayscale or 3-channel (BGR) arrays.")


# --------------------------------------------------------------------------- #
# Loading and saving
# --------------------------------------------------------------------------- #
def load_image_from_bytes(data: bytes, filename: str | None = None) -> np.ndarray:
    """Decode raw JPG/PNG bytes into a BGR image, validating along the way.

    Steps:
        1. Check the size (and the extension when a filename is given).
        2. Peek at the header with Pillow. This does *not* decode pixels, so a
           "decompression bomb" is rejected before it can eat all memory.
        3. Decode with ``cv2.imdecode``. With ``IMREAD_COLOR`` OpenCV also
           applies the EXIF orientation tag, so phone photos arrive upright.

    Raises:
        InvalidImageError: for any problem with the input.
    """
    validate_file_size(len(data))
    if filename:
        validate_extension(filename)

    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_format = probe.format
            width, height = probe.size
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(
            "This file could not be read as an image. It may be corrupted."
        ) from exc

    if image_format not in ALLOWED_FORMATS:
        raise InvalidImageError("Only JPG and PNG images are supported.")
    if width * height > MAX_PIXELS:
        raise InvalidImageError(
            f"The image is {width * height / 1e6:.0f} megapixels; the limit is "
            f"{MAX_PIXELS / 1e6:.0f}. Please use a smaller photo."
        )

    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("OpenCV could not decode this image.")

    if min(image.shape[:2]) < MIN_SIDE_PX:
        raise InvalidImageError(
            f"The image is too small ({image.shape[1]}x{image.shape[0]} px). "
            f"Each side must be at least {MIN_SIDE_PX} px."
        )
    return image


def load_image_from_path(path: str | os.PathLike[str]) -> np.ndarray:
    """Load an image file from disk.

    We read the bytes ourselves instead of calling ``cv2.imread`` because
    ``imread`` fails on Windows paths that contain non-ASCII characters.
    """
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise InvalidImageError(f"Could not open '{os.fspath(path)}': {exc.strerror}") from exc
    return load_image_from_bytes(data, os.fspath(path))


def encode_image(image: np.ndarray, fmt: str = "PNG", jpeg_quality: int = 95) -> bytes:
    """Encode an image as PNG or JPG bytes.

    PNG is lossless. JPG quality runs from 1 to 100; 95 is visually lossless
    for documents while staying much smaller than PNG.
    """
    ensure_valid_image(image)
    fmt = fmt.upper()
    if fmt in ("JPG", "JPEG"):
        extension = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(jpeg_quality, 1, 100))]
    elif fmt == "PNG":
        extension = ".png"
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        raise ValueError(f"Unsupported export format: {fmt!r}")
    ok, encoded = cv2.imencode(extension, image, params)
    if not ok:
        raise ScannerError("The image could not be encoded.")
    return encoded.tobytes()


def decode_image_unchanged(data: bytes) -> np.ndarray:
    """Decode bytes keeping the stored channel count (grayscale stays 2-D)."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise InvalidImageError("Stored image data is corrupted.")
    return image


# --------------------------------------------------------------------------- #
# Resizing and conversion
# --------------------------------------------------------------------------- #
def resize_keep_aspect(
    image: np.ndarray, *, height: int | None = None, max_side: int | None = None
) -> tuple[np.ndarray, float]:
    """Resize while preserving the aspect ratio.

    Pass exactly one of:
        height:   resize so the image is exactly this tall (may enlarge).
        max_side: shrink so the longer side is at most this long (never enlarges).

    Returns:
        ``(resized_image, scale)`` where ``scale = new_size / old_size``.
        To map a point from the resized image back to the original, *divide*
        its coordinates by ``scale``.

    Interpolation: ``INTER_AREA`` when shrinking (averages source pixels, so no
    aliasing) and ``INTER_CUBIC`` when enlarging.
    """
    if (height is None) == (max_side is None):
        raise ValueError("Pass exactly one of 'height' or 'max_side'.")
    ensure_valid_image(image)

    old_h, old_w = image.shape[:2]
    scale = height / old_h if height is not None else min(1.0, max_side / max(old_h, old_w))
    if scale == 1.0:
        return image.copy(), 1.0

    new_w = max(1, round(old_w * scale))
    new_h = max(1, round(old_h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation), scale


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB for display (grayscale is returned as is)."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_pil(image: np.ndarray) -> Image.Image:
    """Convert a BGR or grayscale array to a Pillow image."""
    ensure_valid_image(image)
    return Image.fromarray(to_rgb(image))
