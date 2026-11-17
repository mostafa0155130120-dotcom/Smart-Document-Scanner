import cv2
import numpy as np
import pytest

from utils import (
    MAX_FILE_SIZE_BYTES,
    InvalidImageError,
    decode_image_unchanged,
    encode_image,
    load_image_from_bytes,
    load_image_from_path,
    resize_keep_aspect,
    to_rgb,
    validate_extension,
    validate_file_size,
)
from tests.helpers import scene


def _png_bytes(width=400, height=300, value=200):
    image = np.full((height, width, 3), value, np.uint8)
    return cv2.imencode(".png", image)[1].tobytes()


def test_load_valid_png_and_jpg():
    png = load_image_from_bytes(_png_bytes(), "a.png")
    assert png.shape == (300, 400, 3) and png.dtype == np.uint8

    photo, _ = scene()
    jpg = encode_image(photo, "JPG")
    assert load_image_from_bytes(jpg, "photo.JPG").shape == photo.shape


def test_load_from_path(tmp_path=None):
    import tempfile, os

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "scan é.png")
        with open(path, "wb") as handle:
            handle.write(_png_bytes())
        assert load_image_from_path(path).shape == (300, 400, 3)


def test_invalid_bytes_are_rejected():
    with pytest.raises(InvalidImageError):
        load_image_from_bytes(b"this is definitely not an image")


def test_empty_and_oversized_files_are_rejected():
    with pytest.raises(InvalidImageError):
        load_image_from_bytes(b"")
    with pytest.raises(InvalidImageError):
        validate_file_size(MAX_FILE_SIZE_BYTES + 1)


def test_extension_validation():
    assert validate_extension("scan.JPEG") == ".jpeg"
    for bad in ("notes.pdf", "image.gif", "noextension", None):
        with pytest.raises(InvalidImageError):
            validate_extension(bad)


def test_tiny_image_is_rejected():
    with pytest.raises(InvalidImageError):
        load_image_from_bytes(_png_bytes(width=50, height=50))


def test_gif_content_with_png_name_is_rejected():
    from PIL import Image
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (200, 200), "white").save(buffer, format="GIF")
    with pytest.raises(InvalidImageError):
        load_image_from_bytes(buffer.getvalue(), "fake.png")


def test_resize_keep_aspect_by_height():
    image = np.zeros((1000, 2000, 3), np.uint8)
    small, scale = resize_keep_aspect(image, height=500)
    assert small.shape[:2] == (500, 1000)
    assert scale == 0.5


def test_resize_max_side_never_enlarges():
    image = np.zeros((100, 200, 3), np.uint8)
    same, scale = resize_keep_aspect(image, max_side=1000)
    assert same.shape == image.shape and scale == 1.0
    shrunk, scale = resize_keep_aspect(np.zeros((400, 800, 3), np.uint8), max_side=200)
    assert shrunk.shape[:2] == (100, 200)


def test_resize_requires_exactly_one_target():
    image = np.zeros((100, 100, 3), np.uint8)
    with pytest.raises(ValueError):
        resize_keep_aspect(image)
    with pytest.raises(ValueError):
        resize_keep_aspect(image, height=50, max_side=50)


def test_encode_decode_roundtrip_keeps_grayscale_and_pixels():
    gray = (np.random.default_rng(0).random((120, 160)) * 255).astype(np.uint8)
    decoded = decode_image_unchanged(encode_image(gray, "PNG"))
    assert decoded.ndim == 2 and np.array_equal(decoded, gray)


def test_encode_rejects_unknown_format():
    with pytest.raises(ValueError):
        encode_image(np.zeros((10, 10, 3), np.uint8), "BMP")


def test_to_rgb_swaps_channels():
    bgr = np.zeros((2, 2, 3), np.uint8)
    bgr[..., 0] = 255
    assert to_rgb(bgr)[0, 0].tolist() == [0, 0, 255]
