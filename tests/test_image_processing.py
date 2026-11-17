import numpy as np
import pytest

from image_processing import (
    BINARY_MODES,
    ScanMode,
    ScanSettings,
    adjust_brightness_contrast,
    apply_adaptive_threshold,
    apply_black_white,
    enhance_color,
    normalize_illumination,
    process_scan,
    rotate_image,
    sharpen,
)
from tests.helpers import page


def test_every_mode_returns_valid_uint8_image_of_same_size():
    source = page()
    for mode in ScanMode:
        result = process_scan(source, ScanSettings(mode=mode))
        assert result.dtype == np.uint8, mode
        assert result.shape[:2] == source.shape[:2], mode


def test_binary_modes_contain_only_black_and_white():
    source = page()
    for mode in BINARY_MODES:
        values = set(np.unique(process_scan(source, ScanSettings(mode=mode))).tolist())
        assert values <= {0, 255}, mode


def test_black_and_white_keeps_ink_and_whitens_paper():
    result = apply_black_white(page())
    assert (result == 255).mean() > 0.8
    assert (result == 0).mean() > 0.005


def test_illumination_normalisation_flattens_uneven_light():
    gradient = np.tile(np.linspace(90, 230, 400, dtype=np.uint8), (300, 1))
    flat = normalize_illumination(gradient)
    assert gradient.std() > 35
    assert flat.std() < 5 and flat.mean() > 245


def test_enhanced_color_brightens_paper_and_keeps_colour():
    source = page()
    result = enhance_color(source)
    assert result.mean() > source.mean()
    red_stamp = result[1010 - 40:1010 + 40, 700 - 40:700 + 40].reshape(-1, 3)
    assert red_stamp[:, 2].max() > red_stamp[:, 0].min() + 80


def test_brightness_and_contrast_move_in_the_right_direction():
    source = page()
    assert adjust_brightness_contrast(source, brightness=40).mean() > source.mean()
    assert adjust_brightness_contrast(source, brightness=-40).mean() < source.mean()
    assert adjust_brightness_contrast(source, contrast=60).std() > source.std()
    assert adjust_brightness_contrast(source, contrast=-60).std() < source.std()
    assert np.array_equal(adjust_brightness_contrast(source), source)


def test_brightness_bias_changes_black_white_result():
    source = page()
    lighter = process_scan(source, ScanSettings(mode=ScanMode.BLACK_WHITE, brightness=80))
    darker = process_scan(source, ScanSettings(mode=ScanMode.BLACK_WHITE, brightness=-80))
    assert (lighter == 255).mean() >= (darker == 255).mean()


def test_sharpness_increases_edge_energy_and_zero_is_identity():
    import cv2

    blurred = cv2.GaussianBlur(page(), (0, 0), 2)
    sharper = sharpen(blurred, 1.5)
    assert cv2.Laplacian(sharper, cv2.CV_64F).var() > cv2.Laplacian(blurred, cv2.CV_64F).var()
    assert np.array_equal(sharpen(blurred, 0), blurred)


def test_adaptive_threshold_handles_a_shadow_gradient():
    import cv2

    shaded = (page().astype(np.float32) * np.linspace(0.35, 1.0, page().shape[1])[None, :, None]).astype(np.uint8)
    result = apply_adaptive_threshold(shaded)
    left, right = result[:, :200], result[:, -200:]
    assert (left == 255).mean() > 0.8 and (right == 255).mean() > 0.8


def test_rotation_by_quarter_turns():
    image = np.zeros((100, 200, 3), np.uint8)
    assert rotate_image(image, 90).shape[:2] == (200, 100)
    assert rotate_image(image, 180).shape[:2] == (100, 200)
    assert rotate_image(image, 270).shape[:2] == (200, 100)
    assert np.array_equal(rotate_image(image, 0), image)
    assert np.array_equal(rotate_image(image, 360), image)


def test_rotation_direction_is_clockwise():
    image = np.zeros((10, 20), np.uint8)
    image[0, 0] = 255
    rotated = rotate_image(image, 90)
    assert rotated[0, 9] == 255


def test_fine_rotation_enlarges_canvas_with_white_border():
    image = np.zeros((100, 200, 3), np.uint8)
    rotated = rotate_image(image, 10)
    assert rotated.shape[0] > 100 and rotated.shape[1] > 200
    assert rotated[0, 0].tolist() == [255, 255, 255]


def test_process_scan_applies_rotation_last():
    result = process_scan(page(), ScanSettings(mode=ScanMode.ORIGINAL, quarter_turns=1))
    assert result.shape[:2] == (page().shape[1], page().shape[0])


def test_invalid_settings_and_inputs():
    with pytest.raises(ValueError):
        ScanSettings(adaptive_block_size=10)
    with pytest.raises(Exception):
        process_scan(np.zeros((10, 10, 4), np.uint8))
    with pytest.raises(Exception):
        process_scan(np.zeros((0, 0, 3), np.uint8))
