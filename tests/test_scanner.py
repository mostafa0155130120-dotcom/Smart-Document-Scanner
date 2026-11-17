import itertools

import numpy as np
import pytest

from scanner import (
    DetectionParams,
    DocumentNotFoundError,
    ScannerError,
    detect_document,
    draw_detection,
    four_point_transform,
    full_image_detection,
    order_points,
    scan_document,
)
from tests.helpers import page, scene


def test_order_points_every_permutation():
    expected = np.float32([[10, 10], [200, 20], [210, 300], [5, 280]])
    for permutation in itertools.permutations(range(4)):
        shuffled = expected[list(permutation)]
        assert np.allclose(order_points(shuffled), expected)


def test_order_points_diamond_is_consistent():
    diamond = np.float32([[100, 0], [200, 100], [100, 200], [0, 100]])
    ordered = order_points(diamond[[2, 0, 3, 1]])
    assert np.allclose(ordered[0], [100, 0]) or np.allclose(ordered[0], [0, 100])
    areas = ordered[:, 0] * np.roll(ordered[:, 1], -1) - np.roll(ordered[:, 0], -1) * ordered[:, 1]
    assert areas.sum() > 0


def test_order_points_needs_four_points():
    with pytest.raises(ValueError):
        order_points(np.zeros((3, 2)))


def test_detects_corners_of_synthetic_document():
    photo, truth = scene()
    detection = detect_document(photo)
    assert detection.corners.shape == (4, 2)
    assert np.abs(detection.corners - truth).max() < 12


def test_detects_rotated_and_large_documents():
    for kind in ("rotated", "large"):
        photo, truth = scene(kind)
        detection = detect_document(photo)
        error = np.abs(order_points(detection.corners) - order_points(truth)).max()
        assert error < 15, f"{kind}: corner error {error:.1f}px"


def test_blank_image_raises_with_helpful_message():
    blank = np.full((600, 800, 3), 128, np.uint8)
    with pytest.raises(DocumentNotFoundError, match="No document outline"):
        detect_document(blank)


def test_failure_carries_edge_map_for_debugging():
    blank = np.full((600, 800, 3), 128, np.uint8)
    try:
        detect_document(blank)
    except DocumentNotFoundError as error:
        assert error.edges is not None and error.edges.ndim == 2
    else:
        raise AssertionError("expected DocumentNotFoundError")


def test_min_area_setting_can_reject_a_small_document():
    photo, _ = scene()
    with pytest.raises(DocumentNotFoundError):
        detect_document(photo, DetectionParams(min_area_ratio=0.95))


def test_invalid_detection_params():
    for kwargs in (
        {"blur_kernel": 4},
        {"canny_low": 200, "canny_high": 100},
        {"min_area_ratio": 1.5},
    ):
        with pytest.raises(ValueError):
            DetectionParams(**kwargs)


def test_full_image_detection_uses_photo_bounds():
    photo, _ = scene()
    detection = full_image_detection(photo)
    height, width = photo.shape[:2]
    assert np.allclose(detection.corners, [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])


def test_perspective_transform_recovers_the_flat_page():
    import cv2

    photo, truth = scene()
    flat = four_point_transform(photo, truth)
    reference = page()
    flat_small = cv2.GaussianBlur(cv2.resize(flat, (85, 110), interpolation=cv2.INTER_AREA), (5, 5), 0)
    ref_small = cv2.GaussianBlur(cv2.resize(reference, (85, 110), interpolation=cv2.INTER_AREA), (5, 5), 0)
    error = np.abs(flat_small.astype(int) - ref_small.astype(int)).mean()
    assert error < 12, f"mean abs difference {error:.1f}"


def test_output_size_matches_longest_sides():
    photo, truth = scene()
    flat = four_point_transform(photo, truth)
    tl, tr, br, bl = truth
    expected_w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    expected_h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    assert abs(flat.shape[1] - expected_w) <= 2 and abs(flat.shape[0] - expected_h) <= 2


def test_transform_accepts_unordered_corners_and_caps_size():
    photo, truth = scene()
    flat = four_point_transform(photo, truth[[2, 0, 3, 1]], max_side=500)
    assert max(flat.shape[:2]) == 500


def test_degenerate_quad_is_rejected():
    photo, _ = scene()
    tiny = np.float32([[10, 10], [14, 10], [14, 14], [10, 14]])
    with pytest.raises(ScannerError):
        four_point_transform(photo, tiny)


def test_scan_document_end_to_end_and_overlay():
    photo, _ = scene()
    result = scan_document(photo)
    assert result.warped.ndim == 3 and min(result.warped.shape[:2]) > 400
    overlay = draw_detection(photo, result.detection)
    assert overlay.shape == photo.shape and not np.array_equal(overlay, photo)
    assert np.array_equal(photo, scene()[0])


def test_scan_document_with_full_image_fallback():
    blank = np.full((600, 800, 3), 128, np.uint8)
    result = scan_document(blank, use_full_image=True)
    assert result.detection.method == "full-image"
    assert result.warped.shape[:2] == (600, 800)
