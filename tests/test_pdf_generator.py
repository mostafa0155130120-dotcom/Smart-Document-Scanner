import re

import numpy as np
import pytest

from image_processing import ScanMode, ScanSettings, process_scan
from pdf_generator import PAGE_SIZE_A4, PAGE_SIZE_ORIGINAL, PdfGenerationError, generate_pdf
from tests.helpers import page


def _media_boxes(pdf: bytes):
    return [tuple(float(v) for v in m) for m in re.findall(rb"/MediaBox \[ ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) \]", pdf)]


def _page_count(pdf: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page\b", pdf))


def test_single_page_a4_pdf_is_valid():
    pdf = generate_pdf([page()], PAGE_SIZE_A4)
    assert pdf.startswith(b"%PDF") and pdf.rstrip().endswith(b"%%EOF")
    assert _page_count(pdf) == 1
    (box,) = _media_boxes(pdf)
    assert abs(box[2] - 595.28) < 0.1 and abs(box[3] - 841.89) < 0.1


def test_multipage_pdf_mixes_colour_gray_and_binary():
    source = page()
    pages = [process_scan(source, ScanSettings(mode=mode)) for mode in ScanMode]
    pdf = generate_pdf(pages, PAGE_SIZE_A4)
    assert _page_count(pdf) == len(ScanMode)


def test_landscape_image_gets_landscape_a4_page():
    wide = np.full((400, 900, 3), 220, np.uint8)
    (box,) = _media_boxes(generate_pdf([wide], PAGE_SIZE_A4))
    assert box[2] > box[3]


def test_original_size_page_matches_image_proportions():
    image = np.full((600, 400, 3), 200, np.uint8)
    (box,) = _media_boxes(generate_pdf([image], PAGE_SIZE_ORIGINAL, dpi=100))
    assert abs(box[2] - 400 * 72 / 100) < 0.1 and abs(box[3] - 600 * 72 / 100) < 0.1


def test_binary_pages_are_smaller_than_jpeg_of_same_page():
    binary = process_scan(page(), ScanSettings(mode=ScanMode.BLACK_WHITE))
    assert len(generate_pdf([binary])) < len(generate_pdf([page()]))


def test_empty_list_and_bad_arguments_raise():
    with pytest.raises(PdfGenerationError):
        generate_pdf([])
    with pytest.raises(PdfGenerationError):
        generate_pdf([page()], page_size="Letter")
    with pytest.raises(PdfGenerationError):
        generate_pdf([page()], dpi=0)


def test_invalid_image_is_rejected():
    with pytest.raises(Exception):
        generate_pdf([np.zeros((10, 10, 4), np.uint8)])
