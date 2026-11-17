"""pdf_generator.py — turn scanned pages into a (multi-page) PDF with ReportLab.

Quality strategy
----------------
* Colour and grayscale pages are embedded as JPEG at quality 95 (visually
  lossless for documents, far smaller than raw pixels).
* Pure black-and-white pages are embedded losslessly (PNG/Flate), which is
  both smaller and sharper than JPEG for two-colour images.

Page sizes
----------
* ``"A4"``: every page is A4. Orientation follows the image (landscape scans
  get a landscape page) and the image is scaled to fit, centred, without
  distortion.
* ``"Original Size"``: each page has the image's own proportions. Pixels are
  converted to PDF points (1 pt = 1/72 inch) at ``dpi`` dots per inch.
"""
from __future__ import annotations

import io
import logging
from collections.abc import Sequence

import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from utils import InvalidImageError, ScannerError, ensure_valid_image, to_pil

logger = logging.getLogger(__name__)

PAGE_SIZE_A4 = "A4"
PAGE_SIZE_ORIGINAL = "Original Size"
PAGE_SIZE_OPTIONS: tuple[str, ...] = (PAGE_SIZE_A4, PAGE_SIZE_ORIGINAL)
PDF_MAX_POINTS = 14_400.0


class PdfGenerationError(ScannerError):
    """The PDF could not be created."""


def _is_binary(image: np.ndarray) -> bool:
    """True for a 2-D image containing only 0 and 255."""
    return image.ndim == 2 and not bool(np.any((image != 0) & (image != 255)))


def _encode_for_pdf(image: np.ndarray, jpeg_quality: int) -> io.BytesIO:
    pil_image = to_pil(image)
    buffer = io.BytesIO()
    if _is_binary(image):
        pil_image.save(buffer, format="PNG", optimize=True)
    else:
        pil_image.save(buffer, format="JPEG", quality=jpeg_quality, subsampling=0, optimize=True)
    buffer.seek(0)
    return buffer


def _page_geometry(
    image: np.ndarray, page_size: str, margin_pt: float, dpi: int
) -> tuple[float, float, float, float, float, float]:
    """Return ``(page_w, page_h, x, y, draw_w, draw_h)`` in points."""
    img_h, img_w = image.shape[:2]
    if page_size == PAGE_SIZE_A4:
        page_w, page_h = A4
        if img_w > img_h:
            page_w, page_h = page_h, page_w
        fit = min((page_w - 2 * margin_pt) / img_w, (page_h - 2 * margin_pt) / img_h)
        draw_w, draw_h = img_w * fit, img_h * fit
        return page_w, page_h, (page_w - draw_w) / 2, (page_h - draw_h) / 2, draw_w, draw_h

    page_w, page_h = img_w * 72.0 / dpi, img_h * 72.0 / dpi
    limit = min(1.0, PDF_MAX_POINTS / max(page_w, page_h))
    page_w, page_h = page_w * limit, page_h * limit
    return page_w, page_h, 0.0, 0.0, page_w, page_h


def generate_pdf(
    images: Sequence[np.ndarray],
    page_size: str = PAGE_SIZE_A4,
    *,
    margin_mm: float = 0.0,
    dpi: int = 150,
    jpeg_quality: int = 95,
    title: str = "Scanned document",
) -> bytes:
    """Create a PDF with one page per image and return its bytes.

    Raises:
        PdfGenerationError: for an empty list, an unknown page size or a
                            failure inside ReportLab.
    """
    if not images:
        raise PdfGenerationError("Add at least one page before creating a PDF.")
    if page_size not in PAGE_SIZE_OPTIONS:
        raise PdfGenerationError(f"Unknown page size {page_size!r}. Use one of {PAGE_SIZE_OPTIONS}.")
    if dpi <= 0:
        raise PdfGenerationError("DPI must be positive.")

    output = io.BytesIO()
    try:
        pdf = canvas.Canvas(output, pagesize=A4, pageCompression=1)
        pdf.setTitle(title)
        pdf.setCreator("Smart Document Scanner")
        for number, image in enumerate(images, start=1):
            ensure_valid_image(image)
            page_w, page_h, x, y, draw_w, draw_h = _page_geometry(image, page_size, margin_mm * mm, dpi)
            pdf.setPageSize((page_w, page_h))
            reader = ImageReader(_encode_for_pdf(image, jpeg_quality))
            pdf.drawImage(reader, x, y, width=draw_w, height=draw_h)
            pdf.showPage()
            logger.debug("PDF page %d written (%.0f x %.0f pt)", number, page_w, page_h)
        pdf.save()
    except InvalidImageError:
        raise
    except Exception as exc:
        raise PdfGenerationError(f"The PDF could not be created: {exc}") from exc
    return output.getvalue()
