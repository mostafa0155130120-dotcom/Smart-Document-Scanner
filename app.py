"""app.py — Streamlit interface for the Smart Document Scanner.

Run with:  streamlit run app.py

This file only builds the UI and wires user choices to the computer-vision
modules; all image logic lives in ``scanner.py`` and ``image_processing.py``.

How Streamlit works (important for beginners): every time the user touches a
widget, Streamlit re-runs this whole script from top to bottom. Values that
must survive a re-run live in ``st.session_state``; slow pure functions are
wrapped in ``st.cache_data`` so they are not recomputed needlessly.
"""
from __future__ import annotations

import hashlib
import logging
import re
import uuid
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from image_processing import ScanMode, ScanSettings, process_scan
from pdf_generator import PAGE_SIZE_OPTIONS, PdfGenerationError, generate_pdf
from scanner import (
    DetectionParams,
    DocumentNotFoundError,
    ScanResult,
    draw_detection,
    scan_document,
)
from utils import (
    MAX_FILE_SIZE_MB,
    InvalidImageError,
    ScannerError,
    decode_image_unchanged,
    encode_image,
    load_image_from_bytes,
    resize_keep_aspect,
    to_rgb,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
LOGO_PATH = ROOT / "assets" / "logo.png"
SAMPLE_PATH = ROOT / "samples" / "sample_document.jpg"

MAX_PDF_PAGES = 20
DISPLAY_MAX_SIDE = 1400  # images are shrunk for the browser; exports stay full size
THUMBNAIL_SIDE = 160
SOURCES = ("Upload", "Camera", "Sample")

MODE_HELP = {
    ScanMode.ORIGINAL: "The straightened photo with its colours untouched.",
    ScanMode.ENHANCED: "Removes shadows and turns the paper white while keeping colour.",
    ScanMode.GRAYSCALE: "Shadow-free gray. Keeps shading in stamps and photographs.",
    ScanMode.BLACK_WHITE: "Pure black ink on white. Brightness moves the threshold.",
    ScanMode.ADAPTIVE: "Black and white decided locally. Best for uneven light.",
}

DETECTION_DEFAULTS = {
    "use_full_image": False,
    "blur_kernel": 5,
    "canny_low": 50,
    "canny_high": 150,
    "min_area_pct": 15,
    "show_debug": False,
}
CONTROL_DEFAULTS = {
    "mode": ScanMode.ENHANCED.value,
    "brightness": 0,
    "contrast": 0,
    "sharpness": 0.0,
    "quarter_turns": 0,
    "fine_rotation": 0.0,
    "adaptive_auto": True,
    "adaptive_block": 31,
    "adaptive_c": 10,
}
EXPORT_DEFAULTS = {"export_format": "PNG", "jpg_quality": 95, "pdf_page_size": PAGE_SIZE_OPTIONS[0]}

CSS = """
<style>
.block-container { padding-top: 1.6rem; max-width: 1400px; }
h1 { letter-spacing: -0.02em; margin-bottom: 0.1rem; }
[data-testid="stSidebar"] h2 { font-size: 1.05rem; }
</style>
"""


# --------------------------------------------------------------------------- #
# Cached computer-vision work
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=4)
def cached_load(data: bytes, filename: str | None) -> np.ndarray:
    """Decode and validate the uploaded bytes (cached by the bytes themselves)."""
    return load_image_from_bytes(data, filename)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_scan(
    data: bytes, filename: str | None, params: DetectionParams, use_full_image: bool
) -> ScanResult:
    """Detect the document and flatten it. Only re-runs when the photo or detection settings change."""
    return scan_document(cached_load(data, filename), params, use_full_image=use_full_image)


@st.cache_data(show_spinner="Building PDF…", max_entries=3)
def cached_pdf(page_ids: tuple[str, ...], page_size: str, _png_pages: list[bytes]) -> bytes:
    """Build the PDF. ``_png_pages`` starts with an underscore so Streamlit does not hash it;
    ``page_ids`` identifies the content instead."""
    return generate_pdf([decode_image_unchanged(png) for png in _png_pages], page_size=page_size)


# --------------------------------------------------------------------------- #
# Session state and callbacks
# --------------------------------------------------------------------------- #
def init_state() -> None:
    """Give every widget key a value (``setdefault`` never overwrites)."""
    for defaults in (DETECTION_DEFAULTS, CONTROL_DEFAULTS, EXPORT_DEFAULTS):
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)
    st.session_state.setdefault("pages", [])
    st.session_state.setdefault("image_id", None)


def apply_control_defaults(mode: ScanMode) -> None:
    for key, value in CONTROL_DEFAULTS.items():
        st.session_state[key] = value
    st.session_state["mode"] = mode.value


def reset_to_original() -> None:
    """Back to the plain straightened photo: Original Color and no adjustments."""
    apply_control_defaults(ScanMode.ORIGINAL)


def reset_detection() -> None:
    for key, value in DETECTION_DEFAULTS.items():
        st.session_state[key] = value


def use_whole_photo() -> None:
    st.session_state["use_full_image"] = True


def rotate_quarter(step: int) -> None:
    st.session_state["quarter_turns"] = (st.session_state["quarter_turns"] + step) % 4


def move_page(index: int, delta: int) -> None:
    pages = st.session_state["pages"]
    target = index + delta
    if 0 <= target < len(pages):
        pages[index], pages[target] = pages[target], pages[index]


def remove_page(page_id: str) -> None:
    st.session_state["pages"] = [p for p in st.session_state["pages"] if p["id"] != page_id]


def clear_pages() -> None:
    st.session_state["pages"] = []


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def for_display(image: np.ndarray) -> np.ndarray:
    """Shrink for the browser and convert BGR -> RGB."""
    small, _ = resize_keep_aspect(image, max_side=DISPLAY_MAX_SIDE)
    return to_rgb(small)


def safe_stem(filename: str | None) -> str:
    stem = Path(filename).stem if filename else "scan"
    return re.sub(r"[^\w\-]+", "_", stem).strip("_") or "scan"


def read_input() -> tuple[bytes, str | None] | None:
    """Show the chosen input widget and return ``(bytes, filename)`` or ``None``."""
    source = st.radio("Image source", SOURCES, horizontal=True, key="source")
    if source == "Upload":
        uploaded = st.file_uploader(
            "Upload a photo of a document",
            type=["jpg", "jpeg", "png"],
            help=f"JPG or PNG, up to {MAX_FILE_SIZE_MB} MB.",
        )
        return (uploaded.getvalue(), uploaded.name) if uploaded else None
    if source == "Camera":
        shot = st.camera_input("Take a photo of the document")
        return (shot.getvalue(), None) if shot else None  # camera frames are JPEG; no filename to check
    if SAMPLE_PATH.exists():
        return SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name
    st.warning("The sample image is missing from the samples folder.")
    return None


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar_detection() -> DetectionParams:
    with st.sidebar:
        st.header("Detection")
        st.checkbox("Use the whole photo (skip detection)", key="use_full_image")
        with st.expander("Tune detection"):
            st.slider("Blur strength", 3, 15, step=2, key="blur_kernel",
                      help="Smooths noise before looking for edges. Raise it for grainy photos.")
            st.slider("Edge sensitivity: low threshold", 0, 255, key="canny_low",
                      help="Edges weaker than this are dropped.")
            st.slider("Edge sensitivity: high threshold", 1, 400, key="canny_high",
                      help="Edges stronger than this are always kept. Lower both for low-contrast photos.")
            st.slider("Smallest accepted document (% of photo)", 5, 90, key="min_area_pct",
                      help="Lower it if the page is small in the frame.")
            st.button("Reset detection settings", on_click=reset_detection)
        st.checkbox("Show detection debug view", key="show_debug")

    try:
        return DetectionParams(
            blur_kernel=int(st.session_state["blur_kernel"]),
            canny_low=int(st.session_state["canny_low"]),
            canny_high=int(st.session_state["canny_high"]),
            min_area_ratio=st.session_state["min_area_pct"] / 100.0,
        )
    except ValueError as error:
        st.sidebar.error(str(error))
        return DetectionParams()


def sidebar_scan_settings() -> ScanSettings:
    with st.sidebar:
        st.header("Scan")
        st.radio("Scanning mode", [m.value for m in ScanMode], key="mode")
        mode = ScanMode(st.session_state["mode"])
        st.caption(MODE_HELP[mode])

        st.slider("Brightness", -100, 100, key="brightness")
        st.slider("Contrast", -80, 100, key="contrast", disabled=mode in (ScanMode.BLACK_WHITE, ScanMode.ADAPTIVE))
        st.slider("Sharpness", 0.0, 3.0, step=0.1, key="sharpness")

        left, right = st.columns(2)
        left.button("Rotate left", on_click=rotate_quarter, args=(-1,))
        right.button("Rotate right", on_click=rotate_quarter, args=(1,))
        st.slider("Straighten (degrees)", -15.0, 15.0, step=0.5, key="fine_rotation")

        with st.expander("Adaptive threshold options"):
            st.caption("Used only in Adaptive Threshold mode.")
            st.checkbox("Choose neighbourhood size automatically", key="adaptive_auto")
            st.slider("Neighbourhood size (px)", 3, 151, step=2, key="adaptive_block",
                      disabled=st.session_state["adaptive_auto"])
            st.slider("Speckle filter (C)", 0, 30, key="adaptive_c",
                      help="Higher = cleaner white background, but faint ink may vanish.")

        st.button("Reset to original image", on_click=reset_to_original)

    return ScanSettings(
        mode=mode,
        brightness=int(st.session_state["brightness"]),
        contrast=int(st.session_state["contrast"]),
        sharpness=float(st.session_state["sharpness"]),
        quarter_turns=int(st.session_state["quarter_turns"]),
        fine_rotation=float(st.session_state["fine_rotation"]),
        adaptive_block_size=0 if st.session_state["adaptive_auto"] else int(st.session_state["adaptive_block"]),
        adaptive_c=int(st.session_state["adaptive_c"]),
    )


# --------------------------------------------------------------------------- #
# Main-area sections
# --------------------------------------------------------------------------- #
def render_header(has_image: bool) -> None:
    logo_col, title_col = st.columns([1, 9], vertical_alignment="center")
    if LOGO_PATH.exists():
        logo_col.image(str(LOGO_PATH), width=64)
    with title_col:
        st.title("Smart Document Scanner")
        st.caption(
            "Turn a photo of a page into a flat, readable scan. "
            "It uses classic OpenCV image processing, not an AI model."
        )
    with st.expander("First time here? Four steps", expanded=not has_image):
        st.markdown(
            "1. **Choose a source.** Upload a photo, take one with your camera, or try the sample.\n"
            "2. **Check the detection.** The scan appears next to your photo. If the corners are wrong, "
            "turn on *Show detection debug view* and tune *Detection* in the sidebar.\n"
            "3. **Pick a look.** Choose a scanning mode, then adjust brightness, contrast, sharpness and rotation.\n"
            "4. **Export.** Download a PNG or JPG, or add pages and download one PDF.\n\n"
            "For the best result, photograph the page flat on a darker surface, in even light, "
            "with all four corners in view. The camera needs HTTPS or `localhost` and your permission."
        )


def render_failure(image: np.ndarray, error: DocumentNotFoundError) -> None:
    st.error(str(error))
    left, right = st.columns(2, gap="large")
    left.subheader("Your photo")
    left.image(for_display(image), width="stretch")
    if error.edges is not None:
        right.subheader("What the detector saw")
        right.image(error.edges, width="stretch")
        right.caption("White lines are the edges it found. A closed four-sided outline is needed.")
    st.markdown(
        "**Try this:** lower *Smallest accepted document* or the edge thresholds under *Tune detection*, "
        "photograph on a darker surface, or scan the whole photo as it is."
    )
    again, whole = st.columns(2)
    again.button("Reset detection settings", on_click=reset_detection, key="retry_reset")
    whole.button("Use the whole photo", on_click=use_whole_photo, key="retry_whole")


def render_comparison(image: np.ndarray, result: ScanResult, final: np.ndarray, show_debug: bool) -> None:
    detection = result.detection
    original_col, scan_col = st.columns(2, gap="large")
    with original_col:
        st.subheader("Original")
        shown = draw_detection(image, detection) if show_debug else image
        st.image(for_display(shown), width="stretch")
        if show_debug:
            st.caption(
                "Orange: candidate contours. Green: chosen outline. "
                "Red dots: corners (TL, TR, BR, BL)."
            )
            if detection.edges is not None:
                with st.expander("Edge map used for detection"):
                    st.image(detection.edges, width="stretch")
    with scan_col:
        st.subheader("Scanned")
        st.image(for_display(final), width="stretch")
        st.caption(f"{final.shape[1]} × {final.shape[0]} px")


def render_export(final: np.ndarray, stem: str) -> None:
    st.divider()
    st.subheader("Export")
    image_col, pdf_col = st.columns(2, gap="large")

    with image_col:
        st.markdown("**Image**")
        st.radio("Format", ["PNG", "JPG"], horizontal=True, key="export_format")
        fmt = st.session_state["export_format"]
        st.slider("JPG quality", 60, 100, key="jpg_quality", disabled=fmt != "JPG",
                  help="95 is visually lossless for documents.")
        st.download_button(
            f"Download {fmt}",
            data=encode_image(final, fmt, int(st.session_state["jpg_quality"])),
            file_name=f"{stem}_scan.{fmt.lower()}",
            mime="image/png" if fmt == "PNG" else "image/jpeg",
        )

    with pdf_col:
        st.markdown("**PDF**")
        st.selectbox("Page size", PAGE_SIZE_OPTIONS, key="pdf_page_size")
        pages = st.session_state["pages"]
        full = len(pages) >= MAX_PDF_PAGES
        if st.button("Add this scan to the PDF", disabled=full):
            thumbnail, _ = resize_keep_aspect(final, max_side=THUMBNAIL_SIDE)
            pages.append({"id": uuid.uuid4().hex, "png": encode_image(final, "PNG"), "thumb": to_rgb(thumbnail)})
            st.toast(f"Added page {len(pages)}")
        if full:
            st.caption(f"The PDF is limited to {MAX_PDF_PAGES} pages.")

    render_pdf_pages()


def render_pdf_pages() -> None:
    pages = st.session_state["pages"]
    if not pages:
        st.caption("Pages you add appear here, ready to download as one PDF.")
        return

    st.markdown(f"**PDF pages ({len(pages)})**")
    for index, page in enumerate(pages):
        thumb, label, up, down, remove = st.columns([2, 5, 1, 1, 1], vertical_alignment="center")
        thumb.image(page["thumb"], width=80)
        label.write(f"Page {index + 1}")
        up.button("↑", key=f"up_{page['id']}", on_click=move_page, args=(index, -1),
                  disabled=index == 0, help="Move up")
        down.button("↓", key=f"down_{page['id']}", on_click=move_page, args=(index, 1),
                    disabled=index == len(pages) - 1, help="Move down")
        remove.button("✕", key=f"remove_{page['id']}", on_click=remove_page, args=(page["id"],),
                      help="Remove page")

    try:
        pdf_bytes = cached_pdf(
            tuple(p["id"] for p in pages), st.session_state["pdf_page_size"], [p["png"] for p in pages]
        )
    except ScannerError as error:
        st.error(str(error))
        return
    download, clear = st.columns([1, 1])
    download.download_button("Download PDF", data=pdf_bytes, file_name="scan.pdf",
                             mime="application/pdf", type="primary")
    clear.button("Remove all pages", on_click=clear_pages)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    icon = Image.open(LOGO_PATH) if LOGO_PATH.exists() else "📄"
    st.set_page_config(page_title="Smart Document Scanner", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    render_header(has_image=bool(st.session_state.get("image_id")))
    payload = read_input()
    if payload is None:
        st.info("Choose a source above to begin. The sample document is a quick way to see how it works.")
        sidebar_detection()
        return

    data, filename = payload
    digest = hashlib.sha1(data).hexdigest()
    if st.session_state["image_id"] != digest:  # a new photo: start from the default look
        apply_control_defaults(ScanMode.ENHANCED)
        st.session_state["image_id"] = digest

    params = sidebar_detection()
    settings = sidebar_scan_settings()

    try:
        image = cached_load(data, filename)
    except InvalidImageError as error:
        st.error(str(error))
        return

    try:
        with st.spinner("Looking for the document…"):
            result = cached_scan(data, filename, params, bool(st.session_state["use_full_image"]))
    except DocumentNotFoundError as error:
        render_failure(image, error)
        return
    except ScannerError as error:
        st.error(str(error))
        return

    if result.detection.method == "full-image":
        st.info("Using the whole photo as the document. Turn off *Use the whole photo* to detect the page.")
    else:
        st.success(f"Document found (method: {result.detection.method}). Fine-tune the look in the sidebar.")

    with st.spinner("Processing…"):
        final = process_scan(result.warped, settings)

    render_comparison(image, result, final, bool(st.session_state["show_debug"]))
    render_export(final, safe_stem(filename))


main()
