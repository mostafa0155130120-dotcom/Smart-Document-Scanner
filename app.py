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
DISPLAY_MAX_SIDE = 1400
THUMBNAIL_SIDE = 160
SOURCES = ("Upload", "Camera", "Sample")

MODE_HELP_KEYS = {
    ScanMode.ORIGINAL: "mode_help_original",
    ScanMode.ENHANCED: "mode_help_enhanced",
    ScanMode.GRAYSCALE: "mode_help_gray",
    ScanMode.BLACK_WHITE: "mode_help_bw",
    ScanMode.ADAPTIVE: "mode_help_adaptive",
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

TEXT = {
    "en": {
        "language": "Language", "detection": "Detection", "scan": "Scan", "export": "Export",
        "source": "Image source", "upload": "Upload", "camera": "Camera", "sample": "Sample",
        "upload_prompt": "Upload a photo of a document", "upload_help": "JPG or PNG, up to {size} MB.",
        "sample_missing": "The sample image is missing from the samples folder.",
        "whole_photo": "Use the whole photo (skip detection)", "tune_detection": "Tune detection",
        "blur": "Blur strength", "low": "Edge sensitivity: low threshold",
        "high": "Edge sensitivity: high threshold", "min_area": "Smallest document size (% of photo)",
        "reset_detection": "Reset detection settings", "debug": "Show detection debug view",
        "brightness": "Brightness", "contrast": "Contrast", "sharpness": "Sharpness",
        "rotate_left": "Rotate left", "rotate_right": "Rotate right", "straighten": "Straighten (degrees)",
        "adaptive_options": "Adaptive threshold options", "adaptive_only": "Used only in Adaptive Threshold mode.",
        "adaptive_auto": "Choose neighbourhood size automatically", "neighbourhood": "Neighbourhood size (px)",
        "speckle": "Speckle filter (C)", "reset_image": "Reset to original image",
        "title": "Smart Document Scanner", "subtitle": "Turn a photo of a page into a clear, flat scan.",
        "tagline": "Private by design · Classic computer vision · Ready to export",
        "steps_title": "How to scan", "step1": "Choose Upload, Camera, or Sample to add a page.",
        "step2": "Check the detected corners and adjust the scan in the sidebar.",
        "step3": "Download the finished image or collect pages into one PDF.",
        "choose_source": "Choose a source above to get started. Try the bundled sample for a quick preview.",
        "original": "Original photo", "scanned": "Scanned result", "detector_saw": "What the detector found",
        "edge_explanation": "The outline it found is shown here.",
        "try_again": "Try lowering the minimum document size or edge thresholds, use a darker background, or scan the whole photo.",
        "mode_original": "Original Color", "mode_enhanced": "Enhanced Color", "mode_gray": "Grayscale",
        "mode_bw": "Black & White", "mode_adaptive": "Adaptive Threshold",
        "mode_help_original": "Straightened photo with its original colours.",
        "mode_help_enhanced": "Reduces shadows while keeping colour details.",
        "mode_help_gray": "Clean grayscale while preserving shading.",
        "mode_help_bw": "Pure black ink on white paper.",
        "mode_help_adaptive": "Local black-and-white conversion for uneven lighting.",
        "debug_help": "Orange: candidate outlines · Green: selected page · Red: detected corners.",
        "edge_map": "Detection edge map", "found": "Document detected successfully.",
        "whole_used": "Scanning the full photo. Turn off the option in Detection to find page edges.",
        "not_found": "No clear page outline was found.", "image": "Image file", "format": "Format",
        "jpg_quality": "JPG quality", "quality_help": "Higher quality keeps text edges crisp.",
        "pdf": "PDF document", "page_size": "Page size", "add_pdf": "Add scan to PDF",
        "pdf_limit": "The PDF can contain up to {count} pages.", "pages": "Pages in PDF ({count})",
        "no_pages": "Add a scan above to start a PDF.", "page": "Page {number}",
        "move_up": "Move up", "move_down": "Move down", "remove": "Remove page",
        "download_pdf": "Download PDF", "clear_pages": "Clear PDF pages", "added": "Added page {number}.",
        "file_limit": "Maximum file size: {size} MB · JPG or PNG", "error": "Something went wrong: {error}",
        "processing": "Processing image…", "detecting": "Finding the page…", "image_size": "{width} × {height} px",
    },
    "ar": {
        "language": "اللغة", "detection": "اكتشاف الصفحة", "scan": "تحسين الصورة", "export": "تصدير",
        "source": "مصدر الصورة", "upload": "رفع صورة", "camera": "الكاميرا", "sample": "تجربة نموذج",
        "upload_prompt": "ارفع صورة للمستند", "upload_help": "الأنواع المدعومة JPG وPNG، حتى {size} ميجابايت.",
        "sample_missing": "الصورة التجريبية غير موجودة في مجلد العينات.",
        "whole_photo": "استخدم الصورة كاملة بدون اكتشاف الحواف", "tune_detection": "ضبط اكتشاف الحواف",
        "blur": "تنعيم الصورة", "low": "حساسية الحواف: الحد الأدنى",
        "high": "حساسية الحواف: الحد الأعلى", "min_area": "أصغر حجم للورقة (% من الصورة)",
        "reset_detection": "إعادة ضبط إعدادات الاكتشاف", "debug": "عرض تفاصيل اكتشاف الحواف",
        "brightness": "السطوع", "contrast": "التباين", "sharpness": "الوضوح",
        "rotate_left": "تدوير لليسار", "rotate_right": "تدوير لليمين", "straighten": "استقامة الصورة (درجة)",
        "adaptive_options": "إعدادات الأبيض والأسود الذكي", "adaptive_only": "يُستخدم هذا الخيار مع الأبيض والأسود الذكي فقط.",
        "adaptive_auto": "اختيار حجم المنطقة تلقائيًا", "neighbourhood": "حجم المنطقة (بكسل)",
        "speckle": "تنقية النقاط الصغيرة", "reset_image": "إعادة الصورة لأصلها",
        "title": "الماسح الذكي للمستندات", "subtitle": "حوّل صورة الورقة إلى نسخة واضحة ومستقيمة.",
        "tagline": "خصوصية أولًا · معالجة صور تقليدية · جاهز للتصدير",
        "steps_title": "طريقة الاستخدام", "step1": "اختر رفع صورة أو الكاميرا أو جرّب الصورة النموذجية.",
        "step2": "راجع زوايا الورقة واضبط شكل المسح من القائمة الجانبية.",
        "step3": "نزّل الصورة أو اجمع عدة صفحات في ملف PDF واحد.",
        "choose_source": "ابدأ باختيار مصدر للصورة. جرّب الصورة النموذجية لمعاينة سريعة.",
        "original": "الصورة الأصلية", "scanned": "النتيجة بعد المسح", "detector_saw": "الحواف التي تم اكتشافها",
        "edge_explanation": "الإطار الذي عثر عليه البرنامج ظاهر هنا.",
        "try_again": "جرّب تقليل أصغر حجم للورقة أو حساسية الحواف، وضع الورقة على خلفية أغمق، أو استخدم الصورة كاملة.",
        "mode_original": "الألوان الأصلية", "mode_enhanced": "ألوان محسّنة", "mode_gray": "تدرّج رمادي",
        "mode_bw": "أبيض وأسود", "mode_adaptive": "أبيض وأسود ذكي",
        "mode_help_original": "الصورة بعد تعديل الاستقامة مع الاحتفاظ بألوانها.",
        "mode_help_enhanced": "تقليل الظلال مع الحفاظ على تفاصيل الألوان.",
        "mode_help_gray": "نسخة رمادية واضحة مع الحفاظ على التدرجات.",
        "mode_help_bw": "حبر أسود واضح على خلفية بيضاء.",
        "mode_help_adaptive": "تحويل محلي للأبيض والأسود يناسب الإضاءة غير المتساوية.",
        "debug_help": "البرتقالي: حواف مرشحة · الأخضر: حدود الورقة · الأحمر: الزوايا المكتشفة.",
        "edge_map": "خريطة الحواف المكتشفة", "found": "تم اكتشاف المستند بنجاح.",
        "whole_used": "تتم معالجة الصورة كاملة. أوقف هذا الخيار من قائمة اكتشاف الصفحة للبحث عن حدود الورقة.",
        "not_found": "لم يتم العثور على حدود واضحة للورقة.", "image": "ملف الصورة", "format": "الصيغة",
        "jpg_quality": "جودة JPG", "quality_help": "الجودة الأعلى تحافظ على وضوح أطراف النص.",
        "pdf": "ملف PDF", "page_size": "حجم الصفحة", "add_pdf": "أضف الصورة إلى PDF",
        "pdf_limit": "يمكن أن يحتوي ملف PDF على {count} صفحة كحد أقصى.", "pages": "صفحات ملف PDF ({count})",
        "no_pages": "أضف صورة ممسوحة أعلاه لبدء ملف PDF.", "page": "صفحة {number}",
        "move_up": "نقل لأعلى", "move_down": "نقل لأسفل", "remove": "حذف الصفحة",
        "download_pdf": "تنزيل PDF", "clear_pages": "حذف صفحات PDF", "added": "تمت إضافة الصفحة {number}.",
        "file_limit": "الحد الأقصى {size} ميجابايت · JPG أو PNG", "error": "حدث خطأ: {error}",
        "processing": "جارٍ تحسين الصورة…", "detecting": "جارٍ البحث عن حدود الورقة…", "image_size": "{width} × {height} بكسل",
    },
}

MODE_KEYS = {
    ScanMode.ORIGINAL: "mode_original", ScanMode.ENHANCED: "mode_enhanced",
    ScanMode.GRAYSCALE: "mode_gray", ScanMode.BLACK_WHITE: "mode_bw",
    ScanMode.ADAPTIVE: "mode_adaptive",
}


def is_arabic() -> bool:
    return st.session_state.get("ui_language", "العربية") == "العربية"


def tr(key: str, **values: object) -> str:
    language = "ar" if is_arabic() else "en"
    return TEXT[language][key].format(**values)


def language_css() -> str:
    direction = "rtl" if is_arabic() else "ltr"
    row_direction = "row-reverse" if is_arabic() else "row"
    return f"""
    <style>
    :root {{ --ink:#142b35; --muted:#647982; --teal:#087f8c; --mint:#c9f0e6; --paper:#f4f8f7; }}
    html, body, .stApp {{ direction:ltr; }}
    [data-testid="stMain"], [data-testid="stSidebar"] {{ direction:{direction}; }}
    .stApp {{ background: radial-gradient(ellipse at 8% 0%, #e3f4f0 0, transparent 30%), var(--paper); color:var(--ink); font-family:'DM Sans','Noto Sans Arabic',Tahoma,Arial,sans-serif; }}
    [data-testid="stHeader"] {{ background:transparent; }}
    .block-container {{ max-width:1440px; padding-top:1.25rem; padding-bottom:3rem; }}
    h1,h2,h3 {{ color:var(--ink); letter-spacing:-.025em; }}
    h1 {{ margin:0 0 .3rem 0; font-size:2.15rem; }}
    [data-testid="stSidebar"] {{ background:#fff; border-left:1px solid #e4edeb; border-right:1px solid #e4edeb; }}
    [data-testid="stSidebar"] h2 {{ font-size:1rem; }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap:.65rem; }}
    [data-testid="stRadio"] > div {{ gap:.55rem; }}
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button {{ border-radius:12px; min-height:2.65rem; font-weight:600; border-color:#d9e6e3; }}
    [data-testid="stDownloadButton"] button[kind="primary"], [data-testid="stButton"] button[kind="primary"] {{ background:var(--teal); border-color:var(--teal); color:white; }}
    [data-testid="stSlider"] {{ padding-top:.1rem; }}
    [data-testid="stFileUploader"] section {{ border:1.5px dashed #9bbdb9; border-radius:18px; background:#fbfefd; }}
    [data-testid="stImage"] img {{ border-radius:16px; }}
    [data-testid="stExpander"] {{ border-radius:14px; border-color:#e2ece9; background:rgba(255,255,255,.7); }}
    .hero {{ position:relative; overflow:hidden; border-radius:24px; padding:2rem 2.15rem; margin:.3rem 0 1.3rem; color:white; background:linear-gradient(115deg,#073c49 0%,#087f8c 57%,#25a08f 100%); box-shadow:0 18px 38px rgba(8,86,96,.16); }}
    .hero:after {{ content:' '; position:absolute; width:290px; height:290px; border-radius:50%; border:1px solid rgba(255,255,255,.15); top:-155px; {('left' if is_arabic() else 'right')}:8%; box-shadow:0 0 0 35px rgba(255,255,255,.035),0 0 0 75px rgba(255,255,255,.025); }}
    .hero .eyebrow {{ display:inline-block; padding:.35rem .7rem; border:1px solid rgba(255,255,255,.25); border-radius:99px; color:#d4f7ed; font-size:.82rem; margin-bottom:.8rem; }}
    .hero h1 {{ color:white; font-size:2.2rem; margin:0; }}
    .hero p {{ color:#e1f4f1; margin:.55rem 0 0; font-size:1.05rem; }}
    .hero .tags {{ margin-top:1.2rem; color:#c2e9e1; font-size:.82rem; }}
    .section-title {{ margin:.45rem 0 .65rem; color:#203943; font-weight:700; font-size:1.08rem; }}
    [data-testid="stHorizontalBlock"] {{ flex-direction:{row_direction}; }}
    [dir="rtl"] {{ text-align:right; }}
    @media (max-width: 768px) {{
        .block-container {{ padding:1rem .85rem 2rem; }}
        .hero {{ padding:1.35rem 1.2rem; border-radius:18px; }}
        .hero h1 {{ font-size:1.75rem; }}
        .hero p {{ font-size:1rem; }}
        [data-testid="stHorizontalBlock"] {{ flex-direction:column !important; gap:.85rem; }}
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
            flex:1 1 100% !important;
            width:100% !important;
            min-width:100% !important;
        }}
        [data-testid="stRadio"] > div {{ flex-wrap:wrap; gap:.4rem .65rem; }}
    }}
    </style>
    """


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


def init_state() -> None:
    """Give every widget key a value (``setdefault`` never overwrites)."""
    for defaults in (DETECTION_DEFAULTS, CONTROL_DEFAULTS, EXPORT_DEFAULTS):
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)
    st.session_state.setdefault("pages", [])
    st.session_state.setdefault("image_id", None)
    st.session_state.setdefault("ui_language", "العربية")


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


def for_display(image: np.ndarray) -> np.ndarray:
    """Shrink for the browser and convert BGR -> RGB."""
    small, _ = resize_keep_aspect(image, max_side=DISPLAY_MAX_SIDE)
    return to_rgb(small)


def safe_stem(filename: str | None) -> str:
    stem = Path(filename).stem if filename else "scan"
    return re.sub(r"[^\w\-]+", "_", stem).strip("_") or "scan"


def read_input() -> tuple[bytes, str | None] | None:
    """Show the chosen input widget and return ``(bytes, filename)`` or ``None``."""
    source = st.radio(
        tr("source"),
        SOURCES,
        horizontal=True,
        key="source",
        format_func=lambda value: tr(value.lower()),
    )
    if source == "Upload":
        uploaded = st.file_uploader(
            tr("upload_prompt"),
            type=["jpg", "jpeg", "png"],
            help=tr("upload_help", size=MAX_FILE_SIZE_MB),
        )
        return (uploaded.getvalue(), uploaded.name) if uploaded else None
    if source == "Camera":
        shot = st.camera_input("Take a photo of the document" if not is_arabic() else tr("upload_prompt"))
        return (shot.getvalue(), None) if shot else None
    if SAMPLE_PATH.exists():
        return SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name
    st.warning(tr("sample_missing"))
    return None


def sidebar_detection() -> DetectionParams:
    with st.sidebar:
        st.subheader(tr("detection"))
        st.checkbox(tr("whole_photo"), key="use_full_image")
        with st.expander(tr("tune_detection")):
            st.slider(tr("blur"), 3, 15, step=2, key="blur_kernel",
                      help="Smooths noise before looking for edges. Raise it for grainy photos.")
            st.slider(tr("low"), 0, 255, key="canny_low",
                      help="Edges weaker than this are dropped.")
            st.slider(tr("high"), 1, 400, key="canny_high",
                      help="Edges stronger than this are always kept. Lower both for low-contrast photos.")
            st.slider(tr("min_area"), 5, 90, key="min_area_pct",
                      help="Lower it if the page is small in the frame.")
            st.button(tr("reset_detection"), on_click=reset_detection)
        st.checkbox(tr("debug"), key="show_debug")

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
        st.subheader(tr("scan"))
        st.radio("Scanning mode" if not is_arabic() else "نمط المسح", [m.value for m in ScanMode],
                 key="mode", format_func=lambda value: tr(MODE_KEYS[ScanMode(value)]))
        mode = ScanMode(st.session_state["mode"])
        st.caption(tr(MODE_HELP_KEYS[mode]))

        st.slider(tr("brightness"), -100, 100, key="brightness")
        st.slider(tr("contrast"), -80, 100, key="contrast", disabled=mode in (ScanMode.BLACK_WHITE, ScanMode.ADAPTIVE))
        st.slider(tr("sharpness"), 0.0, 3.0, step=0.1, key="sharpness")

        left, right = st.columns(2)
        left.button(tr("rotate_left"), on_click=rotate_quarter, args=(-1,))
        right.button(tr("rotate_right"), on_click=rotate_quarter, args=(1,))
        st.slider(tr("straighten"), -15.0, 15.0, step=0.5, key="fine_rotation")

        with st.expander(tr("adaptive_options")):
            st.caption(tr("adaptive_only"))
            st.checkbox(tr("adaptive_auto"), key="adaptive_auto")
            st.slider(tr("neighbourhood"), 3, 151, step=2, key="adaptive_block",
                      disabled=st.session_state["adaptive_auto"])
            st.slider(tr("speckle"), 0, 30, key="adaptive_c",
                      help="Higher = cleaner white background, but faint ink may vanish.")

        st.button(tr("reset_image"), on_click=reset_to_original)

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


def render_header(has_image: bool) -> None:
    eyebrow = "ماسح مستنداتك، بشكل أذكى" if is_arabic() else "SMART DOCUMENT WORKSPACE"
    st.markdown(
        f"<div class='hero'><span class='eyebrow'>{eyebrow}</span><h1>{tr('title')}</h1>"
        f"<p>{tr('subtitle')}</p><div class='tags'>{tr('tagline')}</div></div>",
        unsafe_allow_html=True,
    )
    with st.expander(tr("steps_title"), expanded=not has_image):
        first, second, third = st.columns(3)
        first.markdown(f"**01**  {tr('step1')}")
        second.markdown(f"**02**  {tr('step2')}")
        third.markdown(f"**03**  {tr('step3')}")


def render_failure(image: np.ndarray, error: DocumentNotFoundError) -> None:
    st.error(f"{tr('not_found')} {error}")
    left, right = st.columns(2, gap="large")
    left.subheader(tr("original"))
    left.image(for_display(image), width="stretch")
    if error.edges is not None:
        right.subheader(tr("detector_saw"))
        right.image(error.edges, width="stretch")
        right.caption(tr("edge_explanation"))
    st.info(tr("try_again"))
    again, whole = st.columns(2)
    again.button(tr("reset_detection"), on_click=reset_detection, key="retry_reset")
    whole.button(tr("whole_photo"), on_click=use_whole_photo, key="retry_whole")


def render_comparison(image: np.ndarray, result: ScanResult, final: np.ndarray, show_debug: bool) -> None:
    detection = result.detection
    original_col, scan_col = st.columns(2, gap="large")
    with original_col:
        st.subheader(tr("original"))
        shown = draw_detection(image, detection) if show_debug else image
        st.image(for_display(shown), width="stretch")
        if show_debug:
            st.caption(
                tr("debug_help")
            )
            if detection.edges is not None:
                with st.expander(tr("edge_map")):
                    st.image(detection.edges, width="stretch")
    with scan_col:
        st.subheader(tr("scanned"))
        st.image(for_display(final), width="stretch")
        st.caption(f"{final.shape[1]} × {final.shape[0]} px")


def render_export(final: np.ndarray, stem: str) -> None:
    st.divider()
    st.subheader(tr("export"))
    image_col, pdf_col = st.columns(2, gap="large")

    with image_col:
        st.markdown(f"**{tr('image')}**")
        st.radio(tr("format"), ["PNG", "JPG"], horizontal=True, key="export_format")
        fmt = st.session_state["export_format"]
        st.slider(tr("jpg_quality"), 60, 100, key="jpg_quality", disabled=fmt != "JPG",
                  help=tr("quality_help"))
        st.download_button(
            (f"تنزيل {fmt}" if is_arabic() else f"Download {fmt}"),
            data=encode_image(final, fmt, int(st.session_state["jpg_quality"])),
            file_name=f"{stem}_scan.{fmt.lower()}",
            mime="image/png" if fmt == "PNG" else "image/jpeg",
        )

    with pdf_col:
        st.markdown(f"**{tr('pdf')}**")
        st.selectbox(tr("page_size"), PAGE_SIZE_OPTIONS, key="pdf_page_size")
        pages = st.session_state["pages"]
        full = len(pages) >= MAX_PDF_PAGES
        if st.button(tr("add_pdf"), disabled=full):
            thumbnail, _ = resize_keep_aspect(final, max_side=THUMBNAIL_SIDE)
            pages.append({"id": uuid.uuid4().hex, "png": encode_image(final, "PNG"), "thumb": to_rgb(thumbnail)})
            st.toast(tr("added", number=len(pages)))
        if full:
            st.caption(tr("pdf_limit", count=MAX_PDF_PAGES))

    render_pdf_pages()


def render_pdf_pages() -> None:
    pages = st.session_state["pages"]
    if not pages:
        st.caption(tr("no_pages"))
        return

    st.markdown(f"**{tr('pages', count=len(pages))}**")
    for index, page in enumerate(pages):
        thumb, label, up, down, remove = st.columns([2, 5, 1, 1, 1], vertical_alignment="center")
        thumb.image(page["thumb"], width=80)
        label.write(tr("page", number=index + 1))
        up.button("↑", key=f"up_{page['id']}", on_click=move_page, args=(index, -1),
                  disabled=index == 0, help=tr("move_up"))
        down.button("↓", key=f"down_{page['id']}", on_click=move_page, args=(index, 1),
                    disabled=index == len(pages) - 1, help=tr("move_down"))
        remove.button("✕", key=f"remove_{page['id']}", on_click=remove_page, args=(page["id"],),
                      help=tr("remove"))

    try:
        pdf_bytes = cached_pdf(
            tuple(p["id"] for p in pages), st.session_state["pdf_page_size"], [p["png"] for p in pages]
        )
    except ScannerError as error:
        st.error(str(error))
        return
    download, clear = st.columns([1, 1])
    download.download_button(tr("download_pdf"), data=pdf_bytes, file_name="scan.pdf",
                             mime="application/pdf", type="primary")
    clear.button(tr("clear_pages"), on_click=clear_pages)


def main() -> None:
    icon = Image.open(LOGO_PATH) if LOGO_PATH.exists() else "📄"
    st.set_page_config(page_title="Smart Document Scanner", page_icon=icon, layout="wide")
    init_state()
    with st.sidebar:
        st.radio(tr("language"), ["\u0627\u0644\u0639\u0631\u0628\u064a\u0629", "English"], horizontal=True, key="ui_language")
        st.divider()
    st.markdown(language_css(), unsafe_allow_html=True)

    render_header(has_image=bool(st.session_state.get("image_id")))
    payload = read_input()
    if payload is None:
        st.info(tr("choose_source"))
        sidebar_detection()
        return

    data, filename = payload
    digest = hashlib.sha1(data).hexdigest()
    if st.session_state["image_id"] != digest:
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
        st.info(tr("whole_used"))
    else:
        st.success(f"{tr('found')} ({result.detection.method})")

    with st.spinner("Processing…"):
        final = process_scan(result.warped, settings)

    render_comparison(image, result, final, bool(st.session_state["show_debug"]))
    render_export(final, safe_stem(filename))


main()
