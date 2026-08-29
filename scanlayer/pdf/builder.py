"""
Final PDF construction.

Page sized to background image, background drawn with adaptive compression
(JPEG or PNG), invisible text layer overlaid word by word.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from scanlayer import config
from scanlayer.ocr.engine import Word
from scanlayer.pdf.fonts import resolve_font
from scanlayer.utils.errors import PDFBuildError
from scanlayer.utils.logger import get_logger, log_warning, stage_timer

log = get_logger(__name__)

POINTS_PER_INCH = 72.0
INVISIBLE_RENDER_MODE = 3

_MIN_FONT_SIZE_PT = 1.0
_MAX_FONT_SIZE_PT = 400.0
_MIN_HORIZ_SCALE_PCT = 1.0
_MAX_HORIZ_SCALE_PCT = 1000.0


def _is_mostly_grayscale(image: Image.Image, threshold: float = None) -> bool:
    """Return True if the image is visually near-grayscale (low chrominance)."""
    if image.mode == "L":
        return True
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.asarray(image)
    # Approximate saturation: max - min across RGB channels.
    sat = arr.max(axis=2).astype(np.int16) - arr.min(axis=2).astype(np.int16)
    threshold = threshold if threshold is not None else config.PDF_GRAYSCALE_THRESHOLD
    fraction_colored = float((sat > 25).mean())  # sat > ~10% = colored pixel
    log.debug(
        f"Compression detector: colored pixel fraction = {fraction_colored:.4f} "
        f"(grayscale threshold = {threshold})"
    )
    return fraction_colored < threshold


def _is_mostly_binary(image: Image.Image) -> bool:
    """Detect if an image is near-binary (black text on white background).

    For near-binary content, PNG is lighter and lossless vs JPEG.
    """
    if image.mode != "L":
        image = image.convert("L")
    arr = np.asarray(image, dtype=np.float32)
    hist, _ = np.histogram(arr, bins=8, range=(0, 256))
    total = arr.size
    if total == 0:
        return False
    extreme_fraction = (hist[0] + hist[-1]) / total
    is_binary = extreme_fraction > 0.80
    log.debug(
        f"Compression detector: extreme pixel fraction = {extreme_fraction:.4f} "
        f"-> binary={'yes' if is_binary else 'no'}"
    )
    return is_binary


def _as_jpeg_reader(image: Image.Image, quality: int) -> ImageReader:
    """Encodes the image as JPEG in a memory buffer and returns an
    ImageReader for ReportLab."""
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer, format="JPEG", quality=quality, optimize=True,
    )
    buffer.seek(0)
    return ImageReader(buffer)


def _as_png_reader(image: Image.Image) -> ImageReader:
    """Encode image as PNG (lossless) in a memory buffer.

    Converts to L mode if near-grayscale to save ~60% of size.
    """
    if _is_mostly_grayscale(image):
        image = image.convert("L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    return ImageReader(buffer)


def _select_background_reader(
    image: Image.Image, jpeg_quality: int
) -> ImageReader:
    """Chooses the best compression strategy for the PDF background.

    1. If PDF_ADAPTIVE_COMPRESSION and image is near-binary -> PNG (L mode if grayscale).
    2. Otherwise -> JPEG at the configured quality.
    """
    if config.PDF_ADAPTIVE_COMPRESSION and _is_mostly_binary(image):
        log.info("Background compression: PNG (near-binary image detected)")
        return _as_png_reader(image)
    log.info(f"Background compression: JPEG q={jpeg_quality}")
    return _as_jpeg_reader(image, jpeg_quality)


def _font_size_for_height(height_px: float, px_to_pt: float) -> float:
    """Calculate font size so a character vertically occupies height_px pixels."""
    size = height_px * px_to_pt * 0.85
    return min(max(size, _MIN_FONT_SIZE_PT), _MAX_FONT_SIZE_PT)


def _draw_word(
    c: canvas.Canvas,
    word: Word,
    px_to_pt: float,
    page_height_pt: float,
    font_name: str,
) -> None:
    """Draw an invisible word on the PDF canvas with horizontal stretching."""
    if not word.width or not word.height:
        return

    font_size = _font_size_for_height(word.height, px_to_pt)
    natural_width = c.stringWidth(word.text, font_name, font_size)
    target_width_pt = word.width * px_to_pt

    text_obj = c.beginText()
    text_obj.setTextRenderMode(INVISIBLE_RENDER_MODE)
    text_obj.setFont(font_name, font_size)

    pdf_x = word.x * px_to_pt
    pdf_y = page_height_pt - (word.y + word.height) * px_to_pt
    text_obj.setTextOrigin(pdf_x, pdf_y)

    if natural_width > 0 and target_width_pt > 0:
        h_scale = 100.0 * target_width_pt / natural_width
        h_scale = min(max(h_scale, _MIN_HORIZ_SCALE_PCT), _MAX_HORIZ_SCALE_PCT)
        text_obj.setHorizScale(h_scale)

    text_obj.textOut(word.text)
    c.drawText(text_obj)


def _validate_page_size(page_width_pt: float, page_height_pt: float, page_num: int = 1) -> None:
    """Validate PDF page size (1-14400pt per spec)."""
    if page_width_pt < 1 or page_height_pt < 1:
        raise PDFBuildError(
            f"Invalid PDF page {page_num}: {page_width_pt}x{page_height_pt}pt"
        )
    if page_width_pt > 14400 or page_height_pt > 14400:
        log_warning(
            log,
            f"PDF page {page_num} > 14400pt (PDF spec limit): "
            f"{page_width_pt}x{page_height_pt}pt, may be rejected by "
            f"some viewers."
        )


def _draw_page(
    c: canvas.Canvas,
    background: Image.Image,
    words: list[Word],
    dpi: int,
    jpeg_quality: int,
    lang: Optional[str],
    page_num: int = 1,
) -> None:
    """Draw one full page (background + text layer) onto an open canvas."""
    font_name = resolve_font(lang or config.DEFAULT_OCR_LANG)
    px_to_pt = POINTS_PER_INCH / dpi
    page_width_pt = background.width * px_to_pt
    page_height_pt = background.height * px_to_pt
    _validate_page_size(page_width_pt, page_height_pt, page_num)

    c.setPageSize((page_width_pt, page_height_pt))

    bg_reader = _select_background_reader(background, jpeg_quality)
    c.drawImage(
        bg_reader, 0, 0,
        width=page_width_pt, height=page_height_pt,
        preserveAspectRatio=False,
    )

    skipped = 0
    for word in words:
        try:
            _draw_word(c, word, px_to_pt, page_height_pt, font_name)
        except Exception as exc:
            # A single failing word must not break the entire PDF.
            skipped += 1
            log_warning(
                log,
                f"Page {page_num}: word skipped in text layer: "
                f"{word.text!r} ({type(exc).__name__}: {exc})"
            )

    c.showPage()
    log.info(
        f"Page {page_num}: {len(words)} words in text layer"
        + (f", {skipped} skipped" if skipped else "")
    )


def build_searchable_pdf(
    background: Image.Image,
    words: list[Word],
    output_path: str,
    dpi: int,
    jpeg_quality: int = None,
    metadata: Optional[dict] = None,
    lang: Optional[str] = None,
) -> int:
    """Build a single-page searchable PDF. Returns size in bytes."""
    jpeg_quality = jpeg_quality if jpeg_quality is not None else config.PDF_JPEG_QUALITY
    metadata = {**config.PDF_METADATA, **(metadata or {})}

    try:
        with stage_timer(log, "PDF build"):
            c = canvas.Canvas(output_path)
            c.setTitle(metadata.get("title", ""))
            c.setAuthor(metadata.get("author", ""))
            c.setSubject(metadata.get("subject", ""))
            c.setCreator(metadata.get("creator", ""))
            if "keywords" in metadata and metadata["keywords"]:
                c.setKeywords(metadata["keywords"])

            _draw_page(c, background, words, dpi, jpeg_quality, lang)
            c.save()
    except PDFBuildError:
        raise
    except OSError as exc:
        raise PDFBuildError(
            f"Could not write PDF to {output_path}: {exc}"
        ) from exc
    except Exception as exc:
        raise PDFBuildError(
            f"Unexpected failure while building the PDF: {type(exc).__name__}: {exc}"
        ) from exc

    size_bytes = os.path.getsize(output_path)
    log.info(f"PDF generated: {size_bytes / 1024:.1f} KB, 1 page")
    return size_bytes


@dataclass
class PageInput:
    """One page's worth of content for `build_searchable_pdf_multipage`."""
    background: Image.Image
    words: list[Word]
    dpi: int
    lang: Optional[str] = None


def build_searchable_pdf_multipage(
    pages: list[PageInput],
    output_path: str,
    jpeg_quality: int = None,
    metadata: Optional[dict] = None,
) -> int:
    """Build a multi-page searchable PDF from several OCR'd pages.

    Returns size in bytes. Raises PDFBuildError if pages is empty.
    """
    if not pages:
        raise PDFBuildError("build_searchable_pdf_multipage() called with 0 pages.")

    jpeg_quality = jpeg_quality if jpeg_quality is not None else config.PDF_JPEG_QUALITY
    metadata = {**config.PDF_METADATA, **(metadata or {})}

    try:
        with stage_timer(log, f"PDF build ({len(pages)} pages)"):
            c = canvas.Canvas(output_path)
            c.setTitle(metadata.get("title", ""))
            c.setAuthor(metadata.get("author", ""))
            c.setSubject(metadata.get("subject", ""))
            c.setCreator(metadata.get("creator", ""))
            if "keywords" in metadata and metadata["keywords"]:
                c.setKeywords(metadata["keywords"])

            for i, page in enumerate(pages, start=1):
                _draw_page(
                    c, page.background, page.words, page.dpi,
                    jpeg_quality, page.lang, page_num=i,
                )
            c.save()
    except PDFBuildError:
        raise
    except OSError as exc:
        raise PDFBuildError(
            f"Could not write PDF to {output_path}: {exc}"
        ) from exc
    except Exception as exc:
        raise PDFBuildError(
            f"Unexpected failure while building the multi-page PDF: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    size_bytes = os.path.getsize(output_path)
    log.info(f"PDF generated: {size_bytes / 1024:.1f} KB, {len(pages)} pages")
    return size_bytes
