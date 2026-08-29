"""
Library entry point for scanlayer.

For the CLI, see scanlayer.cli.
"""

from __future__ import annotations

import glob
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from scanlayer import config
from scanlayer.layout.columns import reorder_reading_order
from scanlayer.ocr.engine import extract_words
from scanlayer.ocr.export import FORMATS, export_words
from scanlayer.pdf.builder import build_searchable_pdf, build_searchable_pdf_multipage
from scanlayer.preprocessing.enhance import preprocess
from scanlayer.utils.debug_image import build_debug_image
from scanlayer.utils.errors import BlankPageDetectedError
from scanlayer.utils.logger import get_logger, log_success, log_warning
from scanlayer.utils.validators import (
    DependencyError,
    InputFileError,
    validate_all,
    validate_image_readable,
    validate_input_file,
    validate_output_path,
    validate_tesseract_environment,
)


@dataclass
class ConversionResult:
    """Summary of a successful conversion."""
    output_path: str
    words_count: int
    mean_confidence: float
    best_psm: Optional[int]
    early_exited: bool
    pdf_size_bytes: int
    elapsed_ms: float
    language_used: str
    output_format: str = "pdf"
    debug_image_path: Optional[str] = None
    column_count: int = 1


@dataclass
class BatchResult:
    """Summary of a convert_batch() run.

    Check .ok / .failed / .failures after the call. Never raises for
    per-file failures.
    """
    results: list  # list[ConversionResult]
    failures: list  # list[tuple[str, str]]

    @property
    def succeeded(self) -> int:
        return len(self.results)

    @property
    def failed(self) -> int:
        return len(self.failures)

    @property
    def ok(self) -> bool:
        return not self.failures


def _write_export(
    output_format: str,
    words: list,
    output_abs: str,
    mean_confidence: float,
    best_psm: Optional[int],
    language_used: str,
    image_width: int,
    image_height: int,
    source_name: str,
) -> int:
    """Serialize words into the requested non-PDF format. Returns file size in bytes."""
    content = export_words(
        output_format, words,
        mean_confidence=mean_confidence, best_psm=best_psm,
        language_used=language_used, image_width=image_width,
        image_height=image_height, source_name=source_name,
    )
    with open(output_abs, "w", encoding="utf-8") as f:
        f.write(content)
    return os.path.getsize(output_abs)


def _default_output_path(input_path: str, output_format: str) -> str:
    """Derive an output path next to the input file when -o/output_path
    is omitted: same directory, same stem, extension from output_format.
    """
    stem = os.path.splitext(os.path.basename(input_path))[0]
    directory = os.path.dirname(os.path.abspath(input_path))
    return os.path.join(directory, f"{stem}.{output_format}")


def _resolve_output_path(
    output_arg: Optional[str], input_path: str, is_batch: bool, output_format: str = "pdf"
) -> str:
    """Resolve output path for single or batch mode.

    output_arg=None -> write next to the input file (see
    _default_output_path); no directory is created in that case since
    the input's own directory already exists.
    """
    if output_arg is None:
        return _default_output_path(input_path, output_format)
    if is_batch or output_arg.endswith(("/", "\\")) or os.path.isdir(output_arg):
        os.makedirs(output_arg, exist_ok=True)
        stem = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(output_arg, f"{stem}.{output_format}")
    return output_arg


def _debug_image_path(output_abs: str) -> str:
    stem, _ext = os.path.splitext(output_abs)
    return f"{stem}_debug.png"


def _write_debug_image(
    output_abs: str,
    background,
    words: list,
    best_psm: Optional[int],
    mean_confidence: float,
    language_used: str,
    result,
) -> str:
    """Builds and saves the debug overlay image next to `output_abs`.
    Returns the path it was written to.
    """
    rotation_note = ""
    if result.orientation_correction_skipped:
        rotation_note = "(orientation correction disabled)"
    elif result.manual_rotation_applied is not None:
        rotation_note = f"manual rotation={result.manual_rotation_applied:.2f} deg"
    elif result.deskew_angle_applied:
        rotation_note = f"deskew={result.deskew_angle_applied:.2f} deg"

    debug_img = build_debug_image(
        background, words,
        best_psm=best_psm, mean_confidence=mean_confidence,
        language_used=language_used,
        exif_orientation=result.exif_orientation_applied,
        gross_rotation=result.gross_rotation_applied,
        rotation_note=rotation_note,
    )
    path = _debug_image_path(output_abs)
    debug_img.save(path)
    return path


def convert(
    input_path: str,
    output_path: Optional[str] = None,
    lang: Optional[str] = None,
    dpi: Optional[int] = None,
    jpeg_quality: Optional[int] = None,
    char_whitelist: Optional[str] = None,
    char_blacklist: Optional[str] = None,
    pdf_metadata: Optional[dict] = None,
    force: bool = False,
    orientation: "str | float | None" = None,
    output_format: str = "pdf",
    debug_image: bool = False,
) -> ConversionResult:
    """Convert an image to a searchable PDF or export raw OCR results.

    Raises ValidationError if prerequisites are not met.
    Raises BlankPageDetectedError if the image is near-blank and force=False.

    orientation: None=auto, "none"=disable, or float degrees (clockwise).
    output_format: "pdf", "txt", "json", "tsv", or "hocr".
    debug_image: if True, saves a _debug.png with word boxes overlay.
    output_path: if omitted (None), defaults to the input file's own
        directory/stem with the extension for output_format, e.g.
        "invoice.jpg" -> "invoice.pdf". Lets debug_image=True be used
        without having to name an output file.
    """
    if output_format not in FORMATS:
        raise ValueError(
            f"Unsupported output_format: {output_format!r} "
            f"(expected one of {FORMATS})"
        )
    if output_path is None:
        output_path = _default_output_path(input_path, output_format)
    log = get_logger("main")
    t0 = time.perf_counter()

    input_abs, output_abs = validate_all(input_path, output_path, output_format)

    dpi = dpi or config.DEFAULT_DPI
    with Image.open(input_abs) as img:
        image = img.copy()
        if "dpi" in img.info and img.info["dpi"][0] > 10:
            detected_dpi = int(img.info["dpi"][0])
            if dpi != detected_dpi:
                log.info(
                    f"DPI detected in image ({detected_dpi}), used "
                    f"instead of default/config value ({dpi})."
                )
            dpi = detected_dpi

    result = preprocess(image, dpi=dpi, orientation=orientation)

    for w in result.preprocessing_warnings:
        log_warning(log, w)

    if result.likely_blank and not force:
        raise BlankPageDetectedError(
            f"'{input_path}': source image appears blank/near-uniform "
            f"(grayscale std={result.source_std:.2f} < "
            f"{config.DESKEW_MIN_STD}), no {output_format} output "
            f"generated. Pass force=True (or --force on the CLI) to "
            f"generate the output anyway (background-only PDF, or an "
            f"empty result for the other formats)."
        )

    if result.likely_blank and force:
        log_warning(
            log,
            f"'{input_path}': --force used on a likely-blank page "
            f"(std={result.source_std:.2f}), generating {output_format} "
            f"output with no text/words (OCR skipped)."
        )
        words_for_output: list = []
        language_used = lang or config.DEFAULT_OCR_LANG

        if output_format == "pdf":
            output_size_bytes = build_searchable_pdf(
                result.background,
                words_for_output,
                output_abs,
                dpi=dpi,
                jpeg_quality=jpeg_quality,
                metadata=pdf_metadata,
                lang=language_used,
            )
        else:
            output_size_bytes = _write_export(
                output_format, words_for_output, output_abs,
                mean_confidence=0.0, best_psm=None,
                language_used=language_used,
                image_width=result.background.width,
                image_height=result.background.height,
                source_name=os.path.basename(input_path),
            )

        debug_image_path = None
        if debug_image:
            debug_image_path = _write_debug_image(
                output_abs, result.background, words_for_output,
                best_psm=None, mean_confidence=0.0,
                language_used=language_used, result=result,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info(
            f"{output_format} output generated: {output_abs} "
            f"({output_size_bytes / 1024:.1f} KB, {elapsed_ms:.0f} ms)"
        )
        return ConversionResult(
            output_path=output_abs,
            words_count=0,
            mean_confidence=0.0,
            best_psm=None,
            early_exited=False,
            pdf_size_bytes=output_size_bytes,
            elapsed_ms=elapsed_ms,
            language_used=language_used,
            output_format=output_format,
            debug_image_path=debug_image_path,
        )

    ocr_result = extract_words(
        result.ocr_image,
        result.ocr_scale,
        effective_dpi=result.effective_dpi,
        lang=lang,
        char_whitelist=char_whitelist,
        char_blacklist=char_blacklist,
    )

    ocr_result.words, column_count = reorder_reading_order(
        ocr_result.words, page_width=result.background.width,
    )

    mean_conf = (
        sum(w.confidence for w in ocr_result.words) / len(ocr_result.words)
        if ocr_result.words else 0.0
    )

    if output_format == "pdf":
        output_size_bytes = build_searchable_pdf(
            result.background,
            ocr_result.words,
            output_abs,
            dpi=dpi,
            jpeg_quality=jpeg_quality,
            metadata=pdf_metadata,
            lang=ocr_result.language_used,
        )
    else:
        output_size_bytes = _write_export(
            output_format, ocr_result.words, output_abs,
            mean_confidence=mean_conf, best_psm=ocr_result.best_psm,
            language_used=ocr_result.language_used,
            image_width=result.background.width,
            image_height=result.background.height,
            source_name=os.path.basename(input_path),
        )

    debug_image_path = None
    if debug_image:
        debug_image_path = _write_debug_image(
            output_abs, result.background, ocr_result.words,
            best_psm=ocr_result.best_psm, mean_confidence=mean_conf,
            language_used=ocr_result.language_used, result=result,
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if ocr_result.words:
        log_success(
            log,
            f"{len(ocr_result.words)} words detected "
            f"(mean confidence {mean_conf:.1f}%, "
            f"PSM {ocr_result.best_psm}"
            + (" [early-exit]" if ocr_result.early_exited else "")
            + (f", {column_count} columns" if column_count > 1 else "")
            + ")"
        )
    else:
        log_warning(log, "No words detected (blank or unreadable page)")
    log.info(
        f"{output_format} output generated: {output_abs} "
        f"({output_size_bytes / 1024:.1f} KB, {elapsed_ms:.0f} ms)"
    )

    return ConversionResult(
        output_path=output_abs,
        words_count=len(ocr_result.words),
        mean_confidence=mean_conf,
        best_psm=ocr_result.best_psm,
        early_exited=ocr_result.early_exited,
        pdf_size_bytes=output_size_bytes,
        elapsed_ms=elapsed_ms,
        language_used=ocr_result.language_used,
        output_format=output_format,
        debug_image_path=debug_image_path,
        column_count=column_count,
    )


def _expand_pdf_input(path: str, dpi: Optional[int], tmp_dir: str) -> list[str]:
    """Rasterize PDF pages to PNGs. Non-PDF inputs pass through unchanged.

    Requires poppler on PATH. Raises DependencyError if missing.
    """
    if os.path.splitext(path)[1].lower() != ".pdf":
        return [path]

    log = get_logger("main")
    try:
        from pdf2image import convert_from_path
        from pdf2image.exceptions import PDFInfoNotInstalledError
    except ImportError as exc:
        raise DependencyError(
            "PDF input requires the 'pdf2image' package: "
            "pip install pdf2image"
        ) from exc

    raster_dpi = dpi or config.DEFAULT_DPI
    try:
        page_images = convert_from_path(path, dpi=raster_dpi)
    except PDFInfoNotInstalledError as exc:
        raise DependencyError(
            "PDF input requires poppler-utils (the 'pdftoppm'/'pdfinfo' "
            "binaries) on PATH. 'pip install pdf2image' alone is not "
            "enough, poppler is a separate system package "
            "(apt install poppler-utils / brew install poppler / "
            "download poppler for Windows and add it to PATH)."
        ) from exc
    except Exception as exc:
        raise InputFileError(
            f"'{path}': could not read as a PDF ({type(exc).__name__}: {exc})"
        ) from exc

    if not page_images:
        raise InputFileError(f"'{path}': PDF has 0 pages, nothing to convert.")

    stem = os.path.splitext(os.path.basename(path))[0]
    out_paths = []
    for i, page_img in enumerate(page_images, start=1):
        page_path = os.path.join(tmp_dir, f"{stem}_p{i}.png")
        page_img.save(page_path, dpi=(raster_dpi, raster_dpi))
        out_paths.append(page_path)
    log.info(
        f"'{path}': rasterized {len(out_paths)} page(s) from PDF input "
        f"at {raster_dpi} DPI."
    )
    return out_paths


def _process_page_for_merge(
    input_path: str,
    lang: Optional[str],
    dpi: Optional[int],
    char_whitelist: Optional[str],
    char_blacklist: Optional[str],
    orientation: "str | float | None",
    force: bool,
):
    """Run preprocessing + OCR for one page of a --merge run.

    Returns (PageInput, stats_dict). Does not touch the output path.
    """
    from scanlayer.pdf.builder import PageInput

    log = get_logger("main")
    input_abs = validate_input_file(input_path)
    validate_image_readable(input_abs)
    validate_tesseract_environment()

    page_dpi = dpi or config.DEFAULT_DPI
    with Image.open(input_abs) as img:
        image = img.copy()
        if "dpi" in img.info and img.info["dpi"][0] > 10:
            page_dpi = int(img.info["dpi"][0])

    result = preprocess(image, dpi=page_dpi, orientation=orientation)
    for w in result.preprocessing_warnings:
        log_warning(log, w)

    if result.likely_blank and not force:
        raise BlankPageDetectedError(
            f"'{input_path}': source image appears blank/near-uniform "
            f"(grayscale std={result.source_std:.2f} < "
            f"{config.DESKEW_MIN_STD}), pass force=True (or --force) to "
            f"include it as a blank page in the merged PDF."
        )

    if result.likely_blank and force:
        log_warning(
            log,
            f"'{input_path}': --force used on a likely-blank page, "
            f"included in the merged PDF with no text layer.",
        )
        page = PageInput(
            background=result.background, words=[], dpi=page_dpi,
            lang=lang or config.DEFAULT_OCR_LANG,
        )
        return page, {"words": 0, "mean_conf": 0.0, "best_psm": None, "columns": 1}

    ocr_result = extract_words(
        result.ocr_image, result.ocr_scale, effective_dpi=result.effective_dpi,
        lang=lang, char_whitelist=char_whitelist, char_blacklist=char_blacklist,
    )
    ocr_result.words, column_count = reorder_reading_order(
        ocr_result.words, page_width=result.background.width,
    )
    mean_conf = (
        sum(w.confidence for w in ocr_result.words) / len(ocr_result.words)
        if ocr_result.words else 0.0
    )

    page = PageInput(
        background=result.background, words=ocr_result.words,
        dpi=page_dpi, lang=ocr_result.language_used,
    )
    stats = {
        "words": len(ocr_result.words), "mean_conf": mean_conf,
        "best_psm": ocr_result.best_psm, "columns": column_count,
    }
    return page, stats


def convert_merge(
    input_paths: list[str],
    output_path: str,
    lang: Optional[str] = None,
    dpi: Optional[int] = None,
    jpeg_quality: Optional[int] = None,
    char_whitelist: Optional[str] = None,
    char_blacklist: Optional[str] = None,
    pdf_metadata: Optional[dict] = None,
    force: bool = False,
    orientation: "str | float | None" = None,
) -> ConversionResult:
    """Combine several images into one multi-page searchable PDF.

    PDF-only. Raises BlankPageDetectedError if any page is blank and
    force=False.
    """
    if not input_paths:
        raise ValueError("convert_merge() called with no input paths.")


    log = get_logger("main")
    t0 = time.perf_counter()

    output_abs = validate_output_path(output_path, expected_ext=".pdf")

    pages = []
    total_words = 0
    conf_weighted_sum = 0.0
    conf_weight_n = 0
    max_columns = 1
    for input_path in input_paths:
        page, stats = _process_page_for_merge(
            input_path, lang, dpi, char_whitelist, char_blacklist, orientation, force,
        )
        pages.append(page)
        total_words += stats["words"]
        if stats["words"]:
            conf_weighted_sum += stats["mean_conf"] * stats["words"]
            conf_weight_n += stats["words"]
        max_columns = max(max_columns, stats["columns"])
        log.info(
            f"'{input_path}': {stats['words']} words "
            f"(mean confidence {stats['mean_conf']:.1f}%"
            + (f", {stats['columns']} columns" if stats["columns"] > 1 else "")
            + ")"
        )

    output_size_bytes = build_searchable_pdf_multipage(
        pages, output_abs, jpeg_quality=jpeg_quality, metadata=pdf_metadata,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    mean_conf = conf_weighted_sum / conf_weight_n if conf_weight_n else 0.0
    log_success(
        log,
        f"Merged {len(pages)} page(s) into {output_abs} "
        f"({total_words} words total, mean confidence {mean_conf:.1f}%)",
    )

    return ConversionResult(
        output_path=output_abs,
        words_count=total_words,
        mean_confidence=mean_conf,
        best_psm=None,  # not a single meaningful value across merged pages
        early_exited=False,
        pdf_size_bytes=output_size_bytes,
        elapsed_ms=elapsed_ms,
        language_used=lang or config.DEFAULT_OCR_LANG,
        output_format="pdf",
        debug_image_path=None,
        column_count=max_columns,
    )


def convert_batch(
    inputs: "str | list[str]",
    output: Optional[str] = None,
    *,
    merge: bool = False,
    lang: Optional[str] = None,
    dpi: Optional[int] = None,
    jpeg_quality: Optional[int] = None,
    char_whitelist: Optional[str] = None,
    char_blacklist: Optional[str] = None,
    pdf_metadata: Optional[dict] = None,
    force: bool = False,
    orientation: "str | float | None" = None,
    output_format: str = "pdf",
    debug_image: bool = False,
    verbose: bool = False,
    quiet: bool = False,
) -> BatchResult:
    """Batch-convert multiple images (or mixed images + PDFs).

    inputs: single path/pattern or list. Glob-expanded, PDFs rasterized.
    output: folder (one file per input) or file path (with merge=True).
        If omitted (None), each file is written next to its own input
        (same directory/stem, extension from output_format). Not valid
        with merge=True, which needs one explicit shared output path.
    merge: combine all inputs into one multi-page PDF.
    verbose/quiet: set log level for the call.

    Never raises for per-file failures. Check BatchResult.failures.
    """
    if verbose and quiet:
        raise ValueError("convert_batch(): verbose and quiet are mutually exclusive.")
    if merge and output_format != "pdf":
        raise ValueError("convert_batch(): merge=True only supports output_format='pdf'.")
    if merge and output is None:
        raise ValueError(
            "convert_batch(): merge=True requires an explicit output "
            "file path (there's no single input to derive a shared "
            "output name from)."
        )
    if verbose:
        config.configure(log_level="DEBUG")
    elif quiet:
        config.configure(log_level="ERROR")

    if isinstance(inputs, str):
        inputs = [inputs]

    raw_inputs: list[str] = []
    for pat in inputs:
        expanded = glob.glob(pat)
        raw_inputs.extend(expanded if expanded else [pat])

    results: list = []
    failures: list = []

    with tempfile.TemporaryDirectory(prefix="scanlayer_pdf_") as tmp_dir:
        expanded_inputs: list[str] = []
        for input_path in raw_inputs:
            try:
                expanded_inputs.extend(_expand_pdf_input(input_path, dpi, tmp_dir))
            except Exception as exc:
                failures.append((input_path, str(exc)))

        if not expanded_inputs:
            return BatchResult(results=results, failures=failures)

        if merge:
            try:
                results.append(convert_merge(
                    input_paths=expanded_inputs, output_path=output,
                    lang=lang, dpi=dpi, jpeg_quality=jpeg_quality,
                    char_whitelist=char_whitelist, char_blacklist=char_blacklist,
                    pdf_metadata=pdf_metadata, force=force, orientation=orientation,
                ))
            except Exception as exc:
                failures.append(("<merge>", str(exc)))
            return BatchResult(results=results, failures=failures)

        is_batch = len(expanded_inputs) > 1
        for input_path in expanded_inputs:
            if output is None:
                output_path = None  # convert() derives it next to input_path
            else:
                try:
                    output_path = _resolve_output_path(output, input_path, is_batch, output_format)
                except OSError as exc:
                    failures.append((input_path, str(exc)))
                    continue
            try:
                results.append(convert(
                    input_path=input_path, output_path=output_path,
                    lang=lang, dpi=dpi, jpeg_quality=jpeg_quality,
                    char_whitelist=char_whitelist, char_blacklist=char_blacklist,
                    pdf_metadata=pdf_metadata, force=force, orientation=orientation,
                    output_format=output_format, debug_image=debug_image,
                ))
            except Exception as exc:
                failures.append((input_path, str(exc)))

    return BatchResult(results=results, failures=failures)

