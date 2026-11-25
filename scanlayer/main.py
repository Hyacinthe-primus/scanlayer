"""
Library entry point for scanlayer.

For the CLI, see scanlayer.cli.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from scanlayer import config
from scanlayer.layout.columns import reorder_reading_order
from scanlayer.ocr.engine import extract_words, get_ocr_slots, use_configured_tesseract
from scanlayer.ocr.export import FORMATS, export_words
from scanlayer.pdf.builder import build_searchable_pdf, build_searchable_pdf_multipage
from scanlayer.pdf.fonts import resolve_font
from scanlayer.preprocessing.enhance import preprocess
from scanlayer.utils.debug_image import build_debug_image
from scanlayer.utils.errors import BlankPageDetectedError
from scanlayer.utils.logger import get_logger, log_success, log_warning, temporary_log_level
from scanlayer.utils.resources import temporary_cv2_threads
from scanlayer.utils.validators import (
    DependencyError,
    InputFileError,
    OutputPathError,
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
    per-file failures. failure_exceptions is an additive typed view used by
    the CLI to preserve exit-code mapping.
    """
    results: list  # list[ConversionResult]
    failures: list  # list[tuple[str, str]]
    # Typed counterpart of failures, used by the CLI to preserve exit-code
    # mapping without parsing strings. Not part of the original API.
    failure_exceptions: list = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return len(self.results)

    @property
    def failed(self) -> int:
        return len(self.failures)

    @property
    def ok(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class BatchOptions:
    """Immutable conversion options shared by batch planning/execution."""
    lang: Optional[str] = None
    dpi: Optional[int] = None
    jpeg_quality: Optional[int] = None
    char_whitelist: Optional[str] = None
    char_blacklist: Optional[str] = None
    pdf_metadata: Optional[dict] = None
    force: bool = False
    orientation: "str | float | None" = None
    output_format: str = "pdf"
    debug_image: bool = False
    merge: bool = False
    workers: Optional[int] = None
    batch_workers: Optional[int] = None
    opencv_threads: Optional[int] = None


@dataclass(frozen=True)
class BatchJob:
    """One planned conversion."""
    input_path: str
    output_path: str
    source_path: str
    page_number: Optional[int] = None
    output_dir: Optional[str] = None


@dataclass
class BatchPlan:
    """Batch plan.

    Planning does not create output paths. For non-dry-run PDF inputs,
    temporary page images may already exist in the caller's temp directory.
    """
    jobs: list  # list[BatchJob]
    failures: list  # list[tuple[str, Exception]]


@dataclass
class BatchExecution:
    """Sequential execution result, with typed failures for the CLI."""
    results: list  # list[ConversionResult]
    failures: list  # list[tuple[str, Exception]]


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


def _resolve_output_target(
    output_arg: Optional[str], input_path: str, is_batch: bool,
    output_format: str = "pdf",
) -> tuple[str, Optional[str]]:
    """Resolve (output_path, folder_to_create) without filesystem changes.

    output_arg=None -> next to input_path. folder_to_create is set for
    folder-style outputs (batch/folder/trailing separator/existing dir).
    """
    if output_arg is None:
        return _default_output_path(input_path, output_format), None
    if is_batch or output_arg.endswith(("/", "\\")) or os.path.isdir(output_arg):
        stem = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(output_arg, f"{stem}.{output_format}"), output_arg
    return output_arg, None


def _resolve_output_path(
    output_arg: Optional[str], input_path: str, is_batch: bool,
    output_format: str = "pdf", create: bool = True,
) -> str:
    """Resolve output path for single or batch mode.

    create=False is used during planning; create=True preserves the original
    behavior of creating a folder-style output directory.
    """
    output_path, folder = _resolve_output_target(
        output_arg, input_path, is_batch, output_format,
    )
    if create and folder:
        os.makedirs(folder, exist_ok=True)
    return output_path


def _normalize_input_paths(inputs: "str | list[str]") -> list[str]:
    """Expand patterns deterministically and deduplicate resolved paths.

    Pattern order is preserved. Matches within one pattern are sorted;
    duplicate paths keep their first occurrence.
    """
    if isinstance(inputs, str):
        inputs = [inputs]
    normalized: list[str] = []
    seen: set[str] = set()
    for pattern in inputs:
        matches = sorted(glob.glob(pattern), key=lambda p: os.path.normcase(p))
        candidates = matches or [pattern]
        for candidate in candidates:
            key = os.path.normcase(os.path.abspath(os.path.expanduser(candidate)))
            if key in seen:
                continue
            seen.add(key)
            normalized.append(candidate)
    return normalized


def _pdf_page_logical_path(
    pdf_path: str, page_number: int, output_format: str
) -> str:
    """Synthetic page path used only for output naming/planning."""
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(
        os.path.dirname(os.path.abspath(pdf_path)),
        f"{stem}_p{page_number}.{output_format}",
    )


def _pdf_page_count(path: str) -> int:
    """Read PDF page count via pdfinfo without rasterizing pages."""
    try:
        from pdf2image import pdfinfo_from_path
        from pdf2image.exceptions import PDFInfoNotInstalledError
    except ImportError as exc:
        raise DependencyError(
            "PDF input requires the 'pdf2image' package: "
            "pip install pdf2image"
        ) from exc

    try:
        info = pdfinfo_from_path(path)
    except PDFInfoNotInstalledError as exc:
        raise DependencyError(
            "PDF input requires poppler-utils (the 'pdfinfo' binary) on "
            "PATH. 'pip install pdf2image' alone is not enough; install "
            "poppler (apt install poppler-utils / brew install poppler / "
            "download poppler for Windows and add it to PATH)."
        ) from exc
    except Exception as exc:
        raise InputFileError(
            f"'{path}': could not read as a PDF "
            f"({type(exc).__name__}: {exc})"
        ) from exc

    try:
        pages = int(info.get("Pages", 0))
    except (TypeError, ValueError):
        pages = 0
    if pages <= 0:
        raise InputFileError(f"'{path}': PDF has 0 pages, nothing to convert.")
    return pages


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


def _expand_pdf_input(
    path: str, dpi: Optional[int], tmp_dir: str,
    page_prefix: Optional[str] = None,
) -> list[str]:
    """Rasterize PDF pages to PNGs. Non-PDF inputs pass through unchanged.

    page_prefix lets a batch plan keep same-stem PDFs in separate temp files.
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

    stem = page_prefix or os.path.splitext(os.path.basename(path))[0]
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


def _pdftoppm_subprocess_kwargs() -> dict:
    """Hide the poppler child console window on Windows."""
    if sys.platform != "win32":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {"startupinfo": startupinfo}


def _run_pdftoppm(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run one poppler rasterization command.

    Kept as a separate seam so tests can replace the subprocess boundary
    without patching global subprocess state.
    """
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_pdftoppm_subprocess_kwargs(),
    )


def _rasterize_pdf_pages(
    path: str, first_page: int, last_page: int,
    dpi: Optional[int], tmp_dir: str,
) -> tuple[list[str], str]:
    """Rasterize a bounded page range into a unique per-chunk directory.

    This invokes ``pdftoppm`` directly instead of going through
    ``pdf2image.convert_from_path``. The latter runs ``pdfinfo`` again for
    every chunk and adds Python-level path handling, which dominated small
    and medium PDF batches. Direct poppler output is byte-identical PPM.
    """
    raster_dpi = dpi or config.DEFAULT_DPI
    job_dir = tempfile.mkdtemp(prefix="scanlayer_pdf_page_", dir=tmp_dir)
    prefix = os.path.join(job_dir, "page")

    try:
        executable = shutil.which("pdftoppm")
        if not executable:
            raise DependencyError(
                "PDF input requires poppler-utils (the 'pdftoppm'/'pdfinfo' "
                "binaries) on PATH. 'pip install pdf2image' alone is not "
                "enough, poppler is a separate system package "
                "(apt install poppler-utils / brew install poppler / "
                "download poppler for Windows and add it to PATH)."
            )

        cmd = [
            executable,
            "-r", str(raster_dpi),
            "-f", str(first_page),
            "-l", str(last_page),
            path,
            prefix,
        ]
        try:
            proc = _run_pdftoppm(cmd)
        except FileNotFoundError as exc:
            raise DependencyError(
                "PDF input requires poppler-utils (the 'pdftoppm'/'pdfinfo' "
                "binaries) on PATH. 'pip install pdf2image' alone is not "
                "enough, poppler is a separate system package "
                "(apt install poppler-utils / brew install poppler / "
                "download poppler for Windows and add it to PATH)."
            ) from exc

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise InputFileError(
                f"'{path}' pages {first_page}-{last_page}: could not "
                f"rasterize (pdftoppm exited {proc.returncode}"
                + (f": {detail}" if detail else "")
                + ")."
            )

        page_paths = glob.glob(prefix + "*.ppm")

        def page_number(page_path: str) -> int:
            stem = os.path.splitext(os.path.basename(page_path))[0]
            try:
                return int(stem.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                return -1

        page_paths.sort(key=page_number)
        expected = list(range(first_page, last_page + 1))
        if not page_paths or [page_number(p) for p in page_paths] != expected:
            raise InputFileError(
                f"'{path}' pages {first_page}-{last_page}: pdftoppm did not "
                "produce the expected page files."
            )
        return page_paths, job_dir
    except (DependencyError, InputFileError):
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise InputFileError(
            f"'{path}' pages {first_page}-{last_page}: could not rasterize "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    except BaseException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise


def _rasterize_pdf_page(
    path: str, page_number: int, dpi: Optional[int], tmp_dir: str,
) -> tuple[str, str]:
    """Rasterize one page; retained as the single-page helper for callers/tests."""
    page_paths, job_dir = _rasterize_pdf_pages(
        path, page_number, page_number, dpi, tmp_dir,
    )
    return page_paths[0], job_dir


class _PdfPageCache:
    """Lazily rasterize PDF pages in bounded chunks for one source PDF.

    A chunk is rasterized by the existing batch worker that needs its first
    page; other workers wait on this small lock and reuse the files. The
    cache owns the temp directories and removes each page after conversion.
    """

    def __init__(
        self, source_path: str, total_pages: int, dpi: Optional[int],
        tmp_dir: str, chunk_size: int,
    ):
        self.source_path = source_path
        self.total_pages = total_pages
        self.dpi = dpi
        self.tmp_dir = tmp_dir
        self.chunk_size = max(1, chunk_size)
        self._lock = threading.Lock()
        self._pages: dict[int, str] = {}
        self._dirs: dict[int, str] = {}
        self._remaining: dict[int, int] = {}
        self._failures: dict[int, Exception] = {}

    def _chunk_start(self, page: int) -> int:
        return ((page - 1) // self.chunk_size) * self.chunk_size + 1

    def get(self, page: int) -> str:
        with self._lock:
            if page in self._pages:
                return self._pages[page]
            start = self._chunk_start(page)
            if start in self._failures:
                raise self._failures[start]
            last = min(self.total_pages, start + self.chunk_size - 1)
            try:
                paths, job_dir = _rasterize_pdf_pages(
                    self.source_path, start, last, self.dpi, self.tmp_dir,
                )
            except Exception as exc:
                self._failures[start] = exc
                raise
            self._dirs[start] = job_dir
            self._remaining[start] = len(paths)
            for offset, path in enumerate(paths):
                self._pages[start + offset] = path
            if page not in self._pages:
                raise InputFileError(
                    f"'{self.source_path}' page {page}: no rasterized page."
                )
            return self._pages[page]

    def release(self, page: int) -> None:
        with self._lock:
            path = self._pages.pop(page, None)
            if path is not None:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            start = self._chunk_start(page)
            if start not in self._remaining:
                return
            self._remaining[start] -= 1
            if self._remaining[start] <= 0:
                self._remaining.pop(start, None)
                job_dir = self._dirs.pop(start, None)
                if job_dir:
                    shutil.rmtree(job_dir, ignore_errors=True)

    def cleanup(self) -> None:
        with self._lock:
            for job_dir in self._dirs.values():
                shutil.rmtree(job_dir, ignore_errors=True)
            self._dirs.clear()
            self._pages.clear()
            self._remaining.clear()


def plan_batch(
    inputs: "str | list[str]",
    output: Optional[str],
    options: BatchOptions,
    tmp_dir: Optional[str] = None,
    dry_run: bool = False,
) -> BatchPlan:
    """Normalize inputs and build a deterministic conversion plan.

    Glob order is deterministic, duplicate inputs are removed, PDF page order
    is preserved, and output collisions are rejected before conversion.
    Planning does not create output paths. Non-dry-run PDF inputs are
    rasterized into tmp_dir before the plan is returned; dry_run=True reads
    PDF metadata only and never rasterizes pages.
    """
    normalized = _normalize_input_paths(inputs)
    jobs: list[BatchJob] = []
    failures: list[tuple[str, Exception]] = []

    for index, source in enumerate(normalized):
        is_pdf = os.path.splitext(source)[1].lower() == ".pdf"
        if is_pdf:
            try:
                if dry_run or not options.merge:
                    pages = _pdf_page_count(source)
                    page_inputs = [source] * pages
                else:
                    if tmp_dir is None:
                        raise ValueError(
                            "plan_batch() needs a temporary directory for merge PDF input"
                        )
                    stem = os.path.splitext(os.path.basename(source))[0]
                    page_inputs = _expand_pdf_input(
                        source, options.dpi, tmp_dir,
                        page_prefix=f"in{index}_{stem}",
                    )
            except Exception as exc:
                failures.append((source, exc))
                continue
            for page_number, page_input in enumerate(page_inputs, start=1):
                logical = _pdf_page_logical_path(
                    source, page_number, options.output_format,
                )
                jobs.append(BatchJob(
                    input_path=page_input,
                    output_path=logical,
                    source_path=source,
                    page_number=page_number,
                ))
        else:
            jobs.append(BatchJob(
                input_path=source,
                output_path=source,
                source_path=source,
            ))

    if options.merge:
        return BatchPlan(jobs=jobs, failures=failures)

    is_batch = len(jobs) > 1
    collisions: dict[str, BatchJob] = {}
    planned: list[BatchJob] = []
    for job in jobs:
        logical_path = job.output_path
        output_path, output_dir = _resolve_output_target(
            output, logical_path, is_batch, options.output_format,
        )
        key = os.path.normcase(os.path.abspath(output_path))
        if key in collisions:
            other = collisions[key]
            raise OutputPathError(
                "Batch output collision: "
                f"{other.source_path!r} and {job.source_path!r} "
                f"both map to {output_path!r}."
            )
        collisions[key] = job
        planned.append(BatchJob(
            input_path=job.input_path,
            output_path=output_path,
            source_path=job.source_path,
            page_number=job.page_number,
            output_dir=output_dir,
        ))
    return BatchPlan(jobs=planned, failures=failures)


def _execute_job(
    job: BatchJob, options: BatchOptions,
    page_cache: Optional["_PdfPageCache"] = None,
):
    """Run one job, rasterizing a cached PDF page if needed, without raising."""
    input_path = job.input_path
    got_page = False
    try:
        if page_cache is not None:
            input_path = page_cache.get(job.page_number)
            got_page = True
        if job.output_dir:
            try:
                os.makedirs(job.output_dir, exist_ok=True)
            except OSError as exc:
                return job, None, exc
        result = convert(
            input_path=input_path,
            output_path=job.output_path,
            lang=options.lang,
            dpi=options.dpi,
            jpeg_quality=options.jpeg_quality,
            char_whitelist=options.char_whitelist,
            char_blacklist=options.char_blacklist,
            pdf_metadata=options.pdf_metadata,
            force=options.force,
            orientation=options.orientation,
            output_format=options.output_format,
            debug_image=options.debug_image,
        )
        return job, result, None
    except Exception as exc:
        return job, None, exc
    finally:
        if got_page and page_cache is not None:
            page_cache.release(job.page_number)


def _execute_plan_with_dir(
    plan: BatchPlan, options: BatchOptions, workers: int, tmp_dir: str,
) -> BatchExecution:
    results: list = []
    failures: list[tuple[str, Exception]] = []

    page_caches: dict[str, _PdfPageCache] = {}
    for job in plan.jobs:
        if job.page_number is None:
            continue
        cache = page_caches.get(job.source_path)
        if cache is None:
            total = max(
                j.page_number for j in plan.jobs
                if j.source_path == job.source_path and j.page_number is not None
            )
            page_caches[job.source_path] = _PdfPageCache(
                job.source_path, total, options.dpi, tmp_dir, workers,
            )

    def run_job(job):
        cache = page_caches.get(job.source_path) if job.page_number is not None else None
        return _execute_job(job, options, cache)

    try:
        if workers <= 1:
            with temporary_cv2_threads(options.opencv_threads):
                outcomes = [run_job(job) for job in plan.jobs]
        else:
            opencv_threads = (
                options.opencv_threads
                if options.opencv_threads is not None
                else max(1, (os.cpu_count() or 1) // workers)
            )
            with temporary_cv2_threads(opencv_threads):
                # Register the PDF font once before parallel PDF builds. The
                # registry is process-global; preflight avoids first-use races.
                if options.output_format == "pdf":
                    resolve_font(options.lang or config.DEFAULT_OCR_LANG)
                with ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="scanlayer-batch",
                ) as executor:
                    outcomes = list(executor.map(run_job, plan.jobs))
    finally:
        for cache in page_caches.values():
            cache.cleanup()

    for job, result, exc in outcomes:
        if exc is not None:
            failures.append((job.source_path, exc))
        else:
            results.append(result)
    return BatchExecution(results=results, failures=failures)


def execute_plan(
    plan: BatchPlan, options: BatchOptions, workers: int = 1,
    tmp_dir: Optional[str] = None,
) -> BatchExecution:
    """Execute a plan sequentially or with a bounded worker pool.

    PDF page jobs are rasterized one page at a time inside these workers; the
    document is never fully expanded during planning. Results/failures are
    assembled in plan order regardless of completion order. The global OCR
    semaphore, not this pool, remains the hard ceiling on Tesseract passes.
    """
    if tmp_dir is not None:
        return _execute_plan_with_dir(plan, options, workers, tmp_dir)
    with tempfile.TemporaryDirectory(prefix="scanlayer_pdf_pages_") as owned:
        return _execute_plan_with_dir(plan, options, workers, owned)


def _process_page_for_merge(
    input_path: str,
    lang: Optional[str],
    dpi: Optional[int],
    char_whitelist: Optional[str],
    char_blacklist: Optional[str],
    orientation: "str | float | None",
    force: bool,
    ocr_slots: Optional[threading.Semaphore] = None,
):
    """Run preprocessing + OCR for one page of a --merge run.

    Returns (PageInput, stats_dict). Does not touch the output path.
    Safe to call from several threads at once.
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
        ocr_slots=ocr_slots,
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


def _resolve_batch_workers(workers: Optional[int], job_count: int) -> int:
    """How many non-merge conversions run at once (always >= 1).

    None means config.BATCH_WORKERS. Never exceeds the job count or CPU
    count. The global OCR semaphore remains the hard Tesseract ceiling.
    """
    requested = config.BATCH_WORKERS if workers is None else workers
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError(
            f"batch_workers must be an integer >= 1, got {requested!r}."
        )
    return max(1, min(requested, max(1, job_count), os.cpu_count() or 1))


def _resolve_merge_workers(workers: Optional[int], page_count: int) -> int:
    """How many pages a merge run processes at once (always >= 1).

    None means config.MERGE_WORKERS. The result never exceeds
    config.MERGE_WORKERS_LIMIT, the page count, or the CPU count.
    """
    requested = config.MERGE_WORKERS if workers is None else workers
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError(f"workers must be an integer >= 1, got {requested!r}.")

    limit = config.MERGE_WORKERS_LIMIT
    if requested > limit:
        log_warning(
            get_logger("main"),
            f"workers={requested} is above the maximum of {limit}, using {limit}.",
        )
    return max(1, min(requested, limit, page_count, os.cpu_count() or 1))


def _process_pages_for_merge(
    input_paths: list[str],
    workers: int,
    lang: Optional[str],
    dpi: Optional[int],
    char_whitelist: Optional[str],
    char_blacklist: Optional[str],
    orientation: "str | float | None",
    force: bool,
) -> list:
    """Run preprocessing + OCR for every page of a merge run.

    Returns [(PageInput, stats), ...] in input order, whatever order the
    pages finish in. If a page fails, the failure of the first bad page in
    input order is raised and pages that have not started yet are dropped.
    With workers == 1 pages are processed one at a time on the calling
    thread, exactly as before.
    """
    log = get_logger("main")
    page_args = (lang, dpi, char_whitelist, char_blacklist, orientation, force)
    use_configured_tesseract()

    def _done(input_path: str, stats: dict) -> None:
        log.info(
            f"'{input_path}': {stats['words']} words "
            f"(mean confidence {stats['mean_conf']:.1f}%"
            + (f", {stats['columns']} columns" if stats["columns"] > 1 else "")
            + ")"
        )

    results: list = []
    if workers <= 1:
        for input_path in input_paths:
            page, stats = _process_page_for_merge(input_path, *page_args)
            _done(input_path, stats)
            results.append((page, stats))
        return results

    # Every page already races several Tesseract processes. Use the same
    # process-wide OCR budget as normal conversion.
    ocr_slots = get_ocr_slots()
    log.info(f"Processing {len(input_paths)} page(s) with {workers} parallel workers.")

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scanlayer-page")
    try:
        futures = [
            executor.submit(
                _process_page_for_merge, input_path, *page_args, ocr_slots=ocr_slots,
            )
            for input_path in input_paths
        ]
        for input_path, future in zip(input_paths, futures):
            page, stats = future.result()
            _done(input_path, stats)
            results.append((page, stats))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return results


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
    workers: Optional[int] = None,
) -> ConversionResult:
    """Combine several images into one multi-page searchable PDF.

    PDF-only. Raises BlankPageDetectedError if any page is blank and
    force=False.

    workers: pages processed concurrently, 1 to config.MERGE_WORKERS_LIMIT
        (4). None uses config.MERGE_WORKERS (default 3). 1 processes one
        page at a time. It only changes speed: page order and OCR output are
        the same for any value.
    """
    if not input_paths:
        raise ValueError("convert_merge() called with no input paths.")

    worker_count = _resolve_merge_workers(workers, len(input_paths))

    log = get_logger("main")
    t0 = time.perf_counter()

    output_abs = validate_output_path(output_path, expected_ext=".pdf")

    pages = []
    total_words = 0
    conf_weighted_sum = 0.0
    conf_weight_n = 0
    max_columns = 1
    for page, stats in _process_pages_for_merge(
        input_paths, worker_count, lang, dpi, char_whitelist, char_blacklist,
        orientation, force,
    ):
        pages.append(page)
        total_words += stats["words"]
        if stats["words"]:
            conf_weighted_sum += stats["mean_conf"] * stats["words"]
            conf_weight_n += stats["words"]
        max_columns = max(max_columns, stats["columns"])

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


def _execute_batch(
    inputs: "str | list[str]",
    output: Optional[str],
    options: BatchOptions,
) -> BatchResult:
    results: list = []
    failures: list[tuple[str, str]] = []
    failure_exceptions: list[tuple[str, Exception]] = []

    with temporary_cv2_threads(options.opencv_threads):
        with tempfile.TemporaryDirectory(prefix="scanlayer_pdf_") as tmp_dir:
            plan = plan_batch(
                inputs, output, options, tmp_dir=tmp_dir, dry_run=False,
            )
            for source, exc in plan.failures:
                failures.append((source, str(exc)))
                failure_exceptions.append((source, exc))

            if options.merge:
                if plan.jobs:
                    try:
                        results.append(convert_merge(
                            input_paths=[job.input_path for job in plan.jobs],
                            output_path=output,
                            lang=options.lang, dpi=options.dpi,
                            jpeg_quality=options.jpeg_quality,
                            char_whitelist=options.char_whitelist,
                            char_blacklist=options.char_blacklist,
                            pdf_metadata=options.pdf_metadata,
                            force=options.force,
                            orientation=options.orientation,
                            workers=options.workers,
                        ))
                    except Exception as exc:
                        failures.append(("<merge>", str(exc)))
                        failure_exceptions.append(("<merge>", exc))
                return BatchResult(
                    results=results,
                    failures=failures,
                    failure_exceptions=failure_exceptions,
                )

            execution = execute_plan(
                plan, options,
                workers=_resolve_batch_workers(options.batch_workers, len(plan.jobs)),
                tmp_dir=tmp_dir,
            )
            results.extend(execution.results)
            for source, exc in execution.failures:
                failures.append((source, str(exc)))
                failure_exceptions.append((source, exc))

    return BatchResult(
        results=results,
        failures=failures,
        failure_exceptions=failure_exceptions,
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
    workers: Optional[int] = None,
    batch_workers: Optional[int] = None,
    verbose: bool = False,
    quiet: bool = False,
) -> BatchResult:
    """Batch-convert multiple images (or mixed images + PDFs).

    inputs: single path/pattern or list. Glob-expanded, PDFs rasterized.
    output: folder (one file per input) or file path (with merge=True).
        If omitted (None), each file is written next to its own input
        (same directory/stem, extension from output_format).
    merge: combine all inputs into one multi-page PDF.
    workers: with merge=True, pages processed concurrently (see
        convert_merge()). Ignored without merge.
    batch_workers: non-merge conversions processed concurrently. None uses
        config.BATCH_WORKERS (default 2). Capped by input count and CPU count.
        The global OCR semaphore remains the Tesseract ceiling.
    verbose/quiet: set log level for the call; restored afterward.

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
    if not merge:
        _resolve_batch_workers(batch_workers, os.cpu_count() or 1)
    if verbose:
        log_context = temporary_log_level("DEBUG")
    elif quiet:
        log_context = temporary_log_level("ERROR")
    else:
        log_context = nullcontext()

    options = BatchOptions(
        lang=lang,
        dpi=dpi,
        jpeg_quality=jpeg_quality,
        char_whitelist=char_whitelist,
        char_blacklist=char_blacklist,
        pdf_metadata=pdf_metadata,
        force=force,
        orientation=orientation,
        output_format=output_format,
        debug_image=debug_image,
        merge=merge,
        workers=workers,
        batch_workers=batch_workers,
    )

    with log_context:
        return _execute_batch(inputs, output, options)

