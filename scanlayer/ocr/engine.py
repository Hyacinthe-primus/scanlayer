"""
Tesseract wrapper: extracts words with coordinates.

Tries multiple PSM candidates and keeps the one with highest mean
confidence. Supports early-exit and parallel execution via threads.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytesseract
from PIL import Image

from scanlayer import config
from scanlayer.utils.errors import OCRProcessingError
from scanlayer.utils.logger import get_logger, log_warning, stage_timer

log = get_logger(__name__)

_ocr_slots_lock = threading.Lock()
_ocr_slots: "threading.BoundedSemaphore | None" = None
_ocr_slots_limit: "int | None" = None


def get_ocr_slots() -> threading.BoundedSemaphore:
    """Return the process-wide OCR concurrency budget.

    All Tesseract passes acquire this semaphore. The existing per-page PSM
    workers therefore cannot multiply beyond the configured ceiling when
    merge or future batch workers run concurrently.
    """
    global _ocr_slots, _ocr_slots_limit
    configured = config.OCR_MAX_CONCURRENCY
    if (
        isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured < 1
    ):
        limit = max(1, os.cpu_count() or 1)
    else:
        limit = configured

    with _ocr_slots_lock:
        if _ocr_slots is None or _ocr_slots_limit != limit:
            _ocr_slots = threading.BoundedSemaphore(limit)
            _ocr_slots_limit = limit
        return _ocr_slots


@dataclass
class Word:
    text: str
    x: float
    y: float
    width: float
    height: float
    confidence: float
    line_id: int = -1


@dataclass
class PsmAttempt:
    """Result of a single OCR pass for a given PSM, for diagnostics."""
    psm: int
    words: list[Word]
    mean_confidence: float
    elapsed_ms: float


@dataclass
class OcrResult:
    """Enriched OCR result, returns words + statistics."""
    words: list[Word]
    best_psm: int
    best_confidence: float
    attempts: list[PsmAttempt] = field(default_factory=list)
    early_exited: bool = False
    language_used: str = ""


def _is_printable(text: str) -> bool:
    """Return True if text contains only printable characters."""
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] == "C":  # Cc, Cf, Cs, Co, Cn
            return False
    return True


def _build_tess_config(
    tess_base: str,
    psm: int,
    effective_dpi: int,
    char_whitelist: Optional[str],
    char_blacklist: Optional[str],
) -> str:
    """Builds the Tesseract CLI argument string from the given options."""
    parts: list[str] = []
    if tess_base:
        parts.append(tess_base)
    parts.append(f"--psm {psm}")
    parts.append(f"--oem {config.TESSERACT_OEM}")
    parts.append(f"-c user_defined_dpi={effective_dpi}")
    if char_whitelist:
        parts.append(f"-c tessedit_char_whitelist={char_whitelist}")
    if char_blacklist:
        parts.append(f"-c tessedit_char_blacklist={char_blacklist}")
    return " ".join(parts)


def tessdata_config() -> str:
    """CLI fragment for a configured tessdata directory, or empty string."""
    if not config.TESSDATA_DIR:
        return ""
    return f'--tessdata-dir "{config.TESSDATA_DIR}"'


def _write_ocr_image(ocr_image: np.ndarray) -> str:
    """Encode the OCR image once to a temp BMP and return its path.

    pytesseract writes a fresh temp image for every ``image_to_data()`` call.
    With several PSM candidates that re-encodes the same page repeatedly.
    BMP is lossless and avoids PNG encoding/decoding overhead; the file is
    deleted by the caller as soon as all passes finish.
    """
    fd, path = tempfile.mkstemp(prefix="scanlayer_ocr_", suffix=".bmp")
    os.close(fd)
    try:
        Image.fromarray(ocr_image).save(path, format="BMP")
    except Exception:
        _remove_temp_file(path)
        raise
    return path


def _remove_temp_file(path: Optional[str]) -> None:
    if not path:
        return
    # Windows can briefly keep a handle after Tesseract exits (antivirus /
    # indexer). Retry a few times before giving up; cleanup must not turn a
    # successful OCR run into a failure.
    last_error: "OSError | None" = None
    for attempt in range(4):
        try:
            os.unlink(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    log_warning(
        log,
        f"Could not remove temporary OCR image {path}: {last_error}",
    )


def _run_ocr(
    ocr_image: "str | np.ndarray",
    lang: str,
    psm: int,
    tess_base: str,
    effective_dpi: int,
    char_whitelist: Optional[str],
    char_blacklist: Optional[str],
) -> PsmAttempt:
    """Run a single OCR pass with a given PSM. Raises on genuine failure.

    ``ocr_image`` is normally a pre-encoded file path; a NumPy array is still
    accepted for internal compatibility, but then pytesseract re-encodes it.
    """
    t0 = time.perf_counter()

    tess_config = _build_tess_config(
        tess_base, psm, effective_dpi, char_whitelist, char_blacklist,
    )

    try:
        data = pytesseract.image_to_data(
            ocr_image, lang=lang, config=tess_config,
            output_type=pytesseract.Output.DICT,
            timeout=config.OCR_TIMEOUT_SECONDS,
        )
    except pytesseract.TesseractNotFoundError:
        raise
    except pytesseract.TesseractError:
        raise
    except RuntimeError as exc:
        raise OCRProcessingError(
            f"PSM {psm}: Tesseract timeout (> {config.OCR_TIMEOUT_SECONDS}s). "
            "Image likely too large/noisy, or the Tesseract binary is stuck."
        ) from exc

    words: list[Word] = []
    n = len(data["text"])
    for i in range(n):
        raw_text = data["text"][i].strip()
        if not raw_text:
            continue

        if config.DROP_NON_PRINTABLE_WORDS and not _is_printable(raw_text):
            log.debug(f"PSM {psm}: word rejected (non-printable): {raw_text!r}")
            continue

        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0

        if conf < config.MIN_WORD_CONFIDENCE:
            continue

        width = float(data["width"][i])
        height = float(data["height"][i])
        if width <= 0 or height <= 0:
            continue

        try:
            line_id = (
                int(data["block_num"][i]) * 1_000_000
                + int(data["par_num"][i]) * 1_000
                + int(data["line_num"][i])
            )
        except (ValueError, TypeError, KeyError):
            line_id = -1

        words.append(Word(
            text=raw_text,
            x=float(data["left"][i]),
            y=float(data["top"][i]),
            width=width,
            height=height,
            confidence=conf,
            line_id=line_id,
        ))

    mean_conf = sum(w.confidence for w in words) / len(words) if words else 0.0
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return PsmAttempt(
        psm=psm, words=words, mean_confidence=mean_conf, elapsed_ms=elapsed_ms,
    )


def use_configured_tesseract() -> None:
    """Point pytesseract at config.TESSERACT_CMD.

    pytesseract keeps the binary path in a module global, and
    preprocessing's OSD call runs before the first extract_words(), so
    callers that start concurrent work set it once up front.
    """
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD


def _safe_run_ocr(
    *args,
    psm: int,
    ocr_slots: Optional[threading.Semaphore] = None,
    **kwargs,
) -> tuple[PsmAttempt | None, Exception | None]:
    """Run one PSM candidate, converting exceptions into return values.

    With ocr_slots, waits for a free slot first, so the Tesseract timeout
    only starts counting once the process actually runs.
    """
    if ocr_slots is not None:
        ocr_slots.acquire()
    try:
        return _run_ocr(*args, **kwargs), None
    except pytesseract.TesseractNotFoundError as exc:
        return None, exc
    except pytesseract.TesseractError as exc:
        log_warning(log, f"PSM {psm} failed (TesseractError: {exc}), pass skipped.")
        return None, exc
    except OCRProcessingError as exc:
        log_warning(log, f"PSM {psm} failed: {exc}")
        return None, exc
    except Exception as exc:  # genuinely unexpected, still surfaced, not swallowed
        log_warning(log, f"PSM {psm}, unexpected error ({type(exc).__name__}: {exc})")
        return None, exc
    finally:
        if ocr_slots is not None:
            ocr_slots.release()


def extract_words(
    ocr_image: np.ndarray,
    ocr_scale: float,
    effective_dpi: int,
    lang: Optional[str] = None,
    char_whitelist: Optional[str] = None,
    char_blacklist: Optional[str] = None,
    ocr_slots: Optional[threading.Semaphore] = None,
) -> OcrResult:
    """Run OCR with automatic PSM selection.

    The OCR image is encoded to one temporary file and reused by every PSM
    pass. This removes the repeated ``Image.save()`` calls pytesseract would
    otherwise perform for each ``image_to_data()`` invocation.

    ocr_slots, if given, is shared by concurrent callers (convert_merge()
    processing several pages at once) and bounds how many Tesseract passes
    run at the same time. When omitted, the process-wide get_ocr_slots()
    budget is used. It never changes which result is selected.

    Raises OCRProcessingError if every PSM candidate fails.
    """
    lang = lang or config.DEFAULT_OCR_LANG
    wl = char_whitelist if char_whitelist is not None else config.TESSERACT_CHAR_WHITELIST
    bl = char_blacklist if char_blacklist is not None else config.TESSERACT_CHAR_BLACKLIST

    use_configured_tesseract()
    tess_base = tessdata_config()

    psm_candidates = list(config.TESSERACT_PSM_CANDIDATES)
    if not psm_candidates:
        log_warning(log, "No PSM candidates configured, falling back to PSM 3.")
        psm_candidates = [3]

    attempts: list[PsmAttempt] = []
    errors: list[Exception] = []
    early_exited = False
    best: PsmAttempt | None = None

    use_parallel = config.PSM_PARALLEL and len(psm_candidates) > 1
    slots = ocr_slots if ocr_slots is not None else get_ocr_slots()
    log.info(
        f"OCR started: lang={lang}, dpi={effective_dpi}, "
        f"PSM candidates={psm_candidates}, parallel={use_parallel}"
    )

    ocr_path = None
    try:
        with stage_timer(log, "OCR"):
            ocr_path = _write_ocr_image(ocr_image)
            if len(psm_candidates) == 1:
                attempt, error = _safe_run_ocr(
                    ocr_path, lang, psm_candidates[0], tess_base,
                    effective_dpi, wl, bl, psm=psm_candidates[0],
                    ocr_slots=slots,
                )
                if attempt is not None:
                    attempts.append(attempt)
                    best = attempt
                else:
                    errors.append(error)

            elif use_parallel:
                max_workers = config.PSM_MAX_WORKERS or min(32, (os.cpu_count() or 1) + 4)
                max_workers = min(max_workers, len(psm_candidates))
                can_skip_unstarted = max_workers < len(psm_candidates)
                order = {psm: i for i, psm in enumerate(psm_candidates)}

                executor = ThreadPoolExecutor(max_workers=max_workers)
                futures = {
                    executor.submit(
                        _safe_run_ocr, ocr_path, lang, psm, tess_base,
                        effective_dpi, wl, bl, psm=psm, ocr_slots=slots,
                    ): psm
                    for psm in psm_candidates
                }
                pending = set(futures)
                try:
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            attempt, error = future.result()
                            if attempt is not None:
                                attempts.append(attempt)
                                if (
                                    best is None
                                    or attempt.mean_confidence > best.mean_confidence
                                    or (
                                        attempt.mean_confidence == best.mean_confidence
                                        and order[attempt.psm] < order[best.psm]
                                    )
                                ):
                                    best = attempt
                            else:
                                errors.append(error)

                        if (
                            can_skip_unstarted
                            and best is not None
                            and best.mean_confidence >= config.PSM_EARLY_EXIT_CONFIDENCE
                        ):
                            early_exited = True
                            log.info(
                                f"Early-exit: confidence {best.mean_confidence:.1f}% "
                                f">= threshold {config.PSM_EARLY_EXIT_CONFIDENCE}%, "
                                f"skipping {len(pending)} not-yet-started candidate(s)."
                            )
                            break
                finally:
                    # Wait for running passes before deleting the shared OCR
                    # image; otherwise a Tesseract process may still be reading
                    # it. cancel_futures only removes tasks that never started.
                    executor.shutdown(wait=True, cancel_futures=True)

            else:
                for psm in psm_candidates:
                    attempt, error = _safe_run_ocr(
                        ocr_path, lang, psm, tess_base,
                        effective_dpi, wl, bl, psm=psm, ocr_slots=slots,
                    )
                    if attempt is not None:
                        attempts.append(attempt)
                        if best is None or attempt.mean_confidence > best.mean_confidence:
                            best = attempt
                    else:
                        errors.append(error)

                    if (
                        best is not None
                        and best.mean_confidence >= config.PSM_EARLY_EXIT_CONFIDENCE
                    ):
                        early_exited = True
                        log.info(
                            f"Early-exit: confidence {best.mean_confidence:.1f}% "
                            f">= threshold {config.PSM_EARLY_EXIT_CONFIDENCE}%, "
                            f"remaining PSMs skipped."
                        )
                        break
    finally:
        _remove_temp_file(ocr_path)

    if best is None:
        detail = "; ".join(f"{type(e).__name__}: {e}" for e in errors) or "no candidates ran"
        raise OCRProcessingError(
            f"All {len(psm_candidates)} PSM candidate(s) failed: {detail}"
        )

    for a in sorted(attempts, key=lambda x: x.psm):
        log.debug(
            f"PSM {a.psm}: {len(a.words)} words, "
            f"mean confidence {a.mean_confidence:.1f}%, "
            f"{a.elapsed_ms:.0f} ms"
        )
    log.info(
        f"OCR complete: PSM {best.psm} selected "
        f"({len(best.words)} words, confidence {best.mean_confidence:.1f}%)"
        + (" [early-exit]" if early_exited else "")
    )

    result_words: list[Word] = []
    for w in best.words:
        result_words.append(Word(
            text=w.text,
            x=w.x / ocr_scale,
            y=w.y / ocr_scale,
            width=w.width / ocr_scale,
            height=w.height / ocr_scale,
            confidence=w.confidence,
            line_id=w.line_id,
        ))

    return OcrResult(
        words=result_words,
        best_psm=best.psm,
        best_confidence=best.mean_confidence,
        attempts=attempts,
        early_exited=early_exited,
        language_used=lang,
    )
