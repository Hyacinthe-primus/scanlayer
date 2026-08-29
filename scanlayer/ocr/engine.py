"""
Tesseract wrapper: extracts words with coordinates.

Tries multiple PSM candidates and keeps the one with highest mean
confidence. Supports early-exit and parallel execution via threads.
"""

from __future__ import annotations

import os
import time
import unicodedata
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytesseract

from scanlayer import config
from scanlayer.utils.errors import OCRProcessingError
from scanlayer.utils.logger import get_logger, log_warning, stage_timer

log = get_logger(__name__)


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


def _run_ocr(
    ocr_image: np.ndarray,
    lang: str,
    psm: int,
    tess_base: str,
    effective_dpi: int,
    char_whitelist: Optional[str],
    char_blacklist: Optional[str],
) -> PsmAttempt:
    """Run a single OCR pass with a given PSM. Raises on genuine failure."""
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


def _safe_run_ocr(*args, psm: int, **kwargs) -> tuple[PsmAttempt | None, Exception | None]:
    """Run one PSM candidate, converting exceptions into return values."""
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


def extract_words(
    ocr_image: np.ndarray,
    ocr_scale: float,
    effective_dpi: int,
    lang: Optional[str] = None,
    char_whitelist: Optional[str] = None,
    char_blacklist: Optional[str] = None,
) -> OcrResult:
    """Run OCR with automatic PSM selection.

    Raises OCRProcessingError if every PSM candidate fails.
    """
    lang = lang or config.DEFAULT_OCR_LANG
    wl = char_whitelist if char_whitelist is not None else config.TESSERACT_CHAR_WHITELIST
    bl = char_blacklist if char_blacklist is not None else config.TESSERACT_CHAR_BLACKLIST

    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
    tess_base = f'--tessdata-dir "{config.TESSDATA_DIR}"' if config.TESSDATA_DIR else ""

    psm_candidates = list(config.TESSERACT_PSM_CANDIDATES)
    if not psm_candidates:
        log_warning(log, "No PSM candidates configured, falling back to PSM 3.")
        psm_candidates = [3]

    attempts: list[PsmAttempt] = []
    errors: list[Exception] = []
    early_exited = False
    best: PsmAttempt | None = None

    use_parallel = config.PSM_PARALLEL and len(psm_candidates) > 1
    log.info(
        f"OCR started: lang={lang}, dpi={effective_dpi}, "
        f"PSM candidates={psm_candidates}, parallel={use_parallel}"
    )

    with stage_timer(log, "OCR"):
        if len(psm_candidates) == 1:
            attempt, error = _safe_run_ocr(
                ocr_image, lang, psm_candidates[0], tess_base,
                effective_dpi, wl, bl, psm=psm_candidates[0],
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

            executor = ThreadPoolExecutor(max_workers=max_workers)
            futures = {
                executor.submit(
                    _safe_run_ocr, ocr_image, lang, psm, tess_base,
                    effective_dpi, wl, bl, psm=psm,
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
                            if best is None or attempt.mean_confidence > best.mean_confidence:
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
                executor.shutdown(wait=False, cancel_futures=True)

        else:
            for psm in psm_candidates:
                attempt, error = _safe_run_ocr(
                    ocr_image, lang, psm, tess_base,
                    effective_dpi, wl, bl, psm=psm,
                )
                if attempt is not None:
                    attempts.append(attempt)
                    if best is None or attempt.mean_confidence > best.mean_confidence:
                        best = attempt
                else:
                    errors.append(error)

                if best is not None and best.mean_confidence >= config.PSM_EARLY_EXIT_CONFIDENCE:
                    early_exited = True
                    log.info(
                        f"Early-exit: confidence {best.mean_confidence:.1f}% "
                        f">= threshold {config.PSM_EARLY_EXIT_CONFIDENCE}%, "
                        f"remaining PSMs skipped."
                    )
                    break

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
