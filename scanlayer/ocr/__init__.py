"""ocr - Tesseract-based OCR engine."""

from scanlayer.ocr.engine import OcrResult, PsmAttempt, Word, extract_words

__all__ = ["extract_words", "Word", "OcrResult", "PsmAttempt"]
