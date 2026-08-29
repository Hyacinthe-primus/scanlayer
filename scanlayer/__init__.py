"""
scanlayer: turn a scanned or photographed document into a searchable PDF,
or export the raw OCR result as text, JSON, TSV, or hOCR.

Usage:

    import scanlayer

    result = scanlayer.convert("invoice.jpg", "invoice.pdf")
    print(result.words_count, result.mean_confidence)

For batch processing, use convert_batch() which handles mixed extensions
and never raises for a single bad file.

Tesseract is located automatically. Use configure() to override:

    scanlayer.configure(tesseract_cmd="/path/to/tesseract", lang="eng")

See scanlayer.config.configure for the full list of options.
"""

from scanlayer.config import configure, configure_from_file, get_settings, load_config_file
from scanlayer.main import BatchResult, ConversionResult, convert, convert_batch, convert_merge

__all__ = [
    "configure", "configure_from_file", "load_config_file", "get_settings",
    "convert", "ConversionResult",
    "convert_batch", "BatchResult",
    "convert_merge",
]

__version__ = "1.0.0"
