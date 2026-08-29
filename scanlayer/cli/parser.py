"""
CLI argument parser for scanlayer.
"""

from __future__ import annotations

import argparse

from scanlayer import config


class _ScanlayerHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Wider help column and wrapping so long option descriptions stay
    readable instead of collapsing into a single dense block."""

    def __init__(self, prog):
        super().__init__(prog, max_help_position=28, width=100)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scanlayer",
        description="Convert a scanned image into a searchable PDF, or export the raw OCR result.",
        formatter_class=_ScanlayerHelpFormatter,
        epilog=(
            "Examples:\n"
            "  scanlayer invoice.jpg -o invoice.pdf\n"
            "  scanlayer invoice.jpg -o invoice.pdf --lang fra --dpi 300\n"
            "  scanlayer invoice.jpg -o invoice.pdf --verbose\n"
            "  scanlayer invoice.jpg -o invoice.pdf --jpeg-quality 90\n"
            "  scanlayer *.jpg -o ./converted/   (batch: -o is a folder)\n"
            "  scanlayer invoice.jpg -o invoice.pdf --orientation none\n"
            "  scanlayer invoice.jpg -o invoice.pdf --orientation 10\n"
            "  scanlayer invoice.jpg -o invoice.json --format json\n"
            "  scanlayer invoice.jpg -o invoice.pdf --debug-image\n"
            "  scanlayer invoice.jpg --debug-image   (no -o: writes "
            "invoice.pdf + invoice_debug.png next to the input)\n"
            "  scanlayer invoice.jpg -o invoice.pdf --font ./MyFont.ttf\n"
            "  scanlayer *.jpg -o ./converted/ --dry-run   (validate, no OCR)\n"
            "\n"
            "Exit codes:\n"
            "  0 = success\n"
            "  1 = user error (file not found, invalid format, blank page "
            "without --force...)\n"
            "  2 = environment error (Tesseract missing...)\n"
            "  3 = unexpected error (bug)\n"
            "  4 = processing error (OCR/PDF stage failed for a real reason)\n"
            "  5 = partial batch failure (multiple inputs, some failed)\n"
        ),
    )
    parser.add_argument(
        "input", nargs="+",
        help="Path(s) to the source image(s) (jpg, png, tiff...). "
             "Give several and point -o at a folder to batch-convert.",
    )

    io_group = parser.add_argument_group("input / output")
    io_group.add_argument(
        "-o", "--output", required=False, default=None,
        help="Output PDF path (single input) or folder (multiple "
             "inputs). If omitted, each output is written next to its "
             "own input, same directory/stem, extension from --format. "
             "Required when --merge is used.",
    )
    io_group.add_argument(
        "--format", choices=["pdf", "txt", "json", "tsv", "hocr"], default="pdf",
        help="Output format. 'pdf' (default) builds a searchable PDF. "
             "'txt', 'json', 'tsv', 'hocr' export the raw OCR result "
             "instead, no PDF is built. See the README for each "
             "format's schema.",
    )
    io_group.add_argument(
        "--merge", action="store_true",
        help="Combine all inputs into ONE multi-page PDF instead of "
             "one output per input. Works with multiple image "
             "arguments (page order = argument order) or a single "
             "multi-page PDF input. Only with --format pdf. -o must "
             "be a file path, not a folder.",
    )
    io_group.add_argument(
        "--config", default=None,
        help="Path to a .yaml/.yml/.json config profile applied via "
             "configure() before any other option (CLI flags still "
             "override it). Keys match configure()'s kwargs, e.g. "
             '{"tesseract_cmd": "/usr/bin/tesseract", "lang": "eng"}. '
             "See scanlayer.config.configure for the full key list.",
    )

    ocr_group = parser.add_argument_group("OCR options")
    ocr_group.add_argument(
        "--lang", default=None,
        help="Tesseract language(s), e.g. fra, eng, fra+eng "
             "(default: fra+eng)",
    )
    ocr_group.add_argument(
        "--dpi", type=int, default=None,
        help="DPI to use if not detectable in the image",
    )
    ocr_group.add_argument(
        "--psm", type=int, default=None, metavar="N",
        help="Force a single page segmentation mode instead of trying "
             "TESSERACT_PSM_CANDIDATES (default [3, 4, 6, 11]) and "
             "keeping the highest mean confidence. Runs exactly one "
             "OCR pass, no auto-selection. Common values: 3 (fully "
             "automatic), 4 (single column), 6 (single uniform block, "
             "e.g. receipts), 7 (single line), 11 (sparse/scattered "
             "text, e.g. order forms). 0 and 2 are rejected, they "
             "produce no OCR text.",
    )
    ocr_group.add_argument(
        "--orientation", default=None,
        help="Orientation correction mode. Omit for automatic "
             "detection (EXIF + OSD + deskew, the default). 'none' "
             "disables correction and uses the image as loaded. Or "
             "give a precise clockwise angle in degrees (e.g. '10' "
             "or '-3.5') to skip auto-detection.",
    )
    ocr_group.add_argument(
        "--no-column-detection", action="store_true",
        help="Disable multi-column reading-order detection for this "
             "run, e.g. if it misfires on a specific document. "
             "Equivalent to configure(multi_column_detection=False).",
    )
    ocr_group.add_argument(
        "--min-confidence", type=int, default=None,
        help="Confidence threshold (0-100) below which a word is "
             f"dropped (default: {config.MIN_WORD_CONFIDENCE})",
    )
    ocr_group.add_argument(
        "--whitelist", default=None,
        help="Allowed characters for OCR, e.g. "
             "'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'",
    )
    ocr_group.add_argument(
        "--blacklist", default=None,
        help="Forbidden characters for OCR, e.g. '|`~'",
    )
    ocr_group.add_argument(
        "--font", default=None,
        help="TTF font to embed in the invisible text layer, "
             "overriding the automatic pick (CJK CID font / bundled "
             "DejaVu Sans / Helvetica fallback based on --lang). "
             "Useful for scripts the auto-pick misses (Arabic, "
             "Hebrew, Thai, Devanagari...).",
    )

    pdf_group = parser.add_argument_group("PDF output")
    pdf_group.add_argument(
        "--jpeg-quality", type=int, default=None,
        help=f"JPEG quality for the background (default: {config.PDF_JPEG_QUALITY})",
    )
    pdf_group.add_argument(
        "--title", default=None,
        help="Title of the PDF document (metadata)",
    )
    pdf_group.add_argument(
        "--author", default=None,
        help="Author of the PDF document (metadata)",
    )
    pdf_group.add_argument(
        "--subject", default=None,
        help="Subject of the PDF document (metadata)",
    )
    pdf_group.add_argument(
        "--force", action="store_true",
        help="Generate the PDF even if the source image looks "
             "blank/near-uniform (background image only, no text "
             "layer, no OCR run). Without this, a likely-blank page "
             "raises an error instead of silently producing an "
             "empty PDF.",
    )

    diag_group = parser.add_argument_group("diagnostics")
    diag_group.add_argument(
        "--debug-image", action="store_true",
        help="Also save a '<output-stem>_debug.png' next to the "
             "output, showing detected word boxes color-coded by "
             "confidence, per-word confidence scores, and run "
             "metadata. Works with any --format.",
    )
    diag_group.add_argument(
        "--dry-run", action="store_true",
        help="Validate the batch without running OCR or writing any "
             "output: checks inputs exist and are readable, Tesseract "
             "is reachable, and output paths are writable. Exits with "
             "the same codes as a normal run, so scripts can fail "
             "fast before paying for OCR. Combine with --merge to "
             "validate a merge run instead.",
    )
    diag_group.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )
    diag_group.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress all logs except errors (ERROR level)",
    )

    return parser
