"""
Runtime pipeline errors, distinct from `utils.validators.ValidationError`.

Validators catch problems BEFORE any processing starts (missing file, missing
Tesseract binary...). This module covers failures that happen DURING
processing, once every precondition was already confirmed valid: a page
that times out, a language pack that fails to load, a PDF write that dies
mid-way. These are not user mistakes and not pre-flight environment issues,
they're operational failures that still need to surface loudly instead of
being swallowed into a silent "0 words, exit 0" success.
"""


class PipelineError(Exception):
    """Base class: a processing stage started but failed."""


class OCRProcessingError(PipelineError):
    """Every configured PSM candidate failed for a real reason (timeout,
    Tesseract crash, corrupted language data), not just 'no text found'."""


class PDFBuildError(PipelineError):
    """The PDF could not be written (disk full, permissions, corrupted
    background image, reportlab failure)."""


class BlankPageDetectedError(PipelineError):
    """The source image is near-uniform (very low grayscale contrast),
    almost certainly a blank/white page rather than genuine content.

    Raised BEFORE the expensive OCR pipeline runs (not after), since the
    signal (image std) is available early in preprocessing and there is
    no point running denoising/CLAHE/upscale/4-way Tesseract on a page
    that is, with high confidence, blank.

    Not raised as a silent "0 words" success: a blank page is often
    unexpected (wrong file, failed scan, camera fired on nothing) and
    deserves an explicit signal rather than a quietly empty PDF. Pass
    `force=True` to `convert()` (or `--force` on the CLI) to generate
    the PDF anyway, background image only, no fabricated text layer.
    """
