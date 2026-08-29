"""
Input and runtime environment validation.

Centralizes pre-checks to fail early with clear messages when prerequisites
are missing. Each validator raises a specific ValidationError subclass.
"""

from __future__ import annotations

import os
from pathlib import Path

from scanlayer import config
from scanlayer.utils.logger import get_logger

log = get_logger(__name__)


class ValidationError(Exception):
    """Base for user-facing validation errors with actionable messages."""


class InputFileError(ValidationError):
    """The input file does not exist, is not readable, or has an unsupported format."""


class OutputPathError(ValidationError):
    """The output path is invalid (parent directory missing,
    insufficient permissions, etc.)."""


class TesseractEnvironmentError(ValidationError):
    """Tesseract is missing, not found on PATH, or its tessdata is missing."""


class DependencyError(ValidationError):
    """A required Python dependency is not installed or broken."""


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp",
    ".gif", ".webp", ".jp2", ".j2k",
}

_COMMON_LANG_FILES = {"eng.traineddata", "fra.traineddata", "osd.traineddata"}


def validate_input_file(path: str) -> str:
    """Check that the input file exists, is readable, and has a recognized extension.

    Returns the resolved absolute path. Raises InputFileError on failure.
    """
    if not path or not isinstance(path, str):
        raise InputFileError("No input path provided.")

    p = Path(path).expanduser()
    if not p.exists():
        raise InputFileError(
            f"Input file does not exist: {path}\n"
            f"Check the path and make sure you are in the correct directory."
        )
    if not p.is_file():
        raise InputFileError(
            f"The specified path is not a file: {path}"
        )
    if os.access(p, os.R_OK) is False:
        raise InputFileError(
            f"Input file is not readable (permissions): {path}"
        )

    ext = p.suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        log.warning(
            f"Extension '{ext}' not in the supported list "
            f"({', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}). "
            f"Attempting anyway, Pillow will reject if the format is unknown."
        )

    if p.stat().st_size == 0:
        raise InputFileError(f"Input file is empty: {path}")

    return str(p.resolve())


def validate_output_path(path: str, expected_ext: str = ".pdf") -> str:
    """Check that the output path is valid (parent exists, is writable).

    Returns the resolved absolute path. A mismatched extension only logs
    a warning since the file is still written in the requested format.
    Raises OutputPathError on failure.
    """
    if not path or not isinstance(path, str):
        raise OutputPathError("No output path provided.")

    p = Path(path).expanduser()
    if p.exists() and p.is_dir():
        raise OutputPathError(
            f"The output path points to a directory, not a file: {path}"
        )

    parent = p.parent
    if not parent.exists():
        raise OutputPathError(
            f"Parent directory of the output file does not exist: {parent}\n"
            f"Create it first, or fix the output path."
        )
    if os.access(parent, os.W_OK) is False:
        raise OutputPathError(
            f"Output directory is not writable: {parent}"
        )

    ext = p.suffix.lower()
    if ext != expected_ext.lower():
        log.warning(
            f"Output path does not have the {expected_ext} extension "
            f"('{ext}'). The file will still be written in the "
            f"requested format, but the extension is misleading."
        )

    return str(p.resolve())


def validate_tesseract_environment() -> None:
    """Check that Tesseract is installed and tessdata is available.

    Raises TesseractEnvironmentError with an actionable message if not.
    """
    cmd = config.TESSERACT_CMD

    if cmd not in ("tesseract",):
        if not os.path.exists(cmd):
            raise TesseractEnvironmentError(
                f"Tesseract not found at the configured location: {cmd}\n"
                f"Check config.DEV_TESSERACT_OVERRIDE, the "
                f"TESSERACT_CMD environment variable, or scanlayer.configure().\n"
                f"Install Tesseract via your package manager "
                f"(apt install tesseract-ocr / brew install tesseract), "
                f"or the Tesseract installer on Windows."
            )
        if not os.access(cmd, os.X_OK):
            raise TesseractEnvironmentError(
                f"Tesseract is not executable: {cmd} (permissions)"
            )
    else:
        import shutil
        resolved = shutil.which("tesseract")
        if resolved is None:
            raise TesseractEnvironmentError(
                "Tesseract not found on the system PATH.\n"
                "Install Tesseract via your package manager "
                "(apt install tesseract-ocr / brew install tesseract / "
                "or download the tesseract-ocr installer on Windows).\n"
                "Or override the path via the TESSERACT_CMD=/path/to/tesseract "
                "environment variable."
            )

    if config.TESSDATA_DIR:
        td = Path(config.TESSDATA_DIR)
        if not td.exists():
            raise TesseractEnvironmentError(
                f"Configured tessdata directory does not exist: {td}"
            )
        present = {f.name for f in td.glob("*.traineddata")}
        missing_common = _COMMON_LANG_FILES - present
        if missing_common:
            log.warning(
                f"Missing tessdata for: {', '.join(sorted(missing_common))}. "
                f"OCR in those languages will fail."
            )


def validate_image_readable(path: str) -> None:
    """Open the image with Pillow to verify it is not corrupted.

    Raises InputFileError if Pillow cannot identify it.
    """
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()  # does not load into memory, only checks the header
    except Exception as exc:
        raise InputFileError(
            f"Unreadable or corrupted image: {path}\n"
            f"Pillow could not identify it: {type(exc).__name__}: {exc}"
        ) from exc


def validate_all(
    input_path: str, output_path: str, output_format: str = "pdf"
) -> tuple[str, str]:
    """Run all validations in order of fastest first.

    Returns (input_abs, output_abs) if everything is OK.
    """
    input_abs = validate_input_file(input_path)
    validate_tesseract_environment()
    output_abs = validate_output_path(output_path, expected_ext=f".{output_format}")
    validate_image_readable(input_abs)
    return input_abs, output_abs
