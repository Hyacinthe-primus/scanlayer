"""
Central configuration for scanlayer.

Resolves Tesseract path at import time (TESSERACT_CMD / TESSDATA_DIR),
overridable at runtime via configure(). All tunable OCR, preprocessing,
and PDF-output parameters are defined here.
"""

import json
import os
import platform
import shutil
import sys

if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEV_TESSERACT_OVERRIDE = None


def _find_bundled_tesseract():
    """Look for a Tesseract binary bundled next to this package."""
    exe_name = "tesseract.exe" if platform.system() == "Windows" else "tesseract"
    cmd = os.path.join(BASE_DIR, "bin", "tesseract", exe_name)
    if os.path.exists(cmd):
        return cmd, os.path.join(BASE_DIR, "bin", "tesseract", "tessdata")
    return None, None


def _find_system_tesseract():
    """Find Tesseract on PATH or common install locations."""
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path

    system = platform.system()
    if system == "Windows":
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
    elif system == "Darwin":
        candidates = [
            "/opt/homebrew/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]
    else:
        candidates = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return "tesseract"


# Resolution order: env var > DEV_TESSERACT_OVERRIDE > bundled > PATH
_env_override = os.environ.get("TESSERACT_CMD", "").strip()
if _env_override:
    TESSERACT_CMD = _env_override
    TESSDATA_DIR = os.environ.get("TESSDATA_PREFIX") or None
elif DEV_TESSERACT_OVERRIDE and os.path.exists(DEV_TESSERACT_OVERRIDE):
    TESSERACT_CMD = DEV_TESSERACT_OVERRIDE
    TESSDATA_DIR = None
else:
    _bundled_cmd, _bundled_tessdata = _find_bundled_tesseract()
    if _bundled_cmd:
        TESSERACT_CMD = _bundled_cmd
        TESSDATA_DIR = _bundled_tessdata
    else:
        TESSERACT_CMD = _find_system_tesseract()
        TESSDATA_DIR = None


DEFAULT_OCR_LANG = "fra+eng"
DEFAULT_DPI = 300
OCR_UPSCALE_MIN_WIDTH = 2500
OCR_DOWNSCALE_MAX_WIDTH = 3500
MIN_WORD_CONFIDENCE = 35

PSM_EARLY_EXIT_CONFIDENCE = 80.0

# Tesseract releases the GIL, so threads give real speedup.
PSM_PARALLEL = True
PSM_MAX_WORKERS = None

OCR_TIMEOUT_SECONDS = 45

# PSM candidates: engine.py runs one pass per PSM, keeps highest mean confidence.
#   3=fully automatic, 4=single column, 6=uniform block, 11=sparse text
TESSERACT_PSM_CANDIDATES = [3, 4, 6, 11]

# 0=legacy, 1=LSTM, 2=both, 3=auto. 1 is best accuracy/speed tradeoff.
TESSERACT_OEM = 1

TESSERACT_CHAR_BLACKLIST = None
TESSERACT_CHAR_WHITELIST = None

DROP_NON_PRINTABLE_WORDS = True


MULTI_COLUMN_DETECTION = True
COLUMN_MIN_GUTTER_FRACTION = 0.03
COLUMN_FULL_WIDTH_LINE_FRACTION = 0.62
COLUMN_MIN_LINES_FOR_DETECTION = 6
COLUMN_GUTTER_VOTE_FRACTION = 0.6


APPLY_EXIF_ORIENTATION = True

UNSHARP_AMOUNT = 0.5
UNSHARP_SIGMA = 1.2

PREPROCESS_TRY_ADAPTIVE_THRESHOLD = False

DENOISING_H = 10

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

DESKEW_MIN_PIXELS = 100
DESKEW_MIN_STD = 5.0
DESKEW_MIN_ANGLE = 0.3
DESKEW_MAX_ANGLE = 30.0


PDF_JPEG_QUALITY = 82
PDF_ADAPTIVE_COMPRESSION = True
PDF_GRAYSCALE_THRESHOLD = 0.02
FONT_PATH = None

PDF_METADATA = {
    "title": "Image converted to searchable PDF",
    "author": "scanlayer",
    "subject": "Scanned document with OCR text layer",
    "creator": "scanlayer (Tesseract + ReportLab)",
}


LOG_LEVEL = os.environ.get("SCANLAYER_LOG_LEVEL", "INFO")
LOG_TIMING = True


# Runtime configuration API
#
# Precedence: 1. convert() args, 2. configure() calls,
# 3. env vars, 4. defaults here.

_CONFIGURABLE_KEYS = {
    "tesseract_cmd": "TESSERACT_CMD",
    "tessdata_dir": "TESSDATA_DIR",
    "lang": "DEFAULT_OCR_LANG",
    "default_dpi": "DEFAULT_DPI",
    "min_word_confidence": "MIN_WORD_CONFIDENCE",
    "psm_candidates": "TESSERACT_PSM_CANDIDATES",
    "psm_early_exit_confidence": "PSM_EARLY_EXIT_CONFIDENCE",
    "psm_parallel": "PSM_PARALLEL",
    "psm_max_workers": "PSM_MAX_WORKERS",
    "ocr_timeout_seconds": "OCR_TIMEOUT_SECONDS",
    "char_whitelist": "TESSERACT_CHAR_WHITELIST",
    "char_blacklist": "TESSERACT_CHAR_BLACKLIST",
    "jpeg_quality": "PDF_JPEG_QUALITY",
    "font_path": "FONT_PATH",
    "log_level": "LOG_LEVEL",
    "log_timing": "LOG_TIMING",
    "multi_column_detection": "MULTI_COLUMN_DETECTION",
    "column_min_gutter_fraction": "COLUMN_MIN_GUTTER_FRACTION",
    "column_gutter_vote_fraction": "COLUMN_GUTTER_VOTE_FRACTION",
    "column_full_width_line_fraction": "COLUMN_FULL_WIDTH_LINE_FRACTION",
    "column_min_lines_for_detection": "COLUMN_MIN_LINES_FOR_DETECTION",
}


def configure(**overrides) -> None:
    """Override configuration at runtime.

    Tesseract is usually found automatically. Use tesseract_cmd/tessdata_dir
    only to point at a specific install.

    Example:
        scanlayer.configure(lang="eng", min_word_confidence=50)

    Raises ValueError on unknown keywords.
    """
    unknown = set(overrides) - set(_CONFIGURABLE_KEYS)
    if unknown:
        raise ValueError(
            f"configure() got unknown option(s): {sorted(unknown)}. "
            f"Valid options: {sorted(_CONFIGURABLE_KEYS)}"
        )

    module = sys.modules[__name__]
    for key, value in overrides.items():
        attr = _CONFIGURABLE_KEYS[key]
        setattr(module, attr, value)

    if "log_level" in overrides:
        from scanlayer.utils.logger import set_log_level
        set_log_level(overrides["log_level"])


def get_settings() -> dict:
    """Return the current value of every configurable setting."""
    module = sys.modules[__name__]
    return {key: getattr(module, attr) for key, attr in _CONFIGURABLE_KEYS.items()}


def load_config_file(path: str) -> dict:
    """Read a .yaml/.yml/.json config profile and return its contents.

    Raises FileNotFoundError if path doesn't exist, ValueError for
    unsupported extension or unparseable content.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ValueError(
                "Reading a .yaml/.yml config file requires PyYAML "
                "('pip install pyyaml'). Use a .json config file instead."
            ) from exc
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"Could not parse YAML config file {path}: {exc}") from exc
    elif ext == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSON config file {path}: {exc}") from exc
    else:
        raise ValueError(
            f"Unsupported config file extension: {ext!r} (expected .yaml, "
            f".yml, or .json)"
        )

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {path} must contain a mapping of setting "
            f"names to values at the top level, got {type(data).__name__}."
        )
    return data


def configure_from_file(path: str) -> dict:
    """Load config file and apply it via configure(). Returns applied settings.

    Example:
        scanlayer.configure_from_file("profile.yaml")
    """
    settings = load_config_file(path)
    configure(**settings)
    return settings
