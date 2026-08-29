import os
import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont


def _has_tesseract() -> bool:
    """Mirrors validate_tesseract_environment()'s resolution logic
    instead of only checking PATH: config.TESSERACT_CMD is either a
    literal 'tesseract' (resolved via PATH at call time) or an
    explicit path (bundled Windows binary, DEV_TESSERACT_OVERRIDE, or
    the TESSERACT_CMD env var) that must exist on disk. Checking PATH
    alone gives a false "available" on Windows dev checkouts where
    Tesseract is on PATH but config.py still points at the
    not-yet-bundled scanlayer/bin/tesseract/tesseract.exe.
    """
    from scanlayer import config

    cmd = config.TESSERACT_CMD
    if cmd == "tesseract":
        return shutil.which("tesseract") is not None
    return os.path.exists(cmd) and os.access(cmd, os.X_OK)


requires_tesseract = pytest.mark.skipif(
    not _has_tesseract(),
    reason=(
        "tesseract binary not found at config.TESSERACT_CMD "
        "(see config.py for how to point it at your install)"
    ),
)


@pytest.fixture
def text_image(tmp_path):
    """A synthetic image with real, OCR-able black text on white,
    used for integration tests that need Tesseract to find something.
    """
    img = Image.new("RGB", (800, 300), "white")
    draw = ImageDraw.Draw(img)
    try:
        from scanlayer.pdf.fonts import _UNICODE_TTF_PATH
        font = ImageFont.truetype(_UNICODE_TTF_PATH, 48)
    except OSError:
        font = ImageFont.load_default()
    draw.text((40, 100), "INVOICE 12345", fill="black", font=font)
    path = tmp_path / "invoice.png"
    img.save(path, dpi=(300, 300))
    return str(path)


@pytest.fixture
def blank_image(tmp_path):
    """A near-uniform white image, should trip the blank-page gate."""
    img = Image.new("RGB", (600, 400), "white")
    path = tmp_path / "blank.png"
    img.save(path, dpi=(300, 300))
    return str(path)


@pytest.fixture(autouse=True)
def _reset_config():
    """Every test gets a clean `config` module state, so `configure()`
    calls in one test can't leak into another.
    """
    from scanlayer import config

    original = dict(config.get_settings())
    yield
    config.configure(**original)
