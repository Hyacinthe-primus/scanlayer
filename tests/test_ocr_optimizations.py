"""Regression tests for Phase 3A OSD and OCR-image handling."""

import os
import time

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from scanlayer import config
from scanlayer.ocr import engine
from scanlayer.preprocessing import enhance
from scanlayer.utils.errors import OCRProcessingError

from .conftest import requires_tesseract


def _osd(rotate=0, confidence=10.0):
    return {"rotate": rotate, "orientation_conf": confidence}


def _fake_ocr_data():
    return {
        "text": ["word"],
        "conf": [90],
        "width": [10],
        "height": [10],
        "block_num": [1],
        "par_num": [1],
        "line_num": [1],
        "left": [0],
        "top": [0],
    }


def test_osd_uses_configured_tesseract_before_call(monkeypatch):
    config.configure(tesseract_cmd="/custom/tesseract")
    seen = {}

    def fake_osd(image, **kwargs):
        seen["cmd"] = enhance.pytesseract.pytesseract.tesseract_cmd
        seen["kwargs"] = kwargs
        return _osd()

    monkeypatch.setattr(enhance.pytesseract, "image_to_osd", fake_osd)
    result, rotation = enhance._fix_gross_orientation(
        Image.new("RGB", (100, 100), "white")
    )

    assert seen["cmd"] == "/custom/tesseract"
    assert rotation == 0
    assert result.size == (100, 100)


def test_osd_passes_configured_tessdata_and_timeout(monkeypatch):
    config.configure(tessdata_dir="/custom/tessdata", osd_timeout_seconds=12)
    seen = {}

    def fake_osd(image, **kwargs):
        seen.update(kwargs)
        return _osd()

    monkeypatch.setattr(enhance.pytesseract, "image_to_osd", fake_osd)
    enhance._fix_gross_orientation(Image.new("RGB", (100, 100), "white"))

    assert seen["config"] == '--tessdata-dir "/custom/tessdata"'
    assert seen["timeout"] == 12


def test_osd_runs_on_capped_thumbnail(monkeypatch):
    seen = {}

    def fake_osd(image, **kwargs):
        seen["size"] = image.size
        return _osd()

    monkeypatch.setattr(enhance.pytesseract, "image_to_osd", fake_osd)
    enhance._fix_gross_orientation(Image.new("RGB", (3000, 2000), "white"))

    assert max(seen["size"]) <= enhance._OSD_THUMBNAIL_MAX_DIM


def test_osd_timeout_is_graceful(monkeypatch):
    def fake_osd(image, **kwargs):
        raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr(enhance.pytesseract, "image_to_osd", fake_osd)
    image = Image.new("RGB", (100, 100), "white")
    result, rotation = enhance._fix_gross_orientation(image)

    assert result is image
    assert rotation == 0


@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
def test_osd_rotation_application(monkeypatch, rotate):
    monkeypatch.setattr(
        enhance.pytesseract, "image_to_osd",
        lambda image, **kwargs: _osd(rotate=rotate),
    )
    image = Image.new("RGB", (300, 500), "white")
    result, applied = enhance._fix_gross_orientation(image)

    assert applied == rotate
    if rotate in (90, 270):
        assert result.size == (500, 300)
    else:
        assert result.size == (300, 500)


def _text_page():
    image = Image.new("RGB", (1600, 2200), "white")
    draw = ImageDraw.Draw(image)
    try:
        from scanlayer.pdf.fonts import _UNICODE_TTF_PATH
        font = ImageFont.truetype(_UNICODE_TTF_PATH, 34)
    except OSError:
        font = ImageFont.load_default()
    for i in range(40):
        draw.text(
            (100, 80 + i * 50),
            "The quick brown fox jumps over the lazy dog 0123456789",
            fill="black", font=font,
        )
    return image


@requires_tesseract
@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
def test_osd_detects_real_gross_orientation(rotate):
    image = _text_page()
    if rotate:
        image = image.rotate(rotate, expand=True, fillcolor=(255, 255, 255))
    _, applied = enhance._fix_gross_orientation(image)
    assert applied == rotate


def test_extract_words_encodes_image_once_and_cleans_up(monkeypatch):
    created = []
    original_write = engine._write_ocr_image

    def write(image):
        path = original_write(image)
        created.append(path)
        return path

    monkeypatch.setattr(engine, "_write_ocr_image", write)
    seen = []

    def fake_image_to_data(image, **kwargs):
        seen.append((image, os.path.exists(image)))
        return _fake_ocr_data()

    monkeypatch.setattr(engine.pytesseract, "image_to_data", fake_image_to_data)
    config.configure(
        psm_candidates=[3, 4, 6, 11], psm_parallel=False,
        min_word_confidence=0, psm_early_exit_confidence=100.0,
    )

    result = engine.extract_words(
        np.full((20, 20), 255, dtype=np.uint8), 1.0, 300, lang="eng",
    )

    assert len(created) == 1
    assert len(seen) == 4
    assert all(path == created[0] for path, _ in seen)
    assert all(exists for _, exists in seen)
    assert not os.path.exists(created[0])
    assert len(result.attempts) == 4


def test_extract_words_cleans_up_when_all_passes_fail(monkeypatch):
    created = []
    original_write = engine._write_ocr_image

    def write(image):
        path = original_write(image)
        created.append(path)
        return path

    monkeypatch.setattr(engine, "_write_ocr_image", write)

    def fail(image, **kwargs):
        raise engine.pytesseract.TesseractError(1, "boom")

    monkeypatch.setattr(engine.pytesseract, "image_to_data", fail)
    config.configure(psm_candidates=[3, 4], psm_parallel=False)

    with pytest.raises(OCRProcessingError):
        engine.extract_words(
            np.full((20, 20), 255, dtype=np.uint8), 1.0, 300, lang="eng",
        )

    assert len(created) == 1
    assert not os.path.exists(created[0])


def test_extract_words_parallel_early_exit_cleans_up(monkeypatch):
    created = []
    original_write = engine._write_ocr_image

    def write(image):
        path = original_write(image)
        created.append(path)
        return path

    monkeypatch.setattr(engine, "_write_ocr_image", write)

    def slow_ok(image, **kwargs):
        time.sleep(0.05)
        return _fake_ocr_data()

    monkeypatch.setattr(engine.pytesseract, "image_to_data", slow_ok)
    config.configure(
        psm_candidates=[3, 4], psm_parallel=True,
        psm_max_workers=1, psm_early_exit_confidence=0.0,
        min_word_confidence=0,
    )

    result = engine.extract_words(
        np.full((20, 20), 255, dtype=np.uint8), 1.0, 300, lang="eng",
    )

    assert result.early_exited is True
    assert len(created) == 1
    assert not os.path.exists(created[0])


def test_remove_temp_file_retries_transient_windows_lock(monkeypatch, tmp_path):
    path = tmp_path / "ocr.bmp"
    path.write_bytes(b"data")
    calls = []
    real_unlink = os.unlink

    def flaky_unlink(target):
        calls.append(target)
        if len(calls) < 3:
            raise PermissionError("temporarily locked")
        real_unlink(target)

    monkeypatch.setattr(engine.os, "unlink", flaky_unlink)
    engine._remove_temp_file(str(path))

    assert not path.exists()
    assert len(calls) == 3
