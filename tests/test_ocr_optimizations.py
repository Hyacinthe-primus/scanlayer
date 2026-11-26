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


def _fake_tsv_rows():
    return [
        ["1", "1", "1", "1", "1", "1", "0", "0", "10", "10", "90", "word"],
    ]


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
        seen["mode"] = image.mode
        return _osd()

    monkeypatch.setattr(enhance.pytesseract, "image_to_osd", fake_osd)
    enhance._fix_gross_orientation(Image.new("RGB", (3000, 2000), "white"))

    assert max(seen["size"]) <= enhance._OSD_THUMBNAIL_MAX_DIM
    assert seen["mode"] == "L"


def test_osd_receives_grayscale_input(monkeypatch):
    seen = {}

    def fake_osd(image, **kwargs):
        seen["mode"] = image.mode
        return {
            "rotate": 90,
            "orientation_conf": 10.0,
            "script": "Latin",
        }

    monkeypatch.setattr(enhance.pytesseract, "image_to_osd", fake_osd)
    result, rotation = enhance._fix_gross_orientation(
        Image.new("RGB", (300, 500), "white")
    )

    assert seen["mode"] == "L"
    assert rotation == 90
    assert result.size == (500, 300)


def test_exif_orientation_is_preserved_with_grayscale_osd(monkeypatch):
    seen = {}

    def fake_osd(image, **kwargs):
        seen["mode"] = image.mode
        return _osd(rotate=0)

    monkeypatch.setattr(enhance.pytesseract, "image_to_osd", fake_osd)
    image = Image.new("RGB", (100, 200), "white")
    image.getexif()[0x0112] = 6

    result = enhance.preprocess(image, dpi=300, orientation=None)

    assert result.exif_orientation_applied == 6
    assert result.gross_rotation_applied == 0
    assert seen["mode"] == "L"
    result.background.close()


def test_osd_timeout_is_graceful(monkeypatch):
    seen = {}

    def fake_osd(image, **kwargs):
        seen["mode"] = image.mode
        raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr(enhance.pytesseract, "image_to_osd", fake_osd)
    image = Image.new("RGB", (100, 100), "white")
    result, rotation = enhance._fix_gross_orientation(image)

    assert result is image
    assert rotation == 0
    assert seen["mode"] == "L"


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


def test_tesseract_command_preserves_filter_spaces(monkeypatch):
    monkeypatch.setattr(config, "TESSDATA_DIR", None)
    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
    blacklist = "|`~"
    cmd = engine._tesseract_command(
        "page.bmp", "out", "eng", 3, 300, whitelist, blacklist,
    )

    whitelist_arg = "tessedit_char_whitelist=" + whitelist
    blacklist_arg = "tessedit_char_blacklist=" + blacklist
    assert cmd[cmd.index(whitelist_arg) - 1] == "-c"
    assert cmd[cmd.index(blacklist_arg) - 1] == "-c"
    assert whitelist_arg.endswith(" ")


def test_zero_confidence_whitelist_words_are_kept_only_when_allowed(monkeypatch):
    rows = [
        ["1", "1", "1", "1", "1", "1", "0", "0", "10", "10", "0", "quartz"],
        ["1", "1", "1", "1", "1", "1", "0", "0", "10", "10", "0", "quartz,"],
        ["1", "1", "1", "1", "1", "1", "0", "0", "10", "10", "0", "bad1"],
        ["1", "1", "1", "1", "1", "1", "0", "0", "10", "10", "95", "kept"],
    ]
    monkeypatch.setattr(engine, "_run_tesseract_tsv", lambda *a, **k: rows)

    attempt = engine._run_ocr(
        "fake.bmp", "eng", 3, 300,
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ", None,
    )

    assert [word.text for word in attempt.words] == ["quartz", "kept"]


def test_zero_confidence_blacklist_keeps_punctuation_only_words(monkeypatch):
    rows = [
        ["1", "1", "1", "1", "1", "1", "0", "0", "10", "10", "0", "quartz"],
        ["1", "1", "1", "1", "1", "1", "0", "0", "10", "10", "95", "kept"],
    ]
    monkeypatch.setattr(engine, "_run_tesseract_tsv", lambda *a, **k: rows)

    attempt = engine._run_ocr("fake.bmp", "eng", 3, 300, None, ".")

    assert [word.text for word in attempt.words] == ["quartz", "kept"]


def test_zero_confidence_blacklist_drops_alphanumeric_words(monkeypatch):
    rows = [
        ["1", "1", "1", "1", "1", "1", "0", "0", "10", "10", "0", "gebra"],
        ["1", "1", "1", "1", "1", "1", "0", "0", "10", "10", "95", "kept"],
    ]
    monkeypatch.setattr(engine, "_run_tesseract_tsv", lambda *a, **k: rows)

    attempt = engine._run_ocr("fake.bmp", "eng", 3, 300, None, "z")

    assert [word.text for word in attempt.words] == ["kept"]


@requires_tesseract
def test_whitelist_with_space_preserves_word_boundaries():
    image = Image.new("RGB", (900, 220), "white")
    draw = ImageDraw.Draw(image)
    try:
        from scanlayer.pdf.fonts import _UNICODE_TTF_PATH
        font = ImageFont.truetype(_UNICODE_TTF_PATH, 48)
    except OSError:
        font = ImageFont.load_default()
    draw.text((30, 70), "Hello World", fill="black", font=font)

    config.configure(
        psm_candidates=[6], psm_parallel=False, min_word_confidence=35,
    )
    result = engine.extract_words(
        np.array(image.convert("L")), 1.0, 300, lang="eng",
        char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ",
    )

    assert len(result.words) >= 2


def test_extract_words_encodes_image_once_and_cleans_up(monkeypatch):
    created = []
    original_write = engine._write_ocr_image

    def write(image):
        path = original_write(image)
        created.append(path)
        return path

    monkeypatch.setattr(engine, "_write_ocr_image", write)
    seen = []

    def fake_tsv(image, *args, **kwargs):
        seen.append((image, os.path.exists(image)))
        return _fake_tsv_rows()

    monkeypatch.setattr(engine, "_run_tesseract_tsv", fake_tsv)
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

    def fail(image, *args, **kwargs):
        raise engine.pytesseract.TesseractError(1, "boom")

    monkeypatch.setattr(engine, "_run_tesseract_tsv", fail)
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

    def slow_ok(image, *args, **kwargs):
        time.sleep(0.05)
        return _fake_tsv_rows()

    monkeypatch.setattr(engine, "_run_tesseract_tsv", slow_ok)
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
