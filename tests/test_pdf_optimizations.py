"""Regression tests for PDF background embedding and searchability."""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image, ImageDraw
from pypdf import PdfReader

from scanlayer.ocr.engine import Word
from scanlayer.pdf.builder import (
    PageInput,
    _is_mostly_binary,
    _is_mostly_grayscale,
    build_searchable_pdf,
    build_searchable_pdf_multipage,
)
from scanlayer.utils.errors import PDFBuildError


def _text_words():
    return [
        Word(text="Hello", x=20, y=20, width=80, height=24, confidence=95.0, line_id=1),
        Word(text="world", x=110, y=20, width=90, height=24, confidence=95.0, line_id=1),
    ]


def _binary_image(width=400, height=600):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for y in range(40, height - 40, 40):
        draw.rectangle([30, y, width - 30, y + 12], fill="black")
    return image


def _grayscale_image(width=400, height=600):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for y in range(40, height - 40, 80):
        draw.rectangle([30, y, width - 30, y + 12], fill=(40, 40, 40))
    return image


def _color_image(width=400, height=600):
    image = Image.new("RGB", (width, height), (235, 242, 250))
    draw = ImageDraw.Draw(image)
    for y in range(40, height - 40, 40):
        draw.rectangle([30, y, width - 30, y + 12], fill=(20, 80, 160))
    return image


def _page(path):
    return PdfReader(str(path)).pages[0]


def test_pdf_page_dimensions_and_text_searchability(tmp_path):
    out = tmp_path / "binary.pdf"
    build_searchable_pdf(
        _binary_image(), _text_words(), str(out), dpi=200, lang="eng",
    )
    page = _page(out)
    assert float(page.mediabox.width) == pytest.approx(400 * 72 / 200, abs=0.01)
    assert float(page.mediabox.height) == pytest.approx(600 * 72 / 200, abs=0.01)
    text = " ".join((page.extract_text() or "").split())
    assert "Hello world" in text


def test_binary_image_embeds_lossless_grayscale(tmp_path):
    source = _binary_image()
    out = tmp_path / "binary.pdf"
    build_searchable_pdf(source, [], str(out), dpi=200, lang="eng")

    images = list(_page(out).images)
    assert len(images) == 1
    embedded = images[0].image
    assert embedded.mode == "L"
    assert np.array_equal(
        np.asarray(embedded), np.asarray(source.convert("L")),
    )


def test_grayscale_image_embeds_lossless(tmp_path):
    source = _grayscale_image()
    out = tmp_path / "gray.pdf"
    build_searchable_pdf(source, [], str(out), dpi=200, lang="eng")

    embedded = list(_page(out).images)[0].image
    assert embedded.mode == "L"
    assert np.array_equal(
        np.asarray(embedded), np.asarray(source.convert("L")),
    )


def test_binary_color_image_keeps_color_lossless(tmp_path):
    source = _binary_image()
    draw = ImageDraw.Draw(source)
    # A small but non-negligible colored mark: still >80% extreme pixels,
    # but enough color that the image must not be converted to grayscale.
    draw.rectangle([100, 100, 300, 160], fill=(220, 20, 20))
    out = tmp_path / "binary_color.pdf"
    build_searchable_pdf(source, [], str(out), dpi=200, lang="eng")

    embedded = list(_page(out).images)[0].image
    assert embedded.mode == "RGB"
    assert np.array_equal(np.asarray(embedded), np.asarray(source))


def test_color_image_embeds_jpeg_and_preserves_appearance(tmp_path):
    source = _color_image()
    out = tmp_path / "color.pdf"
    build_searchable_pdf(source, [], str(out), dpi=200, lang="eng")

    embedded = list(_page(out).images)[0].image
    assert embedded.mode == "RGB"
    actual = np.asarray(embedded, dtype=np.float32)
    expected = np.asarray(source, dtype=np.float32)
    mean_error = np.abs(actual - expected).mean()
    assert mean_error < 8.0


def test_binary_detection_matches_old_histogram_rule():
    image = Image.new("L", (1000, 1000), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 999, 140], fill=0)  # 14% black -> binary

    arr = np.asarray(image, dtype=np.float32)
    hist, _ = np.histogram(arr, bins=8, range=(0, 256))
    expected = bool((hist[0] + hist[-1]) / arr.size > 0.80)
    assert _is_mostly_binary(image) is expected


def test_binary_detection_with_reused_grayscale_matches_default():
    for image in (_binary_image(), _grayscale_image(), _color_image()):
        grayscale = image if image.mode == "L" else image.convert("L")
        assert _is_mostly_binary(image, grayscale=grayscale) == _is_mostly_binary(image)


def test_grayscale_detection_falls_back_for_ambiguous_sample():
    # 3% colored pixels: sampled fraction is near the 2% threshold, so the
    # exact full-image calculation must decide (and return False).
    arr = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    arr[:30, :, :] = (255, 0, 0)
    image = Image.fromarray(arr)
    assert _is_mostly_grayscale(image) is False


def test_repeated_pdf_generation_same_process(tmp_path):
    for index in range(2):
        out = tmp_path / f"repeat_{index}.pdf"
        build_searchable_pdf(
            _binary_image(), _text_words(), str(out), dpi=200, lang="eng",
        )
        page = _page(out)
        assert "Hello world" in " ".join((page.extract_text() or "").split())
    assert sorted(os.listdir(tmp_path)) == ["repeat_0.pdf", "repeat_1.pdf"]


def test_concurrent_pdf_generation(tmp_path):
    # No process-global ReportLab compression setting is changed, but this
    # catches accidental shared reader/temp-resource state.
    from scanlayer.pdf.fonts import resolve_font

    resolve_font("eng")

    def build(index):
        out = tmp_path / f"concurrent_{index}.pdf"
        build_searchable_pdf(
            _binary_image(), _text_words(), str(out), dpi=200, lang="eng",
        )
        return out

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(build, range(4)))

    for path in paths:
        assert os.path.exists(path)
        assert len(PdfReader(str(path)).pages) == 1


def test_failed_build_raises_without_extra_temp_files(tmp_path):
    bad = Image.new("RGB", (0, 0), "white")
    out = tmp_path / "bad.pdf"
    with pytest.raises(PDFBuildError):
        build_searchable_pdf(bad, [], str(out), dpi=200)
    # No image temp files are created by the direct ImageReader path.
    assert not any(name.endswith((".png", ".jpg", ".bmp")) for name in os.listdir(tmp_path))


def test_multipage_dimensions_order_and_text(tmp_path):
    out = tmp_path / "multi.pdf"
    page_one = PageInput(
        background=_binary_image(400, 600),
        words=[Word("First", 20, 20, 60, 20, 95.0, 1)],
        dpi=200, lang="eng",
    )
    page_two = PageInput(
        background=_binary_image(600, 400),
        words=[Word("Second", 20, 20, 70, 20, 95.0, 1)],
        dpi=200, lang="eng",
    )
    build_searchable_pdf_multipage([page_one, page_two], str(out))

    reader = PdfReader(str(out))
    assert len(reader.pages) == 2
    assert float(reader.pages[0].mediabox.width) == pytest.approx(400 * 72 / 200)
    assert float(reader.pages[0].mediabox.height) == pytest.approx(600 * 72 / 200)
    assert float(reader.pages[1].mediabox.width) == pytest.approx(600 * 72 / 200)
    assert float(reader.pages[1].mediabox.height) == pytest.approx(400 * 72 / 200)
    assert "First" in (reader.pages[0].extract_text() or "")
    assert "Second" in (reader.pages[1].extract_text() or "")


def test_pdf_build_does_not_change_reportlab_ascii85_setting(tmp_path):
    from reportlab import rl_config

    before = rl_config.useA85
    build_searchable_pdf(
        _binary_image(), _text_words(), str(tmp_path / "global.pdf"), dpi=200,
    )
    assert rl_config.useA85 == before


def test_jpeg_background_temp_file_is_removed_after_build(tmp_path, monkeypatch):
    from scanlayer.pdf import builder

    paths = []
    original = builder._as_jpeg_file

    def tracked(image, quality):
        path = original(image, quality)
        paths.append(path)
        return path

    monkeypatch.setattr(builder, "_as_jpeg_file", tracked)
    build_searchable_pdf(
        _color_image(), _text_words(), str(tmp_path / "color.pdf"), dpi=200,
    )

    assert len(paths) == 1
    assert not os.path.exists(paths[0])
    assert str(tmp_path) not in paths[0]


def test_jpeg_background_temp_file_is_removed_on_draw_failure(monkeypatch):
    from scanlayer.pdf import builder

    paths = []
    original = builder._as_jpeg_file

    def tracked(image, quality):
        path = original(image, quality)
        paths.append(path)
        return path

    monkeypatch.setattr(builder, "_as_jpeg_file", tracked)

    with pytest.raises(RuntimeError, match="draw failed"):
        with builder._background_for_pdf(_color_image(), 82) as arg:
            paths.append(arg)
            raise RuntimeError("draw failed")

    assert not os.path.exists(paths[0])


def test_concurrent_jpeg_background_files_are_unique(tmp_path, monkeypatch):
    from scanlayer.pdf import builder

    paths = []
    original = builder._as_jpeg_file

    def tracked(image, quality):
        path = original(image, quality)
        paths.append(path)
        return path

    monkeypatch.setattr(builder, "_as_jpeg_file", tracked)

    from scanlayer.pdf.fonts import resolve_font
    resolve_font("eng")

    def build(index):
        out = tmp_path / f"color_{index}.pdf"
        build_searchable_pdf(_color_image(), _text_words(), str(out), dpi=200)
        return out

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(build, range(4)))

    assert len(paths) == 4
    assert len(set(paths)) == 4
    assert all(not os.path.exists(path) for path in paths)
    assert all(path.exists() for path in outputs)
