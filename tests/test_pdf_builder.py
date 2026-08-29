import os

import pytest
from PIL import Image

from scanlayer.ocr.engine import Word
from scanlayer.pdf.builder import _font_size_for_height, build_searchable_pdf
from scanlayer.utils.errors import PDFBuildError


def _bg(w=800, h=600):
    return Image.new("RGB", (w, h), "white")


def test_build_searchable_pdf_writes_nonempty_file(tmp_path):
    out = tmp_path / "out.pdf"
    words = [Word(text="Hi", x=10, y=10, width=40, height=20, confidence=90.0, line_id=1)]
    size = build_searchable_pdf(_bg(), words, str(out), dpi=300, lang="eng")
    assert out.exists()
    assert size == os.path.getsize(out)
    assert size > 0
    with open(out, "rb") as f:
        assert f.read(5) == b"%PDF-"


def test_build_searchable_pdf_empty_word_list_still_builds(tmp_path):
    out = tmp_path / "out.pdf"
    size = build_searchable_pdf(_bg(), [], str(out), dpi=300)
    assert size > 0


def test_degenerate_bbox_word_is_skipped_not_fatal(tmp_path):
    out = tmp_path / "out.pdf"
    words = [
        Word(text="zero-width", x=5, y=5, width=0, height=0, confidence=90.0, line_id=1),
        Word(text="ok", x=5, y=5, width=20, height=10, confidence=90.0, line_id=1),
    ]
    # Must not raise despite the degenerate word.
    size = build_searchable_pdf(_bg(), words, str(out), dpi=300)
    assert size > 0


def test_invalid_page_size_raises_pdfbuilderror(tmp_path):
    out = tmp_path / "out.pdf"
    tiny = Image.new("RGB", (0, 0), "white")
    with pytest.raises(PDFBuildError):
        build_searchable_pdf(tiny, [], str(out), dpi=300)


def test_font_size_for_height_is_clamped():
    assert _font_size_for_height(0, px_to_pt=1.0) >= 1.0
    huge = _font_size_for_height(1_000_000, px_to_pt=1.0)
    assert huge <= 400.0


def test_cjk_lang_produces_valid_pdf_with_cjk_text(tmp_path):
    out = tmp_path / "cjk.pdf"
    words = [Word(text="\u4f60\u597d", x=10, y=10, width=40, height=20,
                   confidence=90.0, line_id=1)]
    size = build_searchable_pdf(_bg(), words, str(out), dpi=300, lang="chi_sim")
    assert size > 0
