import json

import pytest

from scanlayer.ocr.engine import Word
from scanlayer.ocr.export import export_words, to_hocr, to_json, to_text, to_tsv


def _sample_words():
    return [
        Word(text="Hello", x=10, y=10, width=50, height=20, confidence=95.0, line_id=1),
        Word(text="world", x=65, y=10, width=50, height=20, confidence=90.0, line_id=1),
        Word(text="Second", x=10, y=40, width=60, height=20, confidence=80.0, line_id=2),
    ]


def test_to_text_groups_by_line():
    text = to_text(_sample_words())
    assert text == "Hello world\nSecond"


def test_to_text_splits_same_line_id_across_column_gutter():
    # Regression: a single Tesseract `line_id` can span a column gutter
    # (Tesseract's own column-blind line detection fusing a row, or two
    # column-reordered segments becoming adjacent). Geometrically distinct
    # segments sharing a `line_id` must not be silently re-merged.
    # Geometry matches the real two-column boundary repro case.
    words = [
        Word(text="APPLE", x=100.0, y=426.0, width=89.0, height=20.0,
             confidence=95.0, line_id=1007001),
        Word(text="BANANA", x=206.0, y=426.0, width=106.0, height=20.0,
             confidence=95.0, line_id=1007001),
        Word(text="CHERRY", x=329.0, y=426.0, width=107.0, height=20.0,
             confidence=95.0, line_id=1007001),
        Word(text="ZEBRA", x=901.0, y=426.0, width=84.0, height=20.0,
             confidence=95.0, line_id=1007001),
        Word(text="YAK", x=1001.0, y=426.0, width=51.0, height=20.0,
             confidence=95.0, line_id=1007001),
        Word(text="WALRUS", x=1061.0, y=426.0, width=117.0, height=20.0,
             confidence=95.0, line_id=1007001),
    ]
    assert to_text(words) == "APPLE BANANA CHERRY\nZEBRA YAK WALRUS"


def test_to_text_keeps_normal_wide_spacing_merged():
    # A wide-ish but ordinary inter-word gap (well under the gutter
    # threshold relative to line height) must still merge into one line,
    # so the new geometric guard doesn't over-split legitimate text.
    words = [
        Word(text="a", x=0.0, y=0.0, width=10.0, height=20.0, confidence=95.0, line_id=1),
        Word(text="b", x=50.0, y=0.0, width=10.0, height=20.0, confidence=95.0, line_id=1),
    ]
    assert to_text(words) == "a b"


def test_to_hocr_splits_same_line_id_across_column_gutter():
    words = [
        Word(text="APPLE", x=100.0, y=426.0, width=89.0, height=20.0,
             confidence=95.0, line_id=1007001),
        Word(text="ZEBRA", x=901.0, y=426.0, width=84.0, height=20.0,
             confidence=95.0, line_id=1007001),
    ]
    hocr = to_hocr(words, image_width=1700, image_height=800)
    assert hocr.count('class="ocr_line"') == 2


def test_to_text_empty_words():
    assert to_text([]) == ""


def test_to_json_schema():
    payload = json.loads(
        to_json(_sample_words(), mean_confidence=88.33, best_psm=3,
                language_used="eng", image_width=800, image_height=600)
    )
    assert payload["text"] == "Hello world\nSecond"
    assert payload["mean_confidence"] == 88.33
    assert payload["best_psm"] == 3
    assert payload["language"] == "eng"
    assert len(payload["words"]) == 3
    assert payload["words"][0]["bbox"] == [10.0, 10.0, 60.0, 30.0]


def test_to_tsv_header_and_row_count():
    tsv = to_tsv(_sample_words())
    lines = tsv.split("\n")
    assert lines[0] == "text\tconfidence\tx\ty\twidth\theight\tline_id"
    assert len(lines) == 4  # header + 3 words


def test_to_tsv_escapes_tabs_in_text():
    words = [Word(text="a\tb", x=0, y=0, width=10, height=10, confidence=50.0, line_id=1)]
    row = to_tsv(words).split("\n")[1]
    assert row.startswith("a b\t")


def test_to_hocr_contains_words_and_escapes_xml():
    words = [Word(text="A&B", x=0, y=0, width=10, height=10, confidence=99.0, line_id=1)]
    hocr = to_hocr(words, image_width=100, image_height=100, source_name="scan.png")
    assert "A&amp;B" in hocr
    assert "ocr_page" in hocr
    assert "ocrx_word" in hocr


def test_export_words_dispatch_matches_direct_call():
    words = _sample_words()
    assert export_words("txt", words) == to_text(words)


def test_export_words_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported export format"):
        export_words("xml", _sample_words())
