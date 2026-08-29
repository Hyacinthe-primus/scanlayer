from scanlayer.layout import columns
from scanlayer.ocr.engine import Word


def _w(text, x, y, width=60, height=20, line_id=1):
    return Word(text=text, x=x, y=y, width=width, height=height,
                confidence=95.0, line_id=line_id)


def test_single_column_unchanged():
    words = [
        _w("hello", 10, 10, line_id=1), _w("world", 80, 10, line_id=1),
        _w("second", 10, 40, line_id=2), _w("row", 90, 40, line_id=2),
    ]
    reordered, count = columns.reorder_reading_order(words, page_width=800)
    assert reordered == words
    assert count == 1


def test_two_column_page_reorders_columns_before_rows():
    # Simulates the real bug: Tesseract fuses same-row left/right words
    # into one line_id, so column separation must come from the
    # horizontal gap, not from line_id boundaries.
    words = [
        _w("Left1", 20, 10, line_id=1), _w("Right1", 500, 10, line_id=1),
        _w("Left2", 20, 40, line_id=2), _w("Right2", 500, 40, line_id=2),
        _w("Left3", 20, 70, line_id=3), _w("Right3", 500, 70, line_id=3),
        _w("Left4", 20, 100, line_id=4), _w("Right4", 500, 100, line_id=4),
        _w("Left5", 20, 130, line_id=5), _w("Right5", 500, 130, line_id=5),
        _w("Left6", 20, 160, line_id=6), _w("Right6", 500, 160, line_id=6),
    ]
    reordered, count = columns.reorder_reading_order(words, page_width=800)
    assert count == 2
    texts = [w.text for w in reordered]
    assert texts == [
        "Left1", "Left2", "Left3", "Left4", "Left5", "Left6",
        "Right1", "Right2", "Right3", "Right4", "Right5", "Right6",
    ]


def test_header_placed_before_columns():
    words = [
        _w("TITLE", 20, 0, width=700, line_id=0),
        _w("Left1", 20, 60, line_id=1), _w("Right1", 500, 60, line_id=1),
        _w("Left2", 20, 90, line_id=2), _w("Right2", 500, 90, line_id=2),
        _w("Left3", 20, 120, line_id=3), _w("Right3", 500, 120, line_id=3),
        _w("Left4", 20, 150, line_id=4), _w("Right4", 500, 150, line_id=4),
        _w("Left5", 20, 180, line_id=5), _w("Right5", 500, 180, line_id=5),
        _w("Left6", 20, 210, line_id=6), _w("Right6", 500, 210, line_id=6),
    ]
    reordered, count = columns.reorder_reading_order(words, page_width=800)
    assert reordered[0].text == "TITLE"
    assert count == 2


def test_short_centered_title_and_footer_do_not_kill_detection():
    # Regression: a centered title/footer whose box overlaps the gutter
    # but which is far narrower than COLUMN_FULL_WIDTH_LINE_FRACTION used
    # to weld both columns into one merged run, so no gutter was found
    # and reading order stayed row-interleaved. Gap voting must survive
    # them, and both spanning lines must land outside the column flow.
    words = [
        _w("QUARTERLY BUSINESS REPORT", 481, 0, width=540, line_id=0),
        _w("Prepared by the FCP Finance Team", 82, 400, width=809, line_id=9),
    ]
    for i in range(1, 7):
        y = i * 30
        words.append(_w(f"Left{i}", 80, y, line_id=i))
        words.append(_w(f"Right{i}", 900, y, line_id=i))
    reordered, count = columns.reorder_reading_order(words, page_width=1700)
    assert count == 2
    texts = [w.text for w in reordered]
    assert texts[:2] == ["QUARTERLY BUSINESS REPORT", "Left1"]
    assert texts[-1] == "Prepared by the FCP Finance Team"
    left = [t for t in texts if t.startswith("Left")]
    right = [t for t in texts if t.startswith("Right")]
    assert texts.index(left[-1]) < texts.index(right[0])


def test_word_count_preserved_always():
    words = [_w(f"w{i}", i * 30, (i % 3) * 40, line_id=i % 3) for i in range(20)]
    reordered, _ = columns.reorder_reading_order(words, page_width=800)
    assert len(reordered) == len(words)
    assert sorted(w.text for w in reordered) == sorted(w.text for w in words)


def test_disabled_via_config_returns_unchanged():
    from scanlayer import config
    config.configure(multi_column_detection=False)
    words = [
        _w("Left1", 20, 10, line_id=1), _w("Right1", 500, 10, line_id=1),
        _w("Left2", 20, 40, line_id=2), _w("Right2", 500, 40, line_id=2),
        _w("Left3", 20, 70, line_id=3), _w("Right3", 500, 70, line_id=3),
        _w("Left4", 20, 100, line_id=4), _w("Right4", 500, 100, line_id=4),
        _w("Left5", 20, 130, line_id=5), _w("Right5", 500, 130, line_id=5),
        _w("Left6", 20, 160, line_id=6), _w("Right6", 500, 160, line_id=6),
    ]
    reordered, count = columns.reorder_reading_order(words, page_width=800)
    assert reordered == words
    assert count == 1


def test_too_few_lines_skips_detection():
    words = [_w("only", 20, 10, line_id=1), _w("two", 500, 40, line_id=2)]
    reordered, count = columns.reorder_reading_order(words, page_width=800)
    assert reordered == words
    assert count == 1


def test_empty_words_returns_empty():
    reordered, count = columns.reorder_reading_order([], page_width=800)
    assert reordered == []
    assert count == 1


def test_split_row_into_segments_respects_gap():
    words = [_w("a", 0, 0, width=50), _w("b", 60, 0, width=50), _w("c", 500, 0, width=50)]
    segments = columns._split_row_into_segments(words, min_gap=100)
    assert len(segments) == 2
    assert [w.text for w in segments[0]] == ["a", "b"]
    assert [w.text for w in segments[1]] == ["c"]


def test_narrow_whitespace_within_line_not_treated_as_gutter():
    # A single sentence with a wide-ish inter-word space should not be
    # split if the gap is under the configured gutter threshold.
    words = [_w("word1", 0, 0, width=50), _w("word2", 70, 0, width=50)]
    segments = columns._split_row_into_segments(words, min_gap=100)
    assert len(segments) == 1
