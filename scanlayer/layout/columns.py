"""
Multi-column reading-order reconstruction.

Tesseract's block/paragraph numbering frequently interleaves columns on
two-column pages. This module re-derives reading order geometrically:

1. Group words into rows (Tesseract's vertical grouping is reliable).
2. Split rows into segments at wide gaps (catches column fusions).
3. Detect column gutters by per-row gap voting.
4. Assign segments to gutter-delimited bands, sort top-to-bottom.
5. Place full-width lines (headers/footers) before/after columns.

Tuned for prose-style layouts, not tables. Only produces left-to-right
column order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scanlayer import config
from scanlayer.ocr.engine import Word
from scanlayer.utils.logger import get_logger, log_warning

log = get_logger(__name__)


def _group_into_lines(words: list[Word]) -> list[list[Word]]:
    """Group words, preserving Tesseract line order via Word.line_id."""
    lines: list[list[Word]] = []
    current: list[Word] = []
    current_id = None
    for w in words:
        if current_id is None or w.line_id == current_id:
            current.append(w)
        else:
            lines.append(current)
            current = [w]
        current_id = w.line_id
    if current:
        lines.append(current)
    return lines


@dataclass
class _Line:
    words: list[Word]
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def y_center(self) -> float:
        return (self.y_min + self.y_max) / 2


def _line_bounds(words: list[Word]) -> _Line:
    return _Line(
        words=words,
        x_min=min(w.x for w in words),
        x_max=max(w.x + w.width for w in words),
        y_min=min(w.y for w in words),
        y_max=max(w.y + w.height for w in words),
    )


def _split_row_into_segments(
    row_words: list[Word], min_gap: float
) -> list[list[Word]]:
    """Split a Tesseract line into horizontal segments at wide gaps.

    Under column-blind PSMs, Tesseract fuses same-row text from two
    columns into one line. This re-splits on the horizontal gap signal.
    """
    if not row_words:
        return []
    ordered = sorted(row_words, key=lambda w: w.x)
    segments = [[ordered[0]]]
    for prev, w in zip(ordered, ordered[1:]):
        gap = w.x - (prev.x + prev.width)
        if gap >= min_gap:
            segments.append([w])
        else:
            segments[-1].append(w)
    return segments


def _build_segmented_lines(words: list[Word], page_width: float) -> list[_Line]:
    """Group words into rows, then split each row into segments at wide gaps."""
    min_gap = page_width * config.COLUMN_MIN_GUTTER_FRACTION
    segments: list[_Line] = []
    for row in _group_into_lines(words):
        for segment_words in _split_row_into_segments(row, min_gap):
            segments.append(_line_bounds(segment_words))
    return segments


def _row_gap_votes(
    rows: list[list[Word]], page_width: float
) -> tuple[list[int], float]:
    """Build per-x-bin profile counting rows with word gaps covering each bin."""
    bin_w = max(1.0, page_width / 500)
    n_bins = int(page_width / bin_w) + 2
    votes: list[int] = [0] * n_bins
    for row in rows:
        ordered = sorted(row, key=lambda w: w.x)
        voted: set[int] = set()
        for prev, nxt in zip(ordered, ordered[1:]):
            gap_start = prev.x + prev.width
            gap_end = nxt.x
            if gap_end - gap_start < bin_w:
                continue
            lo = max(0, int(gap_start / bin_w))
            hi = min(n_bins - 1, int(gap_end / bin_w))
            for i in range(lo, hi + 1):
                if i not in voted:
                    voted.add(i)
                    votes[i] += 1
    return votes, bin_w


def detect_column_gutters(
    rows: list[list[Word]], page_width: float
) -> list[float]:
    """Return x-positions of detected column gutters, or empty list if single-column.

    rows are RAW Tesseract line groupings (pre-segmentation).
    """
    if not rows or page_width <= 0:
        return []

    min_gap = page_width * config.COLUMN_MIN_GUTTER_FRACTION
    full_width_threshold = page_width * config.COLUMN_FULL_WIDTH_LINE_FRACTION
    segments = [
        seg for row in rows for seg in _split_row_into_segments(row, min_gap)
    ]
    narrow_lines = [
        ln
        for ln in (_line_bounds(s) for s in segments)
        if ln.width <= full_width_threshold
    ]
    if len(narrow_lines) < config.COLUMN_MIN_LINES_FOR_DETECTION:
        return []

    votes, bin_w = _row_gap_votes(rows, page_width)
    strongest = max(votes, default=0)
    if strongest < 2:
        return []
    min_votes = max(2, math.ceil(strongest * config.COLUMN_GUTTER_VOTE_FRACTION))

    tolerance = page_width * 0.005
    gutters: list[float] = []
    start: int | None = None
    for i, v in enumerate([*votes, 0]):
        if v >= min_votes:
            if start is None:
                start = i
            continue
        if start is None:
            continue
        band_start, band_end = start * bin_w, i * bin_w
        start = None
        if band_end - band_start < min_gap:
            continue
        left_support = sum(
            1 for ln in narrow_lines if ln.x_max <= band_start + tolerance
        )
        right_support = sum(
            1 for ln in narrow_lines if ln.x_min >= band_end - tolerance
        )
        if left_support >= 2 and right_support >= 2:
            gutters.append((band_start + band_end) / 2.0)
    return gutters


def reorder_reading_order(
    words: list[Word], page_width: float
) -> tuple[list[Word], int]:
    """Reorder words into left-to-right, top-to-bottom column reading order.

    Returns (words, 1) unchanged if single-column, not enough text,
    no gutters found, or reorder would drop/duplicate words.
    Otherwise returns (reordered_words, column_count).
    """
    if not config.MULTI_COLUMN_DETECTION or not words or page_width <= 0:
        return words, 1

    lines = _build_segmented_lines(words, page_width)
    if len(lines) < config.COLUMN_MIN_LINES_FOR_DETECTION:
        return words, 1

    gutters = detect_column_gutters(_group_into_lines(words), page_width)
    if not gutters:
        return words, 1

    boundaries = [0.0, *gutters, page_width]
    full_width_threshold = page_width * config.COLUMN_FULL_WIDTH_LINE_FRACTION
    crossing_margin = page_width * 0.005

    columns: list[list[_Line]] = [[] for _ in range(len(boundaries) - 1)]
    full_width_lines: list[_Line] = []
    for line in lines:
        crosses_gutter = any(
            line.x_min < g - crossing_margin and line.x_max > g + crossing_margin
            for g in gutters
        )
        if line.width > full_width_threshold or crosses_gutter:
            full_width_lines.append(line)
            continue
        center = (line.x_min + line.x_max) / 2.0
        col_index = len(columns) - 1
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= center < boundaries[i + 1]:
                col_index = i
                break
        columns[col_index].append(line)

    for col in columns:
        col.sort(key=lambda ln: ln.y_min)

    column_ys = [ln.y_min for col in columns for ln in col] or [0.0]
    column_ys_max = [ln.y_max for col in columns for ln in col] or [0.0]
    top_of_columns = min(column_ys)
    bottom_of_columns = max(column_ys_max)
    midpoint = (top_of_columns + bottom_of_columns) / 2.0

    headers = sorted(
        (ln for ln in full_width_lines if ln.y_center <= midpoint),
        key=lambda ln: ln.y_min,
    )
    footers = sorted(
        (ln for ln in full_width_lines if ln.y_center > midpoint),
        key=lambda ln: ln.y_min,
    )

    ordered_lines = headers
    for col in columns:
        ordered_lines += col
    ordered_lines += footers

    reordered_words = [w for line in ordered_lines for w in line.words]

    if len(reordered_words) != len(words):
        log_warning(
            log,
            "Column reordering produced a different word count than the "
            "input (bug in the layout logic), keeping original word "
            "order as a safety fallback.",
        )
        return words, 1

    non_empty_columns = sum(1 for col in columns if col)
    log.info(
        f"Multi-column layout detected: {non_empty_columns} column(s), "
        f"gutter(s) at x={[round(g, 1) for g in gutters]}px "
        f"(page width {page_width:.0f}px)"
    )
    return reordered_words, non_empty_columns
