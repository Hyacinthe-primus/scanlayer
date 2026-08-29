"""
Structured OCR result export.

Turns the internal `Word` list produced by `ocr.engine.extract_words` into
TXT, JSON, TSV, or hOCR output, independent of PDF generation.

Callers (see `main.py`) run words through `layout.columns.reorder_reading_order`
before handing them to this module, so word order already reflects geometric
multi-column reading order. `_group_by_line` groups by Tesseract's `line_id`
without trusting it blindly: see its docstring for why.
"""

from __future__ import annotations

import json as _json
from typing import Optional

from scanlayer.ocr.engine import Word

FORMATS = ("pdf", "txt", "json", "tsv", "hocr")


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# A horizontal gap above this multiple of the reference word height is a
# column gutter; real inter-word gaps stay well under line height.
_MAX_INTRA_LINE_GAP_HEIGHT_RATIO = 4.0


def _same_visual_line(current: list[Word], w: Word) -> bool:
    """Whether `w` plausibly continues the text line formed by `current`.

    Genuine line-mates sit close horizontally and overlap vertically; a
    pair that straddles a column gutter fails at least one of these checks,
    which is the signal `_group_by_line` uses to refuse the merge.
    """
    prev = current[-1]
    ref_height = max(prev.height, w.height, 1.0)

    gap = w.x - (prev.x + prev.width)
    if gap > ref_height * _MAX_INTRA_LINE_GAP_HEIGHT_RATIO:
        return False

    cur_y0 = min(x.y for x in current)
    cur_y1 = max(x.y + x.height for x in current)
    overlap = min(cur_y1, w.y + w.height) - max(cur_y0, w.y)
    if overlap < -(ref_height * 0.5):
        return False

    return True


def _group_by_line(words: list[Word]) -> list[list[Word]]:
    """Groups words into lines by Tesseract's `line_id`, preserving order.

    Words sharing a `line_id` merge only when consecutive AND
    geometrically consistent with one visual line (see `_same_visual_line`).
    `line_id` adjacency alone is unreliable: column-blind line detection
    can fuse a row that crosses a gutter, and column reordering can bring
    one line's segments from two columns together under the same `line_id`.
    """
    lines: list[list[Word]] = []
    current: list[Word] = []
    current_id = None
    for w in words:
        if current and w.line_id == current_id and _same_visual_line(current, w):
            current.append(w)
        else:
            if current:
                lines.append(current)
            current = [w]
        current_id = w.line_id
    if current:
        lines.append(current)
    return lines


def to_text(words: list[Word]) -> str:
    """Plain extracted text, one line per detected Tesseract text line."""
    lines = _group_by_line(words)
    return "\n".join(" ".join(w.text for w in line) for line in lines)


def to_json(
    words: list[Word],
    mean_confidence: float,
    best_psm: Optional[int],
    language_used: str,
    image_width: int,
    image_height: int,
) -> str:
    payload = {
        "text": to_text(words),
        "mean_confidence": round(mean_confidence, 2),
        "best_psm": best_psm,
        "language": language_used,
        "image_width": image_width,
        "image_height": image_height,
        "words": [
            {
                "text": w.text,
                "confidence": round(w.confidence, 2),
                "bbox": [
                    round(w.x, 1), round(w.y, 1),
                    round(w.x + w.width, 1), round(w.y + w.height, 1),
                ],
            }
            for w in words
        ],
    }
    return _json.dumps(payload, ensure_ascii=False, indent=2)


def to_tsv(words: list[Word]) -> str:
    """Tab-separated word list. This is NOT Tesseract's own TSV column
    layout (level/page/block/par/line/word_num/left/top/width/height/
    conf/text), it is a simpler, flat schema kept intentionally
    minimal for loading straight into a spreadsheet or a pandas
    DataFrame without extra parsing.
    """
    header = "text\tconfidence\tx\ty\twidth\theight\tline_id"
    rows = [header]
    for w in words:
        text = w.text.replace("\t", " ")
        rows.append(
            f"{text}\t{w.confidence:.2f}\t{w.x:.1f}\t{w.y:.1f}\t"
            f"{w.width:.1f}\t{w.height:.1f}\t{w.line_id}"
        )
    return "\n".join(rows)


def to_hocr(
    words: list[Word],
    image_width: int,
    image_height: int,
    source_name: str = "image",
) -> str:
    """Minimal hOCR (a standard OCR layout representation). Covers
    ocr_page, ocr_line, and ocrx_word, no ocr_carea/ocr_par distinction,
    since `Word` does not track paragraph boundaries separately from
    line boundaries.
    """
    lines = _group_by_line(words)
    body_lines = [
        f'<div class="ocr_page" id="page_1" '
        f'style="width:{image_width}px;height:{image_height}px" '
        f'title="image \'{_xml_escape(source_name)}\'; bbox 0 0 '
        f'{image_width} {image_height}">'
    ]
    for i, line in enumerate(lines, start=1):
        if not line:
            continue
        x0 = min(w.x for w in line)
        y0 = min(w.y for w in line)
        x1 = max(w.x + w.width for w in line)
        y1 = max(w.y + w.height for w in line)
        line_h = max(y1 - y0, 1.0)
        font_px = max(round(line_h * 0.82), 6)
        body_lines.append(
            f'<span class="ocr_line" id="line_{i}" '
            f'style="left:{x0:.0f}px;top:{y0:.0f}px;'
            f'width:{(x1 - x0):.0f}px;height:{line_h:.0f}px;'
            f'font-size:{font_px}px" '
            f'title="bbox {x0:.0f} {y0:.0f} {x1:.0f} {y1:.0f}">'
        )
        for j, w in enumerate(line, start=1):
            wx0, wy0 = w.x, w.y
            wx1, wy1 = w.x + w.width, w.y + w.height
            conf_cls = (
                " low-conf" if w.confidence < 60
                else " mid-conf" if w.confidence < 85
                else ""
            )
            body_lines.append(
                f'<span class="ocrx_word{conf_cls}" id="line_{i}_word_{j}" '
                f'title="bbox {wx0:.0f} {wy0:.0f} {wx1:.0f} {wy1:.0f}; '
                f'x_wconf {w.confidence:.0f}">{_xml_escape(w.text)}</span>'
            )
        body_lines.append("</span>")
    body_lines.append("</div>")
    body = "\n".join(body_lines)

    style = """
    :root{ --ink:#1c1c1c; --low:#d64545; --mid:#c98a1f; --hl:#fff3a3; }
    html,body{ margin:0; padding:0; background:#e9e9e9; }
    body{
      padding:2.5rem 1rem;
      font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
    }
    .ocr_page{
      position:relative;
      margin:0 auto;
      background:#fff;
      border:1px solid #d8d8d8;
      box-shadow:0 2px 14px rgba(0,0,0,.10);
    }
    .ocr_line{
      position:absolute;
      white-space:nowrap;
      line-height:1.05;
      color:var(--ink);
      overflow:visible;
    }
    .ocrx_word{
      padding:0 1px;
      border-radius:2px;
      cursor:default;
    }
    .ocrx_word:hover{
      background:var(--hl);
    }
    .ocrx_word.low-conf{ border-bottom:1px dotted var(--low); }
    .ocrx_word.mid-conf{ border-bottom:1px dotted var(--mid); }
    """.strip()

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" '
        '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n'
        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
        "<head>\n"
        "<title>OCR Output</title>\n"
        '<meta http-equiv="Content-Type" content="text/html;charset=utf-8"/>\n'
        '<meta name="ocr-system" content="tesseract via scanlayer" />\n'
        '<meta name="ocr-capabilities" content="ocr_page ocr_line ocrx_word" />\n'
        f"<style>\n{style}\n</style>\n"
        "</head>\n"
        f"<body>\n{body}\n</body>\n</html>\n"
    )


def export_words(fmt: str, words: list[Word], **kwargs) -> str:
    """Dispatch helper used by main.py. `fmt` must be one of "txt",
    "json", "tsv", "hocr" ("pdf" is not handled here, that format
    goes through pdf.builder instead).
    """
    if fmt == "txt":
        return to_text(words)
    if fmt == "json":
        return to_json(
            words,
            mean_confidence=kwargs.get("mean_confidence", 0.0),
            best_psm=kwargs.get("best_psm"),
            language_used=kwargs.get("language_used", ""),
            image_width=kwargs.get("image_width", 0),
            image_height=kwargs.get("image_height", 0),
        )
    if fmt == "tsv":
        return to_tsv(words)
    if fmt == "hocr":
        return to_hocr(
            words,
            image_width=kwargs.get("image_width", 0),
            image_height=kwargs.get("image_height", 0),
            source_name=kwargs.get("source_name", "image"),
        )
    raise ValueError(f"Unsupported export format: {fmt!r}")
