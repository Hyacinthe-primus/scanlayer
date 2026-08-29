"""
Debug visualization for OCR results.

Draws word bounding boxes, confidence scores, and run metadata onto
a copy of the PDF background image. Used with --debug-image CLI flag.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

_HIGH_CONF = 80.0
_MID_CONF = 50.0

_COLOR_HIGH = (0, 150, 0)
_COLOR_MID = (210, 140, 0)
_COLOR_LOW = (200, 0, 0)

_BUNDLED_FONT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "fonts", "DejaVuSans.ttf"
)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(_BUNDLED_FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _color_for_confidence(confidence: float) -> tuple[int, int, int]:
    if confidence >= _HIGH_CONF:
        return _COLOR_HIGH
    if confidence >= _MID_CONF:
        return _COLOR_MID
    return _COLOR_LOW


def build_debug_image(
    background: Image.Image,
    words: list,
    best_psm: "int | None" = None,
    mean_confidence: float = 0.0,
    language_used: str = "",
    exif_orientation: int = 1,
    gross_rotation: int = 0,
    rotation_note: str = "",
) -> Image.Image:
    """Returns a NEW PIL image: a copy of `background` with word boxes,
    per-word confidence, and a run-summary header overlaid. Does not
    modify `background` in place.
    """
    img = background.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    word_font = _load_font(max(10, int(img.width / 150)))
    header_font = _load_font(max(14, int(img.width / 90)))

    for w in words:
        x0, y0 = w.x, w.y
        x1, y1 = w.x + w.width, w.y + w.height
        color = _color_for_confidence(w.confidence)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.text((x0, max(0, y0 - word_font.size - 2)),
                  f"{w.confidence:.0f}", fill=color, font=word_font)

    header_lines = [
        f"PSM {best_psm if best_psm is not None else '-'}  |  "
        f"{len(words)} word(s)  |  mean confidence {mean_confidence:.1f}%  |  "
        f"lang={language_used or '-'}",
        f"EXIF orientation={exif_orientation}  gross_rotation={gross_rotation} deg"
        + (f"  {rotation_note}" if rotation_note else ""),
    ]
    pad = 6
    line_height = header_font.size + 6
    box_height = pad * 2 + line_height * len(header_lines)
    draw.rectangle([0, 0, img.width, box_height], fill=(255, 255, 255))
    y = pad
    for line in header_lines:
        draw.text((pad, y), line, fill=(0, 0, 0), font=header_font)
        y += line_height

    return img
