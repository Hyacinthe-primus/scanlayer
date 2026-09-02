"""
Font selection for the invisible PDF text layer.

Picks a font per document based on OCR lang string:
  1. CJK: reportlab's built-in CID fonts.
  2. Latin/Cyrillic/Greek/Vietnamese: bundled DejaVu Sans TTF.
  3. Other scripts: falls back to Helvetica.

Set config.FONT_PATH to force a specific TTF and skip auto-selection.
"""

from __future__ import annotations

import os
import sys

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont

from scanlayer import config
from scanlayer.utils.logger import get_logger, log_warning

log = get_logger(__name__)

FALLBACK_FONT = "Helvetica"

_UNICODE_TTF_NAME = "ScanlayerUnicode"
_scanlayer_root = os.path.dirname(os.path.dirname(__file__))
_bundle_root = getattr(sys, "_MEIPASS", os.path.dirname(_scanlayer_root))
_UNICODE_TTF_PATH = os.path.join(_bundle_root, "scanlayer", "fonts", "DejaVuSans.ttf")

_CUSTOM_TTF_NAME = "ScanlayerCustom"
_custom_registered_path: str | None = None

_CJK_FONTS = {
    "chi_sim": "STSong-Light",
    "chi_tra": "MSung-Light",
    "jpn": "HeiseiMin-W3",
    "kor": "HYSMyeongJo-Medium",
}

_registered: set[str] = set()


def _register_once(register_fn, font_name: str) -> bool:
    """Register a font the first time it's needed. Returns True on success."""
    if font_name in _registered:
        return True
    try:
        register_fn()
        _registered.add(font_name)
        return True
    except Exception as exc:
        log_warning(
            log,
            f"Could not register font {font_name!r} ({type(exc).__name__}: "
            f"{exc}), falling back to {FALLBACK_FONT}.",
        )
        return False


def resolve_font(lang: str | None) -> str:
    """Return the reportlab font name for the text layer.

    If config.FONT_PATH is set, it wins. Otherwise picks based on lang.
    """
    global _custom_registered_path
    if config.FONT_PATH:
        if config.FONT_PATH != _custom_registered_path:
            _registered.discard(_CUSTOM_TTF_NAME)
            _custom_registered_path = config.FONT_PATH
        ok = _register_once(
            lambda: pdfmetrics.registerFont(TTFont(_CUSTOM_TTF_NAME, config.FONT_PATH)),
            _CUSTOM_TTF_NAME,
        )
        if ok:
            return _CUSTOM_TTF_NAME
        log_warning(
            log,
            f"config.FONT_PATH={config.FONT_PATH!r} could not be "
            f"registered, falling back to automatic font selection.",
        )

    lang = (lang or "").lower()
    components = [c for c in lang.split("+") if c]

    for component in components:
        cid_font = _CJK_FONTS.get(component)
        if cid_font:
            ok = _register_once(
                lambda cid_font=cid_font: pdfmetrics.registerFont(UnicodeCIDFont(cid_font)),
                cid_font,
            )
            if ok:
                return cid_font
            break

    if os.path.exists(_UNICODE_TTF_PATH):
        ok = _register_once(
            lambda: pdfmetrics.registerFont(TTFont(_UNICODE_TTF_NAME, _UNICODE_TTF_PATH)),
            _UNICODE_TTF_NAME,
        )
        if ok:
            return _UNICODE_TTF_NAME
    else:
        log_warning(
            log,
            f"Bundled Unicode font not found at {_UNICODE_TTF_PATH!r}, "
            f"falling back to {FALLBACK_FONT} (Latin-1 only).",
        )

    return FALLBACK_FONT
