from scanlayer import config
from scanlayer.pdf import fonts


def test_font_path_override_wins_over_lang(monkeypatch):
    monkeypatch.setattr(config, "FONT_PATH", fonts._UNICODE_TTF_PATH)
    monkeypatch.setattr(fonts, "_custom_registered_path", None)
    monkeypatch.setattr(fonts, "_registered", set())
    # lang would normally resolve to a CJK CID font, but FONT_PATH must win.
    assert fonts.resolve_font("chi_sim") == fonts._CUSTOM_TTF_NAME


def test_font_path_none_falls_back_to_automatic_selection(monkeypatch):
    monkeypatch.setattr(config, "FONT_PATH", None)
    assert fonts.resolve_font("fra+eng") == fonts._UNICODE_TTF_NAME


def test_bad_font_path_falls_back_to_automatic_selection(monkeypatch):
    monkeypatch.setattr(config, "FONT_PATH", "/nonexistent/font.ttf")
    monkeypatch.setattr(fonts, "_custom_registered_path", None)
    monkeypatch.setattr(fonts, "_registered", set())
    assert fonts.resolve_font("fra+eng") == fonts._UNICODE_TTF_NAME


def test_latin_language_resolves_to_bundled_unicode_font():
    assert fonts.resolve_font("fra+eng") == fonts._UNICODE_TTF_NAME


def test_none_lang_falls_back_to_unicode_font_not_helvetica():
    # Previously this path (no lang info) meant the module-level
    # `FONT_NAME = "Helvetica"` constant, verify we no longer silently
    # default to a Latin-1-only font when lang is unknown.
    assert fonts.resolve_font(None) != "Helvetica"


def test_cjk_language_resolves_to_cid_font():
    assert fonts.resolve_font("chi_sim") == "STSong-Light"
    assert fonts.resolve_font("jpn") == "HeiseiMin-W3"
    assert fonts.resolve_font("kor") == "HYSMyeongJo-Medium"


def test_mixed_cjk_and_latin_prefers_cjk():
    assert fonts.resolve_font("chi_sim+eng") == "STSong-Light"


def test_missing_font_file_falls_back_to_helvetica(monkeypatch):
    monkeypatch.setattr(fonts, "_UNICODE_TTF_PATH", "/nonexistent/path/font.ttf")
    monkeypatch.setattr(fonts, "_registered", set())
    assert fonts.resolve_font("fra") == fonts.FALLBACK_FONT


def test_resolve_font_registers_with_reportlab_pdfmetrics():
    from reportlab.pdfbase import pdfmetrics

    font_name = fonts.resolve_font("rus")  # Cyrillic, DejaVu tier
    # Registering twice must not raise (idempotency via _registered cache).
    fonts.resolve_font("rus")
    assert pdfmetrics.getFont(font_name) is not None
