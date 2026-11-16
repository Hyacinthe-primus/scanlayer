"""Regression tests for preprocessing order and buffer handling."""

import cv2
import numpy as np

from scanlayer.preprocessing import enhance


def test_build_ocr_image_upscales_after_filters(monkeypatch):
    """A small image must be filtered at native size, then upscaled."""
    seen = {}

    def fake_shadow(gray):
        seen["shadow_shape"] = gray.shape
        return gray

    monkeypatch.setattr(enhance, "_remove_shadows", fake_shadow)
    monkeypatch.setattr(enhance, "_sharpen", lambda gray: gray)
    monkeypatch.setattr(
        enhance.cv2, "fastNlMeansDenoising", lambda gray, h: gray,
    )

    image = np.full((200, 1200, 3), 240, dtype=np.uint8)
    ocr, scale, effective_dpi, warnings = enhance._build_ocr_image(image, dpi=300)

    assert seen["shadow_shape"] == (200, 1200)
    assert ocr.shape[1] == enhance.config.OCR_UPSCALE_MIN_WIDTH
    assert scale == enhance.config.OCR_UPSCALE_MIN_WIDTH / 1200
    assert effective_dpi == int(300 * scale)
    assert warnings == ()


def test_build_ocr_image_keeps_resize_first_for_small_upscale(monkeypatch):
    """Near-target widths keep the historical order to preserve OCR input."""
    seen = {}

    def fake_shadow(gray):
        seen["shadow_shape"] = gray.shape
        return gray

    monkeypatch.setattr(enhance, "_remove_shadows", fake_shadow)
    monkeypatch.setattr(enhance, "_sharpen", lambda gray: gray)
    monkeypatch.setattr(
        enhance.cv2, "fastNlMeansDenoising", lambda gray, h: gray,
    )

    image = np.full((200, 2480, 3), 240, dtype=np.uint8)
    ocr, scale, effective_dpi, _ = enhance._build_ocr_image(image, dpi=300)

    assert seen["shadow_shape"][1] == enhance.config.OCR_UPSCALE_MIN_WIDTH
    assert ocr.shape[1] == enhance.config.OCR_UPSCALE_MIN_WIDTH
    assert scale == enhance.config.OCR_UPSCALE_MIN_WIDTH / 2480
    assert effective_dpi == int(300 * scale)


def test_build_ocr_image_downscales_before_filters(monkeypatch):
    """A high-resolution image must be downscaled before filtering."""
    seen = {}

    def fake_shadow(gray):
        seen["shadow_shape"] = gray.shape
        return gray

    monkeypatch.setattr(enhance, "_remove_shadows", fake_shadow)
    monkeypatch.setattr(enhance, "_sharpen", lambda gray: gray)
    monkeypatch.setattr(
        enhance.cv2, "fastNlMeansDenoising", lambda gray, h: gray,
    )

    image = np.full((400, 4000, 3), 240, dtype=np.uint8)
    ocr, scale, effective_dpi, warnings = enhance._build_ocr_image(image, dpi=300)

    assert seen["shadow_shape"][1] == enhance.config.OCR_DOWNSCALE_MAX_WIDTH
    assert ocr.shape[1] == enhance.config.OCR_DOWNSCALE_MAX_WIDTH
    assert scale == enhance.config.OCR_DOWNSCALE_MAX_WIDTH / 4000
    assert effective_dpi == int(300 * scale)
    assert warnings


def test_remove_shadows_preserves_shape_and_dtype():
    gray = np.full((300, 400), 128, dtype=np.uint8)
    result = enhance._remove_shadows(gray)
    assert result.shape == gray.shape
    assert result.dtype == np.uint8
    assert result.min() >= 128


def test_remove_shadows_small_image_uses_full_resolution_path():
    gray = np.full((100, 120), 200, dtype=np.uint8)
    result = enhance._remove_shadows(gray)
    assert result.shape == gray.shape
    assert result.dtype == np.uint8


def test_deskew_handles_foreground_without_column_stack():
    image = np.full((300, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (60, 150), (340, 158), (0, 0, 0), -1)
    result, angle, source_std = enhance._deskew_fine(image)
    assert result.shape[0] >= image.shape[0]
    assert result.shape[1] >= image.shape[1]
    assert source_std > enhance.config.DESKEW_MIN_STD
    assert abs(angle) <= enhance.config.DESKEW_MAX_ANGLE


def test_deskew_blank_image_returns_unchanged():
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    result, angle, source_std = enhance._deskew_fine(image)
    assert result.shape == image.shape
    assert angle == 0.0
    assert source_std < enhance.config.DESKEW_MIN_STD


def test_estimate_skew_angle_returns_none_for_blank():
    gray = np.full((800, 600), 255, dtype=np.uint8)
    assert enhance._estimate_skew_angle(gray) is None
