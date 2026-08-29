"""
Source image preprocessing.

Two outputs:
1. background: straightened image, original colors/resolution.
2. ocr_image: grayscale, illumination-corrected, denoised, contrast-enhanced.

Both share the same coordinate system (ocr_scale maps between them).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps

from scanlayer import config
from scanlayer.utils.logger import get_logger, log_warning, stage_timer

log = get_logger(__name__)


@dataclass
class PreprocessResult:
    background: Image.Image
    ocr_image: np.ndarray
    ocr_scale: float
    effective_dpi: int
    exif_orientation_applied: int = 1
    gross_rotation_applied: int = 0
    deskew_angle_applied: float = 0.0
    manual_rotation_applied: "float | None" = None
    orientation_correction_skipped: bool = False
    source_std: float = 0.0
    likely_blank: bool = False
    preprocessing_warnings: tuple = ()


def _pil_to_cv(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _apply_exif_orientation(image: Image.Image) -> tuple[Image.Image, int]:
    """Apply the EXIF Orientation tag before any other processing.

    PIL's Image.open() never does this automatically.
    """
    if not config.APPLY_EXIF_ORIENTATION:
        return image, 1

    try:
        exif = image.getexif()
        orientation = exif.get(0x0112, 1)  # 0x0112 = Orientation tag
    except Exception:
        orientation = 1

    if orientation == 1:
        return image, 1

    corrected = ImageOps.exif_transpose(image)
    log.info(f"EXIF: orientation tag {orientation} applied")
    return corrected, orientation


def _fix_gross_orientation(image: Image.Image) -> tuple[Image.Image, int]:
    """Correct obvious rotations (0/90/180/270) using Tesseract OSD."""
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        rotate = int(osd.get("rotate", 0))
        confidence = float(osd.get("orientation_conf", 0))
    except pytesseract.TesseractError as exc:
        log_warning(
            log,
            f"OSD unavailable (page too text-sparse?), "
            f"orientation not corrected. Detail: {exc}"
        )
        return image, 0
    except Exception as exc:
        # Non-Tesseract error (most often: missing tessdata, or
        # image too large for OSD which downscales poorly internally).
        log_warning(
            log,
            f"OSD failed ({type(exc).__name__}: {exc}), "
            f"orientation not corrected."
        )
        return image, 0

    if rotate == 0:
        log.debug(f"OSD: orientation OK (conf={confidence:.1f})")
        return image, 0

    # If OSD has low confidence, we log a warning but still apply,
    # a wrong gross angle ruins the entire OCR, better to try.
    if confidence < 5.0:
        log_warning(
            log,
            f"OSD: rotation {rotate}° proposed with very low confidence "
            f"({confidence:.1f}/20), applied anyway, but verify the result."
        )
    else:
        log.info(f"OSD: rotation {rotate}° applied (conf={confidence:.1f}/20)")

    # PIL rotate() rotates counter-clockwise; OSD "rotate" indicates
    # the clockwise angle needed to upright the text.
    rotated = image.rotate(-rotate, expand=True, fillcolor=(255, 255, 255))
    return rotated, rotate


def _rotate_by_angle(cv_image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate cv_image by an exact angle in degrees (clockwise positive)."""
    (h, w) = cv_image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(
        cv_image, matrix, (new_w, new_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _deskew_fine(cv_image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Correct slight tilt via minAreaRect on text pixels.

    Returns (corrected_image, applied_angle, source_std).
    """
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    source_std = float(gray.std())

    if source_std < config.DESKEW_MIN_STD:
        log.debug(
            f"Deskew: image std {source_std:.2f} < {config.DESKEW_MIN_STD} "
            f"(near-uniform image, likely blank), Otsu skipped, no correction."
        )
        return cv_image, 0.0, source_std

    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < config.DESKEW_MIN_PIXELS:
        log.debug(
            f"Deskew: not enough signal ({len(coords)} pixels < "
            f"{config.DESKEW_MIN_PIXELS}), image is probably empty."
        )
        return cv_image, 0.0, source_std

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) > config.DESKEW_MAX_ANGLE:
        log_warning(
            log,
            f"Deskew: angle {angle:.2f}° rejected (beyond "
            f"{config.DESKEW_MAX_ANGLE}°), probably a minAreaRect artifact "
            f"(page border detected instead of text lines). "
            f"Tilt not corrected."
        )
        return cv_image, 0.0, source_std

    if abs(angle) < config.DESKEW_MIN_ANGLE:
        log.debug(f"Deskew: angle {angle:.2f}° < threshold, no correction applied")
        return cv_image, 0.0, source_std

    log.info(f"Deskew: correction of {angle:.2f}° applied")

    (h, w) = cv_image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        cv_image, matrix, (new_w, new_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    ), angle, source_std


def _remove_shadows(gray: np.ndarray) -> np.ndarray:
    """Correct non-uniform illumination (cast shadows, light gradients).

    Divides by a background estimate (Gaussian + morphological closing).
    """
    h, w = gray.shape[:2]

    sigma = max(31.0, min(w, h) / 20.0)
    bg_gauss = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)

    kernel_size = min(51, max(21, min(w, h) // 50))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bg_morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    background = np.maximum(bg_gauss, bg_morph)
    background = np.where(background == 0, 1, background).astype(np.float32)
    normalized = cv2.divide(
        gray.astype(np.float32), background, scale=255.0
    )
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return normalized


def _sharpen(gray: np.ndarray) -> np.ndarray:
    """Light unsharp mask after denoising/CLAHE to recover edge definition."""
    amount = config.UNSHARP_AMOUNT
    if amount <= 0:
        return gray
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=config.UNSHARP_SIGMA)
    return cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)


def _build_ocr_image(
    cv_image: np.ndarray, dpi: int
) -> tuple[np.ndarray, float, int, tuple]:
    """Build the grayscale copy optimized for Tesseract.

    Returns (ocr_image, scale, effective_dpi, warnings).
    """
    warnings_list: list[str] = []
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    orig_width = gray.shape[1]

    scale = 1.0
    if orig_width < config.OCR_UPSCALE_MIN_WIDTH:
        scale = config.OCR_UPSCALE_MIN_WIDTH / orig_width
        gray = cv2.resize(
            gray, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        log.debug(
            f"Upscale x{scale:.2f}: {orig_width}px -> {gray.shape[1]}px "
            f"(below {config.OCR_UPSCALE_MIN_WIDTH}px)"
        )
    elif orig_width > config.OCR_DOWNSCALE_MAX_WIDTH:
        scale = config.OCR_DOWNSCALE_MAX_WIDTH / orig_width
        gray = cv2.resize(
            gray, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        warnings_list.append(
            f"Very wide image ({orig_width}px > {config.OCR_DOWNSCALE_MAX_WIDTH}px) "
            f"- downscaling x{scale:.2f} to avoid excessive processing time."
        )
        log.debug(
            f"Downscale x{scale:.2f}: {orig_width}px -> {gray.shape[1]}px"
        )

    gray = _remove_shadows(gray)

    denoised = cv2.fastNlMeansDenoising(gray, h=config.DENOISING_H)

    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID,
    )
    enhanced = clahe.apply(denoised)
    enhanced = _sharpen(enhanced)

    effective_dpi = max(72, int(dpi * scale))

    return enhanced, scale, effective_dpi, tuple(warnings_list)


def preprocess(
    image: Image.Image, dpi: int, orientation: "str | float | None" = None
) -> PreprocessResult:
    """Full preprocessing pipeline.

    orientation: None=auto (EXIF+OSD+deskew), "none"=disable, or float degrees.
    """
    manual_rotation_applied = None
    orientation_correction_skipped = False

    with stage_timer(log, "Preprocessing"):
        if orientation is not None and str(orientation).strip().lower() == "none":
            log.info("Orientation: correction disabled (orientation='none')")
            exif_orientation = 1
            gross_rot = 0
            orientation_correction_skipped = True
            cv_image = _pil_to_cv(image)
            deskew_angle = 0.0
            source_std = float(
                cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY).std()
            )
        elif orientation is not None:
            angle = float(orientation)
            log.info(f"Orientation: manual angle {angle}° applied (auto-detection skipped)")
            exif_orientation = 1
            gross_rot = 0
            manual_rotation_applied = angle
            cv_image = _pil_to_cv(image)
            cv_image = _rotate_by_angle(cv_image, angle)
            deskew_angle = 0.0
            source_std = float(
                cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY).std()
            )
        else:
            image, exif_orientation = _apply_exif_orientation(image)
            image, gross_rot = _fix_gross_orientation(image)
            cv_image = _pil_to_cv(image)
            cv_image, deskew_angle, source_std = _deskew_fine(cv_image)

        likely_blank = source_std < config.DESKEW_MIN_STD
        if likely_blank:
            log.info(
                f"Preprocessing: source std={source_std:.2f} < "
                f"{config.DESKEW_MIN_STD}, page is likely blank, "
                f"skipping OCR-image build (denoise/CLAHE/upscale)."
            )
            ocr_image = np.zeros((1, 1), dtype=np.uint8)  # unused placeholder
            scale = 1.0
            effective_dpi = max(72, dpi)
            warnings_t: tuple = ()
        else:
            ocr_image, scale, effective_dpi, warnings_t = _build_ocr_image(
                cv_image, dpi
            )

        background = _cv_to_pil(cv_image)

    ocr_image_desc = (
        "skipped (likely blank)"
        if likely_blank
        else f"{ocr_image.shape[1]}x{ocr_image.shape[0]}px"
    )
    log.info(
        f"Preprocessing complete: "
        f"exif_orientation={exif_orientation}, gross_rotation={gross_rot}°, "
        f"background={background.size[0]}x{background.size[1]}px, "
        f"ocr_image={ocr_image_desc}, "
        f"scale=x{scale:.2f}, effective_dpi={effective_dpi}"
    )

    return PreprocessResult(
        background=background,
        ocr_image=ocr_image,
        ocr_scale=scale,
        effective_dpi=effective_dpi,
        exif_orientation_applied=exif_orientation,
        gross_rotation_applied=gross_rot,
        deskew_angle_applied=deskew_angle,
        manual_rotation_applied=manual_rotation_applied,
        orientation_correction_skipped=orientation_correction_skipped,
        source_std=source_std,
        likely_blank=likely_blank,
        preprocessing_warnings=warnings_t,
    )
