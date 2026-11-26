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
from scanlayer.ocr.engine import tessdata_config, use_configured_tesseract
from scanlayer.utils.logger import get_logger, log_warning, stage_timer

log = get_logger(__name__)

# Below this factor, keep the historical resize-before-filter order so
# near-target-width images (e.g. a 2480 px A4 scan targeted at 2500 px) keep
# their exact OCR input. Larger upscales (low-resolution photos/receipts) are
# filtered at native resolution and upscaled afterwards, which avoids running
# expensive filters on interpolated pixels that contain no extra detail.
_DEFER_UPSCALE_FACTOR = 1.5

# Illumination is low-frequency. Estimating it at quarter resolution keeps
# the large-sigma Gaussian/morphology cheap; the estimate is upsampled before
# the division. Images below this side length use the full-resolution path.
_SHADOW_ESTIMATE_SCALE = 0.25
_SHADOW_MIN_ESTIMATE_SIDE = 256

# Skew angle is also scale-invariant. Estimate it on a reduced copy so a
# noisy/shadowed page does not spend ~1 s in minAreaRect over millions of
# foreground pixels; the warp still runs at full resolution.
_DESKEW_ESTIMATE_SCALE = 0.25
_DESKEW_MIN_ESTIMATE_SIDE = 256

# OSD only needs enough text to determine 0/90/180/270. Capping the image
# avoids expensive full-resolution Tesseract startup/decoding; 2000 px was
# accurate on the audit's 0/90/180/270 pages, while 1200 px was not.
_OSD_THUMBNAIL_MAX_DIM = 2000


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
    # OSD is a Tesseract subprocess like any OCR pass; use the same
    # configured binary and tessdata, and cap the image size. Full-resolution
    # OSD costs seconds on A4/noisy pages without improving 0/90/180/270.
    use_configured_tesseract()
    osd_image = image
    owns_osd_image = False
    if osd_image.mode != "L":
        # OSD converts to grayscale internally. Do it here once so the
        # thumbnail and the Tesseract child process handle one channel.
        osd_image = osd_image.convert("L")
        owns_osd_image = True
    if max(osd_image.size) > _OSD_THUMBNAIL_MAX_DIM:
        if not owns_osd_image:
            osd_image = osd_image.copy()
            owns_osd_image = True
        osd_image.thumbnail(
            (_OSD_THUMBNAIL_MAX_DIM, _OSD_THUMBNAIL_MAX_DIM),
            Image.Resampling.LANCZOS,
        )

    rotate = 0
    confidence = 0.0
    try:
        osd = pytesseract.image_to_osd(
            osd_image,
            config=tessdata_config(),
            output_type=pytesseract.Output.DICT,
            timeout=config.OSD_TIMEOUT_SECONDS,
        )
        rotate = int(osd.get("rotate", 0))
        confidence = float(osd.get("orientation_conf", 0))
    except pytesseract.TesseractError as exc:
        log_warning(
            log,
            f"OSD unavailable (page too text-sparse?), "
            f"orientation not corrected. Detail: {exc}"
        )
        return image, 0
    except RuntimeError as exc:
        log_warning(
            log,
            f"OSD timed out or failed after {config.OSD_TIMEOUT_SECONDS}s "
            f"({type(exc).__name__}), orientation not corrected."
        )
        return image, 0
    except Exception as exc:
        # Non-Tesseract error (most often: missing tessdata).
        log_warning(
            log,
            f"OSD failed ({type(exc).__name__}: {exc}), "
            f"orientation not corrected."
        )
        return image, 0
    finally:
        if owns_osd_image:
            osd_image.close()

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


def _estimate_skew_angle(gray: np.ndarray) -> "float | None":
    """Estimate tilt from an Otsu mask, using a reduced-resolution copy.

    ``minAreaRect`` over millions of noisy foreground pixels dominated deskew
    time on shadowed/noisy pages. The angle is scale-invariant, so estimate it
    on a 1/4 copy when the image is large enough; the warp itself still runs
    at full resolution.
    """
    h, w = gray.shape[:2]
    scale = (
        _DESKEW_ESTIMATE_SCALE
        if min(h, w) >= _DESKEW_MIN_ESTIMATE_SIDE else 1.0
    )
    sample = gray
    if scale != 1.0:
        sample = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA,
        )
    thresh = cv2.threshold(
        sample, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    coords = cv2.findNonZero(thresh)
    min_pixels = max(1, int(config.DESKEW_MIN_PIXELS * scale * scale))
    if coords is None or len(coords) < min_pixels:
        log.debug(
            f"Deskew: not enough signal "
            f"({0 if coords is None else len(coords)} pixels < "
            f"{min_pixels}), image is probably empty."
        )
        return None

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle


def _deskew_fine(cv_image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Correct slight tilt via minAreaRect on text pixels.

    Returns (corrected_image, applied_angle, source_std).
    """
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    _, stddev = cv2.meanStdDev(gray)
    source_std = float(stddev[0][0])

    if source_std < config.DESKEW_MIN_STD:
        log.debug(
            f"Deskew: image std {source_std:.2f} < {config.DESKEW_MIN_STD} "
            f"(near-uniform image, likely blank), Otsu skipped, no correction."
        )
        return cv_image, 0.0, source_std

    angle = _estimate_skew_angle(gray)
    if angle is None:
        return cv_image, 0.0, source_std

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
    The estimate is low-frequency, so it is computed on a reduced-resolution
    copy and upsampled; this avoids a full-resolution Gaussian with a very
    large sigma, which dominates preprocessing time on A4/high-res pages.
    """
    h, w = gray.shape[:2]

    sigma = max(31.0, min(w, h) / 20.0)
    kernel_size = min(51, max(21, min(w, h) // 50))
    if kernel_size % 2 == 0:
        kernel_size += 1

    if min(w, h) >= _SHADOW_MIN_ESTIMATE_SIDE:
        scale = _SHADOW_ESTIMATE_SCALE
        small_w = max(1, int(round(w * scale)))
        small_h = max(1, int(round(h * scale)))
        small = cv2.resize(
            gray, (small_w, small_h), interpolation=cv2.INTER_AREA,
        )
        small_sigma = max(1.0, sigma * scale)
        small_kernel_size = max(3, int(round(kernel_size * scale)))
        if small_kernel_size % 2 == 0:
            small_kernel_size += 1
        small_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (small_kernel_size, small_kernel_size),
        )
        bg_gauss = cv2.GaussianBlur(small, (0, 0), sigmaX=small_sigma)
        bg_morph = cv2.morphologyEx(small, cv2.MORPH_CLOSE, small_kernel)
        background = cv2.resize(
            np.maximum(bg_gauss, bg_morph), (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size),
        )
        bg_gauss = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        bg_morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        background = np.maximum(bg_gauss, bg_morph)

    background = np.maximum(background, 1)
    return cv2.divide(gray, background, scale=255)


def _denoise_ocr_image(gray: np.ndarray) -> np.ndarray:
    """Apply non-local-means denoising with validated OpenCV parameters.

    The search window is the dominant cost of this stage. It stays
    configurable so the conservative default and the faster 7-pixel window
    can be compared without changing the algorithm.
    """
    window = config.DENOISING_SEARCH_WINDOW
    if isinstance(window, bool) or not isinstance(window, int):
        raise ValueError(
            "DENOISING_SEARCH_WINDOW must be an odd integer >= 3, "
            f"got {window!r}."
        )
    if window < 3 or window % 2 == 0:
        raise ValueError(
            "DENOISING_SEARCH_WINDOW must be an odd integer >= 3, "
            f"got {window!r}."
        )
    strength = config.DENOISING_H
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise ValueError(
            f"DENOISING_H must be a non-negative number, got {strength!r}."
        )
    if strength < 0:
        raise ValueError(
            f"DENOISING_H must be a non-negative number, got {strength!r}."
        )
    return cv2.fastNlMeansDenoising(
        gray,
        h=strength,
        templateWindowSize=7,
        searchWindowSize=window,
    )


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

    Downscaling happens before the expensive filters; upscaling happens
    after them. The illumination background and denoiser are low-frequency /
    edge-preserving operations and should not run at an upscaled resolution
    that adds no detail.

    Returns (ocr_image, scale, effective_dpi, warnings).
    """
    warnings_list: list[str] = []
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    orig_width = gray.shape[1]

    scale = 1.0
    post_scale = 1.0
    if orig_width > config.OCR_DOWNSCALE_MAX_WIDTH:
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
            f"Downscale x{scale:.2f}: {orig_width}px -> {gray.shape[1]}px "
            f"(before filters)"
        )
    elif orig_width < config.OCR_UPSCALE_MIN_WIDTH:
        post_scale = config.OCR_UPSCALE_MIN_WIDTH / orig_width
        scale = post_scale
        if post_scale < _DEFER_UPSCALE_FACTOR:
            # Near target width: filtering at the target size is cheap and
            # preserves the historical OCR input exactly.
            gray = cv2.resize(
                gray, None, fx=post_scale, fy=post_scale,
                interpolation=cv2.INTER_CUBIC,
            )
            log.debug(
                f"Upscale x{post_scale:.2f}: {orig_width}px -> {gray.shape[1]}px "
                f"(before filters, below {config.OCR_UPSCALE_MIN_WIDTH}px)"
            )
            post_scale = 1.0

    gray = _remove_shadows(gray)

    denoised = _denoise_ocr_image(gray)

    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID,
    )
    enhanced = clahe.apply(denoised)
    enhanced = _sharpen(enhanced)

    if post_scale != 1.0:
        enhanced = cv2.resize(
            enhanced, None, fx=post_scale, fy=post_scale,
            interpolation=cv2.INTER_CUBIC,
        )
        log.debug(
            f"Upscale x{post_scale:.2f}: {orig_width}px -> {enhanced.shape[1]}px "
            f"(after filters, below {config.OCR_UPSCALE_MIN_WIDTH}px)"
        )

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
