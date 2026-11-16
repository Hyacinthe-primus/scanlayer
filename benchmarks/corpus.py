"""Reproducible, versioned synthetic corpus for scanlayer benchmarks.

The corpus is generated on demand into ``benchmarks/.cache/corpus/<version>``
so no large binary images need to be committed. Bump ``CORPUS_VERSION`` when
the generator changes in a way that changes the expected ground truth.
"""

from __future__ import annotations

import hashlib
import json
import os
import textwrap
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CORPUS_VERSION = 1
DPI = 300
PAGE_SIZE = (2480, 3508)
RECEIPT_SIZE = (1200, 1800)

PARAGRAPH = (
    "The quarterly report summarises the operating results of the company for "
    "the period ended thirty first December. Revenue increased by twelve percent "
    "compared with the prior year, driven by strong demand in the northern region. "
    "Operating expenses remained broadly flat, and the board recommends a final "
    "dividend of four pence per ordinary share. Further details are provided in "
    "the notes to the financial statements."
)

RECEIPT_LINES = [
    "CORNER STORE",
    "12 High Street",
    "Invoice 00421",
    "Date 14/03/2026",
    "Coffee        3.20",
    "Sandwich      5.50",
    "Water         1.80",
    "TOTAL        10.50",
    "Thank you",
]

_FONT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scanlayer",
    "fonts",
    "DejaVuSans.ttf",
)


@dataclass(frozen=True)
class CorpusItem:
    name: str
    path: str
    kind: str
    dpi: int
    width: int
    height: int
    expected_words: tuple[str, ...]


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(_FONT_PATH, max(8, int(size)))
    except OSError:
        return ImageFont.load_default()


def _wrapped_lines(width: int) -> list[str]:
    return textwrap.wrap(PARAGRAPH, width=width)


def _draw_filled_page(
    size: tuple[int, int],
    lines: Iterable[str],
    font_size: int,
    margins: tuple[int, int] = (150, 150),
    start_y: int = 150,
) -> tuple[Image.Image, list[str]]:
    img = Image.new("RGB", size, (250, 249, 246))
    draw = ImageDraw.Draw(img)
    font = _font(font_size)
    x, y = margins
    drawn: list[str] = []
    step = int(font_size * 1.55)
    while y <= size[1] - margins[1]:
        for line in lines:
            draw.text((x, y), line, fill=(24, 24, 24), font=font)
            drawn.append(line)
            y += step
            if y > size[1] - margins[1]:
                break
    return img, drawn


def _draw_two_column(size: tuple[int, int]) -> tuple[Image.Image, list[str]]:
    img = Image.new("RGB", size, (250, 249, 246))
    draw = ImageDraw.Draw(img)
    font = _font(32)
    lines = _wrapped_lines(42)
    col_w = 1000
    drawn: list[str] = []
    for col in range(2):
        x = 150 + col * (col_w + 220)
        y = 150
        column_lines: list[str] = []
        while y <= size[1] - 150:
            for line in lines:
                draw.text((x, y), line, fill=(20, 20, 20), font=font)
                column_lines.append(line)
                y += 52
                if y > size[1] - 150:
                    break
        drawn.extend(column_lines)
    return img, drawn


def _draw_receipt(size: tuple[int, int]) -> tuple[Image.Image, list[str]]:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font = _font(40)
    y = 120
    for line in RECEIPT_LINES:
        draw.text((100, y), line, fill="black", font=font)
        y += 90
    return img, list(RECEIPT_LINES)


def _add_scan_artifacts(img: Image.Image, seed: int = 1) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Strong but plausible illumination gradient and a soft shadow band.
    grad = 1.05 - 0.55 * (xx / w) - 0.20 * (yy / h)
    shadow = 1.0 - 0.45 * np.exp(
        -((xx - 0.30 * w) ** 2) / (2 * (0.14 * w) ** 2)
    )
    arr *= np.clip(grad * shadow, 0.18, 1.0)[..., None]
    arr += rng.normal(0, 9.0, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _scaled(size: tuple[int, int], scale: float) -> tuple[int, int]:
    return max(32, int(size[0] * scale)), max(32, int(size[1] * scale))


def _render_corpus(cache_dir: str, scale: float = 1.0) -> list[CorpusItem]:
    os.makedirs(cache_dir, exist_ok=True)
    page = _scaled(PAGE_SIZE, scale)
    receipt_size = _scaled(RECEIPT_SIZE, scale)
    font_scale = max(0.35, scale)

    clean, clean_lines = _draw_filled_page(
        page, _wrapped_lines(78), int(34 * font_scale),
        margins=(int(150 * scale), int(150 * scale)),
        start_y=int(150 * scale),
    )
    clean_path = os.path.join(cache_dir, "a4_clean.png")
    clean.save(clean_path, dpi=(DPI, DPI))

    two_col, two_col_lines = _draw_two_column(page)
    two_col_path = os.path.join(cache_dir, "a4_two_column.png")
    two_col.save(two_col_path, dpi=(DPI, DPI))

    receipt, receipt_lines = _draw_receipt(receipt_size)
    receipt_path = os.path.join(cache_dir, "receipt.png")
    receipt.save(receipt_path, dpi=(DPI, DPI))

    hard = _add_scan_artifacts(clean, seed=1)
    hard_path = os.path.join(cache_dir, "a4_hard_shadow.jpg")
    hard.save(hard_path, quality=45, dpi=(DPI, DPI), optimize=False)

    items = [
        CorpusItem(
            "a4_clean", clean_path, "clean_a4", DPI, clean.width, clean.height,
            tuple(" ".join(clean_lines).split()),
        ),
        CorpusItem(
            "a4_two_column", two_col_path, "two_column", DPI,
            two_col.width, two_col.height,
            tuple(" ".join(two_col_lines).split()),
        ),
        CorpusItem(
            "receipt", receipt_path, "receipt", DPI, receipt.width, receipt.height,
            tuple(" ".join(receipt_lines).split()),
        ),
        CorpusItem(
            "a4_hard_shadow", hard_path, "hard_shadow", DPI, hard.width, hard.height,
            tuple(" ".join(clean_lines).split()),
        ),
    ]
    manifest = {
        "version": CORPUS_VERSION,
        "scale": scale,
        "items": [
            {
                "name": item.name,
                "file": os.path.basename(item.path),
                "kind": item.kind,
                "dpi": item.dpi,
                "width": item.width,
                "height": item.height,
                "expected_words": list(item.expected_words),
                "sha256": _sha256(item.path),
            }
            for item in items
        ],
    }
    with open(os.path.join(cache_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return items


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_corpus(
    cache_dir: "str | None" = None, scale: float = 1.0
) -> list[CorpusItem]:
    """Return the corpus, generating it if the version/scale is stale."""
    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".cache", "corpus", f"v{CORPUS_VERSION}_s{scale:g}",
        )
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("version") == CORPUS_VERSION and float(
                manifest.get("scale", 1.0)
            ) == scale:
                items = []
                for entry in manifest["items"]:
                    path = os.path.join(cache_dir, entry["file"])
                    if not os.path.exists(path):
                        raise FileNotFoundError(path)
                    items.append(CorpusItem(
                        entry["name"], path, entry["kind"], int(entry["dpi"]),
                        int(entry["width"]), int(entry["height"]),
                        tuple(entry["expected_words"]),
                    ))
                return items
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return _render_corpus(cache_dir, scale=scale)
