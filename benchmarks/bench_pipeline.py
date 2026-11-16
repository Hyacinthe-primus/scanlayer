"""End-to-end pipeline benchmark with quality checks.

Usage:
    python -m benchmarks.bench_pipeline --repeats 2 --psm 3

Runs the same stages as ``convert()`` (preprocess -> OCR -> reading order ->
searchable PDF) so the benchmark can report words, WER, PDF size, and text
extraction in addition to latency. It does not benchmark CLI validation.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
import time

from PIL import Image

from benchmarks.corpus import ensure_corpus
from benchmarks.metrics import (
    ResourceSampler,
    environment,
    extract_pdf_text,
    format_mb,
    median,
    text_contains_phrase,
    timestamp,
    word_error_rate,
    write_json,
)
from scanlayer import config
from scanlayer.layout.columns import reorder_reading_order
from scanlayer.ocr.engine import extract_words, use_configured_tesseract
from scanlayer.pdf.builder import build_searchable_pdf
from scanlayer.preprocessing.enhance import preprocess
from scanlayer.utils.logger import get_logger


def _load_image(item):
    with Image.open(item.path) as img:
        image = img.copy()
        dpi = item.dpi
        if "dpi" in img.info and img.info["dpi"][0] > 10:
            dpi = int(img.info["dpi"][0])
    return image, dpi


def run_one(item, out_dir, repeats, warmup, lang):
    for _ in range(warmup):
        image, dpi = _load_image(item)
        result = preprocess(image, dpi=dpi, orientation=None)
        result.background.close()
        image.close()
    gc.collect()

    runs = []
    for _ in range(repeats):
        image, dpi = _load_image(item)
        sampler = ResourceSampler()
        sampler.start()
        start = time.perf_counter()

        t0 = time.perf_counter()
        result = preprocess(image, dpi=dpi, orientation=None)
        preprocess_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        ocr = extract_words(
            result.ocr_image, result.ocr_scale,
            effective_dpi=result.effective_dpi, lang=lang,
        )
        ocr_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        words, column_count = reorder_reading_order(
            ocr.words, page_width=result.background.width,
        )
        reorder_ms = (time.perf_counter() - t0) * 1000.0

        out_path = os.path.join(out_dir, f"{item.name}.pdf")
        t0 = time.perf_counter()
        pdf_size = build_searchable_pdf(
            result.background, words, out_path, dpi=dpi,
            metadata=None, lang=ocr.language_used,
        )
        pdf_ms = (time.perf_counter() - t0) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0
        sampler.stop()

        actual = [w.text for w in words]
        pdf_text = extract_pdf_text(out_path)
        wall_s = total_ms / 1000.0
        total_cpu_s = sampler.total_cpu_s
        runs.append({
            "total_ms": total_ms,
            "preprocess_ms": preprocess_ms,
            "ocr_ms": ocr_ms,
            "reorder_ms": reorder_ms,
            "pdf_ms": pdf_ms,
            "words": len(actual),
            "wer": word_error_rate(item.expected_words, actual),
            "best_psm": ocr.best_psm,
            "mean_confidence": (
                sum(w.confidence for w in words) / len(words) if words else 0.0
            ),
            "column_count": column_count,
            "pdf_size_bytes": pdf_size,
            "pdf_text_ok": (
                None if pdf_text is None
                else text_contains_phrase(pdf_text, item.expected_words[:5])
            ),
            "pdf_text_words": None if pdf_text is None else len(pdf_text.split()),
            "peak_rss_mb": sampler.peak_rss_mb,
            "total_cpu_s": total_cpu_s,
            "cpu_efficiency": (
                total_cpu_s / wall_s if total_cpu_s is not None and wall_s else None
            ),
            "wall_s": wall_s,
        })
        result.background.close()
        image.close()
        gc.collect()

    med = {}
    for key in (
        "total_ms", "preprocess_ms", "ocr_ms", "reorder_ms", "pdf_ms",
        "words", "wer", "pdf_size_bytes", "peak_rss_mb", "total_cpu_s",
        "cpu_efficiency",
    ):
        values = [r[key] for r in runs if r[key] is not None]
        med[key] = median(values) if values else None
    med["pdf_text_ok"] = all(
        r["pdf_text_ok"] for r in runs if r["pdf_text_ok"] is not None
    ) if any(r["pdf_text_ok"] is not None for r in runs) else None
    return {"runs": runs, "median": med}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--images", default=None)
    parser.add_argument("--psm", default="3",
                        help="Comma-separated PSM candidates, e.g. 3 or 3,4,6,11")
    parser.add_argument("--no-psm-parallel", action="store_true")
    parser.add_argument("--lang", default="eng")
    parser.add_argument("--output", default=os.path.join(
        "benchmarks", "results", "pipeline.json"))
    args = parser.parse_args(argv)

    root = get_logger("benchmarks")
    root.setLevel(40)
    original = config.get_settings()
    psm = [int(value) for value in args.psm.split(",") if value.strip()]
    config.configure(
        log_level="ERROR", log_timing=False,
        psm_candidates=psm, psm_parallel=not args.no_psm_parallel,
    )
    use_configured_tesseract()

    items = ensure_corpus(scale=args.scale)
    if args.images:
        wanted = {name.strip() for name in args.images.split(",") if name.strip()}
        items = [item for item in items if item.name in wanted]
    if not items:
        parser.error("no corpus items selected")

    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="scanlayer_bench_") as out_dir:
            for item in items:
                print(f"[pipeline] {item.name} ({item.width}x{item.height}) ...",
                      flush=True)
                data = run_one(
                    item, out_dir, args.repeats, args.warmup, args.lang,
                )
                data.update({
                    "name": item.name,
                    "kind": item.kind,
                    "path": item.path,
                    "width": item.width,
                    "height": item.height,
                    "dpi": item.dpi,
                })
                results.append(data)
                med = data["median"]
                print(
                    f"  median {med['total_ms']:.0f} ms "
                    f"(pre {med['preprocess_ms']:.0f} ocr {med['ocr_ms']:.0f} "
                    f"pdf {med['pdf_ms']:.0f}) "
                    f"{med['words']:.0f}w WER {med['wer']:.3f} "
                    f"PDF {med['pdf_size_bytes']/1024:.0f}KB "
                    f"rss {format_mb(med['peak_rss_mb'])} "
                    f"text_ok={med['pdf_text_ok']}",
                    flush=True,
                )
    finally:
        config.configure(**original)

    payload = {
        "metadata": {
            "kind": "pipeline",
            "timestamp": timestamp(),
            "python": sys.version.split()[0],
            "environment": environment(),
            "repeats": args.repeats,
            "warmup": args.warmup,
            "scale": args.scale,
            "psm": psm,
            "psm_parallel": not args.no_psm_parallel,
            "lang": args.lang,
        },
        "results": results,
    }
    write_json(args.output, payload)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
