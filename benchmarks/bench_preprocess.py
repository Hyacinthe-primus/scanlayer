"""Preprocessing-only benchmark.

Usage:
    python -m benchmarks.bench_preprocess --repeats 3 --warmup 1

Measures the full ``preprocess()`` path (including OSD/deskew by default) and
the expensive internal stages, with parent/child RSS sampling.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

from PIL import Image

from benchmarks.corpus import ensure_corpus
from benchmarks.metrics import (
    ResourceSampler,
    environment,
    format_mb,
    median,
    timestamp,
    write_json,
)
from scanlayer import config
from scanlayer.preprocessing import enhance
from scanlayer.utils.logger import get_logger

_STAGE_NAMES = (
    "_fix_gross_orientation",
    "_deskew_fine",
    "_build_ocr_image",
    "_remove_shadows",
    "_sharpen",
)
_stage_times: dict[str, float] = {}
_originals: dict[str, object] = {}


def _install_stage_timers() -> None:
    for stage_name in _STAGE_NAMES:
        original = getattr(enhance, stage_name)
        _originals[stage_name] = original

        def make_wrapper(name, fn):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    _stage_times[name] = (
                        _stage_times.get(name, 0.0)
                        + (time.perf_counter() - start) * 1000.0
                    )

            return wrapper

        setattr(enhance, stage_name, make_wrapper(stage_name, original))


def _restore_stage_timers() -> None:
    for name, original in _originals.items():
        setattr(enhance, name, original)


def _load_image(item):
    with Image.open(item.path) as img:
        image = img.copy()
        dpi = item.dpi
        if "dpi" in img.info and img.info["dpi"][0] > 10:
            dpi = int(img.info["dpi"][0])
    return image, dpi


def run_one(item, orientation, repeats, warmup):
    image, dpi = _load_image(item)
    for _ in range(warmup):
        result = enhance.preprocess(image, dpi=dpi, orientation=orientation)
        result.background.close()
    image.close()
    gc.collect()

    runs = []
    for _ in range(repeats):
        _stage_times.clear()
        image, dpi = _load_image(item)
        sampler = ResourceSampler()
        sampler.start()
        start = time.perf_counter()
        result = enhance.preprocess(image, dpi=dpi, orientation=orientation)
        elapsed = (time.perf_counter() - start) * 1000.0
        sampler.stop()
        self_cpu_s = sampler.self_cpu_s
        runs.append({
            "total_ms": elapsed,
            "stages_ms": dict(_stage_times),
            "peak_rss_mb": sampler.peak_rss_mb,
            "self_cpu_s": self_cpu_s,
            "cpu_efficiency": (
                self_cpu_s / (elapsed / 1000.0)
                if self_cpu_s is not None and elapsed else None
            ),
            "ocr_shape": list(result.ocr_image.shape),
            "source_std": result.source_std,
            "likely_blank": result.likely_blank,
            "effective_dpi": result.effective_dpi,
            "ocr_scale": result.ocr_scale,
        })
        result.background.close()
        image.close()
        gc.collect()

    med = {
        "total_ms": median([r["total_ms"] for r in runs]),
        "peak_rss_mb": median([
            r["peak_rss_mb"] for r in runs if r["peak_rss_mb"] is not None
        ]) if any(r["peak_rss_mb"] is not None for r in runs) else None,
        "self_cpu_s": median([
            r["self_cpu_s"] for r in runs if r["self_cpu_s"] is not None
        ]) if any(r["self_cpu_s"] is not None for r in runs) else None,
        "cpu_efficiency": median([
            r["cpu_efficiency"] for r in runs if r["cpu_efficiency"] is not None
        ]) if any(r["cpu_efficiency"] is not None for r in runs) else None,
        "stages_ms": {},
    }
    for name in _STAGE_NAMES:
        values = [r["stages_ms"].get(name, 0.0) for r in runs]
        if any(values):
            med["stages_ms"][name] = median(values)
    return {"runs": runs, "median": med}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale synthetic corpus dimensions (1.0 = A4 300 DPI)")
    parser.add_argument("--images", default=None,
                        help="Comma-separated corpus item names")
    parser.add_argument("--orientation", choices=["auto", "none"], default="auto")
    parser.add_argument("--cv2-threads", type=int, default=0,
                        help="0 leaves OpenCV's default; >0 sets cv2.setNumThreads")
    parser.add_argument("--output", default=os.path.join(
        "benchmarks", "results", "preprocess.json"))
    args = parser.parse_args(argv)

    import cv2

    if args.cv2_threads > 0:
        cv2.setNumThreads(args.cv2_threads)

    root = get_logger("benchmarks")
    root.setLevel(40)  # ERROR; suppress scanlayer stage logs
    original = config.get_settings()
    config.configure(log_level="ERROR", log_timing=False)
    orientation = None if args.orientation == "auto" else "none"
    items = ensure_corpus(scale=args.scale)
    if args.images:
        wanted = {name.strip() for name in args.images.split(",") if name.strip()}
        items = [item for item in items if item.name in wanted]
    if not items:
        parser.error("no corpus items selected")

    _install_stage_timers()
    results = []
    try:
        for item in items:
            print(f"[preprocess] {item.name} ({item.width}x{item.height}) ...",
                  flush=True)
            data = run_one(item, orientation, args.repeats, args.warmup)
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
                f"rss {format_mb(med['peak_rss_mb'])} "
                f"stages " + " ".join(
                    f"{k.replace('_', '')}={v:.0f}ms"
                    for k, v in med["stages_ms"].items()
                ),
                flush=True,
            )
    finally:
        _restore_stage_timers()
        config.configure(**original)

    payload = {
        "metadata": {
            "kind": "preprocess",
            "timestamp": timestamp(),
            "python": sys.version.split()[0],
            "environment": environment(),
            "repeats": args.repeats,
            "warmup": args.warmup,
            "scale": args.scale,
            "orientation": args.orientation,
            "cv2_threads": args.cv2_threads,
        },
        "results": results,
    }
    write_json(args.output, payload)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
