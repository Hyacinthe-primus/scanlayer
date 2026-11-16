"""Sequential batch benchmark.

Usage:
    python -m benchmarks.bench_batch --repeats 2 \
      --output benchmarks/results/phase5b_batch.json

This intentionally calls the public sequential ``convert_batch()`` only; it
does not add or exercise any worker pool.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time

import cv2
import psutil

from benchmarks.corpus import ensure_corpus
from benchmarks.metrics import (
    ResourceSampler,
    environment,
    timestamp,
    write_json,
)
from scanlayer import config, convert_batch
from scanlayer.utils.logger import get_logger


class ProcMonitor(threading.Thread):
    """Track maximum Tesseract/child process counts during a batch."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.max_tesseract = 0
        self.max_children = 0

    def run(self):
        proc = psutil.Process()
        while not self.stop.is_set():
            try:
                children = proc.children(recursive=True)
            except Exception:
                children = []
            tess = 0
            for child in children:
                try:
                    if "tesseract" in child.name().lower():
                        tess += 1
                except Exception:
                    pass
            self.max_tesseract = max(self.max_tesseract, tess)
            self.max_children = max(self.max_children, len(children))
            self.stop.wait(0.02)

    def finish(self):
        self.stop.set()
        self.join(timeout=5)


def _selected_items(names):
    items = ensure_corpus()
    if not names:
        return items
    wanted = {name.strip() for name in names.split(",") if name.strip()}
    return [item for item in items if item.name in wanted]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--images", default=None,
                        help="Comma-separated corpus names")
    parser.add_argument("--batch-workers", type=int, default=None,
                        help="Non-merge conversion workers; 1 = sequential")
    parser.add_argument("--ocr-max-concurrency", type=int, default=None,
                        help="Process-wide Tesseract pass limit")
    parser.add_argument("--output", default=os.path.join(
        "benchmarks", "results", "batch.json"))
    args = parser.parse_args(argv)

    get_logger("benchmarks").setLevel(40)
    original = config.get_settings()
    config.configure(log_level="ERROR", log_timing=False)
    if args.ocr_max_concurrency is not None:
        config.configure(ocr_max_concurrency=args.ocr_max_concurrency)
    ocr_setting = config.OCR_MAX_CONCURRENCY
    items = _selected_items(args.images)
    inputs = [item.path for item in items]
    print(f"OpenCV threads: {cv2.getNumThreads()} | "
          f"PSM parallel: {config.PSM_PARALLEL} | "
          f"PSM candidates: {config.TESSERACT_PSM_CANDIDATES} | "
          f"inputs: {len(inputs)}")

    runs = []
    try:
        with tempfile.TemporaryDirectory(prefix="scanlayer_batch_bench_") as root:
            for rep in range(args.repeats):
                outdir = os.path.join(root, f"run{rep}")
                sampler = ResourceSampler()
                sampler.start()
                monitor = ProcMonitor()
                monitor.start()
                start = time.perf_counter()
                result = convert_batch(
                    inputs, outdir, batch_workers=args.batch_workers,
                )
                wall = time.perf_counter() - start
                sampler.stop()
                monitor.finish()
                run = {
                    "wall_s": wall,
                    "throughput_img_s": len(inputs) / wall if wall else 0.0,
                    "peak_tree_rss_mb": sampler.peak_rss_mb,
                    "total_cpu_s": sampler.total_cpu_s,
                    "cpu_efficiency": (
                        sampler.total_cpu_s / wall
                        if sampler.total_cpu_s is not None and wall else None
                    ),
                    "max_tesseract": monitor.max_tesseract,
                    "max_children": monitor.max_children,
                    "results": [
                        {
                            "path": r.output_path,
                            "elapsed_s": r.elapsed_ms / 1000.0,
                            "words": r.words_count,
                            "pdf_kb": r.pdf_size_bytes / 1024.0,
                        }
                        for r in result.results
                    ],
                    "failures": [
                        {"path": path, "error": error}
                        for path, error in result.failures
                    ],
                }
                runs.append(run)
                print(f"rep{rep}: wall={wall:.2f}s "
                      f"throughput={run['throughput_img_s']:.3f} img/s "
                      f"rss={run['peak_tree_rss_mb']:.0f}MB "
                      f"cpu_eff={run['cpu_efficiency']:.2f} "
                      f"max_tesseract={monitor.max_tesseract} "
                      f"ok={result.ok}")
                for r in run["results"]:
                    print(f"  {os.path.basename(r['path']):28s} "
                          f"{r['elapsed_s']:6.2f}s {r['words']:4d}w "
                          f"{r['pdf_kb']:6.0f}KB")
                for failure in run["failures"]:
                    print(f"  FAIL {failure['path']}: {failure['error'][:100]}")
    finally:
        config.configure(**original)

    write_json(args.output, {
        "metadata": {
            "kind": "batch",
            "timestamp": timestamp(),
            "python": sys.version.split()[0],
            "environment": environment(),
            "workers": args.batch_workers,
            "ocr_max_concurrency": ocr_setting,
            "repeats": args.repeats,
            "psm_parallel": config.PSM_PARALLEL,
            "psm_candidates": list(config.TESSERACT_PSM_CANDIDATES),
        },
        "runs": runs,
    })
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
