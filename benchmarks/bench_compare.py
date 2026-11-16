"""Compare two benchmark JSON files (baseline vs candidate).

Usage:
    python -m benchmarks.bench_compare \
        --baseline benchmarks/results/preprocess_baseline.json \
        --candidate benchmarks/results/preprocess_after.json
"""

from __future__ import annotations

import argparse
import os

from benchmarks.metrics import read_json


def _median(result):
    return result.get("median", {})


def _delta(before, after):
    if before is None or after is None or before == 0:
        return "n/a"
    return f"{(after - before) / before * 100:+.1f}%"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args(argv)

    baseline = {r["name"]: r for r in read_json(args.baseline)["results"]}
    candidate = {r["name"]: r for r in read_json(args.candidate)["results"]}
    names = [name for name in baseline if name in candidate]
    if not names:
        parser.error("no matching result names")

    keys = [
        key for key in (
            "total_ms", "preprocess_ms", "ocr_ms", "pdf_ms",
            "words", "wer", "pdf_size_bytes", "peak_rss_mb",
        )
        if any(key in _median(baseline[n]) for n in names)
    ]
    print(f"{'image':22s} " + " ".join(f"{key:>16s}" for key in keys))
    for name in names:
        before = _median(baseline[name])
        after = _median(candidate[name])
        cells = []
        for key in keys:
            b, a = before.get(key), after.get(key)
            if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                cells.append(f"{a:.1f} ({_delta(b, a)})")
            else:
                cells.append(str(a))
        print(f"{name:22s} " + " ".join(f"{cell:>16s}" for cell in cells))

    print(f"\nbaseline:  {os.path.abspath(args.baseline)}")
    print(f"candidate: {os.path.abspath(args.candidate)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
