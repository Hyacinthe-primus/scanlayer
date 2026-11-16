# scanlayer benchmarks

These scripts measure the real pipeline, not theoretical micro-optimizations.
They are intentionally **outside** `tests/` so the default pytest suite stays
fast and non-flaky.

## Setup

```bash
pip install -e ".[test,bench]"
```

`bench` adds `psutil` (process-tree CPU/RSS) and `pypdf` (PDF text extraction).
Neither is a runtime dependency.

## Synthetic corpus

`benchmarks/corpus.py` generates a deterministic, versioned corpus into
`benchmarks/.cache/corpus/`:

- `a4_clean` — 2480×3508, 300 DPI, dense single-column text
- `a4_two_column` — two-column page with known reading order
- `receipt` — sparse receipt/table with known text
- `a4_hard_shadow` — low-quality JPEG with strong shadow and noise

Ground-truth words are stored in `manifest.json`. Bump `CORPUS_VERSION` when
changing the generator. No large images are committed.

## Run

```bash
# Stage timings, 2 measured runs + 1 warmup
python -m benchmarks.bench_preprocess --repeats 2 --warmup 1 \
  --output benchmarks/results/preprocess_baseline.json

# End-to-end with PSM 3 (lower variance); use --psm 3,4,6,11 for production default
python -m benchmarks.bench_pipeline --repeats 2 --warmup 1 --psm 3 \
  --output benchmarks/results/pipeline_baseline.json

# Sequential batch baseline (disable the default 2 batch workers)
python -m benchmarks.bench_batch --repeats 2 --batch-workers 1 \
  --output benchmarks/results/batch_baseline.json

# Bounded parallel batch baseline
python -m benchmarks.bench_batch --repeats 2 --batch-workers 2 \
  --output benchmarks/results/batch_parallel.json

# Compare before/after
python -m benchmarks.bench_compare \
  --baseline benchmarks/results/preprocess_baseline.json \
  --candidate benchmarks/results/preprocess_after.json
```

Useful flags:

- `--images a4_clean,receipt` - subset
- `--batch-workers N` - non-merge batch worker count (`1` = sequential)
- `--ocr-max-concurrency N` - process-wide Tesseract pass ceiling
- `--scale 0.25` - small corpus for a quick smoke run
- `--orientation none` - preprocessing without OSD/deskew
- `--cv2-threads 1` - emulate per-worker OpenCV settings
- `--no-psm-parallel` - sequential PSM + early exit

## What is measured

`bench_preprocess.py`:

- full `preprocess()` wall time (median)
- per-stage time (`_fix_gross_orientation`, `_deskew_fine`,
  `_build_ocr_image`, `_remove_shadows`, `_sharpen`)
- parent RSS (preprocessing has no child processes)

`bench_pipeline.py`:

- load/preprocess/OCR/reading-order/PDF stage times
- end-to-end latency
- word count, mean confidence, selected PSM, WER vs ground truth
- PDF size and extracted-text phrase check
- peak parent + child RSS and child CPU time

`bench_batch.py`:

- sequential batch wall time and throughput
- per-file latency/size/words
- peak process-tree RSS and CPU efficiency
- maximum Tesseract/child process count

## Caveats

- The corpus is synthetic. Speed is structural; OCR accuracy must still be
  validated on real, human-labelled scans before changing adaptive defaults.
- Run benchmarks on an idle machine; use medians, not a single run.
- PDF text extraction is checked with whitespace normalization because the
  current text layer emits one text object per word.
