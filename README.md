<div align="center">
<img src="scanlayer.png" alt="scanlayer" width="110"/>
<h1>scanlayer</h1>
<p><em>Scanned image or photographed document &rarr; searchable PDF, or raw OCR text/JSON/TSV/hOCR.</em></p>
</div>

**Turn a scanned image or photographed document into a searchable PDF**, or
export the raw OCR result as plain text, JSON, TSV, or hOCR. Use it as a
command-line tool or as a Python library; both are the same engine underneath.

[![PyPI version](https://badge.fury.io/py/scanlayer.svg)](https://badge.fury.io/py/scanlayer)
[![codecov](https://codecov.io/gh/Hyacinthe-primus/scanlayer/branch/main/graph/badge.svg)](https://codecov.io/gh/Hyacinthe-primus/scanlayer)
[![Python Support](https://img.shields.io/pypi/pyversions/pypdf.svg)](https://Hyacinthe-primus.github.io/scanlayer/)
![Platforms](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
[![](https://img.shields.io/badge/-documentation-green)](https://Hyacinthe-primus.github.io/scanlayer/)
[![GitHub last commit](https://img.shields.io/github/last-commit/Hyacinthe-primus/scanlayer)](https://github.com/Hyacinthe-primus/scanlayer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

---

## Documentation

[Full documentation](https://Hyacinthe-primus.github.io/scanlayer/) — installation, CLI & API reference, configuration, output formats, examples, and troubleshooting.

## What it does

A scanned invoice, a phone photo of a letter, a stack of photographed pages:
scanlayer runs it through [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
and gives you back either:

- a **searchable PDF**: the original page image, with an invisible, precisely
  positioned text layer over it, so you can select and search text exactly
  where it visually appears, or
- the **raw OCR result** as `txt`, `json`, `tsv`, or `hocr`: text, per-word
  confidence, and bounding boxes, no PDF built at all.

Along the way it automatically straightens rotated/skewed pages, corrects
uneven lighting, denoises and sharpens for OCR accuracy without touching what
you actually see in the output, reconstructs correct reading order on
genuine multi-column pages, and races several OCR configurations against
each other to pick the most confident result.

| | |
|---|---|
| **Two interfaces, one engine** | `scanlayer` CLI and `import scanlayer` call the exact same pipeline |
| **Five output formats** | Searchable `pdf`, or raw `txt` / `json` / `tsv` / `hocr` |
| **Real column detection** | Two-column articles/letters read in correct order, not interleaved |
| **Batch, merge, native PDF input** | Convert a folder in one call, merge pages into one PDF, or `--dry-run` a batch before spending time on OCR |
| **One configuration surface** | `configure()`, a JSON/YAML profile, or CLI flags, documented precedence |
| **Debug overlay** | `--debug-image` draws every word, color-coded by confidence |

See the [feature catalog](https://Hyacinthe-primus.github.io/scanlayer/features.html) for the complete list, and
[Roadmap & Limitations](https://Hyacinthe-primus.github.io/scanlayer/roadmap.html) for what's deliberately out of
scope or not built yet.

## Requirements

- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (a separate, system-level install, see below)
- [poppler](https://poppler.freedesktop.org/) *only if* you feed scanlayer a native `.pdf` file directly

```bash
# Debian / Ubuntu
sudo apt install tesseract-ocr poppler-utils

# macOS (Homebrew)
brew install tesseract poppler

# Windows: Tesseract -> https://github.com/tesseract-ocr/tesseract/wiki
#          poppler    -> download a release, add its bin/ to PATH
```

Full detail, including how scanlayer locates the Tesseract binary
automatically and how to bundle your own, is in
[Installation](https://Hyacinthe-primus.github.io/scanlayer/installation.html) and [Bundling Tesseract](https://Hyacinthe-primus.github.io/scanlayer/bundling-tesseract.html).

## Install

```bash
pip install scanlayer
```

This installs the `scanlayer` console command and makes `import scanlayer`
available anywhere on the machine.

Working on scanlayer itself, or want to run it straight from a checkout with
no install at all?

```bash
git clone https://github.com/Hyacinthe-primus/scanlayer.git
cd scanlayer                        # the repo root, which contains requirements.txt
pip install -r requirements.txt
python -m scanlayer invoice.jpg -o invoice.pdf   # works with no install at all
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor setup
(editable install and running the test suite).

## Quick start

**As a CLI:**

```bash
scanlayer invoice.jpg -o invoice.pdf --lang fra+eng --dpi 300
```

**As a library:**

```python
import scanlayer

result = scanlayer.convert("invoice.jpg", "invoice.pdf", lang="fra+eng", dpi=300)
print(f"{result.words_count} words, {result.mean_confidence:.1f}% confidence")
```

Every CLI flag and library keyword argument in this project are named to
match each other (`--lang` ↔ `lang=`, `--dpi` ↔ `dpi=`, and so on), see
[Examples](https://Hyacinthe-primus.github.io/scanlayer/examples.html) for every
feature shown both ways, side by side, and
[CLI Reference](https://Hyacinthe-primus.github.io/scanlayer/cli-reference.html) /
[Library API](https://Hyacinthe-primus.github.io/scanlayer/library-api.html) for
the complete details of each.

A few more common cases:

```bash
# Batch-convert a folder
scanlayer *.jpg -o ./converted/

# Merge several photographed pages into one searchable PDF
scanlayer page1.jpg page2.jpg page3.jpg -o report.pdf --merge

# Export raw OCR text/JSON instead of a PDF
scanlayer invoice.jpg -o invoice.json --format json

# See what OCR actually detected, color-coded by confidence
scanlayer invoice.jpg --debug-image

# Validate a batch before spending time on OCR: files exist,
# Tesseract reachable, output paths writable
scanlayer *.jpg -o ./converted/ --dry-run
```

```python
import scanlayer

# Batch
result = scanlayer.convert_batch(["*.jpg"], "./converted/")

# Merge
scanlayer.convert_merge(["page1.jpg", "page2.jpg", "page3.jpg"], "report.pdf")

# Raw export
scanlayer.convert("invoice.jpg", "invoice.json", output_format="json")
```

## Repository layout

```
.
├── .github/           # FUNDING.yml
├── scanlayer/         # library source
│   ├── __init__.py
│   ├── __main__.py    # enables `python -m scanlayer`
│   ├── main.py        # public API: convert()/convert_batch()/convert_merge()
│   ├── config.py
│   ├── cli/           # argparse CLI: parser.py, run.py, dry_run.py
│   ├── fonts/         # bundled DejaVu Sans for the PDF text layer
│   ├── preprocessing/
│   ├── ocr/
│   ├── layout/
│   ├── pdf/
│   └── utils/
├── tests/             # pytest suite
├── pyproject.toml
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE.md
```

## Contributing

Bug reports, fixes, and feature discussions are welcome: see
[CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a development
environment and what to include in a pull request. There is a `tests/`
directory (now including `utils/validators.py` and the `--dry-run` flag, with
cross-platform coverage for Tesseract discovery on Windows/macOS/Linux) and it runs with `pytest` automatically in CI on every push and pull request; see [Roadmap & Limitations](https://Hyacinthe-primus.github.io/scanlayer/roadmap.html) and the [Adding tests](CONTRIBUTING.md#adding-tests) section of the contributing guide for where coverage is thinnest.

## License

[MIT](LICENSE.md).
