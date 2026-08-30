# Contributing to scanlayer

Thanks for considering a contribution. This project is small enough that
the process is intentionally lightweight: read this, set up a dev
environment, and open a PR.

## Before you start

There is a `tests/` directory covering most of the library, and a CI
workflow runs it (plus `ruff`) on every push and pull request. Still,
verify your change manually against a real scanned image and include what
you tested in your PR description. If your contribution adds coverage for
one of the gaps listed in [Adding tests](#adding-tests) below, even better.

## Development setup

```bash
git clone https://github.com/Hyacinthe-primus/scanlayer.git
cd scanlayer                            # repo root, contains requirements.txt
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
```

You'll also need Tesseract OCR on your machine (and poppler if you're
touching PDF-input code), see [Installation](https://Hyacinthe-primus.github.io/scanlayer/installation.html)
for the per-OS install commands.

Verify your environment:

```bash
python -m scanlayer --help
python -c "import scanlayer; print(scanlayer.get_settings()['tesseract_cmd'])"
```

## Project layout

```
scanlayer/
├── __init__.py           # public API surface: convert, convert_batch, convert_merge, configure
├── main.py                # the top-level convert()/convert_batch()/convert_merge()
├── config.py               # all tunables, Tesseract/poppler discovery
├── cli/                    # argparse CLI: parser.py, run.py, dry_run.py
├── preprocessing/enhance.py  # EXIF/OSD/deskew, illumination, denoise, CLAHE, sharpen
├── ocr/engine.py            # PSM candidate racing, Tesseract invocation
├── ocr/export.py            # txt/json/tsv/hocr exporters
├── layout/columns.py         # multi-column reading-order reconstruction
├── pdf/builder.py            # searchable PDF construction (single + multi-page)
├── pdf/fonts.py              # automatic font selection for the invisible text layer
└── utils/                    # validators, exceptions, logging, debug-image overlay
```

## Making a change

1. **Open an issue first for anything non-trivial** (new CLI flags, a
   change to a default value, anything touching the exception hierarchy)
   so the approach can be discussed before you invest time in it. Small,
   obviously-correct fixes (typos, doc corrections, an off-by-one) can go
   straight to a PR.
2. **Keep the CLI and library in sync.** Every capability should be
   reachable from both `python -m scanlayer` / `scanlayer` and
   `import scanlayer`, using matching parameter names wherever
   possible (`--lang` ↔ `lang=`, `--dpi` ↔ `dpi=`). If you add a CLI
   flag, add the equivalent keyword argument to `convert()` (and
   `convert_batch()`/`convert_merge()` if it applies to those too), not
   just to the CLI's argument parser (`scanlayer/cli/parser.py`).
3. **Respect the two-tier exception hierarchy.** A precondition that
   fails before any real work starts is a `ValidationError` subclass
   (`scanlayer/utils/validators.py`). A stage that starts and fails for
   an operational reason is a `PipelineError` subclass
   (`scanlayer/utils/errors.py`). If your change can fail in a new way,
   pick the right parent class rather than raising a bare `Exception`,
   and map any new library exception to a CLI exit code in `scanlayer/cli/run.py`.
   See [CLI Reference: Exit codes](https://Hyacinthe-primus.github.io/scanlayer/cli-reference.html#exit-codes).
4. **Never let OCR-only processing touch the visual background.**
   `preprocessing/enhance.py` deliberately keeps `background` (what
   gets drawn into the PDF) and `ocr_image` (what Tesseract sees)
   separate. A denoising or contrast change for OCR accuracy should
   never alter what the reader sees in the final PDF.
5. **Update the docs in the same PR.** This repository's documentation is
   the static HTML site on the `gh-pages` branch (edit it from a worktree:
   `git worktree add <path> gh-pages`), plus `README.md`. It is not
   auto-synced, so if you change a default, add a parameter, or add a CLI
   flag, update the relevant page(s) by hand:
   - New/changed CLI flag → `cli-reference.html` and, if it's a
     common case, an example in `examples.html`.
   - New/changed library parameter → `library-api.html`.
   - New/changed `config.py` default → `configuration.html`.
   - New capability entirely → `features.html`.

## Adding tests

A `tests/` directory already exists — install it together with `pytest` and
`pytest-cov` via `pip install -e ".[test]"` (the `[test]` extra bundles both).
It covers `config.py`, `ocr/export.py`, `layout/columns.py`, `pdf/fonts.py`,
`pdf/builder.py`, and an end-to-end `convert()` test that skips automatically
without a real Tesseract install. Coverage is uploaded to Codecov by CI; to
see it locally, run `pytest --cov=scanlayer --cov-report=html`. Coverage gaps
worth closing next, roughly in priority order:

1. `utils/validators.py`: pure functions, no Tesseract required, easy
   to test with tmp paths and mocked environments, and currently only
   exercised indirectly through the integration test.
2. `preprocessing/enhance.py`: EXIF/OSD/deskew, illumination, denoise;
   needs fixture images with known rotation or lighting defects.
3. `ocr/engine.py`: the PSM-candidate racing logic, with a real or
   mocked Tesseract invocation.
4. More end-to-end cases: a genuine multi-column sample, a rotated
   sample, a blank-page sample, asserting on word count / mean
   confidence ranges rather than exact OCR output, since Tesseract
   output can shift slightly across versions.

## Style

- Match the existing code: type hints on public functions, dataclasses
  for structured returns, docstrings on every public function.
- Keep the CLI (`scanlayer/cli/`) a thin wrapper: argument parsing and
  formatting only. Actual logic belongs in the library functions it
  calls, so the library caller and the CLI user get identical behavior.
- Prefer adding a new, clearly-named `config.py` constant over a magic
  number inline, if the value is even remotely likely to need tuning
  later.

## Submitting

- One logical change per PR.
- Describe what you tested it against (sample image type, OS, Tesseract
  version) in the PR description; CI runs the suite on push, but a manual
  check of your actual scenario is still welcome.
- Reference the issue you opened in step 1, if applicable.

By contributing, you agree your contribution is licensed under this
project's [MIT License](LICENSE.md).
