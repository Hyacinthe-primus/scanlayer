# Security Policy for scanlayer

scanlayer is a small, MIT-licensed, dependency-light OCR library. Most of the
heavy lifting runs in external, system-installed subprocesses (`tesseract`,
and `pdftoppm`/`pdfinfo` from poppler) against user-supplied files. This policy
is intentionally lightweight, matching the project's scope.

## Scope

In scope for reporting:

- Code-level security issues in `scanlayer/` (the Python library and CLI).
- Anything that could cause arbitrary code execution, path traversal, unsafe
  file handling, or unexpected subprocess/argument injection from crafted
  input.
- Unsafe handling of unicode/UUID/UTF-BOM filenames or untrusted content.

Out of scope (please handle separately or upstream):

- Vulnerabilities in the external binaries scanlayer invokes: report
  [Tesseract](https://github.com/tesseract-ocr/tesseract) and
  [Poppler](https://gitlab.freedesktop.org/poppler/poppler) issues to those
  projects directly.
- Vulnerabilities in third-party Python dependencies
  (`PyTesseract`, `Pillow`, `ReportLab`, `OpenCV`, `numpy`, `pdf2image`);
  report those to their respective maintainers.
- Issues that only affect a globally-installed, pinned copy of scanlayer with
  no local user control. This project runs offline and locally; the vast
  majority of risks fall on the machine that chooses to run it.

## Supported versions

Security fixes are reported against the latest release. Only the most recent
version is supported:

| Version | Supported          |
| ------- | ------------------ |
| Latest release | :white_check_mark: |
| Older releases | :x:                |

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities.

- For non-sensitive reports, open a normal issue on
  [the repository](https://github.com/Hyacinthe-primus/scanlayer/issues).
- For anything you believe could be exploited or exposes sensitive behavior,
  report it privately by email to the maintainer, and confirm receipt. Include:
  - The scanlayer version and OS affected.
  - Steps or a minimal reproducer (a sample file is ideal).
  - The impact and a proposed fix, if you have one.

You will receive a response within a reasonable time. Please give the maintainer
time to confirm and prepare a fix before making the details public. This project
has no CI and no dedicated security team, so responsible disclosure is
appreciated.

## Mitigations in the codebase

- Text-layer content is embedded as vector text coordinates computed from OCR
  output; user material is never executed.
- The projection of word coordinates into the PDF is strictly numeric and does
  not interpolate untrusted strings as code.
- Non-printable/control characters are stripped from OCR words before output
  (`config.DROP_NON_PRINTABLE_WORDS`).

If your contribution or dependency change introduces a new trust boundary
(e.g. reading a remote URL, or embedding user-provided template content), flag
it in your PR and update this policy accordingly.

## Licensing

By reporting, you agree that any security fix you contribute is licensed under
this project's [MIT License](LICENSE.md).
