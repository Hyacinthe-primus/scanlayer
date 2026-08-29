"""
CLI run orchestration for scanlayer: argument handling, config wiring,
and the main conversion loop. Maps exceptions to exit codes.
"""

from __future__ import annotations

import glob
import os
import sys
import tempfile

from scanlayer import config
from scanlayer.cli.dry_run import (
    EXIT_OK,
    EXIT_PARTIAL_BATCH,
    EXIT_UNEXPECTED_ERROR,
    EXIT_USER_ERROR,
    _exit_code_for,
    _run_dry_run,
)
from scanlayer.cli.parser import _build_parser
from scanlayer.main import _expand_pdf_input, _resolve_output_path, convert, convert_merge
from scanlayer.utils.logger import get_logger, log_error


def main(argv: list[str] = None) -> int:
    """CLI entry point. Returns a standardized exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"
    elif args.quiet:
        os.environ["LOG_LEVEL"] = "ERROR"

    log = get_logger("main")

    if args.config:
        try:
            file_settings = config.configure_from_file(args.config)
            log.info(f"Applied config file: {args.config} ({sorted(file_settings)})")
        except (FileNotFoundError, ValueError) as exc:
            log_error(log, f"--config: {exc}")
            return EXIT_USER_ERROR

    if args.no_column_detection:
        config.configure(multi_column_detection=False)

    if args.psm is not None:
        if args.psm in (0, 2):
            log_error(
                log,
                f"--psm {args.psm}: PSM 0 and 2 only run orientation/script "
                "detection, they don't produce OCR text, so no words would "
                "be extracted. Pick a PSM that actually reads text, e.g. "
                "3, 4, 6, 7, or 11 (see --help for --psm).",
            )
            return EXIT_USER_ERROR
        if not 0 <= args.psm <= 13:
            log_error(
                log,
                f"--psm must be a Tesseract page segmentation mode, 0-13. "
                f"Got {args.psm}.",
            )
            return EXIT_USER_ERROR
        config.configure(psm_candidates=[args.psm])

    if args.font:
        config.configure(font_path=args.font)

    if args.min_confidence is not None:
        if not 0 <= args.min_confidence <= 100:
            log_error(log, "--min-confidence must be between 0 and 100.")
            return EXIT_USER_ERROR
        config.MIN_WORD_CONFIDENCE = args.min_confidence

    orientation_value: "str | float | None" = None
    if args.orientation is not None:
        if args.orientation.strip().lower() == "none":
            orientation_value = "none"
        else:
            try:
                orientation_value = float(args.orientation)
            except ValueError:
                log_error(
                    log,
                    f"--orientation must be 'none' or a numeric angle in "
                    f"degrees, got: '{args.orientation}'."
                )
                return EXIT_USER_ERROR

    pdf_metadata = {}
    if args.title:
        pdf_metadata["title"] = args.title
    if args.author:
        pdf_metadata["author"] = args.author
    if args.subject:
        pdf_metadata["subject"] = args.subject

    if args.merge and args.format != "pdf":
        log_error(log, "--merge only supports --format pdf.")
        return EXIT_USER_ERROR
    if args.merge and args.output is None:
        log_error(
            log,
            "--merge requires -o/--output (a single explicit file "
            "path): there's no one input to derive a shared output "
            "name from.",
        )
        return EXIT_USER_ERROR
    if args.merge and (args.output.endswith(("/", "\\")) or os.path.isdir(args.output)):
        log_error(log, "--merge requires -o/--output to be a file path, not a folder.")
        return EXIT_USER_ERROR

    raw_inputs: list[str] = []
    for pat in args.input:
        expanded = glob.glob(pat)
        if expanded:
            raw_inputs.extend(expanded)
        else:
            raw_inputs.append(pat)

    with tempfile.TemporaryDirectory(prefix="scanlayer_pdf_") as tmp_dir:
        try:
            expanded_inputs: list[str] = []
            for input_path in raw_inputs:
                expanded_inputs.extend(_expand_pdf_input(input_path, args.dpi, tmp_dir))
        except Exception as exc:
            code = _exit_code_for(exc)
            log_error(log, f"PDF input expansion failed: {exc}")
            return code

        if args.dry_run:
            return _run_dry_run(expanded_inputs, args, log)

        if args.merge:
            try:
                convert_merge(
                    input_paths=expanded_inputs,
                    output_path=args.output,
                    lang=args.lang,
                    dpi=args.dpi,
                    jpeg_quality=args.jpeg_quality,
                    char_whitelist=args.whitelist,
                    char_blacklist=args.blacklist,
                    pdf_metadata=pdf_metadata,
                    force=args.force,
                    orientation=orientation_value,
                )
                return EXIT_OK
            except Exception as exc:
                code = _exit_code_for(exc)
                if code == EXIT_UNEXPECTED_ERROR:
                    import traceback
                    log_error(log, f"--merge: unexpected error: {type(exc).__name__}: {exc}")
                    log.debug(traceback.format_exc())
                else:
                    log_error(log, f"--merge: {exc}")
                return code

        is_batch = len(expanded_inputs) > 1
        failures: list[tuple[str, str]] = []
        last_exit_code = EXIT_OK

        for input_path in expanded_inputs:
            try:
                output_path = _resolve_output_path(
                    args.output, input_path, is_batch, args.format
                )
            except OSError as exc:
                log_error(log, f"'{input_path}': cannot prepare output path: {exc}")
                failures.append((input_path, str(exc)))
                last_exit_code = EXIT_USER_ERROR
                if not is_batch:
                    return last_exit_code
                continue

            try:
                convert(
                    input_path=input_path,
                    output_path=output_path,
                    lang=args.lang,
                    dpi=args.dpi,
                    jpeg_quality=args.jpeg_quality,
                    char_whitelist=args.whitelist,
                    char_blacklist=args.blacklist,
                    pdf_metadata=pdf_metadata,
                    force=args.force,
                    orientation=orientation_value,
                    output_format=args.format,
                    debug_image=args.debug_image,
                )
            except Exception as exc:
                code = _exit_code_for(exc)
                if code == EXIT_UNEXPECTED_ERROR:
                    import traceback
                    log_error(log, f"'{input_path}': unexpected error: {type(exc).__name__}: {exc}")
                    log.debug(traceback.format_exc())
                else:
                    log_error(log, f"'{input_path}': {exc}")
                failures.append((input_path, str(exc)))
                last_exit_code = code
                if not is_batch:
                    return last_exit_code

        if failures:
            if is_batch:
                log_error(log, f"{len(failures)}/{len(expanded_inputs)} file(s) failed.")
                return EXIT_PARTIAL_BATCH
            return last_exit_code

        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
