"""
CLI run orchestration for scanlayer: argument handling, config wiring,
and the main conversion loop. Maps exceptions to exit codes.
"""

from __future__ import annotations

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
from scanlayer.main import (
    BatchOptions,
    convert_batch,
    convert_merge,
    plan_batch,
)
from scanlayer.utils.logger import get_logger, log_error, log_warning, set_log_level


def _configure_logging(verbose: bool, quiet: bool) -> None:
    """Apply CLI verbosity to loggers that already exist.

    Module-level loggers are created at import time, before argv is parsed,
    so setting the LOG_LEVEL environment variable alone does not change them.
    """
    if verbose:
        level = "DEBUG"
    elif quiet:
        level = "ERROR"
    else:
        return
    os.environ["LOG_LEVEL"] = level
    set_log_level(level)


def main(argv: list[str] = None) -> int:
    """CLI entry point. Returns a standardized exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose, args.quiet)

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
    if args.workers is not None:
        if args.workers < 1:
            log_error(log, f"--workers must be 1 or more. Got {args.workers}.")
            return EXIT_USER_ERROR
        if not args.merge:
            log_warning(log, "--workers only applies with --merge, ignoring it.")
    if args.batch_workers is not None:
        if args.batch_workers < 1:
            log_error(
                log,
                f"--batch-workers must be 1 or more. Got {args.batch_workers}.",
            )
            return EXIT_USER_ERROR
        if args.merge:
            log_warning(
                log,
                "--batch-workers only applies without --merge, ignoring it.",
            )

    options = BatchOptions(
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
        merge=args.merge,
        workers=args.workers,
        batch_workers=args.batch_workers,
    )

    if args.dry_run:
        try:
            plan = plan_batch(
                args.input, args.output, options,
                tmp_dir=None, dry_run=True,
            )
        except Exception as exc:
            code = _exit_code_for(exc)
            log_error(log, f"dry-run: {exc}")
            return code
        return _run_dry_run(plan, args, log)

    if args.merge:
        with tempfile.TemporaryDirectory(prefix="scanlayer_pdf_") as tmp_dir:
            try:
                plan = plan_batch(
                    args.input, args.output, options,
                    tmp_dir=tmp_dir, dry_run=False,
                )
            except Exception as exc:
                code = _exit_code_for(exc)
                log_error(log, f"--merge: {exc}")
                return code
            if plan.failures:
                _path, exc = plan.failures[0]
                code = _exit_code_for(exc)
                log_error(log, f"--merge: {exc}")
                return code
            if not plan.jobs:
                log_error(log, "--merge: no input files to process.")
                return EXIT_USER_ERROR

            try:
                convert_merge(
                    input_paths=[job.input_path for job in plan.jobs],
                    output_path=args.output,
                    lang=args.lang,
                    dpi=args.dpi,
                    jpeg_quality=args.jpeg_quality,
                    char_whitelist=args.whitelist,
                    char_blacklist=args.blacklist,
                    pdf_metadata=pdf_metadata,
                    force=args.force,
                    orientation=orientation_value,
                    workers=args.workers,
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

    try:
        batch = convert_batch(
            args.input,
            args.output,
            merge=False,
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
            workers=args.workers,
            batch_workers=args.batch_workers,
        )
    except Exception as exc:
        code = _exit_code_for(exc)
        if code == EXIT_UNEXPECTED_ERROR:
            import traceback
            log_error(log, f"batch: unexpected error: {type(exc).__name__}: {exc}")
            log.debug(traceback.format_exc())
        else:
            log_error(log, f"batch: {exc}")
        return code

    if batch.failures:
        first_code = EXIT_UNEXPECTED_ERROR
        for index, (path, exc) in enumerate(batch.failure_exceptions):
            code = _exit_code_for(exc)
            if index == 0:
                first_code = code
            if code == EXIT_UNEXPECTED_ERROR:
                import traceback
                log_error(
                    log,
                    f"'{path}': unexpected error: {type(exc).__name__}: {exc}",
                )
                log.debug("".join(traceback.format_exception(exc)))
            else:
                log_error(log, f"'{path}': {exc}")
        total = len(batch.results) + len(batch.failures)
        if total > 1:
            log_error(log, f"{len(batch.failures)}/{total} file(s) failed.")
            return EXIT_PARTIAL_BATCH
        return first_code

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
