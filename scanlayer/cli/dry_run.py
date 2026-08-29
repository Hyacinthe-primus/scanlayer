"""
Exit codes and --dry-run validation logic for the scanlayer CLI.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from scanlayer.main import _default_output_path
from scanlayer.utils.errors import BlankPageDetectedError, PipelineError
from scanlayer.utils.logger import log_error
from scanlayer.utils.validators import (
    DependencyError,
    InputFileError,
    OutputPathError,
    TesseractEnvironmentError,
    ValidationError,
    validate_image_readable,
    validate_input_file,
    validate_output_path,
    validate_tesseract_environment,
)

EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_ENV_ERROR = 2
EXIT_UNEXPECTED_ERROR = 3
EXIT_PROCESSING_ERROR = 4
EXIT_PARTIAL_BATCH = 5

def _exit_code_for(exc: Exception) -> int:
    if isinstance(exc, (InputFileError, OutputPathError)):
        return EXIT_USER_ERROR
    if isinstance(exc, (TesseractEnvironmentError, DependencyError)):
        return EXIT_ENV_ERROR
    if isinstance(exc, ValidationError):
        return EXIT_USER_ERROR
    if isinstance(exc, BlankPageDetectedError):
        return EXIT_USER_ERROR  # not a bug/env issue, user needs to pass --force
    if isinstance(exc, PipelineError):
        return EXIT_PROCESSING_ERROR
    return EXIT_UNEXPECTED_ERROR


def _dry_run_resolve_output_path(
    output_arg: str, input_path: str, is_batch: bool, output_format: str = "pdf"
) -> "tuple[str, Optional[str]]":
    """Like _resolve_output_path(), but never touches the filesystem.

    _resolve_output_path() calls os.makedirs() as a side effect for
    folder-style -o, which --dry-run must not do. Returns
    (resolved_output_path, folder_or_None) – folder is set when this
    is the auto-created-folder case, so the caller can validate it
    without creating it.
    """
    if output_arg is None:
        return _default_output_path(input_path, output_format), None
    is_folder = is_batch or output_arg.endswith(("/", "\\")) or os.path.isdir(output_arg)
    if is_folder:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(output_arg, f"{stem}.{output_format}"), output_arg
    return output_arg, None


def _nearest_existing_dir(path: str) -> str:
    """Walk up from `path` to the nearest ancestor that already
    exists. Used to check whether os.makedirs(path) would succeed
    without actually calling it.
    """
    path = os.path.abspath(path)
    while not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            return path
        path = parent
    return path


def _check_output_folder_creatable(folder: str) -> None:
    """Dry-run equivalent of the os.makedirs(exist_ok=True) a real
    batch run does for folder-style -o: confirms the folder (or its
    nearest existing ancestor, standing in for what makedirs would
    need to write into) is writable, without creating anything.
    """
    if os.path.exists(folder):
        if not os.path.isdir(folder):
            raise OutputPathError(
                f"The output path points to a file, not a directory: {folder}"
            )
        target = folder
    else:
        target = _nearest_existing_dir(folder)
    if os.access(target, os.W_OK) is False:
        raise OutputPathError(f"Output directory is not writable: {target}")


def _dry_run_validate_one(
    input_path: str, output_path: str, folder: "Optional[str]", output_format: str
) -> None:
    """Same checks as validate_all(), fastest first, but uses the
    dry-run folder check instead of validate_output_path() when the
    output path is an auto-created batch folder that may not exist
    yet (validate_output_path() would otherwise reject it for a
    reason a real run would just fix by creating the folder).
    """
    input_abs = validate_input_file(input_path)
    validate_tesseract_environment()
    if folder is not None:
        _check_output_folder_creatable(folder)
    else:
        validate_output_path(output_path, expected_ext=f".{output_format}")
    validate_image_readable(input_abs)


def _run_dry_run(
    expanded_inputs: list[str], args: argparse.Namespace, log
) -> int:
    """Validate a batch (files exist/readable, Tesseract reachable,
    output paths writable) without running OCR or writing anything.

    Non-merge: mirrors the exit-code behavior of the real conversion
    loop in main() – a single input's failure returns that failure's
    own exit code immediately, multi-input runs collect every
    failure and return EXIT_PARTIAL_BATCH if any file failed.

    --merge: mirrors convert_merge(), which combines all pages into
    ONE output and aborts on the first bad page rather than
    collecting per-file failures, so dry-run does the same (fail
    fast on the first bad page, no EXIT_PARTIAL_BATCH here).
    """
    is_batch = len(expanded_inputs) > 1

    if args.merge:
        # main() already rejects args.merge with args.output is None
        # before _run_dry_run is reached.
        try:
            validate_output_path(args.output, expected_ext=".pdf")
        except ValidationError as exc:
            log_error(log, f"'--output': {exc}")
            return _exit_code_for(exc)
        try:
            validate_tesseract_environment()
        except ValidationError as exc:
            log_error(log, f"'--merge': {exc}")
            return _exit_code_for(exc)

        for input_path in expanded_inputs:
            try:
                input_abs = validate_input_file(input_path)
                validate_image_readable(input_abs)
            except ValidationError as exc:
                log_error(log, f"'{input_path}': {exc}")
                return _exit_code_for(exc)

        log.info(
            f"dry-run: {len(expanded_inputs)} file(s) OK for --merge "
            f"into {args.output}, no output written."
        )
        return EXIT_OK

    failures: list[tuple[str, str]] = []
    last_exit_code = EXIT_OK

    for input_path in expanded_inputs:
        output_path, folder = _dry_run_resolve_output_path(
            args.output, input_path, is_batch, args.format
        )
        try:
            _dry_run_validate_one(input_path, output_path, folder, args.format)
        except Exception as exc:
            code = _exit_code_for(exc)
            log_error(log, f"'{input_path}': {exc}")
            failures.append((input_path, str(exc)))
            last_exit_code = code
            if not is_batch:
                return last_exit_code

    if failures:
        if is_batch:
            log_error(
                log,
                f"dry-run: {len(failures)}/{len(expanded_inputs)} "
                f"file(s) failed validation.",
            )
            return EXIT_PARTIAL_BATCH
        return last_exit_code

    log.info(f"dry-run: {len(expanded_inputs)} file(s) OK, no output written.")
    return EXIT_OK
