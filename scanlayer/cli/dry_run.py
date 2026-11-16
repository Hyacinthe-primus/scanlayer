"""
Exit codes and --dry-run validation logic for the scanlayer CLI.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from scanlayer.main import _resolve_output_target
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
    """Pure output-path resolution used by dry-run.

    Returns (resolved_output_path, folder_or_None). No directory is created.
    """
    return _resolve_output_target(output_arg, input_path, is_batch, output_format)


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


def _dry_run_validate_job(job, output_format: str) -> None:
    """Validate one planned job without writing anything."""
    if job.page_number is not None:
        input_abs = validate_input_file(job.source_path)
    else:
        input_abs = validate_input_file(job.input_path)
    validate_tesseract_environment()
    if job.output_dir is not None:
        _check_output_folder_creatable(job.output_dir)
    else:
        validate_output_path(job.output_path, expected_ext=f".{output_format}")
    if job.page_number is None:
        validate_image_readable(input_abs)


def _run_dry_run(plan, args: argparse.Namespace, log) -> int:
    """Validate a batch plan without OCR, rasterization, or output writes.

    Non-merge: per-file failures are collected; a single job returns its own
    exit code, multiple jobs return EXIT_PARTIAL_BATCH.
    --merge: fail fast on the first bad input, matching convert_merge().
    """
    total = len(plan.jobs) + len(plan.failures)

    if args.merge:
        if plan.failures:
            path, exc = plan.failures[0]
            log_error(log, f"'{path}': {exc}")
            return _exit_code_for(exc)
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

        for job in plan.jobs:
            try:
                if job.page_number is not None:
                    validate_input_file(job.source_path)
                else:
                    input_abs = validate_input_file(job.input_path)
                    validate_image_readable(input_abs)
            except ValidationError as exc:
                log_error(log, f"'{job.source_path}': {exc}")
                return _exit_code_for(exc)

        log.info(
            f"dry-run: {len(plan.jobs)} page/file(s) OK for --merge "
            f"into {args.output}, no output written."
        )
        return EXIT_OK

    failures = 0
    last_exit_code = EXIT_OK
    for path, exc in plan.failures:
        code = _exit_code_for(exc)
        log_error(log, f"'{path}': {exc}")
        failures += 1
        last_exit_code = code
        if total <= 1:
            return last_exit_code

    for job in plan.jobs:
        try:
            _dry_run_validate_job(job, args.format)
        except Exception as exc:
            code = _exit_code_for(exc)
            log_error(log, f"'{job.source_path}': {exc}")
            failures += 1
            last_exit_code = code
            if total <= 1:
                return last_exit_code

    if failures:
        if total > 1:
            log_error(
                log,
                f"dry-run: {failures}/{total} file(s) failed validation.",
            )
            return EXIT_PARTIAL_BATCH
        return last_exit_code

    log.info(f"dry-run: {total} file(s) OK, no output written.")
    return EXIT_OK
