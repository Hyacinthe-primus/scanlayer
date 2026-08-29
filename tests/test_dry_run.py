"""Tests for the --dry-run CLI flag: validates a batch (files exist,
Tesseract reachable, output paths writable) without running OCR or
writing any output.
"""

import os

import pytest

from scanlayer.cli import main
from scanlayer.cli.dry_run import (
    EXIT_ENV_ERROR,
    EXIT_OK,
    EXIT_PARTIAL_BATCH,
    EXIT_USER_ERROR,
)

from .conftest import requires_tesseract


def test_dry_run_flag_parses():
    from scanlayer.cli.parser import _build_parser

    args = _build_parser().parse_args(["a.jpg", "-o", "out.pdf", "--dry-run"])
    assert args.dry_run is True


def test_dry_run_defaults_to_false():
    from scanlayer.cli.parser import _build_parser

    args = _build_parser().parse_args(["a.jpg", "-o", "out.pdf"])
    assert args.dry_run is False


@requires_tesseract
def test_dry_run_success_writes_nothing(text_image, tmp_path):
    out = tmp_path / "out.pdf"
    code = main([text_image, "-o", str(out), "--dry-run"])
    assert code == EXIT_OK
    assert not out.exists()


@requires_tesseract
def test_dry_run_batch_success_creates_no_output_folder(text_image, tmp_path):
    out_dir = tmp_path / "converted"
    code = main([text_image, "-o", str(out_dir) + os.sep, "--dry-run"])
    assert code == EXIT_OK
    # Real (non-dry-run) batches create the output dir up front via
    # os.makedirs in _resolve_output_path; dry-run must not do that.
    assert not out_dir.exists()


def test_dry_run_missing_input_returns_user_error(tmp_path):
    out = tmp_path / "out.pdf"
    code = main([str(tmp_path / "missing.jpg"), "-o", str(out), "--dry-run"])
    assert code == EXIT_USER_ERROR
    assert not out.exists()


def test_dry_run_bad_tesseract_cmd_returns_env_error(text_image, tmp_path):
    from scanlayer import config

    config.configure(tesseract_cmd="/definitely/not/a/real/binary")
    out = tmp_path / "out.pdf"
    code = main([text_image, "-o", str(out), "--dry-run"])
    assert code == EXIT_ENV_ERROR
    assert not out.exists()


def test_dry_run_batch_partial_failure_returns_partial_batch_code(text_image, tmp_path):
    missing = str(tmp_path / "missing.jpg")
    out_dir = tmp_path / "converted"
    code = main([text_image, missing, "-o", str(out_dir) + os.sep, "--dry-run"])
    assert code == EXIT_PARTIAL_BATCH
    assert not out_dir.exists()


def test_dry_run_does_not_run_ocr(text_image, tmp_path, monkeypatch):
    import scanlayer.main as main_module

    called = []
    monkeypatch.setattr(
        main_module, "extract_words", lambda *a, **kw: called.append(True)
    )
    out = tmp_path / "out.pdf"
    main([text_image, "-o", str(out), "--dry-run"])
    assert called == []


def test_dry_run_merge_success_writes_nothing(text_image, tmp_path):
    out = tmp_path / "merged.pdf"
    code = main([text_image, text_image, "-o", str(out), "--merge", "--dry-run"])
    assert code == EXIT_OK
    assert not out.exists()


def test_dry_run_merge_missing_input_fails_fast_no_partial_batch(text_image, tmp_path):
    # convert_merge() (the real, non-dry-run path) combines all pages
    # into ONE output and aborts on the first bad page - it never
    # returns EXIT_PARTIAL_BATCH, so --merge --dry-run must not
    # either, even with multiple inputs.
    missing = str(tmp_path / "missing.jpg")
    out = tmp_path / "merged.pdf"
    code = main([text_image, missing, "-o", str(out), "--merge", "--dry-run"])
    assert code == EXIT_USER_ERROR
    assert not out.exists()


def test_dry_run_merge_bad_output_path_returns_user_error(text_image, tmp_path):
    bad_out = tmp_path / "no" / "such" / "dir" / "merged.pdf"
    code = main([text_image, "-o", str(bad_out), "--merge", "--dry-run"])
    assert code == EXIT_USER_ERROR
    assert not bad_out.exists()


def test_dry_run_merge_plus_folder_output_still_rejected(text_image, tmp_path):
    # Pre-existing --merge validation (folder output invalid) must
    # still fire before dry-run logic even runs.
    out_dir = tmp_path / "converted"
    out_dir.mkdir()
    code = main([text_image, "-o", str(out_dir), "--merge", "--dry-run"])
    assert code == EXIT_USER_ERROR


@pytest.mark.parametrize("sep", ["/", "\\"])
def test_dry_run_resolve_output_path_accepts_both_separators_no_side_effect(tmp_path, sep):
    # Whichever separator ends -o, dry-run's resolver must recognize
    # it as folder-style output and, unlike the real
    # _resolve_output_path(), must NOT create the directory.
    from scanlayer.cli.dry_run import _dry_run_resolve_output_path

    out_dir = str(tmp_path / "converted") + sep
    resolved, folder = _dry_run_resolve_output_path(out_dir, "invoice.jpg", is_batch=False)
    assert os.path.basename(resolved) == "invoice.pdf"
    assert folder == out_dir
    assert not (tmp_path / "converted").exists()
