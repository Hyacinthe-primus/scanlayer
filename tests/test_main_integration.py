import pytest

from scanlayer.cli.dry_run import (
    EXIT_ENV_ERROR,
    EXIT_UNEXPECTED_ERROR,
    EXIT_USER_ERROR,
    _exit_code_for,
)
from scanlayer.main import ConversionResult, convert
from scanlayer.utils.errors import BlankPageDetectedError
from scanlayer.utils.validators import (
    DependencyError,
    InputFileError,
    TesseractEnvironmentError,
)

from .conftest import requires_tesseract


@requires_tesseract
def test_convert_produces_pdf_with_expected_metadata(text_image, tmp_path):
    out = tmp_path / "out.pdf"
    result = convert(text_image, str(out), lang="eng")

    assert isinstance(result, ConversionResult)
    assert out.exists()
    assert result.output_path == str(out)
    assert result.pdf_size_bytes > 0
    assert result.output_format == "pdf"
    # A clean synthetic "INVOICE 12345" render should OCR with something.
    assert result.words_count >= 1


@requires_tesseract
def test_convert_txt_format_contains_recognized_text(text_image, tmp_path):
    out = tmp_path / "out.txt"
    convert(text_image, str(out), lang="eng", output_format="txt")
    content = out.read_text()
    assert "INVOICE" in content.upper() or "12345" in content


@requires_tesseract
def test_convert_blank_page_raises_without_force(blank_image, tmp_path):
    out = tmp_path / "out.pdf"
    with pytest.raises(BlankPageDetectedError):
        convert(blank_image, str(out))


@requires_tesseract
def test_convert_blank_page_with_force_builds_empty_pdf(blank_image, tmp_path):
    out = tmp_path / "out.pdf"
    result = convert(blank_image, str(out), force=True)
    assert out.exists()
    assert result.words_count == 0


def test_convert_missing_input_raises_inputfileerror(tmp_path):
    with pytest.raises(InputFileError):
        convert(str(tmp_path / "does_not_exist.jpg"), str(tmp_path / "out.pdf"))


def test_convert_bad_tesseract_cmd_raises_environment_error(text_image, tmp_path):
    from scanlayer import config
    config.configure(tesseract_cmd="/definitely/not/a/real/binary")
    with pytest.raises(TesseractEnvironmentError):
        convert(text_image, str(tmp_path / "out.pdf"))


def test_convert_rejects_unsupported_output_format(text_image, tmp_path):
    with pytest.raises(ValueError, match="Unsupported output_format"):
        convert(text_image, str(tmp_path / "out.xml"), output_format="xml")


@pytest.mark.parametrize("exc, expected_code", [
    (InputFileError("x"), EXIT_USER_ERROR),
    (TesseractEnvironmentError("x"), EXIT_ENV_ERROR),
    (DependencyError("x"), EXIT_ENV_ERROR),
    (BlankPageDetectedError("x"), EXIT_USER_ERROR),
    (RuntimeError("boom"), EXIT_UNEXPECTED_ERROR),
])
def test_exit_code_for_maps_exception_types(exc, expected_code):
    assert _exit_code_for(exc) == expected_code
