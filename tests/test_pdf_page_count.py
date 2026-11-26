"""Tests for optional pypdf page counting and the pdfinfo fallback."""

import shutil
import sys
import types

import pypdf
import pytest
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

from scanlayer.main import BatchOptions, _pdf_page_count, plan_batch
from scanlayer.utils.validators import DependencyError, InputFileError


def _make_pdf(path, pages=1):
    c = canvas.Canvas(str(path))
    for index in range(pages):
        c.drawString(30, 100, f"Page {index + 1}")
        c.showPage()
    c.save()
    return str(path)


def test_valid_pdf_uses_pypdf_and_skips_pdfinfo(tmp_path, monkeypatch):
    pdf = _make_pdf(tmp_path / "two.pdf", pages=2)

    def fail_pdfinfo(*args, **kwargs):
        raise AssertionError("pdfinfo should not be called when pypdf succeeds")

    import pdf2image
    monkeypatch.setattr(pdf2image, "pdfinfo_from_path", fail_pdfinfo)

    assert _pdf_page_count(pdf) == 2
    plan = plan_batch(
        [pdf], str(tmp_path / "out"), BatchOptions(), tmp_dir=str(tmp_path / "tmp"),
    )
    assert [job.page_number for job in plan.jobs] == [1, 2]


def test_missing_pypdf_falls_back_to_pdfinfo(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)
    import pdf2image
    monkeypatch.setattr(
        pdf2image, "pdfinfo_from_path", lambda path: {"Pages": 3},
    )

    assert _pdf_page_count(str(tmp_path / "three.pdf")) == 3


def test_unexpected_pypdf_parse_failure_falls_back(tmp_path, monkeypatch):
    def fail(path):
        raise ValueError("bad PDF structure")

    monkeypatch.setattr(pypdf, "PdfReader", fail)
    import pdf2image
    monkeypatch.setattr(
        pdf2image, "pdfinfo_from_path", lambda path: {"Pages": 6},
    )

    assert _pdf_page_count(str(tmp_path / "six.pdf")) == 6


def test_pypdf_page_count_failure_falls_back(tmp_path, monkeypatch):
    class BrokenReader:
        def __init__(self, path):
            pass

        @property
        def pages(self):
            raise RuntimeError("cannot enumerate pages")

    monkeypatch.setattr(pypdf, "PdfReader", BrokenReader)
    import pdf2image
    monkeypatch.setattr(
        pdf2image, "pdfinfo_from_path", lambda path: {"Pages": 7},
    )

    assert _pdf_page_count(str(tmp_path / "seven.pdf")) == 7


def test_pypdf_decryption_failure_falls_back(tmp_path, monkeypatch):
    def fail(path):
        raise pypdf.errors.FileNotDecryptedError("encrypted")

    monkeypatch.setattr(pypdf, "PdfReader", fail)
    import pdf2image
    monkeypatch.setattr(
        pdf2image, "pdfinfo_from_path", lambda path: {"Pages": 5},
    )

    assert _pdf_page_count(str(tmp_path / "five.pdf")) == 5


def test_zero_page_count_is_not_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pypdf, "PdfReader", lambda path: types.SimpleNamespace(pages=[]),
    )
    import pdf2image
    monkeypatch.setattr(
        pdf2image, "pdfinfo_from_path", lambda path: {"Pages": 0},
    )

    with pytest.raises(InputFileError, match="0 pages"):
        _pdf_page_count(str(tmp_path / "zero.pdf"))


def test_pdfinfo_failure_preserves_input_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)
    import pdf2image

    def fail(path):
        raise RuntimeError("boom")

    monkeypatch.setattr(pdf2image, "pdfinfo_from_path", fail)
    with pytest.raises(InputFileError, match="could not read as a PDF"):
        _pdf_page_count(str(tmp_path / "bad.pdf"))


def test_pdfinfo_missing_poppler_preserves_dependency_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)
    import pdf2image

    def fail(path):
        raise pdf2image.exceptions.PDFInfoNotInstalledError("no poppler")

    monkeypatch.setattr(pdf2image, "pdfinfo_from_path", fail)
    with pytest.raises(DependencyError, match="poppler"):
        _pdf_page_count(str(tmp_path / "bad.pdf"))


@pytest.mark.skipif(
    shutil.which("pdfinfo") is None,
    reason="poppler pdfinfo is not installed",
)
def test_malformed_pdf_falls_back_to_pdfinfo_error(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF dummy")

    with pytest.raises(InputFileError):
        _pdf_page_count(str(path))


@pytest.mark.skipif(
    shutil.which("pdfinfo") is None,
    reason="poppler pdfinfo is not installed",
)
def test_encrypted_pdf_falls_back_to_pdfinfo_error(tmp_path):
    plain = _make_pdf(tmp_path / "plain.pdf")
    encrypted = tmp_path / "encrypted.pdf"
    writer = PdfWriter()
    writer.append(PdfReader(plain))
    writer.encrypt("secret")
    writer.write(str(encrypted))

    with pytest.raises(InputFileError):
        _pdf_page_count(str(encrypted))
