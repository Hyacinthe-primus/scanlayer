"""Phase 5D-2 tests for incremental, one-page-at-a-time PDF input."""

import os
import tempfile
import threading
import time

import pytest
from PIL import Image
from pypdf import PdfReader

from scanlayer import main as main_module
from scanlayer.main import (
    BatchOptions,
    ConversionResult,
    _rasterize_pdf_page,
    convert_batch,
    execute_plan,
    plan_batch,
)
from scanlayer.utils.validators import InputFileError, OutputPathError

from .conftest import requires_tesseract


def _result(path):
    return ConversionResult(
        output_path=path, words_count=0, mean_confidence=0.0,
        best_psm=None, early_exited=False, pdf_size_bytes=0,
        elapsed_ms=0.0, language_used="eng", output_format="pdf",
    )


def _dummy_pdf(path, pages=3):
    path.write_bytes(b"%PDF dummy")
    return str(path)


def _plan_pdf(monkeypatch, tmp_path, pages=3):
    pdf = _dummy_pdf(tmp_path / "report.pdf", pages)
    monkeypatch.setattr(main_module, "_pdf_page_count", lambda path: pages)
    plan = plan_batch(
        [pdf], str(tmp_path / "out"), BatchOptions(), tmp_dir=str(tmp_path / "tmp"),
    )
    return pdf, plan


def _make_real_pdf(path, pages=2):
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path))
    for i in range(pages):
        c.setPageSize((300, 220))
        c.setFont("Helvetica-Bold", 26)
        c.drawString(30, 140, f"INVOICE {i + 1} 12345")
        c.showPage()
    c.save()
    return str(path)


def test_planning_pdf_does_not_rasterize(monkeypatch, tmp_path):
    pdf = _dummy_pdf(tmp_path / "report.pdf")
    monkeypatch.setattr(main_module, "_pdf_page_count", lambda path: 3)
    monkeypatch.setattr(
        main_module, "_rasterize_pdf_pages",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("rasterized during planning")
        ),
    )
    plan = plan_batch(
        [pdf], str(tmp_path / "out"), BatchOptions(), tmp_dir=str(tmp_path / "tmp"),
    )
    assert [job.page_number for job in plan.jobs] == [1, 2, 3]
    assert all(job.input_path == pdf for job in plan.jobs)


def test_incremental_rasterization_is_bounded_and_cleans_up(
    monkeypatch, tmp_path
):
    _, plan = _plan_pdf(monkeypatch, tmp_path, pages=4)
    lock = threading.Lock()
    state = {"active": 0, "max": 0}
    page_dirs = []

    def fake_raster(path, first, last, dpi, tmp_dir):
        page_dir = tempfile.mkdtemp(prefix="page_", dir=tmp_dir)
        page_paths = []
        for page in range(first, last + 1):
            page_path = os.path.join(page_dir, f"p{page}.png")
            Image.new("RGB", (20, 20), "white").save(page_path)
            page_paths.append(page_path)
        page_dirs.append(page_dir)
        with lock:
            state["active"] += len(page_paths)
            state["max"] = max(state["max"], state["active"])
        time.sleep(0.03)
        return page_paths, page_dir

    def fake_convert(**kwargs):
        time.sleep(0.03)
        with lock:
            state["active"] -= 1
        return _result(kwargs["output_path"])

    monkeypatch.setattr(main_module, "_rasterize_pdf_pages", fake_raster)
    monkeypatch.setattr(main_module, "convert", fake_convert)
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    execution = execute_plan(
        plan, BatchOptions(), workers=2, tmp_dir=str(pages_dir),
    )

    assert len(execution.results) == 4
    assert state["max"] <= 2
    assert page_dirs
    assert not any(os.path.exists(path) for path in page_dirs)


def test_rasterized_page_cleanup_after_conversion_failure(
    monkeypatch, tmp_path
):
    _, plan = _plan_pdf(monkeypatch, tmp_path, pages=2)
    page_dirs = []

    def fake_raster(path, first, last, dpi, tmp_dir):
        page_dir = tempfile.mkdtemp(prefix="page_", dir=tmp_dir)
        page_paths = []
        for page in range(first, last + 1):
            page_path = os.path.join(page_dir, f"p{page}.png")
            Image.new("RGB", (20, 20), "white").save(page_path)
            page_paths.append(page_path)
        page_dirs.append(page_dir)
        return page_paths, page_dir

    def fail(**kwargs):
        raise RuntimeError("convert failed")

    monkeypatch.setattr(main_module, "_rasterize_pdf_pages", fake_raster)
    monkeypatch.setattr(main_module, "convert", fail)
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    execution = execute_plan(
        plan, BatchOptions(), workers=1, tmp_dir=str(pages_dir),
    )

    assert len(execution.failures) == 2
    assert page_dirs
    assert not any(os.path.exists(path) for path in page_dirs)


def test_rasterization_failure_cleans_owned_directory(monkeypatch, tmp_path):
    pdf = _dummy_pdf(tmp_path / "bad.pdf")
    target = tmp_path / "pages"
    target.mkdir()

    def fail(*args, **kwargs):
        raise RuntimeError("raster failed")

    import pdf2image
    monkeypatch.setattr(pdf2image, "convert_from_path", fail)
    with pytest.raises(InputFileError):
        _rasterize_pdf_page(pdf, 1, 300, str(target))
    assert list(target.iterdir()) == []


def test_same_stem_pdfs_collide_before_execution(monkeypatch, tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    p1 = _dummy_pdf(a / "report.pdf")
    p2 = _dummy_pdf(b / "report.pdf")
    monkeypatch.setattr(main_module, "_pdf_page_count", lambda path: 1)
    with pytest.raises(OutputPathError, match="output collision"):
        plan_batch(
            [p1, p2], str(tmp_path / "out"), BatchOptions(),
            tmp_dir=str(tmp_path / "tmp"),
        )


@requires_tesseract
@pytest.mark.parametrize("workers", [1, 2])
def test_real_multipage_pdf_incremental_conversion(
    tmp_path, monkeypatch, workers
):
    pdf = _make_real_pdf(tmp_path / "report.pdf", pages=3)
    rasterized = []
    original = main_module._rasterize_pdf_pages

    def tracked(path, first, last, dpi, tmp_dir):
        rasterized.extend(range(first, last + 1))
        return original(path, first, last, dpi, tmp_dir)

    monkeypatch.setattr(main_module, "_rasterize_pdf_pages", tracked)
    out = tmp_path / "out"
    batch = convert_batch(
        [pdf], str(out), lang="eng", batch_workers=workers,
    )

    assert batch.ok
    assert len(batch.results) == 3
    assert sorted(rasterized) == [1, 2, 3]
    for page_number in (1, 2, 3):
        path = out / f"report_p{page_number}.pdf"
        assert path.exists()
        reader = PdfReader(str(path))
        assert len(reader.pages) == 1
        text = " ".join((reader.pages[0].extract_text() or "").split())
        assert f"INVOICE {page_number}" in text.upper()


def test_same_stem_pdfs_in_different_directories_use_unique_temp_pages(
    monkeypatch, tmp_path
):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    p1 = _dummy_pdf(a / "report.pdf")
    p2 = _dummy_pdf(b / "report.pdf")
    monkeypatch.setattr(main_module, "_pdf_page_count", lambda path: 1)
    plan = plan_batch(
        [p1, p2], None, BatchOptions(), tmp_dir=str(tmp_path / "tmp"),
    )
    assert len(plan.jobs) == 2
    assert {job.output_path for job in plan.jobs} == {
        str(a / "report_p1.pdf"), str(b / "report_p1.pdf"),
    }

    page_dirs = []

    def fake_raster(path, first, last, dpi, tmp_dir):
        page_dir = tempfile.mkdtemp(prefix="page_", dir=tmp_dir)
        page_paths = []
        for page in range(first, last + 1):
            page_path = os.path.join(page_dir, f"p{page}.png")
            Image.new("RGB", (20, 20), "white").save(page_path)
            page_paths.append(page_path)
        page_dirs.append(page_dir)
        return page_paths, page_dir

    monkeypatch.setattr(main_module, "_rasterize_pdf_pages", fake_raster)
    monkeypatch.setattr(main_module, "convert", lambda **kwargs: _result(kwargs["output_path"]))
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    execute_plan(plan, BatchOptions(), workers=2, tmp_dir=str(pages_dir))
    assert len(page_dirs) == 2
    assert len(set(page_dirs)) == 2
    assert not any(os.path.exists(path) for path in page_dirs)
