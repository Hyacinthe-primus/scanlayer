"""Phase 5B batch planning and deterministic execution tests."""

import os
import tempfile

import pytest
from PIL import Image

from scanlayer import main as main_module
from scanlayer.cli import main as cli_main
from scanlayer.cli.dry_run import EXIT_OK, EXIT_PARTIAL_BATCH, EXIT_USER_ERROR
from scanlayer.main import (
    BatchJob,
    BatchOptions,
    BatchPlan,
    BatchResult,
    ConversionResult,
    convert_batch,
    execute_plan,
    plan_batch,
)
from scanlayer.utils.validators import InputFileError, OutputPathError

from .conftest import requires_tesseract


def _image(path):
    Image.new("RGB", (12, 12), "white").save(path)
    return str(path)


def _result(path):
    return ConversionResult(
        output_path=path, words_count=0, mean_confidence=0.0,
        best_psm=None, early_exited=False, pdf_size_bytes=0,
        elapsed_ms=0.0, language_used="eng", output_format="pdf",
    )


def test_pattern_order_and_sorted_matches(tmp_path):
    for name in ("a2.png", "a1.png", "b1.png"):
        _image(tmp_path / name)
    plan = plan_batch(
        [str(tmp_path / "a*.png"), str(tmp_path / "b*.png")],
        str(tmp_path / "out"),
        BatchOptions(),
        dry_run=True,
    )
    names = [os.path.basename(job.source_path) for job in plan.jobs]
    assert names == ["a1.png", "a2.png", "b1.png"]


def test_duplicate_inputs_keep_first_occurrence(tmp_path):
    source = _image(tmp_path / "one.png")
    plan = plan_batch(
        [source, source, os.path.join(str(tmp_path), "sub", "..", "one.png")],
        str(tmp_path / "out"),
        BatchOptions(),
        dry_run=True,
    )
    assert len(plan.jobs) == 1


def test_same_stem_different_directories_collide(tmp_path):
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    p1 = _image(d1 / "invoice.png")
    p2 = _image(d2 / "invoice.png")
    output = tmp_path / "out"

    with pytest.raises(OutputPathError, match="output collision") as excinfo:
        plan_batch([p1, p2], str(output), BatchOptions(), dry_run=True)

    message = str(excinfo.value)
    assert "d1" in message
    assert "d2" in message
    assert "invoice.pdf" in message
    assert not output.exists()


def test_collision_is_detected_before_conversion(monkeypatch, tmp_path):
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    p1 = _image(d1 / "invoice.png")
    p2 = _image(d2 / "invoice.png")
    called = []
    monkeypatch.setattr(main_module, "convert", lambda **kwargs: called.append(kwargs))

    with pytest.raises(OutputPathError):
        convert_batch([p1, p2], str(tmp_path / "out"))
    assert called == []


def test_pdf_page_order_and_output_names_in_dry_run(monkeypatch, tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF dummy")
    monkeypatch.setattr(main_module, "_pdf_page_count", lambda path: 2)

    plan = plan_batch(
        [str(pdf)], str(tmp_path / "out"),
        BatchOptions(), dry_run=True,
    )
    assert [job.page_number for job in plan.jobs] == [1, 2]
    assert [os.path.basename(job.output_path) for job in plan.jobs] == [
        "report_p1.pdf", "report_p2.pdf",
    ]
    assert not (tmp_path / "out").exists()


def test_pdf_page_order_uses_metadata(monkeypatch, tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF dummy")
    monkeypatch.setattr(main_module, "_pdf_page_count", lambda path: 2)

    plan = plan_batch(
        [str(pdf)], str(tmp_path / "out"),
        BatchOptions(), tmp_dir=str(tmp_path / "tmp"),
    )
    assert [job.page_number for job in plan.jobs] == [1, 2]
    assert [job.input_path for job in plan.jobs] == [str(pdf), str(pdf)]
    assert [os.path.basename(job.output_path) for job in plan.jobs] == [
        "report_p1.pdf", "report_p2.pdf",
    ]


def test_pdf_page_default_output_is_next_to_source(monkeypatch, tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF dummy")
    monkeypatch.setattr(main_module, "_pdf_page_count", lambda path: 2)

    plan = plan_batch(
        [str(pdf)], None, BatchOptions(), dry_run=True,
    )
    assert [job.output_path for job in plan.jobs] == [
        str(tmp_path / "report_p1.pdf"),
        str(tmp_path / "report_p2.pdf"),
    ]


def test_dry_run_does_not_rasterize_pdf(monkeypatch, tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF dummy")
    out = str(tmp_path / "out") + os.sep
    monkeypatch.setattr(main_module, "_pdf_page_count", lambda path: 2)
    monkeypatch.setattr(
        main_module, "_expand_pdf_input",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("PDF rasterized during dry-run")
        ),
    )

    code = cli_main([str(pdf), "-o", str(out), "--dry-run"])
    assert code == EXIT_OK
    assert not (tmp_path / "out").exists()


def test_merge_dry_run_does_not_rasterize_pdf(monkeypatch, tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF dummy")
    out = tmp_path / "merged.pdf"
    monkeypatch.setattr(main_module, "_pdf_page_count", lambda path: 2)
    monkeypatch.setattr(
        main_module, "_expand_pdf_input",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("PDF rasterized during merge dry-run")
        ),
    )

    code = cli_main([str(pdf), "-o", str(out), "--merge", "--dry-run"])
    assert code == EXIT_OK
    assert not out.exists()


def test_execute_plan_preserves_order_and_isolates_failures(monkeypatch, tmp_path):
    failing = str(tmp_path / "bad.png")
    good = str(tmp_path / "good.png")
    jobs = [
        BatchJob(failing, str(tmp_path / "bad.pdf"), failing),
        BatchJob(good, str(tmp_path / "good.pdf"), good),
    ]
    plan = BatchPlan(jobs=jobs, failures=[])

    def fake_convert(**kwargs):
        if kwargs["input_path"] == failing:
            raise InputFileError("bad input")
        return _result(kwargs["output_path"])

    monkeypatch.setattr(main_module, "convert", fake_convert)
    execution = execute_plan(plan, BatchOptions())

    assert [r.output_path for r in execution.results] == [str(tmp_path / "good.pdf")]
    assert [path for path, _ in execution.failures] == [failing]


def test_convert_batch_pdf_page_default_output_next_to_source(
    monkeypatch, tmp_path
):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF dummy")
    monkeypatch.setattr(main_module, "_pdf_page_count", lambda path: 2)

    def fake_raster(path, first, last, dpi, tmp_dir):
        page_dir = tempfile.mkdtemp(prefix="fake_page_", dir=tmp_dir)
        page_paths = []
        for page_number in range(first, last + 1):
            page_path = os.path.join(page_dir, f"p{page_number}.png")
            _image(page_path)
            page_paths.append(page_path)
        return page_paths, page_dir

    monkeypatch.setattr(main_module, "_rasterize_pdf_pages", fake_raster)
    seen = []

    def fake_convert(**kwargs):
        seen.append(kwargs["output_path"])
        return _result(kwargs["output_path"])

    monkeypatch.setattr(main_module, "convert", fake_convert)
    batch = convert_batch([str(pdf)], None)
    assert batch.ok
    assert seen == [
        str(tmp_path / "report_p1.pdf"),
        str(tmp_path / "report_p2.pdf"),
    ]


def test_convert_batch_preserves_failure_order(monkeypatch, tmp_path):
    p1 = _image(tmp_path / "one.png")
    p2 = _image(tmp_path / "two.png")

    def fake_convert(**kwargs):
        if kwargs["input_path"] == p1:
            raise InputFileError("one failed")
        return _result(kwargs["output_path"])

    monkeypatch.setattr(main_module, "convert", fake_convert)
    batch = convert_batch([p1, p2], str(tmp_path / "out"))
    assert len(batch.results) == 1
    assert [path for path, _ in batch.failures] == [p1]
    assert [path for path, _ in batch.failure_exceptions] == [p1]


def test_cli_non_merge_uses_shared_convert_batch(monkeypatch, tmp_path):
    source = _image(tmp_path / "one.png")
    seen = {}

    def fake_batch(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return BatchResult(results=[], failures=[])

    import scanlayer.cli.run as run_module
    monkeypatch.setattr(run_module, "convert_batch", fake_batch)

    code = cli_main([source, "-o", str(tmp_path / "out.pdf")])
    assert code == EXIT_OK
    assert seen["args"][0] == [source]
    assert seen["kwargs"]["merge"] is False


def test_cli_batch_partial_failure_isolated(monkeypatch, tmp_path):
    p1 = _image(tmp_path / "one.png")
    bad = str(tmp_path / "missing.png")

    def fake_batch(*args, **kwargs):
        return BatchResult(
            results=[_result(str(tmp_path / "out.pdf"))],
            failures=[(bad, "missing")],
            failure_exceptions=[(bad, InputFileError("missing"))],
        )

    import scanlayer.cli.run as run_module
    monkeypatch.setattr(run_module, "convert_batch", fake_batch)

    code = cli_main([p1, bad, "-o", str(tmp_path / "out") + os.sep])
    assert code == EXIT_PARTIAL_BATCH


def test_cli_collision_fails_before_conversion(tmp_path):
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    p1 = _image(d1 / "invoice.png")
    p2 = _image(d2 / "invoice.png")
    out = tmp_path / "out"

    code = cli_main([p1, p2, "-o", str(out) + os.sep, "--dry-run"])
    assert code == EXIT_USER_ERROR
    assert not out.exists()

    code = cli_main([p1, p2, "-o", str(out) + os.sep])
    assert code == EXIT_USER_ERROR
    assert not out.exists()


@requires_tesseract
def test_cli_and_library_produce_equivalent_searchable_output(text_image, tmp_path):
    library_out = tmp_path / "library.pdf"
    cli_out = tmp_path / "cli.pdf"

    batch = convert_batch([text_image], str(library_out), lang="eng")
    assert batch.ok
    assert library_out.exists()

    code = cli_main([text_image, "-o", str(cli_out), "--lang", "eng"])
    assert code == EXIT_OK
    assert cli_out.exists()

    from pypdf import PdfReader
    for path in (library_out, cli_out):
        text = " ".join((PdfReader(str(path)).pages[0].extract_text() or "").split())
        assert "INVOICE" in text.upper() or "12345" in text
