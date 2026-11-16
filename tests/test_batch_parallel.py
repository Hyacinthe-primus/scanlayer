"""Phase 5D bounded batch-parallelism tests."""

import os
import shutil
import threading
import time

import cv2
import pytest
from PIL import Image

from scanlayer import config
from scanlayer import main as main_module
from scanlayer.cli import main as cli_main
from scanlayer.cli.dry_run import EXIT_OK, EXIT_USER_ERROR
from scanlayer.cli.parser import _build_parser
from scanlayer.main import (
    BatchJob,
    BatchOptions,
    BatchPlan,
    ConversionResult,
    _resolve_batch_workers,
    convert_batch,
    execute_plan,
)

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


def _jobs(tmp_path, count):
    return [
        BatchJob(
            input_path=str(tmp_path / f"in{i}.png"),
            output_path=str(tmp_path / f"out{i}.pdf"),
            source_path=str(tmp_path / f"in{i}.png"),
        )
        for i in range(count)
    ]


def test_batch_worker_resolution_caps_and_rejects():
    assert _resolve_batch_workers(None, 10) == min(
        config.BATCH_WORKERS, os.cpu_count() or 1,
    )
    assert _resolve_batch_workers(4, 2) == 2
    assert _resolve_batch_workers(1, 10) == 1
    for bad in (0, -1, True, 1.5, "2"):
        with pytest.raises(ValueError):
            _resolve_batch_workers(bad, 10)


def test_sequential_mode_uses_calling_thread(monkeypatch, tmp_path):
    jobs = _jobs(tmp_path, 3)
    caller = threading.get_ident()
    threads = []

    def fake_convert(**kwargs):
        threads.append(threading.get_ident())
        return _result(kwargs["output_path"])

    monkeypatch.setattr(main_module, "convert", fake_convert)
    execution = execute_plan(BatchPlan(jobs=jobs, failures=[]), BatchOptions(), workers=1)

    assert len(execution.results) == 3
    assert threads == [caller, caller, caller]


def test_parallel_workers_are_bounded_and_results_ordered(monkeypatch, tmp_path):
    jobs = _jobs(tmp_path, 5)
    lock = threading.Lock()
    state = {"running": 0, "max": 0}
    seen_cv2 = []
    delays = {0: 0.08, 1: 0.04, 2: 0.01, 3: 0.06, 4: 0.02}

    def fake_convert(**kwargs):
        index = int(os.path.basename(kwargs["input_path"])[2:-4])
        with lock:
            state["running"] += 1
            state["max"] = max(state["max"], state["running"])
            seen_cv2.append(cv2.getNumThreads())
        time.sleep(delays[index])
        with lock:
            state["running"] -= 1
        return _result(kwargs["output_path"])

    monkeypatch.setattr(main_module, "convert", fake_convert)
    previous_cv2 = cv2.getNumThreads()
    execution = execute_plan(
        BatchPlan(jobs=jobs, failures=[]), BatchOptions(), workers=2,
    )

    assert state["max"] == 2
    expected_cv2 = max(1, (os.cpu_count() or 1) // 2)
    assert seen_cv2 and all(value == expected_cv2 for value in seen_cv2)
    assert [r.output_path for r in execution.results] == [
        job.output_path for job in jobs
    ]
    assert cv2.getNumThreads() == previous_cv2


def test_parallel_failure_isolation_and_ordering(monkeypatch, tmp_path):
    jobs = _jobs(tmp_path, 4)

    def fake_convert(**kwargs):
        name = os.path.basename(kwargs["input_path"])
        if name == "in1.png":
            raise RuntimeError("bad one")
        if name == "in2.png":
            time.sleep(0.05)
        return _result(kwargs["output_path"])

    monkeypatch.setattr(main_module, "convert", fake_convert)
    execution = execute_plan(
        BatchPlan(jobs=jobs, failures=[]), BatchOptions(), workers=2,
    )

    assert [r.output_path for r in execution.results] == [
        jobs[0].output_path, jobs[2].output_path, jobs[3].output_path,
    ]
    assert [path for path, _ in execution.failures] == [jobs[1].source_path]


def test_no_duplicate_output_writes_in_parallel(monkeypatch, tmp_path):
    jobs = _jobs(tmp_path, 4)
    writes = []
    lock = threading.Lock()

    def fake_convert(**kwargs):
        with lock:
            writes.append(kwargs["output_path"])
        return _result(kwargs["output_path"])

    monkeypatch.setattr(main_module, "convert", fake_convert)
    execute_plan(BatchPlan(jobs=jobs, failures=[]), BatchOptions(), workers=2)
    assert sorted(writes) == sorted(job.output_path for job in jobs)
    assert len(set(writes)) == len(writes)


def test_convert_batch_workers_validation_before_work(monkeypatch, tmp_path):
    source = _image(tmp_path / "one.png")
    called = []
    monkeypatch.setattr(main_module, "convert", lambda **kwargs: called.append(True))
    with pytest.raises(ValueError):
        convert_batch([source], str(tmp_path / "out"), batch_workers=0)
    assert called == []


def test_cli_batch_workers_parsing_and_validation(monkeypatch, tmp_path):
    source = _image(tmp_path / "one.png")
    parser = _build_parser()
    args = parser.parse_args(
        [source, "-o", str(tmp_path / "out.pdf"), "--batch-workers", "3"]
    )
    assert args.batch_workers == 3

    import scanlayer.cli.run as run_module
    seen = {}

    def fake_batch(*args, **kwargs):
        seen.update(kwargs)
        return main_module.BatchResult(results=[], failures=[])

    monkeypatch.setattr(run_module, "convert_batch", fake_batch)
    assert cli_main([
        source, "-o", str(tmp_path / "out.pdf"), "--batch-workers", "3",
    ]) == EXIT_OK
    assert seen["batch_workers"] == 3

    assert cli_main([
        source, "-o", str(tmp_path / "out.pdf"), "--batch-workers", "0",
    ]) == EXIT_USER_ERROR


def test_cli_merge_workers_semantics_unchanged(monkeypatch, tmp_path):
    source = _image(tmp_path / "one.png")
    import scanlayer.cli.run as run_module
    seen = {}
    monkeypatch.setattr(
        run_module, "convert_merge",
        lambda **kwargs: seen.update(kwargs) or None,
    )
    code = cli_main([
        source, source, "-o", str(tmp_path / "merged.pdf"),
        "--merge", "--workers", "2",
    ])
    assert code == EXIT_OK
    assert seen["workers"] == 2


@requires_tesseract
def test_real_parallel_batch_completes(text_image, tmp_path):
    second = tmp_path / "second.png"
    shutil.copy(text_image, second)
    batch = convert_batch(
        [text_image, str(second)], str(tmp_path / "out"),
        lang="eng", batch_workers=2,
    )
    assert batch.ok
    assert len(batch.results) == 2
    assert all(os.path.exists(r.output_path) for r in batch.results)
