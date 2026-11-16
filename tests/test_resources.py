"""Tests for resource guards: log levels, OpenCV threads, OCR budget."""

import threading
import time

import cv2
import numpy as np
import pytest
from PIL import Image

from scanlayer import config
from scanlayer import main as main_module
from scanlayer.main import BatchJob, BatchOptions, BatchPlan, execute_plan
from scanlayer.ocr import engine
from scanlayer.utils.logger import get_logger, temporary_log_level
from scanlayer.utils.resources import temporary_cv2_threads


def test_temporary_log_level_restores_success():
    logger = get_logger("scanlayer.tests.resources")
    logger.setLevel(10)
    previous_config = config.LOG_LEVEL

    with temporary_log_level("DEBUG"):
        assert logger.level == 10
        assert config.LOG_LEVEL == "DEBUG"

    assert logger.level == 10
    assert config.LOG_LEVEL == previous_config


def test_temporary_log_level_restores_on_exception():
    logger = get_logger("scanlayer.tests.resources")
    logger.setLevel(30)
    previous_config = config.LOG_LEVEL

    with pytest.raises(RuntimeError):
        with temporary_log_level("ERROR"):
            assert logger.level == 40
            raise RuntimeError("boom")

    assert logger.level == 30
    assert config.LOG_LEVEL == previous_config


@pytest.mark.parametrize("verbosity", [{"verbose": True}, {"quiet": True}])
def test_convert_batch_verbosity_restores_after_success(
    monkeypatch, tmp_path, verbosity
):
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 20), "white").save(source)
    logger = get_logger("main")
    old_level = logger.level
    old_config = config.LOG_LEVEL

    monkeypatch.setattr(
        main_module, "convert",
        lambda **kwargs: main_module.ConversionResult(
            output_path=kwargs["output_path"], words_count=0,
            mean_confidence=0.0, best_psm=None, early_exited=False,
            pdf_size_bytes=0, elapsed_ms=0.0, language_used="eng",
        ),
    )
    batch = main_module.convert_batch(
        [str(source)], str(tmp_path / "out.pdf"), **verbosity,
    )
    assert batch.ok
    assert logger.level == old_level
    assert config.LOG_LEVEL == old_config


def test_convert_batch_verbosity_restores_after_failure(monkeypatch, tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 20), "white").save(source)
    logger = get_logger("main")
    old_level = logger.level
    old_config = config.LOG_LEVEL

    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(main_module, "execute_plan", fail)
    with pytest.raises(RuntimeError):
        main_module.convert_batch(
            [str(source)], str(tmp_path / "out.pdf"), verbose=True,
        )

    assert logger.level == old_level
    assert config.LOG_LEVEL == old_config


@pytest.mark.parametrize("failure", [False, True])
def test_temporary_cv2_threads_restores(failure):
    previous = cv2.getNumThreads()
    target = 1 if previous != 1 else 2
    try:
        if failure:
            with pytest.raises(RuntimeError):
                with temporary_cv2_threads(target):
                    assert cv2.getNumThreads() == target
                    raise RuntimeError("boom")
        else:
            with temporary_cv2_threads(target):
                assert cv2.getNumThreads() == target
        assert cv2.getNumThreads() == previous
    finally:
        cv2.setNumThreads(previous)


def test_temporary_cv2_threads_none_leaves_host_unchanged():
    previous = cv2.getNumThreads()
    with temporary_cv2_threads(None):
        assert cv2.getNumThreads() == previous
    assert cv2.getNumThreads() == previous


def test_execute_plan_restores_cv2_threads_on_failure(monkeypatch, tmp_path):
    previous = cv2.getNumThreads()
    target = 1 if previous != 1 else 2
    job = BatchJob(
        input_path=str(tmp_path / "in.png"),
        output_path=str(tmp_path / "out.pdf"),
        source_path=str(tmp_path / "in.png"),
    )

    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(main_module, "convert", fail)
    execution = execute_plan(
        BatchPlan(jobs=[job], failures=[]),
        BatchOptions(opencv_threads=target),
    )
    assert execution.results == []
    assert cv2.getNumThreads() == previous


def test_ocr_budget_is_shared():
    assert engine.get_ocr_slots() is engine.get_ocr_slots()


def test_ocr_budget_default_limit():
    config.configure(ocr_max_concurrency=2)
    slots = engine.get_ocr_slots()
    acquired = sum(1 for _ in range(5) if slots.acquire(blocking=False))
    try:
        assert acquired == 2
    finally:
        for _ in range(acquired):
            slots.release()


def test_extract_words_uses_shared_budget(monkeypatch):
    config.configure(
        ocr_max_concurrency=2,
        psm_candidates=[3, 4, 6, 11],
        psm_parallel=True,
        psm_max_workers=4,
        psm_early_exit_confidence=101.0,
    )
    lock = threading.Lock()
    state = {"running": 0, "max": 0}

    def fake_run_ocr(*args, **kwargs):
        with lock:
            state["running"] += 1
            state["max"] = max(state["max"], state["running"])
        time.sleep(0.03)
        with lock:
            state["running"] -= 1
        return engine.PsmAttempt(
            psm=args[2], words=[], mean_confidence=0.0, elapsed_ms=0.0,
        )

    monkeypatch.setattr(engine, "_run_ocr", fake_run_ocr)
    engine.extract_words(
        np.full((10, 10), 255, dtype=np.uint8), 1.0, 300, lang="eng",
    )
    assert state["max"] <= 2
