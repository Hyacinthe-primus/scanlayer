"""Tests for processing several pages of a --merge run at once.

Everything here fakes the per-page work, so no Tesseract is needed: the
point is the scheduling (order, real concurrency, the worker cap, error
reporting, the shared OCR slot limit), not the OCR itself.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from scanlayer import config
from scanlayer import main as main_module
from scanlayer.cli import main
from scanlayer.cli.dry_run import EXIT_OK, EXIT_USER_ERROR
from scanlayer.ocr import engine
from scanlayer.utils.errors import BlankPageDetectedError, OCRProcessingError

STATS = {"words": 1, "mean_conf": 90.0, "best_psm": 3, "columns": 1}


class FakePages:
    """Stands in for _process_page_for_merge and records how it was called."""

    def __init__(self, sleep=0.0, delays=None, errors=None):
        self.sleep = sleep
        self.delays = delays or {}
        self.errors = errors or {}
        self.lock = threading.Lock()
        self.running = 0
        self.max_running = 0
        self.started = []
        self.thread_ids = set()
        self.slots = []

    def __call__(
        self, input_path, lang, dpi, whitelist, blacklist, orientation, force,
        ocr_slots=None,
    ):
        with self.lock:
            self.started.append(input_path)
            self.running += 1
            self.max_running = max(self.max_running, self.running)
            self.thread_ids.add(threading.get_ident())
            self.slots.append(ocr_slots)
        try:
            time.sleep(self.delays.get(input_path, self.sleep))
            if input_path in self.errors:
                raise self.errors[input_path]
            return f"page:{input_path}", dict(STATS)
        finally:
            with self.lock:
                self.running -= 1


@pytest.fixture
def merge_env(monkeypatch, tmp_path):
    """Fake page processing + PDF writer, 8 CPUs, returns (run, built)."""
    built = []

    def fake_build(pages, output_path, jpeg_quality=None, metadata=None):
        built.append(list(pages))
        return 123

    monkeypatch.setattr(main_module, "build_searchable_pdf_multipage", fake_build)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)

    def run(fake, count, **kwargs):
        monkeypatch.setattr(main_module, "_process_page_for_merge", fake)
        paths = [f"p{i}" for i in range(count)]
        return main_module.convert_merge(paths, str(tmp_path / "out.pdf"), **kwargs)

    return run, built


def test_pages_keep_input_order_when_they_finish_out_of_order(merge_env):
    run, built = merge_env
    fake = FakePages(delays={"p0": 0.15, "p1": 0.1, "p2": 0.05})
    run(fake, 6, workers=3)
    assert built == [[f"page:p{i}" for i in range(6)]]


def test_pages_really_run_concurrently(merge_env):
    run, _ = merge_env
    fake = FakePages(sleep=0.1)
    run(fake, 6, workers=3)
    assert fake.max_running == 3


def test_workers_above_the_limit_are_capped(merge_env):
    run, _ = merge_env
    fake = FakePages(sleep=0.05)
    run(fake, 10, workers=10)
    assert fake.max_running == config.MERGE_WORKERS_LIMIT == 4


def test_one_worker_runs_on_the_calling_thread_without_slots(merge_env):
    run, built = merge_env
    fake = FakePages(sleep=0.01)
    run(fake, 4, workers=1)
    assert fake.max_running == 1
    assert fake.thread_ids == {threading.get_ident()}
    assert fake.slots == [None] * 4
    assert built == [[f"page:p{i}" for i in range(4)]]


def test_default_comes_from_configure(merge_env):
    run, _ = merge_env
    config.configure(merge_workers=2)
    fake = FakePages(sleep=0.1)
    run(fake, 6)
    assert fake.max_running == 2


def test_default_is_three(merge_env):
    run, _ = merge_env
    fake = FakePages(sleep=0.1)
    run(fake, 6)
    assert fake.max_running == 3


def test_workers_never_exceed_the_cpu_count(merge_env, monkeypatch):
    run, _ = merge_env
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    fake = FakePages(sleep=0.1)
    run(fake, 6, workers=4)
    assert fake.max_running == 2


def test_workers_never_exceed_the_page_count(merge_env):
    run, _ = merge_env
    fake = FakePages(sleep=0.05)
    run(fake, 2, workers=4)
    assert fake.max_running == 2


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
def test_invalid_workers_rejected_before_any_work(merge_env, bad):
    run, built = merge_env
    fake = FakePages()
    with pytest.raises(ValueError, match="workers"):
        run(fake, 3, workers=bad)
    assert fake.started == []
    assert built == []


def test_first_bad_page_in_input_order_is_the_one_reported(merge_env):
    run, built = merge_env
    fake = FakePages(
        delays={"p0": 0.2, "p1": 0.1},
        errors={
            "p1": BlankPageDetectedError("p1 is blank"),
            "p2": BlankPageDetectedError("p2 is blank"),
        },
    )
    with pytest.raises(BlankPageDetectedError, match="p1 is blank"):
        run(fake, 4, workers=3)
    assert built == []


def test_a_failure_drops_pages_that_have_not_started(merge_env):
    run, built = merge_env
    fake = FakePages(
        sleep=0.05, delays={"p0": 0.0}, errors={"p0": OCRProcessingError("boom")},
    )
    with pytest.raises(OCRProcessingError, match="boom"):
        run(fake, 20, workers=2)
    assert len(fake.started) < 20
    assert built == []


def test_concurrent_pages_share_one_slot_pool_sized_to_the_cpu_count(merge_env):
    run, _ = merge_env
    fake = FakePages(sleep=0.02)
    run(fake, 5, workers=3)
    slots = fake.slots[0]
    assert slots is not None and all(s is slots for s in fake.slots)
    free = sum(1 for _ in range(20) if slots.acquire(blocking=False))
    assert free == 8


def test_convert_batch_passes_workers_to_merge(monkeypatch, tmp_path, text_image):
    seen = {}

    def fake_merge(**kwargs):
        seen.update(kwargs)
        return "result"

    monkeypatch.setattr(main_module, "convert_merge", fake_merge)
    batch = main_module.convert_batch(
        [text_image, text_image], str(tmp_path / "out.pdf"), merge=True, workers=2,
    )
    assert seen["workers"] == 2
    assert batch.results == ["result"]


def _fake_psm_runner(calls):
    def fake(*args, psm, ocr_slots=None, **kwargs):
        calls.append((psm, ocr_slots))
        return engine.PsmAttempt(psm=psm, words=[], mean_confidence=0.0, elapsed_ms=0.0), None

    return fake


@pytest.mark.parametrize(
    "settings",
    [
        {"psm_candidates": [3]},
        {"psm_candidates": [3, 4, 6, 11], "psm_parallel": True},
        {"psm_candidates": [3, 4, 6, 11], "psm_parallel": False},
    ],
)
def test_every_psm_pass_gets_the_shared_slots(monkeypatch, settings):
    calls = []
    monkeypatch.setattr(engine, "_safe_run_ocr", _fake_psm_runner(calls))
    config.configure(**settings)
    slots = threading.BoundedSemaphore(2)
    engine.extract_words(
        np.zeros((10, 10), dtype=np.uint8), 1.0, 300, ocr_slots=slots,
    )
    assert len(calls) == len(settings["psm_candidates"])
    assert all(s is slots for _, s in calls)


def _racing_psm_runner(confidence, delays):
    def fake(*args, psm, ocr_slots=None, **kwargs):
        time.sleep(delays.get(psm, 0.0))
        attempt = engine.PsmAttempt(
            psm=psm, words=[], mean_confidence=confidence[psm], elapsed_ms=0.0,
        )
        return attempt, None

    return fake


def test_psm_tie_goes_to_the_first_candidate_even_if_it_finishes_last(monkeypatch):
    monkeypatch.setattr(
        engine,
        "_safe_run_ocr",
        _racing_psm_runner({3: 50.0, 4: 50.0, 6: 50.0, 11: 50.0}, {3: 0.15}),
    )
    config.configure(psm_candidates=[3, 4, 6, 11], psm_parallel=True)
    result = engine.extract_words(np.zeros((10, 10), dtype=np.uint8), 1.0, 300)
    assert result.best_psm == 3


def test_psm_higher_confidence_still_beats_an_earlier_candidate(monkeypatch):
    monkeypatch.setattr(
        engine,
        "_safe_run_ocr",
        _racing_psm_runner({3: 40.0, 4: 40.0, 6: 60.0, 11: 40.0}, {6: 0.15}),
    )
    config.configure(psm_candidates=[3, 4, 6, 11], psm_parallel=True)
    result = engine.extract_words(np.zeros((10, 10), dtype=np.uint8), 1.0, 300)
    assert result.best_psm == 6


def test_slots_bound_concurrent_tesseract_passes_and_are_released(monkeypatch):
    lock = threading.Lock()
    state = {"running": 0, "max": 0}

    def fake_run_ocr(*args, **kwargs):
        with lock:
            state["running"] += 1
            state["max"] = max(state["max"], state["running"])
        time.sleep(0.03)
        with lock:
            state["running"] -= 1
        if kwargs.get("fail"):
            raise OCRProcessingError("timeout")
        return engine.PsmAttempt(psm=3, words=[], mean_confidence=0.0, elapsed_ms=0.0)

    monkeypatch.setattr(engine, "_run_ocr", fake_run_ocr)
    slots = threading.BoundedSemaphore(2)

    def one(_):
        return engine._safe_run_ocr(None, "eng", 3, "", 300, None, None, psm=3, ocr_slots=slots)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(one, range(6)))
    assert state["max"] == 2

    failing = engine._safe_run_ocr(
        None, "eng", 3, "", 300, None, None, psm=3, ocr_slots=slots, fail=True,
    )
    assert failing[0] is None
    assert sum(1 for _ in range(5) if slots.acquire(blocking=False)) == 2


def test_workers_flag_parses():
    from scanlayer.cli.parser import _build_parser

    args = _build_parser().parse_args(["a.jpg", "-o", "x.pdf", "--merge", "--workers", "3"])
    assert args.workers == 3
    assert _build_parser().parse_args(["a.jpg"]).workers is None


def test_cli_rejects_zero_workers(text_image, tmp_path):
    out = tmp_path / "merged.pdf"
    code = main([text_image, "-o", str(out), "--merge", "--workers", "0"])
    assert code == EXIT_USER_ERROR


def test_cli_passes_workers_to_convert_merge(monkeypatch, text_image, tmp_path):
    import scanlayer.cli.run as run_module

    seen = {}
    monkeypatch.setattr(run_module, "convert_merge", lambda **kw: seen.update(kw))
    out = tmp_path / "merged.pdf"
    code = main([text_image, text_image, "-o", str(out), "--merge", "--workers", "2"])
    assert code == EXIT_OK
    assert seen["workers"] == 2
