"""Benchmark helpers: metrics, resource sampling, JSON output."""

from __future__ import annotations

import gc
import json
import os
import platform
import threading
import time
from pathlib import Path
from typing import Any, Sequence

try:
    import psutil
except ImportError:  # pragma: no cover - benchmark optional dependency
    psutil = None


def word_error_rate(expected: Sequence[str], actual: Sequence[str]) -> float:
    """Word-level Levenshtein error rate: (S+D+I) / len(expected)."""
    n, m = len(expected), len(actual)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    previous = list(range(m + 1))
    for i in range(1, n + 1):
        current = [i] + [0] * m
        exp = expected[i - 1]
        for j in range(1, m + 1):
            cost = 0 if exp == actual[j - 1] else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
        previous = current
    return previous[m] / n


def normalized_text(text: str) -> str:
    return " ".join(text.split())


def text_contains_phrase(text: str, words: Sequence[str]) -> bool:
    if not words:
        return True
    return normalized_text(" ".join(words)) in normalized_text(text)


def extract_pdf_text(path: str) -> "str | None":
    """Return extracted text, or None if pypdf is not installed."""
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - benchmark optional dependency
        return None
    try:
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    except Exception:
        return None


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


class ResourceSampler(threading.Thread):
    """Sample parent + child RSS and child CPU time.

    Tesseract is a subprocess, so process-only RSS/CPU misses most of the
    real cost. Child CPU is accumulated per PID while the process is alive.
    """

    def __init__(self, interval: float = 0.02):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()
        self.peak_rss_mb: "float | None" = None
        self.child_cpu_s: "float | None" = None
        self._child_cpu: dict[int, float] = {}
        self._process = psutil.Process() if psutil is not None else None
        self._start_self_cpu = (
            sum(self._process.cpu_times()[:2]) if self._process is not None else 0.0
        )

    def run(self) -> None:
        if self._process is None:
            return
        while not self._stop.is_set():
            try:
                rss = self._process.memory_info().rss
                children = self._process.children(recursive=True)
            except Exception:
                children = []
                rss = 0
            for child in children:
                try:
                    times = child.cpu_times()
                    self._child_cpu[child.pid] = max(
                        self._child_cpu.get(child.pid, 0.0),
                        times.user + times.system,
                    )
                    rss += child.memory_info().rss
                except Exception:
                    pass
            self.peak_rss_mb = max(self.peak_rss_mb or 0.0, rss / 2**20)
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=5.0)

    @property
    def self_cpu_s(self) -> "float | None":
        if self._process is None:
            return None
        return sum(self._process.cpu_times()[:2]) - self._start_self_cpu

    @property
    def total_cpu_s(self) -> "float | None":
        self_cpu = self.self_cpu_s
        if self_cpu is None:
            return None
        return self_cpu + sum(self._child_cpu.values())


def collect_garbage() -> None:
    gc.collect()


def write_json(path: str, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_seconds(seconds: "float | None") -> str:
    return "-" if seconds is None else f"{seconds:.2f}s"


def format_mb(mb: "float | None") -> str:
    return "-" if mb is None else f"{mb:.0f}MB"


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def environment() -> dict[str, Any]:
    """Best-effort environment metadata for reproducible result files."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    if psutil is not None:
        try:
            info["physical_cpu_count"] = psutil.cpu_count(logical=False)
            info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        except Exception:
            pass
    for name, module in (
        ("Pillow", "PIL"), ("OpenCV", "cv2"), ("NumPy", "numpy"),
        ("pytesseract", "pytesseract"), ("ReportLab", "reportlab"),
    ):
        try:
            imported = __import__(module)
            info[name] = getattr(imported, "__version__", None) or getattr(
                imported, "Version", None
            )
        except Exception:
            pass
    return info


def relpath(path: str) -> str:
    try:
        return os.path.relpath(path)
    except ValueError:
        return path
