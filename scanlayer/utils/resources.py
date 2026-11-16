"""Small process-local resource guards.

OpenCV's thread count is process-global. Use this context only around a
batch execution, never around individual image conversions.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Optional

import cv2


@contextmanager
def temporary_cv2_threads(threads: Optional[int]) -> Generator[None, None, None]:
    """Temporarily set OpenCV's global thread count and restore it.

    None or <=0 means "leave the host configuration unchanged". Restoring
    happens in finally, including on exceptions.
    """
    if threads is None or threads <= 0:
        yield
        return

    previous = cv2.getNumThreads()
    if threads == previous:
        yield
        return

    cv2.setNumThreads(threads)
    try:
        yield
    finally:
        cv2.setNumThreads(previous)
