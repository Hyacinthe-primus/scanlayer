"""
Structured logger and per-stage timer.

Provides logging configuration, a timing context manager, and uniform
message helpers. The internal `LOG_LEVEL` env var (set by the CLI's
`--verbose`/`--quiet`) overrides `config.LOG_LEVEL`, which itself comes
from the user-facing `SCANLAYER_LOG_LEVEL` env var.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Generator

from scanlayer import config

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


_known_loggers: list[logging.Logger] = []


def _resolve_level() -> int:
    """Return the effective log level.

    Priority: CLI `LOG_LEVEL` env var > `config.LOG_LEVEL` (from
    `SCANLAYER_LOG_LEVEL`) > INFO.
    """
    raw = os.environ.get("LOG_LEVEL", "").strip().upper()
    if not raw:
        raw = str(getattr(config, "LOG_LEVEL", "INFO")).upper()
    return getattr(logging, raw, logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with the configured log level.

    Idempotent: multiple calls won't overwrite the handler setup.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False  # avoid duplicates via the root logger
        _known_loggers.append(logger)
    logger.setLevel(_resolve_level())
    return logger


def set_log_level(level: str) -> None:
    """Apply a log level to all loggers created via get_logger().

    Needed because modules call get_logger(__name__) at import time,
    so a runtime configure() call wouldn't reach them otherwise.
    """
    level_value = getattr(logging, str(level).upper(), None)
    if level_value is None:
        raise ValueError(f"Unknown log level: {level!r}")
    for logger in _known_loggers:
        logger.setLevel(level_value)


@contextmanager
def stage_timer(logger: logging.Logger, stage_name: str) -> Generator[None, None, None]:
    """Context manager that logs the duration of a block.

    Timing is only logged if config.LOG_TIMING is True or DEBUG is active.
    """
    enable = bool(getattr(config, "LOG_TIMING", True)) or logger.isEnabledFor(logging.DEBUG)
    start = time.perf_counter() if enable else None
    try:
        yield
    finally:
        if enable:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(f"{stage_name}: completed in {elapsed_ms:.0f} ms")


def log_success(logger: logging.Logger, msg: str) -> None:
    logger.info(msg)


def log_warning(logger: logging.Logger, msg: str) -> None:
    logger.warning(msg)


def log_error(logger: logging.Logger, msg: str, exc: Exception | None = None) -> None:
    if exc is not None:
        logger.error(f"{msg}: {type(exc).__name__}: {exc}")
    else:
        logger.error(msg)


def log_stage_start(logger: logging.Logger, msg: str) -> None:
    logger.info(msg)
