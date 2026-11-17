"""CLI verbosity flags must reach loggers created at import time."""

import logging

from scanlayer.cli import run
from scanlayer.utils import logger as logger_module
from scanlayer.utils.logger import get_logger


def _restore_levels(levels):
    for logger, level in levels.items():
        logger.setLevel(level)


def test_cli_verbose_updates_existing_loggers(monkeypatch):
    logger = get_logger("tests.cli_verbosity")
    levels = {item: item.level for item in logger_module._known_loggers}
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    try:
        run._configure_logging(verbose=True, quiet=False)
        assert logger.level == logging.DEBUG
    finally:
        _restore_levels(levels)


def test_cli_quiet_updates_existing_loggers(monkeypatch):
    logger = get_logger("tests.cli_verbosity")
    levels = {item: item.level for item in logger_module._known_loggers}
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    try:
        run._configure_logging(verbose=False, quiet=True)
        assert logger.level == logging.ERROR
    finally:
        _restore_levels(levels)


def test_cli_default_verbosity_leaves_existing_levels_alone(monkeypatch):
    logger = get_logger("tests.cli_verbosity")
    logger.setLevel(logging.INFO)
    levels = {item: item.level for item in logger_module._known_loggers}
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    try:
        run._configure_logging(verbose=False, quiet=False)
        assert logger.level == logging.INFO
    finally:
        _restore_levels(levels)
