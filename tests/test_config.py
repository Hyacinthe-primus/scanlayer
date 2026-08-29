import pytest

from scanlayer import config


def test_configure_sets_module_attribute():
    config.configure(tesseract_cmd="/opt/custom/tesseract")
    assert config.TESSERACT_CMD == "/opt/custom/tesseract"


def test_configure_multiple_keys_at_once():
    config.configure(lang="deu", min_word_confidence=50, jpeg_quality=95)
    assert config.DEFAULT_OCR_LANG == "deu"
    assert config.MIN_WORD_CONFIDENCE == 50
    assert config.PDF_JPEG_QUALITY == 95


def test_configure_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown option"):
        config.configure(not_a_real_setting="x")


def test_configure_unknown_key_does_not_partially_apply():
    """A bad key in a multi-key call must not silently apply the good
    keys before raising, configure() validates all keys up front.
    """
    before = config.TESSERACT_CMD
    with pytest.raises(ValueError):
        config.configure(tesseract_cmd="/should/not/stick", bogus="x")
    assert config.TESSERACT_CMD == before


def test_get_settings_round_trips_configure():
    config.configure(tesseract_cmd="/opt/tess2", default_dpi=250)
    settings = config.get_settings()
    assert settings["tesseract_cmd"] == "/opt/tess2"
    assert settings["default_dpi"] == 250


def test_get_settings_covers_every_configurable_key():
    settings = config.get_settings()
    assert set(settings) == set(config._CONFIGURABLE_KEYS)


def test_configure_log_level_updates_existing_loggers():
    import logging

    from scanlayer.utils.logger import get_logger

    log = get_logger("scanlayer.tests.some_module")
    config.configure(log_level="ERROR")
    assert log.level == logging.ERROR

    config.configure(log_level="DEBUG")
    assert log.level == logging.DEBUG


def test_configure_invalid_log_level_raises():
    with pytest.raises(ValueError):
        config.configure(log_level="NOT_A_LEVEL")
