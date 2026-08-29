"""Tests for scanlayer.utils.validators, plus the platform-dependent
Tesseract resolution helpers in scanlayer.config.

The config tests mock platform.system()/shutil.which()/os.path.exists()
so that Windows, macOS (Darwin), and Linux resolution paths are all
exercised regardless of which OS actually runs the suite.
"""

import os

import pytest
from PIL import Image

from scanlayer import config
from scanlayer.utils.validators import (
    InputFileError,
    OutputPathError,
    TesseractEnvironmentError,
    validate_all,
    validate_image_readable,
    validate_input_file,
    validate_output_path,
    validate_tesseract_environment,
)

# --------------------------------------------------------------------
# validate_input_file
# --------------------------------------------------------------------

def test_validate_input_file_missing_raises():
    with pytest.raises(InputFileError, match="does not exist"):
        validate_input_file("/definitely/does/not/exist.jpg")


def test_validate_input_file_directory_raises(tmp_path):
    with pytest.raises(InputFileError, match="not a file"):
        validate_input_file(str(tmp_path))


def test_validate_input_file_empty_raises(tmp_path):
    p = tmp_path / "empty.jpg"
    p.write_bytes(b"")
    with pytest.raises(InputFileError, match="empty"):
        validate_input_file(str(p))


def test_validate_input_file_no_path_raises():
    with pytest.raises(InputFileError):
        validate_input_file("")
    with pytest.raises(InputFileError):
        validate_input_file(None)


def test_validate_input_file_unreadable_raises(tmp_path, monkeypatch):
    p = tmp_path / "locked.jpg"
    p.write_bytes(b"not empty")

    # os.access-based permission check: mocked so the assertion holds
    # the same on Windows (where chmod 0o000 doesn't reliably block
    # the owning user) as on Linux/macOS.
    monkeypatch.setattr(os, "access", lambda *a, **kw: False)
    with pytest.raises(InputFileError, match="not readable"):
        validate_input_file(str(p))


def test_validate_input_file_returns_resolved_absolute_path(tmp_path):
    p = tmp_path / "invoice.jpg"
    p.write_bytes(b"not empty")
    resolved = validate_input_file(str(p))
    assert os.path.isabs(resolved)
    assert os.path.normcase(resolved) == os.path.normcase(str(p.resolve()))


def test_validate_input_file_unsupported_extension_warns_not_raises(tmp_path, caplog):
    p = tmp_path / "scan.xyz"
    p.write_bytes(b"not empty")
    with caplog.at_level("WARNING"):
        validate_input_file(str(p))
    assert "not in the supported list" in caplog.text


def test_validate_input_file_expands_user_home(tmp_path, monkeypatch):
    # `~` expansion must work regardless of the platform's home-dir
    # convention; Path.expanduser() handles that, this just checks
    # scanlayer actually calls it. HOME is what posixpath.expanduser
    # reads; ntpath.expanduser (Windows) ignores HOME entirely and
    # checks USERPROFILE first, so both must be set for this to
    # actually exercise the code path on every platform.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    p = tmp_path / "invoice.jpg"
    p.write_bytes(b"not empty")
    resolved = validate_input_file("~/invoice.jpg")
    assert os.path.normcase(resolved) == os.path.normcase(str(p.resolve()))


# --------------------------------------------------------------------
# validate_output_path
# --------------------------------------------------------------------

def test_validate_output_path_no_path_raises():
    with pytest.raises(OutputPathError):
        validate_output_path("")


def test_validate_output_path_points_to_existing_directory_raises(tmp_path):
    with pytest.raises(OutputPathError, match="directory, not a file"):
        validate_output_path(str(tmp_path))


def test_validate_output_path_missing_parent_raises(tmp_path):
    missing_parent = tmp_path / "does" / "not" / "exist" / "out.pdf"
    with pytest.raises(OutputPathError, match="does not exist"):
        validate_output_path(str(missing_parent))


def test_validate_output_path_parent_not_writable_raises(tmp_path, monkeypatch):
    out = tmp_path / "out.pdf"
    monkeypatch.setattr(os, "access", lambda *a, **kw: False)
    with pytest.raises(OutputPathError, match="not writable"):
        validate_output_path(str(out))


def test_validate_output_path_mismatched_extension_warns_not_raises(tmp_path, caplog):
    out = tmp_path / "out.txt"
    with caplog.at_level("WARNING"):
        resolved = validate_output_path(str(out), expected_ext=".pdf")
    assert "does not have the .pdf extension" in caplog.text
    assert os.path.normcase(resolved) == os.path.normcase(str(out.resolve()))


def test_validate_output_path_returns_resolved_absolute_path(tmp_path):
    out = tmp_path / "out.pdf"
    resolved = validate_output_path(str(out))
    assert os.path.isabs(resolved)


# --------------------------------------------------------------------
# validate_image_readable
# --------------------------------------------------------------------

def test_validate_image_readable_corrupted_raises(tmp_path):
    bad = tmp_path / "corrupt.jpg"
    bad.write_bytes(b"this is not an image")
    with pytest.raises(InputFileError, match="Unreadable or corrupted"):
        validate_image_readable(str(bad))


def test_validate_image_readable_accepts_real_image(tmp_path):
    good = tmp_path / "ok.png"
    Image.new("RGB", (10, 10), "white").save(good)
    validate_image_readable(str(good))  # must not raise


# --------------------------------------------------------------------
# validate_tesseract_environment
# --------------------------------------------------------------------

def test_validate_tesseract_environment_missing_binary_raises(monkeypatch):
    monkeypatch.setattr(config, "TESSERACT_CMD", "/no/such/tesseract")
    with pytest.raises(TesseractEnvironmentError, match="not found"):
        validate_tesseract_environment()


def test_validate_tesseract_environment_not_executable_raises(tmp_path, monkeypatch):
    fake = tmp_path / "tesseract"
    fake.write_bytes(b"")
    monkeypatch.setattr(config, "TESSERACT_CMD", str(fake))
    monkeypatch.setattr(os, "access", lambda *a, **kw: False)
    with pytest.raises(TesseractEnvironmentError, match="not executable"):
        validate_tesseract_environment()


def test_validate_tesseract_environment_not_on_path_raises(monkeypatch):
    import shutil

    monkeypatch.setattr(config, "TESSERACT_CMD", "tesseract")
    monkeypatch.setattr(shutil, "which", lambda *_a, **_kw: None)
    with pytest.raises(TesseractEnvironmentError, match="system PATH"):
        validate_tesseract_environment()


def test_validate_tesseract_environment_on_path_succeeds(monkeypatch):
    import shutil

    monkeypatch.setattr(config, "TESSERACT_CMD", "tesseract")
    monkeypatch.setattr(config, "TESSDATA_DIR", None)
    monkeypatch.setattr(shutil, "which", lambda *_a, **_kw: "/usr/bin/tesseract")
    validate_tesseract_environment()  # must not raise


def test_validate_tesseract_environment_missing_tessdata_dir_raises(tmp_path, monkeypatch):
    fake = tmp_path / "tesseract"
    fake.write_bytes(b"")
    fake.chmod(0o755)
    monkeypatch.setattr(config, "TESSERACT_CMD", str(fake))
    monkeypatch.setattr(os, "access", lambda *a, **kw: True)
    monkeypatch.setattr(config, "TESSDATA_DIR", str(tmp_path / "no_such_tessdata"))
    with pytest.raises(TesseractEnvironmentError, match="tessdata directory"):
        validate_tesseract_environment()


def test_validate_tesseract_environment_missing_lang_files_warns(tmp_path, monkeypatch, caplog):
    fake = tmp_path / "tesseract"
    fake.write_bytes(b"")
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    monkeypatch.setattr(config, "TESSERACT_CMD", str(fake))
    monkeypatch.setattr(os, "access", lambda *a, **kw: True)
    monkeypatch.setattr(config, "TESSDATA_DIR", str(tessdata))
    with caplog.at_level("WARNING"):
        validate_tesseract_environment()  # must not raise
    assert "Missing tessdata" in caplog.text


# --------------------------------------------------------------------
# validate_all
# --------------------------------------------------------------------

def test_validate_all_missing_input_raises_before_touching_output(tmp_path):
    # Output path is deliberately invalid too (missing parent); if
    # validate_all checked output first, we'd get OutputPathError
    # instead, so this also pins the "fastest first" check order.
    with pytest.raises(InputFileError):
        validate_all(
            str(tmp_path / "missing.jpg"),
            str(tmp_path / "no" / "such" / "dir" / "out.pdf"),
        )


def test_validate_all_returns_resolved_paths(tmp_path, monkeypatch):
    import shutil

    src = tmp_path / "invoice.jpg"
    Image.new("RGB", (10, 10), "white").save(src)
    out = tmp_path / "out.pdf"

    monkeypatch.setattr(config, "TESSERACT_CMD", "tesseract")
    monkeypatch.setattr(shutil, "which", lambda *_a, **_kw: "/usr/bin/tesseract")

    input_abs, output_abs = validate_all(str(src), str(out))
    assert os.path.isabs(input_abs)
    assert os.path.isabs(output_abs)


# --------------------------------------------------------------------
# Cross-platform Tesseract discovery (scanlayer.config)
#
# These mock platform.system(), shutil.which(), and os.path.exists()
# so all three OS branches run identically no matter which OS
# actually executes the test suite.
# --------------------------------------------------------------------

@pytest.mark.parametrize("system, expected_name", [
    ("Windows", "tesseract.exe"),
    ("Darwin", "tesseract"),
    ("Linux", "tesseract"),
])
def test_find_bundled_tesseract_uses_platform_exe_name(monkeypatch, system, expected_name):
    import platform

    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(config, "BASE_DIR", "/opt/scanlayer")
    monkeypatch.setattr(os.path, "exists", lambda p: p.endswith(expected_name))

    cmd, tessdata = config._find_bundled_tesseract()
    assert cmd == os.path.join("/opt/scanlayer", "bin", "tesseract", expected_name)
    assert tessdata == os.path.join("/opt/scanlayer", "bin", "tesseract", "tessdata")


def test_find_bundled_tesseract_absent_returns_none(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    cmd, tessdata = config._find_bundled_tesseract()
    assert cmd is None
    assert tessdata is None


def test_find_system_tesseract_prefers_path(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda *_a, **_kw: "/custom/bin/tesseract")
    assert config._find_system_tesseract() == "/custom/bin/tesseract"


@pytest.mark.parametrize("system, candidates", [
    ("Windows", [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]),
    ("Darwin", [
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]),
    ("Linux", [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]),
])
def test_find_system_tesseract_checks_platform_candidates_in_order(
    monkeypatch, system, candidates
):
    import platform
    import shutil

    monkeypatch.setattr(shutil, "which", lambda *_a, **_kw: None)
    monkeypatch.setattr(platform, "system", lambda: system)
    # Only the *second* candidate "exists" -> proves every platform's
    # candidate list is walked in the documented order, not just the
    # first entry.
    monkeypatch.setattr(os.path, "exists", lambda p: p == candidates[1])

    assert config._find_system_tesseract() == candidates[1]


@pytest.mark.parametrize("system", ["Windows", "Darwin", "Linux"])
def test_find_system_tesseract_falls_back_to_bare_command(monkeypatch, system):
    import platform
    import shutil

    monkeypatch.setattr(shutil, "which", lambda *_a, **_kw: None)
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(os.path, "exists", lambda p: False)

    assert config._find_system_tesseract() == "tesseract"
