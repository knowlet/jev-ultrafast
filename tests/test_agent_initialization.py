"""Offline ownership contracts for failures after a browser has been created."""

import base64
import binascii
from pathlib import Path
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent as loop


@pytest.fixture
def browser(monkeypatch):
    instance = Mock(observe=Mock(return_value={"screenshot": base64.b64encode(b"first frame").decode()}))
    monkeypatch.setattr(loop, "Browser", Mock(return_value=instance))
    return instance


def test_invalid_record_path_is_rejected_before_creating_browser(browser):
    with pytest.raises(TypeError):
        loop.Agent("https://example.test/", "Find a book", record_dir=object())

    loop.Browser.assert_not_called()
    browser.close.assert_not_called()


@pytest.mark.parametrize("cleanup_error", [None, RuntimeError, KeyboardInterrupt, SystemExit])
def test_observation_failure_preserves_original_error_and_closes_browser(browser, cleanup_error):
    failure = RuntimeError("observation failed")
    browser.observe.side_effect = failure
    if cleanup_error:
        browser.close.side_effect = cleanup_error("cleanup failed")

    with pytest.raises(RuntimeError) as caught:
        loop.Agent("https://example.test/", "Find a book")

    assert caught.value is failure
    browser.close.assert_called_once_with()


def test_recording_directory_failure_closes_browser(browser, tmp_path):
    record_dir = tmp_path / "existing-file"
    record_dir.write_text("keep this file")

    with pytest.raises(FileExistsError):
        loop.Agent("https://example.test/", "Find a book", record_dir=record_dir)

    browser.close.assert_called_once_with()
    assert record_dir.read_text() == "keep this file"


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_first_frame_write_failure_closes_browser(browser, monkeypatch, tmp_path, cleanup_fails):
    failure = OSError("frame write failed")
    monkeypatch.setattr(Path, "write_bytes", Mock(side_effect=failure))
    if cleanup_fails:
        browser.close.side_effect = RuntimeError("cleanup failed")

    with pytest.raises(OSError) as caught:
        loop.Agent("https://example.test/", "Find a book", record_dir=tmp_path / "frames")

    assert caught.value is failure
    browser.close.assert_called_once_with()


def test_invalid_initial_frame_closes_browser(browser, tmp_path):
    browser.observe.return_value = {"screenshot": "a"}

    with pytest.raises(binascii.Error):
        loop.Agent("https://example.test/", "Find a book", record_dir=tmp_path / "frames")

    browser.close.assert_called_once_with()


def test_interrupted_initialization_closes_browser(browser):
    browser.observe.side_effect = KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        loop.Agent("https://example.test/", "Find a book")

    browser.close.assert_called_once_with()


@pytest.mark.parametrize("record", [False, True])
def test_success_keeps_browser_owned_until_agent_closes(browser, tmp_path, record):
    record_dir = tmp_path / "frames" if record else None
    with loop.Agent("https://example.test/", "Find a book", record_dir=record_dir) as agent:
        browser.observe.assert_called_once_with(screenshot=record)
        browser.close.assert_not_called()
        assert agent.state["record"] is record
        if record:
            assert (record_dir / "000000.jpg").read_bytes() == b"first frame"

    browser.close.assert_called_once_with()
