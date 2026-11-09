"""Offline ownership tests. Never connect to a browser or model provider."""

from unittest.mock import Mock

import pytest

from jev_ultrafast import browser as module

STAGES = [
    "Target.attachToTarget",
    "Emulation.setDeviceMetricsOverride",
    "Emulation.setFocusEmulationEnabled",
    "Page.navigate",
    "Runtime.evaluate",
]


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_initialization_closes_owned_target(monkeypatch, stage, error_type, cleanup_fails):
    failure = error_type("synthetic initialization failure")

    def call(method, **params):
        if method == "Target.createTarget":
            return {"targetId": "owned-test-target"}
        if method == stage:
            raise failure
        if method == "Target.attachToTarget":
            return {"sessionId": "test-session"}
        if method == "Target.closeTarget" and cleanup_fails:
            raise RuntimeError("synthetic cleanup failure")
        return {}

    cdp = Mock(side_effect=call)
    monkeypatch.setattr(module, "ensure_daemon", Mock())
    monkeypatch.setattr(module, "cdp", cdp)
    with pytest.raises(error_type) as caught:
        module.Browser("https://example.test/")
    assert caught.value is failure
    closes = [call for call in cdp.call_args_list if call.args[0] == "Target.closeTarget"]
    assert len(closes) == 1
    assert closes[0].kwargs == {"targetId": "owned-test-target"}


def test_success_retains_owned_target_until_explicit_close(monkeypatch):
    def call(method, **params):
        if method == "Target.createTarget":
            return {"targetId": "owned-test-target"}
        if method == "Target.attachToTarget":
            return {"sessionId": "test-session"}
        if method == "Runtime.evaluate":
            return {"result": {"value": "complete"}}
        return {}

    cdp = Mock(side_effect=call)
    monkeypatch.setattr(module, "ensure_daemon", Mock())
    monkeypatch.setattr(module, "cdp", cdp)
    browser = module.Browser("https://example.test/")
    assert not any(call.args[0] == "Target.closeTarget" for call in cdp.call_args_list)
    browser.close()
    browser.close()
    assert sum(call.args[0] == "Target.closeTarget" for call in cdp.call_args_list) == 1


def test_creation_failure_does_not_close_unknown_target(monkeypatch):
    failure = RuntimeError("creation failed")
    cdp = Mock(side_effect=failure)
    monkeypatch.setattr(module, "ensure_daemon", Mock())
    monkeypatch.setattr(module, "cdp", cdp)
    with pytest.raises(RuntimeError) as caught:
        module.Browser("https://example.test/")
    assert caught.value is failure
    assert cdp.call_count == 1
