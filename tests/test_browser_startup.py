"""Offline startup ownership contracts; no real browser or model calls."""

import pytest

from jev_ultrafast.browser import Browser


class Transport:
    def __init__(self, failure=None, error=None, cleanup_error=None):
        self.failure = failure
        self.error = error or RuntimeError("startup failed")
        self.cleanup_error = cleanup_error
        self.calls = []
        self.targets = {"existing-user-tab"}

    def __call__(self, method, **params):
        self.calls.append((method, params))
        if method == self.failure:
            raise self.error
        if method == "Target.createTarget":
            self.targets.add("owned-target")
            return {"targetId": "owned-target"}
        if method == "Target.attachToTarget":
            return {"sessionId": "owned-session"}
        if method == "Target.closeTarget":
            if self.cleanup_error:
                raise self.cleanup_error
            self.targets.remove(params["targetId"])
            return {"success": True}
        if method == "Runtime.evaluate":
            return {"result": {"value": "complete"}}
        return {}


@pytest.mark.parametrize("stage", [
    "Target.attachToTarget", "Emulation.setDeviceMetricsOverride",
    "Emulation.setFocusEmulationEnabled", "Page.navigate", "Runtime.evaluate",
])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_startup_failure_closes_only_owned_target(stage, error_type):
    error = error_type("original startup error")
    transport = Transport(stage, error)
    with pytest.raises(error_type) as caught:
        Browser("about:blank", transport=transport)
    assert caught.value is error
    assert transport.targets == {"existing-user-tab"}
    assert transport.calls[-1] == ("Target.closeTarget", {"targetId": "owned-target"})
    assert sum(method == "Target.closeTarget" for method, _ in transport.calls) == 1
    assert not any(method in {"Browser.close", "Target.getTargets"} for method, _ in transport.calls)


@pytest.mark.parametrize("cleanup_error", [RuntimeError("cleanup failed"), KeyboardInterrupt("cleanup interrupted")])
def test_cleanup_failure_preserves_original_exception_without_retry(cleanup_error):
    transport = Transport("Page.navigate", cleanup_error=cleanup_error)
    with pytest.raises(RuntimeError) as caught:
        Browser("about:blank", transport=transport)
    assert caught.value is transport.error
    assert sum(method == "Target.closeTarget" for method, _ in transport.calls) == 1


def test_target_creation_failure_does_not_close_unowned_targets():
    transport = Transport("Target.createTarget")
    with pytest.raises(RuntimeError) as caught:
        Browser("about:blank", transport=transport)
    assert caught.value is transport.error
    assert transport.targets == {"existing-user-tab"}
    assert [method for method, _ in transport.calls] == ["Target.createTarget"]


def test_success_retains_owned_target_until_explicit_close():
    transport = Transport()
    browser = Browser("about:blank", transport=transport)
    assert transport.targets == {"existing-user-tab", "owned-target"}
    assert not any(method == "Target.closeTarget" for method, _ in transport.calls)
    assert all(params["session_id"] == "owned-session" for method, params in transport.calls
               if method.startswith(("Emulation.", "Page.", "Runtime.")))
    browser.close()
    browser.close()
    assert transport.targets == {"existing-user-tab"}
    assert sum(method == "Target.closeTarget" for method, _ in transport.calls) == 1
