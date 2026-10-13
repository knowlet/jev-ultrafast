"""Offline isolation contracts; these tests never start Chromium or Browser Harness."""

import json
import os
import signal
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jev_ultrafast import browser as browser_module
from jev_ultrafast.agent import Agent
from jev_ultrafast.isolated_browser import CDPTransport, IsolatedChromium, allowed_url, normalize_fixture_origin

ORIGIN = "http://127.0.0.1:8767"
ENDPOINT = "ws://127.0.0.1:9222/devtools/browser/owned-test"


class FakeWebSocket:
    def __init__(self, messages=()):
        self.messages = list(messages)
        self.sent = []
        self.closed = False

    def send(self, message):
        self.sent.append(json.loads(message))

    def recv(self, *, timeout):
        assert timeout > 0
        return json.dumps(self.messages.pop(0))

    def close(self):
        self.closed = True


@pytest.mark.parametrize("origin", [
    "https://127.0.0.1:8767", "http://localhost:8767", "http://8.8.8.8:8767", "http://127.0.0.1",
    "http://user:password@127.0.0.1:8767", "http://127.0.0.1:8767/path", "http://127.0.0.1:8767?x=1",
])
def test_fixture_origin_requires_explicit_numeric_loopback_http(origin):
    with pytest.raises(ValueError, match="Fixture origin"):
        normalize_fixture_origin(origin)


def test_fixture_origin_normalizes_ipv6_and_trailing_slash():
    assert normalize_fixture_origin(ORIGIN + "/") == ORIGIN
    assert normalize_fixture_origin("http://[::1]:8767/") == "http://[::1]:8767"


@pytest.mark.parametrize("url", [ORIGIN + "/fixture.html?q=Lisbon#detail", "about:blank", "data:text/html,hello"])
def test_only_fixture_and_inert_urls_are_allowed(url):
    assert allowed_url(url, ORIGIN)


@pytest.mark.parametrize("url", [
    "https://example.com", "http://127.0.0.1:8768/", "https://127.0.0.1:8767/", "ws://127.0.0.1:8767/socket",
    "file:///etc/passwd", "about:net-internals", "http://[broken", None,
])
def test_other_origins_schemes_and_malformed_urls_are_rejected(url):
    assert not allowed_url(url, ORIGIN)


@pytest.mark.parametrize("endpoint", [
    "ws://example.com:9222/devtools/browser/test", "ws://127.0.0.1/devtools/browser/test",
    "ws://127.0.0.1:9222/devtools/page/test", "ws://user@127.0.0.1:9222/devtools/browser/test",
])
def test_remote_or_unowned_cdp_endpoint_rejected_before_connect(endpoint):
    with pytest.raises(ValueError, match="CDP endpoint"):
        CDPTransport(endpoint, ORIGIN)


@pytest.mark.parametrize("url,method", [
    (ORIGIN + "/fixture.html", "Fetch.continueRequest"),
    ("https://external.test/", "Fetch.failRequest"),
    ("http://127.0.0.1:9999/private", "Fetch.failRequest"),
    ("ws://127.0.0.1:8767/socket", "Fetch.failRequest"),
])
def test_interception_handles_nested_command_and_out_of_order_response(url, method):
    websocket = FakeWebSocket([
        {"method": "Fetch.requestPaused", "sessionId": "owned-session", "params": {
            "requestId": "paused-request", "request": {"url": url},
        }},
        {"id": 1, "result": {"result": {"value": "complete"}}},
        {"id": 2, "result": {}},
    ])
    transport = CDPTransport(ENDPOINT, ORIGIN, websocket=websocket)
    assert transport("Runtime.evaluate", session_id="owned-session", expression="document.readyState") == {
        "result": {"value": "complete"},
    }
    assert websocket.sent[1]["method"] == method
    assert websocket.sent[1]["sessionId"] == "owned-session"
    assert transport.blocked_requests == int(method == "Fetch.failRequest")
    assert not transport.pending


def test_new_targets_are_intercepted_before_resuming():
    websocket = FakeWebSocket([
        {"method": "Target.attachedToTarget", "params": {"sessionId": "new-session"}},
        {"id": 1, "result": {}}, {"id": 2, "result": {}}, {"id": 3, "result": {}},
    ])
    transport = CDPTransport(ENDPOINT, ORIGIN, websocket=websocket)
    transport("Target.setAutoAttach", autoAttach=True, waitForDebuggerOnStart=True, flatten=True)
    assert [call["method"] for call in websocket.sent] == [
        "Target.setAutoAttach", "Fetch.enable", "Runtime.runIfWaitingForDebugger",
    ]
    assert "new-session" in transport.configured_sessions


def test_direct_navigation_outside_fixture_fails_before_sending():
    websocket = FakeWebSocket()
    transport = CDPTransport(ENDPOINT, ORIGIN, websocket=websocket)
    with pytest.raises(ValueError, match="fixture origin"):
        transport("Page.navigate", session_id="owned-session", url="https://external.test")
    assert not websocket.sent


def test_browser_uses_injected_transport_without_harness(monkeypatch):
    daemon = Mock(side_effect=AssertionError("Default daemon must not be touched"))
    monkeypatch.setattr(browser_module, "ensure_daemon", daemon)
    calls = []

    def transport(method, **params):
        calls.append((method, params))
        return {
            "Target.createTarget": {"targetId": "owned-target"},
            "Target.attachToTarget": {"sessionId": "owned-session"},
            "Runtime.evaluate": {"result": {"value": "complete"}},
        }.get(method, {})

    browser = browser_module.Browser(ORIGIN, transport=transport)
    browser.close()
    daemon.assert_not_called()
    assert calls[-1] == ("Target.closeTarget", {"targetId": "owned-target"})
    assert next(params for method, params in calls if method == "Page.navigate")["session_id"] == "owned-session"


def test_snapshot_uses_injected_transport():
    state = {"url": ORIGIN, "text": "fixture", "actions": [], "scroll": {"y": 0}}
    transport = Mock(return_value={"result": {"value": state}})
    observed = browser_module.browser_operation({"operation": "observe", "session": "owned", "screenshot": False},
                                                transport=transport)
    assert observed["fingerprint"] == browser_module.fingerprint(state)
    assert transport.call_args.kwargs["session_id"] == "owned"


def test_agent_uses_browser_factory_and_closes_owned_browser():
    page = {"url": ORIGIN, "text": "fixture", "actions": [], "scroll": {"y": 0}}
    browser = Mock(observe=Mock(return_value=page))
    factory = Mock(return_value=browser)
    with Agent(ORIGIN, "Search for Lisbon", browser_factory=factory) as agent:
        assert agent.state["page"] == page
    factory.assert_called_once_with(ORIGIN)
    browser.close.assert_called_once_with()


def test_missing_executable_fails_before_startup(tmp_path):
    with pytest.raises(ValueError, match="existing executable"):
        IsolatedChromium(executable=tmp_path / "missing", profile_root=tmp_path, fixture_origin=ORIGIN)


def test_cleanup_only_terminates_the_owned_process_and_profile(monkeypatch):
    chromium = IsolatedChromium.__new__(IsolatedChromium)
    chromium.transport = Mock()
    chromium.process = Mock(pid=4242, returncode=None)
    chromium.proxy = Mock()
    chromium.proxy_thread = Mock()
    chromium.stderr_file = Mock()
    chromium.profile = Mock()
    usage = SimpleNamespace(ru_maxrss=123456, ru_utime=0.3, ru_stime=0.2)
    wait4 = Mock(side_effect=[(0, 0, None), (4242, 15, usage)])
    kill = Mock()
    monkeypatch.setattr(os, "wait4", wait4)
    monkeypatch.setattr(os, "kill", kill)
    process, profile, transport = chromium.process, chromium.profile, chromium.transport
    chromium.close()
    kill.assert_called_once_with(4242, signal.SIGTERM)
    assert wait4.call_args_list == [((4242, os.WNOHANG),), ((4242, os.WNOHANG),)]
    process.poll.assert_not_called()
    process.wait.assert_not_called()
    assert chromium.resource_usage["peak_rss_bytes"] == 123456 * (1 if sys.platform == "darwin" else 1024)
    assert chromium.resource_usage["exit_code"] == -15
    assert chromium.resource_usage["user_seconds"] == 0.3
    assert chromium.resource_usage["system_seconds"] == 0.2
    assert "not simultaneous process-tree total" in chromium.resource_usage["scope"]
    profile.cleanup.assert_called_once_with()
    transport.close.assert_called_once_with()
    chromium.close()


def test_already_exited_owned_process_preserves_usage_without_signalling(monkeypatch):
    chromium = IsolatedChromium.__new__(IsolatedChromium)
    chromium.process = Mock(pid=4242, returncode=None)
    usage = SimpleNamespace(ru_maxrss=98765, ru_utime=0.5, ru_stime=0.1)
    wait4, kill = Mock(return_value=(4242, 0, usage)), Mock()
    monkeypatch.setattr(os, "wait4", wait4)
    monkeypatch.setattr(os, "kill", kill)
    chromium._stop_owned_process()
    assert chromium.resource_usage["exit_code"] == 0
    assert chromium.process.returncode == 0
    kill.assert_not_called()
