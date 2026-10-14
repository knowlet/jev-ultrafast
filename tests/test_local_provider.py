"""Offline local routing contracts. Mock transports are plumbing tests, not performance evidence."""

import json
from unittest.mock import Mock

import httpx
import pytest

from jev_ultrafast import model


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    for name in (
        "S1_DECISION_URL", "S1_TEXT_BASE_URL", "TYPESAFE_API_KEY", "TYPESAFE_MODEL",
        "TEXT_MODEL_API_KEY", "TEXT_MODEL", "TEXT_MODEL_BASE_URL", "TEXT_MODEL_REASONING",
    ):
        monkeypatch.delenv(name, raising=False)


def page(clicks=2):
    return {
        "url": "http://127.0.0.1:8768/", "title": "Local fixture", "text": "Search Zurich",
        "actions": [
            {"id": "name", "kind": "fill", "node": 1, "label": "City", "role": "textbox", "value": ""},
            *[
                {
                    "id": f"button{i}", "kind": "click", "node": i + 2, "label": f"Button {i}",
                    "role": "button", "checked": False,
                }
                for i in range(clicks)
            ],
        ],
    }


def answer(labels, selected):
    labels = list(labels)
    probabilities = {label: 0.2 / (len(labels) - 1) for label in labels}
    probabilities[selected] = 0.8
    return {"choice": selected, "confidence": 0.8, "probabilities": probabilities}


@pytest.fixture
def local_client(monkeypatch):
    clients = []

    def install(handler, **kwargs):
        client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False, **kwargs)
        clients.append(client)
        monkeypatch.setattr(model, "LOCAL_CLIENT", client)
        monkeypatch.setattr(model, "CLIENT", Mock(post=Mock(side_effect=AssertionError("Unexpected remote request"))))
        return client

    yield install
    for client in clients:
        client.close()


def test_local_transport_ignores_proxies_and_disables_redirects():
    assert model.LOCAL_CLIENT.trust_env is False
    assert model.LOCAL_CLIENT.follow_redirects is False
    assert model.LOCAL_CLIENT.timeout.read == 90


@pytest.mark.parametrize("url", [
    "http://localhost:8767/decide", "http://0.0.0.0:8767/decide", "https://api.typesafe.ai/v1/systemone",
    "http://127.0.0.1.example.com/decide", "http://2130706433/decide", "http://[::ffff:127.0.0.1]/decide",
    "http://user:password@127.0.0.1/decide", "http://127.0.0.1/decide?target=external",
    "http://127.0.0.1/decide#fragment", "ftp://127.0.0.1/decide", "",
])
def test_nonliteral_or_ambiguous_local_destination_fails_before_network(monkeypatch, url):
    monkeypatch.setenv("S1_DECISION_URL", url)
    post = Mock(side_effect=AssertionError("Request must not run"))
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="numeric loopback"):
        model.choose(page(), "Find Zurich", [])
    post.assert_not_called()


@pytest.mark.parametrize("url", ["http://127.0.0.1:8767/decide", "http://[::1]:8767/v1/systemone"])
def test_numeric_loopback_urls_are_supported(url):
    assert model.local_url(url, "S1_DECISION_URL") == url


def test_local_native_contract_serialization_and_real_probabilities(monkeypatch, local_client):
    monkeypatch.setenv("S1_DECISION_URL", "http://127.0.0.1:8767/decide")
    monkeypatch.setenv("TYPESAFE_API_KEY", "unused-test-secret")
    requests = []
    target_distribution = {"2": 0.2, "3": 0.8}

    def handle(request):
        requests.append(request)
        body = json.loads(request.content)
        assert set(body) == {"state", "questions"}
        assert "authorization" not in request.headers
        assert json.loads(body["questions"]["operation"]["instructions"])["goal"] == "Find Zürich"
        target = body["questions"]["click_target"]
        assert json.loads(target["instructions"])["operation"] == "CLICK"
        assert json.loads(target["criteria"]["2"])["checked"] is False
        assert "type_text_target" not in body["questions"]
        return httpx.Response(200, json={
            "model": "verified-system-one-test-fixture",
            "latency_ms": 31.25,
            "native_latency_ms": 29.0,
            "native_token_count": 42,
            "preprocessing": {"latency_ms": 1.5},
            "answers": {
                "operation": answer(body["questions"]["operation"]["criteria"], "CLICK"),
                "click_target": {"choice": "3", "confidence": 0.8, "probabilities": target_distribution},
            },
        })

    local_client(handle)
    result = model.choose(page(), "Find Zürich", [])
    assert len(requests) == 1
    assert result["choice"] == "button1"
    assert result["target_source"] == "model"
    assert result["target_probabilities"] == target_distribution
    assert result["target_confidence"] == 0.8
    assert result["probabilities"] == {"button0": 0.2, "button1": 0.8}
    assert result["provider"] == "local_system_one"
    assert "model" not in result["request"]
    assert result["model_latency_ms"] == 31.25
    assert result["native_latency_ms"] == 29.0
    assert result["native_token_count"] == 42
    assert result["preprocessing"] == {"latency_ms": 1.5}
    assert result["raw_response"]["model"] == result["model"]


def test_singleton_target_is_deterministic_without_fabricated_model_head(monkeypatch, local_client):
    monkeypatch.setenv("S1_DECISION_URL", "http://127.0.0.1:8767/v1/systemone")

    def handle(request):
        body = json.loads(request.content)
        assert set(body["questions"]) == {"operation", "click_target"}
        return httpx.Response(200, json={
            "model": "verified-system-one-test-fixture",
            "answers": {"operation": answer(body["questions"]["operation"]["criteria"], "TYPE_TEXT")},
        })

    local_client(handle)
    result = model.choose(page(), "Find Zurich", [])
    assert result["choice"] == "name"
    assert result["target_source"] == "single_legal_target"
    assert result["target_confidence"] is None
    assert result["target_probabilities"] == {}
    assert "type_text_target" not in result["raw_answers"]
    assert result["confidence"] == 0.8
    # This describes the sole legal executor action, not a model probability/confidence.
    assert result["probabilities"] == {"name": 1.0}


def test_more_than_52_choices_fails_without_truncation_or_request(monkeypatch):
    monkeypatch.setenv("S1_DECISION_URL", "http://127.0.0.1:8767/decide")
    post = Mock(side_effect=AssertionError("Request must not run"))
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="at most 52"):
        model.choose(page(clicks=53), "Find Zurich", [])
    post.assert_not_called()


def test_generated_choice_without_probabilities_is_rejected(monkeypatch, local_client):
    monkeypatch.setenv("S1_DECISION_URL", "http://127.0.0.1:8767/decide")
    local_client(lambda request: httpx.Response(200, json={
        "model": "test", "answers": {"operation": {"choice": "TYPE_TEXT"}},
    }))
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.choose(page(), "Find Zurich", [])


@pytest.mark.parametrize("status", [301, 307, 308, 429, 503, 529])
def test_local_failures_are_not_redirected_or_retried(monkeypatch, local_client, status):
    monkeypatch.setenv("S1_DECISION_URL", "http://127.0.0.1:8767/decide")
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(status, headers={"location": "https://api.typesafe.ai/v1/systemone"})

    local_client(handle, follow_redirects=True)
    with pytest.raises(RuntimeError):
        model.choose(page(), "Find Zurich", [])
    assert len(requests) == 1
    assert requests[0].url.host == "127.0.0.1"


def test_local_connection_failure_does_not_fall_back(monkeypatch, local_client):
    monkeypatch.setenv("S1_DECISION_URL", "http://127.0.0.1:8767/decide")

    def handle(request):
        raise httpx.ConnectError("offline fixture", request=request)

    local_client(handle)
    with pytest.raises(RuntimeError, match="Model connection failed"):
        model.choose(page(), "Find Zurich", [])


def test_remote_default_preserves_upstream_contract(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    seen = []

    def post(url, key, body):
        seen.append((url, key, body))
        return {
            "model": "test", "answers": {"operation": answer(body["questions"]["operation"]["criteria"], "DONE")},
        }

    monkeypatch.setattr(model, "post_json", post)
    result = model.choose(page(), "Find Zurich", [])
    assert seen[0][0] == "https://api.typesafe.ai/v1/systemone"
    assert seen[0][1] == "test"
    assert seen[0][2]["model"] == "jev-latest"
    assert isinstance(seen[0][2]["questions"]["operation"]["instructions"], dict)
    assert "type_text_target" in seen[0][2]["questions"]
    assert result["provider"] == "typesafe"


def test_local_decision_blocks_remote_text_helper(monkeypatch):
    monkeypatch.setenv("S1_DECISION_URL", "http://127.0.0.1:8767/decide")
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "unused-test-secret")
    post = Mock(side_effect=AssertionError("Request must not run"))
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="no remote text fallback"):
        model.field_text({"goal": "Find Zurich"})
    post.assert_not_called()


def test_local_text_requires_explicit_model_identifier(monkeypatch):
    monkeypatch.setenv("S1_TEXT_BASE_URL", "http://127.0.0.1:8769/v1")
    stream = Mock(side_effect=AssertionError("Request must not run"))
    monkeypatch.setattr(model, "stream_local_text", stream)
    with pytest.raises(ValueError, match="explicit TEXT_MODEL"):
        model.field_text({"goal": "Find Zurich"})
    stream.assert_not_called()


def sse(events, done=True):
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + ("data: [DONE]\n\n" if done else "")


def test_local_text_stream_uses_no_credentials_and_reports_ttft(monkeypatch, local_client):
    monkeypatch.setenv("S1_TEXT_BASE_URL", "http://127.0.0.1:8769/v1")
    monkeypatch.setenv("TEXT_MODEL", "verified-system-one-test-fixture")
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "unused-test-secret")
    seen = []

    def handle(request):
        seen.append(request)
        assert "authorization" not in request.headers
        body = json.loads(request.content)
        assert body["model"] == "verified-system-one-test-fixture"
        assert body["temperature"] == 0 and body["seed"] == 17 and body["max_tokens"] == 128
        assert body["response_format"] == {"type": "json_object"}
        assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
        assert "reasoning" not in body and "thinking" not in body
        assert json.loads(body["messages"][1]["content"]) == {"goal": "Find Zurich"}
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse([
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"index": 0, "delta": {"content": '{"text":'}}]},
            {"choices": [{"index": 0, "delta": {"content": '"Zurich"}'}}]},
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 7}},
        ]))

    local_client(handle)
    text, info = model.field_text({"goal": "Find Zurich"})
    assert len(seen) == 1 and seen[0].url.path == "/v1/chat/completions"
    assert text == "Zurich"
    assert info["ttft_ms"] is not None and 0 <= info["ttft_ms"] <= info["latency_ms"] + 1
    assert info["usage"] == {"prompt_tokens": 12, "completion_tokens": 7}
    assert info["provider"] == "local_openai_compatible"


@pytest.mark.parametrize("response", [
    httpx.Response(307, headers={"location": "https://api.deepseek.com/v1/chat/completions"}),
    httpx.Response(200, headers={"content-type": "application/json"}, json={"choices": []}),
    httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse([
        {"choices": [{"delta": {"content": '{"text":"Zurich"}'}}]},
    ], done=False)),
    httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse([{"error": "fixture error"}])),
])
def test_local_text_stream_failure_never_falls_back_or_types(monkeypatch, local_client, response):
    monkeypatch.setenv("S1_TEXT_BASE_URL", "http://127.0.0.1:8769/v1")
    monkeypatch.setenv("TEXT_MODEL", "verified-system-one-test-fixture")
    seen = []

    def handle(request):
        seen.append(request)
        return response

    local_client(handle, follow_redirects=True)
    with pytest.raises((RuntimeError, ValueError)):
        model.field_text({"goal": "Find Zurich"})
    assert len(seen) == 1
    assert seen[0].url.host == "127.0.0.1"
