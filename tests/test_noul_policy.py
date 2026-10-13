"""Offline Noul protocol/transport/accounting tests; synthetic scores are not model evidence."""

import asyncio
import hashlib
import json
import math
from copy import deepcopy
from unittest.mock import Mock

import httpx
import pytest

from jev_ultrafast import model
from jev_ultrafast import noul_policy as policy
from scripts.probe_noul_policy import build_request as probe_request

IDENTITY = {"model": "trained-system-one-test", "temperature": 2.5}
URL = "http://127.0.0.1:18775/decide"


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    for name in ("S1_DECISION_URL", "S1_EXPECTED_MODEL", "S1_EXPECTED_TEMPERATURE", "S1_LOCAL_POLICY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(model, "CLIENT", Mock(post=Mock(side_effect=AssertionError("No remote provider"))))


def page(clicks=2, fill=True, select=False, controls=()):
    actions = []
    if fill:
        actions.append({"id": "field", "kind": "fill", "node": 1, "label": "City", "role": "textbox", "value": ""})
    actions.extend({"id": f"button{i}", "kind": "click", "node": i + 2,
                    "label": f"Button {i}", "role": "button", "checked": False} for i in range(clicks))
    if select:
        actions.extend({"id": f"option{i}", "kind": "select", "node": 100, "label": f"Style → Option {i}",
                        "role": "combobox", "current_value": "Option 0", "value": f"v{i}"} for i in range(2))
    actions.extend({"id": name.lower(), "kind": name.lower(), "label": description} for name, description in controls)
    return {"url": "http://127.0.0.1:8768/", "title": "Local page", "text": "Visible current page state",
            "actions": actions, "unused_external_checker": "must stay outside the request"}


def noul(value):
    return {"id": "candidate", "type": "noul", "noul": value, "confidence": max(value, 1 - value),
            "probabilities": {"false": 1 - value, "true": value}}


def choice(labels, selected):
    labels = list(labels)
    probabilities = {label: 0.2 / (len(labels) - 1) for label in labels}
    probabilities[selected] = 0.8
    return {"type": "choice", "choice": selected, "confidence": 0.8, "probabilities": probabilities}


def response(body, score=0.2, target=None, **overrides):
    head, question = next(iter(body["questions"].items()))
    answer = noul(score) if question["type"] == "noul" else choice(question["criteria"], target)
    return httpx.Response(200, json={"status": "ok", **IDENTITY, "answers": {head: answer},
                                    "latency_ms": 3.25, "native_latency_ms": 2.5,
                                    "native_token_count": 42, "preprocessing": {"latency_ms": 0.75}, **overrides})


def operation_of(body):
    question = body["state"]["question"]
    return next(name for name in (
        "CLICK", "TYPE_TEXT", "SELECT", "WAIT", "BACK", "SCROLL_UP", "SCROLL_DOWN", "DONE", "BLOCKED"
    )
                if f"is {name} the appropriate" in question)


@pytest.fixture
def local_transport(monkeypatch):
    original = httpx.AsyncClient
    installed = []

    def install(handler):
        settings = []

        def factory(**kwargs):
            settings.append(kwargs)
            client = original(transport=httpx.MockTransport(handler), **kwargs)
            installed.append(client)
            return client

        monkeypatch.setattr(policy.httpx, "AsyncClient", factory)
        return settings

    yield install
    assert all(client.is_closed for client in installed)


def choose(state, **kwargs):
    return policy.choose_noul(state, "Render the requested view.", [], decision_url=URL,
                              expected_identity=IDENTITY, **kwargs)


def test_source_preserves_baseline_indexing_context_and_exact_probe_format():
    state = page(select=True, controls=(("WAIT", "Wait for an observed pending update."),))
    history = [{"action": f"past{i}", "kind": "click", "text": None, "page_changed": bool(i % 2),
                "gold": "private checker must stay out"} for i in range(12)]
    before = deepcopy((state, history))
    source, targets, controls = policy.prepare_source(state, "Find Zürich and render its requested view.", history)
    assert source["state"]["elements"] == model.action_space(state["actions"])[0]
    assert len(source["state"]["recent_actions"]) == 10
    assert source["state"]["recent_actions"][0]["action"] == "past2"
    assert "gold" not in json.dumps(source) and "checker" not in json.dumps(source)
    assert set(targets) == {"CLICK", "TYPE_TEXT", "SELECT"} and set(controls) == {"WAIT"}
    for operation in source["questions"]["operation"]["criteria"]:
        request = policy.build_request(source, operation)
        assert request == probe_request(source, operation)
        assert request["questions"] == {"candidate": {"type": "noul", "instructions": policy.INSTRUCTIONS}}
        assert "Find Zürich" in request["state"]["passage"]
        original_instructions = source["questions"]["operation"]["instructions"]
        assert original_instructions in request["state"]["passage"]
        assert json.loads(original_instructions)["rules"] == model.NEXT_ACTION
        assert "All legal operation descriptions in original order" in request["state"]["passage"]
    assert (state, history) == before


def test_independent_scores_then_only_selected_target_retains_every_real_mock_call(local_transport):
    requests = []
    scores = {"TYPE_TEXT": 0.7, "CLICK": 0.95, "SELECT": 0.8, "WAIT": 0.6, "DONE": 0.1, "BLOCKED": 0.05}

    def handler(request):
        requests.append(request)
        assert str(request.url) == URL and request.headers.get("authorization") is None
        body = json.loads(request.content)
        assert set(body) == {"state", "questions"} and len(body["questions"]) == 1
        if next(iter(body["questions"].values()))["type"] == "noul":
            return response(body, scores[operation_of(body)])
        assert set(body["questions"]) == {"click_target"}
        assert set(body["questions"]["click_target"]["criteria"]) == {"2", "3"}
        return response(body, target="3")

    client_settings = local_transport(handler)
    result = choose(page(select=True, controls=(("WAIT", "Wait for an observed pending update."),)))
    assert len(requests) == 7
    assert result["choice"] == "button1" and result["target"] == "3"
    assert result["operation_scores"] == scores and sum(result["operation_scores"].values()) > 1
    assert result["operation_probabilities"] is None and result["categorical_operation_probabilities"] is None
    assert result["confidence"] is None and result["target_confidence"] == 0.8
    assert result["target_probabilities"] == {"2": 0.2, "3": 0.8}
    assert result["probabilities"] == {"button0": 0.2, "button1": 0.8}
    assert result["provider"] == "local_system_one_noul" and result["policy_id"] == "noul_v1"
    assert result["model_latency_ms"] == 7 * 3.25 and result["native_token_count"] == 7 * 42
    assert result["native_latency_ms"] == 7 * 2.5
    assert result["native_calls"][-1]["phase"] == "target"
    for call, request in zip(result["native_calls"], requests, strict=True):
        assert call["request"] == json.loads(request.content)
        assert call["request_sha256"] == hashlib.sha256(request.content).hexdigest()
        assert call["raw_response"]["answers"][call["head"]]["probabilities"] == call["actual_distribution"]
        assert call["client_wall_ms"] >= 0 and call["model_execution_confirmed"]
        assert call["transport_attempted"] and call["ok"]
    assert client_settings == [{"http2": True, "trust_env": False, "follow_redirects": False}]


def test_single_legal_target_has_no_fabricated_native_target_head(local_transport):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        op = operation_of(requests[-1])
        return response(requests[-1], score=0.9 if op == "TYPE_TEXT" else 0.1)

    local_transport(handler)
    result = choose(page(clicks=0))
    assert len(requests) == 3 and all(set(body["questions"]) == {"candidate"} for body in requests)
    assert result["choice"] == "field" and result["target_source"] == "single_legal_target"
    assert result["target_confidence"] is None and result["target_probabilities"] == {}
    assert result["probabilities"] == {"field": 1.0}
    assert result["probabilities_source"] == "single_legal_target"
    assert "type_text_target" not in result["raw_answers"]


def test_all_eight_legal_operations_scored_then_one_target_with_max_nine_calls(local_transport):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if "candidate" in body["questions"]:
            return response(body, score=0.95 if operation_of(body) == "CLICK" else 0.2)
        return response(body, target="3")

    local_transport(handler)
    controls = (("SCROLL_UP", "Scroll up"), ("SCROLL_DOWN", "Scroll down"), ("WAIT", "Wait for an update"))
    result = choose(page(select=True, controls=controls), max_native_calls=9)
    assert len(requests) == 9 and len(result["operation_scores"]) == 8
    assert result["settings"]["maximum_native_calls"] == 9
    assert sum(call["phase"] == "operation" for call in result["native_calls"]) == 8
    assert result["native_calls"][-1]["phase"] == "target" and result["choice"] == "button1"


@pytest.mark.parametrize("winner,expected", [("WAIT", "wait"), ("DONE", "DONE"), ("BLOCKED", "BLOCKED")])
def test_control_operation_has_no_fabricated_categorical_probability(local_transport, winner, expected):
    local_transport(lambda request: response(json.loads(request.content),
                                             score=0.9 if operation_of(json.loads(request.content)) == winner else 0.1))
    result = choose(page(clicks=0, fill=False, controls=(("WAIT", "Wait for pending update."),)))
    assert result["choice"] == expected and result["probabilities"] == {expected: None}
    assert result["target"] is None and result["confidence"] is None
    assert len(result["native_calls"]) == 3


def test_ties_follow_declared_operation_order_without_reordering(local_transport):
    local_transport(lambda request: response(json.loads(request.content), score=0.5))
    result = choose(page(clicks=1, fill=False))
    assert result["operation"] == "CLICK"
    assert result["settings"]["operation_order"] == ["CLICK", "DONE", "BLOCKED"]
    assert result["operation_scores"] == {"CLICK": 0.5, "DONE": 0.5, "BLOCKED": 0.5}


@pytest.mark.parametrize("url", [None, "", "http://localhost:18775/decide", "http://0.0.0.0:18775/decide",
                                 "http://[::ffff:127.0.0.1]:18775/decide", "http://127.0.0.1:18775/other",
                                 "http://user:password@127.0.0.1:18775/decide",
                                 "http://127.0.0.1:18775/decide?external=yes", "https://api.typesafe.ai/v1/systemone"])
def test_missing_or_nonliteral_loopback_endpoint_fails_without_transport(monkeypatch, url):
    monkeypatch.setattr(policy.httpx, "AsyncClient", lambda **_: pytest.fail("Must not construct network client"))
    with pytest.raises(policy.NoulPolicyError) as error:
        policy.choose_noul(page(), "A generic goal", [], decision_url=url, expected_identity=IDENTITY)
    assert error.value.native_calls == []


@pytest.mark.parametrize("identity", [{}, {"model": "test", "temperature": True},
                                      {"model": "test", "temperature": float("nan")},
                                      {"model": "", "temperature": 1}, {"model": "test", "temperature": 0}])
def test_identity_required_before_first_request(monkeypatch, identity):
    monkeypatch.setattr(policy.httpx, "AsyncClient", lambda **_: pytest.fail("No model client without identity"))
    with pytest.raises(policy.NoulPolicyError) as error:
        policy.choose_noul(page(), "Generic goal", [], decision_url=URL, expected_identity=identity)
    assert error.value.native_calls == []


def test_explicit_environment_optin_without_secret_headers(monkeypatch, local_transport):
    monkeypatch.setenv("S1_DECISION_URL", URL)
    monkeypatch.setenv("S1_EXPECTED_MODEL", IDENTITY["model"])
    monkeypatch.setenv("S1_EXPECTED_TEMPERATURE", str(IDENTITY["temperature"]))
    monkeypatch.setenv("HTTP_PROXY", "http://outside.invalid:1234")
    monkeypatch.setenv("HTTPS_PROXY", "http://outside.invalid:1234")
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-unused-secret")

    def handler(request):
        assert str(request.url) == URL and "authorization" not in request.headers
        return response(json.loads(request.content), score=0.2)

    client_settings = local_transport(handler)
    result = policy.choose_noul(page(clicks=0, fill=False), "Generic goal", [])
    assert result["expected_native_identity"] == IDENTITY
    assert client_settings[0]["trust_env"] is False


@pytest.mark.parametrize("change", ["nan", "bool", "missing", "sum", "confidence", "noul", "head", "model", "temp"])
def test_bad_response_fails_closed_and_retains_first_attempt(local_transport, change):
    requests = []

    def handler(request):
        requests.append(request)
        raw = json.loads(response(json.loads(request.content), 0.8).content)
        answer = raw["answers"]["candidate"]
        if change == "nan":
            # JSONResponse refuses NaN; deliberately preserve the malformed provider bytes.
            return httpx.Response(200, content=json.dumps({**raw, "answers": {"candidate": {
                **answer, "noul": float("nan")}}}).encode())
        if change == "bool":
            answer["probabilities"]["true"] = True
        elif change == "missing":
            answer.pop("probabilities")
        elif change == "sum":
            answer["probabilities"]["true"] = 0.7
        elif change == "confidence":
            answer["confidence"] = 0.99
        elif change == "noul":
            answer["noul"] = 0.7
        elif change == "head":
            raw["answers"] = {"wrong": answer}
        elif change == "model":
            raw["model"] = "wrong model"
        elif change == "temp":
            raw["temperature"] = 1.0
        return httpx.Response(200, json=raw)

    local_transport(handler)
    with pytest.raises(policy.NoulPolicyError) as error:
        choose(page())
    assert len(requests) == 1
    calls = error.value.native_calls
    assert len(calls) == 1 and calls[0]["transport_attempted"] and not calls[0]["ok"]
    assert calls[0]["raw_response"] is not None and calls[0]["error"]
    assert calls[0]["actual_distribution"] is None


@pytest.mark.parametrize("status", [301, 307, 308, 429, 503, 529])
def test_redirect_errors_no_retry_or_fallback(local_transport, status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, headers={"location": "https://api.typesafe.ai/v1/systemone"})

    local_transport(handler)
    with pytest.raises(policy.NoulPolicyError) as error:
        choose(page())
    assert len(requests) == 1 and len(error.value.native_calls) == 1
    assert error.value.native_calls[0]["http_status"] == status


def test_selected_target_identity_failure_retains_operation_prefix_without_fallback(local_transport):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if "candidate" in body["questions"]:
            return response(body, score=0.9 if operation_of(body) == "CLICK" else 0.1)
        return response(body, target="3", model="wrong target model")

    local_transport(handler)
    with pytest.raises(policy.NoulPolicyError) as error:
        choose(page())
    assert len(requests) == 5
    assert [call["ok"] for call in error.value.native_calls] == [True, True, True, True, False]
    assert error.value.native_calls[-1]["phase"] == "target"


@pytest.mark.parametrize("change", ["nan", "bool", "sum", "missing", "unknown_choice", "id", "type"])
def test_malformed_selected_target_never_selects_an_alternative(local_transport, change):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if "candidate" in body["questions"]:
            return response(body, score=0.9 if operation_of(body) == "CLICK" else 0.1)
        raw = json.loads(response(body, target="3").content)
        answer = raw["answers"]["click_target"]
        if change == "nan":
            answer["probabilities"]["3"] = float("nan")
        elif change == "bool":
            answer["confidence"] = True
        elif change == "sum":
            answer["probabilities"]["2"] = 0.21
        elif change == "missing":
            answer.pop("probabilities")
        elif change == "unknown_choice":
            answer["choice"] = "unobserved"
        elif change == "id":
            answer["id"] = "type_text_target"
        elif change == "type":
            answer["type"] = "noul"
        return httpx.Response(200, content=json.dumps(raw).encode())

    local_transport(handler)
    with pytest.raises(policy.NoulPolicyError) as error:
        choose(page())
    assert len(requests) == 5
    assert error.value.native_calls[-1]["phase"] == "target"
    assert not error.value.native_calls[-1]["ok"]


@pytest.mark.parametrize("budget,attempts", [(0, 0), (1, 1), (3, 3), (4, 4)])
def test_global_native_budget_checks_each_post_without_reserving_unused_calls(local_transport, budget, attempts):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if "candidate" in body["questions"]:
            return response(body, score=0.9 if operation_of(body) == "CLICK" else 0.1)
        return response(body, target="3")

    local_transport(handler)
    with pytest.raises(policy.NativeCallBudgetError) as error:
        choose(page(), max_native_calls=budget)
    assert error.value.budget_exhausted is True
    assert len(requests) == attempts
    assert sum(call["transport_attempted"] for call in error.value.native_calls) == attempts
    assert error.value.native_calls[-1]["transport_attempted"] is False


def test_budget_equal_to_actual_calls_succeeds_without_full_cap_reservation(local_transport):
    local_transport(lambda request: response(json.loads(request.content), score=0.2))
    result = choose(page(clicks=0, fill=False), max_native_calls=2)
    assert len(result["native_calls"]) == 2 and result["settings"]["effective_native_request_budget"] == 2


def test_one_shared_deadline_cancels_request_and_blocks_later_calls(local_transport):
    requests = []

    async def handler(request):
        requests.append(request)
        if len(requests) == 1:
            await asyncio.sleep(0.002)
            return response(json.loads(request.content), score=0.7)
        await asyncio.Event().wait()
        return response(json.loads(request.content), score=0.5)

    local_transport(handler)
    with pytest.raises(policy.NoulPolicyError) as error:
        choose(page(), timeout=0.03)
    calls = error.value.native_calls
    assert len(requests) == 2 and len(calls) == 2
    assert calls[0]["ok"] and not calls[1]["ok"]
    assert calls[1]["error"]["type"] == "CancelledError"
    timeouts = [request.extensions["timeout"]["read"] for request in requests]
    assert 0 < timeouts[1] < timeouts[0] <= 0.03


def test_missing_latency_or_tokens_are_not_invented(local_transport):
    local_transport(lambda request: response(json.loads(request.content), score=0.2,
                                             latency_ms=None, native_token_count=None))
    result = choose(page(clicks=0, fill=False))
    assert result["model_latency_ms"] is None and result["native_token_count"] is None
    assert all(call["model_latency_ms"] is None for call in result["native_calls"])


def test_too_many_offered_operations_or_targets_fail_before_transport(monkeypatch):
    monkeypatch.setattr(policy.httpx, "AsyncClient", lambda **_: pytest.fail("No request after bound failure"))
    controls = tuple((f"CUSTOM{i}", f"Observed control {i}") for i in range(7))
    with pytest.raises(policy.NoulPolicyError, match="operation count"):
        choose(page(clicks=0, fill=False, controls=controls))
    with pytest.raises(policy.NoulPolicyError, match="at most 52"):
        choose(page(clicks=53, fill=False))


def test_sync_in_existing_event_loop_fails_closed_without_thread_fallback():
    async def run():
        with pytest.raises(policy.NoulPolicyError, match="choose_noul_async") as error:
            choose(page())
        assert error.value.native_calls == []
    asyncio.run(run())


def test_unoffered_operation_and_nonfinite_input_never_enter_payload():
    source, _, _ = policy.prepare_source(page(), "A generic goal", [])
    with pytest.raises(ValueError, match="actually offered"):
        policy.build_request(source, "UNSUPPORTED")
    assert math.isfinite(IDENTITY["temperature"])


def test_private_trace_sink_is_the_same_complete_telemetry_without_changing_policy(local_transport):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        return response(body, score=0.8 if operation_of(body) == "DONE" else 0.2)

    local_transport(handler)
    state = page(clicks=0, fill=False)
    original = choose(state)
    first_requests = deepcopy(requests)
    requests.clear()
    sink = []
    traced = choose(state, _trace_calls=sink)
    assert traced["native_calls"] is sink and len(sink) == 2
    assert requests == first_requests
    for key in ("choice", "operation", "target", "operation_scores", "source_request", "request", "settings",
                "confidence", "categorical_operation_probabilities", "expected_native_identity"):
        assert traced[key] == original[key]
    assert all(call["transport_attempted"] and call["model_execution_confirmed"] for call in sink)


@pytest.mark.parametrize("sink", [[{"previous": "immutable"}], {}, (), "not a list"])
def test_private_trace_sink_rejects_existing_or_wrong_storage_before_transport(monkeypatch, sink):
    before = deepcopy(sink)
    monkeypatch.setattr(policy.httpx, "AsyncClient", lambda **_: pytest.fail("Invalid sink must not transport"))
    with pytest.raises(policy.NoulPolicyError, match="initially empty") as error:
        choose(page(), _trace_calls=sink)
    assert error.value.native_calls == [] and sink == before


def test_async_task_cancellation_preserves_attempted_prefix_in_private_sink(local_transport):
    pending, requests, sink = asyncio.Event(), [], []

    async def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return response(json.loads(request.content), score=0.2)
        pending.set()
        await asyncio.Event().wait()

    local_transport(handler)

    async def run():
        task = asyncio.create_task(policy.choose_noul_async(
            page(clicks=0, fill=False), "Generic original goal", [], decision_url=URL,
            expected_identity=IDENTITY, _trace_calls=sink,
        ))
        await pending.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert len(requests) == len(sink) == 2
    assert [call["transport_attempted"] for call in sink] == [True, True]
    assert [call["model_execution_confirmed"] for call in sink] == [True, False]
    assert sink[-1]["error"]["type"] == "CancelledError" and sink[-1]["client_wall_ms"] >= 0
