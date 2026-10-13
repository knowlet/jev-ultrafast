"""Opt-in local Noul operation ranking, followed by only the selected Choice target.

Independent binary scores are never normalized into operation probabilities. The
existing BoolQ temperature does not establish navigation calibration. No browser
checks, external labels, retries, remote providers or model startup enter this policy.
"""

import asyncio
import hashlib
import ipaddress
import json
import math
import os
import time
from copy import deepcopy
from urllib.parse import urlsplit

import httpx

from . import model
from .questions import NEXT_ACTION, TARGET

POLICY_ID = "noul_v1"
HEAD = "candidate"
INSTRUCTIONS = "Answer the question using the supplied passage."
MAX_OPERATIONS = 8
MAX_TIMEOUT = 90
CALIBRATION_SCOPE = "Existing BoolQ Noul temperature; navigation calibration not established"


class NoulPolicyError(RuntimeError):
    """Fail closed while retaining every completed/attempted native request."""

    def __init__(self, message, native_calls):
        super().__init__(message)
        self.native_calls = deepcopy(native_calls)
        self.policy_id = POLICY_ID


class NativeCallBudgetError(NoulPolicyError):
    budget_exhausted = True


def endpoint(value):
    parsed = urlsplit(value or "")
    try:
        address, port = ipaddress.ip_address(parsed.hostname or ""), parsed.port
    except ValueError:
        address, port = None, None
    if (
        parsed.scheme != "http" or not address or not address.is_loopback
        or getattr(address, "ipv4_mapped", None) is not None or port is None or parsed.path != "/decide"
        or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment
    ):
        raise ValueError("Noul policy requires an explicit HTTP numeric-loopback /decide endpoint")
    return value


def build_request(native_choice_request, operation):
    """The exact grounded protocol proven by scripts/probe_noul_policy.build_request."""
    original = deepcopy(native_choice_request)
    if set(original) != {"state", "questions"} or not isinstance(original["state"], dict):
        raise ValueError("Expected the unchanged native Jev choice request with a full observed state")
    question = original["questions"]["operation"]
    if question.get("type") != "choice" or not isinstance(question.get("criteria"), dict):
        raise ValueError("The original operation head must offer named legal choices")
    criteria = question["criteria"]
    if operation not in criteria or not all(isinstance(value, str) for value in criteria.values()):
        raise ValueError("A candidate must be an actually offered operation with its original description")
    instructions = question["instructions"]
    if not isinstance(instructions, str):
        raise ValueError("Expected the baseline's recorded operation instruction string")
    context = json.loads(instructions)
    if not isinstance(context, dict) or not isinstance(context.get("goal"), str) or not context["goal"].strip():
        raise ValueError("The recorded operation instructions must contain the exact original goal")
    if not isinstance(context.get("rules"), (str, list)):
        raise ValueError("Original policy rules must be retained")
    passage = (
        "Original goal:\n" + context["goal"]
        + "\n\nFull observed state (including indexed elements and recent actions):\n"
        + json.dumps(original["state"], ensure_ascii=False)
        + "\n\nOriginal operation instructions and policy rules:\n" + instructions
        + "\n\nAll legal operation descriptions in original order:\n"
        + json.dumps(criteria, ensure_ascii=False)
    )
    claim = (
        "Considering the original goal, policy rules and current observations in the passage, "
        f"is {operation} the appropriate next operation NOW to progress toward or finish the goal? "
        f"Operation description: {criteria[operation]}"
    )
    return {
        "state": {"passage": passage, "question": claim},
        "questions": {HEAD: {"type": "noul", "instructions": INSTRUCTIONS}},
    }


def validate_noul(answer):
    try:
        probabilities = answer["probabilities"]
        values = [*probabilities.values(), answer["noul"], answer["confidence"]]
        valid = (
            set(probabilities) == {"false", "true"}
            and all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in values)
            and abs(sum(probabilities.values()) - 1) <= 1e-6
            and abs(answer["noul"] - probabilities["true"]) <= 1e-6
            and abs(answer["confidence"] - max(probabilities.values())) <= 1e-6
            and answer.get("id", HEAD) == HEAD and answer.get("type", "noul") == "noul"
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid complete native Noul distribution; no operation inferred")
    return answer


def prepare_source(state, goal, history):
    """Reuse the baseline's legal action indexing, state projection and generic rules."""
    elements, targets, controls = model.action_space(state["actions"])
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    if set(controls) & (set(operations) | {"DONE", "BLOCKED"}):
        raise ValueError("Observed control IDs conflict with operation semantics")
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    questions = {
        "operation": {"type": "choice", "criteria": operations, "instructions": {"goal": goal, "rules": NEXT_ACTION}}
    }
    for operation, candidates in targets.items():
        questions[operation.lower() + "_target"] = {
            "type": "choice", "criteria": {
                index: {"element": f"[{index}] {action['label']}",
                        "current_value": action.get("current_value", action.get("value", "")),
                        **{key: action[key] for key in ("role", "checked", "selected", "expanded") if key in action}}
                for index, action in candidates.items()
            },
            "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
        }
    body = {
        "state": {"page": {key: state[key] for key in ("url", "title", "text")}, "elements": elements,
                  "recent_actions": [{key: item.get(key) for key in ("action", "kind", "text", "page_changed")}
                                     for item in history[-10:]]},
        "questions": model.local_questions(questions),
    }
    return body, targets, controls


def identity_settings(expected):
    if expected is None:
        name, temperature = os.environ.get("S1_EXPECTED_MODEL"), os.environ.get("S1_EXPECTED_TEMPERATURE")
        if not name or temperature is None:
            raise ValueError("Noul policy requires explicit S1_EXPECTED_MODEL and S1_EXPECTED_TEMPERATURE")
        expected = {"model": name, "temperature": float(temperature)}
    if (set(expected) != {"model", "temperature"} or not isinstance(expected["model"], str)
            or not expected["model"].strip() or type(expected["temperature"]) not in (int, float)
            or not math.isfinite(expected["temperature"]) or expected["temperature"] <= 0):
        raise ValueError("Expected native model identity requires a named model and positive finite saved temperature")
    return deepcopy(expected)


def remaining(deadline):
    seconds = deadline - time.perf_counter()
    if seconds <= 0:
        raise TimeoutError("Noul policy exceeded its shared decision deadline; no action selected")
    return seconds


async def request_once(client, url, body, operation, phase, head, identity, deadline, calls, budget):
    payload = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    call = {
        "kind": "decision", "phase": phase, "operation": operation, "head": head,
        "request": deepcopy(body), "request_sha256": hashlib.sha256(payload).hexdigest(),
        "raw_response": None, "actual_distribution": None, "ok": False,
        "transport_attempted": False, "model_execution_confirmed": False,
    }
    calls.append(call)
    started = time.perf_counter()
    try:
        if sum(item["transport_attempted"] for item in calls) >= budget:
            call["error"] = {"type": "NativeCallBudgetError", "message": "Actual native request budget exhausted"}
            raise NativeCallBudgetError("Actual native request budget exhausted; no action selected", calls)
        timeout = remaining(deadline)
        call["transport_attempted"] = True
        response = await client.post(
            url, content=payload, headers={"Content-Type": "application/json"},
            timeout=timeout, follow_redirects=False,
        )
        call["http_status"] = response.status_code
        model.response_error(response)
        raw = response.json()
        call["raw_response"] = deepcopy(raw)
        if raw.get("status") != "ok" or any(raw.get(key) != value for key, value in identity.items()):
            raise ValueError("Native model or saved calibration differs from the explicit identity")
        call["model_execution_confirmed"] = True
        if set(raw["answers"]) != {head}:
            raise ValueError("Native response contains unexpected or missing question heads")
        answer = raw["answers"][head]
        if phase == "operation":
            validate_noul(answer)
        else:
            model.validate_choice(answer, body["questions"][head]["criteria"])
            values = list(answer["probabilities"].values())
            if (abs(sum(values) - 1) > 1e-6
                    or abs(answer["confidence"] - max(values)) > 1e-6
                    or answer.get("id", head) != head or answer.get("type", "choice") != "choice"):
                raise ValueError("Invalid complete native target distribution")
        call.update(actual_distribution=deepcopy(answer["probabilities"]),
                    actual_answer=deepcopy(answer), model_latency_ms=raw.get("latency_ms"),
                    native_latency_ms=raw.get("native_latency_ms"), native_token_count=raw.get("native_token_count"),
                    preprocessing=deepcopy(raw.get("preprocessing")))
        remaining(deadline)
        call["ok"] = True
        return answer
    except BaseException as error:
        call["error"] = {"type": type(error).__name__, "message": str(error)[:1000]}
        raise
    finally:
        call["client_wall_ms"] = (time.perf_counter() - started) * 1000


def sum_real(calls, field):
    values = [call.get(field) for call in calls]
    if all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values):
        return sum(values)
    return None


async def choose_noul_async(state, goal, history, *, decision_url=None, expected_identity=None,
                            max_operations=MAX_OPERATIONS, timeout=MAX_TIMEOUT, max_native_calls=None):
    started, calls = time.perf_counter(), []
    try:
        if (type(max_operations) is not int or not 1 <= max_operations <= MAX_OPERATIONS
                or type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= MAX_TIMEOUT):
            raise ValueError("Noul policy bound: at most8 offered operations and one shared <=90-second deadline")
        if max_native_calls is not None and (type(max_native_calls) is not int or max_native_calls < 0):
            raise ValueError("Actual native request budget must be a nonnegative integer")
        url = endpoint(decision_url if decision_url is not None else os.environ.get("S1_DECISION_URL"))
        identity = identity_settings(expected_identity)
        source, targets, controls = prepare_source(state, goal, history)
        operations = source["questions"]["operation"]["criteria"]
        if len(operations) > max_operations:
            raise ValueError("Offered operation count exceeds the explicit Noul policy budget; nothing truncated")
        budget = len(operations) + 1 if max_native_calls is None else min(max_native_calls, len(operations) + 1)
        deadline = started + timeout
        operation_answers, scores, target, target_answer, target_source = {}, {}, None, None, None
        async with asyncio.timeout(remaining(deadline)):
            async with httpx.AsyncClient(http2=True, trust_env=False, follow_redirects=False) as client:
                for operation in operations:
                    answer = await request_once(client, url, build_request(source, operation), operation,
                                                "operation", HEAD, identity, deadline, calls, budget)
                    operation_answers[operation] = deepcopy(answer)
                    scores[operation] = answer["noul"]
                operation = max(scores, key=scores.__getitem__)
                probabilities, probability_source = {}, "no_categorical_operation_probability"
                if operation in targets:
                    candidates = targets[operation]
                    if len(candidates) == 1:
                        target = next(iter(candidates))
                        target_source, probability_source = "single_legal_target", "single_legal_target"
                        choice = candidates[target]["id"]
                        probabilities = {choice: 1.0}
                    else:
                        head = operation.lower() + "_target"
                        body = {"state": deepcopy(source["state"]),
                                "questions": {head: deepcopy(source["questions"][head])}}
                        target_answer = await request_once(client, url, body, operation, "target", head,
                                                           identity, deadline, calls, budget)
                        target, target_source = target_answer["choice"], "model"
                        choice = candidates[target]["id"]
                        probabilities = {action["id"]: target_answer["probabilities"][index]
                                         for index, action in candidates.items()}
                        probability_source = "conditional_selected_operation_target"
                else:
                    choice = controls[operation]["id"] if operation in controls else operation
                    probabilities = {choice: None}
                remaining(deadline)
        raw_answers = {"operation_noul": operation_answers}
        if target_answer is not None:
            raw_answers[operation.lower() + "_target"] = deepcopy(target_answer)
        return {
            "choice": choice, "operation": operation, "target": target, "target_source": target_source,
            "confidence": None, "confidence_scope": "No categorical operation confidence available",
            "probabilities": probabilities, "probabilities_source": probability_source,
            "operation_probabilities": None, "categorical_operation_probabilities": None,
            "operation_scores": scores, "operation_score_kind": "independent_native_noul_p_true",
            "target_probabilities": target_answer["probabilities"] if target_answer else {},
            "target_confidence": target_answer["confidence"] if target_answer else None,
            "raw_answers": raw_answers, "raw_response": [call["raw_response"] for call in calls],
            "native_calls": calls, "model": identity["model"], "temperature": identity["temperature"], "usage": {},
            "model_latency_ms": sum_real(calls, "model_latency_ms"),
            "native_latency_ms": sum_real(calls, "native_latency_ms"),
            "native_token_count": sum_real(calls, "native_token_count"),
            "preprocessing": [call["preprocessing"] for call in calls],
            "latency_ms": (time.perf_counter() - started) * 1000,
            "source_request": source,
            "request": {"policy_id": POLICY_ID, "requests": [call["request"] for call in calls]},
            "provider": "local_system_one_noul", "endpoint": url, "policy_id": POLICY_ID,
            "settings": {"max_operations": max_operations, "maximum_native_calls": len(operations) + 1,
                         "effective_native_request_budget": budget,
                         "operation_order": list(operations), "decision_timeout_seconds": timeout,
                         "retry_count": 0, "tie_policy": "first offered operation"},
            "expected_native_identity": identity, "calibration_scope": CALIBRATION_SCOPE,
        }
    except NativeCallBudgetError as error:
        raise NativeCallBudgetError(str(error), calls) from error
    except Exception as error:
        raise NoulPolicyError(str(error), calls) from error


def choose_noul(state, goal, history, **settings):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(choose_noul_async(state, goal, history, **settings))
    raise NoulPolicyError("Sync Noul policy needs a non-async caller; use choose_noul_async in an existing loop", [])


choose = choose_noul
