"""TypeSafe makes choices; an optional small OpenAI-compatible model writes field values."""

import ipaddress
import json
import math
import os
import time

import httpx

from .questions import NEXT_ACTION, TARGET, TEXT_VALUE

CLIENT = httpx.Client(http2=True, timeout=25)
# Local model requests never inherit HTTP(S)_PROXY or follow provider redirects.
LOCAL_CLIENT = httpx.Client(http2=True, timeout=90, trust_env=False, follow_redirects=False)


def local_url(value, setting):
    """Use numeric loopback addresses so hostname resolution cannot change the destination."""
    try:
        url = httpx.URL(value)
        address = ipaddress.ip_address(url.host)
        valid = (
            url.scheme in {"http", "https"}
            and address.is_loopback
            and getattr(address, "ipv4_mapped", None) is None
            and not url.userinfo
            and not url.query
            and not url.fragment
        )
    except (ValueError, httpx.InvalidURL):
        valid = False
    if not valid:
        raise ValueError(f"{setting} must be an HTTP(S) numeric loopback URL without credentials, query or fragment.")
    return str(url)


def response_error(response):
    if response.is_redirect:
        raise RuntimeError("Model provider redirected; no action executed.")
    if response.is_error:
        raise RuntimeError(f"Model provider returned HTTP {response.status_code}; no action executed.")


def post_json(url, key, body):
    if key is None:
        url = local_url(url, "Local model endpoint")
    client = LOCAL_CLIENT if key is None else CLIENT
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    attempts = 1 if key is None else 3
    for attempt in range(attempts):
        try:
            response = client.post(url, json=body, headers=headers, follow_redirects=False)
        except httpx.HTTPError:
            raise RuntimeError("Model connection failed; no action executed.") from None
        if response.status_code in {429, 529, 503} and attempt < attempts - 1:
            time.sleep(0.5 * 2**attempt)
            continue
        response_error(response)
        return response.json()
    raise RuntimeError("Model unavailable")


def local_questions(questions):
    """Adapt only the native System One request shape; never manufacture model probabilities."""
    adapted = {}
    for name, question in questions.items():
        count = len(question["criteria"])
        if count > 52:
            raise ValueError("Local System One supports at most 52 choices per head; no action executed.")
        if count == 1 and name != "operation":
            # The executor can resolve a sole legal observed target without asking a model.
            continue
        adapted[name] = {
            "type": "choice",
            "instructions": json.dumps(question["instructions"], ensure_ascii=False),
            "criteria": {
                key: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                for key, value in question["criteria"].items()
            },
        }
    return adapted


def stream_local_text(url, body):
    """Measure time to the first content delta, retaining only provider-reported token usage."""
    url = local_url(url, "Local text endpoint")
    started = time.perf_counter()
    ttft_ms, pieces, usage, timings, completed = None, [], {}, {}, False
    try:
        with LOCAL_CLIENT.stream("POST", url, json=body, follow_redirects=False) as response:
            response_error(response)
            if "text/event-stream" not in response.headers.get("content-type", ""):
                raise ValueError("Local text helper must support OpenAI-compatible SSE streaming; nothing typed.")
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    completed = True
                    break
                if not data:
                    continue
                event = json.loads(data)
                if not isinstance(event, dict):
                    raise ValueError("Local text helper returned an invalid stream event; nothing typed.")
                if "error" in event:
                    raise ValueError("Local text helper returned a stream error; nothing typed.")
                if event.get("usage"):
                    usage = event["usage"]
                if event.get("timings"):
                    timings = event["timings"]
                for choice in event.get("choices", []):
                    content = choice.get("delta", {}).get("content")
                    if content:
                        if not isinstance(content, str) or choice.get("index", 0) != 0:
                            raise ValueError("Local text helper returned an invalid content delta; nothing typed.")
                        if ttft_ms is None:
                            ttft_ms = round((time.perf_counter() - started) * 1000, 3)
                        pieces.append(content)
                        if sum(map(len, pieces)) > 10000:
                            raise ValueError("Local text helper returned too much text; nothing typed.")
    except httpx.HTTPError:
        raise RuntimeError("Model connection failed; no action executed.") from None
    if not completed:
        raise ValueError("Local text helper stream ended before [DONE]; nothing typed.")
    return {
        "choices": [{"message": {"content": "".join(pieces)}}], "usage": usage, "timings": timings,
    }, ttft_ms


def validate_choice(answer, ids):
    try:
        probabilities = answer["probabilities"]
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid TypeSafe response; no action executed.")
    return answer


def action_space(actions):
    """One index per observed element; each operation has its own valid target choices."""
    elements, indices, targets, controls = [], {}, {}, {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
    for action in actions:
        kind = action["kind"]
        if kind not in operations:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in ("role", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = operations[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action
    return elements, targets, controls


def choose_noul(state, goal, history, **settings):
    """Explicit local experimental policy; binary scores retain their native meaning."""
    from .noul_policy import choose_noul as local_choose

    return local_choose(state, goal, history, **settings)


def choose(state, goal, history):
    policy = os.environ.get("S1_LOCAL_POLICY", "baseline")
    if policy == "noul_v1":
        return choose_noul(state, goal, history)
    if policy != "baseline":
        raise ValueError("Unsupported S1_LOCAL_POLICY; no model request or action executed.")
    local_endpoint = os.environ.get("S1_DECISION_URL")
    local_mode = local_endpoint is not None
    endpoint = (
        local_url(local_endpoint, "S1_DECISION_URL") if local_mode else "https://api.typesafe.ai/v1/systemone"
    )
    elements, targets, controls = action_space(state["actions"])
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    questions = {
        "operation": {"type": "choice", "criteria": operations, "instructions": {"goal": goal, "rules": NEXT_ACTION}}
    }
    for operation, candidates in targets.items():
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": {
                index: {
                    "element": f"[{index}] {a['label']}",
                    "current_value": a.get("current_value", a.get("value", "")),
                    **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},
                }
                for index, a in candidates.items()
            },
            "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
        }
    body = {
        "model": os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        "state": {
            "page": {k: state[k] for k in ("url", "title", "text")},
            "elements": elements,
            "recent_actions": [
                {k: h.get(k) for k in ("action", "kind", "text", "page_changed")} for h in history[-10:]
            ],
        },
        "questions": questions,
    }
    if local_mode:
        body = {"state": body["state"], "questions": local_questions(questions)}
    started = time.perf_counter()
    result = post_json(endpoint, None if local_mode else os.environ["TYPESAFE_API_KEY"], body)
    operation_answer = validate_choice(result["answers"].get("operation", {}), operations)
    operation = operation_answer["choice"]
    target = None
    target_answer = None
    target_source = None
    probabilities = {}
    if operation in targets:
        # Unused target heads cannot cause an action. Validate the head selected by the operation.
        if local_mode and len(targets[operation]) == 1:
            target = next(iter(targets[operation]))
            target_source = "single_legal_target"
        else:
            target_answer = validate_choice(
                result["answers"].get(operation.lower() + "_target", {}), targets[operation]
            )
            target = target_answer["choice"]
            target_source = "model"
        choice = targets[operation][target]["id"]
        probabilities = (
            {a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()}
            if target_answer else {choice: 1.0}
        )
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "target_source": target_source,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "raw_answers": result["answers"],
        "model": result["model"],
        "usage": result.get("usage", {}),
        "model_latency_ms": result.get("latency_ms"),
        "native_latency_ms": result.get("native_latency_ms"),
        "native_token_count": result.get("native_token_count"),
        "preprocessing": result.get("preprocessing"),
        "raw_response": result,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": body,
        "provider": "local_system_one" if local_mode else "typesafe",
        "endpoint": endpoint,
    }


def field_context(goal, action, page, history):
    return {
        "goal": goal,
        "field": {k: action.get(k) for k in ("label", "role", "value")},
        "page": {"title": page["title"], "text": page["text"][:6000]},
        "recent_actions": [{k: h.get(k) for k in ("action", "text")} for h in history[-6:]],
    }


def field_text(context):
    local_base = os.environ.get("S1_TEXT_BASE_URL")
    local_mode = local_base is not None
    if "S1_DECISION_URL" in os.environ and not local_mode:
        raise ValueError("Local System One TYPE_TEXT needs S1_TEXT_BASE_URL; no remote text fallback is allowed.")
    key = None if local_mode else os.environ.get("TEXT_MODEL_API_KEY")
    if not local_mode and not key:
        raise ValueError("TYPE_TEXT needs TEXT_MODEL_API_KEY; no text is hardcoded or guessed by the executor.")
    base = (
        local_url(local_base, "S1_TEXT_BASE_URL") if local_mode
        else os.environ.get("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1")
    ).rstrip("/")
    model = os.environ.get("TEXT_MODEL", "" if local_mode else "deepseek-chat")
    if not model:
        raise ValueError("Local text helper needs an explicit TEXT_MODEL identifier; nothing typed.")
    reasoning = (
        {} if local_mode else {"thinking": {"type": "disabled"}} if "api.deepseek.com/" in base
        else {"reasoning": {"effort": "low"}}
    )
    if not local_mode and os.environ.get("TEXT_MODEL_REASONING") == "none":
        reasoning = {"reasoning": {"enabled": False}}
    started = time.perf_counter()
    body = {
        "model": model,
        "max_tokens": 128 if local_mode else 1024,
        "response_format": {"type": "json_object"},
        **reasoning,
        "messages": [
            {"role": "system", "content": TEXT_VALUE},
            {
                "role": "user",
                "content": json.dumps(context),
            },
        ],
    }
    ttft_ms = None
    if local_mode:
        body.update(temperature=0, seed=17, stream=True, stream_options={"include_usage": True})
        result, ttft_ms = stream_local_text(base + "/chat/completions", body)
    else:
        result = post_json(base + "/chat/completions", key, body)
    try:
        output = json.loads(result["choices"][0]["message"]["content"])
        value = output["text"]
        if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ValueError("Text helper returned no valid field value; nothing typed.") from None
    return value, {
        "model": model,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": result.get("usage", {}),
        "ttft_ms": ttft_ms,
        "timings": result.get("timings", {}),
        "provider": "local_openai_compatible" if local_mode else "openai_compatible",
        "endpoint": base + "/chat/completions",
        "settings": {key: body[key] for key in ("max_tokens", "temperature", "seed", "stream") if key in body},
    }
