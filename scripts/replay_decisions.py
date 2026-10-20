"""Plan or run at most eight real local fixed-observation decision ablations.

No browser actions or model startup. Default is an offline plan; --execute uses an
explicit numeric-loopback /decide URL and never retries. Gold evaluation metadata
stays outside request bodies. Each variant starts from the unchanged original body.
"""

import argparse
import difflib
import hashlib
import ipaddress
import json
import string
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ABLATIONS = (
    "baseline", "operation-only", "operation-last", "plain-instructions", "reverse-candidates", "state-question"
)
HYPOTHESES = {
    "baseline": "Reproduce the original choice on the exact stored observation.",
    "operation-only": "Does removing speculative target heads affect the operation choice?",
    "operation-last": "Does placing the operation head after target heads affect operation scoring?",
    "plain-instructions": "Does faithful plain-text instruction formatting affect instruction comprehension?",
    "reverse-candidates": "Does legal candidate position affect choice after semantic keys are remapped?",
    "state-question": "Does duplicating the unchanged original goal into the training-style state.question field help?",
}


def request_bytes(body):
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def request_sha256(body):
    """This hashes the exact compact UTF-8 body sent, preserving all candidate order."""
    return hashlib.sha256(request_bytes(body)).hexdigest()


def plain_instructions(encoded):
    value = json.loads(encoded)
    if not isinstance(value, dict) or set(value) - {"goal", "operation", "rules"}:
        raise ValueError("Instructions must be the recorded goal/operation/rules object")
    if not isinstance(value.get("goal"), str) or not isinstance(value.get("rules"), (str, list)):
        raise ValueError("Recorded instructions lack a textual goal or rules")
    rules = value["rules"] if isinstance(value["rules"], list) else [value["rules"]]
    if not all(isinstance(rule, str) for rule in rules):
        raise ValueError("Recorded instruction rules must all be text")
    sections = ["Goal:", value["goal"]]
    if "operation" in value:
        sections.extend(("Operation:", value["operation"]))
    sections.extend(("Rules:", *rules))
    return "\n".join(sections)


def ablate(body, name):
    if name not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {name}")
    variant = deepcopy(body)
    questions = variant["questions"]
    if name == "operation-only":
        variant["questions"] = {"operation": questions["operation"]}
    elif name == "operation-last":
        variant["questions"] = {key: value for key, value in questions.items() if key != "operation"}
        variant["questions"]["operation"] = questions["operation"]
    elif name == "plain-instructions":
        for question in questions.values():
            question["instructions"] = plain_instructions(question["instructions"])
    elif name == "reverse-candidates":
        for question in questions.values():
            question["criteria"] = dict(reversed(list(question["criteria"].items())))
    elif name == "state-question":
        goal = json.loads(body["questions"]["operation"]["instructions"])["goal"]
        if not isinstance(goal, str):
            raise ValueError("Recorded operation goal must be text")
        if "question" in body["state"] and body["state"]["question"] != goal:
            raise ValueError("Cannot replace a different existing state.question")
        variant["state"]["question"] = goal
    applicable = name == "baseline" or request_bytes(variant) != request_bytes(body)
    notes = {
        "applicable": applicable,
        "skip_reason": None if applicable else "Original observation already has this layout; variant is a no-op.",
        "head_order_before": list(body["questions"]), "head_order_after": list(variant["questions"]),
        "candidate_remapping": {
            head: {
                "before_order": list(body["questions"][head]["criteria"]),
                "after_order": list(question["criteria"]),
                "candidate_marker_to_semantic_key": dict(zip(string.ascii_uppercase + string.ascii_lowercase,
                                                            question["criteria"], strict=False)),
                "semantic_keys_unchanged": True,
            }
            for head, question in variant["questions"].items()
        },
    }
    if name == "state-question":
        if any(variant["state"].get(key) != value for key, value in body["state"].items()):
            raise AssertionError("State-question ablation changed an existing observed field")
    elif variant["state"] != body["state"]:
        raise AssertionError("An ablation changed the observed state")
    for head, question in variant["questions"].items():
        if question["criteria"] != body["questions"][head]["criteria"]:
            raise AssertionError("An ablation changed a legal semantic candidate")
    return variant, notes


def extract_case(trace_dir, case, run_index):
    source = trace_dir / "warm/runs.jsonl"
    runs = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    fixture = "click" if case.startswith("click-") else "travel"
    matches = [run for run in runs if run["kind"] == "browser_task" and run["phase"] == "measured"
               and run.get("fixture") == fixture and run["index"] == run_index]
    if len(matches) != 1:
        raise ValueError("Expected exactly one recorded measured browser run for the case and index")
    calls = [(index, call) for index, call in enumerate(matches[0]["model_calls"])
             if call["kind"] == "decision" and call["ok"]]
    candidates = []
    for call_index, call in calls:
        decision = call["result"]
        body = decision["request"]
        page = body["state"]["page"]
        parsed = urlsplit(page["url"])
        if not ipaddress.ip_address(parsed.hostname or "").is_loopback:
            raise ValueError("Replay sources must be the recorded local fixtures")
        if case == "click-pre-complete":
            selected = "Completion not yet confirmed" in page["text"] and not body["state"]["recent_actions"]
        elif case == "click-post-complete":
            selected = "Completion confirmed" in page["text"] and "Completion not yet confirmed" not in page["text"]
        elif case == "travel-pre-open":
            selected = decision["operation"] == "CLICK" and "#casa-flora" not in page["url"] and any(
                element.get("label") == "Stay category" and element.get("value") == "Design"
                for element in body["state"]["elements"]
            )
        else:
            selected = page["url"].endswith("#casa-flora")
        if selected:
            candidates.append((call_index, decision, body))
    if not candidates:
        raise ValueError(f"Recorded case {case} was not found")
    call_index, original, body = candidates[0]
    if set(body) != {"state", "questions"} or "operation" not in body["questions"]:
        raise ValueError("Recorded request does not match the native local System One contract")
    expected = {
        "click-pre-complete": {
            "operation": "CLICK", "allowed_targets": ["1"],
            "reason": "The original goal's requested completion is not yet confirmed; Mark complete can progress.",
        },
        "click-post-complete": {
            "operation": "DONE", "reason": "The original goal's requested completion is visibly confirmed.",
        },
        "travel-pre-open": {
            "operation": "CLICK", "allowed_targets": ["5", "7"],
            "reason": "Applied search and cancellation control remain unsatisfied; either can progress before opening.",
        },
        "travel-detail": {
            "operation": "CLICK", "allowed_targets": ["1", "2", "4"],
            "reason": "The visible filter receipt remains unsatisfied; return to the controls to progress.",
        },
    }[case]
    provenance = {
        "case": case, "measured_run_index": run_index, "model_call_index": call_index,
        "source_runs_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "original_native_request_sha256": request_sha256(body),
        "original_operation": original["operation"], "original_target": original["target"],
        "original_answers": original["raw_answers"], "external_expected_label": expected,
        "original_model_metadata_sha256": hashlib.sha256((trace_dir / "model-metadata.json").read_bytes()).hexdigest(),
        "source_model_metadata": json.loads((trace_dir / "model-metadata.json").read_text()),
    }
    return body, provenance


def endpoint(value):
    parsed = urlsplit(value or "")
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
        port = parsed.port
    except ValueError:
        address, port = None, None
    if (
        parsed.scheme != "http" or not address or not address.is_loopback
        or getattr(address, "ipv4_mapped", None) is not None or port is None or parsed.path != "/decide"
        or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment
    ):
        raise ValueError("--decision-url must be an explicit HTTP numeric-loopback /decide URL without credentials")
    return value


def input_diff(before, after):
    before = json.dumps(before, ensure_ascii=False, indent=2).splitlines()
    after = json.dumps(after, ensure_ascii=False, indent=2).splitlines()
    return "\n".join(difflib.unified_diff(before, after, fromfile="original", tofile="variant", lineterm=""))


def execute_once(plan, url, timeout):
    from jev_ultrafast.model import LOCAL_CLIENT, response_error, validate_choice

    record = {**plan, "ok": False, "executed": True}
    started = time.perf_counter()
    try:
        response = LOCAL_CLIENT.post(url, content=request_bytes(plan["request"]),
                                     headers={"Content-Type": "application/json"},
                                     follow_redirects=False, timeout=timeout)
        record["http_status"] = response.status_code
        response_error(response)
        raw = response.json()
        record["raw_response"] = raw
        answers = raw["answers"]
        for head, question in plan["request"]["questions"].items():
            validate_choice(answers.get(head, {}), question["criteria"])
        operation = answers["operation"]["choice"]
        target_head = operation.lower() + "_target"
        target = answers.get(target_head, {}).get("choice")
        expected = plan["external_expected_label"]
        record.update(
            ok=True, actual_operation=operation, actual_target=target, actual_answers=answers,
            agrees_with_external_operation=operation == expected["operation"],
            agrees_with_external_target=(
                target in expected["allowed_targets"] if "allowed_targets" in expected and target is not None else None
            ),
            adapter_scoring_ms=raw.get("latency_ms"), native_token_count=raw.get("native_token_count"),
        )
    except Exception as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    finally:
        record["client_wall_ms"] = (time.perf_counter() - started) * 1000
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--case", choices=(
        "click-pre-complete", "click-post-complete", "travel-pre-open", "travel-detail"
    ), required=True)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--ablations", default=",".join(ABLATIONS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-calls", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--decision-url")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = args.ablations.split(",")
    try:
        if len(set(names)) != len(names) or any(name not in ABLATIONS for name in names):
            raise ValueError("--ablations must list distinct supported names")
        if not 1 <= args.repeats <= 2 or not 1 <= args.max_calls <= 8 or not 0 < args.timeout <= 90:
            raise ValueError("Bounded replay: repeats 1..2, total actual calls <=8, timeout <=90 seconds")
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("--output must be a new or empty directory; original evidence is never overwritten")
        if args.output.resolve().is_relative_to(args.trace_dir.resolve()):
            raise ValueError("Replay output must be outside the preserved trace directory")
        if args.execute:
            args.decision_url = endpoint(args.decision_url)
        body, provenance = extract_case(args.trace_dir, args.case, args.run_index)
        plans = []
        for name in names:
            variant, notes = ablate(body, name)
            plans.append({
                "ablation": name, "hypothesis": HYPOTHESES[name], **notes,
                "native_request_sha256": request_sha256(variant), "request": variant,
                "raw_input_diff": input_diff(body, variant),
                "external_expected_label": provenance["external_expected_label"],
            })
        count = sum(plan["applicable"] for plan in plans) * args.repeats
        if count > args.max_calls:
            raise ValueError(f"Selected plan needs {count} actual calls, exceeding --max-calls {args.max_calls}")
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "execute": args.execute, "decision_url": args.decision_url if args.execute else None,
        "repeats": args.repeats, "planned_actual_calls": count, "maximum_actual_calls": args.max_calls,
        "per_call_timeout_seconds": args.timeout, "retry_count": 0, "provenance": provenance,
        "evaluation_boundary": "Expected labels and independent evaluation are never inserted into request bodies.",
        "timing_scope": "Adapter reported latency includes preparation/native IPC, not native-only forward time.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with (args.output / "plans.jsonl").open("w") as stream:
        for plan in plans:
            stream.write(json.dumps(plan, ensure_ascii=False) + "\n")
    results = []
    if args.execute:
        # Round-robin variants across repeats; no mutation and no network retry.
        for repeat in range(args.repeats):
            for plan in plans:
                if not plan["applicable"]:
                    continue
                result = execute_once({**plan, "repeat": repeat}, args.decision_url, args.timeout)
                results.append(result)
                with (args.output / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(json.dumps({key: result.get(key) for key in
                                  ("ablation", "repeat", "ok", "actual_operation", "actual_target", "client_wall_ms")}),
                      flush=True)
                if not result["ok"]:
                    break
            if results and not results[-1]["ok"]:
                break
    summary = {
        "planned_actual_calls": count, "executed_calls": len(results),
        "failed_calls": sum(not result["ok"] for result in results),
        "skipped_noop_ablations": [plan["ablation"] for plan in plans if not plan["applicable"]],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return int(any(not result["ok"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
