"""Bounded frozen-state Noul policy experiment using existing local System One.

The default only writes a replay plan. --execute calls an existing numeric-loopback
service without retries. This is a model-use/protocol candidate experiment, not a
single-format ablation, browser executor, calibrated policy or categorical probability.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.replay_decisions import endpoint, extract_case, request_bytes, request_sha256  # noqa: E402

HEAD = "candidate"
INSTRUCTIONS = "Answer the question using the supplied passage."


def build_request(native_choice_request, operation):
    """Retain exact task/policy context; evaluate an actually offered operation claim."""
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
    """Require the provider's genuine complete binary distribution; never reconstruct it."""
    try:
        probabilities = answer["probabilities"]
        values = [*probabilities.values(), answer["noul"], answer["confidence"]]
        valid = (
            set(probabilities) == {"false", "true"}
            and all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in values)
            and abs(sum(probabilities.values()) - 1) <= 1e-6
            and abs(answer["noul"] - probabilities["true"]) <= 1e-6
            and abs(answer["confidence"] - max(probabilities.values())) <= 1e-6
            and answer.get("id", HEAD) == HEAD
            and answer.get("type", "noul") == "noul"
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid native Noul answer; no score inferred or probability fabricated")
    return answer


def execute_once(plan, url, timeout=90):
    """One genuine score call; transport failures never trigger an alternative route."""
    from jev_ultrafast.model import LOCAL_CLIENT, response_error

    url = endpoint(url)
    if not 0 < timeout <= 90:
        raise ValueError("Native Noul replay timeout must be bounded to 90 seconds")
    record = {**plan, "transport_attempted": True, "model_execution_confirmed": False, "ok": False}
    started = time.perf_counter()
    try:
        response = LOCAL_CLIENT.post(
            url, content=request_bytes(plan["request"]), headers={"Content-Type": "application/json"},
            timeout=timeout, follow_redirects=False,
        )
        record["http_status"] = response.status_code
        response_error(response)
        raw = response.json()
        record["raw_response"] = raw
        expected_identity = plan["expected_native_identity"]
        if raw.get("status") != "ok" or any(raw.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("Native model or saved calibration differs from the frozen source identity")
        if set(raw["answers"]) != {HEAD}:
            raise ValueError("Native response question IDs differ from this single Noul request")
        answer = validate_noul(raw["answers"][HEAD])
        record.update(
            ok=True, model_execution_confirmed=True, p_true=answer["noul"], actual_answer=deepcopy(answer),
            adapter_scoring_ms=raw.get("latency_ms"), native_token_count=raw.get("native_token_count"),
        )
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)[:1000]}
    finally:
        record["client_wall_ms"] = (time.perf_counter() - started) * 1000
    return record


def rank_operations(records, operations):
    """Compare raw independent P(true); a sum-to-one operation distribution is not claimed."""
    if not records or any(not record.get("ok") for record in records):
        raise ValueError("Every planned candidate needs a successful real Noul result before ranking")
    actual = [record["operation"] for record in records]
    if len(set(operations)) != len(operations) or len(set(actual)) != len(actual) or set(actual) != set(operations):
        raise ValueError("Every declared candidate needs exactly one score before ranking")
    if len({record["original_native_request_sha256"] for record in records}) != 1:
        raise ValueError("Only candidates grounded in the same original observation can be compared")
    by_operation = {record["operation"]: validate_noul(record["actual_answer"])["noul"] for record in records}
    scores = {operation: by_operation[operation] for operation in operations}
    chosen = max(scores, key=scores.__getitem__)
    return {
        "selected_operation": chosen, "independent_p_true_scores": scores,
        "tie_policy": "first candidate in the declared replay order",
        "selection_source": "argmax_of_independent_native_noul_scores",
        "categorical_probabilities": None,
        "calibration_scope": "Existing BoolQ Noul temperature; navigation calibration not established",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--cases", default="click-pre-complete,click-post-complete")
    parser.add_argument("--operations", default="CLICK,DONE")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--max-calls", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--decision-url")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases, operations = args.cases.split(","), args.operations.split(",")
    try:
        supported_cases = {"click-pre-complete", "click-post-complete", "travel-pre-open", "travel-detail"}
        if len(set(cases)) != len(cases) or set(cases) - supported_cases:
            raise ValueError("--cases must name distinct frozen cases from the existing trace")
        if len(set(operations)) != len(operations) or not all(operations):
            raise ValueError("--operations must list distinct actually offered operation keys")
        calls = len(cases) * len(operations)
        if not 1 <= args.max_calls <= 8 or calls > args.max_calls or not 0 < args.timeout <= 90:
            raise ValueError("Bounded probe: at most8 actual calls, default4; timeout <=90 seconds")
        if args.output.exists():
            raise ValueError("--output must be fresh; prior evidence is never overwritten")
        if args.output.resolve().is_relative_to(args.trace_dir.resolve()):
            raise ValueError("Probe output must remain outside the preserved trace directory")
        if args.execute:
            args.decision_url = endpoint(args.decision_url)
        plans, provenance = [], {}
        for case in cases:
            original, source = extract_case(args.trace_dir, case, args.run_index)
            service_identity = source["source_model_metadata"]["decision_service_identity"]
            expected_identity = {"model": service_identity["model_file"],
                                 "temperature": service_identity["temperature"]}
            # External expected labels and fixture checks never enter the replay payload or passage.
            provenance[case] = {key: value for key, value in source.items() if key != "external_expected_label"}
            for operation in operations:
                request = build_request(original, operation)
                plans.append({
                    "case": case, "operation": operation, "request": request,
                    "original_native_request_sha256": request_sha256(original),
                    "native_request_sha256": request_sha256(request),
                    "expected_native_identity": expected_identity,
                })
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "execute": args.execute, "decision_url": args.decision_url if args.execute else None,
        "planned_calls": calls, "maximum_calls": args.max_calls, "retry_count": 0,
        "per_call_timeout_seconds": args.timeout, "provenance": provenance,
        "experiment": "Model-use/protocol candidate: native trained Noul primitive at frozen observations",
        "input_boundary": (
            "Exact goal/full state/original operation rules/legal descriptions retained; no gold or checker"
        ),
        "timing_scope": "Client wall and adapter preparation/native scoring, not native-only compute",
        "calibration_scope": "Existing BoolQ Noul calibration; no navigation calibration claim",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (args.output / "plans.jsonl").write_text(
        "".join(json.dumps(plan, ensure_ascii=False) + "\n" for plan in plans)
    )
    results, rankings = [], {}
    if args.execute:
        for plan in plans:
            result = execute_once(plan, args.decision_url, args.timeout)
            results.append(result)
            with (args.output / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(json.dumps({key: result.get(key) for key in
                              ("case", "operation", "ok", "p_true", "client_wall_ms")}), flush=True)
            if not result["ok"]:
                break
        for case in cases:
            group = [record for record in results if record["case"] == case]
            if len(group) == len(operations) and all(record["ok"] for record in group):
                rankings[case] = rank_operations(group, operations)
    summary = {
        "planned_calls": calls, "attempted_requests": len(results),
        "confirmed_model_calls": sum(result["model_execution_confirmed"] for result in results),
        "failed_calls": sum(not result["ok"] for result in results), "rankings": rankings,
        "interpretation": "Frozen replay scores only; no browser success or calibrated categorical policy claimed",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return int(any(not result["ok"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
