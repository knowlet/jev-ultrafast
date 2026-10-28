"""Audit the immutable October 14 native packet without loading or running a model.

Raw and probability tolerances are fixed. The default exit status is 1 because
the historical candidate failed parity. --audit-only returns 0 only when that
failed verdict is independently reproduced and the complete archive is intact;
it never certifies a new candidate or permits a browser benchmark.
"""

import argparse
import hashlib
import json
import math
import sys
from functools import wraps
from pathlib import Path

RAW_ATOL = 1e-4
PROBABILITY_ATOL = 1e-5
ARCHIVE_MANIFEST_SHA256 = "41ac5b2fa26d2c9fd563b48d7501d27a5ee4d3b47c60eb50555137cf372d3504"
DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "benchmark/public/native-prefix-parity-2026-10-14"
MODES = ("original", "prefix_off", "prefix_on")
RUNTIME_FIELDS = (
    "autoregressive_tokens", "batch_size", "context_size", "decision_passes", "engine",
    "gpu_layers_requested", "image_preprocessing", "llama_cpp_revision", "logits_semantics",
    "max_output_slots", "media_projection", "microbatch_size", "projection",
)


class AnalysisError(ValueError):
    """Evidence is incomplete, malformed, or internally inconsistent."""


def evidence_schema(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (KeyError, TypeError, IndexError, OverflowError, ValueError) as exc:
            if isinstance(exc, AnalysisError):
                raise
            raise AnalysisError(f"Malformed evidence in {function.__name__}: {exc}") from exc
    return checked


def require(condition, message):
    if not condition:
        raise AnalysisError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def wire_hash(value):
    # Key/choice insertion order is part of the original wire input.
    return digest(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


def finite_number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f"{label}: expected finite number")
    return value


def integers(value, label, *, nonempty=True):
    require(isinstance(value, list) and (bool(value) or not nonempty), f"{label}: expected integer list")
    require(all(type(item) is int and 0 <= item <= 2**31 - 1 for item in value),
            f"{label}: expected nonnegative int32")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise AnalysisError(f"Nonfinite JSON number: {value}")


def parse_json(data):
    try:
        return json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise AnalysisError("Invalid or incomplete JSON evidence") from exc


def read_rows(path):
    lines = path.read_bytes().splitlines()
    require(bool(lines) and all(line.strip() for line in lines), f"{path.name}: empty JSONL record")
    return [parse_json(line) for line in lines]


def verify_archive(source):
    """Verify public hashes, never compare redacted bytes against original hashes."""
    manifest_bytes = (source / "sanitization-manifest.json").read_bytes()
    require(digest(manifest_bytes) == ARCHIVE_MANIFEST_SHA256, "Historical archive manifest hash mismatch")
    manifest = parse_json(manifest_bytes)
    require(len(manifest["files"]) == 28, "Incomplete public archive manifest")
    seen = set()
    for entry in manifest["files"]:
        name = entry["path"]
        path = (source / name).resolve()
        require(path.is_relative_to(source.resolve()) and name not in seen, "Invalid archive member path")
        seen.add(name)
        data = path.read_bytes()
        require(digest(data) == entry["public_sha256"], f"Public SHA-256 mismatch: {name}")
        require(len(data) == entry["public_bytes"], f"Public byte count mismatch: {name}")
    return {"passed": True, "verified_files": len(seen), "manifest_sha256": ARCHIVE_MANIFEST_SHA256}


@evidence_schema
def validate_prepared(prepared):
    """Validate a nonempty ordered corpus of already-tokenized text requests."""
    require(isinstance(prepared, list) and bool(prepared), "Prepared corpus is empty")
    ids = set()
    for row in prepared:
        payload = row["payload"]
        case_id = payload["case_id"]
        require(isinstance(case_id, str) and case_id and case_id not in ids, "Duplicate/invalid prepared case_id")
        ids.add(case_id)
        require(wire_hash(payload) == row["payload_sha256"], "Prepared payload hash mismatch")
        require(wire_hash(row["request"]) == row["request_sha256"], "Prepared request hash mismatch")
        tokens = integers(row["input_ids"], "input_ids")
        require(payload["parts"] == [{"type": "text", "tokens": tokens}], "Expected exactly one text part")
        integers(payload["parts"][0]["tokens"], "payload tokens")
        require(type(payload["source_input_token_count"]) is int, "Invalid source token count")
        require(payload["source_input_token_count"] == len(tokens), "Source token count mismatch")
        slots = integers(payload["slots"], "slots")
        nopts = integers(payload["nopts"], "nopts")
        letters = integers(payload["letters"], "letters")
        require(slots == integers(row["slots"], "prepared slots")
                and nopts == integers(row["nopts"], "prepared nopts")
                and letters == integers(row["letters"], "prepared letters"),
                "Prepared slot/options metadata mismatch")
        require(all(slot < len(tokens) for slot in slots) and slots == sorted(set(slots)),
                "Invalid slots: must be strictly increasing within input bounds")
        questions = payload["question_ids"]
        require(isinstance(questions, list) and len(questions) == len(slots) == len(nopts), "Question shape mismatch")
        require(all(isinstance(q, str) and q for q in questions) and len(set(questions)) == len(questions),
                "Invalid question identities")
        require(len(letters) == len(set(letters)) == 52, "Expected exactly 52 distinct letter token IDs")
        require(all(2 <= count <= 52 for count in nopts), "Invalid option count: expected 2..52")


@evidence_schema
def validate_native(prepared, response):
    payload = prepared["payload"]
    require(response["status"] == "ok", "Native response did not complete successfully")
    require(response["case_id"] == payload["case_id"], "Native case_id mismatch")
    require(response["question_ids"] == payload["question_ids"], "Native question identity mismatch")
    for key in ("slots", "runtime_slots"):
        require(integers(response[key], key) == payload["slots"], f"Native {key} mismatch")
    require(type(response["native_token_count"]) is int, "Invalid native token count")
    require(response["native_token_count"] == len(prepared["input_ids"]), "Native token count mismatch")
    require(finite_number(response["latency_ms"], "latency_ms") >= 0, "Negative native latency")
    runtime = response["runtime"]
    require(all(key in runtime for key in RUNTIME_FIELDS), "Incomplete runtime identity")
    for key in ("batch_size", "microbatch_size", "context_size", "max_output_slots", "decision_passes"):
        require(type(runtime[key]) is int and runtime[key] > 0, f"Invalid runtime {key}")
    logits = response["raw_logits"]
    require(isinstance(logits, list) and len(logits) == len(payload["nopts"]), "Raw logit row count mismatch")
    for values, count in zip(logits, payload["nopts"], strict=True):
        require(isinstance(values, list) and len(values) == count, "Raw logit option count mismatch")
        for value in values:
            finite_number(value, "raw logit")
    return logits


@evidence_schema
def calibrated_probabilities(logits, temperature):
    require(finite_number(temperature, "temperature") > 0, "Temperature must be positive")
    require(isinstance(logits, list) and bool(logits), "Empty logit matrix")
    result = []
    for row in logits:
        require(isinstance(row, list) and bool(row), "Empty logit row")
        for value in row:
            finite_number(value, "raw logit")
        maximum = max(row)
        values = [math.exp((value - maximum) / temperature) for value in row]
        total = sum(values)
        result.append([value / total for value in values])
    return result


@evidence_schema
def compare_native_responses(prepared, baseline, candidate, *, temperature):
    """Compare paired raw responses using fixed tolerances and fresh softmax.

This API can compare future text-native captures. It establishes numerical
equivalence of the supplied pairs only; the caller must separately bind actual
executed binaries, model files, calibration, transport and cache telemetry.
"""
    validate_prepared(prepared)
    require(len(baseline) == len(candidate) == len(prepared), "Missing or extra paired response")
    runtime_identity = {key: baseline[0]["runtime"][key] for key in RUNTIME_FIELDS}
    comparisons = []
    for index, (input_row, original, current) in enumerate(zip(prepared, baseline, candidate, strict=True)):
        a = validate_native(input_row, original)
        b = validate_native(input_row, current)
        require(all(original["runtime"][key] == current["runtime"][key] for key in RUNTIME_FIELDS),
                "Runtime identity mismatch between modes")
        require(all(original["runtime"][key] == value for key, value in runtime_identity.items()),
                "Runtime identity changed within mode")
        pa, pb = calibrated_probabilities(a, temperature), calibrated_probabilities(b, temperature)
        raw_delta = max(abs(x - y) for aa, bb in zip(a, b, strict=True) for x, y in zip(aa, bb, strict=True))
        prob_delta = max(abs(x - y) for aa, bb in zip(pa, pb, strict=True) for x, y in zip(aa, bb, strict=True))
        finite_number(raw_delta, "raw delta")
        finite_number(prob_delta, "probability delta")
        labels_match = all(max(range(len(aa)), key=aa.__getitem__) == max(range(len(bb)), key=bb.__getitem__)
                           for aa, bb in zip(pa, pb, strict=True))
        # Check every option's order, including near-tied non-winning options.
        # Use logits to retain distinctions that softmax underflow can hide.
        option_rankings_match = all(
            sorted(range(len(aa)), key=aa.__getitem__, reverse=True)
            == sorted(range(len(bb)), key=bb.__getitem__, reverse=True)
            for aa, bb in zip(a, b, strict=True)
        )
        comparisons.append({
            "index": index, "case_id": input_row["payload"]["case_id"],
            "max_raw_abs_delta": raw_delta, "max_probability_abs_delta": prob_delta,
            "labels_match": labels_match, "option_rankings_match": option_rankings_match,
            "within_fixed_tolerance": (raw_delta <= RAW_ATOL and prob_delta <= PROBABILITY_ATOL
                                       and labels_match and option_rankings_match),
        })
    return {"raw_abs_atol": RAW_ATOL, "prob_abs_atol": PROBABILITY_ATOL, "rtol": 0,
            "comparisons": comparisons, "numerical_parity_pass": all(c["within_fixed_tolerance"] for c in comparisons)}


@evidence_schema
def validate_mode(prepared, native, records, transport, raw_lines, *, mode, identity):
    """Cross-check one mode's independent native, typed and transport records."""
    requests = [event for event in transport if event["event"] == "request"]
    responses = [event for event in transport if event["event"] == "response"]
    require(len(native) == len(records) == len(requests) == len(responses) == len(raw_lines) == len(prepared),
            "Incomplete native/records/transport pairing")
    require([event["event"] for event in transport] == ["spawn"] + ["request", "response"] * len(prepared),
            "Transport event order mismatch")
    max_typed_error = 0.0
    for index, (prep, raw, record, req, resp, line) in enumerate(
        zip(prepared, native, records, requests, responses, raw_lines, strict=True)
    ):
        require(record["mode"] == mode and type(record["index"]) is int and record["index"] == index,
                "Record mode/index mismatch")
        require(all(record[key] == prep[key] for key in ("case", "operation", "repetition")),
                "Record case/operation/repetition mismatch")
        require(record["raw_response"] == raw, "Record and native stdout disagree")
        require(record["payload_sha256"] == req["payload_sha256"] == prep["payload_sha256"],
                "Transport payload hash mismatch")
        require(req["case_id"] == resp["case_id"] == raw["case_id"], "Transport case_id mismatch")
        require(digest(line) == resp["raw_line_sha256"], "Transport native stdout hash mismatch")
        require(record["status"] == "ok" and record["transport_attempted"] is True
                and record["model_execution_confirmed"] is True, "Unconfirmed model response")
        logits = validate_native(prep, raw)
        require(prep["payload"]["nopts"] == [2], "Historical corpus must have one binary slot")
        typed = record["typed_response"]
        require(typed["status"] == "ok" and typed["model"] == identity["model"], "Typed model identity mismatch")
        require(typed["temperature"] == identity["temperature"], "Calibration temperature mismatch")
        for key in ("runtime", "runtime_slots", "native_token_count", "preprocessing", "latency_ms"):
            require(typed[key] == raw[key], f"Typed/native {key} mismatch")
        require(typed["passes"] == raw["runtime"]["decision_passes"], "Typed pass count mismatch")
        question = prep["payload"]["question_ids"][0]
        require(list(typed["answers"]) == [question], "Typed question mismatch")
        answer = typed["answers"][question]
        require(answer["id"] == question and answer["type"] == "noul", "Typed answer identity mismatch")
        probabilities = answer["probabilities"]
        require(list(probabilities) == ["false", "true"], "Typed probability labels/order mismatch")
        expected = calibrated_probabilities(logits, identity["temperature"])[0]
        for key, value in zip(("false", "true"), expected, strict=True):
            observed = finite_number(probabilities[key], "typed probability")
            require(0 <= observed <= 1, "Probability outside [0,1]")
            error = abs(observed - value)
            # This tighter serialization/math check is separate from the unchanged
            # 1e-5 cross-mode probability acceptance gate.
            require(error <= 1e-12, "Typed probabilities do not derive from raw logits and temperature")
            max_typed_error = max(max_typed_error, error)
        require(record["p_true"] == probabilities["true"] == answer["noul"], "Typed p_true mismatch")
        require(abs(finite_number(answer["confidence"], "confidence") - max(expected)) <= 1e-12,
                "Typed confidence mismatch")
        require(abs(finite_number(answer["margin"], "margin") - abs(expected[1] - expected[0])) <= 1e-12,
                "Typed margin mismatch")
        require(record["truth_label"] == ("true" if expected[1] >= expected[0] else "false"),
                "Recorded truth label mismatch")
        require(finite_number(record["client_wall_ms"], "client wall") >= 0, "Negative client wall")
    return max_typed_error


def geometry(prepared, native):
    previous = []
    result = []
    for index, (prep, raw) in enumerate(zip(prepared, native, strict=True)):
        tokens, slot = prep["input_ids"], min(prep["slots"])
        lcp = 0
        for old, new in zip(previous, tokens):
            if old != new:
                break
            lcp += 1
        cache = raw["preprocessing"]["prefix_cache"]
        cap = min(lcp, slot)
        batch = raw["runtime"]["batch_size"]
        aligned = cap // batch * batch
        require(cache["enabled"] is True and cache["eligible"] is True
                and cache["cache_valid_for_next_request"] is True
                and cache["requested_slots_recomputed"] is True, "Invalid historical cache eligibility")
        require(cache["exact_common_prefix_tokens"] == lcp and cache["previous_input_tokens"] == len(previous),
                "Historical LCP telemetry mismatch")
        require(cache["retained_kv_tokens"] == cap and cache["recomputed_token_positions"] == len(tokens) - cap
                and cache["cache_hit"] is (cap > 0), "Historical retention telemetry mismatch")
        result.append({"index": index, "input_tokens": len(tokens), "exact_lcp": lcp, "first_requested_slot": slot,
                       "observed_retained": cap, "observed_recomputed": len(tokens) - cap,
                       "planned_aligned_retained": aligned, "planned_aligned_recomputed": len(tokens) - aligned,
                       "native_batch_size": batch})
        previous = tokens
    return result


def rankings(prepared, native, temperature):
    groups = {}
    for prep, response in zip(prepared, native, strict=True):
        key = f"{prep['case']}/repeat-{prep['repetition']}"
        scores = groups.setdefault(key, {})
        require(prep["operation"] not in scores, "Duplicate operation in ranking pair")
        scores[prep["operation"]] = calibrated_probabilities(response["raw_logits"], temperature)[0][1]
    require(bool(groups) and all(set(scores) == {"CLICK", "DONE"} for scores in groups.values()),
            "Incomplete CLICK/DONE ranking pair")
    return {key: sorted(scores, key=scores.get, reverse=True) for key, scores in groups.items()}


@evidence_schema
def analyze_archive(source=DEFAULT_SOURCE):
    source = Path(source)
    integrity = verify_archive(source)
    parity = source / "parity"
    manifest = parse_json((parity / "manifest.json").read_bytes())
    identity = manifest["identity"]
    require((manifest["raw_abs_atol"], manifest["prob_abs_atol"], manifest["rtol"])
            == (RAW_ATOL, PROBABILITY_ATOL, 0), "Frozen tolerance metadata changed")
    require(identity["bookend_before"] == manifest["bookend_after"]
            and manifest["package_and_sources_unchanged"] is True,
            "Model/source identity changed across historical run")
    prepared = read_rows(parity / "prepared.jsonl")
    validate_prepared(prepared)
    require(len(prepared) == manifest["calls_per_mode"] == 8 and manifest["planned_calls"] == 24,
            "Incomplete historical plan")
    require([row["payload_sha256"] for row in prepared] == manifest["ordered_payload_sha256"],
            "Execution payload order mismatch")
    native, errors, ranks = {}, {}, {}
    for mode in MODES:
        mode_dir = parity / mode
        lines = (mode_dir / "stdout.log").read_bytes().splitlines()
        native[mode] = [parse_json(line) for line in lines]
        errors[mode] = validate_mode(prepared, native[mode], read_rows(mode_dir / "records.jsonl"),
                                     read_rows(mode_dir / "transport.jsonl"), lines, mode=mode, identity=identity)
        ranks[mode] = rankings(prepared, native[mode], identity["temperature"])
    comparisons = {mode: compare_native_responses(prepared, native["original"], native[mode],
                                                  temperature=identity["temperature"])
                   for mode in MODES[1:]}
    cache_rows = geometry(prepared, native["prefix_on"])
    ranking_agreement = all(ranks[mode] == ranks["original"] for mode in MODES[1:])
    numerical = all(result["numerical_parity_pass"] for result in comparisons.values())
    failures = [row["index"] for row in comparisons["prefix_on"]["comparisons"] if not row["within_fixed_tolerance"]]
    reproduced = (comparisons["prefix_off"]["numerical_parity_pass"] and failures == [1, 3, 5, 7]
                  and ranking_agreement and all(row["labels_match"] for row in comparisons["prefix_on"]["comparisons"]))
    return {
        "schema": "jev-historical-native-prefix-audit-v1", "scope": "offline historical evidence audit",
        "archive_integrity": integrity, "historical_verdict_reproduced": reproduced,
        "historical_numerical_acceptance": numerical, "historical_acceptance": numerical and ranking_agreement,
        "raw_abs_atol": RAW_ATOL, "prob_abs_atol": PROBABILITY_ATOL, "rtol": 0,
        "confirmed_historical_responses": sum(map(len, native.values())), "comparisons": comparisons,
        "maximum_typed_probability_recalculation_delta": max(errors.values()),
        "rankings": ranks, "ranking_agreement": ranking_agreement,
        "historical_cache_effectiveness": any(row["observed_retained"] > 0 for row in cache_rows),
        "geometry": {"aligned_values_are_planned_not_observed": True, "rows": cache_rows},
        "new_native_calls": 0, "new_browser_tasks": 0, "new_jevbench_runs": 0,
        "new_candidate_native_parity": "not_executed", "browser_benchmark_allowed": False,
        "performance_improvement_established": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, help="New derived JSON file outside the immutable source directory")
    parser.add_argument("--audit-only", action="store_true",
                        help="Exit 0 for reproduced historical failure, never parity")
    args = parser.parse_args(argv)
    try:
        if args.output:
            require(not args.output.exists() and not args.output.is_symlink(),
                    "Output already exists; refusing overwrite")
            require(not args.output.resolve().is_relative_to(args.source_dir.resolve()),
                    "Output must be outside archive")
        result = analyze_archive(args.source_dir)
        result["exit_status_mode"] = "historical_reproduction_only" if args.audit_only else "historical_parity_gate"
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(text)
        print(text, end="")
        passed = result["historical_verdict_reproduced"] if args.audit_only else result["historical_acceptance"]
        return 0 if passed else 1
    except (AnalysisError, OSError, KeyError, TypeError, OverflowError) as exc:
        print(json.dumps({"scope": "offline historical evidence audit", "evidence_valid": False,
                          "error": str(exc), "new_candidate_native_parity": "not_executed"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
