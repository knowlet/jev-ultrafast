"""Recompute local benchmark evidence offline, retaining failures and candidate order.

All primary datasets must have finished. This script never invokes model providers,
loads weights, starts a browser, or modifies source evidence. Its JSON output omits
personal paths/owners; original private source-file hashes document that derivation.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_benchmark import distribution, summarize  # noqa: E402

PRIMARY = {
    "q4_decisions": "q4-decisions-20261013",
    "q8_decisions": "q8-decisions-20261013",
    "travel": "q4-travel-20261013",
    "click": "q4-click-20261013",
}


class AnalysisError(ValueError):
    """Source evidence is incomplete or does not support the requested comparison."""


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wire_request_sha256(request):
    """Do not sort keys: the order of legal choices is part of the model's input."""
    encoded = json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path):
    if not path.is_file():
        raise AnalysisError(f"Missing source evidence: {path.parent.name}/{path.name}")
    return json.loads(path.read_text())


def read_jsonl(path):
    if not path.is_file():
        raise AnalysisError(f"Missing source evidence: {path.parent.name}/{path.name}")
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Incomplete or invalid JSONL: {path.parent.name}/{path.name}") from exc


def native_peak_rss(log_path):
    if not log_path.is_file():
        raise AnalysisError(f"Missing native resource log: {log_path.name}")
    values = [int(value) for value in re.findall(r"^\s*(\d+)\s+maximum resident set size\s*$",
                                                log_path.read_text(), re.MULTILINE)]
    if not values:
        raise AnalysisError(f"No completed native time RSS report: {log_path.name}")
    return {
        "peak_rss_bytes": max(values), "n_reports": len(values),
        "scope": "macOS /usr/bin/time -l for the owned native model process; bytes, not total system memory",
    }


def verify_complete(dataset_root, warm_metadata, records):
    if not (dataset_root / "warm/summary.json").is_file():
        raise AnalysisError(f"{dataset_root.name}: incomplete; warm/summary.json has not been written")
    config = warm_metadata["configuration"]
    mode, warmups = config["mode"], config["warmup"]
    expected = Counter()
    if mode in {"decision-only", "all"}:
        expected[("decision_microbenchmark", "warmup")] = warmups
        expected[("decision_microbenchmark", "measured")] = config["decision_repeats"]
    if mode in {"helper-only", "all"} and config["text_repeats"]:
        expected[("helper_microbenchmark", "warmup")] = warmups
        expected[("helper_microbenchmark", "measured")] = config["text_repeats"]
    if mode == "all":
        expected[("browser_task", "warmup")] = warmups
        expected[("browser_task", "measured")] = config["repeats"]
    if mode == "smoke":
        expected[("browser_task", "measured")] = 1
    actual = Counter((r["kind"], r["phase"]) for r in records)
    if +actual != +expected:
        raise AnalysisError(
            f"{dataset_root.name}: incomplete run counts; got {dict(actual)}, expected {dict(expected)}"
        )
    events = read_jsonl(dataset_root / "native-requests.jsonl")
    if not any(event.get("event") == "shutdown" for event in events):
        raise AnalysisError(f"{dataset_root.name}: native service has not recorded shutdown")
    native_peak_rss(dataset_root / "native-decision-time.log.native.log")


def public_manifest(manifest):
    # Keep conversion/tensor/file receipts, with explicit names instead of owner identifiers.
    result = {k: v for k, v in manifest.items() if k not in {"source_hub_repository", "source_model", "commands"}}
    result["trained_project_name"] = manifest.get("source_hub_repository", "").rsplit("/", 1)[-1]
    result["base_model_name"] = manifest.get("source_model", "").rsplit("/", 1)[-1]
    return result


def identity(dataset_root, metadata, phases):
    service = metadata.get("decision_service_identity", {})
    manifest = metadata["manifest"]
    return {
        "artifact_file": manifest["model_file"], "weights_revision": metadata["version"],
        "quantization": metadata["quantization"], "runtime": metadata["runtime"],
        "system_one_repo_head": metadata["s1_repo_head"], "llama_cpp_head": metadata["llama_cpp_head"],
        "native_runner_sha256": service.get("native_runner_sha256"),
        "adapter_source_sha256": service.get("adapter_source_sha256"),
        "conversion_manifest_sha256": service.get("conversion_manifest_sha256"),
        "calibration_sha256": service.get("calibration_sha256"), "calibrated_temperature": service.get("temperature"),
        "model_manifest": public_manifest(manifest), "hardware": metadata["hardware"],
        "runtime_settings": metadata["settings"],
        "executed_sources": {
            phase: {k: source.get(k) for k in (
                "repo_head", "tracked_diff_sha256", "source_sha256", "installed_distributions", "python", "machine"
            )}
            for phase, source in phases.items()
        },
        "private_original_file_sha256": {
            str(path.relative_to(dataset_root)): file_sha256(path)
            for path in sorted(dataset_root.rglob("*.json"))
            if path.name in {"metadata.json", "model-metadata.json"} or path.name.startswith("model-metadata-")
        },
        "redaction": "Personal filesystem paths, repository owner identifiers, and launcher commands omitted.",
    }


def helper_metrics(records):
    calls = [call for record in records if record["phase"] == "measured"
             for call in record.get("model_calls", []) if call["kind"] == "text_helper"]
    successful = [call for call in calls if call["ok"]]
    helpers = [call["result"][1] for call in successful]
    rates = [helper.get("timings", {}).get("predicted_per_second") for helper in helpers]
    return {
        "calls": len(calls), "failed_calls": len(calls) - len(successful),
        "first_content_delta_ms": distribution([h.get("ttft_ms") for h in helpers]),
        "full_request_ms": distribution([call["wall_ms"] for call in calls]),
        "provider_decode_tokens_per_second": distribution(rates),
        "provider_decode_rate_scope": (
            "Actual llama-server timings.predicted_per_second, using its convention excluding the first "
            "prefill-produced token from the decode interval; not completion_tokens/full request time."
        ),
        "provider_token_usage": [helper.get("usage", {}) for helper in helpers],
        "provider_timings": [helper.get("timings", {}) for helper in helpers],
    }


def browser_run_evidence(run):
    verification = run.get("verification", {})
    checks = verification.get("checks", {})
    errors = [key for key in ("error", "verification_error", "teardown_error") if key in run]
    strict_success = (
        run.get("successful_task") is True and run.get("verified") is True
        and bool(checks) and all(checks.values()) and not errors
    )
    failed_checks = [key for key, passed in checks.items() if passed is not True]
    calls = run.get("model_calls", [])
    reasons = [f"independent_check_failed:{key}" for key in failed_checks]
    reasons.extend(f"{key}:{run[key].get('type', 'error')}" for key in errors)
    if not strict_success and not reasons:
        reasons.append("independent_success_not_confirmed")
    text = verification.get("fresh_dom", {}).get("text", "")
    return {
        "phase": run["phase"], "index": run["index"], "strict_success": strict_success,
        "agent_status": run.get("agent_state", {}).get("status"), "verification_checks": checks,
        "failure_reasons": reasons,
        "operations": [call.get("result", {}).get("operation", "request_failed")
                       for call in calls if call["kind"] == "decision"],
        "executed_operations": [h.get("operation") for h in run.get("agent_state", {}).get("history", [])],
        "fixture_filter_statement": next(
            (line for line in text.splitlines() if line.startswith("Your filters:")), None
        ),
        **{key: run.get(key) for key in ("decision_calls", "text_calls", "setup_ms", "task_ms", "verification_ms",
                                       "total_ms", "model_wall_ms", "browser_task_wall_ms")},
    }


def load_dataset(dataset_root, *, require_complete=True):
    metadata = read_json(dataset_root / "model-metadata.json")
    phase_metadata, phase_records = {}, {}
    for phase in ("cold", "warm", "helper-probe"):
        if (dataset_root / phase / "runs.jsonl").is_file():
            phase_metadata[phase] = read_json(dataset_root / phase / "metadata.json")
            phase_records[phase] = read_jsonl(dataset_root / phase / "runs.jsonl")
    if "warm" not in phase_records:
        raise AnalysisError(f"{dataset_root.name}: no warm run evidence")
    if require_complete:
        verify_complete(dataset_root, phase_metadata["warm"], phase_records["warm"])
    native_events = read_jsonl(dataset_root / "native-requests.jsonl")
    native_decisions = [event for event in native_events if event.get("event") == "decision"]
    all_records = [record for records in phase_records.values() for record in records]
    calls = [call for record in all_records for call in record.get("model_calls", [])]
    cold_calls = [call for record in phase_records.get("cold", []) for call in record.get("model_calls", [])]
    measured = [run for run in phase_records["warm"] if run["phase"] == "measured"]
    groups = {kind: [run for run in phase_records["warm"] if run["kind"] == kind]
              for kind in {run["kind"] for run in phase_records["warm"]}}
    browser_runs = [browser_run_evidence(run) for run in measured if run["kind"] == "browser_task"]
    derived = {
        "dataset": dataset_root.name, "identity": identity(dataset_root, metadata, phase_metadata),
        "cold_process_first_request_client_ms": distribution([call["wall_ms"] for call in cold_calls]),
        "startup": metadata.get("startup", {}),
        "startup_scope": "Cold process/start-to-first-request completion; filesystem/OS caches were not cleared.",
        "actual_counts": {
            "native_http_decision_requests": len(native_decisions),
            "native_transport_failures": sum(event.get("event") == "transport_failure" for event in native_events),
            "logical_decision_calls": sum(call["kind"] == "decision" for call in calls),
            "logical_helper_calls": sum(call["kind"] == "text_helper" for call in calls),
            "measured_runs": dict(Counter(run["kind"] for run in measured)),
            "warmup_runs": dict(Counter(run["kind"] for run in phase_records["warm"] if run["phase"] == "warmup")),
        },
        "measured_by_workload": {kind: summarize(records) for kind, records in groups.items()},
        "browser_runs": browser_runs,
        "browser_strict_success_count": sum(run["strict_success"] for run in browser_runs),
        "browser_measured_count": len(browser_runs),
        "browser_helper_metrics": helper_metrics([run for run in measured if run["kind"] == "browser_task"]),
        "helper_probe_metrics": helper_metrics(phase_records.get("helper-probe", [])),
        "chromium": {
            key: phase_metadata["warm"].get(key) for key in
            ("chromium_version", "chromium_startup_ms", "chromium_resource_usage", "chromium_resource_usage_scope",
             "chromium_cdp_blocked_requests", "chromium_proxy_blocked_requests")
        },
        "native_model_peak_rss": native_peak_rss(dataset_root / "native-decision-time.log.native.log"),
        "separate_process_resource_usage": metadata.get("resource_usage", {}),
        "resource_scope": "Per-process lifetime peaks; not simultaneous physical memory and must not be summed.",
        "native_wire_request_sha256_counts": dict(Counter(wire_request_sha256(event["request"])
                                                         for event in native_decisions)),
        "native_input_token_count_counts": dict(Counter(str(event["response"].get("native_token_count"))
                                                       for event in native_decisions)),
        "raw_evidence_sha256": {
            str(path.relative_to(dataset_root)): file_sha256(path)
            for path in sorted(dataset_root.rglob("*")) if path.is_file() and
            (path.name == "runs.jsonl" or path.name == "native-requests.jsonl" or "native-decision-time" in path.name)
        },
    }
    return derived, native_decisions


def compare_microbenchmarks(q4, q8):
    for arm in (q4, q8):
        counts = arm["actual_counts"]
        if counts["native_http_decision_requests"] != 13:
            raise AnalysisError(f"{arm['dataset']}: expected 13 native requests (1 cold + 2 warmup + 10 measured)")
        group = arm["measured_by_workload"].get("decision_microbenchmark", {}).get("groups", {})
        if group.get("decision_microbenchmark", {}).get("runs") != 10:
            raise AnalysisError(f"{arm['dataset']}: expected 10 measured fixed-input decisions")
        if len(arm["native_wire_request_sha256_counts"]) != 1:
            raise AnalysisError(f"{arm['dataset']}: native wire input differs across cold/warm requests")
    hashes = [next(iter(arm["native_wire_request_sha256_counts"])) for arm in (q4, q8)]
    if hashes[0] != hashes[1]:
        raise AnalysisError("Q4/Q8 native wire requests differ, including insertion/candidate order")
    if q4["native_input_token_count_counts"] != q8["native_input_token_count_counts"]:
        raise AnalysisError("Q4/Q8 actual native input token counts differ")
    conditions = (
        "weights_revision", "system_one_repo_head", "llama_cpp_head", "native_runner_sha256",
        "adapter_source_sha256", "hardware", "runtime_settings",
    )
    for key in conditions:
        if q4["identity"].get(key) != q8["identity"].get(key):
            raise AnalysisError(f"Q4/Q8 comparison conditions differ: {key}")
    for key in ("source_checkpoint_sha256", "source_s1_config_sha256"):
        if q4["identity"]["model_manifest"].get(key) != q8["identity"]["model_manifest"].get(key):
            raise AnalysisError(f"Q4/Q8 trained source identity differs: {key}")
    left = q4["measured_by_workload"]["decision_microbenchmark"]["decision_wall_ms_all_attempts"]["p50"]
    right = q8["measured_by_workload"]["decision_microbenchmark"]["decision_wall_ms_all_attempts"]["p50"]
    return {
        "exact_input_match": True, "wire_request_sha256": hashes[0],
        "wire_hash_method": "SHA256 compact JSON UTF-8, ensure_ascii=False, original insertion order, no key sorting",
        "native_requests_per_arm": 13, "measured_requests_per_arm": 10,
        "native_input_token_count_counts": q4["native_input_token_count_counts"],
        "matched_conditions": list(conditions) + ["source_checkpoint_sha256", "source_s1_config_sha256"],
        "q4_relative_median_client_time_reduction": 1 - left / right if right else None,
        "q8_over_q4_median_client_time_ratio": right / left if left else None,
        "comparison_scope": (
            "One sequential Q4 batch and one sequential Q8 batch of the same trained checkpoint; identical wire "
            "inputs, no alternating pairs. Model-specific calibrated temperatures are retained in each identity."
        ),
    }


def analyze(results_dir):
    primary = {key: load_dataset(results_dir / name)[0] for key, name in PRIMARY.items()}
    comparison = compare_microbenchmarks(primary["q4_decisions"], primary["q8_decisions"])
    developmental = {}
    for root in sorted(results_dir.iterdir()):
        if root.is_dir() and root.name not in PRIMARY.values() and (root / "model-metadata.json").is_file():
            try:
                derived, _events = load_dataset(root)
                developmental[root.name] = derived
            except AnalysisError as exc:
                developmental[root.name] = {"status": "incomplete", "reason": str(exc)}
    return {
        "schema_version": 1, "analysis_source_sha256": file_sha256(Path(__file__)),
        "summary_source_sha256": file_sha256(Path(__file__).with_name("local_benchmark.py")),
        "primary": primary, "fixed_input_q4_q8_comparison": comparison, "developmental": developmental,
        "analysis_scope": "Offline recomputation from original runs; failures retained; no model/browser execution.",
        "historical_metric_notice": (
            "Developmental summaries used native_forward/prefill labels for adapter latency. All metrics here are "
            "recomputed with adapter-scoring scope; original raw records and historical summaries remain unchanged."
        ),
        "percentile_scope": "Linear interpolation at (n-1)*q; p95 is descriptive with n=10 or smaller.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(args.results_dir)
        encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    except (AnalysisError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(json.dumps({"written": args.output.name, "comparison": result["fixed_input_q4_q8_comparison"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
