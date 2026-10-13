"""Offline checks for honest measurement boundaries and percentile arithmetic."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "local_benchmark", Path(__file__).resolve().parents[1] / "scripts/local_benchmark.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_percentiles_interpolate_and_report_small_sample():
    stats = benchmark.distribution([10, 30, 20, 40, 50])
    assert stats["n"] == 5
    assert stats["p50"] == 30
    assert stats["p95"] == pytest.approx(48)
    assert benchmark.distribution([])["p95"] is None


def test_summary_retains_failed_timings_excludes_warmups_and_never_estimates_tokens():
    summary = benchmark.summarize([
        {"kind": "decision_microbenchmark", "phase": "warmup", "ok": True,
         "python_peak_rss_bytes_cumulative": 1,
         "model_calls": [{"kind": "decision", "ok": True, "wall_ms": 900}]},
        {"kind": "decision_microbenchmark", "phase": "measured", "ok": False,
         "python_peak_rss_bytes_cumulative": 2,
         "model_calls": [{"kind": "decision", "ok": False, "wall_ms": 100}]},
        {"kind": "helper_microbenchmark", "phase": "measured", "ok": True,
         "python_peak_rss_bytes_cumulative": 3,
         "model_calls": [{"kind": "text_helper", "ok": True, "wall_ms": 250,
                          "result": ["Lisbon", {"ttft_ms": 40, "usage": {}}]}]},
    ])
    assert summary["decision_wall_ms_all_attempts"] == benchmark.distribution([100])
    assert summary["decision_wall_ms_successful"]["n"] == 0
    assert summary["decision_failed_calls"] == 1
    assert summary["helper_first_content_delta_ms"]["p50"] == 40
    assert summary["helper_provider_completion_tokens_per_full_request_second"]["n"] == 0


@pytest.mark.parametrize("url", [
    "https://api.typesafe.ai/v1/systemone", "http://localhost:8000/decide",
    "http://127.0.0.1:8000/decide?key=x", "http://user@127.0.0.1:8000/decide",
])
def test_benchmark_rejects_remote_ambiguous_or_credential_urls(url):
    with pytest.raises(ValueError):
        benchmark.local_endpoint(url, "endpoint")


def test_browser_wrapper_counts_internal_work_once():
    class InnerBrowser:
        def act(self, *_args):
            return self.fresh()

        def fresh(self):
            return True

    events = []
    browser = benchmark.TimedBrowser(InnerBrowser(), events)
    browser.phase = "task"
    assert browser.act({}, {}) is True
    assert [event["method"] for event in events] == ["act"]


def test_distinguishes_adapter_input_scoring_and_text_generation_throughput():
    summary = benchmark.summarize([
        {"kind": "decision_microbenchmark", "phase": "measured", "ok": True,
         "python_peak_rss_bytes_cumulative": 1,
         "model_calls": [{"kind": "decision", "ok": True, "wall_ms": 300,
                          "result": {"model_latency_ms": 250, "native_token_count": 1000}}]},
        {"kind": "helper_microbenchmark", "phase": "measured", "ok": True,
         "python_peak_rss_bytes_cumulative": 1,
         "model_calls": [{"kind": "text_helper", "ok": True, "wall_ms": 250,
                          "result": ["Lisbon", {"usage": {"completion_tokens": 5}}]}]},
    ])
    assert summary["decision_input_tokens_per_scoring_second"]["p50"] == 4000
    assert summary["decision_adapter_scoring_ms_successful"]["p50"] == 250
    assert "decision_native_forward_ms_successful" not in summary
    assert summary["helper_provider_completion_tokens_per_full_request_second"]["p50"] == 20


def test_recorder_retains_late_failure_native_prefix_and_unsent_budget_rejection():
    native = [
        {"transport_attempted": True, "model_execution_confirmed": True, "ok": True,
         "phase": "operation", "client_wall_ms": 10},
        {"transport_attempted": False, "model_execution_confirmed": False, "ok": False,
         "phase": "target", "client_wall_ms": 0},
    ]

    class FailedPolicy(RuntimeError):
        native_calls = native
        budget_exhausted = True

    budget = benchmark.NativeRequestBudget(1)
    recorder = benchmark.ModelRecorder("measured", native_budget=budget)

    def fail(*_args):
        raise FailedPolicy("Budget reached after a real successful prefix")

    recorder.real_choose = fail
    with pytest.raises(FailedPolicy):
        recorder.choose({}, "Original goal", [])
    assert recorder.calls[0]["ok"] is False
    assert recorder.calls[0]["native_calls"] == native
    assert recorder.calls[0]["native_budget_exhausted"] is True
    assert budget.used == 1 and budget.remaining == 0
    native[0]["phase"] = "mutated after capture"
    assert recorder.calls[0]["native_calls"][0]["phase"] == "operation"


def test_candidate_identity_requires_real_metadata_and_never_guesses_model_or_calibration():
    assert benchmark.candidate_identity({"decision_service_identity": {
        "model_file": "Verified training artifact", "temperature": 3.25,
    }}) == {"S1_EXPECTED_MODEL": "Verified training artifact", "S1_EXPECTED_TEMPERATURE": "3.25"}
    with pytest.raises(ValueError):
        benchmark.candidate_identity({})
    with pytest.raises(ValueError):
        benchmark.candidate_identity({"decision_service_identity": {"model_file": "Artifact", "temperature": None}})


def test_serial_remaining_native_budget_is_forwarded_without_reserving_unused_calls(monkeypatch):
    from jev_ultrafast import model

    remaining = []

    def candidate(_page, _goal, _history, *, max_native_calls):
        remaining.append(max_native_calls)
        count = min(2, max_native_calls)
        return {"native_calls": [{"transport_attempted": True, "model_execution_confirmed": True,
                                  "ok": True, "client_wall_ms": 1} for _ in range(count)]}

    monkeypatch.setattr(model, "choose_noul", candidate, raising=False)
    budget = benchmark.NativeRequestBudget(3)
    recorder = benchmark.ModelRecorder("measured", "noul_v1", budget)
    recorder.choose({}, "Original goal", [])
    recorder.choose({}, "Original goal", [])
    assert remaining == [3, 1]
    assert budget.used == 3 and budget.remaining == 0
    with pytest.raises(benchmark.NativeRequestBudgetError):
        recorder.choose({}, "Original goal", [])
    assert remaining == [3, 1]
    assert recorder.calls[-1]["native_calls"] == []


@pytest.mark.parametrize("status,choice,stopped", [
    ("blocked", "BLOCKED", True), ("blocked", "e1", True),
    ("done", "e1", True), ("done", "DONE", False),
])
def test_matching_development_dom_cannot_pass_blocked_or_automatic_stop(status, choice, stopped):
    result = {"verified": True, "stopped_cleanly": stopped,
              "agent_state": {"status": status, "decisions": [{"choice": choice}]}}
    assert benchmark.development_task_success(result) is False


def test_genuine_done_and_independent_matching_dom_pass_without_errors():
    result = {"verified": True, "stopped_cleanly": True,
              "agent_state": {"status": "done", "decisions": [{"choice": "DONE"}]}}
    assert benchmark.development_task_success(result) is True
    for error in ("error", "verification_error", "teardown_error"):
        assert benchmark.development_task_success({**result, error: {"type": "Failure"}}) is False
