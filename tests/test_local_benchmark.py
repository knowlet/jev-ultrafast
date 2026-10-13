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
