"""Offline checks for request identity and byte-accurate native RSS analysis."""

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "analyze_local_results", Path(__file__).resolve().parents[1] / "scripts/analyze_local_results.py"
)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def decision_request():
    return {
        "state": {"title": "里斯本", "text": "Choose a hotel"},
        "questions": [{
            "id": "operation",
            "choices": {"CLICK": "Open hotel", "DONE": "Requirements satisfied"},
        }],
    }


def test_wire_request_hash_matches_compact_utf8_and_equal_clone():
    request = decision_request()
    expected = hashlib.sha256(
        json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert analysis.wire_request_sha256(request) == expected
    assert analysis.wire_request_sha256(deepcopy(request)) == expected


def test_wire_request_hash_preserves_candidate_order():
    request = decision_request()
    reordered = deepcopy(request)
    choices = reordered["questions"][0]["choices"]
    reordered["questions"][0]["choices"] = dict(reversed(list(choices.items())))
    assert analysis.wire_request_sha256(request) != analysis.wire_request_sha256(reordered)


def test_native_peak_rss_is_reported_in_bytes_without_kib_conversion(tmp_path):
    log = tmp_path / "native.log"
    nine_gib = 9 * 1024**3
    log.write_text(
        f"        {8 * 1024**3}  maximum resident set size\n"
        f"        {nine_gib}  maximum resident set size\n",
        encoding="utf-8",
    )
    result = analysis.native_peak_rss(log)
    assert result["peak_rss_bytes"] == nine_gib
    assert result["n_reports"] == 2
    assert isinstance(result["scope"], str) and result["scope"]


def test_native_peak_rss_rejects_log_without_a_time_report(tmp_path):
    log = tmp_path / "native.log"
    log.write_text("native model initialization completed\n", encoding="utf-8")
    with pytest.raises(analysis.AnalysisError):
        analysis.native_peak_rss(log)
