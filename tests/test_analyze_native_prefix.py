"""Offline reproduction of real archived failures and fail-closed corruption checks."""

import importlib.util
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "analyze_native_prefix", Path(__file__).resolve().parents[1] / "scripts/analyze_native_prefix.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
SOURCE = audit.DEFAULT_SOURCE


@pytest.fixture(scope="module")
def packet():
    parity = SOURCE / "parity"
    return {
        "prepared": audit.read_rows(parity / "prepared.jsonl"),
        "native": {mode: audit.read_rows(parity / mode / "stdout.log") for mode in audit.MODES},
        "records": {mode: audit.read_rows(parity / mode / "records.jsonl") for mode in audit.MODES},
        "transport": {mode: audit.read_rows(parity / mode / "transport.jsonl") for mode in audit.MODES},
        "raw_lines": {mode: (parity / mode / "stdout.log").read_bytes().splitlines() for mode in audit.MODES},
        "identity": json.loads((parity / "manifest.json").read_text())["identity"],
    }


def compare(packet, candidate, *, prepared=None, baseline=None, temperature=None):
    return audit.compare_native_responses(
        packet["prepared"] if prepared is None else prepared,
        packet["native"]["original"] if baseline is None else baseline,
        candidate,
        temperature=packet["identity"]["temperature"] if temperature is None else temperature,
    )


def check_mode(packet, mode="prefix_on"):
    return audit.validate_mode(
        packet["prepared"], packet["native"][mode], packet["records"][mode], packet["transport"][mode],
        packet["raw_lines"][mode], mode=mode, identity=packet["identity"],
    )


def test_real_archive_reproduces_failed_gate_without_reading_saved_verdict(monkeypatch):
    parse = audit.parse_json

    def no_saved_summary(data):
        value = parse(data)
        assert not (isinstance(value, dict) and ("parity_pass" in value or "fixed_gate" in value))
        return value

    monkeypatch.setattr(audit, "parse_json", no_saved_summary)
    result = audit.analyze_archive()
    assert result["archive_integrity"]["verified_files"] == 28
    assert result["historical_verdict_reproduced"]
    assert result["historical_numerical_acceptance"] is False
    assert result["historical_acceptance"] is False
    assert result["confirmed_historical_responses"] == 24
    assert result["comparisons"]["prefix_off"]["numerical_parity_pass"]
    on = result["comparisons"]["prefix_on"]["comparisons"]
    assert [r["index"] for r in on if not r["within_fixed_tolerance"]] == [1, 3, 5, 7]
    assert [on[i]["max_raw_abs_delta"] for i in [1, 3, 5, 7]] == pytest.approx([
        0.0065937042236328125, 0.0086212158203125, 0.009721755981445312, 0.017385482788085938,
    ], abs=1e-14)
    assert [on[i]["max_probability_abs_delta"] for i in [1, 3, 5, 7]] == pytest.approx([
        0.0000964140119810164, 0.00003449753789164234, 0.00027493166967917926, 0.0006715392832572775,
    ], abs=1e-14)
    assert result["ranking_agreement"]
    assert all(row["labels_match"] for row in on)
    assert all(row["option_rankings_match"] for row in on)
    assert result["maximum_typed_probability_recalculation_delta"] < 1e-15
    assert result["new_native_calls"] == result["new_browser_tasks"] == result["new_jevbench_runs"] == 0
    assert result["browser_benchmark_allowed"] is False
    assert result["performance_improvement_established"] is False


def test_real_token_geometry_has_conservative_short_prefix_fallback(packet):
    rows = audit.geometry(packet["prepared"], packet["native"]["prefix_on"])
    assert [r["exact_lcp"] for r in rows] == [0, 601, 526, 591, 93, 590, 515, 580]
    assert [r["planned_aligned_retained"] for r in rows] == [0, 512, 512, 512, 0, 512, 512, 512]
    assert [r["planned_aligned_recomputed"] for r in rows] == [601, 89, 79, 79, 590, 78, 68, 68]
    for row in rows:
        assert row["planned_aligned_retained"] <= min(row["exact_lcp"], row["first_requested_slot"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "18.2", True, None])
def test_nonfinite_or_nonnumeric_logits_cannot_pass(packet, value):
    candidate = deepcopy(packet["native"]["prefix_off"])
    candidate[0]["raw_logits"][0][0] = value
    with pytest.raises(audit.AnalysisError):
        compare(packet, candidate)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "3.2"])
def test_invalid_temperature_fails(packet, value):
    with pytest.raises(audit.AnalysisError):
        compare(packet, packet["native"]["prefix_off"], temperature=value)


@pytest.mark.parametrize("change", ["missing", "extra", "swapped", "empty"])
def test_missing_extra_swapped_or_empty_pairs_fail(packet, change):
    candidate = deepcopy(packet["native"]["prefix_off"])
    if change == "missing":
        candidate.pop()
    elif change == "extra":
        candidate.append(deepcopy(candidate[0]))
    elif change == "swapped":
        candidate[0], candidate[1] = candidate[1], candidate[0]
    else:
        with pytest.raises(audit.AnalysisError):
            audit.compare_native_responses([], [], [], temperature=3.2)
        return
    with pytest.raises(audit.AnalysisError):
        compare(packet, candidate)


@pytest.mark.parametrize("change", ["id", "tokens", "slots", "shape", "runtime", "missing_field"])
def test_response_identity_and_shape_are_required(packet, change):
    candidate = deepcopy(packet["native"]["prefix_off"])
    row = candidate[0]
    if change == "id":
        row["question_ids"] = ["wrong"]
    elif change == "tokens":
        row["native_token_count"] -= 1
    elif change == "slots":
        row["runtime_slots"] = [0]
    elif change == "shape":
        row["raw_logits"] = [[]]
    elif change == "runtime":
        row["runtime"]["microbatch_size"] = 256
    else:
        del row["raw_logits"]
    with pytest.raises(audit.AnalysisError):
        compare(packet, candidate)


def test_raw_and_probability_gates_are_independent_and_absolute(packet):
    baseline = deepcopy(packet["native"]["original"])
    baseline[0]["raw_logits"] = [[0.0, 0.0]]
    candidate = deepcopy(baseline)
    candidate[0]["raw_logits"] = [[2e-4, 2e-4]]
    row = compare(packet, candidate, baseline=baseline)["comparisons"][0]
    assert row["max_probability_abs_delta"] == 0
    assert row["within_fixed_tolerance"] is False
    candidate[0]["raw_logits"] = [[-0.99e-4, 0.99e-4]]
    row = compare(packet, candidate, baseline=baseline)["comparisons"][0]
    assert row["max_raw_abs_delta"] < audit.RAW_ATOL
    assert row["max_probability_abs_delta"] > audit.PROBABILITY_ATOL
    assert row["within_fixed_tolerance"] is False
    baseline[0]["raw_logits"] = [[1e6, 1e6]]
    candidate[0]["raw_logits"] = [[1e6 + 0.001, 1e6 + 0.001]]
    assert compare(packet, candidate, baseline=baseline)["comparisons"][0]["within_fixed_tolerance"] is False


def test_nonwinning_near_tie_rank_change_fails_despite_numeric_tolerances(packet):
    # A synthetic three-option payload exercises a failure not covered by binary
    # archived decisions: the winning label remains stable while ranks 2/3 flip.
    prepared = deepcopy(packet["prepared"][:1])
    prepared[0]["nopts"] = prepared[0]["payload"]["nopts"] = [3]
    prepared[0]["payload_sha256"] = audit.wire_hash(prepared[0]["payload"])
    baseline = deepcopy(packet["native"]["original"][:1])
    baseline[0]["raw_logits"] = [[1.0, 0.0, -1e-6]]
    candidate = deepcopy(baseline)
    candidate[0]["raw_logits"] = [[1.0, -1e-6, 0.0]]
    result = compare(packet, candidate, prepared=prepared, baseline=baseline)
    row = result["comparisons"][0]
    assert row["max_raw_abs_delta"] < audit.RAW_ATOL
    assert row["max_probability_abs_delta"] < audit.PROBABILITY_ATOL
    assert row["labels_match"]
    assert row["option_rankings_match"] is False
    assert row["within_fixed_tolerance"] is False
    assert result["numerical_parity_pass"] is False


@pytest.mark.parametrize("change", ["unsorted_slots", "one_option", "too_many_options", "short_letters",
                                    "duplicate_letters", "large_token", "large_slot", "large_letter",
                                    "float_payload_token", "bool_mirror_option"])
def test_prepared_contract_matches_native_bounds_even_with_valid_hash(packet, change):
    prepared = deepcopy(packet["prepared"][:1])
    row = prepared[0]
    payload = row["payload"]
    if change == "unsorted_slots":
        row["slots"] = payload["slots"] = [600, 599]
        row["nopts"] = payload["nopts"] = [2, 2]
        payload["question_ids"] = ["first", "second"]
    elif change in {"one_option", "too_many_options"}:
        row["nopts"] = payload["nopts"] = [1 if change == "one_option" else 53]
    elif change == "short_letters":
        row["letters"] = payload["letters"] = row["letters"][:-1]
    elif change == "duplicate_letters":
        row["letters"][0] = row["letters"][1]
        payload["letters"] = row["letters"].copy()
    elif change == "large_token":
        row["input_ids"][0] = payload["parts"][0]["tokens"][0] = 2**31
    elif change == "large_slot":
        row["slots"] = payload["slots"] = [2**31]
    elif change == "large_letter":
        row["letters"][0] = payload["letters"][0] = 2**31
    elif change == "float_payload_token":
        payload["parts"][0]["tokens"][0] = float(row["input_ids"][0])
    else:
        row["nopts"] = [True]
    row["payload_sha256"] = audit.wire_hash(payload)
    with pytest.raises(audit.AnalysisError):
        audit.validate_prepared(prepared)


@pytest.mark.parametrize("change", ["mode", "probability", "probability_nan", "label", "model", "temperature",
                                    "transport_id", "transport_hash", "transport_order", "missing_response"])
def test_typed_transport_and_model_identity_cannot_be_substituted(packet, change):
    bad = deepcopy(packet)
    row = bad["records"]["prefix_on"][0]
    answer = row["typed_response"]["answers"]["candidate"]
    if change == "mode":
        row["mode"] = "prefix_off"
    elif change == "probability":
        answer["probabilities"]["true"] += 1e-4
    elif change == "probability_nan":
        answer["probabilities"]["true"] = float("nan")
    elif change == "label":
        row["truth_label"] = "false"
    elif change == "model":
        row["typed_response"]["model"] = "unrelated.gguf"
    elif change == "temperature":
        row["typed_response"]["temperature"] = 1
    elif change == "transport_id":
        bad["transport"]["prefix_on"][2]["case_id"] = "wrong"
    elif change == "transport_hash":
        bad["transport"]["prefix_on"][2]["raw_line_sha256"] = "0" * 64
    elif change == "transport_order":
        events = bad["transport"]["prefix_on"]
        events[1], events[2] = events[2], events[1]
    else:
        bad["native"]["prefix_on"].pop()
    with pytest.raises(audit.AnalysisError):
        check_mode(bad)


def test_changed_request_and_duplicate_ids_rejected(packet):
    prepared = deepcopy(packet["prepared"])
    prepared[0]["payload"]["parts"][0]["tokens"][0] += 1
    with pytest.raises(audit.AnalysisError, match="hash"):
        compare(packet, packet["native"]["prefix_off"], prepared=prepared)
    prepared = deepcopy(packet["prepared"])
    prepared[1] = deepcopy(prepared[0])
    with pytest.raises(audit.AnalysisError, match="Duplicate"):
        compare(packet, packet["native"]["prefix_off"], prepared=prepared)


@pytest.mark.parametrize("value", ['{"value":NaN}', '{"value":Infinity}', '{"a":1,"a":2}', '{"a":'])
def test_invalid_json_is_rejected(value):
    with pytest.raises(audit.AnalysisError):
        audit.parse_json(value)


def test_corrupted_public_bytes_fail_before_any_numerical_verdict(tmp_path):
    shutil.copyfile(SOURCE / "sanitization-manifest.json", tmp_path / "sanitization-manifest.json")
    (tmp_path / "build-receipt.json").write_text("{}\n")
    with pytest.raises(audit.AnalysisError, match="Public SHA-256 mismatch"):
        audit.verify_archive(tmp_path)


def test_modified_hash_manifest_cannot_self_certify_corruption(tmp_path):
    manifest = json.loads((SOURCE / "sanitization-manifest.json").read_text())
    manifest["files"] = []
    (tmp_path / "sanitization-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(audit.AnalysisError, match="manifest hash mismatch"):
        audit.verify_archive(tmp_path)


def test_cli_exit_status_distinguishes_audit_from_parity(tmp_path, capsys):
    assert audit.main([]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["exit_status_mode"] == "historical_parity_gate"
    output = tmp_path / "derived.json"
    assert audit.main(["--audit-only", "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(output.read_text())
    assert result["exit_status_mode"] == "historical_reproduction_only"
    assert result["historical_numerical_acceptance"] is False
    before = output.read_bytes()
    assert audit.main(["--audit-only", "--output", str(output)]) == 2
    assert output.read_bytes() == before
    assert "overwrite" in capsys.readouterr().err


def test_cli_refuses_archive_output_and_tolerance_override(capsys):
    assert audit.main(["--output", str(SOURCE / "new-audit.json")]) == 2
    assert "outside archive" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        audit.main(["--raw-atol", "0.1"])
    assert exc.value.code == 2
