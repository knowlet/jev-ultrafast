"""Offline runner contracts; fake agents never measure model/browser performance."""

import json
import os
import signal
import socket
import sys
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_heldout_benchmark as runner

PLAN = json.loads((runner.REPO / "benchmark/heldout-plan.json").read_text())


@pytest.fixture(autouse=True)
def prohibit_live_execution(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Offline heldout tests must not start a process, bind a socket, or access the network")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(runner, "ThreadingHTTPServer", forbidden)
    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    monkeypatch.setattr(runner.subprocess, "check_output", forbidden)


@pytest.fixture
def frozen_copy(tmp_path, monkeypatch):
    original_repo = runner.REPO
    repository = tmp_path / "repository"
    registry = original_repo / "benchmark/heldout-freeze.json"
    freeze = json.loads(registry.read_text())
    for relative in freeze["files"]:
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original_repo / relative).read_bytes())
    freeze_path = repository / "benchmark/heldout-freeze.json"
    freeze_path.write_bytes(registry.read_bytes())
    monkeypatch.setattr(runner, "REPO", repository)
    return repository, repository / "benchmark/heldout-plan.json", freeze_path


def cli_arguments(repository):
    metadata = repository / "model-metadata.json"
    metadata.write_text('{}\n')
    return [
        "--output", str(repository / "results"), "--model-metadata", str(metadata),
        "--chrome-path", str(Path(sys.executable).resolve()),
        "--decision-url", "http://127.0.0.1:18001/decide",
        "--text-base-url", "http://127.0.0.1:18002/v1", "--text-model", "offline-test-model",
    ]


def test_registered_sources_and_rendered_html_match_both_frozen_anchors(frozen_copy):
    _repository, manifest, freeze = frozen_copy
    plan, registry = runner.verify_freeze(manifest, freeze)
    assert len(plan["tasks"]) == 8
    assert len(registry["files"]) == 12


@pytest.mark.parametrize("relative", [
    "benchmark/heldout-plan.json", "benchmark/heldout_fixture.py",
    "benchmark/heldout-fixtures/unknown_code.html", "tests/test_heldout_fixture.py",
])
def test_any_registered_content_change_rejects_execution(frozen_copy, relative):
    repository, manifest, freeze = frozen_copy
    with (repository / relative).open("ab") as stream:
        stream.write(b"\nchanged after freeze\n")
    with pytest.raises(ValueError, match="frozen|Frozen"):
        runner.verify_freeze(manifest, freeze)


def test_registry_cannot_be_rewritten_to_bless_a_changed_fixture(frozen_copy):
    repository, manifest, freeze_path = frozen_copy
    relative = "benchmark/heldout-fixtures/unknown_code.html"
    (repository / relative).write_text("Changed local fixture")
    freeze = json.loads(freeze_path.read_text())
    freeze["files"][relative] = runner.sha256(repository / relative)
    freeze_path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="Freeze registry differs"):
        runner.verify_freeze(manifest, freeze_path)


def test_renderer_mismatch_is_detected_even_when_registered_bytes_are_unchanged(frozen_copy, monkeypatch):
    _repository, manifest, freeze = frozen_copy
    monkeypatch.setattr(runner, "render_case", lambda _case: "Different imported renderer output")
    with pytest.raises(ValueError, match="Rendered fixture differs"):
        runner.verify_freeze(manifest, freeze)


def test_schedule_scores_each_of_eight_cases_three_times_in_the_frozen_order():
    scheduled = runner.schedule(PLAN, "baseline")
    assert len(scheduled) == 24
    assert Counter(run["task_id"] for run in scheduled) == dict.fromkeys(PLAN["task_order"], 3)
    assert len({(run["task_id"], run["repetition"]) for run in scheduled}) == 24
    for repetition in (0, 1, 2):
        assert [run["task_id"] for run in scheduled if run["repetition"] == repetition] == PLAN["task_order"]
    assert {run["policy"] for run in scheduled} == {"baseline"}


def test_only_preserved_baseline_and_explicit_candidate_registry_can_be_resolved(monkeypatch):
    from jev_ultrafast import model

    def candidate(*_args):
        return None
    monkeypatch.setattr(model, "choose_noul", candidate, raising=False)
    assert runner.APPROVED_POLICIES == {"baseline": "choose", "noul_v1": "choose_noul"}
    assert runner.resolve_policy("baseline") is model.choose
    assert runner.resolve_policy("noul_v1") is candidate
    for name in ("candidate", "os:system", "/tmp/arbitrary_policy.py"):
        with pytest.raises(ValueError, match="fixed and approved"):
            runner.resolve_policy(name)


@pytest.mark.parametrize("override", [
    ["--policy", "arbitrary"], ["--policy-path", "/tmp/policy.py"], ["--task", "unknown_code"],
    ["--repeats", "1"], ["--max-decisions", "1"], ["--max-actions", "1"],
    ["--seed", "17"], ["--temperature", "0.5"], ["--task-timeout", "1"],
])
def test_cli_rejects_unapproved_policy_task_or_budget_overrides(frozen_copy, override):
    repository, _manifest, _freeze = frozen_copy
    with pytest.raises(SystemExit) as error:
        runner.parser().parse_args(cli_arguments(repository) + override)
    assert error.value.code == 2


def test_execute_requires_an_explicit_opt_in(frozen_copy):
    repository, _manifest, _freeze = frozen_copy
    arguments = cli_arguments(repository)
    assert runner.parser().parse_args(arguments).execute is False
    assert runner.parser().parse_args(arguments).policy == "baseline"
    assert runner.parser().parse_args(arguments + ["--policy", "noul_v1"]).execute is False
    assert runner.parser().parse_args(arguments + ["--execute"]).execute is True


def test_plan_only_writes_all_runs_without_resolving_policy_or_starting_execution(frozen_copy, monkeypatch):
    repository, _manifest, _freeze = frozen_copy

    def forbidden(*_args, **_kwargs):
        pytest.fail("Plan-only execution must stop before policy/provider/browser/server work")

    for name in ("resolve_policy", "local_configuration", "fixture_server", "warmup", "run_case"):
        monkeypatch.setattr(runner, name, forbidden)
    monkeypatch.setattr(runner, "provenance", lambda *_args: {"offline_test": True})
    monkeypatch.setattr(sys, "argv", ["run_heldout_benchmark", *cli_arguments(repository)])
    assert runner.main() == 0
    output = repository / "results"
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["execution_requested"] is False
    assert json.loads((output / "execution-plan.json").read_text()) == runner.schedule(PLAN, "baseline")
    assert not (output / "runs.jsonl").exists()
    assert not (output / "summary.json").exists()


def test_freeze_mismatch_stops_before_output_or_execution(frozen_copy, monkeypatch):
    repository, _manifest, _freeze = frozen_copy
    (repository / "benchmark/heldout-fixtures/unknown_code.html").write_text("Changed")
    monkeypatch.setattr(sys, "argv", ["run_heldout_benchmark", *cli_arguments(repository), "--execute"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not (repository / "results").exists()


def summary_record(case, *, successful, task_ms, budget=False, decision_ok=True, wall_ms=10):
    call = {"kind": "decision", "ok": decision_ok, "wall_ms": wall_ms}
    if decision_ok:
        call["result"] = {}
    return {
        "task_id": case["id"], "family": case["family"], "successful_task": successful,
        "task_ms": task_ms, "total_ms": 40, "executed_actions": 0,
        "decision_calls": 1, "text_calls": 0, "budget_exhausted": budget,
        "model_wall_ms": wall_ms, "browser_task_wall_ms": 3,
        "model_calls": [call],
    }


def test_summary_keeps_failures_missing_runs_and_all_eight_task_denominators():
    first, second = PLAN["tasks"][:2]
    summary = runner.summarize([
        summary_record(first, successful=True, task_ms=10),
        summary_record(first, successful=False, task_ms=30, budget=True, decision_ok=False, wall_ms=90),
        summary_record(second, successful=False, task_ms=None, decision_ok=False, wall_ms=50),
    ], PLAN, "baseline", batch_error={"type": "Interrupted", "message": "Stopped early"})
    assert summary["expected_runs"] == 24
    assert summary["executed_runs"] == 3
    assert summary["unexecuted_runs"] == 21
    assert summary["failed_runs"] == 2
    assert summary["all_task_success_rate"] == pytest.approx(1 / 24)
    assert summary["complete"] is False
    assert summary["budget_exhaustions"] == 1
    assert summary["decision_wall_ms_all_attempts"] == runner.distribution([10, 90, 50])
    assert summary["task_ms_all_runs"] == runner.distribution([10, 30])
    assert set(summary["by_task"]) == {case["id"] for case in PLAN["tasks"]}
    first_stats = summary["by_task"][first["id"]]
    assert first_stats["expected_runs"] == 3
    assert first_stats["unexecuted_runs"] == 1
    assert first_stats["success_rate"] == pytest.approx(1 / 3)
    assert first_stats["complete"] is False
    for case in PLAN["tasks"][2:]:
        unexecuted = summary["by_task"][case["id"]]
        assert unexecuted["expected_runs"] == unexecuted["unexecuted_runs"] == 3
        assert unexecuted["success_rate"] == 0
        assert unexecuted["complete"] is False
    assert summary["by_family"]["completion"]["expected_runs"] == 6
    assert summary["by_family"]["completion"]["success_rate"] == pytest.approx(1 / 6)


def test_summary_distinguishes_complete_execution_from_complete_success():
    records = [summary_record(case, successful=False, task_ms=1) for case in PLAN["tasks"] for _ in range(3)]
    summary = runner.summarize(records, PLAN, "baseline")
    assert summary["complete"] is True
    assert summary["failed_runs"] == 24
    assert summary["all_task_success_rate"] == 0
    assert summary["unexecuted_runs"] == 0
    assert all(item["complete"] and item["success_rate"] == 0 for item in summary["by_task"].values())
    assert runner.summarize(records, PLAN, "baseline", batch_error={"type": "CleanupError"})["complete"] is False


def test_summary_requires_provider_token_counts_and_keeps_memory_scopes_separate():
    record = summary_record(PLAN["tasks"][0], successful=False, task_ms=1)
    record["model_calls"] = [
        {"kind": "decision", "ok": True, "wall_ms": 400,
         "result": {"native_token_count": 1000, "model_latency_ms": 250}},
        {"kind": "text_helper", "ok": True, "wall_ms": 500,
         "result": ["Generated", {"ttft_ms": 30, "usage": {"completion_tokens": 5}}]},
        {"kind": "text_helper", "ok": True, "wall_ms": 900,
         "result": ["Uncounted text", {"usage": {}}]},
    ]
    record["chromium_resource_usage"] = {"peak_rss_bytes": 5000}
    record["python_peak_rss_bytes_cumulative"] = 9000
    summary = runner.summarize([record], PLAN, "baseline")
    assert summary["helper_first_content_delta_ms"] == runner.distribution([30])
    assert summary["helper_reported_completion_tokens_per_full_request_second"] == runner.distribution([10])
    assert summary["decision_input_tokens_per_adapter_scoring_second"] == runner.distribution([4000])
    assert summary["decision_ttft_ms"] is None
    assert summary["chromium_owned_main_peak_rss_bytes"] == runner.distribution([5000])
    assert summary["python_peak_rss_bytes_cumulative"] == 9000
    assert "not process-tree total" in summary["memory_scope"]


def test_summary_distinguishes_multicall_policy_cycles_physical_requests_and_parent_wall():
    record = summary_record(PLAN["tasks"][0], successful=False, task_ms=80)
    record["native_scoring_requests"] = 2
    record["model_wall_ms"] = 80
    record["model_calls"] = [{
        "kind": "decision", "ok": False, "wall_ms": 80,
        "native_calls": [
            {"transport_attempted": True, "model_execution_confirmed": True, "ok": True,
             "phase": "operation", "client_wall_ms": 20, "native_token_count": 100, "model_latency_ms": 10},
            {"transport_attempted": True, "model_execution_confirmed": False, "ok": False,
             "phase": "target", "client_wall_ms": 40},
            {"transport_attempted": False, "model_execution_confirmed": False, "ok": False,
             "phase": "target", "client_wall_ms": 0},
        ],
    }]
    summary = runner.summarize([record], PLAN, "noul_v1")
    assert summary["logical_decision_cycles"] == 1
    assert summary["decision_requests"] == summary["native_scoring_requests"] == 2
    assert summary["native_model_executions_confirmed"] == 1
    assert summary["native_scoring_client_wall_ms_all_attempts"] == runner.distribution([20, 40])
    assert summary["decision_input_tokens_per_adapter_scoring_second"] == runner.distribution([10000])
    assert summary["model_wall_ms_per_task"] == runner.distribution([80])


def test_frozen_helper_seed_is_both_sent_and_recorded_and_configuration_is_restored(monkeypatch):
    from jev_ultrafast import model

    requests = []

    def fake_stream(url, body):
        requests.append((url, deepcopy(body)))
        return {"choices": [{"message": {"content": '{"text":"Generated field value"}'}}]}, 3

    args = SimpleNamespace(decision_url="http://127.0.0.1:18001/decide",
                           text_base_url="http://127.0.0.1:18002/v1", text_model="offline-test-model")
    monkeypatch.setattr(model, "stream_local_text", fake_stream)
    monkeypatch.setenv("S1_DECISION_URL", "previous-decision-value")
    monkeypatch.delenv("S1_TEXT_BASE_URL", raising=False)
    monkeypatch.setenv("TEXT_MODEL", "previous-model-value")
    with runner.local_configuration(args, PLAN):
        recorder = runner.ModelRecorder("measured")
        value, metadata = recorder.text({"goal": "Use the goal value", "field": {"label": "Title"}})
        assert value == "Generated field value"
        assert requests[0][1]["seed"] == PLAN["model_seed"] == 42
        assert requests[0][1]["temperature"] == PLAN["temperature"] == 0
        assert metadata["settings"]["seed"] == 42
        assert recorder.calls[0]["result"][1]["settings"]["seed"] == 42
    assert model.stream_local_text is fake_stream
    assert os.environ["S1_DECISION_URL"] == "previous-decision-value"
    assert "S1_TEXT_BASE_URL" not in os.environ
    assert os.environ["TEXT_MODEL"] == "previous-model-value"


def observation_for(case):
    expected = case["acceptance"]
    observed = {key: value for key, value in expected.items()
                if key not in {"agent_status", "minimum_submissions", "applied_settings_match"}}
    observed["case_id"] = case["id"]
    if "minimum_submissions" in expected:
        observed["submissions"] = expected["minimum_submissions"]
    if expected.get("applied_settings_match"):
        observed["applied"] = {key: expected[key] for key in ("field", "mode", "checked")}
    return observed


@pytest.mark.parametrize("task_id,status,terminal_choice,actions,execution_error,expected_success", [
    ("unknown_code", "blocked", "e1", 3, False, False),
    ("unknown_code", "blocked", "BLOCKED", 1, False, False),
    ("unknown_code", "blocked", "BLOCKED", 0, False, True),
    ("complete_stamp", "done", "DONE", 1, False, True),
    ("complete_stamp", "done", "DONE", 1, True, False),
])
def test_run_case_forwards_actual_terminal_choice_and_history_only_after_policy_stops(
    monkeypatch, tmp_path, task_id, status, terminal_choice, actions, execution_error, expected_success,
):
    import jev_ultrafast
    from jev_ultrafast import agent as agent_module
    from jev_ultrafast import isolated_browser

    case = next(case for case in PLAN["tasks"] if case["id"] == task_id)
    observed, events, checker_arguments, model_inputs = observation_for(case), [], [], []
    original_choose = agent_module.choose
    original_verify = runner.verify_case

    class FakeBrowser:
        def evaluate(self, expression):
            assert expression == runner.READ_FRESH_DOM
            assert agent_module.choose is original_choose
            events.append("fresh_checker_read")
            return deepcopy(observed)

        def close(self):
            events.append("browser_close")

    class FakeTransport:
        blocked_requests = 0

        def __init__(self):
            self.calls = []

        def __call__(self, method):
            assert method == "Browser.getVersion"
            self.calls.append({"method": method})
            return {"product": "Offline fake; no Chromium process"}

    class FakeChromium:
        def __init__(self, **kwargs):
            assert kwargs["fixture_origin"] == "http://127.0.0.1:18003"
            self.transport, self.proxy = FakeTransport(), None
            self.resource_usage, self.startup_ms = {}, 0

        def start(self):
            events.append("fake_chromium_start")

        def browser(self, _url):
            return FakeBrowser()

        def close(self):
            events.append("fake_chromium_close")

    class FakeAgent:
        def __init__(self, url, goal, *, screenshots, browser_factory):
            assert goal == case["goal"]
            assert screenshots is False
            self.goal, self.browser = goal, browser_factory(url)
            self.state = {"status": "ready", "decisions": [], "history": [{} for _ in range(actions)]}

        def command(self, command):
            assert command == "tick"
            events.append("policy_tick")
            result = agent_module.choose({"text": "Observed UI only"}, self.goal, deepcopy(self.state["history"]))
            self.state["decisions"].append(result)
            self.state["status"] = status
            if execution_error:
                raise RuntimeError("Recorded execution failure")

        def snapshot(self):
            return deepcopy(self.state)

        def close(self):
            self.browser.close()

    def choose(page, goal, history):
        events.append("model_call")
        model_inputs.append((deepcopy(page), goal, deepcopy(history)))
        return {"choice": terminal_choice}

    def verify(case_argument, dom, **kwargs):
        events.append("verify_case")
        assert agent_module.choose is original_choose
        checker_arguments.append(kwargs)
        return original_verify(case_argument, dom, **kwargs)

    monkeypatch.setattr(jev_ultrafast, "Agent", FakeAgent)
    monkeypatch.setattr(isolated_browser, "IsolatedChromium", FakeChromium)
    monkeypatch.setattr(runner, "deadline", lambda _seconds: nullcontext())
    monkeypatch.setattr(runner, "verify_case", verify)
    monkeypatch.setattr(runner, "peak_python_rss_bytes", lambda: 0)
    args = SimpleNamespace(policy="baseline", chrome_path=Path("/offline/no-executable"), output=tmp_path)
    result = runner.run_case(case, 0, args, PLAN, "http://127.0.0.1:18003", choose)
    assert result["terminal_choice"] == checker_arguments[0]["terminal_choice"] == terminal_choice
    assert result["executed_actions"] == checker_arguments[0]["executed_actions"] == actions
    assert checker_arguments[0]["agent_status"] == status
    assert bool(checker_arguments[0]["execution_error"]) is execution_error
    assert result["successful_task"] is expected_success
    assert result["decision_calls"] == 1
    assert len(model_inputs) == 1
    assert model_inputs[0] == ({"text": "Observed UI only"}, case["goal"], [{} for _ in range(actions)])
    assert events.index("model_call") < events.index("fresh_checker_read") < events.index("verify_case")
    assert events[-1] == "fake_chromium_close"


@pytest.mark.parametrize('status', [401, 403, 503])
def test_actual_provider_denial_or_unavailability_stops_without_scanning_page(status):
    record = {'goal': 'Permission denied appears on a synthetic page', 'model_calls': [
        {'kind': 'decision', 'native_calls': [{'http_status': status, 'raw_response': None}]}]}
    assert runner.batch_stop_reason(record)['type'] in ('PolicyDenial', 'NativeServiceUnavailable')
    benign = {'goal': 'Permission denied', 'agent_state': {'status': 'blocked'},
              'model_calls': [{'kind': 'decision', 'input': {'page': {'text': 'Operation not permitted'}},
                               'native_calls': [{'http_status': 200, 'raw_response': {'status': 'ok'}}]}]}
    assert runner.batch_stop_reason(benign) is None


@pytest.mark.parametrize('status', [403, 503])
def test_actual_helper_response_error_stops_warmups_immediately(monkeypatch, status):
    calls = []
    class Recorder:
        def __init__(self, *_args):
            self.calls = []
        def choose(self, *_args):
            calls.append('choose')
        def text(self, *_args):
            calls.append('text')
            raise RuntimeError(f'Model provider returned HTTP {status}; no action executed.')
    monkeypatch.setattr(runner, 'ModelRecorder', Recorder)
    rows = runner.warmup(PLAN, lambda *_: None, policy='noul_v1')
    assert len(rows) == 1 and calls == ['choose', 'text']
    assert rows[0]['batch_stop_reason']['type'] in ('PolicyDenial', 'NativeServiceUnavailable')


def test_external_signal_handler_uses_ordinary_budget_error_and_ignores_second_signal():
    with runner.window_abort_handlers() as state:
        handler = signal.getsignal(signal.SIGINT)
        with pytest.raises(runner.BudgetTimeout, match='External heldout window'):
            handler(signal.SIGINT, None)
        assert state['requested'] and state['signal'] == signal.SIGINT
        handler(signal.SIGINT, None)


def test_interrupted_current_task_is_saved_once_budget_preserved_and24_denominator(frozen_copy, monkeypatch):
    repository, _manifest, _freeze = frozen_copy
    monkeypatch.setattr(runner, 'provenance', lambda *_: {'source_sha256': {}})
    monkeypatch.setattr(runner, 'resolve_policy', lambda *_: lambda *_: None)
    monkeypatch.setattr(runner, 'local_configuration', lambda *_: nullcontext())
    monkeypatch.setattr(runner, 'fixture_server', lambda *_: nullcontext('http://127.0.0.1:18003'))
    monkeypatch.setattr(runner, 'warmup', lambda *_args, **_kwargs: [])
    seen = []
    def interrupted(case, repetition, args, plan, origin, choose, *, native_budget):
        seen.append(case['id'])
        physical = [{'transport_attempted': True, 'model_execution_confirmed': True, 'ok': True},
                    {'transport_attempted': True, 'model_execution_confirmed': False, 'ok': False}]
        native_budget.account(physical)
        record = summary_record(case, successful=False, task_ms=10, decision_ok=False)
        record['model_calls'][0]['native_calls'] = physical
        record['native_scoring_requests'] = 2
        try:
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        except runner.BudgetTimeout as exc:
            record['error'] = runner.error_record(exc)
        return record
    monkeypatch.setattr(runner, 'run_case', interrupted)
    monkeypatch.setattr(sys, 'argv', ['heldout', *cli_arguments(repository), '--execute', '--policy', 'noul_v1'])
    assert runner.main() == 1 and seen == [PLAN['task_order'][0]]
    output = repository / 'results'
    rows = [json.loads(line) for line in (output / 'runs.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['window_interrupted'] and not rows[0]['successful_task']
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['expected_runs'] == 24 and summary['executed_runs'] == 1 and summary['unexecuted_runs'] == 23
    assert summary['native_scoring_requests'] == 2 and not summary['complete']
    metadata = json.loads((output / 'metadata.json').read_text())
    assert metadata['measured_native_budget'] == {'limit': 288, 'transport_attempts': 2,
                                                 'remaining': 286, 'warmups_included': False}
