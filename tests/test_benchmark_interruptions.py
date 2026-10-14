"""Offline interruption regressions; no model, browser, socket or real signal."""

import json
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent as agent_module
from jev_ultrafast import model
from scripts import local_benchmark as bench


@pytest.fixture
def interrupted_provider(monkeypatch):
    stop = KeyboardInterrupt("synthetic stop")
    calls = []

    def choose(*_args, _trace_calls, **_kwargs):
        calls.append("decision")
        _trace_calls.append({"transport_attempted": True, "model_execution_confirmed": False,
                             "ok": False, "client_wall_ms": 1.0,
                             "error": {"type": type(stop).__name__, "message": str(stop)}})
        raise stop

    def text(*_args):
        calls.append("helper")
        raise stop

    monkeypatch.setattr(bench, "resolve_policy", lambda _name: choose)
    monkeypatch.setattr(model, "field_text", text)
    monkeypatch.setattr(bench, "deadline", lambda _seconds: nullcontext())
    return stop, calls


def args(tmp_path):
    metadata = tmp_path / "model.json"
    metadata.write_text(json.dumps({"decision_service_identity": {
        "model_file": "synthetic-existing-model", "temperature": 2.5,
    }}))
    return SimpleNamespace(policy="noul_v1", fixture="click", fixture_url="http://127.0.0.1:8766",
                           task_timeout=1, max_decisions=2, output=tmp_path / "new-evidence",
                           model_metadata=metadata, mode="decision-only", chrome_path=None,
                           repeats=1, warmup=0, decision_repeats=2, text_repeats=0)


@pytest.mark.parametrize("kind", ["decision_microbenchmark", "helper_microbenchmark"])
def test_micro_stop_flushes_prefix_once_and_preserves_exception(interrupted_provider, tmp_path, kind):
    stop, calls = interrupted_provider
    saved = []
    with pytest.raises(KeyboardInterrupt) as raised:
        bench.micro_run(kind, "measured", 0, args(tmp_path), on_interrupt=saved.append)
    assert raised.value is stop
    assert len(saved) == 1 and len(calls) == 1
    row = saved[0]
    assert row["interrupted"] and not row["ok"]
    assert row["error"]["type"] == "KeyboardInterrupt" and not row["model_calls"][0]["ok"]
    if kind == "decision_microbenchmark":
        assert len(row["model_calls"][0]["native_calls"]) == 1
        assert row["model_calls"][0]["native_calls"][0]["transport_attempted"]


def fake_agent(monkeypatch, stop, stage="task"):
    browser = Mock()

    class Agent:
        def __init__(self, _url, goal, *, browser_factory, **_kwargs):
            self.browser = browser_factory(_url)
            self.goal = goal
            self.state = {"status": "ready", "history": [], "decisions": []}
            if stage == "setup":
                raise stop

        def command(self, _name):
            agent_module.choose(deepcopy(bench.CLICK_PAGE), self.goal, [])

        def snapshot(self):
            return deepcopy(self.state)

        def close(self):
            self.browser.close()

    import jev_ultrafast

    monkeypatch.setattr(jev_ultrafast, "Agent", Agent)
    return Mock(browser=Mock(return_value=browser)), browser


@pytest.mark.parametrize("stage", ["setup", "task"])
def test_browser_stop_retains_partial_attempt_skips_dom_and_closes_owned_tab(
    interrupted_provider, monkeypatch, tmp_path, stage,
):
    stop, calls = interrupted_provider
    chromium, browser = fake_agent(monkeypatch, stop, stage)
    verify = Mock(side_effect=AssertionError("No new DOM verification after stop"))
    monkeypatch.setattr(bench, "verify", verify)
    saved = []
    with pytest.raises(KeyboardInterrupt) as raised:
        bench.browser_run(chromium, args(tmp_path), "measured", 0, on_interrupt=saved.append)
    assert raised.value is stop and len(saved) == 1
    row = saved[0]
    assert row["interrupted"] and not row["successful_task"] and not row["stopped_cleanly"]
    assert row["native_scoring_requests"] == (1 if stage == "task" else 0)
    assert row["decision_calls"] == len(calls)
    verify.assert_not_called()
    browser.close.assert_called_once()


def test_partial_save_failure_does_not_replace_stop_exception():
    stop = SystemExit(7)
    save = Mock(side_effect=OSError("synthetic disk error"))
    with pytest.raises(SystemExit) as raised:
        bench.flush_interrupted_run(stop, {"ok": False}, save)
    assert raised.value is stop and stop.code == 7
    assert stop.__notes__ == ["Partial evidence save failed: OSError"]
    save.assert_called_once()


def test_second_stop_during_partial_save_does_not_replace_first():
    stop = SystemExit(7)
    with pytest.raises(SystemExit) as raised:
        bench.flush_interrupted_run(stop, {"ok": False}, Mock(side_effect=KeyboardInterrupt()))
    assert raised.value is stop
    assert stop.__notes__ == ["Partial evidence save failed: KeyboardInterrupt"]


def test_stop_after_provider_return_is_failed_run_with_completed_call(monkeypatch, tmp_path):
    stop = KeyboardInterrupt("synthetic context stop")

    @contextmanager
    def deadline(_seconds):
        yield
        raise stop

    monkeypatch.setattr(bench, "deadline", deadline)
    monkeypatch.setattr(model, "field_text", lambda *_args: ("Lisbon", {"model": "synthetic"}))
    monkeypatch.setattr(bench, "resolve_policy", lambda _name: Mock())
    saved = []
    with pytest.raises(KeyboardInterrupt) as raised:
        bench.micro_run("helper_microbenchmark", "measured", 0, args(tmp_path), on_interrupt=saved.append)
    assert raised.value is stop and len(saved) == 1
    assert not saved[0]["ok"] and saved[0]["model_calls"][0]["ok"]


@pytest.mark.parametrize("stop", [KeyboardInterrupt("synthetic stop"), SystemExit(7)])
def test_main_writes_one_failed_partial_row_and_summary_then_stops(
    interrupted_provider, monkeypatch, tmp_path, stop,
):
    _old_stop, provider_calls = interrupted_provider

    def choose(*_args, _trace_calls, **_kwargs):
        provider_calls.append("decision")
        _trace_calls.append({"transport_attempted": True, "model_execution_confirmed": False,
                             "ok": False, "client_wall_ms": 1.0})
        raise stop

    options = args(tmp_path)
    monkeypatch.setattr(bench, "resolve_policy", lambda _name: choose)
    monkeypatch.setattr(bench, "parser", lambda: Mock(parse_args=Mock(return_value=options)))
    monkeypatch.setattr(bench, "source_metadata", lambda _path: {"offline_fixture": True})
    for name, value in {"S1_DECISION_URL": "http://127.0.0.1:18775/decide",
                        "S1_EXPECTED_MODEL": "synthetic-existing-model",
                        "S1_EXPECTED_TEMPERATURE": "2.5", "S1_LOCAL_POLICY": "noul_v1"}.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(type(stop)) as raised:
        bench.main()
    assert raised.value is stop and provider_calls == ["decision"]
    rows = [json.loads(line) for line in (options.output / "runs.jsonl").read_text().splitlines()]
    summary = json.loads((options.output / "summary.json").read_text())
    metadata = json.loads((options.output / "metadata.json").read_text())
    assert len(rows) == 1 and rows[0]["interrupted"] and not rows[0]["ok"]
    assert summary["groups"]["decision_microbenchmark"]["runs"] == 1
    assert summary["groups"]["decision_microbenchmark"]["failed_runs"] == 1
    assert summary["native_scoring_requests"] == 1
    assert summary["native_model_executions_confirmed"] == 0
    assert summary["batch_error"]["type"] == type(stop).__name__
    assert metadata["batch_error"] == summary["batch_error"]


def test_final_metadata_save_failure_keeps_stop_and_still_attempts_summary(
    interrupted_provider, monkeypatch, tmp_path,
):
    stop, _calls = interrupted_provider
    options = args(tmp_path)
    monkeypatch.setattr(bench, "parser", lambda: Mock(parse_args=Mock(return_value=options)))
    monkeypatch.setattr(bench, "source_metadata", lambda _path: {"offline_fixture": True})
    for name, value in {"S1_DECISION_URL": "http://127.0.0.1:18775/decide",
                        "S1_EXPECTED_MODEL": "synthetic-existing-model",
                        "S1_EXPECTED_TEMPERATURE": "2.5", "S1_LOCAL_POLICY": "noul_v1"}.items():
        monkeypatch.setenv(name, value)
    original = type(options.output).write_text
    writes = []

    def write(path, *positional, **named):
        writes.append(path.name)
        if path.name == "metadata.json" and writes.count("metadata.json") == 2:
            raise OSError("synthetic final metadata failure")
        return original(path, *positional, **named)

    monkeypatch.setattr(type(options.output), "write_text", write)
    with pytest.raises(KeyboardInterrupt) as raised:
        bench.main()
    assert raised.value is stop
    assert "Final metadata save failed: OSError" in stop.__notes__
    assert "summary.json" in writes
    assert json.loads((options.output / "summary.json").read_text())["batch_error"]["type"] == "KeyboardInterrupt"
