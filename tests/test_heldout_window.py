"""Offline watchdog contracts; no socket, model, GPU, browser or child process."""

import io
import json
import signal
import socket
import sys
import time
import urllib.error
from types import SimpleNamespace

import pytest

from scripts import run_heldout_window as window


@pytest.fixture(autouse=True)
def no_live_actions(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Offline window tests cannot start processes, bind sockets, or open network")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(window.subprocess, "Popen", forbidden)
    monkeypatch.setattr(window.urllib.request, "build_opener", forbidden)


@pytest.fixture
def config(tmp_path):
    s1 = tmp_path / "s1"
    package = s1 / "artifacts/exports/s1-boolq-Q4_K_M"
    package.mkdir(parents=True)
    selected = {}
    for name in ("model.gguf", "mmproj.gguf"):
        path = package / name
        path.write_bytes(b"synthetic weights, never loaded")
        selected[name] = {"path": name, "size_bytes": path.stat().st_size, "sha256": window.digest(path)}
    manifest = {"source_hub_repository": "knowlet/Gemma-4-12B-Unified-System-One",
                "source_checkpoint_sha256": "fixture-checkpoint", "source_hub_weights_revision": "fixture-version",
                "status": "converted_and_structurally_verified",
                "quantization": {"language_model": "Q4_K_M", "multimodal_projector": "F16", "direct_conversion": False},
                "model_file": "model.gguf", "mmproj_file": "mmproj.gguf", "files": list(selected.values())}
    (package / "conversion-manifest.json").write_text(json.dumps(manifest))
    calibration = {"temperature": 3.25, "runtime": "llama.cpp", "runtime_format": "gguf",
                   "conversion_manifest_sha256": window.digest(package / "conversion-manifest.json")}
    (package / "s1_config.json").write_text(json.dumps(calibration))
    for relative in ("artifacts/gguf-runtime/bin/s1-gguf",
                     "artifacts/export-tools/llama.cpp/build/bin/llama-server",
                     ".venv/bin/python", "scripts/gguf_adapter.py", "tools/gguf/s1_gguf.cpp"):
        path = s1 / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic never-executed dependency")
        path.chmod(0o700)
    chrome = tmp_path / "chromium"
    chrome.write_text("synthetic never-executed browser")
    chrome.chmod(0o700)
    return window.configuration(SimpleNamespace(s1_repo=s1, chrome_path=chrome))


def health(config, sources):
    return {"status": "starting", "calibrated_inference_verified": False, "completed_requests": 0,
            "adapter_pid": 101, "s1_repo": str(config["s1"]), "model_package": str(config["package"]),
            "model_file": str(config["model_file"]), "native_runner": str(config["native"]),
            "native_runner_sha256": sources[str(config["native"])],
            "bridge_source_sha256": sources[str(window.ROOT / "scripts/local_s1_service.py")],
            "adapter_source_sha256": sources[str(config["s1"] / "scripts/gguf_adapter.py")],
            "conversion_manifest_sha256": sources[str(config["package"] / "conversion-manifest.json")],
            "calibration_sha256": sources[str(config["package"] / "s1_config.json")],
            "temperature": 3.25, "quantization": config["manifest"]["quantization"],
            "source_hub_repository": "knowlet/Gemma-4-12B-Unified-System-One", "reuse_prefix": False,
            "ctx_size": 4096, "batch_size": 512, "threads": 4,
            "model_manifest_identity": config["manifest"]["files"][0],
            "mmproj_manifest_identity": config["manifest"]["files"][1]}


def test_real_quantization_schema_and_exact_native_only_original_command(config, tmp_path):
    assert config["manifest"]["quantization"]["language_model"] == "Q4_K_M"
    argv = window.commands(config, tmp_path, 18001, 18002)
    assert "--reuse-prefix" not in argv["decision-service"]
    assert str(config["native"]) in argv["decision-service"]
    assert "--execute" in argv["runner"] and argv["runner"][argv["runner"].index("--policy") + 1] == "noul_v1"
    assert not any(flag in argv["runner"] for flag in ("--task", "--repeats", "--max-decisions"))
    assert argv["text-service"][argv["text-service"].index("--host") + 1] == "127.0.0.1"
    assert argv["text-service"][argv["text-service"].index("--model") + 1] == str(config["model_file"])
    assert argv["text-service"][argv["text-service"].index("--seed") + 1] == "42"


@pytest.mark.parametrize("change", [None, "quantization", "pid", "temperature", "source", "weights"])
def test_fresh_listening_and_health_bind_same_actual_identity_without_ready_inference(config, change):
    _plan, freeze = window.heldout.verify_freeze(window.ROOT / "benchmark/heldout-plan.json",
                                                window.ROOT / "benchmark/heldout-freeze.json")
    sources = window.source_snapshot(config, freeze)
    actual = health(config, sources)
    listening = dict(actual, event="listening")
    if change == "quantization":
        actual["quantization"] = "Q4_K_M"
    elif change == "pid":
        listening["adapter_pid"] = 999
    elif change == "temperature":
        actual["temperature"] = 1
    elif change == "source":
        actual["bridge_source_sha256"] = "different"
    elif change == "weights":
        actual["model_manifest_identity"] = {"path": "base.gguf"}
    if change:
        with pytest.raises(ValueError):
            window.verify_health(actual, listening, config, sources)
    else:
        assert window.verify_health(actual, listening, config, sources) == 101
        assert not actual["calibrated_inference_verified"] and actual["completed_requests"] == 0


@pytest.mark.parametrize("status,body", [(403, "permission denied"), (401, "unauthorized"),
                                        (404, "wrong service"), (503, "runtime unavailable")])
def test_actual_health_error_is_not_retried_or_rerouted(monkeypatch, status, body):
    attempted = []
    def open_url(url, **_kwargs):
        attempted.append(url)
        raise urllib.error.HTTPError(url, status, "actual denied/unavailable", {}, io.BytesIO(body.encode()))
    monkeypatch.setattr(window.urllib.request, "build_opener", lambda *_: SimpleNamespace(open=open_url))
    owner = SimpleNamespace(poll=lambda _name: None)
    with pytest.raises(RuntimeError, match="no retry or reroute"):
        window.health_get("http://127.0.0.1:18001/healthz", owner, "decision-service", time.monotonic() + 10)
    assert attempted == ["http://127.0.0.1:18001/healthz"]


def test_missing_group_is_retired_never_retargeted_and_unknown_group_refused(tmp_path, monkeypatch):
    owner = window.OwnedProcesses(tmp_path, {})
    owner.groups[111] = {"name": "owned"}
    seen = []
    def absent(pid, sig):
        seen.append((pid, sig))
        raise ProcessLookupError
    monkeypatch.setattr(window.os, "killpg", absent)
    assert owner.alive_groups() == []
    owner.kill_all()
    assert owner.alive_groups() == [] and seen == [(111, 0)]
    with pytest.raises(ValueError, match="unregistered"):
        owner.send_group(999, signal.SIGKILL)


def test_post_spawn_abort_keeps_known_owned_session_and_closes_logs(tmp_path, monkeypatch):
    owner = window.OwnedProcesses(tmp_path, {})
    process = SimpleNamespace(pid=123, returncode=None)
    seen = []
    monkeypatch.setattr(window.subprocess, "Popen", lambda *_args, **kwargs: process
                        if kwargs["start_new_session"] else pytest.fail("Missing owned session"))
    monkeypatch.setattr(window.os, "killpg", lambda pid, sig: seen.append((pid, sig)))
    owner.abort_pending = True
    with pytest.raises(window.WorkExpired, match="registered"):
        owner.start("runner", ["synthetic"])
    assert owner.children["runner"] is process and 123 in owner.groups
    owner.kill_all()
    owner.close_logs()
    assert seen == [(123, signal.SIGKILL)] and all(log.closed for log in owner.handles)


def test_clean_environment_does_not_forward_secrets_or_proxy(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "synthetic-do-not-forward")
    monkeypatch.setenv("HTTPS_PROXY", "synthetic-proxy")
    env = window.clean_environment(tmp_path)
    assert "API_KEY" not in env and "HTTPS_PROXY" not in env and "HOME" not in env
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_plan_only_receipt_keeps24_schedule_and_deadlines_before_any_live_action(tmp_path, config, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    output = root / "benchmark/results/new"
    monkeypatch.setattr(window, "ROOT", root)
    monkeypatch.setattr(window, "configuration", lambda *_: config)
    real_plan = json.loads((window.heldout.REPO / "benchmark/heldout-plan.json").read_text())
    monkeypatch.setattr(window.heldout, "verify_freeze", lambda *_: (real_plan, {"files": {}}))
    monkeypatch.setattr(window, "source_snapshot", lambda *_: {"fixture": "digest"})
    monkeypatch.setattr(sys, "argv", ["window", "--s1-repo", str(config["s1"]),
                                     "--chrome-path", str(config["chrome"]), "--output", str(output)])
    assert window.main() == 0
    receipt = json.loads((output / "window-receipt.json").read_text())
    assert receipt["expected_runs"] == len(receipt["scheduled_runs"]) == 24
    assert receipt["measured_physical_request_cap"] == 288 and receipt["development_warmups"] == 2
    assert (receipt["work_seconds"], receipt["total_seconds"], receipt["cleanup_reserved_seconds"]) == (840, 900, 60)
    assert receipt["ready_inferences"] == 0 and not receipt["execution_requested"]


def test_execute_timer_starts_before_configuration(monkeypatch, tmp_path):
    timers, handlers = [], {}
    monkeypatch.setattr(window.signal, "getsignal", lambda *_: None)
    monkeypatch.setattr(window.signal, "signal", lambda sig, handler: handlers.update({sig: handler}))
    monkeypatch.setattr(window.signal, "setitimer", lambda kind, seconds: timers.append((kind, seconds)))
    def config(_args):
        assert signal.SIGALRM in handlers and timers and 839 < timers[0][1] <= 840
        raise ValueError("Synthetic preflight failure; no model started")
    monkeypatch.setattr(window, "configuration", config)
    monkeypatch.setattr(sys, "argv", ["window", "--execute", "--s1-repo", str(tmp_path),
                                     "--chrome-path", str(tmp_path),
                                     "--output", str(window.ROOT / "benchmark/results/offline-timer-test")])
    with pytest.raises(ValueError, match="preflight"):
        window.main()


def test_cleanup_interrupts_runner_pid_then_only_owned_groups_within_reserve(monkeypatch):
    clock, events = [0.0], []
    monkeypatch.setattr(window.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(window.time, "sleep", lambda duration: clock.__setitem__(0, clock[0] + duration))
    class Owner:
        children = {"runner": SimpleNamespace(pid=11), "decision-service": SimpleNamespace(pid=22),
                    "text-service": SimpleNamespace(pid=33)}
        groups = {11: {}, 22: {}, 33: {}, 44: {"name": "owned nested adapter"}}
        denied_groups = set()
        events = []
        stopped = False
        remaining = {11, 22, 33, 44}
        def poll(self, name):
            return 1 if self.stopped else None
        def send_pid(self, name, sig):
            events.append(("pid", self.children[name].pid, sig))
            if name == "runner":
                self.stopped = True
            self.remaining.discard(self.children[name].pid)
        def alive_groups(self):
            return list(self.remaining)
        def send_group(self, pgid, sig):
            assert pgid in self.groups
            events.append(("group", pgid, sig))
            if sig == signal.SIGKILL:
                self.remaining.discard(pgid)
    report = window.cleanup(Owner(), 58)
    assert events[0] == ("pid", 11, signal.SIGINT)
    assert ("group", 44, signal.SIGKILL) in events
    assert all(pid in (11, 22, 33, 44) for _kind, pid, _sig in events)
    assert not report["remaining_owned_groups"] and clock[0] <= 58
    assert report["cleanup_liveness_verified"] and report["unknown_owned_groups"] == []


def test_deferred_spawn_abort_arms900_hardstop_and_exit_even_killall_fails(tmp_path, config, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    output = root / "benchmark/results/window"
    handlers, timers, exited = {}, [], []
    monkeypatch.setattr(window, "ROOT", root)
    monkeypatch.setattr(window.signal, "getsignal", lambda sig: handlers.get(sig))
    monkeypatch.setattr(window.signal, "signal", lambda sig, handler: handlers.update({sig: handler}))
    monkeypatch.setattr(window.signal, "setitimer", lambda kind, duration: timers.append(duration))
    monkeypatch.setattr(window, "configuration", lambda *_: config)
    plan = json.loads((window.heldout.REPO / "benchmark/heldout-plan.json").read_text())
    monkeypatch.setattr(window.heldout, "verify_freeze", lambda *_: (plan, {"files": {}}))
    monkeypatch.setattr(window, "source_snapshot", lambda *_: {})
    monkeypatch.setattr(window, "unchanged", lambda *_: True)
    monkeypatch.setattr(window, "reserve_ports", lambda: (18001, 18002))
    def exit_now(code):
        exited.append(code)
        raise SystemExit(code)
    monkeypatch.setattr(window.os, "_exit", exit_now)
    class Owner:
        events, groups, children = [], {}, {}
        spawning, abort_pending = False, False
        def __init__(self, *_args):
            pass
        def start(self, *_args):
            self.spawning = True
            handlers[signal.SIGALRM](signal.SIGALRM, None)
            assert self.abort_pending and 899 < timers[-1] <= 900
            self.spawning = False
            with pytest.raises(SystemExit) as error:
                handlers[signal.SIGALRM](signal.SIGALRM, None)
            assert error.value.code == 124
            raise window.WorkExpired("Synthetic registered spawn interrupted")
        def kill_all(self):
            raise RuntimeError("Synthetic process-control failure")
        def close_logs(self):
            pass
    monkeypatch.setattr(window, "OwnedProcesses", Owner)
    monkeypatch.setattr(window, "cleanup", lambda *_: {
        "remaining_owned_groups": [], "denied_groups": [], "unknown_owned_groups": [],
        "cleanup_liveness_verified": True,
    })
    monkeypatch.setattr(sys, "argv", ["window", "--execute", "--s1-repo", str(config["s1"]),
                                     "--chrome-path", str(config["chrome"]), "--output", str(output)])
    assert window.main() == 1 and exited == [124]
    receipt = json.loads((output / "window-receipt.json").read_text())
    assert receipt["summary"]["expected_runs"] == receipt["summary"]["unexecuted_runs"] == 24
    assert timers[-1] == 0 and all(handler is None for handler in handlers.values())


def test_denied_liveness_probe_is_reported_unknown_without_retry_or_alternate_probe(tmp_path, monkeypatch):
    owner = window.OwnedProcesses(tmp_path, {})
    owner.groups[111] = {"name": "synthetic owned group"}
    calls = []
    def denied(pid, sig):
        calls.append((pid, sig))
        raise PermissionError("Synthetic liveness denial; do not retry")
    monkeypatch.setattr(window.os, "killpg", denied)
    report = window.cleanup(owner, time.monotonic() + 1)
    assert report["remaining_owned_groups"] == []
    assert report["denied_groups"] == report["unknown_owned_groups"] == [111]
    assert report["cleanup_liveness_verified"] is False
    assert calls == [(111, 0)]


@pytest.mark.parametrize("verified,denied,unknown,expected", [
    (True, [], [], 0), (False, [111], [111], 1), (True, [111], [], 1),
    (True, [], [111], 1), (False, [], [], 1),
])
def test_completed_suite_cannot_succeed_with_denied_or_unknown_cleanup(
    tmp_path, config, monkeypatch, verified, denied, unknown, expected,
):
    root = tmp_path / "repository"
    root.mkdir()
    output = root / "benchmark/results/window"
    handlers = {}
    monkeypatch.setattr(window, "ROOT", root)
    monkeypatch.setattr(window.signal, "getsignal", lambda sig: handlers.get(sig))
    monkeypatch.setattr(window.signal, "signal", lambda sig, handler: handlers.update({sig: handler}))
    monkeypatch.setattr(window.signal, "setitimer", lambda *_: None)
    monkeypatch.setattr(window, "configuration", lambda *_: config)
    plan = json.loads((window.heldout.REPO / "benchmark/heldout-plan.json").read_text())
    monkeypatch.setattr(window.heldout, "verify_freeze", lambda *_: (plan, {"files": {}}))
    monkeypatch.setattr(window, "source_snapshot", lambda *_: {})
    monkeypatch.setattr(window, "unchanged", lambda *_: True)
    monkeypatch.setattr(window, "reserve_ports", lambda: (18001, 18002))
    healths = iter([{}, {}, {"data": [{"id": window.TEXT_MODEL}]}])
    monkeypatch.setattr(window, "health_get", lambda *_: next(healths))
    monkeypatch.setattr(window, "listening_event", lambda *_: {})
    monkeypatch.setattr(window, "verify_health", lambda *_: 111)
    monkeypatch.setattr(window, "wait_runner", lambda *_: 0)
    class Owner:
        events, groups, children = [], {}, {}
        spawning, abort_pending = False, False
        def __init__(self, *_args):
            pass
        def start(self, *_args):
            pass
        def register(self, *_args):
            pass
        def poll(self, *_args):
            return None
        def kill_all(self):
            pytest.fail("Successful mock suite must not enter fallback process control")
        def close_logs(self):
            pass
    monkeypatch.setattr(window, "OwnedProcesses", Owner)
    cleanup = {"remaining_owned_groups": [], "denied_groups": denied, "unknown_owned_groups": unknown,
               "cleanup_liveness_verified": verified}
    monkeypatch.setattr(window, "cleanup", lambda *_: cleanup)
    monkeypatch.setattr(window, "incomplete_summary", lambda *_: {
        "expected_runs": 24, "executed_runs": 24, "successful_runs": 24,
        "failed_runs": 0, "unexecuted_runs": 0, "complete": True,
    })
    monkeypatch.setattr(sys, "argv", ["window", "--execute", "--s1-repo", str(config["s1"]),
                                     "--chrome-path", str(config["chrome"]), "--output", str(output)])
    assert window.main() == expected
    receipt = json.loads((output / "window-receipt.json").read_text())
    assert receipt["summary"]["complete"] and receipt["summary"]["successful_runs"] == 24
    assert receipt["cleanup"]["cleanup_liveness_verified"] is verified
