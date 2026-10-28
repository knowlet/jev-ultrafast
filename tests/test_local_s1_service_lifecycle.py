"""Owned lifecycle contracts using fake processes/signals only; no runtime execution."""

import io
import json
import signal
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from scripts import local_s1_service as service


@pytest.fixture(autouse=True)
def no_live_operations(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Lifecycle tests cannot spawn, signal or configure real processes/pipes")
    monkeypatch.setattr(service.subprocess, "Popen", forbidden)
    monkeypatch.setattr(service.os, "killpg", forbidden)
    monkeypatch.setattr(service.os, "set_blocking", forbidden)


class SavedLog(io.StringIO):
    def close(self):
        if not self.closed:
            self.saved = self.getvalue()
        super().close()


def owned_adapter():
    adapter = object.__new__(service.Adapter)
    adapter.cleanup_lock = threading.Lock()
    adapter.log_lock = threading.Lock()
    adapter.closed = adapter.logs_closed = False
    adapter.shutdown = None
    adapter.raw_log = SavedLog()
    adapter.request_log = SavedLog()
    adapter.wrapper = SimpleNamespace(unlink=Mock())
    adapter.process = SimpleNamespace(
        pid=4242, stdin=io.StringIO(), stdout=io.StringIO(), wait=Mock(return_value=0),
        poll=Mock(side_effect=AssertionError("No additional process probe")),
    )
    return adapter


def assert_owned_resources_closed(adapter):
    assert adapter.closed and adapter.logs_closed
    assert adapter.process.stdin.closed and adapter.process.stdout.closed
    assert adapter.raw_log.closed and adapter.request_log.closed
    adapter.wrapper.unlink.assert_called_once_with(missing_ok=True)
    adapter.process.poll.assert_not_called()


def test_eof_confirms_only_leader_and_second_close_never_signals_again(monkeypatch):
    adapter = owned_adapter()
    signal_group = Mock()
    monkeypatch.setattr(service.os, "killpg", signal_group)
    outcome = adapter.close()
    assert outcome["exit_code"] == 0 and outcome["leader_exit_confirmed"]
    assert outcome["group_shutdown_confirmed"] is False and not outcome["shutdown_error"]
    assert outcome["owned_resources_closed"] and outcome["group_signal"] == "sigterm_sent"
    adapter.process.wait.assert_called_once_with(timeout=25)
    signal_group.assert_called_once_with(4242, signal.SIGTERM)
    assert_owned_resources_closed(adapter)
    assert adapter.close() is outcome
    signal_group.assert_called_once()
    event = json.loads(adapter.request_log.saved)
    assert event["group_shutdown_confirmed"] is False and event["owned_resources_close_pending"] is True


def test_missing_group_is_reported_without_claiming_all_resources_released(monkeypatch):
    adapter = owned_adapter()
    signal_group = Mock(side_effect=ProcessLookupError)
    monkeypatch.setattr(service.os, "killpg", signal_group)
    outcome = adapter.close()
    assert outcome["group_signal"] == "group_not_found" and outcome["group_shutdown_confirmed"] is False
    signal_group.assert_called_once_with(4242, signal.SIGTERM)
    assert_owned_resources_closed(adapter)


@pytest.mark.parametrize("after_term", [0, subprocess.TimeoutExpired("fake-owned", 5)])
def test_eof_timeout_has_one_sigterm_and_bounded_leader_wait(monkeypatch, after_term):
    adapter = owned_adapter()
    adapter.process.wait.side_effect = [subprocess.TimeoutExpired("fake-owned", 25), after_term]
    signal_group = Mock()
    monkeypatch.setattr(service.os, "killpg", signal_group)
    outcome = adapter.close()
    assert adapter.process.wait.call_args_list == [call(timeout=25), call(timeout=5)]
    signal_group.assert_called_once_with(4242, signal.SIGTERM)
    assert outcome["group_shutdown_confirmed"] is False
    if isinstance(after_term, Exception):
        assert not outcome["leader_exit_confirmed"] and outcome["shutdown_error"]
        assert outcome["method"] == "leader_exit_unconfirmed; no_sigkill_used"
        assert outcome["shutdown_errors"][0]["phase"] == "sigterm_wait"
        assert outcome["shutdown_errors"][0]["type"] == "TimeoutExpired"
    else:
        assert outcome["leader_exit_confirmed"] and not outcome["shutdown_error"]
    assert_owned_resources_closed(adapter)


def test_sigterm_permission_denial_closes_owned_resources_without_retry_or_reroute(monkeypatch):
    adapter = owned_adapter()
    adapter.process.wait.side_effect = subprocess.TimeoutExpired("fake-owned", 25)
    signal_group = Mock(side_effect=PermissionError("synthetic process-control denial"))
    monkeypatch.setattr(service.os, "killpg", signal_group)
    outcome = adapter.close()
    signal_group.assert_called_once_with(4242, signal.SIGTERM)
    adapter.process.wait.assert_called_once_with(timeout=25)
    assert outcome["shutdown_error"] and not outcome["leader_exit_confirmed"]
    assert outcome["group_signal"] == "sigterm_failed" and not outcome["group_shutdown_confirmed"]
    assert outcome["shutdown_errors"][0]["type"] == "PermissionError"
    assert_owned_resources_closed(adapter)
    adapter.close()
    signal_group.assert_called_once()


def test_wait_permission_denial_does_not_replace_denied_control_with_group_signal(monkeypatch):
    adapter = owned_adapter()
    adapter.process.wait.side_effect = PermissionError("synthetic wait denial")
    signal_group = Mock()
    monkeypatch.setattr(service.os, "killpg", signal_group)
    outcome = adapter.close()
    signal_group.assert_not_called()
    assert outcome["shutdown_error"] and outcome["group_signal"] == "not_attempted_after_shutdown_error"
    assert outcome["leader_exit_confirmed"] is outcome["group_shutdown_confirmed"] is False
    assert_owned_resources_closed(adapter)


def test_interrupted_wait_closes_resources_then_preserves_original_interruption(monkeypatch):
    adapter = owned_adapter()
    interruption = KeyboardInterrupt("synthetic interruption, no signal sent")
    adapter.process.wait.side_effect = interruption
    signal_group = Mock()
    monkeypatch.setattr(service.os, "killpg", signal_group)
    with pytest.raises(KeyboardInterrupt) as error:
        adapter.close()
    assert error.value is interruption and adapter.shutdown["shutdown_error"]
    assert adapter.shutdown["leader_exit_confirmed"] is adapter.shutdown["group_shutdown_confirmed"] is False
    signal_group.assert_not_called()
    assert_owned_resources_closed(adapter)


def test_shutdown_record_failure_does_not_skip_owned_file_cleanup(monkeypatch):
    adapter = owned_adapter()
    adapter.record = Mock(side_effect=OSError("synthetic log failure"))
    monkeypatch.setattr(service.os, "killpg", Mock())
    outcome = adapter.close()
    assert outcome["shutdown_error"] and outcome["shutdown_errors"][0]["phase"] == "shutdown_record"
    assert_owned_resources_closed(adapter)


def test_one_resource_close_failure_still_releases_other_owned_files(monkeypatch):
    adapter = owned_adapter()
    adapter.raw_log = SimpleNamespace(close=Mock(side_effect=PermissionError("synthetic log-close denial")))
    monkeypatch.setattr(service.os, "killpg", Mock())
    outcome = adapter.close()
    assert outcome["shutdown_error"] and outcome["owned_resources_closed"] is False
    assert adapter.process.stdout.closed and adapter.request_log.closed and adapter.logs_closed
    adapter.wrapper.unlink.assert_called_once_with(missing_ok=True)
    adapter.raw_log.close.assert_called_once()


@pytest.fixture
def startup(tmp_path, monkeypatch):
    workspace, s1 = tmp_path / "workspace", tmp_path / "existing-s1"
    package = s1 / "package"
    package.mkdir(parents=True)
    native = s1 / "native"
    native.write_bytes(b"synthetic executable never run")
    native.chmod(0o700)
    for relative in (".venv/bin/python", "scripts/gguf_adapter.py"):
        path = s1 / relative
        path.parent.mkdir(parents=True)
        path.write_text("synthetic dependency never imported or run")
    manifest = {"files": [{"path": name} for name in ("model.gguf", "mmproj.gguf")],
                "model_file": "model.gguf", "mmproj_file": "mmproj.gguf",
                "source_checkpoint_sha256": "synthetic-checkpoint", "source_hub_repository": "synthetic-repo",
                "source_hub_weights_revision": "synthetic-version", "quantization": {"language_model": "Q4_K_M"}}
    (package / "conversion-manifest.json").write_text(json.dumps(manifest))
    (package / "s1_config.json").write_text(json.dumps({"temperature": 2.0}))
    monkeypatch.setattr(service, "WORKSPACE", workspace)
    process = SimpleNamespace(pid=4242, stdin=Mock(), stdout=Mock(), wait=Mock(return_value=0))
    process.stdin.fileno.return_value, process.stdout.fileno.return_value = 111, 112
    spawned, adapters = [], []

    def popen(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        spawned.append((args, kwargs))
        return process

    original_close = service.Adapter.close

    def capture_close(adapter):
        adapters.append(adapter)
        return original_close(adapter)

    monkeypatch.setattr(service.subprocess, "Popen", popen)
    monkeypatch.setattr(service.os, "set_blocking", Mock())
    monkeypatch.setattr(service.os, "killpg", Mock())
    monkeypatch.setattr(service.Adapter, "close", capture_close)
    args = SimpleNamespace(s1_repo=s1, model=package, native_runner=native,
                           raw_log=workspace / "raw.log", request_log=workspace / "requests.jsonl",
                           cache_dir=workspace / "cache", ctx_size=4096, batch_size=512, threads=4)
    return args, process, spawned, adapters


@pytest.mark.parametrize("phase", ["stdin_blocking", "stdout_blocking", "launch_record"])
def test_postspawn_setup_failure_preserves_primary_error_and_attempts_owned_cleanup(startup, monkeypatch, phase):
    args, process, spawned, adapters = startup
    primary = OSError("synthetic primary startup failure")
    if phase == "launch_record":
        original_record = service.Adapter.record

        def fail_launch(adapter, event):
            if event["event"] == "launch":
                raise primary
            original_record(adapter, event)

        monkeypatch.setattr(service.Adapter, "record", fail_launch)
    else:
        service.os.set_blocking.side_effect = [primary] if phase == "stdin_blocking" else [None, primary]
    with pytest.raises(OSError) as error:
        service.Adapter(args)
    assert error.value is primary and len(spawned) == len(adapters) == 1
    process.stdin.close.assert_called_once()
    process.stdout.close.assert_called_once()
    process.wait.assert_called_once_with(timeout=25)
    service.os.killpg.assert_called_once_with(4242, signal.SIGTERM)
    adapter = adapters[0]
    assert adapter.closed and adapter.raw_log.closed and adapter.request_log.closed
    assert not adapter.wrapper.exists() and adapter.shutdown["group_shutdown_confirmed"] is False


def test_cleanup_interruption_cannot_replace_original_postspawn_failure(startup):
    args, process, _spawned, adapters = startup
    primary = ValueError("synthetic primary setup failure")
    service.os.set_blocking.side_effect = primary
    process.wait.side_effect = KeyboardInterrupt("synthetic cleanup interruption")
    with pytest.raises(ValueError) as error:
        service.Adapter(args)
    assert error.value is primary and any("KeyboardInterrupt" in note for note in primary.__notes__)
    assert adapters[0].raw_log.closed and adapters[0].request_log.closed and not adapters[0].wrapper.exists()
    service.os.killpg.assert_not_called()


def test_failed_popen_never_signals_a_group_and_still_closes_owned_logs(startup, monkeypatch):
    args, _process, _spawned, adapters = startup
    primary = PermissionError("synthetic launch denial")
    monkeypatch.setattr(service.subprocess, "Popen", Mock(side_effect=primary))
    with pytest.raises(PermissionError) as error:
        service.Adapter(args)
    assert error.value is primary and adapters[0].shutdown["method"] == "no_process_spawned"
    service.os.killpg.assert_not_called()
    assert adapters[0].raw_log.closed and adapters[0].request_log.closed and not adapters[0].wrapper.exists()
