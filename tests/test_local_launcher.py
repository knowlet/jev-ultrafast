"""Pure launcher contracts; no sockets, model processes, browsers or GPU work."""

import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "local_launcher", Path(__file__).resolve().parents[1] / "scripts/run_local_benchmark.py"
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture(autouse=True)
def prohibit_live_execution(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Launcher tests must never open a socket, run a process or access a real URL")

    monkeypatch.setattr(launcher.socket, "socket", forbidden)
    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden)
    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    monkeypatch.setattr(launcher.subprocess, "check_output", forbidden)
    monkeypatch.setattr(launcher.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(launcher.urllib.request, "build_opener", forbidden)


@pytest.mark.parametrize("mode,fixture,required", [
    ("smoke", "click", False), ("smoke", "travel", True),
    ("decision-only", "click", False), ("decision-only", "travel", False),
    ("all", "travel", True), ("all", "click", True),
])
def test_text_model_startup_matches_actual_child_contract(mode, fixture, required):
    # all-mode currently validates a helper endpoint even for click; smoke does not.
    assert launcher.needs_text_service(mode, fixture) is required


@pytest.mark.parametrize("policy", ["baseline", "noul_v1"])
@pytest.mark.parametrize("mode", ["smoke", "decision-only", "helper-only"])
def test_child_argv_preserves_explicit_policy_and_budgets(policy, mode, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", Path("/offline/task"))
    args = SimpleNamespace(policy=policy, repeats=3, task_timeout=180,
                           fixture="click", chrome_path=Path("/offline/chromium"))
    command = launcher.benchmark_command(
        args, mode=mode, destination=Path("/offline/results"), metadata_path=Path("/offline/metadata.json"),
        warmup=0, count=1, fixture_port=18003,
    )
    assert command[command.index("--policy") + 1] == policy
    assert command[command.index("--mode") + 1] == mode
    assert command[command.index("--task-timeout") + 1] == "180"
    assert command[command.index("--fixture-url") + 1] == "http://127.0.0.1:18003"
    assert ("--text-repeats" in command) is (mode == "helper-only")


def test_parser_default_policy_and_original_portable_requirements_are_preserved(monkeypatch):
    definitions = {}

    class ParsedOnly(Exception):
        pass

    class CapturedParser:
        def __init__(self, **_kwargs):
            pass

        def add_argument(self, flag, **options):
            definitions[flag] = options

        def parse_args(self):
            raise ParsedOnly()

    monkeypatch.setattr(launcher.argparse, "ArgumentParser", CapturedParser)
    with pytest.raises(ParsedOnly):
        launcher.main()
    assert definitions["--policy"]["default"] == "baseline"
    assert definitions["--policy"]["choices"] == ["baseline", "noul_v1"]
    if hasattr(launcher, "DEFAULT_S1"):
        assert definitions["--s1-repo"]["default"] == launcher.DEFAULT_S1
        assert definitions["--chrome-path"]["default"] == launcher.DEFAULT_CHROME
    else:
        assert definitions["--s1-repo"]["required"] is True
        assert "default" not in definitions["--chrome-path"]


def test_fixture_readiness_retries_only_startup_and_requires_exact_html(monkeypatch):
    calls, sleeps = [], []

    class FakeOpener:
        def open(self, url, *, timeout):
            calls.append((url, timeout))
            if len(calls) == 1:
                raise OSError("Owned fixture not ready yet")
            return io.BytesIO(b"owned HTML")

    monkeypatch.setattr(launcher.urllib.request, "build_opener", lambda *_args: FakeOpener())
    monkeypatch.setattr(launcher.time, "sleep", sleeps.append)
    process = SimpleNamespace(poll=lambda: None)
    launcher.ready_fixture("http://127.0.0.1:18003/local_click_fixture.html", b"owned HTML",
                           process, launcher.time.monotonic() + 5)
    assert len(calls) == 2 and sleeps == [0.05]
    with pytest.raises(ValueError, match="does not match"):
        launcher.ready_fixture("http://127.0.0.1:18003/local_click_fixture.html", b"different HTML",
                               process, launcher.time.monotonic() + 5)
