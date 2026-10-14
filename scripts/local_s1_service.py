"""Loopback HTTP bridge to an existing, calibrated System One GGUF runtime.

No inference packages are installed here. The existing S1 Python environment
runs its unchanged gguf_adapter.py --predict, including its weight, processor
and calibration checks. Requests and responses cross its persistent JSONL pipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import shlex
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
MAX_BODY_BYTES = 16 * 1024 * 1024
REQUEST_TIMEOUT = 90.0


def workspace_path(value: Path) -> Path:
    path = value.expanduser().resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError(f"Output and cache paths must be inside {WORKSPACE}")
    return path


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_native_runner(value: Path, s1_repo: Path, expected_sha256=None, reuse_prefix=False):
    """Permit a deliberately selected workspace experiment without altering the original repo."""
    runner = value.expanduser().resolve(strict=True)
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise ValueError("--native-runner must be an existing executable")
    original = runner.is_relative_to(s1_repo)
    experiment = runner.is_relative_to(WORKSPACE / "tools/native-prefix")
    if not original and not experiment:
        raise ValueError("Native runner must be inside --s1-repo or workspace tools/native-prefix")
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str) or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("--native-runner-sha256 must be a lowercase SHA-256")
    if not original and expected_sha256 is None:
        raise ValueError("Workspace experimental runner requires --native-runner-sha256")
    if reuse_prefix and original:
        raise ValueError("--reuse-prefix requires a deliberately selected workspace experimental runner")
    actual = file_sha256(runner)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("Native runner differs from the explicit SHA-256; no process started")
    return runner, actual, "original_s1_repository" if original else "workspace_experiment"


def local_environment(cache: Path, s1_repo: Path) -> dict[str, str]:
    directories = {name: cache / name for name in ("hf", "torch", "xdg", "tmp")}
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    # Do not inherit credentials, auth configuration, proxy settings or telemetry
    # configuration from the caller's environment or the user's home directory.
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "en_US.UTF-8",
        "PYTHONPATH": str(s1_repo / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "WANDB_DISABLED": "true",
        "HF_HOME": str(directories["hf"]),
        "HF_HUB_CACHE": str(directories["hf"] / "hub"),
        "TORCH_HOME": str(directories["torch"]),
        "XDG_CACHE_HOME": str(directories["xdg"]),
        "TMPDIR": str(directories["tmp"]),
        "TMP": str(directories["tmp"]),
        "TEMP": str(directories["tmp"]),
    }


class Adapter:
    def __init__(self, args):
        self.lock = threading.Lock()
        self.cleanup_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.failed = False
        self.closed = False
        self.shutdown = None
        self.logs_closed = False
        self.completed = 0
        self.first_success_ms = None
        self.buffer = bytearray()
        self.started = time.monotonic()
        self.raw_path = workspace_path(args.raw_log)
        self.native_raw_path = self.raw_path.with_name(self.raw_path.name + ".native.log")
        self.request_path = workspace_path(args.request_log or self.raw_path.with_name("decisions.jsonl"))
        cache = workspace_path(args.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        s1_repo = args.s1_repo.expanduser().resolve(strict=True)
        package = args.model.expanduser().resolve(strict=True)
        native_runner, native_sha256, native_scope = verified_native_runner(
            args.native_runner, s1_repo, getattr(args, "native_runner_sha256", None),
            getattr(args, "reuse_prefix", False),
        )
        if not package.is_dir() or not package.is_relative_to(s1_repo):
            raise ValueError("--model must be a local GGUF package directory inside --s1-repo")
        python = s1_repo / ".venv/bin/python"
        adapter_script = s1_repo / "scripts/gguf_adapter.py"
        if not python.is_file() or not adapter_script.is_file():
            raise ValueError("--s1-repo requires existing .venv/bin/python and scripts/gguf_adapter.py")
        manifest = json.loads((package / "conversion-manifest.json").read_text())
        calibration = json.loads((package / "s1_config.json").read_text())
        files = {item["path"]: item for item in manifest["files"]}
        outputs = (self.raw_path, self.native_raw_path, self.request_path)
        if len(set(outputs)) != len(outputs) or any(path.exists() for path in outputs):
            raise ValueError("Raw logs must be distinct new paths; existing logs are never overwritten")
        self.identity = {
            "backend": "s1.gguf_adapter --predict",
            "s1_repo": str(s1_repo),
            "model_package": str(package),
            "native_runner": str(native_runner),
            "native_runner_sha256": native_sha256,
            "native_runner_scope": native_scope,
            "reuse_prefix": getattr(args, "reuse_prefix", False),
            "bridge_source_sha256": file_sha256(Path(__file__)),
            "adapter_source_sha256": file_sha256(adapter_script),
            "model_file": str(package / manifest["model_file"]),
            "model_manifest_identity": files[manifest["model_file"]],
            "mmproj_manifest_identity": files[manifest["mmproj_file"]],
            "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
            "source_hub_repository": manifest["source_hub_repository"],
            "source_hub_weights_revision": manifest["source_hub_weights_revision"],
            "quantization": manifest["quantization"],
            "conversion_manifest_sha256": file_sha256(package / "conversion-manifest.json"),
            "calibration_sha256": file_sha256(package / "s1_config.json"),
            "temperature": calibration["temperature"],
            "ctx_size": args.ctx_size,
            "batch_size": args.batch_size,
            "threads": args.threads,
            "request_timeout_seconds": REQUEST_TIMEOUT,
            "raw_log": str(self.raw_path),
            "native_raw_log": str(self.native_raw_path),
            "request_log": str(self.request_path),
            "rss_scope": "Separate /usr/bin/time -l reports for adapter and native model processes; bytes on macOS",
        }
        # The second timer measures the real native model child, rather than
        # mistaking the CPU preprocessing adapter's own RSS for model RSS.
        wrapper = cache / f"native-timed-{os.getpid()}.sh"
        with wrapper.open("x", encoding="utf-8") as stream:
            stream.write(
                "#!/bin/sh\n"
                f"exec 2>>{shlex.quote(str(self.native_raw_path))}\n"
                f"exec /usr/bin/time -l {shlex.quote(str(native_runner))}"
                + (" --reuse-prefix" if getattr(args, "reuse_prefix", False) else "")
                + " \"$@\"\n"
            )
        wrapper.chmod(0o700)
        self.wrapper = wrapper
        for path in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_log = self.raw_path.open("x", encoding="utf-8")
        self.native_raw_path.open("x", encoding="utf-8").close()
        self.request_log = self.request_path.open("x", encoding="utf-8")
        command = [
            "/usr/bin/time", "-l", str(python), "-u", str(adapter_script),
            "--predict", "--model", str(package), "--native-runner", str(wrapper),
            "--ctx-size", str(args.ctx_size), "--batch-size", str(args.batch_size),
            "--threads", str(args.threads),
        ]
        self.process = None
        try:
            self.process = subprocess.Popen(
                command,
                cwd=WORKSPACE,
                env=local_environment(cache, s1_repo),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.raw_log,
                bufsize=0,
                start_new_session=True,
            )
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
            self.record({"event": "launch", "command": command, "identity": self.identity})
        except BaseException as error:
            try:
                outcome = self.close()
                if outcome["shutdown_errors"]:
                    error.add_note("Owned cleanup was incomplete; see the shutdown record.")
            except BaseException as cleanup_error:
                error.add_note(f"Owned cleanup also failed: {type(cleanup_error).__name__}")
            raise

    def record(self, event: dict):
        with self.log_lock:
            if self.logs_closed:
                return
            self.request_log.write(json.dumps(event, allow_nan=False) + "\n")
            self.request_log.flush()

    def health(self) -> dict:
        running = self.process.poll() is None and not self.failed and not self.closed
        return {
            "status": "ready" if running and self.completed else "starting" if running else "unavailable",
            "calibrated_inference_verified": bool(self.completed),
            "completed_requests": self.completed,
            "first_success_since_launch_ms": self.first_success_ms,
            "adapter_pid": self.process.pid,
            "adapter_exit_code": self.process.poll(),
            **self.identity,
        }

    @staticmethod
    def remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("90-second decision deadline exceeded; no retry was attempted")
        return remaining

    def write_request(self, payload: bytes, deadline: float):
        fd = self.process.stdin.fileno()
        while payload:
            if not select.select([], [fd], [], self.remaining(deadline))[1]:
                raise TimeoutError("adapter input deadline exceeded")
            try:
                written = os.write(fd, payload[:4096])
            except BlockingIOError:
                continue
            payload = payload[written:]

    def read_response(self, deadline: float) -> dict:
        fd = self.process.stdout.fileno()
        while b"\n" not in self.buffer:
            if not select.select([fd], [], [], self.remaining(deadline))[0]:
                raise TimeoutError("adapter output deadline exceeded")
            try:
                data = os.read(fd, 65536)
            except BlockingIOError:
                continue
            if not data:
                raise RuntimeError(f"adapter closed stdout (exit={self.process.poll()}); inspect raw logs")
            self.buffer.extend(data)
            if len(self.buffer) > MAX_BODY_BYTES:
                raise RuntimeError("adapter response exceeds the 16 MiB limit")
        line, _, pending = self.buffer.partition(b"\n")
        self.buffer = bytearray(pending)
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("adapter response must be a JSON object")
        return response

    def decide(self, request: dict, started: float) -> tuple[int, dict]:
        deadline = started + REQUEST_TIMEOUT
        if not self.lock.acquire(timeout=self.remaining(deadline)):
            return 504, {"status": "error", "error": "serial queue deadline exceeded; no retry"}
        try:
            if self.failed or self.closed or self.process.poll() is not None:
                return 503, {"status": "error", "error": "S1 adapter is unavailable; no restart or retry"}
            acquired = time.monotonic()
            self.write_request((json.dumps(request, allow_nan=False) + "\n").encode(), deadline)
            response = self.read_response(deadline)
            bridge = {
                "queue_ms": (acquired - started) * 1000,
                "wall_ms": (time.monotonic() - started) * 1000,
                "adapter_transport_ms": (time.monotonic() - acquired) * 1000,
            }
            self.record({"event": "decision", "request": request, "response": response, "bridge": bridge})
            if response.get("status") == "ok":
                self.completed += 1
                if self.first_success_ms is None:
                    self.first_success_ms = (time.monotonic() - self.started) * 1000
                return 200, {**response, "local_bridge": bridge}
            return 502, {**response, "local_bridge": bridge}
        except (OSError, ValueError, RuntimeError, RecursionError, TimeoutError) as error:
            # A timed-out line must never be consumed by a later request.
            self.failed = True
            self.record({"event": "transport_failure", "error": str(error), "request": request})
            threading.Thread(target=self.close, daemon=True).start()
            return (504 if isinstance(error, TimeoutError) else 502), {"status": "error", "error": str(error)}
        finally:
            self.lock.release()

    def close(self):
        with self.cleanup_lock:
            if self.closed:
                return self.shutdown
            self.closed = True
            outcome = {"event": "shutdown", "method": "orderly_eof", "exit_code": None,
                       "leader_exit_confirmed": False, "group_shutdown_confirmed": False,
                       "group_signal": "not_attempted", "shutdown_errors": [], "owned_resources_closed": False}
            self.shutdown = outcome
            interruptions = []
            resources_closed = True

            def failed(phase, error):
                outcome["shutdown_errors"].append({"phase": phase, "type": type(error).__name__,
                                                   "message": str(error)[:1000]})
                if not isinstance(error, Exception):
                    interruptions.append(error)

            def release(stream, phase):
                nonlocal resources_closed
                if stream is None:
                    return
                try:
                    stream.close()
                except BaseException as error:
                    resources_closed = False
                    failed(phase, error)

            try:
                if self.process is None:
                    outcome["method"] = "no_process_spawned"
                else:
                    release(self.process.stdin, "stdin_close")
                    timed_out = False
                    try:
                        # EOF lets the unchanged adapter close its native child and flush its real timers.
                        outcome["exit_code"] = self.process.wait(timeout=25)
                        outcome["leader_exit_confirmed"] = True
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        outcome["method"] = "owned_process_group_sigterm"
                    except BaseException as error:
                        failed("eof_wait", error)
                    denied = any(item["type"] == "PermissionError" for item in outcome["shutdown_errors"])
                    if denied or interruptions:
                        outcome["group_signal"] = "not_attempted_after_shutdown_error"
                    else:
                        # Signal this known owned group once. Neither EOF nor leader exit proves group shutdown.
                        try:
                            os.killpg(self.process.pid, signal.SIGTERM)
                            outcome["group_signal"] = "sigterm_sent"
                        except ProcessLookupError:
                            outcome["group_signal"] = "group_not_found"
                        except BaseException as error:
                            outcome["group_signal"] = "sigterm_failed"
                            failed("group_sigterm", error)
                        if timed_out and outcome["group_signal"] == "sigterm_sent":
                            try:
                                outcome["exit_code"] = self.process.wait(timeout=5)
                                outcome["leader_exit_confirmed"] = True
                            except BaseException as error:
                                failed("sigterm_wait", error)
                    if timed_out and not outcome["leader_exit_confirmed"]:
                        outcome["method"] = "leader_exit_unconfirmed; no_sigkill_used"
                outcome["shutdown_error"] = bool(outcome["shutdown_errors"])
                self.record({**outcome, "owned_resources_close_pending": True})
            except BaseException as error:
                failed("shutdown_record", error)
            finally:
                if self.process is not None:
                    release(self.process.stdout, "stdout_close")
                release(self.raw_log, "raw_log_close")
                with self.log_lock:
                    self.logs_closed = True
                    release(self.request_log, "request_log_close")
                try:
                    self.wrapper.unlink(missing_ok=True)
                except BaseException as error:
                    resources_closed = False
                    failed("wrapper_unlink", error)
                outcome["owned_resources_closed"] = resources_closed
                outcome["shutdown_error"] = bool(outcome["shutdown_errors"])
            if interruptions:
                raise interruptions[0]
            return outcome


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int, adapter: Adapter):
        self.adapter = adapter
        super().__init__(("127.0.0.1", port), Handler)


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload: dict):
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path != "/healthz":
            self.send_json(404, {"status": "error", "error": "unknown route"})
            return
        health = self.server.adapter.health()
        self.send_json(503 if health["status"] == "unavailable" else 200, health)

    def do_POST(self):
        started = time.monotonic()
        if self.path != "/decide":
            self.send_json(404, {"status": "error", "error": "unknown route"})
            return
        try:
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Transfer-Encoding is unsupported; provide Content-Length")
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BODY_BYTES:
                self.send_json(413, {"status": "error", "error": "request must contain 1..16 MiB JSON bytes"})
                return
            content = bytearray()
            deadline = started + REQUEST_TIMEOUT
            while len(content) < size:
                self.connection.settimeout(Adapter.remaining(deadline))
                chunk = self.rfile.read1(min(65536, size - len(content)))
                if not chunk:
                    raise ValueError("incomplete request body")
                content.extend(chunk)
            request = json.loads(content, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if not isinstance(request, dict):
                raise ValueError("request must be the exact S1 DecisionRequest JSON object")
            status, response = self.server.adapter.decide(request, started)
        except TimeoutError as error:
            status, response = 504, {"status": "error", "error": str(error)}
        except (ValueError, RecursionError, OSError) as error:
            status, response = 400, {"status": "error", "error": str(error)}
        self.send_json(status, response)

    def log_message(self, format, *args):
        # Local fixture text and requests belong in the explicit raw JSONL log.
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Existing trained GGUF package directory")
    parser.add_argument("--native-runner", type=Path, required=True)
    parser.add_argument("--native-runner-sha256", help="Required for an explicit workspace experimental binary")
    parser.add_argument("--reuse-prefix", action="store_true", help="Opt-in workspace prefix-cache experiment")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--raw-log", type=Path, required=True, help="New adapter stderr + /usr/bin/time -l log")
    parser.add_argument("--request-log", type=Path, help="New raw request/response JSONL; default beside --raw-log")
    parser.add_argument("--cache-dir", type=Path, default=WORKSPACE / "artifacts/local-s1-cache")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or min(args.ctx_size, args.batch_size, args.threads) < 1:
        parser.error("port must be 1..65535; context, batch and threads must be positive")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _sig, _frame: stop.set())
    adapter = None
    try:
        adapter = Adapter(args)
        if stop.is_set():
            return
        server = Server(args.port, adapter)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        print(
            json.dumps({"event": "listening", "url": f"http://127.0.0.1:{args.port}", **adapter.health()}), flush=True
        )
        while not stop.wait(0.25):
            if adapter.process.poll() is not None or adapter.failed:
                break
        server.shutdown()
        server.server_close()
    finally:
        if adapter is not None:
            adapter.close()


if __name__ == "__main__":
    main()
