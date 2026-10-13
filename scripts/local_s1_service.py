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
        native_runner = args.native_runner.expanduser().resolve(strict=True)
        if not package.is_dir() or not package.is_relative_to(s1_repo):
            raise ValueError("--model must be a local GGUF package directory inside --s1-repo")
        if not native_runner.is_file() or not native_runner.is_relative_to(s1_repo):
            raise ValueError("--native-runner must be an existing file inside --s1-repo")
        if not os.access(native_runner, os.X_OK):
            raise ValueError("--native-runner is not executable")
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
            "native_runner_sha256": file_sha256(native_runner),
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
                f"exec /usr/bin/time -l {shlex.quote(str(native_runner))} \"$@\"\n"
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
        except BaseException:
            self.raw_log.close()
            self.request_log.close()
            self.wrapper.unlink(missing_ok=True)
            raise
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        try:
            self.record({"event": "launch", "command": command, "identity": self.identity})
        except BaseException:
            self.close()
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
                return
            self.closed = True
            if self.process.stdin:
                try:
                    self.process.stdin.close()
                except OSError:
                    pass
            try:
                # EOF runs the adapter's finally/native.close and lets both
                # timers write their real peak RSS on an orderly shutdown.
                self.process.wait(timeout=25)
                shutdown = "orderly_eof"
            except subprocess.TimeoutExpired:
                shutdown = "owned_process_group_sigterm"
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    shutdown = "sigterm_did_not_finish; no_sigkill_used"
            # A crashed adapter or a timed native wrapper can leave descendants
            # alive after the group leader exits. This group was created solely
            # for this launch; terminating leftovers cannot touch other services.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.record({"event": "shutdown", "method": shutdown, "exit_code": self.process.poll()})
            self.process.stdout.close()
            self.raw_log.close()
            with self.log_lock:
                self.logs_closed = True
                self.request_log.close()
            self.wrapper.unlink(missing_ok=True)


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
