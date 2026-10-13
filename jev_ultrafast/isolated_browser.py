"""An owned headless Chromium process for local fixtures, never the user's browser.

Use ``with IsolatedChromium(executable=..., profile_root=..., fixture_origin=...) as chromium``
and pass ``browser_factory=chromium.browser`` to Agent. The executable must already exist;
this module never downloads a browser. ``profile_root`` is a caller-owned task directory.
Only the exact numeric-loopback HTTP fixture origin, about:blank, and data URLs are allowed.
Fetch interception checks page and worker requests. A local refusing proxy is an additional
fail-closed boundary for other Chromium network paths; no system proxy settings are changed.
After close, ``resource_usage`` retains wait4 peak RSS/CPU of the owned main PID, with its scope;
it is not a simultaneous total of browser subprocess memory.
"""

import ipaddress
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .browser import Browser


def normalize_fixture_origin(url):
    """Validate and normalize a numeric-loopback HTTP origin, without DNS lookup."""
    parsed = urlsplit(url)
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Fixture origin must be numeric-loopback HTTP with an explicit port") from exc
    if (
        parsed.scheme != "http" or not address.is_loopback or port is None
        or parsed.username is not None or parsed.password is not None
        or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
    ):
        raise ValueError("Fixture origin must be numeric-loopback HTTP with an explicit port")
    host = f"[{address}]" if address.version == 6 else str(address)
    return f"http://{host}:{port}"


def allowed_url(url, origin):
    if not isinstance(url, str):
        return False
    if url == "about:blank" or url.startswith("data:"):
        return True
    try:
        parsed = urlsplit(url)
        return normalize_fixture_origin(f"{parsed.scheme}://{parsed.netloc}") == origin
    except (ValueError, TypeError):
        return False


class _RefusingProxy(BaseHTTPRequestHandler):
    """Never forward or resolve a request, including HTTPS CONNECT and WebSocket."""

    def refuse(self):
        self.server.blocked_requests += 1
        self.send_response(403)
        self.send_header("Connection", "close")
        self.send_header("Content-Length", "0")
        self.end_headers()
        self.close_connection = True

    do_CONNECT = do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = refuse

    def log_message(self, *_args):
        pass


class CDPTransport:
    """Synchronous cdp(method, session_id=..., **params) compatible transport.

    Incoming Fetch/Target events are serviced while awaiting every command, including
    nested command responses. No external command or paused request is retried.
    """

    def __init__(self, endpoint, origin, *, websocket=None, timeout=15):
        parsed = urlsplit(endpoint)
        try:
            endpoint_local = ipaddress.ip_address(parsed.hostname or "").is_loopback
            endpoint_port = parsed.port
        except ValueError:
            endpoint_local, endpoint_port = False, None
        if (
            parsed.scheme != "ws" or not endpoint_local or endpoint_port is None
            or parsed.username is not None or parsed.password is not None
            or not parsed.path.startswith("/devtools/browser/") or parsed.query or parsed.fragment
        ):
            raise ValueError("CDP endpoint must be an owned numeric-loopback browser websocket")
        if websocket is None:
            from websockets.sync.client import connect

            websocket = connect(endpoint, open_timeout=timeout, close_timeout=2, proxy=None, max_size=16 * 1024 * 1024)
        self.websocket = websocket
        self.origin = normalize_fixture_origin(origin)
        self.timeout = timeout
        self.next_id = 0
        self.pending = {}
        self.configured_sessions = set()
        self.lock = threading.RLock()
        self.blocked_requests = 0
        self.calls = []

    def _prepare_session(self, session_id):
        if session_id in self.configured_sessions:
            return
        self.configured_sessions.add(session_id)
        try:
            self("Fetch.enable", session_id=session_id, patterns=[{"urlPattern": "*", "requestStage": "Request"}])
            self("Runtime.runIfWaitingForDebugger", session_id=session_id)
        except Exception:
            self.configured_sessions.discard(session_id)
            raise

    def _event(self, message):
        method, params = message.get("method"), message.get("params", {})
        if method == "Target.attachedToTarget":
            self._prepare_session(params["sessionId"])
        elif method == "Target.detachedFromTarget":
            self.configured_sessions.discard(params.get("sessionId"))
        elif method == "Fetch.requestPaused":
            session_id = message.get("sessionId")
            if allowed_url(params["request"]["url"], self.origin):
                self("Fetch.continueRequest", session_id=session_id, requestId=params["requestId"])
            else:
                self.blocked_requests += 1
                self(
                    "Fetch.failRequest", session_id=session_id,
                    requestId=params["requestId"], errorReason="BlockedByClient",
                )

    def __call__(self, method, *, session_id=None, **params):
        if method in {"Page.navigate", "Target.createTarget"} and not allowed_url(params.get("url", ""), self.origin):
            raise ValueError("Isolated browser accepts only the fixture origin, about:blank, or data URLs")
        with self.lock:
            self.next_id += 1
            request_id = self.next_id
            request = {"id": request_id, "method": method, "params": params}
            if session_id is not None:
                request["sessionId"] = session_id
            started = time.perf_counter()
            self.websocket.send(json.dumps(request))
            deadline = time.monotonic() + self.timeout
            try:
                while request_id not in self.pending:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"CDP command timed out: {method}")
                    message = json.loads(self.websocket.recv(timeout=remaining))
                    if "id" in message:
                        self.pending[message["id"]] = message
                    else:
                        self._event(message)
                response = self.pending.pop(request_id)
                if "error" in response:
                    raise RuntimeError(f"CDP {method} failed: {response['error'].get('message', 'unknown error')}")
                result = response.get("result", {})
                if method == "Target.attachToTarget":
                    self._prepare_session(result["sessionId"])
                return result
            finally:
                self.calls.append({"method": method, "latency_ms": (time.perf_counter() - started) * 1000})

    def close(self):
        self.websocket.close()


class IsolatedChromium:
    """Own one process, disposable profile, refusing proxy, and CDP connection."""

    def __init__(self, *, executable, profile_root, fixture_origin, startup_timeout=15):
        self.executable = Path(executable).resolve()
        self.profile_root = Path(profile_root).resolve()
        self.origin = normalize_fixture_origin(fixture_origin)
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise ValueError("Supply an existing executable Chromium path; no browser is downloaded")
        if not self.profile_root.is_dir():
            raise ValueError("profile_root must be an existing caller-owned task directory")
        if startup_timeout <= 0 or startup_timeout > 60:
            raise ValueError("startup_timeout must be between 0 and 60 seconds")
        self.startup_timeout = startup_timeout
        self.process = self.transport = self.profile = self.proxy = self.proxy_thread = self.stderr_file = None
        self.startup_ms = None
        self.resource_usage = None

    def _poll_owned_usage(self):
        """Reap only our direct child PID; Popen.poll/wait would discard its rusage."""
        if self.process.returncode is not None:
            if self.resource_usage is None:
                self.resource_usage = {
                    "peak_rss_bytes": None, "user_seconds": None, "system_seconds": None,
                    "exit_code": self.process.returncode,
                    "scope": "Owned main PID was already reaped; wait4 resource usage unavailable",
                }
            return True
        try:
            pid, status, usage = os.wait4(self.process.pid, os.WNOHANG)
        except ChildProcessError:
            # Another owner must not wait on this Popen. Preserve unavailable data honestly if it happened.
            self.resource_usage = {
                "peak_rss_bytes": None, "user_seconds": None, "system_seconds": None,
                "exit_code": self.process.returncode,
                "scope": "Owned main PID was already reaped; wait4 resource usage unavailable",
            }
            return True
        if pid == 0:
            return False
        self.process.returncode = os.waitstatus_to_exitcode(status)
        self.resource_usage = {
            "peak_rss_bytes": int(usage.ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
            "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
            "exit_code": self.process.returncode,
            "scope": (
                "macOS wait4 owned main PID; not simultaneous process-tree total" if sys.platform == "darwin"
                else "wait4 owned main PID; not simultaneous process-tree total"
            ),
        }
        return True

    def _stop_owned_process(self):
        if self._poll_owned_usage():
            return
        for shutdown_signal in (signal.SIGTERM, signal.SIGKILL):
            try:
                # Popen.terminate() calls poll() first and can consume wait4's resource data.
                os.kill(self.process.pid, shutdown_signal)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if self._poll_owned_usage():
                    return
                time.sleep(0.02)
        raise TimeoutError("Owned Chromium did not exit after bounded termination")

    def start(self):
        if self.process is not None:
            raise RuntimeError("Isolated Chromium already started")
        self.resource_usage = None
        started = time.perf_counter()
        try:
            self.profile = tempfile.TemporaryDirectory(prefix="jev-isolated-", dir=self.profile_root)
            self.proxy = ThreadingHTTPServer(("127.0.0.1", 0), _RefusingProxy)
            self.proxy.daemon_threads = True
            self.proxy.blocked_requests = 0
            self.proxy_thread = threading.Thread(target=self.proxy.serve_forever, daemon=True)
            self.proxy_thread.start()
            proxy_port = self.proxy.server_address[1]
            self.stderr_file = open(Path(self.profile.name) / "chromium.stderr.log", "wb")
            # The automatic localhost bypass is removed; only the explicit HTTP fixture origin bypasses the proxy.
            arguments = [
                str(self.executable), "--headless=new", f"--user-data-dir={self.profile.name}",
                "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=0",
                "--no-first-run", "--no-default-browser-check", "--disable-background-networking",
                "--disable-extensions", "--disable-default-apps", "--disable-sync", "--disable-component-update",
                "--disable-domain-reliability", "--disable-client-side-phishing-detection", "--disable-quic",
                "--disable-breakpad", "--metrics-recording-only",
                "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
                "--disable-features=MediaRouter,OptimizationHints,AutofillServerCommunication,DnsOverHttps",
                f"--proxy-server=http://127.0.0.1:{proxy_port}", f"--proxy-bypass-list=<-loopback>;{self.origin}",
                "about:blank",
            ]
            # Forward only ordinary execution locale/path, never API keys or proxy credentials.
            child_env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ}
            self.process = subprocess.Popen(
                arguments, stdout=subprocess.DEVNULL, stderr=self.stderr_file, env=child_env,
            )
            active_port = Path(self.profile.name) / "DevToolsActivePort"
            deadline = time.monotonic() + self.startup_timeout
            while not active_port.is_file():
                if self._poll_owned_usage():
                    raise RuntimeError(f"Owned Chromium exited during startup (code {self.process.returncode})")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Owned Chromium did not expose its loopback DevTools endpoint")
                time.sleep(0.02)
            lines = active_port.read_text().splitlines()
            if len(lines) < 2 or not lines[0].isdigit() or not 1 <= int(lines[0]) <= 65535:
                raise RuntimeError("Invalid owned Chromium DevTools endpoint")
            if not lines[1].startswith("/devtools/browser/") or any(c.isspace() for c in lines[1]):
                raise RuntimeError("Invalid owned Chromium DevTools browser path")
            self.transport = CDPTransport(f"ws://127.0.0.1:{lines[0]}{lines[1]}", self.origin)
            self.transport("Target.setAutoAttach", autoAttach=True, waitForDebuggerOnStart=True, flatten=True)
            self.startup_ms = (time.perf_counter() - started) * 1000
            return self
        except Exception:
            self.close()
            raise

    def browser(self, url):
        if self.transport is None:
            raise RuntimeError("Start the owned Chromium context before creating a browser")
        if not allowed_url(url, self.origin):
            raise ValueError("Browser URL must belong to the local fixture")
        return Browser(url, transport=self.transport)

    def close(self):
        try:
            if self.transport is not None:
                self.transport.close()
        finally:
            self.transport = None
            if self.process is not None:
                self._stop_owned_process()
                self.process = None
            if self.proxy is not None:
                self.proxy.shutdown()
                self.proxy.server_close()
                self.proxy = None
            if self.proxy_thread is not None:
                self.proxy_thread.join(timeout=2)
                self.proxy_thread = None
            if self.stderr_file is not None:
                self.stderr_file.close()
                self.stderr_file = None
            if self.profile is not None:
                self.profile.cleanup()
                self.profile = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_args):
        self.close()
