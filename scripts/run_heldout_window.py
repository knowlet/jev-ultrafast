"""Own one frozen noul_v1/Q4 heldout run in a fixed 900-second local window.

Default writes a receipt only. --execute starts existing original S1 and text
runtimes on numeric loopback. No model download, readiness inference, candidate
native binary, personal browser profile, environment secrets, or policy tuning.
The 840-second work deadline includes startup; 60 seconds are reserved for owned
cleanup and partial evidence. Interrupted/unexecuted runs keep the 24 denominator.
"""

import argparse
import errno
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_heldout_benchmark as heldout  # noqa: E402

TOTAL_SECONDS, WORK_SECONDS, CLEANUP_SECONDS = 900, 840, 60
POLICY, QUANTIZATION, TEXT_MODEL = "noul_v1", "Q4_K_M", "system-one-trained-Q4_K_M"


class WorkExpired(TimeoutError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args):
        raise ValueError("Owned loopback health must not redirect")


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def clean_environment(output):
    cache = output / "cache"
    for name in ("tmp", "huggingface", "s1"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1",
               HF_HOME=str(cache / "huggingface"), XDG_CACHE_HOME=str(cache), TMPDIR=str(cache / "tmp"))
    return env


def configuration(args):
    s1, chrome = args.s1_repo.resolve(strict=True), args.chrome_path.resolve(strict=True)
    if not chrome.is_file() or not os.access(chrome, os.X_OK):
        raise ValueError("An explicit existing Chromium executable is required")
    package = s1 / "artifacts/exports/s1-boolq-Q4_K_M"
    manifest = json.loads((package / "conversion-manifest.json").read_text())
    calibration = json.loads((package / "s1_config.json").read_text())
    if (manifest.get("source_hub_repository") != "knowlet/Gemma-4-12B-Unified-System-One"
            or manifest.get("quantization", {}).get("language_model") != QUANTIZATION
            or manifest.get("status") != "converted_and_structurally_verified"):
        raise ValueError("Only the existing user-trained, verified Q4 System One package is allowed")
    temperature = calibration.get("temperature")
    if (type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0
            or calibration.get("runtime") != "llama.cpp" or calibration.get("runtime_format") != "gguf"
            or calibration.get("conversion_manifest_sha256") != digest(package / "conversion-manifest.json")):
        raise ValueError("The existing GGUF-owned calibration must bind this exact conversion manifest")
    identities = {item["path"]: item for item in manifest["files"]}
    selected = {}
    for key in ("model_file", "mmproj_file"):
        name = manifest[key]
        if not isinstance(name, str) or Path(name).name != name or name not in identities:
            raise ValueError("Selected model/projector must be named local manifest files")
        path = (package / name).resolve(strict=True)
        if path.parent != package or path.stat().st_size != identities[name]["size_bytes"]:
            raise ValueError("Selected model/projector path or size differs from the trained manifest")
        selected[key] = path
    native = (s1 / "artifacts/gguf-runtime/bin/s1-gguf").resolve(strict=True)
    helper = (s1 / "artifacts/export-tools/llama.cpp/build/bin/llama-server").resolve(strict=True)
    for path in (native, helper, s1 / ".venv/bin/python", s1 / "scripts/gguf_adapter.py"):
        if not path.is_file() or not os.access(path, os.X_OK if path.name != "gguf_adapter.py" else os.R_OK):
            raise ValueError("Existing original runtime dependencies are required; no installation attempted")
    return {"s1": s1, "chrome": chrome, "package": package, "manifest": manifest,
            "calibration": calibration, "native": native, "helper": helper, **selected}


def source_snapshot(config, freeze):
    paths = {Path(__file__).resolve(), ROOT / "scripts/run_heldout_benchmark.py",
             ROOT / "scripts/local_benchmark.py", ROOT / "scripts/local_s1_service.py",
             ROOT / "jev_ultrafast/snapshot.js", *ROOT.glob("jev_ultrafast/*.py"),
             config["s1"] / "scripts/gguf_adapter.py", config["native"], config["helper"],
             config["package"] / "conversion-manifest.json", config["package"] / "s1_config.json",
             ROOT / "benchmark/heldout-freeze.json", *(ROOT / path for path in freeze["files"]),
             config["s1"] / "tools/gguf/s1_gguf.cpp"}
    paths.update((config["s1"] / "src/s1").rglob("*.py"))
    paths.update(path for path in config["package"].iterdir() if path.is_file() and path.suffix != ".gguf")
    return {str(path): digest(path) for path in sorted(paths)}


def unchanged(snapshot):
    return all(Path(path).is_file() and digest(path) == expected for path, expected in snapshot.items())


def commands(config, output, decision_port, text_port):
    decision = [sys.executable, str(ROOT / "scripts/local_s1_service.py"),
                "--s1-repo", str(config["s1"]), "--model", str(config["package"]),
                "--native-runner", str(config["native"]), "--port", str(decision_port),
                "--ctx-size", "4096", "--batch-size", "512", "--threads", "4",
                "--cache-dir", str(output / "cache/s1"),
                "--raw-log", str(output / "native-decision-time.log"),
                "--request-log", str(output / "native-requests.jsonl")]
    text = [str(config["helper"]), "--model", str(config["model_file"]),
            "--host", "127.0.0.1", "--port", str(text_port), "--alias", TEXT_MODEL,
            "--ctx-size", "4096", "--batch-size", "512", "--ubatch-size", "512", "--threads", "4",
            "--gpu-layers", "99", "--parallel", "1", "--seed", "42", "--temp", "0",
            "--reasoning", "off", "--no-cache-prompt", "--no-webui", "--offline"]
    run = [sys.executable, str(ROOT / "scripts/run_heldout_benchmark.py"), "--execute", "--policy", POLICY,
           "--manifest", str(ROOT / "benchmark/heldout-plan.json"),
           "--freeze", str(ROOT / "benchmark/heldout-freeze.json"),
           "--output", str(output / "heldout"), "--model-metadata", str(output / "model-metadata.json"),
           "--chrome-path", str(config["chrome"]), "--decision-url", f"http://127.0.0.1:{decision_port}/decide",
           "--text-base-url", f"http://127.0.0.1:{text_port}/v1", "--text-model", TEXT_MODEL, "--fixture-port", "0"]
    return {"decision-service": decision, "text-service": text, "runner": run}


def reserve_ports():
    reservations = []
    try:
        for _ in range(2):
            stream = socket.socket()
            stream.bind(("127.0.0.1", 0))
            reservations.append(stream)
        return [stream.getsockname()[1] for stream in reservations]
    finally:
        for stream in reservations:
            stream.close()


class OwnedProcesses:
    """Only explicitly launched sessions and fresh registered child sessions."""
    def __init__(self, output, env):
        self.output, self.env, self.children, self.groups, self.handles = output, env, {}, {}, []
        self.events = []
        self.denied_groups, self.retired_groups, self.spawning, self.abort_pending = set(), set(), False, False

    def start(self, name, command):
        log = (self.output / f"{name}.log").open("x")
        self.handles.append(log)
        self.spawning = True
        process = None
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=self.env, stdout=log, stderr=log,
                                       start_new_session=True)
            self.children[name] = process
            self.groups[process.pid] = {"name": name, "proof": "owned Popen(start_new_session=True)"}
            self.events.append({"event": "spawn", "name": name, "pid": process.pid, "owned_pgid": process.pid,
                                "utc": utc(), "argv": command})
        except BaseException:
            if process is not None:
                self.children[name] = process
                self.groups[process.pid] = {"name": name, "proof": "owned Popen returned before setup failure"}
                self.send_group(process.pid, signal.SIGTERM)
            raise
        finally:
            self.spawning = False
        if self.abort_pending:
            raise WorkExpired("Abort deferred only until the new owned process was registered")
        return process

    def poll(self, name):
        process = self.children[name]
        if process.returncode is None:
            try:
                pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    process.returncode = os.waitstatus_to_exitcode(status)
                    self.events.append({"event": "exit", "name": name, "pid": pid,
                                        "exit_code": process.returncode, "peak_rss_bytes": usage.ru_maxrss,
                                        "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
                                        "scope": "macOS wait4 direct owned PID; peaks not additive", "utc": utc()})
            except ChildProcessError:
                self.events.append({"event": "usage_unavailable", "name": name, "pid": process.pid})
                process.returncode = process.poll()
        return process.returncode

    def register(self, pid, name, proof):
        if type(pid) is not int or pid <= 1 or os.getpgid(pid) != pid:
            raise ValueError("Registered owned child must have its own known process group")
        self.groups[pid] = {"name": name, "proof": proof}
        self.events.append({"event": "register", "name": name, "owned_pgid": pid, "proof": proof, "utc": utc()})

    def send_pid(self, name, sig):
        if name in self.children and self.poll(name) is None:
            pid = self.children[name].pid
            try:
                os.kill(pid, sig)
                self.events.append({"event": "signal_pid", "name": name, "pid": pid, "signal": sig, "utc": utc()})
            except ProcessLookupError:
                pass
            except PermissionError:
                self.denied_groups.add(pid)
                self.events.append({"event": "control_denied", "name": name, "pid": pid, "utc": utc()})
                raise

    def send_group(self, pgid, sig):
        if pgid not in self.groups:
            raise ValueError("Refusing to signal an unregistered process group")
        if pgid in self.denied_groups or pgid in self.retired_groups:
            return  # Never retry/reroute an actual process-control permission denial.
        try:
            os.killpg(pgid, sig)
            self.events.append({"event": "signal_group", "owned_pgid": pgid, "signal": sig, "utc": utc()})
        except ProcessLookupError:
            self.retired_groups.add(pgid)
        except PermissionError as exc:
            self.denied_groups.add(pgid)
            self.events.append({"event": "control_denied", "owned_pgid": pgid, "message": str(exc), "utc": utc()})
            raise

    def alive_groups(self):
        alive = []
        for pgid in self.groups:
            if pgid in self.retired_groups or pgid in self.denied_groups:
                continue
            try:
                os.killpg(pgid, 0)
                alive.append(pgid)
            except ProcessLookupError:
                self.retired_groups.add(pgid)
            except PermissionError:
                self.denied_groups.add(pgid)
                self.events.append({"event": "control_denied", "owned_pgid": pgid, "utc": utc()})
        return alive

    def kill_all(self):
        for pgid in list(self.groups):
            try:
                self.send_group(pgid, signal.SIGKILL)
            except PermissionError:
                pass

    def close_logs(self):
        for log in self.handles:
            log.close()


def listening_event(path, url):
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Owned service startup log exceeded bounded identity evidence")
    for line in path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event") == "listening" and event.get("url") == url:
            return event
    return None


def verify_health(health, listening, config, sources):
    expected = {"s1_repo": str(config["s1"]), "model_package": str(config["package"]),
                "model_file": str(config["model_file"]), "native_runner": str(config["native"]),
                "native_runner_sha256": sources[str(config["native"])],
                "bridge_source_sha256": sources[str(ROOT / "scripts/local_s1_service.py")],
                "adapter_source_sha256": sources[str(config["s1"] / "scripts/gguf_adapter.py")],
                "conversion_manifest_sha256": sources[str(config["package"] / "conversion-manifest.json")],
                "calibration_sha256": sources[str(config["package"] / "s1_config.json")],
                "temperature": config["calibration"]["temperature"], "quantization": config["manifest"]["quantization"],
                "source_hub_repository": "knowlet/Gemma-4-12B-Unified-System-One",
                "reuse_prefix": False, "ctx_size": 4096, "batch_size": 512, "threads": 4}
    if (not isinstance(health, dict) or not isinstance(listening, dict)
            or health.get("status") not in ("starting", "ready")
            or type(health.get("adapter_pid")) is not int or health["adapter_pid"] <= 1
            or any(health.get(key) != value or listening.get(key) != value for key, value in expected.items())
            or listening.get("adapter_pid") != health["adapter_pid"]):
        raise ValueError("Fresh owned bridge listening/health identity differs; no runner launched")
    files = {item["path"]: item for item in config["manifest"]["files"]}
    if (health.get("model_manifest_identity") != files[config["manifest"]["model_file"]]
            or health.get("mmproj_manifest_identity") != files[config["manifest"]["mmproj_file"]]):
        raise ValueError("Actual bridge selected a different trained artifact")
    return health["adapter_pid"]


def health_get(url, owner, name, deadline):
    if not url.startswith("http://127.0.0.1:"):
        raise ValueError("Only the declared numeric loopback health URL is accepted")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    while time.monotonic() < deadline:
        if owner.poll(name) is not None:
            raise RuntimeError(f"Owned {name} exited during startup; see fresh raw log")
        try:
            with opener.open(url, timeout=min(2, max(0.001, deadline - time.monotonic()))) as response:
                data = response.read(1024 * 1024 + 1)
                if response.status != 200 or len(data) > 1024 * 1024:
                    raise ValueError("Invalid bounded loopback health response")
                if owner.poll(name) is not None:
                    raise RuntimeError("Owned service exited before health verification")
                return json.loads(data)
        except urllib.error.HTTPError as exc:
            body = exc.read(1024 * 1024).decode(errors="replace").lower()
            if (exc.code != 503 or name != "text-service" or "loading model" not in body
                    or any(word in body for word in ("permission denied", "operation not permitted", "access denied"))):
                raise RuntimeError(f"Owned {name} actual health HTTP {exc.code}; no retry or reroute") from exc
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        except urllib.error.URLError as exc:
            reason = exc.reason
            if not isinstance(reason, OSError) or reason.errno != errno.ECONNREFUSED:
                raise RuntimeError(f"Owned {name} startup transport failed; no retry or reroute: {reason}") from exc
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        except (ConnectionRefusedError, TimeoutError):
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    raise WorkExpired(f"Owned {name} health deadline expired; no readiness inference sent")


def wait_runner(owner, deadline):
    while time.monotonic() < deadline:
        result = owner.poll("runner")
        if result is not None:
            return result
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    raise WorkExpired("840-second work budget expired, including service startup")


def cleanup(owner, deadline):
    owner.send_pid("runner", signal.SIGINT)
    # Let the runner save one failed in-flight row and its native-call prefix.
    soft_end = min(time.monotonic() + 35, deadline - 20)
    while "runner" in owner.children and owner.poll("runner") is None and time.monotonic() < soft_end:
        time.sleep(0.05)
    if "runner" in owner.children and owner.poll("runner") is None:
        owner.send_group(owner.children["runner"].pid, signal.SIGTERM)
    for name in ("decision-service", "text-service"):
        owner.send_pid(name, signal.SIGTERM)
    soft_end = deadline - 6
    while time.monotonic() < soft_end and owner.alive_groups():
        for name in owner.children:
            owner.poll(name)
        time.sleep(0.05)
    for pgid in owner.alive_groups():
        owner.send_group(pgid, signal.SIGTERM)
    while time.monotonic() < deadline - 3 and owner.alive_groups():
        for name in owner.children:
            owner.poll(name)
        time.sleep(0.05)
    for pgid in owner.alive_groups():
        owner.send_group(pgid, signal.SIGKILL)
    while time.monotonic() < deadline and owner.alive_groups():
        for name in owner.children:
            owner.poll(name)
        time.sleep(0.02)
    remaining = owner.alive_groups()
    unknown = sorted(owner.denied_groups)
    return {"remaining_owned_groups": remaining, "group_registry": owner.groups,
            "events": owner.events,
            "denied_groups": unknown, "unknown_owned_groups": unknown,
            "cleanup_liveness_verified": not remaining and not unknown,
            "remaining_groups_scope": "Known-present groups only; denied groups have unknown liveness",
            "chromium_scope": "Chromium inherits the owned runner session; Chrome PID is not treated as a PGID",
            "profiles": "Runner owns fresh profiles; forced-stop remnants retained only in this fresh output"}


def incomplete_summary(output, plan, reason):
    """If forced before runner summary save, reaggregate only persisted actual rows."""
    destination = output / "heldout"
    destination.mkdir(exist_ok=True)
    summary_path = destination / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    records, path = [], destination / "runs.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break
    result = heldout.summarize(records, plan, POLICY,
                              batch_error={"type": "WindowIncomplete", "message": reason})
    result["forced_report_scope"] = "Persisted complete task rows only; in-flight wire evidence remains in native logs"
    save(summary_path, result)
    return result


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--s1-repo", type=Path, required=True)
    cli.add_argument("--chrome-path", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--execute", action="store_true")
    return cli


def launch_main():
    started = time.monotonic()
    cli, args = parser(), None
    args = cli.parse_args()
    owner = None
    work_end, total_end = started + WORK_SECONDS, started + TOTAL_SECONDS
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM)}

    def work_expired(*_args):
        if owner is not None and owner.spawning:
            owner.abort_pending = True
            signal.signal(signal.SIGALRM, hard_expired)
            signal.setitimer(signal.ITIMER_REAL, max(0.001, total_end - time.monotonic()))
            return
        raise WorkExpired("Fixed 840-second work window expired")

    def hard_expired(*_args):
        try:
            if owner is not None:
                owner.kill_all()
        finally:
            os._exit(124)  # Never extend the900s process cap, even if cleanup itself fails.

    if args.execute:
        for sig in previous:
            signal.signal(sig, work_expired)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, work_end - time.monotonic()))
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT / "benchmark/results").resolve()):
        cli.error("--output must be fresh beneath this checkout's benchmark/results; no existing evidence overwritten")
    config = configuration(args)
    plan, freeze = heldout.verify_freeze(ROOT / "benchmark/heldout-plan.json", ROOT / "benchmark/heldout-freeze.json")
    if (len(heldout.schedule(plan, POLICY)) != 24 or plan["protocol"]["max_decision_requests_per_variant"] != 288
            or plan["model_seed"] != 42 or plan["temperature"] != 0):
        cli.error("Frozen 24-run/288-physical-request/seed42 protocol differs")
    sources = source_snapshot(config, freeze)
    output.mkdir(parents=True)
    receipt = {"created_at_utc": utc(), "execution_requested": args.execute, "policy": POLICY,
               "runtime": "original S1 Q4 GGUF, no prefix candidate", "total_seconds": TOTAL_SECONDS,
               "work_seconds": WORK_SECONDS, "cleanup_reserved_seconds": CLEANUP_SECONDS,
               "wall_scope": "Launcher main entry through startup, fixed suite and cleanup",
               "expected_runs": 24, "measured_physical_request_cap": 288, "development_warmups": 2,
               "warmup_scope": "Two fixed development observations, each choose plus text; separate from measured cap",
               "ready_inferences": 0, "scheduled_runs": heldout.schedule(plan, POLICY), "freeze": freeze,
               "source_sha256": sources, "model_manifest": config["manifest"], "calibration": config["calibration"],
               "artifact_digest_scope": "Calibrated GGUF digest; original adapter verifies actual bytes before score",
               "bookend_scope": "Frozen inputs, runtime/source, processor sidecars; no post-run GGUF SHA check",
               "startup": {}}
    receipt_path = output / "window-receipt.json"
    save(receipt_path, receipt)  # Before any socket or process.
    if not args.execute:
        print("Frozen window receipt written; no port, service, model or browser started.", flush=True)
        return 0
    owner = OwnedProcesses(output, clean_environment(output))
    result = 1
    try:
        if not unchanged(sources):
            raise ValueError("Source changed after the preexecution receipt")
        decision_port, text_port = reserve_ports()
        argv = commands(config, output, decision_port, text_port)
        receipt["commands"] = argv
        save(receipt_path, receipt)
        began = time.monotonic()
        owner.start("decision-service", argv["decision-service"])
        health = health_get(f"http://127.0.0.1:{decision_port}/healthz", owner,
                            "decision-service", min(work_end, began + 30))
        listening = listening_event(output / "decision-service.log", f"http://127.0.0.1:{decision_port}")
        pid = verify_health(health, listening, config, sources)
        if owner.poll("decision-service") is not None:
            raise RuntimeError("Owned bridge exited before nested child registration")
        owner.register(pid, "decision-adapter", "fresh owned bridge stdout listening matches verified /healthz")
        receipt["startup"]["decision_http_identity_ms"] = (time.monotonic() - began) * 1000
        receipt["decision_service_identity"] = health
        save(receipt_path, receipt)
        began = time.monotonic()
        owner.start("text-service", argv["text-service"])
        text_health = health_get(f"http://127.0.0.1:{text_port}/health", owner,
                                "text-service", min(work_end, began + 90))
        models = health_get(f"http://127.0.0.1:{text_port}/v1/models", owner, "text-service", min(work_end, began + 90))
        if {item["id"] for item in models.get("data", [])} != {TEXT_MODEL}:
            raise ValueError("Owned helper model alias identity differs")
        receipt["startup"]["text_model_ready_ms"] = (time.monotonic() - began) * 1000
        metadata = {"decision_service_identity": health, "manifest": config["manifest"],
                    "calibration": config["calibration"], "quantization": QUANTIZATION, "policy": POLICY,
                    "artifact_path": str(config["model_file"]), "text_runtime_models": models,
                    "text_health": text_health, "source_sha256": sources,
                    "settings": {"context": 4096, "batch": 512, "threads": 4, "gpu_layers": 99,
                                 "seed": 42, "text_temperature": 0, "text_max_tokens": 128},
                    "startup": receipt["startup"], "inference_ready_claim": False}
        save(output / "model-metadata.json", metadata)
        if not unchanged(sources):
            raise ValueError("Source changed before the frozen runner launch")
        owner.start("runner", argv["runner"])
        result = wait_runner(owner, work_end)
    except BaseException as exc:
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        receipt["work_interrupted"] = isinstance(exc, (WorkExpired, KeyboardInterrupt))
    finally:
        for sig in previous:
            signal.signal(sig, hard_expired)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, total_end - time.monotonic()))
        try:
            receipt["cleanup"] = cleanup(owner, min(total_end - 2, time.monotonic() + CLEANUP_SECONDS - 2))
            summary = incomplete_summary(output, plan, "Window ended before runner saved a full summary")
            receipt["summary"] = {key: summary[key] for key in
                                  ("expected_runs", "executed_runs", "successful_runs", "unexecuted_runs", "complete")}
            receipt["sources_unchanged_after"] = unchanged(sources)
            if (not receipt["sources_unchanged_after"] or receipt["cleanup"]["remaining_owned_groups"]
                    or receipt["cleanup"]["denied_groups"]
                    or receipt["cleanup"].get("unknown_owned_groups")
                    or receipt["cleanup"].get("cleanup_liveness_verified") is not True
                    or not summary["complete"]):
                result = 1
        except BaseException as exc:
            owner.kill_all()
            receipt["cleanup_failure"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            result = 1
        receipt["owned_events"] = owner.events
        receipt["elapsed_seconds"] = time.monotonic() - started
        save(receipt_path, receipt)
        owner.close_logs()
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return result


def main():
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM)}
    try:
        return launch_main()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
