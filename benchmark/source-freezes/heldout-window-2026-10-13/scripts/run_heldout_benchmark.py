"""Run the frozen local heldout suite only with explicit --execute.

Default behavior writes an execution plan without binding a port or invoking a model
or browser. Existing local decision/text services and an explicit existing Chromium
executable are required. No downloads, .env loading, external sites or paid providers.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmark.heldout_fixture import READ_FRESH_DOM, render_case, verify_case  # noqa: E402
from scripts.local_benchmark import (  # noqa: E402
    CLICK_GOAL,
    CLICK_PAGE,
    TEXT_CONTEXT,
    BudgetTimeout,
    ModelRecorder,
    NativeRequestBudget,
    TimedBrowser,
    candidate_identity,
    deadline,
    distribution,
    error_record,
    local_endpoint,
    native_calls_for,
    peak_python_rss_bytes,
    scoring_calls_for,
)

# A candidate is added only after development evidence and explicit root coordination.
# Attribute names are fixed here; CLI arguments never import arbitrary modules or code.
APPROVED_POLICIES = {"baseline": "choose", "noul_v1": "choose_noul"}
FROZEN_MANIFEST_SHA256 = "e94986f9429889379b368c5adcbf4ab3fcfc00487879ee54be5133fb37f91196"
FROZEN_FREEZE_SHA256 = "ccbc97f8ecf5b29a3eac75bc5702695f72dcfbb2dc7ec33f217a1d34070c9041"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_freeze(plan_path, freeze_path):
    """Validate every registered file before a socket, subprocess or model request."""
    plan_path, freeze_path = Path(plan_path).resolve(), Path(freeze_path).resolve()
    if sha256(plan_path) != FROZEN_MANIFEST_SHA256:
        raise ValueError("Heldout manifest differs from the preregistered frozen SHA-256")
    if sha256(freeze_path) != FROZEN_FREEZE_SHA256:
        raise ValueError("Freeze registry differs from the preregistered frozen SHA-256")
    plan, freeze = json.loads(plan_path.read_text()), json.loads(freeze_path.read_text())
    if plan["schema"] != "jev-heldout-plan-v1" or freeze["schema"] != "jev-heldout-freeze-v1":
        raise ValueError("Unsupported heldout plan/freeze schema")
    if len(plan["tasks"]) != 8 or len(set(plan["task_order"])) != 8:
        raise ValueError("The fixed suite must contain all eight cases")
    if set(plan["task_order"]) != {case["id"] for case in plan["tasks"]}:
        raise ValueError("Task order must include each fixed task exactly once")
    required = {
        "benchmark/heldout-plan.json", "benchmark/heldout_fixture.py", "tests/test_heldout_fixture.py",
        "docs/heldout-preregistration.md",
        *(f"benchmark/heldout-fixtures/{case['id']}.html" for case in plan["tasks"]),
    }
    if set(freeze["files"]) != required:
        raise ValueError("Freeze must cover exactly the registered task, checker and fixture files")
    for relative, expected in freeze["files"].items():
        path = REPO / relative
        if not path.resolve().is_relative_to(REPO.resolve()) or sha256(path) != expected:
            raise ValueError(f"Frozen source/fixture changed: {relative}")
    for case in plan["tasks"]:
        document = render_case(case).encode()
        path = f"benchmark/heldout-fixtures/{case['id']}.html"
        if hashlib.sha256(document).hexdigest() != freeze["files"][path]:
            raise ValueError(f"Rendered fixture differs from the frozen HTML: {case['id']}")
    return plan, freeze


def schedule(plan, policy):
    return [
        {"task_id": task_id, "repetition": repetition, "policy": policy}
        for repetition in range(plan["protocol"]["repeats_per_task_per_variant"])
        for task_id in plan["task_order"]
    ]


def resolve_policy(name):
    from jev_ultrafast import model

    if name not in APPROVED_POLICIES:
        raise ValueError("Policy has not been fixed and approved from development evidence")
    return getattr(model, APPROVED_POLICIES[name])


@contextmanager
def local_configuration(args, plan):
    """Apply only the declared local provider and generation sampler settings."""
    from jev_ultrafast import model

    keys = {"S1_DECISION_URL": args.decision_url, "S1_TEXT_BASE_URL": args.text_base_url, "TEXT_MODEL": args.text_model}
    policy = getattr(args, "policy", "baseline")
    keys["S1_LOCAL_POLICY"] = policy
    if policy == "noul_v1":
        keys.update(candidate_identity(json.loads(args.model_metadata.read_text())))
    previous = {key: os.environ.get(key) for key in keys}
    original_stream = model.stream_local_text

    def seeded_stream(url, body):
        # The helper reports settings from this same body after streaming; preserve truthful raw seed metadata.
        body.update(seed=plan["model_seed"], temperature=plan["temperature"])
        return original_stream(url, body)

    os.environ.update(keys)
    model.stream_local_text = seeded_stream
    try:
        yield
    finally:
        model.stream_local_text = original_stream
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def fixture_server(plan, port):
    documents = {f"/heldout/{case['id']}.html": render_case(case).encode() for case in plan["tasks"]}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            expected_host = f"127.0.0.1:{self.server.server_address[1]}"
            if self.headers.get("Host") != expected_host:
                self.send_error(400)
                return
            parsed = urlsplit(self.path)
            document = documents.get(parsed.path) if not parsed.query else None
            if document is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(document)))
            self.end_headers()
            self.wfile.write(document)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def provenance(args, plan, freeze):
    def git(*arguments):
        return subprocess.check_output(
            ["git", "-c", "core.fsmonitor=false", *arguments], cwd=REPO, text=True,
        ).strip()

    paths = {Path(__file__), REPO / "scripts/local_benchmark.py", *REPO.glob("jev_ultrafast/*.py"),
             REPO / "jev_ultrafast/snapshot.js"}
    packages = {}
    for name in ("httpx", "websockets", "mlx", "mlx-lm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "repo_path": str(REPO),
        "repo_head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
        "working_tree_status": git("status", "--short"),
        "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
        "source_sha256": {str(path.relative_to(REPO)): sha256(path) for path in sorted(paths)},
        "freeze": freeze, "manifest_sha256": sha256(args.manifest),
        "model_metadata_path": str(args.model_metadata), "model_metadata_sha256": sha256(args.model_metadata),
        "model": json.loads(args.model_metadata.read_text()), "python": sys.version,
        "platform": platform.platform(), "machine": platform.machine(), "installed_distributions": packages,
        "configuration": {
            "policy": args.policy, "decision_url": args.decision_url, "text_base_url": args.text_base_url,
            "text_model": args.text_model, "chromium_executable": str(args.chrome_path),
            "task_seed": plan["task_seed"], "model_seed": plan["model_seed"], "temperature": plan["temperature"],
            "generation_seed_note": "Runner overrides old helper seed17 with frozen seed42 for every compared policy",
            "choice_sampling": "Deterministic offered-choice scoring; native runtime details remain in model metadata",
            "protocol": plan["protocol"], "scheduled_runs": schedule(plan, args.policy),
        },
        "boundaries": {
            "task_budget": (
                "180s covers Chromium startup, version query, tab setup and policy; verification/cleanup separate"
            ),
            "setup_ms": "Agent construction: tab navigation and initial observation; separate from Chromium startup",
            "task_ms": "Policy loop after initial observation through model terminal choice, automatic stop, or error",
            "total_ms": (
                "Entire repetition including owned Chromium startup/version query/setup/loop/verification/cleanup"
            ),
            "model_wall_ms": "Each entire logical policy/helper call counted once; child timing is not added twice",
            "native_scoring_requests": "Actual physical transport attempts, including failed attempts; warmup separate",
            "browser_task_wall_ms": "Outer fresh/observe/act only, including nested CDP exactly once",
            "memory": "Python cumulative own-process peak; Chromium wait4 owned main PID, not a process-tree total",
        },
    }


def assert_source_unchanged(metadata):
    for relative, expected in metadata["source_sha256"].items():
        if sha256(REPO / relative) != expected:
            raise ValueError(f"Selected policy/runtime source changed during the fixed suite: {relative}")


def warmup(plan, choose_function, *, policy="baseline", abort_state=None):
    """Two fixed development observations; no heldout page or goal enters warmup."""
    records = []
    for index in range(2):
        recorder = ModelRecorder("development_warmup", policy)
        recorder.real_choose = choose_function
        result = {"index": index, "phase": "development_warmup", "model_calls": recorder.calls, "ok": False}
        try:
            with deadline(plan["protocol"]["task_timeout_seconds"]):
                recorder.choose(deepcopy(CLICK_PAGE), CLICK_GOAL, [])
                recorder.text(deepcopy(TEXT_CONTEXT))
            result["ok"] = True
        except Exception as exc:
            result["error"] = error_record(exc)
        if abort_state and abort_state["requested"]:
            result["window_interrupted"] = True
        stop_reason = batch_stop_reason(result)
        if stop_reason:
            result["batch_stop_reason"] = stop_reason
        records.append(result)
        if result.get("window_interrupted") or stop_reason:
            break
    return records


@contextmanager
def window_abort_handlers():
    """Preserve native call prefixes through the existing ordinary-error path."""
    state = {"requested": False, "signal": None}

    def interrupt(signum, _frame):
        if state["requested"]:
            return  # A second signal must not interrupt evidence/Chromium cleanup.
        state.update(requested=True, signal=signum)
        raise BudgetTimeout("External heldout window requested bounded abort; no further task")

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGUSR1)}
    try:
        yield state
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def batch_stop_reason(record):
    """Inspect actual errors/responses only; page/goal data and genuine BLOCKED are not denials."""
    errors = [record.get(key) for key in ("error", "verification_error", "teardown_error", "chromium_teardown_error")]
    calls = record.get("model_calls", [])
    for call in calls:
        errors.append(call.get("error"))
        for native in native_calls_for(call):
            if native.get("http_status") in (401, 403):
                return {"type": "PolicyDenial", "message": "Actual local provider HTTP403; no retry or reroute"}
            if native.get("http_status") in (502, 503, 504):
                return {"type": "NativeServiceUnavailable",
                        "message": "Actual local provider unavailable; batch stopped"}
            errors.append(native.get("error"))
            raw = native.get("raw_response")
            if isinstance(raw, dict) and raw.get("status") != "ok":
                errors.extend(raw.get(key) for key in ("error", "failure", "detail", "message"))

    def error_text(value):
        if isinstance(value, dict):
            return " ".join(error_text(value.get(key)) for key in ("type", "message", "detail", "error", "failure"))
        return value if isinstance(value, str) else ""

    text = " ".join(error_text(error) for error in errors).lower()
    if any(phrase in text for phrase in ("permissionerror", "permission denied",
                                        "operation not permitted", "access denied")) \
            or re.search(r"\b403\s+(?:forbidden|client error)", text) \
            or re.search(r"model provider returned http (?:401|403)\b", text):
        return {"type": "PolicyDenial", "message": "Actual operation access denial; no retry or reroute"}
    if any(phrase in text for phrase in ("connecterror", "connectionrefusederror", "connectionreseterror",
                                        "brokenpipeerror", "native process exited", "native process unavailable",
                                        "runtime unavailable", "owned native process closed")) \
            or re.search(r"model provider returned http (?:502|503|504)\b", text):
        return {"type": "NativeServiceUnavailable", "message": "Actual local runtime unavailable; batch stopped"}
    return None


def run_case(case, repetition, args, plan, origin, choose_function, *, native_budget=None):
    from jev_ultrafast import Agent
    from jev_ultrafast.isolated_browser import IsolatedChromium

    recorder, events = ModelRecorder("measured", args.policy, native_budget), []
    recorder.real_choose = choose_function
    result = {
        "kind": "heldout_browser_task", "phase": "measured", "task_id": case["id"], "family": case["family"],
        "repetition": repetition, "policy": args.policy, "url": f"{origin}/heldout/{case['id']}.html",
        "goal": case["goal"], "successful_task": False, "model_calls": recorder.calls, "browser_events": events,
        "task_ms": None, "setup_ms": None, "verified": False,
    }
    chromium = IsolatedChromium(executable=args.chrome_path, profile_root=args.output, fixture_origin=origin)
    agent, timed_browser = None, None
    total_started = time.perf_counter()

    def factory(url):
        nonlocal timed_browser
        started = time.perf_counter()
        event = {"method": "construct", "phase": "setup", "ok": False}
        events.append(event)
        try:
            timed_browser = TimedBrowser(chromium.browser(url), events)
            event["ok"] = True
            return timed_browser
        except Exception as exc:
            event["error"] = error_record(exc)
            raise
        finally:
            event["wall_ms"] = (time.perf_counter() - started) * 1000

    try:
        with deadline(plan["protocol"]["task_timeout_seconds"]), recorder.agent_hooks():
            chromium.start()
            result["chromium_startup_ms"] = chromium.startup_ms
            version_started = time.perf_counter()
            result["chromium_version"] = chromium.transport("Browser.getVersion")
            result["chromium_version_query_ms"] = (time.perf_counter() - version_started) * 1000
            setup_started = time.perf_counter()
            agent = Agent(result["url"], case["goal"], screenshots=False, browser_factory=factory)
            result["setup_ms"] = (time.perf_counter() - setup_started) * 1000
            timed_browser.phase = "task"
            task_started = time.perf_counter()
            try:
                while agent.state["status"] not in {"done", "blocked"}:
                    if sum(call["kind"] == "decision" for call in recorder.calls) >= 12:
                        raise RuntimeError("Reached preregistered budget of 12 decisions")
                    if len(agent.state["history"]) >= 12:
                        raise RuntimeError("Reached preregistered budget of 12 browser actions")
                    agent.command("tick")
            finally:
                result["task_ms"] = (time.perf_counter() - task_started) * 1000
    except Exception as exc:
        result["error"] = error_record(exc)
    finally:
        if agent is not None:
            result["agent_state"] = agent.snapshot()
            decisions = agent.state["decisions"]
            result["terminal_choice"] = decisions[-1]["choice"] if decisions else None
            result["executed_actions"] = len(agent.state["history"])
            timed_browser.phase = "verification"
            started = time.perf_counter()
            try:
                with deadline(15):
                    observed = timed_browser.evaluate(READ_FRESH_DOM)
                    result["verification"] = verify_case(
                        case, observed, agent_status=agent.state["status"],
                        terminal_choice=result["terminal_choice"], executed_actions=result["executed_actions"],
                        execution_error=result.get("error"),
                    )
                    result["verified"] = result["verification"]["verified"]
            except Exception as exc:
                result["verification_error"] = error_record(exc)
            finally:
                result["verification_ms"] = (time.perf_counter() - started) * 1000
            timed_browser.phase = "teardown"
            try:
                with deadline(15):
                    agent.close()
            except Exception as exc:
                result["teardown_error"] = error_record(exc)
        else:
            result["executed_actions"] = 0
        if chromium.transport is not None:
            result["chromium_cdp_blocked_requests"] = chromium.transport.blocked_requests
            result["cdp_calls"] = deepcopy(chromium.transport.calls)
        if chromium.proxy is not None:
            result["chromium_proxy_blocked_requests"] = chromium.proxy.blocked_requests
        try:
            chromium.close()
        except Exception as exc:
            result["chromium_teardown_error"] = error_record(exc)
        result["chromium_resource_usage"] = chromium.resource_usage
        result["total_ms"] = (time.perf_counter() - total_started) * 1000
        result["decision_calls"] = sum(call["kind"] == "decision" for call in recorder.calls)
        result["native_scoring_requests"] = sum(
            call.get("transport_attempted") is True for call in recorder.native_calls
        )
        result["native_model_executions_confirmed"] = sum(
            call.get("model_execution_confirmed") is True for call in recorder.native_calls
        )
        result["text_calls"] = sum(call["kind"] == "text_helper" for call in recorder.calls)
        result["model_wall_ms"] = sum(call["wall_ms"] for call in recorder.calls)
        result["browser_task_wall_ms"] = sum(event["wall_ms"] for event in events if event["phase"] == "task")
        result["python_peak_rss_bytes_cumulative"] = peak_python_rss_bytes()
        result["budget_exhausted"] = result.get("error", {}).get("type") == "BudgetTimeout" or (
            "preregistered budget" in result.get("error", {}).get("message", "")
        ) or any(call.get("native_budget_exhausted") for call in recorder.calls)
        result["successful_task"] = result["verified"] and not any(
            key in result for key in ("error", "verification_error", "teardown_error", "chromium_teardown_error")
        )
    return result


def summarize(records, plan, policy, *, batch_error=None):
    expected = len(plan["tasks"]) * plan["protocol"]["repeats_per_task_per_variant"]
    calls = [call for record in records for call in record["model_calls"]]
    helpers = [call for call in calls if call["kind"] == "text_helper" and call["ok"]]
    helper_ttfts, helper_rates, scoring_rates = [], [], []
    for call in helpers:
        helper = call["result"][1]
        if helper.get("ttft_ms") is not None:
            helper_ttfts.append(helper["ttft_ms"])
        tokens = helper.get("usage", {}).get("completion_tokens")
        if type(tokens) is int and tokens > 0 and call["wall_ms"] > 0:
            helper_rates.append(tokens / (call["wall_ms"] / 1000))
    for call in calls:
        if call["kind"] != "decision":
            continue
        for response in scoring_calls_for(call):
            tokens, milliseconds = response.get("native_token_count"), response.get("model_latency_ms")
            if type(tokens) is int and tokens > 0 and type(milliseconds) in (int, float) and milliseconds > 0:
                scoring_rates.append(tokens / (milliseconds / 1000))

    def group(items, expected_count):
        successes = sum(record["successful_task"] for record in items)
        return {
            "executed_runs": len(items), "successful_runs": successes, "failed_runs": len(items) - successes,
            "expected_runs": expected_count, "unexecuted_runs": expected_count - len(items),
            "success_rate": successes / expected_count, "complete": len(items) == expected_count,
            "task_ms_all_runs": distribution([record.get("task_ms") for record in items]),
            "task_ms_successful": distribution([
                record.get("task_ms") for record in items if record["successful_task"]
            ]),
            "total_ms_all_runs": distribution([record["total_ms"] for record in items]),
            "executed_actions": sum(record["executed_actions"] for record in items),
            "logical_decision_cycles": sum(record["decision_calls"] for record in items),
            "decision_requests": sum(
                record.get("native_scoring_requests", record["decision_calls"]) for record in items
            ),
            "helper_requests": sum(record["text_calls"] for record in items),
            "budget_exhaustions": sum(record["budget_exhausted"] for record in items),
        }

    summary = group(records, expected)
    summary.update(
        policy=policy, expected_runs=expected, unexecuted_runs=expected - len(records),
        complete=len(records) == expected and batch_error is None,
        all_task_success_rate=summary["successful_runs"] / expected,
        by_task={case["id"]: group([record for record in records if record["task_id"] == case["id"]],
                                   plan["protocol"]["repeats_per_task_per_variant"])
                 for case in plan["tasks"]},
        by_family={family: group([record for record in records if record["family"] == family],
                                 sum(case["family"] == family for case in plan["tasks"])
                                 * plan["protocol"]["repeats_per_task_per_variant"])
                   for family in {case["family"] for case in plan["tasks"]}},
        decision_wall_ms_all_attempts=distribution([call["wall_ms"] for call in calls if call["kind"] == "decision"]),
        native_scoring_requests=sum(
            item.get("transport_attempted") is True for call in calls for item in native_calls_for(call)
        ),
        native_model_executions_confirmed=sum(
            item.get("model_execution_confirmed") is True for call in calls for item in native_calls_for(call)
        ),
        native_scoring_client_wall_ms_all_attempts=distribution([
            item.get("client_wall_ms") for call in calls for item in native_calls_for(call)
            if item.get("transport_attempted") is True
        ]),
        helper_wall_ms_all_attempts=distribution([call["wall_ms"] for call in calls if call["kind"] == "text_helper"]),
        helper_first_content_delta_ms=distribution(helper_ttfts),
        helper_reported_completion_tokens_per_full_request_second=distribution(helper_rates),
        decision_input_tokens_per_adapter_scoring_second=distribution(scoring_rates),
        token_throughput_scope=(
            "Helper rates require actual provider completion-token counts. Native choice rates are input scoring "
            "tokens/adapter latency, including preparation and transport; not generated tokens/s or pure GPU compute."
        ),
        model_wall_ms_per_task=distribution([record["model_wall_ms"] for record in records]),
        browser_wall_ms_per_task=distribution([record["browser_task_wall_ms"] for record in records]),
        chromium_startup_ms=distribution([record.get("chromium_startup_ms") for record in records]),
        setup_ms=distribution([record.get("setup_ms") for record in records]),
        verification_ms=distribution([record.get("verification_ms") for record in records]),
        chromium_owned_main_peak_rss_bytes=distribution([
            (record.get("chromium_resource_usage") or {}).get("peak_rss_bytes") for record in records
        ]),
        python_peak_rss_bytes_cumulative=max(
            (record.get("python_peak_rss_bytes_cumulative", 0) for record in records), default=None,
        ),
        batch_error=batch_error,
        percentile_method="linear interpolation at (n - 1) * q; descriptive small-sample p95",
        memory_scope="Python cumulative own-process peak; each Chromium wait4 main PID, not process-tree total",
        decision_ttft_ms=None, decision_ttft_reason="Nonstreaming native scoring has no emitted first-token event",
    )
    return summary


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--execute", action="store_true", help="Explicitly run all frozen tasks; otherwise plan only")
    cli.add_argument("--policy", choices=tuple(APPROVED_POLICIES), default="baseline")
    cli.add_argument("--manifest", type=Path, default=REPO / "benchmark/heldout-plan.json")
    cli.add_argument("--freeze", type=Path, default=REPO / "benchmark/heldout-freeze.json")
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--model-metadata", type=Path, required=True)
    cli.add_argument("--chrome-path", type=Path, required=True)
    cli.add_argument("--decision-url", required=True)
    cli.add_argument("--text-base-url", required=True)
    cli.add_argument("--text-model", required=True)
    cli.add_argument("--fixture-port", type=int, default=0)
    return cli


def main():
    cli, records, batch_error = parser(), [], None
    args = cli.parse_args()
    try:
        for name in ("manifest", "freeze", "output", "model_metadata", "chrome_path"):
            path = getattr(args, name)
            if not path.is_absolute():
                raise ValueError(f"--{name.replace('_', '-')} must use a full absolute path")
            setattr(args, name, path.resolve())
        if not args.output.is_relative_to(REPO):
            raise ValueError("--output must be inside the caller's task repository")
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise ValueError("--output must be a new or empty directory; prior evidence is preserved")
        if not args.chrome_path.is_file() or not os.access(args.chrome_path, os.X_OK):
            raise ValueError("--chrome-path must be an existing executable; no download is attempted")
        if not args.model_metadata.is_file() or not args.text_model or not 0 <= args.fixture_port <= 65535:
            raise ValueError("Existing model metadata, explicit text model and a valid fixture port are required")
        args.decision_url = local_endpoint(args.decision_url, "--decision-url")
        args.text_base_url = local_endpoint(args.text_base_url, "--text-base-url")
        plan, freeze = verify_freeze(args.manifest, args.freeze)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        cli.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = provenance(args, plan, freeze)
    metadata["execution_requested"] = args.execute
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if not args.execute:
        (args.output / "execution-plan.json").write_text(json.dumps(schedule(plan, args.policy), indent=2) + "\n")
        print("Frozen plan verified; no port bound, model call or browser start. Add --execute only when authorized.")
        return 0

    def save(record):
        records.append(record)
        with (args.output / "runs.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({key: record.get(key) for key in
                          ("task_id", "repetition", "policy", "successful_task", "task_ms", "error")}), flush=True)

    native_budget = NativeRequestBudget(plan["protocol"]["max_decision_requests_per_variant"])
    try:
        with window_abort_handlers() as abort_state:
            choose_function = resolve_policy(args.policy)
            with local_configuration(args, plan), fixture_server(plan, args.fixture_port) as origin:
                metadata["fixture_origin"] = origin
                assert_source_unchanged(metadata)
                metadata["development_warmups"] = warmup(plan, choose_function, policy=args.policy,
                                                        abort_state=abort_state)
                (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
                tasks = {case["id"]: case for case in plan["tasks"]}
                for run in schedule(plan, args.policy):
                    warmup_stop = next((reason for item in metadata["development_warmups"]
                                        if (reason := batch_stop_reason(item))), None)
                    if warmup_stop:
                        batch_error = warmup_stop
                        metadata["batch_error"] = batch_error
                        break
                    if abort_state["requested"]:
                        batch_error = {"type": "WindowInterrupted",
                                       "message": "External bounded window stopped the suite"}
                        metadata["batch_error"] = batch_error
                        break
                    # No filters, repetitions override, task skips, or acceptance tuning.
                    if native_budget.remaining == 0:
                        batch_error = {"type": "NativeRequestBudgetError",
                                       "message": "Preregistered 288 physical scoring-request budget exhausted"}
                        metadata["batch_error"] = batch_error
                        break
                    assert_source_unchanged(metadata)
                    record = run_case(tasks[run["task_id"]], run["repetition"], args, plan, origin, choose_function,
                                      native_budget=native_budget)
                    if abort_state["requested"]:
                        record["window_interrupted"] = True
                        record["successful_task"] = False
                    save(record)
                    stop_reason = batch_stop_reason(record)
                    if stop_reason:
                        batch_error = stop_reason
                        metadata["batch_error"] = batch_error
                        break
                    if abort_state["requested"]:
                        batch_error = {"type": "WindowInterrupted",
                                       "message": "Interrupted task saved; no further task"}
                        metadata["batch_error"] = batch_error
                        break
    except BaseException as exc:
        batch_error = error_record(exc)
        metadata["batch_error"] = batch_error
    finally:
        metadata["measured_native_budget"] = {
            "limit": native_budget.limit, "transport_attempts": native_budget.used,
            "remaining": native_budget.remaining, "warmups_included": False,
        }
        summary = summarize(records, plan, args.policy, batch_error=batch_error)
        (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return int(not summary["complete"] or summary["failed_runs"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
