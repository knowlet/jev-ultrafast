"""Bounded, real local System One measurements; no .env loading or remote fallbacks.

The model and static fixture server must already be running on numeric loopback.
Decision-only measures a fixed synthetic observed state, not browser task success.
Browser modes create one owned isolated Chromium process and a fresh tab per task.
Raw failed calls/runs and warmups are retained; only measured runs enter statistics.
"""

import argparse
import hashlib
import importlib.metadata
import ipaddress
import json
import math
import os
import platform
import resource
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TRAVEL_GOAL = (
    "Use the destination search and filters to find Design stays in Lisbon with Free cancellation, "
    "then open Casa Flora."
)
CLICK_GOAL = "Click the Mark complete button. Stop when Completion confirmed is visibly shown."
CLICK_PAGE = {
    "url": "http://127.0.0.1:8766/local_click_fixture.html",
    "title": "Local completion fixture",
    "text": "Local completion fixture\nCompletion not yet confirmed\nMark complete\nReset completion",
    "actions": [
        {"id": "e1", "kind": "click", "label": "Mark complete", "role": "button", "value": "", "node": 1},
        {"id": "e2", "kind": "click", "label": "Reset completion", "role": "button", "value": "", "node": 2},
        {"id": "wait", "kind": "wait", "label": "Wait for the page to update"},
    ],
}
TEXT_CONTEXT = {
    "goal": TRAVEL_GOAL,
    "field": {"label": "Destination", "role": "searchbox", "value": ""},
    "page": {"title": "Forma · Find a place to slow down", "text": "Destination\nFind stays"},
    "recent_actions": [],
}
POLICIES = {"baseline": "choose", "noul_v1": "choose_noul"}


class BudgetTimeout(TimeoutError):
    """The finite wall-clock budget elapsed; never retry a mutation."""


class NativeRequestBudgetError(RuntimeError):
    """A preregistered physical scoring-request budget has been exhausted."""

    budget_exhausted = True


class NativeRequestBudget:
    def __init__(self, limit):
        self.limit, self.used = limit, 0

    @property
    def remaining(self):
        return max(0, self.limit - self.used)

    def account(self, calls):
        count = sum(call.get("transport_attempted") is True for call in calls)
        if count > self.remaining:
            raise NativeRequestBudgetError("Native provider exceeded the preregistered physical request budget")
        self.used += count


def resolve_policy(name):
    from jev_ultrafast import model

    if name not in POLICIES or not hasattr(model, POLICIES[name]):
        raise ValueError("Selected benchmark policy is not available; no fallback is allowed")
    return getattr(model, POLICIES[name])


def candidate_identity(metadata):
    identity = metadata.get("decision_service_identity", {})
    expected_model, temperature = identity.get("model_file"), identity.get("temperature")
    if not isinstance(expected_model, str) or not expected_model:
        raise ValueError("noul_v1 requires the verified decision-service model_file identity")
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("noul_v1 requires the verified decision-service calibration temperature")
    return {"S1_EXPECTED_MODEL": expected_model, "S1_EXPECTED_TEMPERATURE": str(temperature)}


def native_calls_for(call):
    """Read explicit physical-call evidence; never equate a policy cycle with a GPU inference."""
    if "native_calls" in call:
        return call["native_calls"]
    result = call.get("result", {})
    return result.get("native_calls", []) if isinstance(result, dict) else []


def scoring_calls_for(call):
    native = native_calls_for(call)
    if "native_calls" in call or native:
        return [item for item in native if item.get("ok") is True]
    # Preserve old raw trace arithmetic while new recorders provide actual physical-call evidence.
    return [call["result"]] if call.get("kind") == "decision" and call.get("ok") else []


@contextmanager
def deadline(seconds):
    def expire(_signum, _frame):
        raise BudgetTimeout(f"Wall-clock budget of {seconds:g} seconds elapsed")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def percentile(values, fraction):
    """Linear interpolation at (n - 1) * fraction; small n is reported explicitly."""
    values = sorted(values)
    if not values:
        return None
    location = (len(values) - 1) * fraction
    lower = math.floor(location)
    upper = math.ceil(location)
    return values[lower] + (values[upper] - values[lower]) * (location - lower)


def distribution(values):
    values = [v for v in values if type(v) in (int, float) and math.isfinite(v)]
    return {
        "n": len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
    }


def local_endpoint(value, setting, *, origin=False):
    parsed = urlsplit(value or "")
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
        port = parsed.port
    except ValueError:
        address, port = None, None
    if (
        parsed.scheme != "http" or not address or not address.is_loopback or port is None
        or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment
        or (origin and parsed.path not in {"", "/"})
    ):
        raise ValueError(f"{setting} must be an HTTP numeric-loopback URL with explicit port and no credentials")
    return value.rstrip("/")


def peak_python_rss_bytes():
    # macOS reports bytes; Linux reports KiB. This is cumulative for this process.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def error_record(exc):
    return {"type": type(exc).__name__, "message": str(exc)[:1000]}


def source_metadata(model_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()

    files = sorted(
        {Path(__file__), *REPO.glob("jev_ultrafast/*.py"), *REPO.glob("jev_ultrafast/*.js"),
         REPO / "jev_ultrafast/static/fixture.html", REPO / "jev_ultrafast/static/local_click_fixture.html"}
    )
    manifest = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    versions = {}
    for name in ("jev-ultrafast", "browser-harness", "httpx", "websockets", "mlx", "mlx-lm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_path": str(REPO),
        "repo_head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "working_tree_status": git("status", "--short"),
        "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
        "source_sha256": manifest,
        "model_metadata_path": str(model_path.resolve()),
        "model_metadata_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "model": json.loads(model_path.read_text()),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "installed_distributions": versions,
    }


class ModelRecorder:
    """Record real logical provider calls, including failures before Agent can log them."""

    def __init__(self, phase, policy="baseline", native_budget=None):
        from jev_ultrafast import model

        self.phase, self.calls, self.native_calls = phase, [], []
        self.policy, self.native_budget = policy, native_budget
        self.real_choose, self.real_text = resolve_policy(policy), model.field_text

    def call(self, kind, function, *inputs, **kwargs):
        from jev_ultrafast import model

        started = time.perf_counter()
        record = {"kind": kind, "phase": self.phase, "input": deepcopy(inputs), "ok": False}
        self.calls.append(record)
        captured, native, previous_post = [], None, model.LOCAL_CLIENT.post

        def capture_post(url, *args, **options):
            item = {"kind": "decision", "phase": "combined", "transport_attempted": False,
                    "model_execution_confirmed": False, "ok": False}
            captured.append(item)
            if self.native_budget and sum(c.get("transport_attempted") is True for c in captured) >= \
                    self.native_budget.remaining:
                raise NativeRequestBudgetError("Reached preregistered native scoring-request budget")
            item["request"] = deepcopy(options.get("json"))
            if item["request"] is None and options.get("content") is not None:
                item["request"] = json.loads(options["content"])
            item["transport_attempted"] = True
            sent = time.perf_counter()
            try:
                response = previous_post(url, *args, **options)
                item["http_status"] = response.status_code
                try:
                    raw = response.json()
                    item.update(raw_response=raw, model_latency_ms=raw.get("latency_ms"),
                                native_token_count=raw.get("native_token_count"),
                                preprocessing=raw.get("preprocessing"))
                    item["model_execution_confirmed"] = raw.get("status") == "ok"
                    item["ok"] = not response.is_error and not response.is_redirect
                except (ValueError, AttributeError):
                    pass
                return response
            except Exception as exc:
                item["error"] = error_record(exc)
                raise
            finally:
                item["client_wall_ms"] = (time.perf_counter() - sent) * 1000

        capture_baseline = kind == "decision" and self.policy == "baseline"
        if capture_baseline:
            model.LOCAL_CLIENT.post = capture_post
        try:
            if kind == "decision" and self.native_budget and self.native_budget.remaining == 0:
                raise NativeRequestBudgetError("Reached preregistered native scoring-request budget")
            result = function(*inputs, **kwargs)
            record.update(ok=True, result=deepcopy(result))
            if kind == "decision" and isinstance(result, dict) and "native_calls" in result:
                native = result["native_calls"]
            return result
        except Exception as exc:
            record["error"] = error_record(exc)
            if kind == "decision":
                native = getattr(exc, "native_calls", None)
                record["native_budget_exhausted"] = bool(getattr(exc, "budget_exhausted", False))
            raise
        finally:
            if capture_baseline:
                model.LOCAL_CLIENT.post = previous_post
            record["wall_ms"] = (time.perf_counter() - started) * 1000
            if kind == "decision":
                record["native_calls"] = deepcopy(captured if native is None else native)
                self.native_calls.extend(record["native_calls"])
                if self.native_budget:
                    self.native_budget.account(record["native_calls"])

    def choose(self, *inputs):
        kwargs = {"max_native_calls": self.native_budget.remaining} if \
            self.policy == "noul_v1" and self.native_budget else {}
        return self.call("decision", self.real_choose, *inputs, **kwargs)

    def text(self, *inputs):
        return self.call("text_helper", self.real_text, *inputs)

    @contextmanager
    def agent_hooks(self):
        from jev_ultrafast import agent

        previous = agent.choose, agent.field_text
        agent.choose, agent.field_text = self.choose, self.text
        try:
            yield
        finally:
            agent.choose, agent.field_text = previous


class TimedBrowser:
    """Outer Browser calls only: nested CDP/freshness work is counted once."""

    def __init__(self, browser, events):
        self.browser, self.events, self.phase = browser, events, "setup"

    def invoke(self, method, *args, **kwargs):
        started = time.perf_counter()
        record = {"method": method, "phase": self.phase, "ok": False}
        self.events.append(record)
        try:
            result = getattr(self.browser, method)(*args, **kwargs)
            record["ok"] = True
            return result
        except Exception as exc:
            record["error"] = error_record(exc)
            raise
        finally:
            record["wall_ms"] = (time.perf_counter() - started) * 1000

    def fresh(self, *args, **kwargs):
        return self.invoke("fresh", *args, **kwargs)

    def observe(self, *args, **kwargs):
        return self.invoke("observe", *args, **kwargs)

    def act(self, *args, **kwargs):
        return self.invoke("act", *args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.invoke("evaluate", *args, **kwargs)

    def close(self):
        return self.invoke("close")


def verify(browser, fixture):
    # A fresh independent DOM read, without the agent snapshot or DONE assertion.
    if fixture == "travel":
        state = browser.evaluate("({url: location.href, text: document.body.innerText})")
        checks = {
            "opened_casa_flora": state["url"].endswith("#casa-flora"),
            "requested_filters_visible": (
                "Your filters: Design · Free cancellation enabled · Destination Lisbon" in state["text"]
            ),
            "casa_flora_visible": "Casa Flora" in state["text"],
        }
    else:
        state = browser.evaluate(
            "({url: location.href, text: document.body.innerText, "
            "completion: document.getElementById('completion')?.textContent})"
        )
        checks = {"completion_confirmed": state["completion"] == "Completion confirmed"}
    return {"verified": all(checks.values()), "checks": checks, "fresh_dom": state}


def development_task_success(result):
    """A matching DOM alone cannot convert model/automatic BLOCKED into success."""
    state = result.get("agent_state", {})
    decisions = state.get("decisions", [])
    return (
        result.get("stopped_cleanly") is True and result.get("verified") is True
        and state.get("status") == "done" and bool(decisions) and decisions[-1].get("choice") == "DONE"
        and not any(key in result for key in ("error", "verification_error", "teardown_error"))
    )


def browser_run(chromium, args, phase, index):
    from jev_ultrafast import Agent

    recorder, events = ModelRecorder(phase, args.policy), []
    path = "/fixture.html?scenario=travel" if args.fixture == "travel" else "/local_click_fixture.html"
    url = args.fixture_url + path
    goal = TRAVEL_GOAL if args.fixture == "travel" else CLICK_GOAL
    result = {
        "kind": "browser_task", "phase": phase, "index": index, "fixture": args.fixture,
        "url": url, "goal": goal, "verified": False, "stopped_cleanly": False,
        "model_calls": recorder.calls, "browser_events": events,
    }
    agent, timed_browser = None, None
    total_started = time.perf_counter()

    def factory(target_url):
        nonlocal timed_browser
        started = time.perf_counter()
        event = {"method": "construct", "phase": "setup", "ok": False}
        events.append(event)
        try:
            timed_browser = TimedBrowser(chromium.browser(target_url), events)
            event["ok"] = True
            return timed_browser
        except Exception as exc:
            event["error"] = error_record(exc)
            raise
        finally:
            event["wall_ms"] = (time.perf_counter() - started) * 1000

    try:
        with deadline(args.task_timeout), recorder.agent_hooks():
            setup_started = time.perf_counter()
            agent = Agent(url, goal, screenshots=False, browser_factory=factory)
            result["setup_ms"] = (time.perf_counter() - setup_started) * 1000
            timed_browser.phase = "task"
            task_started = time.perf_counter()
            try:
                while agent.state["status"] not in {"done", "blocked"}:
                    if sum(call["kind"] == "decision" for call in recorder.calls) >= args.max_decisions:
                        raise RuntimeError(f"Reached benchmark budget of {args.max_decisions} decisions")
                    agent.command("tick")
                result["stopped_cleanly"] = True
            finally:
                result["task_ms"] = (time.perf_counter() - task_started) * 1000
    except Exception as exc:
        result["error"] = error_record(exc)
    finally:
        if agent is not None:
            result["agent_state"] = agent.snapshot()
            timed_browser.phase = "verification"
            started = time.perf_counter()
            try:
                with deadline(15):
                    result["verification"] = verify(timed_browser, args.fixture)
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
        result["total_ms"] = (time.perf_counter() - total_started) * 1000
        result["python_peak_rss_bytes_cumulative"] = peak_python_rss_bytes()
        result["decision_calls"] = sum(c["kind"] == "decision" for c in recorder.calls)
        result["native_scoring_requests"] = sum(c.get("transport_attempted") is True for c in recorder.native_calls)
        result["native_model_executions_confirmed"] = sum(
            c.get("model_execution_confirmed") is True for c in recorder.native_calls
        )
        result["text_calls"] = sum(c["kind"] == "text_helper" for c in recorder.calls)
        result["model_wall_ms"] = sum(c["wall_ms"] for c in recorder.calls)
        result["browser_task_wall_ms"] = sum(e["wall_ms"] for e in events if e["phase"] == "task")
        result["successful_task"] = development_task_success(result)
    return result


def micro_run(kind, phase, index, args):
    recorder = ModelRecorder(phase, args.policy)
    result = {"kind": kind, "phase": phase, "index": index, "ok": False, "model_calls": recorder.calls}
    try:
        with deadline(args.task_timeout):
            if kind == "decision_microbenchmark":
                page = deepcopy(CLICK_PAGE)
                decision = recorder.choose(page, CLICK_GOAL, [])
                result.update(ok=True, intended_choice=decision["choice"] == "e1")
            else:
                text, _helper = recorder.text(deepcopy(TEXT_CONTEXT))
                result.update(ok=True, intended_text=text == "Lisbon")
    except Exception as exc:
        result["error"] = error_record(exc)
    result["python_peak_rss_bytes_cumulative"] = peak_python_rss_bytes()
    return result


def summarize(records):
    measured = [r for r in records if r["phase"] == "measured"]
    calls = [call for r in measured for call in r.get("model_calls", [])]
    decisions = [c for c in calls if c["kind"] == "decision"]
    helpers = [c for c in calls if c["kind"] == "text_helper"]
    tasks = [r for r in measured if r["kind"] == "browser_task"]
    ttfts, throughput, adapter_latencies, scoring_rates = [], [], [], []
    for call in decisions:
        for response in scoring_calls_for(call):
            adapter_ms = response.get("model_latency_ms")
            tokens = response.get("native_token_count")
            # Adapter latency includes preparation and the native pipe, not pure GPU time.
            if type(adapter_ms) in (int, float) and math.isfinite(adapter_ms) and adapter_ms > 0:
                adapter_latencies.append(adapter_ms)
                if type(tokens) is int and tokens > 0:
                    scoring_rates.append(tokens / (adapter_ms / 1000))
    for call in helpers:
        if not call["ok"]:
            continue
        helper = call["result"][1]
        if helper.get("ttft_ms") is not None:
            ttfts.append(helper["ttft_ms"])
        completion = helper.get("usage", {}).get("completion_tokens")
        if type(completion) is int and completion > 0 and call["wall_ms"] > 0:
            throughput.append(completion / (call["wall_ms"] / 1000))
    groups = {}
    for kind in {r["kind"] for r in measured}:
        group = [r for r in measured if r["kind"] == kind]
        groups[kind] = {
            "runs": len(group),
            "successful_runs": sum(r.get("successful_task", r.get("ok", False)) for r in group),
            "failed_runs": sum(not r.get("successful_task", r.get("ok", False)) for r in group),
        }
        group_calls = [call for run in group for call in run.get("model_calls", [])]
        groups[kind]["decision_wall_ms_all_attempts"] = distribution([
            call["wall_ms"] for call in group_calls if call["kind"] == "decision"
        ])
        groups[kind]["helper_wall_ms_all_attempts"] = distribution([
            call["wall_ms"] for call in group_calls if call["kind"] == "text_helper"
        ])
        if kind == "decision_microbenchmark":
            groups[kind]["intended_choice_runs"] = sum(r.get("intended_choice", False) for r in group)
        if kind == "helper_microbenchmark":
            groups[kind]["intended_text_runs"] = sum(r.get("intended_text", False) for r in group)
    return {
        "groups": groups,
        "warmup_runs": sum(r["phase"] == "warmup" for r in records),
        "warmup_failed_runs": sum(
            r["phase"] == "warmup" and not r.get("successful_task", r.get("ok", False)) for r in records
        ),
        "decision_wall_ms_all_attempts": distribution([c["wall_ms"] for c in decisions]),
        "native_scoring_requests": sum(
            item.get("transport_attempted") is True for call in decisions for item in native_calls_for(call)
        ),
        "native_model_executions_confirmed": sum(
            item.get("model_execution_confirmed") is True for call in decisions for item in native_calls_for(call)
        ),
        "native_scoring_client_wall_ms_all_attempts": distribution([
            item.get("client_wall_ms") for call in decisions for item in native_calls_for(call)
            if item.get("transport_attempted") is True
        ]),
        "decision_wall_ms_successful": distribution([c["wall_ms"] for c in decisions if c["ok"]]),
        "decision_adapter_scoring_ms_successful": distribution(adapter_latencies),
        "decision_input_tokens_per_scoring_second": distribution(scoring_rates),
        "decision_failed_calls": sum(not c["ok"] for c in decisions),
        "decision_ttft_ms": None,
        "decision_ttft_reason": (
            "Native choices are nonstreaming forward/scoring requests, without a first token event."
        ),
        "helper_wall_ms_all_attempts": distribution([c["wall_ms"] for c in helpers]),
        "helper_failed_calls": sum(not c["ok"] for c in helpers),
        "helper_first_content_delta_ms": distribution(ttfts),
        "helper_provider_completion_tokens_per_full_request_second": distribution(throughput),
        "task_ms_all_runs": distribution([r["task_ms"] for r in tasks if "task_ms" in r]),
        "task_ms_successful": distribution([r["task_ms"] for r in tasks if r["successful_task"]]),
        "setup_ms": distribution([r["setup_ms"] for r in tasks if "setup_ms" in r]),
        "verification_ms": distribution([r["verification_ms"] for r in tasks if "verification_ms" in r]),
        "end_to_end_ms": distribution([r["total_ms"] for r in tasks]),
        "browser_task_wall_ms": distribution([r["browser_task_wall_ms"] for r in tasks]),
        "model_wall_ms_per_task": distribution([r["model_wall_ms"] for r in tasks]),
        "python_peak_rss_bytes_cumulative": max(
            (r["python_peak_rss_bytes_cumulative"] for r in records), default=peak_python_rss_bytes()
        ),
        "percentile_method": "linear interpolation at (n - 1) * q; p95 is descriptive for these small samples",
        "memory_scope": "Current Python process cumulative peak only; service/Chromium RSS and cold load are external.",
        "token_throughput_scope": (
            "Text uses provider-reported completion_tokens/full request time. Choices use native_token_count "
            "divided by the typed adapter's reported latency_ms, including HF preparation and native pipe "
            "request/response. This is input scoring throughput, not native-only compute or generated tokens/s."
        ),
    }


def parser():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--mode", choices=("decision-only", "helper-only", "smoke", "all"), default="smoke")
    args.add_argument("--policy", choices=tuple(POLICIES), default="baseline")
    args.add_argument("--output", type=Path, required=True)
    args.add_argument("--model-metadata", type=Path, required=True)
    args.add_argument("--fixture-url", default="http://127.0.0.1:8766")
    args.add_argument("--fixture", choices=("click", "travel"), default="travel")
    args.add_argument("--chrome-path", type=Path)
    args.add_argument("--decision-repeats", type=int, default=10)
    args.add_argument("--text-repeats", type=int, default=0)
    args.add_argument("--repeats", type=int, default=5)
    args.add_argument("--warmup", type=int, default=2)
    args.add_argument("--task-timeout", type=float, default=180)
    args.add_argument("--max-decisions", type=int, default=12)
    return args


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        args.fixture_url = local_endpoint(args.fixture_url, "--fixture-url", origin=True)
        decision_url = local_endpoint(os.environ.get("S1_DECISION_URL"), "S1_DECISION_URL")
        text_url = None
        if args.mode in {"helper-only", "all"} or (args.mode == "smoke" and args.fixture == "travel"):
            text_url = local_endpoint(os.environ.get("S1_TEXT_BASE_URL"), "S1_TEXT_BASE_URL")
            if not os.environ.get("TEXT_MODEL"):
                raise ValueError("TEXT_MODEL must identify the actual local text runtime")
        if not args.model_metadata.is_file():
            raise ValueError("--model-metadata must reference root-verified local model metadata")
        policy_settings = {"S1_LOCAL_POLICY": args.policy}
        if args.policy == "noul_v1":
            policy_settings.update(candidate_identity(json.loads(args.model_metadata.read_text())))
        resolve_policy(args.policy)
        if not 0 <= args.warmup <= 2 or not 1 <= args.repeats <= 5 or not 1 <= args.decision_repeats <= 10:
            raise ValueError("Bounded repetitions: warmup 0..2, browser 1..5, decision 1..10")
        if not 0 <= args.text_repeats <= 3 or not 0 < args.task_timeout <= 180 or not 1 <= args.max_decisions <= 12:
            raise ValueError("Bounded budgets: text 0..3, task timeout <=180 seconds, decisions 1..12")
        if args.mode in {"smoke", "all"} and (not args.chrome_path or not args.chrome_path.is_file()):
            raise ValueError("Browser measurements need --chrome-path pointing to the existing isolated executable")
        if args.mode == "helper-only" and args.text_repeats == 0:
            args.text_repeats = 1
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("--output must be a new or empty directory so prior evidence is preserved")
    except ValueError as exc:
        cli.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(policy_settings)
    metadata = source_metadata(args.model_metadata)
    metadata["configuration"] = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "decision_endpoint": decision_url, "text_base_url": text_url,
        "text_model": os.environ.get("TEXT_MODEL"),
        "text_temperature": 0, "text_seed": 17, "text_max_tokens": 128,
        "choice_sampling": "deterministic full offered-choice scoring; native runtime configuration in model metadata",
        "microbenchmark_input": {
            "page": CLICK_PAGE, "goal": CLICK_GOAL, "history": [],
            "note": "Fixed synthetic observed state; this does not establish browser task success.",
        },
        "boundaries": {
            "setup_ms": "Create fresh owned tab, initial navigation, and first observation",
            "task_ms": "After initial observation through stopped state or failure; models and browser included",
            "verification_ms": "Fresh independent DOM evaluation after task",
            "total_ms": "Per-run setup + task + verification + tab teardown; Chromium launch measured separately",
            "browser_task_wall_ms": "Exclusive outer fresh/observe/act calls; nested CDP is not added twice",
            "model_wall_ms": (
                "Real logical model call wall time including validation and failures; retries internal to provider"
            ),
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    records = []

    def save(record):
        records.append(record)
        with (args.output / "runs.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({k: record.get(k) for k in
                         ("kind", "phase", "index", "ok", "successful_task", "task_ms", "error")}), flush=True)

    browser_error = None
    try:
        if args.mode in {"decision-only", "all"}:
            for phase, count in (("warmup", args.warmup), ("measured", args.decision_repeats)):
                for index in range(count):
                    save(micro_run("decision_microbenchmark", phase, index, args))
        if args.mode in {"helper-only", "all"} and args.text_repeats:
            for phase, count in (("warmup", args.warmup), ("measured", args.text_repeats)):
                for index in range(count):
                    save(micro_run("helper_microbenchmark", phase, index, args))
        if args.mode in {"smoke", "all"}:
            from jev_ultrafast.isolated_browser import IsolatedChromium

            with tempfile.TemporaryDirectory(prefix="jev-local-benchmark-") as profile_root:
                started = time.perf_counter()
                chromium = IsolatedChromium(
                    executable=args.chrome_path, profile_root=Path(profile_root), fixture_origin=args.fixture_url
                )
                try:
                    with chromium:
                        metadata["chromium_startup_ms"] = chromium.startup_ms
                        metadata["chromium_context_enter_ms"] = (time.perf_counter() - started) * 1000
                        try:
                            version_started = time.perf_counter()
                            metadata["chromium_version"] = chromium.transport("Browser.getVersion")
                            metadata["chromium_version_query_ms"] = (time.perf_counter() - version_started) * 1000
                            counts = (("measured", 1),) if args.mode == "smoke" else (
                                ("warmup", args.warmup), ("measured", args.repeats)
                            )
                            for phase, count in counts:
                                for index in range(count):
                                    save(browser_run(chromium, args, phase, index))
                        finally:
                            if chromium.transport is not None:
                                metadata["chromium_cdp_blocked_requests"] = chromium.transport.blocked_requests
                            if chromium.proxy is not None:
                                metadata["chromium_proxy_blocked_requests"] = chromium.proxy.blocked_requests
                finally:
                    # wait4 evidence is only available after the owned process has been reaped.
                    metadata["chromium_resource_usage"] = getattr(chromium, "resource_usage", None)
                    metadata["chromium_resource_usage_scope"] = (
                        "macOS wait4 for the owned Chromium main PID; this is not the sum of its process tree."
                    )
    except Exception as exc:
        browser_error = error_record(exc)
        metadata["batch_error"] = browser_error
    finally:
        summary = summarize(records)
        summary["batch_error"] = browser_error
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    failures = any(not r.get("successful_task", r.get("ok", False)) for r in records)
    return int(browser_error is not None or failures)


if __name__ == "__main__":
    raise SystemExit(main())
