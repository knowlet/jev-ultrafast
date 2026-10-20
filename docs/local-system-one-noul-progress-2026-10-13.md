# Local System One Noul development result — 2026-10-13

The original local Choice operation head repeatedly selected CLICK after the
completion goal was visibly satisfied. An explicit `noul_v1` policy now asks the
trained System One binary primitive about every offered operation, ranks the
actual independent P(true) scores, and asks a real Choice head only for the
selected operation's target. A fresh development browser run completed by
CLICKing Mark complete and then selecting DONE. The independent DOM checker
confirmed the goal. This is an opt-in experimental policy, not a new default.

The same trained Q4_K_M package is used: weights revision
`66626de5cbcf8c5fecb8e9b58710805a02a42f67`, checkpoint identity
`fa96debc68209c193338d8edee4ec4258a9f603e4f2156d4688c16364703d15d`,
GGUF SHA256 `daf56f5a5a0496e4e726e504858ef8dda1f6b02cd1bc7a68e9f1beb2bf178365`,
saved temperature `3.232734824261939`, original native binary SHA256
`d1e81415da71954981ebc44794c90393961a8e150559a18afdf948d60f8506be`.
No weights, calibration or original model checkout were changed.

## Actual results

The previously denied four-call probe was retried once after explicit approval.
All four requests returned verified native responses; the old denied attempt
remains a separate zero-confirmed-execution record.

| Fixed development observation | P(true: CLICK) | P(true: DONE) | Pair ranking |
|---|---:|---:|---|
| Completion not confirmed | 0.9268275324 | 0.1201898586 | CLICK |
| Completion confirmed | 0.2103193726 | 0.8085869091 | DONE |

The subsequent smoke scored all four offered operations, including WAIT and
BLOCKED. It used one fresh owned Chromium profile, no warmups, a 180-second task
budget and at most 12 logical decisions. Per decision the policy permits at most
8 operation requests plus 1 selected-target request, sharing a 90-second deadline.
It supplies no checker result or expected label to the model.

| Metric | Actual smoke |
|---|---:|
| Genuine DONE and fresh DOM goal success | 1 / 1 |
| Browser mutations | 1: Mark complete |
| Logical decisions | 2: CLICK target 1, DONE |
| Actual native requests / confirmed executions | 9 / 9 |
| Task time after initial observation | 104.688563 s |
| Model wall time, counted once per logical call | 104.609600 s |
| Browser task operations | 0.077424 s |
| Native process peak RSS | 9,245,802,496 bytes |

Initial scores were CLICK 0.926282, WAIT 0.100499, DONE 0.124394 and BLOCKED
0.107385. After the click, DONE 0.813833 exceeded CLICK 0.210540, WAIT 0.129282
and BLOCKED 0.162188. These are genuine binary distributions, not categorical
operation probabilities. The selected target retains its genuine conditional
Choice distribution. The UI labels operation scores as independent P(true).

## Limits and resource boundaries

One development run does not establish stable p50/p95, heldout generalization,
navigation calibration or an improvement in latency. The preserved baseline
still reports 0/5 completion and 0/3 travel success; failed-task timings are not
compared to the new successful task as a speedup. This intervention jointly
changes primitive, prompt and head layout, so it does not isolate one factor.

Nine serial scores account for nearly all task time. Adapter-scoring input token
rates are not generation throughput or GPU-only compute. This run used the
unchanged original adapter, which does not return its separate native timer.
Native choices have no streaming TTFT. No text helper was needed for this click
run. Model and fixture services exited cleanly; owned Chromium was reaped after
SIGTERM (exit 143). Its main-process RSS is not a whole-tree peak. The outer
adapter timer includes descendants and reports the same peak as native; peaks
must not be added. VM counter epochs differ across snapshots, so deltas cannot
be attributed to this model run.

A workspace token-prefix experiment is being prepared to reduce repeated
prefill. It was not used in this smoke and has no validated speed claim yet.
Independent heldout cases and their checker remain unchanged and have not been
run with this candidate.

The publication integration passed 200 offline tests, Ruff and JavaScript syntax
checks. Those tests verify provider contracts and accounting; their runtimes are
not model performance measurements.

## Evidence and reproduction

[Paired probe](../benchmark/public/noul-development-2026-10-13/paired/summary.json),
[actual browser trace](../benchmark/public/noul-development-2026-10-13/smoke/runs.jsonl),
[service requests](../benchmark/public/noul-development-2026-10-13/smoke-service/decisions.jsonl),
[source freeze](../benchmark/public/noul-development-2026-10-13/paired-service/development-smoke-freeze.json),
[raw file selection](../benchmark/public/noul-development-2026-10-13/source-selection.json),
and [sanitization manifest](../benchmark/public/noul-development-2026-10-13/sanitization-manifest.json)
retain every actual attempt, distribution and timing. Only infrastructure paths
were redacted; model requests and numeric results are unchanged. The original
baseline corpus is kept separately and is not rewritten.

Start the existing native service as documented in [local-s1-service.md](local-s1-service.md),
and a loopback static fixture server. Use explicit existing paths:

```sh
S1_DECISION_URL=http://127.0.0.1:8770/decide \
  .venv/bin/python scripts/local_benchmark.py \
  --policy noul_v1 --mode smoke --fixture click --warmup 0 \
  --max-decisions 12 --task-timeout 180 \
  --fixture-url http://127.0.0.1:8766 \
  --chrome-path /absolute/path/to/existing/chromium \
  --model-metadata /absolute/path/to/current-verified-model-metadata.json \
  --output benchmark/results/noul-click-fresh
```

The metadata must describe the actual current service and contain
`decision_service_identity.model_file` and `temperature`; sanitized public
paths are provenance placeholders, not executable local model paths. The
policy verifies those explicit values for every reply and has no remote fallback
or retry. The convenience launcher also accepts `--policy noul_v1`; it records
an additional separate process-cold decision probe and owns its services.
