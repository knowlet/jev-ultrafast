# Existing trained System One on this Mac

This integration replaces both Jev's decision provider and its text helper with
explicit loopback routes. It does not call TypeSafe or an external text provider,
load `.env`, train, download weights, or use the user's browser profile. Ordinary
upstream `jev` remains unchanged and is not the entry point for this benchmark.

The checked project is `knowlet/Gemma-4-12B-Unified-System-One`,
HEAD `eef847f01b63c86f05df81027a959cf3c658d54b`. Trained source weights are
`knowlet/Gemma-4-12B-Unified-System-One@66626de5cbcf8c5fecb8e9b58710805a02a42f67`,
checkpoint identity `fa96debc68209c193338d8edee4ec4258a9f603e4f2156d4688c16364703d15d`.
The existing Q4_K_M and Q8_0 packages have separate calibration sidecars. Native
startup verifies the selected files against their manifests before inference.

## Reproduce

Run from this checkout using its installed Python 3.12.11 environment. All output
paths must be fresh; the scripts refuse to overwrite results. Dependencies are
locked in `uv.lock`, including Browser Harness 0.1.13; the isolated path does not
initialize Browser Harness. The existing S1 environment and native binaries are
read-only prerequisites. No model or browser download occurs.

```sh
# Set these to existing local prerequisites. No paths are discovered or downloaded.
s1_repo=/path/to/Gemma-4-12B-Unified-System-One
chromium=/path/to/chrome-headless-shell

# Small real decision smoke (cold request, two warmups, one measured request).
.venv/bin/python scripts/run_local_benchmark.py --mode decision-only \
  --s1-repo "$s1_repo" \
  --decision-repeats 1 --output benchmark/results/repro-smoke

# Matched pure decision workloads; run these sequentially.
.venv/bin/python scripts/run_local_benchmark.py --mode decision-only \
  --s1-repo "$s1_repo" \
  --quantization Q4_K_M --decision-repeats 10 --warmup 2 \
  --output benchmark/results/repro-q4
.venv/bin/python scripts/run_local_benchmark.py --mode decision-only \
  --s1-repo "$s1_repo" \
  --quantization Q8_0 --decision-repeats 10 --warmup 2 \
  --output benchmark/results/repro-q8

# One real travel smoke; fresh DOM verification is mandatory.
.venv/bin/python scripts/run_local_benchmark.py --mode smoke \
  --s1-repo "$s1_repo" --chrome-path "$chromium" \
  --output benchmark/results/repro-travel-smoke

# Bounded task repetitions plus model microbenchmark.
.venv/bin/python scripts/run_local_benchmark.py --mode all --fixture travel \
  --s1-repo "$s1_repo" --chrome-path "$chromium" \
  --warmup 1 --repeats 3 --decision-repeats 3 \
  --output benchmark/results/repro-travel
.venv/bin/python scripts/run_local_benchmark.py --mode all --fixture click \
  --s1-repo "$s1_repo" --chrome-path "$chromium" \
  --warmup 1 --repeats 5 --decision-repeats 3 \
  --output benchmark/results/repro-click
```

`--s1-repo` is required. `--chrome-path` is required for browser modes and is
optional for decision-only measurements. The recorded runs used an existing
headless Chromium 1243 executable. `--quantization`, `--task-timeout` and repeat
budgets are explicit options; runtime defaults are 4096 context, 512 batch,
4 CPU threads and Metal offload. Text is
greedy with seed 17 and at most 128 generated tokens; prompt caching is disabled.
Native decision scoring does not sample or generate tokens. Each task is bounded
to 180 seconds and 12 decisions, with no retry of a browser mutation. The launcher
starts and closes only its own loopback services and isolated browser profile.
The owned Chromium main-process RSS is recorded separately from model RSS.

## Provider contract

`S1_DECISION_URL=http://127.0.0.1:<port>/decide` selects native typed choices.
Jev's structured instructions and candidate descriptions are serialized as JSON
strings for the existing S1 contract; the unsupported request `model` field is
omitted. The chosen operation alone may consume its matching target head. Native
distributions are checked against observed choices. A sole legal target is
selected deterministically, without a model target confidence or invented raw
answer; its compatibility action probability is the degenerate value 1. Heads
with more than 52 choices fail before inference. The package calibration was fit
on BoolQ, and does not establish navigation confidence calibration.

`S1_TEXT_BASE_URL=http://127.0.0.1:<port>/v1` plus an explicit `TEXT_MODEL` selects
the local OpenAI-compatible helper. The launcher starts llama-server using the
same selected trained GGUF, and first probes the original JSON text-helper
contract. Local requests omit authorization and reasoning extensions, reject
redirects, ignore proxies and have no external fallback or retry.

## Measurement boundaries

Cold is process startup through the first real response, including processor and
manifest verification, model loading and first kernel work. It is not a machine
reboot or an emptied filesystem cache. Warm samples exclude recorded warmups.
The fixed synthetic page in decision-only is a model microbenchmark, not browser
task success; both quantizations receive the same exact request and settings.

Decision latency is complete client request and validation time. Native throughput
is reported input/prefill tokens divided by the adapter's timed preparation and
scoring pipeline, never generated tokens per second or native-only compute speed.
The existing typed adapter drops its separate native timing field from its reply.
Decisions have no generation TTFT. Text TTFT is time
to the first streamed content delta; provider timings and actual token usage are
retained separately. Text completion tokens divided by full request duration is
explicitly an end-to-end rate, not decode-only speed.

Task time starts after initial observation and ends on stop or failure. Setup,
independent fresh DOM verification, teardown, browser work and model work are
separate. `DONE` is not success: travel requires Casa Flora plus all requested
filters; click requires the visible completion state. All attempts and warmup
failures remain in `runs.jsonl`; p95 uses linear interpolation and is descriptive
for these small sample sizes.

`/usr/bin/time -l` records adapter and native model RSS separately. `wait4` records
the owned text server and Chromium main PID. Process peaks are not simultaneous
and must not be added to infer total memory. Raw `vm_stat` before/after is retained,
without claiming free pages alone equal available memory. Existing upstream live
Google/TypeSafe/Mercury measurements are not a matched baseline for this setup.

## Checks

The offline checks use deterministic response fixtures only for plumbing tests;
their runtimes are not model performance evidence:

```sh
.venv/bin/ruff check .
.venv/bin/python -m pytest -q
node --check jev_ultrafast/static/app.js
.venv/bin/python -m hatchling build
```

The build backend installed in the isolated environment is Hatchling 1.32.4.
`uv run` was not used after its attempted global cache write was denied. No global
cache or Browser Harness user configuration was modified. See the dated result
report and raw directories for execution outcomes and remaining limitations.

New raw outputs under `benchmark/results/` stay ignored. The public evidence
copies under `benchmark/public/` redact infrastructure paths and document their
original and public file hashes; model inputs and numerical measurements remain
unchanged. The executable source hashes identify the code used for each recorded
run, including any differences from later publication edits.
