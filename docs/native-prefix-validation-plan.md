# Native prefix validation and paired task evaluation

This plan preserves numerical acceptance separately from task quality and timing.
The reconstructed batch-aligned candidate is an opt-in experiment. Its successful
CPU build or offline tests do not enable it by default or authorize a claim of
model, browser, or task improvement. No benchmark below was executed by writing
this plan.

## 1. Native parity first

Use the original trained Q4 package, its original calibrated temperature
`3.232734824261939`, and the original native runtime settings. The historical
package identity is recorded in the October 14 packet; the language-model GGUF
SHA-256 is `daf56f5a5a0496e4e726e504858ef8dda1f6b02cd1bc7a68e9f1beb2bf178365`.
Check the complete package, including projector and conversion/calibration files,
against its manifest before starting. Do not substitute another public Gemma
checkpoint or a freshly quantized file for the trained model.

Build against llama.cpp `5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd` on the target
backend. Retain source, compiler flags, linked-library hashes, binary hash, device,
OS, batch and microbatch settings. A Linux CPU build is source/API verification;
it cannot settle a macOS Metal numerical discrepancy.

Replay the saved `parity/prepared.jsonl` payloads, without preparing a new prompt,
in three separately started serial modes: original, candidate prefix off, candidate
prefix on. Preserve all eight request identities/order per mode and all 24 outputs,
including failures. The original probe limits were 60 seconds for each mode's
first response, 30 seconds for subsequent responses, 900 seconds overall, no
retries and no hidden readiness scores. Use a new output directory and never
overwrite the October 14 packet. Capture the same prepared request bytes/hashes,
native stdout, client wall time, stderr, lifecycle, raw logits and typed outputs.

Require **raw absolute delta <= 1e-4, probability absolute delta <= 1e-5, rtol=0**,
with exact token/slot/question/model/calibration identity, finite shaped outputs,
matching labels and candidate rankings, and genuine retained-KV/fresh-logit
telemetry. A missing result or unsupported identity is a failure, not an omitted
comparison. Probability comparison must precede any benchmark renormalization.
These bounds are immutable in the offline auditor; do not use the much looser
coherence relation tolerance as a native parity threshold.

The repeat cases diagnose a mechanism. They are not representative latency cases:
the 93 earlier browser-derived requests were all distinct. For an improvement
claim, also replay varied nonidentical operation candidates from the same task
observations under a frozen, paired request order. Record cold/load and warm
scoring separately and balance mode order for timing. Do not derive throughput
from the entire prompt length when only a suffix is recomputed.

## 2. Real browser tasks after native acceptance

Use the unchanged goal text, fixtures, policy, model, text helper, action limits,
timeouts and completion verifier for each baseline/candidate pair. Each run starts
with a fresh isolated owned browser profile and reset fixture state. Use one model
process at a time. Preserve the model's actual DONE/BLOCKED decision and verify
the final outcome independently from a fresh DOM observation. Do not special-case
failed tasks, hardcode form values, replay an answer, or change candidate order.

Begin with a bounded development pair. The previous plan allowed 600 seconds for
that pair; record partial/interrupted runs rather than inferring success. Then
complete both arms of a preregistered frozen task population with a sufficient
window. If a run is missing, report both scheduled and executed denominators.
The old 6/24 partial arm has no paired baseline and cannot serve as evidence of
this change's benefit.

Report task wall time (including failed runs and timeouts), model client/native
scoring time, text-helper time, queue time, browser snapshot/execution/wait time,
physical model calls and logical decisions separately. Do not subtract overlapping
client/server measurements or report successful-run latency alone. Percentiles
must include sample size and population; a one-pair smoke test cannot establish
stable p95 or heldout generalization.

## 3. Both JevBench tracks, with distinct answer caches

| Track | Frozen evaluator/data source | Population and reporting |
| --- | --- | --- |
| JevBench coherence | [JevBench/jevbench@e18733694623aa93058e279c3534f8b8e2edefa3](https://github.com/JevBench/jevbench/tree/e18733694623aa93058e279c3534f8b8e2edefa3) | `jevbench-mini`: 50 relations, 1,200 tests, 1,248 distinct requests; retain REP/BAT/MEA/LOG/CHO, coverage and invalid responses. |
| BenchmarkHeaven accuracy | [fstandhartinger/jevbench@bb05a335bc809e61b20c0f745d25499a82b326fc](https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc) | `legacy_public231`: original 72 + easy 48 + hard 111. This is public development data, not sealed P300/S1200 or a leaderboard submission. |

Use the already implemented System One integration at
[`eef847f01b63c86f05df81027a959cf3c658d54b`](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/tree/eef847f01b63c86f05df81027a959cf3c658d54b),
especially [`scripts/run_jevbench_public.py`](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/eef847f01b63c86f05df81027a959cf3c658d54b/scripts/run_jevbench_public.py)
and [`configs/benchmarks/jevbench-public.json`](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/eef847f01b63c86f05df81027a959cf3c658d54b/configs/benchmarks/jevbench-public.json).
It verifies upstream scoring code and dataset hashes and isolates the legacy
scorer's module name. Both projects are called `jevbench` in Python; ordinary
imports after installing both can select the wrong evaluator.

The [mini suite manifest](https://github.com/JevBench/jevbench/blob/e18733694623aa93058e279c3534f8b8e2edefa3/src/jevbench/suites/jevbench-mini/manifest.json)
pins `tests.jsonl.gz` to
`0dce8edcf757a697865899731c2ea68ddaffeae553ceb0c7181008f08f961391`.
The public231 canonical dataset hash is
`dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`.
Validate the remaining source/data hashes from those manifests before execution.

For coherence, use the existing typed `standard_predict(state, questions)` adapter
with `jb.from_callable`, `suite="jevbench-mini"`, `concurrency=1`, and the unchanged
official relation tolerance `0.05`. Use a new, distinct cache directory for every
arm and include the model revision, candidate binary SHA and prefix mode in the
backend name. The evaluator otherwise defaults to `.jevbench-cache` and can reuse
old answers. Preserve within-arm official request de-duplication, but report cache
hits separately from physical native calls and model latency. Any `jb.check()`
preflight is an additional real request and must be counted.

For public231, use the verified runner's `run_public(predict, tasks_root, output,
identity)`. Keep strict validity, renormalization indicators, original probabilities,
errors and all 231 task records. Do not flatten Choice/Score or multi-question
requests into the binary cache path; unsupported requests use full native decode.

Every artifact must say whether it came from new model execution or offline
recalculation. `Report.load()` and replaying historical raw responses are useful
checks, but cannot fill new native/model/browser result slots.
