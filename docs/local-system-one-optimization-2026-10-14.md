# System One / Jev: reliability fixes and real native prefix parity

2026-10-14. A newly authorized read-only check found the three old owned process
groups absent, and one newly authorized CPU build succeeded. A subsequent bounded
probe completed **24 real native scores** on the existing trained System One Q4
package. Candidate prefix reuse **failed the unchanged numerical parity gate**;
no new browser task was run and no accepted speedup is established. The October
13 window remains **6/24 executed, 2 passed, 4 failed, 18 unexecuted**, with no
paired baseline. Model, calibration, fixtures and archived raw results are intact.

The substantive latency target is repeated native prefill. All 93 measured
requests are distinct; 86 are operation scores and 7 are targets. Operations in
the same logical decision carry identical passage data but different candidate
questions. The original native runtime clears KV between requests and decodes
the full input again. Repeated passage serialization accounts for 226,279 of
310,641 operation-body bytes (72.843%); this is **not** a token-prefix size or a
speedup estimate. [Derived evidence](../benchmark/public/offline-optimization-2026-10-14/analysis.json)
contains source pins, request hashes, pointers and a public-only recomputation recipe.

| Existing measured server scope | Total | p50 | p95 | maximum |
|---|---:|---:|---:|---:|
| HF preparation | 352.691 ms | 3.578 ms | 6.260 ms | 8.353 ms |
| Queue | 9,024.625 ms | 0.079 ms | 0.500 ms | 4,062.440 ms |
| Adapter scoring | 719,257.021 ms | 6,313.256 ms | 15,091.703 ms | 15,870.718 ms |

HF preparation contributes about **0.049%** of adapter wall. Optimizing Python
serialization or HTTP connection setup cannot materially resolve the observed
43.745-second logical-decision p50. A response cache would have no measured
exact-request hit. Checkbox controls were present in the selected target head;
the missed checkbox was a model ranking failure, not evidence for a hardcoded
executor repair. Frozen failure answers were not used to change selection.

The bounded reliability patch fixes three implementation defects:

- Own shared-deadline expiry now has a nonempty NoulDecisionTimeoutError with
  budget_exhausted=true. It cannot be confused with an external/provider timeout,
  even when provider cleanup converts or suppresses cancellation. Payloads,
  operation order, target selection and the original 90-second limit are unchanged.
- KeyboardInterrupt/SystemExit save an in-progress benchmark attempt once, then
  stop. Cancelled runs fail while completed provider-call timing stays intact.
  Callback/final-save errors cannot replace the original stop. Browser verification
  is skipped after interruption; the cleanup scope is an Agent tab whose browser
  factory already returned. Broader Browser/Chromium constructor and outer cleanup
  exception boundaries are outside this patch and remain unverified.
- Bridge initialization after Popen is guarded. The original failure survives
  cleanup errors; owned resource closure is attempted in finally and group SIGTERM
  is attempted at most once. Leader exit, group uncertainty and close errors stay
  separate. EOF or closed logs never imply a verified full-group/GPU shutdown.

Validation: **334 offline tests**, Ruff, JavaScript syntax, sdist/wheel build and
independent source reviews pass. Tests use mocked transports/processes/signals;
they establish contract behavior, not actual task completion, safe live shutdown,
GPU release, lower latency or improved success rate. The separate real native
measurements below do not establish improved task completion.

The one performance candidate is **exact-input native KV prefix reuse**, keeping
all request tokens, question order, output slots and calibrated distributions.
It recomputes the changed suffix and every requested logit; it stores no answer.
Simple multihead batching changes the shared state.question and later causal slots,
and prompt trimming changes model input. Neither is an equivalent shortcut.

Four prior client cancellations completed on the server. Three next-case first
requests queued for 1.674/4.062/3.278 seconds (0-based decision-event indices
35/73/86, including warmups). These are separate measurements, not exact postcancel
compute duration. Shortening the bridge read timeout alone could fail closed and
shut down the whole adapter earlier; releasing its lock before draining a response
could misassociate scores. Neither is introduced as a completion optimization.

The [comparison plan](../benchmark/public/offline-optimization-2026-10-14/comparison-plan.json)
is the preserved **pre-execution snapshot**: its zero-call/unexecuted fields describe
preparation before the approvals. The actual execution has a separate
[manifest](../benchmark/public/native-prefix-parity-2026-10-14/parity/manifest.json),
[summary](../benchmark/public/native-prefix-parity-2026-10-14/parity/summary.json) and
[independently recomputed audit](../benchmark/public/native-prefix-parity-2026-10-14/verified-summary.json).
It ran original → candidate-off → candidate-on, eight scores each, serially.
Each mode used preCLICK twice, preDONE twice, postCLICK twice, postDONE twice from
the existing development observations; first response 60 seconds, subsequent
responses 30 seconds, shared 900-second cap, zero retries/readiness calls.
All 24 calls returned status ok with model execution confirmed. Numerical
acceptance requires raw absolute delta ≤1e-4, probability delta ≤1e-5, rtol=0,
matching identity/tokens/slots/labels/rankings, and real KV telemetry. The tolerances
were fixed before execution and have not been relaxed.

Candidate-off agrees exactly with the original in all eight comparisons.
Candidate-on agrees exactly at indices 0/2/4/6. All four identical-repeat inputs
at indices 1/3/5/7 fail both numerical tolerances, despite matching truth labels
and CLICK/DONE rankings:

| Candidate-on index | Retained / recomputed positions | Maximum raw delta | Maximum probability delta |
|---:|---:|---:|---:|
| 1 | 600 / 1 | 0.006593704224 | 0.000096414012 |
| 3 | 590 / 1 | 0.008621215820 | 0.000034497538 |
| 5 | 589 / 1 | 0.009721755981 | 0.000274931670 |
| 7 | 579 / 1 | 0.017385482788 | 0.000671539283 |

The cache-effectiveness gate passes: this is actual KV retention plus fresh
requested logits, with no answer cache. Model/source before-and-after hashes
match. Numerical acceptance is nevertheless false, and the probe exits 1.
The nonidentical transitions retained/recomputed 526/65, 93/497 and 515/65
positions and matched the original numerically. Duplicate-input timing is a
mechanism check; none of the 93 earlier measured requests is an exact duplicate.

The following timings are descriptive for this fixed-order run. Warm means
indices 1–7 (n=7); first client wall includes process startup, load and one real
score, while first native wall covers that score only. Units are milliseconds.

| Mode | First client / native | Warm native p50 / p95 | Peak native lifetime RSS |
|---|---:|---:|---:|
| Original | 11303.498 / 9196.893 | 8170.278 / 8490.242 | 9,297,805,312 bytes |
| Candidate-off | 18143.920 / 3709.318 | 4074.395 / 4177.261 | 9,302,048,768 bytes |
| Candidate-on | 6331.247 / 2875.662 | 169.212 / 2704.876 | 9,301,983,232 bytes |

Thermal state, OS file cache and concurrent lightweight work were not controlled;
the original and candidate-off themselves differ substantially in timing. The
fast on-repeat rows fail numerical parity. These measurements support neither
an accepted/stable speedup nor an end-to-end task improvement. TTFT is unavailable
for this nonstreaming scorer, and autoregressive output tokens are zero. Cached
throughput is not inferred from full input length divided by latency.

The executed candidate binary SHA is
`38f2a1700f7809a6e971cacbeeaa470e32a85d3339233e8cb2891b4a5dad9f1e`;
its [build receipt](../benchmark/public/native-prefix-parity-2026-10-14/build-receipt.json)
binds source `fd9291f18b33c669d8c9157090893a2371a178689d3a4690fb90ad6f0e7089d1`
and header `52aa0ee9a04d5c59513717934bb3a9ca413facc443648eaacea019c84a84d1bb`
to pinned llama.cpp `5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd`.
The existing model GGUF SHA is
`daf56f5a5a0496e4e726e504858ef8dda1f6b02cd1bc7a68e9f1beb2bf178365`;
the trained merged checkpoint SHA is
`fa96debc68209c193338d8edee4ec4258a9f603e4f2156d4688c16364703d15d`,
with unchanged calibration temperature 3.232734824261939. No weights were downloaded,
trained, replaced or recalibrated.

The three mode wrappers exited 0 with forced_cleanup=false. EOF, direct wrapper
exit and reader closure establish closure of those owned modes; the finally
cleanup may send SIGTERM to a residual group and has no individual signal ledger.
Full absence of every new process group and a no-signal claim remain unverified.
The separately authorized [old-group signal-0 receipt](../benchmark/public/native-prefix-parity-2026-10-14/owned-group-signal0.json)
records ESRCH for only 24238/24239/24241, with no termination. It proves absence
at the October 14 check, not the exact October 13 shutdown time.

The original output hash ledger contains 21 entries: all 20 durable outputs
match; one temporary CPU helper no longer exists at audit. Its absence was not
repaired or explained by inference. The numerical conclusion uses the durable
prepared/native/transport records. The public packet contains these outputs,
build/approval/audit receipts and source-selection metadata, with only known
infrastructure paths redacted. Its
[sanitization manifest](../benchmark/public/native-prefix-parity-2026-10-14/sanitization-manifest.json)
maps original/public hashes and preserves model requests and numbers. Original
execution hashes must not be compared directly to redacted file hashes.

The next source-only candidate aligns retained positions downward to the original
512-token batch boundary:
`floor(min(exactLCP, first_requested_slot)/512)*512`.
For these repeats it would retain 512 and recompute 89/79/78/68 positions. The
[source preparation receipt](../benchmark/public/native-prefix-parity-2026-10-14/source-prep-batch-aligned.json)
records an uncompiled/unexecuted variant and independent static review. The
[pinned Metal selector source](https://github.com/ggml-org/llama.cpp/blob/5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd/ggml/src/ggml-metal/ggml-metal-ops.cpp#L2949)
supports batch-dependent dispatch as a mechanism hypothesis; no kernel trace
establishes causation. The change depends only on original chunk geometry, not
scores or labels. It is not yet a verified correction.

The one authorized compiler invocation was consumed: exit 0, 20.437 seconds,
one translation unit, existing libraries, 120-second limit. Additional compilation
is awaiting new relevant human authorization; no reroute or retry was attempted.
The already authorized bounded runtime test remains conditional on a successful
fresh build and the same strict parity gate. Only then may one development task
pair compare prefix off/on with isolated profiles, identical fixture/settings,
one model at a time, 600-second total cap, genuine model DONE plus fresh DOM, and
separate model/browser timing. The pair is currently unexecuted and would not
establish heldout generalization or stable percentiles. Audit/source receipts are
historical snapshots; their broader future-authorization wording does not request
runtime permission again. Local signature verification remains unconfirmed and
has not been retried through another route.
