# Batch-aligned native prefix candidate: compiled, parity still unexecuted

2026-10-28. Starting Jev revision:
`b04d3a3b18166eb7eae48dfec7dd3db11429cf35` (Draft PR #1).

The native candidate is now committed as reviewable source, compiles against the
exact pinned llama.cpp APIs, and passes offline control-flow and archived-token
regressions. **It remains disabled by default. No new model, browser, or JevBench
result was produced, and no performance improvement is claimed.** The historical
numerical failure is independently reproducible at the unchanged `1e-4` raw and
`1e-5` probability absolute tolerances, with `rtol=0`.

## Reproduced discrepancy

The new [stdlib auditor](../scripts/analyze_native_prefix.py) verifies all **28/28**
public packet hashes/byte counts, binds all **24/24** native responses to prepared
inputs, transport and typed records, and recalculates probabilities from raw logits
and the original calibrated temperature. It does not use the saved summary's
verdict as an answer. The original October 14 evidence is unchanged.

| Historical mode/cases | Raw maximum absolute difference | Probability maximum absolute difference | Frozen gate |
| --- | ---: | ---: | --- |
| Candidate off, all 8 | 0 | 0 | 8/8 pass |
| Candidate on, indices 0/2/4/6 | 0 | 0 | 4/4 pass |
| Candidate on, repeat index 1 | 0.006593704224 | 0.000096414012 | Fail both |
| Candidate on, repeat index 3 | 0.008621215820 | 0.000034497538 | Fail both |
| Candidate on, repeat index 5 | 0.009721755981 | 0.000274931670 | Fail both |
| Candidate on, repeat index 7 | 0.017385482788 | 0.000671539283 | Fail both |

All labels and CLICK/DONE rankings agree. That does not override either numerical
bound. Typed probabilities recalculated from the saved logits differ by at most
`2.220446049250313e-16`, so the reported failure is not a typed-response conversion
or file-pairing discrepancy. [Derived audit JSON](../benchmark/public/native-prefix-alignment-2026-10-28/historical-audit.json)
retains every comparison and the distinction between observed and planned counts.

## Source correction and why it is conservative

The PR held receipts and hashes for the old native candidate, but neither its C++
source nor the original parity runner. The new candidate is **reconstructed from
the original System One scorer**, not recovered from the missing snapshot:

- Original source: [`tools/gguf/s1_gguf.cpp@eef847f`](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/eef847f01b63c86f05df81027a959cf3c658d54b/tools/gguf/s1_gguf.cpp), verified SHA-256 `8dd7f74a3ba38fda24d305c2bcbb7793d23eb38011ccb58f4f0230f677caa70e`.
- New implementation/build/test/provenance: [`tools/native-prefix`](../tools/native-prefix/).
- Existing dependency remains llama.cpp `5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd`; MIT attribution is preserved.

The failed repeats retained all but one token. They changed the last query batch
from 89/79/78/68 tokens to one token. The pinned Metal source selects different
kernels based on query batch dimensions. This supports the proposed mechanism,
but there is no kernel trace establishing which operation caused the observed
drift. The [research record](research/native-prefix-batch-alignment.md) documents
the source references and alternatives.

The new planner keeps complete original decode chunks only:

```text
cap = min(exact_token_LCP, current_final_slot, previous_final_slot)
retained = floor(cap / original_batch_size) * original_batch_size
```

The previous-slot cap also prevents retaining an earlier chunk that requested
logits when a later, longer request would use that chunk without outputs. For
example, 512→601 tokens retains zero and 1024→1100 retains 512. This narrows an
untested graph-shape assumption and leaves all historical eight-row plans intact.

| Historical transition | Observed old retained / recomputed | New planned retained / recomputed |
| --- | ---: | ---: |
| Initial 601-token input | 0 / 601 | 0 / 601 |
| Repeat 601 | 600 / 1 | 512 / 89 |
| Changed candidate, 591 tokens, LCP 526 | 526 / 65 | 512 / 79 |
| Repeat 591 | 590 / 1 | 512 / 79 |
| Changed state, 590 tokens, LCP 93 | 93 / 497 | 0 / 590 |
| Repeat 590 | 589 / 1 | 512 / 78 |
| Changed candidate, 580 tokens, LCP 515 | 515 / 65 | 512 / 68 |
| Repeat 580 | 579 / 1 | 512 / 68 |

Those right-hand values are compiled planner results, **not new model telemetry
or a speedup estimate**. Only one text part, one binary question and a final-token
slot are eligible. Actual logical batch and microbatch must match the original
chunk size. Unsupported inputs use full decode. Failed suffix removal, incomplete
KV, invalid input, partial decode, missing/nonfinite logits and JSON parsing errors
clear both KV and metadata. Complete outputs must validate before cache commit.
Every requested logit is recomputed; there is no answer cache.

## Completed verification

| Check | Result and scope |
| --- | --- |
| Existing baseline before edits | 350 Python tests and Ruff passed. |
| Final Python suite | **407 passed**; includes 55 auditor/corruption regressions and two compiled planner integrations. |
| C++ planner | **25,913 assertions**, plus replay of all eight archived token arrays; checks suffix schedules and failure recovery. |
| Actual scorer control flow | Includes the real `s1_prefix.cpp` with fake C APIs; verifies tokens, slots, fresh output, fallback and error/JSON recovery. No inference library or model is used by this test. |
| Pinned CPU native build | GNU C/C++ 13.3.0, CMake 3.31.6, Linux x86_64; full scorer links against real pinned llama/mtmd/ggml libraries. Final source hashes remain unchanged across build. |
| CTest | **2/2 passed** (`prefix-plan`, `native-runtime-offline`). |
| Other required checks | Ruff, both JavaScript syntax checks, sdist/wheel build and whitespace check passed. |
| CI coverage | Existing Python contracts plus a new separate pinned native build and CTest job. Inspect checks for the actual pushed commit; an old green run is not evidence for this patch. |

The local CPU binary SHA-256 is
`31008524dd13f6f41f9b84dd05fee7835d37ae7187b22835b50e9cb4a33f92e2`.
The [source-bound build receipt](../benchmark/public/native-prefix-alignment-2026-10-28/build-receipt.json)
records source, binary and linked-library hashes. The
[offline validation record](../benchmark/public/native-prefix-alignment-2026-10-28/offline-validation.json)
records checks separately from unexecuted inference. Build time is not model latency.

Reproduce the historical audit:

```bash
python scripts/analyze_native_prefix.py
# Expected exit 1: intact historical evidence, numerical parity failed.
python scripts/analyze_native_prefix.py --audit-only
# Exit 0 means that historical failure was reproduced, never candidate acceptance.
python scripts/analyze_native_prefix.py --audit-only --output /new/path/audit.json
# Refuses existing files and outputs within the immutable source archive.
```

## Metrics and remaining execution gap

| Dimension | Current verifiable result | New candidate status |
| --- | --- | --- |
| Native numerical equivalence | Historical candidate-off 8/8 exact; candidate-on four repeat failures | **Unexecuted** on the reconstructed candidate |
| Model latency | Earlier logical decisions p50 **43.745 s**, p95 **76.569 s** | No new model measurement |
| Browser latency | Previous browser evidence remains separate from model timing | No new browser measurement |
| Real task completion | Original general **0/5**, travel **0/3**; frozen arm **6/24 executed, 2 passed, 4 failed, 18 unexecuted**, no paired baseline | No new task execution or completion improvement |
| JevBench coherence | Exact suite/source identified in the validation plan | Not rerun |
| JevBench public231 | Exact evaluator/data/source identified in the validation plan | Not rerun |

This workspace has no located original trained Q4 language model/projector package
or original native model environment, and it has no macOS Metal backend. No weights
were downloaded, substituted, trained or recalibrated. Consequently the strict
native gate remains closed, and browser/JevBench reruns were not started.

The [paired validation plan](native-prefix-validation-plan.md) identifies both
JevBench sources, complete result populations, immutable thresholds, isolated
per-arm answer caches and timing scopes. The next evidence-producing step is a
new source/binary-bound original/off/on native capture on the target environment.
Only a passing native gate can precede complete paired real-task measurements and
both benchmark tracks. No higher-level benchmark score can replace that gate or
independent task completion.
