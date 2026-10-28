# Native prefix batch alignment: implementation decision

2026-10-28. Starting Jev head: `b04d3a3b18166eb7eae48dfec7dd3db11429cf35`.

## Scope and evidence before implementation

Preserve model input tokens, slot positions, candidate order, model/weight identity,
calibration, and the frozen absolute tolerances: raw `1e-4`, probability `1e-5`,
relative tolerance zero. Native parity must pass before new browser or JevBench
claims. Offline geometry tests cannot establish native numerical parity.

The published October 14 packet contains prepared inputs and 24 native responses,
but the old experimental C++ source/planner and parity runner were never committed.
The batch-aligned source receipt is not a recoverable source snapshot. This work
reconstructs a candidate from the original pinned scorer; it does not claim to
recover, compile, or reproduce the missing historical candidate binary.

| Reference | Match, license and decision |
| --- | --- |
| [System One original scorer](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/eef847f01b63c86f05df81027a959cf3c658d54b/tools/gguf/s1_gguf.cpp) | Direct integration source, MIT, source SHA-256 `8dd7f74a3ba38fda24d305c2bcbb7793d23eb38011ccb58f4f0230f677caa70e`. Adapt the existing native scorer, preserve licensing and provenance. |
| [Pinned llama.cpp](https://github.com/ggml-org/llama.cpp/tree/5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd) | Existing dependency, MIT. Use its public memory/decoding API; no dependency upgrade or alternate inference engine. |
| [llama.cpp server prefix handling](https://github.com/ggml-org/llama.cpp/blob/5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd/tools/server/server-context.cpp) | Design reference only. Its general multi-sequence/SWA checkpoint system is broader than this serial single-text scorer and will not be copied. |
| [vLLM cache manager](https://github.com/vllm-project/vllm/blob/b766f15983bd7b926c93015dd8551a6101a2ca20/vllm/v1/core/kv_cache_manager.py) | Apache-2.0. `get_computed_blocks` retains full cache blocks and recomputes the final token/block for fresh logits. Design comparison only: allocation blocks are not this scorer's decode boundaries. |
| [SGLang radix cache](https://github.com/sgl-project/sglang/blob/faa56e2b5c5c5dee60eb5dd7b6251944a21a6104/python/sglang/srt/mem_cache/radix_cache.py) | Apache-2.0. `RadixKey.match_at` finds exact token LCP before page alignment and checks cache identity. Design comparison only; no copied code or new dependency. |

The original `runtime::score` clears KV for every request and decodes each text part
in `opts.batch` chunks. Its context sets both logical batch and physical microbatch
to that value, with full SWA storage. [Metal flash-attention dispatch](https://github.com/ggml-org/llama.cpp/blob/5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd/ggml/src/ggml-metal/ggml-metal-ops.cpp)
and [matrix selection](https://github.com/ggml-org/llama.cpp/blob/5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd/ggml/src/ggml-metal/ggml-metal-common.cpp)
depend on query batch dimensions. The observed repeat failure changes a 68–89-token
last chunk to a one-token decode. That is evidence for testing the batch-shape
hypothesis, not proof of root cause or a numerical guarantee from alignment.
The pinned [server documentation](https://github.com/ggml-org/llama.cpp/blob/5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd/tools/server/README.md)
also warns that `cache_prompt` can produce different logits when batch sizes differ.
Research queries covered native prefix-cache/batch-size logit differences and
prefix-cache numerical invariance, followed by reading the cited implementation
symbols. No response-cache proxy or alternate engine is adopted.

The [memory API](https://github.com/ggml-org/llama.cpp/blob/5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd/include/llama.h)
removes positions in a half-open range and can reject partial removal. Decode errors
can leave previously processed microbatches in memory. Therefore a rejected removal,
request/decode failure, or unsupported input must invalidate prefix state and take
the original full-clear path. Cache state is committed only after all requested
logits are copied and validated. Logical/physical batch geometry must be checked;
alignment alone must never be described as batch-invariant numerical execution.

## Adopted design

Extend the existing scorer in `tools/native-prefix`, disabled unless explicitly
selected with `--reuse-prefix`. Permit only one text part, one binary question,
and a requested slot at the final token. Retain
`floor(min(exact_token_LCP, current_final_slot, previous_final_slot) / opts.batch) * opts.batch`.
Recompute the complete original final chunk and every requested logit. A shorter
prefix, unsupported request, or unavailable complete KV prefix falls back to a full
clear/decode. Retain no answers, probability rows, or selected actions.

The extra previous-slot cap follows independent source review. For a previous
512-token request extended to 601 tokens, retaining all 512 positions would retain
a chunk that previously requested logits, while the new cold prefix chunk requests
none. The pinned [Gemma graph](https://github.com/ggml-org/llama.cpp/blob/5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd/src/models/gemma4.cpp)
performs output-row selection after final-layer attention, and
[`llama_context::get_gf_res_prev`](https://github.com/ggml-org/llama.cpp/blob/5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd/src/llama-context.cpp)
selects different graph-result slots for requests with/without outputs. This is
not evidence of observed numeric failure in that extension case. Excluding the
previous final-output chunk conservatively avoids relying on its untested graph
equivalence. It does not change any of the eight historical planned counts.

## Verification and open gaps

1. Independently verify public packet hashes and pair prepared requests, native
   stdout, transport records, raw logits and calibrated probabilities.
2. Compile and run planner and native control-flow regressions without weights,
   including exact repeats, changed prefixes, batch boundaries, malformed input,
   unsupported multipart/media, failed removal and partial decode recovery.
3. Compile against the exact pinned llama.cpp API if the build toolchain is available.
4. Keep native parity unexecuted until the identical trained Q4 model/package and
   appropriate runtime are accessible. Preserve failed historical results verbatim.
5. After native parity passes, compare complete baseline/candidate task pairs and
   both identified JevBench suites. Separate model, browser and end-to-end time,
   independently verified completion, and numerical equivalence.

At investigation time this workspace is Linux x86_64, with no located trained Q4
weights, GGUF projector, original model runtime, or Metal backend. The original
24-call experiment used macOS/Metal. A CPU build is not its parity replication.
