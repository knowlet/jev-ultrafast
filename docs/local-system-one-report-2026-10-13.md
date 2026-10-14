# Local System One × Jev — October 13, 2026

The existing trained Gemma 4 12B System One ran through Jev's real decision and
text paths on an M1 Max MacBook Pro with 64 GiB memory. The native choice API,
local text helper, isolated Chromium, fixed fixtures and raw measurement pipeline
worked. **End-to-end goal success was 0/3 travel tasks and 0/5 completion tasks.**
The model skipped required travel filters and repeatedly reset the completion
fixture. These failed completion times are retained as failures, not speed wins.

## Reproduction and evidence

- [Startup and benchmark commands](local-benchmark.md)
- [Machine-readable offline recomputation](../benchmark/derived-results-2026-10-13.json)
- [All sanitized raw attempts and resource logs](../benchmark/public/2026-10-13/)
- [Sanitization receipt with original/public SHA256 and unchanged input hashes](../benchmark/public/2026-10-13/sanitization-manifest.json)

From this checkout, replay the numerical analysis without invoking a model:

```sh
.venv/bin/python scripts/analyze_local_results.py \
  --results-dir benchmark/public/2026-10-13 \
  --output benchmark/recomputed-results.json
```

The public corpus contains 104 raw text files plus its sanitization receipt,
5,748,456 bytes (5.48 MiB) total: every cold,
warmup, measured, helper-probe and developmental attempt, including failures.
Only known infrastructure path strings were replaced. Model requests, candidate
order, numeric probabilities, timings and token counts are unchanged. Source
hashes identify the code actually executed; portable CLI paths, offline analysis
and CI were added afterward in the publication checkout. Private local commits
containing home paths were retained locally and excluded from this branch.

## Exact model and runtime

| Item | Recorded identity |
| --- | --- |
| Upstream Jev | `1231850a0bf1a0c0341fe408ef1668dbbfdfac46`, MIT |
| Executed local integration | `3a06fc29ee3e7a7e7542d936fa9f7aa9fb9bbdb8` (Q4 fixed-input), `8f54b32e0d9c1509e5a2d7cb812dae507a245c26` (timing scope correction and later runs); per-file SHA256 retained |
| System One source checkout | `eef847f01b63c86f05df81027a959cf3c658d54b` |
| Trained weights | `knowlet/Gemma-4-12B-Unified-System-One@66626de5cbcf8c5fecb8e9b58710805a02a42f67` |
| Trained checkpoint identity | `fa96debc68209c193338d8edee4ec4258a9f603e4f2156d4688c16364703d15d` |
| Native llama.cpp | `5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd` |
| Text-server build | `0.5.0-dev`, build 11347, commit `5fc4f3c8c`, AppleClang 21.0.0.21000334 / Darwin arm64 |
| Native S1 binary SHA256 | `d1e81415da71954981ebc44794c90393961a8e150559a18afdf948d60f8506be` |
| Existing Python inference environment | Python 3.12.11, Torch 2.8.0, Transformers 5.17.0 |
| Jev dependencies | Browser Harness 0.1.13, httpx 0.28.1, websockets 15.0.1 |
| Owned browser | HeadlessChrome 153.0.8010.12, revision `971a7443b0c9b0a9b2860529b33331b76077ec62` |

Both quantizations bind to the same trained source. The strict existing adapter
rehashed the selected GGUF and F16 projector and checked processor/calibration
identities before real scoring. Q4 is `artifacts/exports/s1-boolq-Q4_K_M/`, Q8 is
`artifacts/exports/s1-boolq-gguf/`, relative to the verified model checkout.

| Package | Language-model bytes | Model SHA256 | Calibration temperature |
| --- | ---: | --- | ---: |
| Q4_K_M | 7,381,383,584 | `daf56f5a5a0496e4e726e504858ef8dda1f6b02cd1bc7a68e9f1beb2bf178365` | 3.232734824261939 |
| Q8_0 | 12,669,647,264 | `9d3bc61851e5b7ed8e595b84327b85271830778d0e62fa9c1349c61843f369ed` | 2.7830344470383452 |

The shared F16 projector is 122,031,744 bytes with SHA256
`cffbff05efd290fef28d084ebcb653de7594f5e26bae0eabbedf30d6c782c1d1`.
These are trained System One exports; the base model identity in conversion
receipts is provenance, not a substitute benchmark model. No weights were
downloaded, trained or overwritten. MLX and BF16 were inventoried, not benchmarked.

## Fixed-input decision comparison

Each arm had one cold request, two recorded warmups and ten measured calls. All
13 requests per arm have the identical native wire SHA256
`6bd913b50a9133510fdd9199683665eb09dc6cb314e31b12a15c5e8a12427df2`,
including candidate order. Each contains 849 native input tokens. The synthetic
indexed page is explicitly a model microbenchmark, not browser task success.

| Metric | Q4_K_M | Q8_0 | Samples |
| --- | ---: | ---: | ---: |
| Correct intended choice | 10/10 | 10/10 | 10 measured per arm |
| Client decision p50 / p95 | 6.294 / 8.289 s | 9.588 / 10.588 s | 10 per arm |
| Input tokens / adapter-scoring second, p50 | 134.937 | 88.587 | 10 per arm |
| First process-cold client request | 16.850 s | 19.132 s | 1 per arm |
| Service launch → first complete response | 17.392 s | 19.568 s | 1 per arm |
| Native model peak RSS | 9,298,345,984 bytes (8.66 GiB) | 14,583,250,944 bytes (13.58 GiB) | one process lifetime per arm |

In these sequential batches Q4's observed median client time was 34.36% lower.
The runs were Q4 followed by Q8, not alternating paired trials. Scheduler,
thermal conditions and OS filesystem caches were not controlled. The result is
limited to this fixed input and settings, and is not an agent-level speed claim.

All runs used context 4096, logical/microbatch 512, 4 CPU threads, requested Metal
offload 99 layers and a fresh per-request KV cache. Decisions score legal
candidates in a causal multi-slot input with no sampled/generated answer tokens;
therefore generation TTFT is not defined. Reported scoring throughput includes
the adapter's validation/preparation/native pipe timing. The baseline typed adapter
dropped the separate native timer, so native-only forward speed is not available
here. The subsequent telemetry fix preserves optional `native_latency_ms` as native
scoring wall time; it changes no policy and leaves all baseline raw samples intact.

## Actual browser tasks

Travel used the unchanged upstream local Lisbon/Design/Free cancellation/Casa
Flora fixture, one browser warmup and three measured fresh tabs. Completion used
the included two-button fixture, one warmup and five measured fresh tabs. The
model's input was one fixed natural-language goal; no site-specific action plan
or executor field value was hardcoded. Each task was bounded to 180 seconds and
12 decisions. Text used the same trained Q4 GGUF, temperature 0, seed 17, at most
128 output tokens and no prompt cache.

| Metric | Travel | Completion |
| --- | ---: | ---: |
| Independent goal success | **0/3** | **0/5** |
| Measured decision calls | 12 | 60 |
| Decision p50 / p95 | 10.906 / 17.618 s | 7.658 / 12.402 s |
| Failed task time p50 / p95 | 46.890 / 55.879 s | 102.274 / 106.642 s |
| Setup + task + verification + tab teardown p50 / p95 | 46.978 / 55.984 s | 102.329 / 106.698 s |
| Model wall time per task p50 / p95 | 46.807 / 55.779 s | 101.916 / 106.251 s |
| Browser work per task p50 / p95 | 0.102 / 0.107 s | 0.361 / 0.396 s |

Every measured travel run chose `TYPE_TEXT → SELECT → CLICK → DONE`. The real
helper generated Lisbon and the executor selected Design and opened Casa Flora,
but the model skipped search submission and the cancellation checkbox. Fresh
DOM verification read `Your filters: Design · Free cancellation off · Destination anywhere`.
Every completion run alternated Mark complete and Reset completion for 12
decisions, then failed its budget with `Completion not yet confirmed` in the DOM.
The browser did perform the real mutations; the policy failed to stop correctly.

The initial exploratory travel smoke was also 0/1 with 87.493 s task time; it is
retained separately and not pooled with the later primary runs. All browser
warmups failed as well. Successful-task timing distributions are empty.

## Text helper and memory

The three measured travel helper calls each reported 412 prompt tokens, 13
completion tokens and zero cached tokens. All returned the correct Lisbon value.
Helper full-request p50/p95 was 2.905/3.387 s, and first streamed content-delta
TTFT was 2.419/2.799 s (n=3). llama-server's own reported decode rate was
18.315/19.088 tokens/s (p50/p95, n=3); its convention excludes the first
prefill-produced token from the decode interval. Raw provider timings and usage
are retained. Completion tokens divided by whole HTTP duration is stored under
its separate, explicitly end-to-end rate.

Travel's native decision process peak RSS was 9,305,620,480 bytes, its text server
8,399,208,448 bytes (7.82 GiB), and the Chromium main PID 114,212,864 bytes
(108.92 MiB). Completion recorded 9,300,377,600 / 8,255,291,392 / 94,437,376 bytes
for those three processes. Native resource reports use macOS `/usr/bin/time -l`;
text and Chromium use `wait4` of owned PIDs. Adapter timers may aggregate waited
descendants. **Independent process peaks are not simultaneous and must not be
added into total physical memory.** Chromium's metric is its main PID, not the
complete process tree. Raw before/after `vm_stat` is included; free pages alone
are not claimed as available memory.

## Limits and checks

The model supports the typed local contract and JSON helper on this runtime, but
its observed goal-completion behavior is unreliable on both fixtures. The saved
temperature was calibrated for BoolQ, not navigation. A singleton target is
explicitly deterministic with no model target confidence; all other reported
candidate probabilities originate in real logits. Heads above 52 choices fail
before inference. Arbitrary websites, personal accounts, shadow roots, frames,
uploads, broader quality populations, calibrated navigation confidence and
matched Q8 browser tasks were not tested.

Cold means a newly started model process with manifest checks and first kernel
work, not reboot/OS-cache-cold. p95 uses linear interpolation and is descriptive
at these sample sizes. The upstream TypeSafe/Mercury/Google Flights timings have
different models, profiles and boundaries and are not a matched baseline.

The publication code passed 111 offline tests, Ruff, JS syntax and package build.
Offline mock tests verify plumbing only. All timing/quality claims above come
from actual local inference and browser execution. No paid model API, personal
cookies, website login, system security/network setting change, non-loopback
listener, model retraining, deployment or merge was used. Owned test processes
and profiles were closed. Denied process/sysctl/global-cache/config-directory
operations were not retried against alternate targets; Browser Harness's unused
default imports were made lazy for the authorized isolated transport.

The model's original checkout and its existing `.agents/` changes were preserved.
This report updates the public Gemma project through a separate report branch;
its newer default-branch source is not presented as the runtime measured here.

## Subsequent development result

The preserved baseline above remains unchanged. A subsequent opt-in Noul policy
completed one fresh development click task with genuine model DONE and matching
fresh DOM: 2 logical decisions, 9 actual native requests, 104.689 seconds. See
the [separate progress report](local-system-one-noul-progress-2026-10-13.md) for
the full distributions, source freeze, raw evidence and limits. Travel and
independent heldout quality remain unverified for this candidate.
