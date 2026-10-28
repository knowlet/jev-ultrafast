# Experimental native prefix reuse

This is a **reconstructed candidate**, adapted from the original System One scorer
at `eef847f01b63c86f05df81027a959cf3c658d54b`. See `origin.json` and `LICENSE`.
The October 14 experimental source was not committed in the public packet. This
directory does not recover that source or reproduce its binary hash.

Prefix reuse is **off by default**. New native numerical parity has not been
established. Keep the frozen `1e-4` raw / `1e-5` probability absolute tolerances,
with relative tolerance zero. A build or an offline test cannot accept the
candidate or establish model/browser performance improvement.

## What changes

The candidate permits caching only one text part with one binary requested slot
at the final input token. It retains exact matching input tokens only through a
complete original batch boundary:

```
cap = min(exact_token_LCP, current_final_slot, previous_final_slot)
retained = floor(cap / batch_size) * batch_size
```

For the archived repeat lengths 601/591/590/580 and batch size 512, this retains
512 tokens and recomputes 89/79/78/68. These are **planned geometry counts**, not
new measured inference counts or evidence of numerical agreement. Every requested
logit is recomputed; no answers, probability rows or actions are cached. The
formula uses the original configured batch size, rather than hardcoding 512.

The additional previous-slot cap excludes the previous requested-logit chunk
when the next input grows across a batch boundary. For example, 512→601 retains
zero and 1024→1100 retains 512. Both the cached and cold executions then have
no requested outputs in every retained chunk. The additional cap does not change
any of the eight archived input plans; graph/output-mask invariance outside this
restriction has not been established.

The native logical batch and physical microbatch must both match the original
chunk size. A complete contiguous KV range must exist before and after suffix
removal. Unsupported requests, changed geometry and failed removal take the
original full-clear/decode path. A failed request, including JSON parsing or
partial decode failure, clears both KV and cache metadata. Cache identity is
committed only after all output logits/response fields validate. The native
runtime preserves the original per-part decode loop for multipart/media input.

Telemetry under `preprocessing.prefix_cache` reports actual retained KV,
recomputed positions, alignment cap, original chunk size, fallback reason and
whether the completed input is cacheable. Existing `raw_logits`, token counts,
slots and runtime metadata remain available. `--reuse-prefix` is an explicit
experimental switch; it does not imply that parity has passed.

## Build against pinned native APIs

Use a pristine llama.cpp checkout at
`5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd`. Configuration verifies the SHA and
tracked-source cleanliness. It does not download source, dependencies or weights.

```bash
cmake -S tools/native-prefix -B tools/native-prefix/build \
  -DLLAMA_CPP_DIR=/absolute/path/to/pinned/llama.cpp \
  -DCMAKE_BUILD_TYPE=Release
cmake --build tools/native-prefix/build --parallel 1
ctest --test-dir tools/native-prefix/build --output-on-failure
```

Use backend settings appropriate to the authorized host. A portable CPU-only
compilation can additionally set `-DGGML_NATIVE=OFF -DGGML_METAL=OFF
-DGGML_CUDA=OFF -DGGML_OPENMP=OFF`. This checks the API/build contract; it does not
replicate the original Metal numerical experiment.

The scorer target is `s1-gguf-prefix`. Run `--help` without loading a model. Actual
inference requires the same local trained model, projector and checked package;
the existing bridge additionally binds an experimental binary to an explicit
SHA-256. Neither this build nor the tests modify model weights or calibration.

## Offline regressions

`test-prefix-plan` exercises the actual planner/cache transaction with fake memory:
changed prefixes, exact repeats, chunk boundaries, unsupported inputs, runtime
geometry changes, incomplete KV, failed/partial removal and incomplete decode.
It compares suffix decode schedules with the original full-decode schedules.

`test-native-runtime` includes the **actual `s1_prefix.cpp`** and supplies fake C
API implementations using the pinned headers. It links no inference libraries.
It checks final-chunk decoding, unchanged tokens and requested positions, fresh
logits, disabled/multipart fallback, native failures, response-construction errors
and malformed-JSON recovery. Its fake logits are not numerical parity evidence.

`pytest tests/test_native_prefix_cpp.py` compiles the planner independently of
llama.cpp and replays all eight archived prepared token arrays. No model, browser,
network access or external service is used by these tests.
