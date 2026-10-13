# Local System One decision service

This stdlib bridge uses the existing trained Gemma System One GGUF package and
its unchanged `scripts/gguf_adapter.py --predict`. It installs nothing and binds
only `127.0.0.1`. The adapter verifies model/projector SHA-256, processor files
and GGUF calibration before serving a prediction. No API key is needed.

From this checkout, using a new log directory for every run:

```bash
s1_repo=/path/to/Gemma-4-12B-Unified-System-One
.venv/bin/python scripts/local_s1_service.py \
  --s1-repo "$s1_repo" \
  --model "$s1_repo/artifacts/exports/s1-boolq-Q4_K_M" \
  --native-runner "$s1_repo/artifacts/gguf-runtime/bin/s1-gguf" \
  --port 8770 --ctx-size 4096 --batch-size 512 --threads 4 \
  --raw-log artifacts/local-s1/run-001/adapter.log \
  --request-log artifacts/local-s1/run-001/decisions.jsonl
```

`GET /healthz` identifies the actual package, expected weight identity, native
runner/source hashes, calibration and settings. `starting` means that the child
exists; only a successful real prediction changes it to `ready`. An unavailable
child returns HTTP 503. Startup hash checks and model loading can therefore be
included in the first decision's wall time.

`POST /decide` takes the exact S1 `DecisionRequest` JSON shape and forwards it
unchanged. Both named questions and the upstream list form are supported. For
example:

```json
{
  "state": "The local fixture shows a Next button at observed index 1.",
  "questions": {
    "operation": {
      "type": "choice",
      "instructions": "Choose the operation that advances to the next page.",
      "criteria": {"CLICK": "Click a visible control", "DONE": "The goal is complete"}
    }
  }
}
```

A successful response preserves the adapter's real typed answers, calibrated
probabilities and telemetry, adding `local_bridge` queue/transport/wall timings.
Malformed HTTP/JSON returns 400, oversized bodies return 413, actual adapter
errors return 502, an unavailable child returns 503, and the deadline returns
504. The 90-second deadline includes body reading, the serial queue and adapter
transport. Requests are never retried. A timed-out submitted request fails the
service closed so a late answer cannot be assigned to another request. Request
bodies and adapter responses are bounded to 16 MiB.

Offline flags and explicit HF/Torch/XDG/temp caches are supplied to the child;
caller credentials, auth variables and proxy settings are not inherited. All
cache and log writes stay in this checkout. Existing log paths are rejected.

The raw JSONL retains launch identities and each exact request/response.
`adapter.log` contains adapter stderr and its `/usr/bin/time -l` report;
`adapter.log.native.log` separately contains native model stderr and its own
timer report. On macOS, maximum resident set size is in bytes. These are process
RSS measurements, not GPU allocation or current system memory. The S1 path
returns decisions without generated answer tokens, so it supplies decision
latency rather than streaming TTFT or generation tokens/second. Its prediction
payload retains `preprocessing.hf_prepare_ms`; subtracting that from its latency
only yields a remainder containing native execution and other overhead.

SIGINT/SIGTERM requests orderly EOF shutdown, allowing both timers to finish.
After a bounded grace period, cleanup sends SIGTERM only to the process group
created for this launch, including any leftover descendants. It never sends
SIGKILL, restarts the model or terminates another service. If forced termination
prevents a timer report, retain that missing measurement as a limitation.
