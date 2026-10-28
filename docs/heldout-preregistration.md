# Untouched task regression plan

This plan is fixed before any model sees these fixtures. The original click and travel
fixtures remain development data. The eight cases are two completion tasks, two forms,
two corrections of existing wrong settings/results, one missing-value abstention, and
one unsupported upload abstention. They contain only synthetic data and local resources.

`benchmark/heldout-plan.json` is the complete task, goal, initial-state and acceptance
manifest. Task order is shuffled once with seed 20261013 and then stored. Model seed is
42 and temperature is zero. Both compared implementations use that exact order and
the same model/runtime settings. Select one candidate implementation using development
observations before reading heldout results; pair baseline and candidate, alternating
their order between repetitions. Do not choose a best variant after heldout testing.
Any subsequent tuning turns these cases into development data and requires new cases.

Run all eight tasks three times per implementation, with a fresh owned Chromium profile
for each repetition of each task. Each run has at most 12 model decisions, 12 browser
actions and 180 seconds; the maximum is 288 decision requests per implementation.
Use development observations for model warmup. Every heldout repetition is scored,
including the first repetition and all timeouts, provider errors and budget failures.

The generic fixture renderer receives only `case.fixture` and the case ID. It never
receives a goal, desired settings, acceptance checks or an executor plan. The agent gets
the fixed natural-language goal and its ordinary DOM observation. Its model, action
space, DOM guards and executor remain responsible for choosing every action.

Call `READ_FRESH_DOM` only after the agent has stopped or failed. Pass its result to
`verify_case` outside the agent policy, together with agent status, the last model
decision's `choice`, executed history length, and any execution error. Do not give
checker output or gold values back to the policy. Completion requires an explicit
model DONE, all requested current and applied values, visible final status, an actual
submission, and no execution error. Negative cases require an explicit model BLOCKED,
zero executed browser actions, zero semantic interactions and the unchanged initial
DOM. Automatic stuck/budget BLOCKED and generic exceptions are failures.

Report success across all 24 runs per implementation and per task/family. Include task
time p50/p95 for all runs and separately for successful runs, all decision latency
p50/p95, actions, decision/helper requests, budget exhaustions, model/browser wall time,
and memory with its actual scope. Keep failed traces. Do not remove difficult cases,
change acceptance, extend budgets, or reclassify an attempted interaction as abstention.
This is a small synthetic regression suite, not a general web reliability estimate.

`benchmark/heldout-freeze.json` records manifest, renderer/checker, test and generated
HTML SHA-256 hashes plus the UTC freeze time. Verify those hashes before model runs.
The existing development result files are not changed. Fixture and checker tests are
offline contract tests and do not count as real-agent performance evidence.
