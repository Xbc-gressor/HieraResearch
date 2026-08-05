# Deterministic execution layer

`hieraresearch` is the only run-level coordinator. Claude is used behind a
small, revision-bound gateway for semantic judgments, bounded research, and
bounded candidate edits; it never owns lifecycle state or chooses the next
transition.

## Start a run

From the repository root, install the package dependencies and run:

```bash
python -m hieraresearch <task-name> <tag> --model <claude-model>
```

Use `--recordings <directory>` for replay, ablation, or model-free execution.
Use `--preflight-only` to validate the task environment without model calls or
objective evaluation. A blocked run requires the explicit `--resume-blocked`
flag before it can continue.

## Upstream API failure recovery

Transient provider/transport failures (HTTP 502/503 and related upstream
signatures such as “temporarily unavailable” or “no available accounts”) are
classified in `hieraresearch.upstream` and handled at the **coordinator**
boundary:

- the run stays `running`;
- durable counters on `state.json` record streak, cumulative backoff seconds,
  and the last upstream error/decision;
- the process sleeps with capped exponential backoff, then re-derives the next
  transition from artifacts (same restart safety as a process crash);
- counters reset only after a **new completed model invocation receipt** is
  observed (or on explicit `--resume-blocked`), never after a no-op artifact
  validation or a purely deterministic transition;
- a call-site `transport retry limit reached` message is coordinator-retryable
  only when the nested cause is itself upstream; candidate authoring then
  re-opens its local transport window on re-entry so backoff is not a pure
  sleep loop;
- the run hard-blocks only after the configured **failure streak** or
  **cumulative backoff wall-clock** budget is exhausted (defaults: 8 failures /
  2 hours). Exhaustion reasons are prefixed
  `upstream_failure_streak_exhausted:` or
  `upstream_backoff_wall_clock_exhausted:`.

Contract rejections, invalid-request errors, artifact corruption, and objective
budget exhaustion are **not** absorbed by this path. Manual `--resume-blocked`
still clears a prior block and resets upstream counters for a fresh budget.
Malformed upstream counter fields on disk are rejected rather than coerced.

## Durable protocol

The coordinator derives transitions from the run artifacts and persists state
under `runs/<task>/<tag>/.orchestrator/`:

1. initialize the run and run the task environment preflight;
2. create or validate the frozen background artifacts;
3. admit the provided baseline or a deterministic graph action;
4. materialize the admitted candidate and obtain a bounded implementation edit;
5. obtain the tuning contract, validate it, and run a no-score preflight;
6. run sequential Phase-A warm evaluations through the existing budget helper;
7. deterministically select at most one candidate for Phase-C deep tuning;
8. finalize tuning through the existing idempotent helper;
9. refresh bounded experience at the quiescent round boundary and stop when
   the ledger says the budget is reached or progress is impossible.

`state.json` is restart bookkeeping, not the source of truth for experiment
results. `ledger.json`, `evaluation_attempts.jsonl`, candidate reports, and the
existing `tools/` helpers remain authoritative for their respective contracts.
Completed round receipts and model invocation journals make irreversible work
auditable and allow a restart to reconcile artifacts instead of trusting chat
history.

## Model boundaries

| Python boundary | Model capability | Deterministic postcondition |
|---|---|---|
| `background.py` | bounded research/edit of frozen background outputs | background contract validation |
| `semantic.py` | structured proposal predictions and candidate idea text | semantic helper selection and ledger admission |
| `candidate.py` | bounded candidate/contract edits and evidenced failure diagnosis | syntax, contract, search-space, and preflight checks |
| `experience.py` | bounded structured evidence synthesis | experience validation and helper-owned state application |
| `tuning.py` | none | candidate selection, worker invocation, and finalization |

Judgment calls use the Messages/API backend and structured schemas. Editing
calls use the Agent SDK with explicit read roots, exact write paths, bounded
turns, and a pre-tool path policy that denies shell access. Every call records
its purpose, schema version, model, input revision, response, and outcome.

### Failed invocation receipts

A failed receipt carries two independent fields. They answer different
questions and must not be conflated:

- `replay_permitted` — may a future process reissue this exact request, bound
  to this exact input revision? Only `false` is load-bearing, and only an
  `InferenceRequestError` sets it: the provider rejected the immutable request,
  so replay would fail identically. The journal refuses such a replay.
- `disposition` — what recovery this failure actually admits, one of
  `upstream_transient`, `contract_correction_eligible`, `contract_terminal`,
  `request_rejected`, `stale_inputs`, `validation_rejected`, `interrupted`,
  `failed`. Classification lives in one place (`llm.py:_failure_disposition`).

Only `upstream_transient` drives coordinator backoff. `replay_permitted: true`
therefore does **not** mean the run will retry — it means the journal does not
forbid a future attempt. These were previously one boolean named `retryable`,
which read as an imminent auto-retry on receipts belonging to permanently
parked runs. Receipts written before the rename are durable and are never
rewritten, so the reader still honours a legacy `retryable: false`.

An evidenced candidate failure may consume one diagnostic reservation per
failure fingerprint. An accepted repair is followed by deterministic syntax,
contract/search-space, and no-score preflight validation before objective work
is retried. Unsupported or stale model output blocks the run; it is never
silently applied to newer artifacts.

## Existing deterministic machinery

The coordinator calls the stable helper interfaces in `tools/` for ledger
mutation, graph and semantic selection, objective admission, evaluation,
tuning, experience validation, and finalization. It does not duplicate those
policies or write `ledger.json` directly.

The old `autoresearch-experiment` and `tuner-orchestrator` runtime prompts are
retired. `autoresearch-hillclimb` remains a separate comparison baseline; it
is not an alternative coordinator for `hieraresearch` runs.

## Verification

```bash
python -m pytest tests -q
python tools/validate_tasks.py
python tools/validate_background.py
python tools/validate_got.py
python tools/validate_search_backends.py
```
