# Design: Run-level upstream API failure recovery

**Date:** 2026-08-03
**Status:** implemented (design A)

## Problem

Smoke run `0802-tb-sn-5-smk7` repeatedly reached durable `blocked` solely because
Claude Messages returned transient **502** / **503** (upstream unavailable, no
available accounts) on LLM-bound transitions. Receipts already marked these
`retryable: true`, but `ExperimentCoordinator.run` treated every
`InferenceError` as a hard stop requiring manual `--resume-blocked`.

## Decision

**Design A — coordinator outer recovery loop**

1. Classify retryable **upstream/transport** failures in a dedicated pure module
   (`hieraresearch.upstream`).
2. On match, stay in `running`, persist streak / cumulative backoff, sleep with
   capped exponential backoff, re-derive the next transition from durable
   artifacts (no side-effect replay beyond what restart already allows).
3. Hard-block only when **consecutive streak** or **cumulative backoff wall
   clock** exceeds policy caps.
4. Reset counters after any successful model-bound step (and on
   `--resume-blocked`).

## Non-goals

- Auto-reopen runs already blocked before this change without `--resume-blocked`.
- Infinite retry without caps.
- Treating contract/request/schema/artifact failures as upstream.
- Changing objective budget accounting.

## Defaults

| Knob | Default |
|------|---------|
| max streak | 8 |
| max cumulative backoff | 2 hours |
| base backoff | 30s |
| max single backoff | 15 min |
| multiplier | 2.0 |

## Ownership

- `upstream.py` — classification + pure decision + strict counter parsers.
- `coordinator.py` — apply decision, sleep, durable counters, block reason;
  reset counters only on completed invocation receipt growth or `--resume-blocked`.
- `candidate.py` — on re-entry after local transport exhaustion, reopen the
  local window when `last_error` is upstream-class so outer backoff can reach
  the provider again.
- `CoordinatorState` — optional fields (schema_version remains 1); strict parse.

## Review fixes (post Codex findings)

1. Do not treat bare local `transport retry limit reached` as upstream; nested
   cause must match provider signatures. Candidate reopens local windows on
   re-entry for upstream last errors.
2. Do not reset upstream counters after no-op `background.ensure` or every
   successful transition — only after a new completed invocation receipt.
3. Reject malformed durable counters (bool, negative, non-finite, numeric strings).
