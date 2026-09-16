# Driver observability

`tools/harness_watch.py` and the Claude Code status lines are retired along
with the `.claude/` runtime. Observability now comes from the driver itself:
human-readable progress lines on stdout plus a structured event log at
`<run_dir>/driver_events.jsonl`. Every event is emitted as one stdout line
(`[driver] <kind>: {...}`) and one JSONL row, so a run can be watched live
from the terminal or after the fact from the log. There is no external
service dependency.

## Event kinds

- `setup` — run directory initialized (`task`, `tag`); emitted by the
  hillclimb loop on fresh runs.
- `objective_brief` — the run's `objective_brief.json` was written
  (`metric`, `aspirational_target_score`, `target_source`); the declared
  aspiration bar is diagnostic only and never a decision input.
- `session_start` — a role SDK session opened (`role`, `invocation_id`,
  `resume`).
- `session_end` — the session returned (`role`, `invocation_id`,
  `session_id`, `is_error`, `num_turns`, `total_cost_usd`, `usage`).
  `session_start`/`session_end` pairs carry per-session cost and token usage.
- `corrective_followup` — a postcondition or receipt check failed and the
  driver sent a corrective message in the SAME session (`role`,
  `invocation_id`, `attempt`, `problems`).
- `reconcile_recovery_row` — hillclimb crash recovery appended a TSV
  recovery row for a reserved attempt that was interrupted before its
  outcome row (`index`).
- `metadata_mismatch` — resume-time provenance drift (model, SDK/CLI
  version, or prompt hashes differ from `run_metadata.json`); record and
  warn, never refuse (`warning`).
- `blocked` — the loop hit a hard stop condition (`reason`).
- `blocked_secondary` — another seat hit a stop condition while the run was
  already blocked (`reason`); only the first block persists the phase.
- `seat_started` / `seat_finished` / `seat_skipped` — one admitted seat's
  implementation on the session channel (`run_id`; `session_wait_seconds`
  is the time it waited for a session slot, `seconds` its wall clock).
- `seats_completed` — a generation's seats all returned (`run_ids`,
  `session_concurrency`, `wall_seconds`); compare with the per-seat seconds
  to read the realized overlap. Throughput diagnostics only.
- `candidate_unevaluated` — the driver resolved a zero-attempt candidate at
  a reached stop condition after the extractor reported it (`run_id`).
- `seat_error_masked` — a seat's non-block error was superseded by another
  seat's block (`error`); diagnostic only, the block still unwinds the run.

Per-job timing (`lease_wait_seconds`, `eval_seconds`, `devices`) lives in the
job record under `<run_dir>/driver_jobs/`.

## Correlation

Receipts, sessions, and events join on `invocation_id`: the accepted receipt
for an invocation lives under `<run_dir>/receipts/`, the session id is
persisted at session init (a killed session can be resumed via
`resume=<session_id>`), and every `session_*` / `corrective_followup` event
carries the same `invocation_id`. `run_metadata.json` pins the run's model,
SDK/CLI version, permission policy, and prompt hashes for A/B provenance.

Token/cost attribution is per session via `session_end`
(`total_cost_usd`/`usage`). Drift attribution across a whole run — the old
harness_watch roll-up — is a non-goal of the driver; it can be rebuilt on
top of the events log if needed.

## Artifact-only fallback

Lifecycle checks still derive from `ledger.json`, `framework_cfg.json`, and
`loop_state.md` (`driver/status.py` derives the same phases the retired
harness_watch snapshot did):

```bash
python tools/ledger.py brief --ledger runs/<task>/<tag>/ledger.json
```

`ledger.py set-phase` is the only supported lifecycle transition. It refuses
premature completion and requires a concrete reason for blocked runs:

```bash
python tools/ledger.py set-phase --ledger runs/<task>/<tag>/ledger.json \
  --phase blocked --stop-condition "<reason>"
```

An explicit evaluation budget passed at completion is persisted in the ledger,
so later monitors can still derive the correct phase without the original
conversation.
