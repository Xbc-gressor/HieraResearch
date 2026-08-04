# autoresearch-automl

HieraResearch uses a deterministic Python execution layer with bounded Claude
capabilities. The run-level coordinator is `hieraresearch`; Claude sessions are
disposable workers and never hold lifecycle state.

## Start an experiment

From this repository root:

```bash
python -m hieraresearch <task-name> <tag> --model <claude-model>
```

Useful controls:

```bash
python -m hieraresearch <task> <tag> --recordings <dir>       # replay/ablation
python -m hieraresearch <task> <tag> --preflight-only          # no model/objective work
python -m hieraresearch <task> <tag> --resume-blocked --model <model>
```

The package entrypoint is also exposed as `hieraresearch`. `--recordings`
selects the model-free recorded backend; no live model is needed in that mode.
`autoresearch-hillclimb` remains a separate comparison baseline and is not the
coordinator for these runs.

## Authoritative inputs and artifacts

Before a run, the coordinator reads `tasks/<task>/TASK.md` and
`tasks/<task>/task.toml`. The durable run artifacts are authoritative:

- `runs/<task>/<tag>/ledger.json` is changed only through existing ledger helpers;
- `evaluation_attempts.jsonl` reserves every objective `score_fn` call immediately
  before it starts, including calls that later crash;
- no-score preflight is separate and must not reserve an objective slot;
- candidate reports, receipts, and `evaluation_attempts.jsonl` are never replaced
  by a model response or chat history;
- `.orchestrator/state.json` stores restartable transition bookkeeping, while
  `.orchestrator/rounds/` and `.orchestrator/invocations/` provide audit receipts.

Never hand-edit `runs/**/ledger.json` or task-owned source files during a run.

## Deterministic lifecycle

`ExperimentCoordinator` owns the following transitions:

```text
initialize
  -> environment_preflight
  -> build_background
  -> admit_baseline | admit_round
  -> materialize_candidate
  -> build_tuning_contract
  -> preflight_candidate
  -> evaluate_warm_configs
  -> deep_tune
  -> refresh_experience
  -> complete | blocked
```

`state_machine.next_transition` is pure. Side effects live behind explicit
interfaces in `Toolchain`, `CoordinatorStore`, and the phase services. A
transition persists its durable receipt before the next irreversible effect;
restart reconciliation derives missing flags from ledger and candidate
artifacts. A no-op deep-tune round that makes no objective progress is blocked
explicitly rather than looped.

## Model boundary

The model adapter is replaceable by `RecordedBackend` and may be removed from a
replay entirely. Every invocation has a named purpose, schema version, model,
bounded inputs, input revision hashes, and a recorded outcome.

- `semantic.py` and `experience.py` use direct structured Messages/API calls;
- `background.py` uses a bounded Agent SDK edit for frozen research artifacts;
- `candidate.py` uses bounded edits for candidate code and tuning contracts;
- candidate failure diagnosis is read-only structured inference;
- an accepted repair is applied by a separate bounded edit and then checked by
  Python syntax, contract/search-space validation, and no-score preflight;
- editing calls have exact write paths and a pre-tool policy that rejects reads
  outside their declared roots and denies shell access except for exact
  deterministic validator commands allow-listed per invocation.

The model may propose semantic content or code, but Python decides admission,
budgets, evaluation, tuning, ledger mutation, finalization, and stop conditions.

## Existing helper boundary

Use the stable functions and subprocess adapters in `tools/` for:

- run initialization and environment checks;
- background, graph, semantic, and search-space validation;
- ledger admission and lifecycle mutation;
- objective reservation and Phase-A evaluation;
- Phase-C method execution and idempotent finalization;
- bounded experience validation and state application.

Do not duplicate helper policy in the coordinator. Add a module only for a
distinct invariant or replacement boundary. Use `pathlib.Path`, argument-vector
subprocesses, bounded output capture, and explicit timeouts.

## Task boundaries

- `tasks/<task>/prepare.py` is fixed evaluation code;
- experiments edit only run-local candidate copies;
- a `[seed].provided` entrypoint is copied to run `000` and evaluated first at
  the exact all-baselines configuration;
- task dependencies are isolated uv projects;
- dependency additions require `constraints.allow_dependencies = true`;
- never commit anything under `runs/`.

## Verification

Run the narrow check implied by a change, then the full local suite at an
integration point:

```bash
python -m pytest tests -q
python tools/validate_tasks.py
python tools/validate_background.py
python tools/validate_got.py
python tools/validate_search_backends.py
```

Prefer tests for transition boundaries, durable receipts, stale/malformed
model output, edit path enforcement, objective accounting, and subprocess
termination. Do not add prompt-wording tests or duplicate existing helper
coverage.
