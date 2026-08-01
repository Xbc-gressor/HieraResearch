# autoresearch-automl

The supported run-level execution path is the deterministic Python coordinator
in `src/hieraresearch/`. Claude supplies bounded semantic judgments, research,
and code edits through `ModelGateway`; it does not orchestrate the experiment.

## Running a task

From the repository root:

```bash
python -m hieraresearch <task-name> <tag> --model <claude-model>
```

Use `--recordings <directory>` for model-free replay or ablation and
`--preflight-only` for startup validation without model calls or objective
evaluation. Use `--resume-blocked` only when intentionally reopening a blocked
run. The same command is exposed as the `hieraresearch` console script.

`autoresearch-hillclimb` remains a separate comparison baseline. It is not an
alternative implementation of the coordinator.

## Authoritative documents

Read the task contract before changing a run:

1. `tasks/<task-name>/TASK.md`
2. `tasks/<task-name>/task.toml`
3. `docs/execution-layer.md`
4. the relevant helper contract in `tools/` or `docs/`

`TASK.md` defines the evaluation surface and behavioral rules. `task.toml`
defines the machine-readable environment, timeout, metric, file, and function
contracts. The execution-layer document describes the Python/Claude boundary.

## Project layout

```text
src/hieraresearch/               deterministic coordinator and model boundary
tools/                           existing ledger, graph, evaluation, and tuning helpers
tasks/<task-name>/               isolated uv task projects
docs/execution-layer.md          durable state and transition contract
runs/<task>/<tag>/               gitignored experiment artifacts
```

Run artifacts, not model sessions or conversation context, are the source of
truth. `ledger.json` is mutated only through existing deterministic helpers;
`evaluation_attempts.jsonl` is the authoritative objective admission log.

## Coordinator responsibilities

`ExperimentCoordinator` derives the next transition from durable state and a
compact ledger brief. `state_machine.next_transition` is pure; tool/process
calls and artifact writes are explicit side effects. The flow is:

```text
environment preflight
  -> frozen background
  -> baseline/round admission
  -> candidate implementation
  -> tuning contract
  -> no-score preflight
  -> Phase-A warm evaluation
  -> one deterministic Phase-C selection/tuning step
  -> idempotent finalization
  -> experience refresh or stop
```

The coordinator persists state after admission, candidate stages, deep-tune
selection, round completion, and blocking. On restart it validates ledger and
candidate artifacts and reconstructs incomplete stage flags. It blocks on
corrupt authoritative state, ambiguous objective accounting, unsupported model
output, and no-progress cycles instead of inventing recovery decisions.

## Model boundary

- `semantic.py` and `experience.py` use direct structured Messages/API calls;
- `background.py` uses a bounded Agent SDK edit for frozen research artifacts;
- `candidate.py` uses bounded edits for candidate code and tuning contracts;
- debugging inference is read-only and accepts only `config_invalid`,
  `code_incompatible`, or `abandon`;
- accepted repairs are followed by deterministic validation and no-score
  preflight before the objective worker is retried;
- `RecordedBackend` supports replay without a live model.

Every call records its purpose, schema version, model, input revision, bounded
request, response, and outcome. Agent edits use exact write paths, bounded read
roots, explicit turn limits, and no shell tool. Python validates all model
output before mutation and never lets a model select lifecycle phases, grant
budget, evaluate scores, or finalize tuning.

## Development and verification

Use existing helpers rather than duplicating ledger, budget, graph, evaluation,
or finalization policy. Keep new modules at real invariant or replacement
boundaries. Candidate and task source edits belong under `runs/`; task
`prepare.py` is fixed.

```bash
python -m pytest tests -q
python tools/validate_tasks.py
python tools/validate_background.py
python tools/validate_got.py
python tools/validate_search_backends.py
```

Focus regression tests on transition/receipt boundaries, stale or malformed
model output, edit path enforcement, objective accounting, and process
termination. Do not test prompt wording or duplicate helper internals.
