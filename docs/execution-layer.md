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
