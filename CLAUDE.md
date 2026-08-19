# autoresearch-automl

Multi-task autoresearch harness. Autonomous loops edit run-local candidate
`train.py` files and minimize the task's configured metric. The `driver/`
Python package is the sole runtime: a deterministic driver sequencing Claude
Agent SDK role sessions. Deterministic state, graph search, evaluation, and
tuning live in `tools/` and are shared. The `.claude/` Claude Code runtime
is retired (deleted in `816690f`).

## Start an experiment

```bash
uv run python -m driver run <task> <tag> --loop experiment \
  --model <model-id> [--max-evaluations N] [--timeout SECONDS]
```

`--loop hillclimb` is the comparison baseline and starts the same way.
`--loop baseline-tune` is the strong tuning baseline: the task-provided
baseline plus ONE HEBO MACE bout spanning the whole `--max-evaluations`
budget (requires `[seed].provided`; no ideation, no scheduler).
`--model` is required for a new run and ignored on resume — the run's
`run_metadata.json` wins. `--max-evaluations` and `--timeout` persist as
`max_evaluations` and `per_runtime_limit` in the run's `framework_cfg.json`
via `tools/init_run.py`; `--timeout` is a pass-through alias for
`init_run.py --per-runtime-limit` — the task-layer per-evaluation limit,
NOT a session watchdog. The same controls can also be set deterministically
beforehand:

```bash
python tools/init_run.py <task> <tag> --max-evaluations <n> --timeout <seconds>
```

## Authoritative Documents

Always read these in this order before doing experiment work:

1. `tasks/<task-name>/TASK.md` and `tasks/<task-name>/task.toml` — task brief
   and machine-readable contract for whichever task is in scope. The task
   contract is authoritative for environment, preparation, editable files,
   dependency permission, timeout, and metric details.
2. `driver/loops/experiment.py`, `driver/loops/hillclimb.py`, and
   `driver/loops/baseline_tune.py` — the
   canonical loop protocols, now deterministic Python. Role behavior lives in
   `driver/prompts/*.md`; each role's own prompt is authoritative for its
   contract. There is no separate protocol document — the repo-root
   `program.md` was removed in `0ba735f`, so ignore any remaining mention of
   it.
3. `driver/prompts/rules/ledger.md` — only when ledger schema detail is
   needed.

## Project Layout

```text
driver/                          the sole runtime: deterministic loops
                                 sequencing Claude Agent SDK role sessions
driver/loops/                    experiment.py, hillclimb.py, and
                                 baseline_tune.py protocols
driver/prompts/                  role prompts (+ rules/ledger.md)
driver/session.py                one role invocation = one SDK session +
                                 verify-repair loop
driver/roles.py                  role registry: prompt, capability set,
                                 receipt schema, postconditions
driver/receipts.py               receipt protocol + in-process MCP server
driver/events.py                 stdout progress lines + driver_events.jsonl
driver/metadata.py               run_metadata.json provenance + drift warnings
contracts/                       versioned shared contracts (dimension catalog)
docs/                            search-space, background, observability notes
tasks/<task-name>/               independent uv task projects
tests/                           pytest suite over tools/ and driver/;
                                 tests/fixtures.py holds the shared toy
                                 search space
tools/                           shared deterministic machinery
runs/<task-name>/<tag>/          local run artifacts (gitignored)
```

Each task is its own uv project. Do not treat `tasks/*` as a uv workspace.
`driver/` is the sole runtime.

## Reference Docs

Read on demand, not by default:

- `docs/search-space.md` — the formal semantic-space model the P2 helpers
  implement. Restates what the code enforces; adds no requirement.
- `docs/background-research.md` — the hierarchical search-space contract,
  evidence and scope semantics, retrieval fallback, validation commands.
- `docs/dimension-induction.md` — only when using the `llm_induced` dimension
  strategy.
- `docs/observability.md` — driver stdout progress lines and
  `driver_events.jsonl` event kinds; receipt↔session↔event correlation.

## Driver

The driver is deterministic Python; LLM judgment enters only through bounded
role sessions. The division of responsibility: **deterministic helpers own
deterministic decisions** (candidate promotion is `tune_tools.py
select-candidate`, tuner method choice is `select-method`, search-space state
transitions are `ledger.py apply-space-state`), **the driver owns sequencing**
(budget checks, escalation chains, crash recovery, keep/revert), and **LLM
sessions own generation** (ideas, code, diagnoses).

Role/receipt model:

- Each role in `driver/roles.py` pins a prompt file under `driver/prompts/`,
  a positive tool capability set (enforced fail-closed by a PreToolUse hook;
  `Agent`/`Task`/`Skill` are always denied), a receipt schema, and
  driver-side postconditions.
- One role invocation = one SDK session (`driver/session.py`). Roles never
  return free text: each session gets an in-process MCP server exposing
  `mcp__receipts__submit_receipt`; the accepted receipt is persisted under
  `<run_dir>/receipts/` linked by `invocation_id`. Failed postconditions
  trigger corrective follow-ups in the SAME session with concrete
  diagnostics, up to `role.corrective_attempts`, then `InvocationFailed`;
  escalation beyond that is the loop's job.
- Children return receipts, not payloads. Pass paths and compact ids; the
  durable run artifact is the payload. Do not collapse role boundaries to
  save time or budget.
- Session ids are persisted at session init, so a killed session can be
  resumed via `resume=<session_id>`; every recovery path also works without
  it.

Provenance: each new run writes `<run_dir>/run_metadata.json` (model,
SDK/CLI version and source, permission policy, prompt hashes). Resuming
with a drifted toolchain or edited prompts emits `metadata_mismatch`
warnings — record and warn, never refuse; a toolchain upgrade must not
orphan an in-flight run.

Orchestration rules that live in no single prompt:

- **Step 2 is decoupled from step 0+1** (design §15). Every candidate stops at
  step 0+1; `tuner-orchestrator` then runs once for the whole round and picks at
  most one candidate. A `none` selection is a valid no-op.
- **Deterministic helpers own deterministic decisions.** An agent proposes;
  the helper decides. Never hand-edit `ledger.json`.
- **Preflight failures are not objective evaluations.** They are no-score
  engineering checks, diagnosed via the `crash-diagnosis` role, and consume no
  budget slot. Objective calls reserve against the run cap in
  `evaluation_attempts.jsonl` immediately before `score_fn`.

## Running A Task

From the repo root:

```bash
uv --directory tasks/<task-name> sync
uv --directory tasks/<task-name> run python <entrypoint.py>
```

Experiments use candidate directories by default. Create a candidate first
with `tools/new_candidate.py` and run the copied entrypoint through the task
environment.

## Validation

```bash
uv run python -m pytest tests -q      # the suite; fast, no GPU, no network
python tools/validate_tasks.py        # task contracts
python tools/validate_background.py   # background round trip, shape
                                      # neutrality, retrieval, lifecycle
python tools/validate_got.py          # graph/ledger invariants
python tools/validate_search_backends.py
```

Keep checks minimal and implied by the touched contract. Do not add
required-wording or forbidden-wording checks over role prompts, rules, or
docs.

## Adding A Task

1. Create `tasks/<task-name>/` mirroring an existing task. Minimum set:
   `TASK.md`, `task.toml`, `pyproject.toml`, plus task code (typically
   `prepare.py` and `train.py`).
2. Add task dependencies to `pyproject.toml` and run
   `uv --directory tasks/<task-name> sync` to produce `uv.lock`.
3. Fill in `TASK.md` (human brief plus the `## Evaluation Contract` section —
   the prose semantics and hard rules for how a candidate trains, scores, and
   reports) and `task.toml` (machine config: the single `[evaluation].score_fn`
   — the one `config → score` function — plus metric, required patterns,
   optional candidate overrides, file constraints). The split is deliberate:
   the function name is config in `task.toml`; its semantics are prose in
   `TASK.md`. There is no separate official run — warm-start eval and Phase C
   tuning both call `score_fn(make_model, params) -> float`, and its return
   value is the candidate's score.
   **Scores are always lower-is-better.** The framework minimizes everywhere
   (keep/discard, percentile, every tuner) and tracks no direction flag — a
   higher-is-better metric must be negated or complemented inside the task's own
   `score_fn` (see `tabular-model-search`, which reports
   `neg_mean_test_accuracy`). A crash scores `+inf`, the worst.
4. Run `python tools/validate_tasks.py`.

## Adding A Role

1. Add `driver/prompts/<role-name>.md` and register the role in
   `driver/roles.py`: prompt file, positive tool capability set, receipt
   schema, postconditions.
2. Use lowercase hyphen-case for `<role-name>`.
3. Keep the prompt scoped to generation and judgment; put deterministic
   logic in `tools/` and sequencing in `driver/loops/`.

## Shell Command Conventions

Do not prepend `cd <project-root> &&` to shell commands. The session is already
at the project root, so it is redundant; it also defeats the project permission
allowlist (which matches by command prefix) and can trigger backslash-escape
safety prompts on absolute paths containing whitespace. Use relative paths from
the project root, or quoted absolute paths, without a leading `cd`.

## Boundaries

- `tasks/<task-name>/prepare.py` is the fixed evaluation surface — do not modify
  during normal experiments.
- `tasks/<task-name>/train.py` is the experiment surface, but experiments edit
  only the copy under `runs/<task-name>/<tag>/candidates/<run_id>/`. Most tasks
  omit a task-root `train.py`; `candidate-writer` generates each candidate's
  `train.py` under `runs/`. The task author does **not** need to provide
  contract-compliant code — `tunable-contract-extractor` extracts the tuner
  contract for every candidate, provided baselines included, at step 0+1.
- A candidate entrypoint declared in `[seed].provided` is copied into run `000`
  and evaluated first at the all-baselines point; seedless tasks bootstrap with
  normal `fresh` candidates.
- Never hand-edit `runs/**/ledger.json` — use `tools/ledger.py`.
- Do not commit anything under `runs/`. Run logs, `ledger.json`, and
  `loop_state.md` are local-only, disposable state.
- Add task dependencies only when `constraints.allow_dependencies = true` in the
  task's `task.toml`.
- The outer loop searches semantic candidates; step 0+1 / step 2 tune numeric
  parameters inside one candidate. Keep those two search levels distinct in
  schemas, metrics, and experiments.
- Keep context bounded. Experience refresh reads
  `got_graph.py render --incremental` with fixed Top/Bottom anchors; never inject
  the unbounded full ledger or global DAG. Retrieve a full record, source, or log
  only when a compact view identifies a specific missing field or bottleneck.
- Use content hashes only for cross-artifact bindings, cache/version keys, and
  content-addressed ids. Do not store a hash beside the complete inline JSON it
  hashes unless another artifact uses that digest as its identity; never add a
  self-hash solely to revalidate its own container. Validate inline fields and
  schemas directly instead.
