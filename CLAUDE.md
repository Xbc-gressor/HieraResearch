# autoresearch-automl

Multi-task autoresearch harness. Autonomous loops edit run-local candidate
`train.py` files and minimize the task's configured metric. Claude Code
(`.claude/`) is the supported interactive runtime; deterministic state, graph
search, evaluation, and tuning live in `tools/` and are shared.

The loop's current bar is beating `autoresearch-hillclimb` — the deliberately
simple edit→run→keep/revert baseline — at matched evaluation budget. It does not
yet. Until it does, prefer diagnosing current run artifacts and simplifying the
loop over extending it.

## Start an experiment

```bash
claude --agent autoresearch-experiment     # from this repo root
```

Then provide `task_name`, `tag`, and optionally `max_evaluations` and `timeout`
(the hard per-evaluation limit in seconds). The same controls can be set
deterministically beforehand:

```bash
python tools/init_run.py <task> <tag> --max-evaluations <n> --timeout <seconds>
```

They persist as `max_evaluations` and `per_runtime_limit` in the run's
`framework_cfg.json`; explicit initialization values override the copied
template. `autoresearch-hillclimb` is the comparison baseline and starts the
same way with `--agent autoresearch-hillclimb`.

## Authoritative Documents

Always read these in this order before doing experiment work:

1. `tasks/<task-name>/TASK.md` and `tasks/<task-name>/task.toml` — task brief
   and machine-readable contract for whichever task is in scope. The task
   contract is authoritative for environment, preparation, editable files,
   dependency permission, timeout, and metric details.
2. `.claude/agents/autoresearch-experiment.md` — the canonical experiment
   protocol (setup, loop, candidate directories, ledger.json, loop_state.md,
   NEVER STOP rules). The agent prompt carries the protocol; there is no
   separate protocol document — the repo-root `program.md` was removed in
   `0ba735f`, so ignore any remaining mention of it.
3. `.claude/rules/ledger.md` — only when ledger schema detail is needed.

## Project Layout

```text
.claude/agents/                  project-local agents; the experiment agent
                                 carries the canonical run protocol
.claude/skills/                  project-local Claude Code skills
.claude/rules/                   path-scoped Claude Code rules and schemas
.claude/settings.json            Agent delegation guard + status lines
contracts/                       versioned shared contracts (dimension catalog)
docs/                            search-space, background, observability notes
tasks/<task-name>/               independent uv task projects
tests/                           pytest suite over tools/; tests/fixtures.py
                                 holds the shared toy search space
tools/                           shared deterministic machinery
runs/<task-name>/<tag>/          local run artifacts (gitignored)
```

Each task is its own uv project. Do not treat `tasks/*` as a uv workspace.
`.opencode/` and `.kimi/` are deprecated runtime mirrors: unmaintained, free to
drift, and not to be read as contracts. `.claude/` is canonical — do not sync
changes into them.

## Reference Docs

Read on demand, not by default:

- `docs/search-space.md` — the formal semantic-space model the P2 helpers
  implement. Restates what the code enforces; adds no requirement.
- `docs/background-research.md` — the hierarchical search-space contract,
  evidence and scope semantics, retrieval fallback, validation commands.
- `docs/dimension-induction.md` — only when using the `llm_induced` dimension
  strategy.
- `docs/observability.md` — `tools/harness_watch.py` token and drift attribution.

## Skills

Project-local skills under `.claude/skills/` are auto-discovered. Match the
user's request to each skill's `description`; do not pick by a name recalled
from training data. Skills here are **capability skills**: pure methodology
followed **inline** in the caller's own context, no spawning, reusable at many
sites. A skill **owns** its protocol — callers invoke it and verify its output;
they do not restate its steps.

- `crash-diagnosis` — diagnose one candidate preflight or objective crash and
  decide recovery: `config_invalid` (fix the config) / `code_incompatible`
  (minimally fix the code, preferred) / `abandon`.

## Agents

Agents under `.claude/agents/` run in their own fresh context. Claude Code
surfaces each one's `description` automatically, and **each agent's own prompt
is authoritative for its contract** — read the prompt, not a summary, before
changing what an agent does.

| agent | role | cadence |
|---|---|---|
| `autoresearch-experiment` | run-level orchestrator for one `task_name + tag + run_dir`; its prompt carries the full run protocol | main thread, one per concurrent run |
| `background-researcher` | freezes `background.md` + `background_retrieval.json`: the run's hierarchical semantic search space | once, before the loop (required) |
| `idea-generator` | graph `SELECT` via `got_select`, then semantic point choice and record/receipt persistence | per round |
| `candidate-writer` | implements one candidate's `train.py` from its own ledger record | per candidate |
| `tunable-contract-extractor` | step 0+1: `PARAM_SCHEMA` refactor, warm configs, `SEARCH_SPACE`, screening evaluation | per candidate |
| `tuner-orchestrator` | step 2: promotion gate, then one progressive-tuning bout for at most one selected candidate in place | once per round |
| `experience-extractor` | regenerates the bounded belief snapshot and requests state transitions | per completed non-empty round |

Orchestration rules that live in no single prompt:

- **Two execution modes.** A default main session follows the protocol in
  `.claude/agents/autoresearch-experiment.md` and spawns the bounded children
  itself; a dedicated session starts with
  `claude --agent autoresearch-experiment`, making that agent the main thread.
- **Never spawn `autoresearch-experiment` as a child agent.** Claude Code
  subagents cannot spawn subagents, so that mode removes the independent
  contexts the design depends on. `.claude/settings.json` enforces this with an
  `Agent` PreToolUse guard (`tools/harness_guard.py`).
- **Step 2 is decoupled from step 0+1** (design §15). Every candidate stops at
  step 0+1; `tuner-orchestrator` then runs once for the whole round and picks at
  most one candidate. A `none` selection is a valid no-op.
- **Deterministic helpers own deterministic decisions.** Candidate promotion is
  `tune_tools.py select-candidate`, tuner method choice is `select-method`,
  search-space state transitions are `ledger.py apply-space-state`. An agent
  proposes; the helper decides. Never hand-edit `ledger.json`.
- **Preflight failures are not objective evaluations.** They are no-score
  engineering checks, diagnosed inline via `crash-diagnosis`, and consume no
  budget slot. Objective calls reserve against the run cap in
  `evaluation_attempts.jsonl` immediately before `score_fn`.
- **Children return receipts, not payloads.** Pass paths and compact ids; the
  durable run artifact is the payload. Do not collapse role boundaries to save
  time or budget.

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
python -m pytest tests -q             # the suite; fast, no GPU, no network
python tools/validate_tasks.py        # task contracts
python tools/validate_background.py   # background round trip, shape
                                      # neutrality, retrieval, lifecycle
python tools/validate_got.py          # graph/ledger invariants
python tools/validate_search_backends.py
```

Keep checks minimal and implied by the touched contract. Do not add
required-wording or forbidden-wording checks over agent prompts, rules, or
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
   — the one `config → score` function — plus metric, parser, required patterns,
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

## Adding A Skill

1. Create `.claude/skills/<skill-name>/SKILL.md`. Mirror an existing skill
   such as `crash-diagnosis` for shape.
2. Use lowercase hyphen-case for `<skill-name>`.
3. Frontmatter must declare `name` (matching the folder) and `description`
   (the trigger Claude Code matches against user requests).
4. Keep the body concise. Put long details in `references/` and deterministic
   helpers in `scripts/`.

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
