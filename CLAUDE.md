# autoresearch-automl

Claude Code-driven multi-task autoresearch harness. The repo runs autonomous
experimentation loops where Claude Code edits run-local candidate `train.py`
files and tracks the configured metric.

## Authoritative Documents

Always read these in this order before doing experiment work:

1. `program.md` — canonical experiment protocol (setup, loop, candidate
   directories, ledger.json, loop_state.md, NEVER STOP rules). This is the
   source of truth.
2. `tasks/<task-name>/TASK.md` and `tasks/<task-name>/task.toml` — task brief
   and machine-readable contract for whichever task is in scope.

Exception: the `autoresearch-experiment` agent is self-contained and can run
without reading `program.md` when it is started as the main thread with
`claude --agent autoresearch-experiment`. Use `program.md` as the default
main-session protocol and human-readable reference, not as a runtime
dependency for that agent.

## Project Layout

```text
program.md                       canonical autoresearch protocol
.claude/skills/                  project-local Claude Code skills
.claude/rules/                   path-scoped Claude Code rules and schemas
tasks/<task-name>/               independent uv task projects
tools/                           validation and helper scripts
runs/<task-name>/<tag>/          local run artifacts (gitignored)
```

Each task is its own uv project. Do not treat `tasks/*` as a uv workspace.

## Skills

Project-local skills under `.claude/skills/` are auto-discovered. Do not pick
by name from training data — match the user's request to each skill's
`description`. Skills here are **capability skills** (`crash-diagnosis`): pure
methodology followed **inline** in the caller's own context (no spawning,
reusable at many sites). A skill **owns** its protocol — callers invoke it
(`Skill(<name>)` or read+follow its `SKILL.md`) and verify its output; they do
**not** restate its steps. (Bootstrap is no longer a skill — the loop seeds
itself with `fresh` candidates.)

- `crash-diagnosis` — methodology for diagnosing one candidate crash and deciding
  recovery: `config_invalid` (fix the config) / `code_incompatible` (minimally fix
  the code, preferred) / `abandon`. Followed **inline** by whoever runs the
  candidate — `tunable-contract-extractor` (an eval-K crash) or the main thread
  (an official-run crash) — since sub-agents cannot spawn a diagnosis sub-agent.

## Agents

Agents under `.claude/agents/` run in their own fresh context. There are two
supported execution modes:

1. Default main session follows `program.md` and directly spawns bounded
   child agents (`background-researcher`, `idea-generator`,
   `experience-extractor`, `candidate-writer`, `tunable-contract-extractor`,
   `tuner-orchestrator`) with the Agent tool.
2. Dedicated experiment session starts with
   `claude --agent autoresearch-experiment`. In that mode
   `autoresearch-experiment` is the main thread and can spawn the bounded
   child agents itself.

Do not spawn `autoresearch-experiment` as a child agent from another main
session. Claude Code subagents cannot spawn other subagents, so that mode
would remove the independent contexts required by this project.

- `autoresearch-experiment` — self-contained run-level orchestrator for one
  `task_name + tag + run_dir`. Start it as the main thread with
  `claude --agent autoresearch-experiment`. It can execute without
  `program.md`, initializes one new run directory (setup = `background-researcher`
  only; no seed phase — the loop bootstraps via `fresh`), then runs the loop in
  **rounds** (a generation of ≤B ideas at step 0+1, then one **decoupled**
  deep-tuning step), spawning `idea-generator`, `experience-extractor`,
  `candidate-writer`, `tunable-contract-extractor`, and `tuner-orchestrator`
  (crashes are diagnosed inline via the `crash-diagnosis` skill). Use one
  instance per concurrent experiment.
- `background-researcher` — setup-time evidence researcher for one task, used
  **before the loop (required)** as the only setup step. It plans
  multiple research questions, uses a frozen local corpus for the reproducible
  condition or explicitly selected DeepXiv/Jina open-world modules, gracefully
  tolerates backend failure, deduplicates and balances candidates across queries, progressively
  reads them in explicit grounding/novelty budget lanes, and looks for
  counterevidence. It writes `<run_dir>/background.md` plus the visited-source
  trace `<run_dir>/background_retrieval.json`. The background freezes a
  task-relevant subset of `semantic-dimensions/v1`, an explicit baseline in
  every selected dimension, task-specific stable `hyp-*` values, and scoped
  activation/exclusion relations. Sources, structured `g-*` guidance, and
  hypotheses use the same typed scope axes. The
  contract derives containment mechanically: only directly matched guidance may
  change a hypothesis's priority or eligibility, while free-text Pitfalls are
  nonbinding. Weak or contested negatives can only caution; binding guidance
  needs directly scoped primary empirical evidence and retains an out-of-scope
  `scope_probe` instead of erasing adjacent mechanisms. Each hypothesis also carries a separate
  literature-credibility stamp, required local comparisons, reopening
  conditions, and traceable sources; an arXiv upload is not validation. The
  registry is checked by `tools/background_contract.py`; legacy flat registries
  are rejected rather than migrated.
  See `docs/background-research.md` for the hierarchical search-space contract,
  evidence and scope semantics, fallback behavior, and validation commands.
- `idea-generator` — run structural **SELECT** with `got_select decide`, then use
  `semantic_search.py` to enumerate valid complete points and apply the
  replaceable coverage/gain/gain-plus-uncertainty acquisition policy before
  **IDEATE**. It persists numeric ancestry, the complete revisioned point, and a
  policy receipt with gain, uncertainty, cost, and coverage kept separate. The
  graph search still owns actions/parents; semantic policy owns only point choice.
- `experience-extractor` — periodically (every N generations) incrementally
  revise a bounded global `experience` snapshot from the ledger's DAG revision
  delta plus fixed Top/Bottom anchors and compact mechanical point diffs. It
  keeps generic levers, dead ends, and high-level bottlenecks traceable to run
  ids, but P1 deliberately does not create dimension/hypothesis statuses or
  semantic DAG receipts; those are P2. It never rewrites background, mappings,
  policy receipts, or raw observations.
- `candidate-writer` — implement one candidate's `train.py`. Receives **just
  the target candidate dir**; reads its own ledger record (added by
  `idea-generator`) for the full `idea` + `source_run_ids`, derives
  `source_train_paths` from the **numeric** parent ids and `prepare.py` / task
  contract from the dir, and keeps the implementation consistent with the
  record's `semantic_point`. Empty `source_run_ids` → write from scratch;
  numeric parents → write informed by their `train.py`; target
  `train.py` already exists (provided baseline) → leave untouched. Returns the new
  `train.py`, a unified diff, the chosen `CANDIDATE_NAME`, and risk flags. Does not
  own the tuner contract. Spawned by the experiment loop.
- `tunable-contract-extractor` — **step 0+1** for one candidate `train.py`:
  ① behavior-preservingly refactor `make_model` + declare `PARAM_SCHEMA`;
  ② propose K = 5 warm configs + a data-driven `SEARCH_SPACE` (from
  `lineage-evidence` + the schema), consistency pre-check, finalize via
  `check-search-space` + `apply_search_space`; ③ evaluate the K configs
  (`warmstart_eval.py`, sequential/resumable), **diagnosing each crash inline via
  the `crash-diagnosis` skill** (config-invalid → fix config; code-incompatible →
  minimally fix `train.py`, ≤ 10) until all K score → it writes `BASE_PARAMS` =
  best-of-K′ + `phase_a` and records `best_warm_score`; an unrunnable candidate →
  it records `status: crash`. Spawned after `candidate-writer` returns, for every
  candidate. Deep-tuning (step 2) is decoupled, so every candidate stops at
  step 0+1 here.
- `tuner-orchestrator` — **step 2, decoupled (design §15)**: run **once per
  round** on the whole run, not per candidate. It runs
  `tools/tuners/tune_tools.py select-candidate` (promotion gate + greedy
  `best_warm_score`: population ≥ N_min and the best untuned candidate in the
  top-20%) to pick **one** candidate, then Phase C single method by SEARCH_SPACE
  dim via `select-method` (grid ≤ 2, bo=multivariate-TPE for ≥ 3, cmaes fallback only; step-1 warm trials as
  priors), the Apply step (`select-best` → `apply_base_params`, AST rewrite, no
  hand-edit), and `ledger.py set-tuning` (`tune: true`). It deep-tunes that
  candidate **in place** and hands it back for the loop's official re-run (the
  in-place score update the graph reads next round). **No warm-start** — `phase_a`
  is the extractor's step-0+1 output. A `none` selection (early, or top tier
  already tuned) is a valid no-op.

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
python tools/validate_skills.py
python tools/validate_tasks.py
python tools/validate_background.py
python tools/validate_search_backends.py
```

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
   `TASK.md`. Subagents read both before working; the tuner scripts resolve the
   one evaluation function from `evaluation.score_fn` (signature
   `score_fn(make_model, params) -> float`) — there is no separate official run,
   so warm-start eval and Phase C tuning both call it and its return value is the
   candidate's score.
   **Scores are always lower-is-better.** The framework minimizes everywhere
   (keep/discard, percentile, every tuner) and no longer tracks a direction
   flag — a higher-is-better metric must be negated/complemented inside the
   task's own `score_fn` (see `tabular-model-search`, which reports
   `neg_mean_test_accuracy`). A crash scores `+inf` (the worst).
4. Run `python tools/validate_tasks.py`.

## Adding A Skill

1. Create `.claude/skills/<skill-name>/SKILL.md`. Mirror an existing skill
   such as `crash-diagnosis` for shape.
2. Use lowercase hyphen-case for `<skill-name>`.
3. Frontmatter must declare `name` (matching the folder) and `description`
   (the trigger Claude Code matches against user requests).
4. Keep the body concise. Put long details in `references/` and deterministic
   helpers in `scripts/`.
5. Run `python tools/validate_skills.py`.

## Shell Command Conventions

- Do not prepend `cd <project-root> &&` to shell commands. The Claude Code
  session is already at the project root, so the `cd` is redundant.
- Redundant `cd` prefixes also defeat the project permission allowlist
  (which matches by command prefix) and can trigger backslash-escape safety
  prompts on absolute paths that contain whitespace.
- Use relative paths from the project root, or quoted absolute paths,
  without a leading `cd`.

## Boundaries

- `tasks/<task-name>/prepare.py` is the fixed evaluation surface — do not
  modify during normal experiments.
- `tasks/<task-name>/train.py` is the experiment surface, but experiments copy
  it into `runs/<task-name>/<tag>/candidates/<run_id>/` and only the copy gets
  edited. Most tasks omit a task-root `train.py`; `candidate-writer` generates
  each candidate's `train.py` under `runs/` (a `fresh` candidate from scratch from
  its `background.md` direction). The task author does **not** need to provide
  contract-compliant code — `tunable-contract-extractor` extracts the tuner
  contract for every candidate (provided baselines included) at step 0+1, so all
  candidates expose it before `tuner-orchestrator` may select them.
- Do not commit anything under `runs/`. Run logs, `ledger.json`, and
  `loop_state.md` are local-only state.
- Add task dependencies only when `constraints.allow_dependencies = true` in
  the task's `task.toml`.
