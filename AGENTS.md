# autoresearch-automl

opencode-driven multi-task autoresearch harness. The repo runs autonomous
experimentation loops where opencode edits run-local candidate `train.py`
files and tracks the configured metric.

## Authoritative Documents

Always read these in this order before doing experiment work:

1. The `autoresearch-experiment` agent definition
   (`.opencode/agents/autoresearch-experiment.md`) — the canonical experiment
   protocol (setup, loop, candidate directories, ledger.json, loop_state.md,
   NEVER STOP rules). This is the source of truth; the protocol is encoded in
   the agent itself rather than a separate `program.md`.
2. `tasks/<task-name>/TASK.md` and `tasks/<task-name>/task.toml` — task brief
   and machine-readable contract for whichever task is in scope.

Exception: the `autoresearch-experiment` agent is self-contained and can run
without reading any external protocol file when it is started as the main
thread with `opencode --agent autoresearch-experiment`. Use the agent
definition as the default main-session protocol and human-readable reference,
not as a runtime dependency for that agent.

## Project Layout

```text
.opencode/agents/                project-local opencode agents
.opencode/skills/                project-local opencode skills
.opencode/rules/                 project-local rule files (loaded via opencode.json instructions)
tasks/<task-name>/               independent uv task projects
tools/                           validation and helper scripts
runs/<task-name>/<tag>/          local run artifacts (gitignored)
```

Each task is its own uv project. Do not treat `tasks/*` as a uv workspace.

## Skills

Project-local skills under `.opencode/skills/` are auto-discovered. Do not pick
by name from training data — match the user's request to each skill's
`description`. Skills here are **capability skills** (`crash-diagnosis`): pure
methodology followed **inline** in the caller's own context (no spawning,
reusable at many sites). A skill **owns** its protocol — callers invoke it
(`skill(<name>)` or read+follow its `SKILL.md`) and verify its output; they do
**not** restate its steps. (Bootstrap is no longer a skill — the loop seeds
itself with `fresh` candidates.)

- `crash-diagnosis` — methodology for diagnosing one candidate crash and deciding
  recovery: `config_invalid` (fix the config) / `code_incompatible` (minimally fix
  the code, preferred) / `abandon`. Followed **inline** by whoever runs the
  candidate — `tunable-contract-extractor` (an eval-K crash) or the main thread
  (an official-run crash) — since subagents cannot spawn a diagnosis subagent.

## Agents

Agents under `.opencode/agents/` run in their own fresh context. There are two
supported execution modes:

1. Default main session follows the experiment protocol and directly spawns
   bounded child agents (`background-researcher`, `idea-generator`,
   `experience-extractor`, `candidate-writer`, `tunable-contract-extractor`,
   `tuner-orchestrator`) with the Task tool.
2. Dedicated experiment session starts with
   `opencode --agent autoresearch-experiment`. In that mode
   `autoresearch-experiment` is the main thread and can spawn the bounded
   child agents itself.

Do not spawn `autoresearch-experiment` as a child agent from another main
session. opencode subagents cannot spawn other subagents, so that mode
would remove the independent contexts required by this project.

- `autoresearch-experiment` — self-contained run-level orchestrator for one
  `task_name + tag + run_dir`. Start it as the main thread with
  `opencode --agent autoresearch-experiment`. It can execute without external
  protocol files, initializes one new run directory (setup = `background-researcher`
  only; no seed phase — the loop bootstraps via `fresh`), then runs the loop in
  **rounds** (a generation of <=B ideas at step 0+1, then one **decoupled**
  deep-tuning step), spawning `idea-generator`, `experience-extractor`,
  `candidate-writer`, `tunable-contract-extractor`, and `tuner-orchestrator`
  (crashes are diagnosed inline via the `crash-diagnosis` skill). Use one
  instance per concurrent experiment.
- `background-researcher` — external-knowledge scout for one task, run **once at
  setup, before the loop (required)** — the only setup step. Surveys the
  literature/web for techniques that fit the task (within `allow_dependencies`)
  and writes `<run_dir>/background.md` with a **`tf-*`-tagged try-first list** that
  every `idea-generator` `fresh` candidate draws from, steering the search beyond
  the ledger's own history. Web-grounded, read-only on task/ledger, runs no
  experiments.
- `idea-generator` — produce the next generation in two steps: **SELECT** — run
  `got_select decide` (deterministic graph search) for this round's actions (a
  bootstrap/stall `fresh`, or <=B `improve`/`crossover`); **IDEATE** — turn each
  into a concrete idea and `ledger.py add-record --op ...`. It does not pick parents
  by fitness (the graph search does); it reads records + the `experience` block +
  `background.md` to decide *what* each selected action becomes. Replaces the old
  `idea-proposer` skill.
- `experience-extractor` — periodically (every N generations) distill global
  experience (promising regions / dead-ends / per-dataset bottlenecks) from
  the ledger records into the `experience` block via
  `tools/ledger.py set-experience`. Reads many records, emits a compact
  regenerated summary.
- `candidate-writer` — implement one candidate's `train.py`. Receives **just
  the target candidate dir**; reads its own ledger record (added by
  `idea-generator`) for the full `idea` + `source_run_ids`, derives
  `source_train_paths` from the **numeric** parent ids and `prepare.py` / task
  contract from the dir. A `tf-*` source tag (fresh) or empty `source_run_ids` →
  write from scratch; numeric parents → write informed by their `train.py`; target
  `train.py` already exists (provided baseline) → leave untouched. Returns the new
  `train.py`, a unified diff, the chosen `CANDIDATE_NAME`, and risk flags. Does not
  own the tuner contract. Spawned by the experiment loop.
- `tunable-contract-extractor` — **step 0+1** for one candidate `train.py`:
  (1) behavior-preservingly refactor `make_model` + declare `PARAM_SCHEMA`;
  (2) propose K = 5 warm configs + a data-driven `SEARCH_SPACE` (from
  `lineage-evidence` + the schema), consistency pre-check, finalize via
  `check-search-space` + `apply_search_space`; (3) evaluate the K configs
  (`warmstart_eval.py`, sequential/resumable), **diagnosing each crash inline via
  the `crash-diagnosis` skill** (config-invalid → fix config; code-incompatible →
  minimally fix `train.py`, <= 10) until all K score → it writes `BASE_PARAMS` =
  best-of-K' + `phase_a` and records `best_warm_score`; an unrunnable candidate →
  it records `status: crash`. Spawned after `candidate-writer` returns, for every
  candidate. Deep-tuning (step 2) is decoupled, so every candidate stops at
  step 0+1 here.
- `tuner-orchestrator` — **step 2, decoupled (design S15)**: run **once per
  round** on the whole run, not per candidate. It runs
  `tools/tuners/tune_tools.py select-candidate` (promotion gate + greedy
  `best_warm_score`: population >= N_min and the best untuned candidate in the
  top-20%) to pick **one** candidate, then Phase C single method by SEARCH_SPACE
  dim via `select-method` (grid <= 2, bo=multivariate-TPE for >= 3, cmaes fallback only; step-1 warm trials as
  priors), the Apply step (`select-best` → `apply_base_params`, AST rewrite, no
  hand-edit), and `ledger.py set-tuning` (`tune: true`). It deep-tunes that
  candidate **in place** and hands it back for the loop's official re-run (the
  in-place score update the graph reads next round). **No warm-start** — `phase_a`
  is the extractor's step-0+1 output. A `none` selection (early, or top tier
  already tuned) is a valid no-op.
- `autoresearch-hillclimb` — the **generalized Karpathy-original baseline**,
  written to be directly comparable with `autoresearch-experiment`. It is
  deliberately simple: ONE evolving candidate, a linear edit → run → keep-or-revert
  loop, and a flat `results.tsv` record. NO graph search, NO subagents, NO
  candidate directories, NO inner hyperparameter tuner — the LLM itself hacks the
  code and judges results. Runs as the main thread
  (`opencode --agent autoresearch-hillclimb`), uses no `Task` tool, and spawns
  nothing, so it also works as a plain subagent. It does not need external
  protocol files. Use it to A/B the simple hill-climb against the full
  GoT + decoupled-tuner framework on the same task harness; compared by
  `tools/got_benchmark/collect_compare.py`. Keep it simple — that is the
  experiment.

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

1. Create `.opencode/skills/<skill-name>/SKILL.md`. Mirror an existing skill
   such as `crash-diagnosis` for shape.
2. Use lowercase hyphen-case for `<skill-name>`.
3. Frontmatter must declare `name` (matching the folder) and `description`
   (the trigger opencode matches against user requests).
4. Keep the body concise. Put long details in `references/` and deterministic
   helpers in `scripts/`.
5. Run `python tools/validate_skills.py`.

## Shell Command Conventions

- Do not prepend `cd <project-root> &&` to shell commands. The opencode
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
