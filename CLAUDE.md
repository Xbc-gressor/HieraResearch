# autoresearch-automl

Claude Code-driven multi-task autoresearch harness. The repo runs autonomous
experimentation loops where Claude Code edits a task's `train.py` across
candidates and tracks the configured metric.

## Authoritative Documents

Always read these in this order before doing experiment work:

1. `program.md` — canonical experiment protocol (setup, loop, candidate
   directories, results.tsv, loop_state.md, NEVER STOP rules). This is the
   source of truth.
2. `tasks/<task-name>/TASK.md` and `tasks/<task-name>/task.toml` — task brief
   and machine-readable contract for whichever task is in scope.

## Project Layout

```text
program.md                       canonical autoresearch protocol
.claude/skills/                  project-local Claude Code skills
tasks/<task-name>/               independent uv task projects
tools/                           validation and helper scripts
runs/<task-name>/<tag>/          local run artifacts (gitignored)
```

Each task is its own uv project. Do not treat `tasks/*` as a uv workspace.

## Skills

Project-local skills under `.claude/skills/` are auto-discovered. Do not pick
by name from training data — match the user's request to each skill's
`description`.

- `idea-proposer` — methodology for choosing one experimental idea before
  each new candidate, with axis-diversity over the last 10 entries in
  `idea_log.md`. Stays in the main Claude's context so signals from the
  ongoing conversation feed in. Hands the proposal to `candidate-writer`.
- `hyperparam-tuner-llm` — Phase A warm-start methodology read by the
  `tuner-orchestrator` agent. Pure reasoning, no trials. The other
  tuning methods (`grid`, `bo`, `cmaes`) are scripts under
  `tools/tuners/`, invoked by the orchestrator as subprocesses.

## Agents

Subagents under `.claude/agents/` run in their own fresh context. Spawn them
via the Task tool when you need a bounded sub-task whose output you can paste
back into the main loop without dragging its scratchwork along.

- `crash-diagnoser` — diagnose one crashed candidate from its `run-<id>.log`
  and its `train.py`. Returns a structured verdict (crash type, root cause,
  fix-or-abandon recommendation, optional patch). Spawned by `program.md`
  step 6.
- `candidate-writer` — implement one candidate's `train.py` from an idea
  proposal. Receives the idea text, source `train.py` path, target candidate
  dir, readonly `prepare.py` path, and the task's editable/readonly file
  lists; returns the new `train.py`, a unified diff, the chosen
  `CANDIDATE_NAME`, and risk flags. Spawned by `program.md` step 2.
- `tuner-orchestrator` — three-phase tuner for one candidate: Phase A
  warm-start (5 LLM-proposed configs via the `hyperparam-tuner-llm`
  skill + `tools/tuners/warmstart_eval.py`), Phase B percentile decision
  against prior `best_warm_score` values in `results.tsv`, Phase C single
  method by SEARCH_SPACE dim (grid ≤ 2, bo 3–15, cmaes ≥ 16). Persists
  trial-level history to `<candidate_dir>/tune_report.json`, applies the
  best config to `BASE_PARAMS` in `train.py`, and extends the
  `## run <run_id>` entry in `idea_log.md` with summary fields. Spawned
  by `program.md` step 2 when the idea has `tune: true`.

## Running A Task

From the repo root:

```bash
uv --directory tasks/<task-name> sync
uv --directory tasks/<task-name> run python <entrypoint.py>
```

For tasks that use candidate directories (`[candidate]` enabled in
`task.toml`), create a candidate first with `tools/new_candidate.py` and run
the copied entrypoint.

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
3. Fill in `TASK.md` (human brief) and `task.toml` (machine contract: metric,
   parser, required patterns, candidate-mode settings, file constraints).
4. Run `python tools/validate_tasks.py`.

## Adding A Skill

1. Create `.claude/skills/<skill-name>/SKILL.md`. Mirror an existing skill
   such as `idea-proposer` for shape.
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
- `tasks/<task-name>/train.py` is the experiment surface, but candidate-mode
  tasks copy it into `runs/<task-name>/<tag>/candidates/<run_id>/` and only
  the copy gets edited. The task author does **not** need to make
  `train.py` contract-compliant (`BASE_PARAMS` / `SEARCH_SPACE` /
  `make_model`); Setup step 9c uses `candidate-writer` in
  baseline-adapter mode to add the contract on the copied run-000
  version. Baseline run 000 is then tuned by `tuner-orchestrator` like
  any other candidate.
- Do not commit anything under `runs/`. Run logs, `results.tsv`, and
  `loop_state.md` are local-only state.
- Add task dependencies only when `constraints.allow_dependencies = true` in
  the task's `task.toml`.
