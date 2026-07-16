# autoresearch-automl (kimi-cli runtime)

Multi-task autoresearch harness. The repo runs autonomous experimentation loops
where an agent edits run-local candidate `train.py` files and tracks the
configured metric. The harness has two side-by-side runtimes:

- `.claude/` — the original Claude Code runtime (see `CLAUDE.md`).
- `.kimi/` — the kimi-cli port documented here. The deterministic machinery in
  `tools/` (~11k lines: ledger, background contract, graph search, tuners) is
  shared and runtime-agnostic.

This file is the kimi-side conventions doc; kimi-cli auto-loads it as
`${KIMI_AGENTS_MD}`.

## Authoritative Documents

Always read these before doing experiment work:

1. `tasks/<task-name>/TASK.md` and `tasks/<task-name>/task.toml` — task brief
   and machine-readable contract for whichever task is in scope.
2. `.kimi/rules/ledger.md` — the ledger schema and the helper-only mutation
   contract (never hand-edit `ledger.json`).

The `autoresearch-experiment` agent is self-contained when started as the main
thread (see below) and is the canonical experiment protocol.

## Project Layout

```text
.kimi/agents/                  kimi-cli agent files (*.yaml + system-prompt *.md)
.kimi/rules/                   referenced schemas (ledger.md)
.kimi/skills/                  project-local kimi skills (crash-diagnosis)
.kimi/kimi-hooks.toml          hook registrations merged at launch
tools/kimi_run.py              launcher: merges hooks into the user config, execs kimi
tools/                         validation and helper scripts (shared, runtime-agnostic)
tasks/<task-name>/             independent uv task projects
runs/<task-name>/<tag>/        local run artifacts (gitignored)
```

Each task is its own uv project. Do not treat `tasks/*` as a uv workspace.

## Skills

Project-local skills under `.kimi/skills/` are auto-discovered by kimi-cli.
Skills here are **capability skills** (`crash-diagnosis`): pure methodology
followed **inline** in the caller's own context (no spawning, reusable at many
sites). kimi-cli has no Skill tool — the caller reads the `SKILL.md` with
ReadFile and follows it.

- `crash-diagnosis` — methodology for diagnosing one candidate crash and deciding
  recovery: `config_invalid` (fix the config) / `code_incompatible` (minimally fix
  the code, preferred) / `abandon`. Followed **inline** by whoever runs the
  candidate — `tunable-contract-extractor` (an eval-K crash) or the main thread
  (an official-run crash) — since sub-agents cannot spawn a diagnosis sub-agent.

## Agents

Agents under `.kimi/agents/` are kimi-cli agent files: each `<name>.yaml`
declares the tool policy (`allowed_tools` / `exclude_tools`) and points at its
system prompt `<name>.md`. There are two supported execution modes:

1. A default main session (plain `kimi` in this repo) follows the loop
   described in `AGENTS.md`/`CLAUDE.md` and spawns bounded child agents — but
   the six role agents are registered only on `autoresearch-experiment`, so
   prefer mode 2 for real runs.
2. Dedicated experiment session:

   ```bash
   python3 tools/kimi_run.py --agent autoresearch-experiment
   ```

   The launcher merges `.kimi/kimi-hooks.toml` into the user config (this
   registers the PreToolUse delegation guard) and execs kimi with
   `--agent-file`. In that mode `autoresearch-experiment` is the main thread
   and spawns the six bounded children itself via the `Agent` tool.

Do not spawn `autoresearch-experiment` as a child agent: kimi-cli launches
subagents without the `Agent` tool, so a nested orchestrator would lose the
independent contexts this project requires.

The role agents (prompts are runtime-adapted copies of `.claude/agents/`):

- `autoresearch-experiment` — self-contained run-level orchestrator for one
  `task_name + tag + run_dir`. Setup = `background-researcher` only; then the
  loop runs in **rounds** (a generation of ≤B ideas at step 0+1, then one
  **decoupled** deep-tuning step). Use one instance per concurrent experiment.
- `autoresearch-hillclimb` — the deliberately simple comparison baseline
  (single working copy, edit → run → keep/revert, `results.tsv`). Launched the
  same way: `python3 tools/kimi_run.py --agent autoresearch-hillclimb`.
- `background-researcher` — setup-time evidence researcher; writes
  `<run_dir>/background.md` + `background_retrieval.json` with scoped,
  credibility-stamped `tf-*` directions. The **only** agent with web tools
  (`SearchWeb`/`FetchURL`); records native fetches via
  `tools/search_backends.py record-visit --backend kimi-fetch`.
- `idea-generator` — SELECT via `tools/got_select.py decide` (deterministic;
  never override op/parents), then IDEATE each action into an idea and persist
  it with `tools/ledger.py add-record`.
- `candidate-writer` — implement one candidate's `train.py` from its ledger
  record. Has **no Shell tool**; cannot run or evaluate anything.
- `tunable-contract-extractor` — step 0+1 for one candidate: `PARAM_SCHEMA` /
  `make_model` refactor, K warm configs + `SEARCH_SPACE`, eval-K with inline
  crash diagnosis (the `crash-diagnosis` skill), records `best_warm_score`.
- `tuner-orchestrator` — once per round on the run dir: the promotion gate
  selects at most one candidate and deep-tunes it in place.
- `experience-extractor` — every 5 rounds: regenerate the ledger's `experience`
  block (levers, lessons, per-`tf-*` run-local direction evidence).

The PreToolUse hook on `Agent` (`tools/harness_guard.py`) deterministically
rejects delegation-boundary violations (e.g. assigning step-0+1/eval work to
`candidate-writer`) and, via `--allowed-subagents`, any `subagent_type` outside
the six role agents — kimi-cli's Agent tool prose always advertises its
built-in `coder`/`explore` types, so the spawn closure is enforced by the hook
rather than by prompt text. kimi-cli's hook JSON uses the same field names as
Claude Code's, so the guard is shared (the allow-list flag is kimi-only;
without it the guard behaves exactly as before).

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
python tools/validate_kimi.py        # the kimi runtime wiring (offline)
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
3. Fill in `TASK.md` (human brief plus the `## Evaluation Contract` section)
   and `task.toml` (machine config: the single `[evaluation].score_fn`,
   metric, parser, required patterns, file constraints). **Scores are always
   lower-is-better** — a higher-is-better metric must be negated/complemented
   inside the task's own `score_fn`. A crash scores `+inf` (the worst).
4. Run `python tools/validate_tasks.py`.

## Shell Command Conventions

- Do not prepend `cd <project-root> &&` to shell commands. The kimi-cli session
  is already at the project root, so the `cd` is redundant.
- Use relative paths from the project root, or quoted absolute paths.
- kimi-cli's Shell tool supports a `timeout` parameter (seconds) — set it
  generously for candidate evaluations instead of letting the default cut a
  long run.

## Boundaries

- `tasks/<task-name>/prepare.py` is the fixed evaluation surface — do not
  modify during normal experiments.
- `tasks/<task-name>/train.py` is the experiment surface, but experiments copy
  it into `runs/<task-name>/<tag>/candidates/<run_id>/` and only the copy gets
  edited. `candidate-writer` generates each candidate's `train.py` under
  `runs/`.
- Do not commit anything under `runs/`. Run logs, `ledger.json`, and
  `loop_state.md` are local-only state.
- Add task dependencies only when `constraints.allow_dependencies = true` in
  the task's `task.toml`.
