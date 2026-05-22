# autoresearch-automl

![teaser](tasks/autoresearch-baseline/progress.png)

This repository is a Claude Code-driven harness for running multiple
independent autoresearch-style tasks. Each task owns its own uv environment
and lockfile, so tasks can use different or conflicting dependencies without
forcing the whole repo into one Python environment.

The original autoresearch example is preserved as:

```text
tasks/autoresearch-baseline/
```

## Quick Start

Requirements for the baseline task: a single NVIDIA GPU, Python 3.10+, and
`uv`.

```bash
uv --directory tasks/autoresearch-baseline sync
uv --directory tasks/autoresearch-baseline run python prepare.py
uv --directory tasks/autoresearch-baseline run python train.py
```

## Project Structure

```text
.claude/                    Claude Code skills
CLAUDE.md                   Project context auto-loaded by Claude Code
program.md                  Canonical autoresearch experiment protocol
pyproject.toml              Lightweight root harness project
tasks/                      Independent uv task projects
  autoresearch-baseline/    Original torch-based autoresearch task
  tabular-model-search/     CPU tabular ML model search task
tools/                      Local validation and task creation helpers
runs/                       Local run artifacts, ignored by git
```

Each task follows this shape:

```text
tasks/<task-name>/
  TASK.md                   Human-readable task brief
  task.toml                 Machine-readable task metadata
  pyproject.toml            Task-local uv environment
  uv.lock                   Task-local lockfile
  <task code>
```

Do not configure `tasks/*` as a uv workspace. Run tasks with `uv --directory`
or by changing into the task directory.

## Claude Code Skills

Project-local skills live under `.claude/skills/<skill-name>/SKILL.md` and are
auto-discovered by Claude Code based on each skill's `description` field.
The current skills are:

- `idea-proposer` — propose one diverse experimental idea per candidate,
  with axis-diversity over recent history
- `hyperparam-tuner-llm` — Phase A warm-start methodology consumed by the
  `tuner-orchestrator` agent

The orchestration logic itself lives in three subagents under
`.claude/agents/`: `candidate-writer` (idea → code), `tuner-orchestrator`
(three-phase hyperparameter tuning), and `crash-diagnoser` (verdict on a
crashed run). See `program.md` for how they compose.

Validate local metadata:

```bash
python tools/validate_skills.py
python tools/validate_tasks.py
```

## Baseline Task

The baseline task keeps the upstream autoresearch contract:

- `program.md` is the core autonomous experiment protocol.
- `tasks/autoresearch-baseline/prepare.py` owns data preparation, tokenizer
  loading, dataloading, and `evaluate_bpb`.
- `tasks/autoresearch-baseline/train.py` is the normal experiment surface.
- The metric is `val_bpb`; lower is better.
- `runs/<task-name>/<tag>/` contains run logs and `results.tsv`; `runs/` is
  local artifact state and is not committed.

See `CLAUDE.md` and `program.md` for details.

## Tabular Model Search

`tasks/tabular-model-search/` is a macOS/CPU-friendly task for classical
tabular classification. Each run-local candidate directory contains a copied
`prepare.py` and `train.py`; experiments modify the copied `train.py` and score
it on fixed train/test splits for noisy synthetic sklearn datasets. The
configured metric is `mean_test_accuracy`, and higher is better.
