# Autoresearch Baseline

This task preserves the original single-GPU autoresearch example. It gives an
agent a small LLM pretraining setup and asks it to improve validation bits per
byte by editing `train.py` while keeping `prepare.py` and the evaluation metric
fixed. It follows the framework's single config→score contract, so it runs
under both `autoresearch-experiment` (tuner loop) and `autoresearch-hillclimb`
(standalone edit/run loop).

## Goal

Minimize `val_bpb`. Lower is better.

The training script runs for a fixed 5-minute training budget, excluding startup
and compilation. VRAM is a soft constraint: some increase is acceptable for a
meaningful `val_bpb` gain, but it should not blow up dramatically.

## Evaluation Contract

Authoritative description of how a candidate must construct, train, score, and
report. `task.toml` holds the machine-readable config (`[evaluation].score_fn`,
`[result]` metric, `[constraints]`); this section holds the prose contract. When
they disagree, `task.toml` wins for values it declares.

There is **one global `config → score` function** and **no separate official
run**: under the experiment loop a candidate is never executed as
`python train.py`. Its score is produced where `make_model` is evaluated
against that function by the tuner scripts.

- **Construct**: `train.py` exposes `make_model(env, params)` returning a
  configured **trainer object** with a `run() -> float` method, plus the tuner
  contract (`PARAM_SCHEMA`, `SEARCH_SPACE`, `BASE_PARAMS`) written by
  `tunable-contract-extractor`. The provided task-root `train.py` already
  carries `make_model` + `PARAM_SCHEMA`, and its `DEFAULT_PARAMS` are the
  original hyperparameter values.
- **Train**: `env` is the task-defined `prepare.PretrainEnv`. The trainer
  trains only via `env.make_dataloader(tokenizer, B, T, "train")`, for at most
  `env.train_budget_seconds` (300 s) of training time — counted after warmup
  steps, excluding startup and compilation, exactly as the original script
  accounted for it. Readable env attributes: `tokenizer`, `make_dataloader`,
  `evaluate_bpb`, `max_seq_len`, `train_budget_seconds`, `vocab_size`,
  `device`, `seed`.
- **Score**: `evaluation.score_fn`
  (`prepare.evaluate_config(make_model, params)`) is the ONE evaluation
  surface — it runs one full budgeted training run and returns the
  post-training `val_bpb` computed by the fixed `env.evaluate_bpb`. Its return
  value **is** the candidate's `final_best_score`.

Rules:

- Keep `prepare.py`, the tokenizer, and the evaluation metric fixed.
- Respect `env.train_budget_seconds`; derive all randomness from `env.seed`.
- VRAM is a soft constraint; do not blow it up dramatically.
- Do not catch broad exceptions to fabricate a score. If a candidate cannot
  build/train/return a finite `val_bpb`, let it fail so the run is recorded
  as `crash`.
- One candidate strategy per `train.py`; do not enumerate competing candidates.

Evaluation cost: one config eval is one full budgeted training run (~300 s of
training plus startup/compilation and the final eval). Set
`init_run.py --timeout` (the per-config `per_runtime_limit` hard kill) to ≈900 s
for this task, and consider lowering `tuner.K` / `tuner.K_eval` in
`framework_cfg.json` — the defaults cost 3 full training runs per candidate at
step 0+1.

## Files

- `prepare.py`: fixed data prep, tokenizer, dataloader, the `PretrainEnv`
  task environment, the fixed `evaluate_bpb` metric, and the single
  `evaluate_config` scoring function. Readonly during experiments.
- `train.py`: model, optimizer, hyperparameters, and training loop behind
  `make_model(env, params)`; also directly runnable (see below).
- `pyproject.toml`: this task's uv environment.
- `uv.lock`: this task's locked dependency resolution.

During autonomous experiments this task uses candidate directories. If a task
root `train.py` is present it is one provided-baseline candidate; otherwise
`candidate-writer` writes each candidate's `train.py` (a `fresh` candidate from
scratch, or informed by parent candidates for `improve`/`crossover`). Only
run-local candidate files are edited.

## Run

Standalone (manual runs and `autoresearch-hillclimb`): `python train.py` runs
`DEFAULT_PARAMS` through the identical `make_model`/trainer path and prints the
parseable summary:

```bash
uv --directory tasks/autoresearch-baseline sync
uv --directory tasks/autoresearch-baseline run python prepare.py
uv --directory tasks/autoresearch-baseline run python train.py
```

Under this mode, redirect training output to a run log under the run directory
and parse the final summary into `ledger.json` (via
`tools/parse_result.py --ledger`).

Under the experiment loop there is **no `python train.py` run**: a candidate is
scored only where the tuner scripts call `evaluate_config`:

```bash
# Requires an existing <run_id> ledger record; also derives _candidate_brief.json.
python tools/new_candidate.py autoresearch-baseline <tag> <run_id> --skip-entrypoint
# after candidate-writer + tunable-contract-extractor produce train.py + _warm_configs.json:
uv --directory tasks/autoresearch-baseline run python tools/tuners/warmstart_eval.py \
  --candidate-path   runs/autoresearch-baseline/<tag>/candidates/<run_id>/train.py \
  --configs-json     runs/autoresearch-baseline/<tag>/candidates/<run_id>/_warm_configs.json \
  --tune-report-json runs/autoresearch-baseline/<tag>/candidates/<run_id>/tune_report.json
```

When using the repository-level run layout, write logs and results under:

```text
runs/autoresearch-baseline/<tag>/
```

## Output Format

`Trainer.run()` prints a summary like this in both modes:

```text
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

The key metric is `val_bpb`, and lower is better. In standalone mode a
completed run should include both `val_bpb:` and `peak_vram_mb:` in the run
log; the generic task parser is declared in `task.toml`.

Under the experiment loop there is no log parse: `tunable-contract-extractor`
records `final_best_score` = `best_warm_score` straight into the candidate's
`ledger.json` record via `tools/ledger.py` (`record-run` + `set-tuning`); the
decoupled `tuner-orchestrator`, if it selects the candidate, lowers
`final_best_score` with the tuned best. One JSON record per run; see
`.claude/rules/ledger.md`.
