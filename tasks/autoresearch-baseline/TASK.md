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
  original hyperparameter values. The trainer also exposes
  **`preflight() -> dict`**, which constructs the real model/optimizer and runs
  exactly one real-shape training step, but never invokes validation or returns
  an objective score.
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
- **Preflight**: before each proposed config may enter `evaluate_config`, the
  framework tuner or standalone hillclimb runner calls the fixed
  `prepare.preflight_config(make_model, params)` in an isolated subprocess. It
  calls the trainer's `preflight()` only; it must not call `evaluate_bpb`,
  inspect validation data, or emit a score. The fixed `PreflightEnv`
  mechanically rejects validation and non-training dataloader access. A failure
  creates a feasibility receipt and is repaired/rejected before `score_fn`, so
  it is reported separately from the objective-call budget.
- **Resource probe**: the framework additionally runs
  `prepare.resource_probe_config(make_model, params)` — the same no-score
  contract, but with the dataloader's `T` pinned to `env.max_seq_len` — to
  measure the worst-case training-shape memory envelope. The search-space clamp
  uses this probe, not `preflight_config`, because a candidate whose `run()`
  ramps sequence length peaks well above its own first step. Both probes are
  no-score and consume no objective budget.

Rules:

- Keep `prepare.py`, the tokenizer, and the evaluation metric fixed.
- Respect `env.train_budget_seconds`; derive all randomness from `env.seed`.
- Express effective batch size with the independent tuner coordinates
  `device_batch_size` and `grad_accum_steps`; derive `total_batch_size` as
  `device_batch_size * env.max_seq_len * grad_accum_steps`. Do not expose
  `total_batch_size` as an independently sampled coordinate.
- VRAM is a soft constraint; do not blow it up dramatically.
- Do not catch broad exceptions to fabricate a score. If a candidate cannot
  build/train/return a finite `val_bpb`, let it fail so the run is recorded
  as `crash`.
- Keep `preflight()` behaviorally aligned with the construction and first
  training step used by `run()`; it may return resource telemetry such as peak
  VRAM, but never `val_bpb` or another validation-derived value. The model must
  be constructible at the full `env.max_seq_len` even when `run()` starts below
  it, because the framework probes that shape for its memory envelope.
- Keep the standalone
  `if __name__ == "__main__": evaluate_config(make_model, DEFAULT_PARAMS)`
  driver structurally unchanged. Candidate preflight reads `DEFAULT_PARAMS`, so
  standalone hyperparameter changes must edit that mapping rather than pass a
  second ad-hoc params dict only from `__main__`. Structural model/trainer
  changes remain in the code reached by `make_model`, where both paths see them.
- One candidate strategy per `train.py`; do not enumerate competing candidates.

Evaluation cost: one config eval is one full budgeted training run (~300 s of
training plus startup/compilation and the final eval). New runs inherit this
task's 900 s `run.timeout_seconds` as their per-config `per_runtime_limit`;
`init_run.py --timeout` may override it. Consider lowering `tuner.K` /
`tuner.K_eval` in `framework_cfg.json` — the defaults cost 3 full training runs
per candidate at step 0+1.

The run-level `tools/preflight_env.py` check and candidate preflight calls do
not consume `max_evaluations` because they never enter `score_fn`. Their
attempts, failures, feasibility rejections, and runtime remain auditable
separately. Every admitted `score_fn` call—including one that later crashes—
atomically consumes one objective slot.

## Files

- `prepare.py`: fixed data prep, tokenizer, dataloader, the `PretrainEnv`
  task environment, the fixed `evaluate_bpb` metric, and the single
  `evaluate_config` scoring function. Readonly during experiments.
- `train.py`: model, optimizer, hyperparameters, and training loop behind
  `make_model(env, params)`; also directly runnable (see below).
- `pyproject.toml`: this task's uv environment.
- `uv.lock`: this task's locked dependency resolution.

During autonomous experiments this task uses candidate directories. Because the
task-root `train.py` is declared by `[seed].provided`, the experiment admits its
unchanged run-local copy first at the all-baselines semantic point. Its exact
`DEFAULT_PARAMS` are the sole step-0+1 baseline evaluation; the normal decoupled
tuner may later tune that same point. Otherwise `candidate-writer` writes each
candidate's `train.py` (a `fresh` candidate from scratch, or informed by parent
candidates for `improve`/`crossover`). Only run-local candidate files are edited.

## Run

Standalone: `python train.py` passes `make_model` + `DEFAULT_PARAMS` through the
same fixed `evaluate_config` score surface and prints the parseable summary.
`autoresearch-hillclimb` first runs `tools/preflight_candidate.py` against the
working copy, atomically reserves one objective slot, and only then invokes this
entrypoint:

```bash
uv --directory tasks/autoresearch-baseline sync
uv --directory tasks/autoresearch-baseline run python prepare.py
uv --directory tasks/autoresearch-baseline run python train.py
```

Under manual mode, read the printed summary directly. Under hillclimb, redirect
it to the run log, validate the required result patterns, and append exactly one
row per reserved objective attempt to `results.tsv`; do not create a framework
ledger or invoke the legacy `parse_result.py --ledger` path.

Under the experiment loop there is **no `python train.py` run**: a candidate is
scored only where the tuner scripts call `evaluate_config`:

```bash
# Run 000 uses --provided-baseline after deterministic all-baselines admission.
python tools/new_candidate.py autoresearch-baseline <tag> 000 --provided-baseline
# Later records derive _candidate_brief.json without copying the entrypoint.
python tools/new_candidate.py autoresearch-baseline <tag> <run_id> --skip-entrypoint
# after candidate-writer + tunable-contract-extractor produce train.py + _warm_configs.json:
# (--project selects the task env without chdir, so the repo-relative paths below resolve)
uv --project tasks/autoresearch-baseline run python tools/tuners/warmstart_eval.py \
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
`driver/prompts/rules/ledger.md`.
