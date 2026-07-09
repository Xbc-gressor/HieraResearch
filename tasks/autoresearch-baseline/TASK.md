# Autoresearch Baseline

This task preserves the original single-GPU autoresearch example. It gives an
agent a small LLM pretraining setup and asks it to improve validation bits per
byte by editing `train.py` while keeping `prepare.py` and the evaluation metric
fixed.

## Goal

Minimize `val_bpb`. Lower is better.

The training script runs for a fixed 5-minute training budget, excluding startup
and compilation. VRAM is a soft constraint: some increase is acceptable for a
meaningful `val_bpb` gain, but it should not blow up dramatically.

## Evaluation Contract

Authoritative description of how a candidate must train, score, and report.
`task.toml` declares the evaluation function name as config
(`evaluation.score_fn`); this section explains its semantics. There is no
`evaluation.tuning_fn` — `prepare.py` exposes no tuning oracle, so this task is
not tuner-ready.

- **Train**: `prepare.make_dataloader(tokenizer, B, T, split)` provides token
  batches; the training budget is the fixed wall-clock limit enforced inside
  `train.py`.
- **Score (official)**: `evaluation.score_fn`
  (`prepare.evaluate_bpb(model, tokenizer, batch_size)`) computes `val_bpb` on
  the validation split after training.
- **Report**: `train.py` prints the final summary lines itself; a completed run
  must include `val_bpb:` and `peak_vram_mb:`.

Rules:

- Keep `prepare.py`, the tokenizer, and the evaluation metric fixed.
- Keep the fixed training-time budget logic intact.
- VRAM is a soft constraint; do not blow it up dramatically.

## Files

- `prepare.py`: fixed data prep, tokenizer, dataloader, and evaluation.
- `train.py`: model, optimizer, hyperparameters, and training loop.
- `pyproject.toml`: this task's uv environment.
- `uv.lock`: this task's locked dependency resolution.

During autonomous experiments this task uses candidate directories. If a task
root `train.py` is present it is one provided-baseline candidate; otherwise
`candidate-writer` writes each candidate's `train.py` (a `fresh` candidate from
scratch, or informed by parent candidates for `improve`/`crossover`). Only
run-local candidate files are edited.

## Run

From the repository root:

```bash
uv --directory tasks/autoresearch-baseline sync
uv --directory tasks/autoresearch-baseline run python prepare.py
uv --directory tasks/autoresearch-baseline run python train.py
```

During autonomous experiments, redirect training output to a run log under the
run directory and parse the final summary into `ledger.json` (via
`tools/parse_result.py --ledger`).

When using the repository-level run layout, write logs and results under:

```text
runs/autoresearch-baseline/<tag>/
```

## Output Format

Once the training script finishes it prints a summary like this:

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

The key metric is `val_bpb`, and lower is better. A completed run should include
both `val_bpb:` and `peak_vram_mb:` in the run log. The generic task parser is
declared in `task.toml`.

The parser records the result into the candidate's `ledger.json` record (one
JSON record per run; see `.claude/rules/ledger.md`). The parsed task metric is
stored as `final_best_score`; `best_warm_score` is written earlier by the
tuner (and stays `null` for untuned seeds — this task is not tuner-ready).
