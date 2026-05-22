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

## Files

- `prepare.py`: fixed data prep, tokenizer, dataloader, and evaluation.
- `train.py`: model, optimizer, hyperparameters, and training loop.
- `pyproject.toml`: this task's uv environment.
- `uv.lock`: this task's locked dependency resolution.

## Run

From the repository root:

```bash
uv --directory tasks/autoresearch-baseline sync
uv --directory tasks/autoresearch-baseline run python prepare.py
uv --directory tasks/autoresearch-baseline run python train.py
```

During autonomous experiments, redirect training output to a run log under the
run directory and parse the final summary into `results.tsv`.

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

The generic parser writes this TSV shape:

```text
run_id	commit	metric	value	best_model	status	description
```
