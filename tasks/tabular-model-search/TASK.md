# Tabular Model Search

This is a CPU-friendly macOS task for classical machine learning on tabular
classification data. Each run tests one candidate training script on fixed,
noisy synthetic sklearn datasets.

## Goal

Maximize `mean_test_accuracy`, the mean accuracy on fixed hidden test splits
averaged across the configured tabular datasets. Higher is better.

## Files

- `train.py`: baseline candidate template. During autonomous experiments, copy
  it into a run-local candidate directory and edit the copied `train.py`.
- `prepare.py`: fixed synthetic datasets, train/test splits, and hidden test
  scoring helpers. Do not modify during normal experiments.
- `pyproject.toml`: task-local uv environment. Dependency changes are allowed
  when a new CPU-friendly tabular model package is needed.
- `tools/parse_result.py`: generic parser for the task summary, configured by
  `task.toml`.
- `task.toml`: machine-readable run and result contract.

## Program Mapping

This task follows the repository-level `program.md` protocol with these
task-specific file roles:

- Baseline template entrypoint: `train.py`
- Normal editable experiment surface: `runs/tabular-model-search/<tag>/candidates/<run_id>/train.py`
- Fixed data split and hidden evaluation layer: `prepare.py`
- Result parser: `tools/parse_result.py`
- Baseline template run command: `uv run python train.py`
- Candidate run command: `uv --directory tasks/tabular-model-search run python <candidate-dir>/train.py`

This task follows the same `prepare.py` / `train.py` split as the LLM
pretraining baseline: `prepare.py` fixes the data and benchmark contract, while
`train.py` is the editable experiment surface. The training process belongs in
`train.py`; after training, `train.py` passes the fitted estimator into
`prepare.test_accuracy()`. The immutable train/test split, hidden test labels,
and final summary formatting belong in `prepare.py`. Candidate code receives
only training arrays. Each hidden test split may be scored once per process; use
training-only validation inside `train.py` for model selection.

Candidate granularity:

- One candidate directory is one candidate.
- Each candidate directory must contain `prepare.py` and `train.py`.
- `prepare.py` is copied from the task root and treated as readonly.
- `train.py` is copied from the current best candidate, or from the task root
  for the baseline, then edited for the new proposal.
- Do not enumerate many competing candidates inside a single `train.py` run.
- Compare the resulting `mean_test_accuracy` across run IDs in the run ledger.

## Search Space

The intended search space is classical tabular classification on CPU. The
benchmark uses noisy, moderately high-dimensional generated datasets so simple
defaults should leave room for improvement. Keep experiments fast enough for
repeated autonomous runs on a Mac.

Reasonable directions include:

- Linear and margin-based models, such as logistic regression and SVM variants.
- Tree-based models, such as random forests, extra trees, boosted trees, and
  XGBoost-style methods.
- Lightweight neural or nearest-neighbor baselines if they are CPU-friendly.
- Preprocessing choices, such as scaling, feature transforms, feature selection,
  or dimensionality reduction.
- Training logic and hyperparameter choices for the single candidate script.
- Ensembles, including voting, averaging, stacking, blending, or dataset-aware
  selection among strong candidates.

Acceptable dependency additions:

- CPU-friendly packages for tabular ML are allowed when they support the search,
  for example additional boosting libraries or hyperparameter search tools.
- Do not add GPU-only dependencies.
- Keep dependency changes task-local in this directory's `pyproject.toml`.

Comparison rules:

- Primary score is the mean of per-dataset held-out test accuracies.
- Higher `mean_test_accuracy` is better.
- Compare different candidates across runs, not inside one run.
- Prefer simpler models when the score is effectively tied.
- Avoid changes that only overfit one dataset while harming the average.
- Do not inspect, reconstruct, or repeatedly query hidden test labels from
  `prepare.py`.
- Do not catch broad training or scoring exceptions and convert them into
  artificial scores. If a candidate cannot train or score, let the process fail
  so the parser records the run as `crash`.

## Run

From the repository root:

```bash
uv --directory tasks/tabular-model-search sync
uv --directory tasks/tabular-model-search run python train.py
```

During autonomous experiments, create a candidate directory first:

```bash
python tools/new_candidate.py tabular-model-search <tag> 000
python tools/new_candidate.py tabular-model-search <tag> 001 --from-candidate 000
```

Then run the copied candidate entrypoint with the task-local uv environment:

```bash
uv --directory tasks/tabular-model-search run python "$(pwd)/runs/tabular-model-search/<tag>/candidates/<run_id>/train.py"
```

Redirect output to a run log under:

```text
runs/tabular-model-search/<tag>/
```

Parse the log with `--commit worktree` because candidate files under `runs/`
are intentionally outside git:

```bash
python tools/parse_result.py \
  runs/tabular-model-search/<tag>/run-<run_id>.log \
  --run-id <run_id> \
  --commit worktree \
  --append runs/tabular-model-search/<tag>/results.tsv \
  --description "<short idea>"
```

## Output Format

The script prints a final summary like:

```text
---
metric:           mean_test_accuracy
score:            0.668122
best_model:       extra_trees_baseline
dataset_scores:   noisy_binary=0.807143 sparse_multiclass=0.634259 high_dimensional=0.562963
fit_seconds:      0.9
num_datasets:     3
num_candidates:   1
```

A completed run must include the patterns listed in `task.toml`.
`prepare.py` owns the summary field names and formatting; candidate `train.py`
files should train estimators, call `test_accuracy(estimator, dataset)`, collect
scores, and call the fixed print helpers. The generic parser writes `run_id`,
`commit`, `metric`, `value`, `best_model`, `status`, and `description` columns.

The `results.tsv` header for this task is:

```text
run_id	commit	metric	value	best_model	status	description
```

When parser status is left as `auto`, completed runs are marked `keep` only if
their `score` strictly improves over the best previous kept value in
`results.tsv`. Completed non-improving runs are marked `discard`; logs without a
complete summary are marked `crash`. When `--append` is used, the parser also
updates `runs/tabular-model-search/<tag>/loop_state.md` from the ledger.
