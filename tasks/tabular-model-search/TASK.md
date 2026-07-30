# Tabular Model Search

This is a CPU-friendly macOS task for classical machine learning on tabular
classification data. Each run tests one candidate training script on fixed,
noisy synthetic sklearn datasets.

## Goal

Maximize accuracy on the fixed hidden test splits, averaged across the
configured tabular datasets. The framework **always minimizes**, so this task
reports `neg_mean_test_accuracy = -mean(accuracy)` and the optimizer minimizes
it — **lower (more negative) is better**, equivalent to higher accuracy.

## Evaluation Contract

Authoritative description of how a candidate must train, score, and report for
this task. `task.toml` holds the machine-readable config (`[evaluation].score_fn`,
`[result]` metric, `[constraints]` file boundaries); this section holds the prose
contract that subagents read before writing or tuning a candidate. When this
section and `task.toml` disagree, `task.toml` wins for values it actually declares.

There is **one global `config → score` function** and **no separate official
run**: a candidate is never executed as `python train.py`. Its score is produced
where `make_model` is evaluated against that function by the tuner scripts.

- **Construct**: the candidate's `train.py` exposes `make_model(dataset, params)`
  returning an unfitted sklearn-style estimator (`.fit` / `.predict`), plus the
  tuner contract (`PARAM_SCHEMA`, `SEARCH_SPACE`, `BASE_PARAMS`) that
  `tunable-contract-extractor` writes at step 0+1.
- **Train**: `prepare.load_datasets()` returns `DatasetSplit` items; train only
  on `dataset.x_train` / `dataset.y_train`.
- **Score**: `evaluation.score_fn` (`prepare.evaluate_config(make_model, params)`)
  is the ONE evaluation surface — the tuner scripts call it for warm-start eval
  and Phase C search. It builds + fits + scores on the held-out test split for
  every dataset and returns the mean negative accuracy (lower is better). The
  value it returns **is** the candidate's `final_best_score`; there is no
  separate official run to re-score it.

Rules:

- Do not inspect, reconstruct, or repeatedly query the hidden test labels.
- Do not catch broad training or scoring exceptions to fabricate a score. If a
  candidate cannot build/fit/score, let it fail so the run is recorded as `crash`.
- One candidate strategy per `train.py`; do not enumerate competing candidates.
  Use training-only validation for any in-candidate model selection.

## Files

- `train.py`: no task-root baseline is provided; `candidate-writer` writes each
  complete candidate at its validated `background.md` semantic point (`fresh`
  from scratch, or informed by numeric parents for `improve`/`crossover`).
- `prepare.py`: fixed synthetic datasets, train/test splits, and the single
  `evaluate_config` scoring function. Do not modify during normal experiments.
- `pyproject.toml`: task-local uv environment. Dependency changes are allowed
  when a new CPU-friendly tabular model package is needed.
- `task.toml`: machine-readable run and result contract.

## Program Mapping

This task follows the repository-level experiment protocol (see
`.claude/agents/autoresearch-experiment.md`) with these task-specific file
roles:

- Editable experiment surface: `runs/tabular-model-search/<tag>/candidates/<run_id>/train.py`
- Fixed data split + the one `config → score` function: `prepare.py`
- Scoring: the tuner scripts call `prepare.evaluate_config(make_model, params)`
  (warm-start eval + Phase C); there is no `python train.py` run.

`prepare.py` fixes the data and the scoring contract; the candidate's `train.py`
exposes `make_model(dataset, params)` + the tuner contract. Candidate code
receives only training arrays. The test split is the optimization target (one
`config → score` function, no separate held-out official run); use training-only
validation inside the candidate for any in-candidate model selection.

Candidate granularity:

- One candidate directory is one candidate.
- Each candidate directory must contain `prepare.py` and `train.py`.
- `prepare.py` is copied from the task root and treated as readonly.
- A candidate's `train.py` is written by `candidate-writer` — from scratch for a
  `fresh` candidate, or informed by the parent candidates' `train.py` for an
  `improve`/`crossover` candidate.
- Do not enumerate many competing candidates inside a single `train.py`.
- Compare the resulting `neg_mean_test_accuracy` across run IDs in the run
  ledger (lower is better).

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

- Primary score is `neg_mean_test_accuracy` = the negative mean of per-dataset
  held-out test accuracies.
- Lower (more negative) `neg_mean_test_accuracy` is better.
- Compare different candidates across runs, not inside one run.
- Prefer simpler models when the score is effectively tied.
- Avoid changes that only overfit one dataset while harming the average.
- Do not inspect, reconstruct, or repeatedly query hidden test labels from
  `prepare.py`.
- Do not catch broad training or scoring exceptions and convert them into
  artificial scores. If a candidate cannot train or score, let the process fail
  so the parser records the run as `crash`.

## Run

There is **no `python train.py` run** for this task: a candidate is scored only
where the tuner scripts call `evaluate_config` (one `config → score` function, no
separate official run). To evaluate a candidate by hand:

```bash
# 1. Sync the task env (once).
uv --directory tasks/tabular-model-search sync

# 2. After ledger.py add-record has persisted <run_id>, create the candidate
#    directory. This copies prepare.py and derives _candidate_brief.json;
#    candidate-writer writes train.py (do NOT pre-copy a baseline train.py).
python tools/new_candidate.py tabular-model-search <tag> <run_id> --skip-entrypoint

# 3. Once train.py + _warm_configs.json exist (candidate-writer +
#    tunable-contract-extractor), score the K warm configs against evaluate_config
#    in the task-local uv env (step 0+1):
#    (--project selects the task env without chdir, so the repo-relative paths below resolve)
uv --project tasks/tabular-model-search run python tools/tuners/warmstart_eval.py \
  --candidate-path   runs/tabular-model-search/<tag>/candidates/<run_id>/train.py \
  --configs-json     runs/tabular-model-search/<tag>/candidates/<run_id>/_warm_configs.json \
  --tune-report-json runs/tabular-model-search/<tag>/candidates/<run_id>/tune_report.json
```

Normally the experiment loop drives this through its agents
(`tunable-contract-extractor` for step 0+1, `tuner-orchestrator` for the decoupled
deep-tuning), not by hand. Candidate files under `runs/` are
intentionally outside git.

## Scoring And Recording

There is **no run-log summary** — a candidate is never run as a script. The tuner
scripts call `prepare.evaluate_config(make_model, params)` (warm-start eval +
Phase C) and the score is written straight to `ledger.json` via `tools/ledger.py`:

- `tunable-contract-extractor` (step 0+1) records the warm-start best as the
  candidate's `final_best_score` (= `best_warm_score`) with `ledger.py record-run`,
  and the warm metadata with `set-tuning` (no `--mark-tuned`).
- `tuner-orchestrator`, if it selects the candidate, calls
  `tools/finalize_tuning.py`. The helper accepts only a terminal Phase-C report,
  applies the global best, and updates `final_best_score`, status, tuning
  metadata, and `tune: true` together. An interrupted search leaves all of
  those downstream fields unchanged.

The candidate's result lives in `runs/tabular-model-search/<tag>/ledger.json`
(one record per run; see `.claude/rules/ledger.md`). `ledger.py record-run`
computes the keep/discard status: a completed run is `keep` only if its
`final_best_score` strictly improves over the best previous kept value in the
ledger, otherwise `discard`; a candidate that cannot be evaluated is `crash`
(`final_best_score` `+inf`). `record-run` also regenerates
`runs/tabular-model-search/<tag>/loop_state.md` from the ledger.
