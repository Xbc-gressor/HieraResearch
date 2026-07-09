# Tabular (Blind)

A CPU-friendly tabular classification task over **three real datasets**, shipped
**anonymized** (`d0`/`d1`/`d2`) — their identities, feature meanings, and which
model/hyperparameters win are deliberately withheld. Nothing here tells you what
wins. **That is the point: discover it by experiment** (try models, tune
hyperparameters, read the held-out score), not by looking anything up. The
datasets were chosen so that careful tuning yields a large benefit that is only
reached after a long search — good tuning/search is rewarded.

## Goal

Maximize **balanced accuracy** on fixed held-out test splits, averaged across the
three datasets. The framework **always minimizes**, so this task reports
`neg_mean_balanced_accuracy = -mean(balanced_accuracy)` — **lower (more negative)
is better**.

There is meaningful headroom above naive baselines (a simple default model is far
from the achievable ceiling), so better search and deeper tuning should yield
meaningfully better scores. How much of that headroom you capture, and how, is
for the search to find out — no hints are given.

## Evaluation Contract

Authoritative description of how a candidate must train, score, and report.
`task.toml` holds the machine-readable config (`[evaluation].score_fn`, `[result]`
metric, `[constraints]`); this section holds the prose contract. When they
disagree, `task.toml` wins for values it declares.

There is **one global `config → score` function** and **no separate official
run**: a candidate is never executed as `python train.py`. Its score is produced
where `make_model` is evaluated against that function by the tuner scripts.

- **Construct**: `train.py` exposes `make_model(dataset, params)` returning an
  unfitted sklearn-style estimator (`.fit` / `.predict`), plus the tuner contract
  (`PARAM_SCHEMA`, `SEARCH_SPACE`, `BASE_PARAMS`) when made tunable.
- **Train**: train only on `dataset.x_train` / `dataset.y_train`.
- **Score**: `evaluation.score_fn` (`prepare.evaluate_config(make_model, params)`)
  is the ONE evaluation surface — it builds + fits + scores on the held-out test
  split for every dataset and returns the mean negative balanced accuracy (lower is
  better). Its return value **is** the candidate's score.

Rules:

- Do not inspect, reconstruct, or repeatedly query the hidden test labels.
- Do not reverse-engineer the data file's provenance to shortcut the search; learn
  structure only from the training split you are given.
- Do not catch broad training/scoring exceptions to fabricate a score. If a
  candidate cannot build/fit/score, let it fail so the run is recorded as `crash`.
- One candidate strategy per `train.py`; use training-only validation for any
  in-candidate model selection.

## Files

- `train.py`: no task-root baseline; each candidate's `train.py` is written by the
  experiment (from scratch, or informed by parents).
- `prepare.py`: loads the fixed opaque datasets and the single `evaluate_config`
  scoring function. Readonly during normal experiments.
- `data/datasets.npz`: the precomputed datasets (opaque). Readonly; do not modify.
- `pyproject.toml`: task-local uv env. CPU-friendly dependency additions allowed.
- `task.toml`: machine-readable run/result contract.

## Approach

Classical CPU tabular classification is in scope — linear models, trees,
gradient boosting, SVMs / kernel methods, MLPs, feature engineering, feature
selection, ensembles/stacking. **Which of these help, and what (if any) feature
engineering matters, is deliberately unstated — determine it empirically** by
running candidates and comparing the held-out score. Use training-only
cross-validation to form hypotheses; let the score decide.

Comparison rules: lower (more negative) `neg_mean_balanced_accuracy` is better;
compare candidates across runs; prefer simpler models when effectively tied.

## Run

There is **no `python train.py` run**: a candidate is scored only where the tuner
scripts (or a hand driver) call `evaluate_config(make_model, params)`. The
experiment loop drives this through its agents.

## Scoring And Recording

The score comes from `prepare.evaluate_config(make_model, params)` and is recorded
to the run's ledger. A completed run is `keep` only if its score strictly improves
over the best previous kept value, otherwise `discard`; an unrunnable candidate is
`crash` (`+inf`).
