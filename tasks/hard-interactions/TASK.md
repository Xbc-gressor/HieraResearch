# Hard Interactions

A CPU-friendly synthetic tabular classification task **deliberately built to be
hard, with large headroom** — so that better search and deeper tuning yield
meaningfully better scores (used to give framework-hyperparameter experiments
real discriminating power, unlike a noise-capped task that plateaus immediately).

## Goal

Maximize accuracy on fixed hidden test splits, averaged across three synthetic
datasets whose targets depend on **high-order feature interactions** (XOR/parity
of binarized features, sign of pairwise products, threshold conjunctions) buried
among many pure-noise features, with only low label noise (~3%). The framework
**always minimizes**, so this task reports `neg_mean_test_accuracy = -mean(accuracy)`
— **lower (more negative) is better**.

Why it has headroom: linear/shallow models cannot represent the parity/XOR terms
→ they stay near ~60%. Only candidates that capture the interactions (deep GBDT
with enough depth, explicit polynomial/interaction features, MLPs, well-tuned)
approach the ~88–92% ceiling. So the outer S-GoT idea search **and** the inner
deep-tune both have room to matter — different framework configurations should
reach different scores.

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
  (`PARAM_SCHEMA`, `SEARCH_SPACE`, `BASE_PARAMS`) written by
  `tunable-contract-extractor`.
- **Train**: train only on `dataset.x_train` / `dataset.y_train`.
- **Score**: `evaluation.score_fn` (`prepare.evaluate_config(make_model, params)`)
  is the ONE evaluation surface — it builds + fits + scores on the held-out test
  split for every dataset and returns the mean negative accuracy (lower is
  better). Its return value **is** the candidate's `final_best_score`.

Rules:

- Do not inspect, reconstruct, or repeatedly query the hidden test labels.
- Do not catch broad training/scoring exceptions to fabricate a score. If a
  candidate cannot build/fit/score, let it fail so the run is recorded as `crash`.
- One candidate strategy per `train.py`; use training-only validation for any
  in-candidate model selection.

## Files

- `train.py`: no task-root baseline; `candidate-writer` writes each candidate's
  `train.py` (a `fresh` candidate from a `background.md` try-first direction, or
  informed by parents for `improve`/`crossover`).
- `prepare.py`: fixed synthetic datasets, splits, and the single `evaluate_config`
  scoring function. Readonly during normal experiments.
- `pyproject.toml`: task-local uv env. CPU-friendly dependency additions allowed.
- `task.toml`: machine-readable run/result contract.

## Search Space

The intended search is classical + interaction-aware tabular classification on
CPU. The targets reward methods that **capture high-order interactions**, so
fruitful directions include:

- Gradient boosting with **enough depth / leaves** to represent order-k
  interactions (XGBoost, LightGBM, HistGradientBoosting, CatBoost), carefully
  regularized and tuned.
- Explicit **feature engineering**: polynomial / interaction features
  (`PolynomialFeatures`), feature crosses, binarization, before a classifier.
- **MLPs** / kernel methods (RBF SVM, Nystroem + linear) that model non-linear
  interactions.
- **Feature selection** to cut the many pure-noise columns (the first ~8–10 of
  each dataset are active; the rest are noise).
- Ensembles / stacking of complementary interaction-capturing models.

Avoid: purely linear models, or very shallow trees, which cannot represent the
parity/XOR structure and will plateau near ~60%.

Comparison rules: lower (more negative) `neg_mean_test_accuracy` is better;
compare candidates across runs; prefer simpler models when effectively tied.

## Run

There is **no `python train.py` run**: a candidate is scored only where the tuner
scripts call `evaluate_config`. To evaluate a candidate by hand:

```bash
uv --directory tasks/hard-interactions sync
python tools/new_candidate.py hard-interactions <tag> <run_id> --skip-entrypoint
# after candidate-writer + tunable-contract-extractor produce train.py + _warm_configs.json:
uv --directory tasks/hard-interactions run python tools/tuners/warmstart_eval.py \
  --candidate-path   runs/hard-interactions/<tag>/candidates/<run_id>/train.py \
  --configs-json     runs/hard-interactions/<tag>/candidates/<run_id>/_warm_configs.json \
  --tune-report-json runs/hard-interactions/<tag>/candidates/<run_id>/tune_report.json
```

Normally the experiment loop drives this through its agents; see `program.md`.

## Scoring And Recording

There is no run-log summary. The tuner scripts call
`prepare.evaluate_config(make_model, params)` and the score is written straight to
`runs/hard-interactions/<tag>/ledger.json` via `tools/ledger.py`
(`tunable-contract-extractor` records `final_best_score` = `best_warm_score`;
`tuner-orchestrator`, if it selects the candidate, lowers it with the tuned best).
A completed run is `keep` only if its `final_best_score` strictly improves over
the best previous kept value, otherwise `discard`; an unrunnable candidate is
`crash` (`+inf`). `record-run` also regenerates `loop_state.md`.
