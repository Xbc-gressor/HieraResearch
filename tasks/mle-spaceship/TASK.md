# Spaceship Titanic

Predict which passengers were transported to an alternate dimension. Pursue the
strongest generalization possible within the available runtime. This task uses
CPU; no GPU is required.

## Evaluation Contract

`make_model(dataset, params)` returns an unfitted estimator implementing `fit`
and `predict`. Inputs are pandas DataFrames carrying the 13 feature columns of
the public training data (identifiers and free-text fields included); labels are
boolean `Transported` values and predictions must be boolean labels. Keep the
tuner contract in `train.py`.

`prepare.evaluate_config` owns a fixed stratified 80/20 split (seed 42) of the
MLE-bench **public training data**. The competition metric is accuracy; the
score returned is `1 - validation accuracy`, so lower is better, and it is the
only feedback used for screening, tuning and candidate selection. Do not
inspect the held-out validation labels from candidate code.

The final selected implementation is refit on all public training rows by
`prepare.export_submission`. It emits `PassengerId,Transported` predictions for
the public test inputs. Official MLE-bench grading happens outside the agent
environment after search and submission generation have finished. Official
scores and medal thresholds must not be used for further candidate selection.

## Data and resources

`MLEBENCH_PUBLIC_DATA` points to a read-only directory containing only the
official prepared public `train.csv`, `test.csv`, `sample_submission.csv`.
An operator prepares these with the pinned MLE-bench checkout. Raw Kaggle
training data is not interchangeable with the prepared public split.
Do not access original Kaggle labels, private answers, prior runs or submissions.

Feature columns contain missing values; fit imputers and any learned
preprocessing on training rows only. Set random states for stochastic models.
The per-evaluation limit is 300 seconds; the cell's total runtime also includes
agent calls, setup and final refitting. Dependencies belong to this task's uv
project. `prepare.py` is fixed; `train.py` is the candidate implementation.

The supplied impute + one-hot logistic-regression control supports the
`baseline-tune` loop. It is a repository-provided control, not a published
Kaggle baseline.
