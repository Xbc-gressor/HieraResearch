# Spooky Author Identification

Build a strong author classifier for English fiction excerpts. Minimize
multiclass log loss and pursue the strongest generalization possible within
the available runtime. This task uses CPU; no GPU is required.

## Evaluation Contract

`make_model(dataset, params)` returns an unfitted estimator implementing `fit`,
`predict_proba` and, after fitting, `classes_`. Inputs are one-dimensional arrays
of text; labels are EAP, HPL, MWS. Keep the tuner contract in `train.py`.

`prepare.evaluate_config` owns a fixed stratified 80/20 split (seed 42) of the
MLE-bench **public training data**. Its validation log loss is lower-is-better
and is the only feedback used for screening, tuning and candidate selection.
Do not inspect the held-out validation labels from candidate code.

The final selected implementation is refit on all public training rows by
`prepare.export_submission`. It emits `id,EAP,HPL,MWS` probabilities for the
public test inputs. Official MLE-bench grading happens outside the agent
environment after search and submission generation have finished. Official
scores and medal thresholds must not be used for further candidate selection.

## Data and resources

`MLEBENCH_PUBLIC_DATA` points to a read-only directory containing only the
official prepared public `train.csv`, `test.csv`, `sample_submission.csv`.
An operator prepares these with the pinned MLE-bench checkout. Raw Kaggle
training data is not interchangeable with the prepared public split.
Do not access original Kaggle labels, private answers, prior runs or submissions.

The dataset is cached in each evaluator process. Fit vectorizers and any learned
preprocessing on training text only. Set random states for stochastic models.
The per-evaluation limit is 300 seconds; the cell's total runtime also includes
agent calls, setup and final refitting. Dependencies belong to this task's uv
project. `prepare.py` is fixed; `train.py` is the candidate implementation.

The supplied word TF-IDF/naive-Bayes control supports the `baseline-tune` loop.
It is a repository-provided control, not a published Kaggle baseline.
