# Aerial Cactus Identification

Predict whether each 32x32 aerial image contains a columnar cactus. Pursue the
strongest generalization possible within the available runtime. One CUDA GPU is available; choose the model and algorithm that best use the budget.

## Evaluation Contract

`make_model(dataset, params)` returns an unfitted estimator implementing `fit`,
`predict_proba` and, after fitting, `classes_`. Inputs are uint8 arrays of shape
`(n, 32, 32, 3)`; labels are binary `has_cactus` values 0/1 and the scored
output is the predicted probability of class 1. Keep the tuner contract in
`train.py`.

`prepare.evaluate_config` owns a fixed stratified 80/20 split (seed 42) of the
MLE-bench **public training data**. The competition metric is ROC AUC; the
score returned is `1 - validation AUC`, so lower is better, and it is the only
feedback used for screening, tuning and candidate selection. Do not inspect the
held-out validation labels from candidate code.

The final selected implementation is refit on all public training rows by
`prepare.export_submission`. It emits `id,has_cactus` probabilities for the
public test images. Official MLE-bench grading happens outside the agent
environment after search and submission generation have finished. Official
scores and medal thresholds must not be used for further candidate selection.

## Data and resources

All MLE tasks use the same `envs/mle` environment, supplying sklearn,
CUDA-enabled torch/torchvision and transformers. Execution reserves one GPU
through the project device lease, with an 8 GiB free-memory preflight minimum
and a 600 second lease-wait limit. CPU-based models are also allowed;
GPU use is available, not a model-family restriction. Declare additional
dependency needs for installation into this shared environment before execution.

`MLEBENCH_PUBLIC_DATA` points to a read-only directory containing only the
official prepared public `train.csv`, `train.zip`, `test.zip` and
`sample_submission.csv`. The zip archives hold bare 32x32 JPEG filenames; the
test image IDs come from `sample_submission.csv`. An operator prepares these
with the pinned MLE-bench checkout. Raw Kaggle competition files are not
interchangeable with the prepared public split. Do not access original Kaggle
labels, private answers, prior runs or submissions.

Images are decoded once and cached in each evaluator process. Fit any learned
preprocessing on training images only. Set random states for stochastic models.
The per-evaluation limit is 300 seconds; the cell's total runtime also includes
agent calls, setup and final refitting. Dependencies are supplied by the shared `envs/mle` uv environment. `prepare.py` is fixed; `train.py` is the candidate implementation.

The supplied raw-pixel logistic-regression control supports the
`baseline-tune` loop. It is a repository-provided control, not a published
Kaggle baseline.
