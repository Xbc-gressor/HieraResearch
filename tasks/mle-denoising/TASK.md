# Denoising Dirty Documents

Recover clean document images from the public dirty document images. Minimize
pixel RMSE within the available runtime. One CUDA GPU is available; choose the model and algorithm that best use the budget.

## Evaluation Contract

`make_model(dataset, params)` returns an unfitted image regressor implementing
`fit` and `predict`. Inputs and targets are lists of float32 grayscale arrays
with shape `(height, width)` and values in `[0, 1]`, one array per document
page. The public split mixes page sizes, so the images are not a single stacked
array and a model must not assume one shape across the batch. `predict` takes
such a list and returns one image per input with that input's own shape. Keep
the tuner contract in `train.py`.

`prepare.evaluate_config` re-splits the prepared public training pages 80/20
(seed 0) and scores pixel RMSE on its held-out clean pages. That holdout is
this project's proxy, not the official MLE-bench test split, whose answers stay
private. Lower is better, and it is the only feedback used for screening,
tuning, and candidate selection. Candidate code receives no held-out images or
targets.

The final selected implementation is refit on all public dirty/clean training
pairs by `prepare.export_submission`. It emits `id,value` rows for the public
test images in `sampleSubmission.csv` order. Official MLE-bench grading happens
outside the agent environment after search and submission generation.

## Data and resources

All MLE tasks use the same `envs/mle` environment, supplying sklearn,
CUDA-enabled torch/torchvision and transformers. Execution reserves one GPU
through the project device lease, with an 8 GiB free-memory preflight minimum
and a 600 second lease-wait limit. CPU-based models are also allowed;
GPU use is available, not a model-family restriction. Declare additional
dependency needs for installation into this shared environment before execution.

`MLEBENCH_PUBLIC_DATA` points to a read-only directory containing only the
prepared public `train/`, `train_cleaned/`, `test/` and `sampleSubmission.csv`.
The public split is the one produced by the pinned MLE-bench preparation script;
raw Kaggle archives and private `answers.csv` must not be accessed.

Images are decoded and cached by the evaluator. Dependencies are supplied by
the shared `envs/mle` environment. `prepare.py` is fixed; `train.py` is the candidate
implementation.
