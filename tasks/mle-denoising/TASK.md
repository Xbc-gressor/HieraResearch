# Denoising Dirty Documents

Recover clean document images from the public dirty document images. Minimize
pixel RMSE within the available runtime. This task uses CPU; no GPU is required.

## Evaluation Contract

`make_model(dataset, params)` returns an unfitted image regressor implementing
`fit` and `predict`. Inputs and targets are float32 grayscale arrays with shape
`(n, height, width)` and values in `[0, 1]`. The model must return one image per
input with the same shape. Keep the tuner contract in `train.py`.

`prepare.evaluate_config` owns the fixed MLE-bench public split (the official
`train.zip` split uses `test_size=0.2`, `random_state=0`) and scores pixel RMSE
on the held-out clean images. Lower is better, and this is the only feedback
used for screening, tuning, and candidate selection. Candidate code receives no
held-out images or targets.

The final selected implementation is refit on all public dirty/clean training
pairs by `prepare.export_submission`. It emits `id,value` rows for the public
test images in `sampleSubmission.csv` order. Official MLE-bench grading happens
outside the agent environment after search and submission generation.

## Data and resources

`MLEBENCH_PUBLIC_DATA` points to a read-only directory containing only the
prepared public `train/`, `train_cleaned/`, `test/` and `sampleSubmission.csv`.
The public split is the one produced by the pinned MLE-bench preparation script;
raw Kaggle archives and private `answers.csv` must not be accessed. The supplied
affine pixel control in `train.py` supports the `baseline-tune` loop.

Images are decoded and cached by the evaluator. Dependencies belong to this
task's uv project. `prepare.py` is fixed; `train.py` is the candidate
implementation.
