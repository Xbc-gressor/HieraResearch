# Detecting Insults in Social Commentary

Predict whether a comment is insulting. The public MLE-bench data contains
the original competition's `Insult`, `Date`, and `Comment` columns. Optimize
generalization to the held-out public validation split under the fixed
evaluation contract below.

## Evaluation Contract

`make_model(dataset, params)` returns an unfitted estimator implementing
`fit`, `predict_proba`, and `classes_`. The dataset's `x_train` is a pandas
frame with `Date` and `Comment` columns; `y_train` is a binary NumPy array.
The positive class is integer `1` (`Insult`). The score is `1 - validation
ROC AUC`, so lower is better. Candidate code must not read files or labels
outside the objects supplied by `prepare.py`.

`prepare.evaluate_config` owns a fixed stratified 80/20 split (seed 42) of
the public training data. `prepare.export_submission` refits the selected
configuration on all public training rows and writes the MLE-bench format:
`Insult,Date,Comment`, preserving the public test row order. The private
answer file is never loaded by this task surface.

## Data and resources

All MLE tasks use the same `envs/mle` environment, supplying sklearn,
CUDA-enabled torch/torchvision and transformers. Execution reserves one GPU
through the project device lease, with an 8 GiB free-memory preflight minimum
and a 600 second lease-wait limit. CPU-based models are also allowed;
GPU use is available, not a model-family restriction. Declare additional
dependency needs for installation into this shared environment before execution.

`MLEBENCH_PUBLIC_DATA` points to a read-only directory containing only the
prepared public `train.csv`, `test.csv`, and `sample_submission_null.csv`.
The test has `Date,Comment`; the sample has `Insult,Date,Comment`. Do not
look for private test labels, raw competition archives, or prior submissions.
This task has a 300 second per-evaluation limit. Dependencies are supplied by the shared `envs/mle` uv environment. `prepare.py` is fixed and read-only; `train.py` is the
candidate implementation.
