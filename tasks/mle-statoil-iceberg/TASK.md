# Statoil/C-CORE Iceberg Classifier Challenge

Decide whether each remotely sensed target in a Sentinel-1 satellite radar
image is an iceberg or a ship. Every target is a 75x75 dual-polarization
(HH and HV) backscatter image in dB with the incidence angle at which it was
captured. This is a binary problem scored by log loss on the predicted
iceberg probability. Pursue the strongest generalization possible within the
available runtime. The task reserves one CUDA GPU and supports GPU training as
well as classical classifiers within the declared resource budget.

## Evaluation Contract

The compatibility adapter uses `make_model(dataset, params)`, which returns an
unfitted estimator implementing `fit(X, y)` and `predict_proba(X)`. Keep the
tuner contract in `train.py`.

`dataset.x_train` is a pandas frame with one image per row. Its index is the
image `id` (string) and its columns are `band_1_0000..band_1_5624`,
`band_2_0000..band_2_5624` (the two bands' pixel values in dB, row-major
75x75, float32) and `inc_angle` (float32, NaN where the published file says
`"na"`), so the frame is numeric and ordinary estimators accept it directly.
`dataset.y_train` is an `(n,)` 0/1 array, 1 marking an iceberg.
`predict_proba(X)` must return one iceberg probability per image: an `(n,)`
or `(n, 1)` array, or the `(n, 2)` layout of binary sklearn classifiers (the
positive column is located through `classes_`). Values must be finite and lie
in `[0, 1]`; official grading rejects anything outside that range.

The index carries the identity a model needs, and `prepare.py` exposes the
published inputs for the training, holdout and test images alike:

- `prepare.bands(frame)` — the pixel columns of any frame this module
  produces as an `(n, 2, 75, 75)` float32 array.
- `prepare.train_frame()`, `prepare.test_frame()` — every public training and
  test image with the columns above, indexed by `id`.
- `prepare.test_records()` — the published test JSON records exactly as
  shipped (`id`, `band_1`, `band_2`, `inc_angle`), parsed on each call.
- `prepare.sample_submission()`.

Training labels reach candidate code only through `dataset.y_train` (the
training partition); the held-out labels are withheld by rule, not by
technical isolation.

Any representation derived from these inputs is fair use. Read data only
through the objects and loaders `prepare.py` supplies.

`prepare.evaluate_config` owns a fixed 80/20 split (seed 42, stratified by
`is_iceberg`) of the MLE-bench **public training images**. Its score is the
binary log loss on the held-out images, lower-is-better, and is the proxy
feedback used for screening and tuning. Do not inspect the held-out labels
from candidate code.

The task-native protocol evaluator, `prepare.evaluate_protocol`, consumes
`protocol_outputs.npz` containing `holdout_ids`, `holdout_predictions`,
`test_ids` and `test_submission`. The two id arrays are image ids as strings
in the task holdout order and in `sample_submission.csv` order; the two
prediction arrays are `(n,)` iceberg probabilities in `[0, 1]`. The evaluator
scores the holdout predictions and separately validates the test output.
Protocol predictions use the model trained on the training partition; full
public-data refitting happens only after search is frozen. Protocol scores
use the same public holdout and are not official grades. Compare candidates
only within the same stage and fidelity.

The final selected implementation is refit on all public training images by
`prepare.export_submission`. It writes `id,is_iceberg` rows in
`sample_submission.csv` order. Official MLE-bench grading happens outside the
agent environment after search and submission generation have finished.
Official scores and medal thresholds must not be used for further candidate
selection.

## Data and resources

All MLE tasks use the same `envs/mle` environment, supplying sklearn,
CUDA-enabled torch/torchvision, transformers, Pillow and py7zr. Execution
reserves one GPU through the project device lease, with an 8 GiB free-memory
preflight minimum and a 600 second lease-wait limit. CPU-based models are also
allowed; GPU use is available, not a model-family restriction. Declare
additional dependency needs for installation into this shared environment
before execution.

`MLEBENCH_PUBLIC_DATA` points to a read-only copy of the official prepared
public tree. An operator prepares it with the pinned MLE-bench checkout; raw
Kaggle data is not interchangeable with the prepared public split. It contains
`description.md` (the competition description) and three 7z archives, each
holding one file:

- `train.json.7z` → `train.json` — a JSON list of 1283 training records with
  fields `id` (8-character hex string), `band_1` and `band_2` (5625 floats
  each: the flattened 75x75 HH and HV backscatter images in dB), `inc_angle`
  (incidence angle in degrees as a float, or the string `"na"` for 105
  records) and `is_iceberg` (1 iceberg, 0 ship; 592 icebergs).
- `test.json.7z` → `test.json` — 321 test records with the same fields except
  `is_iceberg`; 28 of them have `inc_angle` `"na"`.
- `sample_submission.csv.7z` → `sample_submission.csv` — `id,is_iceberg` for
  the 321 test images, every probability set to 0.5.

The public test split was drawn at random from the original Kaggle training
set by the MLE-bench preparer; it contains no machine-generated images.

Do not access original Kaggle labels, private answers, prior runs or
submissions.

The dataset is cached in each evaluator process. Fit any learned preprocessing
on training images only. Set random states for stochastic models. The memory
check is an observation at admission, not a guarantee against other processes
using the device later.

The default per-evaluation limit is 1200 seconds; the run's total runtime also
includes agent calls, setup and final refitting. Dependencies are supplied by
the shared `envs/mle` uv environment. `prepare.py` is evaluator-owned;
`train.py` is the candidate implementation.
