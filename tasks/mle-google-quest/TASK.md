# Google QUEST Q&A Labeling

Predict 30 subjective quality ratings for question-answer pairs collected
from StackExchange sites. Each row is one question (title and body) with one
answer to it and its metadata; the targets are rater-aggregated values in
`[0, 1]` such as `question_well_written`, `question_type_reason_explanation`,
`answer_helpful` and `answer_satisfaction`. This is a multi-target problem
scored by the mean over the 30 targets of Spearman's rank correlation between
predictions and labels. Pursue the strongest generalization possible within
the available runtime. The task reserves one CUDA GPU and supports GPU
training as well as classical text models within the declared resource
budget.

## Evaluation Contract

The compatibility adapter uses `make_model(dataset, params)`, which returns an
unfitted estimator implementing `fit(X, Y)` and `predict(X)`. Keep the tuner
contract in `train.py`.

`dataset.x_train` is a pandas frame with one question-answer pair per row.
Its index is the integer `qa_id` and its columns are the ten published input
columns: `question_title`, `question_body`, `question_user_name`,
`question_user_page`, `answer`, `answer_user_name`, `answer_user_page`,
`url`, `category`, `host` (all strings). `dataset.y_train` is an `(n, 30)`
float array with the targets in `prepare.TARGETS` order (the
`sample_submission.csv` column order). `predict(X)` must return one value per
row and target: an `(n, 30)` array in that order, or a DataFrame carrying the
target names as columns. Values must be finite. Spearman correlation is
rank-based, so the scale of the predictions does not affect the score;
the competition brief asks for values in `[0, 1]`. A prediction column that
is constant over the holdout has no defined correlation and counts as 0 for
that column.

`prepare.py` exposes the published inputs for the training, holdout and test
rows alike:

- `prepare.train_frame()` — every public training row, indexed by `qa_id`,
  with the ten input columns followed by the 30 target columns.
- `prepare.test_frame()` — every public test row with the ten input columns.
- `prepare.inputs(frame)`, `prepare.sample_submission()`, `prepare.TARGETS`,
  `prepare.INPUT_COLUMNS`.

Training labels reach candidate code through `dataset.y_train` (the training
partition); the held-out labels are withheld by rule, not by technical
isolation. Do not inspect them from candidate code.

Any representation derived from these inputs is fair use. Read data only
through the objects and loaders `prepare.py` supplies.

`prepare.evaluate_config` owns a fixed 80/20 split (seed 42) of the MLE-bench
**public training rows**, grouped by question (`url`) so that all answers to
one question fall on the same side. Its score is `1 - mean column-wise
Spearman` on the held-out rows, lower-is-better, and is the proxy feedback
used for screening and tuning.

The task-native protocol evaluator, `prepare.evaluate_protocol`, consumes
`protocol_outputs.npz` containing `holdout_ids`, `holdout_predictions`,
`test_ids` and `test_submission`. The two id arrays are integer `qa_id`
values in the task holdout order and in `sample_submission.csv` order; the
two prediction arrays are `(n, 30)` in `prepare.TARGETS` order. The evaluator
scores the holdout predictions and separately validates the test output.
Protocol predictions use the model trained on the training partition; full
public-data refitting happens only after search is frozen. Protocol scores
use the same public holdout and are not official grades. Compare candidates
only within the same stage and fidelity.

The final selected implementation is refit on all public training rows by
`prepare.export_submission`. It writes `qa_id` plus the 30 target columns in
`sample_submission.csv` order. Official MLE-bench grading happens outside the
agent environment after search and submission generation have finished.
Official scores and medal thresholds must not be used for further candidate
selection.

## Data and resources

All MLE tasks use the same `envs/mle` environment, supplying sklearn, scipy,
CUDA-enabled torch/torchvision and transformers. Execution reserves one GPU
through the project device lease, with an 8 GiB free-memory preflight minimum
and a 600 second lease-wait limit. CPU-based models are also allowed; GPU use
is available, not a model-family restriction. Declare additional dependency
needs for installation into this shared environment before execution.

`MLEBENCH_PUBLIC_DATA` points to a read-only copy of the official prepared
public tree. An operator prepares it with the pinned MLE-bench checkout; raw
Kaggle data is not interchangeable with the prepared public split. It contains
three files:

- `train.csv` — 5471 rows, 41 columns: `qa_id`, the ten input columns and the
  30 targets. The rows cover 3392 distinct questions; a question with several
  answers appears once per answer with the same title, body, `url` and
  question-side metadata. `category` takes five values (TECHNOLOGY,
  STACKOVERFLOW, CULTURE, LIFE_ARTS, SCIENCE) and `host` names one of 63
  StackExchange sites. Question bodies and answers are free text with a
  median length of about 550 characters and a 95th percentile near 2300–2500;
  no field is missing. Each target column takes between 3 and 17 distinct
  values; `question_type_spelling` is nonzero in 10 rows.
- `test.csv` — 608 rows, 11 columns (`qa_id` and the ten input columns),
  covering 577 distinct questions.
- `sample_submission.csv` — `qa_id` plus the 30 target columns for the 608
  test rows in `test.csv` order, filled with placeholder values.

The public test split was drawn at random from the original Kaggle training
set by the MLE-bench preparer.

Do not access original Kaggle labels, private answers, prior runs or
submissions.

The dataset is cached in each evaluator process. Fit any learned preprocessing
on training rows only. Set random states for stochastic models. The memory
check is an observation at admission, not a guarantee against other processes
using the device later.

The default per-evaluation limit is 3600 seconds; the run's total runtime also
includes agent calls, setup and final refitting. Dependencies are supplied by
the shared `envs/mle` uv environment. `prepare.py` is evaluator-owned;
`train.py` is the candidate implementation.
