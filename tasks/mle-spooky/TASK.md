# Spooky Author Identification

Build a strong author classifier for English fiction excerpts. Minimize
multiclass log loss and pursue the strongest generalization possible within
the available runtime. The task reserves one CUDA GPU and supports GPU training
as well as classical text classifiers within the declared resource budget.

## Evaluation Contract

The compatibility adapter uses `make_model(dataset, params)`, which returns an
unfitted estimator implementing `fit`,
`predict_proba` and, after fitting, `classes_`. Inputs are one-dimensional arrays
of text; labels are EAP, HPL, MWS. Keep the tuner contract in `train.py`.

`prepare.evaluate_config` owns a fixed stratified 80/20 split (seed 42) of the
MLE-bench **public training data**. Its validation log loss is lower-is-better
and is the proxy feedback used for screening and tuning.
Do not inspect the held-out validation labels from candidate code.

The task-native protocol evaluator, `prepare.evaluate_protocol`, consumes
`torch_outputs.npz` containing `holdout_ids`, `holdout_predictions`, `test_ids`
and `test_submission`. Predictions must follow the task-provided sample IDs
and class order EAP, HPL, MWS. Both prediction arrays have three probability
columns, and every value must lie in `[0, 1]` as well as sum to one per row;
official grading rejects a negative or above-one entry even when its row sums
correctly. The archive is read with `allow_pickle=False`, so save the two id
arrays as strings rather than as an object array: `np.array(list(ids))` works,
while a bare `frame.id.to_numpy()` on this task's string ids does not reload.
The evaluator scores the holdout predictions and separately validates the test
output. Protocol predictions use the model trained on the training
partition; full public-data refitting happens only after search is frozen.
Protocol scores use the same public holdout and are not official grades.
Compare candidates only within the same stage and fidelity.

The final selected implementation is refit on all public training rows by
`prepare.export_submission`. It emits `id,EAP,HPL,MWS` probabilities for the
public test inputs. Official MLE-bench grading happens outside the agent
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
official prepared public `train.csv`, `test.csv`, `sample_submission.csv`.
An operator prepares these with the pinned MLE-bench checkout. Raw Kaggle
training data is not interchangeable with the prepared public split.
Do not access original Kaggle labels, private answers, prior runs or submissions.

The dataset is cached in each evaluator process. Fit vectorizers and any learned
preprocessing on training text only. Set random states for stochastic models.
The task declares one CUDA device, a free-memory preflight minimum of 8 GiB,
and a lease-wait limit of 600 seconds. GPU execution must use the project's
device lease. The memory check is an observation at admission, not a guarantee
against other processes using the device later.

The default per-evaluation limit is 300 seconds; the run's total runtime also
includes agent calls, setup and final refitting. Dependencies are supplied by the shared `envs/mle` uv environment.
`prepare.py` is evaluator-owned; `train.py` is the candidate implementation.
