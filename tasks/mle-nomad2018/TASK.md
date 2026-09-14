# NOMAD 2018 Predicting Transparent Conductors

Predict `formation_energy_ev_natom` and `bandgap_energy_ev` for materials
from their tabular descriptors. The public MLE-bench preparation also keeps
geometry files for the official task; the fixed baseline uses the tabular
columns so evaluation remains fast and reproducible.

## Evaluation Contract

`make_model(dataset, params)` returns an unfitted estimator implementing
`fit` and `predict`. `dataset.x_train` is a pandas frame containing the
public tabular feature columns (with the two targets removed), and
`dataset.y_train` is a two-column NumPy array in target order
`formation_energy_ev_natom`, `bandgap_energy_ev`. The score is the mean of
the two column RMSLE values, so lower is better. Candidate code must not read
files or labels outside the objects supplied by `prepare.py`.

`prepare.evaluate_config` owns a fixed 80/20 split (seed 42) of the prepared
public training data. `prepare.export_submission` refits on all public rows
and writes `id,formation_energy_ev_natom,bandgap_energy_ev` in public test
order. Private labels and raw competition archives are never loaded.

## Data and resources

All MLE tasks use the same `envs/mle` environment, supplying sklearn,
CUDA-enabled torch/torchvision and transformers. Execution reserves one GPU
through the project device lease, with an 8 GiB free-memory preflight minimum
and a 600 second lease-wait limit. CPU-based models are also allowed;
GPU use is available, not a model-family restriction. Declare additional
dependency needs for installation into this shared environment before execution.

`MLEBENCH_PUBLIC_DATA` points to a read-only directory containing prepared
public `train.csv`, `test.csv`, `sample_submission.csv`, and the public
geometry directories. Training and test IDs are the per-split IDs created by
the official MLE-bench preparation. This task has a 300 second
per-evaluation limit. Dependencies are supplied by the shared `envs/mle` uv environment;
`prepare.py` is fixed and read-only and `train.py` is the candidate file.

The supplied regularized linear control supports the `baseline-tune` loop.
