# NOMAD 2018 Predicting Transparent Conductors

Predict `formation_energy_ev_natom` and `bandgap_energy_ev` for
$(Al_x Ga_y In_z)_{2N}O_{3N}$ materials. Each material is described by
published tabular descriptors and by the atomic structure of its unit cell.
Pursue the strongest generalization possible within the available runtime.

## Evaluation Contract

`make_model(dataset, params)` returns an unfitted estimator implementing
`fit` and `predict`. `dataset.x_train` is a pandas frame of the numeric
tabular feature columns (targets removed), and `dataset.y_train` is a
two-column NumPy array in target order `formation_energy_ev_natom`,
`bandgap_energy_ev`. Predictions are two non-negative values per row. The
score is the mean of the two column RMSLE values, so lower is better.

Each row's frame index is its geometry key, `<split>/<id>`. Passing that
index (or any key from it) to `prepare.load_geometry` / `prepare.load_geometries`
returns that material's `Geometry`: `lattice` as the 3x3 lattice vectors in
angstroms, `positions` as the per-atom cartesian coordinates in angstroms, and
`elements` as the per-atom symbols over Al, Ga, In and O. Cells hold between
10 and 80 atoms. The tabular columns summarize the cell; these files describe
it fully, and any descriptor derived from them is fair use. Estimators
receiving `x_train` see numeric columns only, so a model that wants structural
features looks them up by index inside `fit` and `predict` and builds its own
representation. Read data only through the objects and loaders `prepare.py`
supplies.

`prepare.evaluate_config` owns a fixed 80/20 split (seed 42) of the prepared
public training data; both partitions of it are indexed under `train/`.
`prepare.export_submission` refits on all public rows and writes
`id,formation_energy_ev_natom,bandgap_energy_ev` in public test order.
Private labels and raw competition archives are never loaded.

## Data and resources

All MLE tasks use the same `envs/mle` environment, supplying sklearn,
CUDA-enabled torch/torchvision and transformers. Execution reserves one GPU
through the project device lease, with an 8 GiB free-memory preflight minimum
and a 600 second lease-wait limit. CPU-based models are also allowed;
GPU use is available, not a model-family restriction. Declare additional
dependency needs for installation into this shared environment before execution.

`MLEBENCH_PUBLIC_DATA` points to a read-only directory containing prepared
public `train.csv`, `test.csv`, `sample_submission.csv`, and the
`{train,test}/<id>/geometry.xyz` structure files. Training and test IDs are
the per-split IDs created by the official MLE-bench preparation, so the same
integer appears in both splits and is only a submission key.

<!-- runtime-budget:start -->
This task has a
300 second per-evaluation limit.
<!-- runtime-budget:end -->

Dependencies are supplied by the shared
`envs/mle` uv environment; `prepare.py` is fixed and read-only and `train.py`
is the candidate file.
