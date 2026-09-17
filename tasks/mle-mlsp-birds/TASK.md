# MLSP 2013 Bird Classification Challenge

Predict which of 19 bird species are audible in each ten-second field
recording from the H. J. Andrews Experimental Forest. Recordings hold several
simultaneously vocalizing birds, insects, rain, wind and other background
noise, and some contain no birds at all. This is a multi-label problem scored
by one ROC AUC over every (recording, species) pair. Pursue the strongest
generalization possible within the available runtime. The task reserves one
CUDA GPU and supports GPU training as well as classical classifiers within the
declared resource budget.

## Evaluation Contract

The compatibility adapter uses `make_model(dataset, params)`, which returns an
unfitted estimator implementing `fit(X, Y)` and `predict_proba(X)`. Keep the
tuner contract in `train.py`.

`dataset.x_train` is a pandas frame with one recording per row. Its index is
the recording's integer `rec_id` and its columns are the recording's mono
16 kHz waveform samples as float32 in `[-1, 1]`, so the frame is numeric and
ordinary estimators accept it directly. `dataset.y_train` is an `(n, 19)`
0/1 array, column `j` marking species `j`. `predict_proba(X)` must return one
probability of presence per recording and species: either an `(n, 19)` array
or the per-species list of `(n, k)` blocks that sklearn multi-output
estimators produce (the positive column is located through `classes_`).
Values must be finite and lie in `[0, 1]`; official grading rejects anything
outside that range.

The index carries the identity a model needs to reach every other published
input from inside `fit` and `predict_proba`, for the training, holdout and
test recordings alike:

- `prepare.load_waveform(rec_id)` / `prepare.wav_path(rec_id)` — the raw audio.
- `prepare.load_spectrogram(rec_id, kind)` / `prepare.spectrogram_path(rec_id, kind)`
  with `kind` in `spectrograms`, `filtered_spectrograms`,
  `supervised_segmentation`, `segmentation_examples` — the published BMP
  images (the last kind exists only for a few training recordings).
- `prepare.segment_features()`, `prepare.segment_rectangles()`,
  `prepare.histogram_of_segments()` — the published segment-level and
  recording-level tables, keyed by `rec_id` (and `segment_id`).
- `prepare.recordings()` (filename, fold and species-`labels` per `rec_id`;
  the `labels` column covers every public training recording, holdout rows
  included — do not inspect held-out labels), `prepare.filename_of(rec_id)`,
  `prepare.species_list()`, `prepare.public_dir()`.

Any representation derived from these inputs is fair use. Read data only
through the objects and loaders `prepare.py` supplies.

`prepare.evaluate_config` owns a fixed 80/20 split (seed 42, stratified by the
number of species in the recording, capped at two) of the MLE-bench **public
training recordings**. Its score is `1 - AUC` on the held-out recordings,
lower-is-better, and is the proxy feedback used for screening and tuning. Do
not inspect the held-out labels from candidate code.

The task-native protocol evaluator, `prepare.evaluate_protocol`, consumes
`protocol_outputs.npz` containing `holdout_ids`, `holdout_predictions`,
`test_ids` and `test_submission`. The two id arrays are integer `rec_id`s in
the task holdout order and in `sample_submission.csv` order; the two prediction
arrays are `(n, 19)` probabilities in `[0, 1]`. The evaluator scores the
holdout predictions and separately validates the test output. Protocol
predictions use the model trained on the training partition; full public-data
refitting happens only after search is frozen. Protocol scores use the same
public holdout and are not official grades. Compare candidates only within the
same stage and fidelity.

The final selected implementation is refit on all public training recordings
by `prepare.export_submission`. It writes `Id,Probability` rows in
`sample_submission.csv` order, where `Id = rec_id * 100 + species`. Official
MLE-bench grading happens outside the agent environment after search and
submission generation have finished. Official scores and medal thresholds must
not be used for further candidate selection.

## Data and resources

All MLE tasks use the same `envs/mle` environment, supplying sklearn,
CUDA-enabled torch/torchvision, transformers and Pillow. Execution reserves one
GPU through the project device lease, with an 8 GiB free-memory preflight
minimum and a 600 second lease-wait limit. CPU-based models are also allowed;
GPU use is available, not a model-family restriction. Declare additional
dependency needs for installation into this shared environment before
execution.

`MLEBENCH_PUBLIC_DATA` points to a read-only copy of the official prepared
public tree. An operator prepares it with the pinned MLE-bench checkout; raw
Kaggle data is not interchangeable with the prepared public split. It contains:

- `sample_submission.csv` — `Id,Probability` for the 64 test recordings
  (19 rows each, 1216 rows).
- `essential_data/CVfolds_2.txt` — `rec_id,fold`; fold 0 marks the 258
  labelled training recordings, fold 1 the 64 unlabelled test recordings.
- `essential_data/rec_id2filename.txt` — `rec_id,filename`; the filename
  encodes the recorder site and timestamp and is the stem shared by the wav and
  every spectrogram of that recording.
- `essential_data/rec_labels_test_hidden.txt` — `rec_id,[labels]`; the
  present species ids for training recordings (a bare `rec_id` means no bird),
  `?` for test recordings.
- `essential_data/species_list.txt` — the 19 species with their ids and codes.
- `essential_data/src_wavs/<filename>.wav` — ten-second mono 16 kHz 16-bit
  recordings (160000 samples each), training and test alike.
- `supplemental_data/spectrograms/`, `filtered_spectrograms/`,
  `supervised_segmentation/` — one 1246x256 BMP per recording (time on x,
  frequency 0 to 8 kHz on y): the grayscale spectrogram, the grayscale
  noise-filtered spectrogram, and the RGB spectrogram with the baseline
  segmentation outlines drawn on it.
- `supplemental_data/segmentation_examples/` — RGB pixel-level annotations for
  20 training recordings (red = bird sound, blue = rain or loud wind).
- `supplemental_data/segment_features.txt` — a 38-dimensional descriptor per
  baseline segment (`rec_id,segment_id,...`).
- `supplemental_data/segment_rectangles.txt` — the bounding box of each
  baseline segment in spectrogram pixels.
- `supplemental_data/histogram_of_segments.txt` — a 100-bin codebook histogram
  per recording built from the segment descriptors (all zeros for recordings
  without segments).
- `supplemental_data/segment_clusters.bmp`, `segment_mosaic.bmp` —
  visualizations of the segment codebook and of all segments.

Do not access original Kaggle labels, private answers, prior runs or
submissions.

The dataset is cached in each evaluator process. Fit any learned preprocessing
on training recordings only. Set random states for stochastic models. The
memory check is an observation at admission, not a guarantee against other
processes using the device later.

The default per-evaluation limit is 900 seconds; the run's total runtime also
includes agent calls, setup and final refitting. Dependencies are supplied by
the shared `envs/mle` uv environment. `prepare.py` is evaluator-owned;
`train.py` is the candidate implementation.
