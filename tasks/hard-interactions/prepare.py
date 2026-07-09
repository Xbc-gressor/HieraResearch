"""Fixed data preparation and the single config->score evaluation for
`hard-interactions` — a deliberately HARD synthetic tabular classification task
with large headroom, built to give framework-hyperparameter search real
discriminating power (unlike a noise-capped task that plateaus immediately).

Do not modify during normal experiments.

Design (why it has headroom):
- Targets depend on **high-order feature interactions** — XOR/parity of binarized
  features, sign of pairwise products, and threshold conjunctions — buried among
  many pure-noise features, with only **low label noise** (~3-4%).
- Linear / shallow models cannot represent XOR/parity → stay near ~60%. Only
  candidates that capture interactions (deep GBDT with enough depth, explicit
  polynomial/interaction features, MLPs, well-tuned) approach the ~88-92% ceiling.
- So better search + deeper tuning yields **meaningfully better** scores: the
  outer S-GoT idea search and the inner deep-tune both have room to matter.

Scores follow the framework convention **lower is better**: reports
`neg_mean_test_accuracy = -mean(accuracy)`. `evaluate_config` is the ONE
evaluation surface (warm-start eval + Phase C tuning both call it); there is no
separate official run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
TEST_SIZE = 0.30
METRIC = "neg_mean_test_accuracy"  # negative mean accuracy: lower is better
_LABEL_NOISE = 0.03                # low → high ceiling, large headroom


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray


_TEST_DATA: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_DATASETS_CACHE: list["DatasetSplit"] | None = None


def _linear_plus_products(rng: np.random.Generator, n: int, d: int):
    """Binary. margin = a LINEAR term (marginal signal → models can find the
    active features, naive linear reaches ~70%) + PAIRWISE PRODUCT interactions
    (degree-2; only interaction-aware models capture them → headroom to ~90%)."""
    x = rng.standard_normal((n, d))
    a = x[:, :10]
    linear = 0.9 * a[:, 0] - 0.8 * a[:, 1] + 0.7 * a[:, 2]
    inter = 1.7 * a[:, 3] * a[:, 4] + 1.6 * a[:, 5] * a[:, 6] - 1.5 * a[:, 7] * a[:, 8]
    margin = linear + inter
    y = (margin > np.median(margin)).astype(int)
    return x, y


def _linear_plus_xor(rng: np.random.Generator, n: int, d: int):
    """Binary. linear margin + an XOR-of-2 term (degree-2: learnable by depth>=2
    trees / poly features, NOT by a linear model) + a product term."""
    x = rng.standard_normal((n, d))
    a = x[:, :10]
    xor2 = ((a[:, 0] > 0) ^ (a[:, 1] > 0)).astype(float)
    margin = (0.85 * a[:, 2] - 0.75 * a[:, 3]
              + 2.0 * (xor2 - 0.5) + 1.6 * a[:, 4] * a[:, 5])
    y = (margin > np.median(margin)).astype(int)
    return x, y


def _multiclass_interactions(rng: np.random.Generator, n: int, d: int):
    """3-class. Terciles of a LINEAR term + several PAIRWISE PRODUCTS."""
    x = rng.standard_normal((n, d))
    a = x[:, :12]
    s = (0.7 * a[:, 0] - 0.6 * a[:, 1]
         + 1.5 * a[:, 2] * a[:, 3] + 1.4 * a[:, 4] * a[:, 5]
         - 1.3 * a[:, 6] * a[:, 7] + 1.2 * a[:, 8] * a[:, 9])
    q = np.quantile(s, [1.0 / 3, 2.0 / 3])
    y = np.digitize(s, q).astype(int)  # balanced 3-class
    return x, y


def _make(name, builder, *, n, d, rng_seed) -> DatasetSplit:
    rng = np.random.default_rng(rng_seed)
    x, y = builder(rng, n, d)
    # low label noise (keeps the ceiling high → headroom)
    flip = rng.random(len(y)) < _LABEL_NOISE
    classes = np.unique(y)
    if flip.any():
        y = y.copy()
        y[flip] = rng.choice(classes, size=int(flip.sum()))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    _TEST_DATA[name] = (x_test, y_test)
    return DatasetSplit(name=name, x_train=x_train, y_train=y_train)


def load_datasets() -> list[DatasetSplit]:
    """Three fixed interaction-structured datasets (cached). Many features are
    pure noise (only the first 8-10 are active) so feature selection matters."""
    global _DATASETS_CACHE
    if _DATASETS_CACHE is not None:
        return _DATASETS_CACHE
    _TEST_DATA.clear()
    _DATASETS_CACHE = [
        _make("linear_products", _linear_plus_products, n=3000, d=35, rng_seed=11),
        _make("linear_xor", _linear_plus_xor, n=3000, d=30, rng_seed=23),
        _make("multiclass_inter", _multiclass_interactions, n=3200, d=40, rng_seed=37),
    ]
    return _DATASETS_CACHE


def evaluate_config(make_model, params: dict) -> float:
    """The single `config -> score` evaluation (lower is better). Builds an
    estimator via `make_model(dataset, params)` per dataset, fits on the training
    split, scores on the fixed held-out test split, returns mean negative test
    accuracy. No separate official run — this value IS the candidate's score."""
    datasets = load_datasets()
    scores = []
    for dataset in datasets:
        estimator = make_model(dataset, params)
        estimator.fit(dataset.x_train, dataset.y_train)
        x_test, y_test = _TEST_DATA[dataset.name]
        scores.append(-float(accuracy_score(y_test, estimator.predict(x_test))))
    return float(np.mean(scores))
