"""Provided control: raw-pixel logistic regression with strided downsampling."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PARAM_SCHEMA = {"C": ("float", "log"), "stride": ("categorical", [2, 4])}
DEFAULT_PARAMS = {"C": 1.0, "stride": 2}
BASE_PARAMS = dict(DEFAULT_PARAMS)
SEARCH_SPACE = {"C": ("float", 0.0001, 10.0, "log"), "stride": ("categorical", [2, 4])}


class Flatten(BaseEstimator, TransformerMixin):
    """Downsample uint8 (n, 32, 32, 3) images by striding and flatten to rows."""

    def __init__(self, stride=2):
        self.stride = stride

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)[:, :: self.stride, :: self.stride, :] / 255.0
        return x.reshape(len(x), -1)


def make_model(dataset, params):
    config = {**DEFAULT_PARAMS, **params}
    return make_pipeline(
        Flatten(stride=int(config["stride"])),
        StandardScaler(),
        LogisticRegression(C=float(config["C"]), max_iter=1000),
    )
