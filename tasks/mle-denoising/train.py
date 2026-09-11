"""Provided control: a regularized global affine pixel denoiser."""
import numpy as np


PARAM_SCHEMA = {"alpha": ("float", "log")}
DEFAULT_PARAMS = {"alpha": 1e-3}
BASE_PARAMS = {"alpha": 1e-3}
SEARCH_SPACE = {"alpha": ("float", 1e-6, 10.0, "log")}


class AffineDenoiser:
    def __init__(self, alpha=1e-3):
        self.alpha = float(alpha)

    def fit(self, x, y):
        dirty = np.asarray(x, dtype=np.float64)
        clean = np.asarray(y, dtype=np.float64)
        if dirty.shape != clean.shape or dirty.ndim != 3:
            raise ValueError("denoiser expects paired (n, height, width) arrays")
        x_mean = float(dirty.mean())
        y_mean = float(clean.mean())
        centered_x = dirty - x_mean
        centered_y = clean - y_mean
        denominator = float(np.sum(centered_x * centered_x)) + max(self.alpha, 0.0)
        self.slope_ = float(np.sum(centered_x * centered_y) / denominator)
        self.intercept_ = y_mean - self.slope_ * x_mean
        return self

    def predict(self, x):
        if not hasattr(self, "slope_"):
            raise RuntimeError("fit must be called before predict")
        values = self.slope_ * np.asarray(x, dtype=np.float32) + self.intercept_
        return np.clip(values, 0.0, 1.0)


def make_model(dataset, params):
    config = {**DEFAULT_PARAMS, **params}
    return AffineDenoiser(alpha=config["alpha"])
