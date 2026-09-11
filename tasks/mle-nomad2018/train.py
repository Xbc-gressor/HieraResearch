"""Provided control: imputation, scaling, and multi-output ridge regression."""
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

PARAM_SCHEMA = {"alpha": ("float", "log")}
DEFAULT_PARAMS = {"alpha": 1.0}
BASE_PARAMS = dict(DEFAULT_PARAMS)
SEARCH_SPACE = {"alpha": ("float", 1e-4, 1e4, "log")}


def make_model(dataset, params):
    config = {**DEFAULT_PARAMS, **params}
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MultiOutputRegressor(Ridge(alpha=float(config["alpha"]))),
    )
