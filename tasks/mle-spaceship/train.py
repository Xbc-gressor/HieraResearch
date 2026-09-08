"""Provided control: impute, one-hot encode, and logistic regression."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ("Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck")
CATEGORICAL = ("HomePlanet", "CryoSleep", "Cabin", "Destination", "VIP")

PARAM_SCHEMA = {"C": ("float", "log"), "impute_num": ("categorical", ["median", "mean"])}
DEFAULT_PARAMS = {"C": 1.0, "impute_num": "median"}
BASE_PARAMS = dict(DEFAULT_PARAMS)
SEARCH_SPACE = {"C": ("float", 0.001, 100.0, "log"), "impute_num": ("categorical", ["median", "mean"])}


def make_model(dataset, params):
    config = {**DEFAULT_PARAMS, **params}
    numeric = make_pipeline(
        SimpleImputer(strategy=str(config["impute_num"])),
        StandardScaler(),
    )
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    return Pipeline([
        ("features", ColumnTransformer([
            ("numeric", numeric, list(NUMERIC)),
            ("categorical", categorical, list(CATEGORICAL)),
        ])),
        ("model", LogisticRegression(C=float(config["C"]), max_iter=2000)),
    ])
