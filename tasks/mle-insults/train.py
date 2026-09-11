"""Provided control: word and character TF-IDF logistic regression."""
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PARAM_SCHEMA = {"C": ("float", "log"), "min_df": ("int",), "ngram_max": ("int",)}
DEFAULT_PARAMS = {"C": 2.0, "min_df": 1, "ngram_max": 2}
BASE_PARAMS = dict(DEFAULT_PARAMS)
SEARCH_SPACE = {
    "C": ("float", 0.05, 20.0, "log"),
    "min_df": ("int", 1, 5),
    "ngram_max": ("int", 1, 3),
}


def make_model(dataset, params):
    config = {**DEFAULT_PARAMS, **params}
    features = ColumnTransformer([
        ("word", TfidfVectorizer(ngram_range=(1, int(config["ngram_max"])),
                                  min_df=int(config["min_df"]), sublinear_tf=True), "Comment"),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                  min_df=int(config["min_df"])), "Comment"),
    ])
    return Pipeline([("features", features),
                     ("model", LogisticRegression(C=float(config["C"]), max_iter=1000))])
