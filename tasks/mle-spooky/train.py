"""Provided control: word TF-IDF with multinomial naive Bayes."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

PARAM_SCHEMA = {"alpha": ("float", "log"), "max_ngram": "int"}
DEFAULT_PARAMS = {"alpha": 0.1, "max_ngram": 2}
BASE_PARAMS = dict(DEFAULT_PARAMS)
SEARCH_SPACE = {"alpha": ("float", 0.01, 2.0, "log"), "max_ngram": ("int", 1, 3)}


def make_model(dataset, params):
    config = {**DEFAULT_PARAMS, **params}
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, int(config["max_ngram"])), sublinear_tf=True),
        MultinomialNB(alpha=float(config["alpha"])),
    )
