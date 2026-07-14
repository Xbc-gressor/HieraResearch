"""Phase-1 HPO-benchmark candidate setup.

Writes contract-compliant candidate dirs under
runs/hard-interactions/hpo-bench/candidates/<label>/ (train.py + prepare.py +
_warm_configs.json) spanning the benchmark axes: dimensionality (low/mid/high),
HPO room, sensitivity. 004/005 (the known anchors) are copied in as M1/M2.

After this, run warmstart_eval.py on each to produce tune_report.json (phase_a:
warm scores + the BO priors the benchmark needs).

Run:  uv --directory tasks/hard-interactions run python <abs path>/tools/hpo_benchmark/setup_candidates.py
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "tasks" / "hard-interactions"
BENCH = ROOT / "runs" / "hard-interactions" / "hpo-bench" / "candidates"
SRC_RUN = ROOT / "runs" / "hard-interactions" / "p3-hpotest" / "candidates"

# ---- shared train.py preamble ----
HEAD = '''"""HPO-benchmark candidate: {label} ({note})."""
from __future__ import annotations
import numpy as np
from prepare import load_datasets  # noqa: F401

CANDIDATE_NAME = "{label}"
# BASE_PARAMS is created by warmstart_eval (create-mode, from SEARCH_SPACE keys).
'''

# ---- per-candidate (note, body with make_model/PARAM_SCHEMA/SEARCH_SPACE, warm configs) ----
CANDS: dict[str, dict] = {}

# L1-svm: 2 dims, sensitive
CANDS["L1-svm"] = dict(note="SVM-RBF, 2d, sensitive", body='''
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def make_model(dataset, params):
    return make_pipeline(StandardScaler(),
        SVC(C=params.get("C", 1.0), gamma=params.get("gamma", 0.05),
            kernel="rbf", cache_size=500))

PARAM_SCHEMA = {"C": "float", "gamma": "float"}
SEARCH_SPACE = {"C": ("float", 0.1, 100.0, "log"),
                "gamma": ("float", 1e-4, 1.0, "log")}
''', warm=[
    {"C": 1.0, "gamma": 0.05}, {"C": 10.0, "gamma": 0.01},
    {"C": 0.3, "gamma": 0.2}, {"C": 50.0, "gamma": 0.002}, {"C": 3.0, "gamma": 0.5},
])

# L2-rf: 2 dims, robust
CANDS["L2-rf"] = dict(note="RandomForest, 2d, robust", body='''
from sklearn.ensemble import RandomForestClassifier

def make_model(dataset, params):
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 12),
        n_jobs=-1, random_state=0)

PARAM_SCHEMA = {"n_estimators": "int", "max_depth": "int"}
SEARCH_SPACE = {"n_estimators": ("int", 100, 600),
                "max_depth": ("int", 3, 20)}
''', warm=[
    {"n_estimators": 300, "max_depth": 12}, {"n_estimators": 600, "max_depth": 20},
    {"n_estimators": 150, "max_depth": 5}, {"n_estimators": 500, "max_depth": 8},
    {"n_estimators": 200, "max_depth": 16},
])

# M3-mlp: ~8 dims, sensitive
CANDS["M3-mlp"] = dict(note="MLP, 8d, sensitive", body='''
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def make_model(dataset, params):
    h1 = int(params.get("hidden1", 128)); h2 = int(params.get("hidden2", 64))
    layers = (h1,) if h2 <= 0 else (h1, h2)
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=layers,
        alpha=params.get("alpha", 1e-4),
        learning_rate_init=params.get("learning_rate_init", 1e-3),
        batch_size=int(params.get("batch_size", 128)),
        beta_1=params.get("beta_1", 0.9),
        beta_2=params.get("beta_2", 0.999),
        max_iter=int(params.get("max_iter", 200)),
        early_stopping=True, random_state=0))

PARAM_SCHEMA = {k: "int" for k in ["hidden1","hidden2","batch_size","max_iter"]}
PARAM_SCHEMA.update({k: "float" for k in ["alpha","learning_rate_init","beta_1","beta_2"]})
SEARCH_SPACE = {
    "hidden1": ("int", 32, 256), "hidden2": ("int", 0, 128),
    "alpha": ("float", 1e-6, 1e-1, "log"),
    "learning_rate_init": ("float", 1e-4, 1e-1, "log"),
    "batch_size": ("int", 32, 256), "max_iter": ("int", 100, 400),
    "beta_1": ("float", 0.8, 0.99), "beta_2": ("float", 0.9, 0.9999),
}
''', warm=[
    {"hidden1":128,"hidden2":64,"alpha":1e-4,"learning_rate_init":1e-3,"batch_size":128,"max_iter":200,"beta_1":0.9,"beta_2":0.999},
    {"hidden1":256,"hidden2":128,"alpha":1e-5,"learning_rate_init":3e-3,"batch_size":64,"max_iter":300,"beta_1":0.85,"beta_2":0.999},
    {"hidden1":64,"hidden2":0,"alpha":1e-2,"learning_rate_init":1e-2,"batch_size":256,"max_iter":150,"beta_1":0.9,"beta_2":0.99},
    {"hidden1":200,"hidden2":32,"alpha":1e-3,"learning_rate_init":5e-4,"batch_size":96,"max_iter":250,"beta_1":0.95,"beta_2":0.9999},
    {"hidden1":96,"hidden2":96,"alpha":1e-6,"learning_rate_init":2e-2,"batch_size":160,"max_iter":200,"beta_1":0.8,"beta_2":0.995},
])

# M4-xgb: ~8 dims, robust
CANDS["M4-xgb"] = dict(note="XGBoost full, 8d, robust", body='''
from xgboost import XGBClassifier

def make_model(dataset, params):
    return XGBClassifier(
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=params.get("learning_rate", 0.1),
        n_estimators=int(params.get("n_estimators", 400)),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_lambda=params.get("reg_lambda", 1.0),
        reg_alpha=params.get("reg_alpha", 0.0),
        min_child_weight=params.get("min_child_weight", 1.0),
        tree_method="hist", n_jobs=-1, random_state=0, eval_metric="logloss")

PARAM_SCHEMA = {"max_depth":"int","n_estimators":"int"}
PARAM_SCHEMA.update({k:"float" for k in ["learning_rate","subsample","colsample_bytree","reg_lambda","reg_alpha","min_child_weight"]})
SEARCH_SPACE = {
    "max_depth": ("int", 3, 10), "n_estimators": ("int", 200, 800),
    "learning_rate": ("float", 0.01, 0.3, "log"),
    "subsample": ("float", 0.6, 1.0), "colsample_bytree": ("float", 0.5, 1.0),
    "reg_lambda": ("float", 0.1, 10.0, "log"), "reg_alpha": ("float", 1e-3, 5.0, "log"),
    "min_child_weight": ("float", 0.5, 10.0, "log"),
}
''', warm=[
    {"max_depth":6,"n_estimators":400,"learning_rate":0.1,"subsample":0.9,"colsample_bytree":0.8,"reg_lambda":1.0,"reg_alpha":0.01,"min_child_weight":1.0},
    {"max_depth":8,"n_estimators":600,"learning_rate":0.05,"subsample":0.8,"colsample_bytree":0.6,"reg_lambda":3.0,"reg_alpha":0.1,"min_child_weight":3.0},
    {"max_depth":4,"n_estimators":300,"learning_rate":0.2,"subsample":1.0,"colsample_bytree":1.0,"reg_lambda":0.3,"reg_alpha":0.001,"min_child_weight":0.5},
    {"max_depth":10,"n_estimators":800,"learning_rate":0.02,"subsample":0.7,"colsample_bytree":0.7,"reg_lambda":5.0,"reg_alpha":1.0,"min_child_weight":5.0},
    {"max_depth":5,"n_estimators":500,"learning_rate":0.08,"subsample":0.85,"colsample_bytree":0.9,"reg_lambda":0.5,"reg_alpha":0.05,"min_child_weight":2.0},
])

# H1-bigstack: >=16 dims, blended ensemble
CANDS["H1-bigstack"] = dict(note="big blended ensemble, 16d", body='''
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np
from xgboost import XGBClassifier

class Blend(ClassifierMixin, BaseEstimator):
    def __init__(self, p): self.p = p
    def fit(self, X, y):
        self.classes_ = np.unique(y); p = self.p
        self.xgb = XGBClassifier(max_depth=int(p.get("xgb_max_depth",6)), learning_rate=p.get("xgb_lr",0.1),
            n_estimators=int(p.get("xgb_n",400)), subsample=p.get("xgb_subsample",0.9),
            colsample_bytree=p.get("xgb_colsample",0.8), reg_lambda=p.get("xgb_reg_lambda",1.0),
            tree_method="hist", n_jobs=-1, random_state=0, eval_metric="logloss").fit(X,y)
        self.et = ExtraTreesClassifier(n_estimators=int(p.get("et_n",300)), max_depth=int(p.get("et_max_depth",16)),
            max_features=p.get("et_max_features",0.5), n_jobs=-1, random_state=0).fit(X,y)
        self.hgb = HistGradientBoostingClassifier(max_leaf_nodes=int(p.get("hgb_leaves",31)),
            learning_rate=p.get("hgb_lr",0.1), max_iter=int(p.get("hgb_iter",300)), random_state=0).fit(X,y)
        self.lr = LogisticRegression(C=p.get("lr_C",1.0), max_iter=2000).fit(X,y)
        return self
    def predict_proba(self, X):
        p = self.p
        w = np.array([max(p.get("w_xgb",1.0),0), max(p.get("w_et",1.0),0),
                      max(p.get("w_hgb",1.0),0), max(p.get("w_lr",0.5),0)]); w = w/w.sum()
        return (w[0]*self.xgb.predict_proba(X) + w[1]*self.et.predict_proba(X)
                + w[2]*self.hgb.predict_proba(X) + w[3]*self.lr.predict_proba(X))
    def predict(self, X): return self.classes_[self.predict_proba(X).argmax(1)]

def make_model(dataset, params):
    return Blend(dict(params))

PARAM_SCHEMA = {k:"int" for k in ["xgb_max_depth","xgb_n","et_n","et_max_depth","hgb_leaves","hgb_iter"]}
PARAM_SCHEMA.update({k:"float" for k in ["xgb_lr","xgb_subsample","xgb_colsample","xgb_reg_lambda","et_max_features","hgb_lr","lr_C","w_xgb","w_et","w_hgb","w_lr"]})
SEARCH_SPACE = {
    "xgb_max_depth": ("int",3,10), "xgb_n": ("int",200,700), "xgb_lr": ("float",0.01,0.3,"log"),
    "xgb_subsample": ("float",0.6,1.0), "xgb_colsample": ("float",0.5,1.0), "xgb_reg_lambda": ("float",0.1,10.0,"log"),
    "et_n": ("int",100,500), "et_max_depth": ("int",4,24), "et_max_features": ("float",0.2,1.0),
    "hgb_leaves": ("int",15,127), "hgb_lr": ("float",0.02,0.3,"log"), "hgb_iter": ("int",150,500),
    "lr_C": ("float",0.05,50.0,"log"),
    "w_xgb": ("float",0.0,1.0), "w_et": ("float",0.0,1.0), "w_hgb": ("float",0.0,1.0), "w_lr": ("float",0.0,1.0),
}
''', warm=[
    {"xgb_max_depth":6,"xgb_n":400,"xgb_lr":0.1,"xgb_subsample":0.9,"xgb_colsample":0.8,"xgb_reg_lambda":1.0,"et_n":300,"et_max_depth":16,"et_max_features":0.5,"hgb_leaves":31,"hgb_lr":0.1,"hgb_iter":300,"lr_C":1.0,"w_xgb":0.4,"w_et":0.2,"w_hgb":0.3,"w_lr":0.1},
    {"xgb_max_depth":8,"xgb_n":600,"xgb_lr":0.05,"xgb_subsample":0.8,"xgb_colsample":0.6,"xgb_reg_lambda":3.0,"et_n":400,"et_max_depth":20,"et_max_features":0.7,"hgb_leaves":63,"hgb_lr":0.05,"hgb_iter":400,"lr_C":3.0,"w_xgb":0.5,"w_et":0.2,"w_hgb":0.2,"w_lr":0.1},
    {"xgb_max_depth":4,"xgb_n":300,"xgb_lr":0.2,"xgb_subsample":1.0,"xgb_colsample":1.0,"xgb_reg_lambda":0.3,"et_n":200,"et_max_depth":10,"et_max_features":0.3,"hgb_leaves":15,"hgb_lr":0.2,"hgb_iter":200,"lr_C":0.3,"w_xgb":0.3,"w_et":0.3,"w_hgb":0.3,"w_lr":0.1},
    {"xgb_max_depth":10,"xgb_n":700,"xgb_lr":0.02,"xgb_subsample":0.7,"xgb_colsample":0.7,"xgb_reg_lambda":5.0,"et_n":500,"et_max_depth":24,"et_max_features":1.0,"hgb_leaves":127,"hgb_lr":0.03,"hgb_iter":500,"lr_C":10.0,"w_xgb":0.6,"w_et":0.15,"w_hgb":0.2,"w_lr":0.05},
    {"xgb_max_depth":5,"xgb_n":500,"xgb_lr":0.08,"xgb_subsample":0.85,"xgb_colsample":0.9,"xgb_reg_lambda":0.5,"et_n":350,"et_max_depth":14,"et_max_features":0.6,"hgb_leaves":47,"hgb_lr":0.08,"hgb_iter":350,"lr_C":0.5,"w_xgb":0.35,"w_et":0.25,"w_hgb":0.25,"w_lr":0.15},
])

# H2-deepmlp: >=16 dims, sensitive
CANDS["H2-deepmlp"] = dict(note="deep MLP many HPs, 16d, sensitive", body='''
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def make_model(dataset, params):
    p = params
    h = tuple(int(x) for x in (p.get("h1",128), p.get("h2",64), p.get("h3",0)) if int(x) > 0) or (64,)
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=h,
        activation=p.get("activation","relu"),
        solver="adam",
        alpha=p.get("alpha",1e-4),
        learning_rate_init=p.get("learning_rate_init",1e-3),
        learning_rate=p.get("learning_rate","constant"),
        power_t=p.get("power_t",0.5),
        batch_size=int(p.get("batch_size",128)),
        beta_1=p.get("beta_1",0.9), beta_2=p.get("beta_2",0.999),
        epsilon=p.get("epsilon",1e-8),
        max_iter=int(p.get("max_iter",300)),
        n_iter_no_change=int(p.get("n_iter_no_change",10)),
        tol=p.get("tol",1e-4),
        validation_fraction=p.get("validation_fraction",0.1),
        early_stopping=True, random_state=0))

PARAM_SCHEMA = {k:"int" for k in ["h1","h2","h3","batch_size","max_iter","n_iter_no_change"]}
PARAM_SCHEMA.update({k:"float" for k in ["alpha","learning_rate_init","power_t","beta_1","beta_2","epsilon","tol","validation_fraction"]})
PARAM_SCHEMA.update({k:"categorical" for k in ["activation","learning_rate"]})
SEARCH_SPACE = {
    "h1": ("int",32,256), "h2": ("int",0,160), "h3": ("int",0,128),
    "alpha": ("float",1e-6,1e-1,"log"), "learning_rate_init": ("float",1e-4,1e-1,"log"),
    "power_t": ("float",0.1,0.9), "batch_size": ("int",32,256),
    "beta_1": ("float",0.8,0.99), "beta_2": ("float",0.9,0.9999), "epsilon": ("float",1e-9,1e-6,"log"),
    "max_iter": ("int",150,500), "n_iter_no_change": ("int",5,30),
    "tol": ("float",1e-5,1e-3,"log"), "validation_fraction": ("float",0.05,0.2),
    "activation": ("categorical",["relu","tanh"]), "learning_rate": ("categorical",["constant","adaptive"]),
}
''', warm=[
    {"h1":128,"h2":64,"h3":0,"alpha":1e-4,"learning_rate_init":1e-3,"power_t":0.5,"batch_size":128,"beta_1":0.9,"beta_2":0.999,"epsilon":1e-8,"max_iter":300,"n_iter_no_change":10,"tol":1e-4,"validation_fraction":0.1,"activation":"relu","learning_rate":"constant"},
    {"h1":256,"h2":128,"h3":64,"alpha":1e-5,"learning_rate_init":3e-3,"power_t":0.5,"batch_size":64,"beta_1":0.85,"beta_2":0.999,"epsilon":1e-8,"max_iter":400,"n_iter_no_change":15,"tol":1e-4,"validation_fraction":0.1,"activation":"relu","learning_rate":"adaptive"},
    {"h1":64,"h2":0,"h3":0,"alpha":1e-2,"learning_rate_init":1e-2,"power_t":0.3,"batch_size":256,"beta_1":0.9,"beta_2":0.99,"epsilon":1e-7,"max_iter":200,"n_iter_no_change":8,"tol":1e-3,"validation_fraction":0.15,"activation":"tanh","learning_rate":"constant"},
    {"h1":200,"h2":96,"h3":32,"alpha":1e-3,"learning_rate_init":5e-4,"power_t":0.7,"batch_size":96,"beta_1":0.95,"beta_2":0.9999,"epsilon":1e-8,"max_iter":350,"n_iter_no_change":20,"tol":1e-4,"validation_fraction":0.1,"activation":"relu","learning_rate":"adaptive"},
    {"h1":96,"h2":96,"h3":48,"alpha":1e-6,"learning_rate_init":2e-2,"power_t":0.5,"batch_size":160,"beta_1":0.8,"beta_2":0.995,"epsilon":1e-9,"max_iter":250,"n_iter_no_change":12,"tol":1e-5,"validation_fraction":0.1,"activation":"tanh","learning_rate":"adaptive"},
])


def main() -> int:
    BENCH.mkdir(parents=True, exist_ok=True)
    prepare_src = TASK / "prepare.py"
    for label, spec in CANDS.items():
        d = BENCH / label
        d.mkdir(parents=True, exist_ok=True)
        src = HEAD.format(label=label, note=spec["note"]) + spec["body"]
        (d / "train.py").write_text(src)
        shutil.copy(prepare_src, d / "prepare.py")
        (d / "_warm_configs.json").write_text(json.dumps(spec["warm"], indent=2))
        print(f"wrote {label}: {len(spec['warm'])} warm configs, "
              f"{len(spec['warm'][0])} dims")
    # copy anchors 004->M1-gbdt, 005->M2-stack
    for src_id, label in [("004", "M1-gbdt"), ("005", "M2-stack")]:
        s = SRC_RUN / src_id
        d = BENCH / label
        if s.is_dir():
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d)
            print(f"copied anchor {src_id} -> {label}")
        else:
            print(f"WARN anchor {src_id} missing at {s}")
    print(f"\ncandidates at {BENCH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
