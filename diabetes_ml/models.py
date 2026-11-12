"""Catálogo de modelos (LR, RF, XGBoost, TabNet) como Pipelines sin fuga + espacios Optuna desde YAML."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diabetes_ml.config import load, seed

BASE_MODELS = ("lr", "rf", "xgb", "tabnet")


class TabNetSK(BaseEstimator, ClassifierMixin):
    """Envoltorio sklearn de TabNet. La parada temprana usa un 10 % agrupado del propio entrenamiento,
    nunca la validación externa ni la prueba (corrige la fuga del pipeline v2)."""

    def __init__(self, n_d=16, n_a=16, n_steps=4, gamma=1.3, lambda_sparse=1e-4, lr=0.02,
                 max_epochs=100, patience=15, batch_size=256, virtual_batch_size=128, seed=42):
        self.n_d, self.n_a, self.n_steps, self.gamma = n_d, n_a, n_steps, gamma
        self.lambda_sparse, self.lr, self.max_epochs, self.patience = lambda_sparse, lr, max_epochs, patience
        self.batch_size, self.virtual_batch_size, self.seed = batch_size, virtual_batch_size, seed

    def fit(self, X, y, groups=None):
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
        groups = np.arange(len(y)) if groups is None else np.asarray(groups)
        tr, es = next(GroupShuffleSplit(1, test_size=0.10, random_state=self.seed).split(X, y, groups))
        self.model_ = TabNetClassifier(n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps, gamma=self.gamma,
                                       lambda_sparse=self.lambda_sparse, seed=self.seed, verbose=0,
                                       optimizer_fn=torch.optim.Adam, optimizer_params={"lr": self.lr})
        bs = int(min(self.batch_size, max(32, len(tr) // 4)))
        self.model_.fit(X[tr], y[tr], eval_set=[(X[es], y[es])], eval_metric=["auc"], max_epochs=self.max_epochs,
                        patience=self.patience, batch_size=bs, virtual_batch_size=int(min(self.virtual_batch_size, bs)))
        self.classes_ = np.array([0, 1])
        self.history_ = {k: [float(v) for v in vals] for k, vals in self.model_.history.history.items() if k in ("loss", "val_0_auc")}
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def suggest(trial, space: Dict[str, Any]) -> Dict[str, Any]:
    p = {}
    for name, s in space.items():
        if s["type"] == "float":
            p[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
        elif s["type"] == "int":
            p[name] = trial.suggest_int(name, s["low"], s["high"], step=s.get("step", 1))
        else:
            p[name] = trial.suggest_categorical(name, s["choices"])
    return p


def default_params(name: str) -> Dict[str, Any]:
    return {"lr": {"C": 1.0, "class_weight": "balanced"},
            "rf": {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5, "max_features": "sqrt"},
            "xgb": {"learning_rate": 0.05, "max_depth": 4, "n_estimators": 300, "subsample": 0.8, "colsample_bytree": 0.8},
            "tabnet": dict(load("search_spaces")["tabnet_fijo"])}[name]


def build(name: str, params: Dict[str, Any], pos_weight: float = 1.0) -> Pipeline:
    s = seed()
    imp = ("imp", SimpleImputer(strategy="median"))
    if name == "lr":
        cw = params.get("class_weight", "balanced")
        clf = LogisticRegression(C=params["C"], class_weight=None if cw == "none" else cw, max_iter=5000, random_state=s)
        return Pipeline([imp, ("sc", StandardScaler()), ("clf", clf)])
    if name == "rf":
        mf = params["max_features"]
        mf = float(mf) if isinstance(mf, str) and mf.replace(".", "").isdigit() else mf
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                     min_samples_leaf=params["min_samples_leaf"], max_features=mf,
                                     class_weight="balanced_subsample", random_state=s, n_jobs=-1)
        return Pipeline([imp, ("clf", clf)])
    if name == "xgb":
        import xgboost as xgb
        clf = xgb.XGBClassifier(**params, scale_pos_weight=pos_weight, tree_method="hist", eval_metric="logloss",
                                random_state=s, n_jobs=-1)
        return Pipeline([imp, ("clf", clf)])
    if name == "tabnet":
        params = {**default_params("tabnet"), **params}
        if "n_d" in params:
            params["n_a"] = params["n_d"]
        return Pipeline([imp, ("sc", StandardScaler()), ("clf", TabNetSK(**params, seed=s))])
    raise ValueError(name)


def fit(name: str, params: Dict[str, Any], X, y, groups) -> Pipeline:
    y = np.asarray(y)
    pw = float((y == 0).sum() / max(1, (y == 1).sum()))
    pipe = build(name, params, pw)
    if name == "tabnet":
        pipe.fit(X, y, clf__groups=groups)
    else:
        pipe.fit(X, y)
    return pipe
