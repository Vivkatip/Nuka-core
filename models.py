"""
Модели. Все предсказания возвращаются в исходной шкале SI.
Для моделей с log-таргетом обратное преобразование expm1 делается внутри.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

MODELS = {
    "mean":     {"kind": "mean"},
    "rf_cap":   {"kind": "rf", "cap": 0.95},   # хвост обрезан только в обучении
    "rf_log":   {"kind": "rf", "log": True},   # обучение на log1p(SI)
    "et_log":   {"kind": "et", "log": True},
    "hurdle_5": {"kind": "hurdle", "thr": 5.0},  # классификатор SI>5 * регрессор активных
}
ENSEMBLE = ["rf_cap", "rf_log", "et_log", "hurdle_5"]


def impute(X_tr, X_va):
    """Медианы считаются только по обучающей части."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isnan(X_tr), med, X_tr), np.where(np.isnan(X_va), med, X_va)


def _regressor(kind, seed, n_trees, leaf):
    if kind == "et":
        return ExtraTreesRegressor(
            n_estimators=n_trees, min_samples_leaf=2, max_features=0.5,
            random_state=seed, n_jobs=-1)
    return RandomForestRegressor(
        n_estimators=n_trees, min_samples_leaf=leaf, max_features=0.3,
        random_state=seed, n_jobs=-1)


def fit_predict(name, X_tr, y_tr, X_va, seed=0, n_trees=500):
    spec = MODELS[name]
    n_va = len(X_va)
    if spec["kind"] == "mean":
        return np.full(n_va, float(np.mean(y_tr)))

    Xt, Xv = impute(X_tr, X_va)

    if spec["kind"] == "hurdle":
        act = y_tr > spec["thr"]
        if act.sum() < 5 or (~act).sum() < 5:
            return np.full(n_va, float(np.mean(y_tr)))
        clf = RandomForestClassifier(
            n_estimators=n_trees, min_samples_leaf=3, max_features=0.3,
            random_state=seed, n_jobs=-1)
        clf.fit(Xt, act.astype(int))
        p = clf.predict_proba(Xv)[:, 1]
        cap = np.percentile(y_tr[act], 95)
        reg = _regressor("rf", seed, n_trees, leaf=4)
        reg.fit(Xt[act], np.minimum(y_tr[act], cap))
        pred = p * reg.predict(Xv) + (1 - p) * y_tr[~act].mean()
        return np.clip(pred, 0, None)

    if spec.get("log"):
        y_fit = np.log1p(y_tr)
    else:
        y_fit = y_tr.astype(float)
        if spec.get("cap"):
            y_fit = np.minimum(y_fit, np.quantile(y_fit, spec["cap"]))

    reg = _regressor(spec["kind"], seed, n_trees, leaf=3)
    reg.fit(Xt, y_fit)
    pred = reg.predict(Xv)
    if spec.get("log"):
        pred = np.expm1(pred)
    return np.clip(pred, 0, None)
