"""Evaluation protocols, in decreasing order of information shared between
training and test sets:

- mixed-project: pooled stratified k-fold CV; train and test folds share
  projects (what most prior work reports as "within-project").
- within-project: stratified k-fold CV run inside each single project.
- cross-project (leave-one-project-out): every project held out in turn;
  the deployment scenario.

Accuracy is deliberately not a headline metric: with a 3.6% positive rate a
useless model scores 96% accuracy. We report precision, recall, F1, ROC-AUC,
average precision, and precision@k, plus each fold's class distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .data import Dataset
from .modeling import RANDOM_STATE, make_model
from .modeling import pos_weight as _pos_weight


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Precision among the k tests ranked most likely to be flaky."""
    k = min(k, len(y_score))
    top = np.argsort(-y_score)[:k]
    return float(np.asarray(y_true)[top].mean())


def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "p_at_50": precision_at_k(y_true, y_score, 50),
        "n": len(y_true),
        "n_flaky": int(np.sum(y_true)),
        "flaky_rate": float(np.mean(y_true)),
    }
    # AUC metrics are undefined when a fold has a single class
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["avg_precision"] = average_precision_score(y_true, y_score)
    else:
        out["roc_auc"] = np.nan
        out["avg_precision"] = np.nan
    return out


def _fit_predict(model_name: str, X_train, y_train, X_test):
    model = make_model(model_name, pos_weight=_pos_weight(y_train))
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_test)[:, 1]
    else:
        score = model.predict(X_test).astype(float)
    return model.predict(X_test), score


def mixed_project_cv(
    ds: Dataset, model_name: str, features: list[str] | None = None, n_splits: int = 5
) -> pd.DataFrame:
    """Pooled stratified k-fold CV: train and test folds share projects.

    Most prior work reports this protocol under the name "within-project";
    the pooled folds let a model exploit cross-test regularities of every
    project it will be tested on.
    """
    features = features or ds.feature_names
    X, y = ds.frame[features].values, ds.y.values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        y_pred, y_score = _fit_predict(model_name, X[tr], y[tr], X[te])
        rows.append({"model": model_name, "fold": fold,
                     **compute_metrics(y[te], y_pred, y_score)})
    return pd.DataFrame(rows)


def within_project_cv(
    ds: Dataset, model_name: str, features: list[str] | None = None,
    n_splits: int = 5, min_flaky: int = 10,
) -> pd.DataFrame:
    """True within-project protocol: stratified k-fold CV inside each project.

    Only projects with at least min_flaky flaky tests participate — below
    that, per-fold positive counts are too small for stable metrics. One row
    per (project, fold).
    """
    features = features or ds.feature_names
    rows = []
    for project, group in ds.frame.groupby(ds.projects):
        y = group["flaky"].values
        if y.sum() < min_flaky or (len(y) - y.sum()) < n_splits:
            continue
        X = group[features].values
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=RANDOM_STATE)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            y_pred, y_score = _fit_predict(model_name, X[tr], y[tr], X[te])
            rows.append({"model": model_name, "project": project, "fold": fold,
                         **compute_metrics(y[te], y_pred, y_score)})
    return pd.DataFrame(rows)


def leave_one_project_out(
    ds: Dataset, model_name: str, features: list[str] | None = None,
    min_test_size: int = 50,
) -> pd.DataFrame:
    """Cross-project protocol: hold out every project in turn.

    This is the realistic deployment scenario — the model has never seen the
    project it is scoring. Projects with fewer than min_test_size tests are
    still trained on but skipped as held-out folds (their metrics would be
    too noisy to interpret).
    """
    features = features or ds.feature_names
    X, y, projects = ds.frame[features].values, ds.y.values, ds.projects.values
    rows = []
    for project in sorted(set(projects)):
        mask = projects == project
        if mask.sum() < min_test_size:
            continue
        y_pred, y_score = _fit_predict(model_name, X[~mask], y[~mask], X[mask])
        rows.append({"model": model_name, "project": project,
                     **compute_metrics(y[mask], y_pred, y_score)})
    return pd.DataFrame(rows)


def summarize(per_fold: pd.DataFrame, group: str = "model") -> pd.DataFrame:
    """Mean ± std of each metric across folds/projects."""
    metrics = ["precision", "recall", "f1", "roc_auc", "avg_precision", "p_at_50"]
    agg = per_fold.groupby(group)[metrics].agg(["mean", "std"]).round(3)
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg
