"""Token-vocabulary baseline: bag-of-words over test-body identifiers.

This reproduces the "vocabulary of flaky tests" line of prior work
(Pinto et al., MSR 2020): predict flakiness purely from the tokens that
appear in the test body, with no engineered features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

from .evaluation import RANDOM_STATE, compute_metrics
from .modeling import make_model


def _vectorizer() -> TfidfVectorizer:
    # tokenList is comma-separated; tests are short so cap the vocabulary
    return TfidfVectorizer(
        tokenizer=lambda s: [t for t in s.split(",") if t],
        preprocessor=None,
        lowercase=True,
        max_features=2000,
        token_pattern=None,
    )


def _fit_predict(model_name, texts_train, y_train, texts_test):
    vec = _vectorizer()
    X_train = vec.fit_transform(texts_train)
    X_test = vec.transform(texts_test)
    pos = max(int(y_train.sum()), 1)
    model = make_model(model_name, pos_weight=(len(y_train) - pos) / pos)
    model.fit(X_train, y_train)
    return model.predict(X_test), model.predict_proba(X_test)[:, 1]


def vocab_within_project_cv(
    frame: pd.DataFrame, model_name: str = "xgboost", n_splits: int = 5
) -> pd.DataFrame:
    texts, y = frame["tokenList"].values, frame["flaky"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(texts, y)):
        y_pred, y_score = _fit_predict(model_name, texts[tr], y[tr], texts[te])
        rows.append({"model": f"vocab_{model_name}", "fold": fold,
                     **compute_metrics(y[te], y_pred, y_score)})
    return pd.DataFrame(rows)


def vocab_leave_one_project_out(
    frame: pd.DataFrame, model_name: str = "xgboost", min_test_size: int = 50
) -> pd.DataFrame:
    texts = frame["tokenList"].values
    y = frame["flaky"].values
    projects = frame["project"].values
    rows = []
    for project in sorted(set(projects)):
        mask = projects == project
        if mask.sum() < min_test_size:
            continue
        y_pred, y_score = _fit_predict(model_name, texts[~mask], y[~mask], texts[mask])
        rows.append({"model": f"vocab_{model_name}", "project": project,
                     **compute_metrics(y[mask], y_pred, y_score)})
    return pd.DataFrame(rows)
