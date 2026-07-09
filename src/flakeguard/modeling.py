"""Model zoo: baselines and main classifiers, all imbalance-aware."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42


def make_model(name: str, pos_weight: float = 1.0):
    """Build a fresh, unfitted classifier by name.

    pos_weight (n_negative / n_positive of the training fold) is used by
    XGBoost; the sklearn models use class_weight='balanced' which is
    equivalent per-fold.
    """
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
        ])
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"unknown model: {name}")


MODEL_NAMES = ["majority", "logreg", "random_forest", "xgboost"]
