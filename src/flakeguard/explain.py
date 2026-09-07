"""Explainability: SHAP attributions turned into human-readable risk reasons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import HINDEX_WINDOWS

# Plain-English descriptions of what a high value of each feature means,
# used to turn SHAP attributions into remediation hints.
FEATURE_MEANING = {
    "assertion-roulette": "many assertions without messages (assertion roulette smell)",
    "conditional-test-logic": "control flow (if/loops) inside the test body",
    "eager-test": "the test exercises several methods of the class under test",
    "fire-and-forget": "the test launches background work it never waits on",
    "indirect-testing": "the test asserts on objects other than the class under test",
    "mystery-guest": "the test depends on external resources (files, databases)",
    "resource-optimism": "the test assumes an external resource is in a given state",
    "test-run-war": "the test competes with others for a shared resource",
    "testLength": "long test body",
    "numAsserts": "many assertions",
    "numCoveredLines": "large amount of covered code",
    "ExecutionTime": "long execution time",
    "projectSourceLinesCovered": "covers a large share of project source lines",
    "projectSourceClassesCovered": "touches many project classes",
    "num_third_party_libs": "exercises many third-party libraries",
}
for _w in HINDEX_WINDOWS:
    FEATURE_MEANING[f"hIndexModificationsPerCoveredLine_window{_w}"] = (
        f"covered code was modified frequently in the last {_w} commits"
    )


def shap_values_for(model, X: pd.DataFrame):
    """TreeExplainer SHAP values for the positive (flaky) class."""
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):  # sklearn RF returns [class0, class1]
        values = values[1]
    if values.ndim == 3:  # (n, features, classes)
        values = values[:, :, 1]
    return values


def risk_reasons(
    shap_row: np.ndarray, feature_names: list[str], top_n: int = 3
) -> list[str]:
    """The top_n features pushing this test's prediction towards 'flaky'."""
    order = np.argsort(-shap_row)
    reasons = []
    for i in order[:top_n]:
        if shap_row[i] <= 0:
            break
        name = feature_names[i]
        reasons.append(FEATURE_MEANING.get(name, name))
    return reasons


def global_importance(shap_matrix: np.ndarray, feature_names: list[str]) -> pd.Series:
    """Mean |SHAP| per feature — the headline importance ranking."""
    return (
        pd.Series(np.abs(shap_matrix).mean(axis=0), index=feature_names)
        .sort_values(ascending=False)
    )
