"""Smoke tests for the research pipeline on a small synthetic dataset."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flakeguard.data import ALL_FEATURES, LABEL, PROJECT, Dataset
from flakeguard.evaluation import (
    compute_metrics,
    leave_one_project_out,
    precision_at_k,
    within_project_cv,
)


@pytest.fixture
def synthetic_dataset() -> Dataset:
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(rng.random((n, len(ALL_FEATURES))), columns=ALL_FEATURES)
    # make the label learnable from one feature
    df[LABEL] = (df["ExecutionTime"] > 0.8).astype(int)
    df[PROJECT] = np.repeat(["a", "b", "c"], n // 3)
    return Dataset(frame=df, feature_names=list(ALL_FEATURES))


def test_precision_at_k():
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    assert precision_at_k(y, scores, 2) == 0.5
    assert precision_at_k(y, scores, 10) == 0.5  # k capped at n


def test_metrics_handle_single_class_fold():
    m = compute_metrics(np.zeros(10), np.zeros(10), np.zeros(10))
    assert np.isnan(m["roc_auc"])
    assert m["precision"] == 0.0


def test_within_project_cv_learns_synthetic_signal(synthetic_dataset):
    res = within_project_cv(synthetic_dataset, "random_forest", n_splits=3)
    assert len(res) == 3
    assert res["f1"].mean() > 0.8


def test_lopo_produces_one_row_per_project(synthetic_dataset):
    res = leave_one_project_out(synthetic_dataset, "logreg", min_test_size=10)
    assert sorted(res["project"]) == ["a", "b", "c"]
