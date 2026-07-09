"""Loading and preparing the FlakeFlagger dataset.

The dataset (Alshammari et al., "FlakeFlagger: Predicting Flakiness Without
Rerunning Tests", ICSE 2021) contains 22k+ JUnit tests from 24 open-source
Java projects, each labelled flaky/not-flaky by 10,000 reruns.

Feature families follow the taxonomy in the original paper's
FlakeFlaggerFeaturesTypes.csv, with `testLength` added to the static family:

- smells:   boolean test-smell detectors (assertion roulette, mystery guest, ...)
- size:     test length and assertion count
- dynamic:  coverage counts, execution time, covered-line churn (hIndex windows),
            third-party library count
- vocab:    Java keyword counts + identifier tokens from the test body
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

SMELL_FEATURES = [
    "assertion-roulette",
    "conditional-test-logic",
    "eager-test",
    "fire-and-forget",
    "indirect-testing",
    "mystery-guest",
    "resource-optimism",
    "test-run-war",
]

SIZE_FEATURES = ["testLength", "numAsserts"]

DYNAMIC_FEATURES = [
    "numCoveredLines",
    "ExecutionTime",
    "projectSourceLinesCovered",
    "projectSourceClassesCovered",
    "hIndexModificationsPerCoveredLine_window5",
    "hIndexModificationsPerCoveredLine_window10",
    "hIndexModificationsPerCoveredLine_window25",
    "hIndexModificationsPerCoveredLine_window50",
    "hIndexModificationsPerCoveredLine_window75",
    "hIndexModificationsPerCoveredLine_window100",
    "hIndexModificationsPerCoveredLine_window500",
    "hIndexModificationsPerCoveredLine_window10000",
    "num_third_party_libs",
]

STATIC_FEATURES = SMELL_FEATURES + SIZE_FEATURES

FEATURE_FAMILIES: dict[str, list[str]] = {
    "smells": SMELL_FEATURES,
    "size": SIZE_FEATURES,
    "dynamic": DYNAMIC_FEATURES,
}

ALL_FEATURES = STATIC_FEATURES + DYNAMIC_FEATURES

LABEL = "flaky"
PROJECT = "project"


@dataclass
class Dataset:
    """Feature table plus label and project grouping."""

    frame: pd.DataFrame
    feature_names: list[str]

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[self.feature_names]

    @property
    def y(self) -> pd.Series:
        return self.frame[LABEL]

    @property
    def projects(self) -> pd.Series:
        return self.frame[PROJECT]


def load_flakeflagger(path: Path | None = None) -> Dataset:
    """Load the engineered-feature table (one row per test)."""
    path = path or DATA_DIR / "processed_data.csv"
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df[LABEL] = df[LABEL].astype(int)
    bool_like = [c for c in SMELL_FEATURES if df[c].dtype == object]
    for c in bool_like:
        df[c] = df[c].map({"True": 1, "False": 0, True: 1, False: 0}).astype(int)
    df[ALL_FEATURES] = df[ALL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    return Dataset(frame=df, feature_names=list(ALL_FEATURES))


def load_vocabulary(path: Path | None = None, features_path: Path | None = None) -> pd.DataFrame:
    """Load per-test token lists, joined with project via the feature table.

    Returns a frame with test_name, project, flaky, and tokenList (a
    comma-separated string of identifier tokens from the test body).
    """
    import ast

    path = path or DATA_DIR / "processed_data_with_vocabulary_per_test.csv"
    vocab = pd.read_csv(path, usecols=["test_name", "flakyStatus", "tokenList"])
    base = load_flakeflagger(features_path).frame[["test_name", PROJECT, LABEL]]
    merged = vocab.merge(base, on="test_name", how="inner")

    def normalize(raw) -> str:
        # tokenList is stored as the repr of a Python list of strings
        if not isinstance(raw, str) or not raw.startswith("["):
            return ""
        try:
            return ",".join(ast.literal_eval(raw))
        except (ValueError, SyntaxError):
            return ""

    merged["tokenList"] = merged["tokenList"].map(normalize)
    return merged


def dataset_summary(ds: Dataset) -> pd.DataFrame:
    """Per-project test and flaky-test counts, for the write-up."""
    g = ds.frame.groupby(PROJECT)[LABEL]
    out = pd.DataFrame({"tests": g.size(), "flaky": g.sum()})
    out["flaky_rate"] = (out["flaky"] / out["tests"]).round(4)
    out.loc["TOTAL"] = [out["tests"].sum(), out["flaky"].sum(),
                        round(out["flaky"].sum() / out["tests"].sum(), 4)]
    return out.astype({"tests": int, "flaky": int})
