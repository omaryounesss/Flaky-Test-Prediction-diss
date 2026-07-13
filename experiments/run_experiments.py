"""Run the full experimental grid and write results/ tables + figures.

Usage: python experiments/run_experiments.py [--skip-vocab]

Produces:
  results/dataset_summary.csv          per-project counts (write-up Table 1)
  results/mixed_project_folds.csv      pooled stratified CV, per-fold metrics
  results/mixed_project_summary.csv    mean±std per model
  results/within_project_folds.csv     true per-project CV, per (project, fold)
  results/within_project_summary.csv   mean±std per model
  results/cross_project_folds.csv      leave-one-project-out, per held-out project
  results/cross_project_summary.csv    mean±std per model
  results/lopo_per_project_detail.csv  per-project class distribution + PR metrics
  results/ablations.csv                feature-family ablations
  results/significance.csv             Wilcoxon signed-rank model comparisons
  results/figures/*.png                figures for the write-up
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flakeguard.data import (  # noqa: E402
    ALL_FEATURES,
    DYNAMIC_FEATURES,
    FEATURE_FAMILIES,
    STATIC_FEATURES,
    dataset_summary,
    load_flakeflagger,
    load_vocabulary,
)
from flakeguard.evaluation import (  # noqa: E402
    leave_one_project_out,
    mixed_project_cv,
    summarize,
    within_project_cv,
)
from flakeguard.modeling import MODEL_NAMES  # noqa: E402
from flakeguard.vocab import (  # noqa: E402
    vocab_leave_one_project_out,
    vocab_mixed_project_cv,
)

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_models(ds, skip_vocab: bool):
    mixed, within, cross = [], [], []
    for name in MODEL_NAMES:
        log(f"mixed-project CV: {name}")
        mixed.append(mixed_project_cv(ds, name))
        log(f"within-project CV: {name}")
        within.append(within_project_cv(ds, name))
        log(f"cross-project LOPO: {name}")
        cross.append(leave_one_project_out(ds, name))
    if not skip_vocab:
        vocab = load_vocabulary()
        log(f"vocabulary baseline: {len(vocab)} tests matched")
        log("mixed-project CV: vocab_xgboost")
        mixed.append(vocab_mixed_project_cv(vocab))
        log("cross-project LOPO: vocab_xgboost")
        cross.append(vocab_leave_one_project_out(vocab))
    return (pd.concat(mixed, ignore_index=True),
            pd.concat(within, ignore_index=True),
            pd.concat(cross, ignore_index=True))


def lopo_detail(cross: pd.DataFrame) -> pd.DataFrame:
    """Per held-out project: class distribution + precision/recall/F1 per model.

    F1 and ROC-AUC paint different pictures when flaky tests are rare, so the
    write-up reports this table alongside the aggregates."""
    base = (cross.groupby("project")[["n", "n_flaky", "flaky_rate"]].first()
            .round({"flaky_rate": 4}))
    pivot = cross.pivot_table(index="project", columns="model",
                              values=["precision", "recall", "f1"]).round(3)
    pivot.columns = [f"{model}_{metric}" for metric, model in pivot.columns]
    return base.join(pivot).sort_values("flaky_rate", ascending=False)


def run_ablations(ds) -> pd.DataFrame:
    """XGBoost with feature families removed / in isolation, both protocols."""
    variants: dict[str, list[str]] = {
        "all": ALL_FEATURES,
        "static_only": STATIC_FEATURES,
        "dynamic_only": DYNAMIC_FEATURES,
    }
    for family, cols in FEATURE_FAMILIES.items():
        variants[f"without_{family}"] = [f for f in ALL_FEATURES if f not in cols]

    rows = []
    for variant, features in variants.items():
        log(f"ablation: {variant} ({len(features)} features)")
        w = mixed_project_cv(ds, "xgboost", features=features)
        w["protocol"], w["variant"] = "mixed_project", variant
        c = leave_one_project_out(ds, "xgboost", features=features)
        c["protocol"], c["variant"] = "cross_project", variant
        rows += [w, c]
    return pd.concat(rows, ignore_index=True)


def significance_tests(cross: pd.DataFrame) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank on per-project F1 between models."""
    pivot = cross.pivot_table(index="project", columns="model", values="f1")
    rows = []
    for a, b in combinations([m for m in pivot.columns if m != "majority"], 2):
        paired = pivot[[a, b]].dropna()
        if (paired[a] - paired[b]).abs().sum() == 0:
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(paired[a], paired[b])
        rows.append({"model_a": a, "model_b": b,
                     "median_f1_a": paired[a].median(), "median_f1_b": paired[b].median(),
                     "wilcoxon_stat": stat, "p_value": p, "n_projects": len(paired)})
    return pd.DataFrame(rows).round(4)


def make_figures(mixed: pd.DataFrame, within: pd.DataFrame, cross: pd.DataFrame,
                 ablations: pd.DataFrame):
    sns.set_theme(style="whitegrid")

    # Fig 1: F1 by protocol and model — the generalization-gap headline
    m = mixed.groupby("model")["f1"].mean().rename("mixed-project (pooled CV)")
    w = within.groupby("model")["f1"].mean().rename("within-project (per-project CV)")
    c = cross.groupby("model")["f1"].mean().rename("cross-project (LOPO)")
    all_three = pd.concat([m, w, c], axis=1).drop(index="majority", errors="ignore")
    ax = all_three.plot.bar(rot=20, figsize=(8.5, 4.5),
                            color=["#4c72b0", "#55a868", "#dd8452"])
    ax.set_ylabel("Mean F1")
    ax.set_title("F1 by evaluation protocol (the generalization gap)")
    plt.tight_layout()
    plt.savefig(FIGURES / "generalization_gap.png", dpi=200)
    plt.close()

    # Fig 2: per-project F1 spread under LOPO
    order = cross.groupby("project")["f1"].max().sort_values().index
    plt.figure(figsize=(9, 6))
    sns.stripplot(data=cross[cross["model"] != "majority"], y="project", x="f1",
                  hue="model", order=order, size=6)
    plt.title("Per-project F1 under leave-one-project-out")
    plt.tight_layout()
    plt.savefig(FIGURES / "lopo_per_project.png", dpi=200)
    plt.close()

    # Fig 3: ablations
    ab = (ablations.groupby(["protocol", "variant"])["f1"].mean()
          .unstack("protocol").sort_values("cross_project"))
    ax = ab.plot.barh(figsize=(8, 4.5), color=["#dd8452", "#4c72b0"])
    ax.set_xlabel("Mean F1 (XGBoost)")
    ax.set_title("Feature-family ablations")
    plt.tight_layout()
    plt.savefig(FIGURES / "ablations.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-vocab", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    ds = load_flakeflagger()
    dataset_summary(ds).to_csv(RESULTS / "dataset_summary.csv")
    log(f"dataset: {len(ds.frame)} tests, {int(ds.y.sum())} flaky, "
        f"{ds.projects.nunique()} projects")

    mixed, within, cross = run_models(ds, skip_vocab=args.skip_vocab)
    mixed.to_csv(RESULTS / "mixed_project_folds.csv", index=False)
    within.to_csv(RESULTS / "within_project_folds.csv", index=False)
    cross.to_csv(RESULTS / "cross_project_folds.csv", index=False)
    summarize(mixed).to_csv(RESULTS / "mixed_project_summary.csv")
    summarize(within).to_csv(RESULTS / "within_project_summary.csv")
    summarize(cross).to_csv(RESULTS / "cross_project_summary.csv")
    lopo_detail(cross).to_csv(RESULTS / "lopo_per_project_detail.csv")

    ablations = run_ablations(ds)
    ablations.to_csv(RESULTS / "ablations.csv", index=False)

    significance_tests(cross).to_csv(RESULTS / "significance.csv", index=False)
    make_figures(mixed, within, cross, ablations)

    log("done — see results/")
    print("\n=== MIXED-PROJECT (pooled 5-fold stratified CV) ===")
    print(summarize(mixed).to_string())
    print("\n=== WITHIN-PROJECT (per-project 5-fold CV) ===")
    print(summarize(within).to_string())
    print("\n=== CROSS-PROJECT (leave-one-project-out) ===")
    print(summarize(cross).to_string())


if __name__ == "__main__":
    main()
