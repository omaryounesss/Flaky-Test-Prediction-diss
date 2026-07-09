"""SHAP analysis of the XGBoost model: global importance + per-test reasons.

Usage: python experiments/run_explainability.py

Produces:
  results/shap_global_importance.csv   mean |SHAP| per feature
  results/example_risk_reasons.md      per-test explanations for 10 flaky tests
  results/figures/shap_summary.png     SHAP beeswarm for the write-up
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flakeguard.data import load_flakeflagger  # noqa: E402
from flakeguard.explain import global_importance, risk_reasons, shap_values_for  # noqa: E402
from flakeguard.modeling import make_model  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def main() -> None:
    import shap

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    ds = load_flakeflagger()
    X, y = ds.X, ds.y.values
    pos = int(y.sum())
    model = make_model("xgboost", pos_weight=(len(y) - pos) / pos)
    model.fit(X, y)

    # SHAP on a stratified sample to keep this fast
    rng = np.random.default_rng(42)
    flaky_idx = np.flatnonzero(y == 1)
    ok_idx = rng.choice(np.flatnonzero(y == 0), size=2000, replace=False)
    sample = X.iloc[np.concatenate([flaky_idx, ok_idx])]

    values = shap_values_for(model, sample)
    global_importance(values, ds.feature_names).to_csv(
        RESULTS / "shap_global_importance.csv", header=["mean_abs_shap"])

    shap.summary_plot(values, sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    # per-test explanations for the 10 highest-scored known-flaky tests
    scores = model.predict_proba(X.iloc[flaky_idx])[:, 1]
    top = flaky_idx[np.argsort(-scores)[:10]]
    lines = ["# Example risk explanations (known flaky tests)\n"]
    flaky_values = values[: len(flaky_idx)]
    for rank, idx in enumerate(top):
        row = ds.frame.iloc[idx]
        pos_in_sample = int(np.flatnonzero(flaky_idx == idx)[0])
        reasons = risk_reasons(flaky_values[pos_in_sample], ds.feature_names)
        lines.append(f"## {rank + 1}. `{row['test_name']}` ({row['project']})")
        lines.append(f"predicted risk: {scores[np.argsort(-scores)[rank]]:.2f}\n")
        lines += [f"- {r}" for r in reasons] + [""]
    (RESULTS / "example_risk_reasons.md").write_text("\n".join(lines))
    print("wrote SHAP importance, summary plot, and example explanations")
    print(global_importance(values, ds.feature_names).head(10).to_string())


if __name__ == "__main__":
    main()
