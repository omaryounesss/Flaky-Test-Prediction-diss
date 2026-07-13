"""Convert results/*.csv into booktabs LaTeX tables for the dissertation.

Usage: python experiments/make_latex_tables.py
Writes dissertation/tables/*.tex, which chapters \\input directly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "dissertation" / "tables"

METRICS = ["precision", "recall", "f1", "roc_auc", "p_at_50"]
NICE = {"precision": "Precision", "recall": "Recall", "f1": "F1",
        "roc_auc": "ROC-AUC", "p_at_50": "P@50"}


def tex_escape(s: str) -> str:
    return str(s).replace("_", "\\_")


def mean_std_table(src: str, dst: str) -> None:
    df = pd.read_csv(RESULTS / src, index_col=0)
    out = pd.DataFrame(index=df.index.map(tex_escape))
    for m in METRICS:
        out[NICE[m]] = (df[f"{m}_mean"].map("{:.2f}".format)
                        + " $\\pm$ " + df[f"{m}_std"].map("{:.2f}".format)).values
    out.index.name = "Model"
    (TABLES / dst).write_text(
        out.to_latex(escape=False).replace("nan $\\pm$ nan", "--"))


def dataset_table() -> None:
    df = pd.read_csv(RESULTS / "dataset_summary.csv", index_col=0)
    df.columns = ["Tests", "Flaky", "Flaky rate"]
    df["Flaky rate"] = df["Flaky rate"].map(lambda r: f"{r:.1%}".replace("%", "\\%"))
    df.index = df.index.map(tex_escape)
    df.index.name = "Project"
    (TABLES / "dataset_summary.tex").write_text(df.to_latex(escape=False))


def significance_table() -> None:
    df = pd.read_csv(RESULTS / "significance.csv")
    df.columns = ["Model A", "Model B", "Median F1 (A)", "Median F1 (B)",
                  "$W$", "$p$", "$n$"]
    for col in ("Model A", "Model B"):
        df[col] = df[col].map(tex_escape)
    (TABLES / "significance.tex").write_text(df.to_latex(index=False, escape=False))


def lopo_detail_table() -> None:
    df = pd.read_csv(RESULTS / "lopo_per_project_detail.csv", index_col=0)
    keep = ["n", "n_flaky", "flaky_rate",
            "logreg_precision", "logreg_recall", "logreg_f1"]
    df = df[[c for c in keep if c in df.columns]]
    df.columns = ["Tests", "Flaky", "Rate", "Prec.", "Rec.", "F1"]
    df["Rate"] = df["Rate"].map(lambda r: f"{r:.1%}".replace("%", "\\%"))
    df.index = df.index.map(tex_escape)
    df.index.name = "Held-out project"
    (TABLES / "lopo_per_project_detail.tex").write_text(df.to_latex(escape=False))


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    mean_std_table("mixed_project_summary.csv", "mixed_project_summary.tex")
    mean_std_table("within_project_summary.csv", "within_project_summary.tex")
    mean_std_table("cross_project_summary.csv", "cross_project_summary.tex")
    dataset_table()
    significance_table()
    lopo_detail_table()
    print(f"wrote {len(list(TABLES.glob('*.tex')))} tables to {TABLES}")


if __name__ == "__main__":
    main()
