# How to run everything

Complete operating manual for the project. Every command is run from the
repository root. If you only remember one thing: `make all` reproduces the
entire study from a clean checkout.

## 0. Prerequisites (one-time, macOS)

- **Python 3.13** — `brew install python@3.13`. Do *not* use 3.14: as of
  July 2026 shap/numba wheels for it stall pip.
- **libomp** — `brew install libomp` (XGBoost crashes without it; `make setup`
  handles this).

## 1. Setup

```bash
make setup          # venv + editable install of the flakeguard package
source .venv/bin/activate   # optional; the Makefile uses .venv/bin explicitly
```

## 2. Get the data

```bash
make data
```

Downloads into `data/raw/` (gitignored, ~23 MB total):

| file | what it is |
|---|---|
| `processed_data.csv` | 22,236 tests × 23 engineered features + flaky label + project (FlakeFlagger, ICSE 2021) |
| `processed_data_with_vocabulary_per_test.csv` | same tests with per-test identifier token lists |
| `idoft_pr_data.csv` | IDoFT flaky-test registry (for the extension study) |

## 3. Run the study

```bash
make experiments    # ~6 minutes on a laptop
```

This runs, with fixed seed 42:

1. **Within-project protocol** — 5-fold stratified CV, four models
   (majority, logistic regression, random forest, XGBoost).
2. **Cross-project protocol** — leave-one-project-out over all 24 projects.
3. **Vocabulary baseline** — TF-IDF over test-body tokens + XGBoost, both protocols.
4. **Ablations** — XGBoost with each feature family removed / in isolation.
5. **Significance tests** — Wilcoxon signed-rank over per-project F1.

Outputs land in `results/` (see the table in section 5).

```bash
make explain        # SHAP analysis, ~1 minute
make model          # trains models/flakeguard.joblib for the CLI
make test           # 8 unit tests, ~3 s
```

## 4. Use the tool

```bash
flakeguard scan path/to/any/java/repo          # ranked risk report with reasons
flakeguard scan . --format markdown            # for GitHub Action summaries
flakeguard scan . --format json                # machine-readable
flakeguard scan . --fail-above 0.9             # CI gate: exit 1 if breached
flakeguard scan . --model ""                   # heuristic-only mode (no model)
make demo                                      # scores examples/PaymentServiceTest.java
```

How it works: `extract.py` finds `@Test`-annotated methods with a
brace-matching parser, `cli.tokenize_java` turns each body into identifier
tokens matching the dataset's conventions, the trained TF-IDF+XGBoost model
scores them, and the risky-API detectors (Thread.sleep, network, filesystem,
time, randomness, shared state, ...) supply the human-readable reasons.

The GitHub Action (`.github/workflows/flaky-risk-gate.yml`) runs the markdown
scan on every PR that touches `*Test*.java` files and posts the report to the
PR's step summary.

## 5. Where each result lives

| artifact | dissertation use |
|---|---|
| `results/dataset_summary.csv` | Table 1 — per-project test/flaky counts |
| `results/within_project_summary.csv` | RQ1 headline table (mean ± std per model) |
| `results/cross_project_summary.csv` | RQ3 headline table |
| `results/within_project_folds.csv`, `cross_project_folds.csv` | per-fold/per-project raw metrics (for variance reporting) |
| `results/ablations.csv` | RQ2 / feature-family analysis |
| `results/significance.csv` | Wilcoxon model comparisons |
| `results/shap_global_importance.csv` | RQ4 feature ranking |
| `results/example_risk_reasons.md` | RQ4 worked examples |
| `results/figures/generalization_gap.png` | headline figure |
| `results/figures/lopo_per_project.png` | per-project variance figure |
| `results/figures/ablations.png` | ablation figure |
| `results/figures/shap_summary.png` | SHAP beeswarm |

## 6. Troubleshooting

- `XGBoostError: libomp.dylib` → `brew install libomp`.
- pip hangs for 10+ minutes → you are on Python 3.14; recreate the venv with 3.13.
- `flakeguard: command not found` → venv not active; use `.venv/bin/flakeguard`
  or `source .venv/bin/activate`.
- Flat model scores on a new repo → the model file predates the tokenizer fix;
  retrain with `make model`.
- Numbers differ slightly from previously generated results → check you ran
  with an unmodified `RANDOM_STATE = 42` and the same library versions
  (`pip freeze` vs a fresh `make setup`).
