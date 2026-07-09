# One-command reproduction for the flakeguard dissertation project.
#
#   make setup        create .venv and install everything (needs python3.13 + brew)
#   make data         download the datasets into data/raw/
#   make experiments  run the full model x protocol grid -> results/
#   make explain      run the SHAP analysis -> results/
#   make model        train the shippable CLI model -> models/flakeguard.joblib
#   make test         run the pytest suite
#   make all          everything above, in order
#   make demo         score the bundled example test file with the CLI

PY      := .venv/bin/python
PIP     := .venv/bin/pip
DATA    := data/raw
FF_BASE := https://raw.githubusercontent.com/AlshammariA/FlakeFlagger/master/flakiness-predicter

.PHONY: setup data experiments explain model test all demo

setup:
	python3.13 -m venv .venv
	$(PIP) install -e ".[dev]"
	@command -v brew >/dev/null && brew list libomp >/dev/null 2>&1 || brew install libomp

data:
	mkdir -p $(DATA)
	curl -sL -o $(DATA)/processed_data.csv $(FF_BASE)/result/processed_data.csv
	curl -sL -o $(DATA)/processed_data_with_vocabulary_per_test.csv $(FF_BASE)/result/processed_data_with_vocabulary_per_test.csv
	curl -sL -o $(DATA)/idoft_pr_data.csv https://raw.githubusercontent.com/TestingResearchIllinois/idoft/main/pr-data.csv

experiments:
	$(PY) experiments/run_experiments.py

explain:
	$(PY) experiments/run_explainability.py

model:
	.venv/bin/flakeguard train --out models/flakeguard.joblib

test:
	$(PY) -m pytest tests/ -q

all: setup data test experiments explain model

demo:
	.venv/bin/flakeguard scan examples/ --model models/flakeguard.joblib
