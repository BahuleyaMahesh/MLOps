# AI-Powered Study Assistant — Phase 1

Phase 1 explores the EngSAF short-answer dataset and establishes reproducible
baseline evaluation. It uses DVC for local dataset versioning, MLflow for
experiment tracking, and scikit-learn for the baseline. Docker is not required
for this phase.

## Data and approval status

The project reads `IsmaelMousa/engsaf` from Hugging Face. The dataset page has
conflicting licence statements; see [docs/data_licences.md](docs/data_licences.md).
Do not push data or derived artifacts to a shared DVC remote until the team has
resolved the permission question. The pipeline stores the downloaded splits
locally under `data/raw/`, which is excluded from Git.

## Setup

Use Python 3.10 or newer. Create and activate a virtual environment, then run:

```powershell
python -m pip install -r requirements.txt
git init
dvc init
```

If this folder is already a Git repository or DVC has already been initialized,
skip the corresponding initialization command.

## Run Phase 1

```powershell
dvc repro
```

The pipeline:

1. Downloads the train, validation, and test splits to local DVC-managed data (`data/raw/`).
2. Cleans text, standardizes whitespaces/linebreaks, unescapes HTML entities, strips markdown formatting while preserving math/code syntax, and validates score labels into `data/processed/` (see [docs/data_cleaning_and_preprocessing.md](docs/data_cleaning_and_preprocessing.md) and [reports/cleaning_report.json](reports/cleaning_report.json)).
3. Writes an aggregate-only profile of the processed data to `reports/data_profile.json`.
4. Trains a majority-class baseline, a validation-tuned cosine-similarity
   threshold baseline, and a TF-IDF plus engineered-feature Logistic Regression
   baseline using the cleaned dataset.
5. Evaluates both on the supplied test split and a grouped-by-question
   holdout carved only from the supplied training split.
6. Logs parameters and metrics to the local SQLite-backed MLflow store in
   `mlflow.db`.

Open the experiment UI with:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`params.yaml` pins the Hugging Face dataset to a commit SHA, while DVC records
the downloaded content hashes locally. A DVC remote is intentionally not
configured until data permissions and access controls are approved.

## Baseline safeguards

The classifier features use question, student answer, and reference answer.
`score` is the target; `rationale` and `mark_scheme` are excluded from inputs to
avoid target leakage. Metrics include macro-F1, accuracy, and quadratic
weighted kappa. The grouped holdout is important because multiple answers can
belong to the same question.

This is an educational research baseline, not a validated grading system.
Reported test results must remain untouched during model selection, and
performance should be interpreted alongside the dataset's small question pool,
single reference answers, and unresolved licensing notice.
