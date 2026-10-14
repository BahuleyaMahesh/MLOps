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

Use Python 3.10 or 3.11 (Python 3.11 recommended). Create and activate a virtual environment, then install dependencies:

```powershell
python -m pip install -r requirements.txt
```

*(Note: Git and DVC are already initialized in this repository. Do not run `git init` or `dvc init`.)*

## Run Phase 1

```powershell
dvc repro
```

The pipeline:

1. Downloads the train, validation, and test splits to local DVC-managed data (`data/raw/engsaf`).
2. Cleans text, standardizes whitespaces/linebreaks, unescapes HTML entities, strips markdown formatting while preserving math/code syntax, and validates score labels into `data/processed/engsaf` (see [docs/data_cleaning_and_preprocessing.md](docs/data_cleaning_and_preprocessing.md) and [reports/cleaning_report.json](reports/cleaning_report.json)).
3. Writes an aggregate-only profile and EDA report of the processed data to `reports/data_profile.json` and `reports/engsaf_eda.md`.
4. Trains and evaluates five answer-scoring baseline architectures:
   - **M2-A (Majority Class Baseline)**: Naive majority-class predictor (class 2 floor).
   - **M2-B (Cosine Similarity Baseline)**: Token-overlap / TF-IDF cosine similarity thresholding with thresholds tuned solely on validation data.
   - **M2-C (Engineered Features + Logistic Regression)**: Character/word length ratios, Jaccard/Dice overlaps, and dense semantic interaction embeddings.
   - **M2-D (Non-Linear Tree Baseline / HistGradientBoosting)**: HistGradientBoosting classifier capturing non-linear interactions across engineered signals.
   - **M2-E (Ordinal Regression / Frank-Hall)**: Decomposes 3-class ordinal scoring into thresholded binary classifiers ($P(\text{score} > 0)$ and $P(\text{score} > 1)$).
5. Executes diagnostic ablations:
   - *Without question text*: Verifies whether models memorize questions or assess true student-reference alignment.
   - *Length-only baseline*: Tests if predictions merely reward student verbosity.
   - *Copy-the-question sanity check*: Tests model resilience against verbatim question repetition.
6. Evaluates generalizability using 5-fold `GroupKFold` cross-validation partitioned strictly by question ID (zero question leakage across folds).
7. Logs all parameters and metrics to MLflow in the `answer-scorer-baselines` experiment (`mlflow.db`).
8. Generates comprehensive comparison and error analysis markdown reports in `reports/model2_comparison.md` and `reports/error_analysis.md`.

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
