# Scripts Directory (`scripts/`)

## Role of Command-Line Scripts

This directory houses deterministic, executable Python scripts that orchestrate end-to-end data processing, training, and evaluation pipelines.

### Design Principles

1. **Deterministic & Parameterized:**
   - Every script must read configurations and hyperparameter settings directly from `configs/params.yaml` or parse explicit CLI flags.
   - Global random seeds must be explicitly set from configuration (`global.random_seed`).
2. **DVC Pipeline Compatibility:**
   - Scripts serve as the executable entrypoints for `dvc.yaml` stages:
     - `ingest_data.py`: Downloads or loads raw data and saves to `data/raw/`.
     - `preprocess_topic.py`: Cleans text, applies taxonomy mapping, and writes `data/processed/topic_train.parquet`, etc.
     - `preprocess_scorer.py`: Strips `rationale` and `mark_scheme`, computes embeddings/features, outputs question-grouped splits.
     - `train_topic.py`: Trains Model 1 candidates and logs metrics/artifacts to MLflow.
     - `train_scorer.py`: Trains Model 2 candidates on both provided and question-grouped splits.
     - `evaluate.py`: Generates comparative evaluation summaries, confusion matrices, and metrics.
3. **MLflow Integration:**
   - Training scripts must log all hyperparameters, metrics, Git commit SHAs, and DVC data hashes to the appropriate MLflow experiment (`topic-baselines` or `answer-scorer-baselines`).
