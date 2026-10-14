# Notebooks Directory (`notebooks/`)

## Guidelines for Jupyter Notebooks

This directory is reserved exclusively for **Exploratory Data Analysis (EDA)**, data profiling, and hypothesis generation.

### Core Principles

1. **Exploration Only, Not Pipeline Logic:**
   - Notebooks are meant for visualizing distributions, checking missing values, examining class balance, and verifying schema.
   - **Reusable preprocessing, feature engineering, and model training logic MUST NOT reside solely in notebooks.** All reproducible pipelines must be implemented in Python modules under `src/study_assistant/` and executed via `scripts/` or `dvc.yaml`.
2. **Notebook Hygiene & Reproducibility:**
   - Always run **Kernel -> Restart and Run All Cells** before saving.
   - Clear large cell outputs (especially raw text dumps or images) before committing if they bloat Git diffs.
   - Never commit API tokens, passwords, or `.env` credentials in notebook cells.
3. **Naming Convention:**
   - Prefix notebooks with sequential numbering and clear descriptions:
     - `01_stackpulse_data_profiling.ipynb` (Member 1)
     - `02_engsaf_eda_and_split_audit.ipynb` (Member 2)
     - `03_topic_taxonomy_validation.ipynb` (Member 1)
     - `04_baseline_error_analysis.ipynb` (Member 2 / Member 4)
