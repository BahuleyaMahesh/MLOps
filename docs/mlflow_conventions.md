# MLflow Experiment Tracking Conventions (Phase 1)

**Owner:** Member 4<br>
**Scope:** Phase 1 Development & Reproducibility

---

## 1. Experiment Naming Architecture

To maintain a clean and structured experiment hierarchy across local environments and shared servers (e.g., DagsHub MLflow), all runs must be logged to one of two dedicated experiments:

| Experiment Name | Target Model | Scope of Runs |
| :--- | :--- | :--- |
| `topic-baselines` | Model 1 (Topic Classifier) | Baseline comparison runs M1-A to M1-F on stratified Stack data |
| `answer-scorer-baselines` | Model 2 (Answer Scorer) | Baseline comparison runs M2-A to M2-E on provided & grouped splits |

```python
import mlflow

# Always set experiment at the start of a training script
mlflow.set_experiment("topic-baselines")  # or "answer-scorer-baselines"
```

---

## 2. Standardized Run Naming

Every run name must follow the template:
```text
{candidate_id}_{model_family}_{split_type}_{variant}
```
*Examples:*
- `M1-C_tfidf_logreg_stratified100k_v1`
- `M2-C_minilm_logreg_provided_split_v1`
- `M2-D_minilm_xgboost_grouped_fold1_v1`

---

## 3. Mandatory Logged Items

Every training run logged to MLflow must record the following items:

### 3.1 Parameters (`mlflow.log_params`)
- `model_name`: Short model name (e.g., `tfidf_logreg`, `xgboost_pairwise`).
- `candidate_id`: ID from Phase 1 plan (e.g., `M1-A`, `M1-C`, `M2-D`).
- `random_seed`: Seed value from `configs/params.yaml` (e.g., `42`).
- `dataset_id`: Hugging Face dataset identifier.
- `sample_size`: Number of training examples used.
- `split_type`: `provided_split` or `question_grouped_split`.
- `feature_set`: Summary of features used (e.g., `use_question_text=False`, `cosine_only`, `tfidf_unigram`).
- Hyperparameters: Algorithm-specific settings (e.g., `C`, `penalty`, `max_depth`, `learning_rate`, `n_estimators`).

### 3.2 Metrics (`mlflow.log_metrics`)
- **Primary Metrics (Reported in Leaderboards):**
  - `val_macro_f1` (Primary for Model 1 and Model 2)
  - `val_qwk` (Quadratic Weighted Kappa; primary for Model 2)
- **Secondary Metrics:**
  - `val_macro_precision`
  - `val_macro_recall`
  - `val_accuracy` (Never reported in isolation due to class imbalance)
  - `val_mae` (Mean Absolute Error for Model 2)
- **Per-Class Metrics:**
  - `val_recall_class_{0,1,2}` or `val_recall_{topic_id}`

### 3.3 Tags (`mlflow.set_tags`)
- `git_commit`: Active Git commit SHA (`git rev-parse HEAD`).
- `dvc_data_hash`: Hash of the dataset version recorded in DVC.
- `author`: Member ID executing the run (e.g., `Member_1`).
- `phase`: `phase1-development`.
- `hardware`: `local_cpu`, `colab_cpu`, or `colab_t4_gpu`.

### 3.4 Artifacts (`mlflow.log_artifact`)
- `confusion_matrix.png`: Normalized confusion matrix visualization.
- `classification_report.json`: Full JSON dictionary of per-class precision, recall, and F1.
- `params.yaml`: Copy of the exact parameter configuration used.

---

## 4. Standard Logging Template

```python
import subprocess
import mlflow
from study_assistant.evaluation.metrics import evaluate

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"

def log_experiment_run(
    experiment_name: str,
    run_name: str,
    candidate_id: str,
    params: dict,
    metrics: dict,
    dvc_hash: str,
    author: str,
    artifacts: list = None
):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        # 1. Log Tags
        mlflow.set_tags({
            "candidate_id": candidate_id,
            "git_commit": get_git_commit(),
            "dvc_data_hash": dvc_hash,
            "author": author,
            "phase": "phase1-development",
        })

        # 2. Log Parameters
        mlflow.log_params(params)

        # 3. Log Metrics
        mlflow.log_metrics(metrics)

        # 4. Log Artifacts
        if artifacts:
            for artifact_path in artifacts:
                mlflow.log_artifact(artifact_path)
```
