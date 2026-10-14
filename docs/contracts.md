# Inter-Team Technical Contracts (Phase 1)

This document formalizes the internal interface agreements between all four team members to ensure modular, conflict-free collaboration and reproducibility.

---

## 1. Data Schema Contracts

### 1.1 Model 1: Topic Classification (Cleaned & Processed Data)

**Stage Output:** `data/processed/topic_data_{train|val|test}.parquet`<br>
**Responsible Member:** Member 1

| Column Name | Data Type | Description | Constraints |
| :--- | :--- | :--- | :--- |
| `id` | `string` / `int` | Unique post identifier from raw dataset | Primary key |
| `text` | `string` | Concatenated and cleaned question text (e.g., `title` + `body`) | Stripped HTML, code tokens normalized, min length 10 |
| `raw_tags` | `string` / `list[str]` | Original tags from source dataset (for taxonomy derivation) | Retained for reference, NEVER passed as a model feature |
| `topic` | `string` | Target coarse topic label mapped from `topic_map.yaml` | Exactly one of 8-12 approved topics |
| `topic_id` | `int` | Integer encoded target label | `0 <= topic_id < N_topics` |

> **Anti-Leakage Barrier:** Model features must be extracted solely from `text`. Tags are used strictly to assign ground truth `topic` and must never be provided as input features.

---

### 1.2 Model 2: Student Answer Scorer (Cleaned & Processed Data)

**Stage Output:** `data/processed/engsaf_{train|val|test}.parquet` and `data/processed/engsaf_grouped_folds.parquet`<br>
**Responsible Member:** Member 2

| Column Name | Data Type | Description | Constraints |
| :--- | :--- | :--- | :--- |
| `question_id` | `int` | Unique identifier for each distinct question prompt | Used for `GroupKFold` splitting |
| `question` | `string` | Text of the technical exam question | Mixed engineering topics |
| `student_answer` | `string` | Student submitted short answer text | Input feature source |
| `reference_answer` | `string` | Model/instructor reference answer text | Input feature source |
| `score` | `int` | Ground truth evaluation score | Value in `{0, 1, 2}` (0: incorrect, 1: partially correct, 2: correct) |
| `fold` | `int` | Assigned fold index for question-grouped cross-validation | `0 <= fold < 5` (GroupKFold) |

> **Strict Target Leakage Rule:**
> - `rationale` is the human grader's justification written *after* assigning the score. It contains direct label leakage and **must never be present in feature sets**.
> - `mark_scheme` is a static constant string across rows and carries zero predictive signal; it must be excluded.

---

## 2. Configuration Conventions (`configs/params.yaml`)

**Responsible Member:** Member 3

All executable scripts must decouple parameters from code:
- No hard-coded random seeds; scripts must read `global.random_seed`.
- Storage locations must be referenced via `paths.*`.
- Feature engineering toggles (e.g., `use_question_text`) must be configured in `params.yaml` to enable programmatic ablation runs.

---

## 3. MLflow Experiment Naming & Tracking

**Responsible Member:** Member 4

To avoid collision and maintain clean experiment trees on the shared MLflow server:

| Workstream | Standard Experiment Name | Primary Purpose |
| :--- | :--- | :--- |
| Model 1 (Topic Classifier) | `topic-baselines` | Track M1-A to M1-F baseline runs, hyperparameters, and macro-F1 |
| Model 2 (Answer Scorer) | `answer-scorer-baselines` | Track M2-A to M2-E runs on standard and question-grouped splits |

---

## 4. Shared Evaluation Function Interface

**Module Location:** `src/study_assistant/evaluation/metrics.py`<br>
**Responsible Member:** Member 4

All model training scripts must call this standardized evaluation harness to ensure uniform metric calculation across team members.

### Function Signature:
```python
from typing import Any, Dict, List, Optional
import numpy as np

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[List[Any]] = None,
    is_ordinal: bool = False,
) -> Dict[str, Any]:
    """Computes standardized evaluation metrics for project models.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth integer or categorical labels.
    y_pred : np.ndarray
        Model predicted class labels.
    y_proba : Optional[np.ndarray], default=None
        Predicted class probability distributions (for log-loss/ROC if applicable).
    labels : Optional[List[Any]], default=None
        Explicit list of class labels or indices.
    is_ordinal : bool, default=False
        If True (Model 2 answer scorer), computes Quadratic Weighted Kappa (QWK)
        and Mean Absolute Error (MAE).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "macro_f1": float
        - "macro_precision": float
        - "macro_recall": float
        - "accuracy": float (supplementary, never reported alone)
        - "per_class": Dict[str, Dict[str, float]]
        - "confusion_matrix": List[List[int]]
        - "qwk": Optional[float] (populated when is_ordinal=True)
        - "mae": Optional[float] (populated when is_ordinal=True)
    """
```

---

## 5. DVC Pipeline Stages Contract

**Pipeline File:** `dvc.yaml`<br>
**Responsible Member:** Member 3

| Stage Name | Command Script | Primary Inputs | Primary Outputs |
| :--- | :--- | :--- | :--- |
| `ingest_stack` | `python scripts/ingest_data.py --dataset stack` | `configs/params.yaml` | `data/raw/stackpulse/` |
| `ingest_engsaf` | `python scripts/ingest_data.py --dataset engsaf` | `configs/params.yaml` | `data/raw/engsaf/` |
| `clean_stack` | `python scripts/preprocess_topic.py` | `data/raw/stackpulse/`, `configs/topic_map.yaml` | `data/processed/topic_*.parquet` |
| `features_engsaf` | `python scripts/preprocess_scorer.py` | `data/raw/engsaf/` | `data/processed/engsaf_*.parquet` |
| `train_topic` | `python scripts/train_topic.py` | `data/processed/topic_*.parquet`, `configs/params.yaml` | `models/topic_model.pkl`, `metrics/topic_val.json` |
| `train_scorer` | `python scripts/train_scorer.py` | `data/processed/engsaf_*.parquet`, `configs/params.yaml` | `models/scorer_model.pkl`, `metrics/scorer_val.json` |
| `evaluate` | `python scripts/evaluate.py` | `models/`, `data/processed/` | `docs/evaluation_summary.json` |
