# MLflow Experiment Tracking and Model Registry Conventions

This document specifies the standard conventions for MLflow experiment tracking, tagging, parameter/metric logging, artifact storage, and model registry lifecycle within the **AI-Powered Study Assistant** project.

---

## 1. Experiment Naming Architecture

All team runs must be organized into distinct, scoped experiments:

| Experiment Name | Workstream / Owner | Purpose |
|---|---|---|
| `answer-scorer-baselines` | Workstream 2 (Member 2 & 3) | Phase 1 EngSAF short-answer grading models (M2-A through M2-E), ablations, and GroupKFold cross-validation. |
| `topic-clf` | Workstream 1 (Member 1) | Question topic classification taxonomy models (M1-A through M1-F). |
| `retrain-and-monitoring` | Workstream 3 & 4 (Member 3 & 4) | Automated retraining triggers, drift response experiments, and promotion gate audits. |
| `integration-evaluation` | Workstream 4 (Member 4) | End-to-end multi-step pipeline evaluation (Topic Gate → Coverage Gate → Scorer). |

---

## 2. Standard Run Tags

Every MLflow run must record the following mandatory tags:

```python
mlflow.set_tags({
    "project": "ai-powered-study-assistant",
    "phase": "phase-1",
    "workstream": "workstream-2",
    "author": "member-2",  # or member-1, member-3, member-4
    "model_id": "M2-C",
    "model_family": "logistic-regression",
    "split_type": "question-grouped-kfold",  # or "standard-hf", "ablation"
    "git_commit": git_commit_sha,
    "dvc_revision": dvc_lock_hash,
    "stage": "development",  # "development", "staging", "production"
})
```

---

## 3. Required Parameters and Metrics Schema

### Logged Parameters
1. **Dataset Parameters**:
   - `dataset_name`: e.g. `IsmaelMousa/engsaf`
   - `dataset_revision`: pinned Git commit SHA on Hugging Face
   - `train_rows`, `val_rows`, `test_rows`
2. **Preprocessor Hyperparameters**:
   - `tfidf_max_features`: e.g. `30000`
   - `tfidf_ngram_max`: e.g. `2`
   - `svd_components`: e.g. `16`
3. **Model Hyperparameters**:
   - `classifier`: e.g. `LogisticRegression`, `HistGradientBoostingClassifier`, `FrankHallOrdinal`
   - `logistic_regression_c`: e.g. `1.0`
   - `random_state`: e.g. `42`

### Logged Metrics
All scoring models must log the following core metrics (both validation and test sets):

| Metric Name | Meaning | Target Range / Direction |
|---|---|---|
| `macro_f1` | Unweighted average of F1 across classes 0, 1, 2 | Higher is better ($0.0 \to 1.0$) |
| `quadratic_weighted_kappa` | Cohen's Kappa with quadratic penalty for severe misgradings | Higher is better ($-1.0 \to 1.0$) |
| `accuracy` | Overall correct score proportion | Higher is better |
| `mae` | Mean Absolute Error across ordinal scores (0, 1, 2) | Lower is better ($0.0 \to 2.0$) |

For GroupKFold runs, prefix metrics with `gkf_` and log both `_mean` and `_std`:
- `gkf_macro_f1_mean`, `gkf_macro_f1_std`
- `gkf_qwk_mean`, `gkf_qwk_std`
- `gkf_accuracy_mean`, `gkf_accuracy_std`
- `gkf_mae_mean`, `gkf_mae_std`

---

## 4. Artifact Logging Standards

Runs producing models must log:
1. `model/`: Serialized scikit-learn pipeline via `mlflow.sklearn.log_model` or joblib artifact.
2. `confusion_matrix.png`: Visual confusion matrix plot.
3. `classification_report.json`: Precision, recall, and support per class (0, 1, 2).
4. `feature_importance.json` / `feature_importance.png`: Top positive and negative coefficients.

---

## 5. Model Registry & Aliasing

The MLflow Model Registry uses **registered models** with role-based aliases rather than legacy stage transitions:

- **Registered Models**:
  - `answer-scorer`: The unified short-answer scoring model.
  - `topic-classifier`: The topic prediction model.
- **Aliases**:
  - `@Candidate`: Newly trained model that passed the local CI test suite.
  - `@Challenger`: Model that passed promotion gating and is undergoing drift shadow evaluation.
  - `@Production`: Currently active production service model served by FastAPI.

### Promotion Gate Criteria (Candidate → Challenger / Production):
1. **Macro-F1**: $\text{Macro-F1}_{\text{candidate}} \ge \text{Macro-F1}_{\text{production}} + 0.01$ (or non-inferior within 0.005 on grouped holdout).
2. **QWK**: $\text{QWK}_{\text{candidate}} \ge 0.50$ on question-grouped holdout.
3. **Severe Error Rate**: Severe error rate (true 0 scored 2 or true 2 scored 0) must be $\le 5\%$.
4. **Zero Question Leakage**: Strictly evaluated via 5-fold `GroupKFold` on unseen question IDs.
