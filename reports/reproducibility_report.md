# Independent Pipeline Reproducibility Audit Record

**Workstream**: Member 3 — Git, DVC & Reproducibility Backbone  
**Audit Timestamp**: `2026-10-14 09:04:34 UTC`  
**Git Commit SHA**: `9e70aec180a797f98895763090e7d22bad77e24d`  
**Environment**: Python `3.10.0` on `Windows 10 (AMD64)`  
**Strict Numerical Tolerance**: `Δ ≤ 0.001`  

---

## 1. Executive Summary

> [!IMPORTANT]
> **AUDIT STATUS: FULLY REPRODUCIBLE (PASSED)**  
> All 40 evaluated metrics across candidate models (M2-A through M2-E) and 5-fold GroupKFold 
cross-validation reproduced with deterministic fidelity within Δ ≤ 0.001.

---

## 2. Metric-by-Metric Verification Comparison

| Evaluation Slice | Metric Name | Baseline (Golden) | Reproduced | Absolute Diff (Δ) | Verification Status |
|---|---|---|---|---|---|
| `Standard Split - majority` | `macro_f1` | 0.1902 | 0.1902 | 0.00000 | ✅ PASSED |
| `Standard Split - majority` | `quadratic_weighted_kappa` | 0.0000 | 0.0000 | 0.00000 | ✅ PASSED |
| `Standard Split - majority` | `accuracy` | 0.3992 | 0.3992 | 0.00000 | ✅ PASSED |
| `Standard Split - majority` | `mae` | 0.8241 | 0.8241 | 0.00000 | ✅ PASSED |
| `Standard Split - cosine_similarity_threshold` | `macro_f1` | 0.4990 | 0.4990 | 0.00000 | ✅ PASSED |
| `Standard Split - cosine_similarity_threshold` | `quadratic_weighted_kappa` | 0.3439 | 0.3439 | 0.00000 | ✅ PASSED |
| `Standard Split - cosine_similarity_threshold` | `accuracy` | 0.5059 | 0.5059 | 0.00000 | ✅ PASSED |
| `Standard Split - cosine_similarity_threshold` | `mae` | 0.5791 | 0.5791 | 0.00000 | ✅ PASSED |
| `Standard Split - logistic_regression` | `macro_f1` | 0.5141 | 0.5141 | 0.00000 | ✅ PASSED |
| `Standard Split - logistic_regression` | `quadratic_weighted_kappa` | 0.4292 | 0.4292 | 0.00000 | ✅ PASSED |
| `Standard Split - logistic_regression` | `accuracy` | 0.5296 | 0.5296 | 0.00000 | ✅ PASSED |
| `Standard Split - logistic_regression` | `mae` | 0.5791 | 0.5791 | 0.00000 | ✅ PASSED |
| `Standard Split - gradient_boosting` | `macro_f1` | 0.5160 | 0.5160 | 0.00000 | ✅ PASSED |
| `Standard Split - gradient_boosting` | `quadratic_weighted_kappa` | 0.4394 | 0.4394 | 0.00000 | ✅ PASSED |
| `Standard Split - gradient_boosting` | `accuracy` | 0.5257 | 0.5257 | 0.00000 | ✅ PASSED |
| `Standard Split - gradient_boosting` | `mae` | 0.5494 | 0.5494 | 0.00000 | ✅ PASSED |
| `Standard Split - ordinal_regression` | `macro_f1` | 0.4608 | 0.4608 | 0.00000 | ✅ PASSED |
| `Standard Split - ordinal_regression` | `quadratic_weighted_kappa` | 0.4262 | 0.4262 | 0.00000 | ✅ PASSED |
| `Standard Split - ordinal_regression` | `accuracy` | 0.5079 | 0.5079 | 0.00000 | ✅ PASSED |
| `Standard Split - ordinal_regression` | `mae` | 0.6186 | 0.6186 | 0.00000 | ✅ PASSED |
| `GroupKFold - majority` | `macro_f1_mean` | 0.2000 | 0.2000 | 0.00000 | ✅ PASSED |
| `GroupKFold - majority` | `qwk_mean` | 0.0000 | 0.0000 | 0.00000 | ✅ PASSED |
| `GroupKFold - majority` | `accuracy_mean` | 0.4325 | 0.4325 | 0.00000 | ✅ PASSED |
| `GroupKFold - majority` | `mae_mean` | 0.7814 | 0.7814 | 0.00000 | ✅ PASSED |
| `GroupKFold - logistic_regression` | `macro_f1_mean` | 0.5016 | 0.5016 | 0.00000 | ✅ PASSED |
| `GroupKFold - logistic_regression` | `qwk_mean` | 0.3749 | 0.3749 | 0.00000 | ✅ PASSED |
| `GroupKFold - logistic_regression` | `accuracy_mean` | 0.5225 | 0.5225 | 0.00000 | ✅ PASSED |
| `GroupKFold - logistic_regression` | `mae_mean` | 0.5928 | 0.5928 | 0.00000 | ✅ PASSED |
| `GroupKFold - gradient_boosting` | `macro_f1_mean` | 0.5131 | 0.5131 | 0.00000 | ✅ PASSED |
| `GroupKFold - gradient_boosting` | `qwk_mean` | 0.3747 | 0.3747 | 0.00000 | ✅ PASSED |
| `GroupKFold - gradient_boosting` | `accuracy_mean` | 0.5392 | 0.5392 | 0.00000 | ✅ PASSED |
| `GroupKFold - gradient_boosting` | `mae_mean` | 0.5626 | 0.5626 | 0.00000 | ✅ PASSED |
| `GroupKFold - ordinal_regression` | `macro_f1_mean` | 0.4583 | 0.4583 | 0.00000 | ✅ PASSED |
| `GroupKFold - ordinal_regression` | `qwk_mean` | 0.3678 | 0.3678 | 0.00000 | ✅ PASSED |
| `GroupKFold - ordinal_regression` | `accuracy_mean` | 0.5073 | 0.5073 | 0.00000 | ✅ PASSED |
| `GroupKFold - ordinal_regression` | `mae_mean` | 0.6431 | 0.6431 | 0.00000 | ✅ PASSED |
| `GroupKFold - cosine_similarity_threshold` | `macro_f1_mean` | 0.4334 | 0.4334 | 0.00000 | ✅ PASSED |
| `GroupKFold - cosine_similarity_threshold` | `qwk_mean` | 0.2983 | 0.2983 | 0.00000 | ✅ PASSED |
| `GroupKFold - cosine_similarity_threshold` | `accuracy_mean` | 0.4724 | 0.4724 | 0.00000 | ✅ PASSED |
| `GroupKFold - cosine_similarity_threshold` | `mae_mean` | 0.6271 | 0.6271 | 0.00000 | ✅ PASSED |

---

## 3. Pipeline Stages Audited

| Stage | Command | Inputs / Dependencies | Tracked Outputs |
|---|---|---|---|
| `ingest` | `python -m src.acquire_data` | `src/acquire_data.py`, `params.yaml` | `data/raw/engsaf` |
| `clean` | `python -m src.clean_data` | `src/clean_data.py`, `data/raw/engsaf` | `data/processed/engsaf`, `reports/cleaning_report.json` |
| `explore` | `python -m src.profile_data` | `src/profile_data.py`, `data/processed/engsaf` | `reports/data_profile.json`, `reports/engsaf_eda.md` |
| `features` | `python -m src.extract_features` | `src/features.py`, `src/extract_features.py` | `data/features/engsaf` |
| `train` | `python -m src.train` | `src/train.py`, `data/features/engsaf` | `models/` |
| `evaluate` | `python -m src.evaluate` | `src/evaluate.py`, `models/`, `data/features/engsaf` | `reports/baseline_metrics.json`, `reports/model2_comparison.md`, `reports/error_analysis.md` |

---

## 4. DVC Remote and Cache Verification

- **DVC Remote**: `local_remote` (`../../dvc_storage`) configured in `.dvc/config`.
- **DVC Status**: `dvc status` verified clean with all stages and data caches synchronized.
- **Data Integrity**: SHA-256 fingerprinting on splits confirms zero corruption.
- **Sign-off**: Member 3 (Reproducibility & Infrastructure Lead).
