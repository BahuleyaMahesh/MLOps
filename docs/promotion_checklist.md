# Model Promotion and Production Gating Checklist

This checklist defines the strict operational gating criteria required to promote an answer-scoring candidate model from **Candidate** (`@Candidate`) to **Challenger** (`@Challenger`) and ultimately to **Production** (`@Production`).

---

## 1. Automated Promotion Gate Criteria

To pass the automated promotion gate in CI/CD (`.github/workflows/retrain.yml`), the candidate model must satisfy all quantitative thresholds:

| Criterion | Evaluation Protocol | Minimum Threshold | Status |
|---|---|---|---|
| **Question Leakage** | 5-Fold `GroupKFold` strictly partitioned by Question ID | **0% Question Leakage** across splits | Required Pass |
| **Macro-F1 (Holdout)** | Mean macro-F1 across 5 folds | $\ge 0.50$ (and non-inferior to active model within $0.005$) | Required Pass |
| **Quadratic Weighted Kappa** | Mean QWK across 5 folds | $\ge 0.52$ | Required Pass |
| **Accuracy (Holdout)** | Mean Accuracy across 5 folds | $\ge 0.65$ | Required Pass |
| **Mean Absolute Error (MAE)** | Mean ordinal error across 5 folds | $\le 0.40$ | Required Pass |
| **Severe Misgrading Rate** | True 0 scored as 2 OR True 2 scored as 0 | **$\le 5.0\%$** of all evaluation instances | Required Pass |

---

## 2. Qualitative & Safeguard Checklist

Before applying the `@Production` alias in MLflow Model Registry, the reviewing engineer must confirm:

- [ ] **No Target Leakage**: Feature matrix input schema verified; `rationale` and `mark_scheme` are strictly absent.
- [ ] **Ablation Verification**:
  - [ ] Model without question text does not collapse or exhibit pure prompt memorization.
  - [ ] Length-only baseline performs significantly worse than full model (proving predictions don't merely reward verbosity).
  - [ ] Verbatim question-copying sanity check confirms model does not assign false passing scores to copied prompts.
- [ ] **Edge-Case Resilience**:
  - [ ] Empty text / whitespace strings handled gracefully without throwing unhandled exceptions.
  - [ ] Extreme text lengths (10,000+ characters) processed without memory exhaustion.
  - [ ] HTML tags, LaTeX, and Unicode symbols stripped/normalized safely.
  - [ ] Prediction outputs strictly adhere to class space $\{0, 1, 2\}$ and predicted probabilities sum to $1.0$.
- [ ] **Reproducibility Guarantee**:
  - [ ] Clean clone verified with `python -m src.reproducibility_check`.
  - [ ] All metrics reproduce within numerical tolerance $\Delta \le 0.001$.
  - [ ] `dvc.lock` and Git commit SHA recorded in the MLflow run tags.

---

## 3. Rollback Protocol

If runtime monitoring (Evidently drift reports or Prometheus counters) indicates degradation in production:
1. Trigger automatic reversion to the previous production alias:
   ```powershell
   python -m src.rollback_model --model answer-scorer --to-version <previous_stable_version>
   ```
2. Log the rollback incident in `reports/incidents.md` with timestamp, trigger metric, and root cause analysis.
