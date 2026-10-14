## Pull Request Description

### Summary of Changes
<!-- Provide a concise description of the purpose of this PR and what was changed. -->

### Workstream / Member Alignment
- [ ] Member 1: Topic Data + Model 1 (Classification & Taxonomy)
- [ ] Member 2: Answer Data + Model 2 (EngSAF Baselines, GroupKFold, Error Analysis)
- [ ] Member 3: MLOps Backbone (Git, DVC, CI/CD, Reproducibility, Security)
- [ ] Member 4: Integration, Serving, Monitoring (FastAPI, Docker, Evidently, Prometheus)

---

### MLOps & Reproducibility Checklist
Please verify each of the following before requesting a review:

- [ ] **No Data in Git**: Verified that `data/raw/` and `data/processed/` are untracked by Git and managed exclusively by DVC.
- [ ] **DVC Pipeline Integrity**: Ran `dvc repro` successfully from a clean tree and updated `dvc.lock`.
- [ ] **Data Leakage Safeguards**:
  - [ ] Confirmed forbidden inputs (`mark_scheme`, `rationale`) are excluded from feature engineering.
  - [ ] Grouped evaluation strictly partitions questions by ID (`GroupKFold`) with 0% train/val/test question leakage.
- [ ] **MLflow Tracking**: Runs logged with standardized experiment names, tags (`model_type`, `author`, `git_commit`), hyperparameters, and metrics (`macro_f1`, `qwk`, `accuracy`).
- [ ] **Automated Tests**: All unit and contract tests in `pytest tests/` pass (including edge-case and security tests).
- [ ] **Dependency Hygiene**: Pinned new packages in `requirements.txt` / `requirements-dev.txt` without unconstrained versions.
- [ ] **Documentation**: Updated relevant documentation in `docs/` and `README.md`.

---

### Linked Issues / Experiments
- Closes #<!-- Issue Number -->
- MLflow Run ID: `<!-- e.g., 67bae282-... -->`
