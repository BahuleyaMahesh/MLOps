# Branching Rules and Repository Governance

This document outlines the branching strategy, protection rules, and collaboration standards for the **AI-Powered Study Assistant** repository.

---

## 1. Branch Strategy

The team adopts a modified **GitHub Flow** with structured branch prefixes:

| Branch Type | Naming Convention | Description / Target |
|---|---|---|
| **Production / Trunk** | `main` | Production-ready, fully reproducible code. Direct pushes are disabled. |
| **Feature Branches** | `feature/<member>-<feature-description>` | Work on new models, preprocessing steps, or pipeline stages. |
| **Chore / Setup** | `chore/<member>-<task-description>` | Tooling, CI/CD, dependency updates, and DVC infrastructure. |
| **Bug Fixes** | `bugfix/<issue-number>-<description>` | Patching regressions, pipeline broken states, or test failures. |
| **Hotfixes** | `hotfix/<description>` | Urgent emergency patches applied directly against `main`. |

### Examples:
- `feature/member1-topic-classifier`
- `feature/member2-engsaf-baselines`
- `chore/member3-dvc-and-reproducibility`
- `feature/member4-fastapi-service`

---

## 2. Protected Branch Rules (`main`)

The `main` branch is permanently protected under the following rules:

1. **Pull Request Required**:
   - No direct pushes permitted to `main`.
   - Every change must go through a pull request from a feature/chore branch.
2. **Mandatory Code Reviews**:
   - At least **1 peer review approval** from another team member is required prior to merge.
   - Code author cannot approve their own PR.
3. **Automated Status Checks (CI) Must Pass**:
   - `lint-and-security`: Flake8/Black code style and `pip-audit` / secret scans must pass.
   - `unit-and-contract-tests`: All Pytest test suites across Python 3.10 and 3.11 must exit with 0.
   - `dvc-pipeline-check`: DVC pipeline integrity must be intact (`dvc repro` validates without errors).
4. **Clean Commits & History**:
   - Merge strategy is configured as **Squash and Merge** or **Rebase and Merge** to maintain a linear and clean Git history.
   - Branch must be rebased and up to date with `main` before merging.

---

## 3. Data and Artifact Governance

1. **Strictly No Data in Git**:
   - Never commit raw or processed datasets (`*.csv`, `*.parquet`, `*.jsonl`) into Git.
   - Verify `.gitignore` contains `data/raw/` and `data/processed/`.
   - All dataset versions are managed exclusively via DVC hashes tracked in `dvc.lock`.
2. **No Model Weights in Git**:
   - Serialized models (`models/*.joblib`, `*.pt`, `*.bin`, `*.onnx`) are ignored by Git.
   - Model versions and metrics are logged to MLflow and serialized into DVC-tracked `models/` outputs.
3. **No Target Leakage**:
   - Never use `rationale` or `mark_scheme` as inputs for model features.
   - The CI test suite actively checks for leakage prevention.

---

## 4. Pull Request Review Criteria

Reviewers must inspect:
- [ ] PR description uses `.github/pull_request_template.md`.
- [ ] `dvc repro` reproduces deterministically from scratch.
- [ ] Grouped evaluation uses `GroupKFold` on question IDs (zero cross-split question leakage).
- [ ] MLflow experiment run ID is cited and accessible in `mlflow.db`.
- [ ] New packages are pinned with exact versions in `requirements.txt`.
