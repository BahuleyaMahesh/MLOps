# AI-Powered Study Assistant: Phase 1 Plan (Development & Reproducibility)

**Course:** AI252TA, Machine Learning Operations (2026-27) · **Project #1**, SDG 4 · **Team:** 4 members
**Scope:** Phase 1 only. Based on `PRD.md` v0.1 and the faculty circular.

**Phase 1 requirements (from the circular):** dataset acquisition/approval · data exploration · feature engineering · baseline models · Git repository · DVC dataset versioning · MLflow experiment tracking · initial model evaluation.

> Tags: **[Confirmed]** = read from a source on 14 Oct 2026 · **[Verify]** = the team must check · **[Decision]** = needs team sign-off.

---

## 1. Dataset access check (dataset acquisition)

| Dataset | What I could check | Status |
|---|---|---|
| **EngSAF** (`IsmaelMousa/engsaf`) | Full dataset card and first page of the viewer | **[Confirmed]** readable. 5.58k rows; train 4.74k / val 338 / test 506. Columns: `question`, `student_answer`, `reference_answer`, `mark_scheme`, `score` (0/1/2), `rationale`. 107 distinct questions in the viewer |
| **StackPulse** (`Omarrran/StackPulse_778K_QnA_Code_dataset`) | The page returned HTTP 429 (rate limit) twice; only search snippets were readable | **Partly confirmed.** Stack Overflow programming Q&A. Hub licence tag apache-2.0. Raw files are per-language JSONL (e.g. `stackoverflow_java.jsonl`, 203 MB). Viewer shows ~1.95M rows although the title says 778K. **Columns and schema not verified** |

My sandbox cannot reach huggingface.co, so I could not download either dataset. **First task: one teammate runs `load_dataset(...)` in Colab for both and commits the schema, row counts and `df.head()` to `docs/`.**

### StackPulse is not the dataset the PRD assumed
The PRD used `habedi/stack-exchange-dataset` (CS / Data Science / Political Science Stack Exchange, with a tags column). StackPulse is **Stack Overflow code questions**, so Model 1 becomes a *programming-topic* classifier (web, data/ML, databases, concurrency, systems, algorithms, etc.). Whether a usable topic label exists depends on the schema, which is unverified. **[Verify]** that tags (or equivalent) exist.

**[Decision D0, first meeting]** Keep StackPulse, or switch to the PRD's dataset? If your faculty approved a specific dataset, follow that. The dataset approval step in the circular makes this the first thing to settle.

### EngSAF facts that affect Phase 1
- **[Confirmed]** The author re-split val/test to be semantically dissimilar from train (90% embedding-similarity cut-off). The split is **not grouped by question**, so build your own question-grouped split for a fair second evaluation.
- **[Confirmed]** The original paper's "unseen questions" split had only 12 unique questions, so question-held-out results will be noisy. Use `GroupKFold` and report spread.
- **[Confirmed]** Licence conflict: Hub tag says apache-2.0, card text says academic/research use only, no redistribution without permission. Follow the stricter text.
- **Observed in the preview:** some rationales contradict the score, which suggests label noise. `rationale` is derived from the label and **must never be a model input**.
- Subjects are mixed (OS, C++, hydrology, image processing, manufacturing, statistics, finance, ML, chemistry), mostly not programming.

---

## 2. Decisions before starting

| ID | Decision | Recommendation |
|---|---|---|
| D0 | Which Stack dataset | See §1 |
| D2 | Data size for Model 1 | Stratified sample of 100k to 200k rows; full data is too big for free Colab. Store the sample in DVC |
| D5 | Shared storage for DVC and MLflow | **DagsHub** can host both (**[Verify]** current free limits). Alternative: Google Drive DVC remote + MLflow on one member's machine (fragile) |
| D6 | Python / pins | Python 3.11; `requirements.txt` with pinned versions |

---

## 3. Phase 1 setup checklist

- [ ] GitHub repo created; everyone has access; Hugging Face accounts + read token (Colab secret, never in code)
- [ ] Branch rules: feature branches, one reviewer from a different member, PR template
- [ ] Repo skeleton: `src/`, `data/` (DVC-tracked, git-ignored), `configs/`, `notebooks/` (EDA only), `docs/`, `dvc.yaml`, `params.yaml`, `requirements.txt`, `.gitignore` (covers `data/`, `mlruns/`, `.env`)
- [ ] `dvc init`; remote configured (D5); every member confirms `dvc pull` works from a clean clone
- [ ] MLflow tracking server reachable from Colab and laptops; each member logs one dummy run
- [ ] Colab template: `git clone` repo, install pinned deps, set MLflow URI, run a script (training lives in scripts, not only notebooks)
- [ ] Schema + sample rows for both datasets committed to `docs/`
- [ ] `docs/data_licences.md` drafted
- [ ] Shared interface agreed in `docs/contracts.md` (see §6)

---

## 4. Models to train and compare (baselines + initial evaluation)

### 4.1 Selection protocol (write it down before training)
1. Fix the primary metric and tie-breakers before touching any test set.
2. Tune and compare on **validation** only (grouped CV for Model 2).
3. Every run is logged to MLflow with params, metrics, plots, the DVC data hash and the git commit.
4. The test set is evaluated **once**, for the shortlisted best model(s).
5. Record the winner and the runner-up in the Phase 1 report.

### 4.2 Model 1: question topic classifier

| ID | Model | Purpose |
|---|---|---|
| M1-A | Majority class | Floor |
| M1-B | Tag/keyword rules from `topic_map.yaml` | Non-ML baseline |
| M1-C | TF-IDF + Logistic Regression | Main baseline |
| M1-D | TF-IDF + Linear SVM | Strong linear model |
| M1-E | TF-IDF + Complement Naive Bayes | Fast, handles imbalance |
| M1-F | Sentence-embedding (MiniLM) + Logistic Regression | Semantic features |

Metrics: macro-F1, per-class precision/recall, confusion matrix. A fine-tuned transformer is optional and only if Colab GPU time allows.

### 4.3 Model 2: student answer scorer

| ID | Model | Purpose |
|---|---|---|
| M2-A | Majority class | Floor |
| M2-B | Cosine-similarity threshold(s) | Non-learned baseline |
| M2-C | Features (cosine, \|u−v\|, u⊙v, length ratios) + Logistic Regression | Main baseline |
| M2-D | Same features + XGBoost | Non-linear |
| M2-E | Same features + ordinal regression variant | Scores are ordered 0<1<2 |

Required checks: **with vs without question text** (the question pool is tiny, so models can memorise it), a length-only sanity baseline, and results on **both** the provided split and a question-grouped split.
Metrics: macro-F1, Quadratic Weighted Kappa, per-class recall, confusion matrix, MAE.

---

## 5. Work division (Phase 1)

### Member 1: Topic data + Model 1
- Settle D0; profile the Stack dataset (schema, tags, duplicates, class balance, text length)
- Build `topic_map.yaml` (tag phrase → topic, 8 to 12 topics)
- Clean text (strip HTML, handle code blocks); build the stratified sample (D2)
- Train M1-A to M1-F; log all runs to MLflow
- **Deliverables:** EDA notebook/report, taxonomy, preprocessing + training scripts, Model 1 comparison table, initial evaluation

### Member 2: Answer data + Model 2
- Profile EngSAF fully: score distribution, per-question counts, overlap of questions across splits, answer lengths, duplicates
- Build the question-grouped split and `GroupKFold`
- Feature engineering (embeddings, similarity, length features); confirm no `rationale` leaks into features
- Train M2-A to M2-E with the required checks; log all runs to MLflow
- **Deliverables:** EDA report, feature code, grouped split code, Model 2 results on both split types, initial error analysis

### Member 3: Git + DVC + reproducibility
- Repo skeleton, branch rules, PR template, `.gitignore`
- DVC remote, DVC-tracked raw and processed data, `dvc.yaml` stages (ingest → clean → features → train → evaluate), `params.yaml`
- Dependency pinning; Colab template
- Reproducibility check: a member who did not write the pipeline runs `dvc pull` and `dvc repro` on a clean clone and compares metrics within a stated tolerance
- **Deliverables:** working `dvc repro`, versioned datasets, README with clean-clone setup, reproducibility check record

### Member 4: MLflow + evaluation harness + dataset approval
- MLflow setup (shared server, D5), experiment naming, required params/tags, short `docs/mlflow_conventions.md`
- Shared evaluation code used by both models (macro-F1, QWK, confusion matrix plots, logging helper) so results are comparable
- Dataset approval record: `docs/data_licences.md` for both datasets, EngSAF conflict noted, source credits
- MLflow comparison tables for both models; collate the Phase 1 report (data summary, features, baselines, initial evaluation, limitations)
- **Deliverables:** tracked experiment history, evaluation module, licence/approval doc, Phase 1 report draft

**Interface (agree in Week 1, change only by PR):** processed data paths and column names; `params.yaml` keys; MLflow experiment names (`topic-baselines`, `answer-scorer-baselines`); the shared metrics function signature.

**Review pairs:** Member 1 ↔ Member 2 review each other's data and model code; Member 3 ↔ Member 4 review each other's pipeline and tracking code.

---

## 6. Phase 1 timeline (relative weeks; align to your faculty's review date)

| Week | Focus |
|---|---|
| 0 (now) | Decisions D0, D2, D5, D6; setup checklist §3; dataset schema check |
| 1 | EDA on both datasets; licence doc; DVC + MLflow working for all four; taxonomy v0; majority/rule/cosine baselines |
| 2 | Feature engineering; remaining models; grouped split; MLflow comparison; one clean-clone `dvc repro`; test-set run for the shortlisted models; Phase 1 report |

---

## 7. Phase 1 acceptance checklist

- [ ] Both datasets documented (source, fields, licence, limitations) and the dataset choice approved
- [ ] EDA completed for both datasets
- [ ] Features documented; no `rationale` in any feature set
- [ ] Baselines and candidate models trained, each run in MLflow with data hash and commit
- [ ] Model 2 reported on provided and question-grouped splits
- [ ] Data versioned in DVC; clean-clone `dvc repro` reproduces metrics within tolerance
- [ ] Initial evaluation written up honestly, including limitations and label noise

## 8. Risks (Phase 1)

| Risk | Mitigation |
|---|---|
| StackPulse schema or tags unsuitable for a topic label | Week-0 audit; fall back to the PRD's dataset |
| Tiny question pool inflates Model 2 results | Grouped split, with/without question text, report both |
| Remote MLflow/DVC not reachable from Colab | Test with a dummy run in Week 0 before any real work |
| Free Colab limits | Use a sample for Model 1; skip transformer models if needed |
| Licence misuse (EngSAF academic-only) | Keep raw data out of Git and public places; document use |
