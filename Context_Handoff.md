# LLM Context Handoff: AI-Powered Study Assistant (Phase 1)

> **Purpose of this file.** You are an LLM assistant that has been given three things: (1) `PRD.md`, (2) `MLOps_Project_Setup_and_Work_Division.md` (the Phase 1 plan), and (3) this file. You have no memory of the conversation that produced them. This file explains the situation, resolves conflicts between the other two documents, and states what is verified, what is not, and what needs to be done. **Read this file first, then the other two.**
>
> **Precedence if documents disagree:** this file > `MLOps_Project_Setup_and_Work_Division.md` (Phase 1 plan) > `PRD.md`. The PRD is the long-term design; the Phase 1 plan is the current scope; this file records corrections.
>
> **Date context:** written 14 Oct 2026. Dataset facts below were read on that date and may have changed.

---

## 1. Who and what

- **Who:** a team of 4 undergraduate students (Semester V, Department of AI & ML, CSE(AIML)), course **AI252TA, Machine Learning Operations**, tutorials 2026-27. Team member names are not recorded; refer to them as Member 1 to 4.
- **What:** a graded course project. The faculty circular lists 50 candidate projects; the team chose **#1, "AI-Powered Study Assistant for Engineering Students"** (SDG 4, Quality Education).
- **The point of the project is the MLOps lifecycle, not model accuracy.** The circular's design principle: every project must demonstrate the complete lifecycle (data exploration, feature engineering, experimentation, versioning/reproducibility, testing, model risk/security, registry, containers, CI/CD, deployment, monitoring, drift detection, retraining, governance). Choosing and *using* appropriate MLOps tools well is the largest rubric item.
- **Grading rubric (50 marks, reduced to 20 for internal assessment):**

| Criterion | Marks |
|---|---|
| Identification of problem and ML model | 5 |
| Design, development and evaluation of ML model | 10 |
| Selection and usage of appropriate MLOps tools | 15 |
| Presentation and reporting | 10 |
| Testing, evaluation and validation | 10 |

### Phases (from the circular)
| Phase | Content | Status |
|---|---|---|
| **1 – Development & Reproducibility** | Dataset acquisition/approval; data exploration; feature engineering; baseline models; Git repository; DVC dataset versioning; MLflow experiment tracking; initial model evaluation | **CURRENT. Only this is in scope.** |
| 2 – Productionization & Deployment | PyTest; model-risk/quality checks; MLflow model registry; FastAPI; Docker; GitHub Actions CI/CD; deployment | Not started. Do not build yet |
| 3 – Monitoring, Feedback & Governance | Evidently drift monitoring; Python logging; Prometheus/Grafana or Power BI; simulated production data; drift/degradation experiment; retraining trigger; model comparison; model card; final report and demo | Not started. Do not build yet |

**Rule for you:** if asked for something belonging to Phase 2 or 3 (API, Docker, CI, dashboard, drift, registry, retraining), say it is out of Phase 1 scope and offer to help only if the team explicitly says they are moving on. You may mention that a Phase 1 choice should not block a later phase (for example, keep training in scripts, not notebooks).

---

## 2. The project in one paragraph

Two small ML models linked by a topic taxonomy, not merged datasets:
- **Model 1 (topic classifier):** takes a technical question's text and predicts a coarse topic. Trained on a Stack Overflow/Stack Exchange-style dataset.
- **Model 2 (answer scorer):** takes a question, a student's short answer and a reference answer, and predicts a score in {0 = incorrect, 1 = partially correct, 2 = correct}. Trained on EngSAF.

The long-term system (PRD §5) would route a student's question through Model 1, then to Model 2 if the topic is supported. **None of that integration is in Phase 1.** In Phase 1 the two models are trained and evaluated independently.

---

## 3. Datasets: verified facts and unverified items

### 3.1 EngSAF (`IsmaelMousa/engsaf`): Model 2
**Verified (dataset card and first viewer page read on 14 Oct 2026):**
- 5.58k rows. Splits: train 4,740 (card text says 4,735 after cleaning) / validation 338 / test 506.
- Columns: `question`, `student_answer`, `reference_answer`, `mark_scheme`, `score` (int 0 to 2), `rationale`.
- `mark_scheme` is one constant string mapping 0/1/2 to incorrect/partial/correct. It carries no information; drop it from features.
- `rationale` is the grader's feedback text, derived from the label. **It must never be a model input** (target leakage).
- 107 distinct `question` values and 108 distinct `reference_answer` values in the viewer, so about one reference answer per question. The raw data reportedly has 119 questions.
- Class balance: the card describes the cleaned train set as nearly balanced, label 2 slightly over-represented, label 0 least. **Re-verify** with a count.
- Val/test were built by the dataset author from entries *not* semantically similar to train (embedding model `all-MiniLM-L6-v2`, FAISS, 90% cosine cut-off). Overlapping entries were merged back into train. The split is **not grouped by question**, so the same questions probably appear in train, val and test. **Verify** how many.
- The original paper's "unseen questions" split had only 12 unique questions. A question-held-out evaluation will be small and noisy.
- Licence conflict: Hub tag says `apache-2.0`; the card text says released strictly for academic and research purposes, no commercial use or redistribution without explicit permission. **Treat the stricter statement as binding.** Paper: arXiv 2407.12818 (AIED 2025).
- Subjects in the preview are mixed: operating systems, C++, hydrology, image processing, manufacturing, statistics, finance, machine learning, chemistry, materials, water treatment, space image compression.
- Observed in the preview: some rationales contradict the score (e.g. a rationale criticising the justification on a score-2 row). Treat as label noise; it limits achievable accuracy. Some rationales contain artefacts such as `</END>`.

### 3.2 StackPulse (`Omarrran/StackPulse_778K_QnA_Code_dataset`): Model 1
**Verified (only from search-result snippets; the dataset page itself returned HTTP 429 twice):**
- Stack Overflow programming Q&A. Tags on the Hub: stackoverflow, code, nlp, question-answering, programming, python.
- Hub licence tag: `apache-2.0`. The underlying Stack Overflow content has its own terms (attribution, share-alike). **Verify and record.**
- Raw data is stored as per-language JSONL files, e.g. `data/stackoverflow_java.jsonl` (203 MB). A Parquet conversion exists under `refs/convert/parquet`, with a `python/train` folder.
- The viewer lists about 1.95M rows although the title says 778K. Unexplained; **verify** (possibly several configs or files counted together).

**NOT verified (do not assume):**
- Column names, whether tags exist, whether there is a question title vs body, whether answers are included, whether code and HTML are present, language and date coverage, duplicates, class balance.

### 3.3 Dataset conflict between the PRD and the team's current choice
The PRD was written around **`habedi/stack-exchange-dataset`** (82,158 rows; columns `id`, `title`, `body`, `tags`, `label`; Computer Science, Data Science and Political Science Stack Exchange up to May 2021; Hub licence `cc-by-4.0`). Its "[Confirmed]" statements in PRD §4.1 and §4.2 refer to **that** dataset, not StackPulse. In particular, its `label` column means resolved/not resolved and is not a topic; tags are a space-separated string with multi-word tags.

The team then listed **StackPulse** as their dataset. Consequences:
- Model 1 becomes a **programming-topic** classifier (web, data/ML, databases, concurrency, systems, algorithms, testing, mobile, etc.), not a CS-theory classifier.
- PRD §4.2 items 1, 2 and 4 (label meaning, tag parsing, source site) do not transfer until StackPulse's schema is known.
- The PRD's Political-Science out-of-domain test no longer exists. If out-of-domain testing is ever needed, EngSAF's non-programming questions (hydrology, manufacturing, finance) are a natural substitute. This is a later-phase idea only.
- Overlap between StackPulse topics and EngSAF subjects is narrow (C++, OS-style, ML). Document as a limitation.

**Open decision D0:** keep StackPulse, or return to the PRD's dataset? If the faculty approved a specific dataset, that wins. Until D0 is made, write Model 1 code so the dataset name and column mapping live in `params.yaml` or `configs/`, not hard-coded.

### 3.4 You cannot assume data access
The assistant that prepared these documents could not download either dataset (its sandbox cannot reach huggingface.co). The first real task is for a teammate to run `datasets.load_dataset(...)` in Colab for both and commit the schema, row counts and `df.head()` to `docs/`. If you can run code with network access, do this and report the results rather than guessing.

---

## 4. Phase 1 scope in detail

### 4.1 Deliverables (what "done" means)
1. Both datasets documented: source, fields, licence, limitations, preprocessing. Dataset choice approved (D0).
2. EDA for both datasets (notebooks are fine for EDA only).
3. Feature engineering documented and implemented in scripts. No `rationale` or `mark_scheme` in any feature set. TF-IDF/scalers fit on train only.
4. Baselines and candidate models trained for both tasks, every run logged to MLflow with parameters, metrics, plots, the DVC data hash and the git commit.
5. Model 2 evaluated on (a) the provided split and (b) a question-grouped split.
6. Data and processed artifacts versioned with DVC; `dvc repro` works on a clean clone and reproduces metrics within a stated tolerance, verified by a member who did not write the pipeline.
7. Git repository with README, branch/PR workflow, and reviews.
8. Initial evaluation written up honestly, including limitations, label noise and the small question pool.

### 4.2 Explicitly NOT in Phase 1
FastAPI, Docker, GitHub Actions CI/CD, pytest suite, model registry, deployment, Evidently, Prometheus/Grafana/Power BI, drift or degradation experiments, retraining triggers, model card, integration logic (topic gate, coverage map, bank-question matching), abstention tuning, LLM-fallback study, fine-tuned transformer models (optional only if GPU time allows).

**Small exception that is not a violation:** a trivial `pytest`-free sanity check inside the pipeline (e.g. an assertion that scores are in {0,1,2}) is fine. A formal test suite is Phase 2.

### 4.3 Model 1: topic classifier
Input: cleaned question text (title + body if available). Output: one of 8 to 12 coarse topics defined in a versioned `topic_map.yaml` (tag phrase → topic). If a question's tags map to several topics, keep it only when one topic clearly dominates; otherwise drop it and log the count.

| ID | Candidate | Role |
|---|---|---|
| M1-A | Majority class | Floor |
| M1-B | Tag/keyword rules from `topic_map.yaml` | Non-ML baseline |
| M1-C | TF-IDF + Logistic Regression | Main baseline |
| M1-D | TF-IDF + Linear SVM | Strong linear |
| M1-E | TF-IDF + Complement Naive Bayes | Handles imbalance |
| M1-F | MiniLM sentence embeddings + Logistic Regression | Semantic features |

Metrics: macro-F1 (primary), per-class precision/recall, confusion matrix. Use a stratified sample of 100k to 200k rows (decision D2) so free Colab can cope; version the sample in DVC.

Caution: if tags are used both to *build the label* and as *features*, the model learns the rule instead of the topic. Use only title/body as input.

### 4.4 Model 2: answer scorer
Input: question, student answer, reference answer. Output: score in {0,1,2}.

| ID | Candidate | Role |
|---|---|---|
| M2-A | Majority class | Floor |
| M2-B | Cosine-similarity threshold(s) between student and reference embeddings | Non-learned baseline |
| M2-C | Features (cosine, \|u−v\|, u⊙v, length ratios) + Logistic Regression | Main baseline |
| M2-D | Same features + XGBoost | Non-linear |
| M2-E | Same features + ordinal regression variant | Scores are ordered |

Embeddings: Sentence-Transformers (e.g. `all-MiniLM-L6-v2`). Required checks:
- **With vs without question text** as a feature (the question pool is tiny; models can memorise questions).
- A **length-only** sanity baseline.
- Results on the provided split **and** a **question-grouped split** (whole questions held out) plus `GroupKFold` for tuning. Report the gap; do not hide it.
Metrics: macro-F1 and Quadratic Weighted Kappa (primary), per-class recall, confusion matrix, MAE (secondary). Report spread or confidence intervals for the grouped split because it is small.

### 4.5 Model selection protocol
1. Fix primary metric and tie-breakers **before** looking at any test set.
2. Compare on validation only (grouped CV for Model 2).
3. Log every run to MLflow.
4. Evaluate the test set **once**, for the shortlisted best model(s).
5. Record winner and runner-up in the Phase 1 report. The team wants several models trained and the best chosen, not just one.
6. Proposed targets in the PRD (e.g. macro-F1 ≥ 0.70 for Model 1) are **proposals**, to be ratified only after baselines exist. Never state performance numbers you have not computed.

---

## 5. Work division (Phase 1)

| Member | Workstream | Main tasks | Deliverables |
|---|---|---|---|
| **1** | Topic data + Model 1 | Settle D0; profile Stack data; build `topic_map.yaml`; clean text (strip HTML, handle code); stratified sample; train M1-A to M1-F | EDA report, taxonomy, preprocessing + training scripts, Model 1 comparison table |
| **2** | Answer data + Model 2 | Profile EngSAF (score distribution, per-question counts, split overlap, lengths, duplicates); grouped split; features; train M2-A to M2-E with checks | EDA report, feature code, grouped-split code, Model 2 results on both splits, initial error analysis |
| **3** | Git + DVC + reproducibility | Repo skeleton, branch rules, PR template; DVC remote and tracked data; `dvc.yaml` stages (ingest → clean → features → train → evaluate); `params.yaml`; pinned deps; Colab template; clean-clone reproducibility check | Working `dvc repro`, versioned data, README, reproducibility record |
| **4** | MLflow + evaluation harness + dataset approval | Shared MLflow server; naming conventions; shared metrics module (macro-F1, QWK, confusion-matrix plots, logging helper); `docs/data_licences.md`; MLflow comparison tables; Phase 1 report | Experiment history, evaluation module, licence/approval doc, report draft |

Review pairs: Member 1 ↔ 2; Member 3 ↔ 4. Every PR gets one review from a different member. Each member needs meaningful commits in their own area.

### 5.1 Interface agreements (to be fixed in Week 1 in `docs/contracts.md`)
- Processed-data paths and column names.
- `params.yaml` keys.
- MLflow experiment names: `topic-baselines`, `answer-scorer-baselines`.
- Shared metrics function signature, for example:
```python
def evaluate(y_true, y_pred, y_proba=None, labels=None) -> dict:
    """Returns {'macro_f1':..., 'qwk':... (Model 2 only), 'per_class':..., 'confusion':...}"""
```
- Sketch of `dvc.yaml` stages (names are suggestions): `ingest_stack`, `ingest_engsaf`, `clean_stack`, `features_engsaf`, `train_topic`, `train_scorer`, `evaluate`.
- Suggested repo layout:
```
src/        data/, features/, models/, evaluation/
configs/    params.yaml (or root), topic_map.yaml
data/       raw/, processed/   (DVC-tracked, git-ignored)
notebooks/  EDA only
docs/       data_licences.md, contracts.md, mlflow_conventions.md, phase1_report.md
```
- Suggested MLflow params to log on every run: dataset name and DVC hash, split type, random seed, model name and hyperparameters, feature set flags (e.g. `use_question_text`), git commit.

---

## 6. Decisions

| ID | Decision | Status / recommendation |
|---|---|---|
| D0 | Which Stack dataset | **Open.** Follow faculty approval; else StackPulse with a tag-based taxonomy if tags exist |
| D2 | Model 1 data size | Stratified 100k to 200k sample; version in DVC |
| D5 | Shared storage for DVC and MLflow | **Open.** DagsHub can host both (verify current free limits). Alternative: Google Drive DVC remote + MLflow on one member's machine (fragile) |
| D6 | Python and pins | Python 3.11; pinned `requirements.txt` |
| D1 (later) | Where the reference answer comes from in the integrated system | Deferred to Phase 2/3. Do not work on it now |

---

## 7. Setup checklist (Week 0)

- [ ] GitHub repo created, all four have access; Hugging Face accounts + read token (Colab secret and local `.env`, never committed)
- [ ] Branch rules and PR template agreed
- [ ] Repo skeleton from §5.1 committed; `.gitignore` covers `data/`, `mlruns/`, `.env`
- [ ] `dvc init`; remote configured; all four confirm `dvc pull` works from a clean clone
- [ ] MLflow server reachable from Colab and laptops; each member logs a dummy run
- [ ] Colab template: clone repo, install pinned deps, set MLflow tracking URI, run a script
- [ ] Schema, row counts and sample rows of both datasets committed to `docs/`
- [ ] `docs/data_licences.md` drafted
- [ ] `docs/contracts.md` written

## 8. Timeline (relative; align to the faculty's review date, which the circular does not give)

| Week | Focus |
|---|---|
| 0 | Decisions D0, D2, D5, D6; setup checklist; dataset schema check |
| 1 | EDA on both; licence doc; DVC + MLflow working for all four; taxonomy v0; majority/rule/cosine baselines |
| 2 | Features; remaining models; grouped split; MLflow comparison; one clean-clone `dvc repro`; single test-set run for shortlisted models; Phase 1 report |

---

## 9. Rules for how you (the LLM) should help

1. **Honesty over polish.** Never invent dataset facts, metrics, library behaviour or results. If a fact is not in §3 as verified, say it is unverified and suggest how to check it (code snippet or what to look at).
2. **Respect the scope.** Only help with Phase 1 unless told otherwise (§1).
3. **Reproducibility first.** Prefer scripts with `params.yaml` and fixed seeds over notebook-only code. Suggest DVC stages and MLflow logging in any training code you write.
4. **Guard against leakage.** Fit vectorizers and scalers on train only; never use `rationale`, `mark_scheme` or tag-derived labels as features; keep question-grouped splits for Model 2; flag any step that could leak.
5. **Use macro metrics** for imbalanced classes; QWK for Model 2. Do not report accuracy alone.
6. **Respect licences.** EngSAF is academic/research only; keep raw data out of Git and public places. StackPulse's underlying Stack Overflow terms must be checked.
7. **Keep it small enough for free Colab.** Prefer samples, CPU-friendly models, and pinned lightweight dependencies. Treat transformer fine-tuning as optional.
8. **Do not decide for the team.** For open decisions (D0, D5), present options and trade-offs and let them decide.
9. **Give runnable, concise code** with the exact library calls the team would use (`datasets`, `pandas`, `scikit-learn`, `sentence-transformers`, `xgboost`, `mlflow`, `dvc`), and say which member's workstream the code belongs to.
10. **If asked "what next?"**, check the setup checklist (§7) and the acceptance checklist (§10) and name the first unchecked item.

## 10. Phase 1 acceptance checklist

- [ ] Both datasets documented (source, fields, licence, limitations) and dataset choice approved
- [ ] EDA completed for both datasets
- [ ] Features documented; no `rationale` or `mark_scheme` in any feature set
- [ ] Baselines and candidate models trained for both tasks; each run in MLflow with data hash and commit
- [ ] Model 2 reported on provided and question-grouped splits, with and without question text
- [ ] Data versioned in DVC; clean-clone `dvc repro` reproduces metrics within tolerance
- [ ] Test set evaluated once, for shortlisted models only
- [ ] Initial evaluation written up, including limitations and label noise; no claim exceeds what was run

## 11. Known risks

| Risk | Mitigation |
|---|---|
| StackPulse schema or tags unsuitable for a topic label | Week-0 audit; fall back to the PRD's dataset |
| Tiny EngSAF question pool inflates Model 2 results | Grouped split; with/without question text; report both with spread |
| Label noise in EngSAF caps accuracy | Document as ceiling; do error analysis on contradictory rationales |
| Remote MLflow/DVC unreachable from Colab | Test with a dummy run in Week 0 |
| Free Colab limits | Sample for Model 1; skip transformers if needed |
| Licence misuse (EngSAF academic-only) | Raw data out of Git; document use |

## 12. Glossary

- **EDA:** exploratory data analysis. **DVC:** Data Version Control (versions data and pipelines on top of Git). **MLflow:** experiment tracking (params, metrics, artifacts). **QWK:** Quadratic Weighted Kappa, an agreement metric that penalises larger score errors more. **Macro-F1:** F1 averaged equally over classes. **GroupKFold:** cross-validation that keeps all rows of a group (here, a question) in the same fold. **Leakage:** information in training features that would not be available at prediction time, or overlap between train and evaluation data. **OOD:** out-of-domain.
