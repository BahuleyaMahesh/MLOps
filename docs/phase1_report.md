# Phase 1 Evaluation & Reproducibility Report

**Project:** AI-Powered Study Assistant for Engineering Students<br>
**Course:** AI252TA — Machine Learning Operations (2026-27) · Project #1 (SDG 4: Quality Education)<br>
**Authors:** Team 4 Members (Roles: Member 1, Member 2, Member 3, Member 4)<br>
**Phase:** Phase 1 — Development & Reproducibility<br>
**Status:** Report Outline & Empirical Template (Pre-computation Scaffold)

> **Transparency Note:** In accordance with course guidelines, this document serves as the standardized report template. All quantitative metrics, baseline scores, and statistical distributions below are structured to receive empirical results from pipeline runs; no synthetic or unverified scores have been fabricated.

---

## 1. Executive Summary & Problem Formulation

### 1.1 Problem Statement
The goal is to develop an AI-powered assistant to assist engineering students with technical questions and provide objective automated grading for short technical answers.
In **Phase 1 (Development & Reproducibility)**, the system focuses strictly on establishing two independent, fully reproducible ML baselines with end-to-end data versioning (DVC) and experiment tracking (MLflow):

1. **Model 1 (Topic Classifier):** Given technical question text, predict a coarse engineering/programming topic from an 8–12 topic taxonomy.
2. **Model 2 (Student Answer Scorer):** Given an exam question, a student's short answer, and a reference answer, predict an ordinal evaluation score in `{0: incorrect, 1: partially correct, 2: correct}`.

*(System integration, topic-gated routing, API endpoints, and containerization are explicitly deferred to Phase 2/3).*

### 1.2 Evaluation Rubric Mapping
This report addresses the Phase 1 criteria (50 marks scaled to 20 for internal assessment):
- **Problem & Model Identification (5 marks):** Task formulation, taxonomy definition, and baseline model architectures.
- **Design & Development (10 marks):** Pipeline design, feature engineering, and anti-leakage barriers.
- **MLOps Tooling (15 marks):** DVC data versioning, MLflow tracking, parameter management, and reproducibility.
- **Testing, Validation & Error Analysis (10 marks):** Provided vs question-grouped evaluation, ablation studies, and label noise analysis.
- **Reporting & Governance (10 marks):** Licensing compliance, data contracts, and transparent limitation reporting.

---

## 2. Dataset Acquisition, Provenance & Licensing

### 2.1 Model 1: Technical Q&A Dataset
- **Candidate Dataset:** `Omarrran/StackPulse_778K_QnA_Code_dataset` (provisional; Decision D0 pending faculty sign-off).
- **Alternative Considered:** `habedi/stack-exchange-dataset` (PRD baseline).
- **Licensing & Compliance:** Raw Stack Overflow posts fall under CC BY-SA terms requiring attribution. Formal faculty approval is pending.
- **Sampling Strategy:** 100,000-row stratified sample (Decision D2) to ensure reproducibility on constrained hardware.

### 2.2 Model 2: EngSAF Dataset
- **Source:** `IsmaelMousa/engsaf` (arXiv:2407.12818, AIED 2025).
- **Scope & Size:** 5,580 rows across engineering disciplines (operating systems, C++, hydrology, image processing, chemistry, thermodynamics, statistics).
- **Licensing Restriction:** Academic and research use only; strictly no commercial redistribution. All raw and processed data are tracked exclusively via access-controlled DVC remotes and excluded from Git.

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Model 1 (StackPulse / Stack Exchange) Profiling
- *[To be populated by Member 1 post-profiling]:*
  - Schema verification (confirmed columns: title, body, tags).
  - Class frequency and tag co-occurrence matrix.
  - Text length distributions (character/token counts before and after code snippet filtering).
  - Multi-topic question frequency and drop rates.

### 3.2 Model 2 (EngSAF) Profiling
- *[To be populated by Member 2 post-profiling]:*
  - Score distribution across classes `{0, 1, 2}` in provided train (4,740), val (338), and test (506) splits.
  - Total count of distinct questions (~107-119 distinct prompts).
  - Analysis of question overlap across provided splits (estimating semantic vs literal overlap).
  - Student answer length distribution across score tiers.

---

## 4. Feature Engineering & Target Leakage Prevention

### 4.1 Strict Leakage Prevention Barriers
- **Exclusion of `rationale`:** The dataset contains a grader's explanation written with post-hoc knowledge of the score. Including `rationale` would result in catastrophic target leakage. It is strictly dropped during data ingestion.
- **Exclusion of `mark_scheme`:** Constant string mapping that carries no predictive signal.
- **Isolation of Tag Information (Model 1):** Tags are used solely for ground-truth label assignment and are excluded from model input text.

### 4.2 Feature Pipelines
- **Model 1 Features:** Cleaned question text (HTML stripped, code blocks converted to `<CODE>` tokens), unigram/bigram TF-IDF representations (max 10,000 features), and dense Sentence-BERT (`all-MiniLM-L6-v2`) embeddings.
- **Model 2 Features:** Dense embeddings computed for student answer ($u$) and reference answer ($v$), complemented by pairwise similarity metrics:
  $$\text{Cosine Similarity} = \frac{u \cdot v}{\|u\| \|v\|}$$
  $$\text{Manhattan Distance} = \|u - v\|_1$$
  $$\text{Elementwise Product} = u \odot v$$
  along with student-to-reference text length ratios.
- **Split Integrity:** All scalers, normalizers, and TF-IDF vectorizers are fit exclusively on the training split.

---

## 5. Model Candidate Evaluation & Baseline Comparisons

### 5.1 Model 1: Question-Topic Classifier

| Candidate ID | Model Architecture | Feature Representation | Val Macro-F1 | Val Precision | Val Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M1-A** | Majority Class | N/A (Floor) | *[Pending]* | *[Pending]* | *[Pending]* |
| **M1-B** | Keyword Rules | Tag Taxonomy Match | *[Pending]* | *[Pending]* | *[Pending]* |
| **M1-C** | Logistic Regression | TF-IDF (1-2 ngrams) | *[Pending]* | *[Pending]* | *[Pending]* |
| **M1-D** | Linear SVM | TF-IDF (1-2 ngrams) | *[Pending]* | *[Pending]* | *[Pending]* |
| **M1-E** | Complement Naive Bayes | TF-IDF (Imbalance tuned)| *[Pending]* | *[Pending]* | *[Pending]* |
| **M1-F** | Logistic Regression | MiniLM Dense Embeddings| *[Pending]* | *[Pending]* | *[Pending]* |

*Primary metric: Macro-F1 (unweighted across classes).*

### 5.2 Model 2: Student Answer Scorer (Provided Split)

| Candidate ID | Model Architecture | Feature Representation | Val Macro-F1 | Val QWK | Val MAE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M2-A** | Majority Class | N/A (Floor) | *[Pending]* | *[Pending]* | *[Pending]* |
| **M2-B** | Cosine Threshold | MiniLM Embedding Cosine| *[Pending]* | *[Pending]* | *[Pending]* |
| **M2-C** | Logistic Regression | MiniLM Pairwise + Length| *[Pending]* | *[Pending]* | *[Pending]* |
| **M2-D** | XGBoost Classifier | MiniLM Pairwise + Length| *[Pending]* | *[Pending]* | *[Pending]* |
| **M2-E** | Ordinal Logistic Reg | MiniLM Pairwise + Length| *[Pending]* | *[Pending]* | *[Pending]* |

*Primary metrics: Macro-F1 and Quadratic Weighted Kappa (QWK).*

---

## 6. Question-Grouped Evaluation & Generalization Analysis (Model 2)

### 6.1 Motivation for Grouped Cross-Validation
Because EngSAF contains only ~107-119 distinct question prompts, models evaluated on standard random splits may inadvertently memorize specific question phrasing rather than learning generic student-reference semantic alignment.

### 6.2 Provided Split vs Question-Grouped Evaluation

| Model Candidate | Metric | Provided Split (Val) | 5-Fold GroupKFold (Mean ± Std) | Generalization Gap |
| :--- | :--- | :--- | :--- | :--- |
| **M2-C (LogReg)** | Macro-F1 | *[Pending]* | *[Pending]* | *[Pending]* |
| **M2-C (LogReg)** | QWK | *[Pending]* | *[Pending]* | *[Pending]* |
| **M2-D (XGBoost)** | Macro-F1 | *[Pending]* | *[Pending]* | *[Pending]* |
| **M2-D (XGBoost)** | QWK | *[Pending]* | *[Pending]* | *[Pending]* |

### 6.3 Ablation Study: Question Text Inclusion
- **Without Question Text:** ($u = \text{student}$, $v = \text{reference}$): Forces model to evaluate answer alignment independently of question memorization.
- **With Question Text:** ($u = \text{question} + \text{student}$, $v = \text{question} + \text{reference}$): Tests whether question context provides genuine disambiguation or spurious cues.
- *[Empirical ablation findings to be inserted post-run]*

---

## 7. MLOps Tooling, Reproducibility & Pipeline Audit

### 7.1 Pipeline Versioning with DVC
- DVC pipeline stages defined in `dvc.yaml`: `ingest_stack`, `ingest_engsaf`, `clean_stack`, `features_engsaf`, `train_topic`, `train_scorer`, `evaluate`.
- All raw and intermediate datasets tracked via `.dvc` files.
- Reproduction verified via:
  ```bash
  dvc repro
  ```

### 7.2 Experiment Tracking with MLflow
- Central tracking server configured for team access.
- Every experiment logs: Git commit SHA, DVC data hash, random seed, model hyperparameters, macro-F1, QWK, and confusion matrix artifacts.

### 7.3 Clean-Clone Verification Protocol
- Step-by-step audit performed by Member 3: Clean git clone, environment creation (`python 3.11`), dependency installation from pinned `requirements.txt`, `dvc pull`, and `dvc repro`.

---

## 8. Limitations, Known Risks & Error Analysis

1. **Label Noise in EngSAF:**
   - Preliminary inspection reveals instances where grader rationales appear inconsistent with assigned scores (e.g., negative critique on score-2 answers). This inherent label noise imposes an upper bound on model convergence.
2. **Domain Mismatch:**
   - StackPulse covers broad programming Q&A, while EngSAF covers formal engineering exam questions across multiple disciplines (hydrology, materials, chemistry). There is narrow subject overlap.
3. **Small Question Pool:**
   - With ~119 distinct questions in EngSAF, question-held-out splits have higher statistical variance; confidence intervals must be reported alongside point estimates.

---

## 9. Conclusions & Phase 2 Roadmap

- *[To be drafted upon completion of model training and selection]*
- Candidate model shortlisted for productionization in Phase 2 (FastAPI packaging, Docker containerization, and automated PyTest validation).
