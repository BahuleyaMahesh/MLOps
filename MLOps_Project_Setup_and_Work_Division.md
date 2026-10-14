# AI-Powered Study Assistant: Setup Plan and Work Division

**Course:** AI252TA, Machine Learning Operations (2026-27) · **Project #1**, SDG 4 · **Team:** 4 members
**Stage:** pre-project setup (no modelling yet) · **Based on:** `PRD.md` v0.1 + faculty circular

> Tags used below: **[Confirmed]** = read from a source on 14 Oct 2026 · **[Verify]** = must be checked by the team · **[Decision]** = needs team sign-off.

---

## 1. Can we access the datasets?

| Dataset | What I could check | Status |
|---|---|---|
| **EngSAF** (`IsmaelMousa/engsaf`) | Full dataset card and first page of the viewer | **[Confirmed]** readable. 5.58k rows; train 4.74k / val 338 / test 506. Columns: `question`, `student_answer`, `reference_answer`, `mark_scheme`, `score` (0/1/2), `rationale`. 107 distinct questions in the viewer. |
| **StackPulse** (`Omarrran/StackPulse_778K_QnA_Code_dataset`) | The page itself returned an HTTP 429 (rate limit) twice. Only search-result snippets were readable | **Partly confirmed.** It is a **Stack Overflow programming Q&A** dataset (tags: stackoverflow, code, python, programming). Hub licence tag: apache-2.0. Raw files are per-language JSONL (e.g. `stackoverflow_java.jsonl`, 203 MB). The viewer shows ~1.95M rows although the title says 778K. **Column names and schema are not verified.** |

I also cannot download either dataset from my sandbox (huggingface.co is not on its network allow-list). **A teammate must run `load_dataset(...)` in Colab and paste the schema + `df.head()` into the repo as the first task.**

### Important: this is not the dataset the PRD assumed
The PRD was written around `habedi/stack-exchange-dataset` (Computer Science / Data Science / Political Science Stack Exchange, with a tags column). **StackPulse is Stack Overflow code questions.** That changes Model 1:

- The question domain is *programming* (languages, frameworks, libraries), not CS theory (OS, networks, DBMS).
- Political Science rows no longer exist, so out-of-domain (OOD) negatives need another source. **Good option: the non-CS EngSAF questions** (hydrology, manufacturing, finance, image processing, chemistry) are natural OOD negatives for a programming-topic classifier.
- If the file-per-language layout holds, the language is a free label, but it is also a *shallow* one. A better target is a topic taxonomy built from tags (e.g. web, data/ML, databases, concurrency, OS/systems, algorithms, testing, mobile). **[Verify]** tags exist in the schema.
- Overlap with EngSAF is narrow (C++, OS-style questions, ML, maybe crypto). The coverage map will therefore be small. This is a limitation to document, not hide.

**[Decision D0]** Pick one: (a) keep StackPulse and update the PRD, or (b) switch back to `habedi/stack-exchange-dataset` as in the PRD. If the faculty approved a specific dataset, follow that. Decide in the first meeting; everything in Workstream 1 depends on it. The rest of this document works for either.

### What the EngSAF page already tells us
- **[Confirmed]** The author re-split val/test to be *semantically dissimilar* from train (90% embedding-similarity cut-off). Test scores are therefore less inflated than the PRD feared, but the split is **not grouped by question**, so you still need your own grouped split.
- **[Confirmed]** The original paper's "unseen questions" split had only **12 unique questions**. A question-held-out evaluation is very small and noisy. Report confidence intervals and use `GroupKFold`.
- **[Confirmed]** The card says *academic and research use only; no commercial use or redistribution without permission*, while the Hub tag says apache-2.0. Follow the stricter text.
- **Observed in the preview:** some rationales contradict the score (e.g. a rationale saying the justification is inaccurate on a score-2 row). Treat this as label noise and a ceiling on achievable accuracy. `rationale` must never be a model input.
- Subjects in the preview: operating systems, C++, hydrology, image processing, manufacturing, statistics, finance, ML, chemistry, materials, water treatment, space image compression. Most are not programming topics.

---

## 2. Gaps between the PRD and the circular (fix before starting)

The circular grades **tool selection and usage at 15/50 marks, the biggest block**. The PRD covers most of it but misses some items:

| Circular requirement | In PRD? | Action |
|---|---|---|
| Deployed prediction service | Only `docker run` locally | **Add** a real deployment (see D3). |
| Prometheus/Grafana **or** Power BI dashboard | Evidently reports only | **Add** a live dashboard (see D4). |
| Retraining **trigger** + redeployment evidence | PRD says manual script only, no automatic triggers | **Conflict.** Add a simple threshold-based trigger (drift flag → GitHub Actions retrain → compare → promote/reject). Keep it small but real. |
| Deliberately introduced drift/degradation scenario | "Simulated shifted batch" | Make it an explicit named experiment with before/after metrics (see §5.3). |
| Model-risk and **security** considerations | Responsible AI covered, security not | **Add** a short security section: input validation, size limits, dependency scan (`pip-audit`), secrets handling, PII in logs. |
| Feedback loop | Not mentioned | Add a `/feedback` endpoint that logs "was this score fair?" (stored, used in the retrain demo). |
| Model registry | MLflow registry | Fine. Use aliases (`candidate`, `production`). |
| Governance / model card | Yes | Fine. |

---

## 3. Decisions needed before starting

| ID | Decision | Recommendation |
|---|---|---|
| D0 | Which Stack dataset (see §1) | Whatever the faculty approved; else StackPulse with a tag-based taxonomy |
| D1 | Where the reference answer comes from (PRD §5) | Match the question to the nearest EngSAF bank question; return "scoring unavailable" otherwise. Optional "teacher mode" where the user supplies the reference |
| D2 | Data size for Model 1 | Stratified sample of 100k to 200k rows. The full dataset is too big for free Colab and gives little gain over a sample |
| D3 | Deployment target | Hugging Face Spaces (Docker) or Render. **[Verify]** current free-tier limits and cold-start behaviour. Fallback: local Docker + a public tunnel for the demo |
| D4 | Dashboard | Prometheus + Grafana via `docker-compose` (FastAPI exposes `/metrics`). Power BI only if the team has it working already |
| D5 | Shared storage | DVC remote + MLflow server. **DagsHub** can host both (**[Verify]** free limits). Alternative: Google Drive remote with a service account + MLflow on a team member's machine (fragile) |
| D6 | Python / library pins | Python 3.11; one `requirements.txt` split into `base`, `train`, `serve`, `dev` |

---

## 4. Setup checklist (do now, before any modelling)

**Everyone**
- [ ] GitHub accounts, Hugging Face accounts + read token (store as Colab secret and GitHub secret, never in code)
- [ ] Agree branch/PR rules: feature branches, 1 reviewer from a different member, squash merge
- [ ] Weekly sync + shared task board (GitHub Projects), one issue per deliverable in §6
- [ ] Read the dataset licences and write `docs/data_licences.md` (Workstream 2 drafts, all review)

**Workstream 3 owner leads, all review**
- [ ] Create the repo and skeleton: `src/`, `tests/`, `data/` (DVC-tracked), `configs/`, `notebooks/` (EDA only), `docs/`, `.github/workflows/`, `Dockerfile`, `dvc.yaml`, `params.yaml`
- [ ] `pre-commit` with ruff; `.gitignore` covers `data/`, `mlruns/`, `.env`
- [ ] Protect `main` (require PR, require CI check)
- [ ] `dvc init`; configure the remote; teammates confirm `dvc pull` works from a clean clone
- [ ] MLflow tracking server reachable from Colab and laptops; test one dummy run from each member
- [ ] Hello-world CI: lint + one passing pytest
- [ ] Colab template that does `git clone`, installs pinned deps, sets MLflow URI, runs a script (no notebook-only training)

**Workstreams 1 and 2**
- [ ] First data load in Colab; commit schema + sample rows + row counts to `docs/`
- [ ] Write the interface contracts (§6) in `docs/contracts.md`

---

## 5. Model plan: train several, pick the best

### 5.1 Selection protocol (write this down before training)
1. Fix the primary metric, tie-breakers and deployment limits **before** looking at any test set.
2. Tune and compare on **validation** only (grouped CV for Model 2). Every run is logged to MLflow with params, data version (DVC hash) and git commit.
3. Apply the deployment constraints: Docker image size, CPU-only inference, p95 latency per request, memory. A heavier model must win by a margin that justifies the cost.
4. The test set is evaluated **once**, for the shortlisted finalist(s).
5. Register the winner as `production` and the runner-up as `candidate`. The runner-up becomes the challenger in the promotion-gate and retrain demos.

### 5.2 Candidate models

**Model 1: question topic classifier** (Workstream 1)

| ID | Model | Purpose |
|---|---|---|
| M1-A | Majority class | Floor |
| M1-B | Tag/keyword rules from `topic_map.yaml` | Non-ML baseline |
| M1-C | TF-IDF + Logistic Regression | Main baseline |
| M1-D | TF-IDF + Linear SVM (calibrated) | Strong linear model |
| M1-E | TF-IDF + Complement Naive Bayes | Fast, handles imbalance |
| M1-F | Sentence-embedding (MiniLM) + Logistic Regression | Semantic features, small |
| M1-G | Fine-tuned DistilBERT *(optional, Colab GPU)* | Upper bound; keep only if latency/size allow |

All with calibrated probabilities and an abstention threshold chosen on validation. Metrics: macro-F1, per-class P/R, confusion matrix, calibration curve, abstention-vs-accuracy curve, OOD rejection rate.

**Model 2: answer scorer** (Workstream 2)

| ID | Model | Purpose |
|---|---|---|
| M2-A | Majority class | Floor |
| M2-B | Cosine-similarity threshold(s) | Non-learned baseline |
| M2-C | Features (cosine, \|u−v\|, u⊙v, length ratios) + Logistic Regression | Main baseline |
| M2-D | Same features + XGBoost (and LightGBM) | Non-linear |
| M2-E | Same features + ordinal regression variant | Scores are ordered 0<1<2 |
| M2-F | Cross-encoder fine-tune (MiniLM or DeBERTa-v3-small) on [question, reference, student] *(Colab GPU)* | Likely strongest; check size/latency |

Required ablations: **with vs without the question text** (the question pool is tiny, so models can memorise it), and **length-only** and **copy-the-question** sanity baselines. Metrics: macro-F1, QWK, per-class recall, MAE; report on (a) the provided split and (b) a question-grouped split.

### 5.3 The drift / degradation experiment (circular requirement)
Define it up front so the work is planned, not improvised:
- **Model 1 drift:** feed a batch of non-programming questions (EngSAF non-CS questions) and a batch with a different style (very short or non-English). Expect confidence to drop and abstention to rise.
- **Model 2 degradation:** feed answers from questions outside the bank, answers with shifted length distribution, and an injected label-noise set once feedback labels exist.
- **Detect:** Evidently data-drift + prediction-drift reports; custom Prometheus counters (abstention rate, mean confidence).
- **Trigger → respond:** drift above threshold sets a flag → GitHub Actions retrain workflow → new model compared with `production` on fixed criteria → promoted or rejected, with the decision recorded.
- Label everything *simulated* in the report. There are no real users.

---

## 6. Work division (4 workstreams)

**Interface contracts (agree in Week 1, change only via PR to `docs/contracts.md`):**
`topic_map.yaml` format · `coverage_map.csv` columns (`question_id`, `question_text`, `topic`) · `predict_topic(text) -> {topic, confidence, abstain}` · `score_answer(question, student, reference) -> {score, probs, label}` · MLflow registered names `topic-clf` and `answer-scorer` · log record schema (JSON fields) for requests.

### Member 1: Workstream 1, Topic data + Model 1
- Resolve D0; profile the Stack dataset; verify schema, tags, duplicates, HTML/code handling
- Build `topic_map.yaml` (tag phrase → topic, 8 to 12 topics) and review it manually with Member 2
- Clean text (strip HTML, handle code blocks), build stratified sample (D2)
- Train M1-A to M1-F (and M1-G if time), calibrate, set abstention threshold on validation, OOD test using EngSAF non-CS questions
- Error analysis, top terms per topic, model-card section
- **Deliverables:** EDA report, taxonomy, training script, Model 1 comparison table in MLflow, final metrics, model card section
- **Needs from others:** scaffolding from Member 3; taxonomy review from Member 2

### Member 2: Workstream 2, Answer data + Model 2
- Profile EngSAF fully: score distribution, per-question counts, question overlap across splits, answer-length stats, duplicates
- Resolve licence conflict; write `docs/data_licences.md`
- Manually label the ~107 questions into the Member 1 taxonomy → `coverage_map.csv`; mark questions with no matching topic as "unsupported"
- Build grouped split + `GroupKFold`; feature code; train M2-A to M2-F with ablations
- Error analysis per question/topic, label-noise review, human-grader ceiling note, model-card section
- **Deliverables:** EDA report, coverage map, feature code, Model 2 results on both split types, error analysis, model card section
- **Needs from others:** taxonomy from Member 1; scaffolding from Member 3

### Member 3: Workstream 3, MLOps backbone
*(starts first, so finish setup in §4 within Week 1)*
- Repo, DVC remote and `dvc.yaml` stages (ingest → clean → features → train → evaluate), `params.yaml`
- MLflow conventions (experiment names, tags, required params, registry aliases); document in `docs/mlflow_conventions.md`
- Pytest suite for data validation (schema, no `rationale` in features, score range), tag mapping, features, prediction shape/range, edge cases (empty text, very long text, HTML, non-English)
- GitHub Actions: lint + test on every PR; build-Docker job; **retrain workflow** and **promotion gate** (compare candidate vs production on fixed criteria)
- Security/model-risk checks (`pip-audit`, secrets scan, input-limit tests)
- **Deliverables:** reproducible `dvc repro`, CI badge, test suite, conventions doc, promotion checklist, retrain workflow
- **Needs from others:** model code from Members 1 and 2 to test; drift flag format from Member 4

### Member 4: Workstream 4, Integration, serving, monitoring
- Implement the §5 workflow of the PRD: topic gate → coverage gate → bank-question matching (τ tuned on validation) → scoring; each failure branch returns an explicit message
- FastAPI: `/predict-topic`, `/score-answer`, `/feedback`, `/health`, `/metrics`; request validation and size limits
- Dockerfile + `docker-compose` (API + Prometheus + Grafana); deployment to the D3 target
- Structured request logging; Evidently reports; Grafana dashboard; simulated-drift generator (§5.3)
- End-to-end tests and the validation report (each unsupported/low-confidence/no-match case has a passing test)
- LLM-fallback study document (investigation only, not wired into the default path)
- **Deliverables:** running container + deployed service, integration tests, dashboard, drift report, validation report, fallback study
- **Needs from others:** models + `coverage_map` from Members 1 and 2; CI from Member 3

### Shared work (split by section owner, one editor)
| Item | Owner of draft | Reviewer |
|---|---|---|
| Model card | Members 1 and 2 (own sections) | Member 3 |
| Governance, limitations, security section | Member 3 | Member 4 |
| Final report assembly + evaluation-vs-validation write-up | Member 4 | All |
| Demo script + slides | Rotate; each presents their own part | All |

**Balance note:** Members 1 and 2 carry heavy modelling early; Members 3 and 4 carry heavy tooling and integration later. If Member 1 finishes early, they should take the Model 1 pieces of the drift experiment and the `/metrics` custom counters.

---

## 7. Timeline mapped to the circular's three review phases

Week numbers are relative; align them to your faculty's review dates (the circular does not give dates).

| Week | Focus | Circular phase |
|---|---|---|
| 0 (now) | Setup §4, decisions D0 to D6, access check, contracts | n/a |
| 1 | Profiling, licences, taxonomy v0, repo + DVC + MLflow + CI skeleton, baselines M1-A/B/C, M2-A/B/C | **Phase 1:** Development and Reproducibility |
| 2 | Remaining candidate models, coverage map, D1 integration design, first MLflow comparison tables, model selection | Phase 1 review ready |
| 3 | Tests, registry, FastAPI, Docker, CI/CD, deployment, integration logic | **Phase 2:** Productionization and Deployment |
| 4 | Logging, Evidently, dashboard, drift experiment, retrain trigger, promote/reject, model card, final report, demo | **Phase 3:** Monitoring, Feedback and Governance |

---

## 8. Alignment with the evaluation rubric (50 marks)

| Rubric item | Marks | Where it is covered |
|---|---|---|
| Problem and ML model identification | 5 | PRD §1 to §4; §1 of this file (dataset fit) |
| Design, development, evaluation of ML model | 10 | §5 candidate models, ablations, selection protocol |
| Selection and usage of MLOps tools | 15 | Git, DVC, MLflow, pytest, GitHub Actions, Docker, FastAPI, Evidently, Prometheus/Grafana |
| Presentation and reporting | 10 | Model card, final report, demo |
| Testing, evaluation, validation | 10 | pytest suite, grouped splits, validation report, drift experiment |

---

## 9. What can be done now vs later

| Can be done now | Needs a decision first | Blocked until later |
|---|---|---|
| Repo, branch protection, CI skeleton | D0 dataset choice | Final taxonomy (needs profiling) |
| DVC remote + MLflow server (D5 first) | D3 deployment target | Coverage map (needs taxonomy) |
| Load both datasets in Colab, record schema | D1 reference-answer source | Abstention threshold (needs baselines) |
| Licence document | D4 dashboard | Promotion gate numbers (need runs) |
| Interface contracts | Whether to attempt M1-G / M2-F (GPU time) | Drift thresholds (need production-like logs) |
| EngSAF EDA (data is readable) | | Final report, model card |

## 10. Top risks

| Risk | Mitigation |
|---|---|
| StackPulse schema or tags unsuitable for a topic label | Week-0 audit with a go/no-go; fall back to the PRD's dataset |
| Tiny question pool makes Model 2 results look better than they are | Grouped split, ablation without question text, report both |
| Free Colab GPU limits stop transformer runs | Treat M1-G and M2-F as optional; linear/XGBoost models must already meet the targets |
| Free-tier hosting limits (sleep, memory) | Check in Week 0; keep a local Docker demo as backup |
| Licence misuse (EngSAF academic-only) | Keep raw data out of Git and out of the public image; document use |
| Claims exceed what is built | Rule from PRD: every statement in docs must be backed by a run, a test or a log |
