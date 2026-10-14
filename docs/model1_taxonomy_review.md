# Model 1 Topic Taxonomy & Label Coverage Review

**Document ID:** `DOC-M1-TAXONOMY-20261014`
**Date:** 14 October 2026
**Workstream:** Member 1 — Question-Topic Classification (Model 1)
**Project:** AI-Powered Study Assistant for Engineering Students (AI252TA — SDG 4)
**Git Branch:** `feature/atul-topic-classifier`
**Referenced Documents:**
- [docs/model1_dataset_audit.md](model1_dataset_audit.md)
- [docs/model1_eda_findings.md](model1_eda_findings.md)
- [notebooks/01_model1_eda.ipynb](../notebooks/01_model1_eda.ipynb)
- [configs/topic_map.yaml](../configs/topic_map.yaml)
- [configs/params.yaml](../configs/params.yaml)
**Status:** Taxonomy Review Complete · Label-Construction Policy Formulated · Governance Pending

---

## 1. Executive Summary & Objective

Before initiating model baseline implementations (M1-A through M1-F) or materializing large training samples, Member 1 must resolve the **topic-label coverage problem**.

In StackPulse (`Omarrran/StackPulse_778K_QnA_Code_dataset`), no ready-made topic column exists; ground-truth supervisory labels must be constructed programmatically from post tags. The preliminary EDA revealed two major challenges:
1. **The 54% Unmapped Gap:** More than half of StackPulse questions are tagged only with language names (`python`, `javascript`, `java`, `c#`) or generic syntax tokens (`string`, `arrays`, `list`) without functional domain tags.
2. **Extreme Class Imbalance:** Among cleanly mapped questions, Web Frontend and Mobile Development account for over $55\%$, whereas core theoretical engineering topics—especially `algorithms_and_data_structures`—account for only $\sim 1.1\%$.

To evaluate taxonomy coverage with statistical power, we analyzed the **first 10,000 sequential records** ($N = 10{,}000$) streamed directly from the head of `data/stackoverflow_778k_full.jsonl` without downloading the full dataset. This document details the mapping evidence, analyzes unmapped and conflicting tag structures, compares taxonomy options, and formalizes a defensible label-construction policy.
*(Important caveat: Because records were retrieved sequentially from the file head rather than through uniform random sampling, all population-level extrapolations to the full 778,929 rows are strictly unverified exploratory projections).*

---

## 2. Audit of Proposed Tag-to-Category Mappings

The EDA notebook introduced a conservative **anchor tag dictionary** to ground the 10 provisional categories defined in [configs/topic_map.yaml](../configs/topic_map.yaml). We evaluated every proposed mapping against empirical tag co-occurrences in the 10,000-row sample:

### 2.1 Supported Category Mappings & Evidence

| Category ID | Display Name | Grounded Anchor Tags | Empirical Evidence & Signal Strength |
| :--- | :--- | :--- | :--- |
| `web_frontend` | Web Frontend Development | `html`, `css`, `reactjs`, `angularjs`, `angular`, `vue.js`, `frontend`, `dom`, `bootstrap`, `jquery` | **Extremely Strong.** Tags denote client-side rendering, styling, and browser DOM manipulation. Co-occurrences are highly consistent. |
| `mobile_development` | Mobile App Development | `android`, `ios`, `swift`, `kotlin`, `flutter`, `react-native`, `xcode`, `android-studio`, `xamarin` | **Extremely Strong.** Distinct platform APIs and SDKs. Strong intra-domain clustering with negligible ambiguity. |
| `web_backend_and_apis` | Web Backend & APIs | `node.js`, `django`, `flask`, `fastapi`, `express`, `spring-boot`, `asp.net`, `rest`, `api`, `graphql` | **Strong.** Clear server-side application frameworks. Frequently pairs with database tags, resolved via dominance rules. |
| `databases_and_storage` | Databases & Storage | `sql`, `mysql`, `postgresql`, `mongodb`, `sqlite`, `redis`, `database`, `nosql` | **Strong.** Standard relational and NoSQL engines. Clear, unambiguous technical boundaries. |
| `data_science_and_ml` | Data Science & ML | `pandas`, `numpy`, `dataframe`, `machine-learning`, `deep-learning`, `tensorflow`, `pytorch`, `scikit-learn` | **Strong.** Distinct numerical computing and ML library ecosystem. High semantic coherence. |
| `devops_cloud_and_infrastructure` | DevOps & Cloud | `docker`, `kubernetes`, `amazon-web-services`, `azure`, `git`, `ci-cd` | **Moderate-Strong.** Clear modern infrastructure tooling, though lower overall frequency in StackPulse. |
| `operating_systems_and_systems` | OS & Systems Programming | `linux`, `unix`, `shell`, `bash`, `posix`, `memory-management`, `concurrency`, `multithreading` | **Moderate.** Strong theoretical relevance. Low representation in StackPulse; posts often focus on build/environment issues. |
| `computer_networks_and_security` | Networks & Security | `networking`, `sockets`, `security`, `cryptography`, `ssl`, `tls`, `authentication` | **Moderate.** Essential engineering curriculum subject. Low question frequency on Stack Overflow. |
| `software_engineering_and_testing` | SE & Testing | `unit-testing`, `testing`, `pytest`, `junit`, `design-patterns` | **Moderate-Low.** Clear semantics, but low volume on StackPulse (often overshadowed by functional framework tags). |
| `algorithms_and_data_structures` | Algorithms & Data Structures | `algorithms`, `data-structures`, `sorting`, `recursion`, `tree`, `binary-tree`, `graph`, `time-complexity`, `dynamic-programming` | **Sparse (<1.5%).** The semantics are rigorous, but raw volume is very low in practical developer Q&A. |

### 2.2 Deliberately Excluded / Unsupported Mappings
To maintain label integrity and avoid arbitrary assignments, the following high-frequency tags were **explicitly excluded** from the 10 domain buckets:

1. **Standalone Language Tags (`python`, `javascript`, `java`, `c#`, `c++`, `c`, `php`, `r`):**
   - *Reason for Exclusion:* A programming language is an implementation medium, not a functional domain. For instance, `python` is used for data science, web backends, desktop utilities, automation scripts, and CS101 education. Mapping `python` solely to `data_science_and_ml` would corrupt the target label for thousands of basic syntax or script questions.
2. **Generic Data Structure / Syntax Terms (`arrays`, `string`, `list`, `json`, `regex`, `loops`):**
   - *Reason for Exclusion:* While `arrays` relates to data structures, questions tagged `['javascript', 'arrays']` are almost exclusively basic language manipulation tasks (e.g., filtering an array in JS), not algorithmic complexity proofs.
3. **General Development Tools (`visual-studio-code`, `windows`, `excel`, `powerbi`, `wordpress`):**
   - *Reason for Exclusion:* These represent IDEs, operating environments, or business intelligence tools that do not map to the core 10 engineering subjects.

---

## 3. Empirical Analysis on the 10,000-Row Sample

Analyzing the expanded bounded stream of 10,000 questions yielded 27,666 total tag instances across 6,369 distinct tags.

### 3.1 Unmapped Tag Frequency Analysis
A total of **5,614 questions (56.14%)** in the 10,000 sample have no matching anchor tag under the provisional 10-category dictionary. The top unmapped tags are:

| Tag | Count in Unmapped ($N=5,614$) | % of Unmapped Questions | Overall Sample Share (%) | Technical Context |
| :--- | :--- | :--- | :--- | :--- |
| `python` | 664 | 11.8% | 6.6% | Multi-paradigm language |
| `javascript` | 391 | 7.0% | 3.9% | Language (syntax, ES6 features) |
| `java` | 292 | 5.2% | 2.9% | Language (syntax, collections, OOP) |
| `c#` | 221 | 3.9% | 2.2% | Language / .NET runtime |
| `php` | 189 | 3.4% | 1.9% | Language (general web scripting) |
| `r` | 162 | 2.9% | 1.6% | Statistical programming |
| `wordpress` | 134 | 2.4% | 1.3% | CMS ecosystem |
| `python-3.x` | 125 | 2.2% | 1.2% | Version tag |
| `c++` | 70 | 1.2% | 0.7% | Language (pointers, OOP) |
| `excel` | 69 | 1.2% | 0.7% | Spreadsheet manipulation / VBA |
| `visual-studio-code`| 69 | 1.2% | 0.7% | Editor configuration |
| `typescript` | 67 | 1.2% | 0.7% | Typed JS superset |
| `json` | 66 | 1.2% | 0.7% | Data serialization format |
| `powerbi` | 66 | 1.2% | 0.7% | BI analytics platform |
| `ruby-on-rails` | 60 | 1.1% | 0.6% | Web framework (not in v0 anchor) |
| `arrays` | 58 | 1.0% | 0.6% | In-memory data representation |

### 3.2 Common Unmapped Tag Combinations
The most frequent tag sets among unmapped questions are single-tag or double-tag pure language posts:
- `['python']` (68 posts), `['javascript']` (45 posts), `['r']` (26 posts), `['java']` (26 posts)
- `['python', 'python-3.x']` (15 posts)
- `['dax', 'powerbi']` (15 posts)
- `['wordpress']` (22 posts)
- `['visual-studio-code']` (18 posts)

These posts represent legitimate programming questions, but they address general language syntax, tool configuration, or unmapped platforms rather than the 10 coarse categories.

### 3.3 Conflict & Multi-Topic Analysis
Ambiguous multi-topic conflicts occurred in **377 questions (3.77%)** where tags matched multiple distinct categories and failed the $60\%$ dominance threshold. The top conflict pairs are:
1. `['web_backend_and_apis', 'web_frontend']`: 78 questions (e.g. full-stack `node.js` + `reactjs`).
2. `['databases_and_storage', 'web_backend_and_apis']`: 68 questions (e.g. backend ORM/database integration `django` + `postgresql`, `node.js` + `mongodb`).
3. `['mobile_development', 'web_frontend']`: 39 questions (e.g. cross-platform hybrid `reactjs` + `react-native`).
4. `['databases_and_storage', 'mobile_development']`: 17 questions (e.g. mobile local storage `android` + `sqlite`).
5. `['computer_networks_and_security', 'web_backend_and_apis']`: 15 questions (e.g. backend security `express` + `jwt` / `oauth`).

---

## 4. Quantitative Comparison of Taxonomy Options

We simulated two structural options on the 10,000-row sample to provide the team with empirical data for Decision D0:

### 4.1 Option A: Retain Current 10 Categories (Filter Unmapped & Ambiguous)
Under Option A, the 10 provisional categories in [configs/topic_map.yaml](../configs/topic_map.yaml) are preserved. Questions that do not match a functional anchor tag (or produce an unresolvable cross-domain tie) are excluded from the training set.

- **Cleanly Mapped Questions:** **4,009 / 10,000 (40.09%)**
- **Ambiguous Conflicts (Dropped):** **377 / 10,000 (3.77%)**
- **Unmapped Posts (Excluded):** **5,614 / 10,000 (56.14%)**

#### Option A Class Distribution (Cleanly Mapped Subset, $N = 4{,}009$):
| Topic Category | Count in Sample | Exploratory Share (%) |
| :--- | :--- | :--- |
| `web_frontend` | 1,139 | **28.41%** |
| `mobile_development` | 1,071 | **26.71%** |
| `web_backend_and_apis` | 608 | **15.17%** |
| `databases_and_storage` | 374 | **9.33%** |
| `data_science_and_ml` | 314 | **7.83%** |
| `devops_cloud_and_infrastructure` | 203 | **5.06%** |
| `operating_systems_and_systems` | 108 | **2.69%** |
| `computer_networks_and_security` | 77 | **1.92%** |
| `software_engineering_and_testing` | 70 | **1.75%** |
| `algorithms_and_data_structures` | 45 | **1.12%** |

### 4.2 Option B: Propose 11th Category (`programming_fundamentals_and_syntax`)
Under Option B, we simulate an expanded taxonomy that captures posts lacking functional domain tags but containing language or fundamental syntax tokens (`python`, `javascript`, `java`, `c#`, `string`, `arrays`, `list`, `loops`, `dict`).

- **Cleanly Mapped Questions:** **6,215 / 10,000 (62.15%)**
- **Ambiguous Conflicts (Dropped):** **377 / 10,000 (3.77%)**
- **Unmapped Posts:** **3,408 / 10,000 (34.08%)**

#### Option B Class Distribution (Cleanly Mapped Subset, $N = 6{,}215$):
| Topic Category | Count in Sample | Exploratory Share (%) |
| :--- | :--- | :--- |
| `programming_fundamentals_and_syntax` *(Proposed)* | 2,206 | **35.49%** |
| `web_frontend` | 1,139 | **18.33%** |
| `mobile_development` | 1,071 | **17.23%** |
| `web_backend_and_apis` | 608 | **9.78%** |
| `databases_and_storage` | 374 | **6.02%** |
| `data_science_and_ml` | 314 | **5.05%** |
| `devops_cloud_and_infrastructure` | 203 | **3.27%** |
| `operating_systems_and_systems` | 108 | **1.74%** |
| `computer_networks_and_security` | 77 | **1.24%** |
| `software_engineering_and_testing` | 70 | **1.13%** |
| `algorithms_and_data_structures` | 45 | **0.72%** |

### 4.3 Trade-Off Assessment
| Evaluation Criterion | Option A (Strict 10 Categories) | Option B (11 Categories with Fundamentals) |
| :--- | :--- | :--- |
| **Label Purity & Defensibility** | **Very High:** Every mapped question has clear, unambiguous domain anchor tags. | Moderate-High: Distinguishes syntax from domains, but "fundamentals" is broad. |
| **Projected Data Yield on 778K Pool (Unverified Extrapolation)** | $\sim 40\%$ sample yield (unverified linear extrapolation: $\approx 311{,}000$ clean rows across full pool). | $\sim 62\%$ sample yield (unverified linear extrapolation: $\approx 480{,}000$ clean rows across full pool). |
| **Alignment with PRD Scope** | **Exact:** Perfectly aligns with the 10 coarse engineering subjects in [configs/topic_map.yaml](../configs/topic_map.yaml). | Requires updating [configs/topic_map.yaml](../configs/topic_map.yaml) and downstream contracts. |
| **Minority Class Dilution** | Algorithms represents $1.12\%$ of mapped data. | Algorithms drops to $0.72\%$ of mapped data. |

---

## 5. Identification of Categories with Insufficient Examples

In the 10,000-row sample, four categories show low representation:
1. **`algorithms_and_data_structures`:** 45 examples ($1.12\%$ of mapped data).
2. **`software_engineering_and_testing`:** 70 examples ($1.75\%$ of mapped data).
3. **`computer_networks_and_security`:** 77 examples ($1.92\%$ of mapped data).
4. **`operating_systems_and_systems`:** 108 examples ($2.69\%$ of mapped data).

### Feasibility for 100,000-Row Stratified Sample (Unverified Population Extrapolations)

> [!WARNING]
> **Sampling Limitation & Unverified Extrapolations:** The sample analyzed consists strictly of the **first 10,000 sequential records** streamed from the head of the raw JSONL file, not a randomized or representative cross-sectional sample. Because StackPulse records may exhibit temporal or sequential ordering bias, **all population-level extrapolations to the full 778,929-record corpus are unverified theoretical projections**. Actual full-population distributions may differ once full-index or uniform random profiling is conducted.

- **Hypothetical Linear Projection (Unverified):** If the sample's $40.09\%$ clean yield held uniformly across the full **778,929 rows**, it would theoretically provide approximately **311,000 cleanly labeled candidate questions**.
- **Unverified Projections of Category Volume in 778K Population:**
  - `algorithms_and_data_structures`: $\sim 3{,}500$ questions (unverified linear extrapolation: $0.45\% \times 778{,}929$).
  - `software_engineering_and_testing`: $\sim 5{,}400$ questions (unverified linear extrapolation: $0.70\% \times 778{,}929$).
  - `computer_networks_and_security`: $\sim 6{,}000$ questions (unverified linear extrapolation: $0.77\% \times 778{,}929$).
  - `operating_systems_and_systems`: $\sim 8{,}400$ questions (unverified linear extrapolation: $1.08\% \times 778{,}929$).
- **Critical Strategy:** Uniform random sampling would produce severe class scarcity in validation and test folds. When drawing the 100,000-row sample for Decision D2, **stratified quota allocation** must be applied, enforcing a minimum floor (e.g., at least 2,500–3,000 examples for every category, including algorithms), with verification against full-dataset indexes.

---

## 6. Recommended Label-Construction Policy

To avoid arbitrary heuristic guesses and minimize supervisory label noise, Member 1 recommends adopting the following **single, unified policy**:

```
[Raw Tags]
    │
    ├── Step 1: Match against Verified Functional Anchor Dictionary (anchor_map)
    │           (Deliberately omit standalone language names and generic syntax tokens)
    │
    ├── Step 2: Evaluate Matched Categories
    │     ├── 0 Matches ───────────────> Exclude from training (UNMAPPED)
    │     ├── 1 Category Match ────────> Assign Category Label (100% agreement)
    │     └── 2+ Matches ──────────────> Step 3: Conflict Dominance Check
    │
    └── Step 3: Dominance Evaluation
          ├── Top Category Votes >= 60% ─> Assign Dominant Category
          └── Top Category Votes < 60% ──> Drop sample & log (AMBIGUOUS_CONFLICT)
```

### Policy Rules:
1. **Keep Option A (10 Categories) for Phase 1 Baseline Reproducibility:** Do not prematurely modify [configs/topic_map.yaml](../configs/topic_map.yaml) without team and faculty sign-off. Even under conservative yield projections, the candidate pool offers substantial volume for a 100k sample, but true population totals remain unverified until uniform sampling is executed.
2. **Exclusion of Ambiguous Posts:** True 50/50 ties (e.g. `node.js` + `mongodb`) are cleanly dropped ($3.77\%$ drop rate), protecting the classifier from contradictory supervisory signals.
3. **Strict Stratified Sampling:** Enforce a minimum class floor during DVC ingestion so minority categories have sufficient training and evaluation instances.

---

## 7. Anti-Leakage Verification

```
┌────────────────────────────────────────────────────────┐
│                   TRAINING PIPELINE                    │
│                                                        │
│  [Question Text (title + body_text)]                  │
│                │                                       │
│                ▼                                       │
│     Feature Extraction (TF-IDF / MiniLM)               │
│                │                                       │
│                ▼                                       │
│          Model Prediction                              │
│                │                                       │
│                ▼                                       │
│          Loss Calculation ◄────────┐                   │
│                                    │                   │
├────────────────────────────────────┼───────────────────┤
│         OFFLINE LABELING           │                   │
│                                    │                   │
│  [Question Tags] ──► [topic_map] ──┴─► [Target Label]  │
│  (STRIPPED & NEVER PASSED TO MODEL)                    │
└────────────────────────────────────────────────────────┘
```

- **Verification:** Tags are consumed exclusively during the offline data labeling stage to synthesize the supervisory target label $y \in \{0, \dots, 9\}$.
- **Input Scrubbing:** In `src/study_assistant/data/preprocess.py` (and data ingestion stages), all tag columns (`tags`, `tag_list`, `tag_count`) are completely stripped from feature vectors.
- **Model Invariance:** Model 1 inputs are restricted strictly to cleaned question text (`title` + `body_text`).

---

## 8. Notebook Verification & Code Hygiene

The companion exploratory notebook [notebooks/01_model1_eda.ipynb](../notebooks/01_model1_eda.ipynb) was verified for compliance with repository standards:
- **No Embedded Raw Question Records:** All cell outputs display only aggregate summary statistics, frequency tables, and distributions. No raw question titles, bodies, or student records are written to Git.
- **No Credentials or API Tokens:** All calls use public unauthenticated HTTP streams. No passwords, tokens, or `.env` secrets exist.
- **No Machine-Specific Paths:** All imports and configuration paths use relative paths (`../configs/topic_map.yaml`).
- **Reproducibility:** All 7 code cells execute sequentially in $4.22$ seconds with clean exit status.

---

## 9. Governance & Outstanding Decisions

1. **Faculty Approval Status (Decision D0):** **Explicitly pending.** Course review deadlines have not yet arrived, and formal faculty approval has not been entered into repository records. This report provides the empirical basis to request faculty sign-off during the Week 1 milestone.
2. **Licensing Suitability Status:** StackPulse source content remains subject to upstream Stack Overflow **CC BY-SA 3.0 / 4.0** terms. Licensing suitability for academic coursework remains **unresolved** pending institutional and faculty guidance.
3. **Taxonomy Status:** [configs/topic_map.yaml](../configs/topic_map.yaml) remains marked `status: "provisional_scaffold"` and [configs/params.yaml](../configs/params.yaml) remains marked `approval_status: "pending_review_and_profiling"`. Neither configuration was modified during this task.
