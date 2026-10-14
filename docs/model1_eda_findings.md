# Model 1 Exploratory Data Analysis & Topic-Label Feasibility Report

**Document ID:** `DOC-M1-EDA-20261014`
**Date of Audit & EDA:** 14 October 2026
**Workstream:** Member 1 — Question-Topic Classification (Model 1)
**Project:** AI-Powered Study Assistant for Engineering Students (AI252TA — SDG 4)
**Git Branch:** `feature/atul-topic-classifier`
**Companion Artifact:** [notebooks/01_model1_eda.ipynb](../notebooks/01_model1_eda.ipynb)
**Status:** Initial Exploratory Profiling Complete · Decision D0 Explicitly Pending Faculty Review

---

## 1. Executive Summary & Purpose

This document details the exploratory data analysis (EDA) and label-construction feasibility assessment for Model 1 (Question-Topic Classifier). Model 1 is designed to predict coarse engineering/programming topic categories from cleaned question text (`title` + `body_text`).

To evaluate whether the provisional dataset **StackPulse** (`Omarrran/StackPulse_778K_QnA_Code_dataset`, `full` configuration) can reliably support this task, we profiled a bounded streaming sample ($N = 2{,}500$ to $5{,}000$ rows) directly from the Hugging Face JSONL source. This avoided downloading the full 778K-row corpus while providing high-fidelity statistical observations of schema integrity, text lengths, tag structures, and taxonomy alignment.

### Key EDA Takeaways
1. **High Structural Data Quality:** Zero missing values across all 24 columns; zero duplicate question IDs; 100% availability for `title`, `question_body`, and pre-cleaned `body_text`.
2. **Deterministic Tag Syntax:** Tags are pipe-delimited (`|`), with multi-word concepts preserved via hyphens (`android-studio`, `react-native`, `data-structures`), completely avoiding space-splitting ambiguities. Questions average $2.67$–$2.83$ tags.
3. **The 54% "Language-Only" Unmapped Gap:** Over 54% of questions in StackPulse are tagged solely with programming language names (`python`, `javascript`, `java`, `c#`, `php`) and generic constructs (`arrays`, `string`, `list`, `json`) without functional domain tags. Assigning standalone language names to specialized topic buckets (e.g., mapping `python` exclusively to ML or `javascript` to frontend) would cause severe target label corruption.
4. **Extreme Category Imbalance in StackPulse:** Among cleanly mapped questions, **Mobile Development** (~30–35%) and **Web Frontend** (~27–29%) dominate, whereas **Algorithms & Data Structures** represents only ~0.7–1.1%. Stack Overflow is inherently oriented toward applied frameworks rather than academic algorithmic proofs.
5. **Anti-Leakage Mandate:** Tags are used exclusively to derive the supervisory target label $y$. **Tags must never be passed to the model as input features.** Input features must derive strictly from cleaned question text (`title` + `body_text`).

---

## 2. Sampling Methodology & Environment Safety

Per course guidelines and [notebooks/README.md](../notebooks/README.md), exploratory profiling must not jeopardize memory limits on free Google Colab (12 GB RAM) or bloat Git history:
- **Source Configuration:** `Omarrran/StackPulse_778K_QnA_Code_dataset` (`full` config; raw file: `data/stackoverflow_778k_full.jsonl`).
- **Access Method:** Streaming over HTTP (`urllib.request`) line-by-line with an early break upon collecting the bounded sample.
- **Sequential Head Sampling:** Records were retrieved sequentially from the file head. While this is computationally efficient and sufficient for schema and syntax profiling, it does not constitute a randomized cross-sectional sample. Any population-level extrapolations to the full 778,929 rows remain strictly unverified exploratory projections.
- **Memory Footprint:** The in-memory DataFrame for the 2,500-row sample consumed just **6.17 MB** RAM (5,000 rows consumed 12.3 MB).
- **Execution Speed:** 2,500 rows streamed in $4.22$ seconds; 5,000 rows streamed in $7.70$ seconds.
- **Data Hygiene:** No raw question dumps or large binary cache files are written to Git.

---

## 3. Data Integrity, Schema & Missing Values

We audited all 24 schema columns across the sampled rows:

| Check | Result | Assessment |
| :--- | :--- | :--- |
| **Missing / Null Values** | 0 across all 24 columns | **Passed:** Perfect data completeness. |
| **Duplicate Primary Keys (`id`)** | 0 duplicates found | **Passed:** Each record represents a distinct Stack Overflow question. |
| **Title Presence** | 100% populated | **Passed:** Zero empty titles. |
| **Body Presence (`body_text`)** | 100% populated | **Passed:** Pre-normalized text available for all questions. |
| **Raw HTML Body (`question_body`)** | 100% populated | **Passed:** Raw `<pre><code>` blocks preserved for reference. |
| **Code Block Prevalence (`has_code`)** | 63.7% – 76.5% | Questions frequently contain inline or fenced code snippets. |

---

## 4. Question Text & Length Distributions

Model 1 inputs will consist of concatenated `title` and `body_text`. We examined character and word length statistics across fields:

| Field Name | Mean Characters | Median Characters | P95 Characters | Mean Words | Median Words | P95 Words | Min / Max Words |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `title` | 57.8 | 54.0 | 99.0 | 9.3 | 9.0 | 16.0 | 1 / 34 |
| `body_text` (Cleaned) | 515.1 | 416.5 | 1,233.2 | 88.9 | 72.0 | 207.0 | 1 / 909 |
| `question_body` (Raw HTML) | 1,467.7 | 895.5 | 4,632.2 | 152.2 | 112.0 | 379.0 | 7 / 3,323 |

### Text Preprocessing Insights
1. **Pre-Normalized Advantage:** `body_text` already strips HTML tags and replaces `<pre><code>...</code></pre>` blocks with the placeholder token `[CODE]`. This eliminates the need for expensive HTML parsing pipelines during initial model training.
2. **Compact Lengths for Fast Vectorization:** The median combined input text length (`title` + `body_text`) is $\sim 470$ characters ($\sim 81$ words). This allows fast TF-IDF unigram/bigram tokenization and prevents truncation issues when using sentence transformers (such as `all-MiniLM-L6-v2`, max sequence length 256 tokens).
3. **Outlier Filtering Threshold:** 95% of questions have `body_text` under 207 words. We recommend setting a minimum text filter (`min_text_length: 10` characters) as configured in [configs/params.yaml](../configs/params.yaml) to remove anomalous one-word bodies.

---

## 5. Tag Distribution & Delimitation Profiling

### 5.1 Tag Count per Question
Stack Overflow permits 1 to 5 tags per question. In our sample:
- **1 tag:** 18.0% (449 rows)
- **2 tags:** 25.6% (639 rows)
- **3 tags:** 25.1% (628 rows)
- **4 tags:** 18.2% (454 rows)
- **5 tags:** 13.2% (330 rows)
- **Mean tags per question:** 2.83 (Median: 3.0)

### 5.2 Most Frequent Observed Tags
Out of 7,077 total tag occurrences across 2,522 unique tags in the sample, the top 20 most frequent tags are:

| Rank | Tag Name | Sample Count | Share of Questions (%) | Domain Category |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `javascript` | 272 | 10.9% | Language (Ambiguous / Frontend / Node) |
| 2 | `android` | 239 | 9.6% | **Mobile Development** |
| 3 | `python` | 189 | 7.6% | Language (Ambiguous / ML / Scripts / Backend) |
| 4 | `java` | 175 | 7.0% | Language (Ambiguous / Android / Backend) |
| 5 | `php` | 139 | 5.6% | Language / Web Backend |
| 6 | `html` | 116 | 4.6% | **Web Frontend** |
| 7 | `c#` | 107 | 4.3% | Language (Ambiguous / .NET / Desktop / Mobile) |
| 8 | `jquery` | 94 | 3.8% | **Web Frontend** |
| 9 | `ios` | 90 | 3.6% | **Mobile Development** |
| 10 | `node.js` | 65 | 2.6% | **Web Backend & APIs** |
| 11 | `reactjs` | 61 | 2.4% | **Web Frontend** |
| 12 | `css` | 60 | 2.4% | **Web Frontend** |
| 13 | `mysql` | 53 | 2.1% | **Databases & Storage** |
| 14 | `swift` | 52 | 2.1% | **Mobile Development** |
| 15 | `angularjs` | 44 | 1.8% | **Web Frontend** |
| 16 | `angular` | 41 | 1.6% | **Web Frontend** |
| 17 | `arrays` | 40 | 1.6% | General Syntax / Data Structure |
| 18 | `r` | 37 | 1.5% | Language (Data Science / Stats) |
| 19 | `react-native` | 33 | 1.3% | **Mobile Development** |
| 20 | `android-studio` | 33 | 1.3% | **Mobile Development** |

---

## 6. Evaluation of Provisional Taxonomy ([configs/topic_map.yaml](../configs/topic_map.yaml))

We evaluated the 10 provisional coarse categories against observed tags using **conservative anchor mappings** (mapping only unambiguous, domain-specific tags and deliberately avoiding ungrounded guesses):

| Provisional Category | Representative Observed Anchor Tags | Observed Volume | Representation Assessment |
| :--- | :--- | :--- | :--- |
| `mobile_development` | `android`, `ios`, `swift`, `kotlin`, `flutter`, `react-native`, `xcode`, `android-studio` | **Very High** | Highly viable, clearly delineated, robust volume. |
| `web_frontend` | `html`, `css`, `reactjs`, `angularjs`, `angular`, `vue.js`, `jquery`, `bootstrap`, `dom` | **Very High** | Strongest presence in dataset; easily separable. |
| `web_backend_and_apis` | `node.js`, `django`, `flask`, `fastapi`, `express`, `spring-boot`, `rest`, `api` | **Moderate** | Viable, but frequently overlaps with database tags. |
| `databases_and_storage` | `sql`, `mysql`, `postgresql`, `mongodb`, `sqlite`, `redis`, `database`, `nosql` | **Moderate** | Solid representation; clear vocabulary. |
| `data_science_and_ml` | `pandas`, `numpy`, `dataframe`, `machine-learning`, `deep-learning`, `tensorflow`, `pytorch` | **Moderate** | Clean anchor tags exist, but volume is moderate. |
| `devops_cloud_and_infrastructure` | `docker`, `kubernetes`, `amazon-web-services`, `azure`, `git`, `ci-cd` | **Low-Moderate** | Clear signal, modern relevance, moderate frequency. |
| `computer_networks_and_security` | `networking`, `sockets`, `security`, `cryptography`, `ssl`, `tls`, `authentication` | **Low** | Critical engineering topic, but lower question volume. |
| `operating_systems_and_systems` | `linux`, `unix`, `shell`, `bash`, `posix`, `memory-management`, `multithreading` | **Low** | Present, but often conflated with build/tooling issues. |
| `software_engineering_and_testing`| `unit-testing`, `testing`, `pytest`, `junit`, `design-patterns` | **Low** | Narrow coverage; rarely primary subject on Stack Overflow. |
| `algorithms_and_data_structures` | `algorithms`, `data-structures`, `sorting`, `recursion`, `tree`, `graph`, `time-complexity` | **Very Low (<1.5%)** | **Severely under-represented.** Questions favor library API calls over algorithmic theory. |

---

## 7. Multi-Topic Overlap & Ambiguity Analysis

When evaluating the sample with defensible anchor tags, the status breakdown is:
- **Cleanly Mapped to 1 Topic:** 1,057 questions (**42.3%**)
- **Ambiguous Multi-Topic Conflict:** 84 questions (**3.4%**)
- **Unmapped (No Anchor Tag):** 1,359 questions (**54.4%**)

### 7.1 Nature of Ambiguous Multi-Topic Questions
Questions often span two distinct technical domains:
- `['node.js', 'mongodb']` $\rightarrow$ Web Backend vs. Databases & Storage (50/50 tie).
- `['reactjs', 'react-native']` $\rightarrow$ Web Frontend vs. Mobile Development (50/50 tie).
- `['java', 'android', 'rest', 'retrofit2']` $\rightarrow$ Mobile Development vs. Web Backend.
- `['javascript', 'design-patterns', 'architecture', 'angular']` $\rightarrow$ Web Frontend vs. Software Engineering.

**Resolution Policy Performance:**
Under the policy in [configs/topic_map.yaml](../configs/topic_map.yaml):
- When one topic dominates with $\ge 60\%$ of matched tag votes, the dominant topic is assigned.
- In true 50/50 ties (e.g., full-stack `node.js` + `mongodb`), the sample is dropped from training and logged as `AMBIGUOUS_CONFLICT` ($3.4\%$ drop rate). This policy successfully prevents injecting noisy, contradictory labels into the training set.

### 7.2 Why Are 54.4% of Questions Unmapped?
Inspection of the 1,359 unmapped questions reveals:
1. **Pure Language Questions:** The single most common tags on unmapped questions are `python` (251), `javascript` (200), `java` (134), `c#` (113), and `php` (108).
2. **Syntax and Data Types:** Questions like *"How to slice a string in Python"* or *"Javascript loop over object keys"* have tags like `['python', 'string', 'list', 'replace']`. They describe fundamental programming syntax rather than Web, Mobile, or Data Science.
3. **Ecosystem Tools:** Tags like `wordpress`, `excel`, `visual-studio-code`, `.net`, and `powerbi` appear frequently but do not map to the 10 engineering categories.

---

## 8. Exploratory Class Distribution & Imbalance Analysis

> [!WARNING]
> **Exploratory Notice:** The figures below reflect the observed distribution within the cleanly mapped subset ($N = 1{,}057$) using provisional anchor mappings on the bounded sample. They are **exploratory estimates** to guide taxonomy design, NOT final training statistics.

| Topic Category | Sample Count ($N=1,057$) | Exploratory Share (%) | Imbalance Ratio (vs. Minimum) |
| :--- | :--- | :--- | :--- |
| `mobile_development` | 369 | **34.9%** | $52.7 \times$ |
| `web_frontend` | 303 | **28.7%** | $43.3 \times$ |
| `web_backend_and_apis` | 110 | **10.4%** | $15.7 \times$ |
| `databases_and_storage` | 107 | **10.1%** | $15.3 \times$ |
| `data_science_and_ml` | 59 | **5.6%** | $8.4 \times$ |
| `devops_cloud_and_infrastructure` | 51 | **4.8%** | $7.3 \times$ |
| `computer_networks_and_security` | 20 | **1.9%** | $2.9 \times$ |
| `operating_systems_and_systems` | 19 | **1.8%** | $2.7 \times$ |
| `software_engineering_and_testing` | 12 | **1.1%** | $1.7 \times$ |
| `algorithms_and_data_structures` | 7 | **0.7%** | **$1.0 \times$ (Baseline Minimum)** |

### Implications of Class Imbalance
- **Extreme Skew:** The top 2 categories (`mobile_development` and `web_frontend`) account for **63.6%** of all cleanly mapped samples.
- **Sparse Minority Classes:** `algorithms_and_data_structures`, `software_engineering_and_testing`, and `operating_systems_and_systems` together account for less than $4\%$ of the sample.
- **Modeling Requirement:** Standard accuracy will be completely deceptive. Macro-averaged F1 (`macro_f1`) must be the primary metric, and models capable of handling severe class imbalance (e.g., Complement Naive Bayes M1-E, class-weighted Logistic Regression M1-C) must be prioritized.

---

## 9. Strict Anti-Leakage Protocol

A primary risk in tag-derived supervisory labeling is data leakage into model features:

```
[Question Text] --------> [Model 1 Features (TF-IDF / Embeddings)] --------> [Predicted Topic]
                                                                                   ^
                                                                                   |  (Loss)
                                                                                   v
[Question Tags] --------> [topic_map.yaml Taxonomy Mapping] -------------> [Supervisory Label]
```

1. **Isolation of Tags:** Tags are used exclusively during the offline data preprocessing stage to generate ground truth `topic`.
2. **Feature Scrubbing:** All raw tag fields (`tags`, `tag_list`, `tag_count`) must be excluded from feature vectors.
3. **Text-Only Input:** Model 1 must learn topic boundaries solely from natural language question text (`title` + `body_text`).

---

## 10. Operational Blockers, Licensing & Governance Status

1. **Licensing Status (Unresolved):** As established in [docs/model1_dataset_audit.md](model1_dataset_audit.md), StackPulse carries an `apache-2.0` Hub tag, but its source content is subject to upstream Stack Overflow **CC BY-SA 3.0 / 4.0** terms. The licensing suitability of using this dataset for coursework remains **unresolved** pending institutional/faculty guidance.
2. **Faculty Approval (Decision D0):** **Explicitly pending.** Neither dataset has received formal faculty approval in available documentation.
3. **Hardware / Free Colab Safety:** Bounded sample streaming confirms that a 100,000-row sample will consume $\sim 45\text{--}65\text{ MB}$ compressed Parquet, fitting easily within free Google Colab (12 GB RAM) and local CPU bounds.

---

## 11. Final Decision & Actionable Next Steps

### 11.1 Can StackPulse Support Defensible Topic Labels?
**Yes, conditionally.** StackPulse can support defensible topic labels, but **only if** the following two critical challenges are addressed prior to dataset freezing:
1. **Handling the 54% Unmapped "Language-Only" Questions:** Rather than discarding over half the dataset or corrupting specialized categories by blindly assigning `python` or `javascript`, the team must either:
   - Introduce a 11th category: `programming_fundamentals_and_syntax` (e.g. for syntax, data structures, strings, loops, basic I/O), OR
   - Filter out questions that have only generic language tags and no domain-specific tags during the sampling stage.
2. **Addressing Extreme Algorithms Sparsity:** If the faculty expects the study assistant to heavily assist with core theoretical subjects (Algorithms, Data Structures, OS Theory), StackPulse provides very sparse signal ($\sim 1\%$). The team should consider whether to:
   - Stratify sampling aggressively to oversample `algorithms_and_data_structures`, OR
   - Use Candidate B (`habedi/stack-exchange-dataset`, which is dominated by theoretical CS) as a specialized supplemental split for algorithms.

### 11.2 What Evidence is Needed Before Baseline Training?
Before Member 1 begins baseline model training (M1-A through M1-F):
1. **Decision D0 Faculty Sign-Off:** Faculty approval of the dataset choice and licensing posture.
2. **Taxonomy Ratification:** Review and team sign-off on the expanded tag dictionary in [configs/topic_map.yaml](../configs/topic_map.yaml), specifically ratifying how standalone language tags and the unmapped subset are handled.
3. **Stratified Sample Generation:** Materializing the versioned 100,000-row sample via DVC with verified class distributions across all 8–12 target topics.
