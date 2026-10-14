# Model 1 (Topic Classifier) Dataset Audit & Verification Report

**Document ID:** `DOC-M1-AUDIT-20261014`
**Date of Audit:** 14 October 2026
**Workstream:** Member 1 — Question-Topic Classification (Model 1)
**Project:** AI-Powered Study Assistant for Engineering Students (AI252TA — SDG 4)
**Git Branch:** `feature/atul-topic-classifier` (Base: `chore/phase1-project-skeleton`)
**Status:** Initial Audit Complete · Decision D0 Explicitly Pending Faculty Review

---

## 1. Executive Summary & Audit Scope

This document provides a rigorous, empirical dataset audit for the **Question-Topic Classification** workstream (Model 1). Model 1 is tasked with classifying incoming technical questions into one of 8–12 coarse technical/engineering topics based on cleaned question text (`title` + `body`).

### Project Workstream Allocation & Dataset Context
The Phase 1 development scope divides machine learning modeling into two independent workstreams:
- **Member 1 — Question-Topic Classifier (Model 1):** Classifies question text into coarse engineering/programming topics. The current project configuration provisionally selects the [StackPulse (`Omarrran/StackPulse_778K_QnA_Code_dataset`)](https://huggingface.co/datasets/Omarrran/StackPulse_778K_QnA_Code_dataset) dataset as its primary candidate. The [habedi/stack-exchange-dataset](https://huggingface.co/datasets/habedi/stack-exchange-dataset) is retained solely as the original PRD's alternative candidate for Model 1, **not** as Member 2's dataset.
- **Member 2 — Student Answer Scorer (Model 2):** Independently implements the short-answer scoring model evaluated on the [EngSAF (`IsmaelMousa/engsaf`)](https://huggingface.co/datasets/IsmaelMousa/engsaf) dataset. (Member 2's workstream is executed independently; EngSAF is not audited in this report).

> [!IMPORTANT]
> **Governance & Licensing Status:** Formal faculty approval for dataset selection remains **explicitly pending** across both workstreams (Decision D0 for Model 1). Furthermore, critical licensing questions—such as upstream CC BY-SA copyleft/attribution obligations on StackPulse content and academic-only redistribution restrictions on EngSAF—remain **unresolved** pending formal institutional and faculty guidance. Neither dataset may be treated as formally approved.

Two candidate datasets for Model 1 are evaluated in this audit:
1. **Candidate A (Provisional Selection):** `Omarrran/StackPulse_778K_QnA_Code_dataset` (Stack Overflow developer Q&A)
2. **Candidate B (PRD Alternative Reference):** `habedi/stack-exchange-dataset` (Stack Exchange CS, Data Science, and Political Science)

### Key Audit Findings at a Glance
- **Discrepancy Resolved (StackPulse 778K vs 1.95M):** The apparent discrepancy between the 778K title and ~1.95M viewer rows is an artifact of Hugging Face's dataset server summing row counts across all 9 configurations (`full`, `unanswered`, `with_code`, `high_quality`, `python`, `javascript`, `java`, `csharp`, `android`). The primary deduplicated `full` configuration contains **exactly 778,929 rows**.
- **No Native Topic Labels Exist:** Neither candidate contains an off-the-shelf target topic label. Ground-truth topic targets must be constructed programmatically from tags.
- **Label Trap Identified (Habedi):** The `label` column in `habedi/stack-exchange-dataset` represents *resolved vs unresolved* question status (0/1), **not a topic category**. Blindly treating `label` as a target would cause catastrophic formulation error.
- **Tag Representation Contrast:** StackPulse uses cleanly delimited pipe-separated tags (`|`) with standard hyphenated compound tags (`android-7.0-nougat`), whereas Habedi uses space-separated strings (`database theory relational algebra`) that complicate tokenization.
- **Licensing & Legal Reality:** While StackPulse carries an `apache-2.0` Hub tag, its underlying source content originates from Stack Overflow community contributions governed by **Creative Commons Attribution-ShareAlike (CC BY-SA 3.0 / 4.0)**. Downstream usage and redistribution inherit attribution and share-alike constraints.
- **Faculty Approval Status (Decision D0):** **Neither dataset has been formally approved by the course faculty in available project records.** Per governance rules, Decision D0 remains **explicitly pending**.

---

## 2. Side-by-Side Candidate Comparison

The following table contrasts the empirical properties of Candidate A and Candidate B obtained via Hugging Face Hub metadata APIs and low-cost sample inspection.

| Attribute | Candidate A: StackPulse | Candidate B: Habedi Stack Exchange | Verification Status |
| :--- | :--- | :--- | :--- |
| **Exact Identifier** | `Omarrran/StackPulse_778K_QnA_Code_dataset` | `habedi/stack-exchange-dataset` | Verified via HF API |
| **Curator / Author** | Omar Haq Nawaz Malik (BITS Pilani) | habedi (Academic Q&A Research) | Verified (Card metadata) |
| **Publication Date** | February 2026 | ~2022 (Data through May 2021) | Verified (Git/Card metadata) |
| **Stated Hub Licence** | `apache-2.0` | `cc-by-4.0` | Verified (Hub tag) |
| **Underlying Source** | Stack Overflow programming Q&A | CS, Data Science & Politics Stack Exchange | Verified (Card / README) |
| **Governing Legal Terms**| **CC BY-SA 3.0 / 4.0** (Upstream Stack Overflow) | **CC BY-4.0** / CC BY-SA (Stack Exchange) | Legal analysis of source dump |
| **Total Row Count** | **778,929** (base `full` config); 1,950,497 aggregated | **82,158** (single `train` split) | Verified via HF `/size` API |
| **Available Configs** | 9 (`full`, `unanswered`, `with_code`, `high_quality`, `python`, `javascript`, `java`, `csharp`, `android`) | 1 (`default`) | Verified via HF `/info` API |
| **Available Splits** | `train` only (per configuration) | `train` only | Verified via HF `/info` API |
| **Number of Columns** | 24 columns | 5 columns (`id`, `title`, `body`, `tags`, `label`) | Verified via schema inspection |
| **Question Title** | Yes (`title`: `string`) | Yes (`title`: `string`) | Verified |
| **Question Body** | Yes (`question_body`: HTML; `body_text`: plain) | Yes (`body`: raw HTML `string`) | Verified |
| **Answer Text Present**| **No** (Only `answer_count` and `is_unanswered`) | **No** (Questions only) | Verified (Schema inspection) |
| **Tag Field & Format** | `tags`: Pipe-separated string (`tag1\|tag2\|tag3`) | `tags`: Space-separated string | Verified via first-rows sample |
| **Multi-Word Tags** | Hyphenated tokens preserved (`xamarin.forms`, `time-complexity`) | Space-delimited phrases (`relational algebra`) | Verified via first-rows sample |
| **Pre-existing Target** | **None** (`score_bucket` is upvote tier) | **None** (`label` is resolved status) | Verified via schema inspection |
| **Sampling Feasibility**| Feasible for 100k–200k stratified sample | **Infeasible** for 100k+ (total rows = 82k) | Calculated against D2 scope |
| **Domain Scope** | Applied programming, tools, libraries, SDKs | Theoretical Computer Science, Data Science | Verified from titles & tags |
| **OOD Evaluation Set** | None built-in (all programming/dev) | Built-in via `p.tsv` (Political Science) | Verified from README & repo files |
| **Faculty Approval** | **Pending** (Decision D0) | **Pending** (Decision D0) | Verified against project records |

---

## 3. Candidate A Deep-Dive: StackPulse (`Omarrran/StackPulse_778K_QnA_Code_dataset`)

### 3.1 Metadata, File Inventory & Configurations
Candidate A is hosted as a multi-configuration dataset. An inspection of repository files (`siblings`) reveals:
- **Base Full Dataset:**
  - `data/stackoverflow_778k_full.csv` (CSV format)
  - `data/stackoverflow_778k_full.jsonl` (JSONL format)
- **Filtered Language/Quality Subsets:**
  - `data/stackoverflow_python.jsonl` / `.csv` (107,083 rows)
  - `data/stackoverflow_javascript.jsonl` / `.csv` (87,367 rows)
  - `data/stackoverflow_java.jsonl` / `.csv` (54,077 rows)
  - `data/stackoverflow_android.jsonl` / `.csv` (42,004 rows)
  - `data/stackoverflow_csharp.jsonl` / `.csv` (40,420 rows)
  - `data/stackoverflow_with_code.jsonl` / `.csv` (595,679 rows)
  - `data/stackoverflow_unanswered.jsonl` / `.csv` (224,733 rows)
  - `data/stackoverflow_high_quality.jsonl` / `.csv` (20,205 rows)

### 3.2 Resolution of the 778K vs 1.95M Discrepancy
The context handoff noted an unresolved contradiction: the dataset title specifies 778K rows, while the Hugging Face dataset viewer displayed approximately 1.95 million rows.

**Empirical Resolution:**
The Hugging Face Dataset Server (`https://datasets-server.huggingface.co/size`) sums rows across all exposed configurations to compute total repository size:

$$\begin{aligned}
\text{Total Aggregated Rows} &= \text{full } (778,929) + \text{unanswered } (224,733) + \text{with\_code } (595,679) \\
&\quad + \text{python } (107,083) + \text{javascript } (87,367) + \text{java } (54,077) \\
&\quad + \text{android } (42,004) + \text{csharp } (40,420) + \text{high\_quality } (20,205) \\
&= \mathbf{1,950,497\text{ rows}}
\end{aligned}$$

The 8 sub-configurations are non-disjoint, filtered views extracted directly from `stackoverflow_778k_full`. The base dataset contains **778,929 distinct questions** (deduplicated by ID). There is no missing data or unrecorded explosion of rows.

### 3.3 Verified Schema & Feature Types
The base `full` configuration exposes 24 features:

| Column Name | Storage Type | Semantics & Role in Model 1 |
| :--- | :--- | :--- |
| `id` | `int64` | Unique Stack Overflow question ID. Used as primary key. |
| `title` | `string` | Question title. **Primary text input feature.** |
| `question_body` | `string` | Raw HTML question body including `<pre><code>` blocks. |
| `body_text` | `string` | Cleaned plain text body. HTML stripped, code blocks normalized to `[CODE]`. |
| `tags` | `string` | Pipe-delimited tag string (e.g., `python\|pandas\|dataframe`). **Source for topic label derivation.** |
| `score` | `int64` | Net community upvote score. Metadata only. |
| `creation_date` | `string` | ISO 8601 UTC timestamp (2015-02-14 to 2022-09-25). |
| `year`, `month`, `hour`, `dayofweek` | `int32` / `int64` | Extracted temporal features. |
| `view_count` | `int64` | Total question views. Highly skewed. |
| `answer_count` | `int64` | Number of answers. (Zero in 28.85% of rows). |
| `body_len`, `title_len` | `int64` | Character lengths of cleaned text fields. |
| `tag_count` | `int64` | Number of attached tags (range 1–5; mean 3.01). |
| `code_block_cnt` | `int64` | Number of `<pre>` blocks present. |
| `has_code` | `bool` | True if question contains code (`code_block_cnt >= 1`). (76.47% positive). |
| `is_unanswered` | `bool` | True if `answer_count == 0`. |
| `is_popular`, `is_viral` | `bool` | High-percentile view count flags (P95, P99). |
| `is_highly_voted`, `is_negative` | `bool` | Score threshold flags (`score >= 10`, `score < 0`). |
| `score_bucket` | `string` | Categorical upvote tier: `"negative"`, `"zero"`, `"low"`, `"medium"`, `"high"`. |

### 3.4 Key Strengths of StackPulse for Model 1
1. **Pre-Normalized Text Available:** The presence of `body_text` with HTML already stripped and code converted to `[CODE]` aligns directly with our preprocessing specification in [configs/params.yaml](../configs/params.yaml) (`code_block_handling: replace_with_token`).
2. **Deterministic Tag Delimitation:** Tags are delimited by pipes (`|`), which completely eliminates whitespace splitting ambiguity.
3. **Hyphenated Compound Tags:** Multi-word terms retain their boundaries (`android-7.0-nougat`, `react-native`, `data-structures`).
4. **Volume:** With 778,929 rows, extracting a stratified sample of 100,000–200,000 rows (Decision D2) is straightforward.

### 3.5 Inconsistencies & Dataset Risks
1. **Sampling Gaps:** Years 2019 and 2021 are completely absent due to sampling artifacts during original collection.
2. **Temporal Imbalance:** Over 53.3% of all questions (415,121 rows) are from year 2022. A naive random sample will over-represent 2022 frameworks and libraries. Stratification must account for temporal balance if temporal generalization is tested.
3. **Extreme Metric Skew:** 59.31% of questions have `score == 0`. The mean score (0.69) is heavily pulled by a few viral posts.
4. **No Native Topic Label:** Ground truth must be constructed from `tags`.

---

## 4. Candidate B Deep-Dive: Habedi (`habedi/stack-exchange-dataset`)

### 4.1 Metadata & Schema
Candidate B is stored as 3 TSV files on Hugging Face:
- `cs.tsv` (Computer Science Stack Exchange)
- `ds.tsv` (Data Science Stack Exchange)
- `p.tsv` (Political Science Stack Exchange)

The Hugging Face default config merges these into a single `train` split of **82,158 rows** with 5 columns:

| Column Name | Type | Description | Observations & Risks |
| :--- | :--- | :--- | :--- |
| `id` | `int64` | Question ID | Unique within site. |
| `title` | `string` | Question title | Clean text. |
| `body` | `string` | Question body | Raw HTML containing `<p>`, `<code>`, `<span>`, math tags. Must be parsed. |
| `tags` | `string` | Space-delimited tags | e.g., `database theory relational algebra finite model theory`. |
| `label` | `int64` | **Resolved status (0/1)** | **DO NOT USE AS TOPIC TARGET.** (0 = unresolved, 1 = resolved). |

### 4.2 The "Label Trap" in Habedi
Planning documents explicitly warned against assuming `label` is a topic category. Our schema verification confirms that `label` is a binary integer representing whether a question was answered/resolved in the research study. Treating `label` as a target for topic classification would be a severe formulation error.

### 4.3 Tag Parsing Challenges
Because tags are space-separated without internal delimiters, compound concepts such as `database theory`, `relational algebra`, and `finite model theory` appear as a single unpunctuated string. A naive whitespace split splits `database theory` into two unrelated tokens (`database` and `theory`), leading to inaccurate topic mappings unless a greedy longest-match phrase dictionary is employed.

### 4.4 Strengths of Habedi
1. **Curriculum Domain Match:** Questions originate from Computer Science and Data Science Stack Exchange communities, covering theoretical topics (complexity theory, formal grammars, computability, algorithms, operating systems concepts) that closely parallel university engineering exam questions.
2. **Natural Out-of-Domain (OOD) Split:** The `p.tsv` file (Political Science) provides ~11,000 legitimate non-engineering questions, ideal for testing Model 1's abstention capabilities (PRD Objective O2).

### 4.5 Limitations of Habedi
1. **Insufficient Volume for D2:** The entire dataset is only 82,158 rows. A 100,000–200,000 row sample cannot be created from this dataset.
2. **Space-Separated Tags:** Parsing multi-word tags requires complex heuristic matching.
3. **Raw HTML Only:** The dataset does not provide a pre-cleaned text field.

---

## 5. Licensing, Governance & Legal Analysis

### 5.1 StackPulse (`Omarrran/StackPulse_778K_QnA_Code_dataset`)
- **Hub Metadata Licence:** `apache-2.0`
- **Underlying Provenance:** Content extracted from Stack Overflow public dumps.
- **Legal Reality:**
  - Stack Overflow content contributed by users is licensed under **Creative Commons Attribution-ShareAlike (CC BY-SA 3.0 / 4.0)** depending on the date of contribution.
  - An individual curator cannot unilaterally extinguish upstream CC BY-SA obligations by attaching an Apache-2.0 tag to a Hugging Face repository.
  - **Downstream Requirements:**
    1. **Attribution:** Any public dissemination or published derivative work must credit original Stack Overflow authors and the platform.
    2. **ShareAlike (Copyleft):** If derivative datasets or adaptations are distributed, they must be shared under compatible CC BY-SA terms.
    3. **Licensing Suitability Status:** The licensing suitability of using and adapting StackPulse content under upstream CC BY-SA terms remains **unresolved** for this project. Determining whether CC BY-SA obligations permit internal coursework usage without public ShareAlike redistribution requires appropriate institutional or faculty guidance. Unsupported legal assumptions must not be treated as resolution.

### 5.2 Habedi (`habedi/stack-exchange-dataset`)
- **Hub Metadata Licence:** `cc-by-4.0`
- **Underlying Provenance:** Stack Exchange network dumps.
- **Legal Reality:** Well-aligned with Stack Exchange terms. CC BY-4.0 requires attribution to the author and Stack Exchange, permitting adaptation and academic research without strict ShareAlike encumbrance.

---

## 6. Label Construction Feasibility & Anti-Leakage Protocol

### 6.1 Target Label Construction Feasibility
Neither dataset provides a ready-made coarse topic target. We must construct ground-truth labels using a tag taxonomy mapping:

$$\text{Raw Tags} \xrightarrow{\text{Taxonomy Rule in [configs/topic_map.yaml](../configs/topic_map.yaml)}} \text{Target Topic } y \in \{1, \dots, C\}$$

- **In StackPulse:** High feasibility. Tags are cleanly delimited by `|`. Each post has 1–5 tags (mean 3.01). Compound tags (`react-native`, `data-structures`) are preserved.
- **In Habedi:** Moderate feasibility. Requires a curated multi-word vocabulary to avoid splitting phrases like `operating systems` into `operating` and `systems`.

### 6.2 Multi-Topic Conflict Resolution Policy
Technical questions frequently span multiple engineering topics. In our sample inspection:
- `c#|android|xaml|xamarin.forms` matches both **Mobile Development** and general language frameworks.
- `python|json|amazon-web-services|lambda|amazon-dynamodb` matches **DevOps / Cloud Infrastructure**, **Databases & Storage**, and **Web Backend**.
- `ios|swift|data-structures|enums|swift2` matches **Mobile Development** and **Algorithms & Data Structures**.

**Operational Rule (per [configs/topic_map.yaml](../configs/topic_map.yaml)):**
1. Map each tag in a question's tag list to candidate coarse topics.
2. Compute the topic frequency distribution for the question.
3. If the top topic achieves a dominance ratio $\ge 0.60$ (or strictly dominates all other candidate matches), assign that topic as ground truth.
4. If a question exhibits an unresolvable tie or broad multi-domain ambiguity (e.g., 50/50 split across two distinct topics), **drop the question from the training set and log the drop event**.

### 6.3 Strict Target Leakage Prevention Barrier
> [!CAUTION]
> **Anti-Leakage Mandate:** Tags are used exclusively to construct the supervisory training target $y$. Once $y$ is assigned, the tag strings and individual tag tokens **must be completely stripped from the input feature space**. Model 1 inputs must be restricted strictly to cleaned question text (`title` + `body_text`). Passing tags as input features would allow the classifier to memorize tag-matching rules rather than learning semantic representations from natural text.

---

## 7. Evaluation of Provisional Taxonomy ([configs/topic_map.yaml](../configs/topic_map.yaml))

[configs/topic_map.yaml](../configs/topic_map.yaml) outlines 10 provisional coarse categories. We evaluated their representation against empirical tag frequencies observed in the dataset samples and documentation:

| Provisional Topic ID | Display Name | Representation in StackPulse | Representation in Habedi | Assessment & Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| `web_frontend` | Web Frontend Development | **Very High** (`javascript`, `html`, `css`, `angularjs`, `reactjs`, `typescript`) | Minimal (Off-topic on CS SE) | Retain for StackPulse; well-supported. |
| `data_science_and_ml` | Data Science & ML | **High** (`python`, `pandas`, `numpy`, `r`, `dataframe`, `scikit-learn`) | **High** (Dominant in `ds.tsv`) | Retain; strong overlap in both datasets. |
| `mobile_development` | Mobile App Development | **High** (`android`, `ios`, `swift`, `xamarin.forms`, `flutter`) | None (Off-topic on CS SE) | Retain for StackPulse; clearly separable. |
| `devops_cloud_and_infrastructure` | DevOps & Cloud | **High** (`docker`, `kubernetes`, `aws`, `lambda`, `git`, `ci/cd`) | None | Retain for StackPulse; modern engineering relevance. |
| `databases_and_storage` | Databases & Storage | **Medium-High** (`sql`, `dynamodb`, `mongodb`, `indexing`) | **High** (`database theory`, `relational algebra`) | Retain; fundamental across both practical & theoretical CS. |
| `operating_systems_and_systems` | OS & Systems | **Medium** (`windows`, `shell`, `cygwin`, `c++`, `posix`) | **Very High** (`operating systems`, `scheduling`, `memory`) | Retain; core engineering subject. |
| `algorithms_and_data_structures` | Algorithms & Data Structures| **Medium** (`arrays`, `sorting`, `trees`, `data-structures`) | **Dominant** (`algorithms`, `complexity theory`, `sorting`) | Retain; central to both datasets. |
| `web_backend_and_apis` | Web Backend & APIs | **Medium** (`django`, `flask`, `rest`, `api`, `node.js`, `express`) | Minimal | Retain; distinct from frontend. |
| `software_engineering_and_testing` | SE & Testing | **Low-Medium** (`unit-testing`, `design-patterns`, `junit`) | **Low** (`software engineering`) | Requires careful tag curation; lower raw frequency. |
| `computer_networks_and_security` | Networks & Security | **Low-Medium** (`sockets`, `tcp`, `ssl`, `cryptography`) | **Medium** (`computer networks`, `cryptography`) | Retain; essential engineering topic. |

### Critical Taxonomy Observations & Gaps
1. **The "General Programming Language" Ambiguity in StackPulse:**
   - StackPulse contains hundreds of thousands of generic syntax and data-type questions tagged only with language names and generic constructs (e.g., `python|string|list|replace` or `javascript|arrays`).
   - If `python` is arbitrarily mapped to `data_science_and_ml`, or `javascript` exclusively to `web_frontend`, basic programming questions will introduce severe label noise.
   - *Recommendation:* Tags like `python` or `javascript` must be treated as context-dependent modifiers, or a 11th category (e.g., `programming_fundamentals_and_syntax`) should be considered if StackPulse is ratified.
2. **Missing Theoretical Topics in Habedi:**
   - If Habedi is chosen, `theory_of_computation_and_formal_languages` (`automata`, `formal grammars`, `computability`, `turing machines`) is one of the largest tag clusters on CS Stack Exchange and is currently absent from [configs/topic_map.yaml](../configs/topic_map.yaml).

---

## 8. Sample-Size Feasibility (Decision D2)

Planning documents propose an initial stratified sample of **100,000–200,000 rows** for Model 1 (Decision D2).

### 8.1 Measured & Verified Dataset Dimensions (Ground Truth from APIs)
The following metrics are verified directly from Hugging Face dataset metadata and `/size` APIs:
- **StackPulse Base Row Count:** 778,929 rows (in `full` configuration).
- **StackPulse Verified Storage Size:** 997,103,974 bytes (~997.1 MB) in compressed Parquet files across shards on Hugging Face; uncompressed memory footprint reported at 2,043,807,304 bytes (~2.04 GB).
- **Habedi Base Row Count:** 82,158 rows (in `default` configuration).
- **Habedi Verified Storage Size:** 102,693,671 bytes (~102.7 MB) raw original files; 53,247,727 bytes (~53.2 MB) in Parquet; uncompressed memory footprint 103,491,121 bytes (~103.5 MB).

### 8.2 Feasibility Assessment
- **StackPulse Suitability:** Fully suitable for sample-size targets. The verified 778,929 rows provide ample volume to draw 100,000–200,000 stratified rows while preserving large independent test sets.
- **Habedi Suitability:** **Infeasible for targets $\ge 100\text{k}$.** The entire dataset has only 82,158 rows. After filtering non-technical posts (e.g., Political Science with ~11k rows) and unmapped tags, only approximately 50,000–60,000 candidate rows remain. If Habedi is chosen, Decision D2 must be updated to an adjusted target of 50,000 rows.

### 8.3 Resource Requirements: Projected Engineering Estimates vs Measured Data
> [!NOTE]
> **Distinction:** Model training and sample extraction have **not** been executed during this audit. The following resource figures are **projected engineering estimates** derived from proportional scaling and standard text-processing benchmarks, not measured empirical runtime metrics.

- **Projected 100,000-Row Sample File Size (Estimated):**
  - If all 24 columns are retained: Projected at $\sim 125\text{--}130\text{ MB}$ compressed Parquet (scaled linearly from the verified 997.1 MB for 778.9k rows: $100{,}000 / 778{,}929 \times 997.1\text{ MB} \approx 128\text{ MB}$).
  - If pruned to Model 1 pipeline features (`id`, `title`, `body_text`, `tags`, derived `topic`): Projected at $\sim 45\text{--}65\text{ MB}$ compressed Parquet.
- **Projected In-Memory DataFrame (Estimated):**
  - Projected at $\sim 250\text{--}350\text{ MB}$ RAM for pruned columns.
- **Projected Vectorization Memory Overhead (Estimated):**
  - A sparse TF-IDF matrix (100,000 rows $\times$ 10,000 vocabulary unigram/bigram features) with $\sim 100$ non-zero entries per document requires $\sim 10^7$ floats, consuming $\sim 80\text{--}120\text{ MB}$ in SciPy CSR format.
  - Peak processing memory during tokenization and n-gram extraction is estimated at $\sim 1.0\text{--}1.5\text{ GB}$ RAM, which is well within free Google Colab (12 GB RAM) and commodity local CPU memory.
- **Projected Training Execution Time (Estimated):**
  - Linear baselines (M1-A Majority, M1-C Logistic Regression, M1-D Linear SVM, M1-E Complement NB) on a 100k sparse matrix are projected to train in under 2–5 minutes on modern CPU cores.
- **Recommendation:** Ratify `sample_size: 100000` in [configs/params.yaml](../configs/params.yaml) once Decision D0 is formally signed off.

---

## 9. Governance & Outstanding Decisions

### 9.1 Status of Decision D0 (Dataset Choice)
- **Status:** **EXPLICITLY PENDING FACULTY REVIEW.**
- No formal faculty approval is recorded in repository documentation or commit history.
- [configs/params.yaml](../configs/params.yaml) correctly maintains `approval_status: "pending_review_and_profiling"`.
- [configs/topic_map.yaml](../configs/topic_map.yaml) correctly maintains `taxonomy_metadata.status: "provisional_scaffold"`.

### 9.2 Recommendation for the Team
| Dimension | Candidate A: StackPulse | Candidate B: Habedi Stack Exchange |
| :--- | :--- | :--- |
| **Course Lifecycle Fit** | **Superior:** Supports 100k sample, clean pipe tags, pre-normalized text, modern ML engineering scale. | Moderate: Clean license, but small row count (82k) and space-delimited tags. |
| **Curriculum Alignment** | Moderate: Applied software engineering & framework questions. | **Superior:** Direct alignment with theoretical computer science exam questions. |
| **Operational Risk** | Upstream CC BY-SA terms; licensing suitability unresolved pending institutional/faculty guidance. | Tag parsing complexity; inability to meet 100k sample size. |

**Recommended Next Step:**
Submit this audit document to the course instructor/faculty supervisor during the Week 1 review to obtain formal sign-off on Decision D0. The team should recommend **Candidate A (StackPulse)** for Model 1 due to its superior schema cleanliness, tag delimitation, and scale, while noting that if the faculty desires pure academic exam theory, **Candidate B (Habedi)** can be adopted with an adjusted sample size (50k rows).

---

## 10. Audit Execution Log & Verification Trail

All findings in this report were produced using the following verified, reproducible commands and API calls:

1. **Local Git State Audit:**
   ```bash
   git status --short --branch
   git branch --show-current
   git remote -v
   git log -5 --oneline --decorate
   ```
2. **Hugging Face Hub Metadata API Queries:**
   - StackPulse Hub Info: `https://huggingface.co/api/datasets/Omarrran/StackPulse_778K_QnA_Code_dataset`
   - Habedi Hub Info: `https://huggingface.co/api/datasets/habedi/stack-exchange-dataset`
3. **Hugging Face Datasets-Server Size & Info API Queries:**
   - StackPulse Configurations & Splits: `https://datasets-server.huggingface.co/info?dataset=Omarrran/StackPulse_778K_QnA_Code_dataset`
   - StackPulse Exact Split Sizes: `https://datasets-server.huggingface.co/size?dataset=Omarrran/StackPulse_778K_QnA_Code_dataset`
   - Habedi Dataset Info & Sizes: `https://datasets-server.huggingface.co/size?dataset=habedi/stack-exchange-dataset`
4. **Streaming First-Rows Sample Inspection (without downloading full data):**
   - StackPulse Sample: `https://datasets-server.huggingface.co/first-rows?dataset=Omarrran/StackPulse_778K_QnA_Code_dataset&config=full&split=train`
   - Habedi Sample: `https://datasets-server.huggingface.co/first-rows?dataset=habedi/stack-exchange-dataset&config=default&split=train`
5. **Raw Documentation Extraction:**
   - Direct inspection of repository `README.md` files for author provenance, sampling dates, known issues, and licensing disclaimers.
