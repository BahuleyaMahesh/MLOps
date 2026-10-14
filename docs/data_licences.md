# Dataset Licensing, Governance & Compliance Record

**Project:** AI-Powered Study Assistant for Engineering Students (AI252TA — SDG 4)<br>
**Status:** Audit & Verification in Progress (Phase 1)

---

## 1. Overview & Regulatory Posture

As part of the academic MLOps course rubric, all data sources must undergo rigorous licensing audits, governance checks, and formal faculty approval prior to model training.

> **CRITICAL COMPLIANCE NOTICE:**
> No dataset is currently marked as formally approved. Formal faculty sign-off on dataset acquisition is pending resolution of Decision D0. Unresolved licenses must never be claimed as approved.

---

## 2. Dataset Licences & Known Concerns

### 2.1 EngSAF (`IsmaelMousa/engsaf`) — Model 2 (Answer Scorer)

| Attribute | Details |
| :--- | :--- |
| **Source URL** | Hugging Face: `IsmaelMousa/engsaf` |
| **Associated Paper** | *EngSAF: Short Answer Feedback Dataset for Engineering Education* (arXiv:2407.12818, AIED 2025) |
| **Hugging Face Hub Tag** | `apache-2.0` |
| **Dataset Card Text** | "Strictly for academic and research purposes. No commercial use or redistribution without explicit permission." |
| **Licence Conflict & Risk** | **Direct Conflict between Hub metadata and Author card text.** The permissive `apache-2.0` metadata tag contradicts the restrictive card body. |
| **Governing Policy** | **Treat the stricter statement as binding.** |
| **Actionable Constraints** | 1. Raw and processed EngSAF data must **NEVER** be committed to public GitHub repositories or public cloud buckets.<br>2. Storage must be restricted to access-controlled DVC remotes (e.g., private DagsHub storage).<br>3. Appropriate citation to arXiv:2407.12818 must be included in all reports. |

---

### 2.2 StackPulse (`Omarrran/StackPulse_778K_QnA_Code_dataset`) — Model 1 (Topic Classifier)

| Attribute | Details |
| :--- | :--- |
| **Source URL** | Hugging Face: `Omarrran/StackPulse_778K_QnA_Code_dataset` |
| **Hugging Face Hub Tag** | `apache-2.0` |
| **Underlying Data Origin** | Stack Overflow programming Q&A community dump. |
| **Downstream Licensing Risk** | Stack Overflow user-contributed content is governed by **Creative Commons Attribution-ShareAlike (CC BY-SA 3.0 / 4.0)**.<br>An Apache 2.0 tag on Hugging Face cannot unilaterally override CC BY-SA obligations attached to the underlying Stack Exchange data dump. |
| **Attribution & SA Terms** | CC BY-SA requires explicit author attribution, source link preservation, and sharing downstream adaptations under compatible terms. |
| **Current Status** | **UNVERIFIED.** Exact terms, schema, and dump extraction metadata have not yet been audited. |

---

### 2.3 Alternative Dataset: `habedi/stack-exchange-dataset` (PRD Original)

| Attribute | Details |
| :--- | :--- |
| **Source URL** | Hugging Face: `habedi/stack-exchange-dataset` |
| **Hub Licence Tag** | `cc-by-4.0` |
| **Coverage** | 82,158 questions from CS, Data Science, and Political Science Stack Exchange (up to May 2021). |
| **Licence Status** | Well-documented CC BY-4.0 (requires attribution, allows adaptation). |

---

## 3. Outstanding Verification & Compliance Checklist

- [ ] **Task L1 (Member 1):** Ingest a minimal sample of StackPulse in Colab, inspect metadata files and JSONL headers, and determine whether CC BY-SA attribution requirements are documented.
- [ ] **Task L2 (Member 2):** Document exact institutional usage constraints for EngSAF with course faculty.
- [ ] **Task L3 (Team Decision D0):** Review trade-offs between StackPulse (programming focus, uncertain schema/licence) vs `habedi/stack-exchange-dataset` (clear CC BY-4.0, verified schema).
- [ ] **Task L4 (Member 4):** Prepare formal dataset acquisition and approval memo for faculty sign-off.
- [ ] **Task L5 (Member 3):** Ensure DVC remote authentication is restricted to authenticated team members to avoid unintended public redistribution of EngSAF.
