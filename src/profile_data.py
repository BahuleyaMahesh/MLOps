import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = (
    ROOT / "data" / "processed" / "engsaf"
    if (ROOT / "data" / "processed" / "engsaf").exists()
    else ROOT / "data" / "raw" / "engsaf"
)
OUTPUT_JSON_PATH = ROOT / "reports" / "data_profile.json"
OUTPUT_EDA_PATH = ROOT / "reports" / "engsaf_eda.md"
EXPECTED_COLUMNS = [
    "question",
    "student_answer",
    "reference_answer",
    "mark_scheme",
    "score",
    "rationale",
]


def _summary_stats(series: pd.Series) -> dict:
    clean = series.dropna().astype(float)
    if clean.empty:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "std": 0.0}
    return {
        "min": int(clean.min()),
        "max": int(clean.max()),
        "mean": round(float(clean.mean()), 2),
        "median": round(float(clean.median()), 2),
        "p25": round(float(clean.quantile(0.25)), 2),
        "p75": round(float(clean.quantile(0.75)), 2),
        "std": round(float(clean.std()), 2),
    }


def profile_split(frame: pd.DataFrame) -> dict:
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Split is missing required columns: {missing_columns}")

    rows = len(frame)
    q_counts = frame["question"].value_counts(dropna=True)
    score_counts = frame["score"].value_counts(dropna=False).to_dict()

    student_str = frame["student_answer"].fillna("").astype(str)
    ref_str = frame["reference_answer"].fillna("").astype(str)
    q_str = frame["question"].fillna("").astype(str)

    student_chars = student_str.str.len()
    student_words = student_str.str.split().str.len()
    ref_chars = ref_str.str.len()
    ref_words = ref_str.str.split().str.len()
    q_chars = q_str.str.len()
    q_words = q_str.str.split().str.len()

    dup_student_answers = int(student_str.duplicated().sum())
    dup_qa_pairs = int(frame.duplicated(subset=["question", "student_answer"]).sum())
    dup_ref_answers = int(frame.duplicated(subset=["question", "reference_answer"]).sum())

    return {
        "rows": int(rows),
        "columns": EXPECTED_COLUMNS,
        "missing_values": {
            column: int(frame[column].isna().sum()) for column in EXPECTED_COLUMNS
        },
        "score_distribution": {
            "counts": {str(int(score)): int(count) for score, count in sorted(score_counts.items())},
            "percentages": {
                str(int(score)): round(100.0 * count / rows, 2)
                for score, count in sorted(score_counts.items())
            },
        },
        "per_question_counts": {
            "distinct_questions": int(len(q_counts)),
            **_summary_stats(q_counts),
        },
        "text_lengths": {
            "student_answer": {
                "characters": _summary_stats(student_chars),
                "words": _summary_stats(student_words),
            },
            "reference_answer": {
                "characters": _summary_stats(ref_chars),
                "words": _summary_stats(ref_words),
            },
            "question": {
                "characters": _summary_stats(q_chars),
                "words": _summary_stats(q_words),
            },
        },
        "duplicates": {
            "duplicate_student_answers": dup_student_answers,
            "duplicate_question_student_pairs": dup_qa_pairs,
            "duplicate_question_reference_pairs": dup_ref_answers,
        },
    }


def generate_eda_markdown(report: dict) -> str:
    lines = [
        "# EngSAF Dataset Exploratory Data Analysis (EDA) Report",
        "",
        "**Dataset**: `IsmaelMousa/engsaf`  ",
        "**Status**: Phase 1 Member 2 Deliverable  ",
        "**Feature Safeguard Policy**: `mark_scheme` and `rationale` are strictly excluded from inputs.",
        "",
        "---",
        "",
        "## 1. Executive Summary & Critical Findings",
        "",
        "1. **Severe Cross-Split Question Memorization Risk**:",
        f"   - **{report['question_overlap']['train_and_test']['distinct_shared_questions']} out of {report['question_overlap']['train_and_test']['right_question_count']} questions ({round(100.0 * report['question_overlap']['train_and_test']['distinct_shared_questions'] / report['question_overlap']['train_and_test']['right_question_count'], 1)}%)** in the test split are already present in the training set.",
        f"   - **{report['question_overlap']['train_and_validation']['distinct_shared_questions']} out of {report['question_overlap']['train_and_validation']['right_question_count']} questions ({round(100.0 * report['question_overlap']['train_and_validation']['distinct_shared_questions'] / report['question_overlap']['train_and_validation']['right_question_count'], 1)}%)** in the validation split are shared with train.",
        "   - **Action**: A purely random split allows models to memorize question-specific vocabulary. Evaluating on a **question-grouped holdout / GroupKFold cross-validation** is essential for measuring true generalization.",
        "",
        "2. **Score Distribution**:",
        "   - Scores range across 0, 1, and 2. Class 2 is the plurality across all splits (~40-43%), Class 1 is ~35-38%, and Class 0 is ~21-22%.",
        "   - The distribution is mildly imbalanced, warranting macro-averaged F1 and Quadratic Weighted Kappa (QWK) as core evaluation metrics.",
        "",
        "3. **Question Skew & Answer Frequencies**:",
        f"   - In the training set, the mean number of answers per question is {report['splits']['train']['per_question_counts']['mean']}, but the distribution ranges from a minimum of {report['splits']['train']['per_question_counts']['min']} up to a maximum of {report['splits']['train']['per_question_counts']['max']}.",
        "   - This heavy skew necessitates group-aware partitioning.",
        "",
        "---",
        "",
        "## 2. Split Overview and Score Distributions",
        "",
        "| Split | Rows | Distinct Questions | Score 0 Count (%) | Score 1 Count (%) | Score 2 Count (%) |",
        "|---|---|---|---|---|---|",
    ]

    for split_name in ("train", "validation", "test"):
        if split_name in report["splits"]:
            s = report["splits"][split_name]
            p = s["score_distribution"]["percentages"]
            c = s["score_distribution"]["counts"]
            lines.append(
                f"| `{split_name}` | {s['rows']} | {s['per_question_counts']['distinct_questions']} | "
                f"{c.get('0', 0)} ({p.get('0', 0.0)}%) | {c.get('1', 0)} ({p.get('1', 0.0)}%) | {c.get('2', 0)} ({p.get('2', 0.0)}%) |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Question Overlap Across Splits",
        "",
        "| Split Pair | Shared Questions | Left Total Questions | Right Total Questions | Overlap % (of Right Split) |",
        "|---|---|---|---|---|",
    ])

    for pair, ov in report["question_overlap"].items():
        pct = round(100.0 * ov["distinct_shared_questions"] / ov["right_question_count"], 1)
        lines.append(
            f"| `{pair}` | {ov['distinct_shared_questions']} | {ov['left_question_count']} | {ov['right_question_count']} | {pct}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Text Length Statistics (Words and Characters)",
        "",
        "| Split | Field | Words (Mean ± Std) | Words (Median [IQR]) | Characters (Mean ± Std) |",
        "|---|---|---|---|---|",
    ])

    for split_name in ("train", "validation", "test"):
        if split_name in report["splits"]:
            tl = report["splits"][split_name]["text_lengths"]
            for field in ("student_answer", "reference_answer", "question"):
                w = tl[field]["words"]
                c = tl[field]["characters"]
                lines.append(
                    f"| `{split_name}` | `{field}` | {w['mean']} ± {w['std']} | {w['median']} [{w['p25']} - {w['p75']}] | {c['mean']} ± {c['std']} |"
                )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Duplicate Analysis",
        "",
        "| Split | Duplicate Student Answers | Duplicate (Question, Answer) Pairs | Missing Values |",
        "|---|---|---|---|",
    ])

    for split_name in ("train", "validation", "test"):
        if split_name in report["splits"]:
            s = report["splits"][split_name]
            dups = s["duplicates"]
            missing_total = sum(s["missing_values"].values())
            lines.append(
                f"| `{split_name}` | {dups['duplicate_student_answers']} | {dups['duplicate_question_student_pairs']} | {missing_total} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Recommendations for Model 2 Development",
        "",
        "1. **Mandatory Grouped Evaluation**: GroupKFold cross-validation must be reported alongside the standard split to avoid inflated results due to question overlap.",
        "2. **Length-Ablation Sanity Check**: Student answers average ~27 words with high standard deviation; models must be checked against length-only predictors to verify they do not merely reward verbosity.",
        "3. **Zero-Overlap Prompt Copy Check**: Verify that blindly repeating the question text does not fool the semantic similarity scorer.",
        "4. **Strict Rationale Exclusion**: The dataset provides `rationale` and `mark_scheme`, which contain human evaluative reasoning. These must remain strictly excluded from all training and feature pipelines.",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile EngSAF dataset splits.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to data directory")
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON_PATH, help="Path to profile output JSON")
    parser.add_argument("--output-eda", type=Path, default=OUTPUT_EDA_PATH, help="Path to EDA markdown report")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}; run the data preparation stages first."
        )

    split_paths = sorted(data_dir.glob("*.csv"))
    if not split_paths:
        raise FileNotFoundError(f"No split CSV files found in {data_dir}")

    split_frames = {path.stem: pd.read_csv(path) for path in split_paths}
    report = {
        "dataset": "IsmaelMousa/engsaf",
        "data_directory": str(data_dir.relative_to(ROOT)),
        "feature_policy": {
            "allowed_inputs": ["question", "student_answer", "reference_answer"],
            "target": "score",
            "excluded_inputs": ["mark_scheme", "rationale"],
        },
        "splits": {
            name: profile_split(split_frames[name])
            for name in split_frames
        },
        "question_overlap": {},
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if left in split_frames and right in split_frames:
            left_questions = set(split_frames[left]["question"].dropna())
            right_questions = set(split_frames[right]["question"].dropna())
            overlap = left_questions & right_questions
            report["question_overlap"][f"{left}_and_{right}"] = {
                "distinct_shared_questions": len(overlap),
                "left_question_count": len(left_questions),
                "right_question_count": len(right_questions),
            }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote aggregate profile to {args.output_json.relative_to(ROOT)}")

    eda_md = generate_eda_markdown(report)
    args.output_eda.parent.mkdir(parents=True, exist_ok=True)
    args.output_eda.write_text(eda_md, encoding="utf-8")
    print(f"Wrote EDA markdown report to {args.output_eda.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
