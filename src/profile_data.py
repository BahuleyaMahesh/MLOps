import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = (
    ROOT / "data" / "processed" / "engsaf"
    if (ROOT / "data" / "processed" / "engsaf").exists()
    else ROOT / "data" / "raw" / "engsaf"
)
OUTPUT_PATH = ROOT / "reports" / "data_profile.json"
EXPECTED_COLUMNS = [
    "question",
    "student_answer",
    "reference_answer",
    "mark_scheme",
    "score",
    "rationale",
]


def profile_split(path: Path) -> dict:
    frame = pd.read_csv(path)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{path.name} is missing columns: {missing_columns}")

    return {
        "rows": int(len(frame)),
        "columns": EXPECTED_COLUMNS,
        "missing_values": {
            column: int(frame[column].isna().sum()) for column in EXPECTED_COLUMNS
        },
        "score_counts": {
            str(int(score)): int(count)
            for score, count in frame["score"].value_counts(dropna=False).items()
        },
        "distinct_questions": int(frame["question"].nunique(dropna=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile EngSAF dataset splits.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to data directory")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Path to profile output JSON")
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
            name: profile_split(data_dir / f"{name}.csv")
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote aggregate-only profile to {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
