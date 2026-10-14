"""Pytest suite for dataset validation, schema constraints, and score ranges."""

from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "engsaf"
RAW_DATA_DIR = ROOT / "data" / "raw" / "engsaf"

REQUIRED_COLUMNS = [
    "question",
    "student_answer",
    "reference_answer",
    "score",
]
EXPECTED_SCORES = {0, 1, 2}


@pytest.fixture(scope="module")
def processed_splits() -> dict[str, pd.DataFrame]:
    splits = {}
    for name in ("train", "validation", "test"):
        path = DATA_DIR / f"{name}.csv"
        if not path.exists():
            pytest.skip(f"Processed dataset split not found at {path}")
        splits[name] = pd.read_csv(path)
    return splits


def test_processed_splits_schema(processed_splits):
    """Verify all splits contain the required columns with no unexpected missing column names."""
    for name, df in processed_splits.items():
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        assert not missing, f"Split {name} is missing required columns: {missing}"
        assert len(df) > 0, f"Split {name} has zero rows"


def test_processed_scores_range_and_type(processed_splits):
    """Verify all scores are strictly integer values within {0, 1, 2}."""
    for name, df in processed_splits.items():
        assert df["score"].notna().all(), f"Split {name} contains null scores"
        unique_scores = set(df["score"].astype(int).unique())
        invalid_scores = unique_scores - EXPECTED_SCORES
        assert not invalid_scores, f"Split {name} contains invalid scores: {invalid_scores}"


def test_processed_text_non_null(processed_splits):
    """Verify question, student_answer, and reference_answer are non-null and valid strings."""
    for name, df in processed_splits.items():
        for col in ("question", "reference_answer"):
            # Questions and reference answers must not be blank
            assert df[col].notna().all(), f"Split {name} has NaN in {col}"
            assert (df[col].astype(str).str.strip().str.len() > 0).all(), f"Split {name} has blank {col}"


def test_no_identical_duplicate_rows(processed_splits):
    """Verify duplicate rows across question, student_answer, reference_answer are bounded per EDA findings."""
    for name, df in processed_splits.items():
        subset = ["question", "student_answer", "reference_answer"]
        duplicates = df.duplicated(subset=subset).sum()
        # Per reports/engsaf_eda.md, train has 22 benign identical short answers across different students
        assert duplicates <= 25, f"Split {name} contains unexpectedly high ({duplicates}) duplicate rows"


def test_score_distribution_non_empty(processed_splits):
    """Verify all score classes (0, 1, 2) have representation in train split."""
    train_scores = set(processed_splits["train"]["score"].unique())
    assert train_scores == {0, 1, 2}, f"Train split missing some score classes: {train_scores}"
