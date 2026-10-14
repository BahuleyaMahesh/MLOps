import pandas as pd
import pytest

from src.grouped_split import create_question_grouped_holdout, get_group_kfold_splits


def test_group_kfold_splits_have_zero_group_leakage():
    frame = pd.DataFrame(
        {
            "question": ["Q1", "Q1", "Q2", "Q2", "Q3", "Q3", "Q4", "Q4", "Q5", "Q5"],
            "student_answer": ["A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2", "E1", "E2"],
            "score": [2, 1, 2, 0, 1, 1, 2, 0, 0, 1],
        }
    )

    splits = get_group_kfold_splits(frame, group_col="question", n_splits=5)
    assert len(splits) == 5

    for train_idx, val_idx in splits:
        train_groups = set(frame.iloc[train_idx]["question"])
        val_groups = set(frame.iloc[val_idx]["question"])
        assert len(train_groups & val_groups) == 0
        assert len(train_idx) + len(val_idx) == len(frame)


def test_question_grouped_holdout_zero_leakage():
    frame = pd.DataFrame(
        {
            "question": [f"Q{i}" for i in range(10) for _ in range(3)],
            "student_answer": [f"ans_{i}" for i in range(30)],
            "score": [1] * 30,
        }
    )

    train_df, holdout_df = create_question_grouped_holdout(
        frame, group_col="question", test_size=0.3, random_state=42
    )

    train_qs = set(train_df["question"])
    holdout_qs = set(holdout_df["question"])

    assert len(train_qs & holdout_qs) == 0
    assert len(train_df) + len(holdout_df) == 30


def test_group_kfold_rejects_fewer_questions_than_splits():
    frame = pd.DataFrame(
        {
            "question": ["Q1", "Q1", "Q2"],
            "student_answer": ["A", "B", "C"],
            "score": [0, 1, 2],
        }
    )

    with pytest.raises(ValueError, match="Cannot create 5 group folds"):
        get_group_kfold_splits(frame, group_col="question", n_splits=5)
