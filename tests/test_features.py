import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, build_features


def test_features_include_overlap_and_length_signals():
    frame = pd.DataFrame(
        {
            "question": ["What is a tree?"],
            "student_answer": ["A tree has root and leaves"],
            "reference_answer": ["A tree has a root and leaves"],
            "score": [2],
            "rationale": ["leakage should not enter features"],
            "mark_scheme": ["also excluded"],
        }
    )

    features = build_features(frame)

    assert tuple(features.columns) == FEATURE_COLUMNS
    assert features.loc[0, "student_reference_jaccard"] > 0
    assert features.loc[0, "answer_reference_length_ratio"] > 0
    assert "rationale" not in features
    assert "mark_scheme" not in features


def test_features_handle_missing_text_as_empty():
    frame = pd.DataFrame(
        {
            "question": [None],
            "student_answer": [""],
            "reference_answer": [None],
        }
    )

    features = build_features(frame)

    assert features.loc[0, "student_reference_jaccard"] == 0
    assert features.loc[0, "student_reference_overlap"] == 0
    assert features.loc[0, "answer_reference_length_ratio"] == 0


def test_features_reject_missing_required_inputs():
    with pytest.raises(ValueError, match="question"):
        build_features(pd.DataFrame({"student_answer": [""], "reference_answer": [""]}))
