"""Text and numeric features for the answer-scoring baseline."""

import re

import pandas as pd


TEXT_INPUTS = ("question", "student_answer", "reference_answer")
FORBIDDEN_INPUTS = ("mark_scheme", "rationale")
FEATURE_COLUMNS = (
    "question",
    "student_answer",
    "reference_answer",
    "student_reference_jaccard",
    "student_reference_overlap",
    "answer_reference_length_ratio",
    "student_question_overlap",
)
TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def _tokens(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.casefold()))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select safe text inputs and add simple lexical/length features."""
    missing = sorted(set(TEXT_INPUTS) - set(frame.columns))
    if missing:
        raise ValueError(f"Required text columns are missing: {missing}")

    features = frame.loc[:, list(TEXT_INPUTS)].fillna("").astype(str).copy()
    jaccard_scores = []
    overlap_scores = []
    length_ratios = []
    question_overlap_scores = []

    for row in features.itertuples(index=False, name=None):
        question, student_answer, reference_answer = row
        student_tokens = _tokens(student_answer)
        reference_tokens = _tokens(reference_answer)
        question_tokens = _tokens(question)
        intersection = student_tokens & reference_tokens

        jaccard_scores.append(
            _ratio(len(intersection), len(student_tokens | reference_tokens))
        )
        overlap_scores.append(_ratio(len(intersection), len(student_tokens)))
        length_ratios.append(
            _ratio(len(student_tokens), len(reference_tokens))
        )
        question_overlap_scores.append(
            _ratio(len(student_tokens & question_tokens), len(student_tokens))
        )

    features["student_reference_jaccard"] = jaccard_scores
    features["student_reference_overlap"] = overlap_scores
    features["answer_reference_length_ratio"] = length_ratios
    features["student_question_overlap"] = question_overlap_scores
    return features
