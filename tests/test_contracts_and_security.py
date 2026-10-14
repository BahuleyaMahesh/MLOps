"""Pytest suite for security boundaries, contract validation, and edge-case handling.

Validates:
1. Leakage prevention: rationale and mark_scheme must NEVER enter features.
2. Prediction contracts: shape consistency, discrete class boundaries {0, 1, 2}, valid probabilities.
3. Edge cases: empty text, extreme length text, malicious HTML/scripts, math LaTeX, unicode/emojis.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from src.features import FORBIDDEN_INPUTS, build_features
from src.train_baseline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


@pytest.fixture
def params() -> dict:
    with (ROOT / "params.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def trained_pipeline(params) -> object:
    model_path = MODELS_DIR / "m2_c_logistic_regression.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    # Fallback to in-memory trained pipeline if model artifact not yet saved
    pipe = make_pipeline("logistic_regression", params)
    dummy_data = pd.DataFrame(
        {
            "question": ["What is recursion?"] * 6,
            "student_answer": [
                "Recursion is a function calling itself until base case.",
                "Looping repeatedly.",
                "Don't know.",
                "Function calls itself.",
                "A data structure.",
                "Base case and recursive step.",
            ],
            "reference_answer": ["Recursion is a method where a function calls itself."] * 6,
        }
    )
    y = np.array([2, 1, 0, 2, 0, 2])
    feats = build_features(dummy_data, fit_semantic=True)
    pipe.fit(feats, y)
    return pipe


def test_strict_no_rationale_leakage():
    """Ensure that even if forbidden columns (rationale, mark_scheme) exist in raw data, they are stripped."""
    dirty_data = pd.DataFrame(
        {
            "question": ["Define polymorphism."],
            "student_answer": ["Polymorphism allows objects to take multiple forms."],
            "reference_answer": ["The ability of different classes to respond to the same message."],
            "rationale": ["Full marks: student correctly identified multiple forms."],
            "mark_scheme": ["2 marks for multiple forms; 1 mark for vague answer."],
            "score": [2],
            "secret_teacher_notes": ["Student did well."],
        }
    )
    features = build_features(dirty_data, fit_semantic=True)

    for forbidden in FORBIDDEN_INPUTS:
        assert forbidden not in features.columns, f"CRITICAL LEAKAGE: {forbidden} found in features!"
    assert "secret_teacher_notes" not in features.columns
    assert "score" not in features.columns


def test_prediction_contract_shape_and_classes(trained_pipeline):
    """Verify prediction output adheres to shape, discrete values {0, 1, 2}, and valid probabilities."""
    sample_df = pd.DataFrame(
        {
            "question": [
                "What is an algorithm?",
                "Explain inheritance.",
                "What is a binary tree?",
            ],
            "student_answer": [
                "A step-by-step procedure to solve a computational problem.",
                "Children inherit properties from parent classes.",
                "A tree where each node has at most two children.",
            ],
            "reference_answer": [
                "An algorithm is a finite sequence of well-defined instructions.",
                "Mechanism where one class derives attributes from another.",
                "A hierarchical data structure with up to two children per node.",
            ],
        }
    )
    feats = build_features(sample_df, fit_semantic=True)

    preds = trained_pipeline.predict(feats)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (3,)
    assert set(preds).issubset({0, 1, 2}), f"Predictions contained invalid classes: {set(preds)}"

    if hasattr(trained_pipeline, "predict_proba"):
        probs = trained_pipeline.predict_proba(feats)
        assert probs.shape == (3, 3)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
        # Row sums equal 1.0 within numerical precision
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(3), rtol=1e-5)


def test_edge_case_empty_and_whitespace_text(trained_pipeline):
    """Test model resilience against empty strings, pure whitespaces, and tabs."""
    empty_df = pd.DataFrame(
        {
            "question": ["What is a pointer?", "Explain memory allocation."],
            "student_answer": ["", "   \t\n   "],
            "reference_answer": ["A variable that stores a memory address.", "Reserving memory blocks."],
        }
    )
    feats = build_features(empty_df, fit_semantic=True)
    preds = trained_pipeline.predict(feats)
    assert len(preds) == 2
    assert set(preds).issubset({0, 1, 2})


def test_edge_case_extreme_text_length(trained_pipeline):
    """Test robustness when student answer is excessively long (10,000 words)."""
    long_answer = "recursion base case termination stack overflow " * 2000  # 10,000 words
    long_df = pd.DataFrame(
        {
            "question": ["Explain recursion."],
            "student_answer": [long_answer],
            "reference_answer": ["A function that calls itself with a termination condition."],
        }
    )
    feats = build_features(long_df, fit_semantic=True)
    preds = trained_pipeline.predict(feats)
    assert len(preds) == 1
    assert preds[0] in {0, 1, 2}


def test_edge_case_html_and_script_injection(trained_pipeline):
    """Test resilience against raw HTML tags, embedded scripts, and formatting."""
    injection_df = pd.DataFrame(
        {
            "question": ["What is HTML? <script>alert('test')</script>"],
            "student_answer": [
                "<b>HyperText</b> Markup Language with <img src=x onerror=alert(1)> and <div>tags</div>."
            ],
            "reference_answer": ["Standard markup language for documents designed for web browsers."],
        }
    )
    feats = build_features(injection_df, fit_semantic=True)
    preds = trained_pipeline.predict(feats)
    assert len(preds) == 1
    assert preds[0] in {0, 1, 2}


def test_edge_case_math_latex_and_unicode_emojis(trained_pipeline):
    """Test handling of LaTeX math equations and Unicode characters."""
    unicode_df = pd.DataFrame(
        {
            "question": ["State Euler's formula: $e^{i\\pi} + 1 = 0$"],
            "student_answer": ["$e^{i\\pi} = -1$ 🚀✨ Amazing result in mathematics! 计算机科学"],
            "reference_answer": ["Euler's identity establishes relation between exponential and trig functions."],
        }
    )
    feats = build_features(unicode_df, fit_semantic=True)
    preds = trained_pipeline.predict(feats)
    assert len(preds) == 1
    assert preds[0] in {0, 1, 2}
