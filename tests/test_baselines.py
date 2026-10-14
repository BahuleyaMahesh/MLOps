import numpy as np
import pandas as pd
import pytest

from src.train_baseline import (
    FrankHallOrdinalClassifier,
    make_pipeline,
    score_from_similarity,
    tune_similarity_thresholds,
)


def test_similarity_thresholds_map_to_ordered_scores():
    scores = score_from_similarity(np.array([0.1, 0.4, 0.8]), 0.3, 0.7)
    assert scores.tolist() == [0, 1, 2]


def test_threshold_tuning_uses_validation_labels():
    similarities = np.array([0.05, 0.15, 0.45, 0.55, 0.85, 0.95])
    labels = pd.Series([0, 0, 1, 1, 2, 2])

    low, high, predictions = tune_similarity_thresholds(similarities, labels)

    assert low <= high
    assert predictions.tolist() == labels.tolist()


def test_frank_hall_ordinal_classifier():
    X = np.array([[0.1], [0.2], [1.5], [1.8], [4.5], [5.0]])
    y = np.array([0, 0, 1, 1, 2, 2])

    clf = FrankHallOrdinalClassifier(C=1.0, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    assert probs.shape == (6, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert set(preds).issubset({0, 1, 2})


def test_make_pipeline_ablations():
    params = {
        "model": {
            "tfidf_max_features": 100,
            "tfidf_ngram_max": 2,
            "logistic_regression_c": 1.0,
            "random_state": 42,
        }
    }

    pipe_all = make_pipeline("logistic_regression", params, ablation="all")
    pipe_no_q = make_pipeline("logistic_regression", params, ablation="without_question")
    pipe_len = make_pipeline("logistic_regression", params, ablation="length_only")
    pipe_gb = make_pipeline("gradient_boosting", params, ablation="all")
    pipe_ord = make_pipeline("ordinal_regression", params, ablation="all")

    assert pipe_all is not None
    assert pipe_no_q is not None
    assert pipe_len is not None
    assert pipe_gb is not None
    assert pipe_ord is not None
