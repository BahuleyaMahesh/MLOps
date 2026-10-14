import numpy as np
import pandas as pd

from src.train_baseline import score_from_similarity, tune_similarity_thresholds


def test_similarity_thresholds_map_to_ordered_scores():
    scores = score_from_similarity(np.array([0.1, 0.4, 0.8]), 0.3, 0.7)

    assert scores.tolist() == [0, 1, 2]


def test_threshold_tuning_uses_validation_labels():
    similarities = np.array([0.05, 0.15, 0.45, 0.55, 0.85, 0.95])
    labels = pd.Series([0, 0, 1, 1, 2, 2])

    low, high, predictions = tune_similarity_thresholds(similarities, labels)

    assert low <= high
    assert predictions.tolist() == labels.tolist()
