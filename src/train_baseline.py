"""Train and evaluate simple answer-scoring baselines with MLflow tracking."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.features import FEATURE_COLUMNS, TEXT_INPUTS, build_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    ROOT / "data" / "processed" / "engsaf"
    if (ROOT / "data" / "processed" / "engsaf").exists()
    else ROOT / "data" / "raw" / "engsaf"
)
OUTPUT_PATH = ROOT / "reports" / "baseline_metrics.json"
NUMERIC_COLUMNS = [
    "student_reference_jaccard",
    "student_reference_overlap",
    "answer_reference_length_ratio",
    "student_question_overlap",
]


def read_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required dataset split not found: {path}")
    return pd.read_csv(path)


def validate_splits(splits: dict[str, pd.DataFrame]) -> None:
    for name, frame in splits.items():
        if not set(TEXT_INPUTS + ("score",)) <= set(frame.columns):
            raise ValueError(f"{name} split does not have required model columns.")
        if frame["score"].isna().any() or not frame["score"].isin([0, 1, 2]).all():
            raise ValueError(f"{name} split has missing or invalid score labels.")
        if any(column in FEATURE_COLUMNS for column in ("rationale", "mark_scheme")):
            raise ValueError("A leakage-prone column was added to model features.")


def make_model(params: dict) -> Pipeline:
    text_transformers = [
        (
            f"tfidf_{column}",
            TfidfVectorizer(
                max_features=params["model"]["tfidf_max_features"],
                ngram_range=(1, params["model"]["tfidf_ngram_max"]),
                min_df=2,
                sublinear_tf=True,
            ),
            column,
        )
        for column in TEXT_INPUTS
    ]
    preprocessor = ColumnTransformer(
        text_transformers
        + [
            (
                "engineered",
                StandardScaler(with_mean=False),
                NUMERIC_COLUMNS,
            )
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        C=params["model"]["logistic_regression_c"],
        class_weight="balanced",
        max_iter=1000,
        random_state=params["model"]["random_state"],
    )
    return Pipeline([("features", preprocessor), ("classifier", classifier)])


def evaluate(y_true, predictions) -> dict:
    return {
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, predictions, weights="quadratic")
        ),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1, 2],
            output_dict=True,
            zero_division=0,
        ),
    }


def fingerprint_splits(splits: dict[str, pd.DataFrame]) -> str:
    digest = hashlib.sha256()
    for name, frame in splits.items():
        digest.update(name.encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def run_grouped_holdout(
    train: pd.DataFrame,
    params: dict,
) -> dict:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=params["model"]["test_size"],
        random_state=params["model"]["random_state"],
    )
    train_indices, holdout_indices = next(
        splitter.split(train, groups=train["question"])
    )
    fit_frame = train.iloc[train_indices]
    holdout_frame = train.iloc[holdout_indices]
    model = make_model(params)
    model.fit(
        build_features(fit_frame),
        fit_frame["score"].astype(int),
    )
    predictions = model.predict(build_features(holdout_frame))
    return {
        "fit_rows": int(len(fit_frame)),
        "holdout_rows": int(len(holdout_frame)),
        "fit_questions": int(fit_frame["question"].nunique()),
        "holdout_questions": int(holdout_frame["question"].nunique()),
        **evaluate(holdout_frame["score"].astype(int), predictions),
    }


def cosine_similarity_scores(
    fit_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
) -> np.ndarray:
    """Score student/reference similarity with a vectorizer fit on training text."""
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    fit_text = pd.concat(
        [fit_frame["student_answer"], fit_frame["reference_answer"]],
        ignore_index=True,
    ).fillna("").astype(str)
    vectorizer.fit(fit_text)
    student_vectors = vectorizer.transform(
        target_frame["student_answer"].fillna("").astype(str)
    )
    reference_vectors = vectorizer.transform(
        target_frame["reference_answer"].fillna("").astype(str)
    )
    return np.asarray(student_vectors.multiply(reference_vectors).sum(axis=1)).ravel()


def score_from_similarity(
    similarities: np.ndarray,
    score_zero_below: float,
    score_two_at_or_above: float,
) -> np.ndarray:
    return np.where(
        similarities < score_zero_below,
        0,
        np.where(similarities < score_two_at_or_above, 1, 2),
    )


def tune_similarity_thresholds(
    similarities: np.ndarray,
    labels: pd.Series,
) -> tuple[float, float, np.ndarray]:
    candidates = np.unique(np.quantile(similarities, np.linspace(0, 1, 21)))
    best_macro_f1 = -1.0
    best_thresholds = (float(candidates[0]), float(candidates[-1]))
    best_predictions = score_from_similarity(
        similarities, *best_thresholds
    )
    for low_index, low in enumerate(candidates):
        for high in candidates[low_index:]:
            predictions = score_from_similarity(similarities, float(low), float(high))
            macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
            if macro_f1 > best_macro_f1:
                best_macro_f1 = float(macro_f1)
                best_thresholds = (float(low), float(high))
                best_predictions = predictions
    return best_thresholds[0], best_thresholds[1], best_predictions


def main() -> None:
    global DATA_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    with (ROOT / "params.yaml").open(encoding="utf-8") as file:
        params = yaml.safe_load(file)

    splits = {
        "train": read_split("train"),
        "validation": read_split("validation"),
        "test": read_split("test"),
    }
    validate_splits(splits)

    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    train_x = build_features(train)
    validation_x = build_features(validation)
    test_x = build_features(test)
    train_y = train["score"].astype(int)

    dataset_hash = fingerprint_splits(splits)
    validation_similarities = cosine_similarity_scores(train, validation)
    test_similarities = cosine_similarity_scores(train, test)
    low_threshold, high_threshold, validation_similarity_predictions = (
        tune_similarity_thresholds(
            validation_similarities,
            validation["score"].astype(int),
        )
    )
    test_similarity_predictions = score_from_similarity(
        test_similarities,
        low_threshold,
        high_threshold,
    )
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("engsaf-answer-scorer")

    results = {
        "dataset_sha256": dataset_hash,
        "target": "score",
        "feature_inputs": list(TEXT_INPUTS) + NUMERIC_COLUMNS,
        "excluded_columns": ["rationale", "mark_scheme"],
        "splits": {
            name: {
                "rows": int(len(frame)),
                "distinct_questions": int(frame["question"].nunique()),
            }
            for name, frame in splits.items()
        },
        "models": {},
    }
    results["models"]["cosine_similarity_threshold"] = {
        "threshold_score_zero_below": low_threshold,
        "threshold_score_two_at_or_above": high_threshold,
        "validation": evaluate(
            validation["score"].astype(int),
            validation_similarity_predictions,
        ),
        "test": evaluate(
            test["score"].astype(int),
            test_similarity_predictions,
        ),
    }
    model = make_model(params)
    baseline = DummyClassifier(strategy="most_frequent")
    train_constant = pd.DataFrame({"constant": [0] * len(train)})
    validation_constant = pd.DataFrame({"constant": [0] * len(validation)})
    test_constant = pd.DataFrame({"constant": [0] * len(test)})

    with mlflow.start_run(run_name="phase1-baselines") as run:
        mlflow.log_params(
            {
                "dataset": params["dataset"]["name"],
                "dataset_revision": params["dataset"]["revision"],
                "dataset_sha256": dataset_hash,
                "tfidf_max_features": params["model"]["tfidf_max_features"],
                "tfidf_ngram_max": params["model"]["tfidf_ngram_max"],
                "logistic_regression_c": params["model"]["logistic_regression_c"],
                "random_state": params["model"]["random_state"],
            }
        )
        mlflow.log_params(
            {
                "cosine_score_zero_below": low_threshold,
                "cosine_score_two_at_or_above": high_threshold,
            }
        )
        mlflow.log_metrics(
            {
                "cosine_validation_macro_f1": results["models"][
                    "cosine_similarity_threshold"
                ]["validation"]["macro_f1"],
                "cosine_validation_accuracy": results["models"][
                    "cosine_similarity_threshold"
                ]["validation"]["accuracy"],
                "cosine_validation_qwk": results["models"][
                    "cosine_similarity_threshold"
                ]["validation"]["quadratic_weighted_kappa"],
                "cosine_test_macro_f1": results["models"][
                    "cosine_similarity_threshold"
                ]["test"]["macro_f1"],
                "cosine_test_accuracy": results["models"][
                    "cosine_similarity_threshold"
                ]["test"]["accuracy"],
                "cosine_test_qwk": results["models"][
                    "cosine_similarity_threshold"
                ]["test"]["quadratic_weighted_kappa"],
            }
        )
        for name, estimator in (
            ("majority", baseline),
            ("tfidf_logistic_regression", model),
        ):
            fit_x = train_constant if name == "majority" else train_x
            estimator.fit(fit_x, train_y)
            validation_predictions = estimator.predict(
                validation_constant
                if name == "majority"
                else validation_x
            )
            test_predictions = estimator.predict(
                test_constant
                if name == "majority"
                else test_x
            )
            validation_metrics = evaluate(
                validation["score"].astype(int), validation_predictions
            )
            test_metrics = evaluate(test["score"].astype(int), test_predictions)
            results["models"][name] = {
                "validation": validation_metrics,
                "test": test_metrics,
            }
            mlflow.log_metrics(
                {
                    f"{name}_validation_macro_f1": validation_metrics["macro_f1"],
                    f"{name}_validation_accuracy": validation_metrics["accuracy"],
                    f"{name}_validation_qwk": validation_metrics[
                        "quadratic_weighted_kappa"
                    ],
                    f"{name}_test_macro_f1": test_metrics["macro_f1"],
                    f"{name}_test_accuracy": test_metrics["accuracy"],
                    f"{name}_test_qwk": test_metrics["quadratic_weighted_kappa"],
                }
            )

        results["models"]["grouped_question_holdout"] = run_grouped_holdout(
            train, params
        )
        grouped_metrics = results["models"]["grouped_question_holdout"]
        mlflow.log_metrics(
            {
                "grouped_holdout_macro_f1": grouped_metrics["macro_f1"],
                "grouped_holdout_accuracy": grouped_metrics["accuracy"],
                "grouped_holdout_qwk": grouped_metrics["quadratic_weighted_kappa"],
            }
        )
        results["mlflow_run_id"] = run.info.run_id

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote baseline metrics to {args.output}")


if __name__ == "__main__":
    main()
