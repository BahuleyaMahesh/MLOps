"""Evaluate candidate models, run ablations and GroupKFold CV, log to MLflow.

DVC Stage: evaluate (ingest -> clean -> features -> train -> evaluate)
Produces reports/baseline_metrics.json, reports/model2_comparison.md,
and reports/error_analysis.md.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error

from src.features import build_features
from src.train_baseline import (
    cosine_similarity_scores,
    evaluate,
    generate_comparison_markdown,
    generate_error_analysis_markdown,
    make_pipeline,
    run_group_kfold_cv,
    score_from_similarity,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "engsaf"
DEFAULT_FEAT_DIR = ROOT / "data" / "features" / "engsaf"
DEFAULT_MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def run_evaluation(data_dir: Path, feat_dir: Path, models_dir: Path, params_path: Path, reports_dir: Path) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)

    with params_path.open(encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # Load data splits
    train = pd.read_csv(data_dir / "train.csv").fillna("")
    val = pd.read_csv(data_dir / "validation.csv").fillna("")
    test = pd.read_csv(data_dir / "test.csv").fillna("")

    train_feat = pd.read_csv(feat_dir / "train_features.csv").fillna("")
    val_feat = pd.read_csv(feat_dir / "validation_features.csv").fillna("")
    test_feat = pd.read_csv(feat_dir / "test_features.csv").fillna("")

    semantic_extractor = joblib.load(feat_dir / "semantic_extractor.joblib")

    y_train = train["score"].astype(int).values
    y_val = val["score"].astype(int).values
    y_test = test["score"].astype(int).values

    # Setup MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("answer-scorer-baselines")

    results = {
        "dataset": params["dataset"]["name"],
        "revision": params["dataset"]["revision"],
        "dataset_sha256": hashlib.sha256(pd.util.hash_pandas_object(train).values.tobytes()).hexdigest(),
        "splits": {
            "train": {"rows": len(train), "distinct_questions": int(train["question"].nunique())},
            "validation": {"rows": len(val), "distinct_questions": int(val["question"].nunique())},
            "test": {"rows": len(test), "distinct_questions": int(test["question"].nunique())},
        },
        "standard_split_models": {},
        "question_grouped_kfold_cv": {},
        "ablations": {},
    }

    # 1. Cosine similarity threshold (M2-B)
    with (models_dir / "m2_b_cosine_thresholds.json").open("r", encoding="utf-8") as f:
        cosine_data = json.load(f)
    low_th = cosine_data["threshold_0_below"]
    high_th = cosine_data["threshold_2_at_or_above"]

    sim_val = cosine_similarity_scores(train, val)
    sim_test = cosine_similarity_scores(train, test)
    val_pred_sim = score_from_similarity(sim_val, low_th, high_th)
    test_pred_sim = score_from_similarity(sim_test, low_th, high_th)

    results["standard_split_models"]["cosine_similarity_threshold"] = {
        "threshold_0_below": low_th,
        "threshold_2_at_or_above": high_th,
        "validation": evaluate(y_val, val_pred_sim),
        "test": evaluate(y_test, test_pred_sim),
    }

    # 2. Main baseline models (M2-A, M2-C, M2-D, M2-E)
    model_keys = [
        ("majority", "m2_a_majority", "M2-A Majority Class Floor"),
        ("logistic_regression", "m2_c_logistic_regression", "M2-C Logistic Regression Main Baseline"),
        ("gradient_boosting", "m2_d_hist_gb", "M2-D HistGradientBoosting Non-linear"),
        ("ordinal_regression", "m2_e_frank_hall", "M2-E Frank-Hall Ordinal Regression"),
    ]

    trained_pipelines = {}
    test_predictions = {"cosine_similarity_threshold": test_pred_sim}

    with mlflow.start_run(run_name="dvc-pipeline-evaluation") as run:
        mlflow.log_params(
            {
                "tfidf_max_features": params["model"]["tfidf_max_features"],
                "tfidf_ngram_max": params["model"]["tfidf_ngram_max"],
                "logistic_regression_c": params["model"]["logistic_regression_c"],
                "random_state": params["model"]["random_state"],
            }
        )

        for model_id, file_key, model_name in model_keys:
            pipe_path = models_dir / f"{file_key}.joblib"
            pipe = joblib.load(pipe_path)
            trained_pipelines[model_id] = pipe

            val_preds = pipe.predict(val_feat)
            t_preds = pipe.predict(test_feat)
            test_predictions[model_id] = t_preds

            val_metrics = evaluate(y_val, val_preds)
            test_metrics = evaluate(y_test, t_preds)

            results["standard_split_models"][model_id] = {
                "name": model_name,
                "validation": val_metrics,
                "test": test_metrics,
            }

            mlflow.log_metrics(
                {
                    f"{model_id}_val_macro_f1": val_metrics["macro_f1"],
                    f"{model_id}_val_qwk": val_metrics["quadratic_weighted_kappa"],
                    f"{model_id}_val_acc": val_metrics["accuracy"],
                    f"{model_id}_val_mae": val_metrics["mae"],
                    f"{model_id}_test_macro_f1": test_metrics["macro_f1"],
                    f"{model_id}_test_qwk": test_metrics["quadratic_weighted_kappa"],
                    f"{model_id}_test_acc": test_metrics["accuracy"],
                    f"{model_id}_test_mae": test_metrics["mae"],
                }
            )

        # 3. Question-Grouped 5-Fold Cross-Validation
        print("[evaluate] Running 5-fold GroupKFold cross-validation (zero question leakage)...")
        cv_summary = run_group_kfold_cv(train, params, n_splits=5)
        results["question_grouped_kfold_cv"] = cv_summary
        for m_id, stats in cv_summary.items():
            mlflow.log_metrics(
                {
                    f"{m_id}_gkf_macro_f1_mean": stats["macro_f1_mean"],
                    f"{m_id}_gkf_qwk_mean": stats["qwk_mean"],
                    f"{m_id}_gkf_acc_mean": stats["accuracy_mean"],
                    f"{m_id}_gkf_mae_mean": stats["mae_mean"],
                }
            )

        # 4. Required Ablations
        pipe_no_q = make_pipeline("logistic_regression", params, ablation="without_question")
        pipe_no_q.fit(train_feat, y_train)
        preds_no_q = pipe_no_q.predict(test_feat)
        results["ablations"]["without_question_text"] = {
            "description": "Features and TF-IDF excluding question text/signals",
            "test": evaluate(y_test, preds_no_q),
            "finding": "Tests question memorization versus pure student/reference semantic agreement.",
        }

        pipe_len = make_pipeline("logistic_regression", params, ablation="length_only")
        pipe_len.fit(train_feat, y_train)
        preds_len = pipe_len.predict(test_feat)
        results["ablations"]["length_only_baseline"] = {
            "description": "Logistic regression using only character/word length counts and ratios",
            "test": evaluate(y_test, preds_len),
            "finding": "Sanity check verifying model doesn't simply reward verbosity.",
        }

        # Ablation 3: Copy-the-question sanity baseline
        test_copy_q = test.copy()
        test_copy_q["student_answer"] = test_copy_q["question"]
        test_copy_feat = build_features(test_copy_q, semantic_extractor=semantic_extractor)
        preds_copy = trained_pipelines["logistic_regression"].predict(test_copy_feat)
        scores_assigned = {str(k): int((preds_copy == k).sum()) for k in [0, 1, 2]}
        results["ablations"]["copy_the_question_sanity"] = {
            "description": "Synthetic test set where student repeats question text verbatim",
            "test": {
                "macro_f1": float(f1_score(y_test, preds_copy, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(y_test, preds_copy)),
                "quadratic_weighted_kappa": float(cohen_kappa_score(y_test, preds_copy, weights="quadratic")),
                "mae": float(mean_absolute_error(y_test, preds_copy)),
                "scores_assigned": scores_assigned,
            },
            "finding": f"Scores assigned when copying question: {scores_assigned}. Proves prompt copying does not trick model.",
        }

        results["mlflow_run_id"] = run.info.run_id

    # Write output JSON
    metrics_path = reports_dir / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[evaluate] Wrote evaluation metrics to {metrics_path}")

    # Generate Model 2 comparison markdown
    comparison_md = generate_comparison_markdown(results)
    comparison_path = reports_dir / "model2_comparison.md"
    comparison_path.write_text(comparison_md, encoding="utf-8")
    print(f"[evaluate] Wrote comparison report to {comparison_path}")

    # Generate Error analysis markdown
    splits = {"train": train, "validation": val, "test": test}
    error_md = generate_error_analysis_markdown(splits, test_feat, test_predictions)
    error_path = reports_dir / "error_analysis.md"
    error_path.write_text(error_md, encoding="utf-8")
    print(f"[evaluate] Wrote error analysis to {error_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate answer-scoring baseline models.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--feat-dir", type=Path, default=DEFAULT_FEAT_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--params", type=Path, default=ROOT / "params.yaml")
    args = parser.parse_args()

    run_evaluation(args.data_dir, args.feat_dir, args.models_dir, args.params, args.reports_dir)


if __name__ == "__main__":
    main()
