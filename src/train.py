"""Train candidate answer-scoring baseline models (M2-A through M2-E).

DVC Stage: train (ingest -> clean -> features -> train -> evaluate)
Trains models on extracted features and logs parameters/artifacts to MLflow.
Saves serialized model pipelines to `models/`.
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml

from src.features import NUMERIC_COLUMNS
from src.train_baseline import (
    FrankHallOrdinalClassifier,
    cosine_similarity_scores,
    make_pipeline,
    score_from_similarity,
    tune_similarity_thresholds,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "engsaf"
DEFAULT_FEAT_DIR = ROOT / "data" / "features" / "engsaf"
DEFAULT_MODELS_DIR = ROOT / "models"


def train_all_models(data_dir: Path, feat_dir: Path, models_dir: Path, params_path: Path) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)

    with params_path.open(encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # Load raw text and processed features
    train_raw = pd.read_csv(data_dir / "train.csv").fillna("")
    val_raw = pd.read_csv(data_dir / "validation.csv").fillna("")

    train_feat = pd.read_csv(feat_dir / "train_features.csv").fillna("")
    val_feat = pd.read_csv(feat_dir / "validation_features.csv").fillna("")

    y_train = train_raw["score"].astype(int).values
    y_val = val_raw["score"].astype(int).values

    # Setup MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("answer-scorer-baselines")

    print("[train] Starting training for M2-A, M2-B, M2-C, M2-D, M2-E...")

    # 1. Cosine similarity thresholds (M2-B)
    sim_val = cosine_similarity_scores(train_raw, val_raw)
    low_thresh, high_thresh, _ = tune_similarity_thresholds(sim_val, y_val)
    cosine_model = {
        "model_id": "M2-B",
        "name": "Cosine Similarity Thresholds",
        "threshold_0_below": float(low_thresh),
        "threshold_2_at_or_above": float(high_thresh),
    }
    with (models_dir / "m2_b_cosine_thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(cosine_model, f, indent=2)
    print(f"[train] Saved M2-B thresholds to {models_dir / 'm2_b_cosine_thresholds.json'}")

    # 2. Main baseline models (M2-A, M2-C, M2-D, M2-E)
    model_configs = [
        ("m2_a_majority", "majority", "M2-A Majority Class Floor"),
        ("m2_c_logistic_regression", "logistic_regression", "M2-C Logistic Regression Main Baseline"),
        ("m2_d_hist_gb", "gradient_boosting", "M2-D HistGradientBoosting Non-linear"),
        ("m2_e_frank_hall", "ordinal_regression", "M2-E Frank-Hall Ordinal Regression"),
    ]

    saved_models = {"m2_b_cosine_thresholds": cosine_model}

    for file_key, model_type, desc in model_configs:
        print(f"[train] Fitting {desc}...")
        pipeline = make_pipeline(model_type, params)
        pipeline.fit(train_feat, y_train)

        out_path = models_dir / f"{file_key}.joblib"
        joblib.dump(pipeline, out_path)
        saved_models[file_key] = {
            "path": str(out_path),
            "model_type": model_type,
            "description": desc,
        }
        print(f"[train] Saved {desc} to {out_path}")

    # Save training metadata
    train_metadata = {
        "dataset_name": params["dataset"]["name"],
        "dataset_revision": params["dataset"]["revision"],
        "random_state": params["model"]["random_state"],
        "trained_models": saved_models,
    }
    with (models_dir / "train_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(train_metadata, f, indent=2)

    print(f"[train] All models trained and saved to {models_dir}")
    return saved_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train answer-scoring baseline models.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--feat-dir", type=Path, default=DEFAULT_FEAT_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--params", type=Path, default=ROOT / "params.yaml")
    args = parser.parse_args()

    train_all_models(args.data_dir, args.feat_dir, args.models_dir, args.params)


if __name__ == "__main__":
    main()
