"""Extract and cache engineered features for EngSAF answer scoring.

DVC Stage: features (ingest -> clean -> features -> train -> evaluate)
Extracts lexical, length, and dense semantic interaction features while
strictly enforcing no-leakage boundaries (excluding rationale and mark_scheme).
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd
import yaml

from src.features import FEATURE_COLUMNS, FORBIDDEN_INPUTS, TEXT_INPUTS, build_features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "engsaf"
DEFAULT_OUT_DIR = ROOT / "data" / "features" / "engsaf"


def extract_all_features(data_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for name in ("train", "validation", "test"):
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing processed split: {path}")
        df = pd.read_csv(path)
        splits[name] = df

    # Strict leakage validation: confirm input does not leak target or forbidden text
    for name, df in splits.items():
        forbidden = set(FORBIDDEN_INPUTS) & set(df.columns)
        # Note: raw df might have mark_scheme or rationale, but feature extractor must strip them
        print(f"[{name}] Read {len(df)} rows from {data_dir / f'{name}.csv'}")

    # Build features: fit semantic extractor on train only
    train_feat, semantic_extractor = build_features(splits["train"], fit_semantic=True, return_extractor=True)
    val_feat = build_features(splits["validation"], semantic_extractor=semantic_extractor)
    test_feat = build_features(splits["test"], semantic_extractor=semantic_extractor)

    feature_dfs = {
        "train": train_feat,
        "validation": val_feat,
        "test": test_feat,
    }

    # Verify no forbidden inputs leaked into features
    for name, feat_df in feature_dfs.items():
        leaked = set(FORBIDDEN_INPUTS) & set(feat_df.columns)
        if leaked:
            raise ValueError(f"CRITICAL: Forbidden columns leaked into {name} features: {leaked}")

    # Save features as CSV and parquet (or CSV for standard DVC compatibility)
    paths = {}
    for name, feat_df in feature_dfs.items():
        out_csv = out_dir / f"{name}_features.csv"
        feat_df.to_csv(out_csv, index=False)
        paths[name] = str(out_csv)
        print(f"[{name}] Saved {len(feat_df)} rows x {len(feat_df.columns)} features to {out_csv}")

    # Save fitted semantic extractor
    extractor_path = out_dir / "semantic_extractor.joblib"
    joblib.dump(semantic_extractor, extractor_path)
    print(f"Saved fitted semantic extractor to {extractor_path}")

    return {
        "feature_columns": list(train_feat.columns),
        "shapes": {name: list(df.shape) for name, df in feature_dfs.items()},
        "extractor_path": str(extractor_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract engineered features for answer scoring.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to cleaned dataset directory")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Path to output feature directory")
    args = parser.parse_args()

    extract_all_features(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
