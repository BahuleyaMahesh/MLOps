"""Question-grouped splitting utilities and GroupKFold cross-validation for EngSAF."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "processed" / "engsaf" / "train.csv"


def get_group_kfold_splits(
    data: pd.DataFrame,
    group_col: str = "question",
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate GroupKFold train/validation index pairs with zero group overlap.

    Guarantees that no question in the validation fold appeared in the training fold.
    """
    if group_col not in data.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame.")

    groups = data[group_col].astype(str)
    n_distinct = groups.nunique()
    if n_distinct < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} group folds with only {n_distinct} distinct questions."
        )

    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(data, groups=groups)):
        train_groups = set(groups.iloc[train_idx])
        val_groups = set(groups.iloc[val_idx])
        overlap = train_groups & val_groups
        if overlap:
            raise RuntimeError(
                f"Fold {fold} group leakage detected! Overlapping groups: {len(overlap)}"
            )
        splits.append((train_idx, val_idx))
    return splits


def create_question_grouped_holdout(
    data: pd.DataFrame,
    group_col: str = "question",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition a dataset into train and holdout splits grouped by question."""
    if group_col not in data.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, holdout_idx = next(splitter.split(data, groups=data[group_col]))
    train_frame = data.iloc[train_idx].copy().reset_index(drop=True)
    holdout_frame = data.iloc[holdout_idx].copy().reset_index(drop=True)

    overlap = set(train_frame[group_col]) & set(holdout_frame[group_col])
    if overlap:
        raise RuntimeError(f"Group leakage detected! Shared questions: {len(overlap)}")

    return train_frame, holdout_frame


def evaluate_group_splits(
    data: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    group_col: str = "question",
) -> list[dict]:
    """Inspect summary statistics for each generated fold."""
    summaries = []
    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        train_q = data.iloc[train_idx][group_col].nunique()
        val_q = data.iloc[val_idx][group_col].nunique()
        summaries.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "val_rows": int(len(val_idx)),
                "train_questions": int(train_q),
                "val_questions": int(val_q),
                "question_leakage": len(
                    set(data.iloc[train_idx][group_col])
                    & set(data.iloc[val_idx][group_col])
                ),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify or export GroupKFold cross-validation splits on EngSAF."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to CSV containing questions.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold cross-validation splits.",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Data file not found at {args.data_path}")

    frame = pd.read_csv(args.data_path)
    splits = get_group_kfold_splits(frame, group_col="question", n_splits=args.n_splits)
    summaries = evaluate_group_splits(frame, splits, group_col="question")

    print(f"Generated {len(splits)}-fold GroupKFold successfully on {args.data_path.name}:")
    for s in summaries:
        print(
            f"  Fold {s['fold']}: Train={s['train_rows']} rows ({s['train_questions']} Qs), "
            f"Val={s['val_rows']} rows ({s['val_questions']} Qs), Leakage={s['question_leakage']}"
        )


if __name__ == "__main__":
    main()
