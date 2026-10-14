"""Download and persist the EngSAF splits for local DVC versioning."""

from pathlib import Path

import yaml
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"
OUTPUT_DIR = ROOT / "data" / "raw" / "engsaf"


def main() -> None:
    with PARAMS_PATH.open(encoding="utf-8") as file:
        params = yaml.safe_load(file)

    dataset = load_dataset(
        params["dataset"]["name"],
        revision=params["dataset"]["revision"],
    )
    expected_splits = {"train", "validation", "test"}
    missing_splits = expected_splits - set(dataset.keys())
    if missing_splits:
        raise ValueError(f"Acquired dataset is missing required splits: {sorted(missing_splits)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split in dataset.items():
        split.to_csv(str(OUTPUT_DIR / f"{split_name}.csv"))

    print(
        "Saved local EngSAF splits:",
        {name: len(split) for name, split in dataset.items()},
    )


if __name__ == "__main__":
    main()
