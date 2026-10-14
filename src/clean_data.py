"""Data cleaning and text preprocessing pipeline stage for EngSAF dataset."""

import argparse
import html
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "data" / "raw" / "engsaf"
PROCESSED_DATA_DIR = ROOT / "data" / "processed" / "engsaf"
REPORT_PATH = ROOT / "reports" / "cleaning_report.json"
PARAMS_PATH = ROOT / "params.yaml"

REQUIRED_COLUMNS = [
    "question",
    "student_answer",
    "reference_answer",
    "mark_scheme",
    "score",
    "rationale",
]

MARKDOWN_BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")
MARKDOWN_UNDERSCORE_PATTERN = re.compile(r"__(.+?)__")
MULTIPLE_SPACES_PATTERN = re.compile(r"[ \t]+")
EXCESSIVE_NEWLINES_PATTERN = re.compile(r"\n{3,}")
HTML_ENTITY_PATTERN = re.compile(r"&[a-zA-Z0-9#]+;")


def clean_text(
    text: str | None,
    unescape_html: bool = True,
    normalize_unicode: bool = True,
    normalize_whitespace: bool = True,
    clean_markdown: bool = True,
) -> str:
    """Preprocess and clean raw text while preserving domain syntax (math, code)."""
    if text is None or pd.isna(text):
        return ""

    cleaned = str(text)

    # 1. HTML entity unescaping (e.g. &gt; -> >)
    if unescape_html and "&" in cleaned:
        cleaned = html.unescape(cleaned)

    # 2. Unicode normalization (NFKC normalizes full-width characters, ligature forms, etc.)
    if normalize_unicode:
        cleaned = unicodedata.normalize("NFKC", cleaned)

    # 3. Clean markdown formatting syntax (e.g. **word** -> word, __word__ -> word)
    # Carefully preserve single asterisks for pointers (Shape*) or math operations
    if clean_markdown:
        cleaned = MARKDOWN_BOLD_PATTERN.sub(r"\1", cleaned)
        cleaned = MARKDOWN_UNDERSCORE_PATTERN.sub(r"\1", cleaned)

    # 4. Whitespace and newline standardization
    if normalize_whitespace:
        # Standardize carriage returns to standard line breaks
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple horizontal whitespace characters per line and trim line margins
        lines = [
            MULTIPLE_SPACES_PATTERN.sub(" ", line).strip()
            for line in cleaned.split("\n")
        ]
        # Rejoin and collapse excessive consecutive blank lines (max 2 consecutive newlines)
        cleaned = "\n".join(lines)
        cleaned = EXCESSIVE_NEWLINES_PATTERN.sub("\n\n", cleaned)
        cleaned = cleaned.strip()

    return cleaned


def clean_split(
    frame: pd.DataFrame,
    options: dict,
) -> tuple[pd.DataFrame, dict]:
    """Clean all text columns, enforce valid labels, and produce audit stats."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # Validate score values
    if frame["score"].isna().any():
        raise ValueError("Missing score values found in dataset.")

    valid_scores = {0, 1, 2}
    scores_set = set(frame["score"].unique())
    if not scores_set.issubset(valid_scores):
        raise ValueError(f"Invalid scores found: {scores_set - valid_scores}")

    cleaned_df = frame.copy()
    cleaned_df["score"] = cleaned_df["score"].astype(int)

    # Check for empty or missing student answers cleanly
    stats = {
        "total_rows": int(len(frame)),
        "empty_student_answers_before": int(
            frame["student_answer"].isna().sum()
            + frame["student_answer"].dropna().astype(str).str.strip().eq("").sum()
        ),
        "text_columns_modified": {},
        "html_entities_resolved": 0,
        "markdown_markers_cleaned": 0,
    }

    # Track transformations across text columns
    text_cols = ["question", "student_answer", "reference_answer", "rationale", "mark_scheme"]
    for col in text_cols:
        # Fill missing values with empty string rather than converting NaN to "nan"
        raw_series = frame[col].fillna("").astype(str)
        # Count HTML entities in raw
        stats["html_entities_resolved"] += int(
            raw_series.map(lambda s: len(HTML_ENTITY_PATTERN.findall(s))).sum()
        )
        if col == "student_answer":
            stats["markdown_markers_cleaned"] += int(
                raw_series.map(
                    lambda s: len(MARKDOWN_BOLD_PATTERN.findall(s))
                    + len(MARKDOWN_UNDERSCORE_PATTERN.findall(s))
                ).sum()
            )

        cleaned_series = raw_series.map(
            lambda s: clean_text(
                s,
                unescape_html=options.get("unescape_html", True),
                normalize_unicode=options.get("normalize_unicode", True),
                normalize_whitespace=options.get("normalize_whitespace", True),
                clean_markdown=options.get("clean_markdown_formatting", True),
            )
        )
        modified_count = int((raw_series != cleaned_series).sum())
        stats["text_columns_modified"][col] = modified_count
        cleaned_df[col] = cleaned_series

    stats["empty_student_answers_after"] = int(
        cleaned_df["student_answer"].eq("").sum()
    )
    return cleaned_df, stats


def clean_dataset(
    raw_dir: Path,
    output_dir: Path,
    options: dict,
) -> dict:
    """Read all raw splits, clean texts, save processed CSVs, and return summary report."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found at {raw_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source_directory": str(raw_dir),
        "target_directory": str(output_dir),
        "options": options,
        "splits": {},
    }

    split_files = sorted(raw_dir.glob("*.csv"))
    if not split_files:
        raise FileNotFoundError(f"No CSV split files found in {raw_dir}")

    for file_path in split_files:
        split_name = file_path.stem
        raw_df = pd.read_csv(file_path)
        cleaned_df, split_stats = clean_split(raw_df, options)

        target_file = output_dir / f"{split_name}.csv"
        cleaned_df.to_csv(target_file, index=False, encoding="utf-8")
        report["splits"][split_name] = split_stats

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and preprocess EngSAF dataset splits.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR, help="Path to raw data directory")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR, help="Path to processed output directory")
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH, help="Path to cleaning report JSON")
    parser.add_argument("--params-path", type=Path, default=PARAMS_PATH, help="Path to params.yaml")
    args = parser.parse_args()

    options = {
        "unescape_html": True,
        "normalize_unicode": True,
        "normalize_whitespace": True,
        "clean_markdown_formatting": True,
    }
    if args.params_path.exists():
        with args.params_path.open(encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
            if "preprocessing" in params:
                options.update(params["preprocessing"])

    report = clean_dataset(args.raw_dir, args.output_dir, options)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Data cleaning completed successfully. Output written to {args.output_dir}")
    print(f"Cleaning audit report written to {args.report_path}")


if __name__ == "__main__":
    main()
