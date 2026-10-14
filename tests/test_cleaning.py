import pandas as pd
import pytest

from src.clean_data import clean_dataset, clean_split, clean_text


def test_clean_text_unescapes_html_entities():
    raw = "Under which circumstances is a network (&gt;1 layer) desirable &amp; valid?"
    expected = "Under which circumstances is a network (>1 layer) desirable & valid?"
    assert clean_text(raw) == expected


def test_clean_text_normalizes_whitespace_and_newlines():
    raw = "   First   line    with  spaces.   \r\n\r\n\n\n\n   Second line.   "
    expected = "First line with spaces.\n\nSecond line."
    assert clean_text(raw) == expected


def test_clean_text_handles_empty_and_none():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("   \n\n\t   ") == ""


def test_clean_text_cleans_markdown_bold_and_italics():
    raw = " **TRUE**, context switch allows **syscalls** and __ready__ processes."
    expected = "TRUE, context switch allows syscalls and ready processes."
    assert clean_text(raw) == expected


def test_clean_text_preserves_code_syntax_and_math_operators():
    code_text = "vector<Shape*> shapes;\nshapes.push_back(new Rectangle());\nRectangle::draw();"
    assert clean_text(code_text) == code_text

    math_text = "for class-1 -> g(x) > 0 and class-2 -> g(x) < 0 with S <= 0.95"
    assert clean_text(math_text) == math_text


def test_clean_split_validates_scores_and_missing_columns():
    invalid_cols_df = pd.DataFrame(
        {
            "question": ["What is AI?"],
            "student_answer": ["Artificial Intelligence"],
        }
    )
    with pytest.raises(ValueError, match="missing required columns"):
        clean_split(invalid_cols_df, {})

    invalid_score_df = pd.DataFrame(
        {
            "question": ["Q1"],
            "student_answer": ["A1"],
            "reference_answer": ["R1"],
            "mark_scheme": ["M1"],
            "score": [5],  # Invalid score label
            "rationale": ["Rat1"],
        }
    )
    with pytest.raises(ValueError, match="Invalid scores"):
        clean_split(invalid_score_df, {})


def test_clean_split_processes_data_and_counts_modifications():
    df = pd.DataFrame(
        {
            "question": ["  Q1 with space  ", "Network (&gt;1)"],
            "student_answer": [" **TRUE** answer ", "\n\n"],
            "reference_answer": [" Ref answer ", "Ref2"],
            "mark_scheme": ["MS1", "MS2"],
            "score": [2, 0],
            "rationale": [" Good job ", "Empty"],
        }
    )
    cleaned_df, stats = clean_split(df, {})

    assert stats["total_rows"] == 2
    assert stats["empty_student_answers_before"] == 1
    assert stats["empty_student_answers_after"] == 1
    assert stats["html_entities_resolved"] == 1
    assert stats["markdown_markers_cleaned"] == 1
    assert cleaned_df.loc[0, "student_answer"] == "TRUE answer"
    assert cleaned_df.loc[1, "student_answer"] == ""
    assert cleaned_df.loc[1, "question"] == "Network (>1)"
    assert cleaned_df["score"].tolist() == [2, 0]


def test_clean_dataset_end_to_end(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()

    sample_df = pd.DataFrame(
        {
            "question": [" What is Python? "],
            "student_answer": ["  A programming language  "],
            "reference_answer": ["Python is a language"],
            "mark_scheme": ["Full marks"],
            "score": [2],
            "rationale": ["Correct"],
        }
    )
    sample_df.to_csv(raw_dir / "train.csv", index=False)

    report = clean_dataset(raw_dir, processed_dir, {})
    assert "train" in report["splits"]
    assert (processed_dir / "train.csv").exists()

    result_df = pd.read_csv(processed_dir / "train.csv")
    assert result_df.loc[0, "question"] == "What is Python?"
    assert result_df.loc[0, "student_answer"] == "A programming language"
