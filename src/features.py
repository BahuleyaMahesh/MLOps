"""Feature engineering for answer scoring.

Extracts lexical overlap, text length statistics, dense semantic embeddings,
and interaction features (|u - v|, u ⊙ v) while strictly guaranteeing no
evaluative leakage ('rationale' or 'mark_scheme') enters the feature pipeline.
"""

import re
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


TEXT_INPUTS = ("question", "student_answer", "reference_answer")
FORBIDDEN_INPUTS = ("mark_scheme", "rationale")

LEXICAL_COLUMNS = [
    "student_reference_jaccard",
    "student_reference_overlap",
    "reference_student_overlap",
    "student_reference_f1",
    "student_question_overlap",
    "student_exact_match",
]

LENGTH_COLUMNS = [
    "student_word_len",
    "reference_word_len",
    "question_word_len",
    "student_char_len",
    "reference_char_len",
    "question_char_len",
    "answer_reference_length_ratio",
    "char_length_ratio",
    "char_length_diff",
]

N_EMBEDDING_COMPONENTS = 16
EMBEDDING_INTERACTION_COLUMNS = (
    ["embedding_cosine_similarity", "embedding_euclidean_distance", "student_question_emb_sim"]
    + [f"emb_diff_{i}" for i in range(N_EMBEDDING_COMPONENTS)]
    + [f"emb_prod_{i}" for i in range(N_EMBEDDING_COMPONENTS)]
)

NUMERIC_COLUMNS = LEXICAL_COLUMNS + LENGTH_COLUMNS + EMBEDDING_INTERACTION_COLUMNS
FEATURE_COLUMNS = tuple(list(TEXT_INPUTS) + NUMERIC_COLUMNS)

TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def _tokens(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.casefold()))


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


class DenseSemanticExtractor(BaseEstimator, TransformerMixin):
    """Extracts dense semantic representations and interaction features (cosine, |u-v|, u⊙v)."""

    def __init__(self, n_components: int = N_EMBEDDING_COMPONENTS, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        self.is_fitted = False
        self.is_empty_vocab = False

    def fit(self, X: pd.DataFrame, y=None):
        corpus = pd.concat(
            [X["student_answer"].fillna(""), X["reference_answer"].fillna(""), X["question"].fillna("")],
            ignore_index=True,
        ).astype(str)

        has_words = any(TOKEN_PATTERN.search(doc) for doc in corpus)
        if not has_words:
            self.vectorizer.fit(["dummy token for empty corpus"])
            self.svd = TruncatedSVD(n_components=1, random_state=self.random_state)
            self.svd.fit(self.vectorizer.transform(["dummy token for empty corpus"]))
            self.is_fitted = True
            self.is_empty_vocab = True
            return self

        self.is_empty_vocab = False
        tfidf_mat = self.vectorizer.fit_transform(corpus)
        n_comps = min(self.n_components, max(1, tfidf_mat.shape[1] - 1, tfidf_mat.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=n_comps, random_state=self.random_state)
        self.svd.fit(tfidf_mat)
        self.is_fitted = True
        return self

    def _embed(self, series: pd.Series) -> np.ndarray:
        if getattr(self, "is_empty_vocab", False):
            return np.zeros((len(series), self.n_components))

        mat = self.vectorizer.transform(series.fillna("").astype(str))
        if mat.nnz == 0:
            return np.zeros((len(series), self.n_components))

        emb = self.svd.transform(mat)
        if emb.shape[1] < self.n_components:
            padding = np.zeros((emb.shape[0], self.n_components - emb.shape[1]))
            emb = np.hstack([emb, padding])
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("DenseSemanticExtractor must be fitted before calling transform.")

        u = self._embed(X["student_answer"])
        v = self._embed(X["reference_answer"])
        q = self._embed(X["question"])

        cos_sim = np.sum(u * v, axis=1)
        euclid_dist = np.linalg.norm(u - v, axis=1)
        q_sim = np.sum(u * q, axis=1)

        abs_diff = np.abs(u - v)
        elem_prod = u * v

        features = pd.DataFrame(index=X.index)
        features["embedding_cosine_similarity"] = cos_sim
        features["embedding_euclidean_distance"] = euclid_dist
        features["student_question_emb_sim"] = q_sim

        for i in range(self.n_components):
            features[f"emb_diff_{i}"] = abs_diff[:, i]
        for i in range(self.n_components):
            features[f"emb_prod_{i}"] = elem_prod[:, i]

        return features


def extract_lexical_and_length_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract lexical overlap and length statistics."""
    records = []
    for row in frame.itertuples(index=False):
        q = str(getattr(row, "question", "") or "")
        s = str(getattr(row, "student_answer", "") or "")
        r = str(getattr(row, "reference_answer", "") or "")

        s_tokens = _tokens(s)
        r_tokens = _tokens(r)
        q_tokens = _tokens(q)

        intersection = s_tokens & r_tokens
        union = s_tokens | r_tokens
        jaccard = _ratio(len(intersection), len(union))
        s_overlap = _ratio(len(intersection), len(s_tokens))
        r_overlap = _ratio(len(intersection), len(r_tokens))
        denom = s_overlap + r_overlap
        f1 = (2.0 * s_overlap * r_overlap / denom) if denom > 0 else 0.0
        q_overlap = _ratio(len(s_tokens & q_tokens), len(s_tokens))
        exact_match = 1.0 if s.strip().casefold() == r.strip().casefold() and s.strip() else 0.0

        s_words = len(s.split())
        r_words = len(r.split())
        q_words = len(q.split())
        s_chars = len(s)
        r_chars = len(r)
        q_chars = len(q)

        records.append(
            {
                "student_reference_jaccard": jaccard,
                "student_reference_overlap": s_overlap,
                "reference_student_overlap": r_overlap,
                "student_reference_f1": f1,
                "student_question_overlap": q_overlap,
                "student_exact_match": exact_match,
                "student_word_len": float(s_words),
                "reference_word_len": float(r_words),
                "question_word_len": float(q_words),
                "student_char_len": float(s_chars),
                "reference_char_len": float(r_chars),
                "question_char_len": float(q_chars),
                "answer_reference_length_ratio": _ratio(s_words, r_words),
                "char_length_ratio": _ratio(s_chars, r_chars),
                "char_length_diff": float(abs(s_chars - r_chars)),
            }
        )
    return pd.DataFrame.from_records(records, index=frame.index)


def build_features(
    frame: pd.DataFrame,
    semantic_extractor: Optional[DenseSemanticExtractor] = None,
    fit_semantic: bool = False,
    return_extractor: bool = False,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, DenseSemanticExtractor]]:
    """Build full feature matrix and ensure no forbidden columns leak into features."""
    for forbidden in FORBIDDEN_INPUTS:
        if forbidden in frame.columns:
            frame = frame.drop(columns=[forbidden])

    missing = sorted(set(TEXT_INPUTS) - set(frame.columns))
    if missing:
        raise ValueError(f"Required text columns are missing: {missing}")

    text_df = frame.loc[:, list(TEXT_INPUTS)].fillna("").astype(str).copy()
    lex_len_df = extract_lexical_and_length_features(text_df)

    if semantic_extractor is None:
        semantic_extractor = DenseSemanticExtractor()
        semantic_extractor.fit(text_df)
    elif fit_semantic:
        semantic_extractor.fit(text_df)

    sem_df = semantic_extractor.transform(text_df)
    combined = pd.concat([text_df, lex_len_df, sem_df], axis=1)

    for forbidden in FORBIDDEN_INPUTS:
        if forbidden in combined.columns:
            raise RuntimeError(f"Critical security violation: {forbidden} leaked into features!")

    if return_extractor:
        return combined, semantic_extractor
    return combined
