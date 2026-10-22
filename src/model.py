"""Similarity scoring helpers for the two assignment baselines."""
from __future__ import annotations

from typing import Sequence
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def make_tfidf_vectorizer() -> TfidfVectorizer:
    """Create the project TF-IDF setup (fit it on training text only)."""
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")


def paired_tfidf_cosine(vectorizer: TfidfVectorizer, headlines: Sequence[str], articles: Sequence[str]) -> np.ndarray:
    """Return one headline/article cosine score for each aligned pair."""
    if len(headlines) != len(articles):
        raise ValueError("headlines and articles must have equal lengths")
    h = vectorizer.transform(headlines)
    a = vectorizer.transform(articles)
    return cosine_similarity(h, a, dense_output=True).diagonal()


def paired_embedding_cosine(model, headlines: Sequence[str], articles: Sequence[str], batch_size: int = 32) -> np.ndarray:
    """Encode aligned pairs and return cosine scores; vectors are normalized."""
    if len(headlines) != len(articles):
        raise ValueError("headlines and articles must have equal lengths")
    h = model.encode(list(headlines), batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    a = model.encode(list(articles), batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return np.sum(h * a, axis=1)
