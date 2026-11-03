"""Recipe representations and a weighted multi-field retriever.

Each field (title, ingredients, steps) gets its own encoder that maps text to
L2-normalized row vectors, so a dot product is cosine similarity. The retriever
scores candidates with a weighted sum of per-field cosine similarities; the
weights are hyperparameters tuned on a dev split (see scripts/evaluate_retrieval.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class TfidfEncoder:
    def __init__(self, **tfidf_kwargs):
        self.vectorizer = TfidfVectorizer(**tfidf_kwargs)

    def fit_transform(self, texts: Sequence[str]):
        return self.vectorizer.fit_transform(texts).astype(np.float32)

    def transform(self, texts: Sequence[str]):
        return self.vectorizer.transform(texts).astype(np.float32)


class LsaEncoder:
    """TF-IDF followed by truncated SVD (latent semantic analysis)."""

    def __init__(self, n_components: int = 200, seed: int = 0, **tfidf_kwargs):
        self.tfidf = TfidfVectorizer(**tfidf_kwargs)
        self.svd = TruncatedSVD(n_components=n_components, random_state=seed)

    def fit_transform(self, texts: Sequence[str]):
        z = self.svd.fit_transform(self.tfidf.fit_transform(texts))
        return normalize(z).astype(np.float32)

    def transform(self, texts: Sequence[str]):
        return normalize(self.svd.transform(self.tfidf.transform(texts))).astype(np.float32)


class SentenceEncoder:
    """Pretrained sentence embeddings (optional: pip install '.[embeddings]')."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 128):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.batch_size = batch_size

    def fit_transform(self, texts: Sequence[str]):
        return self.transform(texts)

    def transform(self, texts: Sequence[str]):
        return self.model.encode(list(texts), batch_size=self.batch_size,
                                 normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def _rows_dot(a, b) -> np.ndarray:
    out = a @ b.T
    return out.toarray() if sparse.issparse(out) else np.asarray(out)


@dataclass
class FieldRetriever:
    """Holds one encoded matrix per field and scores queries with tunable weights."""

    matrices: Dict[str, object] = field(default_factory=dict)

    def add(self, name: str, matrix) -> None:
        self.matrices[name] = matrix

    def field_sims(self, query_ids: np.ndarray, names: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        """Cosine similarity of each query row to every corpus row, per field."""
        names = list(self.matrices) if names is None else names
        return {n: _rows_dot(self.matrices[n][query_ids], self.matrices[n]) for n in names}

    @staticmethod
    def combine(sims: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        total: Optional[np.ndarray] = None
        for name, w in weights.items():
            if w == 0:
                continue
            total = w * sims[name] if total is None else total + w * sims[name]
        if total is None:
            raise ValueError("at least one field weight must be non-zero")
        return total

    def scores(self, query_ids: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
        names = [n for n, w in weights.items() if w != 0]
        return self.combine(self.field_sims(query_ids, names), weights)
