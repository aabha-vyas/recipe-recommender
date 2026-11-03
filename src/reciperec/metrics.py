"""Ranking metrics with binary relevance."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def top_k(scores: np.ndarray, k: int, exclude: Sequence[int]) -> np.ndarray:
    """Indices of the k highest scores per row, never returning the row's own item."""
    s = scores.copy()
    s[np.arange(len(exclude)), np.asarray(exclude)] = -np.inf
    part = np.argpartition(-s, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def per_query_ndcg(rel: np.ndarray, n_relevant: np.ndarray) -> np.ndarray:
    k = rel.shape[1]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (rel * discounts).sum(axis=1)
    ideal = np.array([discounts[: min(int(n), k)].sum() for n in n_relevant])
    return np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)


def ranking_metrics(rel: np.ndarray, n_relevant: np.ndarray) -> Dict[str, float]:
    """rel: (queries, k) 0/1 matrix of relevance in rank order.
    n_relevant: number of relevant items per query in the whole corpus."""
    k = rel.shape[1]
    first_hit = np.where(rel.any(axis=1), rel.argmax(axis=1) + 1, np.inf)
    return {
        f"precision@{k}": float(rel.mean()),
        f"ndcg@{k}": float(per_query_ndcg(rel, n_relevant).mean()),
        "mrr": float(np.mean(1.0 / first_hit)),
        f"hit@{k}": float(rel.any(axis=1).mean()),
    }


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """95% CI for mean(a - b) over queries, resampling queries with replacement."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
