"""Offline evaluation of retrieval using subcategory labels as relevance.

A recommendation counts as relevant if it shares at least one subcategory with
the query recipe (e.g. both are filed under "Beef Stews"). Labels are used only
for evaluation and tuning, never as model input.
"""
from __future__ import annotations

import itertools
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer

from .metrics import per_query_ndcg, ranking_metrics, top_k
from .models import FieldRetriever


class RelevanceJudge:
    def __init__(self, df: pd.DataFrame, label_col: str = "subcategories"):
        mlb = MultiLabelBinarizer(sparse_output=True)
        self.labels = sparse.csr_matrix(mlb.fit_transform(df[label_col]), dtype=np.float32)
        self.labels_t = self.labels.T.tocsr()

    def n_relevant(self, query_ids: np.ndarray) -> np.ndarray:
        overlap = self.labels[query_ids] @ self.labels_t
        return np.asarray((overlap > 0).sum(axis=1)).ravel() - 1  # minus the query itself

    def relevance(self, query_ids: np.ndarray, ranked: np.ndarray) -> np.ndarray:
        k = ranked.shape[1]
        q = self.labels[np.repeat(query_ids, k)]
        c = self.labels[ranked.ravel()]
        shared = np.asarray(q.multiply(c).sum(axis=1)).ravel()
        return (shared > 0).astype(np.float32).reshape(ranked.shape)


def evaluate_scores(score_fn: Callable[[np.ndarray], np.ndarray], query_ids: np.ndarray,
                    judge: RelevanceJudge, k: int = 10, batch: int = 500):
    """Returns (summary metrics, per-query nDCG@k)."""
    rels, n_rel = [], []
    for start in range(0, len(query_ids), batch):
        q = query_ids[start:start + batch]
        ranked = top_k(score_fn(q), k, exclude=q)
        rels.append(judge.relevance(q, ranked))
        n_rel.append(judge.n_relevant(q))
    rel, n = np.vstack(rels), np.concatenate(n_rel)
    return ranking_metrics(rel, n), per_query_ndcg(rel, n)


def weight_grid(fields: List[str], values: Iterable[float] = (0, 0.25, 0.5, 1.0)) -> List[Dict[str, float]]:
    grid = []
    for combo in itertools.product(list(values), repeat=len(fields)):
        if sum(combo) == 0:
            continue
        total = sum(combo)
        w = {f: round(c / total, 4) for f, c in zip(fields, combo)}
        if w not in grid:
            grid.append(w)
    return grid


def tune_weights(retriever: FieldRetriever, fields: List[str], dev_ids: np.ndarray,
                 judge: RelevanceJudge, k: int = 10, metric: str = "ndcg@10",
                 values: Iterable[float] = (0, 0.25, 0.5, 1.0)):
    """Grid-search field weights on the dev queries. Returns (best_weights, best_score, n_tried)."""
    sims = {f: retriever.field_sims(dev_ids, [f])[f] for f in fields}
    ranked_rel = judge.n_relevant(dev_ids)
    best_w, best = None, -1.0
    grid = weight_grid(fields, values)
    for w in grid:
        scores = FieldRetriever.combine(sims, w)
        ranked = top_k(scores, k, exclude=dev_ids)
        m = ranking_metrics(judge.relevance(dev_ids, ranked), ranked_rel)[metric]
        if m > best:
            best_w, best = w, m
    return best_w, best, len(grid)

