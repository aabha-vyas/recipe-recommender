"""Turn retrieval scores into recommendations a person can use."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import FieldRetriever
from .text import ingredient_keywords, normalize_ingredient

PREFERENCES = ("none", "fewer_ingredients", "fewer_steps", "healthier")


@dataclass
class Recommendation:
    id: int
    title: str
    similarity: float
    score: float
    num_ingredients: int
    num_steps: int
    health: Optional[float]
    new_ingredients: List[str]
    summary: str


def preference_signal(query: pd.Series, cands: pd.DataFrame, preference: str,
                      health: Optional[np.ndarray]) -> np.ndarray:
    if preference == "fewer_ingredients":
        return np.clip(query["num_ingredients"] - cands["num_ingredients"].to_numpy(), -5, 5).astype(float)
    if preference == "fewer_steps":
        return np.clip(query["num_steps"] - cands["num_steps"].to_numpy(), -5, 5).astype(float)
    if preference == "healthier":
        if health is None:
            raise ValueError("preference 'healthier' needs a trained health model")
        return health[cands.index.to_numpy()]
    return np.zeros(len(cands))


def _minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def summarize(sentences: List[str], vectorizer: TfidfVectorizer, n: int = 2) -> str:
    """Pick the n highest-TF-IDF sentences and keep them in their original order."""
    if len(sentences) <= n:
        return " ".join(sentences)
    weights = np.asarray(vectorizer.transform(sentences).sum(axis=1)).ravel()
    keep = sorted(np.argsort(-weights)[:n])
    return " ".join(sentences[i] for i in keep)


def new_ingredients(have: List[str], candidate: List[str]) -> List[str]:
    """Ingredients in the candidate that share no key word with anything in `have`."""
    known = set().union(*(ingredient_keywords(normalize_ingredient(x)) for x in have)) if have else set()
    out = []
    for ing in candidate:
        norm = normalize_ingredient(ing)
        if norm and not (ingredient_keywords(norm) & known) and norm not in out:
            out.append(norm)
    return out


class Recommender:
    def __init__(self, df: pd.DataFrame, retriever: FieldRetriever, weights: Dict[str, float],
                 health_model=None, pref_weight: float = 0.3, n_candidates: int = 50):
        self.df = df
        self.retriever = retriever
        self.weights = weights
        self.pref_weight = pref_weight
        self.n_candidates = n_candidates
        self.health = (health_model.predict_proba(df["ingredients_text"])[:, 1]
                       if health_model is not None else None)
        self.summary_vec = TfidfVectorizer(stop_words="english", min_df=2).fit(df["steps_text"])

    def find(self, title_query: str, limit: int = 5) -> pd.DataFrame:
        hits = self.df[self.df["title"].str.contains(title_query, case=False, regex=False)]
        return hits[["id", "title"]].head(limit)

    def recommend(self, recipe_id: int, preference: str = "none", k: int = 5) -> List[Recommendation]:
        if preference not in PREFERENCES:
            raise ValueError(f"preference must be one of {PREFERENCES}")
        q = self.df.iloc[recipe_id]
        sims = self.retriever.scores(np.array([recipe_id]), self.weights)[0]
        sims[recipe_id] = -np.inf
        top = np.argpartition(-sims, self.n_candidates)[: self.n_candidates]
        cands = self.df.iloc[top]
        sim = sims[top]
        pref = preference_signal(q, cands, preference, self.health)
        w = self.pref_weight if preference != "none" else 0.0
        score = (1 - w) * _minmax(sim) + w * _minmax(pref)
        order = np.argsort(-score)[:k]

        out = []
        for i in order:
            row = cands.iloc[i]
            new = new_ingredients(q["ingredients_list"], row["ingredients_list"])
            out.append(Recommendation(
                id=int(row["id"]), title=row["title"], similarity=float(sim[i]), score=float(score[i]),
                num_ingredients=int(row["num_ingredients"]), num_steps=int(row["num_steps"]),
                health=None if self.health is None else float(self.health[row["id"]]),
                new_ingredients=new[:5], summary=summarize(row["sentences"], self.summary_vec),
            ))
        return out

