"""Healthiness: the original keyword heuristic and a learned classifier.

Label: 1 if the recipe is filed under any nutrition category (see data.HEALTH_CATEGORIES).
Input: ingredients only, so the model can't read "healthy" off the title or description.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .text import ingredient_analyzer

# From the CS 410 version of the project.
UNHEALTHY_KEYWORDS = {"butter", "sugar", "cream", "bacon", "oil", "cheese", "fried"}
HEALTHY_KEYWORDS = {"broccoli", "spinach", "kale", "tomato", "beans", "lentil", "chickpea", "avocado", "apple", "banana"}


def keyword_score(ingredients: List[str]) -> float:
    if not ingredients:
        return 0.5
    text = " ".join(ingredients).lower()
    healthy = sum(text.count(k) for k in HEALTHY_KEYWORDS)
    unhealthy = sum(text.count(k) for k in UNHEALTHY_KEYWORDS)
    score = 0.5 + 0.25 * (healthy - unhealthy) / max(len(ingredients), 1)
    return float(min(max(score, 0.0), 1.0))


def keyword_scores(ingredient_lists: Iterable[List[str]]) -> np.ndarray:
    return np.array([keyword_score(x) for x in ingredient_lists])


def make_classifier(C: float = 1.0, seed: int = 0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer=ingredient_analyzer, min_df=3, sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)),
    ])


def top_features(model: Pipeline, n: int = 15):
    names = model.named_steps["tfidf"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_.ravel()
    order = np.argsort(coef)
    return [names[i] for i in order[::-1][:n]], [names[i] for i in order[:n]]
