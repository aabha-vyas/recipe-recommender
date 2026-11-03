"""Load the raw CSV and collapse duplicate rows into unique, multi-label recipes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .text import ingredient_tokens, parse_list, split_sentences

# Categories that describe nutrition. A recipe filed under any of them counts as
# "health-tagged" for the healthiness model.
HEALTH_CATEGORIES = frozenset({
    "Healthy Recipes", "Low Calorie", "Low Fat", "Low Sodium", "Low Cholesterol",
    "High Fiber", "Diabetic", "Low Glycemic Impact",
})


def load_recipes(csv_path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    raw["ingredients_list"] = raw["ingredients"].map(parse_list)
    raw["steps_list"] = raw["directions"].map(parse_list)
    # Same ingredients and steps = same recipe, even if filed under another category.
    raw["_key"] = raw["ingredients_list"].map(tuple).astype(str) + "||" + raw["steps_list"].map(tuple).astype(str)

    grouped = raw.groupby("_key", sort=False)
    first = grouped.first()
    df = pd.DataFrame({
        "title": first["recipe_title"].astype(object).fillna("").astype(str),
        "description": first["description"].astype(object).fillna("").astype(str),
        "ingredients_list": first["ingredients_list"],
        "steps_list": first["steps_list"],
        "categories": grouped["category"].agg(lambda s: sorted(set(s.dropna()))),
        "subcategories": grouped["subcategory"].agg(lambda s: sorted(set(s.dropna()))),
    }).reset_index(drop=True)

    df = df[df["ingredients_list"].map(len) > 0].reset_index(drop=True)
    df["sentences"] = df["steps_list"].map(split_sentences)
    df["num_ingredients"] = df["ingredients_list"].map(len)
    df["num_steps"] = df["steps_list"].map(len)
    df["ingredients_text"] = df["ingredients_list"].map(ingredient_tokens)
    df["title_text"] = df["title"].str.lower()
    df["steps_text"] = df["steps_list"].map(" ".join)
    df["health_label"] = df["categories"].map(lambda cs: int(bool(HEALTH_CATEGORIES & set(cs))))
    df["id"] = np.arange(len(df))
    return df


def split_ids(n: int, fractions=(0.5, 0.5), seed: int = 0) -> list[np.ndarray]:
    """Shuffle 0..n-1 and cut it into parts with the given fractions."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(n)
    cuts = np.cumsum([int(round(f * n)) for f in fractions[:-1]])
    return np.split(ids, cuts)
