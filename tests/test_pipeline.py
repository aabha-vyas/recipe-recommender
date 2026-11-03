"""End-to-end check on a tiny synthetic CSV (no real data needed)."""
import json

import numpy as np
import pandas as pd
import pytest

from reciperec.data import load_recipes
from reciperec.evaluation import RelevanceJudge
from reciperec.features import build_retriever
from reciperec.recommend import Recommender


def _row(title, cat, sub, ings, steps):
    return {"recipe_title": title, "category": cat, "subcategory": sub, "description": "",
            "ingredients": json.dumps(ings), "directions": json.dumps(steps),
            "num_ingredients": len(ings), "num_steps": len(steps)}


@pytest.fixture
def recipes(tmp_path):
    stew = ["2 pounds beef chuck", "3 carrots", "2 potatoes", "1 onion", "2 cups beef broth"]
    rows = [
        _row("Beef Stew", "Beef Recipes", "Beef Stews", stew, ["Brown the beef. Add vegetables.", "Simmer 2 hours."]),
        _row("Beef Stew", "Comfort Food", "Beef Stews", stew, ["Brown the beef. Add vegetables.", "Simmer 2 hours."]),
        _row("Easy Beef Stew", "Beef Recipes", "Beef Stews", stew[:4], ["Brown beef.", "Simmer with vegetables."]),
        _row("Hearty Beef Stew", "Beef Recipes", "Beef Stews", stew + ["1 cup red wine"], ["Brown beef. Deglaze with wine.", "Simmer."]),
        _row("Chocolate Cake", "Cakes", "Chocolate Cakes", ["2 cups flour", "1 cup sugar", "1/2 cup cocoa", "2 eggs"], ["Mix.", "Bake 30 minutes."]),
        _row("Fudgy Chocolate Cake", "Cakes", "Chocolate Cakes", ["2 cups flour", "1 cup sugar", "1 cup cocoa", "3 eggs"], ["Whisk.", "Bake 35 minutes."]),
        _row("Green Salad", "Healthy Recipes", "Green Salads", ["1 head lettuce", "1 cucumber", "2 tomatoes"], ["Chop.", "Toss."]),
    ]
    path = tmp_path / "recipes.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return load_recipes(path)


def test_duplicate_rows_collapse_and_keep_all_labels(recipes):
    assert len(recipes) == 6
    stew = recipes[recipes["title"] == "Beef Stew"].iloc[0]
    assert stew["categories"] == ["Beef Recipes", "Comfort Food"]
    assert recipes.loc[recipes["title"] == "Green Salad", "health_label"].item() == 1


def test_relevance_counts_shared_subcategories(recipes):
    judge = RelevanceJudge(recipes)
    stew_id = int(recipes.index[recipes["title"] == "Beef Stew"][0])
    assert judge.n_relevant(np.array([stew_id]))[0] == 2  # the two other stews


def test_recommender_ranks_same_dish_first(recipes):
    retriever = build_retriever(recipes, ["tfidf"], cache_dir=None)
    rec = Recommender(recipes, retriever, {"title": 0.3, "ing_words": 0.4, "steps": 0.3}, n_candidates=4)
    stew_id = int(recipes.index[recipes["title"] == "Beef Stew"][0])
    out = rec.recommend(stew_id, k=2)
    assert {r.title for r in out} == {"Easy Beef Stew", "Hearty Beef Stew"}
    hearty = next(r for r in out if r.title == "Hearty Beef Stew")
    assert hearty.new_ingredients == ["red wine"]


def test_fewer_ingredients_preference_prefers_simpler_recipe(recipes):
    retriever = build_retriever(recipes, ["tfidf"], cache_dir=None)
    rec = Recommender(recipes, retriever, {"title": 0.3, "ing_words": 0.4, "steps": 0.3},
                      pref_weight=0.9, n_candidates=4)
    stew_id = int(recipes.index[recipes["title"] == "Beef Stew"][0])
    assert rec.recommend(stew_id, "fewer_ingredients", k=1)[0].title == "Easy Beef Stew"


def test_new_ingredients_ignores_plurals_and_variants():
    from reciperec.recommend import new_ingredients
    have = ["2 yellow onions", "2 1/2 pounds beef chuck", "3 tablespoons olive oil", "1 cup tomato sauce"]
    cand = ["1 onion, cut into chunks", "4 pounds stew beef", "5 cloves garlic, minced",
            "1/2 cup red wine", "2 tablespoons hot sauce", "3 tomatoes"]
    assert new_ingredients(have, cand) == ["garlic", "red wine", "hot sauce"]
