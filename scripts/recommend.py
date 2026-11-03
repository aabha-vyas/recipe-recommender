"""Recommend recipes similar to one you pick.

    python scripts/recommend.py --title "beef stew" --preference fewer_steps
    python scripts/recommend.py --id 1234 --preference healthier

Uses field weights tuned on the dev split (results/retrieval.json) and the
health model from scripts/train_health.py (artifacts/health_model.pkl).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

from reciperec.data import load_recipes
from reciperec.features import build_retriever
from reciperec.recommend import PREFERENCES, Recommender

MODELS = {"tfidf": "TF-IDF, tuned weights", "embeddings": "sentence embeddings, tuned"}
FALLBACK_WEIGHTS = {"title": 0.2857, "ing_words": 0.1429, "steps": 0.5714}


def load_weights(path: Path, model: str) -> dict:
    """Field weights tuned on the dev split for the chosen model."""
    if not path.exists():
        if model != "tfidf":
            sys.exit(f"{path} not found; run scripts/evaluate_retrieval.py --embed first")
        return FALLBACK_WEIGHTS
    result = json.loads(path.read_text())
    chosen = next(r for r in result["results"] if r["model"] == MODELS[model])
    return {k: v for k, v in chosen["weights"].items() if v}


def match_bar(similarity: float, top: float) -> str:
    filled = max(1, round(10 * similarity / top)) if top > 0 else 0
    return "[" + "#" * filled + "." * (10 - filled) + "]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/recipes.csv")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--title", help="search recipe titles and use the first match")
    group.add_argument("--id", type=int, help="recipe id")
    ap.add_argument("--preference", choices=PREFERENCES, default="none")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--model", choices=MODELS, default="tfidf",
                    help="tfidf needs no extra installs; embeddings needs requirements-embeddings.txt")
    ap.add_argument("--weights", default="results/retrieval.json")
    ap.add_argument("--health-model", default="artifacts/health_model.pkl")
    args = ap.parse_args()

    weights = load_weights(Path(args.weights), args.model)
    df = load_recipes(args.data)
    groups = {"embed" if f.startswith("emb_") else "lsa" if f.startswith("lsa_") else "tfidf" for f in weights}
    retriever = build_retriever(df, groups)
    health = None
    if Path(args.health_model).exists():
        with open(args.health_model, "rb") as f:
            health = pickle.load(f)
    elif args.preference == "healthier":
        sys.exit("run scripts/train_health.py first to use --preference healthier")
    rec = Recommender(df, retriever, weights, health_model=health)

    if args.title:
        hits = rec.find(args.title)
        if hits.empty:
            sys.exit(f"no recipe title contains {args.title!r}")
        recipe_id = int(hits.iloc[0]["id"])
    else:
        recipe_id = args.id
    q = df.iloc[recipe_id]
    print(f"\nBecause you liked: {q['title']}  ({q['num_ingredients']} ingredients, {q['num_steps']} steps)")
    print(f"Preference: {args.preference}\n")

    results = rec.recommend(recipe_id, args.preference, args.k)
    top = max(r.similarity for r in results)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.title}")
        line = f"   match {match_bar(r.similarity, top)}  {r.num_ingredients} ingredients, {r.num_steps} steps"
        if r.health is not None:
            line += f", health score {r.health:.2f}"
        print(line)
        if r.new_ingredients:
            print(f"   adds: {', '.join(r.new_ingredients)}")
        print(f"   summary: {r.summary}\n")


if __name__ == "__main__":
    main()
