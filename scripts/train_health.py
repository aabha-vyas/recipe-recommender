"""Train the healthiness classifier and compare it with the original keyword heuristic.

Split: 70% train, 15% validation (to pick C), 15% test (reported once).

    python scripts/train_health.py --data data/recipes.csv
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from reciperec.data import HEALTH_CATEGORIES, load_recipes
from reciperec.features import field_texts
from reciperec.health import keyword_scores, make_classifier, top_features


def scores(y, p):
    return {"roc_auc": float(roc_auc_score(y, p)), "avg_precision": float(average_precision_score(y, p))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/recipes.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model-out", default="artifacts/health_model.pkl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_recipes(args.data)
    X = field_texts(df)["ing_phrase"].to_numpy()
    y = df["health_label"].to_numpy()
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=args.seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=args.seed)
    print(f"{len(df)} recipes, {y.mean():.1%} health-tagged; train {len(tr)}, val {len(va)}, test {len(te)}")

    search = {}
    for C in (0.1, 0.3, 1.0, 3.0, 10.0):
        m = make_classifier(C, args.seed).fit(X[tr], y[tr])
        search[C] = scores(y[va], m.predict_proba(X[va])[:, 1])["avg_precision"]
        print(f"C={C:<5} val avg precision={search[C]:.3f}")
    best_C = max(search, key=search.get)

    model = make_classifier(best_C, args.seed).fit(X[np.concatenate([tr, va])], y[np.concatenate([tr, va])])
    learned = scores(y[te], model.predict_proba(X[te])[:, 1])
    heuristic = scores(y[te], keyword_scores(df["ingredients_list"].iloc[te]))
    base_rate = float(y[te].mean())
    healthy_terms, unhealthy_terms = top_features(model)

    result = {
        "label": "filed under any of: " + ", ".join(sorted(HEALTH_CATEGORIES)),
        "input": "normalized ingredients only (words and within-ingredient word pairs)",
        "n_train": int(len(tr) + len(va)), "n_test": int(len(te)), "test_base_rate": base_rate,
        "C_search_val_avg_precision": search, "chosen_C": best_C,
        "test": {"keyword heuristic (CS 410)": heuristic, "logistic regression": learned},
        "most_healthy_terms": healthy_terms, "least_healthy_terms": unhealthy_terms,
    }
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "health.json").write_text(json.dumps(result, indent=2))
    md = [
        "| model | ROC-AUC | average precision |", "|---|---|---|",
        f"| always guess the base rate | 0.500 | {base_rate:.3f} |",
        f"| keyword heuristic (CS 410) | {heuristic['roc_auc']:.3f} | {heuristic['avg_precision']:.3f} |",
        f"| logistic regression (C={best_C}) | {learned['roc_auc']:.3f} | {learned['avg_precision']:.3f} |",
        "", f"Terms pushing toward health-tagged: {', '.join(healthy_terms)}",
        "", f"Terms pushing away: {', '.join(unhealthy_terms)}",
    ]
    (out / "health.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
