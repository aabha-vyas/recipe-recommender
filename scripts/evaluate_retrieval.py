"""Compare recipe representations on held-out queries.

Field weights are tuned on dev queries and every number reported is on separate
test queries. Relevance = the recommended recipe shares a subcategory with the query.

    python scripts/evaluate_retrieval.py --data data/recipes.csv [--embed]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from reciperec.data import load_recipes, split_ids
from reciperec.evaluation import RelevanceJudge, evaluate_scores, tune_weights
from reciperec.features import EMBED_FIELDS, LSA_FIELDS, TFIDF_FIELDS, build_retriever
from reciperec.metrics import paired_bootstrap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/recipes.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--dev", type=int, default=1500)
    ap.add_argument("--test", type=int, default=3000)
    ap.add_argument("--embed", action="store_true", help="also evaluate sentence embeddings")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    df = load_recipes(args.data)
    judge = RelevanceJudge(df)
    print(f"{len(df)} unique recipes ({time.time() - t0:.0f}s)")

    # Queries need at least one other recipe with a shared subcategory.
    has_rel = np.where(judge.n_relevant(np.arange(len(df))) > 0)[0]
    order = split_ids(len(has_rel), (0.5, 0.5), seed=args.seed)
    rng = np.random.default_rng(args.seed)
    dev_ids = np.sort(rng.choice(has_rel[order[0]], args.dev, replace=False))
    test_ids = np.sort(rng.choice(has_rel[order[1]], args.test, replace=False))

    groups = ["tfidf", "course", "lsa"] + (["embed"] if args.embed else [])
    t0 = time.time()
    retriever = build_retriever(df, groups, seed=args.seed)
    print(f"built fields {list(retriever.matrices)} ({time.time() - t0:.0f}s)")

    n = len(df)
    pop = np.array([len(s) for s in df["subcategories"]], dtype=np.float32)
    pop += np.random.default_rng(1).random(n).astype(np.float32) * 1e-3
    baselines = {
        "random": lambda q: np.random.default_rng(len(q) + int(q[0])).random((len(q), n)).astype(np.float32),
        "popularity": lambda q: np.tile(pop, (len(q), 1)),
    }

    experiments = [
        ("course project (TF-IDF, fixed 0.85/0.15)", {"course_search": 0.85, "course_steps": 0.15}, None),
        ("TF-IDF, ingredients only", {"ing_phrase": 1.0}, None),
        ("TF-IDF, tuned weights", None, list(TFIDF_FIELDS)),
        ("TF-IDF, tuned, no title", None, ["ing_words", "ing_phrase", "steps"]),
        ("LSA, tuned weights", None, list(LSA_FIELDS)),
    ]
    if args.embed:
        experiments += [
            ("sentence embeddings, tuned", None, list(EMBED_FIELDS)),
            ("hybrid TF-IDF + embeddings, tuned", None, ["title", "ing_words", "steps", "emb_ing", "emb_steps"]),
        ]

    rows, per_query = [], {}
    for name, fn in baselines.items():
        m, pq = evaluate_scores(fn, test_ids, judge)
        rows.append({"model": name, "weights": None, **m})
        per_query[name] = pq
        print(f"{name:45s} ndcg@10={m['ndcg@10']:.3f}")

    for name, weights, tune_fields in experiments:
        t0 = time.time()
        dev_score = None
        if tune_fields is not None:
            # Coarser grid when there are many fields, to keep the search small.
            values = (0, 0.25, 0.5, 1.0) if len(tune_fields) <= 4 else (0, 0.5, 1.0)
            weights, dev_score, tried = tune_weights(retriever, tune_fields, dev_ids, judge, values=values)
        m, pq = evaluate_scores(lambda q, w=weights: retriever.scores(q, w), test_ids, judge)
        rows.append({"model": name, "weights": weights, "dev_ndcg@10": dev_score, **m})
        per_query[name] = pq
        print(f"{name:45s} ndcg@10={m['ndcg@10']:.3f}  weights={weights}  ({time.time() - t0:.0f}s)")

    model_rows = [r for r in rows if r["weights"] is not None]
    best = max(model_rows, key=lambda r: r.get("dev_ndcg@10") or -1)
    base = "course project (TF-IDF, fixed 0.85/0.15)"
    diff, lo, hi = paired_bootstrap(per_query[best["model"]], per_query[base])

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    result = {
        "n_recipes": len(df), "n_dev_queries": len(dev_ids), "n_test_queries": len(test_ids),
        "relevance": "shares at least one subcategory with the query recipe",
        "selected_by_dev": best["model"],
        "selected_vs_course_ndcg@10": {"diff": diff, "ci95": [lo, hi]},
        "results": rows,
    }
    (out / "retrieval.json").write_text(json.dumps(result, indent=2))
    lines = ["| model | nDCG@10 | P@10 | MRR | hit@10 |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['ndcg@10']:.3f} | {r['precision@10']:.3f} | {r['mrr']:.3f} | {r['hit@10']:.3f} |")
    lines.append("")
    lines.append(f"Selected on dev: **{best['model']}**. Test nDCG@10 vs. the course setup: "
                 f"{diff:+.3f} (95% bootstrap CI {lo:+.3f} to {hi:+.3f}).")
    (out / "retrieval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
