# Recipe Recommender

A content-based recipe recommender, rebuilt as a measured machine learning project.
Pick a recipe you like and it finds similar ones, then re-ranks them by fewer
ingredients, fewer steps or a learned healthiness score.

This started as my CS 410 (Text Information Systems) project. The
course version used TF-IDF with hand-picked weights and a keyword-based health score,
and was judged by reading its output. This version adds:

- **Offline evaluation** with held-out queries and ranking metrics, so changes are measured.
- **Model selection** across TF-IDF, LSA and sentence embeddings, with field weights
  tuned on a dev split and reported on a separate test split.
- **A learned healthiness classifier** that replaces the keyword count.

## Data

25,021 unique recipes. The raw CSV has 62,126 rows because the same recipe is often
filed under several categories; those rows are merged into one recipe that keeps all of
its labels. The CSV isn't in the repo (about 80 MB). See [data/README.md](data/README.md).

## 1. Which representation finds the best matches?

**Relevance label.** A recommendation counts as relevant if it shares a subcategory with
the query recipe (for example both are "Beef Stews"). Labels are only used to score and
tune, never as model input.

**Protocol.** Queries are split into 1,500 dev and 3,000 test recipes. Each model
retrieves its top 10 from all 25,021 recipes. Field weights (title, ingredients, steps)
are grid-searched on dev; every number below is on test.

| model | nDCG@10 | P@10 | MRR | hit@10 |
|---|---|---|---|---|
| random | 0.003 | 0.003 | 0.009 | 0.029 |
| popularity | 0.028 | 0.026 | 0.055 | 0.111 |
| course project (TF-IDF, fixed 0.85/0.15) | 0.363 | 0.346 | 0.549 | 0.796 |
| TF-IDF, ingredients only | 0.160 | 0.145 | 0.327 | 0.583 |
| TF-IDF, tuned weights | 0.452 | 0.430 | 0.639 | 0.852 |
| TF-IDF, tuned, no title | 0.358 | 0.335 | 0.570 | 0.815 |
| LSA, tuned weights | 0.399 | 0.378 | 0.598 | 0.825 |
| **sentence embeddings, tuned** | **0.470** | **0.450** | **0.649** | **0.852** |
| hybrid TF-IDF + embeddings, tuned | 0.460 | 0.441 | 0.640 | 0.853 |

Sentence embeddings (`all-MiniLM-L6-v2`) scored best on dev and were selected. On test
they beat the course setup by **+0.107 nDCG@10 (95% paired bootstrap CI +0.098 to +0.115)**.

What I learned:

- **Tuning mattered more than the model.** Re-weighting the same TF-IDF features took
  nDCG@10 from 0.363 to 0.452. The tuned weights put most weight on cooking steps
  (0.57), which the course version weighted at 0.15.
- **Ingredients alone are a weak signal** (0.160). Many dishes share pantry staples;
  how a recipe is cooked separates a stew from a soup.
- **The title helps a lot** (0.452 vs. 0.358 without it). That's partly fair, since
  titles name the dish, and partly a caveat: subcategory names often appear in titles,
  so this label rewards title matching.
- **Embeddings vs. tuned TF-IDF is close** (0.470 vs. 0.452). I didn't run a
  significance test between those two. TF-IDF needs no PyTorch and is the default in
  the command-line tool.
- The hybrid's grid was coarser (weights in {0, 0.5, 1}) and left out the title
  embedding to keep the search small, so it isn't a fair test of combining the two.

## 2. Can a model judge healthiness better than keywords?

**Label.** 1 if the recipe is filed under any nutrition category (Healthy Recipes, Low
Calorie, Low Fat, Low Sodium, Low Cholesterol, High Fiber, Diabetic, Low Glycemic
Impact). 9.4% of recipes are positive.

**Model.** Logistic regression on TF-IDF of the ingredients only (no title or
description, so it can't read the word "healthy"). Word pairs stay inside one
ingredient. Split 70/15/15; regularization chosen on validation.

| model | ROC-AUC | average precision |
|---|---|---|
| always guess the base rate | 0.500 | 0.094 |
| keyword heuristic (CS 410) | 0.674 | 0.188 |
| **logistic regression** | **0.796** | **0.321** |

Ingredients that push toward health-tagged: banana, onion, carrots, nonfat, whole wheat,
sweetener, sugar free, cucumber, chickpeas. Away: cheese, butter, sausage, chocolate,
cream, margarine, mayonnaise, shortening.

Caveat: a missing tag doesn't mean a recipe is unhealthy, only that it wasn't filed
that way, so the label is noisy and the score is a ranking signal, not nutrition advice.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
# put the CSV at data/recipes.csv

python scripts/train_health.py                   # health model + results/health.*
python scripts/evaluate_retrieval.py             # retrieval comparison + results/retrieval.*
python scripts/recommend.py --title "beef stew" --preference healthier
pytest
```

Example output (abridged):

```
Because you liked: Hawaiian-Style Beef Stew  (15 ingredients, 7 steps)
Preference: healthier

1. Hawaiian Beef Stew
   match [##########]  16 ingredients, 4 steps, health score 0.42
   adds: garlic, red wine, water, white sugar, bay leaves
   summary: Add tomato paste, salt, sugar, bay leaves, and pepper. Place carrots into
   stew; cook until slightly tender, about 10 minutes.
```

The match bar is relative to the best result shown. "adds" lists ingredients the
original recipe doesn't have, and the summary is the two highest-TF-IDF sentences
from the steps.

**Sentence embeddings (optional).** Install `requirements-embeddings.txt` in a separate
environment (it pins `numpy<2` for PyTorch 2.2.2, the last release with Intel-Mac
wheels), then run `evaluate_retrieval.py --embed` or `recommend.py --model embeddings`.
Embeddings are cached in `artifacts/`. Encoding the steps took about three hours on an
Intel Mac CPU.

## Layout

```
src/reciperec/
  data.py        load the CSV, merge duplicate rows, build labels
  text.py        ingredient normalization and tokenizers
  models.py      TF-IDF, LSA and sentence-embedding encoders; weighted field retriever
  features.py    builds every field used in the experiments
  evaluation.py  relevance labels, evaluation loop, weight tuning
  metrics.py     nDCG, precision, MRR, paired bootstrap
  health.py      keyword heuristic and the healthiness classifier
  recommend.py   re-ranking by preference, summaries, new ingredients
scripts/         evaluate_retrieval.py, train_health.py, recommend.py
results/         the numbers above, as JSON and Markdown
tests/           unit tests and a small end-to-end test
```

## Limits

- Relevance is a proxy: sharing a subcategory isn't the same as a person liking the
  recommendation. There are no user ratings in the data.
- Duplicates are merged only when ingredients and steps match exactly, so near-duplicate
  recipes can still land in both dev and test.
- One dataset and one random split; the numbers come with a bootstrap interval but no
  repeated runs.
