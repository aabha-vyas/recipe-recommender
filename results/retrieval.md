| model | nDCG@10 | P@10 | MRR | hit@10 |
|---|---|---|---|---|
| random | 0.003 | 0.003 | 0.009 | 0.029 |
| popularity | 0.028 | 0.026 | 0.055 | 0.111 |
| course project (TF-IDF, fixed 0.85/0.15) | 0.363 | 0.346 | 0.549 | 0.796 |
| TF-IDF, ingredients only | 0.160 | 0.145 | 0.327 | 0.583 |
| TF-IDF, tuned weights | 0.452 | 0.430 | 0.639 | 0.852 |
| TF-IDF, tuned, no title | 0.358 | 0.335 | 0.570 | 0.815 |
| LSA, tuned weights | 0.399 | 0.378 | 0.598 | 0.825 |
| sentence embeddings, tuned | 0.470 | 0.450 | 0.649 | 0.852 |
| hybrid TF-IDF + embeddings, tuned | 0.460 | 0.441 | 0.640 | 0.853 |

Selected on dev: **sentence embeddings, tuned**. Test nDCG@10 vs. the course setup: +0.107 (95% bootstrap CI +0.098 to +0.115).
