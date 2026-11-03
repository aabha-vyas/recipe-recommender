import numpy as np
import pytest

from reciperec.evaluation import weight_grid
from reciperec.metrics import paired_bootstrap, ranking_metrics, top_k


def test_top_k_never_returns_the_query_itself():
    scores = np.array([[9.0, 1.0, 5.0, 3.0], [0.0, 9.0, 8.0, 7.0]])
    ranked = top_k(scores, 2, exclude=[0, 1])
    assert ranked.tolist() == [[2, 3], [2, 3]]


def test_perfect_ranking_scores_one():
    m = ranking_metrics(np.ones((3, 10)), n_relevant=np.array([20, 20, 20]))
    assert m["ndcg@10"] == pytest.approx(1.0)
    assert m["precision@10"] == 1.0 and m["mrr"] == 1.0


def test_ndcg_rewards_hits_near_the_top():
    early = np.zeros((1, 10)); early[0, 0] = 1
    late = np.zeros((1, 10)); late[0, 9] = 1
    n = np.array([1])
    assert ranking_metrics(early, n)["ndcg@10"] > ranking_metrics(late, n)["ndcg@10"]
    assert ranking_metrics(early, n)["ndcg@10"] == pytest.approx(1.0)


def test_no_hits():
    m = ranking_metrics(np.zeros((2, 10)), np.array([5, 5]))
    assert m["ndcg@10"] == 0 and m["mrr"] == 0 and m["hit@10"] == 0


def test_bootstrap_ci_contains_mean_difference():
    rng = np.random.default_rng(0)
    a, b = rng.random(500) + 0.1, rng.random(500)
    diff, lo, hi = paired_bootstrap(a, b)
    assert lo < diff < hi and lo > 0


def test_weight_grid_is_normalized_and_deduplicated():
    grid = weight_grid(["a", "b"], values=(0, 0.5, 1.0))
    assert all(sum(w.values()) == pytest.approx(1.0) for w in grid)
    assert len(grid) == len({tuple(sorted(w.items())) for w in grid}) == 5  # 0:1, 1:2, 1:1, 2:1, 1:0
