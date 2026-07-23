"""Sanity tests for ranking metrics. Run with: python -m pytest tests/ -v"""

import math

from src.metrics.ranking import (
    AggregateMetrics,
    UserResult,
    hit_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_ndcg_perfect_ranking():
    ranked = ["a", "b", "c", "d", "e"]
    relevant = {"a", "b"}
    # Both relevant at positions 0 and 1 → perfect
    assert ndcg_at_k(ranked, relevant, k=5) == 1.0


def test_ndcg_worst_ranking():
    ranked = ["x", "y", "z", "a", "b"]
    relevant = {"a", "b"}
    # Relevant items at positions 3 and 4 (not 0 and 1)
    # NDCG should be < 1
    val = ndcg_at_k(ranked, relevant, k=5)
    assert 0.0 < val < 1.0


def test_ndcg_no_relevant_in_topk():
    ranked = ["x", "y", "z"]
    relevant = {"a", "b"}
    assert ndcg_at_k(ranked, relevant, k=3) == 0.0


def test_ndcg_empty_relevant():
    ranked = ["a", "b"]
    assert ndcg_at_k(ranked, set(), k=10) == 0.0


def test_recall_at_k():
    ranked = ["a", "b", "c", "d"]
    relevant = {"a", "c", "e"}
    # 2 of 3 relevant items recovered in top-4
    assert math.isclose(recall_at_k(ranked, relevant, k=4), 2 / 3)


def test_precision_at_k():
    ranked = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    # 2 of top-4 are relevant
    assert math.isclose(precision_at_k(ranked, relevant, k=4), 0.5)


def test_hit_at_k():
    ranked = ["x", "y", "a"]
    relevant = {"a"}
    assert hit_at_k(ranked, relevant, k=3) == 1.0
    assert hit_at_k(ranked, relevant, k=2) == 0.0


def test_mrr():
    ranked = ["x", "a", "y"]
    relevant = {"a"}
    assert math.isclose(mrr(ranked, relevant), 0.5)


def test_mrr_no_hit():
    ranked = ["x", "y", "z"]
    relevant = {"a"}
    assert mrr(ranked, relevant) == 0.0


def test_user_result_compute():
    r = UserResult(
        user_id="u1",
        ranked=["a", "b", "c"],
        relevant={"a"},
        cold_bin="1",
    )
    r.compute()
    assert r.hit_at_10 == 1.0
    assert r.mrr == 1.0
    assert math.isclose(r.ndcg_at_10, 1.0)


def test_aggregate_metrics_per_bin():
    r1 = UserResult("u1", ["a"], {"a"}, cold_bin="1", total_tokens=100)
    r2 = UserResult("u2", ["x", "b"], {"b"}, cold_bin="3", total_tokens=300, escalated=True)
    r1.compute()
    r2.compute()
    agg = AggregateMetrics.from_user_results([r1, r2])
    assert agg.n_users == 2
    assert agg.escalation_rate == 0.5
    assert agg.avg_tokens_per_user == 200.0
    assert "1" in agg.per_bin
    assert "3" in agg.per_bin
    assert agg.per_bin["1"]["n_users"] == 1
    assert agg.per_bin["3"]["escalation_rate"] == 1.0
