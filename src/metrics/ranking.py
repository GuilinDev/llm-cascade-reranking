"""
Ranking metrics for CascadeRec.

All metrics operate on ranked item-id lists plus a ground-truth set or dict.
No PyTorch. Pure numpy so that metrics are cheap and dependency-light.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set


def _dcg(relevances: Sequence[float]) -> float:
    return sum(
        rel / math.log2(i + 2)  # positions are 0-indexed → log2(i+2)
        for i, rel in enumerate(relevances)
    )


def ndcg_at_k(ranked: Sequence[str], relevant: Set[str], k: int = 10) -> float:
    """Binary-relevance NDCG@k."""
    if not relevant:
        return 0.0
    top_k = ranked[:k]
    gains = [1.0 if item in relevant else 0.0 for item in top_k]
    dcg = _dcg(gains)
    ideal_hits = min(len(relevant), k)
    idcg = _dcg([1.0] * ideal_hits)
    return dcg / idcg if idcg > 0 else 0.0


def ndcg_at_k_graded(
    ranked: Sequence[str],
    relevance_dict: Dict[str, float],
    k: int = 10,
) -> float:
    """Graded-relevance NDCG@k (useful for rating-based ground truth)."""
    top_k = ranked[:k]
    gains = [relevance_dict.get(item, 0.0) for item in top_k]
    dcg = _dcg(gains)
    ideal_gains = sorted(relevance_dict.values(), reverse=True)[:k]
    idcg = _dcg(ideal_gains)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Set[str], k: int = 20) -> float:
    if not relevant:
        return 0.0
    top_k = set(ranked[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: Set[str], k: int = 10) -> float:
    if k == 0:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def hit_at_k(ranked: Sequence[str], relevant: Set[str], k: int = 10) -> float:
    return 1.0 if any(item in relevant for item in ranked[:k]) else 0.0


def mrr(ranked: Sequence[str], relevant: Set[str]) -> float:
    for i, item in enumerate(ranked):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def intra_list_diversity(
    ranked: Sequence[str],
    item_embeddings: Dict[str, Sequence[float]],
    k: int = 10,
) -> float:
    """Average pairwise cosine distance over the top-k items."""
    import numpy as np

    top_k = [item for item in ranked[:k] if item in item_embeddings]
    if len(top_k) < 2:
        return 0.0
    vecs = np.asarray([item_embeddings[item] for item in top_k], dtype=float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = vecs / norms
    sims = normalized @ normalized.T
    n = len(top_k)
    total_distance = 0.0
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_distance += 1.0 - sims[i, j]
            pair_count += 1
    return total_distance / pair_count if pair_count > 0 else 0.0


def catalog_coverage(all_rankings: Sequence[Sequence[str]], catalog_size: int, k: int = 10) -> float:
    surfaced: Set[str] = set()
    for ranked in all_rankings:
        surfaced.update(ranked[:k])
    return len(surfaced) / catalog_size if catalog_size > 0 else 0.0


@dataclass
class UserResult:
    """Metrics + cost for a single user."""

    user_id: str
    ranked: List[str]
    relevant: Set[str]
    cold_bin: str = "all"
    ndcg_at_10: float = 0.0
    ndcg_at_20: float = 0.0
    recall_at_20: float = 0.0
    mrr: float = 0.0
    hit_at_10: float = 0.0
    precision_at_10: float = 0.0
    # compute cost for this user (set by CascadeRec harness)
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    escalated: bool = False
    # LLM output parsing diagnostics (None = no LLM call at that tier)
    tier1_parse_ok: bool = None
    tier2_parse_ok: bool = None
    fallback: bool = False  # exception path: first-stage ranking used as-is

    def compute(self) -> None:
        self.ndcg_at_10 = ndcg_at_k(self.ranked, self.relevant, k=10)
        self.ndcg_at_20 = ndcg_at_k(self.ranked, self.relevant, k=20)
        self.recall_at_20 = recall_at_k(self.ranked, self.relevant, k=20)
        self.mrr = mrr(self.ranked, self.relevant)
        self.hit_at_10 = hit_at_k(self.ranked, self.relevant, k=10)
        self.precision_at_10 = precision_at_k(self.ranked, self.relevant, k=10)


@dataclass
class AggregateMetrics:
    """Mean metrics across a set of users plus compute totals."""

    n_users: int = 0
    ndcg_at_10: float = 0.0
    ndcg_at_20: float = 0.0
    recall_at_20: float = 0.0
    mrr: float = 0.0
    hit_at_10: float = 0.0
    precision_at_10: float = 0.0
    avg_tokens_per_user: float = 0.0
    avg_latency_ms_per_user: float = 0.0
    escalation_rate: float = 0.0
    tier1_parse_failure_rate: float = 0.0
    tier2_parse_failure_rate: float = 0.0
    fallback_rate: float = 0.0
    per_bin: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @staticmethod
    def _parse_failure_rate(flags: List) -> float:
        judged = [f for f in flags if f is not None]
        if not judged:
            return 0.0
        return sum(1 for f in judged if not f) / len(judged)

    @classmethod
    def from_user_results(cls, results: List[UserResult]) -> "AggregateMetrics":
        if not results:
            return cls()
        agg = cls(n_users=len(results))
        n = float(len(results))
        agg.ndcg_at_10 = sum(r.ndcg_at_10 for r in results) / n
        agg.ndcg_at_20 = sum(r.ndcg_at_20 for r in results) / n
        agg.recall_at_20 = sum(r.recall_at_20 for r in results) / n
        agg.mrr = sum(r.mrr for r in results) / n
        agg.hit_at_10 = sum(r.hit_at_10 for r in results) / n
        agg.precision_at_10 = sum(r.precision_at_10 for r in results) / n
        agg.avg_tokens_per_user = sum(r.total_tokens for r in results) / n
        agg.avg_latency_ms_per_user = sum(r.total_latency_ms for r in results) / n
        agg.escalation_rate = sum(1 for r in results if r.escalated) / n
        agg.tier1_parse_failure_rate = cls._parse_failure_rate([r.tier1_parse_ok for r in results])
        agg.tier2_parse_failure_rate = cls._parse_failure_rate([r.tier2_parse_ok for r in results])
        agg.fallback_rate = sum(1 for r in results if r.fallback) / n
        # per cold-start bin breakdown
        bins: Dict[str, List[UserResult]] = {}
        for r in results:
            bins.setdefault(r.cold_bin, []).append(r)
        for bin_name, bucket in bins.items():
            m = float(len(bucket))
            agg.per_bin[bin_name] = {
                "n_users": len(bucket),
                "ndcg_at_10": sum(r.ndcg_at_10 for r in bucket) / m,
                "ndcg_at_20": sum(r.ndcg_at_20 for r in bucket) / m,
                "recall_at_20": sum(r.recall_at_20 for r in bucket) / m,
                "mrr": sum(r.mrr for r in bucket) / m,
                "escalation_rate": sum(1 for r in bucket if r.escalated) / m,
                "avg_tokens": sum(r.total_tokens for r in bucket) / m,
                "tier1_parse_failure_rate": cls._parse_failure_rate(
                    [r.tier1_parse_ok for r in bucket]
                ),
                "fallback_rate": sum(1 for r in bucket if r.fallback) / m,
            }
        return agg

    def as_dict(self) -> dict:
        return {
            "n_users": self.n_users,
            "ndcg_at_10": self.ndcg_at_10,
            "ndcg_at_20": self.ndcg_at_20,
            "recall_at_20": self.recall_at_20,
            "mrr": self.mrr,
            "hit_at_10": self.hit_at_10,
            "precision_at_10": self.precision_at_10,
            "avg_tokens_per_user": self.avg_tokens_per_user,
            "avg_latency_ms_per_user": self.avg_latency_ms_per_user,
            "escalation_rate": self.escalation_rate,
            "tier1_parse_failure_rate": self.tier1_parse_failure_rate,
            "tier2_parse_failure_rate": self.tier2_parse_failure_rate,
            "fallback_rate": self.fallback_rate,
            "per_bin": self.per_bin,
        }
