"""Shared dataset types for CascadeRec.

All three dataset loaders (MovieLens, Amazon, LaMP-3) produce the same
`UserEvalSample` objects so that CascadeRec code is dataset-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


COLD_START_BINS = [1, 3, 5, 10, 20, 50]  # last bin = 50+


def cold_start_bin(history_length: int) -> str:
    """Return the bin label for a given history length."""
    for upper in COLD_START_BINS[:-1]:
        if history_length <= upper:
            return str(upper)
    return "50+"


@dataclass
class ItemRecord:
    """Item metadata record used by LLM rerankers to build prompts."""

    item_id: str
    title: str = ""
    category: str = ""
    description: str = ""
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserEvalSample:
    """
    One user evaluation instance.

    - history: list of (item_id, rating, timestamp) tuples already observed
    - candidate_items: the top-N from a first-stage retriever (to be reranked)
    - relevant_items: ground-truth positive items from the held-out split
    - cold_bin: string bin label (see `cold_start_bin`)
    """

    user_id: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    candidate_items: List[str] = field(default_factory=list)
    relevant_items: Set[str] = field(default_factory=set)
    cold_bin: str = "all"
    dataset: str = ""
    # optional task-specific fields
    input_text: str = ""  # LaMP-3 query text
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def history_length(self) -> int:
        return len(self.history)

    def with_cold_bin(self) -> "UserEvalSample":
        self.cold_bin = cold_start_bin(self.history_length)
        return self


def truncate_history(
    sample: UserEvalSample,
    k: int,
    strategy: str = "recent",
) -> UserEvalSample:
    """
    Return a *new* UserEvalSample whose history is truncated to k items.

    Used to simulate cold-start: even on datasets like MovieLens-1M where
    every user has ≥20 interactions, we can create artificial cold-start
    samples by keeping only k items per user.

    strategy:
        "recent"  — keep the k most recent items (standard)
        "random"  — keep k items uniformly at random
        "oldest"  — keep the k oldest items
    """
    import random as _random

    hist = sample.history
    if len(hist) <= k:
        new_hist = list(hist)
    elif strategy == "recent":
        new_hist = list(hist[-k:])
    elif strategy == "oldest":
        new_hist = list(hist[:k])
    elif strategy == "random":
        new_hist = _random.sample(hist, k)
        new_hist.sort(key=lambda x: x.get("timestamp", 0))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return UserEvalSample(
        user_id=f"{sample.user_id}@k{k}",
        history=new_hist,
        candidate_items=list(sample.candidate_items),
        relevant_items=set(sample.relevant_items),
        cold_bin=str(k),
        dataset=sample.dataset,
        input_text=sample.input_text,
        extra={**sample.extra, "original_user_id": sample.user_id, "truncated_to": k},
    )


def build_cold_start_eval_set(
    samples: List["UserEvalSample"],
    ks: List[int],
    max_users_per_k: Optional[int] = None,
    seed: int = 42,
) -> List["UserEvalSample"]:
    """
    From a list of "warm" user samples, build a cold-start evaluation set
    with multiple k-values.

    For each k ∈ ks, take (up to) `max_users_per_k` users whose original
    history length ≥ k+1, and emit a truncated sample. This yields a balanced
    per-bin evaluation where each bin has the same number of users.
    """
    import random as _random

    rng = _random.Random(seed)
    eligible_by_k: Dict[int, List["UserEvalSample"]] = {}
    for k in ks:
        eligible_by_k[k] = [s for s in samples if s.history_length >= k + 1]

    out: List["UserEvalSample"] = []
    for k in ks:
        pool = eligible_by_k[k]
        if max_users_per_k is not None and len(pool) > max_users_per_k:
            pool = rng.sample(pool, max_users_per_k)
        for s in pool:
            out.append(truncate_history(s, k, strategy="recent"))
    return out
