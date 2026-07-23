"""
Router variants for CascadeRec.

Each router takes a UserEvalSample + Tier-1 output and decides:
  escalate=True  → send to Tier-2 (expensive)
  escalate=False → accept Tier-1 output (cheap)

Router protocol:
    .should_escalate(sample, tier1_ranking, confidence, budget) -> bool
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..datasets.base import UserEvalSample
from ..llm.ollama_client import ComputeBudget, LLMResponse, OllamaClient


# ---------------------------------------------------------------------------
# R0: Heuristic (history length threshold)
# ---------------------------------------------------------------------------

class HeuristicRouter:
    """
    R0: escalate if user history length <= threshold.

    Cheapest possible router — zero LLM calls, zero latency.
    This is the "what if we just used a simple rule?" ablation baseline.
    """

    def __init__(self, threshold: int = 5) -> None:
        self.threshold = threshold

    def should_escalate(
        self,
        sample: UserEvalSample,
        tier1_ranking: List[str],
        confidence: Optional[str] = None,
        budget: Optional[ComputeBudget] = None,
    ) -> bool:
        k = sample.extra.get("truncated_to", sample.history_length)
        return k <= self.threshold


# ---------------------------------------------------------------------------
# R1: Self-confidence (parse Tier-1's verbalized confidence)
# ---------------------------------------------------------------------------

class SelfConfidenceRouter:
    """
    R1: escalate if Tier-1 reports LOW confidence.

    Requires Tier-1 to output CONFIDENCE: {LOW|MEDIUM|HIGH}.
    Zero additional LLM calls — just parses what Tier-1 already generated.
    """

    def __init__(self, escalate_on: str = "LOW") -> None:
        self.escalate_on = {s.strip().upper() for s in escalate_on.split(",")}

    def should_escalate(
        self,
        sample: UserEvalSample,
        tier1_ranking: List[str],
        confidence: Optional[str] = None,
        budget: Optional[ComputeBudget] = None,
    ) -> bool:
        if confidence is None:
            return True  # no confidence → assume uncertain → escalate
        return confidence.upper() in self.escalate_on


# ---------------------------------------------------------------------------
# R2: Logistic router (trained on held-out split)
# ---------------------------------------------------------------------------

class LogisticRouter:
    """
    R2: lightweight logistic regression on cheap features.

    Features:
      - history_length (int)
      - n_unique_categories_in_history (int)
      - avg_rating_in_history (float)
      - tier1_rank_entropy (float, higher = more uncertain ordering)

    Training: on a held-out set where we know the oracle escalation label
    (escalate iff tier2 NDCG > tier1 NDCG for this user). See
    scripts/train_router.py.

    At test time: predict P(escalate) and threshold at τ=0.5.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    def _features(self, sample: UserEvalSample, tier1_ranking: List[str]) -> np.ndarray:
        history = sample.history
        h_len = len(history)
        cats = set(h.get("category", h.get("item_id", "")) for h in history)
        n_cats = len(cats)
        ratings = [h.get("rating", 0) for h in history if h.get("rating")]
        avg_rating = np.mean(ratings) if ratings else 0.0
        # rank entropy: how "spread out" is the ranking? (proxy for uncertainty)
        n = len(tier1_ranking)
        rank_entropy = np.log(max(n, 1))  # placeholder — real entropy needs logprobs
        return np.array([h_len, n_cats, avg_rating, rank_entropy], dtype=np.float32)

    def fit(self, features_matrix: np.ndarray, labels: np.ndarray) -> None:
        """Fit logistic regression (sklearn-free to avoid extra dependency)."""
        # simple gradient descent logistic regression
        n_features = features_matrix.shape[1]
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0
        lr = 0.01
        for _ in range(500):
            z = features_matrix @ w + b
            pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            error = pred - labels
            w -= lr * (features_matrix.T @ error) / len(labels)
            b -= lr * error.mean()
        self.weights = w.astype(np.float32)
        self.bias = float(b)

    def should_escalate(
        self,
        sample: UserEvalSample,
        tier1_ranking: List[str],
        confidence: Optional[str] = None,
        budget: Optional[ComputeBudget] = None,
    ) -> bool:
        if self.weights is None:
            return True  # not trained yet → always escalate
        feat = self._features(sample, tier1_ranking)
        z = float(feat @ self.weights + self.bias)
        prob = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return prob >= self.threshold


# ---------------------------------------------------------------------------
# R3: LLM Coordinator (adapted from Paper A's DualAgent-Rec)
# ---------------------------------------------------------------------------

class LLMCoordinatorRouter:
    """
    R3: an LLM reads user stats + Tier-1 summary and decides whether
    to escalate to Tier-2.

    Costs one additional LLM call (on the small model), but can make
    more nuanced decisions than R0/R1/R2.
    """

    COORDINATOR_PROMPT = """You are a recommendation system router. Your job is to decide whether a user needs a more powerful recommendation model.

## User Profile:
- History length: {history_length} items
- Categories in history: {categories}
- Average rating: {avg_rating:.1f}

## Tier-1 Recommendation Quality Signal:
- Tier-1 model confidence: {confidence}
- Number of candidates considered: {n_candidates}

## Decision Criteria:
- If the user has very few interactions (1-3 items), the basic model likely struggles → ESCALATE
- If the user's interests span many categories, the basic model may miss nuances → ESCALATE
- If Tier-1 reported LOW confidence → ESCALATE
- If the user has clear, focused preferences and Tier-1 is confident → ACCEPT

## Your decision (respond with exactly one word: ESCALATE or ACCEPT):"""

    def __init__(
        self,
        model: str = "mistral-small3.2:24b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.client = OllamaClient(
            model=model,
            base_url=base_url,
            temperature=0.1,  # deterministic routing
            max_tokens=20,
        )

    def should_escalate(
        self,
        sample: UserEvalSample,
        tier1_ranking: List[str],
        confidence: Optional[str] = None,
        budget: Optional[ComputeBudget] = None,
    ) -> bool:
        history = sample.history
        cats = list(set(h.get("category", h.get("item_id", "")) for h in history))
        ratings = [h.get("rating", 0) for h in history if h.get("rating")]
        avg_rating = float(np.mean(ratings)) if ratings else 0.0

        prompt = self.COORDINATOR_PROMPT.format(
            history_length=len(history),
            categories=", ".join(cats[:10]),
            avg_rating=avg_rating,
            confidence=confidence or "UNKNOWN",
            n_candidates=len(tier1_ranking),
        )
        resp = self.client.generate(prompt)
        if budget:
            budget.add(resp)

        text = resp.text.strip().upper()
        # strip think tags
        text = re.sub(r"<THINK>.*?</THINK>", "", text, flags=re.DOTALL).strip()
        return "ESCALATE" in text
