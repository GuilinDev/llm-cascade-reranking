"""
CascadeRec: Adaptive LLM Compute Allocation for Personalized Reranking.

Main orchestrator that chains:
  1. First-stage candidates (loaded from cache)
  2. Tier-1 LLM rerank (small model)
  3. Router decision (escalate or accept)
  4. Tier-2 LLM rerank (large/SC/multipath, only if escalated)
  5. Metrics + cost accounting
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..datasets.base import ItemRecord, UserEvalSample
from ..llm.ollama_client import ComputeBudget, LLMResponse
from ..metrics.ranking import AggregateMetrics, UserResult
from .reranker import (
    LLMReranker,
    MultipathReranker,
    RerankerConfig,
    SelfConsistencyReranker,
)
from .tier1_cache import Tier1Cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CascadeRec Pipeline
# ---------------------------------------------------------------------------

@dataclass
class CascadeConfig:
    """Full cascade pipeline configuration."""

    # Tier-1 (small/fast model)
    tier1_model: str = "mistral-small3.2:24b"
    tier1_max_candidates: int = 20
    tier1_max_history: int = 10
    tier1_temperature: float = 0.5
    tier1_with_confidence: bool = True

    # Tier-2 variant
    tier2_type: str = "large"  # "large" | "self_consistency" | "multipath"
    tier2_model: str = "llama3.1:70b"
    tier2_n_samples: int = 3  # for SC and multipath

    # Router
    router_type: str = "heuristic"  # "heuristic" | "self_confidence" | "logistic" | "llm_coordinator"
    router_threshold: int = 5  # for heuristic: history_len <= threshold → escalate

    # General
    base_url: str = "http://localhost:11434"
    max_tokens: int = 512
    output_top_k: int = 10
    seed: Optional[int] = None  # deterministic LLM sampling when set
    strict_output: bool = False  # harden prompts (essay-prone models, e.g. qwen)
    # Path to a shared Tier-1 output cache. All methods reading the same
    # cache see identical Tier-1 rankings (fixes RecSys R1-W6) and skip
    # redundant Tier-1 LLM calls.
    tier1_cache_path: Optional[str] = None


class CascadeRec:
    """Main CascadeRec pipeline."""

    def __init__(self, config: CascadeConfig) -> None:
        self.config = config
        self.budget = ComputeBudget()

        # Build Tier-1 reranker
        self.tier1 = LLMReranker(
            RerankerConfig(
                model=config.tier1_model,
                temperature=config.tier1_temperature,
                max_tokens=config.max_tokens,
                max_candidates=config.tier1_max_candidates,
                max_history=config.tier1_max_history,
                with_confidence=config.tier1_with_confidence,
                base_url=config.base_url,
                seed=config.seed,
                strict_output=config.strict_output,
            )
        )
        self.tier1_cache = Tier1Cache(config.tier1_cache_path) if config.tier1_cache_path else None

        # Build Tier-2 reranker
        t2_config = RerankerConfig(
            model=config.tier2_model,
            temperature=config.tier1_temperature,
            max_tokens=config.max_tokens,
            max_candidates=config.tier1_max_candidates,
            max_history=config.tier1_max_history,
            with_confidence=False,
            base_url=config.base_url,
            seed=config.seed,
            strict_output=config.strict_output,
        )
        if config.tier2_type == "large":
            self.tier2 = LLMReranker(t2_config)
        elif config.tier2_type == "self_consistency":
            self.tier2 = SelfConsistencyReranker(t2_config, n_samples=config.tier2_n_samples)
        elif config.tier2_type == "multipath":
            self.tier2 = MultipathReranker(t2_config)
        else:
            raise ValueError(f"Unknown tier2_type: {config.tier2_type}")

        # Build Router
        self.router = self._build_router()

    def _build_router(self):
        from ..routers.router import (
            HeuristicRouter,
            LLMCoordinatorRouter,
            LogisticRouter,
            SelfConfidenceRouter,
        )

        rt = self.config.router_type
        if rt == "heuristic":
            return HeuristicRouter(threshold=self.config.router_threshold)
        if rt == "self_confidence":
            return SelfConfidenceRouter(escalate_on="LOW")
        if rt == "logistic":
            return LogisticRouter(threshold=0.5)
        if rt == "llm_coordinator":
            return LLMCoordinatorRouter(
                model=self.config.tier1_model,
                base_url=self.config.base_url,
            )
        raise ValueError(f"Unknown router_type: {rt}")

    def rerank_user(
        self,
        sample: UserEvalSample,
        candidates: List[str],
        catalog: Dict[str, ItemRecord],
    ) -> UserResult:
        """Run the full cascade for a single user."""
        user_budget = ComputeBudget()

        # --- Tier-1 (cache-first) ---
        cached = (
            self.tier1_cache.get_response(sample.user_id)
            if self.tier1_cache is not None
            else None
        )
        if cached is not None:
            tier1_ranking, confidence, tier1_resp = cached
            user_budget.add(tier1_resp)
        else:
            tier1_ranking, confidence, tier1_resp = self.tier1.rerank(
                sample, candidates, catalog, budget=user_budget
            )
            if self.tier1_cache is not None:
                self.tier1_cache.put(sample.user_id, tier1_ranking, confidence, tier1_resp)

        # --- Router ---
        escalate = self.router.should_escalate(
            sample, tier1_ranking, confidence, budget=user_budget
        )

        # --- Tier-2 (conditional) ---
        tier2_parse_ok = None
        if escalate:
            tier2_out = self.tier2.rerank(
                sample, candidates, catalog, budget=user_budget
            )
            final_ranking = tier2_out[0]
            # tier2_out[2] is one LLMResponse (T2-Large) or a list (SC/MP)
            t2_resp = tier2_out[2]
            if isinstance(t2_resp, list):
                oks = [r.parse_ok for r in t2_resp if r.parse_ok is not None]
                tier2_parse_ok = all(oks) if oks else None
            else:
                tier2_parse_ok = t2_resp.parse_ok
        else:
            final_ranking = tier1_ranking

        # --- Build UserResult ---
        result = UserResult(
            user_id=sample.user_id,
            ranked=final_ranking[: self.config.output_top_k],
            relevant=sample.relevant_items,
            cold_bin=sample.cold_bin,
            total_tokens=user_budget.total_prompt_tokens + user_budget.total_completion_tokens,
            total_latency_ms=user_budget.total_latency_ms,
            escalated=escalate,
            tier1_parse_ok=tier1_resp.parse_ok,
            tier2_parse_ok=tier2_parse_ok,
        )
        result.compute()

        # accumulate global budget
        self.budget.add_bulk(user_budget)

        return result

    def run_experiment(
        self,
        samples: List[UserEvalSample],
        candidates_map: Dict[str, List[str]],
        catalog: Dict[str, ItemRecord],
        show_progress: bool = True,
    ) -> List[UserResult]:
        """Run cascade on all users. Returns list of UserResult."""
        from tqdm import tqdm

        results: List[UserResult] = []
        errors = 0
        iterator = tqdm(samples, desc="CascadeRec") if show_progress else samples
        for sample in iterator:
            cands = candidates_map.get(sample.user_id, [])
            if not cands:
                logger.warning(f"No candidates for user {sample.user_id}")
                continue
            try:
                result = self.rerank_user(sample, cands, catalog)
                results.append(result)
            except Exception as exc:
                errors += 1
                logger.warning(f"User {sample.user_id} failed: {exc}")
                # Fallback: use first-stage ranking as-is
                fallback = UserResult(
                    user_id=sample.user_id,
                    ranked=cands[:self.config.output_top_k],
                    relevant=sample.relevant_items,
                    cold_bin=sample.cold_bin,
                    fallback=True,
                )
                fallback.compute()
                results.append(fallback)
        if errors:
            logger.warning(f"{errors}/{len(samples)} users had errors (fell back to first-stage)")
        if self.tier1_cache is not None:
            self.tier1_cache.flush()
        return results


# Patch ComputeBudget to support bulk add
def _add_bulk(self, other: ComputeBudget) -> None:
    self.total_calls += other.total_calls
    self.total_prompt_tokens += other.total_prompt_tokens
    self.total_completion_tokens += other.total_completion_tokens
    self.total_latency_ms += other.total_latency_ms
    for model, stats in other.by_model.items():
        mine = self.by_model.setdefault(
            model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}
        )
        for k, v in stats.items():
            mine[k] += v


ComputeBudget.add_bulk = _add_bulk


# ---------------------------------------------------------------------------
# Baseline wrappers (always-small, always-large, no-rerank)
# ---------------------------------------------------------------------------

class AlwaysSmallBaseline:
    """B1: always use Tier-1 (small model), never escalate."""

    def __init__(self, config: CascadeConfig) -> None:
        cfg = CascadeConfig(
            tier1_model=config.tier1_model,
            tier1_max_candidates=config.tier1_max_candidates,
            tier1_max_history=config.tier1_max_history,
            tier1_temperature=config.tier1_temperature,
            # Same prompt as the cascade's Tier-1 (confidence line included,
            # then ignored) so both share one Tier-1 cache and non-escalated
            # cascade users match B1 exactly.
            tier1_with_confidence=True,
            router_type="heuristic",
            router_threshold=-1,  # never escalate (history_len >= 0 always)
            base_url=config.base_url,
            max_tokens=config.max_tokens,
            output_top_k=config.output_top_k,
            seed=config.seed,
            strict_output=config.strict_output,
            tier1_cache_path=config.tier1_cache_path,
        )
        self.cascade = CascadeRec(cfg)

    def run_experiment(self, *args, **kwargs):
        return self.cascade.run_experiment(*args, **kwargs)


class AlwaysLargeBaseline:
    """B2: always use Tier-2 (large model) for every user."""

    def __init__(self, config: CascadeConfig) -> None:
        cfg = CascadeConfig(
            tier1_model=config.tier2_model,  # tier-1 IS the large model
            tier1_max_candidates=config.tier1_max_candidates,
            tier1_max_history=config.tier1_max_history,
            tier1_temperature=config.tier1_temperature,
            tier1_with_confidence=False,
            router_type="heuristic",
            router_threshold=-1,
            base_url=config.base_url,
            max_tokens=config.max_tokens,
            output_top_k=config.output_top_k,
            seed=config.seed,
            tier1_cache_path=config.tier1_cache_path,
        )
        self.cascade = CascadeRec(cfg)

    def run_experiment(self, *args, **kwargs):
        return self.cascade.run_experiment(*args, **kwargs)


class NoRerankBaseline:
    """B0: just return first-stage ranking as-is. Zero LLM calls."""

    def run_experiment(
        self,
        samples: List[UserEvalSample],
        candidates_map: Dict[str, List[str]],
        catalog: Dict[str, ItemRecord],
        **kwargs,
    ) -> List[UserResult]:
        results = []
        for sample in samples:
            cands = candidates_map.get(sample.user_id, [])
            result = UserResult(
                user_id=sample.user_id,
                ranked=cands[:10],
                relevant=sample.relevant_items,
                cold_bin=sample.cold_bin,
            )
            result.compute()
            results.append(result)
        return results
