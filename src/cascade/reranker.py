"""
LLM-based reranker for CascadeRec.

Takes a list of candidate items (from the first-stage retriever) + user
history, asks an LLM to rerank them, and parses the output into a ranked
item-id list.

Design choices for RecSys:
  - Candidate list is truncated to top-20 (not 100) because even 70B models
    struggle with 100-item context.  The first-stage top-20 Recall is the
    effective ceiling.
  - Prompt asks the model to output ONLY a numbered list of item IDs.
  - Parse is robust: tries regex, falls through to original order on failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..datasets.base import ItemRecord, UserEvalSample
from ..llm.ollama_client import ComputeBudget, LLMResponse, OllamaClient


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

RERANK_PROMPT_TEMPLATE = """You are a recommendation expert. Given a user's interaction history and a list of candidate items, rerank the candidates from MOST to LEAST relevant for this user.

## User History (most recent items):
{history_section}

## Candidate Items to Rerank:
{candidates_section}

## Instructions:
- Rank ALL candidate items from most to least relevant for this user.
- Output ONLY a numbered list of item IDs, nothing else.
- Format: one line per item, e.g.:
1. ID_123
2. ID_456
...

## Your ranking:"""


RERANK_PROMPT_WITH_CONFIDENCE = """You are a recommendation expert. Given a user's interaction history and a list of candidate items, rerank the candidates from MOST to LEAST relevant for this user.

## User History (most recent items):
{history_section}

## Candidate Items to Rerank:
{candidates_section}

## Instructions:
- Rank ALL candidate items from most to least relevant for this user.
- After the ranking, state your confidence (LOW, MEDIUM, or HIGH) in this ranking.
- Format:
1. ID_123
2. ID_456
...
CONFIDENCE: HIGH

## Your ranking:"""


# Appended for models that ignore the list-only instruction and write essays
# instead (observed: qwen3.5 burns its whole token budget on reasoning prose,
# 97% parse failure). Verified to restore compliant output on qwen3.5.
STRICT_OUTPUT_SUFFIX = (
    "\n\nIMPORTANT: Output ONLY the numbered list of item IDs in brackets. "
    "No explanation, no reasoning, no titles. Your entire response must "
    "start with '1. ' and contain nothing but the list"
)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _format_history(
    sample: UserEvalSample,
    catalog: Dict[str, ItemRecord],
    max_items: int = 10,
) -> str:
    recent = sample.history[-max_items:]
    if not recent:
        return "(No history available — new user)"
    lines = []
    for h in recent:
        iid = h.get("item_id", "")
        item = catalog.get(iid)
        rating = h.get("rating", "")
        if item:
            lines.append(f"- {item.title} [{item.category}] (rated {rating})")
        else:
            lines.append(f"- Item {iid} (rated {rating})")
    return "\n".join(lines)


def _format_candidates(
    item_ids: List[str],
    catalog: Dict[str, ItemRecord],
) -> str:
    lines = []
    for idx, iid in enumerate(item_ids, 1):
        item = catalog.get(iid)
        if item:
            lines.append(f"{idx}. [{iid}] {item.title} — {item.category}")
        else:
            lines.append(f"{idx}. [{iid}] (no metadata)")
    return "\n".join(lines)


def build_rerank_prompt(
    sample: UserEvalSample,
    candidates: List[str],
    catalog: Dict[str, ItemRecord],
    max_history: int = 10,
    with_confidence: bool = False,
    strict_output: bool = False,
) -> str:
    history_section = _format_history(sample, catalog, max_items=max_history)
    candidates_section = _format_candidates(candidates, catalog)
    template = RERANK_PROMPT_WITH_CONFIDENCE if with_confidence else RERANK_PROMPT_TEMPLATE
    prompt = template.format(
        history_section=history_section,
        candidates_section=candidates_section,
    )
    if strict_output:
        suffix = STRICT_OUTPUT_SUFFIX
        if with_confidence:
            suffix += " and the final CONFIDENCE line"
        prompt += suffix + "."
    return prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_reranked_ids(
    response_text: str,
    original_candidates: List[str],
) -> Tuple[List[str], Optional[str], bool]:
    """
    Parse numbered list of item IDs from LLM output.

    Returns:
        (ranked_ids, confidence, parse_ok) where confidence ∈ {LOW, MEDIUM,
        HIGH, None} and parse_ok=False means the numbered-list parse failed
        and we fell back to substring matching / original order.
    """
    text = response_text.strip()

    # extract confidence
    confidence: Optional[str] = None
    conf_match = re.search(r"CONFIDENCE:\s*(LOW|MEDIUM|HIGH)", text, re.IGNORECASE)
    if conf_match:
        confidence = conf_match.group(1).upper()

    # strip <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # try to find item IDs from lines like "1. ID_123" or "1. [ID_123]"
    candidate_set = set(original_candidates)
    found: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # patterns: "1. ID", "1. [ID] ...", "1) ID", "- ID"
        m = re.match(r"^\d+[\.\)]\s*\[?([^\]\s]+)\]?", line)
        if m:
            iid = m.group(1)
            if iid in candidate_set and iid not in found:
                found.append(iid)

    # If parsing failed, try to find any candidate IDs mentioned in the text
    parse_ok = len(found) >= 2
    if not parse_ok:
        found = []
        for iid in original_candidates:
            if iid in text and iid not in found:
                found.append(iid)

    # pad with remaining candidates in original order
    seen = set(found)
    for iid in original_candidates:
        if iid not in seen:
            found.append(iid)
            seen.add(iid)

    return found, confidence, parse_ok


# ---------------------------------------------------------------------------
# Reranker classes
# ---------------------------------------------------------------------------

@dataclass
class RerankerConfig:
    """Config for a single-pass LLM reranker (Tier-1 or Tier-2)."""

    model: str = "mistral-small3.2:24b"
    temperature: float = 0.5
    max_tokens: int = 512
    max_candidates: int = 20  # truncate first-stage top-N to this
    max_history: int = 10
    with_confidence: bool = False
    base_url: str = "http://localhost:11434"
    seed: Optional[int] = None  # deterministic sampling when set
    strict_output: bool = False  # harden prompt for essay-prone models (qwen)


class LLMReranker:
    """Single-pass LLM reranker. Used as Tier-1 and as T2-Large."""

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self.client = OllamaClient(
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )

    def rerank(
        self,
        sample: UserEvalSample,
        candidates: List[str],
        catalog: Dict[str, ItemRecord],
        budget: Optional[ComputeBudget] = None,
        seed: Optional[int] = None,
    ) -> Tuple[List[str], Optional[str], LLMResponse]:
        """
        Rerank candidates for a user.

        Returns:
            (ranked_ids, confidence, raw_response) — raw_response.parse_ok
            records whether the numbered-list parse succeeded.
        """
        top_n = candidates[: self.config.max_candidates]
        prompt = build_rerank_prompt(
            sample,
            top_n,
            catalog,
            max_history=self.config.max_history,
            with_confidence=self.config.with_confidence,
            strict_output=self.config.strict_output,
        )
        response = self.client.generate(prompt, seed=seed)
        if budget:
            budget.add(response)
        ranked, confidence, parse_ok = parse_reranked_ids(response.text, top_n)
        response.parse_ok = parse_ok
        return ranked, confidence, response


class SelfConsistencyReranker:
    """
    T2-SelfConsistency: run the SAME rerank prompt N times, aggregate via
    Borda count.
    """

    def __init__(self, base_config: RerankerConfig, n_samples: int = 3) -> None:
        self.base = LLMReranker(base_config)
        self.n_samples = n_samples

    def rerank(
        self,
        sample: UserEvalSample,
        candidates: List[str],
        catalog: Dict[str, ItemRecord],
        budget: Optional[ComputeBudget] = None,
    ) -> Tuple[List[str], Optional[str], List[LLMResponse]]:
        rankings: List[List[str]] = []
        responses: List[LLMResponse] = []
        base_seed = self.base.config.seed
        for i in range(self.n_samples):
            # distinct-but-deterministic seed per sample when a seed is set
            sample_seed = (base_seed + i) if base_seed is not None else None
            ranked, _, resp = self.base.rerank(
                sample, candidates, catalog, budget, seed=sample_seed
            )
            rankings.append(ranked)
            responses.append(resp)
        aggregated = _borda_aggregate(rankings, candidates[: self.base.config.max_candidates])
        return aggregated, None, responses


class MultipathReranker:
    """
    T2-Multipath: run N prompts with DIFFERENT persona strategies (inherited
    from MPPR), aggregate via Borda count.
    """

    STRATEGIES = {
        "conservative": (
            "Be conservative. Only rely on preferences clearly supported by "
            "the user's history. Avoid assumptions."
        ),
        "contrastive": (
            "Think about what this user is NOT interested in based on their "
            "history, then rank items that contrast with those dislikes."
        ),
        "analytical": (
            "Identify patterns in the user's ratings and categories. "
            "Which items fit the statistical profile of their preferences?"
        ),
    }

    def __init__(self, base_config: RerankerConfig) -> None:
        self.base_config = base_config
        self.strategies = list(self.STRATEGIES.values())

    def rerank(
        self,
        sample: UserEvalSample,
        candidates: List[str],
        catalog: Dict[str, ItemRecord],
        budget: Optional[ComputeBudget] = None,
    ) -> Tuple[List[str], Optional[str], List[LLMResponse]]:
        top_n = candidates[: self.base_config.max_candidates]
        rankings: List[List[str]] = []
        responses: List[LLMResponse] = []
        client = OllamaClient(
            model=self.base_config.model,
            base_url=self.base_config.base_url,
            temperature=self.base_config.temperature,
            max_tokens=self.base_config.max_tokens,
            seed=self.base_config.seed,
        )
        for strategy_instruction in self.strategies:
            prompt = build_rerank_prompt(
                sample, top_n, catalog, max_history=self.base_config.max_history,
                strict_output=self.base_config.strict_output,
            )
            # inject strategy at the start
            prompt = f"STRATEGY: {strategy_instruction}\n\n{prompt}"
            resp = client.generate(prompt)
            if budget:
                budget.add(resp)
            ranked, _, parse_ok = parse_reranked_ids(resp.text, top_n)
            resp.parse_ok = parse_ok
            rankings.append(ranked)
            responses.append(resp)
        aggregated = _borda_aggregate(rankings, top_n)
        return aggregated, None, responses


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _borda_aggregate(rankings: List[List[str]], all_items: List[str]) -> List[str]:
    """
    Borda count aggregation: each ranking gives n-rank points to each item.
    Higher = better.
    """
    scores: Dict[str, float] = {iid: 0.0 for iid in all_items}
    n = len(all_items)
    for ranking in rankings:
        for rank_pos, iid in enumerate(ranking):
            if iid in scores:
                scores[iid] += n - rank_pos
    sorted_items = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_items
