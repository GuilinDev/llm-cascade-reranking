"""
Tier-1 output cache for CascadeRec.

Motivation (RecSys 2026 R1-W6): Tier-1 was rerun stochastically for every
experiment, so non-escalated users in a cascade run did not match the
Always-Small baseline exactly. Caching the Tier-1 pass per
(dataset, model, seed, max_candidates) guarantees that every method
consuming the same cache sees identical Tier-1 rankings — and cuts GPU
cost, since all cascade variants / routers reuse one Tier-1 pass.

Cache file: one JSON dict keyed by user_id. Only successful LLM responses
are cached; errors/timeouts are retried on the next run.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from ..llm.ollama_client import LLMResponse


def default_tier1_cache_path(
    dataset: str, model: str, seed: int, max_candidates: int,
    cache_dir: str = "data/cache/tier1",
    variant: str = "conf",
) -> str:
    # `variant` distinguishes prompt variants (with/without the confidence
    # line) so runs with different prompts never share cache entries.
    safe_model = model.replace(":", "_").replace("/", "_")
    return os.path.join(
        cache_dir,
        f"{dataset}_{safe_model}_seed{seed}_c{max_candidates}_{variant}.json",
    )


class Tier1Cache:
    """In-memory dict backed by a JSON file, flushed periodically."""

    FLUSH_EVERY = 20

    def __init__(self, path: str) -> None:
        self.path = path
        self._data: Dict[str, dict] = {}
        self._pending = 0
        if os.path.isfile(path):
            with open(path) as f:
                self._data = json.load(f)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, user_id: str) -> Optional[dict]:
        return self._data.get(user_id)

    def get_response(self, user_id: str) -> Optional[tuple]:
        """
        Returns (ranked, confidence, LLMResponse) rebuilt from the cache,
        or None on miss. The replayed response carries the original token
        and latency cost so budget accounting stays truthful.
        """
        entry = self._data.get(user_id)
        if entry is None:
            return None
        response = LLMResponse(
            text="[CACHED]",
            model=entry["model"],
            prompt_tokens=entry["prompt_tokens"],
            completion_tokens=entry["completion_tokens"],
            latency_ms=entry["latency_ms"],
            parse_ok=entry.get("parse_ok"),
            cached=True,
        )
        return entry["ranked"], entry.get("confidence"), response

    def put(
        self,
        user_id: str,
        ranked: List[str],
        confidence: Optional[str],
        response: LLMResponse,
    ) -> None:
        if not response.is_ok:
            return  # never freeze errors/timeouts into the cache
        self._data[user_id] = {
            "ranked": ranked,
            "confidence": confidence,
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "latency_ms": response.latency_ms,
            "parse_ok": response.parse_ok,
        }
        self._pending += 1
        if self._pending >= self.FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        if self._pending == 0 and os.path.isfile(self.path):
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._data, f)
        os.replace(tmp, self.path)
        self._pending = 0
