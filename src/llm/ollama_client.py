"""
Ollama LLM client for CascadeRec.

Ported from mppr-personalization/src/baselines/llm_inference.py.
Adapted to track compute cost (tokens + latency) explicitly so that
CascadeRec can compute Pareto curves of NDCG vs cost.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional

import requests
from tqdm import tqdm


@dataclass
class LLMResponse:
    """One LLM call result, with cost tracking for cascade accounting."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None
    # set post-hoc by rerankers: did the numbered-list parse succeed?
    parse_ok: Optional[bool] = None
    # set when the response was replayed from a Tier-1 cache
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def is_ok(self) -> bool:
        return self.error is None and not self.text.startswith("[ERROR")


@dataclass
class ComputeBudget:
    """Running total of LLM compute spent during an experiment."""

    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: float = 0.0
    by_model: dict = field(default_factory=dict)

    def add(self, response: LLMResponse) -> None:
        self.total_calls += 1
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        self.total_latency_ms += response.latency_ms
        model_stats = self.by_model.setdefault(
            response.model,
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0},
        )
        model_stats["calls"] += 1
        model_stats["prompt_tokens"] += response.prompt_tokens
        model_stats["completion_tokens"] += response.completion_tokens
        model_stats["latency_ms"] += response.latency_ms

    def as_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_latency_ms": self.total_latency_ms,
            "by_model": self.by_model,
        }


class OllamaClient:
    """Client for local Ollama LLM inference with robust timeout + retry."""

    def __init__(
        self,
        model: str = "qwen2.5:14b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.5,
        max_tokens: int = 512,
        timeout: int = 60,
        max_retries: int = 2,
        retry_delay: float = 3.0,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.seed = seed

    def generate(self, prompt: str, seed: Optional[int] = None) -> LLMResponse:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        for attempt in range(self.max_retries + 1):
            start_time = time.time()
            try:
                # Hard wall-clock timeout via thread — catches ALL hang modes
                pool = ThreadPoolExecutor(max_workers=1)
                future = pool.submit(self._generate_streaming, prompt, seed)
                try:
                    result = future.result(timeout=self.timeout)
                finally:
                    pool.shutdown(wait=False)
                if result.is_ok:
                    return result
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    continue
                return result
            except FuturesTimeout:
                # Don't retry hard timeouts — same prompt will hang again
                return LLMResponse(
                    text="[TIMEOUT: hard wall clock]",
                    model=self.model,
                    latency_ms=(time.time() - start_time) * 1000,
                    error="hard_timeout",
                )
            except Exception as exc:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    continue
                return LLMResponse(
                    text=f"[ERROR: {exc}]",
                    model=self.model,
                    latency_ms=(time.time() - start_time) * 1000,
                    error=str(exc),
                )

    def _generate_streaming(self, prompt: str, seed: Optional[int] = None) -> LLMResponse:
        """
        Stream response from Ollama with per-chunk timeout.

        This catches the case where Ollama accepts the connection and starts
        generating, then hangs mid-generation — which a plain `timeout`
        parameter on requests.post() does NOT catch.
        """
        import json as _json

        start_time = time.time()
        chunks: list = []
        prompt_tokens = 0
        completion_tokens = 0

        # timeout=(connect, read_between_bytes): if no data for 30s, abort.
        # This is the REAL fix for Ollama hanging mid-generation.
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    **(
                        {"seed": seed if seed is not None else self.seed}
                        if (seed is not None or self.seed is not None)
                        else {}
                    ),
                },
                "think": False,
                "keep_alive": "10m",
            },
            timeout=(10, 30),  # 10s connect, 30s between ANY bytes
            stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            # Wall-clock hard limit
            if time.time() - start_time > self.timeout:
                response.close()
                return LLMResponse(
                    text="".join(chunks).strip() or "[TIMEOUT: wall clock]",
                    model=self.model,
                    latency_ms=(time.time() - start_time) * 1000,
                    error="wall_timeout",
                )
            if not line:
                continue
            try:
                data = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            token_text = data.get("response", "")
            if token_text:
                chunks.append(token_text)
            if data.get("done", False):
                prompt_tokens = data.get("prompt_eval_count", 0)
                completion_tokens = data.get("eval_count", 0)
                break

        latency = (time.time() - start_time) * 1000
        full_text = "".join(chunks).strip()
        return LLMResponse(
            text=full_text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency,
        )

    def generate_batch(
        self,
        prompts: List[str],
        max_workers: int = 1,
        show_progress: bool = True,
    ) -> List[LLMResponse]:
        results: List[Optional[LLMResponse]] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.generate, prompt): idx
                for idx, prompt in enumerate(prompts)
            }
            iterator = as_completed(future_to_idx)
            if show_progress:
                iterator = tqdm(iterator, total=len(prompts), desc=f"LLM:{self.model}")
            for future in iterator:
                idx = future_to_idx[future]
                results[idx] = future.result()
        return [r for r in results if r is not None]

    def check_health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            models = response.json().get("models", [])
            names = [m.get("name", "") for m in models]
            base = self.model.split(":")[0]
            return self.model in names or any(base in m for m in names)
        except Exception:
            return False


class OpenAIClient:
    """OpenAI fallback (only used if Ollama unavailable)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")

    def generate(self, prompt: str) -> LLMResponse:
        import openai  # lazy import

        client = openai.OpenAI(api_key=self.api_key)
        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            latency = (time.time() - start_time) * 1000
            return LLMResponse(
                text=response.choices[0].message.content.strip(),
                model=self.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                latency_ms=latency,
            )
        except Exception as exc:
            return LLMResponse(
                text=f"[ERROR: {exc}]",
                model=self.model,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(exc),
            )


def get_llm_client(
    provider: str = "ollama",
    model: Optional[str] = None,
    **kwargs,
):
    """Factory for LLM clients."""
    if provider == "ollama":
        model = model or "qwen2.5:14b"
        return OllamaClient(model=model, **kwargs)
    if provider == "openai":
        model = model or "gpt-4o-mini"
        return OpenAIClient(model=model, **kwargs)
    raise ValueError(f"Unknown provider: {provider}")
