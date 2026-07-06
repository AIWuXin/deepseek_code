"""Lightweight token estimation and cost accounting.

We don't ship a tokenizer — a cheap char-based heuristic is enough to decide
when to compact and to drive the on-screen meter. Exact accounting comes from
the API's usage fields (prompt_cache_hit_tokens / prompt_cache_miss_tokens),
which the LLM client records after each call.
"""

from __future__ import annotations

import json

# Chars per token heuristic.  DeepSeek V4 / GPT-5 tokenizers average:
#   English prose         ~4.0
#   Chinese text          ~1.5–2.0
#   code                  ~3.5
#   JSON / structured     ~2.5–3.0
# We use a conservative 2.8 so we *over*-estimate for mixed workloads,
# which is the safe direction for a compaction trigger — better to compact
# a tiny bit early than to hit the API's hard context limit.
CHARS_PER_TOKEN = 2.8


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def estimate_messages(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            total += estimate_tokens(json.dumps(content, ensure_ascii=False))
        # Tool calls carried on assistant messages.
        for tc in m.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            total += estimate_tokens(fn.get("name", "") + fn.get("arguments", ""))
        total += 4  # per-message role/format overhead
    return total


class CostMeter:
    """Accumulates spend from real API usage fields."""

    def __init__(self, price: dict):
        self.price = price
        self.cache_hit = 0
        self.cache_miss = 0
        self.output = 0

    def add(self, hit: int, miss: int, out: int) -> None:
        self.cache_hit += hit
        self.cache_miss += miss
        self.output += out

    @property
    def usd(self) -> float:
        p = self.price
        return (
            self.cache_hit / 1e6 * p["cache_hit"]
            + self.cache_miss / 1e6 * p["cache_miss"]
            + self.output / 1e6 * p["output"]
        )

    @property
    def hit_rate(self) -> float:
        total_in = self.cache_hit + self.cache_miss
        return self.cache_hit / total_in if total_in else 0.0
