"""DeepSeek V4 client wrapper (OpenAI-compatible, streaming, tool calls).

Responsibilities:
  - Stream deltas so the TUI can render as tokens arrive.
  - Accumulate streamed tool-call fragments into complete calls.
  - Record real usage (cache hit/miss/output) for the cost meter.
  - Strip reasoning_content from what we persist (DeepSeek 400s if it's echoed
    back in a later request).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from openai import OpenAI


@dataclass
class StreamDelta:
    """One streamed chunk handed to the UI."""

    content: str = ""
    reasoning: str = ""


@dataclass
class Completion:
    """Final assembled result of a streamed completion."""

    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = ""
    # Usage for cost accounting.
    cache_hit: int = 0
    cache_miss: int = 0
    output_tokens: int = 0


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str, model: str, thinking: bool = False):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.thinking = thinking

    def stream(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> Iterator[StreamDelta | Completion]:
        """Yield StreamDelta chunks, then a final Completion as the last item."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        # tool_calls accumulate by index across chunks.
        tool_acc: dict[int, dict] = {}
        finish_reason = ""
        usage = None

        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
                yield StreamDelta(reasoning=reasoning)

            if delta.content:
                content_parts.append(delta.content)
                yield StreamDelta(content=delta.content)

            for tc in delta.tool_calls or []:
                # DeepSeek normally sends a stable integer index per tool call.
                # Fall back if a chunk ever arrives without one — otherwise every
                # index-less chunk collapses into key None and overwrites the
                # prior call, silently dropping tool calls.
                idx = tc.index if tc.index is not None else len(tool_acc)
                slot = tool_acc.setdefault(
                    idx,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["function"]["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["function"]["arguments"] += tc.function.arguments

        comp = Completion(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            tool_calls=[tool_acc[i] for i in sorted(tool_acc)],
            finish_reason=finish_reason,
        )
        if usage is not None:
            comp.output_tokens = getattr(usage, "completion_tokens", 0) or 0
            # DeepSeek-specific cache accounting.
            comp.cache_hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
            comp.cache_miss = getattr(usage, "prompt_cache_miss_tokens", 0) or 0
            if not (comp.cache_hit or comp.cache_miss):
                # Fallback when cache fields are absent.
                comp.cache_miss = getattr(usage, "prompt_tokens", 0) or 0
        yield comp

    def complete(self, messages: list[dict]) -> str:
        """Non-streaming one-off completion (used for compaction summaries)."""
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=False
        )
        return resp.choices[0].message.content or ""
