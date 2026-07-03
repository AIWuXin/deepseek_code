"""The message store that keeps DeepSeek's prefix cache warm.

Cache rule (https://api-docs.deepseek.com/guides/kv_cache): only an input
*prefix* that matches byte-for-byte a previously seen prefix is billed at the
cheap cache-hit rate (~98% cheaper). So the golden rule of this whole project:

    NEVER mutate or reorder anything but the tail of the message list.

Layout, front to back:
    [system]            fixed, byte-stable
    [ ...history... ]   append-only
    [new user input]    tail

When we must reclaim space we do it in two escalating stages, both of which
touch only *old* messages far from the tail:
    1. Lossless pruning: replace bulky old tool-result bodies with a stub.
       (Cheap; the stub still marks that the call happened.)
    2. Compaction: summarize the early history into one block and drop it.
       (Lossy; only when pruning isn't enough — see compaction.py.)

Note both stages *do* break the cached prefix for one turn — that's the
unavoidable cost of reclaiming space. We delay them as long as possible so the
prefix stays stable across the majority of turns.
"""

from __future__ import annotations

from .tokens import estimate_messages

# Tool results older than this many messages from the tail are eligible for
# lossless pruning once we cross the soft budget.
PRUNE_KEEP_RECENT = 8
PRUNE_MIN_CHARS = 1500  # only prune results big enough to be worth it
STUB = "[old tool result cleared to save context]"


class ContextManager:
    def __init__(self, system_prompt: str, limit: int):
        self.limit = limit
        # System message is the stable head of the prefix. Never touched.
        self._system = {"role": "system", "content": system_prompt}
        self._messages: list[dict] = []

    # -- message construction -------------------------------------------------

    def add_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_assistant(self, content: str, tool_calls: list[dict] | None = None) -> None:
        # reasoning_content is intentionally dropped: DeepSeek returns 400 if it
        # appears in a subsequent request's messages.
        msg: dict = {"role": "assistant", "content": content or ""}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self._messages.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )

    # -- rendering ------------------------------------------------------------

    def render(self) -> list[dict]:
        """Full message list for the API: stable system head + history tail."""
        return [self._system, *self._messages]

    def estimated_tokens(self) -> int:
        return estimate_messages(self.render())

    # -- reclamation ----------------------------------------------------------

    def maybe_reclaim(self) -> str | None:
        """Reclaim space if over budget. Returns a note if action was taken."""
        if self.estimated_tokens() <= self.limit:
            return None
        pruned = self._prune_old_tool_results()
        if self.estimated_tokens() <= self.limit:
            if pruned:
                return f"pruned {pruned} old tool result(s)"
            return None
        # Still over budget after pruning → caller should compact.
        return "needs_compaction"

    def _prune_old_tool_results(self) -> int:
        """Replace bulky old tool-result bodies with a stub. Lossless-ish."""
        cutoff = len(self._messages) - PRUNE_KEEP_RECENT
        pruned = 0
        for i in range(max(0, cutoff)):
            m = self._messages[i]
            if m.get("role") == "tool" and m.get("content") not in (STUB, None):
                if len(m["content"]) >= PRUNE_MIN_CHARS:
                    m["content"] = STUB
                    pruned += 1
        return pruned

    def replace_history(self, summary: str, keep_recent: int) -> None:
        """Compaction hook: swap early history for a summary, keep recent tail."""
        tail = self._messages[-keep_recent:] if keep_recent else []
        summary_msg = {
            "role": "user",
            "content": f"[Summary of earlier conversation]\n{summary}",
        }
        self._messages = [summary_msg, *tail]

    # -- accessors for compaction ---------------------------------------------

    def restore(self, messages: list[dict]) -> None:
        """Replace all history with loaded session messages."""
        self._messages = list(messages)

    @property
    def messages(self) -> list[dict]:
        return self._messages
