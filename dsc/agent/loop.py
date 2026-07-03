"""The agent main loop: a tool-use while-loop with explicit stop conditions.

One turn = the user says something, then we repeatedly:
  stream a completion → if it wants tools, run them and append results, loop →
  else, we're done.

Events are yielded out so any front-end (TUI or plain CLI) can render streaming
text, tool activity, and cost without the loop knowing about the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from ..config import Config
from ..context.compaction import KEEP_RECENT, build_summary_request
from ..context.manager import ContextManager
from ..context.tokens import CostMeter
from ..session import SessionStore
from ..tools import ToolRegistry
from .llm import Completion, DeepSeekClient, StreamDelta
from .prompts import SYSTEM_PROMPT, initial_environment


@dataclass
class LoopEvent:
    """A single thing that happened in the loop, for the UI to render."""

    kind: str  # "reasoning" | "text" | "tool_start" | "tool_end" | "notice" | "done" | "error"
    text: str = ""
    display: str = ""
    is_error: bool = False


# Naming is done by a throwaway, isolated call — its messages never enter
# self.ctx, so the main conversation's cached prefix stays untouched.
_NAMING_SYSTEM = (
    "You write terse titles for coding sessions. Given the user's first request, "
    "reply with ONLY a 2-6 word title naming the task. No quotes, no trailing "
    "punctuation, no explanation. Match the language of the request."
)


def _clean_title(raw: str) -> str:
    text = (raw or "").strip()
    first = text.splitlines()[0] if text else ""
    return first.strip().strip('"').strip("'").rstrip(".。").strip()[:48]


class AgentLoop:
    def __init__(self, config: Config, registry: ToolRegistry, cwd: str, session_name: str | None = None):
        self.config = config
        self.registry = registry
        self.client = DeepSeekClient(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            thinking=config.thinking,
        )
        self.ctx = ContextManager(SYSTEM_PROMPT, config.context_limit)
        self.meter = CostMeter(config.price())
        self._tools = self.registry.schemas()

        # Session persistence.
        self._store = SessionStore(session_name)
        # A model-generated title (sidecar file); None until the first turn of a
        # fresh session names it. On resume it's already present.
        self.title = self._store.read_title()

        if session_name:
            # Resume: load existing session messages into context.
            stored = self._store.load()
            if stored:
                self.ctx.restore(stored)
            else:
                # Fallback: fresh start.
                self.ctx.add_user(initial_environment(cwd))
                self._store.append({"role": "user", "content": initial_environment(cwd)})
        else:
            # Fresh session: seed dynamic environment as the first user turn.
            self.ctx.add_user(initial_environment(cwd))
            self._store.append({"role": "user", "content": initial_environment(cwd)})

    def send(self, user_text: str) -> Iterator[LoopEvent]:
        """Run one full user turn, yielding events until the model stops."""
        self.ctx.add_user(user_text)
        self._store.append({"role": "user", "content": user_text})

        for _ in range(self.config.max_iterations):
            yield from self._reclaim_if_needed()

            comp = yield from self._stream_once()

            # Persist the assistant message (content + any tool calls).
            assistant_msg: dict = {"role": "assistant", "content": comp.content or ""}
            if comp.tool_calls:
                assistant_msg["tool_calls"] = comp.tool_calls
            self.ctx.add_assistant(comp.content, comp.tool_calls or None)
            self._store.append(assistant_msg)
            self.meter.add(comp.cache_hit, comp.cache_miss, comp.output_tokens)

            if not comp.tool_calls:
                yield LoopEvent("done", text=comp.content)
                return

            # Execute each requested tool and append results to the tail.
            for tc in comp.tool_calls:
                fn = tc["function"]
                name, args = fn["name"], fn["arguments"]
                yield LoopEvent("tool_start", display=f"{name}({_preview(args)})")
                result = self.registry.execute(name, args)
                self.ctx.add_tool_result(tc["id"], result.content)
                self._store.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": result.content}
                )
                yield LoopEvent(
                    "tool_end",
                    text=result.content,
                    display=result.display or name,
                    is_error=result.is_error,
                )
            # Loop back: feed tool results to the model.

        yield LoopEvent(
            "notice",
            text=f"Stopped after {self.config.max_iterations} iterations (max reached).",
        )

    def generate_title(self, first_user_text: str) -> str | None:
        """Name the session with one isolated LLM call.

        Runs on a throwaway message list — nothing here is added to self.ctx, so
        the main conversation's cached prefix is never disturbed. Persists the
        result to the session's sidecar file. Safe to call from a background
        thread (writes a different file than the JSONL turn log). No-op if the
        session already has a title.
        """
        if self.title or self._store.read_title():
            return None
        messages = [
            {"role": "system", "content": _NAMING_SYSTEM},
            {"role": "user", "content": first_user_text[:600]},
        ]
        try:
            raw = self.client.complete(messages)
        except Exception:
            return None  # naming is best-effort; never break the session
        title = _clean_title(raw)
        if title:
            self.title = title
            self._store.save_title(title)
        return title or None

    # -- internals ------------------------------------------------------------

    def _stream_once(self) -> Iterator[LoopEvent]:
        """Stream one completion, yielding deltas; return the final Completion."""
        messages = self.ctx.render()
        final: Completion | None = None
        try:
            for item in self.client.stream(messages, self._tools):
                if isinstance(item, StreamDelta):
                    if item.reasoning:
                        yield LoopEvent("reasoning", text=item.reasoning)
                    if item.content:
                        yield LoopEvent("text", text=item.content)
                else:
                    final = item
        except Exception as e:  # network / API error
            yield LoopEvent("error", text=f"API error: {e}", is_error=True)
            raise
        assert final is not None
        return final

    def _reclaim_if_needed(self) -> Iterator[LoopEvent]:
        note = self.ctx.maybe_reclaim()
        if note is None:
            return
        if note != "needs_compaction":
            yield LoopEvent("notice", text=f"Context: {note}.")
            return
        # Lossless pruning wasn't enough → summarize early history.
        yield LoopEvent("notice", text="Context near limit — compacting history…")
        req = build_summary_request(self.ctx.messages, KEEP_RECENT)
        try:
            summary = self.client.complete(req)
            self.ctx.replace_history(summary, KEEP_RECENT)
            yield LoopEvent("notice", text="History compacted.")
        except Exception as e:
            yield LoopEvent("notice", text=f"Compaction failed ({e}); continuing.")


def _preview(args: str, n: int = 60) -> str:
    s = args.replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")
