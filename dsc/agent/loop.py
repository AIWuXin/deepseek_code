"""The agent main loop: a tool-use while-loop with explicit stop conditions.

One turn = the user says something, then we repeatedly:
  stream a completion → if it wants tools, run them and append results, loop →
  else, we're done.

Events are yielded out so any front-end (TUI or plain CLI) can render streaming
text, tool activity, and cost without the loop knowing about the UI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

from ..config import Config
from ..context.compaction import build_summary_request, format_compact_summary, _flatten
from ..context.manager import ContextManager
from ..context.repair import repair_dangling_tool_calls
from ..context.tokens import CostMeter
from ..session import SessionStore
from ..session.store import ArchiveBlock
from ..tools import ToolRegistry, ToolResult
from .llm import Completion, DeepSeekClient, StreamDelta
from .prompts import SYSTEM_PROMPT, initial_environment
from ..debug import log


@dataclass
class LoopEvent:
    """A single thing that happened in the loop, for the UI to render."""

    kind: str  # "reasoning" | "text" | "tool_start" | "tool_end" | "notice" | "done" | "error"
    text: str = ""
    display: str = ""
    is_error: bool = False


# Archive heuristic: a turn shorter than this isn't worth a sub-agent call.
# A task that produced a real artifact (file edit + tool result + summary)
# is almost always > 800 chars; pure queries and quick clarifications are
# usually < 500. This is a cheap pre-filter, not a semantic judgment.
_ARCHIVE_MIN_CHARS = 800

# Naming is done by a throwaway, isolated call — its messages never enter
# self.ctx, so the main conversation's cached prefix stays untouched.
_NAMING_SYSTEM = (
    "You write terse titles for coding sessions. Given the user's first request, "
    "reply with ONLY a 2-6 word title naming the task. No quotes, no trailing "
    "punctuation, no explanation. Match the language of the request."
)

# Isolated sub-agent for task archiving.
_ARCHIVE_SYSTEM = (
    "You decide whether a completed turn should be archived and, if so, "
    "produce its metadata. Given the conversation below, output a JSON object "
    "with exactly these four fields:\n"
    '- "archive": boolean. true ONLY if this turn produced a lasting artifact '
    "(a modified/created file, a meaningful command run, a concrete problem "
    "resolved). Return false for pure queries, chit-chat, clarifications, or "
    "trivial exchanges even if a tool was called.\n"
    '- "summary": one-line summary (max 80 chars)\n'
    '- "keywords": comma-separated search terms (max 10, include filenames, '
    "symbols, tech terms)\n"
    '- "in_context_summary": 2-3 sentence recap (max 300 chars) that lets the '
    "agent continue working without the original transcript.\n"
    "Only output the JSON object, nothing else.")

# Isolated sub-agent for compressing read_archive results.
_READ_COMPRESS_SYSTEM = (
    "You are compressing a retrieved archive block for a coding agent. "
    "The content below is the full conversation transcript of a past task. "
    "Extract ONLY the key information that the agent needs to continue working: "
    "decisions made, files changed, conclusions reached, and any unresolved items. "
    "Omit chit-chat, raw tool output dumps, and repetitive debugging steps. "
    "Return a concise summary (max 500 characters) that captures the essence. "
    "Only output the summary, nothing else.")


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

        # Wire up the archive dir for ReadArchiveTool (session name is now known).
        arc_tool = self.registry.get("read_archive")
        if arc_tool:
            arc_tool.set_archive_root(self._store.name)

        # V2 archive state (Phase 1).
        # Seed from disk so resume doesn't overwrite existing blocks —
        # archive_block() writes {id:04d}.json unconditionally.
        existing = self._store.list_blocks()
        self._archive_next_id: int = max((b["id"] for b in existing), default=-1) + 1
        # Index in ctx.messages where the current user turn started; set by send().
        self._turn_start_idx: int = 0

        if session_name:
            # Resume: load existing session messages into context.
            stored = self._store.load()
            if stored:
                # Repair any dangling tool_calls left by an earlier crash so the
                # session is valid for the API again; persist the fix.
                stored, repaired = repair_dangling_tool_calls(stored)
                if repaired:
                    self._store.replace(stored)
                    log("resume: repaired dangling tool_calls in session")
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
        self._turn_start_idx = len(self.ctx.messages)
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
                self._maybe_archive_turn(comp.content)
                return

            # Execute each requested tool and append results to the tail.
            for tc in comp.tool_calls:
                fn = tc["function"]
                name, args = fn["name"], fn["arguments"]
                log(f"loop: yield tool_start name={name} id={tc.get('id')}")
                yield LoopEvent("tool_start", display=f"{name}({_preview(args)})")
                # Everything from execute through post-processing is wrapped so a
                # bug here can NEVER skip add_tool_result below — an assistant
                # message with tool_calls that isn't answered by a tool message
                # corrupts the session (DeepSeek 400s on the next request).
                try:
                    result = self.registry.execute(name, args)
                    log(f"loop: executed {name} error={result.is_error} bytes={len(result.content)}")

                    # Phase 2: compress read_archive(id) results via sub-agent.
                    if name == "read_archive" and not result.is_error:
                        compressed = self._compress_read_archive(result.content)
                        if compressed is not None:
                            result = ToolResult(
                                content=compressed,
                                display=result.display,
                                is_error=False,
                            )
                            log(f"loop: compressed read_archive result to {len(compressed)} bytes")
                except Exception as e:
                    log(f"loop: tool handler crashed for {name}: {e!r}")
                    result = ToolResult(content=f"tool handler error: {e}", is_error=True)

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

    # -- V2 archive (Phase 1) -------------------------------------------------

    def _compress_read_archive(self, content: str) -> str | None:
        """Sub-agent: compress a read_archive result to save context space.

        Called in the same iteration, before the model sees the result.
        Returns the compressed string or ``None`` on failure (fall back to raw).
        """
        if len(content) < 2000:
            return content  # small enough, no compression needed
        req = [
            {"role": "system", "content": _READ_COMPRESS_SYSTEM},
            {"role": "user", "content": content},
        ]
        try:
            compressed = self.client.complete(req)
        except Exception:
            return None
        if compressed:
            return compressed.strip()[:500]
        return None

    def _maybe_archive_turn(self, assistant_text: str) -> None:
        """Archive this turn if it represents a completed task.

        Three gates, cheapest first:
          1. Structural: did the turn invoke any tools? Pure Q&A has no
             ``role=="tool"`` messages → skip (no sub-agent call).
          2. Length: a turn shorter than ``_ARCHIVE_MIN_CHARS`` isn't worth
             archiving → skip.
          3. Semantic: the archive sub-agent decides whether the turn produced
             a lasting artifact and returns ``should_archive`` accordingly.

        Missing a boundary just means the messages stay verbose until the next
        compaction pass — this is best-effort and never breaks the session.
        """
        if not assistant_text:
            return

        # Collect messages for this turn (user → assistant(s) → tools).
        # Reclamation earlier in the turn may have shrunk the message list, so
        # ``_turn_start_idx`` (captured at turn start) can now point past the
        # end — clamp it to stay in bounds (J2).
        start_idx = min(self._turn_start_idx, len(self.ctx.messages))
        turn_msgs = list(self.ctx.messages[start_idx:])
        if len(turn_msgs) < 2:  # need at least user + assistant
            return

        # Gate 1: structural — only archive turns that invoked tools. A pure
        # conversational exchange (clarification, Q&A) carries no artifact worth
        # archiving and should stay in the live context. This check is free.
        if not any(m.get("role") == "tool" for m in turn_msgs):
            return

        # Gate 2: length — very short turns, even with a tool call, rarely
        # produce enough substance to justify the sub-agent cost.
        if sum(len(m.get("content") or "") for m in turn_msgs) < _ARCHIVE_MIN_CHARS:
            return

        # Gate 3 + write + mark: shared with the cleanup-driven archiver.
        start = self.ctx.find_clean_task_boundary(self.ctx.messages, start_idx)
        end = len(self.ctx.messages)
        self._archive_range(start, end)

    def _archive_range(self, start: int, end: int) -> bool:
        """Archive the message range ``[start, end)`` if the sub-agent approves.

        Runs the semantic gate (``_archive_task``), writes the block to disk, and
        marks the range so later compression can fold it into a summary. Returns
        ``True`` when a block was written. Best-effort: any failure logs and
        returns ``False`` without disturbing the session. ``start`` must land on
        a ``user`` boundary; callers align it (``find_clean_task_boundary`` /
        ``next_unarchived_old_turn``).
        """
        msgs = self.ctx.messages
        if not (0 <= start < end <= len(msgs)) or msgs[start].get("role") != "user":
            return False
        turn_msgs = list(msgs[start:end])

        try:
            result = self._archive_task(turn_msgs)
        except Exception:
            log("archive_task: sub-agent call failed (non-fatal)")
            return False  # best-effort; never break the session
        if result is None:
            return False

        should_archive, summary, keywords, in_context = result
        if not should_archive:
            log("archive: sub-agent vetoed (no lasting artifact)")
            return False

        aid = self._archive_next_id
        self._archive_next_id += 1
        self._store.archive_block(ArchiveBlock(
            id=aid,
            summary=summary,
            keywords=keywords,
            in_context_summary=in_context,
            messages=turn_msgs,
        ))
        self.ctx.mark_archived(start, end, aid)
        log(f"archive: task #{aid} — {summary}")
        return True

    def _archive_task(self, messages: list[dict]) -> tuple[bool, str, str, str] | None:
        """Isolated sub-agent that generates archive metadata for a completed task.

        Returns ``(should_archive, summary, keywords, in_context_summary)`` or
        ``None`` on failure. ``should_archive`` defaults to ``True`` when the
        model omits the field, since callers already pre-filter to high-probability
        candidates (tool-bearing turns of meaningful length).
        """
        transcript = _flatten(messages)
        req = [
            {"role": "system", "content": _ARCHIVE_SYSTEM},
            {"role": "user", "content": transcript},
        ]
        try:
            raw = self.client.complete(req)
        except Exception:
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON in the response (model sometimes wraps in markdown).
            import re
            m = re.search(r"\{[^}]+\}", raw, re.DOTALL)
            if not m:
                return None
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return None
        return (
            bool(data.get("archive", True)),
            data.get("summary", ""),
            data.get("keywords", ""),
            data.get("in_context_summary", ""),
        )

    def export(self, cwd: str) -> str:
        """Write the full stored conversation to a Markdown file in ``cwd``.

        Reads from disk (the complete transcript) rather than the in-context
        messages, so compaction doesn't shrink the export. Returns the path.
        """
        from pathlib import Path
        from ..session.export import to_markdown

        messages = self._store.load() or list(self.ctx.messages)
        title = self.title or self._store.name
        md = to_markdown(title, messages)
        out = Path(cwd) / f"dsc-{self._store.name}.md"
        out.write_text(md, encoding="utf-8")
        return str(out)

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

        # Phase 3: try V2 cleaning first (compress archived + smart cleanup).
        if self.config.v2_cleaning:
            yield from self._reclaim_v2()
            return

        # Fallback: old compaction.
        yield from self._reclaim_old()

    def _reclaim_v2(self) -> Iterator[LoopEvent]:
        """V2 cleaning: compress archived ranges, then smart-cleanup the tail."""
        yield LoopEvent("notice", text="Context near limit — compressing archives…")

        # Ensure archive summaries are loaded from disk.
        self.ctx.populate_archives(str(self._store.archive_dir))

        # Step 1: compress archived ranges (replace big original text with tiny summaries).
        n = self.ctx.compress_archived_ranges()
        if n:
            yield LoopEvent("notice", text=f"Compressed {n} archived task(s).")

        # Step 2: cleanup tail (delete backed-up / read_archive results).
        before = self.ctx.estimated_tokens()
        deleted_bu, deleted_ra, _archived_new = self.ctx.cleanup_tail()
        now = self.ctx.estimated_tokens()
        saved = before - now

        total_deleted = deleted_bu + deleted_ra
        if total_deleted:
            yield LoopEvent(
                "notice",
                text=f"Cleaned {total_deleted} messages ({saved} tokens saved).",
            )
        # Step 3: if still over budget, archive old un-archived tool-bearing
        # turns that cleanup could only skip (no disk backup to fall back on),
        # then compress them. This closes the loop cleanup left open — otherwise
        # those turns sit in context until full compaction wipes everything.
        archived_new = 0
        if now > self.ctx.high_water:
            archived_new = yield from self._archive_stale_turns()
            if archived_new:
                extra = self.ctx.compress_archived_ranges()
                if extra:
                    yield LoopEvent("notice", text=f"Archived + compressed {archived_new} stale task(s).")
                now = self.ctx.estimated_tokens()

        # Sync store whenever the in-context history changed — including when
        # only compression fired (n>0) but nothing was deleted. Otherwise resume
        # reloads the old, uncompressed messages and silently loses the work (J3).
        if n or total_deleted or archived_new:
            self._store.replace(self.ctx.messages)

        if now > self.ctx.high_water:
            # Still above high-water after V2 → fall back to old compaction,
            # which drops to the last few turns and lands us well below the
            # low-water target.
            yield LoopEvent("notice", text="V2 cleaning insufficient — falling back to compaction.")
            yield from self._reclaim_old()

    def _archive_stale_turns(self, max_turns: int = 4) -> Iterator[LoopEvent]:
        """Archive up to ``max_turns`` old un-archived tool-bearing turns.

        Yields nothing to the UI itself (caller summarizes); ``return``s the
        count archived so the caller can compress and report. Bounded so one
        reclamation pass can't fire a dozen sub-agent calls.
        """
        archived = 0
        for _ in range(max_turns):
            rng = self.ctx.next_unarchived_old_turn()
            if rng is None:
                break
            start, end = rng
            if self._archive_range(start, end):
                archived += 1
            else:
                # Sub-agent vetoed or failed; stop to avoid re-scanning the same
                # un-archivable turn forever (mark_archived didn't fire, so
                # next_unarchived_old_turn would return it again).
                break
        return archived
        yield  # make this a generator (never reached)

    def _reclaim_old(self) -> Iterator[LoopEvent]:
        """Original compaction: summarize early history (old behaviour)."""
        yield LoopEvent("notice", text="Compacting history…")
        before = self.ctx.estimated_tokens()
        req = build_summary_request(self.ctx.messages)
        try:
            summary = format_compact_summary(self.client.complete(req))
            self.ctx.replace_history(summary)
            self._store.replace(self.ctx.messages)
            after = self.ctx.estimated_tokens()
            saved = before - after
            limit = self.ctx.limit or 1
            yield LoopEvent(
                "notice",
                text=(
                    f"History compacted — saved {saved:,} tokens "
                    f"({before * 100 // limit}% → {after * 100 // limit}%)."
                ),
            )
        except Exception as e:
            yield LoopEvent("notice", text=f"Compaction failed ({e}); continuing.")


def _preview(args: str, n: int = 60) -> str:
    s = args.replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")
