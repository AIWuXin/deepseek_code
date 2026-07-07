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

from dataclasses import dataclass, field

from .compaction import KEEP_TURNS, find_clean_tail_start
from .tokens import estimate_messages

# Tool results older than this many messages from the tail are eligible for
# lossless pruning once we cross the soft budget.
PRUNE_KEEP_RECENT = 8
PRUNE_MIN_CHARS = 1500  # only prune results big enough to be worth it
STUB = "[old tool result cleared to save context]"

# V2 cleaning gateway threshold: total saved tokens must exceed this ratio
# of context_limit before we risk breaking the mid-stream cache.
COMPRESS_THRESHOLD_RATIO = 0.1

# Reclamation watermarks (fractions of ``limit``).
#
# Every reclamation pass breaks the cached prefix for one turn — it rewrites or
# drops messages that are no longer at the tail. So the goal is to break the
# cache *rarely and deeply*: trigger at the high-water mark (before the hard
# limit, so we never nibble near the ceiling every turn) and reclaim all the way
# down toward the low-water target, buying many stable, cache-hitting turns
# before the next break.
HIGH_WATER_RATIO = 0.92
LOW_WATER_RATIO = 0.70


@dataclass
class ArchiveInfo:
    """Lightweight in-memory cache of an archived task (messages excluded)."""

    archive_id: int
    in_context_summary: str


class ContextManager:
    def __init__(self, system_prompt: str, limit: int):
        self.limit = limit
        # System message is the stable head of the prefix. Never touched.
        self._system = {"role": "system", "content": system_prompt}
        self._messages: list[dict] = []

        # V2 archive marking (Phase 1).  Per-message parallel list; None =
        # unarchived, int = archive block ID.  NEVER renders into API calls.
        self._archive_id: list[int | None] = []
        # Lightweight cache of archive summaries (full messages stay on disk).
        self._archives: dict[int, ArchiveInfo] = {}

        # Token accounting cache. estimate_messages is fully additive per
        # message, so tokens(system + msgs) == tokens(system) + tokens(msgs).
        # The system head never changes → estimate it once. The message sum is
        # maintained incrementally on append (the hot path: checked every loop
        # iteration) and lazily recomputed after any structural/content mutation
        # via the dirty flag. Turning an O(n²)-per-session scan into O(1) checks.
        self._system_tokens = estimate_messages([self._system])
        self._msg_tokens = 0
        self._tokens_dirty = False

    # -- message construction -------------------------------------------------

    def add_user(self, text: str) -> None:
        msg = {"role": "user", "content": text}
        self._messages.append(msg)
        self._archive_id.append(None)
        self._msg_tokens += estimate_messages([msg])

    def add_assistant(self, content: str, tool_calls: list[dict] | None = None) -> None:
        # reasoning_content is intentionally dropped: DeepSeek returns 400 if it
        # appears in a subsequent request's messages.
        msg: dict = {"role": "assistant", "content": content or ""}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self._messages.append(msg)
        self._archive_id.append(None)
        self._msg_tokens += estimate_messages([msg])

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        msg = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        self._messages.append(msg)
        self._archive_id.append(None)
        self._msg_tokens += estimate_messages([msg])

    # -- rendering ------------------------------------------------------------

    def render(self) -> list[dict]:
        """Full message list for the API: stable system head + history tail."""
        return [self._system, *self._messages]

    def estimated_tokens(self) -> int:
        if self._tokens_dirty:
            # A mutation touched message bodies/structure — recompute the sum
            # once and clear the flag. Rare (only on reclamation).
            self._msg_tokens = estimate_messages(self._messages)
            self._tokens_dirty = False
        return self._system_tokens + self._msg_tokens

    # -- reclamation ----------------------------------------------------------

    @property
    def high_water(self) -> int:
        """Trigger reclamation once the estimate crosses this (below the limit)."""
        return int(self.limit * HIGH_WATER_RATIO)

    @property
    def low_water(self) -> int:
        """Target to reclaim down toward, so the next break is many turns away."""
        return int(self.limit * LOW_WATER_RATIO)

    def maybe_reclaim(self) -> str | None:
        """Reclaim space if over the high-water mark. Returns a note if acted.

        We trigger at ``high_water`` rather than the hard ``limit`` so a single
        cache-breaking pass has room to reclaim deeply instead of the agent
        oscillating just under the ceiling — pruning a sliver and breaking the
        cache every single turn.
        """
        if self.estimated_tokens() <= self.high_water:
            return None
        pruned = self._prune_old_tool_results()
        if self.estimated_tokens() <= self.high_water:
            if pruned:
                return f"pruned {pruned} old tool result(s)"
            return None
        # Still above high-water after pruning → caller should compact deeply.
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
        if pruned:
            self._tokens_dirty = True  # bodies changed in place
        return pruned

    def replace_history(self, summary: str, keep_turns: int = KEEP_TURNS) -> None:
        """Swap summarised early history for a summary; retain recent turns.

        Uses ``find_clean_tail_start`` to cut at a complete turn boundary so
        ``tool`` messages are never orphaned from their ``assistant`` caller.
        """
        tail_start = find_clean_tail_start(self._messages, keep_turns)
        tail = self._messages[tail_start:]
        tail_aid = self._archive_id[tail_start:]  # keep matching archive marks
        summary_msg = {
            "role": "system",
            "content": f"[Summary of earlier conversation]\n{summary}",
        }
        self._messages = [summary_msg, *tail]
        self._archive_id = [None, *tail_aid]
        self._tokens_dirty = True

    @staticmethod
    def find_clean_task_boundary(messages: list[dict], target_idx: int) -> int:
        """Walk backwards from ``target_idx`` to the nearest preceding ``user`` message.

        Returns ``-1`` if no user message is found.  Clamped to ``len-1`` so
        ``target_idx == len(messages)`` does not index past the end.
        """
        i = min(target_idx, len(messages) - 1) if messages else 0
        while i >= 0 and messages[i].get("role") != "user":
            i -= 1
        return i

    @staticmethod
    def _next_turn_boundary(messages: list[dict], target_idx: int) -> int:
        """Walk *forward* from ``target_idx`` to the next ``user`` message.

        The mirror of ``find_clean_task_boundary`` (which walks backward).  Used
        to extend an archived range's right edge to a complete-turn boundary so a
        range never splits a turn and orphans its ``tool`` results.  Returns
        ``len(messages)`` when no later ``user`` message exists (the range runs
        to the end of history).  When ``messages[target_idx]`` is already a
        ``user`` message it returns ``target_idx`` unchanged.
        """
        n = len(messages)
        i = max(0, target_idx)
        while i < n and messages[i].get("role") != "user":
            i += 1
        return i

    # -- accessors for compaction ---------------------------------------------

    def restore(self, messages: list[dict]) -> None:
        """Replace all history with loaded session messages."""
        self._messages = list(messages)
        # Reset archive marks on restore — they are rebuilt on-demand.
        self._archive_id = [None] * len(self._messages)
        self._archives.clear()
        self._tokens_dirty = True

    # -- V2 archive marking (Phase 1) -----------------------------------------

    def mark_archived(self, start: int, end: int, archive_id: int) -> None:
        """Mark messages [start, end) as belonging to archive ``archive_id``.

        ``start`` **must** be on a ``user`` turn boundary — the caller is
        responsible for aligning it (see ``find_clean_task_boundary``).
        """
        for i in range(start, min(end, len(self._archive_id))):
            self._archive_id[i] = archive_id

    # -- Phase 3: batch compression + smart cleanup ---------------------------

    def _collect_compressible_ranges(self) -> list[tuple[int, int, int]]:
        """Scan ``_archive_id`` for contiguous ranges and return ``(start, end, archive_id)``.

        Only returns ranges whose total token savings exceed
        ``COMPRESS_THRESHOLD_RATIO * context_limit``.
        """
        estimated_saved = 0
        ranges: list[tuple[int, int, int]] = []
        i = 0
        while i < len(self._messages):
            aid = self._archive_id[i]
            if aid is None or self._messages[i].get("role") != "user":
                i += 1
                continue

            # Found the start of an archived range at a user boundary.
            j = i
            while j < len(self._messages) and self._archive_id[j] == aid:
                j += 1

            start = self.find_clean_task_boundary(self._messages, i)
            # Right end must extend *forward* to the next turn boundary; using
            # the backward finder here (old C1 bug) pulled ``end`` back into the
            # range, splitting the turn and orphaning its tool results.
            end = self._next_turn_boundary(self._messages, j)
            if start < 0 or start >= end:
                i = j
                continue

            # Estimate token savings: sum all messages in [start, end).
            raw_len = sum(
                estimate_messages([self._messages[k]]) for k in range(start, end)
            )
            estimated_saved += max(0, raw_len - 10)

            ranges.append((start, end, aid))
            i = end  # skip past this range

        threshold = int(self.limit * COMPRESS_THRESHOLD_RATIO)
        if estimated_saved < threshold:
            return []
        return ranges

    def compress_archived_ranges(self) -> int:
        """Batch-replace contiguous archived message ranges with their summaries.

        Scans ``_archive_id`` in real time.  Returns the number of ranges
        compressed (0 means no threshold met or nothing to compress).
        """
        ranges = self._collect_compressible_ranges()
        if not ranges:
            return 0

        replaced = 0
        # Process from the end backward so indices stay valid.
        for start, end, aid in reversed(ranges):
            archive = self._archives.get(aid)
            if archive is None:
                continue
            summary_msg = {
                "role": "system",
                "content": f"[Task summary] {archive.in_context_summary}",
            }
            self._messages[start:end] = [summary_msg]
            self._archive_id[start:end] = [None]
            replaced += 1
        if replaced:
            self._tokens_dirty = True
        return replaced

    def cleanup_tail(self) -> tuple[int, int, int]:
        """Smart cleanup of the tail when over budget.

        Returns ``(deleted_backed_up, deleted_read_archive, archived_new)``.
        Does NOT touch ``[Task summary]`` messages (those are mid-term
        summaries that survive until explicit demotion).
        """
        deleted_bu = 0   # backed-up original messages
        deleted_ra = 0   # read_archive results
        archived_new = 0  # newly archived unarchived messages

        # Work from the front of the message list (oldest first).
        # Protect the tail, snapped to a ``user`` turn boundary so the retained
        # tail always starts on a turn boundary and no ``tool`` result is
        # orphaned from its assistant caller (C2 bug → API 400).
        #
        # Two levels of protection:
        #   * long history → keep ~8 recent messages (backward-snapped cutoff);
        #   * short history (the whole list fits in ~8) → the backward snap
        #     collapses to 0, so fall back to protecting just the most recent
        #     *complete* turn.  Archived / read_archive messages older than that
        #     are on disk and always safe to drop, so we still reclaim them
        #     instead of bailing out entirely (C2 short-history regression).
        n = len(self._messages)
        if n < 2:
            return 0, 0, 0
        soft = self.find_clean_task_boundary(self._messages, max(0, n - 8))
        keep_tail = soft if soft > 0 else self.find_clean_task_boundary(
            self._messages, n - 1
        )
        if keep_tail <= 0:
            # Only the current turn exists → nothing older is safe to delete.
            return 0, 0, 0
        i = 0
        while i < keep_tail:
            msg = self._messages[i]
            aid = self._archive_id[i] if i < len(self._archive_id) else None
            content = msg.get("content", "")

            # 1. Backed-up original → delete.
            if aid is not None and not content.startswith("[Task summary]"):
                self._messages.pop(i)
                self._archive_id.pop(i)
                deleted_bu += 1
                keep_tail -= 1
                continue

            # 2. read_archive result → delete (disk has the original).
            if content.startswith("[Archive #"):
                self._messages.pop(i)
                self._archive_id.pop(i)
                deleted_ra += 1
                keep_tail -= 1
                continue

            # 3. Unarchived original → skip (can't auto-archive in cleanup
            #    without a sub-agent call; caller handles this).
            i += 1

        if deleted_bu or deleted_ra:
            self._tokens_dirty = True
        return deleted_bu, deleted_ra, archived_new

    def next_unarchived_old_turn(self) -> tuple[int, int] | None:
        """Find the oldest un-archived complete turn in the reclaimable region.

        ``cleanup_tail`` can delete messages already backed by an archive, but it
        can only *skip* old originals that were never archived (case 3) — there's
        nothing on disk to fall back to, so deleting them would lose data. This
        method hands the loop such a turn as a clean ``[start, end)`` range so it
        can archive it (via the sub-agent) and then reclaim it on the next pass.

        Returns the first ``user``-bounded turn that:
          * lies entirely before the protected tail (same boundary cleanup uses),
          * carries no archive id on any of its messages,
          * and actually contains a tool message (worth archiving; pure Q&A is
            left in context, matching ``_maybe_archive_turn``'s gate 1).

        Returns ``None`` when there's no such turn.
        """
        n = len(self._messages)
        if n < 2:
            return None
        # Same protected-tail computation as cleanup_tail.
        soft = self.find_clean_task_boundary(self._messages, max(0, n - 8))
        keep_tail = soft if soft > 0 else self.find_clean_task_boundary(
            self._messages, n - 1
        )
        if keep_tail <= 0:
            return None

        i = 0
        while i < keep_tail:
            if self._messages[i].get("role") != "user" or self._archive_id[i] is not None:
                i += 1
                continue
            # Start of a candidate turn at a user boundary. Extend to the next
            # user boundary (or the protected-tail edge, whichever comes first).
            end = self._next_turn_boundary(self._messages, i + 1)
            end = min(end, keep_tail)
            block = self._messages[i:end]
            # Only offer turns that are fully un-archived and tool-bearing.
            if (
                all(a is None for a in self._archive_id[i:end])
                and any(m.get("role") == "tool" for m in block)
            ):
                return (i, end)
            i = end
        return None

    # -- Phase 3 helpers -------------------------------------------------------

    def populate_archives(self, archive_dir_path: str) -> None:
        """Load ``ArchiveInfo`` from disk for all known archive IDs."""
        from pathlib import Path
        import json

        p = Path(archive_dir_path)
        if not p.exists():
            return
        for f in sorted(p.glob("*.json")):
            try:
                with f.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            aid = data.get("id")
            summary = data.get("in_context_summary", "")
            if aid is not None and summary:
                self._archives[aid] = ArchiveInfo(
                    archive_id=aid,
                    in_context_summary=summary,
                )

    @property
    def messages(self) -> list[dict]:
        return self._messages
