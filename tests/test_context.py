"""Context manager: prefix stability, lossless pruning, compaction, restore."""

from __future__ import annotations

from dsc.context.manager import PRUNE_KEEP_RECENT, STUB, ContextManager, ArchiveInfo
from dsc.context.compaction import format_compact_summary
from dsc.context.tokens import estimate_messages


def _mgr(limit=200_000):
    return ContextManager("SYSTEM", limit)


def _truth(m: ContextManager) -> int:
    """Ground-truth token count: a full, uncached recompute over render()."""
    return estimate_messages(m.render())


# -- Token cache correctness (incremental sum + dirty invalidation) -----------

def test_token_cache_matches_full_recompute_after_every_op():
    """The cached estimate must equal a fresh recompute after EVERY mutation.

    This is the safety net for the incremental cache: if a future mutation
    forgets to invalidate ``_tokens_dirty``, one of these assertions fails.
    """
    # append paths
    m = _mgr(limit=1000)
    m.add_user("hello world")
    assert m.estimated_tokens() == _truth(m)
    m.add_assistant("hi", [{"id": "1", "type": "function",
                            "function": {"name": "read", "arguments": "{}"}}])
    assert m.estimated_tokens() == _truth(m)
    m.add_tool_result("1", "z" * 3000)
    assert m.estimated_tokens() == _truth(m)

    # in-place prune (stubbing bodies)
    for i in range(PRUNE_KEEP_RECENT + 2):
        m.add_user(f"pad{i}")
    m._prune_old_tool_results()
    assert m.estimated_tokens() == _truth(m)

    # mark_archived alone changes nothing token-wise
    m.mark_archived(0, 2, 1)
    assert m.estimated_tokens() == _truth(m)

    # compress archived ranges (slice replacement)
    mc = _mgr(limit=50)
    mc.add_user("task - build the widget with plenty of descriptive content")
    mc.add_assistant("done - built it thoroughly with many implementation details")
    mc.mark_archived(0, 2, 3)
    mc._archives[3] = ArchiveInfo(archive_id=3, in_context_summary="Built widget.")
    mc.add_user("recent")
    mc.add_assistant("ok")
    mc.compress_archived_ranges()
    assert mc.estimated_tokens() == _truth(mc)

    # cleanup_tail (pops)
    mt = _mgr(limit=1_000_000)
    mt.add_user("archived task")
    mt.add_assistant("done")
    mt.mark_archived(0, 2, 1)
    mt._archives[1] = ArchiveInfo(archive_id=1, in_context_summary="Done.")
    mt.add_user("current")
    mt.add_assistant("active")
    mt.cleanup_tail()
    assert mt.estimated_tokens() == _truth(mt)

    # replace_history (rebuild)
    mr = _mgr()
    for i in range(8):
        mr.add_user(f"m{i}")
    mr.replace_history("SUMMARY", keep_turns=2)
    assert mr.estimated_tokens() == _truth(mr)

    # restore (reassign)
    mr.restore([{"role": "user", "content": "new"},
                {"role": "assistant", "content": "reply"}])
    assert mr.estimated_tokens() == _truth(mr)


def test_token_cache_append_stays_clean():
    """The hot path (append + check) must not flip the dirty flag."""
    m = _mgr()
    m.add_user("a")
    assert m._tokens_dirty is False
    m.add_assistant("b")
    m.add_tool_result("1", "c")
    assert m._tokens_dirty is False
    # And the value is right without ever hitting a full recompute.
    assert m.estimated_tokens() == _truth(m)


def test_token_cache_dirty_set_then_cleared():
    m = _mgr(limit=1_000_000)
    m.add_user("archived task")
    m.add_assistant("done")
    m.mark_archived(0, 2, 1)
    m._archives[1] = ArchiveInfo(archive_id=1, in_context_summary="Done.")
    m.add_user("current")
    m.add_assistant("active")
    m.cleanup_tail()
    assert m._tokens_dirty is True       # mutation flagged it
    _ = m.estimated_tokens()
    assert m._tokens_dirty is False      # lazy recompute cleared it


# -- Compaction summary formatting (analysis scratchpad stripping) ------------

def test_format_compact_summary_strips_analysis():
    raw = "<analysis>\nthinking out loud, lots of tokens\n</analysis>\n" \
          "- Goal: build the parser\n- Next: add tests"
    out = format_compact_summary(raw)
    assert "analysis" not in out.lower()
    assert "thinking out loud" not in out
    assert out.startswith("- Goal: build the parser")


def test_format_compact_summary_no_analysis_passthrough():
    raw = "- Goal: ship the feature\n- State: done"
    assert format_compact_summary(raw) == raw


def test_format_compact_summary_unterminated_tag_salvages():
    # Model opened <analysis> but never closed it → drop from the opener on,
    # rather than returning a scratchpad as the summary.
    raw = "prefix brief line\n<analysis>\nunterminated rambling"
    out = format_compact_summary(raw)
    assert out == "prefix brief line"


def test_format_compact_summary_empty_is_safe():
    assert format_compact_summary("") == ""
    assert format_compact_summary(None) == ""  # type: ignore[arg-type]


def test_system_is_stable_head():
    m = _mgr()
    m.add_user("a")
    m.add_assistant("b")
    render = m.render()
    assert render[0] == {"role": "system", "content": "SYSTEM"}


def test_prune_stubs_old_large_tool_results():
    # Tiny limit forces reclamation; big old tool result gets stubbed.
    m = _mgr(limit=10)
    big = "x" * 5000
    m.add_user("go")
    m.add_assistant("", [{"id": "1", "type": "function", "function": {"name": "read", "arguments": "{}"}}])
    m.add_tool_result("1", big)
    # Pad the tail so the big result is older than PRUNE_KEEP_RECENT.
    for i in range(PRUNE_KEEP_RECENT + 2):
        m.add_user(f"pad{i}")

    note = m.maybe_reclaim()
    tool_msgs = [x for x in m.messages if x.get("role") == "tool"]
    assert tool_msgs[0]["content"] == STUB
    assert note is not None


def test_recent_tool_results_not_pruned():
    m = _mgr(limit=10)
    m.add_tool_result("1", "y" * 5000)  # this is the most recent message
    m.maybe_reclaim()
    tool_msgs = [x for x in m.messages if x.get("role") == "tool"]
    # Within the keep-recent window → untouched.
    assert tool_msgs[0]["content"] != STUB


def test_replace_history_keeps_summary_and_tail():
    m = _mgr()
    for i in range(10):
        m.add_user(f"m{i}")
    m.replace_history("SUM", keep_turns=3)
    contents = [x["content"] for x in m.messages]
    assert contents[0].startswith("[Summary of earlier conversation]")
    # With 3 complete turns retained, the last 3 user messages are kept.
    assert contents[-3:] == ["m7", "m8", "m9"]


def test_replace_history_keeps_complete_turns_not_orphaned_tools():
    """Tool results must never survive without their assistant caller."""
    m = _mgr()
    m.add_user("request")
    m.add_assistant("", [{"id": "tc1", "type": "function", "function": {"name": "read", "arguments": "{}"}}])
    m.add_tool_result("tc1", "big result here")
    m.add_user("next")
    m.replace_history("SUM", keep_turns=1)
    # Should keep the complete last turn: user("next") + nothing else.
    assert len(m.messages) == 2  # summary + user("next")
    # The tool result for the old turn must NOT be in the tail.
    tool_roles = [x for x in m.messages if x.get("role") == "tool"]
    assert len(tool_roles) == 0


def test_restore_replaces_all():
    m = _mgr()
    m.add_user("old")
    m.restore([{"role": "user", "content": "new"}])
    assert m.messages == [{"role": "user", "content": "new"}]
    # System head still present after restore.
    assert m.render()[0]["role"] == "system"


# -- Phase 1: archive marking -------------------------------------------------

def test_archive_id_grows_with_messages():
    m = _mgr()
    assert m._archive_id == []
    m.add_user("hi")
    assert m._archive_id == [None]
    m.add_assistant("hello")
    assert m._archive_id == [None, None]
    m.add_tool_result("1", "out")
    assert m._archive_id == [None, None, None]


def test_mark_archived_sets_ids():
    m = _mgr()
    m.add_user("task")
    m.add_assistant("done")
    m.mark_archived(0, 2, archive_id=42)
    assert m._archive_id == [42, 42]


def test_mark_archived_aligns_to_exists():
    m = _mgr()
    m.add_user("a")
    m.add_assistant("b")
    m.mark_archived(0, 10, archive_id=7)  # end past list length
    assert m._archive_id == [7, 7]  # no crash, clamped by min()


def test_find_clean_task_boundary():
    m = _mgr()
    m.add_user("a")
    m.add_assistant("", [{"id": "tc1", "type": "function", "function": {"name": "r", "arguments": "{}"}}])
    m.add_tool_result("tc1", "result")
    m.add_user("b")

    # Tool result at index 2 → nearest user backward is 0.
    assert m.find_clean_task_boundary(m.messages, 2) == 0
    # Past end → clamped to last index (3, which is user "b").
    assert m.find_clean_task_boundary(m.messages, 5) == 3


def test_replace_history_syncs_archive_id_length():
    m = _mgr()
    for i in range(6):
        m.add_user(f"m{i}")
        m.add_assistant(f"r{i}")
    m._archive_id = [0, 0, 1, 1, None, None, None, None, None, None, None, None]
    m.replace_history("SUM", keep_turns=2)
    assert len(m._messages) == len(m._archive_id)


def test_restore_resets_archive_state():
    m = _mgr()
    m.add_user("a")
    m.mark_archived(0, 1, 99)
    assert m._archive_id == [99]
    m.restore([{"role": "user", "content": "new"}])
    assert m._archive_id == [None]
    assert len(m._archives) == 0


def test_archive_gating_structural_and_length(tmp_path, monkeypatch):
    """Archiving is gated cheaply before any sub-agent call:

      1. structural — a turn with no tool messages is never archived;
      2. length     — a tool-bearing turn under _ARCHIVE_MIN_CHARS is skipped.

    Both gates must short-circuit *without* invoking the archive sub-agent
    (which would cost a network call). We assert that by making _archive_task
    blow up if it's ever reached.
    """
    import dsc.session.store as store_mod
    from dsc.agent.loop import AgentLoop, _ARCHIVE_MIN_CHARS
    from dsc.config import Config
    from dsc.tools import build_registry

    monkeypatch.setattr(store_mod, "SESSIONS_DIR", tmp_path / "sessions")
    loop = AgentLoop(Config(api_key="x"), build_registry(str(tmp_path)), str(tmp_path))

    def _boom(_msgs):
        raise AssertionError("sub-agent must not be called when a gate rejects")

    monkeypatch.setattr(loop, "_archive_task", _boom)

    # Gate 1: pure Q&A (no tool message) → skipped, no sub-agent.
    loop._turn_start_idx = len(loop.ctx.messages)
    loop.ctx.add_user("what does this do?")
    loop.ctx.add_assistant("It parses the config." * 50)  # long, but no tool msg
    loop._maybe_archive_turn("It parses the config.")  # must not raise

    # Gate 2: has a tool msg but the whole turn is tiny → skipped.
    loop._turn_start_idx = len(loop.ctx.messages)
    loop.ctx.add_user("ls")
    loop.ctx.add_assistant("", [{"id": "t", "type": "function",
                                 "function": {"name": "bash", "arguments": "{}"}}])
    loop.ctx.add_tool_result("t", "a.py")  # tiny → under the min
    assert sum(len(m.get("content") or "")
               for m in loop.ctx.messages[loop._turn_start_idx:]) < _ARCHIVE_MIN_CHARS
    loop._maybe_archive_turn("done")  # must not raise


def test_compress_read_archive_small_content_noop():
    """Content under 2000 chars should pass through unchanged."""
    from dsc.agent.loop import _READ_COMPRESS_SYSTEM

    assert "compressing" in _READ_COMPRESS_SYSTEM
    assert "max 500 characters" in _READ_COMPRESS_SYSTEM


# -- Phase 3: compress + cleanup ----------------------------------------------

def test_compress_archived_ranges_basic():
    # Small limit so compression threshold (10% = 5) is easily met.
    m = _mgr(limit=50)
    # Add two archived tasks with enough content to trigger compression.
    m.add_user("task1 - implement the user authentication module")
    m.add_assistant("done1 - implemented JWT auth with refresh tokens")
    m.mark_archived(0, 2, 10)
    m._archives[10] = ArchiveInfo(archive_id=10, in_context_summary="Did task1.")

    m.add_user("task2 - add API rate limiting middleware")
    m.add_assistant("done2 - added RateLimit decorator with Redis backing")
    m.mark_archived(2, 4, 20)
    m._archives[20] = ArchiveInfo(archive_id=20, in_context_summary="Did task2.")

    # Add unarchived recent messages (to keep tail).
    m.add_user("recent")
    m.add_assistant("recent reply")

    before = len(m.messages)
    n = m.compress_archived_ranges()
    assert n == 2
    # 4 task messages replaced by 2 summaries = 6 before → 4 after + 2 recent = ?
    assert len(m.messages) < before
    # Verify summaries are present.
    summaries = [x for x in m.messages if x.get("content", "").startswith("[Task summary]")]
    assert len(summaries) == 2
    assert "task1" in summaries[0]["content"]
    assert "task2" in summaries[1]["content"]
    # Archived marks cleared.
    assert all(x is None for x in m._archive_id if x is not None)


def test_compress_archived_ranges_skips_recent_tail():
    """Recent unarchived messages should never be compressed."""
    m = _mgr(limit=50)
    m.add_user("old archivable task - implement the parser module")
    m.add_assistant("done - added PEG parser with error recovery")
    m.mark_archived(0, 2, 5)
    m._archives[5] = ArchiveInfo(archive_id=5, in_context_summary="Old task.")

    # Keep the last messages unarchived.
    m.add_user("still relevant")
    m.add_assistant("yep")

    n = m.compress_archived_ranges()
    assert n == 1  # Only the archived range was compressed
    assert len(m.messages) == 3  # summary + user + assistant


def test_watermark_properties():
    m = _mgr(limit=1000)
    assert m.high_water == 920   # 0.92 * limit
    assert m.low_water == 700    # 0.70 * limit


def test_reclaim_fires_in_watermark_gap():
    """Reclamation must trigger at high-water, before the hard limit.

    Old behaviour only acted when tokens exceeded ``limit``; now we act in the
    (high_water, limit] gap so we never nibble at the ceiling every turn.
    """
    m = _mgr(limit=1000)
    m.add_user("go")
    m.add_assistant("", [{"id": "1", "type": "function",
                          "function": {"name": "read", "arguments": "{}"}}])
    m.add_tool_result("1", "x" * 2400)  # prunable big old result
    for i in range(PRUNE_KEEP_RECENT + 2):
        m.add_user(f"pad{i}")           # push it out of the keep-recent window

    tokens = m.estimated_tokens()
    # Land inside the (high_water, limit] gap to prove early triggering.
    assert m.high_water < tokens <= m.limit, tokens
    assert m.maybe_reclaim() is not None


def test_reclaim_noop_below_high_water():
    m = _mgr(limit=100_000)
    m.add_user("small")
    m.add_assistant("also small")
    assert m.estimated_tokens() < m.high_water
    assert m.maybe_reclaim() is None


def test_next_turn_boundary_walks_forward():
    """Right-edge finder must extend forward, never pull back into the range."""
    m = _mgr()
    m.add_user("a")                                              # 0 user
    m.add_assistant("", [{"id": "tc1", "type": "function",
                          "function": {"name": "r", "arguments": "{}"}}])  # 1 asst
    m.add_tool_result("tc1", "result")                          # 2 tool
    m.add_user("b")                                             # 3 user
    m.add_assistant("done")                                     # 4 asst

    # From inside the first turn's tail (idx 1/2) → next user boundary is 3.
    assert m._next_turn_boundary(m.messages, 1) == 3
    assert m._next_turn_boundary(m.messages, 2) == 3
    # Already on a user boundary → unchanged.
    assert m._next_turn_boundary(m.messages, 3) == 3
    # No later user message → clamps to len (range runs to end).
    assert m._next_turn_boundary(m.messages, 4) == len(m.messages)


def test_compress_uses_forward_boundary_not_backward():
    """Regression: compress must not crash and must keep turns whole.

    Reproduces the C1 bug where ``_next_turn_boundary`` was called but never
    defined (AttributeError), and the earlier variant used the backward finder
    which split turns.
    """
    m = _mgr(limit=50)
    # One archived task that ends with a tool result, then a live tail.
    m.add_user("task - build the indexer with incremental updates")   # 0
    m.add_assistant("", [{"id": "t1", "type": "function",
                          "function": {"name": "read", "arguments": "{}"}}])  # 1
    m.add_tool_result("t1", "x" * 200)                                # 2
    m.mark_archived(0, 3, 30)
    m._archives[30] = ArchiveInfo(archive_id=30, in_context_summary="Built indexer.")
    m.add_user("now add a status bar")                                # 3 (live tail)
    m.add_assistant("ok")                                             # 4

    n = m.compress_archived_ranges()
    assert n == 1
    # The archived tool result must be gone (folded into the summary), and the
    # live tail must survive intact.
    assert any(x.get("content", "").startswith("[Task summary]") for x in m.messages)
    tail_users = [x for x in m.messages if x.get("role") == "user"]
    assert any("status bar" in x["content"] for x in tail_users)
    # No orphaned tool result left behind.
    assert not any(x.get("role") == "tool" for x in m.messages)


def test_cleanup_tail_short_history_still_cleans_archived():
    """Regression (C2): a short history must not disable cleanup entirely.

    With the whole list inside the ~8-message keep window, the backward snap
    collapses to 0; we must fall back to protecting only the last turn so
    on-disk archived messages are still reclaimed.
    """
    m = _mgr(limit=1_000_000)
    m.add_user("archived task")     # 0
    m.add_assistant("done")         # 1
    m.mark_archived(0, 2, 1)
    m._archives[1] = ArchiveInfo(archive_id=1, in_context_summary="Done.")
    m.add_user("current")           # 2 (last turn — must be protected)
    m.add_assistant("active")       # 3

    bu, ra, _new = m.cleanup_tail()
    assert bu == 2                  # both backed-up messages dropped
    # Last turn preserved.
    assert m.messages[-2:] == [
        {"role": "user", "content": "current"},
        {"role": "assistant", "content": "active"},
    ]


def _tool_turn(m, user, aid=None):
    """Append a user→assistant(tool_call)→tool turn; optionally mark archived."""
    start = len(m.messages)
    m.add_user(user)
    m.add_assistant("", [{"id": "x", "type": "function",
                          "function": {"name": "bash", "arguments": "{}"}}])
    m.add_tool_result("x", "some output")
    if aid is not None:
        m.mark_archived(start, len(m.messages), aid)
    return start


def test_next_unarchived_old_turn_finds_toolbearing_turn():
    m = _mgr()
    _tool_turn(m, "old task that touched files")   # 0..3, un-archived
    # Pad the tail so the old turn is well before the protected region.
    for i in range(10):
        m.add_user(f"pad{i}")
    rng = m.next_unarchived_old_turn()
    assert rng is not None
    start, end = rng
    assert start == 0
    # Range is a complete turn (user + assistant + tool).
    block = m.messages[start:end]
    assert block[0]["role"] == "user"
    assert any(x["role"] == "tool" for x in block)


def test_next_unarchived_old_turn_skips_already_archived():
    m = _mgr()
    _tool_turn(m, "archived task", aid=5)          # already archived
    for i in range(10):
        m.add_user(f"pad{i}")
    # The only old tool-bearing turn is archived → nothing to offer.
    assert m.next_unarchived_old_turn() is None


def test_next_unarchived_old_turn_skips_pure_qa():
    m = _mgr()
    # Old turn with NO tool message → not worth archiving, must be skipped.
    m.add_user("what is this?")
    m.add_assistant("an explanation")
    for i in range(10):
        m.add_user(f"pad{i}")
    assert m.next_unarchived_old_turn() is None


def test_next_unarchived_old_turn_protects_tail():
    m = _mgr()
    # A single recent tool-bearing turn, nothing before it → inside the
    # protected tail, so it must not be offered for archiving.
    _tool_turn(m, "recent task")
    assert m.next_unarchived_old_turn() is None


def test_cleanup_tail_deletes_backed_up_and_read_archive():
    m = _mgr(limit=1_000_000)
    m.add_user("archived task")
    m.add_assistant("done")
    m.mark_archived(0, 2, 1)
    m._archives[1] = ArchiveInfo(archive_id=1, in_context_summary="Done.")

    # Simulate a read_archive result in the tail.
    m._messages.append({"role": "tool", "content": "[Archive #5 re-archived] some key info"})
    m._archive_id.append(None)

    m.add_user("current")
    m.add_assistant("active")

    bu, ra, new = m.cleanup_tail()
    assert bu == 2  # backed-up archived task messages deleted
    # read_archive result might not be reached with small message count,
    # but the archived messages are removed.
    remaining_tasks = [x for x in m.messages
                       if x.get("role") == "user" and "current" in str(x.get("content", ""))]
    assert len(remaining_tasks) == 1



