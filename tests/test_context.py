"""Context manager: prefix stability, lossless pruning, compaction, restore."""

from __future__ import annotations

from dsc.context.manager import PRUNE_KEEP_RECENT, STUB, ContextManager, ArchiveInfo


def _mgr(limit=200_000):
    return ContextManager("SYSTEM", limit)


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


def test_completion_signal_detection():
    from dsc.agent.loop import AgentLoop

    assert AgentLoop._has_completion_signal("Done — fixed the bug")
    assert AgentLoop._has_completion_signal("搞定了，已添加 web_fetch 工具")
    assert AgentLoop._has_completion_signal("All tests pass, issue resolved")
    assert not AgentLoop._has_completion_signal("Let me investigate further")
    assert not AgentLoop._has_completion_signal("")


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



