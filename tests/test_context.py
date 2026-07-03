"""Context manager: prefix stability, lossless pruning, compaction, restore."""

from __future__ import annotations

from dsc.context.manager import PRUNE_KEEP_RECENT, STUB, ContextManager


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
    m.replace_history("SUM", keep_recent=3)
    contents = [x["content"] for x in m.messages]
    assert contents[0].startswith("[Summary of earlier conversation]")
    assert contents[-3:] == ["m7", "m8", "m9"]


def test_restore_replaces_all():
    m = _mgr()
    m.add_user("old")
    m.restore([{"role": "user", "content": "new"}])
    assert m.messages == [{"role": "user", "content": "new"}]
    # System head still present after restore.
    assert m.render()[0]["role"] == "system"
