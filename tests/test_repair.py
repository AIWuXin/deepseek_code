"""Transcript repair: dangling tool_calls must never break a session.

Regression for the bug where a crash between a tool call and its recorded
result left an assistant ``tool_calls`` message unanswered, after which every
request 400'd. Covers the pure repair function plus resume self-healing.
"""

from __future__ import annotations

import dsc.session.store as store_mod
from dsc.agent.loop import AgentLoop
from dsc.config import Config
from dsc.context.repair import RECOVERED_STUB, repair_dangling_tool_calls
from dsc.tools import build_registry


def _asst(*ids):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": i, "type": "function", "function": {"name": "read_archive", "arguments": "{}"}}
            for i in ids
        ],
    }


def _tool(tid, content="ok"):
    return {"role": "tool", "tool_call_id": tid, "content": content}


# -- pure repair --------------------------------------------------------------

def test_wellformed_is_unchanged():
    msgs = [{"role": "user", "content": "hi"}, _asst("a"), _tool("a"),
            {"role": "assistant", "content": "done"}]
    out, changed = repair_dangling_tool_calls(msgs)
    assert changed is False
    assert out == msgs


def test_dangling_before_user_gets_stub():
    msgs = [_asst("a"), {"role": "user", "content": "next"}]
    out, changed = repair_dangling_tool_calls(msgs)
    assert changed is True
    assert out[0]["role"] == "assistant"
    assert out[1] == {"role": "tool", "tool_call_id": "a", "content": RECOVERED_STUB}
    assert out[2]["role"] == "user"


def test_trailing_dangling_gets_stub():
    msgs = [{"role": "user", "content": "hi"}, _asst("a")]
    out, changed = repair_dangling_tool_calls(msgs)
    assert changed is True
    assert out[-1] == {"role": "tool", "tool_call_id": "a", "content": RECOVERED_STUB}


def test_partial_multi_call_stubs_only_missing():
    msgs = [_asst("a", "b"), _tool("a")]
    out, changed = repair_dangling_tool_calls(msgs)
    assert changed is True
    answered = {m["tool_call_id"] for m in out if m["role"] == "tool"}
    assert answered == {"a", "b"}


def test_orphan_tool_message_dropped():
    msgs = [{"role": "user", "content": "hi"}, _tool("ghost")]
    out, changed = repair_dangling_tool_calls(msgs)
    assert changed is True
    assert all(m.get("role") != "tool" for m in out)


# -- resume self-heals a broken session --------------------------------------

def test_resume_repairs_broken_session(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "SESSIONS_DIR", tmp_path / "sessions")

    # Write a broken transcript: assistant tool_calls with no tool result.
    store = store_mod.SessionStore("broken")
    store.append({"role": "user", "content": "<environment>\ncwd\n</environment>"})
    store.append({"role": "user", "content": "search archives"})
    store.append(_asst("c1"))  # dangling — no tool result follows

    loop = AgentLoop(Config(api_key="x"), build_registry(str(tmp_path)),
                     str(tmp_path), session_name="broken")

    # In-context transcript must now answer c1.
    answered = [m.get("tool_call_id") for m in loop.ctx.messages if m.get("role") == "tool"]
    assert "c1" in answered

    # And the fix was persisted to disk.
    reloaded = store_mod.SessionStore("broken").load()
    answered_disk = [m.get("tool_call_id") for m in reloaded if m.get("role") == "tool"]
    assert "c1" in answered_disk
