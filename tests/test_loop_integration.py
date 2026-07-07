"""End-to-end loop test with a mock LLM client (no network).

Proves the risky integration path: stream → tool_call → execute → append
result → loop → done, plus cost accounting.
"""

from __future__ import annotations

import json

from dsc.agent.llm import Completion, StreamDelta
from dsc.agent.loop import AgentLoop
from dsc.config import Config
from dsc.tools import build_registry


class FakeClient:
    """Turn 1 asks to write a file via a tool; turn 2 finishes."""

    def __init__(self):
        self.model = "deepseek-v4-flash"
        self.calls = 0

    def stream(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            yield StreamDelta(content="Creating the file. ")
            yield Completion(
                content="Creating the file. ",
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "write",
                            "arguments": json.dumps({"path": "out.txt", "content": "hi\n"}),
                        },
                    }
                ],
                finish_reason="tool_calls",
                cache_hit=100,
                cache_miss=50,
                output_tokens=20,
            )
        else:
            yield StreamDelta(content="Done.")
            yield Completion(
                content="Done. Wrote out.txt.",
                finish_reason="stop",
                cache_hit=200,
                cache_miss=10,
                output_tokens=8,
            )

    def complete(self, messages):
        return "summary"


def test_full_turn_executes_tool(tmp_path):
    cfg = Config(api_key="x", model="deepseek-v4-flash")
    loop = AgentLoop(cfg, build_registry(str(tmp_path)), str(tmp_path))
    loop.client = FakeClient()

    kinds = []
    for ev in loop.send("create out.txt with hi"):
        kinds.append(ev.kind)

    # The tool actually ran and wrote the file.
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hi\n"
    # Loop went: tool_start → tool_end → ... → done.
    assert "tool_start" in kinds
    assert "tool_end" in kinds
    assert kinds[-1] == "done"
    # Two model calls happened (initial + after tool result).
    assert loop.client.calls == 2
    # Cost meter accumulated from both calls.
    assert loop.meter.cache_hit == 300
    assert loop.meter.usd > 0


class CrashToolClient:
    """Turn 1 asks for a tool; the handler will be forced to crash. Turn 2 ends."""

    def __init__(self):
        self.model = "deepseek-v4-flash"
        self.calls = 0

    def stream(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            yield Completion(
                content="",
                tool_calls=[{
                    "id": "c1", "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }],
                finish_reason="tool_calls",
            )
        else:
            yield Completion(content="recovered", finish_reason="stop")

    def complete(self, messages):
        return "summary"


def test_tool_handler_crash_never_dangles(tmp_path, monkeypatch):
    """Even if the tool handler raises, the tool_call must still be answered.

    Regression for the `ToolResult` NameError that skipped add_tool_result and
    left a dangling tool_calls message → API 400 on the next request.
    """
    loop = AgentLoop(Config(api_key="x"), build_registry(str(tmp_path)), str(tmp_path))
    loop.client = CrashToolClient()

    def boom(name, args):
        raise RuntimeError("simulated handler crash")

    monkeypatch.setattr(loop.registry, "execute", boom)

    list(loop.send("go"))

    msgs = loop.ctx.messages
    called = {tc["id"] for m in msgs if m.get("role") == "assistant"
              for tc in (m.get("tool_calls") or [])}
    answered = {m.get("tool_call_id") for m in msgs if m.get("role") == "tool"}
    assert called and called <= answered           # every call answered
    assert loop.client.calls == 2                  # loop continued past the crash


def test_archive_stale_turns_closes_cleanup_gap(tmp_path, monkeypatch):
    """cleanup can only skip un-archived old turns; the loop must archive them.

    Builds an old, un-archived, tool-bearing turn, then drives the archiver and
    asserts it wrote a block and marked the range — the gap that used to leave
    ``archived_new`` permanently 0.
    """
    import dsc.session.store as store_mod
    monkeypatch.setattr(store_mod, "SESSIONS_DIR", tmp_path / "sessions")

    loop = AgentLoop(Config(api_key="x"), build_registry(str(tmp_path)), str(tmp_path))

    # Sub-agent approves and returns metadata (no network).
    monkeypatch.setattr(
        loop, "_archive_task",
        lambda msgs: (True, "did the thing", "foo,bar", "recap of the thing"),
    )

    # One old tool-bearing turn, then padding to push it out of the tail.
    start = len(loop.ctx.messages)
    loop.ctx.add_user("refactor the parser")
    loop.ctx.add_assistant("", [{"id": "t1", "type": "function",
                                 "function": {"name": "bash", "arguments": "{}"}}])
    loop.ctx.add_tool_result("t1", "changed parser.py")
    for i in range(10):
        loop.ctx.add_user(f"later message {i}")

    n = list(loop._archive_stale_turns())  # drain the generator
    # It archived exactly one block...
    blocks = loop._store.list_blocks()
    assert len(blocks) == 1 and blocks[0]["summary"] == "did the thing"
    # ...and marked that range so compression can fold it.
    assert any(a is not None for a in loop.ctx._archive_id[start:start + 3])


def test_archive_stale_turns_stops_on_veto(tmp_path, monkeypatch):
    """A vetoed turn must stop the loop, not spin on the same un-archivable turn."""
    import dsc.session.store as store_mod
    monkeypatch.setattr(store_mod, "SESSIONS_DIR", tmp_path / "sessions")

    loop = AgentLoop(Config(api_key="x"), build_registry(str(tmp_path)), str(tmp_path))
    calls = {"n": 0}

    def veto(msgs):
        calls["n"] += 1
        return (False, "", "", "")  # sub-agent says: not worth archiving

    monkeypatch.setattr(loop, "_archive_task", veto)

    loop.ctx.add_user("some task")
    loop.ctx.add_assistant("", [{"id": "t", "type": "function",
                                 "function": {"name": "bash", "arguments": "{}"}}])
    loop.ctx.add_tool_result("t", "output")
    for i in range(10):
        loop.ctx.add_user(f"pad {i}")

    list(loop._archive_stale_turns())
    # Vetoed once, then bailed — did not re-scan the same turn forever.
    assert calls["n"] == 1
    assert loop._store.list_blocks() == []


def test_prefix_stability(tmp_path):
    """System message stays byte-identical; history only grows at the tail."""
    cfg = Config(api_key="x")
    loop = AgentLoop(cfg, build_registry(str(tmp_path)), str(tmp_path))
    loop.client = FakeClient()

    sys0 = loop.ctx.render()[0]["content"]
    list(loop.send("hi"))
    render = loop.ctx.render()
    assert render[0]["content"] == sys0  # system prompt unchanged
    assert render[0]["role"] == "system"
