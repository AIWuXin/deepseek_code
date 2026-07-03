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
