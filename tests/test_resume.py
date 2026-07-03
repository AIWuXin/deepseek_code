"""Resume roundtrip: a turn is persisted, then reloaded into a new AgentLoop."""

from __future__ import annotations

import json

import pytest

from dsc.agent.llm import Completion, StreamDelta
from dsc.agent.loop import AgentLoop
from dsc.config import Config
from dsc.session import store as store_mod
from dsc.tools import build_registry


@pytest.fixture(autouse=True)
def _tmp_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "SESSIONS_DIR", tmp_path / "sessions")


class OneShotClient:
    """Answers with plain text, no tool calls."""

    def __init__(self, reply="done"):
        self.model = "deepseek-v4-flash"
        self._reply = reply

    def stream(self, messages, tools):
        yield StreamDelta(content=self._reply)
        yield Completion(content=self._reply, finish_reason="stop",
                         cache_hit=10, cache_miss=5, output_tokens=3)

    def complete(self, messages):
        return "summary"


class NamingClient(OneShotClient):
    """Records the messages passed to complete() so we can assert isolation."""

    def __init__(self):
        super().__init__()
        self.complete_messages = None

    def complete(self, messages):
        self.complete_messages = messages
        return "Fix Login Crash"


def test_generate_title_is_isolated(tmp_path):
    cfg = Config(api_key="x")
    loop = AgentLoop(cfg, build_registry(str(tmp_path)), str(tmp_path), session_name="named")
    client = NamingClient()
    loop.client = client
    list(loop.send("the login page crashes on click"))
    ctx_len = len(loop.ctx.messages)

    title = loop.generate_title("the login page crashes on click")

    assert title == "Fix Login Crash"
    assert loop.title == "Fix Login Crash"
    # The naming call must NOT touch the main conversation context.
    assert len(loop.ctx.messages) == ctx_len
    # It used a throwaway 2-message list (system + user), not the main history.
    assert [m["role"] for m in client.complete_messages] == ["system", "user"]
    # Persisted to a sidecar, never into the JSONL turn log.
    assert loop._store.read_title() == "Fix Login Crash"
    roles = [r.get("role") for r in loop._store.load()]
    assert all(r in ("user", "assistant", "tool") for r in roles)


def test_generate_title_noop_when_already_named(tmp_path):
    cfg = Config(api_key="x")
    loop = AgentLoop(cfg, build_registry(str(tmp_path)), str(tmp_path), session_name="named2")
    loop._store.save_title("Existing Title")
    loop.title = "Existing Title"
    loop.client = NamingClient()
    assert loop.generate_title("something") is None


def test_turn_is_persisted(tmp_path):
    cfg = Config(api_key="x")
    loop = AgentLoop(cfg, build_registry(str(tmp_path)), str(tmp_path), session_name="run1")
    loop.client = OneShotClient("hello there")
    list(loop.send("first question"))

    # The JSONL file holds env seed + user turn + assistant reply.
    records = loop._store.load()
    roles = [r["role"] for r in records]
    assert roles == ["user", "user", "assistant"]
    assert records[1]["content"] == "first question"
    assert records[2]["content"] == "hello there"


def test_resume_restores_context(tmp_path):
    cfg = Config(api_key="x")
    root = str(tmp_path)

    # Session 1: one turn.
    loop1 = AgentLoop(cfg, build_registry(root), root, session_name="run2")
    loop1.client = OneShotClient("answer one")
    list(loop1.send("q1"))
    n_before = len(loop1.ctx.messages)

    # Session 2: resume the same name — context should be restored from disk.
    loop2 = AgentLoop(cfg, build_registry(root), root, session_name="run2")
    assert len(loop2.ctx.messages) == n_before
    assert loop2.ctx.messages[1]["content"] == "q1"

    # Continuing appends to the same file rather than starting over.
    loop2.client = OneShotClient("answer two")
    list(loop2.send("q2"))
    roles = [r["role"] for r in loop2._store.load()]
    # env, q1, answer1, q2, answer2
    assert roles == ["user", "user", "assistant", "user", "assistant"]
