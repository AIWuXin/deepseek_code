"""Copy-to-clipboard affordance on conversation blocks.

Uses Textual's headless test harness (``App.run_test``) to actually click the
copy icon and assert the right text is handed to the clipboard. Driven with
``asyncio.run`` so we don't need pytest-asyncio.
"""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll

import dsc.tui.widgets as widgets_mod
from dsc.tui import clipboard
from dsc.tui.widgets import (
    AssistantMessage,
    CopyButton,
    Notice,
    ReasoningBlock,
    ToolLine,
    ToolOutput,
    UserMessage,
)


class _Harness(App):
    """Minimal app that mounts one block in a scroll container."""

    def __init__(self, block):
        super().__init__()
        self._block = block

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield self._block


def _capture(monkeypatch) -> list[str]:
    captured: list[str] = []
    monkeypatch.setattr(
        widgets_mod, "copy_to_clipboard", lambda app, text: captured.append(text)
    )
    return captured


def _click_copy(block) -> None:
    async def scenario():
        app = _Harness(block)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.click(CopyButton)
            await pilot.pause()

    asyncio.run(scenario())


# -- per-block copy payloads --------------------------------------------------

def test_user_message_copies_raw_text(monkeypatch):
    captured = _capture(monkeypatch)
    _click_copy(UserMessage("hello world"))
    assert captured == ["hello world"]


def test_tool_line_copies_line(monkeypatch):
    captured = _capture(monkeypatch)
    _click_copy(ToolLine("✓ read(foo.py)"))
    assert captured == ["✓ read(foo.py)"]


def test_notice_copies_text(monkeypatch):
    captured = _capture(monkeypatch)
    _click_copy(Notice("history compacted"))
    assert captured == ["history compacted"]


def test_tool_output_error_starts_expanded():
    assert ToolOutput("boom happened", error=True)._collapsible.collapsed is False


def test_tool_output_short_starts_expanded():
    assert ToolOutput("a\nb\nc")._collapsible.collapsed is False  # 3 lines < 10


def test_tool_output_long_success_stays_collapsed():
    assert ToolOutput("x\n" * 30)._collapsible.collapsed is True  # 31 lines, ok


def test_tool_output_explicit_collapsed_overrides_heuristic():
    assert ToolOutput("x\n" * 30, collapsed=False)._collapsible.collapsed is False
    assert ToolOutput("short", collapsed=True)._collapsible.collapsed is True


def test_tool_output_copy_bar_reachable_while_collapsed(monkeypatch):
    """The copy bar sits outside the (collapsed) Collapsible, so one click grabs
    the full output without expanding it."""
    captured = _capture(monkeypatch)
    payload = "line1\nline2\n" * 40
    _click_copy(ToolOutput(payload))
    assert captured and captured[0] == payload


def test_reasoning_block_copies_thinking(monkeypatch):
    captured = _capture(monkeypatch)
    block = ReasoningBlock()

    async def scenario():
        app = _Harness(block)
        async with app.run_test() as pilot:
            block.append("step 1\nstep 2")
            await pilot.pause()
            await pilot.click(CopyButton)
            await pilot.pause()

    asyncio.run(scenario())
    assert captured == ["step 1\nstep 2"]


def test_assistant_message_payload_tracks_stream(monkeypatch):
    """Mid-stream and final copies both grab the text shown so far."""
    captured = _capture(monkeypatch)
    block = AssistantMessage("")

    async def scenario():
        app = _Harness(block)
        async with app.run_test() as pilot:
            block.append_text("Hello ")
            block.append_text("world")
            block.finalize()
            await pilot.pause()
            await pilot.click(CopyButton)
            await pilot.pause()

    asyncio.run(scenario())
    assert captured == ["Hello world"]


def test_copy_button_shows_feedback(monkeypatch):
    """Clicking flips the label to a confirmation."""
    _capture(monkeypatch)
    block = UserMessage("x")

    async def scenario():
        app = _Harness(block)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.click(CopyButton)
            await pilot.pause()
            btn = app.query_one(CopyButton)
            assert "copied" in btn.label

    asyncio.run(scenario())


# -- clipboard helper (no Textual) --------------------------------------------

def test_clipboard_helper_fires_both_mechanisms(monkeypatch):
    calls = {"osc": 0, "pyp": 0}

    class FakeApp:
        def copy_to_clipboard(self, t):
            calls["osc"] += 1

    class FakePyperclip:
        def copy(self, t):
            calls["pyp"] += 1

    monkeypatch.setattr(clipboard, "pyperclip", FakePyperclip())
    clipboard.copy_to_clipboard(FakeApp(), "text")
    assert calls == {"osc": 1, "pyp": 1}


def test_clipboard_helper_empty_is_noop(monkeypatch):
    class FakeApp:
        def copy_to_clipboard(self, t):
            raise AssertionError("should not be called for empty text")

    clipboard.copy_to_clipboard(FakeApp(), "")


def test_clipboard_helper_never_raises(monkeypatch):
    class Boom:
        def copy_to_clipboard(self, t):
            raise RuntimeError("osc failed")

    class BoomPyp:
        def copy(self, t):
            raise RuntimeError("pyperclip failed")

    monkeypatch.setattr(clipboard, "pyperclip", BoomPyp())
    # Must swallow both failures — copying can never crash the UI.
    clipboard.copy_to_clipboard(Boom(), "text")
