"""Prompt input: mouse-sequence sanitizing and editing-key behavior."""

from __future__ import annotations

import asyncio

from dsc.tui.prompt import _sanitize


def test_sanitize_strips_sgr_mouse_reports():
    garbage = "[<35;47;22M[<35;46;22M[<35;45;21M"
    assert _sanitize(garbage) == ""


def test_sanitize_strips_mouse_with_esc():
    assert _sanitize("\x1b[<35;47;22Mhello") == "hello"


def test_sanitize_keeps_normal_text_and_cjk():
    assert _sanitize("修复 foo.py 的 bug") == "修复 foo.py 的 bug"


def test_sanitize_keeps_newline_and_tab_drops_other_control():
    assert _sanitize("line1\nline2\tend\x07") == "line1\nline2\tend"


def test_editing_keys_and_paste_filter():
    """Backspace deletes, Enter submits stripped text, mouse paste is filtered."""
    import tempfile

    from textual import events

    from dsc.config import Config
    from dsc.tools import build_registry
    from dsc.tui.app import DSCApp
    from dsc.tui.prompt import PromptInput

    async def scenario():
        d = tempfile.mkdtemp()
        app = DSCApp(Config(api_key="x"), build_registry(d), d)
        app.run_turn = lambda text: setattr(app, "_sent", text)  # stub network turn
        async with app.run_test(size=(100, 24)) as pilot:
            pi = app.query_one(PromptInput)
            pi.focus()
            await pilot.press("h", "i", "!")
            await pilot.press("backspace")
            await pilot.pause()
            assert pi.text == "hi"
            # mouse garbage arriving as a paste must not enter the box
            pi.post_message(events.Paste("[<35;47;22M"))
            await pilot.pause()
            assert pi.text == "hi"
            # Enter submits and clears
            await pilot.press("enter")
            await pilot.pause()
            assert app._sent == "hi"
            assert pi.text == ""

    asyncio.run(scenario())
