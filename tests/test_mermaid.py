"""Mermaid detection + rendering, and the MermaidBlock widget.

Extraction and the fallback logic are pure and always tested. The render tests
require termaid (the optional ``mermaid`` extra); they skip cleanly if it's
absent so the suite passes either way.
"""

from __future__ import annotations

import asyncio

import pytest

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll

from dsc.tui import mermaid
from dsc.tui.mermaid import extract_mermaid_blocks, safe_render, termaid_available
from dsc.tui.widgets import CopyButton, MermaidBlock

_SIMPLE = "graph LR\n A --> B --> C"


# -- extraction (pure) --------------------------------------------------------

def test_extract_none_when_no_block():
    assert extract_mermaid_blocks("just some prose") == []
    assert extract_mermaid_blocks("") == []


def test_extract_single_block():
    text = f"Here is a diagram:\n\n```mermaid\n{_SIMPLE}\n```\n\nDone."
    blocks = extract_mermaid_blocks(text)
    assert blocks == [_SIMPLE]


def test_extract_multiple_blocks():
    text = (
        "```mermaid\ngraph LR\n A --> B\n```\n"
        "text between\n"
        "```mermaid\ngraph TD\n X --> Y\n```\n"
    )
    blocks = extract_mermaid_blocks(text)
    assert len(blocks) == 2
    assert "A --> B" in blocks[0] and "X --> Y" in blocks[1]


def test_extract_ignores_non_mermaid_fence():
    text = "```python\nprint('hi')\n```"
    assert extract_mermaid_blocks(text) == []


def test_extract_tolerates_case_and_trailing_info():
    text = "```Mermaid  \ngraph LR\n A --> B\n```"
    assert extract_mermaid_blocks(text) == ["graph LR\n A --> B"]


# -- safe_render fallback (pure logic, no termaid needed) ---------------------

def test_safe_render_empty_source_is_none():
    assert safe_render("") is None
    assert safe_render("   ") is None


def test_safe_render_returns_none_when_termaid_missing(monkeypatch):
    monkeypatch.setattr(mermaid, "_termaid", None)
    assert safe_render(_SIMPLE) is None


def test_safe_render_none_on_empty_render(monkeypatch):
    """termaid's silent-failure mode (empty Text) must map to None, not a block."""
    from rich.text import Text

    class FakeTermaid:
        def render_rich(self, src):
            return Text("")  # what termaid returns for invalid input

    monkeypatch.setattr(mermaid, "_termaid", FakeTermaid())
    assert safe_render("garbage") is None


def test_safe_render_none_on_exception(monkeypatch):
    class BoomTermaid:
        def render_rich(self, src):
            raise RuntimeError("kaboom")

    monkeypatch.setattr(mermaid, "_termaid", BoomTermaid())
    assert safe_render(_SIMPLE) is None


def test_safe_render_success(monkeypatch):
    from rich.text import Text

    class FakeTermaid:
        def render_rich(self, src):
            return Text("┌───┐\n│ A │\n└───┘")

    monkeypatch.setattr(mermaid, "_termaid", FakeTermaid())
    out = safe_render(_SIMPLE)
    assert out is not None and "A" in out.plain


# -- widget -------------------------------------------------------------------

class _Harness(App):
    def __init__(self, block):
        super().__init__()
        self._block = block

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield self._block


def test_mermaid_block_copies_source(monkeypatch):
    from rich.text import Text

    captured: list[str] = []
    import dsc.tui.widgets as widgets_mod
    monkeypatch.setattr(widgets_mod, "copy_to_clipboard", lambda app, t: captured.append(t))

    block = MermaidBlock(Text("┌─┐\n│A│\n└─┘"), _SIMPLE)

    async def scenario():
        app = _Harness(block)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.click(CopyButton)
            await pilot.pause()

    asyncio.run(scenario())
    # Copies the SOURCE, not the ASCII art.
    assert captured == [_SIMPLE]


# -- real termaid (skips if the extra isn't installed) ------------------------

@pytest.mark.skipif(not termaid_available(), reason="termaid extra not installed")
def test_real_termaid_renders_simple_graph():
    out = safe_render(_SIMPLE)
    assert out is not None
    # Box-drawing chars present → an actual diagram was produced.
    assert any(ch in out.plain for ch in "┌─┐│└┘►▶")


@pytest.mark.skipif(not termaid_available(), reason="termaid extra not installed")
def test_real_termaid_invalid_returns_none():
    assert safe_render("total nonsense not a diagram") is None
