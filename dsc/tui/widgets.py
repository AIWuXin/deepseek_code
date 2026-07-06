"""Conversation widgets: user/assistant messages, tool lines, notices.

Assistant text streams in as plain Text (cheap, throttled to ~20fps) and is
swapped to a full Markdown render only when the message finishes — re-parsing
Markdown on every token froze the entire UI on long outputs.
Reasoning (deepseek thinking) is shown dimmed until the answer starts.

Note: do NOT name any method ``_render`` — Textual's ``Static`` already defines
an internal ``_render`` that the layout engine calls to measure height. Shadowing
it returns None and crashes reflow with ``NoneType has no attribute get_height``.
"""

from __future__ import annotations

import time

from rich.markdown import Markdown
from rich.text import Text
from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.widgets import Collapsible, Static


class UserMessage(Static):
    def __init__(self, text: str):
        super().__init__(Text(f"› {text}", style="bold"))


class ReasoningBlock(Collapsible):
    """Collapsible container for deepseek 'thinking' output.

    Collapsed by default (thinking is noise once the answer lands), but click
    the title to expand and read the chain-of-thought. The inner Static is
    updated in place as reasoning streams in.
    """

    def __init__(self) -> None:
        self._body = Static(Text("", style="dim italic"))
        super().__init__(self._body, title="💭 thinking", collapsed=True)

    def append(self, reasoning: str) -> None:
        self._body.update(Text(reasoning, style="dim italic"))


class AssistantMessage(Static):
    """Live-updating assistant bubble.

    During streaming we repaint with plain Text (cheap, O(1)) and throttle to
    ~20fps, so the UI thread is never stuck re-parsing Markdown on every token
    — that re-parse is what froze the *entire* interface on long outputs (not
    just the bubble: keyboard and status bar died too). Only when the message
    finishes do we swap in the full Markdown render via finalize().
    """

    _STREAM_INTERVAL = 0.05  # seconds between plain-text repaints while streaming

    def __init__(self, text: str = ""):
        self._text = text
        self._dirty = False
        self._last_render = 0.0
        # Pass the initial renderable to Static; never call update() pre-mount.
        super().__init__(Markdown(text) if text else Text(""))

    def append_text(self, chunk: str) -> bool:
        """Append a streamed chunk.

        Returns True when a repaint actually fired, so the caller can throttle
        scroll (and thus log reflow) to the same cadence as the repaint.
        """
        self._text += chunk
        self._dirty = True
        now = time.monotonic()
        if now - self._last_render >= self._STREAM_INTERVAL:
            self._last_render = now
            self._dirty = False
            self.update(Text(self._text))  # plain text: cheap while streaming
            return True
        return False

    def finalize(self) -> None:
        """Replace the streaming plain text with the full Markdown render."""
        if self._text:
            self.update(Markdown(self._text))
        self._dirty = False
        self._last_render = time.monotonic()


class ToolLine(Static):
    """One compact line summarizing a tool call, to save screen space."""

    def __init__(self, text: str, running: bool = False, error: bool = False):
        style = "red" if error else ("cyan" if running else "dim")
        super().__init__(Text(text, style=style))


class Notice(Static):
    def __init__(self, text: str):
        super().__init__(Text(f"• {text}", style="yellow"))


class ToolOutput(Collapsible):
    """Collapsible block showing the full output of a tool call."""

    def __init__(self, output: str, collapsed: bool = True):
        # No fixed id: several tool outputs can be mounted in one turn, and a
        # shared id would collide in the DOM. Let Textual assign its own.
        self._body = Static(Syntax(output, "text", word_wrap=True))
        super().__init__(self._body, title="📄 output", collapsed=collapsed)
