"""Conversation widgets: user/assistant messages, tool lines, notices.

Assistant text is rendered as Markdown and updated in place while streaming.
Reasoning (deepseek thinking) is shown dimmed until the answer starts.

Note: do NOT name any method ``_render`` — Textual's ``Static`` already defines
an internal ``_render`` that the layout engine calls to measure height. Shadowing
it returns None and crashes reflow with ``NoneType has no attribute get_height``.
"""

from __future__ import annotations

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
    """Live-updating assistant bubble holding streaming Markdown text."""

    def __init__(self, text: str = ""):
        self._text = text
        # Pass the initial renderable to Static; never call update() pre-mount.
        super().__init__(self._build())

    def _build(self):
        return Markdown(self._text) if self._text else Text("")

    def append_text(self, chunk: str) -> None:
        self._text += chunk
        self.update(self._build())


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
        self._body = Static(Syntax(output, "text", word_wrap=True), id="tool-output")
        super().__init__(self._body, title="📄 output", collapsed=collapsed)
