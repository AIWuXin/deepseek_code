"""Conversation widgets: user/assistant messages, tool lines, notices.

Every block is a small ``Vertical`` container: the content on top, and a
right-aligned "⧉ copy" affordance docked at the bottom. Clicking the affordance
copies that block's raw text to the clipboard — the reliable way to grab text
out of a Textual app, which captures the mouse and so disables the terminal's
native drag-to-select.

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
from textual import events
from textual.containers import Horizontal, Vertical
from textual.widgets import Collapsible, Static

from .clipboard import copy_to_clipboard

# ── display helper ───────────────────────────────────────────────────────
# On CJK terminals circled digits (①②③) render at 1 cell wide.  When mixed
# with 2-cell-wide Han characters they visually merge.  We add one space
# after each circled digit for display; the copy button still copies raw text.
_CIRCLED = frozenset(chr(cp) for cp in range(0x2460, 0x2500)) | \
           frozenset(chr(cp) for cp in range(0x2776, 0x2794))


def _display_text(raw: str) -> str:
    """Insert a space after each circled digit for visual spacing."""
    if not raw:
        return raw
    parts = []
    for ch in raw:
        parts.append(ch)
        if ch in _CIRCLED:
            parts.append(" ")
    return "".join(parts)


class CopyButton(Static):
    """A tiny clickable affordance; clicking copies its payload to the clipboard."""

    DEFAULT_CSS = """
    CopyButton {
        width: auto;
        height: 1;
        color: $text-muted;
        padding: 0 1;
    }
    CopyButton:hover {
        color: $accent;
        text-style: bold;
        background: $boost;
    }
    """

    _IDLE = "⧉ copy"
    _DONE = "✓ copied"

    def __init__(self, payload: str = "") -> None:
        super().__init__(self._IDLE)
        self._payload = payload
        # Current label, tracked explicitly (Static doesn't expose its content).
        self.label = self._IDLE

    def set_payload(self, text: str) -> None:
        self._payload = text

    def _set_label(self, text: str) -> None:
        self.label = text
        self.update(text)

    def on_click(self, event: events.Click) -> None:
        # Stop propagation so a click never also toggles a nearby Collapsible.
        event.stop()
        copy_to_clipboard(self.app, self._payload)
        self._set_label(self._DONE)
        self.set_timer(1.2, lambda: self._set_label(self._IDLE))


class _CopyBar(Horizontal):
    """A 1-row bar that right-aligns a single ``CopyButton``."""

    DEFAULT_CSS = """
    _CopyBar {
        height: 1;
        width: 100%;
        align-horizontal: right;
    }
    """

    def __init__(self, payload: str = "") -> None:
        self._button = CopyButton(payload)
        super().__init__(self._button)

    def set_payload(self, text: str) -> None:
        self._button.set_payload(text)


class _Block(Vertical):
    """Base for a multi-line block: content on top, copy bar docked below."""

    DEFAULT_CSS = """
    _Block { height: auto; width: 1fr; }
    """


class _InlineBlock(Horizontal):
    """Base for a single-line block: content fills the row, copy icon at line end.

    Cheaper vertically than ``_Block`` — a short status line doesn't get a whole
    extra row for the copy affordance, the icon just rides at the right edge.
    """

    DEFAULT_CSS = """
    _InlineBlock { height: auto; width: 1fr; }
    _InlineBlock .inline-body { width: 1fr; }
    """

    def __init__(self, body: Static, payload: str) -> None:
        body.add_class("inline-body")
        self._body = body
        self._button = CopyButton(payload)
        super().__init__(self._body, self._button)


class UserMessage(_Block):
    def __init__(self, text: str):
        self._body = Static(Text(f"› {_display_text(text)}", style="bold"))
        self._bar = _CopyBar(text)
        super().__init__(self._body, self._bar)


class ReasoningBlock(_Block):
    """Collapsible container for deepseek 'thinking' output, with a copy bar.

    Collapsed by default (thinking is noise once the answer lands); click the
    title to expand and read the chain-of-thought. The copy bar lives below the
    collapsible so it stays reachable even while collapsed. The inner Static is
    updated in place as reasoning streams in.
    """

    def __init__(self) -> None:
        self._body = Static(Text("", style="dim italic"))
        self._collapsible = Collapsible(self._body, title="💭 thinking", collapsed=True)
        self._bar = _CopyBar("")
        super().__init__(self._collapsible, self._bar)

    def append(self, reasoning: str) -> None:
        self._body.update(Text(_display_text(reasoning), style="dim italic"))
        self._bar.set_payload(reasoning)


class AssistantMessage(_Block):
    """Live-updating assistant bubble with a copy bar.

    During streaming we repaint the inner body with plain Text (cheap, O(1)) and
    throttle to ~20fps, so the UI thread is never stuck re-parsing Markdown on
    every token — that re-parse is what froze the *entire* interface on long
    outputs (not just the bubble: keyboard and status bar died too). Only when
    the message finishes do we swap in the full Markdown render via finalize().
    """

    _STREAM_INTERVAL = 0.05  # seconds between plain-text repaints while streaming

    def __init__(self, text: str = ""):
        self._text = text
        self._dirty = False
        self._last_render = 0.0
        display = _display_text(text)
        self._body = Static(Markdown(display) if display else Text(""))
        self._bar = _CopyBar(text)
        super().__init__(self._body, self._bar)

    def append_text(self, chunk: str) -> bool:
        """Append a streamed chunk.

        Returns True when a repaint actually fired, so the caller can throttle
        scroll (and thus log reflow) to the same cadence as the repaint.
        """
        self._text += chunk
        self._dirty = True
        # Keep the copy payload current so a mid-stream copy grabs what's shown.
        self._bar.set_payload(self._text)
        now = time.monotonic()
        if now - self._last_render >= self._STREAM_INTERVAL:
            self._last_render = now
            self._dirty = False
            self._body.update(Text(_display_text(self._text)))  # spaces for display
            return True
        return False

    def finalize(self) -> None:
        """Replace the streaming plain text with the full Markdown render."""
        display = _display_text(self._text)
        if display:
            self._body.update(Markdown(display))
        self._dirty = False
        self._last_render = time.monotonic()
        self._bar.set_payload(self._text)


class ToolLine(_InlineBlock):
    """One compact line summarizing a tool call; copy icon rides at the line end."""

    def __init__(self, text: str, running: bool = False, error: bool = False):
        style = "red" if error else ("cyan" if running else "dim")
        super().__init__(Static(Text(_display_text(text), style=style)), text)


class Notice(_InlineBlock):
    def __init__(self, text: str):
        super().__init__(Static(Text(f"• {_display_text(text)}", style="yellow")), text)


class ToolOutput(_Block):
    """Collapsible block showing the full output of a tool call, with a copy bar.

    The copy bar sits below the collapsible so the full output is one click away
    without having to expand it first.

    Collapse is decided by usefulness, not a blanket default: errors and short
    results start expanded (you almost always want to read them); long, healthy
    output stays folded to keep the log scannable. Pass ``collapsed`` explicitly
    to override the heuristic.
    """

    # Outputs with fewer lines than this start expanded even when successful.
    _SHORT_LINES = 10

    def __init__(self, output: str, collapsed: bool | None = None, error: bool = False):
        if collapsed is None:
            line_count = output.count("\n") + 1
            collapsed = not (error or line_count < self._SHORT_LINES)
        # No fixed id: several tool outputs can be mounted in one turn, and a
        # shared id would collide in the DOM. Let Textual assign its own.
        self._body = Static(Syntax(output, "text", word_wrap=True))
        self._collapsible = Collapsible(self._body, title="📄 output", collapsed=collapsed)
        self._bar = _CopyBar(output)
        super().__init__(self._collapsible, self._bar)


class MermaidBlock(_Block):
    """A rendered ```mermaid diagram plus its collapsed source.

    The diagram (a Rich ``Text`` from termaid) sits on top; the original mermaid
    source is tucked into a collapsed block below so it's still there to read or
    copy. The copy bar copies the *source*, not the ASCII art — the source is
    what you paste back into mermaid.live or a doc.
    """

    def __init__(self, diagram, source: str):
        self._body = Static(diagram)
        self._collapsible = Collapsible(
            Static(Syntax(source, "text", word_wrap=True)),
            title="⌗ mermaid source",
            collapsed=True,
        )
        self._bar = _CopyBar(source)
        super().__init__(self._body, self._collapsible, self._bar)
