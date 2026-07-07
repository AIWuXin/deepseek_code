"""Read-only file preview, shown as a modal over the conversation.

Opened from the sidebar's 👁 action button. Renders the file with syntax
highlighting (guessed from the extension) and line numbers, scrollable, and
dismissed by clicking the ✕ / pressing Esc / clicking the dimmed backdrop. It
never calls the LLM — it's a cheap "let me glance at this file" affordance.
"""

from __future__ import annotations

from pathlib import Path

from rich.syntax import Syntax
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

# Cap what we load so opening a huge file never stalls the UI. 200 KB is far
# more than fits on screen; the tail is elided with a note.
_MAX_BYTES = 200_000


class PreviewScreen(ModalScreen[None]):
    """Modal, scrollable, syntax-highlighted read-only view of one file."""

    BINDINGS = [("escape", "dismiss", "Close")]

    CSS = """
    PreviewScreen { align: center middle; }
    #preview-box {
        width: 85%;
        height: 85%;
        border: round $accent;
        background: $surface;
    }
    #preview-title {
        dock: top;
        height: 1;
        background: $panel;
        color: $text;
        padding: 0 1;
    }
    #preview-scroll { padding: 0 1; scrollbar-gutter: stable; }
    #preview-hint {
        dock: bottom;
        height: 1;
        background: $panel;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, path: str, rel: str | None = None) -> None:
        super().__init__()
        self._path = path
        self._rel = rel or path

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical

        with Vertical(id="preview-box"):
            yield Static(f"📄 {self._rel}", id="preview-title")
            with VerticalScroll(id="preview-scroll"):
                yield Static(self._build_render())
            yield Static("Esc / 点击外部关闭", id="preview-hint")

    def _build_render(self):
        """Return a Rich renderable for the file, or an error notice."""
        p = Path(self._path)
        try:
            raw = p.read_bytes()
        except OSError as e:
            return Text(f"无法读取 {self._rel}: {e}", style="red")
        truncated = len(raw) > _MAX_BYTES
        data = raw[:_MAX_BYTES]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return Text(f"{self._rel} 不是文本文件（二进制内容已跳过）。", style="yellow")
        if truncated:
            text += "\n\n… （文件过大，仅显示前 200 KB）"
        try:
            # from_path picks a lexer by extension; fall back to plain on error.
            return Syntax(
                text,
                Syntax.guess_lexer(str(p), code=text),
                line_numbers=True,
                word_wrap=True,
                indent_guides=True,
            )
        except Exception:
            return Syntax(text, "text", line_numbers=True, word_wrap=True)

    def on_click(self, event: events.Click) -> None:
        # A click on the dimmed backdrop (outside the box) closes the preview.
        if event.widget is self:
            self.dismiss(None)

    def action_dismiss(self) -> None:
        self.dismiss(None)
