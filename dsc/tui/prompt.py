"""Multi-line, auto-growing prompt input.

A single-line Input hides earlier text once you type past its width. This
subclasses TextArea so the box wraps and grows with the content (from a few
lines up to a cap), while keeping a chat-like key scheme:

    Enter        → submit
    Shift+Enter  → newline
    (Ctrl+J also inserts a newline, for terminals that swallow Shift+Enter.)
"""

from __future__ import annotations

import re

from textual import events
from textual.message import Message
from textual.widgets import TextArea

MIN_LINES = 3
MAX_LINES = 12

# SGR / X10 mouse-report escape sequences. During streaming the terminal can
# flush buffered mouse moves as input; if they slip past the app's mouse
# handling they arrive here as literal text like "\x1b[<35;47;22M" (or with the
# ESC already stripped, "[<35;47;22M"). Strip them so they never reach the box.
_MOUSE_SEQ = re.compile(r"\x1b?\[<?\d+;\d+;\d+[Mm]")
# Also drop any other stray control chars except tab/newline.
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _sanitize(text: str) -> str:
    text = _MOUSE_SEQ.sub("", text)
    text = _CTRL.sub("", text)
    return text


class PromptInput(TextArea):
    """TextArea that submits on Enter and grows to fit its content."""

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    def __init__(self) -> None:
        super().__init__(soft_wrap=True, show_line_numbers=False)
        self._sync_height()

    def _on_paste(self, event: events.Paste) -> None:
        # Intercept paste to strip mouse-report/control noise before it lands.
        cleaned = _sanitize(event.text)
        event.prevent_default()
        event.stop()
        if cleaned:
            self.insert(cleaned)
            self._sync_height()

    async def _on_key(self, event: events.Key) -> None:
        # NOTE: do not filter control characters here — Backspace/Delete arrive
        # with control-char values (\x7f etc.) and must pass through to the base
        # TextArea. Mouse-report noise comes via Paste, handled in _on_paste.
        # Enter submits; Shift+Enter / Ctrl+J insert a real newline.
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = _sanitize(self.text).strip()
            if text:
                self.post_message(self.Submitted(text))
            self.text = ""
            self._sync_height()
            return
        if event.key in ("shift+enter", "ctrl+j"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
            self._sync_height()
            return
        await super()._on_key(event)
        # Any other key may have changed the line count.
        self._sync_height()

    def _sync_height(self) -> None:
        """Grow/shrink the box to fit content, clamped to [MIN, MAX] lines."""
        lines = self.text.count("\n") + 1
        rows = max(MIN_LINES, min(lines, MAX_LINES))
        # +2 accounts for the top/bottom border rows.
        self.styles.height = rows + 2
