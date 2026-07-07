"""Multi-line, auto-growing prompt input.

A single-line Input hides earlier text once you type past its width. This
subclasses TextArea so the box wraps and grows with the content (from a few
lines up to a cap), while keeping a chat-like key scheme:

    Enter        → submit
    Ctrl+Enter   → newline
"""

from __future__ import annotations

import re

from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea

try:
    import pyperclip
except ImportError:
    pyperclip = None  # fallback to Textual's OSC 52

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

    # Remap ctrl+a to select_all (Textual defaults it to cursor_line_start).
    BINDINGS = [
        Binding("ctrl+a", "select_all", "Select all"),
        *[b for b in TextArea.BINDINGS
          if b.action != "cursor_line_start" or "ctrl+a" not in b.key],
    ]

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
        #
        # Enter submits; Shift+Enter / Ctrl+Enter / Ctrl+J insert a newline.
        # Textual 8.2.8 events.Key has __slots__ = ['key', 'character'] only —
        # no .shift attribute. On Windows most terminals report Shift+Enter as
        # key="enter" (same as plain Enter), making it impossible to distinguish
        # at the widget level.  Ctrl+Enter may also be reported as "enter" on
        # some terminals, but at least one of these combos typically works.
        #
        # We handle every possible form explicitly:
        #   key="shift+enter" / "ctrl+enter" / "ctrl+j" → newline
        #   key="enter" with no match above → submit
        if event.key == "enter":
            # On old Textual with .shift attr, check it
            if getattr(event, "shift", False):
                event.prevent_default()
                event.stop()
                self.insert("\n")
                self._sync_height()
                return
            # Plain Enter → submit
            event.prevent_default()
            event.stop()
            text = _sanitize(self.text).strip()
            if text:
                self.post_message(self.Submitted(text))
            self.text = ""
            self._sync_height()
            return
        if event.key in ("shift+enter", "ctrl+enter", "ctrl+j"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
            self._sync_height()
            return
        await super()._on_key(event)
        # Any other key may have changed the line count.
        self._sync_height()

    def action_copy(self) -> None:
        """Copy selection via pyperclip (more reliable on Windows than OSC 52)."""
        selected_text = self.selected_text
        if not selected_text:
            from textual.actions import SkipAction
            raise SkipAction()
        if pyperclip is not None:
            pyperclip.copy(selected_text)
        else:
            self.app.copy_to_clipboard(selected_text)

    def action_cut(self) -> None:
        """Cut selection via pyperclip (more reliable on Windows than OSC 52)."""
        if self.read_only:
            return
        start, end = self.selection
        if start == end:
            edit_result = self._delete_cursor_line()
        else:
            edit_result = self._delete_via_keyboard(start, end)
        if edit_result is not None:
            text = edit_result.replaced_text
            if pyperclip is not None:
                pyperclip.copy(text)
            else:
                self.app.copy_to_clipboard(text)

    def _sync_height(self) -> None:
        """Grow/shrink the box to fit content, clamped to [MIN, MAX] lines."""
        lines = self.text.count("\n") + 1
        rows = max(MIN_LINES, min(lines, MAX_LINES))
        # +2 accounts for the top/bottom border rows.
        self.styles.height = rows + 2
