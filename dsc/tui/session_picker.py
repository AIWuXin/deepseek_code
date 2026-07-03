"""Session picker screen: shows saved sessions as a clickable list."""

from __future__ import annotations

import datetime

from rich.text import Text
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Label, ListItem, ListView

from ..session import SessionStore


def _format_sessions() -> list[tuple[str, Text]]:
    """Return (session_name, rich_label) for each saved session.

    The label leads with the session's title (its first request) and last
    reply, so users recognize sessions by content — not by the numeric name.
    """
    out: list[tuple[str, Text]] = []
    for info in SessionStore.infos():
        when = datetime.datetime.fromtimestamp(info.mtime).strftime("%m-%d %H:%M")
        label = Text()
        label.append(f"  {info.title}\n", style="bold")
        meta = f"    {info.count} msgs · {when}"
        if info.summary:
            meta += f" · {info.summary}"
        label.append(meta, style="dim")
        out.append((info.name, label))
    return out


class SessionPickerScreen(Screen[str | None]):
    """Modal screen listing saved sessions. Enter resumes; 'd' (twice) deletes."""

    TITLE = "Saved Sessions"

    # NOTE: do NOT bind "enter" here. ListView consumes Enter and emits a
    # ListView.Selected message (which on_list_view_selected handles); a
    # screen-level enter binding never fires, so selection would silently break.
    BINDINGS = [
        ("d", "delete", "Delete"),
        ("escape", "cancel", "Cancel"),
    ]

    CSS = """
    ListView { height: 1fr; }
    ListItem { padding: 1 2; height: auto; }
    ListItem:hover { background: $accent 30%; }
    """

    def compose(self) -> ComposeResult:
        # Session names are timestamps like "20260703-140000" — they start with
        # a digit, which is NOT a valid Textual widget id. Map rows back to names
        # by index instead of using id=.
        self._pending_delete: int | None = None  # index awaiting confirm
        self._refreshing = False  # suppress highlight events during rebuild
        self._rebuild_data()
        yield ListView(*self._make_items())
        yield Footer()

    # -- data / rendering -----------------------------------------------------

    def _rebuild_data(self) -> None:
        sessions = _format_sessions()
        self._names = [name for name, _ in sessions]
        self._labels = [label for _, label in sessions]

    def _make_items(self) -> list[ListItem]:
        if not self._names:
            return [ListItem(Label("  No saved sessions."))]
        return [ListItem(Label(label)) for label in self._labels]

    def _refresh_list(self, keep_index: int = 0) -> None:
        lv = self.query_one(ListView)
        self._refreshing = True
        lv.clear()
        for item in self._make_items():
            lv.mount(item)
        if self._names:
            lv.index = min(keep_index, len(self._names) - 1)
        self._refreshing = False

    # -- selection / actions --------------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Fires on Enter-in-list and on mouse click — the real resume path."""
        index = self.query_one(ListView).index
        if self._names and index is not None and 0 <= index < len(self._names):
            self.dismiss(self._names[index])

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_delete(self) -> None:
        index = self.query_one(ListView).index
        if not self._names or index is None or not (0 <= index < len(self._names)):
            return
        if self._pending_delete == index:
            # Second press on the same row → actually delete.
            SessionStore.delete(self._names[index])
            self._pending_delete = None
            self._rebuild_data()
            self._refresh_list(keep_index=index)
        else:
            # First press → arm confirmation on this row.
            self._pending_delete = index
            self._labels[index] = Text("  ⚠ press d again to delete", style="bold red")
            self._refresh_list(keep_index=index)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        # Moving off the armed row cancels the pending delete. Ignore the
        # highlight churn our own _refresh_list produces.
        if self._refreshing or self._pending_delete is None:
            return
        if self.query_one(ListView).index != self._pending_delete:
            self._pending_delete = None
            self._rebuild_data()
            self._refresh_list(keep_index=self.query_one(ListView).index or 0)
