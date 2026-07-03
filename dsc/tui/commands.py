"""Command palette screen: shows all available commands alphabetically, clickable."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Label, ListItem, ListView

COMMANDS: list[tuple[str, str]] = [
    ("/clear", "Clear the conversation log"),
    ("/commands", "Show this command menu"),
    ("/exit", "Exit the application"),
    ("/help", "Show this command menu"),
    ("/model <name>", "Switch to a different model"),
    ("/quit", "Exit the application"),
    ("/q", "Exit the application"),
    ("/resume", "Resume a previous session"),
    ("/sessions", "List all saved sessions"),
]


def _sorted_unique() -> list[tuple[str, str]]:
    """Return sorted unique display entries, omitting aliases of the same command."""
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for cmd, desc in COMMANDS:
        key = desc.split(" application")[0]  # group aliases by meaning
        if key not in seen:
            seen.add(key)
            result.append((cmd, desc))
    result.sort(key=lambda x: x[0])
    return result


class CommandScreen(Screen[str | None]):
    """Modal screen listing all commands. Dismisses with the selected command string."""

    TITLE = "Commands"

    CSS = """
    ListView { height: 1fr; }
    ListItem { padding: 0 2; }
    ListItem:hover { background: $accent 30%; }
    """

    def compose(self) -> ComposeResult:
        # Keep the ordered command list so we can map a selected row back to its
        # command string by index. Command strings like "/clear" or "/model
        # <name>" are NOT valid Textual widget ids, so we must not use id=.
        self._entries = _sorted_unique()
        items = [
            ListItem(Label(f"  {cmd:<22}  —  {desc}"))
            for cmd, desc in self._entries
        ]
        yield ListView(*items)
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = event.list_view.index
        if index is not None and 0 <= index < len(self._entries):
            cmd = self._entries[index][0]
            # Drop any "<placeholder>" so the caller gets a runnable prefix.
            cmd = cmd.split(" <")[0]
            self.dismiss(cmd)
        else:
            self.dismiss(None)

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)
