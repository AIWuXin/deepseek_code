"""Command palette: unique ordering and index→command mapping (no invalid ids)."""

from __future__ import annotations

import asyncio
import tempfile

from dsc.config import Config
from dsc.tools import build_registry
from dsc.tui.app import DSCApp
from dsc.tui.commands import CommandScreen, _sorted_unique


def test_sorted_unique_has_no_duplicate_meanings():
    entries = _sorted_unique()
    metaphors = [desc for _, desc in entries]
    assert len(metaphors) == len(set(metaphors))
    # Sorted by command string.
    assert entries == sorted(entries, key=lambda x: x[0])


def test_command_screen_selection_maps_to_stripped_command():
    """Selecting a row dismisses with the command, placeholder stripped."""

    async def scenario():
        d = tempfile.mkdtemp()
        app = DSCApp(Config(api_key="x"), build_registry(d), d)
        async with app.run_test(size=(100, 30)) as pilot:
            result = {}
            app.push_screen(CommandScreen(), lambda c: result.__setitem__("cmd", c))
            await pilot.pause()
            screen = app.screen
            # Compose must succeed despite command strings like "/clear".
            assert len(screen._entries) > 0
            lv = screen.query_one("ListView")
            lv.focus()
            # Find the "/model <name>" row and select it; expect "/model" back.
            target = next(i for i, (c, _) in enumerate(screen._entries) if c.startswith("/model"))
            lv.index = target
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert result["cmd"] == "/model"

    asyncio.run(scenario())
