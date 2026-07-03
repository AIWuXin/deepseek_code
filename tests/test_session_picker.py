"""Session picker: selection resumes, and the two-press delete works.

Regression guard: a screen-level "enter" binding never fires (ListView eats
Enter and emits Selected), so selection must go through on_list_view_selected.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from dsc.config import Config
from dsc.session import store as store_mod
from dsc.session import SessionStore
from dsc.tools import build_registry
from dsc.tui.app import DSCApp
from dsc.tui.session_picker import SessionPickerScreen


def _seed_sessions(tmp: Path, names):
    store_mod.SESSIONS_DIR = tmp / "sessions"
    for n in names:
        s = SessionStore(n)
        s.append({"role": "user", "content": f"task {n}"})
        s.save_title(f"Title {n}")


def test_enter_resumes_selected_session(tmp_path):
    _seed_sessions(tmp_path, ["alpha", "beta", "gamma"])

    async def scenario():
        d = tempfile.mkdtemp()
        app = DSCApp(Config(api_key="x"), build_registry(d), d)
        async with app.run_test(size=(100, 30)) as pilot:
            result = {}
            app.push_screen(SessionPickerScreen(), lambda r: result.__setitem__("v", r))
            await pilot.pause()
            sp = app.screen
            lv = sp.query_one("ListView")
            lv.focus()
            lv.index = sp._names.index("beta")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert result["v"] == "beta"

    asyncio.run(scenario())


def test_two_press_delete_removes_session(tmp_path):
    _seed_sessions(tmp_path, ["one", "two"])

    async def scenario():
        d = tempfile.mkdtemp()
        app = DSCApp(Config(api_key="x"), build_registry(d), d)
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(SessionPickerScreen())
            await pilot.pause()
            sp = app.screen
            lv = sp.query_one("ListView")
            lv.focus()
            target = sp._names.index("two")
            lv.index = target
            await pilot.pause()
            before = list(sp._names)
            await pilot.press("d")  # arm
            await pilot.pause()
            assert sp._pending_delete == target
            await pilot.press("d")  # confirm
            await pilot.pause()
            assert "two" not in sp._names
            assert len(sp._names) == len(before) - 1

    asyncio.run(scenario())

    # The file is actually gone from disk.
    assert SessionStore.from_name("two") is None


def test_escape_cancels(tmp_path):
    _seed_sessions(tmp_path, ["solo"])

    async def scenario():
        d = tempfile.mkdtemp()
        app = DSCApp(Config(api_key="x"), build_registry(d), d)
        async with app.run_test(size=(100, 30)) as pilot:
            result = {"v": "unset"}
            app.push_screen(SessionPickerScreen(), lambda r: result.__setitem__("v", r))
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert result["v"] is None

    asyncio.run(scenario())
