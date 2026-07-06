"""App-level TUI behaviours: welcome banner, activity caption, auto-scroll lock.

Constructs the real DSCApp against a temp sessions dir and a dummy API key
(no network is touched — the client is only *created*, never called). Driven
with asyncio.run + Textual's headless harness.
"""

from __future__ import annotations

import asyncio

from textual.containers import VerticalScroll

import dsc.session.store as store_mod
from dsc.config import Config
from dsc.tools import build_registry
from dsc.tui.app import DSCApp
from dsc.tui.widgets import CopyButton, UserMessage


def _make_app(tmp_path, monkeypatch) -> DSCApp:
    monkeypatch.setattr(store_mod, "SESSIONS_DIR", tmp_path / "sessions")
    cfg = Config(api_key="dummy-key")
    reg = build_registry(str(tmp_path))
    return DSCApp(cfg, reg, str(tmp_path), session_name=None)


def test_welcome_banner_on_start(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            payloads = [b._payload for b in app.query(CopyButton)]
            assert any("DeepSeek Code" in p for p in payloads)

    asyncio.run(scenario())


def test_activity_caption_updates_and_clears(tmp_path, monkeypatch):
    from dsc.tui.prompt import PromptInput

    app = _make_app(tmp_path, monkeypatch)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            prompt = app.query_one(PromptInput)
            assert app._PROMPT_HINT in str(prompt.border_title)
            app._set_activity("thinking…")
            assert "thinking" in str(prompt.border_title)
            app._clear_activity()
            assert app._PROMPT_HINT in str(prompt.border_title)

    asyncio.run(scenario())


def test_autoscroll_follows_only_when_enabled(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)

    async def scenario():
        async with app.run_test(size=(80, 10)) as pilot:
            for i in range(40):
                app._append(UserMessage(f"line {i}"))
            await pilot.pause()
            log = app.query_one("#log", VerticalScroll)

            # follow ON → stays pinned to the bottom
            app._follow = True
            app._scroll()
            await pilot.pause()
            assert log.scroll_offset.y >= log.max_scroll_y - 1

            # user scrolled up + follow OFF → new content must NOT yank the view
            app._follow = False
            log.scroll_to(y=0, animate=False)
            await pilot.pause()
            app._append(UserMessage("arrived while reading history"))
            await pilot.pause()
            assert log.scroll_offset.y <= 2

    asyncio.run(scenario())


def test_agentloop_export_writes_markdown_file(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "SESSIONS_DIR", tmp_path / "sessions")
    from pathlib import Path
    from dsc.agent.loop import AgentLoop

    loop = AgentLoop(Config(api_key="dummy-key"), build_registry(str(tmp_path)),
                     str(tmp_path), session_name=None)
    loop._store.append({"role": "user", "content": "do the thing"})
    loop._store.append({"role": "assistant", "content": "thing done"})

    path = Path(loop.export(str(tmp_path)))
    assert path.exists() and path.suffix == ".md"
    text = path.read_text(encoding="utf-8")
    assert "do the thing" in text and "thing done" in text


def test_mouse_scroll_up_disables_follow(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._follow is True
            app.on_mouse_scroll_up(None)
            assert app._follow is False

    asyncio.run(scenario())
