"""Web search tool: result formatting and error handling (no live network)."""

from __future__ import annotations

import sys
import types

from dsc.tools.web_search import SNIPPET_CHARS, WebSearchTool


def _install_fake_ddgs(monkeypatch, hits=None, error=None):
    """Install a fake `ddgs` module so tests never hit the network."""
    mod = types.ModuleType("ddgs")

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results=5):
            if error:
                raise error
            return list(hits or [])

    mod.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", mod)


def test_formats_results(monkeypatch):
    _install_fake_ddgs(
        monkeypatch,
        hits=[
            {"title": "T1", "href": "https://a.com", "body": "snippet one"},
            {"title": "T2", "href": "https://b.com", "body": "snippet two"},
        ],
    )
    res = WebSearchTool(".").run(query="python asyncio")
    assert not res.is_error
    assert "1. T1" in res.content
    assert "https://a.com" in res.content
    assert "2 results" in res.display


def test_snippet_is_truncated(monkeypatch):
    _install_fake_ddgs(
        monkeypatch,
        hits=[{"title": "Long", "href": "https://x.com", "body": "z" * 5000}],
    )
    res = WebSearchTool(".").run(query="q")
    assert "…" in res.content
    # Body is capped well under the raw length.
    assert len(res.content) < 5000


def test_no_results(monkeypatch):
    _install_fake_ddgs(monkeypatch, hits=[])
    res = WebSearchTool(".").run(query="nothing")
    assert not res.is_error
    assert "No web results" in res.content


def test_search_error_is_handled(monkeypatch):
    _install_fake_ddgs(monkeypatch, error=RuntimeError("network down"))
    res = WebSearchTool(".").run(query="q")
    assert res.is_error
    assert "network down" in res.content


def test_max_results_clamped(monkeypatch):
    captured = {}

    mod = types.ModuleType("ddgs")

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results=5):
            captured["n"] = max_results
            return []

    mod.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", mod)

    WebSearchTool(".").run(query="q", max_results=999)
    assert captured["n"] == 10  # clamped to MAX_RESULTS
