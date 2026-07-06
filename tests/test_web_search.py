"""Web tool: result formatting and error handling for search mode (no live network)."""

from __future__ import annotations

import sys
import types

from dsc.tools.web import SNIPPET_CHARS, WebTool


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
    res = WebTool(".").run(query="python asyncio")
    assert not res.is_error
    assert "1. T1" in res.content
    assert "https://a.com" in res.content
    assert "2 results" in res.display


def test_snippet_is_truncated(monkeypatch):
    _install_fake_ddgs(
        monkeypatch,
        hits=[{"title": "Long", "href": "https://x.com", "body": "z" * 5000}],
    )
    res = WebTool(".").run(query="q")
    assert "…" in res.content
    # Body is capped well under the raw length.
    assert len(res.content) < 5000


def test_no_results(monkeypatch):
    _install_fake_ddgs(monkeypatch, hits=[])
    res = WebTool(".").run(query="nothing")
    assert not res.is_error
    assert "No web results" in res.content


def test_search_error_is_handled(monkeypatch):
    _install_fake_ddgs(monkeypatch, error=RuntimeError("network down"))
    res = WebTool(".").run(query="q")
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

    WebTool(".").run(query="q", max_results=999)
    assert captured["n"] == 10  # clamped to MAX_RESULTS


def test_fetch_empty_list_is_error():
    """An empty urls list is rejected."""
    res = WebTool(".").run(urls=[])
    assert res.is_error


def test_both_query_and_urls_prefers_search(monkeypatch):
    """If model sends both, search wins (common first step)."""
    _install_fake_ddgs(
        monkeypatch,
        hits=[{"title": "T", "href": "https://a.com", "body": "body"}],
    )
    res = WebTool(".").run(query="python", urls=["https://example.com"])
    assert not res.is_error
    assert "1. T" in res.content  # search result, not fetch error


def test_neither_query_nor_urls_is_error():
    res = WebTool(".").run()
    assert res.is_error
