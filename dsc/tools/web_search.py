"""Web search via DuckDuckGo (no API key required).

Returns a compact, numbered list of results — title, URL, and a truncated
snippet — rather than full page content, keeping token cost low. The model can
follow up by reading a URL if it needs the full text.
"""

from __future__ import annotations

from .base import Tool, ToolResult

DEFAULT_RESULTS = 5
MAX_RESULTS = 10
SNIPPET_CHARS = 300


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web (DuckDuckGo) and return the top results as title, URL, "
        "and a short snippet. Use for current information, docs, error messages, "
        "or anything outside the workspace. Returns snippets, not full pages — "
        "use web_fetch to read the full text of any result."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "max_results": {
                "type": "integer",
                "description": f"Number of results (1-{MAX_RESULTS}, default {DEFAULT_RESULTS}).",
                "default": DEFAULT_RESULTS,
            },
        },
        "required": ["query"],
    }

    def run(self, query: str, max_results: int = DEFAULT_RESULTS) -> ToolResult:
        n = max(1, min(int(max_results), MAX_RESULTS))
        try:
            from ddgs import DDGS
        except ImportError:
            return ToolResult(
                "web_search unavailable: the 'ddgs' package is not installed.",
                is_error=True,
            )

        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=n))
        except Exception as e:
            return ToolResult(f"web_search failed: {e}", is_error=True)

        if not hits:
            return ToolResult(f"No web results for: {query}", display=f"web_search '{query}' — 0")

        blocks = []
        for i, h in enumerate(hits, start=1):
            title = (h.get("title") or "").strip()
            url = (h.get("href") or "").strip()
            body = " ".join((h.get("body") or "").split())
            if len(body) > SNIPPET_CHARS:
                body = body[:SNIPPET_CHARS] + "…"
            blocks.append(f"{i}. {title}\n   {url}\n   {body}")

        out = "\n\n".join(blocks)
        return ToolResult(out, display=f"web_search '{query}' — {len(hits)} results")
