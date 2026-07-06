"""Web search and page fetching — combined into one tool.

Two modes, selected by which parameter you pass:
  query      → DuckDuckGo search (returns title, URL, snippet per result)
  urls       → fetch full page content as Markdown (1-5 URLs)

Prefer search first to find relevant pages, then fetch for details.
"""

from __future__ import annotations

from .base import Tool, ToolResult

DEFAULT_RESULTS = 5
MAX_RESULTS = 10
SNIPPET_CHARS = 300
MAX_URLS = 5
MAX_CHARS = 80_000  # per-URL cap, roughly 20k tokens
TRUNCATED_WARNING = "[truncated — content exceeds limit]"


class WebTool(Tool):
    name = "web"
    description = (
        "Search the web or fetch page content. "
        "Provide 'query' to search DuckDuckGo (returns title, URL, snippet), "
        "or provide 'urls' to fetch full page content as Markdown (1-5 URLs). "
        "Prefer search first to find relevant pages, then fetch for details."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — returns matching result titles, URLs, and snippets.",
            },
            "max_results": {
                "type": "integer",
                "description": f"Number of search results (1-{MAX_RESULTS}, default {DEFAULT_RESULTS}). Only used with 'query'.",
                "default": DEFAULT_RESULTS,
            },
            "urls": {
                "type": "array",
                "items": {"type": "string", "format": "uri"},
                "description": f"URL(s) to fetch full content (1–{MAX_URLS}). Only used with 'urls'.",
                "minItems": 1,
                "maxItems": MAX_URLS,
            },
        },
    }

    def run(
        self,
        query: str | None = None,
        max_results: int = DEFAULT_RESULTS,
        urls: list[str] | None = None,
    ) -> ToolResult:
        # If both provided, prefer query (search is the common first step).
        if query and urls:
            return self._search(query, max_results)
        if query is not None:
            return self._search(query, max_results)
        if urls is not None:
            return self._fetch(urls)
        return ToolResult(
            "web: provide 'query' (search) or 'urls' (fetch content).",
            is_error=True,
        )

    # -- search ----------------------------------------------------------------

    def _search(self, query: str, max_results: int) -> ToolResult:
        n = max(1, min(int(max_results), MAX_RESULTS))
        try:
            from ddgs import DDGS
        except ImportError:
            return ToolResult(
                "web search unavailable: the 'ddgs' package is not installed.",
                is_error=True,
            )
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=n))
        except Exception as e:
            return ToolResult(f"web search failed: {e}", is_error=True)

        if not hits:
            return ToolResult(
                f"No web results for: {query}",
                display=f"web search '{query}' — 0",
            )

        blocks = []
        for i, h in enumerate(hits, start=1):
            title = (h.get("title") or "").strip()
            url = (h.get("href") or "").strip()
            body = " ".join((h.get("body") or "").split())
            if len(body) > SNIPPET_CHARS:
                body = body[:SNIPPET_CHARS] + "…"
            blocks.append(f"{i}. {title}\n   {url}\n   {body}")

        out = "\n\n".join(blocks)
        return ToolResult(out, display=f"web search '{query}' — {len(hits)} results")

    # -- fetch ----------------------------------------------------------------

    def _fetch(self, urls: list[str]) -> ToolResult:
        if not isinstance(urls, list):
            return ToolResult("web: 'urls' must be a list of strings.", is_error=True)

        urls = urls[:MAX_URLS]
        blocks: list[str] = []
        errors: list[str] = []
        total_chars = 0

        for url in urls:
            try:
                content = self._fetch_url(url, MAX_CHARS)
            except Exception as e:
                errors.append(f"{url}: {e}")
                continue

            if content is None:
                errors.append(f"{url}: no extractable content found")
                continue

            total_chars += len(content)
            header = f"## {url}\n\n" if len(urls) > 1 else ""
            blocks.append(f"{header}{content}")

        out_parts: list[str] = []
        if blocks:
            out_parts.append("\n\n---\n\n".join(blocks))
        if errors:
            out_parts.append(
                "\n\n### Errors\n\n" + "\n".join(f"- {e}" for e in errors)
            )

        if not out_parts:
            return ToolResult(
                "web: all URLs failed — see errors above.",
                is_error=True,
            )

        result = "\n\n".join(out_parts)
        display = f"web fetch {len(blocks)}/{len(urls)} pages"
        return ToolResult(result, display=display)

    def _fetch_url(self, url: str, max_chars: int) -> str | None:
        """Fetch a single URL and return its Markdown-extracted content."""
        import requests

        resp = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
        )
        resp.raise_for_status()
        html = resp.text

        try:
            import trafilatura
        except ImportError:
            return self._fallback_extract(html, max_chars)

        extracted = trafilatura.extract(
            html,
            output_format="html",
            include_links=True,
            include_tables=True,
            include_images=False,
            favor_recall=True,
            no_fallback=False,
        )
        if not extracted:
            return self._fallback_extract(html, max_chars)

        try:
            import markdownify
        except ImportError:
            return self._html_to_text(extracted, max_chars)

        md = markdownify.markdownify(
            extracted,
            heading_style="ATX",
            bullets="-",
            strip=["img", "script", "style", "nav", "footer"],
        )
        return self._truncate(md, max_chars)

    def _fallback_extract(self, html: str, max_chars: int) -> str | None:
        """Fallback when trafilatura is not installed: extract <body> text."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return self._truncate(text, max_chars) if text.strip() else None

    def _html_to_text(self, html: str, max_chars: int) -> str:
        """Crude HTML-to-text when markdownify is absent."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return self._truncate(text, max_chars) if text.strip() else ""

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + f"\n\n{TRUNCATED_WARNING}"
