"""Fetch the full text content of one or more web pages.

Uses ``requests`` to fetch the page, ``trafilatura`` to extract the main
article content (stripping navigation, ads, sidebars), and ``markdownify`` to
convert the result to Markdown so the model reads clean prose, not raw HTML.
"""

from __future__ import annotations

from .base import Tool, ToolResult

MAX_URLS = 5
MAX_CHARS = 80_000  # per-URL cap, roughly 20k tokens
TRUNCATED_WARNING = "[truncated — content exceeds limit]"


class WebFetchTool(Tool):
    name = "web_fetch"
    description = (
        "Fetch and read the full text content of a web page. "
        "Extracts the main article (strips ads, nav, sidebars) and returns it "
        "as Markdown. Pass up to 5 URLs at once."
    )
    parameters = {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string", "format": "uri"},
                "description": f"URL(s) to fetch (1–{MAX_URLS}).",
                "minItems": 1,
                "maxItems": MAX_URLS,
            },
        },
        "required": ["urls"],
    }

    def run(self, urls: list[str]) -> ToolResult:
        if not urls or not isinstance(urls, list):
            return ToolResult("web_fetch: 'urls' must be a non-empty list of strings.", is_error=True)

        urls = urls[:MAX_URLS]
        blocks: list[str] = []
        errors: list[str] = []
        total_chars = 0

        for url in urls:
            try:
                content = self._fetch(url, MAX_CHARS)
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
                "web_fetch: all URLs failed — see logs above.",
                is_error=True,
            )

        result = "\n\n".join(out_parts)
        display = f"web_fetch {len(blocks)}/{len(urls)} pages"
        return ToolResult(result, display=display)

    # -- helpers ---------------------------------------------------------------

    def _fetch(self, url: str, max_chars: int) -> str | None:
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

        # trafilatura extracts the main article content as HTML.
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

        # Convert extracted HTML to Markdown for the model.
        try:
            import markdownify
        except ImportError:
            # Return plain text if markdownify isn't available.
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

    def _truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + f"\n\n{TRUNCATED_WARNING}"
