"""Search and read archived task blocks from disk.

Two modes:
  search(query)  — search archive block summaries/keywords, return matching
                   block IDs and one-line summaries.
  read(id)       — load a full archive block (including original messages)
                   and return its content.  Pre-compression is handled by
                   the loop layer (not inside this tool) so the tool stays
                   stateless.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import CONFIG_DIR
from .base import Tool, ToolResult


class ReadArchiveTool(Tool):
    name = "read_archive"
    description = (
        "Search and read archived task blocks. "
        "Use `read_archive(search=...)` to find relevant blocks by keyword "
        "and get a list of matching block IDs with summaries. "
        "Use `read_archive(id=...)` to load the full content of one block. "
        "Archives let you revisit earlier decisions without keeping every "
        "raw message in context."
    )
    parameters = {
        "type": "object",
        "properties": {
            "search": {
                "type": "string",
                "description": "Search query — returns matching block IDs + summaries. "
                               "Pass the archive ID from the search result to `read`.",
            },
            "id": {
                "type": "integer",
                "description": "Archive block ID to read (returned by `search`). "
                               "Returns the full archived conversation.",
            },
        },
        # Provide exactly one of search or id (enforced in run()).
    }

    def __init__(self, root: str, archive_dir: str = ""):
        super().__init__(root)
        self._archive_dir = Path(archive_dir) if archive_dir else CONFIG_DIR / "sessions"

    def set_archive_root(self, session_name: str) -> None:
        """Set the archive directory from a session name (called after store init)."""
        self._archive_dir = CONFIG_DIR / "sessions" / f"{session_name}_arc"

    # -- tool run -------------------------------------------------------------

    def run(self, search: str | None = None, id: int | None = None) -> ToolResult:
        if search is not None:
            return self._search(search)
        if id is not None:
            return self._read(id)
        return ToolResult("read_archive: provide either 'search' or 'id'.", is_error=True)

    def _search(self, query: str) -> ToolResult:
        """Search archive summaries/keywords and return matching blocks."""
        terms = [t.lower() for t in query.strip().split()]
        if not terms:
            return ToolResult("read_archive: empty search query.", is_error=True)

        hits = []
        for p in sorted(self._archive_dir.glob("*.json")):
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            text = (
                data.get("summary", "") + " " + data.get("keywords", "")
            ).lower()
            if all(t in text for t in terms):
                hits.append({"id": data["id"], "summary": data.get("summary", "")})

        if not hits:
            return ToolResult(
                f"No archived tasks match: {query}",
                display=f"read_archive search: 0",
            )

        lines = [f"Matched archived tasks for '{query}':"]
        for h in hits:
            lines.append(f"  #{h['id']} — {h['summary']}")
        return ToolResult(
            "\n".join(lines),
            display=f"read_archive search: {len(hits)}",
        )

    def _read(self, block_id: int) -> ToolResult:
        """Load a full archive block and return its content (raw, pre-compression
        happens in the loop layer)."""
        p = self._archive_dir / f"{block_id:04d}.json"
        if not p.exists():
            return ToolResult(
                f"read_archive: block #{block_id} not found.",
                is_error=True,
            )
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            return ToolResult(f"read_archive: failed to load block #{block_id}: {e}", is_error=True)

        # Reconstruct the conversation from stored messages.
        messages = data.get("messages", [])
        summary = data.get("summary", "")
        key_info = data.get("in_context_summary", "")
        lines = [f"[Archive #{block_id}] {summary}", f"  Key: {key_info}", ""]
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content", "")
            if role == "tool" and len(content) > 500:
                content = content[:500] + "\n  ... [truncated]"
            lines.append(f"  [{role}] {content}")
        out = "\n".join(lines)
        return ToolResult(
            out,
            display=f"read_archive #{block_id} — {summary}",
        )
