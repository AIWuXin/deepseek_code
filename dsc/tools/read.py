"""Read a file with line/character truncation to bound token cost."""

from __future__ import annotations

from .base import Tool, ToolResult

MAX_LINES = 2000
MAX_LINE_CHARS = 2000


class ReadTool(Tool):
    name = "read"
    description = (
        "Read a text file from the workspace. Returns numbered lines. "
        "Long files are truncated; pass 'offset' and 'limit' to page through. "
        "Prefer grep/glob to locate content before reading whole files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (relative to workspace or absolute)."},
            "offset": {"type": "integer", "description": "1-based line to start from.", "default": 1},
            "limit": {"type": "integer", "description": f"Max lines to read (<= {MAX_LINES}).", "default": MAX_LINES},
        },
        "required": ["path"],
    }

    def run(self, path: str, offset: int = 1, limit: int = MAX_LINES) -> ToolResult:
        p = self.resolve(path)
        if not p.exists():
            return ToolResult(f"File not found: {path}", is_error=True)
        if p.is_dir():
            return ToolResult(f"Is a directory, not a file: {path}", is_error=True)

        offset = max(1, offset)
        limit = max(1, min(limit, MAX_LINES))
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return ToolResult(f"Could not read {path}: {e}", is_error=True)

        lines = text.splitlines()
        total = len(lines)
        window = lines[offset - 1 : offset - 1 + limit]

        out = []
        for i, line in enumerate(window, start=offset):
            if len(line) > MAX_LINE_CHARS:
                line = line[:MAX_LINE_CHARS] + f"… [+{len(line) - MAX_LINE_CHARS} chars truncated]"
            out.append(f"{i:>6}\t{line}")

        body = "\n".join(out) if out else "(empty range)"
        end = offset - 1 + len(window)
        note = ""
        if end < total:
            note = f"\n\n[showing lines {offset}-{end} of {total}; use offset={end + 1} to continue]"
        display = f"read {path} ({len(window)} of {total} lines)"
        return ToolResult(body + note, display=display)
