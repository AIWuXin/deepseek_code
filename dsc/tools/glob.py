"""Filename matching by glob pattern, sorted by modification time."""

from __future__ import annotations

from .base import Tool, ToolResult

MAX_RESULTS = 200
# Directories we never want to walk into — noise that wastes tokens.
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".mypy_cache"}


class GlobTool(Tool):
    name = "glob"
    description = (
        "Find files by glob pattern (e.g. '**/*.py', 'src/**/*.ts'). "
        "Returns paths sorted by modification time (newest first)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern relative to the search path."},
            "path": {"type": "string", "description": "Base directory to search (default: workspace root)."},
        },
        "required": ["pattern"],
    }

    def run(self, pattern: str, path: str | None = None) -> ToolResult:
        base = self.resolve(path) if path else self.root
        if not base.exists():
            return ToolResult(f"Path not found: {path}", is_error=True)

        matches = []
        for p in base.glob(pattern):
            if not p.is_file():
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            matches.append(p)

        if not matches:
            return ToolResult(f"No files match {pattern}.", display=f"glob {pattern} — 0 files")

        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        shown = matches[:MAX_RESULTS]
        rels = []
        for p in shown:
            try:
                rels.append(str(p.relative_to(self.root)))
            except ValueError:
                rels.append(str(p))

        body = "\n".join(rels)
        if len(matches) > MAX_RESULTS:
            body += f"\n\n[showing {MAX_RESULTS} of {len(matches)}; narrow the pattern]"
        return ToolResult(body, display=f"glob {pattern} — {len(matches)} files")
