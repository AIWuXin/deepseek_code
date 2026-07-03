"""Create or overwrite a whole file."""

from __future__ import annotations

from .base import Tool, ToolResult


class WriteTool(Tool):
    name = "write"
    description = (
        "Create a new file or overwrite an existing one with the given content. "
        "For modifying part of an existing file, prefer 'edit' to save tokens."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write."},
            "content": {"type": "string", "description": "Full file content."},
        },
        "required": ["path", "content"],
    }

    def run(self, path: str, content: str) -> ToolResult:
        p = self.resolve(path)
        existed = p.exists()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except OSError as e:
            return ToolResult(f"Could not write {path}: {e}", is_error=True)

        lines = content.count("\n") + 1
        verb = "Overwrote" if existed else "Created"
        return ToolResult(f"{verb} {path} ({lines} lines).", display=f"write {path} ({lines} lines)")
