"""Search-and-replace file editing (content-anchored, no line numbers).

This format costs far fewer tokens than rewriting whole files and is more
reliable than unified diffs, which break on line-number drift. To survive
minor whitespace mismatches from the model we apply a small ladder of
fallbacks, from strictest to loosest:

  1. Exact substring match (must be unique).
  2. Trailing-whitespace-insensitive match per line.
  3. Leading+trailing (full strip) per-line match.

If more than one location matches at any level we refuse, so an ambiguous
edit never silently changes the wrong place.
"""

from __future__ import annotations

from .base import Tool, ToolResult


def _find_unique(haystack: str, needle: str) -> tuple[int, int] | str:
    """Return (start, end) of a unique match, or an error string."""
    # Level 1: exact.
    count = haystack.count(needle)
    if count == 1:
        start = haystack.index(needle)
        return start, start + len(needle)
    if count > 1:
        return f"'old_str' appears {count} times; add surrounding context to make it unique."

    # Levels 2 & 3: normalize whitespace per line and match on line boundaries.
    for strip in (str.rstrip, str.strip):
        span = _match_normalized(haystack, needle, strip)
        if isinstance(span, tuple):
            return span
        if span == "ambiguous":
            return "'old_str' matches multiple locations after whitespace normalization; add context."

    return "not found"


def _match_normalized(haystack: str, needle: str, strip) -> tuple[int, int] | str | None:
    hay_lines = haystack.split("\n")
    ndl_lines = needle.split("\n")
    n = len(ndl_lines)
    if n == 0:
        return None
    norm_ndl = [strip(l) for l in ndl_lines]

    # Precompute character offsets of each line start.
    offsets = []
    pos = 0
    for line in hay_lines:
        offsets.append(pos)
        pos += len(line) + 1  # +1 for the '\n'

    hits = []
    for i in range(0, len(hay_lines) - n + 1):
        if [strip(l) for l in hay_lines[i : i + n]] == norm_ndl:
            start = offsets[i]
            last = i + n - 1
            end = offsets[last] + len(hay_lines[last])
            hits.append((start, end))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return "ambiguous"
    return None


class EditTool(Tool):
    name = "edit"
    description = (
        "Replace an exact block of text in a file. 'old_str' must match the "
        "current file content (including indentation) and be unique. To insert, "
        "include enough surrounding context in old_str. Set replace_all=true to "
        "replace every occurrence of old_str."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File to edit."},
            "old_str": {"type": "string", "description": "Exact text to find (empty to create/overwrite is not allowed; use 'write')."},
            "new_str": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["path", "old_str", "new_str"],
    }

    def run(self, path: str, old_str: str, new_str: str, replace_all: bool = False) -> ToolResult:
        p = self.resolve(path)
        if not p.exists():
            return ToolResult(f"File not found: {path}. Use 'write' to create it.", is_error=True)
        if old_str == new_str:
            return ToolResult("old_str and new_str are identical; nothing to do.", is_error=True)
        if old_str == "":
            return ToolResult("old_str is empty; use the 'write' tool to create files.", is_error=True)

        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            return ToolResult(f"Could not read {path}: {e}", is_error=True)

        if replace_all:
            count = text.count(old_str)
            if count == 0:
                return ToolResult("old_str not found in file.", is_error=True)
            new_text = text.replace(old_str, new_str)
            self._write(p, new_text)
            return ToolResult(f"Replaced {count} occurrence(s) in {path}.", display=f"edit {path} ({count}×)")

        span = _find_unique(text, old_str)
        if isinstance(span, str):
            return ToolResult(f"Edit failed: {span}", is_error=True)
        start, end = span
        new_text = text[:start] + new_str + text[end:]
        self._write(p, new_text)

        added = new_str.count("\n") + 1
        removed = text[start:end].count("\n") + 1
        return ToolResult(
            f"Edited {path} (-{removed}/+{added} lines).",
            display=f"edit {path} (-{removed}/+{added})",
        )

    @staticmethod
    def _write(p, text: str) -> None:
        p.write_text(text, encoding="utf-8")
