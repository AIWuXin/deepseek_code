"""Content search backed by ripgrep.

Returns matching lines (optionally with context) rather than whole files, so
we spend tokens only on what matched. Falls back to a pure-Python scan if
ripgrep is not on PATH.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from .base import Tool, ToolResult

MAX_MATCHES = 200
# Directories the Python fallback should never descend into.
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".mypy_cache"}


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search file contents with a regular expression (ripgrep). "
        "Returns matching lines with file:line prefixes, not whole files. "
        "Use 'glob' to filter by path and 'context' for surrounding lines."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for."},
            "path": {"type": "string", "description": "Directory or file to search (default: workspace root)."},
            "glob": {"type": "string", "description": "Glob to filter files, e.g. '*.py'."},
            "context": {"type": "integer", "description": "Lines of context before/after each match.", "default": 0},
            "ignore_case": {"type": "boolean", "default": False},
        },
        "required": ["pattern"],
    }

    def run(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        context: int = 0,
        ignore_case: bool = False,
    ) -> ToolResult:
        target = self.resolve(path) if path else self.root
        rg = shutil.which("rg") or shutil.which("rg.exe")
        if rg is None:
            # No ripgrep on PATH — fall back to a pure-Python scan so search
            # always works, just slower and without .gitignore awareness.
            return self._python_grep(pattern, target, glob, context, ignore_case)

        cmd = [rg, "--line-number", "--no-heading", "--color", "never"]
        if ignore_case:
            cmd.append("--ignore-case")
        if context and context > 0:
            cmd += ["--context", str(min(context, 5))]
        if glob:
            cmd += ["--glob", glob]
        cmd += ["--max-count", str(MAX_MATCHES), "--", pattern, str(target)]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return ToolResult("grep timed out after 30s.", is_error=True)

        if proc.returncode not in (0, 1):  # 1 == no matches, not an error
            return ToolResult(f"grep error: {proc.stderr.strip()}", is_error=True)

        out = proc.stdout.rstrip("\n")
        if not out:
            return ToolResult(f"No matches for /{pattern}/.", display=f"grep /{pattern}/ — 0 matches")

        lines = out.splitlines()
        truncated = ""
        if len(lines) >= MAX_MATCHES:
            truncated = f"\n\n[truncated at {MAX_MATCHES} matches; narrow the pattern or path]"
        display = f"grep /{pattern}/ — {len(lines)} lines"
        return ToolResult(out + truncated, display=display)

    def _python_grep(
        self,
        pattern: str,
        target: Path,
        glob: str | None,
        context: int,
        ignore_case: bool,
    ) -> ToolResult:
        try:
            rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            return ToolResult(f"Invalid regex: {e}", is_error=True)

        if target.is_file():
            files = [target]
        else:
            pat = glob or "**/*"
            files = [
                p
                for p in target.glob(pat)
                if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)
            ]

        ctx = min(context or 0, 5)
        out_lines: list[str] = []
        for f in files:
            try:
                lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            try:
                rel = f.relative_to(self.root)
            except ValueError:
                rel = f
            for i, line in enumerate(lines):
                if rx.search(line):
                    lo = max(0, i - ctx)
                    hi = min(len(lines), i + ctx + 1)
                    for j in range(lo, hi):
                        sep = ":" if j == i else "-"
                        out_lines.append(f"{rel}{sep}{j + 1}{sep}{lines[j]}")
                    if len(out_lines) >= MAX_MATCHES:
                        break
            if len(out_lines) >= MAX_MATCHES:
                break

        if not out_lines:
            return ToolResult(f"No matches for /{pattern}/.", display=f"grep /{pattern}/ — 0 matches")
        body = "\n".join(out_lines[:MAX_MATCHES])
        truncated = f"\n\n[truncated at {MAX_MATCHES} matches; narrow the pattern]" if len(out_lines) >= MAX_MATCHES else ""
        return ToolResult(body + truncated, display=f"grep /{pattern}/ — {len(out_lines)} lines (py)")
