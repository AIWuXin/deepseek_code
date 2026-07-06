"""Tool abstraction and registry.

A Tool exposes an OpenAI-compatible function schema and a synchronous
``run`` method. Results carry both the text handed back to the model and an
optional short display line for the TUI, so we never have to re-derive a
human summary from the raw payload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolResult:
    """Outcome of a tool call.

    content: text returned to the model (already truncated to budget).
    display: one-line summary for the TUI (e.g. "read foo.py (120 lines)").
    is_error: marks failures so the model can react.
    """

    content: str
    display: str = ""
    is_error: bool = False


class Tool:
    name: str = ""
    description: str = ""
    # JSON Schema for parameters (OpenAI function-calling format).
    parameters: dict = {}

    def __init__(self, root: str):
        # All relative paths resolve against this working-directory root.
        self.root = Path(root).resolve()

    def resolve(self, path: str) -> Path:
        """Resolve a possibly-relative path under the workspace root."""
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        return p

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, **kwargs) -> ToolResult:  # pragma: no cover - interface
        raise NotImplementedError


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._order: list[str] = []

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._order.append(tool.name)

    def schemas(self) -> list[dict]:
        """Tool schemas in stable registration order (prefix-cache friendly)."""
        return [self._tools[n].schema() for n in self._order]

    def execute(self, name: str, arguments: str | dict) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"Unknown tool: {name}", is_error=True)
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError as e:
            return ToolResult(f"Invalid tool arguments JSON: {e}", is_error=True)
        try:
            return tool.run(**args)
        except TypeError as e:
            return ToolResult(f"Bad arguments for {name}: {e}", is_error=True)
        except Exception as e:  # tools must never crash the loop
            return ToolResult(f"{name} failed: {e}", is_error=True)

    def get(self, name: str) -> Tool | None:
        """Look up a registered tool by name."""
        return self._tools.get(name)
