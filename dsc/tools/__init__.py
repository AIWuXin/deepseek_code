"""Tool implementations for the agent.

The registry order is fixed and stable so the serialized tool schemas — which
sit at the very front of every request — stay byte-identical across turns and
keep hitting DeepSeek's prefix cache.
"""

from __future__ import annotations

from .base import Tool, ToolRegistry, ToolResult
from .read import ReadTool
from .grep import GrepTool
from .glob import GlobTool
from .edit import EditTool
from .write import WriteTool
from .bash import BashTool
from .web import WebTool
from .read_archive import ReadArchiveTool


def build_registry(root: str) -> ToolRegistry:
    """Build the default tool registry rooted at the given working directory."""
    reg = ToolRegistry()
    # Order matters: keep it deterministic for prefix-cache stability.
    reg.register(ReadTool(root))
    reg.register(GrepTool(root))
    reg.register(GlobTool(root))
    reg.register(EditTool(root))
    reg.register(WriteTool(root))
    reg.register(BashTool(root))
    reg.register(WebTool(root))
    reg.register(ReadArchiveTool(root))
    return reg


__all__ = ["Tool", "ToolRegistry", "ToolResult", "build_registry"]
