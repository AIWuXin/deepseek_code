"""Optional diagnostic log for tracking down intermittent bugs.

Off by default — zero overhead when ``DSC_DEBUG`` is unset. Enable with:

    DSC_DEBUG=1 dsc ...

Events from the agent loop and the TUI event handler are appended to
``~/.dsc/debug.log`` with timestamps. When a "tool call vanished" repro comes
in, diff the two layers: every ``loop: yield tool_start`` should be followed by
a ``ui: recv tool_start``. The layer where the line goes missing is the bug.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

_enabled: bool | None = None
_path: Path | None = None


def _resolve() -> tuple[bool, Path | None]:
    """Lazily read the env var once and prepare the log path."""
    global _enabled, _path
    if _enabled is None:
        _enabled = bool(os.environ.get("DSC_DEBUG"))
        if _enabled:
            _path = Path.home() / ".dsc" / "debug.log"
            try:
                _path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                _path = None
                _enabled = False
    return _enabled, _path


def log(msg: str) -> None:
    """Append a timestamped line if DSC_DEBUG is set; otherwise a no-op."""
    enabled, path = _resolve()
    if not enabled or path is None:
        return
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except OSError:
        pass  # logging never breaks the app
