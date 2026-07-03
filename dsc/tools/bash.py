"""Shell command execution with truncated, budget-bounded output.

On Windows the model tends to emit Linux-style commands (ls, grep, cat, rm -rf,
&&-chaining), which cmd.exe/PowerShell reject. So we run commands through Git
Bash when it's available, giving a consistent POSIX shell across platforms.
Falls back to the platform default shell if Git Bash isn't found.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from .base import Tool, ToolResult

MAX_OUTPUT_CHARS = 30_000
DEFAULT_TIMEOUT = 120

# Common Git-for-Windows install locations, checked in order. We prefer the
# top-level bin\bash.exe (the Git Bash launcher) over usr\bin\bash.exe (the raw
# MSYS shell) so the environment matches what the user sees in a Git Bash tab.
# The env override DSC_BASH_PATH wins if set.
_GIT_BASH_CANDIDATES = [
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files (x86)\Git\bin\bash.exe",
    r"D:\Git\bin\bash.exe",
    r"D:\Program Files\Git\bin\bash.exe",
]


def _find_bash() -> str | None:
    """Locate a bash.exe (prefer Git Bash's bin\\bash.exe) on Windows, else None."""
    override = os.environ.get("DSC_BASH_PATH")
    if override and os.path.isfile(override):
        return override
    # Prefer known Git-Bash launcher paths before whatever is on PATH (which is
    # often the MSYS usr\bin\bash inside this very session).
    for cand in _GIT_BASH_CANDIDATES:
        if os.path.isfile(cand):
            return cand
    return shutil.which("bash")


def _truncate_middle(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the head and tail, drop the middle — errors usually live at both ends."""
    if len(text) <= limit:
        return text
    half = limit // 2
    head = text[:half]
    tail = text[-half:]
    dropped = len(text) - 2 * half
    return f"{head}\n\n… [{dropped} chars truncated] …\n\n{tail}"


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a shell command and return combined stdout/stderr. On Windows this "
        "runs in Git Bash, so use POSIX/Linux-style commands (ls, grep, cat, "
        "&&-chaining, forward-slash paths). Output is truncated in the middle if "
        "very long. Use for builds, tests, git, and file operations. Avoid "
        "interactive commands."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute (POSIX/bash syntax)."},
            "timeout": {"type": "integer", "description": f"Seconds before kill (default {DEFAULT_TIMEOUT}).", "default": DEFAULT_TIMEOUT},
        },
        "required": ["command"],
    }

    def __init__(self, root: str):
        super().__init__(root)
        self._bash = _find_bash()

    def run(self, command: str, timeout: int = DEFAULT_TIMEOUT) -> ToolResult:
        timeout = max(1, min(int(timeout), 600))
        if self._bash:
            # -c runs the command string; skip -l so login-profile noise doesn't
            # pollute output. We still get the full Git Bash PATH.
            argv = [self._bash, "-c", command]
            popen_kwargs = {"cwd": str(self.root)}
        else:
            # Fall back to the platform default shell.
            argv = command
            popen_kwargs = {"cwd": str(self.root), "shell": True}

        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                # Force UTF-8; the Windows default (GBK here) crashes on UTF-8
                # output from tools. errors='replace' avoids hard failures.
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                **popen_kwargs,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(f"Command timed out after {timeout}s.", is_error=True)
        except OSError as e:
            return ToolResult(f"Failed to run command: {e}", is_error=True)

        combined = (proc.stdout or "") + (proc.stderr or "")
        combined = _truncate_middle(combined.strip())
        code = proc.returncode
        header = f"(exit {code})\n" if code != 0 else ""
        body = combined if combined else "(no output)"
        display = f"bash: {command[:50]}{'…' if len(command) > 50 else ''} (exit {code})"
        return ToolResult(header + body, display=display, is_error=(code != 0))
