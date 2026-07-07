"""Read `git status` into a {relative-path: status-code} map for the tree.

Kept deliberately small and dependency-free: one subprocess call, a porcelain
parser, and a graceful empty result when the workspace isn't a git repo (or git
isn't installed). The TUI colours tree labels from this map; a missing map just
means no colouring, never an error.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Coarse status buckets the UI cares about. We collapse git's two-column XY
# porcelain codes into one of these — the tree only needs a colour, not the full
# staged/unstaged distinction.
MODIFIED = "modified"
ADDED = "added"      # staged new file
UNTRACKED = "untracked"
DELETED = "deleted"
RENAMED = "renamed"


def _classify(xy: str) -> str | None:
    """Map a porcelain XY status pair to one of our buckets.

    X is the staged (index) state, Y the worktree state. We report the most
    salient change; untracked ('??') and deletions win over plain modification
    so they stand out in the tree.
    """
    if xy == "??":
        return UNTRACKED
    x, y = xy[0], xy[1]
    if "D" in (x, y):
        return DELETED
    if "R" in (x, y):
        return RENAMED
    if "A" in (x, y):
        return ADDED
    if "M" in (x, y) or "T" in (x, y):
        return MODIFIED
    return MODIFIED  # any other tracked change → treat as modified


def parse_porcelain(output: str) -> dict[str, str]:
    """Parse `git status --porcelain` text into {path: bucket}.

    Paths are returned exactly as git prints them (repo-root-relative, forward
    slashes). Rename entries ("R  old -> new") are keyed on the new path.
    """
    result: dict[str, str] = {}
    for line in output.splitlines():
        if len(line) < 4:
            continue
        xy = line[:2]
        rest = line[3:]
        # Renames/copies are "old -> new"; the new name is what exists on disk.
        if " -> " in rest:
            rest = rest.split(" -> ", 1)[1]
        # git quotes paths with unusual chars; strip the surrounding quotes.
        path = rest.strip().strip('"')
        bucket = _classify(xy)
        if bucket:
            result[path] = bucket
    return result


def git_status(cwd: str) -> dict[str, str]:
    """Return {repo-relative-path: bucket} for `cwd`, or {} if not a git repo.

    Never raises: any failure (not a repo, git missing, timeout) yields {} so
    the caller can treat "no git info" and "clean tree" identically.
    """
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "-uall"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0:
        return {}  # e.g. "not a git repository"
    return parse_porcelain(proc.stdout)


def git_root(cwd: str) -> Path | None:
    """Absolute path of the repo root containing `cwd`, or None if not a repo.

    Needed to turn git's repo-relative paths into absolute paths that match the
    tree's node paths (the tree may be rooted at a subdirectory of the repo).
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    top = proc.stdout.strip()
    return Path(top) if top else None
