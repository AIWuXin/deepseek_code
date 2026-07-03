"""Append-only JSONL session log, so a conversation can be inspected or resumed.

One JSON object per line keeps writes cheap and crash-safe (a truncated last
line is simply skipped on load). This is deliberately separate from the live
message list in ContextManager: the store is the full record, while the context
manager holds the possibly-compacted working set.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import CONFIG_DIR

SESSIONS_DIR = CONFIG_DIR / "sessions"


def _first_line(text: str, limit: int) -> str:
    line = " ".join((text or "").split())
    return line[:limit] + ("…" if len(line) > limit else "")


@dataclass
class SessionInfo:
    """Human-facing description of a session, derived from its content.

    The filename stays an opaque timestamp (stable + unique), but users pick
    sessions by a title (their first request) and a summary (the agent's last
    reply) — not by the numeric name.
    """

    name: str
    title: str
    summary: str
    count: int
    mtime: float


def _title_path(jsonl_path: Path) -> Path:
    """Sidecar file holding a model-generated title (kept out of the JSONL so it
    never gets loaded back into the conversation on resume)."""
    return jsonl_path.with_suffix(".title")


def read_title(jsonl_path: Path) -> str | None:
    tp = _title_path(jsonl_path)
    if tp.exists():
        text = tp.read_text(encoding="utf-8").strip()
        return text or None
    return None


def describe_session(path: Path) -> SessionInfo:
    """Read a session file and derive a title + last-reply summary.

    Prefers the model-generated sidecar title; falls back to the first user
    request when no title has been generated yet.
    """
    generated = read_title(path)
    title = ""
    summary = ""
    count = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                count += 1
                role = rec.get("role")
                content = rec.get("content") or ""
                # Title = first real user turn (skip the <environment> seed).
                if not title and role == "user" and not content.startswith("<environment>"):
                    title = _first_line(content, 48)
                # Summary = latest non-empty assistant reply.
                if role == "assistant" and content.strip():
                    summary = _first_line(content, 60)
    except OSError:
        pass
    return SessionInfo(
        name=path.stem,
        title=generated or title or "(no title)",
        summary=summary,
        count=count,
        mtime=path.stat().st_mtime if path.exists() else 0.0,
    )


class SessionStore:
    def __init__(self, name: str | None = None):
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        # A monotonic-ish name without relying on wall clock for uniqueness.
        stamp = name or time.strftime("%Y%m%d-%H%M%S")
        self.path = SESSIONS_DIR / f"{stamp}.jsonl"
        self._name = self.path.stem  # filename without .jsonl

    @property
    def name(self) -> str:
        return self._name

    def append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_title(self) -> str | None:
        return read_title(self.path)

    def save_title(self, title: str) -> None:
        _title_path(self.path).write_text(title.strip(), encoding="utf-8")

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip a torn final line
        return out

    @classmethod
    def list_sessions(cls) -> list[Path]:
        """Return session paths sorted newest-first."""
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        paths = sorted(SESSIONS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        return paths

    @classmethod
    def infos(cls) -> list[SessionInfo]:
        """Human-facing descriptions of all sessions, newest-first."""
        return [describe_session(p) for p in cls.list_sessions()]

    @classmethod
    def latest(cls) -> SessionStore | None:
        """Return the most recently modified session, if any."""
        paths = cls.list_sessions()
        if not paths:
            return None
        store = cls.__new__(cls)
        store.path = paths[0]
        store._name = store.path.stem
        return store

    @classmethod
    def delete(cls, name: str) -> bool:
        """Delete a session's JSONL log and its title sidecar. Returns success."""
        path = SESSIONS_DIR / f"{name}.jsonl"
        existed = path.exists()
        path.unlink(missing_ok=True)
        _title_path(path).unlink(missing_ok=True)
        return existed

    @classmethod
    def from_name(cls, name: str) -> SessionStore | None:
        """Return a session by name (stem of the .jsonl file)."""
        path = SESSIONS_DIR / f"{name}.jsonl"
        if not path.exists():
            return None
        store = cls.__new__(cls)
        store.path = path
        store._name = name
        return store
