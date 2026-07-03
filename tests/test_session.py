"""Session persistence: JSONL append/load roundtrip and discovery helpers."""

from __future__ import annotations

import pytest

from dsc.session import store as store_mod
from dsc.session.store import SessionStore, describe_session


@pytest.fixture(autouse=True)
def _tmp_sessions(tmp_path, monkeypatch):
    """Redirect the sessions dir to a tmp folder so tests never touch ~/.dsc."""
    monkeypatch.setattr(store_mod, "SESSIONS_DIR", tmp_path / "sessions")


def test_append_load_roundtrip():
    s = SessionStore("sess-a")
    records = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "ok"},
    ]
    for r in records:
        s.append(r)
    assert SessionStore("sess-a").load() == records


def test_load_skips_torn_final_line():
    s = SessionStore("torn")
    s.append({"role": "user", "content": "one"})
    # Simulate a crash mid-write leaving an incomplete JSON line.
    with s.path.open("a", encoding="utf-8") as f:
        f.write('{"role": "assistant", "content": "trunc')
    loaded = SessionStore("torn").load()
    assert loaded == [{"role": "user", "content": "one"}]


def test_name_property_matches_stem():
    assert SessionStore("my-session").name == "my-session"


def test_list_sessions_newest_first():
    import os

    older = SessionStore("older")
    older.append({"role": "user", "content": "x"})
    newer = SessionStore("newer")
    newer.append({"role": "user", "content": "y"})
    # Force a clear mtime ordering.
    os.utime(older.path, (1, 1))
    os.utime(newer.path, (2, 2))
    names = [p.stem for p in SessionStore.list_sessions()]
    assert names.index("newer") < names.index("older")


def test_from_name_missing_returns_none():
    assert SessionStore.from_name("does-not-exist") is None


def test_from_name_and_latest():
    s = SessionStore("only")
    s.append({"role": "user", "content": "x"})
    assert SessionStore.from_name("only").name == "only"
    assert SessionStore.latest().name == "only"


def test_latest_none_when_empty():
    assert SessionStore.latest() is None


def test_describe_session_title_and_summary():
    s = SessionStore("described")
    s.append({"role": "user", "content": "<environment>\ncwd: /tmp"})  # seed, skipped
    s.append({"role": "user", "content": "fix the login crash"})
    s.append({"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]})  # empty, skipped
    s.append({"role": "tool", "tool_call_id": "1", "content": "..."})
    s.append({"role": "assistant", "content": "Done — patched the null check."})

    info = describe_session(s.path)
    assert info.name == "described"
    # Title is the first *real* user turn, not the environment seed.
    assert info.title == "fix the login crash"
    # Summary is the latest non-empty assistant reply.
    assert info.summary == "Done — patched the null check."
    assert info.count == 5


def test_describe_session_defaults_when_no_content():
    s = SessionStore("empty-ish")
    s.append({"role": "user", "content": "<environment>\ncwd: /tmp"})
    info = describe_session(s.path)
    assert info.title == "(no title)"
    assert info.summary == ""


def test_delete_removes_jsonl_and_title_sidecar():
    s = SessionStore("to-delete")
    s.append({"role": "user", "content": "x"})
    s.save_title("Some Title")
    assert s.path.exists()
    assert s.read_title() == "Some Title"

    assert SessionStore.delete("to-delete") is True
    assert not s.path.exists()
    assert s.read_title() is None
    assert "to-delete" not in [i.name for i in SessionStore.infos()]


def test_delete_missing_returns_false():
    assert SessionStore.delete("never-existed") is False
