"""Session persistence: JSONL append/load roundtrip and discovery helpers."""

from __future__ import annotations

import pytest

from dsc.session import store as store_mod
from dsc.session.store import SessionStore, describe_session, ArchiveBlock


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


def test_replace_rewrites_jsonl():
    s = SessionStore("replace-test")
    s.append({"role": "user", "content": "old"})
    s.replace([
        {"role": "user", "content": "new1"},
        {"role": "assistant", "content": "new2"},
    ])
    assert s.load() == [
        {"role": "user", "content": "new1"},
        {"role": "assistant", "content": "new2"},
    ]
    assert s.path.read_text(encoding="utf-8").count("\n") == 2


# -- Phase 1: archive ---------------------------------------------------------

def test_archive_block_roundtrip():
    s = SessionStore("arc-test")
    block = ArchiveBlock(
        id=0,
        summary="Added web_fetch",
        keywords="web_fetch, trafilatura",
        in_context_summary="Added a web_fetch tool using trafilatura.",
        messages=[{"role": "user", "content": "add it"}],
    )
    s.archive_block(block)

    loaded = s.load_block(0)
    assert loaded is not None
    assert loaded["summary"] == "Added web_fetch"
    assert loaded["keywords"] == "web_fetch, trafilatura"
    assert loaded["in_context_summary"] == "Added a web_fetch tool using trafilatura."
    assert len(loaded["messages"]) == 1


def test_search_blocks_by_keyword():
    s = SessionStore("arc-search")
    b1 = ArchiveBlock(0, "Fixed login bug", "login, auth, bug", "Fixed null ptr in login", [])
    b2 = ArchiveBlock(1, "Added API endpoint", "api, endpoint, flask", "Created /users endpoint", [])
    s.archive_block(b1)
    s.archive_block(b2)

    hits = s.search_blocks("login")
    assert len(hits) == 1
    assert hits[0]["id"] == 0

    hits = s.search_blocks("api endpoint")
    assert len(hits) == 1
    assert hits[0]["id"] == 1

    hits = s.search_blocks("nonexistent")
    assert len(hits) == 0


def test_list_blocks_returns_all():
    s = SessionStore("arc-list")
    s.archive_block(ArchiveBlock(0, "A", "a", "a", []))
    s.archive_block(ArchiveBlock(1, "B", "b", "b", []))
    blocks = s.list_blocks()
    assert [b["id"] for b in blocks] == [0, 1]


# -- Phase 2: read_archive tool -----------------------------------------------

def test_read_archive_tool_schema():
    from dsc.tools.read_archive import ReadArchiveTool
    tool = ReadArchiveTool(root=".")
    schema = tool.schema()
    assert schema["function"]["name"] == "read_archive"
    props = schema["function"]["parameters"]["properties"]
    assert "search" in props
    assert "id" in props


def test_read_archive_search(tmp_path):
    from dsc.tools.read_archive import ReadArchiveTool

    # Create an archive block.
    arc_dir = tmp_path / "sess1_arc"
    arc_dir.mkdir(parents=True)
    (arc_dir / "0000.json").write_text(
        '{"id": 0, "summary": "Added web_fetch", "keywords": "tool, fetch, web", "in_context_summary": "", "messages": []}',
        encoding="utf-8",
    )

    tool = ReadArchiveTool(root=".")
    tool._archive_dir = arc_dir
    result = tool.run(search="web_fetch")
    assert not result.is_error
    assert "Added web_fetch" in result.content

    result = tool.run(search="nonexistent")
    assert not result.is_error
    assert "0" in result.display  # "read_archive search: 0"


def test_read_archive_read_block(tmp_path):
    from dsc.tools.read_archive import ReadArchiveTool

    arc_dir = tmp_path / "sess2_arc"
    arc_dir.mkdir(parents=True)
    (arc_dir / "0001.json").write_text(
        '{"id": 1, "summary": "Fixed bug", "in_context_summary": "Fixed null ptr", '
        '"messages": [{"role": "user", "content": "fix it"}, {"role": "assistant", "content": "done"}]}',
        encoding="utf-8",
    )

    tool = ReadArchiveTool(root=".")
    tool._archive_dir = arc_dir
    result = tool.run(id=1)
    assert not result.is_error
    assert "[Archive #1]" in result.content
    assert "Fixed bug" in result.content

    # Missing block.
    result = tool.run(id=99)
    assert result.is_error


