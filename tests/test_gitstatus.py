"""Unit tests for the porcelain parser that drives sidebar git colouring."""

from __future__ import annotations

from dsc.tui import gitstatus


def test_parse_porcelain_buckets():
    out = gitstatus.parse_porcelain(
        " M dsc/app.py\n"
        "?? new.txt\n"
        "A  staged.py\n"
        " D gone.py\n"
        "R  old.py -> renamed.py\n"
    )
    assert out == {
        "dsc/app.py": gitstatus.MODIFIED,
        "new.txt": gitstatus.UNTRACKED,
        "staged.py": gitstatus.ADDED,
        "gone.py": gitstatus.DELETED,
        "renamed.py": gitstatus.RENAMED,
    }


def test_parse_porcelain_ignores_blank_and_short_lines():
    assert gitstatus.parse_porcelain("") == {}
    assert gitstatus.parse_porcelain("\n\nx\n") == {}


def test_parse_porcelain_strips_quoted_paths():
    out = gitstatus.parse_porcelain(' M "spaced name.py"\n')
    assert out == {"spaced name.py": gitstatus.MODIFIED}


def test_rename_keys_on_new_path():
    out = gitstatus.parse_porcelain("R  a/old.py -> a/new.py\n")
    assert "a/new.py" in out and "a/old.py" not in out


def test_git_status_non_repo_returns_empty(tmp_path):
    # A bare temp dir isn't a git repo → graceful empty, never raises.
    assert gitstatus.git_status(str(tmp_path)) == {}
    assert gitstatus.git_root(str(tmp_path)) is None
