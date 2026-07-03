"""Bash tool: middle truncation and command execution (git bash or fallback)."""

from __future__ import annotations

from dsc.tools.bash import MAX_OUTPUT_CHARS, BashTool, _truncate_middle


def test_truncate_middle_keeps_head_and_tail():
    text = "A" * 20_000 + "B" * 20_000
    out = _truncate_middle(text, limit=1000)
    assert out.startswith("A")
    assert out.rstrip().endswith("B")
    assert "truncated" in out
    assert len(out) < len(text)


def test_short_output_not_truncated():
    assert _truncate_middle("hello", limit=1000) == "hello"


def test_echo_runs(tmp_path):
    tool = BashTool(str(tmp_path))
    res = tool.run(command="echo hello-from-shell")
    assert not res.is_error
    assert "hello-from-shell" in res.content


def test_nonzero_exit_marked_error(tmp_path):
    tool = BashTool(str(tmp_path))
    res = tool.run(command="exit 3")
    assert res.is_error
    assert "exit 3" in res.content


def test_cwd_is_workspace_root(tmp_path):
    (tmp_path / "marker.txt").write_text("x", encoding="utf-8")
    tool = BashTool(str(tmp_path))
    res = tool.run(command="ls")
    assert "marker.txt" in res.content
