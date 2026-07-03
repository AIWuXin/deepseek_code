"""Reliability tests for the search-replace edit tool.

The edit tool is the highest-risk component: a wrong match silently corrupts
the user's code. These tests pin down the match ladder and the refusal cases.
"""

from __future__ import annotations

import pytest

from dsc.tools.edit import EditTool, _find_unique


def test_exact_unique():
    text = "line1\nhello world\nline3\n"
    span = _find_unique(text, "hello world")
    assert isinstance(span, tuple)
    assert text[span[0] : span[1]] == "hello world"


def test_duplicate_refused():
    text = "x = 1\ny = 1\n"
    span = _find_unique(text, "= 1")
    assert isinstance(span, str)
    assert "appears" in span


def test_not_found():
    assert _find_unique("abc", "xyz") == "not found"


def test_trailing_whitespace_fallback():
    # Model omits trailing spaces that exist in the file.
    text = "def f():   \n    return 1\n"
    span = _find_unique(text, "def f():\n    return 1")
    assert isinstance(span, tuple)


def test_edit_applies(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def f():\n    return 1\n", encoding="utf-8")
    tool = EditTool(str(tmp_path))
    res = tool.run(path="a.py", old_str="return 1", new_str="return 2")
    assert not res.is_error
    assert f.read_text(encoding="utf-8") == "def f():\n    return 2\n"


def test_edit_missing_file(tmp_path):
    tool = EditTool(str(tmp_path))
    res = tool.run(path="nope.py", old_str="a", new_str="b")
    assert res.is_error
    assert "not found" in res.content.lower()


def test_edit_identical_refused(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x\n", encoding="utf-8")
    tool = EditTool(str(tmp_path))
    res = tool.run(path="a.py", old_str="x", new_str="x")
    assert res.is_error


def test_replace_all(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("a\na\na\n", encoding="utf-8")
    tool = EditTool(str(tmp_path))
    res = tool.run(path="a.py", old_str="a", new_str="b", replace_all=True)
    assert not res.is_error
    assert f.read_text(encoding="utf-8") == "b\nb\nb\n"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
