"""Markdown export of a stored conversation (pure formatter)."""

from __future__ import annotations

from dsc.session.export import to_markdown


def _sample():
    return [
        {"role": "user", "content": "<environment>\ncwd=/x\n</environment>"},  # skipped
        {"role": "user", "content": "fix the parser bug"},
        {
            "role": "assistant",
            "content": "On it.",
            "tool_calls": [
                {"function": {"name": "read", "arguments": '{"path": "a.py"}'}}
            ],
        },
        {"role": "tool", "content": "line1\nline2"},
        {"role": "assistant", "content": "Done — bug fixed."},
        {"role": "system", "content": "[Summary of earlier conversation]\nearlier stuff"},
    ]


def test_to_markdown_structure():
    md = to_markdown("My Session", _sample())
    assert md.startswith("# My Session")
    assert "<environment>" not in md          # seed turn skipped
    assert "## 🧑 User" in md and "fix the parser bug" in md
    assert "## 🤖 Assistant" in md and "Done — bug fixed." in md
    assert "🔧 `read(" in md                   # tool call rendered
    assert "tool result" in md                # tool output in a <details>
    assert "earlier stuff" in md              # system summary kept
    assert md.endswith("\n")


def test_to_markdown_truncates_long_tool_output():
    big = "y" * 5000
    md = to_markdown("t", [{"role": "tool", "content": big}])
    assert "[truncated]" in md
    assert len(md) < 5000 + 300


def test_to_markdown_empty():
    assert to_markdown("Empty", []).strip() == "# Empty"
