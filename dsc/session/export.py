"""Render a stored conversation as a readable Markdown transcript.

Pure formatting — no I/O — so it's trivially testable. The agent loop wires it
to a file on disk (see ``AgentLoop.export``).
"""

from __future__ import annotations

# Cap each raw tool result so an export of a long session stays openable.
_MAX_TOOL_CHARS = 4000


def to_markdown(title: str, messages: list[dict]) -> str:
    """Format ``messages`` (the stored JSONL turns) as a Markdown document."""
    out: list[str] = [f"# {title}", ""]
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").rstrip()

        if role == "user":
            if content.startswith("<environment>"):
                continue  # skip the seed environment turn
            out += ["## 🧑 User", "", content, ""]

        elif role == "assistant":
            if content.strip():
                out += ["## 🤖 Assistant", "", content, ""]
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                args = fn.get("arguments", "")
                out += [f"> 🔧 `{fn.get('name', '?')}({args})`", ""]

        elif role == "tool":
            body = content[:_MAX_TOOL_CHARS]
            if len(content) > _MAX_TOOL_CHARS:
                body += "\n… [truncated]"
            out += [
                "<details><summary>🔧 tool result</summary>",
                "",
                "```",
                body,
                "```",
                "",
                "</details>",
                "",
            ]

        elif role == "system":
            # Summary / task-summary blocks injected by compaction.
            out += [f"> _{content}_", ""]

    return "\n".join(out).rstrip() + "\n"
