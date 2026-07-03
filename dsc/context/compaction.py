"""Structured summarization of early history (lossy, last-resort reclamation).

Triggered only when lossless pruning can't get us under budget. We ask the
model to distill everything except the most recent turns into a compact brief,
then replace that early history with the summary. Keeping the most recent turns
verbatim preserves the working state the agent needs to continue.
"""

from __future__ import annotations

import json

KEEP_RECENT = 6  # verbatim messages retained after the summary

SUMMARY_INSTRUCTION = (
    "You are compacting a coding-agent conversation to save context. "
    "Summarize everything below into a dense brief that lets the agent continue "
    "without the original transcript. Preserve: the user's goal and constraints, "
    "files and symbols touched, decisions made, current state, and any pending "
    "next steps. Omit chit-chat and raw tool dumps. Use terse bullet points."
)


def build_summary_request(messages: list[dict], keep_recent: int = KEEP_RECENT) -> list[dict]:
    """Build a one-off request that asks the model to summarize old history."""
    to_summarize = messages[:-keep_recent] if keep_recent else messages
    transcript = _flatten(to_summarize)
    return [
        {"role": "system", "content": SUMMARY_INSTRUCTION},
        {"role": "user", "content": transcript},
    ]


def _flatten(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        line = f"### {role}\n{content}"
        for tc in m.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            line += f"\n[tool_call {fn.get('name')}({fn.get('arguments', '')})]"
        parts.append(line)
    return "\n\n".join(parts)
