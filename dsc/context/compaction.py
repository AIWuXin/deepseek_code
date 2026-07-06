"""Structured summarization of early history (lossy, last-resort reclamation).

Triggered only when lossless pruning can't get us under budget. We ask the
model to distill everything except the most recent turns into a compact brief,
then replace that early history with the summary. Keeping the most recent turns
verbatim preserves the working state the agent needs to continue.
"""

from __future__ import annotations

import json

# Number of *complete user→assistant→(tools)* turns to keep verbatim.
KEEP_TURNS = 3

SUMMARY_INSTRUCTION = (
    "You are compacting a coding-agent conversation to save context. "
    "Summarize everything below into a dense brief that lets the agent continue "
    "without the original transcript. Preserve: the user's goal and constraints, "
    "files and symbols touched, decisions made, current state, and any pending "
    "next steps. Omit chit-chat and raw tool dumps. Use terse bullet points."
)

# Safety cap: never send more than this many characters to the summary model.
# ~50k chars ≈ 15k tokens — leaves plenty of room for system prompt + output.
MAX_SUMMARY_INPUT_CHARS = 50_000


def find_clean_tail_start(messages: list[dict], keep_turns: int = KEEP_TURNS) -> int:
    """Walk backwards to find the start of the ``keep_turns``-th complete turn.

    A complete turn begins with a ``user`` message.  If the assistant replied
    with ``tool_calls``, the subsequent ``tool`` results belong to the same turn
    and must be kept together.  Returns the index of the first message to keep.
    """
    turns_found = 0
    i = len(messages) - 1
    while i >= 0 and turns_found < keep_turns:
        if messages[i].get("role") == "user":
            turns_found += 1
        i -= 1
    return i + 1


def build_summary_request(messages: list[dict], keep_turns: int = KEEP_TURNS) -> list[dict]:
    """Build a one-off request that asks the model to summarize old history.

    Automatically truncates the input to ``MAX_SUMMARY_INPUT_CHARS`` to avoid
    exceeding the model's own context window.
    """
    tail_start = find_clean_tail_start(messages, keep_turns)
    to_summarize = messages[:tail_start]
    transcript = _flatten(to_summarize)
    if len(transcript) > MAX_SUMMARY_INPUT_CHARS:
        transcript = (
            transcript[:MAX_SUMMARY_INPUT_CHARS]
            + "\n\n[truncated — rest of history omitted]"
        )
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
