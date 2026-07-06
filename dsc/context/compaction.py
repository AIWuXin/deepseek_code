"""Structured summarization of early history (lossy, last-resort reclamation).

Triggered only when lossless pruning can't get us under budget. We ask the
model to distill everything except the most recent turns into a compact brief,
then replace that early history with the summary. Keeping the most recent turns
verbatim preserves the working state the agent needs to continue.
"""

from __future__ import annotations

import json
import re

# Number of *complete user→assistant→(tools)* turns to keep verbatim.
KEEP_TURNS = 3

# The compaction summary must survive as the agent's only memory of the early
# conversation, so the instruction is structured (fixed dimensions, nothing
# silently dropped) and guards against the two classic failure modes:
#   * the model calling a tool instead of writing text (wastes the one turn);
#   * task drift — the summary paraphrasing the request until intent shifts,
#     countered by quoting the most recent request verbatim.
# The <analysis> block is a scratchpad to lift generation quality; it is stripped
# by ``format_compact_summary`` before the summary enters context, so it never
# costs us the tokens we are trying to reclaim.
SUMMARY_INSTRUCTION = (
    "You are compacting a coding-agent conversation to free up context while "
    "preserving everything needed to continue the work.\n\n"
    "Respond with TEXT ONLY. Do NOT call any tools — you have exactly one turn "
    "and a tool call wastes it.\n\n"
    "First, think inside a <analysis>...</analysis> block: scan the whole "
    "conversation and note what matters. That block is a scratchpad and will be "
    "DISCARDED, so every fact you want to keep must appear in the final brief "
    "AFTER the closing </analysis> tag.\n\n"
    "The brief MUST cover these dimensions (drop a heading only if truly empty):\n"
    "1. Goal & constraints — what the user ultimately wants; any hard rules.\n"
    "2. Key technical concepts and decisions, with their rationale.\n"
    "3. Files & symbols touched — paths, functions, classes.\n"
    "4. Errors encountered and how they were fixed.\n"
    "5. Current state — what is done and working right now.\n"
    "6. Pending next steps.\n"
    "7. Recent context — quote the most recent user request and the task "
    "currently in progress VERBATIM so intent does not drift.\n\n"
    "Be terse: bullet points, no chit-chat, no raw tool-output dumps."
)

_ANALYSIS_RE = re.compile(r"<analysis>.*?</analysis>", re.DOTALL | re.IGNORECASE)


def format_compact_summary(raw: str) -> str:
    """Strip the model's ``<analysis>`` scratchpad, keeping only the final brief.

    Falls back to the raw text if stripping would leave nothing (e.g. the model
    put everything inside the block, or emitted an unterminated tag).
    """
    text = raw or ""
    # Drop complete <analysis>...</analysis> blocks, then any unterminated opener
    # (model forgot to close the tag → everything after it is scratchpad).
    cleaned = _ANALYSIS_RE.sub("", text)
    cleaned = re.sub(r"(?is)<analysis>.*$", "", cleaned).strip()
    # If stripping left nothing (the whole reply was the scratchpad), salvage the
    # raw text rather than handing back an empty summary.
    return cleaned or text.strip()

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
