"""Repair a stored transcript so it stays valid for the chat API.

DeepSeek (and the OpenAI schema) require that an ``assistant`` message carrying
``tool_calls`` is immediately followed by one ``tool`` message per
``tool_call_id``. A crash — or a bug — between issuing a tool call and recording
its result can leave a *dangling* tool call in the saved session, after which
every request 400s with:

    "An assistant message with 'tool_calls' must be followed by tool messages
     responding to each 'tool_call_id'."

``repair_dangling_tool_calls`` makes such a transcript loadable again by
inserting a stub ``tool`` message for any unanswered call. It also drops orphan
``tool`` messages that have no matching call (the mirror-image corruption).
"""

from __future__ import annotations

RECOVERED_STUB = "[tool result missing — session recovered]"


def repair_dangling_tool_calls(messages: list[dict]) -> tuple[list[dict], bool]:
    """Return ``(repaired, changed)`` with every tool_call answered exactly once.

    - Missing tool response → insert a stub tool message right after the caller.
    - Orphan tool message (no preceding open call for its id) → drop it.
    """
    out: list[dict] = []
    changed = False
    # tool_call_ids that have been announced by an assistant but not yet answered.
    open_ids: set[str] = set()
    n = len(messages)

    for i, m in enumerate(messages):
        role = m.get("role")

        if role == "assistant" and m.get("tool_calls"):
            # Before moving past a previous assistant's calls, flush any that the
            # upcoming non-tool message would leave unanswered.
            if open_ids:
                for tid in list(open_ids):
                    out.append({"role": "tool", "tool_call_id": tid, "content": RECOVERED_STUB})
                    changed = True
                open_ids.clear()
            out.append(m)
            open_ids = {tc.get("id") for tc in m["tool_calls"] if tc.get("id")}
            continue

        if role == "tool":
            tid = m.get("tool_call_id")
            if tid in open_ids:
                out.append(m)
                open_ids.discard(tid)
            else:
                # Orphan tool result with no open call — drop it.
                changed = True
            continue

        # Any other message (user / assistant-without-calls / system) closes the
        # current tool-call group: flush stubs for whatever is still unanswered.
        if open_ids:
            for tid in list(open_ids):
                out.append({"role": "tool", "tool_call_id": tid, "content": RECOVERED_STUB})
                changed = True
            open_ids.clear()
        out.append(m)

    # Trailing unanswered calls (transcript ended right after a tool_calls msg).
    if open_ids:
        for tid in list(open_ids):
            out.append({"role": "tool", "tool_call_id": tid, "content": RECOVERED_STUB})
            changed = True

    return out, changed
