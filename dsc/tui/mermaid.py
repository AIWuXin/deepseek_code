"""Detect and render ```mermaid blocks from assistant replies.

termaid is an optional dependency (the ``mermaid`` extra). Everything here is
written so that when termaid is absent — or a diagram fails to render — the
caller simply shows the original code block and nothing breaks.

A subtlety confirmed by testing termaid 0.7.1: it does NOT raise on bad input.
Invalid or truncated mermaid returns an *empty or near-empty* string/Text rather
than an exception. So ``safe_render`` treats an empty result as failure (returns
None), in addition to guarding against any future exception.
"""

from __future__ import annotations

import re

try:  # optional dependency — the `mermaid` extra
    import termaid as _termaid
except Exception:  # not installed
    _termaid = None

# A fenced ```mermaid ... ``` block. Tolerates leading spaces on the fence and
# an optional language-info suffix; captures the inner source.
_MERMAID_BLOCK = re.compile(
    r"^[ \t]*```[ \t]*mermaid[^\n]*\n(.*?)^[ \t]*```[ \t]*$",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)

# A rendered diagram shorter than this (after stripping) is treated as a failed
# render — termaid returns empty/near-empty text for invalid input.
_MIN_RENDER_CHARS = 4


def termaid_available() -> bool:
    return _termaid is not None


def extract_mermaid_blocks(text: str) -> list[str]:
    """Return the source of every ```mermaid block in ``text`` (in order)."""
    if not text or "mermaid" not in text.lower():
        return []
    return [m.group(1).strip() for m in _MERMAID_BLOCK.finditer(text) if m.group(1).strip()]


def safe_render(source: str):
    """Render mermaid ``source`` to a Rich ``Text``, or ``None`` on any failure.

    Failure = termaid missing, an exception, or an empty/near-empty result
    (termaid's silent-failure mode for invalid syntax).
    """
    if _termaid is None or not source.strip():
        return None
    try:
        rich_text = _termaid.render_rich(source)
    except Exception:
        return None
    plain = getattr(rich_text, "plain", None)
    if plain is None:
        # Not a Rich Text (unexpected API shape) — bail rather than guess.
        return None
    if len(plain.strip()) < _MIN_RENDER_CHARS:
        return None  # silent-failure empty render → treat as no diagram
    return rich_text
