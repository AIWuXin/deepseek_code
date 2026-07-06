"""Copy text to the user's clipboard from the TUI.

Two mechanisms, both best-effort, fired together:

  * OSC 52 (via Textual ``App.copy_to_clipboard``) — an escape sequence the
    *terminal* interprets, so it lands in the clipboard of whichever machine the
    user is actually sitting at. This is the one that works over SSH.
  * ``pyperclip`` — writes the local OS clipboard directly. Rock-solid when dsc
    runs locally (e.g. Windows Terminal), but useless on a headless SSH host.

We fire OSC 52 unconditionally and then also try pyperclip. Locally the two hit
the same clipboard with identical content (idempotent); over SSH pyperclip just
fails quietly and OSC 52 has already done the job. Neither path can raise into
the caller — copying must never crash the UI.
"""

from __future__ import annotations

try:
    import pyperclip
except Exception:  # not installed, or no platform backend available
    pyperclip = None  # type: ignore[assignment]


def copy_to_clipboard(app, text: str) -> None:
    """Best-effort copy ``text`` to the user's clipboard. Never raises."""
    if not text:
        return
    # OSC 52 first — correct target both locally and over SSH.
    try:
        app.copy_to_clipboard(text)
    except Exception:
        pass
    # pyperclip second — reliable local fallback; a harmless duplicate locally,
    # a quiet no-op on a headless remote.
    if pyperclip is not None:
        try:
            pyperclip.copy(text)
        except Exception:
            pass
