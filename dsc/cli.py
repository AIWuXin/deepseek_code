"""Entry point: `dsc`.

Modes:
  dsc                 → launch the Textual TUI (default)
  dsc -p "prompt"     → headless: run one prompt, stream to stdout, exit
  dsc --plain         → plain REPL without the TUI (useful for debugging)
  dsc --resume        → resume the most recent session
  dsc --session NAME  → resume a specific session by name
  dsc --list-sessions → list saved sessions and exit
"""

from __future__ import annotations

import argparse
import os
import sys

# ── bundle bootstrap (only activates in Nuitka-compiled builds) ──────────
# When running as a compiled exe, third-party packages live in a `packages/`
# directory next to the exe.  If missing, we download them on first launch.

try:
    from ._build_config import BUNDLE_VERSION, BUNDLE_DOWNLOAD_BASE
except ImportError:
    BUNDLE_VERSION = None
    BUNDLE_DOWNLOAD_BASE = None


def _bundle_bootstrap() -> None:
    """Ensure third-party ``packages/`` dir exists next to the compiled exe.

    Only runs when ``sys.frozen`` is set (Nuitka / PyInstaller builds).
    Downloads ``packages.zip`` from the release page and extracts it if
    the directory is missing.
    """
    if not getattr(sys, "frozen", False):
        return  # running from source — packages are installed by pip

    exe_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    packages_dir = os.path.join(exe_dir, "packages")

    # Already set up?
    if os.path.isdir(packages_dir) and any(
        fname.endswith((".py", ".pyc", ".pyd"))
        for fname in os.listdir(packages_dir)
    ):
        sys.path.insert(0, packages_dir)
        return

    if not BUNDLE_DOWNLOAD_BASE or not BUNDLE_VERSION:
        print(
            "error: this is a bundled exe but no download URL was configured.\n"
            "       Please re-download from the official release page.",
            file=sys.stderr,
        )
        sys.exit(1)

    import urllib.request
    import zipfile

    # Priority 1: env var override.
    url = os.environ.get("DSC_PACKAGES_URL") or (
        f"{BUNDLE_DOWNLOAD_BASE}/packages.zip" if BUNDLE_DOWNLOAD_BASE else None
    )

    # Priority 2: local packages.zip next to the exe.
    local_zip = os.path.join(exe_dir, "packages.zip")

    if url and not os.path.isfile(local_zip):
        zip_path = os.path.join(exe_dir, "packages.zip.tmp")
        print(f"\n📦  First launch — downloading dependencies ({BUNDLE_VERSION})…")
        print(f"   {url}\n")
        try:
            urllib.request.urlretrieve(url, zip_path, _bundle_progress)
            print()
            _extract_zip(zip_path, packages_dir)
        except Exception as exc:
            print(f"\n   Download failed: {exc}", file=sys.stderr)
            sys.exit(1)
        finally:
            if os.path.isfile(zip_path):
                os.remove(zip_path)
    elif os.path.isfile(local_zip):
        print(f"\n📦  Extracting local packages.zip…")
        _extract_zip(local_zip, packages_dir)
    else:
        print(
            "error: dependencies not found. Place packages.zip next to dsc.exe\n"
            "       or set DSC_PACKAGES_URL to a download link.",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.path.insert(0, packages_dir)
    print("   Done.\n")


def _extract_zip(zip_path: str, target_dir: str) -> None:
    import zipfile
    import os
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def _bundle_progress(block: int, block_size: int, total: int) -> None:
    """Simple progress indicator for ``urlretrieve``."""
    if total > 0 and block % 8 == 0:  # update ~every 64 KB
        pct = min(block * block_size * 100 // total, 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r   [{bar}] {pct}%", end="", file=sys.stderr)


# ── main ─────────────────────────────────────────────────────────────────

from .config import load_config
from .session import SessionStore
from .tools import build_registry

# ── display helper ───────────────────────────────────────────────────────
# On CJK Windows terminals circled digits (①②③) render at 1 cell wide.
# When adjacent to 2-cell-wide Han characters they visually merge.  We add
# a space after each circled digit purely for display.
_CIRCLED = frozenset(chr(cp) for cp in range(0x2460, 0x2500)) | \
           frozenset(chr(cp) for cp in range(0x2776, 0x2794))


def _space_circled(text: str) -> str:
    """Insert a space after each circled digit (display only)."""
    if not text:
        return text
    parts = []
    for ch in text:
        parts.append(ch)
        if ch in _CIRCLED:
            parts.append(" ")
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    _bundle_bootstrap()
    parser = argparse.ArgumentParser(prog="dsc", description="DeepSeek Code — terminal coding agent.")
    parser.add_argument("-p", "--prompt", help="Run a single prompt headlessly and exit.")
    parser.add_argument("--plain", action="store_true", help="Plain REPL without the TUI.")
    parser.add_argument("-C", "--cwd", default=os.getcwd(), help="Workspace directory.")
    parser.add_argument("--model", help="Override model (e.g. deepseek-v4-pro).")
    parser.add_argument("--resume", action="store_true", help="Resume the most recent session.")
    parser.add_argument("--session", help="Resume a specific session by name (see --list-sessions).")
    parser.add_argument("--list-sessions", action="store_true", help="List saved sessions and exit.")
    parser.add_argument("--delete", metavar="NAME", help="Delete a saved session by name and exit.")
    args = parser.parse_args(argv)

    if args.list_sessions:
        _list_sessions()
        return 0

    if args.delete:
        ok = SessionStore.delete(args.delete)
        print(f"Deleted session '{args.delete}'." if ok else f"Session '{args.delete}' not found.")
        return 0 if ok else 1

    config = load_config()
    if args.model:
        config.model = args.model
    if not config.api_key:
        print("error: no API key. Set DEEPSEEK_API_KEY or ~/.dsc/config.toml.", file=sys.stderr)
        return 2

    cwd = os.path.abspath(args.cwd)
    registry = build_registry(cwd)

    # Determine session to resume.
    session_name: str | None = None
    if args.session:
        store = SessionStore.from_name(args.session)
        if store is None:
            print(f"error: session '{args.session}' not found.", file=sys.stderr)
            return 2
        session_name = args.session
    elif args.resume:
        latest = SessionStore.latest()
        if latest is None:
            print("error: no sessions to resume.", file=sys.stderr)
            return 2
        session_name = latest.name

    if args.prompt is not None:
        return _run_headless(config, registry, cwd, args.prompt, session_name)
    if args.plain:
        return _run_plain(config, registry, cwd, session_name)

    # Default: TUI. Imported lazily so headless mode doesn't need Textual.
    from .tui.app import run_tui

    return run_tui(config, registry, cwd, session_name)


def _list_sessions() -> None:
    import datetime

    infos = SessionStore.infos()
    if not infos:
        print("No saved sessions.")
        return
    for info in infos:
        when = datetime.datetime.fromtimestamp(info.mtime).strftime("%Y-%m-%d %H:%M")
        print(f"{info.title}")
        meta = f"    {info.name}  ·  {info.count} msgs  ·  {when}"
        if info.summary:
            meta += f"  ·  {info.summary}"
        print(meta)
        # Pass --session <name> to resume a specific one.


def _run_headless(config, registry, cwd: str, prompt: str, session_name: str | None = None) -> int:
    from .agent.loop import AgentLoop

    loop = AgentLoop(config, registry, cwd, session_name)
    for ev in loop.send(prompt):
        if ev.kind == "text":
            sys.stdout.write(_space_circled(ev.text))
            sys.stdout.flush()
        elif ev.kind == "tool_end":
            marker = "✗" if ev.is_error else "✓"
            sys.stderr.write(f"\n  {marker} {ev.display}\n")
        elif ev.kind == "notice":
            sys.stderr.write(f"\n[{ev.text}]\n")
        elif ev.kind == "done":
            sys.stdout.write("\n")
    _print_cost(loop)
    return 0


def _run_plain(config, registry, cwd: str, session_name: str | None = None) -> int:
    from .agent.loop import AgentLoop

    loop = AgentLoop(config, registry, cwd, session_name)
    print(f"DeepSeek Code (plain) — {config.model} @ {cwd}")
    print("Type your request. Ctrl-C or empty line + Ctrl-D to quit.\n")
    while True:
        try:
            user = input("\x1b[1m›\x1b[0m ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user.strip():
            continue
        try:
            for ev in loop.send(user):
                _emit_plain(ev)
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue
        except Exception as e:
            print(f"\n[error] {e}")
            continue
        _print_cost(loop)
    return 0


def _emit_plain(ev) -> None:
    if ev.kind == "reasoning":
        sys.stdout.write(f"\x1b[2m{_space_circled(ev.text)}\x1b[0m")
    elif ev.kind == "text":
        sys.stdout.write(_space_circled(ev.text))
    elif ev.kind == "tool_start":
        sys.stdout.write(f"\n\x1b[36m→ {ev.display}\x1b[0m\n")
    elif ev.kind == "tool_end":
        marker = "\x1b[31m✗" if ev.is_error else "\x1b[32m✓"
        sys.stdout.write(f"\x1b[2m  {marker} {ev.display}\x1b[0m\n")
    elif ev.kind == "notice":
        sys.stdout.write(f"\n\x1b[33m[{ev.text}]\x1b[0m\n")
    elif ev.kind == "done":
        sys.stdout.write("\n")
    sys.stdout.flush()


def _print_cost(loop) -> None:
    m = loop.meter
    print(
        f"\x1b[2m  ${m.usd:.4f} · cache {m.hit_rate * 100:.0f}% hit "
        f"({m.cache_hit}/{m.cache_hit + m.cache_miss} in, {m.output} out)\x1b[0m"
    )


if __name__ == "__main__":
    raise SystemExit(main())
