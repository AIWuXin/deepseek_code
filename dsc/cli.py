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

from .config import load_config
from .session import SessionStore
from .tools import build_registry


def main(argv: list[str] | None = None) -> int:
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
            sys.stdout.write(ev.text)
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
        sys.stdout.write(f"\x1b[2m{ev.text}\x1b[0m")
    elif ev.kind == "text":
        sys.stdout.write(ev.text)
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
