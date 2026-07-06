"""Textual application: scrollable conversation + input + status bar.

The agent loop is synchronous and blocking, so we run each turn in a worker
thread and marshal events back to the UI thread via call_from_thread. Streaming
text lands in a single live Markdown widget that we update in place, which lets
Textual's diff renderer repaint only what changed.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Static

from ..agent.loop import AgentLoop, _preview
from ..session import SessionStore, describe_session
from .commands import CommandScreen
from .prompt import PromptInput
from .session_picker import SessionPickerScreen
from .widgets import (
    AssistantMessage,
    MermaidBlock,
    Notice,
    ReasoningBlock,
    ToolLine,
    ToolOutput,
    UserMessage,
)
from ..debug import log


class StatusBar(Static):
    """Top bar: session title, model, context usage, cache hit rate, spend."""

    def update_stats(
        self, model: str, tokens: int, limit: int, hit_rate: float, usd: float, title: str | None = None
    ) -> None:
        pct = int(tokens / limit * 100) if limit else 0
        prefix = f"[b]{title}[/b]  ·  " if title else ""
        self.update(
            f"{prefix}{model}  ·  ctx {tokens:,}/{limit:,} ({pct}%)  ·  "
            f"cache [green]{hit_rate * 100:.0f}%[/green] hit  ·  [yellow]${usd:.4f}[/yellow]"
        )


class DSCApp(App):
    # rose-pine-moon is a built-in Textual theme; noticeably nicer than default.
    THEME = "rose-pine-moon"

    CSS = """
    Screen { layout: vertical; }
    StatusBar { dock: top; height: 1; background: $panel; color: $text; padding: 0 1; }
    #log {
        height: 1fr;
        padding: 0 1;
        /* Reserve a column for the scrollbar so it never overlaps text on the
           right edge (the cause of the truncated-looking wrap). */
        scrollbar-gutter: stable;
    }
    /* Cap content width so long lines wrap in a readable column instead of
       running into the right border. */
    UserMessage, AssistantMessage, ReasoningBlock, ToolLine, Notice, ToolOutput, MermaidBlock {
        max-width: 100;
    }
    PromptInput {
        dock: bottom;
        height: 5;
        max-height: 14;
        width: 100%;
        /* Only a bottom gap. A left/right margin is NOT subtracted from the
           docked 1fr width, so it pushes the right border off-screen. Keep the
           box full-width (border-box: borders sit inside the width). */
        margin: 0 0 1 0;
        padding: 0 1;
        border: round $accent;
        background: $panel;
    }
    PromptInput:focus { border: round $accent-lighten-1; }
    Footer { dock: bottom; height: 1; }
    UserMessage { color: $accent; margin: 1 0 0 0; }
    AssistantMessage { margin: 0 0 1 0; }
    ReasoningBlock, ToolOutput { margin: 0 0 1 0; }
    ToolLine { color: $text-muted; }
    Notice { color: $warning; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("escape", "interrupt", "Interrupt"),
        Binding("f1", "show_commands", "Commands", priority=True),
        Binding("ctrl+p", "show_commands", "Commands", priority=True),
    ]

    # Idle prompt caption; also restored after each turn (see _clear_activity).
    _PROMPT_HINT = "Enter send · Shift+Enter newline · F1 commands · Esc interrupt · Ctrl+C quit"

    def __init__(self, config, registry, cwd: str, session_name: str | None = None):
        super().__init__()
        self.config = config
        self.cwd = cwd
        self.loop = AgentLoop(config, registry, cwd, session_name)
        self._resumed_name = session_name  # non-None → paint history on mount
        # A resumed session already has its title; a fresh one gets named after
        # the first turn by an isolated background call.
        self._titled = self.loop.title is not None
        self._live: AssistantMessage | None = None
        self._reasoning: ReasoningBlock | None = None
        self._reasoning_buf = ""
        self._busy = False
        # Auto-scroll follows the tail until the user scrolls up; it resumes when
        # they scroll back to the bottom or send a new message.
        self._follow = True

    def compose(self) -> ComposeResult:
        yield StatusBar()
        yield VerticalScroll(id="log")
        prompt = PromptInput()
        prompt.border_title = self._PROMPT_HINT
        yield prompt
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "rose-pine-moon"
        if self._resumed_name:
            self._append(Notice(f"Resumed '{self._resumed_name}' — earlier history below:"))
            self._render_history(self.loop.ctx.messages)
        else:
            self._welcome()
        self._refresh_status()
        self.query_one(PromptInput).focus()

    def _welcome(self) -> None:
        """A short banner with the key bindings — shown on start and after /clear."""
        self._append(Notice(
            f"DeepSeek Code · {self.loop.config.model}\n"
            "  Enter 发送 · Shift+Enter 换行 · F1 命令 · Esc 中断\n"
            "  /help 查看全部命令 · /export 导出对话"
        ))

    def _render_history(self, messages: list[dict]) -> None:
        """Paint stored messages so a resumed session shows what came before.

        Reconstructs the conversation compactly: user turns and assistant text
        are rendered in full; tool calls become one dim line each; raw tool
        results are omitted (they can be huge and are implied by the call).
        """
        for m in messages:
            role = m.get("role")
            content = m.get("content") or ""
            if role == "user":
                if content.startswith("<environment>"):
                    continue  # skip the seed
                self._append(UserMessage(content))
            elif role == "assistant":
                if content.strip():
                    self._append(AssistantMessage(content))
                for tc in m.get("tool_calls", []) or []:
                    fn = tc.get("function", {})
                    preview = _preview(fn.get("arguments", ""))
                    self._append(ToolLine(f"✓ {fn.get('name', '?')}({preview})"))
            # tool results intentionally not rendered

    # -- input handling -------------------------------------------------------

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        if self._busy:
            self._append(Notice("busy — wait for the current turn to finish"))
            return

        if text.startswith("/"):
            if self._handle_command(text):
                return

        self._follow = True  # sending a message means "show me the reply"
        self._append(UserMessage(text))
        self._busy = True
        self._live = None
        self._set_activity("thinking…")
        self.run_turn(text)

        # First turn of a fresh session → name it in the background (isolated
        # call, does not touch the main context or block the answer).
        if not self._titled:
            self._titled = True
            self.name_session(text)

    def _handle_command(self, text: str) -> bool:
        cmd = text[1:].split()[0].lower()
        if cmd in ("quit", "exit", "q"):
            self.exit()
            return True
        if cmd in ("help", "commands"):
            self.action_show_commands()
            return True
        if cmd == "clear":
            self.query_one("#log", VerticalScroll).remove_children()
            self._welcome()
            return True
        if cmd == "export":
            try:
                path = self.loop.export(self.cwd)
                self._append(Notice(f"Exported conversation → {path}"))
            except Exception as e:
                self._append(Notice(f"Export failed: {e}"))
            return True
        if cmd == "model":
            parts = text.split()
            if len(parts) > 1:
                self.loop.config.model = parts[1]
                self.loop.client.model = parts[1]
                self._append(Notice(f"model → {parts[1]}"))
                self._refresh_status()
            return True
        if cmd == "sessions":
            paths = SessionStore.list_sessions()
            if not paths:
                self._append(Notice("No saved sessions."))
            else:
                lines = []
                for p in paths:
                    count = sum(1 for _ in p.open("r", encoding="utf-8") if _.strip())
                    import datetime
                    mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%H:%M")
                    lines.append(f"  {p.stem:<22} {count:>3} msgs  {mtime}")
                self._append(Notice("Sessions:\n" + "\n".join(lines)))
            return True
        if cmd == "resume":
            parts = text.split()
            if len(parts) > 1:
                # Power user: directly specify session name.
                self._resume_session(parts[1])
            else:
                self._open_session_picker()
            return True
        return False

    # -- command menu & session picker ---------------------------------------

    def action_show_commands(self) -> None:
        """Open the command palette (bound to F1 / Ctrl+P / click in Footer)."""
        self.push_screen(CommandScreen(), self._on_command_selected)

    def _on_command_selected(self, cmd: str | None) -> None:
        if cmd is None:
            return
        if cmd == "/resume":
            self._open_session_picker()
        elif cmd == "/model":
            # Needs an argument: pre-fill the prefix for the user to complete.
            prompt = self.query_one(PromptInput)
            prompt.text = "/model "
            prompt.focus()
        elif cmd.startswith("/"):
            # Synthesize the command text and handle it.
            self._handle_command(cmd)

    def _open_session_picker(self) -> None:
        self.push_screen(SessionPickerScreen(), self._on_session_selected)

    def _on_session_selected(self, session_name: str | None) -> None:
        if session_name:
            self._resume_session(session_name)

    def _resume_session(self, name: str) -> None:
        store = SessionStore.from_name(name)
        if store is None:
            self._append(Notice(f"Session '{name}' not found."))
            return
        msgs = store.load()
        if not msgs:
            self._append(Notice(f"Session '{name}' is empty."))
            return
        old_meter = self.loop.meter
        self.loop = AgentLoop(self.config, self.loop.registry, self.cwd, store.name)
        self.loop.meter = old_meter  # keep cost history
        self._titled = self.loop.title is not None
        self._follow = True
        self.query_one("#log", VerticalScroll).remove_children()
        info = describe_session(store.path)
        self._append(Notice(f"Resumed '{info.title}' ({len(msgs)} messages) — history below:"))
        self._render_history(self.loop.ctx.messages)
        self._refresh_status()

    # -- worker: run one agent turn off the UI thread -------------------------

    from textual import work

    @work(thread=True, exclusive=True)
    def run_turn(self, text: str) -> None:
        try:
            for ev in self.loop.send(text):
                self.call_from_thread(self._handle_event, ev)
        except Exception as e:
            self.call_from_thread(self._append, Notice(f"error: {e}"))
        finally:
            self.call_from_thread(self._finish_turn)

    @work(thread=True, group="naming", exclusive=True)
    def name_session(self, first_text: str) -> None:
        """Generate the session title off the UI thread (isolated LLM call)."""
        title = self.loop.generate_title(first_text)
        if title:
            self.call_from_thread(self._on_titled, title)

    def _on_titled(self, title: str) -> None:
        self._refresh_status()  # title now shows in the status bar

    def _handle_event(self, ev) -> None:
        log(f"ui: recv {ev.kind} display={ev.display!r}")
        try:
            self._dispatch_event(ev)
        except Exception as e:
            # A rendering failure in one event must never silently swallow the
            # rest of the turn — that's exactly how a tool call "vanishes".
            log(f"ui: ERROR handling {ev.kind}: {e!r}")
            self._append(Notice(f"render error ({ev.kind}): {e}"))
        self._refresh_status()

    def _dispatch_event(self, ev) -> None:
        if ev.kind == "reasoning":
            self._reasoning_buf += ev.text
            if self._reasoning is None:
                self._reasoning = ReasoningBlock()
                self._append(self._reasoning)
                self._set_activity("thinking…")
            self._reasoning.append(self._reasoning_buf)
        elif ev.kind == "text":
            if self._live is None:
                self._live = AssistantMessage("")
                self._append(self._live)
                self._set_activity("streaming…")
            # append_text returns True only when it actually repainted; tie
            # scroll to the same cadence so we don't reflow the log per token.
            if self._live.append_text(ev.text):
                self._scroll()
        elif ev.kind == "tool_start":
            self._append(ToolLine(f"→ {ev.display}", running=True))
            self._set_activity(f"→ {ev.display}")
        elif ev.kind == "tool_end":
            self._append(ToolLine(("✗ " if ev.is_error else "✓ ") + ev.display, error=ev.is_error))
            if ev.text:
                # Errors and short results auto-expand; long output stays folded.
                self._append(ToolOutput(ev.text, error=ev.is_error))
            # Next assistant text starts a fresh message bubble.
            self._reset_live()
        elif ev.kind == "notice":
            self._append(Notice(ev.text))
        elif ev.kind == "done":
            self._reset_live()

    def _reset_live(self) -> None:
        if self._live is not None:
            self._live.finalize()  # streaming plain text → full Markdown render
            self._render_mermaid(self._live._text)
            self._scroll()
        self._live = None
        self._reasoning = None
        self._reasoning_buf = ""

    def _render_mermaid(self, text: str) -> None:
        """Append a MermaidBlock for each ```mermaid block in a finished reply.

        Best-effort: if termaid isn't installed or a diagram fails to render,
        the block is skipped and the original code stays in the message. Only
        runs after the message is complete (no half-streamed diagrams).
        """
        from .mermaid import extract_mermaid_blocks, safe_render

        for source in extract_mermaid_blocks(text):
            diagram = safe_render(source)
            if diagram is not None:
                self._append(MermaidBlock(diagram, source))

    def _finish_turn(self) -> None:
        self._busy = False
        self._clear_activity()
        self._refresh_status()

    # -- busy / activity indicator -------------------------------------------

    def _set_activity(self, text: str) -> None:
        """Show what the agent is doing on the input box caption."""
        try:
            self.query_one(PromptInput).border_title = f"⏳ {text}"
        except Exception:
            pass  # UI teardown / not mounted — never break the turn

    def _clear_activity(self) -> None:
        """Restore the idle key-binding caption once the turn ends."""
        try:
            self.query_one(PromptInput).border_title = self._PROMPT_HINT
        except Exception:
            pass

    # -- auto-scroll follow ---------------------------------------------------

    def on_mouse_scroll_up(self, event) -> None:
        # The user is reading back — stop yanking the view to the tail.
        self._follow = False

    def on_mouse_scroll_down(self, event) -> None:
        # Re-enable follow once they roll back down to (near) the bottom.
        log = self.query_one("#log", VerticalScroll)
        if log.scroll_offset.y >= log.max_scroll_y - 1:
            self._follow = True

    # -- helpers --------------------------------------------------------------

    def _append(self, widget) -> None:
        self.query_one("#log", VerticalScroll).mount(widget)
        self._scroll()

    def _scroll(self) -> None:
        if not self._follow:
            return
        self.query_one("#log", VerticalScroll).scroll_end(animate=False)

    def _refresh_status(self) -> None:
        m = self.loop.meter
        self.query_one(StatusBar).update_stats(
            self.loop.config.model,
            self.loop.ctx.estimated_tokens(),
            self.loop.config.context_limit,
            m.hit_rate,
            m.usd,
            title=self.loop.title,
        )

    def action_interrupt(self) -> None:
        # Cooperative: cancel the worker; the loop checks between iterations.
        workers = [w for w in self.workers if w.is_running]
        for w in workers:
            w.cancel()
        if workers:
            self._append(Notice("interrupted"))
            self._busy = False


def run_tui(config, registry, cwd: str, session_name: str | None = None) -> int:
    DSCApp(config, registry, cwd, session_name).run()
    return 0
