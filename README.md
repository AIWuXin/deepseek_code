# DeepSeek Code (`dsc`)

A token-frugal, long-context terminal coding agent, purpose-built for **DeepSeek V4**.

It looks like Claude Code, but every design decision is bent toward two goals:
**spend as few tokens as possible**, and **stay coherent over long sessions**.

## Why it's cheap

DeepSeek bills a matched input *prefix* at the cache-hit rate — about **98%
cheaper** than a cache miss. `dsc` is built around keeping that prefix stable:

- **Byte-stable prefix** — the system prompt and tool schemas never change
  across turns (no timestamps baked in), so they always hit the cache.
- **Append-only history** — we only ever add to the tail of the message list,
  never edit the middle, so earlier turns stay cached.
- **Tiered reclamation** — when context fills up we first *losslessly* stub out
  old tool-result bodies, and only *summarize* (compact) as a last resort.
- **Tool results are bounded** — `read` pages at 2000 lines, `bash` truncates
  at 30k chars, `grep` returns matching lines only. No whole-file dumps.
- **Search-replace edits** — changes are content-anchored `edit` blocks, not
  whole-file rewrites, cutting edit tokens ~10×.

A live status bar shows model, context usage, cache-hit rate, and spend in USD.

## Install

```bash
uv sync
export DEEPSEEK_API_KEY=sk-...     # or put it in ~/.dsc/config.toml
```

## Use

```bash
dsc                       # launch the TUI
dsc --plain               # plain REPL (no TUI) — good for debugging
dsc -p "fix the bug in foo.py and run the tests"   # headless, one-shot
dsc --model deepseek-v4-pro        # switch model for harder tasks
dsc --resume              # resume the most recent session
dsc --session NAME        # resume a specific session
dsc --list-sessions       # list saved sessions
dsc --delete NAME         # delete a saved session
```

Inside the TUI: `Enter` sends · `Shift+Enter` newline · `F1` / `Ctrl+P` opens the
command palette · `Esc` interrupts. Slash commands: `/clear`, `/model <name>`,
`/sessions`, `/resume`, `/help`, `/quit`. In the `/resume` picker, press `d`
twice to delete the highlighted session. Every conversation is auto-saved to
`~/.dsc/sessions/*.jsonl` (with a model-generated title) and can be resumed
later.

### `~/.dsc/config.toml`

```toml
api_key = "sk-..."
model = "deepseek-v4-flash"   # or deepseek-v4-pro
context_limit = 200000        # tokens before compaction
max_iterations = 25
thinking = false              # enable V4 thinking mode
```

## Tools

`read` · `grep` (ripgrep) · `glob` · `edit` (search-replace) · `write` ·
`bash` (Git Bash on Windows) · `web_search` (DuckDuckGo, no API key)

## Architecture

```
dsc/
  agent/      main loop, DeepSeek client, system prompt
  context/    message layout, token accounting, compaction
  tools/      the six tools above
  tui/        Textual app + widgets
  session/    JSONL session log
```

The core is UI-agnostic: `agent.loop.AgentLoop.send()` yields events that the
TUI, the plain REPL, and headless mode all render differently.

## Tests

```bash
uv run pytest -q
```

Covers the highest-risk pieces: search-replace edit matching, the tool-use
loop, context pruning/compaction/restore, session persistence + resume
roundtrip, the input sanitizer, the command palette mapping, and the bash tool.

## Status

MVP. Working: agent loop, six tools (bash runs in Git Bash on Windows),
streaming TUI with a collapsible thinking block, V4 integration, prefix-cache
optimization, tiered context reclamation, session persistence + resume, command
palette. Planned: sub-agent isolation, repo map, FIM completion.
