"""Configuration loading for DeepSeek Code.

Config is resolved with this precedence (highest first):
  1. Environment variables (DEEPSEEK_API_KEY, DSC_MODEL, ...)
  2. ~/.dsc/config.toml
  3. Built-in defaults

Nothing here changes between requests, so it never affects the prompt prefix
that DeepSeek caches — see context/manager.py for why that matters.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path.home() / ".dsc"
CONFIG_PATH = CONFIG_DIR / "config.toml"

# DeepSeek V4 (deepseek-chat / deepseek-reasoner are deprecated 2026-07-24).
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

# Pricing per 1M tokens (USD), used only for the on-screen cost meter.
# Source: https://api-docs.deepseek.com/quick_start/pricing (2026-07).
PRICING = {
    "deepseek-v4-flash": {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
    "deepseek-v4-pro": {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
}


@dataclass
class Config:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    # Context window budget in tokens before compaction kicks in.
    # V4 supports 1M; we stay well under to protect recall quality (context rot).
    context_limit: int = 200_000
    # Hard stop for the agent loop, guards against runaway tool cycles.
    max_iterations: int = 25
    # Phase 3: use V2 cleaning (compress archived ranges + smart cleanup)
    # instead of the old compaction.  Default off until battle-tested.
    v2_cleaning: bool = False
    # Enable deepseek-reasoner-style thinking. V4 flash/pro both support it.
    thinking: bool = False
    extra: dict = field(default_factory=dict)

    def price(self) -> dict:
        return PRICING.get(self.model, PRICING[DEFAULT_MODEL])


def _load_file() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def load_config() -> Config:
    data = _load_file()
    cfg = Config(
        api_key=os.environ.get("DEEPSEEK_API_KEY", data.get("api_key", "")),
        base_url=os.environ.get("DSC_BASE_URL", data.get("base_url", DEFAULT_BASE_URL)),
        model=os.environ.get("DSC_MODEL", data.get("model", DEFAULT_MODEL)),
        context_limit=int(data.get("context_limit", 200_000)),
        max_iterations=int(data.get("max_iterations", 25)),
        thinking=bool(data.get("thinking", False)),
        v2_cleaning=bool(data.get("v2_cleaning", False)),
        extra=data,
    )
    return cfg


def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
