"""System-prompt cache invariants and environment date injection."""

from __future__ import annotations

import re
from datetime import datetime

from dsc.agent.prompts import SYSTEM_PROMPT, initial_environment


def test_system_prompt_has_no_timestamp():
    """The system prompt is the cached prefix head — it must never contain a
    date/time, or every session would break DeepSeek's prefix cache."""
    assert not re.search(r"\b20\d{2}\b", SYSTEM_PROMPT)
    assert not re.search(r"\d{1,2}:\d{2}", SYSTEM_PROMPT)


def test_environment_carries_current_date():
    env = initial_environment("/work", now=datetime(2026, 7, 3, 15, 50))
    assert "Working directory: /work" in env
    assert "Current date: Friday, 2026-07-03 15:50" in env


def test_environment_date_defaults_to_now():
    # Without an explicit time it still injects a date line.
    assert "Current date:" in initial_environment("/work")


def test_system_prompt_tells_model_to_use_env_date():
    assert "current date" in SYSTEM_PROMPT.lower()
    assert "environment" in SYSTEM_PROMPT.lower()
