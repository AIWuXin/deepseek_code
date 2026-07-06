"""Config loading — regression guards for V2 feature flags.

Phase 3 V2 cleaning was once silently dead because ``load_config`` didn't
propagate ``v2_cleaning`` from config.toml (it stayed at the dataclass default
``False`` forever). These tests pin the wiring so that gate can't silently
break again.
"""

from __future__ import annotations

from dsc import config as config_module
from dsc.config import load_config


def test_load_config_reads_v2_cleaning_true(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text("v2_cleaning = true\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg)
    assert load_config().v2_cleaning is True


def test_load_config_v2_cleaning_defaults_true(tmp_path, monkeypatch):
    """Absent the key, V2 is now ON by default — turns are archived every turn
    anyway, so V2 reuses that work to reclaim context cheaply."""
    cfg = tmp_path / "config.toml"
    cfg.write_text("", encoding="utf-8")
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg)
    assert load_config().v2_cleaning is True


def test_load_config_v2_cleaning_opt_out(tmp_path, monkeypatch):
    """Explicit false in config.toml must be respected (opt-out path)."""
    cfg = tmp_path / "config.toml"
    cfg.write_text("v2_cleaning = false\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg)
    assert load_config().v2_cleaning is False
