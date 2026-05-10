"""Verify the new Pydantic ``Settings`` and the legacy ``config.X`` proxy."""

from __future__ import annotations

import config
from config import get_settings, reset_settings, set_settings


def test_defaults() -> None:
    s = get_settings()
    assert s.debug_mode is False
    assert s.debug_level == "full"
    assert s.enable_rate_limiting is True
    assert s.log_dir is None
    assert s.debug_scopes == {"agents": ["all"], "tools": ["all"]}


def test_set_settings_pydantic_names() -> None:
    set_settings(debug_mode=True, log_dir="/tmp/x")
    s = get_settings()
    assert s.debug_mode is True
    assert s.log_dir == "/tmp/x"


def test_set_settings_legacy_names() -> None:
    set_settings(DEBUG_MODE=True, LOG_DIR="/tmp/y", DEBUG_LEVEL="output")
    s = get_settings()
    assert s.debug_mode is True
    assert s.log_dir == "/tmp/y"
    assert s.debug_level == "output"


def test_pep562_legacy_reads() -> None:
    """Existing code that does ``config.DEBUG_MODE`` still works."""
    set_settings(debug_mode=True, debug_scopes={"agents": ["CoachAgent"], "tools": ["all"]})
    assert config.DEBUG_MODE is True
    assert config.DEBUG_SCOPES == {"agents": ["CoachAgent"], "tools": ["all"]}


def test_unknown_attr_raises() -> None:
    try:
        _ = config.NOT_A_THING  # type: ignore[attr-defined]
    except AttributeError as e:
        assert "NOT_A_THING" in str(e)
    else:
        raise AssertionError("Expected AttributeError")


def test_reset_settings_round_trip() -> None:
    set_settings(debug_mode=True)
    assert get_settings().debug_mode is True
    reset_settings()
    assert get_settings().debug_mode is False  # back to default
