"""MealGraph configuration.

This module exposes a Pydantic-Settings ``Settings`` singleton plus a small
backward-compatibility shim so existing code can still read ``config.DEBUG_MODE``,
``config.LOG_DIR``, etc. New code should import :func:`get_settings` directly::

    from config import get_settings
    settings = get_settings()
    if settings.debug_mode:
        ...

Mutation must go through :func:`set_settings` (the legacy ``config.X = y`` write
pattern would otherwise silently shadow the Pydantic value).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration.

    Values are loaded from (in order of precedence): direct ``set_settings``
    calls, environment variables prefixed with ``MEALGRAPH_``, the ``.env``
    file in the project root, and the defaults declared here.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEALGRAPH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_assignment=True,
    )

    # --- Logging / persistence -------------------------------------------------
    log_dir: Optional[str] = None
    persistence_dir: Optional[str] = None

    # --- Debug switches --------------------------------------------------------
    debug_mode: bool = False
    debug_level: str = "full"  # 'full' or 'output'
    debug_scopes: Dict[str, List[str]] = Field(
        default_factory=lambda: {"agents": ["all"], "tools": ["all"]}
    )

    # --- LLM / rate limiting ---------------------------------------------------
    enable_rate_limiting: bool = True
    gemini_api_keys: List[str] = Field(default_factory=list)


# Singleton holder. Instantiated lazily so tests can set env vars before first read.
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the process-wide ``Settings`` instance, creating it on first call."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Drop the cached singleton so the next ``get_settings`` call re-reads env.

    Intended for use in tests.
    """
    global _settings
    _settings = None


def set_settings(**updates: Any) -> Settings:
    """Update fields on the singleton ``Settings``.

    Accepts both legacy upper-case names (``DEBUG_MODE``) and Pydantic field
    names (``debug_mode``). Returns the updated settings instance.
    """
    s = get_settings()
    for raw_key, value in updates.items():
        attr = _LEGACY_ATTR_MAP.get(raw_key, raw_key.lower())
        if not hasattr(s, attr):
            raise AttributeError(f"Settings has no attribute {attr!r}")
        setattr(s, attr, value)
    return s


# --- Legacy attribute proxy ----------------------------------------------------
# Existing code does ``import config`` then reads ``config.DEBUG_MODE`` etc.
# PEP 562 ``__getattr__`` lets us forward those reads to the singleton.
_LEGACY_ATTR_MAP: Dict[str, str] = {
    "DEBUG_MODE": "debug_mode",
    "DEBUG_LEVEL": "debug_level",
    "DEBUG_SCOPES": "debug_scopes",
    "LOG_DIR": "log_dir",
    "PERSISTENCE_DIR": "persistence_dir",
    "ENABLE_RATE_LIMITING": "enable_rate_limiting",
}


def __getattr__(name: str) -> Any:  # noqa: D401 — module-level dunder
    """PEP 562: forward legacy CONST-style reads to the Settings singleton."""
    if name in _LEGACY_ATTR_MAP:
        return getattr(get_settings(), _LEGACY_ATTR_MAP[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Settings", "get_settings", "reset_settings", "set_settings"]
