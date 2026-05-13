"""Centralised logging for MealGraph.

Agents and tools used to ``print`` directly to stdout. That worked in a notebook
but coupled the agentic system to the I/O layer. This module provides a single
``get_logger`` entrypoint so:

* user-mode emoji status lines flow through ``logger.info`` (visible by default),
* debug-mode raw LLM dumps flow through ``logger.debug`` (hidden unless
  ``settings.debug_mode`` is True),
* later phases can attach extra handlers (SSE event stream for the API, JSON
  file handler for trace persistence, etc.) without touching agent code.

Idempotent: calling :func:`configure_logging` more than once is a no-op unless
``force=True``.
"""

from __future__ import annotations

import logging
import sys

from config import get_settings

_BASE = "mealgraph"
_CONFIGURED = False


def configure_logging(*, force: bool = False) -> None:
    """Wire up the ``mealgraph`` logger tree.

    Reads ``settings.debug_mode`` to choose between INFO (user mode) and DEBUG.
    Safe to call from library code; only the first call attaches a handler.

    Reconfigures ``sys.stdout`` to UTF-8 when possible (Windows defaults to
    cp1252 which chokes on the emoji in agent status messages). Falls back to
    a 'replace' error handler so a stray glyph never crashes a log call.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    settings = get_settings()
    level = logging.DEBUG if settings.debug_mode else logging.INFO

    # Best-effort: re-encode stdout to UTF-8 so the emoji status lines render.
    reconf = getattr(sys.stdout, "reconfigure", None)
    if callable(reconf):
        try:
            reconf(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - depends on stream type
            pass

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger(_BASE)
    root.handlers = [handler]
    root.setLevel(level)
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a sub-logger under the ``mealgraph`` namespace.

    Conventional names: ``agents.coach``, ``agents.medical``, ``tools.computation``,
    ``utils.api_pool``.
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(f"{_BASE}.{name}")


def refresh_level() -> None:
    """Re-read ``settings.debug_mode`` and adjust handler levels in place.

    Call this after toggling debug mode at runtime.
    """
    settings = get_settings()
    level = logging.DEBUG if settings.debug_mode else logging.INFO
    root = logging.getLogger(_BASE)
    root.setLevel(level)
    for handler in root.handlers:
        handler.setLevel(level)


__all__ = ["configure_logging", "get_logger", "refresh_level"]
