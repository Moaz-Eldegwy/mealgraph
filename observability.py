"""Observability: LangSmith passthrough + lightweight in-process metrics.

Three pieces:

1. **LangSmith tracing** — opt-in via the standard ``LANGCHAIN_TRACING_V2``
   environment variable. LangGraph picks it up automatically; this module
   only surfaces a one-line confirmation at startup.

2. **MetricsCollector** — wraps :class:`utils.ParseMetrics` and adds
   per-agent latency, call count, and fallback-rate counters. The Gradio
   app renders these as a live system-health panel.

3. **Span context manager** — ad-hoc timing inside agents, logged through
   the standard ``nutrition_mas`` logger so the timing line is filterable
   like everything else.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Iterator, Optional

from logging_setup import get_logger
from utils import get_parse_metrics

_logger = get_logger("observability")


# ---------------------------------------------------------------------------
# LangSmith env passthrough
# ---------------------------------------------------------------------------
def init_langsmith(project: Optional[str] = None) -> bool:
    """If LangSmith env vars are set, log that tracing is on. Returns True."""
    if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() not in {"true", "1", "yes"}:
        return False
    api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    proj = project or os.environ.get("LANGCHAIN_PROJECT", "Nutrition-MAS")
    if not api_key:
        _logger.warning("LANGCHAIN_TRACING_V2 set but no LANGCHAIN_API_KEY; skipping.")
        return False
    os.environ["LANGCHAIN_PROJECT"] = proj
    _logger.info("📈 LangSmith tracing enabled (project=%s, key=…%s)", proj, api_key[-4:])
    return True


# ---------------------------------------------------------------------------
# In-process metrics
# ---------------------------------------------------------------------------
@dataclass
class AgentMetric:
    calls: int = 0
    total_seconds: float = 0.0
    last_seconds: float = 0.0
    errors: int = 0


@dataclass
class MetricsCollector:
    """Aggregate per-agent + per-tool counters. Process-singleton."""

    agents: Dict[str, AgentMetric] = field(default_factory=dict)
    tools: Dict[str, AgentMetric] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def record_agent(self, name: str, seconds: float, *, error: bool = False) -> None:
        with self._lock:
            m = self.agents.setdefault(name, AgentMetric())
            m.calls += 1
            m.total_seconds += seconds
            m.last_seconds = seconds
            if error:
                m.errors += 1

    def record_tool(self, name: str, seconds: float, *, error: bool = False) -> None:
        with self._lock:
            m = self.tools.setdefault(name, AgentMetric())
            m.calls += 1
            m.total_seconds += seconds
            m.last_seconds = seconds
            if error:
                m.errors += 1

    def snapshot(self) -> Dict[str, Dict[str, dict]]:
        """Return a JSON-serialisable snapshot including parse metrics."""
        pm = get_parse_metrics()
        with self._lock:
            return {
                "agents": {k: vars(v).copy() for k, v in self.agents.items() if k != "_lock"},
                "tools": {k: vars(v).copy() for k, v in self.tools.items() if k != "_lock"},
                "parsing": {
                    "native": pm.native_parses,
                    "fallback": pm.fallback_parses,
                    "failure": pm.schema_failures,
                    "by_model": dict(pm.by_model),
                },
            }

    def reset(self) -> None:
        with self._lock:
            self.agents.clear()
            self.tools.clear()


_collector = MetricsCollector()


def get_metrics() -> MetricsCollector:
    return _collector


# ---------------------------------------------------------------------------
# Timing span
# ---------------------------------------------------------------------------
@contextmanager
def span(label: str, *, kind: str = "agent") -> Iterator[None]:
    """Time a block; record into MetricsCollector and emit a debug log line.

    ``kind`` is one of 'agent' | 'tool' | 'misc' (misc only logs, no metric).
    """
    start = time.perf_counter()
    error = False
    try:
        yield
    except Exception:
        error = True
        raise
    finally:
        dur = time.perf_counter() - start
        if kind == "agent":
            _collector.record_agent(label, dur, error=error)
        elif kind == "tool":
            _collector.record_tool(label, dur, error=error)
        _logger.debug("⏱️  %s span %r took %.3fs (err=%s)", kind, label, dur, error)


__all__ = [
    "AgentMetric",
    "MetricsCollector",
    "get_metrics",
    "init_langsmith",
    "span",
]
