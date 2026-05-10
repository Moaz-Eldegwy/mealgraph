"""Shared pytest fixtures.

The mock LLM here is the workhorse for offline tests — it lets us run agents
end-to-end without paying for Gemini calls. Phase 1 will give us schema-typed
agent responses; until then, each test passes the raw JSON string the agent
expects to receive.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from config import reset_settings, set_settings
from utils import APIPoolManager, LLM


class MockLLM(LLM):
    """LLM stub that returns canned responses in order.

    Tests construct it with a list of strings (or dicts, auto-JSON-serialised)
    and each call pops the next one. Out-of-script calls raise so missing
    fixtures are noisy rather than silent.
    """

    def __init__(self, responses: List[Any]) -> None:
        self.responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self.calls: List[str] = []

    def __call__(self, prompt: str, **_: Any) -> List[str]:
        self.calls.append(prompt)
        if not self.responses:
            raise AssertionError(
                f"MockLLM ran out of canned responses (call #{len(self.calls)}). "
                f"Last prompt:\n{prompt[:300]}"
            )
        return [self.responses.pop(0)]

    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


@pytest.fixture
def mock_llm_factory():
    """Factory to build a MockLLM from a list of canned responses."""
    return MockLLM


@pytest.fixture(autouse=True)
def fresh_settings():
    """Reset the Settings singleton before/after each test for isolation."""
    reset_settings()
    set_settings(debug_mode=False, log_dir=None, persistence_dir=None)
    yield
    reset_settings()


@pytest.fixture
def api_pool_no_limits():
    """An APIPoolManager with rate limiting disabled — for unit tests that
    don't care about throttling."""
    return APIPoolManager(["test-key-1", "test-key-2"], rate_limits=None)
