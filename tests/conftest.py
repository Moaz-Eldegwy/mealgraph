"""Shared pytest fixtures.

The mock LLM in this module is the workhorse for offline tests: it lets
agents run end-to-end without paying for Gemini calls. Each test passes
either typed Pydantic decisions (via ``call_typed``) or raw JSON strings
(legacy paths) that the agent expects to receive.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from config import reset_settings, set_settings
from utils import APIPoolManager, LLM


class MockLLM(LLM):
    """LLM stub that returns canned responses in order.

    Tests construct it with a list of either:

    * a JSON-string (returned as-is for the untyped __call__ path),
    * a dict (JSON-serialised on push; validated against the requested schema
      on call_typed),
    * a Pydantic ``BaseModel`` instance (returned as-is from call_typed; its
      ``.model_dump_json()`` is used for __call__).

    Each call pops the next item. Out-of-script calls raise so missing
    fixtures are noisy rather than silent.
    """

    def __init__(self, responses: List[Any]) -> None:
        self._responses: List[Any] = list(responses)
        self.calls: List[str] = []
        self.typed_calls: List[tuple[str, type]] = []

    def _next(self, prompt: str) -> Any:
        self.calls.append(prompt)
        if not self._responses:
            raise AssertionError(
                f"MockLLM ran out of canned responses (call #{len(self.calls)}). "
                f"Last prompt:\n{prompt[:300]}"
            )
        return self._responses.pop(0)

    def __call__(self, prompt: str, **_: Any) -> List[str]:
        item = self._next(prompt)
        if hasattr(item, "model_dump_json"):
            return [item.model_dump_json()]
        if isinstance(item, dict):
            return [json.dumps(item)]
        return [str(item)]

    def call_typed(self, prompt: str, response_model: type, **_: Any):
        from pydantic import BaseModel

        self.typed_calls.append((prompt, response_model))
        item = self._next(prompt)
        if isinstance(item, BaseModel):
            return item if isinstance(item, response_model) else None
        if isinstance(item, dict):
            try:
                return response_model.model_validate(item)
            except Exception:
                return None
        if isinstance(item, str):
            try:
                return response_model.model_validate_json(item)
            except Exception:
                return None
        return None

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
