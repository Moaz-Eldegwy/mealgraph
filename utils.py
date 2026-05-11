"""Utilities: LLM wrapper, API-key pool with rate limiting, JSON helpers, and a
LangGraph file checkpointer.

Phase 0 cleanup notes:

* Removed the duplicate ``GeminiLLM`` definition (the second class silently
  shadowed the first; both remained import-visible).
* Dropped ``from google.colab import userdata`` so the module imports cleanly
  outside Colab. API keys come in via ``create_llm_instances`` or env.
* Replaced ``print(...)`` calls with module loggers under ``nutrition_mas.*``.
* Routed all reads of ``config.X`` through :func:`config.get_settings`.

Larger refactors (Pydantic-typed agent IO, native Gemini ``response_schema``,
async ``acall``) land in Phase 1.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from google import genai
from google.genai import types
from json_repair import repair_json
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, ValidationError

from config import get_settings
from logging_setup import get_logger

_logger = get_logger("utils")
_llm_logger = get_logger("llm.gemini")
_pool_logger = get_logger("utils.api_pool")

T = TypeVar("T", bound=BaseModel)


# --- Phase 1 fallback metrics --------------------------------------------------
@dataclass
class ParseMetrics:
    """Counts native-vs-fallback parses across the process.

    Phase 1's goal is to drive ``fallback_parses`` to zero. Phase 2 will surface
    these via the eval harness.
    """

    native_parses: int = 0  # response.parsed worked first try
    fallback_parses: int = 0  # had to invoke extract_and_parse_json
    schema_failures: int = 0  # output failed Pydantic validation altogether
    by_model: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record(self, model: str, kind: str) -> None:
        if kind == "native":
            self.native_parses += 1
        elif kind == "fallback":
            self.fallback_parses += 1
        elif kind == "failure":
            self.schema_failures += 1
        slot = self.by_model.setdefault(model, {"native": 0, "fallback": 0, "failure": 0})
        slot[kind] = slot.get(kind, 0) + 1


_parse_metrics = ParseMetrics()


def get_parse_metrics() -> ParseMetrics:
    """Return the global parse-metrics singleton (read-only-ish)."""
    return _parse_metrics


# --- Debug-scope helper --------------------------------------------------------
def should_debug(scope: str, name: str) -> bool:
    """Return True when this scope/name is enabled in ``settings.debug_scopes``."""
    settings = get_settings()
    if not settings.debug_mode:
        return False
    if scope not in settings.debug_scopes:
        return False
    scopes_list = settings.debug_scopes[scope]
    return "all" in scopes_list or name in scopes_list


# --- Filesystem logging --------------------------------------------------------
def save_to_json(data: Dict[str, Any], filename: str, subdirectory: Optional[str] = None) -> None:
    """Persist a structured payload to ``settings.log_dir`` if logging is on."""
    settings = get_settings()
    if settings.log_dir is None:
        return
    log_dir = os.path.join(settings.log_dir, subdirectory) if subdirectory else settings.log_dir
    os.makedirs(log_dir, exist_ok=True)
    # Filenames may contain ``:`` from ISO timestamps which is invalid on Windows.
    safe_name = filename.replace(":", "-")
    filepath = os.path.join(log_dir, safe_name)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# --- LLM abstractions ----------------------------------------------------------
class LLM:
    """Minimal LLM contract: callable returning a list with one string."""

    def __call__(self, prompt: str, **kwargs: Any) -> list[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def format_prompt(self, messages: List[Dict[str, str]]) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class GeminiLLM(LLM):
    """Synchronous Gemini wrapper with API-key pooling.

    Phase 1 will add an ``acall`` async path and replace the JSON-in-text
    contract with native ``response_schema`` Pydantic models.
    """

    def __init__(
        self,
        model_name: str,
        structured_output: bool = False,
        thinking_budget: int = 300,
        manager: Optional["APIPoolManager"] = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.structured_output = structured_output
        self.thinking_budget = thinking_budget
        self.kwargs = kwargs
        self.manager = manager
        self.is_gemma = "gemma" in model_name.lower()
        if self.is_gemma:
            # Gemma family doesn't support thinking_config or JSON response schema.
            self.structured_output = False
            self.thinking_budget = None

    def __call__(self, prompt: str, **kwargs: Any) -> list[str]:
        """Untyped streaming call. Returns ``[response_text]``.

        Backwards-compat path used by code that still parses JSON-from-text.
        Prefer :meth:`call_typed` when a Pydantic schema is available.
        """
        text, _ = self._invoke(prompt, response_schema=None, **kwargs)
        return [text]

    def call_typed(
        self,
        prompt: str,
        response_model: Type[T],
        **kwargs: Any,
    ) -> Optional[T]:
        """Call Gemini with constrained-decoded JSON matching ``response_model``.

        Returns a parsed instance of ``response_model``, or ``None`` if every
        parse strategy failed (in which case the parse-metrics ``schema_failures``
        counter is incremented so the eval harness can spot it).
        """
        text, parsed = self._invoke(prompt, response_schema=response_model, **kwargs)

        # Strategy 1: SDK already parsed it for us via response_schema.
        if isinstance(parsed, response_model):
            _parse_metrics.record(self.model_name, "native")
            return parsed

        # Strategy 2: SDK gave us a dict; try to validate it.
        if isinstance(parsed, dict):
            try:
                instance = response_model.model_validate(parsed)
                _parse_metrics.record(self.model_name, "native")
                return instance
            except ValidationError as e:
                _llm_logger.debug("response.parsed dict failed Pydantic validation: %s", e)

        # Strategy 3: regex / json_repair fallback on the raw text.
        try:
            data = extract_and_parse_json(text)
            instance = response_model.model_validate(data)
            _parse_metrics.record(self.model_name, "fallback")
            _llm_logger.warning(
                "Used JSON-repair fallback for %s on model %s — fix the prompt or schema",
                response_model.__name__,
                self.model_name,
            )
            return instance
        except (ValidationError, Exception) as e:  # noqa: BLE001
            _parse_metrics.record(self.model_name, "failure")
            _llm_logger.error(
                "Failed to parse %s from %s response: %s",
                response_model.__name__,
                self.model_name,
                str(e),
            )
            return None

    def call_grounded(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Tuple[str, List[Dict[str, str]], List[str]]:
        """Single grounded call using Gemini's built-in ``google_search`` tool.

        Gemini handles the whole search loop internally: it generates queries,
        runs them against Google Search, synthesises an answer, and returns
        ``groundingMetadata`` with the sources it relied on.

        Returns ``(text, citations, queries)`` where ``citations`` is a list
        of ``{"title": str, "uri": str}`` derived from
        ``grounding_chunks`` and ``queries`` is the actual list of search
        strings Gemini ran (useful for debugging).
        """
        if self.manager is None:
            raise ValueError("APIPoolManager must be provided for rate limiting.")
        if self.is_gemma:
            raise ValueError("Gemma models do not support google_search grounding.")

        merged_kwargs = {**self.kwargs, **kwargs}
        api_key = self.manager.get_next_key(self.model_name)

        try:
            client = genai.Client(api_key=api_key)
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
            generate_content_config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                max_output_tokens=merged_kwargs.get("max_tokens", 5120),
                temperature=merged_kwargs.get("temperature", 0.3),
            )

            start_time = time.time()
            response = client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            )
            completion_time = time.time()
            if self.manager.rate_limits is not None:
                self.manager.record_usage(api_key, self.model_name, completion_time)

            text = (response.text or "").strip()
            citations: List[Dict[str, str]] = []
            queries: List[str] = []
            try:
                candidate = response.candidates[0]
                gm = getattr(candidate, "grounding_metadata", None)
                if gm is not None:
                    for chunk in getattr(gm, "grounding_chunks", None) or []:
                        web = getattr(chunk, "web", None)
                        if web and getattr(web, "uri", None):
                            citations.append(
                                {"title": web.title or web.uri, "uri": web.uri}
                            )
                    queries = list(getattr(gm, "web_search_queries", None) or [])
            except (AttributeError, IndexError):
                pass

            _llm_logger.debug(
                "Grounded LLM call completed for %s using key …%s in %.2fs (%d citations, %d queries)",
                self.model_name,
                api_key[-4:],
                completion_time - start_time,
                len(citations),
                len(queries),
            )
            return text, citations, queries

        except Exception as e:  # noqa: BLE001
            _llm_logger.warning(
                "Grounded LLM call failed for %s using key …%s: %s",
                self.model_name,
                api_key[-4:],
                str(e),
            )
            return f"Error: grounded LLM call failed - {str(e)}", [], []

    def _invoke(
        self,
        prompt: str,
        response_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Tuple[str, Any]:
        """Single Gemini round-trip. Returns ``(text, response.parsed)``.

        ``parsed`` is whatever the SDK populated on ``response.parsed`` —
        usually a Pydantic instance when ``response_schema`` is supplied, ``None``
        otherwise.
        """
        if self.manager is None:
            raise ValueError("APIPoolManager must be provided for rate limiting.")

        merged_kwargs = {**self.kwargs, **kwargs}
        api_key = self.manager.get_next_key(self.model_name)

        try:
            client = genai.Client(api_key=api_key)
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
            generate_content_config = self._build_config(merged_kwargs, response_schema=response_schema)

            start_time = time.time()
            # Non-streaming when we want response.parsed (the streaming API
            # doesn't populate it). Streaming for free-text plain calls.
            if response_schema is not None:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=generate_content_config,
                )
                response_text = response.text or ""
                parsed = getattr(response, "parsed", None)
            else:
                response_text = ""
                parsed = None
                for chunk in client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if chunk.text:
                        response_text += chunk.text

            completion_time = time.time()
            if self.manager.rate_limits is not None:
                self.manager.record_usage(api_key, self.model_name, completion_time)

            _llm_logger.debug(
                "LLM call completed for %s using key …%s in %.2fs (schema=%s)",
                self.model_name,
                api_key[-4:],
                completion_time - start_time,
                response_schema.__name__ if response_schema else "none",
            )
            return response_text.strip(), parsed

        except Exception as e:  # noqa: BLE001 — narrow this in Phase 4 (per-error retries)
            _llm_logger.warning(
                "LLM call failed for %s using key …%s: %s",
                self.model_name,
                api_key[-4:],
                str(e),
            )
            return f"Error: LLM call failed - {str(e)}", None

    def _build_config(
        self,
        merged_kwargs: Dict[str, Any],
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> types.GenerateContentConfig:
        max_tokens = merged_kwargs.get("max_tokens", 5120)
        temperature = merged_kwargs.get("temperature", 0.3)

        if self.is_gemma:
            # Gemma can't do thinking_config or response_schema.
            return types.GenerateContentConfig(
                response_mime_type="text/plain",
                max_output_tokens=max_tokens,
                temperature=temperature,
            )

        thinking_cfg = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        if response_schema is not None:
            return types.GenerateContentConfig(
                thinking_config=thinking_cfg,
                response_mime_type="application/json",
                response_schema=response_schema,
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        mime = "application/json" if self.structured_output else "text/plain"
        return types.GenerateContentConfig(
            thinking_config=thinking_cfg,
            response_mime_type=mime,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant:"
        return prompt


# --- API key pool with optional rate limiting ----------------------------------
class APIPoolManager:
    """Round-robin Gemini API keys with per-key RPM/RPD enforcement.

    ``rate_limits`` is ``{model_name: (rpm, rpd)}``. When ``None``, the pool
    just rotates keys without any throttling.
    """

    def __init__(
        self,
        api_keys: List[str],
        rate_limits: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> None:
        self.api_keys = list(api_keys)
        self.active_keys = list(api_keys)
        self.rate_limits = rate_limits
        self.usage: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.current_index = 0
        self.lock = Lock()

        if rate_limits is not None:
            for key in api_keys:
                self.usage[key] = {}
                for model, (rpm, _rpd) in rate_limits.items():
                    self.usage[key][model] = {
                        "timestamps": deque(maxlen=max(1, rpm)),
                        "daily_requests": 0,
                        "last_day": date.today(),
                    }

    # --- internal helpers ------------------------------------------------------
    def _refresh_daily(self, key: str, model: str) -> None:
        usage = self.usage[key][model]
        today = date.today()
        if usage["last_day"] < today:
            usage["daily_requests"] = 0
            usage["last_day"] = today

    def _key_is_rpd_ok(self, key: str, model: str) -> bool:
        if self.rate_limits is None:
            return True
        self._refresh_daily(key, model)
        _, rpd = self.rate_limits[model]
        if self.usage[key][model]["daily_requests"] >= rpd:
            if key in self.active_keys:
                self.active_keys.remove(key)
            return False
        return True

    def _key_wait_info(self, key: str, model: str) -> Tuple[float, float]:
        if self.rate_limits is None:
            return 0.0, 0.0
        rpm, _ = self.rate_limits[model]
        usage = self.usage[key][model]
        now = time.time()

        timestamps = usage["timestamps"]
        while timestamps and now - timestamps[0] > 60:
            timestamps.popleft()

        wait_slot = 0.0
        if len(timestamps) >= rpm:
            oldest = timestamps[0]
            wait_slot = max(0.0, 60.0 - (now - oldest))

        wait_spacing = 0.0
        if timestamps:
            time_since_last = now - timestamps[-1]
            min_interval = 60.0 / rpm if rpm > 0 else 0.0
            wait_spacing = max(0.0, min_interval - time_since_last)

        return wait_slot, wait_spacing

    def can_use_now(self, key: str, model: str) -> bool:
        if key not in self.active_keys:
            return False
        if not self._key_is_rpd_ok(key, model):
            return False
        wait_slot, wait_spacing = self._key_wait_info(key, model)
        return wait_slot <= 0.0 and wait_spacing <= 0.0

    # --- public API ------------------------------------------------------------
    def get_next_key(self, model: str, max_sleep_once: bool = True) -> str:
        with self.lock:
            if not self.active_keys:
                raise RuntimeError("No available API keys left due to rate limits.")

            n = len(self.active_keys)
            for i in range(n):
                idx = (self.current_index + i) % n
                key = self.active_keys[idx]
                if self.can_use_now(key, model):
                    self.current_index = (idx + 1) % max(1, len(self.active_keys))
                    return key

            min_wait: Optional[float] = None
            for key in list(self.active_keys):
                if not self._key_is_rpd_ok(key, model):
                    continue
                wait_slot, wait_spacing = self._key_wait_info(key, model)
                wait = max(wait_slot, wait_spacing)
                if min_wait is None or wait < min_wait:
                    min_wait = wait

            if min_wait is None:
                raise RuntimeError("No available API keys left (RPD exhausted).")

        if min_wait and min_wait > 0:
            _pool_logger.debug("Waiting %.2fs for next API slot", min_wait)
            time.sleep(min_wait)
        return self.get_next_key(model, max_sleep_once=True)

    def record_usage(self, key: str, model: str, timestamp: Optional[float] = None) -> None:
        if self.rate_limits is None:
            return
        t = timestamp or time.time()
        with self.lock:
            if key not in self.active_keys:
                return
            self._refresh_daily(key, model)
            self.usage[key][model]["timestamps"].append(t)
            self.usage[key][model]["daily_requests"] += 1
            _, rpd = self.rate_limits[model]
            if self.usage[key][model]["daily_requests"] >= rpd:
                if key in self.active_keys:
                    self.active_keys.remove(key)


# --- Factory -------------------------------------------------------------------
def create_llm(config: dict, manager: APIPoolManager) -> LLM:
    """Instantiate an LLM from a config dict."""
    if config["type"] == "gemini":
        return GeminiLLM(
            model_name=config["model_name"],
            structured_output=config.get("structured_output", False),
            thinking_budget=config.get("thinking_budget", 300),
            manager=manager,
            **config.get("params", {}),
        )
    raise ValueError(f"Unknown LLM type: {config['type']}")


# --- JSON helpers --------------------------------------------------------------
def extract_and_parse_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction with a chain of fallbacks.

    Phase 1 makes this a measured *fallback* path only — agents will use
    Gemini's native ``response_schema`` for guaranteed structure. Until then,
    this remains the primary parser.
    """
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    braces = re.search(r"\{.*\}", text, re.DOTALL)
    if braces:
        try:
            return json.loads(repair_json(braces.group(0)))
        except Exception:
            pass

    try:
        return json.loads(repair_json(text))
    except Exception as e:
        _logger.warning("All JSON parsing strategies failed: %s", str(e))
        return {
            "thought": f"JSON parsing failed: {str(e)}",
            "action": "compose_response",
            "params": {"text": f"I encountered an error processing your request. Original response: {text[:200]}..."},
            "_parse_error": True,
            "_original_text": text,
        }


def set_nested(d: Dict[str, Any], key: str, value: Any) -> None:
    """Assign ``value`` at a dotted-path key inside a nested dict."""
    keys = key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def get_memory_summary(memory: Dict[str, Any], partitions: Optional[List[str]] = None) -> str:
    """Format selected memory partitions as JSON for prompt embedding."""
    if partitions is None:
        partitions = ["user_profile", "medical_history", "flags_and_assessments", "plans"]
    summary: Dict[str, Any] = {}
    for partition in partitions:
        summary[partition] = memory[partition] if partition in memory and memory[partition] else "empty"
    return json.dumps(summary, indent=2, default=str)


def update_memory_partition(memory: Dict[str, Any], partition: str, data: Any) -> None:
    """Merge ``data`` into ``memory[partition]`` (or assign when types disagree)."""
    if partition not in memory:
        memory[partition] = {}
    if isinstance(data, dict) and isinstance(memory[partition], dict):
        memory[partition].update(data)
    else:
        memory[partition] = data
    _logger.debug("Updated memory partition %r with new data", partition)


# --- Checkpointer --------------------------------------------------------------
class FileCheckpointSaver(BaseCheckpointSaver):
    """Pickle LangGraph checkpoints to ``directory/checkpoint_<thread_id>.pkl``."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def put(self, config: Dict[str, Any], checkpoint: Dict[str, Any]) -> None:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        filepath = os.path.join(self.directory, f"checkpoint_{thread_id}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint, f)

    def get(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        filepath = os.path.join(self.directory, f"checkpoint_{thread_id}.pkl")
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return pickle.load(f)
        return None


__all__ = [
    "APIPoolManager",
    "FileCheckpointSaver",
    "GeminiLLM",
    "LLM",
    "create_llm",
    "extract_and_parse_json",
    "get_memory_summary",
    "save_to_json",
    "set_nested",
    "should_debug",
    "update_memory_partition",
]
