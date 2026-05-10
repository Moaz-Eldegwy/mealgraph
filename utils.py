import os
import json
import re
import config
from typing import TypedDict, List, Optional, Dict, Any, Tuple
import pickle
from langgraph.checkpoint.base import BaseCheckpointSaver
from google import genai
from google.genai import types
from datetime import datetime, date
import time
from google.colab import userdata
from json_repair import repair_json
from collections import deque
from threading import Lock

# LANGSMITH SETUP FOR DEBUGGING
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = userdata.get("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_PROJECT"] = "Nutrition-MAS-v1"

def should_debug(scope: str, name: str) -> bool:
    if not config.DEBUG_MODE:
        return False
    if scope not in config.DEBUG_SCOPES:
        return False
    scopes_list = config.DEBUG_SCOPES[scope]
    return 'all' in scopes_list or name in scopes_list

def save_to_json(data: Dict[str, Any], filename: str, subdirectory: str = None):
    if config.LOG_DIR is None:
        # print("Logging is disabled. Skipping save_to_json.")
        return
    if subdirectory:
        log_dir = os.path.join(config.LOG_DIR, subdirectory)
    else:
        log_dir = config.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    filepath = os.path.join(log_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

class LLM:
    def __call__(self, prompt: str, **kwargs) -> list[str]:
        pass

    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        pass

class GeminiLLM(LLM):
    def __init__(self, model_name: str, structured_output: bool = False, thinking_budget: int = 300, manager=None, **kwargs):
        self.model_name = model_name
        self.structured_output = structured_output
        self.thinking_budget = thinking_budget
        self.kwargs = kwargs
        self.manager = manager

    def __call__(self, prompt: str, **kwargs) -> list[str]:
        if self.manager is None:
            raise ValueError("APIPoolManager must be provided for rate limiting.")

        merged_kwargs = {**self.kwargs, **kwargs}

        # Get next available API key
        api_key = self.manager.get_next_key(self.model_name)

        try:
            client = genai.Client(api_key=api_key)

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ]

            if self.structured_output:
                generate_content_config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=self.thinking_budget,
                    ),
                    response_mime_type="application/json",
                    max_output_tokens=merged_kwargs.get("max_tokens", 5120),
                    temperature=merged_kwargs.get("temperature", 0.3),
                )
            else:
                generate_content_config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=self.thinking_budget,
                    ),
                    response_mime_type="text/plain",
                    max_output_tokens=merged_kwargs.get("max_tokens", 5120),
                    temperature=merged_kwargs.get("temperature", 0.3),
                )

            response_text = ""
            start_time = time.time()
            for chunk in client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    response_text += chunk.text

            # Record usage only on successful completion
            completion_time = time.time()
            if self.manager.rate_limits is not None:
                self.manager.record_usage(api_key, self.model_name, completion_time)

            if config.DEBUG_MODE:
                print(f"LLM call completed for {self.model_name} using key {api_key[-4:]} in {completion_time - start_time:.2f}s")

            return [response_text.strip()]

        except Exception as e:
            # Do not record usage on error to avoid inflating limits for failed calls
            # print(f"LLM call failed for {self.model_name} using key {api_key[-4:]}: {str(e)}")
            return [f"Error: LLM call failed - {str(e)}"]

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

# In utils.py, update the GeminiLLM class as follows:

class GeminiLLM(LLM):
    def __init__(self, model_name: str, structured_output: bool = False, thinking_budget: int = 300, manager=None, **kwargs):
        self.model_name = model_name
        self.structured_output = structured_output
        self.thinking_budget = thinking_budget
        self.kwargs = kwargs
        self.manager = manager
        self.is_gemma = "gemma" in model_name.lower()
        if self.is_gemma:
            self.structured_output = False
            self.thinking_budget = None
        # No self.client or self.api_key; created dynamically

    def __call__(self, prompt: str, **kwargs) -> list[str]:
        if self.manager is None:
            raise ValueError("APIPoolManager must be provided for rate limiting.")

        merged_kwargs = {**self.kwargs, **kwargs}

        # Get next available API key
        api_key = self.manager.get_next_key(self.model_name)

        try:
            client = genai.Client(api_key=api_key)

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ]

            if self.is_gemma:
                generate_content_config = types.GenerateContentConfig(
                    response_mime_type="text/plain",
                    max_output_tokens=merged_kwargs.get("max_tokens", 5120),
                    temperature=merged_kwargs.get("temperature", 0.3),
                )
            else:
                if self.structured_output:
                    generate_content_config = types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=self.thinking_budget,
                        ),
                        response_mime_type="application/json",
                        max_output_tokens=merged_kwargs.get("max_tokens", 5120),
                        temperature=merged_kwargs.get("temperature", 0.3),
                    )
                else:
                    generate_content_config = types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=self.thinking_budget,
                        ),
                        response_mime_type="text/plain",
                        max_output_tokens=merged_kwargs.get("max_tokens", 5120),
                        temperature=merged_kwargs.get("temperature", 0.3),
                    )

            response_text = ""
            start_time = time.time()
            for chunk in client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    response_text += chunk.text

            # Record usage only on successful completion
            completion_time = time.time()
            if self.manager.rate_limits is not None:
                self.manager.record_usage(api_key, self.model_name, completion_time)

            if config.DEBUG_MODE:
                print(f"LLM call completed for {self.model_name} using key {api_key[-4:]} in {completion_time - start_time:.2f}s")

            return [response_text.strip()]

        except Exception as e:
            # Do not record usage on error to avoid inflating limits for failed calls
            # print(f"LLM call failed for {self.model_name} using key {api_key[-4:]}: {str(e)}")
            return [f"Error: LLM call failed - {str(e)}"]

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


class APIPoolManager:
    def __init__(self, api_keys: List[str], rate_limits: Optional[Dict[str, Tuple[int, int]]] = None):
        """
        rate_limits: { model_name: (RPM, RPD) }
        usage: { api_key: { model: { "timestamps": deque(maxlen=rpm), "daily_requests": int, "last_day": date } } }
        """
        self.api_keys = list(api_keys)
        self.active_keys = list(api_keys)
        self.rate_limits = rate_limits
        self.usage: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.current_index = 0
        self.lock = Lock()

        if rate_limits is not None:
            for key in api_keys:
                self.usage[key] = {}
                for model, (rpm, rpd) in rate_limits.items():
                    self.usage[key][model] = {
                        "timestamps": deque(maxlen=max(1, rpm)),
                        "daily_requests": 0,
                        "last_day": date.today()
                    }
        else:
            self.usage = {}

    def _refresh_daily(self, key: str, model: str):
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
            # drop this api key
            if key in self.active_keys:
                self.active_keys.remove(key)
            return False
        return True

    def _key_wait_info(self, key: str, model: str) -> Tuple[float, float]:
        """
        Return tuple (wait_slot_seconds, wait_spacing_seconds)
        - wait_slot_seconds: time until an RPM slot frees because deque is full (0 if slot available)
        - wait_spacing_seconds: time until spacing interval satisfied relative to last timestamp (0 if spacing ok)
        """
        if self.rate_limits is None:
            return 0.0, 0.0
        rpm, _ = self.rate_limits[model]
        usage = self.usage[key][model]
        now = time.time()

        # Clean old timestamps > 60s
        timestamps = usage["timestamps"]
        while timestamps and now - timestamps[0] > 60:
            timestamps.popleft()

        wait_slot = 0.0
        if len(timestamps) >= rpm:
            oldest = timestamps[0]
            wait_slot = max(0.0, 60.0 - (now - oldest))

        wait_spacing = 0.0
        if len(timestamps) > 0:
            time_since_last = now - timestamps[-1]
            min_interval = 60.0 / rpm if rpm > 0 else 0.0
            wait_spacing = max(0.0, min_interval - time_since_last)

        return wait_slot, wait_spacing

    def can_use_now(self, key: str, model: str) -> bool:
        """
        True if key is active, RPD ok, and both slot and spacing waits are zero.
        """
        if key not in self.active_keys:
            return False
        if not self._key_is_rpd_ok(key, model):
            return False
        wait_slot, wait_spacing = self._key_wait_info(key, model)
        return wait_slot <= 0.0 and wait_spacing <= 0.0

    def get_next_key(self, model: str, max_sleep_once: bool = True) -> str:
        """
        Choose an API key that can be used immediately for the given model.
        If none available now, compute minimum sleep needed across all keys, sleep once,
        then re-evaluate. Loop until a key is found or no keys left.
        """
        with self.lock:
            if not self.active_keys:
                raise RuntimeError("No available API keys left due to rate limits.")

            # Quick pass: try to find an immediately-available key starting from current_index
            n = len(self.active_keys)
            for i in range(n):
                idx = (self.current_index + i) % n
                key = self.active_keys[idx]
                if self.can_use_now(key, model):
                    # advance pointer fairly to next key for next call
                    self.current_index = (idx + 1) % max(1, len(self.active_keys))
                    return key

            # If we reach here: no key is available *right now*
            # compute minimal wait across active keys
            min_wait = None
            for key in list(self.active_keys):  # list() to be safe if removal happens
                if not self._key_is_rpd_ok(key, model):
                    continue
                wait_slot, wait_spacing = self._key_wait_info(key, model)
                wait = max(wait_slot, wait_spacing)
                if min_wait is None or wait < min_wait:
                    min_wait = wait

            if min_wait is None:
                # No keys left after RPD filtering
                raise RuntimeError("No available API keys left (RPD exhausted).")

        if min_wait and min_wait > 0:
            if max_sleep_once:
                time.sleep(min_wait)
            else:
                time.sleep(min_wait)
        return self.get_next_key(model, max_sleep_once=True)

    def record_usage(self, key: str, model: str, timestamp: Optional[float] = None):
        """
        Call this after you receive the response to record actual usage/time.
        timestamp default is now (time of completion).
        """
        if self.rate_limits is None:
            return
        t = timestamp or time.time()
        with self.lock:
            if key not in self.active_keys:
                # safety - if key was removed in-between, ignore or re-add depending on policy
                return
            self._refresh_daily(key, model)
            self.usage[key][model]["timestamps"].append(t)
            self.usage[key][model]["daily_requests"] += 1
            # Remove if daily limit reached
            _, rpd = self.rate_limits[model]
            if self.usage[key][model]["daily_requests"] >= rpd:
                if key in self.active_keys:
                    self.active_keys.remove(key)


def create_llm(config: dict, manager) -> LLM:
    if config["type"] == "gemini":
        structured_output = config.get("structured_output", False)
        thinking_budget = config.get("thinking_budget", 300)
        llm = GeminiLLM(
            model_name=config["model_name"],
            structured_output=structured_output,
            thinking_budget=thinking_budget,
            manager=manager,
            **config.get("params", {})
        )
        return llm
    else:
        raise ValueError(f"Unknown LLM type: {config['type']}")

def extract_and_parse_json(text: str) -> Dict[str, Any]:
    """Enhanced JSON extraction and parsing with multiple fallback strategies"""
    try:
        return json.loads(text.strip())
    except:
        pass

    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass

    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            repaired_json = repair_json(json_match.group(0))
            return json.loads(repaired_json)
        except:
            pass

    try:
        repaired_json = repair_json(text)
        return json.loads(repaired_json)
    except Exception as e:
        print(f"All JSON parsing strategies failed: {str(e)}")
        return {
            "thought": f"JSON parsing failed: {str(e)}",
            "action": "compose_response",
            "params": {
                "text": f"I encountered an error processing your request. Original response: {text[:200]}..."
            },
            "_parse_error": True,
            "_original_text": text
        }

def set_nested(d: Dict[str, Any], key: str, value: Any):
    keys = key.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value

def get_memory_summary(memory: Dict[str, Any], partitions: List[str] = None) -> str:
    """Get a formatted summary of specific memory partitions"""
    if partitions is None:
        partitions = ["user_profile", "medical_history", "flags_and_assessments", "plans"]

    summary = {}
    for partition in partitions:
        if partition in memory and memory[partition]:
            summary[partition] = memory[partition]
        else:
            summary[partition] = "empty"

    return json.dumps(summary, indent=2)

def update_memory_partition(memory: Dict[str, Any], partition: str, data: Any) -> None:
    """Safely update a memory partition with new data"""
    if partition not in memory:
        memory[partition] = {}

    if isinstance(data, dict) and isinstance(memory[partition], dict):
        memory[partition].update(data)
    else:
        memory[partition] = data
    if config.DEBUG_MODE:
        print(f"Updated memory partition '{partition}' with new data")

class FileCheckpointSaver(BaseCheckpointSaver):
    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def put(self, config: Dict[str, Any], checkpoint: Dict[str, Any]) -> None:
        """Save checkpoint to file"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        filepath = os.path.join(self.directory, f"checkpoint_{thread_id}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

    def get(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load checkpoint from file"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        filepath = os.path.join(self.directory, f"checkpoint_{thread_id}.pkl")
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None
