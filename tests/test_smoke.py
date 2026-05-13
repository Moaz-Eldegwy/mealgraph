"""Top-level smoke tests: every module must import cleanly outside Colab."""

from __future__ import annotations


def test_imports_work_outside_colab() -> None:
    """Every module imports cleanly in a plain Python process."""
    import agents  # noqa: F401
    import config  # noqa: F401
    import logging_setup  # noqa: F401
    import nutritionmas  # noqa: F401
    import state  # noqa: F401
    import tools  # noqa: F401
    import utils  # noqa: F401
    import workflow  # noqa: F401


def test_only_one_geminillm_class_in_utils() -> None:
    """A single ``GeminiLLM`` is exported from :mod:`utils` (no duplicates)."""
    import inspect

    import utils

    geminis = [
        cls
        for name, cls in inspect.getmembers(utils, inspect.isclass)
        if name == "GeminiLLM" and cls.__module__ == "utils"
    ]
    assert len(geminis) == 1


def test_initialize_empty_memory_shape() -> None:
    from state import initialize_empty_memory

    mem = initialize_empty_memory()
    assert set(mem.keys()) == {"user_profile", "medical_history", "flags_and_assessments", "plans"}
    assert all(v == {} for v in mem.values())


def test_default_model_configs_present() -> None:
    """Model topology is a contract the rest of the system depends on."""
    from nutritionmas import DEFAULT_MODEL_CONFIGS

    expected = {
        "main",
        "agents_llm",
        "tools_llm",
        "planner_agent",
        "validation_agent",
        "user_simulator",
    }
    assert set(DEFAULT_MODEL_CONFIGS.keys()) == expected


def test_create_llm_instances_requires_keys() -> None:
    import pytest

    from nutritionmas import create_llm_instances

    with pytest.raises(ValueError, match="At least one API key"):
        create_llm_instances([])
