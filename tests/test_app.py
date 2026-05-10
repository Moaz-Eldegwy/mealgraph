"""Lightweight smoke tests for app.py.

Building the Gradio Blocks doesn't require the system to be initialised
(no API keys), so we can verify the UI compiles cleanly and the per-call
helpers work without ever launching a server.
"""

from __future__ import annotations

import json

import pytest


def test_app_imports() -> None:
    import app  # noqa: F401


def test_build_demo_compiles() -> None:
    """Calling build_demo() must not raise — catches Gradio API drift."""
    pytest.importorskip("gradio")
    from app import build_demo

    demo = build_demo()
    assert demo is not None


def test_build_user_profile_round_trip() -> None:
    from app import build_user_profile

    payload = build_user_profile(
        name="Test",
        age=30,
        sex="male",
        height_cm=175,
        weight_kg=72,
        activity="moderately active",
        goal="maintain weight",
        allergies="peanut, shrimp",
        dislikes="okra",
        country="Egypt",
        conditions="hypertension",
        medications="lisinopril",
    )
    # Round-trip via JSON to mirror what the hidden Textbox carries.
    serialised = json.dumps(payload)
    parsed = json.loads(serialised)

    assert parsed["user_profile"]["name"] == "Test"
    assert parsed["user_profile"]["allergies"] == ["peanut", "shrimp"]
    assert parsed["medical_history"]["conditions"] == ["hypertension"]


def test_render_metrics_is_markdown() -> None:
    from app import _render_metrics

    snap = {
        "agents": {"Coach": {"calls": 1, "total_seconds": 0.5, "errors": 0, "last_seconds": 0.5}},
        "tools": {"QuantitiesFinder": {"calls": 2, "total_seconds": 0.1, "errors": 0, "last_seconds": 0.05}},
        "parsing": {"native": 5, "fallback": 0, "failure": 0, "by_model": {}},
    }
    md = _render_metrics(snap)
    assert "Coach" in md
    assert "QuantitiesFinder" in md
    assert "native=5" in md


def test_session_state_default_shape() -> None:
    from app import SessionState

    s = SessionState()
    assert s.initialised is False
    assert s.memory == {
        "user_profile": {},
        "medical_history": {},
        "flags_and_assessments": {},
        "plans": {},
    }
    assert s.conversation_history == []


def test_chat_handles_uninitialised_system() -> None:
    """Calling chat() before init must not crash; returns a friendly error."""
    pytest.importorskip("gradio")
    from app import SessionState, chat

    # Make sure nutritionmas.APP is None so we hit the guard.
    import nutritionmas
    nutritionmas.APP = None

    history, log, metrics, session = chat(
        user_message="hi", history=[], session=SessionState(), profile_json=""
    )
    # messages-format chatbot: list of {role, content} dicts
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"].startswith("❌ System not initialised")
