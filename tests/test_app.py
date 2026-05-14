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
        lab_results="HbA1c 6.1%, LDL 145 mg/dL",
    )
    # Round-trip via JSON to mirror what the hidden Textbox carries.
    serialised = json.dumps(payload)
    parsed = json.loads(serialised)

    assert parsed["user_profile"]["name"] == "Test"
    assert parsed["user_profile"]["allergies"] == ["peanut", "shrimp"]
    assert parsed["medical_history"]["conditions"] == ["hypertension"]
    assert parsed["medical_history"]["lab_results"] == "HbA1c 6.1%, LDL 145 mg/dL"


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
    """Calling chat() before init must not crash; returns a friendly error.

    ``chat`` is a generator that streams (chat, trace, metrics, session,
    progress) tuples — drain to the last yield and inspect the final
    chatbot history.
    """
    pytest.importorskip("gradio")
    from app import SessionState, chat

    # Make sure mealgraph.APP is None so we hit the guard.
    import mealgraph
    mealgraph.APP = None

    last = None
    for last in chat(
        user_message="hi", history=[], session=SessionState(), profile_json=""
    ):
        pass
    assert last is not None
    history, _log, _metrics, _session, progress = last
    # messages-format chatbot: list of {role, content} dicts
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"].startswith("❌ System not initialised")
    # Progress panel should explain the early exit, not be empty.
    assert "system not initialised" in progress.lower()


def test_request_stop_with_no_active_run() -> None:
    """Pressing Stop with no in-flight run is a friendly no-op."""
    pytest.importorskip("gradio")
    from app import request_stop, _set_current_bus

    # Make sure no leftover bus from another test is hanging around.
    _set_current_bus(None)
    msg = request_stop()
    assert "No run is in progress" in msg


def test_progress_bus_check_stop_raises_after_set() -> None:
    """ProgressBus.check_stop is the gate every agent / tool wrapper hits
    on entry. Once the stop flag is flipped, the next check raises so the
    in-flight workflow unwinds at the next agent / tool boundary."""
    pytest.importorskip("gradio")
    from app import ProgressBus, StopRequested

    bus = ProgressBus()
    # Fresh bus: no raise.
    bus.check_stop()
    bus.stop_event.set()
    with pytest.raises(StopRequested):
        bus.check_stop()
