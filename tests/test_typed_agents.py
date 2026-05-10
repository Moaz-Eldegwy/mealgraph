"""End-to-end-ish tests of the typed agent path with MockLLM.

These don't hit Gemini; they verify that an agent which received a typed
``CoachDecision`` / ``MedicalAssessmentDecision`` / ``PlannerDecision`` from
its LLM produces the expected state mutations.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from agents import CoachAgent, MedicalAssessmentAgent, PlannerAgent
from schemas import (
    Calculations,
    CoachDecision,
    MacroTargets,
    MedicalAssessmentDecision,
    MedicalAssessmentResult,
    PlannerDecision,
)
from state import initialize_empty_memory


# ---- Coach -----------------------------------------------------------------
def test_coach_emits_call_agent_action(mock_llm_factory) -> None:
    canned = CoachDecision(
        observation="needs assessment",
        thought="route to medical",
        response_steps=[],
        action="call_agent",
        params={"agent_name": "MedicalAssessmentAgent", "task": "assess"},
    )
    coach = CoachAgent(mock_llm_factory([canned]))
    state: Dict[str, Any] = {
        "memory": initialize_empty_memory(),
        "user_question": "make me a plan",
        "conversation_history": [{"role": "user", "content": "make me a plan"}],
        "current_action": None,
        "agent_result": None,
        "num_turns": 0,
        "max_turns": 10,
        "previous_actions": [],
        "response_steps": [],
    }
    out = coach.handle_task(state)
    assert out["current_action"]["action"] == "call_agent"
    assert out["current_action"]["params"]["agent_name"] == "MedicalAssessmentAgent"
    assert out["num_turns"] == 1


def test_coach_falls_back_when_decision_unparseable(mock_llm_factory) -> None:
    coach = CoachAgent(mock_llm_factory(["{not even close to JSON"]))
    state: Dict[str, Any] = {
        "memory": initialize_empty_memory(),
        "user_question": "anything",
        "conversation_history": [],
        "current_action": None,
        "agent_result": None,
        "num_turns": 0,
        "max_turns": 10,
        "previous_actions": [],
        "response_steps": [],
    }
    out = coach.handle_task(state)
    # Coach injects a compose_response with _parse_error so the workflow can short-circuit
    assert out["current_action"]["action"] == "compose_response"
    assert out["current_action"].get("_parse_error") is True


# ---- Medical ---------------------------------------------------------------
def test_medical_assessment_complete_writes_memory(mock_llm_factory) -> None:
    """A single assessment_complete decision should land in memory partition."""
    result = MedicalAssessmentResult(
        assessment_summary="healthy adult",
        flags_to_set=["maintenance"],
        recommendations=["balanced diet"],
        requires_professional_consultation=False,
        calculations=Calculations(
            BMI=22.0,
            BMR=1600,
            TDEE=2400,
            daily_target_calories=2400,
            macro_targets=MacroTargets(protein_g=150, fat_g=70, carbohydrates_g=300),
        ),
        evidence_sources=["who.int"],
        trace="Medical agent ran one iteration",
    )
    canned = MedicalAssessmentDecision(
        medical_reasoning="single-shot",
        observation="all data present",
        risk_assessment_priorities=["maintenance"],
        assessment_plan=[],
        action_type="assessment_complete",
        result=result,
    )
    # Need a stub for the tools (won't be called in single-iteration assessment_complete)
    class _StubTool:
        def handle_task(self, _: str) -> str:
            return ""

    agent = MedicalAssessmentAgent(mock_llm_factory([canned]), _StubTool(), _StubTool())
    memory = initialize_empty_memory()
    memory["user_profile"] = {
        "age": 30,
        "sex": "male",
        "height": 180,
        "weight": 75,
        "activity_level": "moderate",
        "allergies": [],
        "medications": [],
    }
    summary = agent.handle_task("assess this user", memory)

    assert summary == "healthy adult"
    fa = memory["flags_and_assessments"]
    assert fa["assessment_status"] == "assessment_complete"
    assert fa["calculations"]["macro_targets"]["protein_g"] == 150


def test_medical_ask_user_returns_field_list(mock_llm_factory) -> None:
    canned = MedicalAssessmentDecision(
        medical_reasoning="missing weight + height",
        observation="incomplete",
        risk_assessment_priorities=[],
        assessment_plan=[],
        action_type="ask_user",
        fields=["weight", "height"],
    )

    class _StubTool:
        def handle_task(self, _: str) -> str:
            return ""

    agent = MedicalAssessmentAgent(mock_llm_factory([canned]), _StubTool(), _StubTool())
    out = agent.handle_task("assess", initialize_empty_memory())
    assert "weight" in out and "height" in out


# ---- Planner ---------------------------------------------------------------
def test_planner_provide_plan_stores_to_memory(mock_llm_factory) -> None:
    canned = PlannerDecision(
        observation="ready",
        thought="finalising",
        planning_steps=[],
        action_type="provide_plan",
        final_plan={
            "days": [{"breakfast": "oats", "lunch": "chicken+rice"}],
            "trace": "Planner one-shot",
        },
    )

    class _StubTool:
        def handle_task(self, _: str) -> str:
            return ""

    agent = PlannerAgent(mock_llm_factory([canned]), _StubTool(), _StubTool(), _StubTool())
    memory = initialize_empty_memory()
    memory["flags_and_assessments"] = {"assessment_status": "assessment_complete"}
    out = agent.handle_task("make me a one-day plan", memory)

    assert "trace" in out
    assert memory["plans"]["current_plan"]["days"][0]["breakfast"] == "oats"
    assert "plan_timestamp" in memory["plans"]


def test_planner_error_payload_short_circuits(mock_llm_factory) -> None:
    canned = PlannerDecision(
        observation="missing assessment",
        thought="precondition violated",
        planning_steps=[],
        action_type="provide_plan",
        final_plan={"error": "flags_and_assessments empty; run MedicalAssessmentAgent first"},
    )

    class _StubTool:
        def handle_task(self, _: str) -> str:
            return ""

    agent = PlannerAgent(mock_llm_factory([canned]), _StubTool(), _StubTool(), _StubTool())
    out = agent.handle_task("make a plan", initialize_empty_memory())
    assert "error" in out
    assert "MedicalAssessmentAgent" in out
