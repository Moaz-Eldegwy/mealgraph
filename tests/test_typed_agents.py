"""End-to-end-ish tests of the typed agent path with MockLLM.

These don't hit Gemini; they verify that an agent which received a typed
``CoachDecision`` / ``MedicalAssessmentDecision`` / ``PlannerDecision`` from
its LLM produces the expected state mutations.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agents import CoachAgent, MedicalAssessmentAgent, PlannerAgent
from schemas import (
    CoachDecision,
    MedicalAssessmentDecision,
    MedicalAssessmentResult,
    PlannerDecision,
)
from state import initialize_empty_memory


def _profile(**overrides: Any) -> Dict[str, Any]:
    base = {
        "age": 30,
        "sex": "male",
        "height": 180,
        "weight": 75,
        "activity_level": "moderately active",
        "goal": "maintain weight",
        "allergies": [],
        "medications": [],
    }
    base.update(overrides)
    return base


class _StubTool:
    def __init__(self, responses: List[str] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: List[str] = []

    def handle_task(self, task: str) -> str:
        self.calls.append(task)
        if not self._responses:
            return ""
        return self._responses.pop(0)


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
    assert out["current_action"]["action"] == "compose_response"
    assert out["current_action"].get("_parse_error") is True


# ---- Medical ---------------------------------------------------------------
def test_medical_short_circuits_when_critical_fields_missing(mock_llm_factory) -> None:
    """Pre-LLM check: if anthropometric fields are missing, we shouldn't
    waste a Gemini call."""
    llm = mock_llm_factory([])  # no canned responses; LLM must not be called
    agent = MedicalAssessmentAgent(llm, _StubTool())
    memory = initialize_empty_memory()
    # Leave user_profile empty.
    out = agent.handle_task("assess", memory)
    assert "Missing critical fields" in out
    assert llm.typed_calls == []  # never called


def test_medical_overwrites_calculations_with_deterministic_values(
    mock_llm_factory,
) -> None:
    """Even if the LLM emits the wrong calculations, the agent overwrites
    them with the deterministic result of ``full_assessment``."""
    # LLM tries to claim BMI=99.9 — but the deterministic math says ~22.
    result = MedicalAssessmentResult(
        assessment_summary="healthy adult",
        flags_to_set=["maintenance"],
        recommendations=["balanced diet"],
        requires_professional_consultation=False,
        calculations={
            "BMI": 99.9,
            "BMR": 99.9,
            "TDEE": 99.9,
            "daily_target_calories": 99,
            "macro_targets": {"protein_g": 1, "fat_g": 1, "carbohydrates_g": 1},
        },
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

    agent = MedicalAssessmentAgent(mock_llm_factory([canned]), _StubTool())
    memory = initialize_empty_memory()
    memory["user_profile"] = _profile()

    summary = agent.handle_task("assess this user", memory)
    assert summary == "healthy adult"

    fa = memory["flags_and_assessments"]
    assert fa["assessment_status"] == "assessment_complete"
    calc = fa["calculations"]
    # Deterministic math: full_assessment(75, 180, 30, male, moderately active,
    # maintain weight) -> BMI ~23.15, not 99.9.
    assert 20.0 < calc["BMI"] < 25.0
    # LLM's bogus 99 kcal target is overwritten with the real one.
    assert calc["daily_target_calories"] > 2000


def test_medical_deterministic_fallback_when_llm_never_finalises(
    mock_llm_factory,
) -> None:
    """If the LLM never returns ``assessment_complete``, the agent persists
    a deterministic-only assessment so the Coach can still proceed."""
    # Single canned decision that calls a tool — never finalises.
    canned = MedicalAssessmentDecision(
        medical_reasoning="just exploring",
        observation="...",
        risk_assessment_priorities=[],
        assessment_plan=[],
        action_type="call_tool",
        tool_task="look up something",
    )
    # Repeat the same decision MAX_ITERATIONS times so the loop exhausts.
    llm = mock_llm_factory(
        [canned] * MedicalAssessmentAgent.MAX_ITERATIONS
    )
    agent = MedicalAssessmentAgent(llm, _StubTool(["search result"]))
    memory = initialize_empty_memory()
    memory["user_profile"] = _profile()

    out = agent.handle_task("assess", memory)
    assert "Deterministic assessment only" in out
    fa = memory["flags_and_assessments"]
    assert fa["assessment_status"] == "assessment_complete"
    assert fa["data_confidence"] == 0.6


def test_medical_ask_user_returns_field_list(mock_llm_factory) -> None:
    """When fields are missing, the agent precheck short-circuits with a
    clear list of what's needed."""

    agent = MedicalAssessmentAgent(mock_llm_factory([]), _StubTool())
    memory = initialize_empty_memory()
    memory["user_profile"] = {"name": "incomplete"}
    out = agent.handle_task("assess", memory)
    assert "Missing critical fields" in out


# ---- Planner ---------------------------------------------------------------
def test_planner_provide_plan_stores_to_memory(mock_llm_factory) -> None:
    """A plan that doesn't trigger any deterministic issues is persisted
    and returned wrapped in the {plan, revisions, unresolved_issues} envelope."""
    canned = PlannerDecision(
        observation="ready",
        thought="finalising",
        planning_steps=[],
        action_type="provide_plan",
        final_plan={
            "days": [{"name": "oats and chicken", "calories": 500, "protein_g": 30}],
            "trace": "Planner one-shot",
        },
    )

    agent = PlannerAgent(mock_llm_factory([canned]), _StubTool(), _StubTool())
    memory = initialize_empty_memory()
    memory["flags_and_assessments"] = {"assessment_status": "assessment_complete"}
    out = agent.handle_task("make me a one-day plan", memory)

    # The envelope contains plan + revisions + unresolved_issues.
    import json as _json

    payload = _json.loads(out)
    assert "plan" in payload and "revisions" in payload
    assert payload["revisions"] == 0
    assert memory["plans"]["current_plan"]["days"][0]["name"] == "oats and chicken"
    assert "plan_timestamp" in memory["plans"]


def test_planner_error_payload_short_circuits(mock_llm_factory) -> None:
    canned = PlannerDecision(
        observation="missing assessment",
        thought="precondition violated",
        planning_steps=[],
        action_type="provide_plan",
        final_plan={"error": "flags_and_assessments empty; run MedicalAssessmentAgent first"},
    )

    agent = PlannerAgent(mock_llm_factory([canned]), _StubTool(), _StubTool())
    out = agent.handle_task("make a plan", initialize_empty_memory())
    assert "error" in out
    assert "MedicalAssessmentAgent" in out


def test_planner_post_lp_allergy_triggers_revision(mock_llm_factory) -> None:
    """When the first finalised plan contains an allergen, the Planner
    revises internally without involving the Coach. The second canned
    decision (peanut-free) is the one that lands in memory."""
    bad = PlannerDecision(
        observation="first attempt",
        thought="finalising",
        planning_steps=[],
        action_type="provide_plan",
        final_plan={
            "days": [
                {"name": "peanut butter oats", "calories": 500, "protein_g": 20}
            ],
            "trace": "first draft",
        },
    )
    good = PlannerDecision(
        observation="revision",
        thought="dropped allergen",
        planning_steps=[],
        action_type="provide_plan",
        final_plan={
            "days": [
                {"name": "almond butter oats", "calories": 500, "protein_g": 20}
            ],
            "trace": "second draft (allergen removed)",
        },
    )

    agent = PlannerAgent(mock_llm_factory([bad, good]), _StubTool(), _StubTool())
    memory = initialize_empty_memory()
    memory["user_profile"] = {"allergies": ["peanut"]}
    memory["flags_and_assessments"] = {"assessment_status": "assessment_complete"}

    out = agent.handle_task("plan", memory)
    import json as _json

    payload = _json.loads(out)
    # First plan was rejected -> revisions == 1.
    assert payload["revisions"] == 1
    # The persisted plan is the revised one, with no peanut.
    persisted = memory["plans"]["current_plan"]
    assert "peanut" not in persisted["days"][0]["name"].lower()
    # No high-severity issue left on the final plan.
    assert payload["unresolved_issues"] == []
