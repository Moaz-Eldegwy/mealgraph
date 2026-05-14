"""Validate the Pydantic schemas that anchor every agent decision."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas import (
    Calculations,
    CoachDecision,
    FinalPlan,
    FoodItem,
    MacroTargets,
    MedicalAssessmentDecision,
    MedicalAssessmentResult,
    PlannerDecision,
    ResponseStep,
)


# ---- Coach -----------------------------------------------------------------
def test_coach_decision_call_agent() -> None:
    d = CoachDecision(
        observation="user wants a plan",
        thought="need assessment first",
        response_steps=[
            ResponseStep(id=1, actor="MedicalAssessmentAgent", description="assess"),
        ],
        action="call_agent",
        params={"agent_name": "MedicalAssessmentAgent", "task": "assess user"},
    )
    assert d.action == "call_agent"
    assert d.params["agent_name"] == "MedicalAssessmentAgent"


def test_coach_decision_invalid_action_rejected() -> None:
    with pytest.raises(ValidationError):
        CoachDecision(
            observation="x",
            thought="x",
            response_steps=[],
            action="not_a_real_action",  # type: ignore[arg-type]
            params={},
        )


def test_coach_decision_call_tool_rejected() -> None:
    """The Coach has no call_tool action in the 3-agent topology — tools
    are owned by the worker agents. A Coach decision that tries to call a
    tool directly must fail at parse time so the system can't be tricked
    into bypassing the agent's safety checks."""
    with pytest.raises(ValidationError):
        CoachDecision(
            observation="x",
            thought="x",
            response_steps=[],
            action="call_tool",  # type: ignore[arg-type]
            params={"tool_name": "QuantitiesFinder", "task": "..."},
        )


# ---- Medical assessment ----------------------------------------------------
def test_medical_assessment_complete_round_trip() -> None:
    payload = {
        "medical_reasoning": "BMI within normal range; protein target raised for muscle gain",
        "observation": "all fields present",
        "risk_assessment_priorities": ["maintain micronutrient adequacy"],
        "assessment_plan": [],
        "action_type": "assessment_complete",
        "result": {
            "assessment_summary": "healthy male, hypertrophy goal",
            "flags_to_set": [],
            "recommendations": ["maintain hydration"],
            "requires_professional_consultation": False,
            "calculations": {
                "BMI": 23.4,
                "BMR": 1750,
                "TDEE": 2700,
                "daily_target_calories": 2900,
                "macro_targets": {"protein_g": 180, "fat_g": 70, "carbohydrates_g": 360},
            },
        },
    }
    decision = MedicalAssessmentDecision.model_validate(payload)
    assert decision.action_type == "assessment_complete"
    assert isinstance(decision.result, MedicalAssessmentResult)
    assert decision.result.calculations.macro_targets.protein_g == 180


def test_calculations_negative_values_rejected() -> None:
    with pytest.raises(ValidationError):
        Calculations(
            BMI=-1,  # negative not allowed
            BMR=1700,
            TDEE=2500,
            daily_target_calories=2200,
            macro_targets=MacroTargets(protein_g=120, fat_g=60, carbohydrates_g=250),
        )


# ---- Planner ---------------------------------------------------------------
def test_planner_provide_plan_with_dict_final_plan() -> None:
    decision = PlannerDecision(
        observation="all data ready",
        thought="returning final plan",
        planning_steps=[],
        action_type="provide_plan",
        final_plan={"days": [{"breakfast": "oats"}], "trace": "Coach->Planner"},
    )
    assert decision.action_type == "provide_plan"
    assert decision.final_plan is not None


def test_food_item_strict_grams_non_negative() -> None:
    with pytest.raises(ValidationError):
        FoodItem(
            name="oats",
            grams=-10,
            calories=389,
            protein_g=17,
            fat_g=7,
            carbohydrates_g=66,
        )


def test_final_plan_minimal() -> None:
    plan = FinalPlan(
        days=[
            [
                FoodItem(
                    name="oats",
                    grams=80,
                    calories=311,
                    protein_g=14,
                    fat_g=5,
                    carbohydrates_g=53,
                )
            ]
        ],
        daily_totals={"calories": 311},
    )
    assert plan.days[0][0].name == "oats"


# ---- Planner tool name (Literal enforcement) -------------------------------
def test_planner_decision_unknown_tool_rejected() -> None:
    """``tool_name`` is a strict Literal so a misroute fails at parse time."""
    with pytest.raises(ValidationError):
        PlannerDecision(
            observation="x",
            thought="x",
            planning_steps=[],
            action_type="call_tool",
            tool_name="ComputationTool",  # type: ignore[arg-type]
            tool_task="...",
        )
