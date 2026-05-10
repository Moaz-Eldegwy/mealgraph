"""Tests for the ValidationAgent critic loop.

Most of the value lives in the deterministic checks — they are pure code,
require no LLM, and can be exercised cheaply across edge cases.
"""

from __future__ import annotations

from typing import Any, Dict

from schemas import ValidationDecision
from state import initialize_empty_memory
from validation import ValidationAgent


# A minimal deterministic-only stub LLM for the cases that should never need
# the LLM layer (allergy violation -> verdict "reject" short-circuits LLM).
class _NeverCalledLLM:
    def call_typed(self, *args: Any, **kwargs: Any):
        raise AssertionError("LLM should not be called when deterministic check rejects.")


# ---------------------------------------------------------------------------
def _build_memory(
    *,
    allergies=None,
    dislikes="",
    target_calories=2000,
    macros=(150, 70, 200),
) -> Dict[str, Any]:
    memory = initialize_empty_memory()
    memory["user_profile"] = {
        "name": "Test",
        "country": "Egypt",
        "allergies": allergies or [],
        "food_dislikes": dislikes,
    }
    memory["flags_and_assessments"] = {
        "assessment_status": "assessment_complete",
        "calculations": {
            "BMI": 22,
            "BMR": 1600,
            "TDEE": 2000,
            "daily_target_calories": target_calories,
            "macro_targets": {
                "protein_g": macros[0],
                "fat_g": macros[1],
                "carbohydrates_g": macros[2],
            },
        },
        "flags": [],
        "recommendations": [],
        "requires_professional_consultation": False,
    }
    return memory


def _set_plan(memory: Dict[str, Any], plan: Dict[str, Any]) -> None:
    memory.setdefault("plans", {})["current_plan"] = plan


# ---- deterministic-only paths ----------------------------------------------
def test_passes_when_plan_within_tolerances(mock_llm_factory) -> None:
    memory = _build_memory(target_calories=2000, macros=(150, 70, 200))
    _set_plan(
        memory,
        {
            "days": [
                {
                    "name": "chicken_breast",
                    "calories": 1000,
                    "protein_g": 100,
                    "fat_g": 30,
                    "carbohydrates_g": 100,
                },
                {
                    "name": "rice",
                    "calories": 1000,
                    "protein_g": 50,
                    "fat_g": 40,
                    "carbohydrates_g": 100,
                },
            ],
        },
    )
    # Pre-supply a "no-issues" LLM verdict so the LLM layer is happy.
    llm = mock_llm_factory([ValidationDecision(verdict="pass", issues=[])])
    agent = ValidationAgent(llm)
    out = agent.handle_task("validate plan", memory)

    decision = ValidationDecision.model_validate_json(out)
    assert decision.verdict == "pass"
    assert decision.issues == []
    assert memory["flags_and_assessments"]["last_validation"]["verdict"] == "pass"


def test_allergy_violation_rejected_without_llm() -> None:
    memory = _build_memory(allergies=["peanut"])
    _set_plan(
        memory,
        {
            "days": [
                {
                    "name": "peanut butter sandwich",
                    "calories": 400,
                    "protein_g": 15,
                    "fat_g": 20,
                    "carbohydrates_g": 40,
                }
            ]
        },
    )
    agent = ValidationAgent(_NeverCalledLLM())
    out = agent.handle_task("validate", memory)
    decision = ValidationDecision.model_validate_json(out)

    assert decision.verdict == "reject"
    assert any(i.code == "allergy_violation" for i in decision.issues)


def test_calorie_deviation_triggers_revise(mock_llm_factory) -> None:
    memory = _build_memory(target_calories=2000)
    _set_plan(
        memory,
        {
            "days": [
                {
                    "name": "tiny salad",
                    "calories": 800,  # way under 2000 target -> 60% deviation
                    "protein_g": 30,
                    "fat_g": 20,
                    "carbohydrates_g": 60,
                }
            ]
        },
    )
    llm = mock_llm_factory([ValidationDecision(verdict="pass", issues=[])])
    agent = ValidationAgent(llm)
    out = agent.handle_task("validate", memory)
    decision = ValidationDecision.model_validate_json(out)

    assert decision.verdict == "revise"
    assert any(i.code == "calorie_deviation" for i in decision.issues)


def test_disliked_food_only_low_severity_still_passes(mock_llm_factory) -> None:
    memory = _build_memory(dislikes="okra", target_calories=2000)
    _set_plan(
        memory,
        {
            "days": [
                {
                    "name": "okra stew",
                    "calories": 2000,
                    "protein_g": 150,
                    "fat_g": 70,
                    "carbohydrates_g": 200,
                }
            ]
        },
    )
    llm = mock_llm_factory([ValidationDecision(verdict="pass", issues=[])])
    agent = ValidationAgent(llm)
    decision = ValidationDecision.model_validate_json(agent.handle_task("validate", memory))

    # Low-severity issues alone don't escalate the verdict.
    assert decision.verdict == "pass"
    assert any(i.code == "disliked_food" and i.severity == "low" for i in decision.issues)


def test_missing_plan_rejected() -> None:
    memory = _build_memory()
    # Intentionally no current_plan set
    agent = ValidationAgent(_NeverCalledLLM())
    decision = ValidationDecision.model_validate_json(agent.handle_task("validate", memory))
    assert decision.verdict == "reject"
    assert any(i.code == "missing_plan" for i in decision.issues)


def test_requires_human_review_propagates(mock_llm_factory) -> None:
    memory = _build_memory()
    memory["flags_and_assessments"]["requires_professional_consultation"] = True
    _set_plan(
        memory,
        {
            "days": [
                {
                    "name": "balanced meal",
                    "calories": 2000,
                    "protein_g": 150,
                    "fat_g": 70,
                    "carbohydrates_g": 200,
                }
            ]
        },
    )
    llm = mock_llm_factory([ValidationDecision(verdict="pass", issues=[])])
    agent = ValidationAgent(llm)
    decision = ValidationDecision.model_validate_json(agent.handle_task("validate", memory))
    # Even on a clean pass, HITL flag must propagate from the assessment.
    assert decision.requires_human_review is True
