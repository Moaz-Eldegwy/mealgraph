"""Pydantic models for agent inputs and outputs.

This module is the contract between the LLM, the orchestration layer, and the
test suite. Every agent's decision now passes through one of these models — so:

* Gemini's ``response_schema`` (constrained decoding) returns guaranteed-shape
  JSON; we no longer rely on regex / ``json_repair`` for the high-stakes path.
* Tests can construct decisions directly without hand-crafted JSON strings.
* Phase 2 can split agent loops into LangGraph nodes that pass typed objects
  between them.

Where Gemini's schema support is fussy (e.g. discriminated unions with
``$ref``), we keep the outer envelope strict and leave per-action ``params``
as a free dict — the agent dispatcher validates it at use time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / leaf types
# ---------------------------------------------------------------------------
StepStatus = Literal["pending", "in_progress", "completed", "skipped", "failed"]


class ResponseStep(BaseModel):
    """A single step in the Coach's response plan."""

    id: int
    actor: str = Field(
        description="Who executes this step. Examples: 'CoachAgent', 'MedicalAssessmentAgent', "
        "'PlannerAgent', 'ValidationAgent', 'user'.",
    )
    description: str
    prerequisites: List[str] = Field(default_factory=list)
    status: StepStatus = "pending"


class MacroTargets(BaseModel):
    """Daily macronutrient targets in grams (single integer values)."""

    protein_g: int = Field(ge=0)
    fat_g: int = Field(ge=0)
    carbohydrates_g: int = Field(ge=0)


class Calculations(BaseModel):
    """Derived anthropometric + nutritional values from the assessment."""

    BMI: float = Field(ge=0)
    BMR: float = Field(ge=0)
    TDEE: float = Field(ge=0)
    daily_target_calories: int = Field(ge=0)
    macro_targets: MacroTargets


# ---------------------------------------------------------------------------
# Coach Agent
# ---------------------------------------------------------------------------
CoachActionType = Literal[
    "call_agent",
    "call_tool",
    "ask_user",
    "write_memory",
    "compose_response",
]


class CoachDecision(BaseModel):
    """Single turn of the Coach orchestrator.

    Outer shape is strict; ``params`` is left as a dict because Gemini's
    schema layer struggles with deeply discriminated unions. The dispatcher
    in :mod:`workflow` validates ``params`` against the action type.
    """

    observation: str
    thought: str
    response_steps: List[ResponseStep] = Field(default_factory=list)
    action: CoachActionType
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Action-specific parameters. Required keys per action: "
            "call_agent={agent_name, task}, call_tool={tool_name, task}, "
            "ask_user={prompt}, write_memory={partition, data}, "
            "compose_response={text}."
        ),
    )


# ---------------------------------------------------------------------------
# Medical Assessment Agent
# ---------------------------------------------------------------------------
MedicalActionType = Literal["call_tool", "ask_user", "assessment_complete"]


class MedicalAssessmentResult(BaseModel):
    """Final payload stored in ``memory.flags_and_assessments``."""

    assessment_summary: str
    flags_to_set: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    requires_professional_consultation: bool = False
    calculations: Calculations
    evidence_sources: List[str] = Field(default_factory=list)
    trace: str = ""
    requires_tool_retry: bool = False
    data_confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class MedicalAssessmentDecision(BaseModel):
    """Per-iteration output of the Medical Assessment Agent loop."""

    medical_reasoning: str
    observation: str
    risk_assessment_priorities: List[str] = Field(default_factory=list)
    assessment_plan: List[ResponseStep] = Field(default_factory=list)

    action_type: MedicalActionType
    # action-specific fields (kept flat — see CoachDecision rationale)
    tool_name: Optional[str] = None
    tool_task: Optional[str] = None
    fields: List[str] = Field(default_factory=list)  # for ask_user
    result: Optional[MedicalAssessmentResult] = None  # for assessment_complete


# ---------------------------------------------------------------------------
# Planner Agent
# ---------------------------------------------------------------------------
PlannerActionType = Literal["call_tool", "draft_plan", "provide_plan"]


class FoodItem(BaseModel):
    """A single ingredient on the plan, post-solver."""

    name: str
    grams: float = Field(ge=0)
    calories: float = Field(ge=0)
    protein_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)
    carbohydrates_g: float = Field(ge=0)
    meal_group: Optional[str] = None


class FinalPlan(BaseModel):
    """The shape stored in ``memory.plans.current_plan``."""

    days: List[List[FoodItem]] = Field(
        description="One inner list per day. Most plans return a single day.",
    )
    daily_totals: Dict[str, float] = Field(default_factory=dict)
    notes: str = ""
    sources: List[str] = Field(default_factory=list)
    trace: str = ""


class PlannerDecision(BaseModel):
    """Per-iteration output of the Planner Agent loop."""

    observation: str
    thought: str
    planning_steps: List[ResponseStep] = Field(default_factory=list)

    action_type: PlannerActionType
    tool_name: Optional[str] = None
    tool_task: Optional[str] = None
    drafted_plan: Optional[Dict[str, Any]] = None  # free shape pre-solver
    final_plan: Optional[Dict[str, Any]] = None  # free shape until validation lands in Phase 2


# ---------------------------------------------------------------------------
# Validation Agent (lands in Phase 2; defined here so Phase 1 schemas are
# the single source of truth)
# ---------------------------------------------------------------------------
ValidationVerdict = Literal["pass", "revise", "reject"]
ValidationSeverity = Literal["low", "medium", "high"]


class ValidationIssue(BaseModel):
    code: str = Field(description="Stable error code, e.g. 'allergy_violation'.")
    description: str
    severity: ValidationSeverity = "medium"


class ValidationDecision(BaseModel):
    verdict: ValidationVerdict
    issues: List[ValidationIssue] = Field(default_factory=list)
    notes: str = ""
    requires_human_review: bool = False


__all__ = [
    "Calculations",
    "CoachActionType",
    "CoachDecision",
    "FinalPlan",
    "FoodItem",
    "MacroTargets",
    "MedicalActionType",
    "MedicalAssessmentDecision",
    "MedicalAssessmentResult",
    "PlannerActionType",
    "PlannerDecision",
    "ResponseStep",
    "StepStatus",
    "ValidationDecision",
    "ValidationIssue",
    "ValidationSeverity",
    "ValidationVerdict",
]
