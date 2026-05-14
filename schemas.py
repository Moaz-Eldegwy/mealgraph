"""Pydantic models for agent inputs and outputs.

These models are the contract between the LLM, the orchestration layer,
and the test suite:

* Gemini's ``response_schema`` (constrained decoding) returns guaranteed-shape
  JSON; the regex / ``json_repair`` path is a measured fallback only.
* Tests construct decisions as typed objects, with no hand-crafted JSON.
* Each agent loop can pass typed payloads between LangGraph nodes.

Where Gemini's schema layer is limited (deep discriminated unions over
``$ref`` definitions), the outer envelope is strict and ``params`` stays a
free dict whose keys are validated by the dispatcher at use time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Shared / leaf types
# ---------------------------------------------------------------------------
StepStatus = Literal["pending", "in_progress", "completed", "skipped", "failed"]


class ResponseStep(BaseModel):
    """A single step in the Coach's response plan."""

    id: int
    actor: str = Field(
        description="Who executes this step. Examples: 'CoachAgent', "
        "'MedicalAssessmentAgent', 'PlannerAgent', 'user'.",
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
# Coach dispatches agents, asks the user, writes memory, or composes. Tools
# are exclusively internal to the worker agents — the Coach never calls one
# directly, so ``call_tool`` is intentionally absent.
CoachActionType = Literal[
    "call_agent",
    "ask_user",
    "write_memory",
    "compose_response",
]


class CoachDecision(BaseModel):
    """Single turn of the Coach orchestrator.

    The outer envelope is strictly typed; ``params`` stays a free dict so a
    single schema covers every action variant — the workflow dispatcher
    validates the keys at use time. ``json_schema_extra`` declares the
    possible keys to Gemini so structured decoding fills them rather than
    returning an empty object (Gemini's ``response_schema`` does not accept
    ``additionalProperties``, so the property list is the only available
    hint).
    """

    observation: str
    thought: str
    response_steps: List[ResponseStep] = Field(default_factory=list)
    action: CoachActionType
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Action-specific parameters. Required keys per action: "
            "call_agent={agent_name, task}, ask_user={prompt}, "
            "write_memory={partition, data}, compose_response={text}."
        ),
        json_schema_extra={
            "properties": {
                "agent_name": {"type": "string"},
                "task": {"type": "string"},
                "prompt": {"type": "string"},
                "partition": {"type": "string"},
                "data": {"type": "object"},
                "text": {"type": "string"},
            },
        },
    )


# ---------------------------------------------------------------------------
# Medical Assessment Agent
# ---------------------------------------------------------------------------
# Medical has exactly one optional tool (WebSearchTool) and is otherwise
# driven by deterministic Python (``nutrition_formulas.full_assessment``),
# so ``tool_name`` is unnecessary in the decision schema.
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
    # WebSearchTool is the only tool Medical can invoke; the agent fills in
    # the tool name itself when dispatching, so the schema only carries the
    # research question.
    tool_task: Optional[str] = None
    fields: List[str] = Field(default_factory=list)  # for ask_user
    result: Optional[MedicalAssessmentResult] = None  # for assessment_complete

    @model_validator(mode="before")
    @classmethod
    def _infer_action_type(cls, data: Any) -> Any:
        """Derive ``action_type`` from the populated discriminator fields.

        Constrained decoding will sometimes emit ``tool_task`` or ``result``
        without the matching ``action_type`` discriminator; infer it from
        whatever the model did populate so the dispatch logic stays simple.
        """
        if not isinstance(data, dict) or data.get("action_type"):
            return data
        if data.get("result"):
            data["action_type"] = "assessment_complete"
        elif data.get("tool_task"):
            data["action_type"] = "call_tool"
        elif data.get("fields"):
            data["action_type"] = "ask_user"
        return data


# ---------------------------------------------------------------------------
# Planner Agent
# ---------------------------------------------------------------------------
PlannerActionType = Literal["call_tool", "draft_plan", "provide_plan"]

# Planner has two tools; the schema enforces the choice at parse time so
# the agent's dispatch can't be tricked into calling something else.
PlannerToolName = Literal["WebSearchTool", "QuantitiesFinder"]


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


_PLAN_SCHEMA_HINT = {
    "properties": {
        # Gemini's ``response_schema`` requires ``items`` on every array
        # and explicit ``properties`` on every nested object — without
        # them, constrained decoding emits ``days: [{}]``. The Planner's
        # internal check_plan walks ``days`` recursively, so the flat-food
        # list documented here is interchangeable with the nested
        # ``List[List[FoodItem]]`` form defined by :class:`FinalPlan`.
        "days": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "meal_group": {"type": "string"},
                    "grams": {"type": "number"},
                    "calories": {"type": "number"},
                    "protein_g": {"type": "number"},
                    "fat_g": {"type": "number"},
                    "carbohydrates_g": {"type": "number"},
                    "saturated_fat_g": {"type": "number"},
                    "fiber_g": {"type": "number"},
                },
            },
        },
        "daily_totals": {
            "type": "object",
            "properties": {
                "calories": {"type": "number"},
                "protein_g": {"type": "number"},
                "fat_g": {"type": "number"},
                "carbohydrates_g": {"type": "number"},
            },
        },
        "notes": {"type": "string"},
        "sources": {
            "type": "array",
            "items": {"type": "string"},
        },
        "trace": {"type": "string"},
        "error": {"type": "string"},
    },
}


class PlannerDecision(BaseModel):
    """Per-iteration output of the Planner Agent loop."""

    observation: str
    thought: str
    planning_steps: List[ResponseStep] = Field(default_factory=list)

    action_type: PlannerActionType
    tool_name: Optional[PlannerToolName] = None
    tool_task: Optional[str] = None
    drafted_plan: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Free-shape draft plan (pre-solver). Typical shape: "
            '{"days": [...]} or {"meals": [...]}.'
        ),
        json_schema_extra=_PLAN_SCHEMA_HINT,
    )
    final_plan: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Free-shape final plan (post-solver). Typical shape: "
            '{"days": [...], "daily_totals": {...}, "notes": "...", '
            '"sources": [...], "trace": "..."}.'
        ),
        json_schema_extra=_PLAN_SCHEMA_HINT,
    )

    @model_validator(mode="before")
    @classmethod
    def _infer_action_type(cls, data: Any) -> Any:
        """Derive ``action_type`` from the populated discriminator fields.

        Mirrors :meth:`MedicalAssessmentDecision._infer_action_type`:
        constrained decoding will sometimes emit
        ``{observation, thought, planning_steps, final_plan}`` without an
        ``action_type``; infer the intent so the agent loop can dispatch.
        """
        if not isinstance(data, dict) or data.get("action_type"):
            return data
        if data.get("final_plan"):
            data["action_type"] = "provide_plan"
        elif data.get("drafted_plan"):
            data["action_type"] = "draft_plan"
        elif data.get("tool_name") or data.get("tool_task"):
            data["action_type"] = "call_tool"
        return data


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
    "PlannerToolName",
    "ResponseStep",
    "StepStatus",
]
