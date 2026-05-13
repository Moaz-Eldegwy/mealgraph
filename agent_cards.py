"""Agent cards — A2A-style capability descriptors.

The Agent-to-Agent (A2A) pattern standardises how agents discover each
other's capabilities. Each agent publishes a "card" that declares: name,
description, IO schemas, supported intents, and escalation policy. A
registry collects them; an orchestrator (here, the Coach) looks up cards
rather than hardcoding agent names.

This module ships the data model and an in-process registry. The same
registry can later be served over HTTP at
``/.well-known/agent-card.json`` to enable cross-system collaboration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CapabilityIO(BaseModel):
    """A description of one IO surface (input shape or output shape)."""

    description: str
    json_schema: Optional[Dict[str, Any]] = None
    example: Optional[Dict[str, Any]] = None


class Capability(BaseModel):
    """One thing the agent can do."""

    name: str = Field(description="Stable identifier, e.g. 'assess_user'.")
    description: str
    input: CapabilityIO
    output: CapabilityIO
    side_effects: List[str] = Field(
        default_factory=list,
        description="Free-text list of memory partitions or tools the capability touches.",
    )


class AgentCard(BaseModel):
    """The card every agent publishes.

    Inspired by Google's A2A spec + OpenAI's plugin manifest format.
    """

    name: str
    version: str = "0.1.0"
    description: str
    role: str = Field(
        description=(
            "Free-text role label, e.g. 'orchestrator', 'specialist:medical', "
            "'specialist:nutrition_planning', 'critic', 'retrieval'."
        )
    )
    capabilities: List[Capability]
    requires_human_review: bool = Field(
        default=False,
        description="True for medically sensitive specialists; gates HITL escalation.",
    )
    contact: Optional[str] = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class AgentRegistry:
    """Trivial in-process registry. Real systems would back this with a DB."""

    def __init__(self) -> None:
        self._cards: Dict[str, AgentCard] = {}

    def register(self, card: AgentCard) -> None:
        self._cards[card.name] = card

    def get(self, name: str) -> Optional[AgentCard]:
        return self._cards.get(name)

    def list(self) -> List[AgentCard]:
        return list(self._cards.values())

    def by_role(self, role: str) -> List[AgentCard]:
        return [c for c in self._cards.values() if c.role == role]


# ---------------------------------------------------------------------------
# Default cards for every agent in this system
# ---------------------------------------------------------------------------
def default_cards() -> List[AgentCard]:
    return [
        AgentCard(
            name="CoachAgent",
            description="Central orchestrator: turns user intent into a workflow of agent/tool calls.",
            role="orchestrator",
            capabilities=[
                Capability(
                    name="orchestrate",
                    description="Plan and dispatch one action per turn until compose_response.",
                    input=CapabilityIO(description="NutritionState (full graph state)."),
                    output=CapabilityIO(description="Updated NutritionState with current_action set."),
                    side_effects=["response_steps", "previous_actions"],
                )
            ],
        ),
        AgentCard(
            name="MedicalAssessmentAgent",
            description="Produces evidence-based medical assessment + clinical flags + calculations.",
            role="specialist:medical",
            requires_human_review=True,
            capabilities=[
                Capability(
                    name="assess_user",
                    description="Compute BMI/BMR/TDEE, set clinical flags, attach evidence sources.",
                    input=CapabilityIO(description="task: str, memory: dict"),
                    output=CapabilityIO(description="assessment_summary: str (memory side-effect: flags_and_assessments)"),
                    side_effects=["memory.flags_and_assessments"],
                )
            ],
        ),
        AgentCard(
            name="PlannerAgent",
            description="Personalised meal plans constrained by the medical assessment.",
            role="specialist:nutrition_planning",
            capabilities=[
                Capability(
                    name="plan_meals",
                    description=(
                        "Draft a plan, batch-fetch nutrition facts, run QuantitiesFinder LP, "
                        "finalise."
                    ),
                    input=CapabilityIO(description="task: str, memory: dict"),
                    output=CapabilityIO(description="JSON-serialised final_plan or error payload"),
                    side_effects=["memory.plans"],
                )
            ],
        ),
        AgentCard(
            name="ValidationAgent",
            description="Critic: deterministic + LLM-graded checks; can demand revision or reject.",
            role="critic",
            capabilities=[
                Capability(
                    name="validate_plan",
                    description="Grade memory.plans.current_plan; return ValidationDecision.",
                    input=CapabilityIO(description="task: str, memory: dict"),
                    output=CapabilityIO(description="ValidationDecision JSON"),
                    side_effects=["memory.flags_and_assessments.last_validation"],
                )
            ],
        ),
        AgentCard(
            name="KnowledgeAgent",
            description="Citation-first retrieval (USDA / WHO / ADA / EFSA via biased web search).",
            role="retrieval",
            capabilities=[
                Capability(
                    name="lookup",
                    description="Answer 'kind' x 'query' with at least one source URL.",
                    input=CapabilityIO(
                        description='task: JSON {"kind": "nutrition"|"guideline"|"drug_interaction"|"general", "query": "..."}',
                    ),
                    output=CapabilityIO(description='JSON {"answer": "...", "citations": [url, ...]}'),
                )
            ],
        ),
    ]


def build_default_registry() -> AgentRegistry:
    reg = AgentRegistry()
    for c in default_cards():
        reg.register(c)
    return reg


__all__ = [
    "AgentCard",
    "AgentRegistry",
    "Capability",
    "CapabilityIO",
    "build_default_registry",
    "default_cards",
]
