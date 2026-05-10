"""Tests for the A2A-style agent cards / registry."""

from __future__ import annotations

from agent_cards import (
    AgentCard,
    AgentRegistry,
    Capability,
    CapabilityIO,
    build_default_registry,
    default_cards,
)


def test_default_cards_cover_every_agent() -> None:
    names = {c.name for c in default_cards()}
    assert {
        "CoachAgent",
        "MedicalAssessmentAgent",
        "PlannerAgent",
        "ValidationAgent",
        "KnowledgeAgent",
    } <= names


def test_registry_register_and_get() -> None:
    reg = AgentRegistry()
    card = AgentCard(
        name="TestAgent",
        description="An agent for tests.",
        role="critic",
        capabilities=[
            Capability(
                name="grade",
                description="Grade something.",
                input=CapabilityIO(description="text"),
                output=CapabilityIO(description="grade"),
            )
        ],
    )
    reg.register(card)
    assert reg.get("TestAgent") is card
    assert reg.get("MissingAgent") is None
    assert card in reg.list()


def test_registry_by_role() -> None:
    reg = build_default_registry()
    specialists = reg.by_role("specialist:medical")
    assert len(specialists) == 1
    assert specialists[0].name == "MedicalAssessmentAgent"


def test_medical_card_marks_human_review() -> None:
    """MedicalAssessmentAgent must declare requires_human_review=True so the
    Coach / Validator can route to HITL when appropriate."""
    reg = build_default_registry()
    medical = reg.get("MedicalAssessmentAgent")
    assert medical is not None
    assert medical.requires_human_review is True

    # And no other agent should accidentally inherit that flag.
    others = [c for c in reg.list() if c.name != "MedicalAssessmentAgent"]
    assert all(not c.requires_human_review for c in others)


def test_capability_io_round_trip() -> None:
    cap = Capability(
        name="x",
        description="d",
        input=CapabilityIO(description="i", json_schema={"type": "object"}),
        output=CapabilityIO(description="o", example={"k": 1}),
        side_effects=["memory.x"],
    )
    serialised = cap.model_dump()
    re_built = Capability.model_validate(serialised)
    assert re_built == cap
